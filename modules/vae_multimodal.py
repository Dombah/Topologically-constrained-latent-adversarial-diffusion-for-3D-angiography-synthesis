"""Multimodal VAE for the T1/T2/PD condition: three encoders and three decoders sharing one
8-channel latent space, trained with L1 + KL + a cross-modality latent-alignment term.

Contents: the model (VAEv2Multimodal), its paired patch dataset, the loss, the training and
validation loops (train_vaev2_multimodal), tiled deterministic encoding, checkpoint loading
and the split-folder latent export (extract_vae_latents_from_splits).

Carried over unchanged from the original modules/VAEv2.py and modules/utils.py; the MRA-only
variants, attention variants and test-set preview machinery were dropped.
"""

import gc
import json
import math
import re
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Mapping

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def _progress_bar(iterable, enabled=True, **kwargs):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, **kwargs)


def _set_progress_postfix(progress, values):
    if hasattr(progress, "set_postfix"):
        progress.set_postfix(values)


def _training_dataloader(
    dataset,
    batch_size,
    shuffle,
    num_workers,
    pin_memory,
):
    # In Windows notebooks, persistent worker reuse can fail after many epochs.
    # Rebuild workers normally, and fall back to num_workers=0 if spawn fails.
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=False,
    )


def _training_progress_with_worker_fallback(
    dataset,
    batch_size,
    shuffle,
    num_workers,
    pin_memory,
    show_progress,
    total,
    desc,
):
    active_workers = int(num_workers)
    while True:
        loader = _training_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=active_workers,
            pin_memory=pin_memory,
        )
        try:
            progress = _progress_bar(
                loader,
                enabled=show_progress,
                total=total,
                desc=desc,
                leave=False,
            )
            return loader, progress, active_workers
        except (OSError, RuntimeError) as exc:
            if active_workers <= 0:
                raise
            print(
                "DataLoader worker startup failed; retrying this epoch with "
                f"num_workers=0. Original error: {type(exc).__name__}: {exc}"
            )
            del loader
            gc.collect()
            active_workers = 0


def _json_safe(value):
    if torch.is_tensor(value):
        return _json_safe(value.detach().cpu())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(record), sort_keys=True) + "\n")


def _resolve_resume_path(checkpoint_dir, resume_path=None, resume_from_best=False):
    checkpoint_dir = Path(checkpoint_dir)
    if resume_path is False:
        return None
    if isinstance(resume_path, str) and resume_path.lower() in {"scratch", "none", "false", "new"}:
        return None

    resume_key = resume_path.lower() if isinstance(resume_path, str) else None
    if resume_key == "latest":
        candidates = [checkpoint_dir / "latest.pt"]
    elif resume_key == "best":
        candidates = [checkpoint_dir / "best.pt"]
    elif resume_path is not None:
        candidate = Path(resume_path)
        if not candidate.is_absolute() and not candidate.exists():
            candidate = checkpoint_dir / candidate
        candidates = [candidate]
    elif resume_from_best:
        candidates = [checkpoint_dir / "best.pt"]
    else:
        candidates = [checkpoint_dir / "latest.pt", checkpoint_dir / "best.pt"]

    for candidate in candidates:
        if candidate.exists():
            return candidate
    if len(candidates) == 1:
        print(f"Resume checkpoint not found, starting from scratch: {candidates[0]}")
    else:
        joined = ", ".join(str(candidate) for candidate in candidates)
        print(f"No resume checkpoint found, starting from scratch. Checked: {joined}")
    return None


def _kl_weight_for_epoch(
    epoch,
    start_weight,
    final_weight,
    warmup_epochs=0,
    warmup_start_epoch=1,
):
    epoch = int(epoch)
    start_weight = float(start_weight)
    final_weight = float(final_weight)
    warmup_epochs = int(warmup_epochs or 0)
    warmup_start_epoch = int(warmup_start_epoch or 1)

    if warmup_epochs <= 0:
        return final_weight
    if epoch <= warmup_start_epoch:
        return start_weight
    if epoch >= warmup_start_epoch + warmup_epochs:
        return final_weight

    alpha = (epoch - warmup_start_epoch) / float(warmup_epochs)
    return start_weight + alpha * (final_weight - start_weight)


def _as_clamp_bounds(bounds, name):
    if bounds is None:
        return None
    if len(bounds) != 2:
        raise ValueError(f"{name} must be None or a 2-item (min, max) tuple")
    lower, upper = bounds
    return float(lower), float(upper)


def _optional_clamp(tensor, bounds):
    if bounds is None:
        return tensor
    lower, upper = bounds
    return tensor.clamp(lower, upper)


def _kl_decomposition(mu, logvar, mu_clamp=(-20.0, 20.0), logvar_clamp=(-30.0, 20.0)):
    mu = _optional_clamp(mu.float(), mu_clamp)
    logvar = _optional_clamp(logvar.float(), logvar_clamp)
    var = logvar.exp()
    kl_mu = 0.5 * mu.pow(2).mean()
    kl_var = 0.5 * (var - 1.0 - logvar).mean()
    return kl_mu, kl_var, kl_mu + kl_var


def _cuda_memory_gb(device):
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return {}
    device_obj = torch.device(device)
    return {
        "vram_alloc": f"{torch.cuda.memory_allocated(device_obj) / (1024 ** 3):.2f}G",
        "vram_reserved": f"{torch.cuda.memory_reserved(device_obj) / (1024 ** 3):.2f}G",
    }


def _metric_data_range(target, data_range=None, eps=1e-8):
    if data_range is not None:
        return torch.as_tensor(float(data_range), device=target.device, dtype=target.dtype)
    reduce_dims = tuple(range(1, target.dim()))
    value_range = target.amax(dim=reduce_dims, keepdim=True) - target.amin(dim=reduce_dims, keepdim=True)
    return value_range.clamp_min(eps)


def _psnr_3d(recon, target, data_range=None, eps=1e-8):
    recon = recon.float()
    target = target.float()
    reduce_dims = tuple(range(1, target.dim()))
    mse = torch.mean((recon - target).pow(2), dim=reduce_dims).clamp_min(eps)
    value_range = _metric_data_range(target, data_range=data_range, eps=eps)
    if value_range.dim() > 0:
        value_range = value_range.flatten()
    return 20.0 * torch.log10(value_range.clamp_min(eps)) - 10.0 * torch.log10(mse)


def _ssim_3d(recon, target, data_range=None, window_size=7, eps=1e-8):
    recon = recon.float()
    target = target.float()
    if window_size in (None, 0, "global"):
        return _global_ssim_3d(recon, target, data_range=data_range, eps=eps)
    if recon.dim() != 5 or target.dim() != 5:
        raise ValueError("SSIM expects tensors with shape [B, C, D, H, W]")
    spatial = target.shape[-3:]
    window_size = min(int(window_size), *spatial)
    if window_size < 1:
        raise ValueError(f"Invalid SSIM window size for shape {tuple(target.shape)}")
    if window_size % 2 == 0:
        window_size -= 1
    padding = window_size // 2

    mu_x = F.avg_pool3d(recon, window_size, stride=1, padding=padding, count_include_pad=False)
    mu_y = F.avg_pool3d(target, window_size, stride=1, padding=padding, count_include_pad=False)
    sigma_x = F.avg_pool3d(recon * recon, window_size, stride=1, padding=padding, count_include_pad=False) - mu_x.pow(2)
    sigma_y = F.avg_pool3d(target * target, window_size, stride=1, padding=padding, count_include_pad=False) - mu_y.pow(2)
    sigma_xy = F.avg_pool3d(recon * target, window_size, stride=1, padding=padding, count_include_pad=False) - mu_x * mu_y

    value_range = _metric_data_range(target, data_range=data_range, eps=eps)
    c1 = (0.01 * value_range).pow(2)
    c2 = (0.03 * value_range).pow(2)
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2)
    ssim_map = numerator / denominator.clamp_min(eps)
    return ssim_map.mean(dim=tuple(range(1, ssim_map.dim())))


def _global_ssim_3d(recon, target, data_range=None, eps=1e-8):
    recon = recon.float()
    target = target.float()
    reduce_dims = tuple(range(1, target.dim()))
    mu_x = recon.mean(dim=reduce_dims)
    mu_y = target.mean(dim=reduce_dims)
    var_x = recon.var(dim=reduce_dims, unbiased=False)
    var_y = target.var(dim=reduce_dims, unbiased=False)
    cov_xy = ((recon - mu_x.view(-1, *([1] * (recon.dim() - 1)))) *
              (target - mu_y.view(-1, *([1] * (target.dim() - 1))))).mean(dim=reduce_dims)
    value_range = _metric_data_range(target, data_range=data_range, eps=eps)
    if value_range.dim() > 0:
        value_range = value_range.flatten()
    c1 = (0.01 * value_range).pow(2)
    c2 = (0.03 * value_range).pow(2)
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2)
    denominator = (mu_x.pow(2) + mu_y.pow(2) + c1) * (var_x + var_y + c2)
    return numerator / denominator.clamp_min(eps)


def _reconstruction_metrics(recon, target, data_range=None, ssim_window_size=7):
    with torch.no_grad():
        recon = recon.float()
        target = target.float()

        reduce_dims = tuple(range(1, target.dim()))

        l1_per_sample = torch.mean(torch.abs(recon - target), dim=reduce_dims)
        mse_per_sample = torch.mean((recon - target).pow(2), dim=reduce_dims)

        psnr = _psnr_3d(recon, target, data_range=data_range)
        ssim = _ssim_3d(
            recon,
            target,
            data_range=data_range,
            window_size=ssim_window_size
        )

    return {
        "l1": float(l1_per_sample.mean().detach().cpu()),
        "mae": float(l1_per_sample.mean().detach().cpu()),  # optional alias
        "mse": float(mse_per_sample.mean().detach().cpu()),
        "psnr": float(psnr.mean().detach().cpu()),
        "ssim": float(ssim.mean().detach().cpu()),
    }


def _mip_l1_loss(recon, target, axes=(2, 3, 4)):
    recon = recon.float()
    target = target.float()
    losses = []
    for axis in axes:
        axis = int(axis)
        losses.append(F.l1_loss(recon.amax(dim=axis), target.amax(dim=axis)))
    if not losses:
        return recon.new_zeros(())
    return torch.stack(losses).mean()


def _central_slice_mse_loss(recon, target):
    recon = recon.float()
    target = target.float()
    if recon.dim() != 5 or target.dim() != 5:
        raise ValueError("Central slice MSE expects tensors with shape [B, C, D, H, W]")
    d = target.shape[2] // 2
    h = target.shape[3] // 2
    w = target.shape[4] // 2
    return (
        F.mse_loss(recon[:, :, d, :, :], target[:, :, d, :, :])
        + F.mse_loss(recon[:, :, :, h, :], target[:, :, :, h, :])
        + F.mse_loss(recon[:, :, :, :, w], target[:, :, :, :, w])
    ) / 3.0


def _batch_quantile_threshold(flat, quantile, max_values=1048576):
    flat = flat.detach().float()
    if flat.dim() != 2:
        flat = flat.reshape(flat.shape[0], -1)
    quantile = float(np.clip(quantile, 0.0, 1.0))
    if flat.shape[1] == 0:
        return flat.new_zeros((flat.shape[0],))
    if quantile <= 0.0:
        return flat.min(dim=1).values
    if quantile >= 1.0:
        return flat.max(dim=1).values
    max_values = int(max_values)
    if flat.shape[1] > max_values:
        stride = max(1, int(math.ceil(flat.shape[1] / float(max_values))))
        flat = flat[:, ::stride][:, :max_values]
    return torch.quantile(flat, quantile, dim=1)


def _vessel_weighted_l1_loss(recon, target, vessel_percentile=95.0, vessel_weight=5.0):
    recon = recon.float()
    target = target.float()
    vessel_percentile = float(np.clip(vessel_percentile, 0.0, 100.0))
    vessel_weight = float(vessel_weight)
    flat = target.flatten(start_dim=1)
    threshold = _batch_quantile_threshold(flat, vessel_percentile / 100.0)
    threshold = threshold.view(-1, *([1] * (target.dim() - 1)))
    vessel_mask = target >= threshold
    abs_error = (recon - target).abs()
    loss = abs_error.mean()
    if vessel_weight != 0.0 and vessel_mask.any():
        loss = loss + vessel_weight * abs_error[vessel_mask].sum() / abs_error.numel()
    return loss


def _background_false_positive_loss(recon, target, mask=None, background_threshold=0.05):
    recon = recon.float()
    target = target.float()
    if mask is not None:
        background_mask = mask.to(device=recon.device).float() <= 0.5
    else:
        background_mask = target <= float(background_threshold)
    if not background_mask.any():
        return recon.new_zeros(())
    false_positive = F.relu(recon - target)
    return false_positive[background_mask].mean()


def _inside_dark_false_negative_loss(recon, target, mask=None, dark_margin=0.03):
    recon = recon.float()
    target = target.float()
    if mask is not None:
        inside_mask = mask.to(device=recon.device).float() > 0.5
    else:
        inside_mask = target > float(dark_margin)
    if not inside_mask.any():
        return recon.new_zeros(())
    false_negative = F.relu(target - recon - float(dark_margin))
    return false_negative[inside_mask].mean()


def _init_latent_stats():
    return {
        name: {"sum": 0.0, "sumsq": 0.0, "count": 0, "min": math.inf, "max": -math.inf}
        for name in ("mu", "logvar", "std")
    }


def _update_tensor_stats(acc, name, tensor):
    tensor = tensor.detach().float()
    values = acc[name]
    values["sum"] += float(tensor.sum().cpu())
    values["sumsq"] += float(tensor.pow(2).sum().cpu())
    values["count"] += int(tensor.numel())
    values["min"] = min(values["min"], float(tensor.min().cpu()))
    values["max"] = max(values["max"], float(tensor.max().cpu()))


def _update_latent_stats(acc, mu, logvar):
    with torch.no_grad():
        _update_tensor_stats(acc, "mu", mu)
        _update_tensor_stats(acc, "logvar", logvar)
        _update_tensor_stats(acc, "std", torch.exp(0.5 * logvar))


def _finalize_latent_stats(acc):
    output = {}
    for name, values in acc.items():
        count = max(values["count"], 1)
        mean = values["sum"] / count
        variance = max(values["sumsq"] / count - mean * mean, 0.0)
        output[f"{name}_mean"] = mean
        output[f"{name}_std"] = math.sqrt(variance)
        output[f"{name}_min"] = values["min"] if values["count"] else float("nan")
        output[f"{name}_max"] = values["max"] if values["count"] else float("nan")
    return output


_LATENT_PERCENTILES = (
    ("p0_1", 0.001),
    ("p1", 0.01),
    ("p50", 0.50),
    ("p99", 0.99),
    ("p99_9", 0.999),
)


def _init_latent_extra_stats():
    return {
        "channel_vars": [],
        "percentiles": {
            name: {label: [] for label, _ in _LATENT_PERCENTILES}
            for name in ("mu", "logvar")
        },
    }


def _channel_variance(mu):
    mu = mu.detach().float()
    if mu.dim() >= 5:
        return mu.var(dim=(0, 2, 3, 4), unbiased=False)
    if mu.dim() >= 2:
        reduce_dims = tuple(dim for dim in range(mu.dim()) if dim != 1)
        return mu.var(dim=reduce_dims, unbiased=False)
    return mu.reshape(1).var(dim=0, unbiased=False).reshape(1)


def _sample_for_percentiles(tensor, max_values=262144):
    flat = tensor.detach().float().flatten()
    if flat.numel() <= int(max_values):
        return flat
    stride = max(1, int(math.ceil(flat.numel() / float(max_values))))
    return flat[::stride][:int(max_values)]


def _update_latent_extra_stats(acc, mu, logvar, include_percentiles=True):
    with torch.no_grad():
        acc["channel_vars"].append(_channel_variance(mu).detach().cpu())
        if not include_percentiles:
            return
        for name, tensor in (("mu", mu), ("logvar", logvar)):
            flat = _sample_for_percentiles(tensor)
            if flat.numel() == 0:
                continue
            qs = torch.tensor([q for _, q in _LATENT_PERCENTILES], device=flat.device, dtype=flat.dtype)
            values = torch.quantile(flat, qs).detach().cpu().tolist()
            for (label, _), value in zip(_LATENT_PERCENTILES, values):
                acc["percentiles"][name][label].append(float(value))


def _finalize_latent_extra_stats(
    acc,
    suffix=None,
    include_percentiles=True,
    active_threshold=1e-4,
):
    def out_key(name):
        return f"{name}_{suffix}" if suffix else name

    output = {}
    if acc["channel_vars"]:
        channel_vars = torch.stack(acc["channel_vars"]).float().mean(dim=0)
        output[out_key("active_channels")] = int((channel_vars > float(active_threshold)).sum().item())
        output[out_key("channel_var_min")] = float(channel_vars.min().item())
        output[out_key("channel_var_mean")] = float(channel_vars.mean().item())
        output[out_key("channel_var_max")] = float(channel_vars.max().item())
    else:
        output[out_key("active_channels")] = 0
        output[out_key("channel_var_min")] = float("nan")
        output[out_key("channel_var_mean")] = float("nan")
        output[out_key("channel_var_max")] = float("nan")

    if include_percentiles:
        for name, percentile_values in acc["percentiles"].items():
            for label, values in percentile_values.items():
                output[out_key(f"{name}_{label}")] = (
                    sum(values) / len(values) if values else float("nan")
                )
    return output


def _patient_id_from_sample(sample):
    return str(sample.get("patient") or sample.get("name") or "unknown")


def _scanner_from_patient(patient_id):
    parts = str(patient_id).split("-")
    if len(parts) >= 2:
        return parts[1]
    for scanner in ("Guys", "HH", "IOP"):
        if scanner.lower() in str(patient_id).lower():
            return scanner
    return "unknown"


def _slice_preview(volume, axis=2):
    array = np.asarray(volume, dtype=np.float32)
    array = np.squeeze(array)
    if array.ndim != 3:
        return array
    axis = int(axis)
    index = array.shape[axis] // 2
    return np.take(array, index, axis=axis)


def _preview_limits(*arrays, percentiles=(1.0, 99.0)):
    values = []
    for array in arrays:
        finite = np.asarray(array, dtype=np.float32)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            values.append(finite)
    if not values:
        return 0.0, 1.0
    finite = np.concatenate(values)
    vmin, vmax = np.percentile(finite, percentiles)
    if vmax <= vmin:
        vmin, vmax = float(finite.min()), float(finite.max())
    if vmax <= vmin:
        return 0.0, 1.0
    return float(vmin), float(vmax)


def _normalize_preview_slice(array, vmin, vmax):
    array = np.asarray(array, dtype=np.float32)
    if vmax <= vmin:
        return np.zeros_like(array, dtype=np.float32)
    array = np.clip(array, vmin, vmax)
    return (array - vmin) / (vmax - vmin)


def _normalize_preview_pair(target_slice, recon_slice):
    vmin, vmax = _preview_limits(target_slice, recon_slice)
    return (
        _normalize_preview_slice(target_slice, vmin, vmax),
        _normalize_preview_slice(recon_slice, vmin, vmax),
    )


def _select_preview_entries(entries, max_previews=16, required_scanners=("Guys", "HH", "IOP")):
    if not entries or max_previews is None or int(max_previews) <= 0:
        return []
    max_previews = int(max_previews)
    selected = []
    selected_ids = set()
    for scanner in required_scanners or ():
        for idx, entry in enumerate(entries):
            if idx in selected_ids:
                continue
            if str(entry.get("scanner", "")).lower() == str(scanner).lower():
                selected.append(entry)
                selected_ids.add(idx)
                break
    for idx, entry in enumerate(entries):
        if len(selected) >= max_previews:
            break
        if idx not in selected_ids:
            selected.append(entry)
            selected_ids.add(idx)
    return selected[:max_previews]


def _save_validation_preview(
    entries,
    output_path,
    max_previews=16,
    required_scanners=("Guys", "HH", "IOP"),
):
    selected = _select_preview_entries(entries, max_previews=max_previews, required_scanners=required_scanners)
    if not selected:
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not available; validation preview image was not saved")
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cases_per_row = 2
    cols = cases_per_row * 2
    rows = math.ceil(len(selected) / cases_per_row)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.3, rows * 3.7), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")

    for index, entry in enumerate(selected):
        row = index // cases_per_row
        col = (index % cases_per_row) * 2
        target_preview, recon_preview = _normalize_preview_pair(entry["target_slice"], entry["recon_slice"])
        axes[row, col].imshow(target_preview.T, cmap="gray", origin="lower")
        axes[row, col].set_title(
            f"{entry['patient_id']} {entry['modality']} GT\n"
            f"SSIM {entry['ssim']:.4f} PSNR {entry['psnr']:.2f}",
            fontsize=8,
        )
        axes[row, col + 1].imshow(recon_preview.T, cmap="gray", origin="lower")
        axes[row, col + 1].set_title(f"{entry['modality']} Recon", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _preview_modalities(requested, available_modalities):
    available = tuple(available_modalities)
    if requested is None:
        return available
    if isinstance(requested, str):
        if requested.lower() in {"all", "*"}:
            return available
        requested = (requested,)
    selected = []
    available_lookup = {name.lower(): name for name in available}
    for modality in requested:
        key = str(modality).lower()
        if key not in available_lookup:
            raise ValueError(f"Unknown preview modality '{modality}'. Available: {available}")
        selected.append(available_lookup[key])
    return tuple(selected)


def _compact_validation_metrics(metrics):
    keep = {
        "loss",
        "recon_loss",
        "slice_mse_loss",
        "mip_l1_loss",
        "vessel_l1_loss",
        "background_penalty_loss",
        "inside_dark_loss",
        "kl_loss",
        "kl_mu_term",
        "kl_var_term",
        "latent_alignment_loss",
        "ssim",
        "psnr",
        "steps",
        "preview_path",
        "preview_paths",
        "mip_preview_path",
        "mip_l1_3view",
        "mip_mse_3view",
        "mip_ssim",
        "mip_psnr",
        "mra_quality_score",
        "inside_mask_l1",
        "outside_mask_mean",
        "outside_mask_abs_mean",
        "mu_mean",
        "mu_std",
        "mu_min",
        "mu_max",
        "logvar_mean",
        "logvar_std",
        "logvar_min",
        "logvar_max",
        "std_mean",
        "std_std",
        "std_min",
        "std_max",
        "active_channels",
        "channel_var_min",
        "channel_var_mean",
        "channel_var_max",
        "mu_p0_1",
        "mu_p1",
        "mu_p50",
        "mu_p99",
        "mu_p99_9",
        "logvar_p0_1",
        "logvar_p1",
        "logvar_p50",
        "logvar_p99",
        "logvar_p99_9",
    }
    compact = {key: metrics.get(key) for key in keep if key in metrics}
    for key, value in metrics.items():
        if key.startswith((
            "l1_",
            "ssim_",
            "psnr_",
            "kl_mu_",
            "kl_var_",
            "active_channels_",
            "channel_var_min_",
            "channel_var_mean_",
            "channel_var_max_",
        )):
            compact[key] = value
    return compact


def _initial_best_metric(mode):
    return -math.inf if mode == "max" else math.inf


def _metric_is_better(value, best, mode):
    if mode == "max":
        return value > best
    if mode == "min":
        return value < best
    raise ValueError("best_metric_mode must be 'max' or 'min'")


def _make_single_modality_mask(modality_idx, num_modalities, device):
    mask = torch.zeros(1, num_modalities, dtype=torch.float32, device=device)
    mask[0, int(modality_idx)] = 1.0
    return mask


def _volume_tensor(volume, device):
    return torch.from_numpy(np.asarray(volume, dtype=np.float32)).unsqueeze(0).unsqueeze(0).to(device)


def _round_overlap_to_factor(tile_size, factor, fraction=0.5):
    overlap = []
    for dim in tile_size:
        dim = int(dim)
        raw = int(round(dim * float(fraction)))
        rounded = (raw // int(factor)) * int(factor)
        if rounded >= dim:
            rounded = ((dim - int(factor)) // int(factor)) * int(factor)
        overlap.append(max(0, int(rounded)))
    return tuple(overlap)


def _resolve_tile_overlap(tile_size, factor, overlap=None, fraction=0.5):
    tile_size = _as_3tuple(tile_size, "tile_size")
    if overlap is None:
        return _round_overlap_to_factor(tile_size, factor, fraction=fraction)
    overlap = _as_3tuple(overlap, "overlap")
    if any(value < 0 for value in overlap):
        raise ValueError(f"overlap must be non-negative, got {overlap}")
    if any(value >= dim for value, dim in zip(overlap, tile_size)):
        raise ValueError(f"overlap {overlap} must be smaller than tile_size {tile_size}")
    return overlap


def _blend_weight_1d(size, device, dtype, blend_mode="hann", blend_floor=1e-3):
    size = int(size)
    blend_mode = str(blend_mode).lower()
    if size <= 1 or blend_mode in {"uniform", "none", "flat"}:
        return torch.ones(size, device=device, dtype=dtype)
    if blend_mode == "hann":
        weight = torch.hann_window(size, periodic=False, device=device, dtype=dtype)
    elif blend_mode == "cosine":
        coords = torch.linspace(-1.0, 1.0, size, device=device, dtype=dtype)
        weight = torch.cos(coords * (math.pi / 2.0)).clamp_min(0.0)
    else:
        raise ValueError("blend_mode must be one of: 'hann', 'cosine', 'uniform'")
    weight = weight.clamp_min(float(blend_floor))
    return weight / weight.max().clamp_min(1e-8)


def _blend_weight_3d(shape, device, dtype, blend_mode="hann", blend_floor=1e-3):
    if str(blend_mode).lower() in {"uniform", "none", "flat"}:
        return torch.ones(tuple(int(value) for value in shape), device=device, dtype=dtype)
    wd = _blend_weight_1d(shape[0], device, dtype, blend_mode=blend_mode, blend_floor=blend_floor)
    wh = _blend_weight_1d(shape[1], device, dtype, blend_mode=blend_mode, blend_floor=blend_floor)
    ww = _blend_weight_1d(shape[2], device, dtype, blend_mode=blend_mode, blend_floor=blend_floor)
    return wd[:, None, None] * wh[None, :, None] * ww[None, None, :]


def _encode_deterministic(model, x, modality_mask=None):
    model._validate_spatial_shape(x)
    if hasattr(model, "encoder"):
        return model.encoder(x)

    indices = model._modality_indices(modality_mask, x.shape[0], x.device)
    mu_parts = [None] * x.shape[0]
    logvar_parts = [None] * x.shape[0]
    for modality_idx, encoder in enumerate(model.encoders):
        mask = indices == modality_idx
        if not mask.any():
            continue
        batch_indices = mask.nonzero(as_tuple=False).flatten()
        mu, logvar = encoder(x[batch_indices])
        for local_idx, batch_idx in enumerate(batch_indices.tolist()):
            mu_parts[batch_idx] = mu[local_idx:local_idx + 1]
            logvar_parts[batch_idx] = logvar[local_idx:local_idx + 1]
    return torch.cat(mu_parts, dim=0), torch.cat(logvar_parts, dim=0)


def _encode_tiled_deterministic(
    model,
    x,
    modality_mask=None,
    tile_size=None,
    overlap=None,
    blend_mode="hann",
    blend_floor=1e-3,
):
    tile_size = _as_3tuple(tile_size or model.patch_size, "tile_size")
    factor = model.downsampling_factor
    overlap = _resolve_tile_overlap(tile_size, factor, overlap=overlap)
    if any(v % factor != 0 for v in tile_size + overlap):
        raise ValueError("tile_size and overlap values must be divisible by downsampling factor")

    spatial = tuple(int(v) for v in x.shape[-3:])
    if any(dim % factor != 0 for dim in spatial):
        raise ValueError(f"Input spatial shape {spatial} must be divisible by {factor}")

    starts_d, starts_h, starts_w = model._tile_starts(spatial, tile_size, overlap)
    latent_shape = tuple(dim // factor for dim in spatial)
    mu_acc = logvar_acc = weight = None

    for d0 in starts_d:
        for h0 in starts_h:
            for w0 in starts_w:
                tile = x[
                    ...,
                    d0:d0 + tile_size[0],
                    h0:h0 + tile_size[1],
                    w0:w0 + tile_size[2],
                ]
                mu_tile, logvar_tile = _encode_deterministic(model, tile, modality_mask)
                if mu_acc is None:
                    out_shape = (x.shape[0], mu_tile.shape[1], *latent_shape)
                    mu_acc = torch.zeros(out_shape, device=x.device, dtype=mu_tile.dtype)
                    logvar_acc = torch.zeros_like(mu_acc)
                    weight = torch.zeros_like(mu_acc)

                ld0, lh0, lw0 = d0 // factor, h0 // factor, w0 // factor
                ld, lh, lw = mu_tile.shape[-3:]
                region = (..., slice(ld0, ld0 + ld), slice(lh0, lh0 + lh), slice(lw0, lw0 + lw))
                blend_weight = _blend_weight_3d(
                    (ld, lh, lw),
                    device=mu_tile.device,
                    dtype=mu_tile.dtype,
                    blend_mode=blend_mode,
                    blend_floor=blend_floor,
                ).view(1, 1, ld, lh, lw)
                mu_acc[region] += mu_tile * blend_weight
                logvar_acc[region] += logvar_tile * blend_weight
                weight[region] += blend_weight

    weight = weight.clamp_min(1e-8)
    return mu_acc / weight, logvar_acc / weight


def _reconstruct_volume_tiled(
    model,
    volume,
    device,
    modality_mask=None,
    tile_size=None,
    overlap=None,
    tile_overlap=None,
    blend_mode="hann",
    blend_floor=1e-3,
):
    x = _volume_tensor(volume, device)
    original_spatial = tuple(int(v) for v in x.shape[-3:])
    x_work, _ = model._pad_spatial_to_factor(x)
    effective_overlap = tile_overlap if tile_overlap is not None else overlap
    mu, logvar = _encode_tiled_deterministic(
        model,
        x_work,
        modality_mask=modality_mask,
        tile_size=tile_size,
        overlap=effective_overlap,
        blend_mode=blend_mode,
        blend_floor=blend_floor,
    )
    latent_overlap = None
    if effective_overlap is not None:
        factor = model.downsampling_factor
        latent_overlap = tuple(int(value) // factor for value in _as_3tuple(effective_overlap, "tile_overlap"))
    recon = model.decode_tiled(
        mu,
        modality_mask=modality_mask,
        latent_overlap=latent_overlap,
        blend_mode=blend_mode,
        blend_floor=blend_floor,
    )
    recon = model._crop_recon_to_spatial(recon, original_spatial)
    return recon, x, mu, logvar


def _as_3tuple(value, name):
    if isinstance(value, int):
        return (value, value, value)
    if len(value) != 3:
        raise ValueError(f"{name} must have 3 values, got {value}")
    return tuple(int(v) for v in value)


def _normalize(channels, num_groups=32):
    groups = min(num_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels, eps=1e-6)


def _volume_to_numpy(path, dtype=np.float32):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        volume = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        loaded = np.load(path, allow_pickle=False)
        key = "array" if "array" in loaded.files else loaded.files[0]
        volume = loaded[key]
    else:
        image = sitk.ReadImage(str(path))
        volume = sitk.GetArrayFromImage(image).transpose(2, 1, 0)
    return np.asarray(volume, dtype=dtype)


def _load_array_file(path, dtype=None, mmap_mode="r"):
    path = Path(path)
    if path.suffix.lower() == ".npz":
        loaded = np.load(path, allow_pickle=False)
        key = "array" if "array" in loaded.files else loaded.files[0]
        array = loaded[key]
    else:
        array = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
    if dtype is None:
        return array
    return np.asarray(array, dtype=dtype)


def _load_cached_array(path, dtype=None):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if dtype is not None and np.dtype(dtype) != array.dtype:
            return np.asarray(array, dtype=dtype)
        return array
    return _load_array_file(path, dtype=dtype, mmap_mode=None)


def _save_array_file(path, array, cache_format="npy"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if cache_format == "npy":
        np.save(path, array, allow_pickle=False)
    elif cache_format == "npz":
        np.savez_compressed(path, array=array)
    else:
        raise ValueError("cache_format must be 'npy' or 'npz'")


def _cache_path(cache_dir, cache_name, suffix, cache_format="npy"):
    extension = ".npy" if cache_format == "npy" else ".npz"
    return Path(cache_dir) / f"{cache_name}_{suffix}{extension}"


def _normalize_minmax(volume):
    volume = volume.astype(np.float32, copy=False)
    vmin = float(np.min(volume))
    vmax = float(np.max(volume))
    if vmax > vmin:
        volume = (volume - vmin) / (vmax - vmin)
    return volume.astype(np.float32, copy=False)


def _normalize_percentile(volume, low=1.0, high=99.0):
    volume = volume.astype(np.float32, copy=False)
    p_low = float(np.percentile(volume, low))
    p_high = float(np.percentile(volume, high))
    if p_high > p_low:
        volume = np.clip(volume, p_low, p_high)
        volume = (volume - p_low) / (p_high - p_low)
    return volume.astype(np.float32, copy=False)


def _normalize_volume(volume, mode):
    if mode == "minmax":
        return _normalize_minmax(volume)
    if mode == "percentile":
        return _normalize_percentile(volume)
    if mode in (None, "none"):
        return volume.astype(np.float32, copy=False)
    raise ValueError(f"Unknown normalize mode: {mode}")


def _strip_nii_extension(filename):
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    if filename.endswith(".nii"):
        return filename[:-4]
    return Path(filename).stem


def extract_ixi_patient_id(path, modality):
    name = _strip_nii_extension(Path(path).name)

    # IXI002-Guys-0828-T1_mask -> IXI002-Guys-0828-T1
    name = re.sub(r"[_-]mask$", "", name, flags=re.IGNORECASE)

    # IXI002-Guys-0828-T1 -> IXI002-Guys-0828
    # IXI002-Guys-0828-PD -> IXI002-Guys-0828
    # IXI002-Guys-0828-T2 -> IXI002-Guys-0828
    name = re.sub(rf"[-_]{re.escape(modality)}$", "", name, flags=re.IGNORECASE)

    return name


def _collect_modality_files(directory, modality, image_extensions=(".nii.gz", ".nii")):
    directory = Path(directory)
    files = defaultdict(list)

    # Accept:
    # IXI002-Guys-0828-T1.nii.gz / IXI002-Guys-0828-T1.npy
    # IXI002-Guys-0828-T1_mask.nii.gz / IXI002-Guys-0828-T1_mask.npy
    pattern = re.compile(
        rf"[-_]{re.escape(modality)}(?:[_-]mask)?(?:\.[a-z0-9]+(?:\.[a-z0-9]+)?)?$",
        re.IGNORECASE,
    )

    for path in directory.rglob("*"):
        if not path.is_file():
            continue

        if not any(path.name.endswith(ext) for ext in image_extensions):
            continue

        if not pattern.search(path.name):
            continue

        patient_id = extract_ixi_patient_id(path, modality)
        files[patient_id].append(path)

    return dict(files)


def _make_cache_name(name, target_shape, patch_size, normalize_mode, tag):
    shape_tag = "x".join(str(x) for x in target_shape)
    patch_tag = "x".join(str(x) for x in patch_size)
    return f"{name}_s{shape_tag}_p{patch_tag}_{normalize_mode}_{tag}"


def _ensure_shape(volume, target_shape, name):
    if tuple(volume.shape) != tuple(target_shape):
        raise ValueError(f"[{name}] Expected shape {target_shape}, got {tuple(volume.shape)}")


def _extract_patch(volume, start, patch_size):
    d0, h0, w0 = (int(v) for v in start)
    pd, ph, pw = patch_size
    return volume[d0:d0 + pd, h0:h0 + ph, w0:w0 + pw].copy()


def _valid_random_start(shape, patch_size):
    return tuple(
        np.random.randint(0, dim - patch + 1) if dim > patch else 0
        for dim, patch in zip(shape, patch_size)
    )


def _center_to_start(center, shape, patch_size):
    return tuple(
        int(np.clip(int(c) - patch // 2, 0, dim - patch))
        for c, dim, patch in zip(center, shape, patch_size)
    )


def _one_hot(index, num_classes):
    mask = torch.zeros(num_classes, dtype=torch.float32)
    mask[int(index)] = 1.0
    return mask


class ConvBlock3D(nn.Module):
    """Plain convolutional block without residual or attention shortcuts."""

    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            _normalize(out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            _normalize(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Downsample3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class VAEv2Encoder3D(nn.Module):
    def __init__(
        self,
        in_channels=1,
        base_channels=32,
        channel_multipliers=(1, 2, 4),
        latent_channels=8,
        dropout=0.0,
        blocks_per_level=1,
        logvar_clamp=(-10.0, 5.0),
    ):
        super().__init__()
        channels = [base_channels * mult for mult in channel_multipliers]
        self.downsampling_factor = 2 ** max(0, len(channels) - 1)
        self.logvar_clamp = _as_clamp_bounds(logvar_clamp, "logvar_clamp")

        self.conv_in = nn.Conv3d(in_channels, channels[0], 3, padding=1)
        self.blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        in_ch = channels[0]
        for level, out_ch in enumerate(channels):
            level_blocks = []
            for block_idx in range(blocks_per_level):
                block_in = in_ch if block_idx == 0 else out_ch
                level_blocks.append(ConvBlock3D(block_in, out_ch, dropout=dropout))
            self.blocks.append(nn.Sequential(*level_blocks))
            self.downsamples.append(Downsample3D(out_ch) if level < len(channels) - 1 else nn.Identity())
            in_ch = out_ch

        self.mid = ConvBlock3D(channels[-1], channels[-1], dropout=dropout)
        self.norm_out = _normalize(channels[-1])
        self.conv_out = nn.Conv3d(channels[-1], 2 * latent_channels, 1)

    def forward(self, x):
        h = self.conv_in(x)
        for block, downsample in zip(self.blocks, self.downsamples):
            h = block(h)
            h = downsample(h)
        h = self.mid(h)
        h = self.conv_out(F.silu(self.norm_out(h)))
        mu, logvar = h.chunk(2, dim=1)
        return mu, _optional_clamp(logvar, self.logvar_clamp)


class VAEv2Decoder3D(nn.Module):
    def __init__(
        self,
        out_channels=1,
        base_channels=32,
        channel_multipliers=(1, 2, 4),
        latent_channels=8,
        dropout=0.0,
        blocks_per_level=1,
        output_activation="sigmoid",
    ):
        super().__init__()
        channels = [base_channels * mult for mult in channel_multipliers]
        self.downsampling_factor = 2 ** max(0, len(channels) - 1)
        self.output_activation = output_activation

        self.conv_in = nn.Conv3d(latent_channels, channels[-1], 3, padding=1)
        self.mid = ConvBlock3D(channels[-1], channels[-1], dropout=dropout)

        self.blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        in_ch = channels[-1]
        for level in reversed(range(len(channels))):
            out_ch = channels[level]
            level_blocks = []
            for block_idx in range(blocks_per_level):
                block_in = in_ch if block_idx == 0 else out_ch
                level_blocks.append(ConvBlock3D(block_in, out_ch, dropout=dropout))
            self.blocks.append(nn.Sequential(*level_blocks))
            self.upsamples.append(Upsample3D(out_ch) if level > 0 else nn.Identity())
            in_ch = out_ch

        self.norm_out = _normalize(channels[0])
        self.conv_out = nn.Conv3d(channels[0], out_channels, 3, padding=1)

    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid(h)
        for block, upsample in zip(self.blocks, self.upsamples):
            h = block(h)
            h = upsample(h)
        h = self.conv_out(F.silu(self.norm_out(h)))
        if self.output_activation == "sigmoid":
            return torch.sigmoid(h)
        if self.output_activation in (None, "none"):
            return h
        raise ValueError(f"Unknown output_activation: {self.output_activation}")


class _VAEv2TilingMixin:
    @property
    def downsampling_factor(self):
        return self._downsampling_factor

    def _validate_spatial_shape(self, x):
        spatial = tuple(int(v) for v in x.shape[-3:])
        factor = self.downsampling_factor
        if any(dim % factor != 0 for dim in spatial):
            raise ValueError(
                f"Input spatial shape {spatial} must be divisible by downsampling factor {factor}"
            )

    def _pad_spatial_to_factor(self, x):
        spatial = tuple(int(v) for v in x.shape[-3:])
        factor = self.downsampling_factor
        pad_d = (factor - spatial[0] % factor) % factor
        pad_h = (factor - spatial[1] % factor) % factor
        pad_w = (factor - spatial[2] % factor) % factor
        if pad_d == 0 and pad_h == 0 and pad_w == 0:
            return x, False
        return F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d)), True

    def _crop_recon_to_spatial(self, recon, spatial):
        return recon[..., :spatial[0], :spatial[1], :spatial[2]]

    def _crop_forward_result(self, result, spatial, was_padded):
        if not was_padded:
            return result
        recon, mu, logvar, z = result
        return self._crop_recon_to_spatial(recon, spatial), mu, logvar, z

    def _default_overlap(self, tile_size):
        factor = self.downsampling_factor
        return _resolve_tile_overlap(tile_size, factor, overlap=None)

    def _tile_starts(self, shape, tile_size, overlap):
        starts_by_dim = []
        for dim, tile, ov in zip(shape, tile_size, overlap):
            if tile > dim:
                raise ValueError(f"Tile size {tile_size} exceeds input shape {shape}")
            stride = tile - ov
            if stride <= 0:
                raise ValueError(f"Overlap {overlap} must be smaller than tile size {tile_size}")
            starts = list(range(0, max(dim - tile, 0) + 1, stride))
            if starts[-1] != dim - tile:
                starts.append(dim - tile)
            starts_by_dim.append(starts)
        return starts_by_dim

    def encode_tiled(
        self,
        x,
        modality_mask=None,
        tile_size=None,
        overlap=None,
        tile_overlap=None,
        blend_mode="hann",
        blend_floor=1e-3,
    ):
        tile_size = _as_3tuple(tile_size or self.patch_size, "tile_size")
        factor = self.downsampling_factor
        effective_overlap = tile_overlap if tile_overlap is not None else overlap
        overlap = _resolve_tile_overlap(tile_size, factor, overlap=effective_overlap)
        if any(v % factor != 0 for v in tile_size + overlap):
            raise ValueError("tile_size and overlap values must be divisible by downsampling factor")

        spatial = tuple(int(v) for v in x.shape[-3:])
        if any(dim % factor != 0 for dim in spatial):
            raise ValueError(f"Input spatial shape {spatial} must be divisible by {factor}")

        starts_d, starts_h, starts_w = self._tile_starts(spatial, tile_size, overlap)
        latent_shape = tuple(dim // factor for dim in spatial)
        z_acc = mu_acc = logvar_acc = weight = None

        for d0 in starts_d:
            for h0 in starts_h:
                for w0 in starts_w:
                    tile = x[
                        ...,
                        d0:d0 + tile_size[0],
                        h0:h0 + tile_size[1],
                        w0:w0 + tile_size[2],
                    ]
                    z_tile, mu_tile, logvar_tile = self.encode(tile, modality_mask)
                    if z_acc is None:
                        out_shape = (x.shape[0], z_tile.shape[1], *latent_shape)
                        z_acc = torch.zeros(out_shape, device=x.device, dtype=z_tile.dtype)
                        mu_acc = torch.zeros_like(z_acc)
                        logvar_acc = torch.zeros_like(z_acc)
                        weight = torch.zeros_like(z_acc)

                    ld0, lh0, lw0 = d0 // factor, h0 // factor, w0 // factor
                    ld, lh, lw = z_tile.shape[-3:]
                    region = (..., slice(ld0, ld0 + ld), slice(lh0, lh0 + lh), slice(lw0, lw0 + lw))
                    blend_weight = _blend_weight_3d(
                        (ld, lh, lw),
                        device=z_tile.device,
                        dtype=z_tile.dtype,
                        blend_mode=blend_mode,
                        blend_floor=blend_floor,
                    ).view(1, 1, ld, lh, lw)
                    z_acc[region] += z_tile * blend_weight
                    mu_acc[region] += mu_tile * blend_weight
                    logvar_acc[region] += logvar_tile * blend_weight
                    weight[region] += blend_weight

        weight = weight.clamp_min(1e-8)
        return z_acc / weight, mu_acc / weight, logvar_acc / weight

    def decode_tiled(
        self,
        z,
        modality_mask=None,
        latent_tile_size=None,
        latent_overlap=None,
        blend_mode="hann",
        blend_floor=1e-3,
    ):
        factor = self.downsampling_factor
        if latent_tile_size is None:
            latent_tile_size = tuple(dim // factor for dim in self.patch_size)
        latent_tile_size = _as_3tuple(latent_tile_size, "latent_tile_size")
        if latent_overlap is None:
            latent_overlap = _round_overlap_to_factor(latent_tile_size, factor=1, fraction=0.5)
        latent_overlap = _as_3tuple(latent_overlap, "latent_overlap")

        latent_spatial = tuple(int(v) for v in z.shape[-3:])
        starts_d, starts_h, starts_w = self._tile_starts(latent_spatial, latent_tile_size, latent_overlap)
        recon_acc = weight = None

        for d0 in starts_d:
            for h0 in starts_h:
                for w0 in starts_w:
                    z_tile = z[
                        ...,
                        d0:d0 + latent_tile_size[0],
                        h0:h0 + latent_tile_size[1],
                        w0:w0 + latent_tile_size[2],
                    ]
                    recon_tile = self.decode(z_tile, modality_mask)
                    if recon_acc is None:
                        recon_shape = (
                            z.shape[0],
                            recon_tile.shape[1],
                            latent_spatial[0] * factor,
                            latent_spatial[1] * factor,
                            latent_spatial[2] * factor,
                        )
                        recon_acc = torch.zeros(recon_shape, device=z.device, dtype=recon_tile.dtype)
                        weight = torch.zeros_like(recon_acc)

                    rd0, rh0, rw0 = d0 * factor, h0 * factor, w0 * factor
                    rd, rh, rw = recon_tile.shape[-3:]
                    region = (..., slice(rd0, rd0 + rd), slice(rh0, rh0 + rh), slice(rw0, rw0 + rw))
                    blend_weight = _blend_weight_3d(
                        (rd, rh, rw),
                        device=recon_tile.device,
                        dtype=recon_tile.dtype,
                        blend_mode=blend_mode,
                        blend_floor=blend_floor,
                    ).view(1, 1, rd, rh, rw)
                    recon_acc[region] += recon_tile * blend_weight
                    weight[region] += blend_weight

        return recon_acc / weight.clamp_min(1e-8)

    @torch.no_grad()
    def forward_full_volume(
        self,
        x,
        modality_mask=None,
        use_tiling_fallback=True,
        tile_size=None,
        overlap=None,
        tile_overlap=None,
        blend_mode="hann",
        blend_floor=1e-3,
    ):
        original_spatial = tuple(int(v) for v in x.shape[-3:])
        x_work, was_padded = self._pad_spatial_to_factor(x)
        try:
            self._validate_spatial_shape(x_work)
            result = self.forward(x_work, modality_mask) if modality_mask is not None else self.forward(x_work)
            return self._crop_forward_result(result, original_spatial, was_padded)
        except RuntimeError as error:
            if not use_tiling_fallback or "out of memory" not in str(error).lower():
                raise
            if x.is_cuda:
                torch.cuda.empty_cache()
        except ValueError:
            if not use_tiling_fallback:
                raise

        effective_overlap = tile_overlap if tile_overlap is not None else overlap
        z, mu, logvar = self.encode_tiled(
            x_work,
            modality_mask,
            tile_size=tile_size,
            overlap=effective_overlap,
            blend_mode=blend_mode,
            blend_floor=blend_floor,
        )
        latent_overlap = None
        if effective_overlap is not None:
            latent_overlap = tuple(int(value) // self.downsampling_factor for value in _as_3tuple(effective_overlap, "tile_overlap"))
        recon = self.decode_tiled(
            z,
            modality_mask,
            latent_overlap=latent_overlap,
            blend_mode=blend_mode,
            blend_floor=blend_floor,
        )
        recon = self._crop_recon_to_spatial(recon, original_spatial)
        return recon, mu, logvar, z


class VAEv2Multimodal(nn.Module, _VAEv2TilingMixin):
    def __init__(
        self,
        patch_size=(192, 192, 64),
        modalities=("T1", "T2", "PD"),
        in_channels=1,
        base_channels=32,
        channel_multipliers=(1, 2, 4),
        latent_channels=8,
        dropout=0.0,
        blocks_per_level=1,
        output_activation="sigmoid",
    ):
        super().__init__()
        self.patch_size = _as_3tuple(patch_size, "patch_size")
        self.modalities = tuple(modalities)
        self.modality_to_index = {name: idx for idx, name in enumerate(self.modalities)}
        self.latent_channels = latent_channels

        self.encoders = nn.ModuleList(
            [
                VAEv2Encoder3D(
                    in_channels=in_channels,
                    base_channels=base_channels,
                    channel_multipliers=channel_multipliers,
                    latent_channels=latent_channels,
                    dropout=dropout,
                    blocks_per_level=blocks_per_level,
                )
                for _ in self.modalities
            ]
        )
        self.decoders = nn.ModuleList(
            [
                VAEv2Decoder3D(
                    out_channels=in_channels,
                    base_channels=base_channels,
                    channel_multipliers=channel_multipliers,
                    latent_channels=latent_channels,
                    dropout=dropout,
                    blocks_per_level=blocks_per_level,
                    output_activation=output_activation,
                )
                for _ in self.modalities
            ]
        )
        self._downsampling_factor = self.encoders[0].downsampling_factor
        if any(dim % self.downsampling_factor != 0 for dim in self.patch_size):
            raise ValueError(
                f"patch_size {self.patch_size} must be divisible by {self.downsampling_factor}"
            )

    def _modality_indices(self, modality_mask, batch_size, device):
        if modality_mask is None:
            raise ValueError("VAEv2Multimodal requires modality_mask")
        if not torch.is_tensor(modality_mask):
            modality_mask = torch.as_tensor(modality_mask, device=device)
        modality_mask = modality_mask.to(device)
        if modality_mask.dim() == 1:
            if modality_mask.numel() == len(self.modalities) and torch.isclose(
                modality_mask.float().sum(),
                torch.ones((), device=device),
            ):
                indices = modality_mask.float().argmax(dim=0).repeat(batch_size).long()
            elif modality_mask.numel() == batch_size and modality_mask.dtype in (
                torch.long,
                torch.int64,
                torch.int32,
            ):
                indices = modality_mask.long()
            else:
                raise ValueError(f"Invalid modality_mask shape: {tuple(modality_mask.shape)}")
        elif modality_mask.dim() == 2:
            if modality_mask.shape != (batch_size, len(self.modalities)):
                raise ValueError(
                    f"Expected modality_mask shape {(batch_size, len(self.modalities))}, "
                    f"got {tuple(modality_mask.shape)}"
                )
            indices = modality_mask.float().argmax(dim=1).long()
        else:
            raise ValueError(f"Invalid modality_mask shape: {tuple(modality_mask.shape)}")
        if indices.min() < 0 or indices.max() >= len(self.modalities):
            raise ValueError(f"Unknown modality index in {indices.detach().cpu().tolist()}")
        return indices

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def encode(self, x, modality_mask=None):
        self._validate_spatial_shape(x)
        indices = self._modality_indices(modality_mask, x.shape[0], x.device)
        z_parts = [None] * x.shape[0]
        mu_parts = [None] * x.shape[0]
        logvar_parts = [None] * x.shape[0]

        for modality_idx, encoder in enumerate(self.encoders):
            mask = indices == modality_idx
            if not mask.any():
                continue
            batch_indices = mask.nonzero(as_tuple=False).flatten()
            mu, logvar = encoder(x[batch_indices])
            z = self.reparameterize(mu, logvar)
            for local_idx, batch_idx in enumerate(batch_indices.tolist()):
                z_parts[batch_idx] = z[local_idx:local_idx + 1]
                mu_parts[batch_idx] = mu[local_idx:local_idx + 1]
                logvar_parts[batch_idx] = logvar[local_idx:local_idx + 1]

        return torch.cat(z_parts, dim=0), torch.cat(mu_parts, dim=0), torch.cat(logvar_parts, dim=0)

    def decode(self, z, modality_mask=None):
        indices = self._modality_indices(modality_mask, z.shape[0], z.device)
        recon_parts = [None] * z.shape[0]

        for modality_idx, decoder in enumerate(self.decoders):
            mask = indices == modality_idx
            if not mask.any():
                continue
            batch_indices = mask.nonzero(as_tuple=False).flatten()
            recon = decoder(z[batch_indices])
            for local_idx, batch_idx in enumerate(batch_indices.tolist()):
                recon_parts[batch_idx] = recon[local_idx:local_idx + 1]

        return torch.cat(recon_parts, dim=0)

    def forward(self, x, modality_mask):
        z, mu, logvar = self.encode(x, modality_mask)
        recon = self.decode(z, modality_mask)
        return recon, mu, logvar, z


class VAEv2PairedMultimodalPatchDataset(Dataset):
    def __init__(
        self,
        image_dirs: Mapping[str, str],
        mask_dirs: Mapping[str, str],
        modalities=("T1", "T2", "PD"),
        patch_size=(192, 192, 64),
        target_shape=(512, 512, 128),
        patches_per_patient=64,
        brain_patch_ratio=0.7,
        cache_dir="Dataset/cache/vaev2_multimodal_patchdataset",
        rebuild_cache=False,
        n=-1,
        image_extensions=(".nii.gz", ".nii"),
        normalize_mode="percentile",
        mask_threshold=0.5,
        ram_cache_size=4,
        cache_dtype=np.float32,
        cache_format="npy",
    ):
        self.modalities = tuple(modalities)
        self.patch_size = _as_3tuple(patch_size, "patch_size")
        self.target_shape = _as_3tuple(target_shape, "target_shape")
        self.patches_per_patient = int(patches_per_patient)
        self.brain_patch_ratio = float(brain_patch_ratio)
        self.brain_slots_per_patient = int(round(self.patches_per_patient * self.brain_patch_ratio))
        self.normalize_mode = normalize_mode
        self.mask_threshold = float(mask_threshold)
        self.ram_cache_size = int(ram_cache_size)
        self.ram_cache = OrderedDict()
        self.cache_dtype = np.dtype(cache_dtype)
        self.cache_format = str(cache_format).lower()
        if self.cache_format not in {"npy", "npz"}:
            raise ValueError("cache_format must be 'npy' or 'npz'")
        self.sample_order = None
        self.sample_order = None

        if not 0.0 <= self.brain_patch_ratio <= 1.0:
            raise ValueError("brain_patch_ratio must be between 0 and 1")

        self.image_dirs = {m: Path(image_dirs[m]) for m in self.modalities}
        self.mask_dirs = {m: Path(mask_dirs[m]) for m in self.modalities}
        for m in self.modalities:
            if not self.image_dirs[m].is_dir():
                raise NotADirectoryError(f"Missing image directory for {m}: {self.image_dirs[m]}")
            if not self.mask_dirs[m].is_dir():
                raise NotADirectoryError(f"Missing mask directory for {m}: {self.mask_dirs[m]}")

        image_files = {
            m: _collect_modality_files(self.image_dirs[m], m, image_extensions=image_extensions)
            for m in self.modalities
        }
        mask_files = {
            m: _collect_modality_files(self.mask_dirs[m], m, image_extensions=image_extensions)
            for m in self.modalities
        }
        patients = sorted(set.intersection(*(set(image_files[m]) for m in self.modalities)))
        patients = [p for p in patients if all(p in mask_files[m] for m in self.modalities)]
        if n != -1 and n is not None:
            patients = patients[:n]
        if not patients:
            raise ValueError("No complete T1/T2/PD patients with matching masks were found")

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.samples = []
        for patient in patients:
            sample = {"patient": patient, "vol_files": {}, "union_mask_file": None, "brain_coords_file": None}
            union_mask = None
            for modality in self.modalities:
                image_path = image_files[modality][patient][0]
                mask_path = mask_files[modality][patient][0]
                cache_name = _make_cache_name(
                    f"{patient}_{modality}",
                    self.target_shape,
                    self.patch_size,
                    normalize_mode,
                    "paired",
                )
                vol_file = _cache_path(self.cache_dir, cache_name, "vol", self.cache_format)
                mask_file = _cache_path(self.cache_dir, cache_name, "mask", self.cache_format)
                if rebuild_cache or not vol_file.exists() or not mask_file.exists():
                    volume = _volume_to_numpy(image_path, dtype=np.float32)
                    mask = _volume_to_numpy(mask_path, dtype=np.float32) > self.mask_threshold
                    _ensure_shape(volume, self.target_shape, image_path.name)
                    _ensure_shape(mask, self.target_shape, mask_path.name)
                    volume = _normalize_volume(volume, normalize_mode).astype(self.cache_dtype, copy=False)
                    _save_array_file(vol_file, volume, cache_format=self.cache_format)
                    _save_array_file(mask_file, mask.astype(np.bool_), cache_format=self.cache_format)
                    del volume, mask
                sample["vol_files"][modality] = str(vol_file)
                mask = _load_array_file(mask_file, dtype=np.bool_, mmap_mode=None)
                union_mask = mask if union_mask is None else np.logical_or(union_mask, mask)

            union_name = _make_cache_name(patient, self.target_shape, self.patch_size, normalize_mode, "union")
            union_mask_file = _cache_path(self.cache_dir, union_name, "mask", self.cache_format)
            brain_coords_file = _cache_path(self.cache_dir, union_name, "brain_coords", self.cache_format)
            if rebuild_cache or not union_mask_file.exists() or not brain_coords_file.exists():
                brain_coords = np.argwhere(union_mask).astype(np.int32)
                _save_array_file(union_mask_file, union_mask.astype(np.bool_), cache_format=self.cache_format)
                _save_array_file(brain_coords_file, brain_coords, cache_format=self.cache_format)
                del brain_coords

            sample["union_mask_file"] = str(union_mask_file)
            sample["brain_coords_file"] = str(brain_coords_file)
            self.samples.append(sample)

        self.sample_order = np.arange(len(self.samples), dtype=np.int64)
        print(f"VAEv2 paired multimodal dataset ready: {len(self.samples)} patients")
        print(
            f"Patches per patient: {self.patches_per_patient} "
            f"({self.brain_slots_per_patient} brain / "
            f"{self.patches_per_patient - self.brain_slots_per_patient} background)"
        )

    def __len__(self):
        return len(self.samples) * self.patches_per_patient

    def shuffle_samples(self, seed=None):
        rng = np.random.default_rng(seed)
        self.sample_order = rng.permutation(len(self.samples)).astype(np.int64, copy=False)

    def _cache_key(self, sample):
        return sample["patient"]

    def _get_cached_sample(self, sample):
        key = self._cache_key(sample)
        if key in self.ram_cache:
            self.ram_cache.move_to_end(key)
            return self.ram_cache[key]

        volumes = {
            modality: _load_cached_array(sample["vol_files"][modality])
            for modality in self.modalities
        }
        union_mask = _load_cached_array(sample["union_mask_file"])
        brain_coords = _load_cached_array(sample["brain_coords_file"])
        self.ram_cache[key] = (volumes, union_mask, brain_coords)
        self.ram_cache.move_to_end(key)
        if len(self.ram_cache) > self.ram_cache_size:
            self.ram_cache.popitem(last=False)
        return volumes, union_mask, brain_coords

    def _sample_brain_start(self, brain_coords, shape):
        if len(brain_coords) == 0:
            return _valid_random_start(shape, self.patch_size)
        center = brain_coords[np.random.randint(len(brain_coords))]
        return _center_to_start(center, shape, self.patch_size)

    def _sample_background_start(self, union_mask, shape):
        for _ in range(32):
            start = _valid_random_start(shape, self.patch_size)
            center = tuple(start[i] + self.patch_size[i] // 2 for i in range(3))
            if not union_mask[center]:
                return start
        return _valid_random_start(shape, self.patch_size)

    def __getitem__(self, idx):
        patient_idx = idx // self.patches_per_patient
        patch_slot = idx % self.patches_per_patient
        if self.sample_order is not None:
            patient_idx = int(self.sample_order[patient_idx])
        sample = self.samples[patient_idx]
        volumes, union_mask, brain_coords = self._get_cached_sample(sample)
        use_brain = patch_slot < self.brain_slots_per_patient
        if use_brain:
            start = self._sample_brain_start(brain_coords, self.target_shape)
        else:
            start = self._sample_background_start(union_mask, self.target_shape)

        patches = []
        modality_masks = []
        for modality_idx, modality in enumerate(self.modalities):
            patch = _extract_patch(volumes[modality], start, self.patch_size).astype(np.float32, copy=False)
            patches.append(torch.from_numpy(patch).unsqueeze(0))
            modality_masks.append(_one_hot(modality_idx, len(self.modalities)))
        return torch.stack(patches, dim=0), torch.stack(modality_masks, dim=0)

    def clear_ram_cache(self):
        self.ram_cache.clear()
        gc.collect()


class VAEv2Loss(nn.Module):
    def __init__(
        self,
        kl_weight=1e-6,
        latent_alignment_weight=0.1,
        slice_mse_weight=0.0,
        mip_l1_weight=0.0,
        vessel_l1_weight=0.0,
        vessel_percentile=95.0,
        vessel_weight=5.0,
        background_penalty_weight=0.0,
        background_threshold=0.05,
        inside_dark_loss_weight=0.0,
        inside_dark_margin=0.03,
        mip_axes=(2, 3, 4),
        kl_mu_clamp=(-20.0, 20.0),
        kl_logvar_clamp=(-30.0, 20.0),
    ):
        super().__init__()
        self.kl_weight = float(kl_weight)
        self.latent_alignment_weight = float(latent_alignment_weight)
        self.slice_mse_weight = float(slice_mse_weight)
        self.mip_l1_weight = float(mip_l1_weight)
        self.vessel_l1_weight = float(vessel_l1_weight)
        self.vessel_percentile = float(vessel_percentile)
        self.vessel_weight = float(vessel_weight)
        self.background_penalty_weight = float(background_penalty_weight)
        self.background_threshold = float(background_threshold)
        self.inside_dark_loss_weight = float(inside_dark_loss_weight)
        self.inside_dark_margin = float(inside_dark_margin)
        self.mip_axes = tuple(int(axis) for axis in mip_axes)
        self.kl_mu_clamp = _as_clamp_bounds(kl_mu_clamp, "kl_mu_clamp")
        self.kl_logvar_clamp = _as_clamp_bounds(kl_logvar_clamp, "kl_logvar_clamp")

    def forward(self, recon, target, mu, logvar, paired_mu=None, mask=None):
        recon_loss = F.l1_loss(recon, target)
        kl_mu_term, kl_var_term, kl_loss = _kl_decomposition(
            mu,
            logvar,
            mu_clamp=self.kl_mu_clamp,
            logvar_clamp=self.kl_logvar_clamp,
        )
        slice_mse_loss = recon_loss.new_zeros(())
        if self.slice_mse_weight > 0:
            slice_mse_loss = _central_slice_mse_loss(recon, target)
        mip_l1_loss = recon_loss.new_zeros(())
        if self.mip_l1_weight > 0:
            mip_l1_loss = _mip_l1_loss(recon, target, axes=self.mip_axes)
        vessel_l1_loss = recon_loss.new_zeros(())
        if self.vessel_l1_weight > 0:
            vessel_l1_loss = _vessel_weighted_l1_loss(
                recon,
                target,
                vessel_percentile=self.vessel_percentile,
                vessel_weight=self.vessel_weight,
            )
        background_penalty_loss = recon_loss.new_zeros(())
        if self.background_penalty_weight > 0:
            background_penalty_loss = _background_false_positive_loss(
                recon,
                target,
                mask=mask,
                background_threshold=self.background_threshold,
            )
        inside_dark_loss = recon_loss.new_zeros(())
        if self.inside_dark_loss_weight > 0:
            inside_dark_loss = _inside_dark_false_negative_loss(
                recon,
                target,
                mask=mask,
                dark_margin=self.inside_dark_margin,
            )
        alignment_loss = torch.zeros((), device=recon.device, dtype=recon.dtype)
        if paired_mu is not None:
            if paired_mu.dim() < 3:
                raise ValueError("paired_mu must have shape [B, modalities, ...]")
            pairs = []
            num_modalities = paired_mu.shape[1]
            for i in range(num_modalities):
                for j in range(i + 1, num_modalities):
                    pairs.append(F.mse_loss(paired_mu[:, i], paired_mu[:, j]))
            if pairs:
                alignment_loss = torch.stack(pairs).mean()
        total = (
            recon_loss
            + self.slice_mse_weight * slice_mse_loss
            + self.mip_l1_weight * mip_l1_loss
            + self.vessel_l1_weight * vessel_l1_loss
            + self.background_penalty_weight * background_penalty_loss
            + self.inside_dark_loss_weight * inside_dark_loss
            + self.kl_weight * kl_loss
            + self.latent_alignment_weight * alignment_loss
        )
        return {
            "loss": total,
            "recon_loss": recon_loss,
            "slice_mse_loss": slice_mse_loss,
            "mip_l1_loss": mip_l1_loss,
            "vessel_l1_loss": vessel_l1_loss,
            "background_penalty_loss": background_penalty_loss,
            "inside_dark_loss": inside_dark_loss,
            "kl_loss": kl_loss,
            "kl_mu_term": kl_mu_term,
            "kl_var_term": kl_var_term,
            "latent_alignment_loss": alignment_loss,
        }


def save_vaev2_checkpoint(path, model, optimizer, scheduler, epoch, best_metric, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "extra": extra or {},
    }
    torch.save(checkpoint, path)


def load_vaev2_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


@torch.no_grad()
def validate_vaev2_multimodal(
    model,
    dataset,
    criterion,
    batch_size=1,
    device="cuda",
    max_batches=None,
    show_progress=True,
    num_workers=0,
    metric_data_range=None,
    ssim_window_size=None,
    preview_dir=None,
    preview_epoch=None,
    preview_count=16,
    preview_required_scanners=("Guys", "HH", "IOP"),
    preview_modality=None,
    tile_overlap=None,
    blend_mode="hann",
    blend_floor=1e-3,
):
    del batch_size, num_workers
    model.eval()
    totals = {
        "loss": 0.0,
        "recon_loss": 0.0,
        "slice_mse_loss": 0.0,
        "mip_l1_loss": 0.0,
        "vessel_l1_loss": 0.0,
        "background_penalty_loss": 0.0,
        "inside_dark_loss": 0.0,
        "kl_loss": 0.0,
        "kl_mu_term": 0.0,
        "kl_var_term": 0.0,
        "latent_alignment_loss": 0.0,
    }
    per_modality_l1 = {m: [] for m in model.modalities}
    per_modality_ssim = {m: [] for m in model.modalities}
    per_modality_psnr = {m: [] for m in model.modalities}
    per_modality_kl_mu = {m: [] for m in model.modalities}
    per_modality_kl_var = {m: [] for m in model.modalities}
    per_modality_extra = {m: _init_latent_extra_stats() for m in model.modalities}
    ssim_values = []
    psnr_values = []
    latent_acc = _init_latent_stats()
    latent_extra_acc = _init_latent_extra_stats()
    preview_modalities = _preview_modalities(preview_modality, model.modalities)
    preview_entries = {modality: [] for modality in preview_modalities}
    steps = 0

    total = len(dataset.samples) if max_batches is None else min(len(dataset.samples), int(max_batches))
    progress = _progress_bar(dataset.samples[:total], enabled=show_progress, total=total, desc="val multimodal volumes", leave=False)
    for sample in progress:
        patient_id = _patient_id_from_sample(sample)
        scanner = _scanner_from_patient(patient_id)
        volumes, _, _ = dataset._get_cached_sample(sample)
        recons = []
        targets = []
        mus = []
        logvars = []

        for modality_idx, modality in enumerate(model.modalities):
            modality_mask = _make_single_modality_mask(modality_idx, len(model.modalities), device)
            recon, target, mu, logvar = _reconstruct_volume_tiled(
                model,
                volumes[modality],
                device=device,
                modality_mask=modality_mask,
                tile_size=model.patch_size,
                tile_overlap=tile_overlap,
                blend_mode=blend_mode,
                blend_floor=blend_floor,
            )
            modality_l1 = float(F.l1_loss(recon, target).detach().cpu())
            modality_metrics = _reconstruction_metrics(
                recon,
                target,
                data_range=metric_data_range,
                ssim_window_size=ssim_window_size,
            )
            per_modality_l1[modality].append(modality_l1)
            per_modality_ssim[modality].append(modality_metrics["ssim"])
            per_modality_psnr[modality].append(modality_metrics["psnr"])
            _update_latent_extra_stats(per_modality_extra[modality], mu, logvar, include_percentiles=False)
            kl_mu, kl_var, _ = _kl_decomposition(mu, logvar)
            per_modality_kl_mu[modality].append(float(kl_mu.detach().cpu()))
            per_modality_kl_var[modality].append(float(kl_var.detach().cpu()))
            if preview_dir is not None and modality in preview_entries:
                preview_entries[modality].append(
                    {
                        "patient_id": patient_id,
                        "scanner": scanner,
                        "modality": modality,
                        "ssim": modality_metrics["ssim"],
                        "psnr": modality_metrics["psnr"],
                        "target_slice": _slice_preview(target[0, 0].detach().cpu().numpy()),
                        "recon_slice": _slice_preview(recon[0, 0].detach().cpu().numpy()),
                    }
                )
            recons.append(recon)
            targets.append(target)
            mus.append(mu)
            logvars.append(logvar)

        recon_all = torch.cat(recons, dim=0)
        target_all = torch.cat(targets, dim=0)
        mu_all = torch.cat(mus, dim=0)
        logvar_all = torch.cat(logvars, dim=0)
        paired_mu = mu_all.unsqueeze(0)
        _update_latent_stats(latent_acc, mu_all, logvar_all)
        _update_latent_extra_stats(latent_extra_acc, mu_all, logvar_all)
        losses = criterion(recon_all, target_all, mu_all, logvar_all, paired_mu=paired_mu)
        for key in totals:
            totals[key] += float(losses[key].detach().cpu())
        metrics = _reconstruction_metrics(
            recon_all,
            target_all,
            data_range=metric_data_range,
            ssim_window_size=ssim_window_size,
        )
        ssim_values.append(metrics["ssim"])
        psnr_values.append(metrics["psnr"])
        steps += 1
        _set_progress_postfix(
            progress,
            {
                "loss": f"{totals['loss'] / steps:.4f}",
                "ssim": f"{sum(ssim_values) / len(ssim_values):.4f}",
                "psnr": f"{sum(psnr_values) / len(psnr_values):.2f}",
                **_cuda_memory_gb(device),
            },
        )
        del recons, targets, mus, logvars, recon_all, target_all, mu_all, logvar_all, paired_mu

    if steps == 0:
        return {**totals, "steps": 0}
    metrics = {key: value / steps for key, value in totals.items()}
    metrics["ssim"] = sum(ssim_values) / len(ssim_values) if ssim_values else float("nan")
    metrics["psnr"] = sum(psnr_values) / len(psnr_values) if psnr_values else float("nan")
    metrics.update(_finalize_latent_stats(latent_acc))
    metrics.update(_finalize_latent_extra_stats(latent_extra_acc))
    for modality, values in per_modality_l1.items():
        metrics[f"l1_{modality}"] = sum(values) / len(values) if values else float("nan")
        metrics[f"ssim_{modality}"] = (
            sum(per_modality_ssim[modality]) / len(per_modality_ssim[modality])
            if per_modality_ssim[modality]
            else float("nan")
        )
        metrics[f"psnr_{modality}"] = (
            sum(per_modality_psnr[modality]) / len(per_modality_psnr[modality])
            if per_modality_psnr[modality]
            else float("nan")
        )
        metrics[f"kl_mu_{modality}"] = (
            sum(per_modality_kl_mu[modality]) / len(per_modality_kl_mu[modality])
            if per_modality_kl_mu[modality]
            else float("nan")
        )
        metrics[f"kl_var_{modality}"] = (
            sum(per_modality_kl_var[modality]) / len(per_modality_kl_var[modality])
            if per_modality_kl_var[modality]
            else float("nan")
        )
        metrics.update(
            _finalize_latent_extra_stats(
                per_modality_extra[modality],
                suffix=modality,
                include_percentiles=False,
            )
        )
    metrics["steps"] = steps
    if preview_dir is not None:
        preview_paths = {}
        for modality, entries in preview_entries.items():
            preview_name = (
                f"epoch_{int(preview_epoch):04d}_multimodal_{modality}.png"
                if preview_epoch is not None
                else f"validation_multimodal_{modality}.png"
            )
            preview_paths[modality] = _save_validation_preview(
                entries,
                Path(preview_dir) / preview_name,
                max_previews=preview_count,
                required_scanners=preview_required_scanners,
            )
        metrics["preview_paths"] = preview_paths
        metrics["preview_path"] = next((path for path in preview_paths.values() if path), None)
    return metrics


def train_vaev2_multimodal(
    model,
    dataset,
    val_dataset=None,
    criterion=None,
    num_epochs=100,
    batch_size=1,
    lr=1e-4,
    weight_decay=1e-2,
    num_workers=0,
    device="cuda",
    checkpoint_dir="checkpoints/vaev2_multimodal",
    resume_path=None,
    resume_from_best=False,
    checkpoint_every=10,
    val_every=5,
    max_val_batches=None,
    max_train_batches=None,
    val_num_workers=0,
    best_metric_name="ssim",
    best_metric_mode="max",
    metric_data_range=None,
    ssim_window_size=None,
    save_validation_previews=True,
    validation_preview_count=16,
    validation_preview_required_scanners=("Guys", "HH", "IOP"),
    validation_preview_modality=None,
    tile_overlap=None,
    blend_mode="hann",
    blend_floor=1e-3,
    kl_start_weight=None,
    kl_final_weight=None,
    kl_warmup_epochs=0,
    kl_warmup_start_epoch=1,
    use_amp=True,
    grad_clip=1.0,
    validate_on_resume=True,
    show_progress=True,
    dataloader_shuffle=False,
    shuffle_patients_each_epoch=True,
):
    best_metric_mode = str(best_metric_mode).lower()
    if best_metric_mode not in {"max", "min"}:
        raise ValueError("best_metric_mode must be 'max' or 'min'")
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    criterion = criterion or VAEv2Loss()
    model = model.to(device)
    criterion = criterion.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    start_epoch = 0
    best_metric = _initial_best_metric(best_metric_mode)
    resolved_resume_path = _resolve_resume_path(checkpoint_dir, resume_path, resume_from_best=resume_from_best)
    if resolved_resume_path is not None:
        checkpoint = load_vaev2_checkpoint(resolved_resume_path, model, optimizer, scheduler, map_location=device)
        start_epoch = int(checkpoint.get("epoch", 0))
        best_metric = float(checkpoint.get("best_metric", best_metric))
        if not math.isfinite(best_metric):
            best_metric = _initial_best_metric(best_metric_mode)
        print(f"Resumed from {resolved_resume_path} at epoch {start_epoch}")

    kl_start_weight = float(criterion.kl_weight if kl_start_weight is None else kl_start_weight)
    kl_final_weight = float(criterion.kl_weight if kl_final_weight is None else kl_final_weight)

    train_log_path = Path(checkpoint_dir) / "training.jsonl"
    val_log_path = Path(checkpoint_dir) / "validation.jsonl"
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device_type == "cuda")

    if validate_on_resume and resolved_resume_path is not None and start_epoch > 0 and val_dataset is not None:
        resume_kl_weight = _kl_weight_for_epoch(
            start_epoch,
            start_weight=kl_start_weight,
            final_weight=kl_final_weight,
            warmup_epochs=kl_warmup_epochs,
            warmup_start_epoch=kl_warmup_start_epoch,
        )
        criterion.kl_weight = resume_kl_weight
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Running validation for resumed checkpoint epoch {start_epoch}")
        metrics = validate_vaev2_multimodal(
            model,
            val_dataset,
            criterion,
            batch_size=batch_size,
            device=device,
            max_batches=max_val_batches,
            show_progress=show_progress,
            num_workers=val_num_workers,
            metric_data_range=metric_data_range,
            ssim_window_size=ssim_window_size,
            preview_dir=Path(checkpoint_dir) / "validation_previews" if save_validation_previews else None,
            preview_epoch=start_epoch,
            preview_count=validation_preview_count,
            preview_required_scanners=validation_preview_required_scanners,
            preview_modality=validation_preview_modality,
            tile_overlap=tile_overlap,
            blend_mode=blend_mode,
            blend_floor=blend_floor,
        )
        compact_metrics = _compact_validation_metrics(metrics)
        print(f"Resume validation: {compact_metrics}")
        if best_metric_name not in metrics:
            raise KeyError(f"Validation metric '{best_metric_name}' not found. Available metrics: {sorted(metrics)}")
        current_metric = float(metrics[best_metric_name])
        improved = _metric_is_better(current_metric, best_metric, best_metric_mode)
        if improved:
            best_metric = current_metric
            save_vaev2_checkpoint(
                Path(checkpoint_dir) / "best.pt",
                model,
                optimizer,
                scheduler,
                start_epoch,
                best_metric,
                extra={
                    "metrics": metrics,
                    "best_metric_name": best_metric_name,
                    "best_metric_mode": best_metric_mode,
                    "resume_validation": True,
                },
            )
        val_record = {
            "epoch": start_epoch,
            "mode": "multimodal",
            "kl_weight": resume_kl_weight,
            "metric_name": best_metric_name,
            "metric_value": current_metric,
            "best_metric": best_metric,
            "improved": improved,
            "resume_validation": True,
        }
        val_record.update(compact_metrics)
        _append_jsonl(val_log_path, val_record)

    for epoch in range(start_epoch, num_epochs):
        current_kl_weight = _kl_weight_for_epoch(
            epoch + 1,
            start_weight=kl_start_weight,
            final_weight=kl_final_weight,
            warmup_epochs=kl_warmup_epochs,
            warmup_start_epoch=kl_warmup_start_epoch,
        )
        criterion.kl_weight = current_kl_weight
        if shuffle_patients_each_epoch and hasattr(dataset, "shuffle_samples"):
            dataset.shuffle_samples(seed=epoch)
        model.train()
        epoch_loss = 0.0
        train_totals = {
            "loss": 0.0,
            "recon_loss": 0.0,
            "slice_mse_loss": 0.0,
            "mip_l1_loss": 0.0,
            "vessel_l1_loss": 0.0,
            "background_penalty_loss": 0.0,
            "inside_dark_loss": 0.0,
            "kl_loss": 0.0,
            "kl_mu_term": 0.0,
            "kl_var_term": 0.0,
            "latent_alignment_loss": 0.0,
        }
        steps = 0
        t0 = time.time()
        total_batches = math.ceil(len(dataset) / int(batch_size))
        if max_train_batches is not None:
            total_batches = min(total_batches, int(max_train_batches))
        loader, progress, active_num_workers = _training_progress_with_worker_fallback(
            dataset,
            batch_size=batch_size,
            shuffle=dataloader_shuffle,
            num_workers=num_workers,
            pin_memory=str(device).startswith("cuda"),
            show_progress=show_progress,
            total=total_batches,
            desc=f"epoch {epoch + 1}/{num_epochs} multimodal",
        )
        if active_num_workers != int(num_workers):
            num_workers = active_num_workers
        last_step_end = time.time()
        data_time_avg = None
        step_time_avg = None
        for batch_idx, (patches, modality_masks) in enumerate(progress):
            if max_train_batches is not None and batch_idx >= int(max_train_batches):
                break
            data_time = time.time() - last_step_end
            step_start = time.time()
            patches = patches.to(device, non_blocking=True)
            modality_masks = modality_masks.to(device, non_blocking=True)
            batch_groups, num_modalities = patches.shape[:2]
            flat_patches = patches.reshape(batch_groups * num_modalities, *patches.shape[2:])
            flat_masks = modality_masks.reshape(batch_groups * num_modalities, num_modalities)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                recon, mu, logvar, _ = model(flat_patches, flat_masks)
                paired_mu = mu.reshape(batch_groups, num_modalities, *mu.shape[1:])
                losses = criterion(recon, flat_patches, mu, logvar, paired_mu=paired_mu)
                loss = losses["loss"]

            if not torch.isfinite(loss):
                print(f"Skipping non-finite loss at epoch {epoch + 1}, step {steps + 1}")
                continue

            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            step_time = time.time() - step_start
            last_step_end = time.time()
            data_time_avg = data_time if data_time_avg is None else 0.9 * data_time_avg + 0.1 * data_time
            step_time_avg = step_time if step_time_avg is None else 0.9 * step_time_avg + 0.1 * step_time
            for key in train_totals:
                train_totals[key] += float(losses[key].detach().cpu())
            epoch_loss += float(loss.detach().cpu())
            steps += 1
            _set_progress_postfix(
                progress,
                {
                    "loss": f"{epoch_loss / steps:.4f}",
                    "recon": f"{float(losses['recon_loss'].detach().cpu()):.4f}",
                    "slice": f"{float(losses['slice_mse_loss'].detach().cpu()):.4f}",
                    "kl": f"{float(losses['kl_loss'].detach().cpu()):.4f}",
                    "kl_w": f"{current_kl_weight:.1e}",
                    "align": f"{float(losses['latent_alignment_loss'].detach().cpu()):.4f}",
                    "data_s": f"{data_time_avg:.2f}",
                    "step_s": f"{step_time_avg:.2f}",
                    **_cuda_memory_gb(device),
                },
            )

        scheduler.step()
        avg_loss = epoch_loss / max(steps, 1)
        epoch_time = time.time() - t0
        train_metrics = {key: value / max(steps, 1) for key, value in train_totals.items()}
        train_record = {
            "epoch": epoch + 1,
            "mode": "multimodal",
            "steps": steps,
            "time_sec": epoch_time,
            "lr": optimizer.param_groups[0]["lr"],
            "kl_weight": current_kl_weight,
            "loss": train_metrics["loss"],
            "recon_loss": train_metrics["recon_loss"],
            "slice_mse_loss": train_metrics["slice_mse_loss"],
            "mip_l1_loss": train_metrics["mip_l1_loss"],
            "vessel_l1_loss": train_metrics["vessel_l1_loss"],
            "background_penalty_loss": train_metrics["background_penalty_loss"],
            "inside_dark_loss": train_metrics["inside_dark_loss"],
            "kl_loss": train_metrics["kl_loss"],
            "kl_mu_term": train_metrics["kl_mu_term"],
            "kl_var_term": train_metrics["kl_var_term"],
            "latent_alignment_loss": train_metrics["latent_alignment_loss"],
        }
        _append_jsonl(train_log_path, train_record)
        print(f"Epoch {epoch + 1}/{num_epochs} | loss={avg_loss:.5f} | time={epoch_time:.0f}s")
        if checkpoint_every is not None and checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            save_vaev2_checkpoint(
                Path(checkpoint_dir) / f"epoch_{epoch + 1}.pt",
                model,
                optimizer,
                scheduler,
                epoch + 1,
                best_metric,
            )

        latest_extra = {
            "mode": "multimodal",
            "train": train_record,
            "best_metric_name": best_metric_name,
            "best_metric_mode": best_metric_mode,
        }
        save_vaev2_checkpoint(
            Path(checkpoint_dir) / "latest.pt",
            model,
            optimizer,
            scheduler,
            epoch + 1,
            best_metric,
            extra=latest_extra,
        )

        if val_dataset is not None and (epoch + 1) % val_every == 0:
            if str(device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            metrics = validate_vaev2_multimodal(
                model,
                val_dataset,
                criterion,
                batch_size=batch_size,
                device=device,
                max_batches=max_val_batches,
                show_progress=show_progress,
                num_workers=val_num_workers,
                metric_data_range=metric_data_range,
                ssim_window_size=ssim_window_size,
                preview_dir=Path(checkpoint_dir) / "validation_previews" if save_validation_previews else None,
                preview_epoch=epoch + 1,
                preview_count=validation_preview_count,
                preview_required_scanners=validation_preview_required_scanners,
                preview_modality=validation_preview_modality,
                tile_overlap=tile_overlap,
                blend_mode=blend_mode,
                blend_floor=blend_floor,
            )
            compact_metrics = _compact_validation_metrics(metrics)
            print(f"Validation: {compact_metrics}")
            if best_metric_name not in metrics:
                raise KeyError(f"Validation metric '{best_metric_name}' not found. Available metrics: {sorted(metrics)}")
            current_metric = float(metrics[best_metric_name])
            improved = _metric_is_better(current_metric, best_metric, best_metric_mode)
            if improved:
                best_metric = current_metric
                save_vaev2_checkpoint(
                    Path(checkpoint_dir) / "best.pt",
                    model,
                    optimizer,
                    scheduler,
                    epoch + 1,
                    best_metric,
                    extra={
                        "metrics": metrics,
                        "best_metric_name": best_metric_name,
                        "best_metric_mode": best_metric_mode,
                    },
                )
            val_record = {
                "epoch": epoch + 1,
                "mode": "multimodal",
                "kl_weight": current_kl_weight,
                "metric_name": best_metric_name,
                "metric_value": current_metric,
                "best_metric": best_metric,
                "improved": improved,
            }
            val_record.update(compact_metrics)
            _append_jsonl(val_log_path, val_record)
            latest_extra.update({"validation": val_record, "metrics": metrics})
            save_vaev2_checkpoint(
                Path(checkpoint_dir) / "latest.pt",
                model,
                optimizer,
                scheduler,
                epoch + 1,
                best_metric,
                extra=latest_extra,
            )
    return model


NIFTI_EXTENSIONS = (".nii.gz", ".nii")


LATENT_INPUT_EXTENSIONS = (".npy", ".npz", ".nii.gz", ".nii")


def _has_known_extension(filename, extensions=NIFTI_EXTENSIONS):
    filename = str(filename).lower()
    return any(filename.endswith(extension.lower()) for extension in extensions)


def _strip_known_extension(filename, extensions=NIFTI_EXTENSIONS):
    filename = str(filename)
    filename_lower = filename.lower()
    for extension in sorted(extensions, key=len, reverse=True):
        if filename_lower.endswith(extension.lower()):
            return filename[: -len(extension)]
    return Path(filename).stem


def _normalized_image_key(path, modality=None, extensions=NIFTI_EXTENSIONS):
    stem = _strip_known_extension(Path(path).name, extensions)
    stem_lower = stem.lower()

    for suffix in ("_mask", "_stripped"):
        if stem_lower.endswith(suffix):
            stem = stem[: -len(suffix)]
            stem_lower = stem.lower()

    if modality is None:
        return stem

    token = f"-{modality}".lower()
    index = stem_lower.rfind(token)
    if index < 0:
        return None
    return stem[:index]


def _collect_nifti_by_key(folder, modality=None, recursive=False, extensions=NIFTI_EXTENSIONS):
    folder = Path(folder)
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    files_by_key = defaultdict(list)
    ignored_files = []

    for path in sorted(iterator):
        if not path.is_file():
            continue
        if not _has_known_extension(path.name, extensions):
            ignored_files.append(path)
            continue

        key = _normalized_image_key(path, modality=modality, extensions=extensions)
        if key is None:
            ignored_files.append(path)
            continue
        files_by_key[key].append(path)

    return files_by_key, ignored_files


def _read_volume_for_vae(path, dtype=np.float32):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        volume = np.load(path, mmap_mode="r", allow_pickle=False)
    elif suffix == ".npz":
        loaded = np.load(path, allow_pickle=False)
        key = "array" if "array" in loaded.files else loaded.files[0]
        volume = loaded[key]
    else:
        image = sitk.ReadImage(str(path))
        volume = sitk.GetArrayFromImage(image).transpose(2, 1, 0)

    volume = np.asarray(volume, dtype=dtype)
    if volume.ndim == 4 and volume.shape[0] == 1:
        volume = volume[0]
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume at {path}, got shape {tuple(volume.shape)}")
    return volume


def _make_modality_mask(modality, model_modalities, device):
    model_modalities = tuple(str(item) for item in model_modalities)
    lookup = {name.upper(): index for index, name in enumerate(model_modalities)}
    key = str(modality).upper()
    if key not in lookup:
        raise ValueError(f"Modality {modality!r} is not in model modalities {model_modalities}")
    mask = np.zeros((1, len(model_modalities)), dtype=np.float32)
    mask[0, lookup[key]] = 1.0
    import torch

    return torch.from_numpy(mask).to(device)


def _output_path_for_latent(
    input_path,
    input_folder,
    output_folder,
    output_format="npy",
    filename_suffix="",
):
    input_path = Path(input_path)
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    relative_parent = input_path.parent.relative_to(input_folder)
    stem = _strip_known_extension(input_path.name, LATENT_INPUT_EXTENSIONS)
    extension = str(output_format).lower().lstrip(".")
    if extension not in {"npy", "npz", "pt"}:
        raise ValueError("output_format must be one of: 'npy', 'npz', 'pt'")
    return output_folder / relative_parent / f"{stem}{filename_suffix}.{extension}"


def _save_latent_tensor(
    latent,
    output_path,
    output_format="npy",
    output_dtype=np.float16,
    include_batch_dim=False,
):
    import torch

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_format = str(output_format).lower().lstrip(".")
    latent = latent.detach().cpu()
    if not include_batch_dim and latent.ndim == 5 and latent.shape[0] == 1:
        latent = latent[0]

    if output_format == "pt":
        if output_dtype is not None:
            torch_dtype = getattr(torch, str(np.dtype(output_dtype)), None)
            if torch_dtype is not None:
                latent = latent.to(torch_dtype)
            elif np.dtype(output_dtype) == np.dtype(np.float16):
                latent = latent.half()
            elif np.dtype(output_dtype) == np.dtype(np.float32):
                latent = latent.float()
        torch.save(latent, output_path)
        return tuple(int(value) for value in latent.shape), str(latent.dtype)

    array = latent.numpy()
    if output_dtype is not None:
        array = array.astype(output_dtype, copy=False)
    if output_format == "npy":
        np.save(output_path, array, allow_pickle=False)
    elif output_format == "npz":
        np.savez_compressed(output_path, array=array)
    else:
        raise ValueError("output_format must be one of: 'npy', 'npz', 'pt'")
    return tuple(int(value) for value in array.shape), str(array.dtype)


def _encode_volume_with_frozen_vae(
    model,
    volume,
    vae_type,
    modality,
    modalities,
    device,
    latent_kind="mu",
    prefer_tiled=True,
    tile_size=None,
    overlap=None,
    pad_to_factor=True,
):
    import torch

    vae_type = str(vae_type).lower().strip()
    latent_kind = str(latent_kind).lower().strip()
    if latent_kind not in {"mu", "z", "logvar"}:
        raise ValueError("latent_kind must be one of: 'mu', 'z', 'logvar'")
    if vae_type not in {"multimodal", "mra"}:
        raise ValueError("vae_type must be 'multimodal' or 'mra'")

    x = torch.from_numpy(np.asarray(volume, dtype=np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    modality_mask = None
    if vae_type == "multimodal":
        model_modalities = getattr(model, "modalities", modalities)
        modality_mask = _make_modality_mask(modality, model_modalities, device)

    if pad_to_factor and hasattr(model, "_pad_spatial_to_factor"):
        x, _ = model._pad_spatial_to_factor(x)

    with torch.inference_mode():
        if prefer_tiled:
            z, mu, logvar = model.encode_tiled(
                x,
                modality_mask=modality_mask,
                tile_size=tile_size,
                overlap=overlap,
            )
        else:
            try:
                z, mu, logvar = model.encode(x, modality_mask=modality_mask)
            except RuntimeError as error:
                if "out of memory" not in str(error).lower():
                    raise
                if str(device).startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                z, mu, logvar = model.encode_tiled(
                    x,
                    modality_mask=modality_mask,
                    tile_size=tile_size,
                    overlap=overlap,
                )
            except ValueError:
                z, mu, logvar = model.encode_tiled(
                    x,
                    modality_mask=modality_mask,
                    tile_size=tile_size,
                    overlap=overlap,
                )

    if latent_kind == "mu":
        return mu
    if latent_kind == "z":
        return z
    return logvar


def extract_vae_latents_from_folders(
    image_folders,
    output_root,
    model,
    modalities=("T1", "T2", "PD"),
    vae_type="multimodal",
    checkpoint_path=None,
    device=None,
    image_extensions=LATENT_INPUT_EXTENSIONS,
    recursive=False,
    overwrite=False,
    max_images=None,
    latent_kind="mu",
    output_format="npy",
    output_dtype=np.float16,
    filename_suffix="",
    include_batch_dim=False,
    prefer_tiled=True,
    tile_size=None,
    overlap=None,
    pad_to_factor=True,
    manifest_path=None,
    show_progress=True,
    clear_cuda_cache=True,
    print_stats=True,
):
    """
    Encode one set of modality folders with a frozen VAE and save one latent per image.

    Parameters
    ----------
    image_folders:
        Mapping from modality name to folder path, or a single folder when one
        modality is passed.
    output_root:
        Root folder where modality subfolders with latent files are written.
    model:
        A constructed VAEv2Multimodal or VAEv2MRA instance. If checkpoint_path
        is provided, its state dict is loaded into this model before extraction.
    modalities:
        Modalities to process. Use a one-element list such as ["MRA"] for MRA.
    vae_type:
        "multimodal" routes each volume with a one-hot modality mask. "mra"
        calls the single-modality encoder.
    latent_kind:
        Usually "mu". "z" is sampled and therefore not deterministic.
    """
    import torch
    from modules.vae_multimodal import load_vaev2_checkpoint

    modalities = tuple(str(modality) for modality in modalities)
    if not modalities:
        raise ValueError("modalities must contain at least one modality")

    if isinstance(image_folders, (str, Path)):
        if len(modalities) != 1:
            raise ValueError("A single image folder can only be used with one modality")
        image_folders = {modalities[0]: image_folders}
    else:
        image_folders = {str(key): value for key, value in dict(image_folders).items()}

    missing_modalities = [modality for modality in modalities if modality not in image_folders]
    if missing_modalities:
        raise KeyError(f"Missing image folders for modalities: {missing_modalities}")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_root = Path(output_root)
    if checkpoint_path is not None:
        load_vaev2_checkpoint(checkpoint_path, model, map_location=device)
    model = model.to(device)
    was_training = model.training
    model_parameters = list(model.parameters())
    requires_grad_state = [parameter.requires_grad for parameter in model_parameters]
    model.eval()
    for parameter in model_parameters:
        parameter.requires_grad_(False)

    manifest_handle = None
    if manifest_path is not None:
        manifest_path = Path(manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_handle = manifest_path.open("w", encoding="utf-8")

    processed = 0
    skipped = 0
    failed = []
    ignored = {}
    started = time.time()

    try:
        for modality in modalities:
            image_folder = Path(image_folders[modality])
            if not image_folder.is_dir():
                raise NotADirectoryError(f"Image folder does not exist for {modality}: {image_folder}")
            output_folder = output_root / modality
            files_by_key, ignored_files = _collect_nifti_by_key(
                image_folder,
                modality=modality,
                recursive=recursive,
                extensions=image_extensions,
            )
            ignored[modality] = [str(path) for path in ignored_files]
            items = []
            for patient_id, paths in files_by_key.items():
                for path in paths:
                    items.append((patient_id, path))
            items.sort(key=lambda item: str(item[1]))
            if max_images is not None:
                items = items[: int(max_images)]

            progress = _progress_bar(
                items,
                enabled=show_progress,
                desc=f"extract {modality} latents",
                leave=False,
            )
            for patient_id, image_path in progress:
                output_path = _output_path_for_latent(
                    image_path,
                    image_folder,
                    output_folder,
                    output_format=output_format,
                    filename_suffix=filename_suffix,
                )
                if output_path.exists() and not overwrite:
                    skipped += 1
                    continue

                try:
                    volume = _read_volume_for_vae(image_path, dtype=np.float32)
                    latent = _encode_volume_with_frozen_vae(
                        model=model,
                        volume=volume,
                        vae_type=vae_type,
                        modality=modality,
                        modalities=modalities,
                        device=device,
                        latent_kind=latent_kind,
                        prefer_tiled=prefer_tiled,
                        tile_size=tile_size,
                        overlap=overlap,
                        pad_to_factor=pad_to_factor,
                    )
                    latent_shape, saved_dtype = _save_latent_tensor(
                        latent,
                        output_path,
                        output_format=output_format,
                        output_dtype=output_dtype,
                        include_batch_dim=include_batch_dim,
                    )
                    processed += 1
                    if manifest_handle is not None:
                        manifest_handle.write(
                            json.dumps(
                                {
                                    "patient_id": patient_id,
                                    "modality": modality,
                                    "input_path": str(image_path),
                                    "output_path": str(output_path),
                                    "latent_kind": latent_kind,
                                    "latent_shape": latent_shape,
                                    "dtype": saved_dtype,
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                except Exception as error:
                    failed.append(
                        {
                            "patient_id": patient_id,
                            "modality": modality,
                            "input_path": str(image_path),
                            "output_path": str(output_path),
                            "error": repr(error),
                        }
                    )
                finally:
                    if clear_cuda_cache and str(device).startswith("cuda") and torch.cuda.is_available():
                        torch.cuda.empty_cache()
    finally:
        if manifest_handle is not None:
            manifest_handle.close()
        for parameter, requires_grad in zip(model_parameters, requires_grad_state):
            parameter.requires_grad_(requires_grad)
        model.train(was_training)

    summary = {
        "output_root": str(output_root),
        "modalities": list(modalities),
        "vae_type": str(vae_type),
        "latent_kind": str(latent_kind),
        "output_format": str(output_format),
        "processed_images": processed,
        "skipped_existing_images": skipped,
        "failed_images": failed,
        "ignored_files": ignored,
        "time_sec": time.time() - started,
    }

    if print_stats:
        print("VAE latent extraction finished")
        print(f"Output root: {summary['output_root']}")
        print(f"VAE type: {summary['vae_type']}")
        print(f"Modalities: {summary['modalities']}")
        print(f"Latent kind: {summary['latent_kind']}")
        print(f"Processed images: {processed}")
        print(f"Skipped existing images: {skipped}")
        print(f"Failed images: {len(failed)}")
        print(f"Time: {summary['time_sec']:.1f}s")
        if failed:
            print(f"Failed image preview: {failed[:5]}")
    return summary


def extract_vae_latents_from_splits(
    split_root,
    output_root,
    model,
    modalities=("T1", "T2", "PD"),
    vae_type="multimodal",
    splits=("train", "val", "test"),
    checkpoint_path=None,
    device=None,
    image_extensions=LATENT_INPUT_EXTENSIONS,
    recursive=False,
    overwrite=False,
    max_images=None,
    latent_kind="mu",
    output_format="npy",
    output_dtype=np.float16,
    filename_suffix="",
    include_batch_dim=False,
    prefer_tiled=True,
    tile_size=None,
    overlap=None,
    pad_to_factor=True,
    manifest_name="latent_manifest.jsonl",
    show_progress=True,
    clear_cuda_cache=True,
    print_stats=True,
):
    """
    Encode train/val/test split folders with a frozen VAE and save latents.

    Expected input structure:
        split_root/train/T1/*.npy
        split_root/train/T2/*.npy
        split_root/train/PD/*.npy

    For MRA-only extraction, call with modalities=("MRA",) and vae_type="mra".

    Output structure:
        output_root/train/T1/*.npy
        output_root/val/T1/*.npy
        output_root/test/T1/*.npy
    """
    split_root = Path(split_root)
    output_root = Path(output_root)
    modalities = tuple(str(modality) for modality in modalities)
    summaries = {}

    for split in splits:
        split = str(split)
        split_input = split_root / split
        if not split_input.is_dir():
            raise NotADirectoryError(f"Split folder does not exist: {split_input}")
        image_folders = {modality: split_input / modality for modality in modalities}
        split_output = output_root / split
        manifest_path = split_output / manifest_name if manifest_name is not None else None
        summaries[split] = extract_vae_latents_from_folders(
            image_folders=image_folders,
            output_root=split_output,
            model=model,
            modalities=modalities,
            vae_type=vae_type,
            checkpoint_path=checkpoint_path,
            device=device,
            image_extensions=image_extensions,
            recursive=recursive,
            overwrite=overwrite,
            max_images=max_images,
            latent_kind=latent_kind,
            output_format=output_format,
            output_dtype=output_dtype,
            filename_suffix=filename_suffix,
            include_batch_dim=include_batch_dim,
            prefer_tiled=prefer_tiled,
            tile_size=tile_size,
            overlap=overlap,
            pad_to_factor=pad_to_factor,
            manifest_path=manifest_path,
            show_progress=show_progress,
            clear_cuda_cache=clear_cuda_cache,
            print_stats=print_stats,
        )
        checkpoint_path = None

    total_processed = sum(summary["processed_images"] for summary in summaries.values())
    total_skipped = sum(summary["skipped_existing_images"] for summary in summaries.values())
    total_failed = sum(len(summary["failed_images"]) for summary in summaries.values())
    summary = {
        "split_root": str(split_root),
        "output_root": str(output_root),
        "splits": list(summaries.keys()),
        "modalities": list(modalities),
        "vae_type": str(vae_type),
        "processed_images": total_processed,
        "skipped_existing_images": total_skipped,
        "failed_images": total_failed,
        "split_summaries": summaries,
    }

    if print_stats:
        print("Split VAE latent extraction finished")
        print(f"Split root: {summary['split_root']}")
        print(f"Output root: {summary['output_root']}")
        print(f"Splits: {summary['splits']}")
        print(f"Modalities: {summary['modalities']}")
        print(f"Processed images: {total_processed}")
        print(f"Skipped existing images: {total_skipped}")
        print(f"Failed images: {total_failed}")
    return summary


def build_multimodal_vae(config: dict) -> "VAEv2Multimodal":
    """The model described by configs/vae_multimodal.json (the thesis conditioning VAE)."""
    m = config["model"]
    return VAEv2Multimodal(patch_size=tuple(m["patch_size"]), modalities=tuple(config["data"]["modalities"]),
                           in_channels=1, base_channels=int(m["base_channels"]),
                           channel_multipliers=tuple(m["channel_multipliers"]),
                           latent_channels=int(m["latent_channels"]),
                           blocks_per_level=int(m["blocks_per_level"]),
                           output_activation=m["output_activation"])


def load_multimodal_vae(config: dict, checkpoint, device: str) -> "VAEv2Multimodal":
    model = build_multimodal_vae(config)
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model_state_dict"])
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def encode_multimodal_mu(model, volume: np.ndarray, modality_index: int, device: str) -> np.ndarray:
    """Deterministic latent mean of one whole volume (the same path the Table 5.2 metrics use);
    falls back to tiled encoding if the whole volume does not fit on the GPU."""
    mask = _make_single_modality_mask(modality_index, len(model.modalities), device)
    x = _volume_tensor(np.asarray(volume, np.float32), device)
    with torch.inference_mode():
        x, _ = model._pad_spatial_to_factor(x)
        try:
            mu, _ = _encode_deterministic(model, x, mask)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            mu, _ = _encode_tiled_deterministic(model, x, mask)
    return mu[0].float().cpu().numpy()
