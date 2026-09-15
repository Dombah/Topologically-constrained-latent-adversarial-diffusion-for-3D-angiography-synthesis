"""Image, vessel and topology metrics (skimage-based, brain-masked where it matters)."""
from __future__ import annotations

import math

import numpy as np


def masked_ssim_and_psnr(target: np.ndarray, recon: np.ndarray, brain: np.ndarray) -> tuple[float, float, float]:
    from skimage.metrics import structural_similarity

    recon = np.clip(recon, 0.0, 1.0).astype(np.float32)
    target = target.astype(np.float32)
    value, ssim_map = structural_similarity(target, recon, data_range=1.0, full=True)
    inner = (slice(3, -3),) * 3
    masked = float(ssim_map[inner][brain[inner]].mean()) if brain[inner].any() else float("nan")
    mse = float(np.mean((target - recon) ** 2))
    return float(value), masked, 10.0 * math.log10(1.0 / max(mse, 1e-12))


def vessel_scores(target: np.ndarray, recon: np.ndarray) -> tuple[float, float]:
    recon = np.clip(recon, 0.0, 1.0)
    threshold = np.percentile(target, 99.0)
    true_band, pred_band = target >= threshold, recon >= threshold
    dice = 2.0 * float((true_band & pred_band).sum()) / max(float(true_band.sum() + pred_band.sum()), 1.0)
    return dice, float(np.abs(target - recon)[true_band].mean())


def mip_ssim(target: np.ndarray, recon: np.ndarray) -> float:
    from skimage.metrics import structural_similarity

    recon = np.clip(recon, 0.0, 1.0)
    return float(np.mean([structural_similarity(target.max(axis=axis), recon.max(axis=axis), data_range=1.0)
                          for axis in range(3)]))


def hard_cldice(target: np.ndarray, recon: np.ndarray) -> float:
    from skimage.morphology import skeletonize

    threshold = np.percentile(target, 99.0)
    true_band, pred_band = target >= threshold, np.clip(recon, 0.0, 1.0) >= threshold
    skeleton_pred, skeleton_true = skeletonize(pred_band), skeletonize(true_band)
    precision = float((skeleton_pred & true_band).sum()) / max(float(skeleton_pred.sum()), 1.0)
    sensitivity = float((skeleton_true & pred_band).sum()) / max(float(skeleton_true.sum()), 1.0)
    return float(2.0 * precision * sensitivity / max(precision + sensitivity, 1e-8))


def latent_diagnostics(latents: list[np.ndarray]) -> dict:
    """The properties that decide whether the LDM can model these latents."""
    from scipy import stats as scipy_stats

    per_channel = np.stack([latent.reshape(latent.shape[0], -1).std(axis=1) for latent in latents]).mean(axis=0)
    scale = float(np.concatenate([latent.ravel()[::17] for latent in latents]).std())
    normalised = [latent / max(scale, 1e-8) for latent in latents]
    flat = np.concatenate([latent.ravel()[::7] for latent in normalised])
    high_freq = []
    for latent in normalised[:8]:
        spectrum = np.abs(np.fft.fftshift(np.fft.fftn(latent, axes=(1, 2, 3)), axes=(1, 2, 3))) ** 2
        power = spectrum.mean(axis=0)
        grids = np.meshgrid(*[np.fft.fftshift(np.fft.fftfreq(dim)) for dim in power.shape], indexing="ij")
        radius = np.sqrt(sum(grid ** 2 for grid in grids))
        high_freq.append(float(power[radius > 0.25].sum() / power.sum()))
    return {
        "latent_per_channel_std_min": float(per_channel.min()),
        "latent_per_channel_std_max": float(per_channel.max()),
        "latent_global_std": scale,
        "latent_kurtosis": float(scipy_stats.kurtosis(flat)),
        "latent_max_abs_sigma": float(np.max([np.abs(latent).max() for latent in normalised])),
        "latent_frac_gt4": float((np.abs(flat) > 4).mean()),
        "latent_high_freq_fraction": float(np.mean(high_freq)),
    }


MIN_VOX = 20


def betti(mask: np.ndarray) -> tuple[int, int, int]:
    from skimage.measure import euler_number, label
    labels = label(mask, connectivity=3)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= MIN_VOX
    keep[0] = False
    cleaned = keep[labels]
    b0 = int(keep.sum())
    if b0 == 0:
        return 0, 0, 0
    background = label(~cleaned, connectivity=1)
    border = np.unique(np.concatenate([
        background[0].ravel(), background[-1].ravel(), background[:, 0].ravel(),
        background[:, -1].ravel(), background[:, :, 0].ravel(), background[:, :, -1].ravel()]))
    b2 = int(background.max()) - int((border > 0).sum())
    chi = int(euler_number(cleaned, connectivity=3))
    return b0, b0 + b2 - chi, b2
