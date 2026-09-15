"""Frangi vesselness weight maps at latent resolution, used to weight the bridge's x0 losses."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


TARGET_MODALITY = "MRA"


def latent_patient_id(path: str | Path, modality: str) -> str:
    name = Path(path).stem
    if name.endswith(".nii"):
        name = Path(name).stem
    suffix = f"-{modality}"
    return name[: -len(suffix)] if name.endswith(suffix) else name


def collect_latent_files(latent_root: str | Path, split: str, modality: str) -> dict[str, Path]:
    folder = Path(latent_root) / modality / split
    if not folder.exists():
        return {}
    return {latent_patient_id(path, modality): path for path in sorted(folder.glob("*.npy"))}


def _mra_patient_id(path: str | Path) -> str:
    name = Path(path).stem

    if name.endswith("_mask"):
        name = name[: -len("_mask")]

    if name.endswith("-mask"):
        name = name[: -len("-mask")]

    suffix = f"-{TARGET_MODALITY}"
    return name[: -len(suffix)] if name.endswith(suffix) else name


def _collect_mra_image_files(dataset_root: str | Path, split: str) -> dict[str, Path]:
    folder = Path(dataset_root) / split / TARGET_MODALITY
    if not folder.exists():
        return {}

    return {
        _mra_patient_id(path): path
        for path in sorted(folder.glob("*.npy"))
    }


def _collect_mra_mask_files(dataset_root: str | Path, split: str) -> dict[str, Path]:
    folder = Path(dataset_root) / split / "masks" / TARGET_MODALITY
    if not folder.exists():
        return {}

    return {
        _mra_patient_id(path): path
        for path in sorted(folder.glob("*.npy"))
    }


def _as_volume3d(arr: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(arr)

    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim != 3:
        raise ValueError(
            f"Expected {name} to have shape [D, H, W] or [1, D, H, W], "
            f"got {arr.shape}."
        )

    return np.nan_to_num(
        arr.astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def _resize_volume3d(
    arr: np.ndarray,
    target_shape: Sequence[int],
    *,
    mode: str,
) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(arr, dtype=np.float32))[None, None]

    if mode == "nearest":
        resized = F.interpolate(
            tensor,
            size=tuple(int(v) for v in target_shape),
            mode=mode,
        )
    else:
        resized = F.interpolate(
            tensor,
            size=tuple(int(v) for v in target_shape),
            mode=mode,
            align_corners=False,
        )

    return resized[0, 0].cpu().numpy()


def _as_downsample_tuple(value: int | Sequence[int]) -> tuple[int, int, int]:
    if isinstance(value, int):
        factors = (int(value), int(value), int(value))
    else:
        factors = tuple(int(v) for v in value)

    if len(factors) != 3:
        raise ValueError(
            f"frangi_downsample must be an int or length-3 sequence, got {value}."
        )

    if any(v < 1 for v in factors):
        raise ValueError(
            f"frangi_downsample factors must be >= 1, got {factors}."
        )

    return factors


def _downsample_shape(
    shape: Sequence[int],
    factors: Sequence[int],
) -> tuple[int, int, int]:
    return tuple(
        max(1, int(round(int(size) / int(factor))))
        for size, factor in zip(shape, factors)
    )


def _masked_values(
    arr: np.ndarray,
    mask: np.ndarray | None,
    *,
    positive_only: bool = False,
) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)

    if mask is not None and np.any(mask > 0.5):
        values = arr[mask > 0.5]
    else:
        values = arr.reshape(-1)

    values = values[np.isfinite(values)]

    if positive_only:
        values = values[values > 0]

    return values.astype(np.float32, copy=False)


def _normalize_image_for_frangi(
    image: np.ndarray,
    mask: np.ndarray | None,
    *,
    low_percentile: float = 1.0,
    high_percentile: float = 99.5,
) -> np.ndarray:
    image = np.nan_to_num(
        np.asarray(image, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # If there is no mask, ignore exact zero background.
    values = _masked_values(
        image,
        mask,
        positive_only=(mask is None),
    )

    if values.size == 0:
        return np.zeros_like(image, dtype=np.float32)

    lo = float(np.percentile(values, low_percentile))
    hi = float(np.percentile(values, high_percentile))

    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)

    out = np.clip((image - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

    if mask is not None:
        out *= (mask > 0.5).astype(np.float32)

    return out


def _suppress_weak_responses(
    weight: np.ndarray,
    mask: np.ndarray | None,
    *,
    floor_percentile: float | None,
    eps: float = 1e-8,
) -> np.ndarray:
    weight = np.nan_to_num(
        np.asarray(weight, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    weight = np.clip(weight, 0.0, None)

    if floor_percentile is None or floor_percentile <= 0:
        return weight.astype(np.float32)

    values = _masked_values(
        weight,
        mask,
        positive_only=True,
    )

    if values.size == 0:
        return np.zeros_like(weight, dtype=np.float32)

    floor = float(np.percentile(values, floor_percentile))

    if floor <= eps:
        return weight.astype(np.float32)

    weight = np.clip(weight - floor, 0.0, None)

    if mask is not None:
        weight *= (mask > 0.5).astype(np.float32)

    return weight.astype(np.float32)


def _normalize_weight_map(
    weight: np.ndarray,
    mask: np.ndarray | None,
    *,
    robust_percentile: float,
    gamma: float = 1.0,
    eps: float = 1e-8,
) -> np.ndarray:
    weight = np.nan_to_num(
        np.asarray(weight, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    weight = np.clip(weight, 0.0, None)

    values = _masked_values(
        weight,
        mask,
        positive_only=True,
    )

    if values.size == 0:
        return np.zeros_like(weight, dtype=np.float32)

    denom = float(np.percentile(values, robust_percentile))

    if denom <= eps:
        return np.zeros_like(weight, dtype=np.float32)

    weight = np.clip(weight / denom, 0.0, 1.0)

    if gamma != 1.0:
        weight = np.power(weight, float(gamma))

    if mask is not None:
        weight *= (mask > 0.5).astype(np.float32)

    return np.clip(weight, 0.0, 1.0).astype(np.float32)


def _postprocess_weight_map(
    weight: np.ndarray,
    mask: np.ndarray | None,
    *,
    min_value: float | None,
    renormalize: bool,
    robust_percentile: float,
) -> np.ndarray:
    weight = np.nan_to_num(
        np.asarray(weight, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    weight = np.clip(weight, 0.0, 1.0)

    if min_value is not None and float(min_value) > 0:
        weight = np.where(weight >= float(min_value), weight, 0.0).astype(np.float32)

    if mask is not None:
        weight *= (mask > 0.5).astype(np.float32)

    if renormalize:
        weight = _normalize_weight_map(
            weight,
            mask,
            robust_percentile=robust_percentile,
            gamma=1.0,
        )

    return np.clip(weight, 0.0, 1.0).astype(np.float32)


def _frangi_weight(
    image: np.ndarray,
    mask: np.ndarray | None,
    *,
    sigmas: Sequence[float],
    black_ridges: bool,
    robust_percentile: float,
    floor_percentile: float | None,
    gamma: float,
) -> np.ndarray:
    try:
        from skimage.filters import frangi
    except Exception as error:
        raise ImportError(
            "generate_frangi_vessel_weight_cache requires scikit-image."
        ) from error

    image_norm = _normalize_image_for_frangi(image, mask)

    vesselness = frangi(
        image_norm,
        sigmas=tuple(float(s) for s in sigmas),
        black_ridges=bool(black_ridges),
    )

    vesselness = np.nan_to_num(
        vesselness.astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if mask is not None:
        vesselness *= (mask > 0.5).astype(np.float32)

    vesselness = _suppress_weak_responses(
        vesselness,
        mask,
        floor_percentile=floor_percentile,
    )

    vesselness = _normalize_weight_map(
        vesselness,
        mask,
        robust_percentile=robust_percentile,
        gamma=gamma,
    )

    return vesselness


def generate_frangi_vessel_weight_cache(
    *,
    dataset_root: str | Path,
    latent_root: str | Path,
    output_root: str | Path,
    splits: Sequence[str] = ("train", "val"),
    max_cases: int | None = None,
    overwrite: bool = False,
    sigmas: Sequence[float] = (1.0, 2.0, 3.0),
    black_ridges: bool = False,
    robust_percentile: float = 99.7,
    frangi_downsample: int | Sequence[int] = 2,
    floor_percentile: float | None = 50.0,
    frangi_gamma: float = 1.25,
    latent_renormalize: bool = True,
    latent_gamma: float = 1.0,
    post_min_value: float | None = None,
    post_renormalize: bool = False,
    show_progress: bool = True,
) -> dict[str, dict[str, int]]:
    """
    Generate latent-resolution Frangi vessel weights from real MRA volumes.

    Saved output shape:
        [1, D_lat, H_lat, W_lat]

    Intended training use:
        weight = 1.0 + alpha * vessel_weight
    """

    dataset_root = Path(dataset_root)
    latent_root = Path(latent_root)
    output_root = Path(output_root)

    downsample_factors = _as_downsample_tuple(frangi_downsample)
    summary: dict[str, dict[str, int]] = {}

    for split in splits:
        image_files = _collect_mra_image_files(dataset_root, split)
        mask_files = _collect_mra_mask_files(dataset_root, split)
        latent_files = collect_latent_files(latent_root, split, TARGET_MODALITY)

        patients = sorted(set(image_files) & set(latent_files))

        if not patients:
            raise FileNotFoundError(
                f"No paired MRA images and latents found for split '{split}' under "
                f"{dataset_root} and {latent_root}."
            )

        split_root = output_root / split
        split_root.mkdir(parents=True, exist_ok=True)

        existing = {path.stem for path in split_root.glob("*.npy")}
        patient_set = set(patients)
        missing = [patient for patient in patients if overwrite or patient not in existing]
        to_process = len(missing) if max_cases is None else min(len(missing), int(max_cases))
        if overwrite:
            print(
                f"WARNING: overwrite=True for Frangi vessel cache split '{split}'. "
                f"Existing cache files in {split_root} may be replaced."
            )
        print(
            f"Frangi vessel cache {split}: paired={len(patients)}, "
            f"existing={len(existing & patient_set)}, missing={len(missing)}, "
            f"will_process={to_process}, overwrite={bool(overwrite)}"
        )

        counts = {
            "processed": 0,
            "skipped": 0,
            "missing_masks": 0,
        }

        iterator = patients
        if show_progress:
            try:
                from tqdm.auto import tqdm

                iterator = tqdm(patients, desc=f"Frangi {split}", leave=True)
            except Exception:
                iterator = patients

        for patient in iterator:
            out_path = split_root / f"{patient}.npy"

            if out_path.exists() and not overwrite:
                counts["skipped"] += 1
                if hasattr(iterator, "set_postfix"):
                    iterator.set_postfix(
                        processed=counts["processed"],
                        skipped=counts["skipped"],
                        remaining=max(0, to_process - counts["processed"]),
                    )
                continue

            if max_cases is not None and counts["processed"] >= int(max_cases):
                break

            image = _as_volume3d(
                np.load(image_files[patient], mmap_mode="r"),
                name="MRA image",
            )

            mask = None
            if patient in mask_files:
                mask = _as_volume3d(
                    np.load(mask_files[patient], mmap_mode="r"),
                    name="MRA mask",
                )

                if mask.shape != image.shape:
                    mask = _resize_volume3d(
                        mask,
                        image.shape,
                        mode="nearest",
                    )

                mask = (mask > 0.5).astype(np.float32)
            else:
                counts["missing_masks"] += 1

            latent = np.load(latent_files[patient], mmap_mode="r")
            latent_shape = tuple(int(v) for v in latent.shape[-3:])

            if downsample_factors != (1, 1, 1):
                frangi_shape = _downsample_shape(
                    image.shape,
                    downsample_factors,
                )

                image_for_frangi = _resize_volume3d(
                    image,
                    frangi_shape,
                    mode="trilinear",
                )

                if mask is not None:
                    mask_for_frangi = _resize_volume3d(
                        mask,
                        frangi_shape,
                        mode="nearest",
                    )
                    mask_for_frangi = (mask_for_frangi > 0.5).astype(np.float32)
                else:
                    mask_for_frangi = None
            else:
                image_for_frangi = image
                mask_for_frangi = mask

            weight_frangi = _frangi_weight(
                image_for_frangi,
                mask_for_frangi,
                sigmas=sigmas,
                black_ridges=black_ridges,
                robust_percentile=robust_percentile,
                floor_percentile=floor_percentile,
                gamma=frangi_gamma,
            )

            weight_latent = _resize_volume3d(
                weight_frangi,
                latent_shape,
                mode="trilinear",
            )

            latent_mask = None
            if mask_for_frangi is not None:
                latent_mask = _resize_volume3d(
                    mask_for_frangi,
                    latent_shape,
                    mode="nearest",
                )
                latent_mask = (latent_mask > 0.5).astype(np.float32)

            if latent_renormalize:
                weight_latent = _normalize_weight_map(
                    weight_latent,
                    latent_mask,
                    robust_percentile=robust_percentile,
                    gamma=latent_gamma,
                )
            else:
                weight_latent = np.clip(weight_latent, 0.0, 1.0).astype(np.float32)

            if latent_mask is not None:
                weight_latent *= latent_mask

            weight_latent = _postprocess_weight_map(
                weight_latent,
                latent_mask,
                min_value=post_min_value,
                renormalize=bool(post_renormalize),
                robust_percentile=robust_percentile,
            )

            np.save(
                out_path,
                np.clip(weight_latent[None], 0.0, 1.0).astype(np.float16),
            )

            counts["processed"] += 1
            if hasattr(iterator, "set_postfix"):
                iterator.set_postfix(
                    processed=counts["processed"],
                    skipped=counts["skipped"],
                    remaining=max(0, to_process - counts["processed"]),
                )

        summary[str(split)] = counts

        print(
            f"Frangi vessel cache {split}: "
            f"processed={counts['processed']}, "
            f"skipped={counts['skipped']}, "
            f"missing_masks={counts['missing_masks']}, "
            f"downsample={downsample_factors}, "
            f"sigmas={tuple(float(s) for s in sigmas)}, "
            f"robust_percentile={robust_percentile}, "
            f"floor_percentile={floor_percentile}, "
            f"frangi_gamma={frangi_gamma}, "
            f"latent_renormalize={latent_renormalize}, "
            f"latent_gamma={latent_gamma}, "
            f"post_min_value={post_min_value}, "
            f"post_renormalize={bool(post_renormalize)}"
        )

    return summary
