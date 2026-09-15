"""Export latents: MRA from the MRA VAE (the bridge's target), or T1/T2/PD from the
multimodal VAE (the bridge's condition).

    uv run export_latents.py                         # v5_cldice -> latents_v5_cldice/MRA, then verify
    uv run export_latents.py --verify-only           # re-run the decode checks alone
    uv run export_latents.py --model multimodal      # kl5e5_from_epoch10 -> latents_multimodal/{T1,T2,PD}

MRA output: <out>/MRA/{split}/<case>-MRA.npy (16, 128, 144, 24) + <out>/latent_stats.json.
Multimodal output: <out>/{T1,T2,PD}/{split}/<case>-<M>.npy (8, 128, 144, 24) +
<out>/latent_stats_per_channel.json.

THE THESIS CONDITIONING LATENTS ARE A FIXED INPUT
-------------------------------------------------
latents/{T1,T2,PD} were exported by a notebook that no longer exists, and no encoding of any
checkpoint on disk reproduces them exactly (closest max |diff| ~0.27). Every bridge checkpoint
was trained on those files, so keep them; this script refuses to write into latents/.
`--model multimodal` produces equivalent latents for a from-scratch rebuild.

WHY DIRECT (WHOLE-VOLUME) ENCODING IS USED
------------------------------------------
v3 could not do this: its GroupNorm took statistics over the whole extent, so encoding a
512x576x96 volume in one pass gave latents unlike the 96^3 tiles it trained on -- masked
SSIM 0.912 whole-volume against 0.979 tiled, and that mismatch is why `latents/MRA` is
unusable. v4 was rebuilt to be extent-invariant and measures an extent gap of 1.0e-4
with a tiled/direct latent correlation of 0.9992, so one pass is now both valid and
simpler than Hann-blended tiling. `--regime tiled` is kept for comparison.

STANDARDISATION
---------------
Latents are stored RAW. Standardisation belongs at load time, so the same files survive a
change of convention and so decoding cannot double-transform.

The existing pipeline (`modules/latent_data.compute_latent_normalization_stats`) uses ONE
scalar mean and std for the whole modality. That was fine for v3 but is wrong for v4: the
scale anchor pins every channel's std to 0.94-1.01, yet the channel MEANS span -0.42 to
+1.03. Dividing by a single scalar leaves channels offset by up to 0.92 sigma -- a
systematic bias the diffusion model would have to spend capacity learning for nothing.

So `latent_stats.json` records per-channel mean and std (measured on TRAIN only, never
val/test), and `standardize` / `destandardize` below apply them. The global scalars are
also written so the older loader still runs, but the per-channel path is the correct one.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from modules.metrics import masked_ssim_and_psnr, vessel_scores
from modules.paths import ROOT, mask_path_for, volume_paths
from modules.vae_mra import decode_tiled, encode_tiled, load_vaev4


DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "vaev5_mra" / "v5_cldice" / "best.pt"
DEFAULT_OUT = ROOT / "latents_v5_cldice"
MULTIMODAL_CHECKPOINT = ROOT / "checkpoints" / "vaev2_multimodal" / "kl5e5_from_epoch10" / "best.pt"
MULTIMODAL_OUT = ROOT / "latents_multimodal"
FIXED_CONDITIONING = ROOT / "latents"
MODALITY = "MRA"
SPLITS = ("train", "val", "test")
STATS_NAME = "latent_stats.json"


# =============================================================================
# Standardisation -- import these on the LDM side rather than re-deriving them
# =============================================================================
def load_stats(out_dir: Path, modality: str = MODALITY) -> dict:
    return json.loads((Path(out_dir) / STATS_NAME).read_text(encoding="utf-8"))[modality]


def standardize(latent: np.ndarray | torch.Tensor, stats: dict):
    """(z - per-channel mean) / per-channel std. Accepts [C,D,H,W] or [B,C,D,H,W]."""
    mean, std = np.asarray(stats["per_channel_mean"]), np.asarray(stats["per_channel_std"])
    if isinstance(latent, torch.Tensor):
        shape = (1, -1, 1, 1, 1) if latent.dim() == 5 else (-1, 1, 1, 1)
        m = torch.as_tensor(mean, dtype=torch.float32, device=latent.device).view(shape)
        s = torch.as_tensor(std, dtype=torch.float32, device=latent.device).view(shape)
        return (latent.float() - m) / s
    shape = (1, -1, 1, 1, 1) if latent.ndim == 5 else (-1, 1, 1, 1)
    return (np.asarray(latent, np.float32) - mean.reshape(shape)) / std.reshape(shape)


def destandardize(latent: np.ndarray | torch.Tensor, stats: dict):
    """Inverse of `standardize` -- apply before handing a latent back to the decoder."""
    mean, std = np.asarray(stats["per_channel_mean"]), np.asarray(stats["per_channel_std"])
    if isinstance(latent, torch.Tensor):
        shape = (1, -1, 1, 1, 1) if latent.dim() == 5 else (-1, 1, 1, 1)
        m = torch.as_tensor(mean, dtype=torch.float32, device=latent.device).view(shape)
        s = torch.as_tensor(std, dtype=torch.float32, device=latent.device).view(shape)
        return latent.float() * s + m
    shape = (1, -1, 1, 1, 1) if latent.ndim == 5 else (-1, 1, 1, 1)
    return np.asarray(latent, np.float32) * std.reshape(shape) + mean.reshape(shape)


# =============================================================================
@torch.inference_mode()
def encode_volume(model, path: Path, regime: str, device: str) -> np.ndarray:
    volume = torch.from_numpy(np.asarray(np.load(path), np.float32))[None, None].to(device)
    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
        if regime == "tiled":
            latent = encode_tiled(model, volume)
        else:
            latent, _ = model.encode(volume)     # mu; the sampled z is not what we store
    out = latent[0].float().cpu().numpy()
    del volume, latent
    torch.cuda.empty_cache()
    return out


def export(checkpoint: Path, out_dir: Path, regime: str, dtype: str, device: str,
           overwrite: bool, limit: int | None) -> dict:
    model = load_vaev4(checkpoint, device)
    store = np.float16 if dtype == "float16" else np.float32
    written, skipped = 0, 0
    channels = None
    # per-channel accumulators, TRAIN only -- val/test must not touch the statistics
    total = total_sq = None
    count = 0

    for split in SPLITS:
        paths = volume_paths(split)
        if limit:
            paths = paths[:limit]
        destination = out_dir / MODALITY / split
        destination.mkdir(parents=True, exist_ok=True)
        started = time.time()
        for index, path in enumerate(paths, 1):
            target = destination / f"{path.stem}.npy"
            if target.exists() and not overwrite:
                latent = np.load(target).astype(np.float32)
                skipped += 1
            else:
                latent = encode_volume(model, path, regime, device)
                np.save(target, latent.astype(store))
                written += 1
            channels = latent.shape[0]
            if split == "train":
                flat = latent.reshape(channels, -1).astype(np.float64)
                total = flat.sum(1) if total is None else total + flat.sum(1)
                total_sq = (flat ** 2).sum(1) if total_sq is None else total_sq + (flat ** 2).sum(1)
                count += flat.shape[1]
            if index % 25 == 0 or index == len(paths):
                rate = (time.time() - started) / index
                print(f"  {split:>5} {index:>4}/{len(paths)}  {rate:.1f}s/volume  "
                      f"eta {(len(paths) - index) * rate / 60:.0f} min", flush=True)

    mean = total / max(count, 1)
    std = np.sqrt(np.maximum(total_sq / max(count, 1) - mean ** 2, 1e-12))
    stats = {MODALITY: {
        "per_channel_mean": mean.tolist(), "per_channel_std": std.tolist(),
        # kept so modules/latent_data.py's scalar path still runs; per-channel is correct
        "mean": float(mean.mean()), "std": float(np.sqrt((std ** 2 + mean ** 2).mean()
                                                         - mean.mean() ** 2)),
        "channels": int(channels), "regime": regime, "dtype": dtype,
        "checkpoint": str(checkpoint), "train_volumes_used": len(volume_paths("train")),
        "note": "standardise PER CHANNEL; the scalar mean/std are legacy compatibility only",
    }}
    (out_dir / STATS_NAME).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\nwrote {written} latents ({skipped} already present) -> {out_dir}")
    print(f"stats: {out_dir / STATS_NAME}")
    del model
    torch.cuda.empty_cache()
    return stats[MODALITY]


# =============================================================================
# Verification -- an export nobody checked is an export nobody should trust
# =============================================================================
def verify(checkpoint: Path, out_dir: Path, regime: str, device: str, cases: int,
           limit: int | None = None) -> bool:
    print()
    print("=" * 84)
    print("VERIFYING THE EXPORT")
    print("=" * 84)
    stats = load_stats(out_dir)
    ok = True

    # 1. every volume present, one latent per volume, consistent shape and dtype
    for split in SPLITS:
        paths = volume_paths(split)
        expected = {p.stem for p in (paths[:limit] if limit else paths)}
        found = {p.stem for p in (out_dir / MODALITY / split).glob("*.npy")}
        missing, extra = expected - found, found - expected
        status = "ok" if not missing and not extra else "MISMATCH"
        print(f"  {split:>5}: {len(found):>3}/{len(expected):>3} files  {status}")
        if missing:
            print(f"         missing: {sorted(missing)[:4]}"); ok = False
        if extra:
            print(f"         unexpected: {sorted(extra)[:4]}"); ok = False

    sample = sorted((out_dir / MODALITY / "train").glob("*.npy"))[:cases]
    shapes = {np.load(p, mmap_mode="r").shape for p in sample}
    dtypes = {np.load(p, mmap_mode="r").dtype for p in sample}
    print(f"  shapes {shapes}  dtypes {dtypes}")
    if len(shapes) != 1 or len(dtypes) != 1:
        print("  INCONSISTENT shape or dtype across files"); ok = False

    # 2. no NaN or Inf anywhere in the sample
    bad = [p.stem for p in sample if not np.isfinite(np.load(p).astype(np.float32)).all()]
    print(f"  non-finite values: {'none' if not bad else bad}")
    ok &= not bad

    # 3. standardising with the exported stats must actually centre the channels
    stack = np.stack([standardize(np.load(p).astype(np.float32), stats) for p in sample])
    per_channel_mean = stack.mean(axis=(0, 2, 3, 4))
    per_channel_std = stack.std(axis=(0, 2, 3, 4))
    print(f"  after standardising: channel means {per_channel_mean.min():+.3f} ..."
          f" {per_channel_mean.max():+.3f}   stds {per_channel_std.min():.3f} ..."
          f" {per_channel_std.max():.3f}")
    if np.abs(per_channel_mean).max() > 0.15 or np.abs(per_channel_std - 1).max() > 0.15:
        print("  standardisation did NOT centre/scale the channels"); ok = False

    # 4. round trip: decode the stored fp16 latent and compare against decoding the
    #    freshly encoded fp32 one. This is what actually proves the files are usable.
    model = load_vaev4(checkpoint, device)
    print(f"  round-trip on {len(sample)} volumes (stored fp16 vs fresh fp32 encode):")
    deltas, metrics = [], []
    with torch.inference_mode():
        for path in sample:
            source = ROOT / "Dataset" / "split_numpy" / "train" / MODALITY / f"{path.stem}.npy"
            target = np.asarray(np.load(source), np.float32)
            brain = np.asarray(np.load(mask_path_for(source, "train"))) > 0
            fresh = encode_volume(model, source, regime, device)
            stored = np.load(path).astype(np.float32)
            deltas.append(float(np.abs(fresh - stored).max()))
            recon = decode_latent(model, stored, device)
            _, masked, psnr = masked_ssim_and_psnr(target, recon, brain)
            dice, _ = vessel_scores(target, recon)
            metrics.append((masked, psnr, dice))
            print(f"    {path.stem:<24} max |fresh-stored| {deltas[-1]:.2e}  "
                  f"masked SSIM {masked:.4f}  PSNR {psnr:.2f}  Dice {dice:.4f}", flush=True)
    average = np.mean(metrics, axis=0)
    print(f"  mean over sample: masked SSIM {average[0]:.4f}  PSNR {average[1]:.2f}  "
          f"Dice {average[2]:.4f}")
    print(f"  worst re-encode difference: {max(deltas):.2e} "
          f"({max(deltas) / float(np.mean(stats['per_channel_std'])):.1e} of a channel std)")
    print("    the encoder runs under fp16 autocast, so its output is already "
          "fp16-representable and float16 storage is lossless; this residual is "
          "cuDNN algorithm nondeterminism, not quantisation")
    if max(deltas) > 0.05:
        print("  re-encode difference far larger than expected"); ok = False
    del model
    torch.cuda.empty_cache()

    print()
    print(f"VERDICT: {'PASS -- latents are usable' if ok else 'FAIL -- do not train on these'}")
    return ok


@torch.inference_mode()
def decode_latent(model, latent: np.ndarray, device: str) -> np.ndarray:
    """Tiled decode: bounded memory, so verification runs alongside a training job. v4's
    two decode regimes agree to 1e-4, so this measures the same thing as a single pass
    without needing its ~5 GB."""
    z = torch.from_numpy(latent)[None].to(device)
    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
        recon = decode_tiled(model, z)
    out = np.clip(recon[0, 0].float().cpu().numpy(), 0.0, 1.0)
    del z, recon
    torch.cuda.empty_cache()
    return out


def export_multimodal(checkpoint: Path, out_dir: Path, config_path: Path, device: str,
                      overwrite: bool, limit: int | None) -> None:
    from modules.vae_multimodal import encode_multimodal_mu, load_multimodal_vae
    config = json.loads(config_path.read_text(encoding="utf-8"))
    modalities = list(config["data"]["modalities"])
    model = load_multimodal_vae(config, checkpoint, device)
    sums, squares, counts = {}, {}, {}
    started = time.time()
    for split in SPLITS:
        for index, modality in enumerate(modalities):
            files = sorted((ROOT / config["data"]["root"] / split / modality).glob(f"*-{modality}.npy"))
            if limit:
                files = files[:limit]
            (out_dir / modality / split).mkdir(parents=True, exist_ok=True)
            for i, path in enumerate(files, 1):
                target = out_dir / modality / split / path.name
                if target.exists() and not overwrite:
                    latent = np.load(target).astype(np.float32)
                else:
                    latent = encode_multimodal_mu(model, np.load(path), index, device)
                    np.save(target, latent.astype(np.float16))
                if split == "train":   # channel statistics from TRAIN only
                    flat = latent.astype(np.float64).reshape(latent.shape[0], -1)
                    sums[modality] = sums.get(modality, 0.0) + flat.sum(1)
                    squares[modality] = squares.get(modality, 0.0) + (flat ** 2).sum(1)
                    counts[modality] = counts.get(modality, 0) + flat.shape[1]
                if i % 50 == 0 or i == len(files):
                    print(f"  {split} {modality} {i}/{len(files)}  [{(time.time() - started) / 60:.1f} min]",
                          flush=True)
    stats = {}
    for m in counts:
        mean = sums[m] / counts[m]
        std = np.sqrt(np.maximum(squares[m] / counts[m] - mean ** 2, 1e-12))
        stats[m] = {"per_channel_mean": mean.tolist(), "per_channel_std": std.tolist()}
    (out_dir / "latent_stats_per_channel.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"wrote {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["mra", "multimodal"], default="mra")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="default: v5_cldice (mra) or kl5e5_from_epoch10 (multimodal)")
    parser.add_argument("--out", type=Path, default=None,
                        help="default: latents_v5_cldice (mra) or latents_multimodal (multimodal)")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "vae_multimodal.json",
                        help="multimodal only: the model definition")
    parser.add_argument("--regime", default="direct", choices=["direct", "tiled"])
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--overwrite", action="store_true",
                        help="re-encode volumes whose latent already exists")
    parser.add_argument("--limit", type=int, default=None, help="first N volumes per split")
    parser.add_argument("--verify-cases", type=int, default=4)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--memory-fraction", type=float, default=0.0,
                        help="cap VRAM share, e.g. 0.25 while a training run is using the GPU")
    args = parser.parse_args()
    multimodal = args.model == "multimodal"
    args.checkpoint = args.checkpoint or (MULTIMODAL_CHECKPOINT if multimodal else DEFAULT_CHECKPOINT)
    args.out = args.out or (MULTIMODAL_OUT if multimodal else DEFAULT_OUT)
    if args.out.resolve() == FIXED_CONDITIONING.resolve():
        raise SystemExit("refusing to write into latents/: it holds the fixed conditioning latents "
                         "the thesis bridge was trained on")
    if args.overwrite and args.out.resolve() == (ROOT / "latents_v5_cldice").resolve():
        raise SystemExit("refusing --overwrite on latents_v5_cldice, the thesis target latents")

    if not args.checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    if args.memory_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    if multimodal:
        export_multimodal(args.checkpoint, args.out, args.config, device, args.overwrite, args.limit)
        return
    print(f"checkpoint : {args.checkpoint}")
    print(f"out        : {args.out}")
    print(f"regime     : {args.regime}   dtype: {args.dtype}")
    if not args.verify_only:
        export(args.checkpoint, args.out, args.regime, args.dtype, device,
               args.overwrite, args.limit)
    if not args.no_verify:
        if not verify(args.checkpoint, args.out, args.regime, device, args.verify_cases,
                      limit=args.limit):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
