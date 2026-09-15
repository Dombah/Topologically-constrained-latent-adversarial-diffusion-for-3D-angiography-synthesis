"""Validate any VAEv4-family checkpoint (v4 or a finetuned arm) and optionally save
visual previews.

    uv run evaluate_vae.py
    uv run evaluate_vae.py --checkpoint checkpoints/vaev5_mra/v5_cldice/best.pt --previews
    uv run evaluate_vae.py --split test --cases-per-scanner 8 --previews

WHAT THIS REPORTS THAT THE TRAINER DOES NOT
-------------------------------------------
1. Per-scanner metrics. IOP (GE) reconstructs far worse than Guys/HH (Philips) --
   masked SSIM 0.948 vs 0.997 for v4 -- and a single average hides it.
2. Both weightings. `pick_validation_cases` samples equally per scanner, so the
   trainer's numbers weight IOP at 33 % against its natural 12.5 %. The natural row is
   the one that belongs in the thesis; the equal row is what the acceptance gate saw.
3. clDice on every case rather than the first 8. `evaluate()` computes it only for
   `index < skeleton_cases`, and cases are ordered by scanner, so the trainer's clDice
   is measured on Guys alone -- v4's accepted 0.9393 is the Guys-only figure, against
   0.9246 natural-weighted.

While a training run is using the GPU, pass --memory-fraction 0.25 so this cannot
starve it.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from modules.metrics import (hard_cldice, latent_diagnostics, masked_ssim_and_psnr, mip_ssim,
                             vessel_scores)
from modules.paths import ROOT, mask_path_for, scanner_of, volume_paths
from modules.vae_mra import ACCEPT, load_vaev4, reconstruct


SCANNERS = ("Guys", "HH", "IOP")
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "vaev4_mra" / "v4_c16_fixed" / "best.pt"
METRICS = ("masked_ssim", "ssim", "psnr", "mip_ssim", "vessel_dice", "cldice", "vessel_l1")


def natural_weights(split: str) -> dict[str, float]:
    """Scanner proportions of the split itself, so the weighted row reflects the data
    rather than the sampling."""
    counts = defaultdict(int)
    for path in volume_paths(split):
        counts[scanner_of(path)] += 1
    total = sum(counts.values())
    return {s: counts.get(s, 0) / total for s in SCANNERS}


def select_cases(split: str, per_scanner: int) -> list[Path]:
    paths = volume_paths(split)
    chosen: list[Path] = []
    for scanner in SCANNERS:
        chosen += [p for p in paths if scanner_of(p) == scanner][:per_scanner]
    return chosen


@torch.inference_mode()
def score_case(model, path: Path, split: str, regime: str, with_cldice: bool):
    target = np.asarray(np.load(path), dtype=np.float32)
    brain = np.asarray(np.load(mask_path_for(path, split))) > 0
    volume = torch.from_numpy(target)[None, None].cuda()
    recon, latent = reconstruct(model, volume, regime)
    recon = np.clip(recon, 0.0, 1.0).astype(np.float32)
    ssim, masked, psnr = masked_ssim_and_psnr(target, recon, brain)
    dice, vessel_l1 = vessel_scores(target, recon)
    row = {"masked_ssim": masked, "ssim": ssim, "psnr": psnr,
           "mip_ssim": mip_ssim(target, recon), "vessel_dice": dice,
           "vessel_l1": vessel_l1,
           "cldice": hard_cldice(target, recon) if with_cldice else float("nan")}
    del volume
    torch.cuda.empty_cache()
    return row, target, recon, brain, latent


# =============================================================================
# Previews
# =============================================================================
def save_preview(target, recon, brain, latent, path: Path, row: dict, out_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    error = np.abs(target - recon)
    mid = [s // 2 for s in target.shape]
    panels = [
        ("axial MIP", target.max(axis=2), recon.max(axis=2), error.max(axis=2)),
        ("coronal MIP", target.max(axis=1), recon.max(axis=1), error.max(axis=1)),
        ("sagittal MIP", target.max(axis=0), recon.max(axis=0), error.max(axis=0)),
        ("mid axial slice", target[:, :, mid[2]], recon[:, :, mid[2]], error[:, :, mid[2]]),
    ]
    # row heights follow each panel's own aspect, or the thin coronal/sagittal MIPs
    # leave most of the figure blank
    ratios = [p[1].shape[1] / p[1].shape[0] for p in panels]   # after the rot90 below
    width = 10.5
    figure, axes = plt.subplots(len(panels), 3, figsize=(width, 0.6 + width / 3 * sum(ratios)),
                                gridspec_kw={"height_ratios": ratios})
    for r, (label, gt, rc, er) in enumerate(panels):
        # a shared scale for target/recon so differences are real, not autoscaled away
        top = float(max(gt.max(), rc.max())) or 1.0
        for c, (image, title, kwargs) in enumerate((
                (gt, "target", {"cmap": "gray", "vmin": 0, "vmax": top}),
                (rc, "reconstruction", {"cmap": "gray", "vmin": 0, "vmax": top}),
                (er, "|error| (x5)", {"cmap": "inferno", "vmin": 0, "vmax": top / 5.0}))):
            axis = axes[r, c]
            axis.imshow(np.rot90(image), **kwargs)
            axis.set_axis_off()
            if r == 0:
                axis.set_title(title, fontsize=10)
        axes[r, 0].set_ylabel(label)
        axes[r, 0].text(-0.04, 0.5, label, rotation=90, va="center", ha="right",
                        transform=axes[r, 0].transAxes, fontsize=9)
    figure.suptitle(f"{path.stem}   masked SSIM {row['masked_ssim']:.4f}  "
                    f"PSNR {row['psnr']:.2f}  vessel Dice {row['vessel_dice']:.4f}  "
                    f"clDice {row['cldice']:.4f}", fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    out = out_dir / f"{path.stem}.png"
    figure.savefig(out, dpi=110)
    plt.close(figure)

    # latent channels, mid slice -- the thing the LDM actually sees
    channels = latent.shape[0]
    columns = min(8, channels)
    rows = int(np.ceil(channels / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(2.0 * columns, 2.1 * rows))
    axes = np.atleast_2d(axes)
    z = latent.shape[-1] // 2
    for index in range(rows * columns):
        axis = axes[index // columns, index % columns]
        axis.set_axis_off()
        if index >= channels:
            continue
        plane = latent[index, :, :, z]
        limit = float(np.abs(plane).max()) or 1.0
        axis.imshow(np.rot90(plane), cmap="RdBu_r", vmin=-limit, vmax=limit)
        axis.set_title(f"ch {index}  sd {plane.std():.2f}", fontsize=8)
    figure.suptitle(f"{path.stem} -- latent channels, mid slice", fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    out_latent = out_dir / f"{path.stem}_latent.png"
    figure.savefig(out_latent, dpi=110)
    plt.close(figure)
    return out


# =============================================================================
def validate_vaev4(checkpoint: Path, split: str = "val", cases_per_scanner: int = 4,
                   regime: str = "tiled", previews: bool = False,
                   out_dir: Path | None = None, preview_cases: int = 1,
                   with_cldice: bool = True, device: str = "cuda") -> dict:
    """Score a checkpoint per scanner, both weightings, optionally writing previews.
    Returns the report dict and writes validation_report.json under out_dir."""
    out_dir = Path(out_dir) if out_dir else checkpoint.parent / f"validation_{split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = out_dir / "previews"
    if previews:
        preview_dir.mkdir(exist_ok=True)

    model = load_vaev4(checkpoint, device)
    cases = select_cases(split, cases_per_scanner)
    print(f"checkpoint : {checkpoint}")
    print(f"split      : {split}  ({len(cases)} cases, {cases_per_scanner} per scanner)")
    print(f"regime     : {regime}")
    print(f"clDice     : {'every case' if with_cldice else 'skipped'}", flush=True)

    per_scanner: dict[str, list[dict]] = defaultdict(list)
    latents: list[np.ndarray] = []
    saved_previews = defaultdict(int)
    for index, path in enumerate(cases, 1):
        scanner = scanner_of(path)
        row, target, recon, brain, latent = score_case(model, path, split, regime, with_cldice)
        per_scanner[scanner].append(row)
        latents.append(latent)
        print(f"  [{index:>2}/{len(cases)}] {path.stem:<24} masked {row['masked_ssim']:.4f} "
              f"PSNR {row['psnr']:>6.2f} Dice {row['vessel_dice']:.4f} "
              f"clDice {row['cldice']:.4f}", flush=True)
        if previews and saved_previews[scanner] < preview_cases:
            saved = save_preview(target, recon, brain, latent, path, row, preview_dir)
            saved_previews[scanner] += 1
            print(f"        preview -> {saved.name}", flush=True)

    means = {s: {m: float(np.nanmean([r[m] for r in rows])) for m in METRICS}
             for s, rows in per_scanner.items() if rows}
    weights = natural_weights(split)
    equal = {m: float(np.nanmean([means[s][m] for s in means])) for m in METRICS}
    total = sum(weights[s] for s in means) or 1.0
    natural = {m: float(sum(weights[s] * means[s][m] for s in means) / total) for m in METRICS}

    print()
    header = f"{'scanner':>9}{'n':>4}" + "".join(f"{m:>13}" for m in METRICS)
    print("=" * len(header))
    print(header)
    print("=" * len(header))
    for scanner in SCANNERS:
        if scanner in means:
            print(f"{scanner:>9}{len(per_scanner[scanner]):>4}"
                  + "".join(f"{means[scanner][m]:>13.4f}" for m in METRICS))
    print("-" * len(header))
    print(f"{'equal':>9}{'':>4}" + "".join(f"{equal[m]:>13.4f}" for m in METRICS)
          + "   <- acceptance-gate weighting")
    print(f"{'natural':>9}{'':>4}" + "".join(f"{natural[m]:>13.4f}" for m in METRICS)
          + f"   <- {', '.join(f'{s} {weights[s]:.1%}' for s in SCANNERS)}")

    diagnostics = latent_diagnostics(latents)
    print()
    print("latent: " + "  ".join(f"{k.replace('latent_', '')} {v:.4f}"
                                 for k, v in sorted(diagnostics.items())))

    # acceptance criteria, evaluated on the equal-weighted row the gate was defined for
    gate = {f"{regime}_{m}": equal[m] for m in METRICS}
    gate.update({f"{regime}_{k}": v for k, v in diagnostics.items()})
    print()
    print(f"{'criterion':>34}{'value':>12}{'threshold':>12}{'':>6}")
    passed = True
    for metric, (threshold, direction) in ACCEPT.items():
        if metric not in gate:
            continue
        value = gate[metric]
        if not np.isfinite(value):      # a skipped metric is not a failed one
            print(f"{metric:>34}{'--':>12}{threshold:>12.4f}{'  SKIP':>6}")
            continue
        ok = value >= threshold if direction == "ge" else value <= threshold
        passed &= ok
        print(f"{metric:>34}{value:>12.4f}{threshold:>12.4f}{'  PASS' if ok else '  FAIL':>6}")

    report = {"checkpoint": str(checkpoint), "split": split, "regime": regime,
              "cases": len(cases), "per_scanner": means, "equal": equal,
              "natural": natural, "natural_weights": weights,
              "latent": diagnostics, "acceptance_passed": bool(passed)}
    (out_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2, default=float), encoding="utf-8")
    print()
    print(f"written: {out_dir / 'validation_report.json'}")
    if previews:
        print(f"previews: {preview_dir}  ({sum(saved_previews.values())} cases)")
    del model
    torch.cuda.empty_cache()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--cases-per-scanner", type=int, default=4)
    parser.add_argument("--regime", default="tiled", choices=["tiled", "direct"])
    parser.add_argument("--previews", action="store_true",
                        help="save target/reconstruction/error MIPs and a latent montage")
    parser.add_argument("--preview-cases", type=int, default=1,
                        help="how many cases per scanner to render (default 1)")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-cldice", action="store_true",
                        help="skip clDice (skeletonisation costs ~15 s per case)")
    parser.add_argument("--memory-fraction", type=float, default=0.0,
                        help="cap this process's VRAM share, e.g. 0.25 while a run trains")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    if args.memory_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)

    validate_vaev4(args.checkpoint, split=args.split,
                   cases_per_scanner=args.cases_per_scanner, regime=args.regime,
                   previews=args.previews, out_dir=args.out,
                   preview_cases=args.preview_cases, with_cldice=not args.no_cldice)


if __name__ == "__main__":
    main()
