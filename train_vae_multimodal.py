"""Train the multimodal (T1/T2/PD) VAE whose latents condition the Brownian bridge.

    uv run train_vae_multimodal.py --smoke          # a few steps + one validation volume, in checkpoints/_smoke
    uv run train_vae_multimodal.py                  # full run (~23 h on an RTX 5080), resumable

Three modality-specific encoders and decoders share one 8-channel latent space; every patch
is presented with a one-hot modality mask. Loss: L1 reconstruction + KL (linear warm-up) +
a latent-alignment term that pulls the three modalities of the same patch together.

All settings are in configs/vae_multimodal.json and reproduce the thesis checkpoint
checkpoints/vaev2_multimodal/kl5e5_from_epoch10/best.pt (best by validation SSIM, epoch 75).
After training, export the conditioning latents:

    uv run export_latents.py --model multimodal --checkpoint checkpoints/rebuild/vae_multimodal/<run>/best.pt
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from modules.paths import REBUILD_ROOT, ROOT, refuse_protected
from modules.vae_multimodal import (VAEv2Loss, VAEv2PairedMultimodalPatchDataset, build_multimodal_vae,
                                    train_vaev2_multimodal)


def build_dataset(cfg: dict, split: str, n: int = -1) -> VAEv2PairedMultimodalPatchDataset:
    d = cfg["data"]
    root = ROOT / d["root"] / split
    return VAEv2PairedMultimodalPatchDataset(
        image_dirs={m: str(root / m) for m in d["modalities"]},
        mask_dirs={m: str(root / "masks" / m) for m in d["modalities"]},
        modalities=tuple(d["modalities"]), patch_size=tuple(d["patch_size"]),
        target_shape=tuple(d["target_shape"]), patches_per_patient=int(d["patches_per_patient"]),
        brain_patch_ratio=float(d["brain_patch_ratio"]), cache_dir=str(ROOT / d["cache_dir"] / split),
        n=n, image_extensions=(".npy",), normalize_mode=d["normalize_mode"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "vae_multimodal.json")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="default checkpoints/rebuild/vae_multimodal/<run_name>")
    parser.add_argument("--resume", type=Path, default=None, help="checkpoint to resume from")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    t, loss = cfg["training"], cfg["loss"]
    run_dir = args.run_dir or REBUILD_ROOT / "vae_multimodal" / cfg["run_name"]
    if args.smoke:
        run_dir = args.run_dir or ROOT / "checkpoints" / "_smoke" / "vae_multimodal"
    run_dir = refuse_protected(run_dir, "the multimodal VAE run")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    random.seed(t["seed"]); np.random.seed(t["seed"]); torch.manual_seed(t["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = 2 if args.smoke else -1
    train_set, val_set = build_dataset(cfg, "train", n), build_dataset(cfg, "val", 1 if args.smoke else -1)
    model = build_multimodal_vae(cfg)
    print(f"multimodal VAE: {sum(p.numel() for p in model.parameters())/1e6:.3f} M parameters, "
          f"{len(train_set)} patches per epoch -> {run_dir}", flush=True)

    train_vaev2_multimodal(
        model=model, dataset=train_set, val_dataset=val_set,
        criterion=VAEv2Loss(kl_weight=float(loss["kl_start_weight"]),
                            latent_alignment_weight=float(loss["latent_alignment_weight"])),
        num_epochs=1 if args.smoke else int(t["epochs"]), batch_size=int(t["batch_size"]),
        lr=float(t["lr"]), weight_decay=float(t["weight_decay"]), num_workers=int(t["num_workers"]),
        device=device, checkpoint_dir=str(run_dir), resume_path=str(args.resume) if args.resume else None,
        checkpoint_every=int(t["checkpoint_every"]), val_every=1 if args.smoke else int(t["val_every"]),
        max_train_batches=4 if args.smoke else None, max_val_batches=1 if args.smoke else None,
        best_metric_name=t["best_metric_name"], best_metric_mode="max",
        kl_start_weight=float(loss["kl_start_weight"]), kl_final_weight=float(loss["kl_final_weight"]),
        kl_warmup_epochs=int(loss["kl_warmup_epochs"]),
        kl_warmup_start_epoch=int(loss["kl_warmup_start_epoch"]),
        use_amp=bool(t["use_amp"]), grad_clip=float(t["grad_clip"]))


if __name__ == "__main__":
    main()
