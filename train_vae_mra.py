"""VAEv4: MRA autoencoder trained to be extent-invariant and diffusable.

Run it with no arguments:

    uv run train_vae_mra.py

It trains until whichever comes first: --epochs or --hours (default 9 h), then runs the
acceptance evaluation and writes a PASS/FAIL report. Ctrl-C saves and exits cleanly;
re-running resumes from latest.pt.

WHY v4 EXISTS (see REVIEW_2026-09-02.md)
----------------------------------------
v3 reconstructs beautifully in the regime it was trained in (96^3 tiles: brain-masked
SSIM 0.979, vessel Dice 0.89) and much worse in the regime it is deployed in (whole
volume: 0.912 / 0.66). Its GroupNorms use one or two channels per group, i.e. instance
normalisation whose statistics are taken over the whole input extent, so a 512x576x96
volume with 64 % zero background produces activations unlike anything seen in training.
Its latents are also hard to diffuse: kurtosis 8.4, tails to 42 sigma, per-channel std
0.40-0.92 under one global scalar, and 65 % of the power above half-Nyquist.

Four changes, none of which grows the model (1.08 M parameters, same as v3):

  1. Extent-invariant normalisation. Channel RMS norm normalises across channels at each
     voxel, so its statistics cannot depend on the size of the input. Training also draws
     patches at five different extents, from 64^3 to 192x192x96, including background-only
     crops. Validation scores tiled AND whole-volume every cycle: the gap between them is
     the thing v4 exists to close.
  2. Latent scale and tails. An adaptive KL controller holds the per-channel latent std
     near 1 (MAISI's criterion) instead of guessing a weight, and a tail penalty pushes
     down excursions beyond 4 sigma, which is where the diffusion loss currently spends
     most of its error budget. Per-channel standardisation statistics are exported for the
     LDM stage.
  3. Equivariance / diffusability regularisation. EQ-VAE (Kouzelis 2025) and Skorokhodov
     2025: decode a rescaled and rotated latent and ask it to match the equally transformed
     image. This is what removes the high-frequency latent noise floor.
  4. 8 latent channels instead of 16, and a 2.5-D VGG perceptual term. Fewer channels are
     easier to model (Yao 2025); the perceptual term pays for the lost capacity where it
     matters. If the acceptance report fails on vessel Dice, rerun with --channels 16.

The acceptance gate at the end is the decision rule from the review: if v4 does not clear
it, keep v3 with tiled Hann decoding and spend the GPU time on the LDM instead.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path


import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from modules.metrics import masked_ssim_and_psnr
from modules.paths import (REBUILD_ROOT, refuse_protected, append_jsonl, mask_path_for,
                           pick_validation_cases)
from modules.vae_mra import (CACHE_ROOT, DECAY_START_EPOCH, EQ_SCALES, FINAL_CASES,
                             FINAL_SKELETON_CASES, GRAD_CLIP, KL_ADJUST_RATE, KL_BOUNDS, KL_INIT,
                             KL_STD_TOLERANCE, KL_TARGET_STD, KL_WARMUP_EPOCHS, LATENT_STATS_CASES,
                             LR, MRACropDataset, MultiExtentBatchSampler, P_EQUIVARIANCE,
                             Perceptual25D, SIZE_SPECS, STEPS_PER_EPOCH, TAIL_SIGMA,
                             V3_REFERENCE_FROM_TEST_SPLIT, VAEv4, VAL_CASES_PER_SCANNER, VAL_EVERY,
                             WARMUP_STEPS, WEIGHT_DECAY, W_BACKGROUND, W_EQUIVARIANCE, W_L1, W_MIP,
                             W_PERCEPTUAL, W_SCALE, W_TAIL, W_VESSEL_L1, acceptance_report,
                             evaluate, export_latent_channel_stats, mip_l1, reconstruct,
                             save_checkpoint, selection_score, transform_pair, vessel_weight_map)


                           # Set it to the resume epoch to append a decay phase to a run that
                           # was trained at a constant rate (warmup-stable-decay).


def train(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_dir = refuse_protected(args.run_dir or REBUILD_ROOT / "vae_mra" / args.run_name, "the VAE run")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run directory: {run_dir}")

    model = VAEv4(latent_channels=args.channels, norm=args.norm).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"VAEv4: {parameters/1e6:.3f} M parameters, {args.channels} latent channels, norm={args.norm}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    perceptual = None
    if not args.no_perceptual:
        try:
            perceptual = Perceptual25D().to(device)
            print("perceptual loss: 2.5-D VGG16 (relu1_2, relu2_2, relu3_3)")
        except Exception as error:  # missing weights, no network: train without it
            print(f"perceptual loss disabled ({type(error).__name__}: {error})")

    start_epoch, best_metric, kl_weight = 0, -float("inf"), (KL_INIT if args.kl_adaptive else args.kl_weight)
    latest_path = run_dir / "latest.pt"
    if latest_path.exists() and not args.scratch:
        blob = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(blob["model_state_dict"])
        if blob.get("optimizer_state_dict"):
            optimizer.load_state_dict(blob["optimizer_state_dict"])
        start_epoch = int(blob.get("epoch", 0))
        best_metric = float(blob.get("best_metric", -float("inf")))
        kl_weight = float(blob.get("kl_weight", KL_INIT))
        print(f"resumed from {latest_path} at epoch {start_epoch} (best {best_metric:.5f})")

    cached = len(list(CACHE_ROOT.glob("*.npz"))) if CACHE_ROOT.exists() else 0
    print(f"loading crop caches ({cached} already built; the first run builds ~455, "
          f"about 5 minutes) ...", flush=True)
    dataset = MRACropDataset("train", seed=args.seed)
    sampler = MultiExtentBatchSampler(len(dataset), steps=args.steps, seed=args.seed)
    validation_cases = pick_validation_cases("val", VAL_CASES_PER_SCANNER)
    print(f"{len(dataset)} training volumes, {len(validation_cases)} validation cases, "
          f"{args.steps} steps/epoch")

    config = {"latent_channels": args.channels, "norm": args.norm, "lr": args.lr, "steps": args.steps,
              "lr_schedule": {"warmup_steps": WARMUP_STEPS, "decay_start_epoch": args.decay_start_epoch,
                              "shape": "cosine_to_zero"},
              "epochs": args.epochs, "hours": args.hours, "size_specs": [list(spec) for spec in SIZE_SPECS],
              "weights": {"l1": W_L1, "vessel_l1": W_VESSEL_L1, "mip": W_MIP, "background": W_BACKGROUND,
                          "perceptual": 0.0 if perceptual is None else args.perceptual_weight,
                          "equivariance": W_EQUIVARIANCE, "tail": W_TAIL, "scale": W_SCALE,
                          "noise_aug": args.noise_aug, "noise_aug_weight": args.noise_aug_weight},
              # record the weight actually used, not the module default: config.json disagreeing
              # with the run is the exact failure mode flagged as A5 in REVIEW_2026-09-02.md
              "kl": {"weight": args.kl_weight, "adaptive": bool(args.kl_adaptive),
                     "target_std": KL_TARGET_STD, "bounds": list(KL_BOUNDS)},
              "parameters": int(parameters), "gan": bool(args.gan)}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    workers = args.workers
    started = time.time()
    deadline = started + args.hours * 3600.0
    global_step = start_epoch * args.steps
    total_steps = args.epochs * args.steps
    decay_from_step = min(max(args.decay_start_epoch, 0), args.epochs) * args.steps
    last_completed_epoch = start_epoch   # so an interrupt cannot record an unfinished epoch
    stop_reason = "epochs"

    try:
        for epoch in range(start_epoch + 1, args.epochs + 1):
            sampler.set_epoch(epoch)
            while True:
                try:
                    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=workers,
                                        pin_memory=(device == "cuda"), persistent_workers=False)
                    iterator = iter(loader)
                    break
                except (OSError, RuntimeError) as error:
                    if workers == 0:
                        raise
                    print(f"dataloader workers failed ({type(error).__name__}); falling back to 0")
                    workers = 0

            totals: dict[str, float] = {}
            steps = 0
            epoch_started = time.time()
            latent_std_accumulator = 0.0
            for batch in iterator:
                image = batch["image"].to(device, non_blocking=True)
                brain = batch["mask"].to(device, non_blocking=True)
                p99 = batch["p99"].to(device, non_blocking=True)
                p999 = batch["p999"].to(device, non_blocking=True)

                warm = min(1.0, (global_step + 1) / max(WARMUP_STEPS, 1))
                progress = (global_step - decay_from_step) / max(total_steps - decay_from_step, 1)
                cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
                learning_rate = args.lr * warm * cosine
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate

                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                    recon, mu, logvar, z = model(image)
                    weight = vessel_weight_map(image, p99, p999)
                    l1 = (weight * (recon - image).abs()).mean()
                    loss = W_L1 * l1
                    terms = {"l1": l1.detach()}

                    if W_MIP > 0:
                        term = mip_l1(recon, image)
                        loss = loss + W_MIP * term
                        terms["mip"] = term.detach()
                    if W_BACKGROUND > 0:
                        outside = brain <= 0.5
                        term = (F.relu(recon - image)[outside].mean() if outside.any()
                                else recon.new_zeros(()))
                        loss = loss + W_BACKGROUND * term
                        terms["background"] = term.detach()
                    if perceptual is not None and args.perceptual_weight > 0:
                        term = perceptual(recon, image)
                        loss = loss + args.perceptual_weight * term
                        terms["perceptual"] = term.detach()

                    # KL, split so the controller can watch the scale of mu directly
                    mu32, logvar32 = mu.float(), logvar.float()
                    kl_mu = 0.5 * mu32.pow(2).mean()
                    kl_var = 0.5 * (logvar32.exp() - 1.0 - logvar32).mean()
                    loss = loss + kl_weight * (kl_mu + kl_var)
                    terms["kl"] = (kl_mu + kl_var).detach()

                    channel_std = mu32.transpose(0, 1).reshape(mu32.shape[1], -1).std(dim=1)
                    # Anchor the per-channel scale at 1. This is MAISI's criterion, and it also
                    # removes a degenerate escape route: the c16 run's tail penalty normalised by
                    # a DETACHED std, so shrinking mu lowered it within a step while the ratio
                    # reset the next step. The latent shrank 4000x, the penalty got worse (2.5 ->
                    # 7.2) and reconstruction never started. Never normalise a penalty by a
                    # detached statistic of the quantity being penalised.
                    if W_SCALE > 0:
                        scale_term = channel_std.clamp_min(1e-8).log().pow(2).mean()
                        loss = loss + W_SCALE * scale_term
                        terms["scale"] = scale_term.detach()
                    if W_TAIL > 0:
                        # std is pinned at 1, so this is an absolute threshold with nothing to game
                        tail = F.relu(mu32.abs() - TAIL_SIGMA).pow(2).mean()
                        loss = loss + W_TAIL * tail
                        terms["tail"] = tail.detach()

                    # Decoder robustness to off-manifold latents. The collapsed posterior
                    # injects none (std 0.0067), while the diffusion sampler's latents sit
                    # ~1 sigma from the truth, far outside anything the decoder trains on.
                    # Added as an extra term so the clean reconstruction objective, and the
                    # metric the acceptance gate reads, are unchanged.
                    if args.noise_aug > 0:
                        sigma = random.uniform(0.0, args.noise_aug)
                        perturbed = z + torch.randn_like(z) * sigma * channel_std.detach().view(1, -1, 1, 1, 1)
                        term = F.l1_loss(model.decode(perturbed), image)
                        loss = loss + args.noise_aug_weight * term
                        terms["noise_aug"] = term.detach()

                    if W_EQUIVARIANCE > 0 and random.random() < P_EQUIVARIANCE:
                        scale = random.choice(EQ_SCALES)
                        rotations = random.choice((0, 1, 2, 3))
                        z_t, x_t = transform_pair(z, image, scale, rotations)
                        term = (recon.new_tensor(0.0) if min(z_t.shape[-3:]) < 4
                                else F.l1_loss(model.decode(z_t), x_t))
                        loss = loss + W_EQUIVARIANCE * term
                        terms["equivariance"] = term.detach()

                if not torch.isfinite(loss):
                    print(f"non-finite loss at epoch {epoch} step {steps + 1}; skipping batch")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                if torch.isfinite(grad_norm):
                    optimizer.step()

                latent_std_accumulator += float(channel_std.mean().detach())
                totals["loss"] = totals.get("loss", 0.0) + float(loss.detach())
                for key, value in terms.items():
                    totals[key] = totals.get(key, 0.0) + float(value)
                steps += 1
                global_step += 1

                if steps % 100 == 0:
                    elapsed = time.time() - started
                    print(f"  epoch {epoch} step {steps}/{args.steps} loss {totals['loss']/steps:.4f} "
                          f"latent_std {latent_std_accumulator/steps:.3f} kl_w {kl_weight:.2e} "
                          f"[{elapsed/3600:.2f} h]", flush=True)

            mean_latent_std = latent_std_accumulator / max(steps, 1)
            # Caveat observed in the c8 run: once the controller drives kl_weight to its lower
            # bound the KL term stops being the binding constraint and the latent scale is set
            # by the reconstruction objective instead, so the controller loses authority and
            # the std settles wherever it likes (0.23 there, not the 1.0 target). That is not
            # harmful because the LDM standardises with the exported per-channel statistics,
            # but do not read a converged kl_weight of 1e-7 as "the target was met".
            if args.kl_adaptive and epoch > KL_WARMUP_EPOCHS:  # hold the latent scale near 1
                if mean_latent_std > KL_TARGET_STD + KL_STD_TOLERANCE:
                    kl_weight = min(kl_weight * KL_ADJUST_RATE, KL_BOUNDS[1])
                elif mean_latent_std < KL_TARGET_STD - KL_STD_TOLERANCE:
                    kl_weight = max(kl_weight / KL_ADJUST_RATE, KL_BOUNDS[0])

            record = {"epoch": epoch, "steps": steps, "time_sec": time.time() - epoch_started,
                      "lr": optimizer.param_groups[0]["lr"], "kl_weight": kl_weight,
                      "latent_std": mean_latent_std,
                      **{key: value / max(steps, 1) for key, value in totals.items()}}
            append_jsonl(run_dir / "training.jsonl", record)
            print(f"epoch {epoch}: loss {record.get('loss', float('nan')):.4f} "
                  f"latent_std {mean_latent_std:.3f} kl_w {kl_weight:.2e} "
                  f"({record['time_sec']:.0f}s)", flush=True)
            save_checkpoint(latest_path, model, optimizer, epoch, best_metric, kl_weight)
            last_completed_epoch = epoch

            out_of_time = time.time() > deadline
            if epoch % args.val_every == 0 or epoch == args.epochs or out_of_time:
                summary = evaluate(model, validation_cases, "val", device)
                score = selection_score(summary)
                summary.update({"epoch": epoch, "selection_score": score})
                append_jsonl(run_dir / "validation.jsonl", summary)
                print(f"  validation: tiled masked SSIM {summary.get('tiled_masked_ssim', float('nan')):.4f} "
                      f"vessel Dice {summary.get('tiled_vessel_dice', float('nan')):.3f} | "
                      f"direct {summary.get('direct_masked_ssim', float('nan')):.4f} / "
                      f"{summary.get('direct_vessel_dice', float('nan')):.3f} | "
                      f"gap {summary.get('extent_gap_masked_ssim', float('nan')):.4f} | "
                      f"score {score:.4f}", flush=True)
                if score > best_metric:
                    best_metric = score
                    save_checkpoint(run_dir / "best.pt", model, optimizer, epoch, best_metric, kl_weight,
                                    extra={"validation": summary})
                    print(f"  new best {best_metric:.5f} at epoch {epoch}", flush=True)

            if epoch % args.checkpoint_every == 0:
                save_checkpoint(run_dir / f"epoch_{epoch:04d}.pt", model, None, epoch, best_metric, kl_weight)
            if out_of_time:
                stop_reason = "time budget"
                print(f"stopping after epoch {epoch}: {args.hours} h budget reached", flush=True)
                break
    except KeyboardInterrupt:
        stop_reason = "interrupted"
        print("\ninterrupted; saving and skipping the final evaluation", flush=True)
        # `epoch` is the one in progress, which did not finish; recording it would make the
        # next resume skip it entirely.
        save_checkpoint(latest_path, model, optimizer, last_completed_epoch, best_metric, kl_weight)
        return

    # -------------------------------------------------------------- final report
    print(f"\ntraining finished ({stop_reason}) after {(time.time() - started)/3600:.2f} h; "
          f"running the acceptance evaluation", flush=True)
    best_path = run_dir / "best.pt" if (run_dir / "best.pt").exists() else latest_path
    blob = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(blob["model_state_dict"])
    final_cases = pick_validation_cases("val", FINAL_CASES // 3)
    summary = evaluate(model, final_cases, "val", device, skeleton_cases=FINAL_SKELETON_CASES)
    stats = export_latent_channel_stats(model, device, run_dir, LATENT_STATS_CASES)
    passed, rows = acceptance_report(summary)

    report = {"run": args.run_name, "checkpoint": str(best_path), "epoch": int(blob.get("epoch", -1)),
              "cases": len(final_cases), "stop_reason": stop_reason, "hours": (time.time() - started) / 3600.0,
              "passed": bool(passed), "criteria": rows, "summary": summary, "latent_channel_stats": stats}
    (run_dir / "acceptance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"VAEv4 ACCEPTANCE  ({best_path.name}, epoch {report['epoch']}, {len(final_cases)} val cases)")
    print("=" * 78)
    print(f"{'metric':>28}{'v4':>10}{'threshold':>12}{'v3':>10}   verdict")
    for row in rows:
        arrow = ">=" if row["direction"] == "ge" else "<="
        marker = "*" if row["metric"] in V3_REFERENCE_FROM_TEST_SPLIT else " "
        v3 = "-" if row["v3"] is None else f"{row['v3']:.4f}{marker}"
        print(f"{row['metric']:>28}{row['value']:>10.4f}{arrow + ' ' + format(row['threshold'], '.4f'):>12}"
              f"{v3:>11}  {'PASS' if row['pass'] else 'FAIL'}")
    print(f"{'':>28}{'':>10}{'':>12}{'* test split':>11}")
    print("-" * 78)
    print(f"tiled  masked SSIM {summary.get('tiled_masked_ssim', float('nan')):.4f}  "
          f"PSNR {summary.get('tiled_psnr', float('nan')):.2f}  "
          f"vessel Dice {summary.get('tiled_vessel_dice', float('nan')):.3f}  "
          f"MIP SSIM {summary.get('tiled_mip_ssim', float('nan')):.4f}")
    print(f"direct masked SSIM {summary.get('direct_masked_ssim', float('nan')):.4f}  "
          f"PSNR {summary.get('direct_psnr', float('nan')):.2f}  "
          f"vessel Dice {summary.get('direct_vessel_dice', float('nan')):.3f}  "
          f"latent corr {summary.get('extent_latent_correlation', float('nan')):.4f}")
    print(f"latent per-channel std {min(stats['per_channel_std']):.3f}..{max(stats['per_channel_std']):.3f}  "
          f"kurtosis {summary.get('tiled_latent_kurtosis', float('nan')):.2f}  "
          f"HF power {summary.get('tiled_latent_high_freq_fraction', float('nan')):.3f}")
    print("=" * 78)
    print("VERDICT: " + ("PASS - adopt v4; re-export latents and retrain the LDM on them."
                         if passed else
                         "FAIL - keep v3 with tiled Hann decoding and spend the GPU time on the LDM.\n"
                         "         If only vessel Dice failed, rerun with --channels 16."))
    print(f"written: {run_dir / 'acceptance.json'}")


def smoke(args) -> None:
    """A couple of minutes on real data: shapes, both regimes, and that the loss moves."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VAEv4(latent_channels=args.channels, norm=args.norm).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters())/1e6:.3f} M")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    dataset = MRACropDataset("train", seed=0)
    sampler = MultiExtentBatchSampler(len(dataset), steps=args.smoke_steps, seed=0)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    perceptual = None if args.no_perceptual else Perceptual25D().to(device)
    first = last = None
    for step, batch in enumerate(loader):
        image = batch["image"].to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            recon, mu, logvar, z = model(image)
            weight = vessel_weight_map(image, batch["p99"].to(device), batch["p999"].to(device))
            loss = (weight * (recon - image).abs()).mean() + W_MIP * mip_l1(recon, image)
            if perceptual is not None:
                loss = loss + W_PERCEPTUAL * perceptual(recon, image)
            z_t, x_t = transform_pair(z, image, 0.5, 1)
            loss = loss + W_EQUIVARIANCE * F.l1_loss(model.decode(z_t), x_t)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        value = float(loss.detach())
        first = value if first is None else first
        last = value
        if step % 5 == 0:
            print(f"  step {step:3d} {tuple(image.shape)} loss {value:.4f} "
                  f"latent {tuple(mu.shape[1:])} peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)
    print(f"loss {first:.4f} -> {last:.4f}")
    case = pick_validation_cases("val", 1)[0]
    target = np.asarray(np.load(case), dtype=np.float32)
    brain = np.asarray(np.load(mask_path_for(case, "val"))) > 0
    volume = torch.from_numpy(target)[None, None].to(device)
    for regime in ("tiled", "direct"):
        started = time.time()
        recon, latent = reconstruct(model, volume, regime)
        _, masked, psnr = masked_ssim_and_psnr(target, recon, brain)
        print(f"  {regime:>6}: recon {recon.shape} latent {latent.shape} masked SSIM {masked:.4f} "
              f"PSNR {psnr:.2f} ({time.time()-started:.1f}s)")
    print("smoke test finished (untrained network, so the numbers only prove the plumbing).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VAEv4 for the MRA latent-diffusion pipeline.")
    parser.add_argument("--run-name", default="v4_c16")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="default checkpoints/rebuild/vae_mra/<run-name>")
    parser.add_argument("--channels", type=int, default=16, help="latent channels (the thesis model uses 16)")
    parser.add_argument("--norm", default="channel_rms", choices=["channel_rms", "group"])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--hours", type=float, default=9.0, help="wall-clock budget; stops cleanly and evaluates")
    parser.add_argument("--steps", type=int, default=STEPS_PER_EPOCH)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--decay-start-epoch", type=int, default=DECAY_START_EPOCH,
                        help="epoch at which the cosine decay to 0 begins (0 = decay across the "
                             "whole run). Set it to the epoch you resume at to keep the rate "
                             "continuous across the restart.")
    parser.add_argument("--val-every", type=int, default=VAL_EVERY)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kl-weight", type=float, default=1e-7,
                        help="fixed KL weight. This loss mean-reduces BOTH the KL and the "
                             "reconstruction term, so the number is not comparable with the 1e-6/1e-7 "
                             "quoted elsewhere. Converted to this convention: Stable Diffusion ~2e-8 "
                             "(sum/sum), MONAI 3D-LDM ~1.4e-2 (mean recon vs SUM kl, which multiplies "
                             "the KL by the latent element count), v3 here 1e-3. Judge it by the KL's "
                             "measured share of the objective and by whether logvar sits on its clamp")
    parser.add_argument("--kl-adaptive", action="store_true",
                        help="restore the std-targeting controller; in the c8 run it pinned at its "
                             "lower bound by epoch 40 and lost authority over the latent scale")
    parser.add_argument("--noise-aug", type=float, default=0.0,
                        help="max per-channel sigma of latent noise before an extra decode; "
                             "trains the decoder for the off-manifold latents the sampler "
                             "produces. Try 0.5 once the channel count is settled (costs ~30%% "
                             "step time and confounds a clean channel comparison)")
    parser.add_argument("--noise-aug-weight", type=float, default=0.25)
    parser.add_argument("--no-perceptual", action="store_true")
    parser.add_argument("--perceptual-weight", type=float, default=W_PERCEPTUAL,
                        help="at the default 0.05 the VGG term was ~50%% of the c8 objective; "
                             "lower it to trade perceptual similarity for voxel fidelity")
    parser.add_argument("--gan", action="store_true", help="reserved; the v4 run is GAN-free by default")
    parser.add_argument("--scratch", action="store_true", help="ignore latest.pt and start over")
    parser.add_argument("--smoke", action="store_true", help="short plumbing check, then exit")
    parser.add_argument("--smoke-steps", type=int, default=30)
    args = parser.parse_args()

    if args.gan:
        print("note: --gan is not implemented in this script; run the P2 fine-tune separately.")
    if args.smoke:
        smoke(args)
        return
    train(args)


if __name__ == "__main__":
    main()
