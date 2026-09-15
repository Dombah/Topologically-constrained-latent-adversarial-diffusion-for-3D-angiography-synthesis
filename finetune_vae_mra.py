"""Finetune the accepted VAEv4 with topology (soft clDice) and adversarial terms.

Run the whole ablation with no arguments:

    uv run finetune_vae_mra.py

That trains three arms sequentially from the same v4 checkpoint -- cldice, adv, both --
then scores each one, including under latent perturbation. v4 itself is the fourth
(no-op) cell of the table and is re-scored for free.

One experiment at a time, or a chosen subset, with an explicit epoch budget:

    uv run finetune_vae_mra.py --arms cldice -n 30
    uv run finetune_vae_mra.py --arms adv both -n 30
    uv run finetune_vae_mra.py --arms all -n 30          # same as no arguments

-n is the budget PER ARM. Give every arm the same number: arms trained for different
numbers of epochs are not comparable, and comparing them is the point of the table.

Each arm has its own run directory and its own latest.pt, so arms are independent:
re-running an arm resumes it, and running one arm never touches another's checkpoints.
`--baseline-only` re-scores v4 alone and trains nothing.

WHY A FINETUNE RATHER THAN THREE RUNS FROM SCRATCH
--------------------------------------------------
All arms start from checkpoints/vaev4_mra/v4_c16_fixed/best.pt, so the difference
between them is the added objective and not a different optimisation trajectory. A
from-scratch comparison would confound the two.

WHY THE PERTURBED EVALUATION EXISTS
-----------------------------------
v4 already reconstructs at masked SSIM 0.9911 / clDice 0.9246 (natural weighting), so
clean-reconstruction metrics have very little headroom and may not separate the arms
above noise across 23 cases. The decoder's actual downstream job is to decode latents
produced by the diffusion sampler, which are never exact. `evaluate_perturbed` adds
Gaussian noise to mu at several sigmas and measures vessel Dice and clDice there. A
topology-aware decoder should hold up better as sigma grows even if the clean numbers
are indistinguishable, and that is the claim the paper needs.

WHAT IS DELIBERATELY UNCHANGED FROM v4
--------------------------------------
Every v4 loss term (l1, vessel-weighted L1, MIP, background, perceptual, KL, scale
anchor, tail penalty, equivariance) is kept at the same weight, and `selection_score`
is the same function, so arms stay comparable with each other and with v4. Only the
clDice and adversarial terms are added.
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

from modules.adversarial import PatchDiscriminator3D, adaptive_weight, hinge_d_loss, hinge_g_loss
from modules.paths import REBUILD_ROOT, refuse_protected, ROOT, append_jsonl, pick_validation_cases
from modules.topology import CLDICE_ITERS, soft_cldice_loss, vessel_probability_percentile
from modules.vae_mra import (PERTURB_SIGMAS, evaluate_perturbed, EQ_SCALES, GRAD_CLIP,
                             MRACropDataset, MultiExtentBatchSampler, P_EQUIVARIANCE, Perceptual25D,
                             STEPS_PER_EPOCH, TAIL_SIGMA, VAL_EVERY, W_BACKGROUND, W_EQUIVARIANCE,
                             W_L1, W_MIP, W_SCALE, W_TAIL, evaluate, export_latent_channel_stats,
                             load_vaev4, mip_l1, selection_score, transform_pair, vessel_weight_map)


V4_CHECKPOINT = ROOT / "checkpoints" / "vaev4_mra" / "v4_c16_fixed" / "best.pt"
V5_ROOT = ROOT / "checkpoints" / "vaev5_mra"

# Finetune schedule. 10x below v4's 2e-4: the network is converged and these terms are
# meant to refine vessel topology and texture, not to move the representation.
FT_LR = 2e-5
FT_EPOCHS = 30
FT_STEPS = STEPS_PER_EPOCH
# 10x the generator's rate, and deliberately decoupled from it: the generator is
# CONVERGED and wants a tiny step, but D starts from random init and needs a normal one.
# At 2e-5 (an earlier mistake here) D was still emitting a constant after 2000 steps --
# D(real) +0.2364 vs D(fake) +0.2356, a separation of 0.0008 against its own 0.019 noise --
# so the adversarial gradient carried no information and reconstruction l1 drifted up 7.7 %.
DISC_LR = 2e-4
DISC_START_STEPS = 1000    # train D alone first; a chance-level D would otherwise push
                           # noise straight into a converged encoder/decoder
# Calibrated against THIS objective, not against the literature. Shit et al. combine
# clDice with a Dice loss of magnitude ~0.5, so their alpha of 0.1-0.5 is balanced. Here
# the v4 reconstruction loss converged to ~0.0088 while the clDice loss starts near 0.40,
# so a weight of 0.15 makes the topology term 6x the entire reconstruction objective --
# measured in a smoke run, where vessel Dice fell 0.9224 -> 0.8937 in four steps. 0.005
# puts it at roughly 20 % of the total, which is the intended refinement.
W_CLDICE = 0.005
# The adversarial term has the same scale hazard and a magnitude that drifts as D learns,
# so it is balanced by gradient norms at the decoder's output layer instead of a fixed
# number (Esser et al., VQGAN). W_ADV is only the base multiplier on that ratio.
W_ADV = 0.25
ADV_RAMP_STEPS = 1000      # linear ramp of W_ADV after DISC_START_STEPS

                                    # 0.5 the vessel tree is destroyed for every arm
                                    # (Dice ~0.10) and the comparison stops resolving.
PERTURB_CASES = 6

BASELINE_KEY = "v4 (none)"

ARMS = {
    "cldice": {"cldice": True, "adversarial": False},
    "adv":    {"cldice": False, "adversarial": True},
    "both":   {"cldice": True, "adversarial": True},
}


# =============================================================================
# Finetune
# =============================================================================
def finetune(arm: str, args, dataset, sampler, validation_cases, perturb_cases,
             perceptual, device: str) -> dict:
    config = ARMS[arm]
    run_dir = V5_ROOT / f"v5_{arm}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print()
    print("=" * 92)
    print(f"ARM '{arm}'  clDice={config['cldice']}  adversarial={config['adversarial']}"
          f"  -> {run_dir}")
    print("=" * 92, flush=True)

    model = load_vaev4(args.checkpoint, device)
    # load_vaev4 freezes parameters and calls eval() -- it exists for the LDM's inference
    # use. Undo both before building the optimiser, or the first backward has no graph.
    model.requires_grad_(True)
    model.train()
    # Fresh optimiser: v4's Adam moments belong to lr 2e-4 and a different objective.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    discriminator = disc_optimizer = None
    if config["adversarial"]:
        discriminator = PatchDiscriminator3D().to(device)
        disc_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=args.disc_lr,
                                           betas=(0.5, 0.9))
        print(f"discriminator: {sum(p.numel() for p in discriminator.parameters())/1e6:.3f} M "
              f"(training only; not exported)", flush=True)

    start_epoch, best_metric = 0, -float("inf")
    latest_path = run_dir / "latest.pt"
    if latest_path.exists() and not args.scratch:
        blob = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(blob["model_state_dict"])
        if blob.get("optimizer_state_dict"):
            optimizer.load_state_dict(blob["optimizer_state_dict"])
        if discriminator is not None and blob.get("discriminator_state_dict"):
            discriminator.load_state_dict(blob["discriminator_state_dict"])
            if blob.get("disc_optimizer_state_dict"):
                disc_optimizer.load_state_dict(blob["disc_optimizer_state_dict"])
        start_epoch = int(blob.get("epoch", 0))
        best_metric = float(blob.get("best_metric", -float("inf")))
        print(f"resumed from {latest_path} at epoch {start_epoch} (best {best_metric:.5f})",
              flush=True)

    (run_dir / "config.json").write_text(json.dumps({
        "arm": arm, "finetuned_from": str(args.checkpoint),
        "lr": args.lr, "disc_lr": args.disc_lr, "epochs": args.epochs, "steps": args.steps,
        "w_cldice": args.cldice_weight if config["cldice"] else 0.0,
        "cldice_iterations": CLDICE_ITERS,
        "w_adv": args.adv_weight if config["adversarial"] else 0.0,
        "disc_start_steps": args.disc_start, "adv_ramp_steps": ADV_RAMP_STEPS,
        "kl_weight": args.kl_weight, "perceptual_weight": args.perceptual_weight,
        "inherited_v4_weights": {"l1": W_L1, "mip": W_MIP, "background": W_BACKGROUND,
                                 "equivariance": W_EQUIVARIANCE, "scale": W_SCALE,
                                 "tail": W_TAIL},
        "parameters": int(sum(p.numel() for p in model.parameters())),
    }, indent=2), encoding="utf-8")

    global_step = start_epoch * args.steps
    total_steps = args.epochs * args.steps
    last_completed_epoch = start_epoch
    started = time.time()

    try:
        for epoch in range(start_epoch + 1, args.epochs + 1):
            sampler.set_epoch(epoch)
            loader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.workers,
                                pin_memory=(device == "cuda"), persistent_workers=False)
            totals: dict[str, float] = {}
            steps = 0
            latent_std_accumulator = 0.0
            epoch_started = time.time()
            model.train()

            for batch in loader:
                image = batch["image"].to(device, non_blocking=True)
                brain = batch["mask"].to(device, non_blocking=True)
                p99 = batch["p99"].to(device, non_blocking=True)
                p999 = batch["p999"].to(device, non_blocking=True)

                progress = global_step / max(total_steps, 1)
                learning_rate = args.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate

                # ---------------------------------------------------- generator
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

                    mu32, logvar32 = mu.float(), logvar.float()
                    kl_mu = 0.5 * mu32.pow(2).mean()
                    kl_var = 0.5 * (logvar32.exp() - 1.0 - logvar32).mean()
                    loss = loss + args.kl_weight * (kl_mu + kl_var)
                    terms["kl"] = (kl_mu + kl_var).detach()

                    channel_std = mu32.transpose(0, 1).reshape(mu32.shape[1], -1).std(dim=1)
                    if W_SCALE > 0:      # keep the latent statistics the LDM depends on
                        scale_term = channel_std.clamp_min(1e-8).log().pow(2).mean()
                        loss = loss + W_SCALE * scale_term
                        terms["scale"] = scale_term.detach()
                    if W_TAIL > 0:
                        tail = F.relu(mu32.abs() - TAIL_SIGMA).pow(2).mean()
                        loss = loss + W_TAIL * tail
                        terms["tail"] = tail.detach()

                    if W_EQUIVARIANCE > 0 and random.random() < P_EQUIVARIANCE:
                        scale = random.choice(EQ_SCALES)
                        rotations = random.choice((0, 1, 2, 3))
                        z_t, x_t = transform_pair(z, image, scale, rotations)
                        term = (recon.new_tensor(0.0) if min(z_t.shape[-3:]) < 4
                                else F.l1_loss(model.decode(z_t), x_t))
                        loss = loss + W_EQUIVARIANCE * term
                        terms["equivariance"] = term.detach()

                # ------------------------------------------------ the new terms
                if config["cldice"]:
                    # fp32 outside autocast: the skeleton is built from many min/max ops and
                    # bf16's 7 mantissa bits blur the thin structures this term exists to keep
                    term = soft_cldice_loss(vessel_probability_percentile(recon.float(), p99, p999),
                                            vessel_probability_percentile(image.float(), p99, p999))
                    loss = loss + args.cldice_weight * term
                    terms["cldice"] = term.detach()

                adv_weight = 0.0
                if config["adversarial"] and global_step >= args.disc_start:
                    ramp = min(1.0, (global_step - args.disc_start) / max(ADV_RAMP_STEPS, 1))
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                        term = hinge_g_loss(discriminator(recon))
                    lam = adaptive_weight(loss, term, model.decoder.conv_out.weight)
                    adv_weight = float(args.adv_weight * ramp * lam)
                    loss = loss + adv_weight * term
                    terms["adv_g"] = term.detach()
                    terms["adv_lambda"] = lam

                if not torch.isfinite(loss):
                    print(f"non-finite loss at epoch {epoch} step {steps + 1}; skipping batch")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                if torch.isfinite(grad_norm):
                    optimizer.step()

                # ------------------------------------------------ discriminator
                if config["adversarial"]:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                        real_logits = discriminator(image)
                        fake_logits = discriminator(recon.detach())
                        d_loss = hinge_d_loss(real_logits, fake_logits)
                    disc_optimizer.zero_grad(set_to_none=True)
                    d_loss.backward()
                    d_norm = torch.nn.utils.clip_grad_norm_(discriminator.parameters(), GRAD_CLIP)
                    if torch.isfinite(d_norm):
                        disc_optimizer.step()
                    terms["adv_d"] = d_loss.detach()
                    terms["adv_weight"] = torch.tensor(adv_weight)
                    # whether D is doing anything at all: a working discriminator drives
                    # this positive and growing, a dead one sits at 0. Reuse the logits
                    # from the D step -- recomputing them outside autocast fed bf16
                    # activations to float32 weights, and cost two extra forward passes.
                    terms["d_real"] = real_logits.float().mean().detach()
                    terms["d_fake"] = fake_logits.float().mean().detach()
                    terms["d_separation"] = terms["d_real"] - terms["d_fake"]

                latent_std_accumulator += float(channel_std.mean().detach())
                totals["loss"] = totals.get("loss", 0.0) + float(loss.detach())
                for key, value in terms.items():
                    totals[key] = totals.get(key, 0.0) + float(value)
                steps += 1
                global_step += 1

                if steps % 200 == 0:
                    print(f"  [{arm}] epoch {epoch} step {steps}/{args.steps} "
                          f"loss {totals['loss']/steps:.4f} "
                          f"latent_std {latent_std_accumulator/steps:.3f} "
                          f"[{(time.time()-started)/3600:.2f} h]", flush=True)

            record = {"arm": arm, "epoch": epoch, "lr": learning_rate,
                      "latent_std": latent_std_accumulator / max(steps, 1),
                      "steps": steps, "time_sec": time.time() - epoch_started,
                      **{k: v / max(steps, 1) for k, v in totals.items()}}
            append_jsonl(run_dir / "training.jsonl", record)
            print(f"[{arm}] epoch {epoch}: loss {record['loss']:.5f} "
                  f"l1 {record.get('l1', float('nan')):.5f} "
                  f"cldice {record.get('cldice', float('nan')):.4f} "
                  f"adv_d {record.get('adv_d', float('nan')):.4f} "
                  f"d_sep {record.get('d_separation', float('nan')):+.4f} "
                  f"latent_std {record['latent_std']:.3f} ({record['time_sec']:.0f}s)", flush=True)

            save_v5(latest_path, model, optimizer, discriminator, disc_optimizer,
                    epoch, best_metric, arm)
            last_completed_epoch = epoch

            if epoch % args.val_every == 0 or epoch == args.epochs:
                summary = evaluate(model, validation_cases, "val", device)
                score = selection_score(summary)
                summary.update({"arm": arm, "epoch": epoch, "selection_score": score})
                append_jsonl(run_dir / "validation.jsonl", summary)
                print(f"  [{arm}] validation: masked SSIM "
                      f"{summary.get('tiled_masked_ssim', float('nan')):.4f} "
                      f"Dice {summary.get('tiled_vessel_dice', float('nan')):.4f} "
                      f"PSNR {summary.get('tiled_psnr', float('nan')):.2f} "
                      f"score {score:.4f}", flush=True)
                if score > best_metric:
                    best_metric = score
                    save_v5(run_dir / "best.pt", model, optimizer, discriminator,
                            disc_optimizer, epoch, best_metric, arm, extra={"validation": summary})
                    print(f"  [{arm}] new best {best_metric:.5f} at epoch {epoch}", flush=True)
    except KeyboardInterrupt:
        print(f"[{arm}] interrupted; saving", flush=True)
        save_v5(latest_path, model, optimizer, discriminator, disc_optimizer,
                last_completed_epoch, best_metric, arm)
        raise

    # `best.pt` is chosen by the v4 selection score, which is reconstruction-weighted and
    # can under-rate an adversarial arm that trades a little PSNR for texture. Keep the
    # final weights too so that trade is inspectable rather than discarded.
    save_v5(run_dir / "final.pt", model, optimizer, discriminator, disc_optimizer,
            args.epochs, best_metric, arm)
    return score_arm(arm, run_dir, validation_cases, perturb_cases, device, args)


def save_v5(path: Path, model, optimizer, discriminator, disc_optimizer,
            epoch: int, best: float, arm: str, extra: dict | None = None) -> None:
    blob = {"model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "epoch": int(epoch), "best_metric": float(best), "kl_weight": 0.0,
            "model_config": model.config, "arm": arm, "extra": extra or {}}
    if discriminator is not None:
        blob["discriminator_state_dict"] = discriminator.state_dict()
        blob["disc_optimizer_state_dict"] = disc_optimizer.state_dict()
    torch.save(blob, path)


def score_arm(arm: str, run_dir: Path, validation_cases, perturb_cases, device, args) -> dict:
    """Clean metrics plus the perturbed sweep, on the arm's best checkpoint."""
    checkpoint = next((run_dir / name for name in ("best.pt", "final.pt", "latest.pt")
                       if (run_dir / name).exists()), None)
    if checkpoint is None:
        raise SystemExit(f"no checkpoint to score in {run_dir}")
    model = load_vaev4(checkpoint, device)
    summary = evaluate(model, validation_cases, "val", device, skeleton_cases=8)
    summary.update(evaluate_perturbed(model, perturb_cases, "val", device))
    summary["arm"] = arm
    (run_dir / "arm_results.json").write_text(json.dumps(summary, indent=2, default=float),
                                              encoding="utf-8")
    export_latent_channel_stats(model, device, run_dir, cases=40)
    del model
    torch.cuda.empty_cache()
    return summary


# =============================================================================
def main() -> None:
    global V5_ROOT
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", nargs="+", default=["all"],
                        choices=sorted(ARMS) + ["all"], metavar="ARM",
                        help="experiments to run, in the order given: "
                             f"{', '.join(sorted(ARMS))}, or 'all' for every arm "
                             "sequentially (default)")
    parser.add_argument("--checkpoint", type=Path, default=V4_CHECKPOINT)
    parser.add_argument("-n", "--epochs", type=int, default=FT_EPOCHS,
                        help=f"epochs per arm (default {FT_EPOCHS}); each arm gets this budget")
    parser.add_argument("--steps", type=int, default=FT_STEPS)
    parser.add_argument("--lr", type=float, default=FT_LR)
    parser.add_argument("--disc-lr", type=float, default=DISC_LR)
    parser.add_argument("--disc-start", type=int, default=DISC_START_STEPS)
    parser.add_argument("--cldice-weight", type=float, default=W_CLDICE)
    parser.add_argument("--adv-weight", type=float, default=W_ADV,
                        help="base multiplier on the gradient-balanced lambda, not a raw weight")
    parser.add_argument("--kl-weight", type=float, default=1e-7)
    parser.add_argument("--perceptual-weight", type=float, default=0.02)
    parser.add_argument("--no-perceptual", action="store_true")
    parser.add_argument("--val-every", type=int, default=VAL_EVERY)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scratch", action="store_true", help="ignore latest.pt")
    parser.add_argument("--rescore-baseline", action="store_true",
                        help="recompute v4's cached baseline numbers instead of reusing them")
    parser.add_argument("--baseline-only", action="store_true",
                        help="only re-score v4 under the perturbed sweep, train nothing")
    parser.add_argument("--smoke", action="store_true",
                        help="a few steps of every arm, no validation")
    parser.add_argument("--smoke-steps", type=int, default=4)
    parser.add_argument("--out", type=Path, default=None,
                        help="folder for the v5_<arm> runs (default checkpoints/rebuild/vae_mra_v5; "
                             "a smoke run defaults to checkpoints/_smoke/vaev5)")
    args = parser.parse_args()
    if args.out is None:
        args.out = ROOT / "checkpoints" / "_smoke" / "vaev5" if args.smoke else REBUILD_ROOT / "vae_mra_v5"
    V5_ROOT = refuse_protected(args.out, "the v5 arms")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    V5_ROOT.mkdir(parents=True, exist_ok=True)

    if not args.checkpoint.exists():
        raise SystemExit(f"v4 checkpoint not found: {args.checkpoint}")

    if args.smoke:
        args.epochs, args.steps, args.val_every = 1, args.smoke_steps, 999
        args.disc_start = 2

    dataset = MRACropDataset("train", seed=args.seed)
    sampler = MultiExtentBatchSampler(len(dataset), steps=args.steps, seed=args.seed)
    validation_cases = pick_validation_cases("val", 1 if args.smoke else 8)
    perturb_cases = (pick_validation_cases("val", 1)[:1] if args.smoke
                     else pick_validation_cases("val", 2)[:PERTURB_CASES])
    print(f"{len(dataset)} training volumes | {len(validation_cases)} validation cases | "
          f"{len(perturb_cases)} perturbation cases | device {device}")

    perceptual = None
    if not args.no_perceptual:
        try:
            perceptual = Perceptual25D().to(device)
        except Exception as error:
            print(f"perceptual loss disabled ({type(error).__name__}: {error})")

    results: dict[str, dict] = {}
    print()
    results[BASELINE_KEY] = baseline_summary(args, validation_cases, perturb_cases, device)

    arms = list(ARMS) if "all" in args.arms else list(dict.fromkeys(args.arms))
    if not args.baseline_only:
        for arm in arms:
            results[arm] = finetune(arm, args, dataset, sampler, validation_cases,
                                    perturb_cases, perceptual, device)

    report(results)


def baseline_summary(args, validation_cases, perturb_cases, device) -> dict:
    """v4's own numbers, cached: running the arms in separate launches would otherwise
    re-score the same frozen checkpoint every time, ~12 min each. The cache records the
    case counts it was measured on and is ignored if they differ."""
    cache = V5_ROOT / "baseline_v4.json"
    want = {"validation_cases": len(validation_cases), "perturb_cases": len(perturb_cases),
            "checkpoint": str(args.checkpoint)}
    if cache.exists() and not args.rescore_baseline and not args.smoke:
        stored = json.loads(cache.read_text(encoding="utf-8"))
        if all(stored.get(k) == v for k, v in want.items()):
            print(f"baseline: reusing {cache.name} "
                  f"(--rescore-baseline to recompute)", flush=True)
            return stored
        print("baseline: cached under a different case set, re-scoring", flush=True)

    print("baseline: scoring VAEv4 (the no-op arm of the table)", flush=True)
    model = load_vaev4(args.checkpoint, device)
    summary = evaluate(model, validation_cases, "val", device, skeleton_cases=8)
    summary.update(evaluate_perturbed(model, perturb_cases, "val", device))
    summary.update(want)
    summary["arm"] = BASELINE_KEY
    del model
    torch.cuda.empty_cache()
    if not args.smoke:
        cache.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    return summary


def report(results: dict[str, dict]) -> None:
    """Merge whatever is on disk with this launch's results, so running the arms one at a
    time still produces the full table. Without this, `--arms adv` would overwrite
    ablation.json with a table containing only adv."""
    merged: dict[str, dict] = {}
    cache = V5_ROOT / "baseline_v4.json"
    if cache.exists():
        merged[BASELINE_KEY] = json.loads(cache.read_text(encoding="utf-8"))
    for arm in ARMS:
        stored = V5_ROOT / f"v5_{arm}" / "arm_results.json"
        if stored.exists():
            merged[arm] = json.loads(stored.read_text(encoding="utf-8"))
    merged.update(results)          # this launch's numbers win over anything stale
    order = [BASELINE_KEY] + [a for a in ARMS if a in merged]
    results = {k: merged[k] for k in order if k in merged}

    out = V5_ROOT / "ablation.json"
    out.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")

    clean = [("masked SSIM", "tiled_masked_ssim"), ("vessel Dice", "tiled_vessel_dice"),
             ("clDice", "tiled_cldice"), ("PSNR", "tiled_psnr"), ("MIP SSIM", "tiled_mip_ssim")]
    print()
    print("=" * 92)
    print("ABLATION -- clean reconstruction (validation, tiled)")
    print("=" * 92)
    print(f"{'arm':>12}" + "".join(f"{label:>14}" for label, _ in clean))
    for arm, summary in results.items():
        print(f"{arm:>12}" + "".join(f"{summary.get(key, float('nan')):>14.4f}"
                                    for _, key in clean))

    print()
    print("=" * 92)
    print("ABLATION -- decoding perturbed latents (vessel Dice / clDice)")
    print("   the decoder's real operating point: the diffusion sampler never hands it an")
    print("   exact latent, so this is where a topology-aware decoder should separate")
    print("=" * 92)
    header = "".join(f"{f'sigma {s}':>20}" for s in PERTURB_SIGMAS)
    print(f"{'arm':>12}{header}")
    for arm, summary in results.items():
        cells = "".join(
            f"{summary.get(f'perturbed_{s}_vessel_dice', float('nan')):>9.4f}"
            f"{summary.get(f'perturbed_{s}_cldice', float('nan')):>11.4f}"
            for s in PERTURB_SIGMAS)
        print(f"{arm:>12}{cells}")
    print()
    print(f"written: {out}")
    print("pick ONE arm to carry forward: re-export latents from it, then train the LDM.")
    print("Changing the VAE afterwards invalidates any LDM trained on its latents.")


if __name__ == "__main__":
    main()
