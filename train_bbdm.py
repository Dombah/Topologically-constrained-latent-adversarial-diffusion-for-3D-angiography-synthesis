"""Brownian Bridge latent diffusion: T1/T2/PD -> MRA, diffusing data-to-data.

    uv run train_bbdm.py                       # reads configs/bbdm.json

WHAT IS DIFFERENT FROM train_ldm_v2.py
--------------------------------------
Standard diffusion maps data <-> NOISE: at t=T the state is pure Gaussian and generation
starts from a tractable prior. A Brownian bridge maps data <-> DATA:

    q(x_t | x_0, y) = N( (1-m_t) x_0 + m_t y ,  delta_t I )
    m_t = t/T          delta_t = 2 (m_t - m_t^2)

At t=0 the mean is x_0 (the MRA latent) with delta=0; at t=T the mean is y (a projection
of the source latents), also with delta=0. Variance peaks in the middle and vanishes at
both ends -- the process is pinned at both endpoints, which is what makes it a bridge.

WHY IT MIGHT HELP HERE
----------------------
Every model in this project has been limited by the same thing: T1/T2/PD localise vessels
only weakly, so a model generating from noise must INVENT vessel placement, and hedges when
uncertain. A bridge never starts from noise -- it starts from a learned estimate of the
target and refines it, so the uncertainty it has to resolve is much smaller.

CORRECTED 7 Sep 2026 -- the earlier "linear probe R^2 0.012" was measuring the PROBE.
A per-voxel LINEAR readout cannot detect a vessel even in principle: a vessel is a tubular
arrangement, and "dark on T2 AND mid on T1 AND mid on PD" is a conjunction. Re-measured on
raw T1/T2/PD against the whole-volume p99 vessel band (12 train / 3 val volumes):

    probe                      R^2      AUC
    per-voxel linear        -0.1113   0.5181   <- the old figure's regime: chance
    per-voxel MLP           +0.0172   0.7083   <- nonlinearity alone
    context CNN (rf ~25)    +0.0813   0.7610   <- + spatial context

So the conditioning carries real vessel information, far more than 0.012 implied -- though
0.761 is still well short of the >0.95 a competent segmenter reaches, which is consistent
with the models recovering the major tree and missing the distal one. Report AUC, not R^2:
the band is ~1 % of voxels, and the linear probe scores a NEGATIVE R^2 while sitting at
chance.

THE ENDPOINT, AND WHAT IT COSTS CONCEPTUALLY
--------------------------------------------
The sources are 24 channels and the target is 16, so the bridge needs a learned projection
y = P(cond). That projection is a latent-space REGRESSOR, trained here with an explicit L1
term against x_0 as well as through the bridge.

This is worth being clear about: a regression estimate is now structurally inside the
model. The claim "sampling beats one-step regression", which v2 demonstrated (2.16x Dice,
5.6x MIP SSIM, where the v3-era A1 model failed the same test), is NOT the same claim under
a bridge -- the endpoint IS the regression. What a bridge can claim is that the reverse
process improves on its own endpoint, which is weaker but still meaningful, and the
endpoint's own score is logged every validation so the improvement is measurable.

PARAMETERISATION
----------------
Following Li et al. 2023, the network predicts (x_t - x_0):

    L = || m_t (y - x_0) + sqrt(delta_t) eps  -  eps_theta(x_t, t, cond) ||^2
      = || x_t - x_0 - eps_theta ||^2      so   x0_pred = x_t - eps_theta

Sampling is the deterministic (DDIM-style) bridge update: estimate x_0, recover the noise
that is consistent with the forward process, and step to t-1 with the same formula. At
t=0 both m and delta are zero, so the last step returns x_0_pred exactly.

NO CLASSIFIER-FREE GUIDANCE
---------------------------
CFG in v2 works because dropping the condition leaves a coherent unconditional model. Here
the condition also defines the bridge ENDPOINT, so dropping it removes the process itself.
Guidance would need a separate design; it is deliberately absent rather than forgotten.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from modules.bridge import BridgeProjector, BrownianBridge, validate
from modules.latent_data import PairedLatentDataset
from modules.losses import frangi_weighted_l1, frangi_weighted_mip_l1, vessel_weighted_gradient_l1
from modules.paths import refuse_protected, ROOT
from modules.unet import build_model
from modules.vae_mra import load_vaev4


# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "bbdm.json")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--scratch", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--sanity", type=int, default=0,
                        help="run N epochs, abort unless the loss is finite and falling")
    parser.add_argument("--memory-fraction", type=float, default=0.0)
    args = parser.parse_args()

    if args.memory_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    training = config["training"]
    if args.smoke and not args.run_dir:
        args.run_dir = ROOT / "checkpoints" / "_smoke" / "bbdm"   # never touch a real run
    run_dir = refuse_protected(args.run_dir or ROOT / config["output"]["run_dir"], "the bridge run")
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, run_dir / "config.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(training["seed"]); np.random.seed(training["seed"])
    random.seed(training["seed"])
    if args.smoke:
        training.update({"epochs": 1, "steps_per_epoch": 4, "workers": 0})
        config["validation"].update({"cases": 1, "steps": [4]})
    if args.sanity:
        training["epochs"] = args.sanity
        config["validation"]["every"] = 10 ** 9

    print("loading latents", flush=True)
    train_set = PairedLatentDataset(config, "train", seed=training["seed"])
    val_set = PairedLatentDataset(config, "val", seed=training["seed"])
    source_channels = sum(len(train_set.stats[m]["per_channel_mean"]) for m in train_set.sources)
    target_channels = len(train_set.stats[train_set.target_modality]["per_channel_mean"])

    model = build_model(config, target_channels, source_channels, device)
    projector = BridgeProjector(source_channels, target_channels,
                                base=int(config["model"]["projector_base"]),
                                blocks=int(config["model"]["projector_blocks"])).to(device)
    print(f"UNet {sum(p.numel() for p in model.parameters())/1e6:.2f} M  +  "
          f"projector {sum(p.numel() for p in projector.parameters())/1e6:.2f} M", flush=True)
    bridge = BrownianBridge(int(config["bridge"]["steps"]),
                            float(config["bridge"]["max_variance"]), device)

    parameters = list(model.parameters()) + list(projector.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=training["lr"],
                                  weight_decay=training["weight_decay"])
    ema = copy.deepcopy(model).eval()
    ema_projector = copy.deepcopy(projector).eval()
    for p in list(ema.parameters()) + list(ema_projector.parameters()):
        p.requires_grad_(False)
    vae = load_vaev4(ROOT / config["vae"]["checkpoint"], device) \
        if config["validation"].get("decode", True) else None

    start_epoch, best = 0, -float("inf")
    latest = run_dir / "latest.pt"
    if latest.exists() and not args.scratch:
        blob = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(blob["model"]); projector.load_state_dict(blob["projector"])
        ema.load_state_dict(blob["ema"]); ema_projector.load_state_dict(blob["ema_projector"])
        optimizer.load_state_dict(blob["optimizer"])
        start_epoch, best = int(blob["epoch"]), float(blob["best"])
        print(f"resumed from epoch {start_epoch}", flush=True)

    amp = torch.bfloat16 if training["amp_dtype"] == "bfloat16" else torch.float16
    losses_spec = config["losses"]
    m_gate = float(losses_spec["x0_m_gate"])
    workers = int(training["workers"])
    loader = DataLoader(train_set, batch_size=int(training["batch_size"]), shuffle=True,
                        num_workers=workers, drop_last=True, pin_memory=(device == "cuda"))
    total_steps = int(training["epochs"]) * int(training["steps_per_epoch"])
    global_step = start_epoch * int(training["steps_per_epoch"])
    started = time.time()
    deadline = started + float(training["hours"]) * 3600.0
    history = []

    for epoch in range(start_epoch + 1, int(training["epochs"]) + 1):
        model.train(); projector.train()
        totals: dict[str, float] = {}
        steps = 0
        epoch_started = time.time()
        iterator = iter(loader)
        while steps < int(training["steps_per_epoch"]):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader); continue
            x0 = batch["target"].to(device, non_blocking=True)
            condition = batch["source"].to(device, non_blocking=True)
            frangi = batch["frangi"].to(device, non_blocking=True)

            warm = min(1.0, (global_step + 1) / max(int(training["warmup_steps"]), 1))
            progress = global_step / max(total_steps, 1)
            lr = training["lr"] * warm * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
            for group in optimizer.param_groups:
                group["lr"] = lr

            t = bridge.sample_timesteps(x0.shape[0], device)
            noise = torch.randn_like(x0)
            with torch.amp.autocast("cuda", dtype=amp, enabled=(device == "cuda")):
                y = projector(condition)
                x_t, target = bridge.add_noise(x0, y.float(), t, noise)
                prediction = model(x_t.to(x0.dtype), t, condition)
                bridge_loss = F.mse_loss(prediction.float(), target.float())
                loss = float(losses_spec["bridge_mse"]) * bridge_loss
                terms = {"bridge_mse": bridge_loss.detach()}

                # The endpoint must be a real estimate, not an arbitrary anchor.
                endpoint_loss = F.l1_loss(y.float(), x0.float())
                loss = loss + float(losses_spec["endpoint_l1"]) * endpoint_loss
                terms["endpoint_l1"] = endpoint_loss.detach()

                # x0-space auxiliaries, gated by position along the bridge. Near m_t=1 the
                # state IS the endpoint and x_0 is only knowable through the conditioning,
                # which is the regression regime -- the analogue of v2's SNR gate.
                gate = bridge.m[t] <= m_gate
                terms["aux_active_fraction"] = gate.float().mean().detach()
                if bool(gate.any()):
                    x0_pred = bridge.to_x0(x_t, prediction.float())[gate]
                    x0_true, vessel = x0[gate], frangi[gate]
                    alpha = float(losses_spec["frangi_alpha"])
                    for key, fn in (
                        ("x0_smooth_l1", lambda: F.smooth_l1_loss(x0_pred, x0_true, beta=0.5)),
                        ("frangi_l1", lambda: frangi_weighted_l1(x0_pred, x0_true, vessel, alpha)),
                        ("frangi_mip", lambda: frangi_weighted_mip_l1(x0_pred, x0_true, vessel, alpha)),
                        ("vessel_gradient", lambda: vessel_weighted_gradient_l1(
                            x0_pred, x0_true, vessel, float(losses_spec["gradient_alpha"]))),
                    ):
                        w = float(losses_spec.get(key, 0.0))
                        if w > 0:
                            term = fn(); loss = loss + w * term; terms[key] = term.detach()

            if not torch.isfinite(loss):
                print(f"non-finite loss at epoch {epoch} step {steps}; skipping", flush=True)
                optimizer.zero_grad(set_to_none=True); continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, float(training["grad_clip"]))
            if torch.isfinite(norm):
                optimizer.step()
            with torch.no_grad():
                decay = float(training["ema_decay"])
                for tgt_p, src_p in zip(ema.parameters(), model.parameters()):
                    tgt_p.mul_(decay).add_(src_p.detach(), alpha=1.0 - decay)
                for tgt_p, src_p in zip(ema_projector.parameters(), projector.parameters()):
                    tgt_p.mul_(decay).add_(src_p.detach(), alpha=1.0 - decay)
                for tgt_b, src_b in zip(ema.buffers(), model.buffers()):
                    tgt_b.copy_(src_b)

            totals["loss"] = totals.get("loss", 0.0) + float(loss.detach())
            for k, v in terms.items():
                totals[k] = totals.get(k, 0.0) + float(v)
            steps += 1; global_step += 1
            if steps % 100 == 0:
                print(f"  epoch {epoch} step {steps}/{training['steps_per_epoch']} "
                      f"loss {totals['loss']/steps:.5f} lr {lr:.2e} "
                      f"[{(time.time()-started)/3600:.2f} h]", flush=True)

        record = {"epoch": epoch, "lr": lr, "time_sec": time.time() - epoch_started,
                  **{k: v / max(steps, 1) for k, v in totals.items()}}
        history.append(record["loss"])
        with (run_dir / "training.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        print(f"epoch {epoch}: loss {record['loss']:.5f}  bridge {record['bridge_mse']:.5f}  "
              f"endpoint {record['endpoint_l1']:.5f}  ({record['time_sec']:.0f}s)", flush=True)
        torch.save({"model": model.state_dict(), "projector": projector.state_dict(),
                    "ema": ema.state_dict(), "ema_projector": ema_projector.state_dict(),
                    "optimizer": optimizer.state_dict(), "epoch": epoch, "best": best,
                    "config": config}, latest)

        if args.sanity and epoch == args.sanity:
            ok = all(np.isfinite(history)) and history[-1] < history[0]
            print(f"\nSANITY {'PASS' if ok else 'FAIL'}: loss {history[0]:.5f} -> "
                  f"{history[-1]:.5f} over {len(history)} epochs", flush=True)
            raise SystemExit(0 if ok else 2)

        out_of_time = time.time() > deadline
        if epoch % int(config["validation"]["every"]) == 0 or epoch == training["epochs"] or out_of_time:
            summary = validate(ema, ema_projector, bridge, val_set, config, vae, device)
            summary["epoch"] = epoch
            with (run_dir / "validation.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(summary, default=float) + "\n")
            counts = config["validation"]["steps"]
            counts = [counts] if isinstance(counts, int) else list(counts)
            key = f"bridge{max(counts)}"
            dice = summary.get(f"{key}_vessel_dice", 0.0)
            end = summary.get("endpoint_vessel_dice", 0.0)
            ladder = "  ".join(f"bridge-{k} {summary.get(f'bridge{k}_vessel_dice', float('nan')):.4f}"
                               for k in counts)
            print(f"  validation: {ladder} | endpoint {end:.4f} | "
                  f"bridge advantage {dice-end:+.4f} | "
                  f"mSSIM {summary.get(key+'_masked_ssim', float('nan')):.4f} | "
                  f"p99.9 {summary.get(key+'_p999', float('nan')):.3f}", flush=True)
            if dice < end:
                print("    WARNING: the reverse process is not improving on its own endpoint "
                      "-- the bridge is doing no work", flush=True)
            if dice > best:
                best = dice
                torch.save({"ema": ema.state_dict(), "ema_projector": ema_projector.state_dict(),
                            "epoch": epoch, "best": best, "config": config,
                            "validation": summary}, run_dir / "best.pt")
                print(f"  new best {best:.4f} at epoch {epoch}", flush=True)
        if epoch % int(config["output"]["checkpoint_every"]) == 0:
            torch.save({"ema": ema.state_dict(), "ema_projector": ema_projector.state_dict(),
                        "epoch": epoch, "config": config}, run_dir / f"epoch_{epoch:04d}.pt")
        if out_of_time:
            print(f"stopping: {training['hours']} h budget reached", flush=True)
            break

    print(f"\nfinished. run directory: {run_dir}")


if __name__ == "__main__":
    main()
