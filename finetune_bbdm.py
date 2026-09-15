"""Finetune the trained Brownian-bridge model with topology and adversarial terms.

    uv run finetune_bbdm.py --arms cldice --smoke                 # plumbing check, writes to checkpoints/_smoke
    uv run finetune_bbdm.py --arms cldice_alone cldice adv_alone both -n 40
    uv run finetune_bbdm.py --arms delivered -n 40                # the model used in the thesis

Arms and the folder each writes (under output.run_dir unless the arm sets its own):

    cldice_alone  clDice only (alpha 1.0)                        ft_cldice_alone
    cldice        alpha*clDice + (1-alpha)*soft Dice             ft_cldice
    adv_alone     MIP discriminator only                         ft_adv_alone
    both          cldice + MIP discriminator                     ft_both
    delivered     cldice + frangi_l1 4.0 + vessel_gradient 0.2   <run_dir>_vessel/ft_cldice

Outputs default to checkpoints/rebuild/ (configs/bbdm.json output.run_dir). The thesis checkpoints
in checkpoints/ldm_bbdm and checkpoints/ldm_bbdm_vessel are protected and never written.

The loss machinery (soft clDice, MIP discriminator, VQGAN adaptive lambda) was first written
for a v-prediction latent diffusion model and is reused unchanged; only the forward/reverse
process differs:

    v-prediction            bridge
    ------------            ------
    x_t from (x0, noise)    x_t from (x0, y=P(cond), noise), y the regression endpoint
    predicts v              predicts (x_t - x0)
    to_x0 needs t           to_x0 is static: x_t - prediction
    gate on SNR             gate on m_t (position along the bridge)

WHY THE CROP IS A SLAB
----------------------
The LDM arms decoded 32x32x16 latent crops (~128x128x64 voxels) and the clDice arm made
clDice WORSE (-0.0050) while the adversarial arm, whose discriminator sees whole MIP
projections, made it better (+0.0032). The leading explanation is field of view: at that
crop size vessels are severed at every boundary, so a connectivity loss cannot be
penalised for breaks it never observes, and the discriminator's "MIP" is a 128x128 patch
with no vessel context.

A slab -- full in-plane extent, thin in depth -- fixes both. Its axial MIP is a whole-brain
projection, which is exactly the view the metric is computed on, and in-plane vessel runs
stay intact. Depth is the cheap axis to sacrifice: the axial MIP collapses it anyway.

Set --crop to a slab (e.g. 128 144 4 in latent space -> 512x576x16 decoded). Peak memory
scales with the decoded voxel count; 128x144x4 fits in 16 GB.

THE PROJECTOR IS FROZEN BY DEFAULT
----------------------------------
The bridge's endpoint y = P(cond) is a latent-space regressor and everything downstream is
anchored to it. These arms ask "does an added objective improve the REVERSE process", so
holding the endpoint fixed removes a confound -- and keeps an adversarial gradient from
corrupting the anchor. --train-projector restores the base behaviour.

THE VESSEL BAND MUST MATCH THE METRIC'S SUPPORT
-----------------------------------------------
`vessel_scores` thresholds at np.percentile(target, 99.0) over the WHOLE volume, background
included (~0.338 on this data). finetune_ldm.py's clDice loss used a FIXED band of
(0.3935, 0.99) -- the training set's BRAIN-MASKED (p99, p99.9), a different support. The
loss and the metric therefore disagreed badly:

    64 % of the voxels the metric scores as vessel got EXACTLY ZERO weight in the loss,
    and the ramp only saturated above the brain-masked p99.9, so the soft mask carried
    just 15 % of the hard mask's mass. Soft clDice was supervising the few brightest
    proximal arteries -- never the problem -- and was blind to the distal vessels it was
    added to fix.

Measured over 25 training volumes, soft-mask mass / hard-mask voxel count (want ~1.0):

    fixed (0.3935, 0.99)          0.15   zero-weighted 64.4 %   mean p 0.114
    per-case (p99,   p99.9)       0.24   zero-weighted  0.8 %   mean p 0.240
    per-case (p98,   p99.5)       1.03   zero-weighted  0.0 %   mean p 0.839   <- used
    per-case centred tau +- d/4   2.28   zero-weighted  0.0 %   mean p 0.741

So the band is now PER CASE and on the metric's own support: whole-volume (p98, p99.5),
precomputed into Dataset/vessel_bands.json. This is very likely a bigger effect than the
crop-shape change, and it applies to the LDM arms too -- their clDice result (-0.0050) was
measured under the broken band and should be read as inconclusive rather than negative.

clDICE ALONE IS NOT A VALID OBJECTIVE -- IT MUST BE PAIRED WITH DICE
--------------------------------------------------------------------
The first run with the fixed band "improved" every metric and produced visibly worse
images. Measured on 6 validation cases against the target's own vessel definition:

                  band ratio  precision  recall  parenchyma  contrast   Dice
    target             1.00      1.000    1.000      0.2220    0.2059
    base ep170         0.61      0.3945   0.1942      0.2255    0.0959  0.2304
    clDice ep10        1.83      0.2253   0.3657      0.2336    0.1205  0.2612
    clDice ep40        2.77      0.1546   0.4198      0.2387    0.1268  0.2234

The model was not finding vessels, it was THICKENING them -- a vessel tree up to 2.77x
too large, precision collapsing 0.39 -> 0.15. clDice permits this by construction:
skeletonising a vessel three times too thick yields nearly the same medial axis, so
topology precision barely moves while sensitivity rises monotonically with coverage.
Thickening is a free win, and no other term here (bridge MSE, intensity L1s) constrains
extent. Dice, clDice AND p99.9 all inflate under over-segmentation, so their agreeing
proved nothing -- epoch 10 was already 1.83x over-segmented.

Hence the topology term is now alpha*clDice + (1-alpha)*softDice, which is what Shit et
al. specify: clDice regularises Dice rather than replacing it. soft Dice carries pred.sum()
in its denominator, so over-thickening is penalised directly. `band_ratio` is logged every
epoch as the early-warning signal -- it should sit near 1.0.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

import torch.nn.functional as F
from torch.utils.data import DataLoader

from modules.adversarial import (MIPDiscriminator2D, adaptive_weight, hinge_d_loss, hinge_g_loss,
                                 soft_mip)
from modules.bridge import BridgeProjector, BrownianBridge, validate
from modules.latent_data import PairedLatentDataset, crop_latent, destandardize_t
from modules.losses import frangi_weighted_l1, frangi_weighted_mip_l1, vessel_weighted_gradient_l1
from modules.paths import refuse_protected, ROOT
from modules.topology import soft_cldice_loss, soft_dice_loss, vessel_probability_banded
from modules.unet import build_model
from modules.vae_mra import load_vaev4


ARMS = {
    "cldice_alone": {"cldice": True,  "adv": False, "alpha": 1.0},
    "cldice":       {"cldice": True,  "adv": False},
    "adv_alone":    {"cldice": False, "adv": True},
    "both":         {"cldice": True,  "adv": True},
    # The model used in the thesis: `cldice` with the vessel-weighted latent losses raised.
    "delivered":    {"cldice": True,  "adv": False, "dir": "ft_cldice",
                     "run_dir": "checkpoints/rebuild/bbdm_vessel",
                     "losses": {"frangi_l1": 4.0, "vessel_gradient": 0.2}},
}


class BandedLatentDataset(PairedLatentDataset):
    """PairedLatentDataset plus the per-case vessel band. Subclassed rather than editing
    train_ldm.py, which train_ldm/train_ldm_v2/train_bbdm/finetune_ldm all depend on."""

    def __init__(self, config: dict, split: str, seed: int, bands: dict) -> None:
        super().__init__(config, split, seed=seed)
        missing = [c for c in self.cases if c not in bands]
        if missing:
            raise KeyError(f"{len(missing)} cases have no vessel band "
                           f"(first: {missing[0]}). Regenerate Dataset/vessel_bands.json.")
        self.bands = {c: bands[c] for c in self.cases}

    def __getitem__(self, index: int):
        item = super().__getitem__(index)
        case = self.cases[index % len(self.cases)]
        item["band"] = torch.tensor(self.bands[case], dtype=torch.float32)
        return item


def best_dice(summary: dict) -> float:
    """Highest sampled vessel Dice. The endpoint is the bridge's own regression estimate,
    not a sample, so it must not win the selection."""
    return max((v for k, v in summary.items()
                if k.endswith("_vessel_dice") and not k.startswith("endpoint")), default=0.0)


def load_base(path: Path, config: dict, train_set, device: str):
    base = torch.load(path, map_location=device, weights_only=False)
    source_channels = sum(len(train_set.stats[m]["per_channel_mean"]) for m in train_set.sources)
    target_channels = len(train_set.stats[train_set.target_modality]["per_channel_mean"])
    model = build_model(config, target_channels, source_channels, device)
    model.load_state_dict(base["ema"])
    projector = BridgeProjector(source_channels, target_channels,
                                base=int(config["model"]["projector_base"]),
                                blocks=int(config["model"]["projector_blocks"])).to(device)
    projector.load_state_dict(base["ema_projector"])
    return base, model, projector


# =============================================================================
def finetune(arm: str, args, config: dict, train_set, val_set, vae, device: str) -> dict:
    spec = ARMS[arm]
    config = copy.deepcopy(config)
    if "run_dir" in spec and not args.run_dir:
        config["output"]["run_dir"] = spec["run_dir"]
    for key, value in spec.get("losses", {}).items():
        print(f"arm loss override: {key} {config['losses'][key]} -> {value}", flush=True)
        config["losses"][key] = float(value)
    topo_alpha = float(spec.get("alpha", args.alpha))   # not `alpha`: the loop below uses that name for frangi_alpha
    run_dir = refuse_protected(ROOT / config["output"]["run_dir"] / spec.get("dir", f"ft_{arm}"),
                               "the fine-tuning arm")
    run_dir.mkdir(parents=True, exist_ok=True)
    print()
    print("=" * 88)
    print(f"ARM '{arm}'  cldice={spec['cldice']} (alpha {topo_alpha})  adversarial={spec['adv']}  "
          f"crop={tuple(args.crop)}  -> {run_dir}")
    print("=" * 88, flush=True)

    base, model, projector = load_base(args.checkpoint, config, train_set, device)
    print(f"loaded base epoch {base['epoch']}  "
          f"({sum(p.numel() for p in model.parameters())/1e6:.1f} M UNet + "
          f"{sum(p.numel() for p in projector.parameters())/1e6:.2f} M projector)", flush=True)

    trainable = list(model.parameters())
    if args.train_projector:
        trainable += list(projector.parameters())
    else:
        projector.eval()
        for p in projector.parameters():
            p.requires_grad_(False)
        print("projector FROZEN -- the arms measure the reverse process alone", flush=True)
    # Fresh optimiser: Adam moments accumulated under the base objective are wrong for a
    # different one -- the same reasoning as the VAE and LDM finetunes.
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    ema = copy.deepcopy(model).eval()
    ema_projector = copy.deepcopy(projector).eval()
    for p in list(ema.parameters()) + list(ema_projector.parameters()):
        p.requires_grad_(False)
    bridge = BrownianBridge(int(config["bridge"]["steps"]),
                            float(config["bridge"]["max_variance"]), device)

    discriminator = disc_optimizer = None
    if spec["adv"]:
        discriminator = MIPDiscriminator2D(base=args.disc_base).to(device)
        disc_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=args.disc_lr,
                                           betas=(0.5, 0.9))
        print(f"discriminator: {sum(p.numel() for p in discriminator.parameters())/1e6:.3f} M "
              f"at lr {args.disc_lr:.0e} (10x the generator -- it starts from random init "
              f"against a converged model)", flush=True)

    start_epoch, best = 0, -float("inf")
    latest = run_dir / "latest.pt"
    if latest.exists() and not args.scratch:
        # map_location="cpu", and DELETE the blob afterwards. Loading straight to the GPU
        # kept a full duplicate of model + projector + both EMAs + optimizer moments +
        # discriminator resident for the whole run -- about 1 GB. On a 16 GB card already
        # at ~15.8 GB that tipped the allocator into thrashing: a resumed `both` arm ran at
        # 12.2 s/step against a fresh run's 1.0 s/step. Only the resume path hit it.
        blob = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model"]); ema.load_state_dict(blob["ema"])
        projector.load_state_dict(blob["projector"])
        ema_projector.load_state_dict(blob["ema_projector"])
        optimizer.load_state_dict(blob["optimizer"])
        if discriminator is not None and blob.get("discriminator") is not None:
            discriminator.load_state_dict(blob["discriminator"])
            disc_optimizer.load_state_dict(blob["disc_optimizer"])
        start_epoch, best = int(blob["epoch"]), float(blob["best"])
        del blob
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"resumed at epoch {start_epoch}", flush=True)

    (run_dir / "config.json").write_text(json.dumps(
        {**config, "_finetune": {"arm": arm, "base": str(args.checkpoint),
                                 "base_epoch": int(base["epoch"]), "lr": args.lr,
                                 "disc_lr": args.disc_lr, "disc_start": args.disc_start,
                                 "cldice_weight": args.cldice_weight, "alpha": topo_alpha,
                                 "losses": spec.get("losses", {}),
                                 "adv_weight": args.adv_weight, "crop": list(args.crop),
                                 "every": args.every,
                                 "train_projector": bool(args.train_projector)}},
        indent=2), encoding="utf-8")

    losses_spec = config["losses"]
    training = config["training"]
    amp = torch.bfloat16 if training["amp_dtype"] == "bfloat16" else torch.float16
    m_gate = float(losses_spec.get("x0_m_gate", 1.0))
    stats = train_set.stats[train_set.target_modality]
    loader = DataLoader(train_set, batch_size=1, shuffle=True,
                        num_workers=int(training["workers"]), drop_last=True,
                        pin_memory=(device == "cuda"))
    global_step = start_epoch * args.steps
    started = time.time()

    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        if args.train_projector:
            projector.train()
        totals: dict[str, float] = {}
        steps = 0
        epoch_started = time.time()
        iterator = iter(loader)
        while steps < args.steps:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader); continue
            x0 = batch["target"].to(device, non_blocking=True)
            condition = batch["source"].to(device, non_blocking=True)
            frangi = batch["frangi"].to(device, non_blocking=True)
            band = batch["band"].to(device, non_blocking=True)

            lr = args.lr * 0.5 * (1.0 + math.cos(
                math.pi * min(1.0, global_step / max(args.epochs * args.steps, 1))))
            for group in optimizer.param_groups:
                group["lr"] = lr

            t = bridge.sample_timesteps(x0.shape[0], device)
            noise = torch.randn_like(x0)
            with torch.amp.autocast("cuda", dtype=amp, enabled=(device == "cuda")):
                if args.train_projector:
                    y = projector(condition)
                else:
                    with torch.no_grad():
                        y = projector(condition)
                x_t, target = bridge.add_noise(x0, y.float(), t, noise)
                prediction = model(x_t.to(x0.dtype), t, condition)
                bridge_loss = F.mse_loss(prediction.float(), target.float())
                loss = float(losses_spec["bridge_mse"]) * bridge_loss
                terms = {"bridge_mse": bridge_loss.detach()}

                endpoint_loss = F.l1_loss(y.float(), x0.float())
                terms["endpoint_l1"] = endpoint_loss.detach()
                if args.train_projector:
                    loss = loss + float(losses_spec["endpoint_l1"]) * endpoint_loss

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

            # VQGAN's lambda balances the adversarial gradient against the RECONSTRUCTION
            # gradient, which here is the diffusion objective -- not the decoded topology
            # terms. Capturing it before those are added is both more faithful to the
            # method and far cheaper: passing the full loss made autograd.grad backprop
            # through the VAE decoder and the iterative soft-skeletonisation twice per
            # call (retain_graph=True), which took the `both` arm from 191 s/epoch to
            # 1823 s the moment the adversarial gate opened. `cldice` never calls lambda
            # and `adv` has no decoded term in its loss, so only the combination hit it.
            diffusion_loss = loss

            # ---------------- decoded-crop terms (clDice / adversarial) ----------------
            decoded_pred = decoded_true = None
            use_decode = (spec["cldice"] or spec["adv"]) \
                and bool(gate.any()) and (global_step % max(args.every, 1) == 0)
            if use_decode:
                zp, zt = crop_latent(bridge.to_x0(x_t, prediction.float())[gate][:1],
                                     x0[gate][:1], size=tuple(args.crop))
                with torch.amp.autocast("cuda", dtype=amp, enabled=(device == "cuda")):
                    decoded_pred = vae.decode(destandardize_t(zp, stats).to(zp.dtype))
                with torch.inference_mode():
                    with torch.amp.autocast("cuda", dtype=amp, enabled=(device == "cuda")):
                        decoded_true = vae.decode(destandardize_t(zt, stats).to(zt.dtype))
                decoded_true = decoded_true.clone().float()

            if spec["cldice"] and use_decode:
                # Same gating and [:1] selection the crop used, so the band belongs to the
                # case this crop came from.
                band_sel = band[gate][:1]
                p_pred = vessel_probability_banded(decoded_pred.float(), band_sel)
                p_true = vessel_probability_banded(decoded_true, band_sel)
                # Shit et al.'s actual formulation: clDice REGULARISES Dice, it does not
                # replace it. Used alone it is maximised by thickening (see docstring).
                cl = soft_cldice_loss(p_pred, p_true)
                sd = soft_dice_loss(p_pred, p_true)
                term = topo_alpha * cl + (1.0 - topo_alpha) * sd
                loss = loss + args.cldice_weight * term
                terms["cldice"] = cl.detach()
                terms["sdice"] = sd.detach()
                # The diagnostic that exposed the blob: soft mass of the prediction over
                # the target's. 1.0 is correct; the clDice-only arm reached 2.77.
                terms["band_ratio"] = (p_pred.sum() / p_true.sum().clamp_min(1e-6)).detach()

            # Diagnostic that costs nothing: whether an arm without the topology term is
            # buying its score by brightening (the failure the MIP discriminator rewarded).
            if use_decode:
                with torch.no_grad():
                    if not spec["cldice"]:
                        band_sel = band[gate][:1]
                        terms["band_ratio"] = (
                            vessel_probability_banded(decoded_pred.detach().float(), band_sel).sum()
                            / vessel_probability_banded(decoded_true, band_sel).sum().clamp_min(1e-6))

            adv_w = 0.0
            if spec["adv"] and use_decode and global_step >= args.disc_start:
                parts = []
                if discriminator is not None:
                    mips_fake = [soft_mip(decoded_pred.float(), ax, args.tau, gumbel=False)
                                 for ax in args.mip_axes]
                    with torch.amp.autocast("cuda", dtype=amp, enabled=(device == "cuda")):
                        g_mip = torch.stack([hinge_g_loss(discriminator(m, t[gate][:1]))
                                             for m in mips_fake]).mean()
                    parts.append((1.0, g_mip))
                    terms["adv_g"] = g_mip.detach()
                g_adv = sum(weight * term for weight, term in parts)
                lam = adaptive_weight(diffusion_loss, g_adv, model.conv_out.weight)
                adv_w = float(args.adv_weight * lam)
                loss = loss + adv_w * g_adv
                terms["adv_lambda"] = lam

            if not torch.isfinite(loss):
                print(f"non-finite loss at epoch {epoch} step {steps}; skipping", flush=True)
                optimizer.zero_grad(set_to_none=True); continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(trainable, float(training["grad_clip"]))
            if torch.isfinite(norm):
                optimizer.step()

            # ---------------- discriminator step ----------------
            if spec["adv"] and use_decode and discriminator is not None:
                with torch.amp.autocast("cuda", dtype=amp, enabled=(device == "cuda")):
                    real_logits, fake_logits = [], []
                    for ax in args.mip_axes:
                        real_logits.append(discriminator(decoded_true.amax(dim=ax), t[gate][:1]))
                        fake_logits.append(discriminator(
                            soft_mip(decoded_pred.detach().float(), ax, args.tau, gumbel=False),
                            t[gate][:1]))
                    d_loss = torch.stack([hinge_d_loss(r, f)
                                          for r, f in zip(real_logits, fake_logits)]).mean()
                disc_optimizer.zero_grad(set_to_none=True)
                d_loss.backward()
                d_norm = torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                if torch.isfinite(d_norm):
                    disc_optimizer.step()
                terms["adv_d"] = d_loss.detach()
                with torch.no_grad():
                    terms["d_sep"] = (torch.stack([r.float().mean() for r in real_logits]).mean()
                                      - torch.stack([f.float().mean() for f in fake_logits]).mean())

            if spec["adv"] and use_decode:
                terms["adv_weight"] = torch.tensor(adv_w)

            with torch.no_grad():
                decay = float(training["ema_decay"])
                for tp, sp in zip(ema.parameters(), model.parameters()):
                    tp.mul_(decay).add_(sp.detach(), alpha=1.0 - decay)
                for tb, sb in zip(ema.buffers(), model.buffers()):
                    tb.copy_(sb)
                if args.train_projector:
                    for tp, sp in zip(ema_projector.parameters(), projector.parameters()):
                        tp.mul_(decay).add_(sp.detach(), alpha=1.0 - decay)

            totals["loss"] = totals.get("loss", 0.0) + float(loss.detach())
            for k, v in terms.items():
                totals[k] = totals.get(k, 0.0) + float(v)
            steps += 1; global_step += 1
            if steps % 100 == 0:
                # NO empty_cache() here. It was tried against an apparent epoch-time blowup
                # and made it far worse (411 s -> 1706 s): with expandable_segments the
                # allocator has to re-expand the segments it just released. The blowup was
                # self-inflicted anyway -- GPU-side test scripts were run alongside training
                # and competed for the card.
                print(f"  [{arm}] epoch {epoch} step {steps}/{args.steps} "
                      f"loss {totals['loss']/steps:.5f} [{(time.time()-started)/3600:.2f} h]",
                      flush=True)

        record = {"arm": arm, "epoch": epoch, "lr": lr, "time_sec": time.time() - epoch_started,
                  **{k: v / max(steps, 1) for k, v in totals.items()}}
        with (run_dir / "training.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        print(f"[{arm}] epoch {epoch}: loss {record['loss']:.5f} "
              f"bridge {record['bridge_mse']:.5f} "
              f"cldice {record.get('cldice', float('nan')):.4f} "
              f"sdice {record.get('sdice', float('nan')):.4f} "
              f"band {record.get('band_ratio', float('nan')):.2f} "
              f"d_sep {record.get('d_sep', float('nan')):+.4f} "
              f"({record['time_sec']:.0f}s)", flush=True)
        torch.save({"model": model.state_dict(), "projector": projector.state_dict(),
                    "ema": ema.state_dict(), "ema_projector": ema_projector.state_dict(),
                    "optimizer": optimizer.state_dict(), "epoch": epoch, "best": best,
                    "config": config, "arm": arm,
                    "discriminator": None if discriminator is None else discriminator.state_dict(),
                    "disc_optimizer": None if disc_optimizer is None else disc_optimizer.state_dict()},
                   latest)

        # Periodic snapshots. latest.pt is overwritten every epoch and best.pt tracks Dice
        # only, so without these any epoch that is neither the Dice peak nor the last one is
        # unrecoverable -- which cost the cldice arm its epoch-20 and epoch-30 weights when
        # they were wanted for previews. EMA weights only, ~187 MB each.
        if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
            torch.save({"ema": ema.state_dict(), "ema_projector": ema_projector.state_dict(),
                        "epoch": epoch, "config": config, "arm": arm},
                       run_dir / f"epoch_{epoch:04d}.pt")

        if epoch % args.val_every == 0 or epoch == args.epochs:
            # Release around validation, NOT inside the step loop. Validation samples and
            # decodes whole volumes, leaving large cached blocks that do not fit the training
            # working set. Measured on `both`: epochs 1-10 ran 342-353 s, epoch 11 onward
            # 1172-1250 s -- a permanent 3.5x step change immediately after the first
            # validation, in a FRESH run. `cldice`, same val_every but no discriminator or
            # lambda autograd, is unaffected (172 s -> 188 s), so it is head-room, not a leak.
            # Twice per 10 epochs costs nothing; the earlier every-100-steps version fought
            # expandable_segments and made things 4x worse.
            if device == "cuda":
                torch.cuda.empty_cache()
            summary = validate(ema, ema_projector, bridge, val_set, config, vae, device)
            if device == "cuda":
                torch.cuda.empty_cache()
            summary.update({"arm": arm, "epoch": epoch})
            with (run_dir / "validation.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(summary, default=float) + "\n")
            score = best_dice(summary)
            # Report from the LONGEST sampler in the ladder, whatever it is -- hardcoding
            # "bridge50" silently prints nan the moment validation.steps changes.
            longest = max((int(k[6:].split("_")[0]) for k in summary
                           if k.startswith("bridge") and k.endswith("_vessel_dice")), default=0)
            print(f"  [{arm}] validation: best-sampler Dice {score:.4f} | "
                  f"endpoint {summary.get('endpoint_vessel_dice', float('nan')):.4f} | "
                  f"clDice {summary.get(f'bridge{longest}_cldice', float('nan')):.4f} | "
                  f"p99.9 {summary.get(f'bridge{longest}_p999', float('nan')):.4f}", flush=True)
            if score > best:
                best = score
                torch.save({"ema": ema.state_dict(), "ema_projector": ema_projector.state_dict(),
                            "epoch": epoch, "best": best, "config": config, "arm": arm,
                            "validation": summary}, run_dir / "best.pt")
                print(f"  [{arm}] new best {best:.4f}", flush=True)
    return {"arm": arm, "run_dir": str(run_dir), "best": best}


# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", nargs="+", default=["delivered"],
                        choices=sorted(ARMS) + ["all"], metavar="ARM")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/ldm_bbdm/best.pt")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "bbdm.json")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="override output.run_dir so arms land beside the base run")
    parser.add_argument("-n", "--epochs", type=int, default=40)
    parser.add_argument("--steps", type=int, default=455)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--disc-lr", type=float, default=2e-4)
    parser.add_argument("--disc-base", type=int, default=32)
    parser.add_argument("--disc-start", type=int, default=500)
    parser.add_argument("--cldice-weight", type=float, default=0.25,
                        help="weight on the combined topology term")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="clDice fraction of the topology term; the rest is soft Dice. "
                             "alpha=1.0 reproduces the clDice-only arm that over-segmented "
                             "2.77x. Shit et al. use clDice as a regulariser on Dice.")
    parser.add_argument("--adv-weight", type=float, default=0.25)
    parser.add_argument("--loss", action="append", default=None, metavar="KEY=VALUE",
                        help="override a config['losses'] weight, repeatable. Distal vessels "
                             "are partially predictable (probe AUC 0.83) but are ~1 %% of "
                             "voxels, so they may simply be outweighted rather than "
                             "unlearnable: --loss frangi_l1=4.0")
    parser.add_argument("--crop", type=int, nargs=3, default=(128, 144, 4),
                        help="latent crop; default is a SLAB (full in-plane, thin in depth) "
                             "so the axial MIP is a whole-brain projection")
    parser.add_argument("--mip-axes", type=int, nargs="+", default=(-1,),
                        help="axes to project for the adversarial term. -1 is axial, the "
                             "view the metrics use; a slab's other two MIPs are degenerate")
    parser.add_argument("--every", type=int, default=2)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--bands", type=Path, default=ROOT / "Dataset" / "vessel_bands.json",
                        help="per-case whole-volume (p98, p99.5) of the target MRA -- the "
                             "support vessel_scores thresholds on. See the module docstring "
                             "for why a fixed brain-masked band was wrong.")
    parser.add_argument("--val-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=10,
                        help="also snapshot EMA weights every N epochs, so a checkpoint that "
                             "is neither the Dice peak nor the final one stays recoverable; "
                             "0 disables")
    parser.add_argument("--train-projector", action="store_true")
    parser.add_argument("--scratch", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--memory-fraction", type=float, default=0.0)
    args = parser.parse_args()

    if args.memory_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.run_dir:
        config["output"]["run_dir"] = str(args.run_dir)
    for item in (args.loss or []):
        key, _, value = item.partition("=")
        if key not in config["losses"]:
            raise SystemExit(f"unknown loss key '{key}'. Available: "
                             f"{sorted(k for k in config['losses'] if not k.startswith('_'))}")
        old = config["losses"][key]
        config["losses"][key] = float(value)
        print(f"loss override: {key} {old} -> {config['losses'][key]}", flush=True)
    if args.smoke:
        # A smoke run must never write into a real run directory.
        args.run_dir = args.run_dir or Path("checkpoints/_smoke")
        config["output"]["run_dir"] = str(args.run_dir)
        args.epochs, args.steps, args.val_every, args.disc_start = 1, 6, 999, 2
        config["training"]["workers"] = 0
        config["validation"].update({"cases": 1, "steps": [2]})
    torch.manual_seed(config["training"]["seed"]); np.random.seed(config["training"]["seed"])
    random.seed(config["training"]["seed"])

    if not args.bands.exists():
        raise SystemExit(
            f"missing {args.bands}. The clDice loss needs each case's own vessel band on the\n"
            f"metric's support -- the old fixed band left 64 % of scored vessel voxels at zero\n"
            f"weight. Generate it first (see the module docstring).")
    bands = json.loads(args.bands.read_text(encoding="utf-8"))

    print("loading latents", flush=True)
    seed = config["training"]["seed"]
    train_set = BandedLatentDataset(config, "train", seed=seed, bands=bands)
    vae = load_vaev4(ROOT / config["vae"]["checkpoint"], device)
    lo = np.array([train_set.bands[c][0] for c in train_set.cases])
    hi = np.array([train_set.bands[c][1] for c in train_set.cases])
    print(f"vessel band: per case, whole-volume (p98, p99.5) -- "
          f"lo {lo.mean():.4f}+-{lo.std():.4f}, hi {hi.mean():.4f}+-{hi.std():.4f}", flush=True)

    val_set = BandedLatentDataset(config, "val", seed=seed, bands=bands)
    arms = list(ARMS) if "all" in args.arms else list(dict.fromkeys(args.arms))
    results = []
    for arm in arms:
        results.append(finetune(arm, args, config, train_set, val_set, vae, device))
    print()
    for r in results:
        print(f"  {r['arm']:>8}  best-sampler Dice {r['best']:.4f}  {r['run_dir']}")


if __name__ == "__main__":
    main()
