"""Brownian-bridge diffusion (BBDM) between the projected condition and the MRA latent."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.paths import ROOT, mask_path_for
from modules.metrics import masked_ssim_and_psnr, vessel_scores, mip_ssim, hard_cldice
from modules.vae_mra import decode_tiled
from modules.latent_data import PairedLatentDataset, destandardize_t
from modules.unet import Normalize


class BridgeProjector(nn.Module):
    """y = P(cond): maps the 24 source channels into the 16-channel target space, giving
    the bridge somewhere to start. Trained both through the bridge and by an explicit L1
    against x_0, so it learns to be a genuine estimate rather than an arbitrary anchor."""

    def __init__(self, in_channels: int, out_channels: int, base: int = 64,
                 blocks: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv3d(in_channels, base, 3, padding=1)]
        for _ in range(blocks):
            layers += [Normalize(base), nn.SiLU(), nn.Conv3d(base, base, 3, padding=1)]
        layers += [Normalize(base), nn.SiLU(), nn.Conv3d(base, out_channels, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        return self.net(condition)


class BrownianBridge:
    def __init__(self, steps: int, max_variance: float = 1.0, device: str = "cuda") -> None:
        self.steps = int(steps)
        t = torch.arange(self.steps, dtype=torch.float64) / max(self.steps - 1, 1)
        m = t.clamp(0.0, 1.0)
        # delta_t = 2 s (m - m^2): zero at both endpoints, peaking at m = 1/2.
        # `max_variance` scales the peak; Li et al. use s = 1, giving delta_max = 0.5.
        delta = 2.0 * float(max_variance) * (m - m * m)
        self.m = m.float().to(device)
        self.delta = delta.clamp_min(0.0).float().to(device)
        self.sqrt_delta = self.delta.sqrt()

    def sample_timesteps(self, batch: int, device: str) -> torch.Tensor:
        return torch.randint(0, self.steps, (batch,), device=device)

    def _view(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (-1,) + (1,) * (x.dim() - 1)
        return self.m[t].view(shape), self.sqrt_delta[t].view(shape)

    def add_noise(self, x0: torch.Tensor, y: torch.Tensor, t: torch.Tensor,
                  noise: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (x_t, target) where target = x_t - x_0, the quantity the network predicts."""
        m, sd = self._view(x0, t)
        x_t = (1.0 - m) * x0 + m * y + sd * noise
        return x_t, x_t - x0

    @staticmethod
    def to_x0(x_t: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        return x_t - prediction

    @torch.inference_mode()
    def sample(self, model, projector, condition: torch.Tensor, shape, device: str,
               steps: int, clamp: float = 6.0, eta: float = 0.0, generator=None,
               eta_mask: torch.Tensor | None = None,
               eta_anneal_from: float | None = None) -> torch.Tensor:
        """Bridge sampling. Start at x_T = y (no noise: delta_T = 0), then walk down. At each
        step estimate x_0, recover the noise consistent with the forward process at t, and
        re-apply the forward formula at t-1.

        `eta` interpolates deterministic -> stochastic, exactly as DDIM's eta does:

            noise = sqrt(1 - eta^2) * eps_recovered  +  eta * z,     z ~ N(0, I)

        The two terms are independent and the weights are the legs of a unit right triangle,
        so the injected variance is (1 - eta^2) + eta^2 = 1 and the state keeps the forward
        marginal N((1-m)x_0 + m*y, delta_t I) at every step. eta=0 is the original
        deterministic path; eta=1 redraws the noise each step.

        WHY THIS EXISTS: with eta=0 nothing in the reverse process is random, so the network
        must PREDICT parenchymal texture from the conditioning -- and texture is not
        predictable from T1/T2/PD, so an MSE objective averages it away. Measured: the VAE
        round-trip preserves 98.6 % of the target's texture amplitude, the bridge endpoint
        11 %, and the eta=0 sampler only 42.6 %. The deficit is above ~0.05 cycles/voxel."""
        y = projector(condition)
        x = y.clone()
        schedule = torch.linspace(self.steps - 1, 0, steps, device=device).long()
        for index, t in enumerate(schedule):
            bt = t.repeat(shape[0])
            x0 = self.to_x0(x, model(x, bt, condition)).clamp(-clamp, clamp)
            m, sd = self._view(x, bt)
            # noise implied by (x, x0, y) at this t; zero when delta_t is zero
            eps = torch.where(sd > 1e-6, (x - (1.0 - m) * x0 - m * y) / sd.clamp_min(1e-6),
                              torch.zeros_like(x))
            nxt = schedule[index + 1] if index + 1 < len(schedule) else None
            if nxt is None:
                x = x0
                break
            bn = nxt.repeat(shape[0])
            m_n, sd_n = self._view(x, bn)
            # A UNIFORM eta buys texture and wrecks vessels: measured 42.6 % -> 75 % of the
            # target's texture, but the vessel band inflated to 2.66x and every fidelity
            # metric fell. The deficit is in PARENCHYMA while the damage is in VESSELS, so
            # both knobs here aim the noise away from vessels:
            #   eta_mask         per-voxel scale in [0,1] -- pass (1 - vesselness) to inject
            #                    only where there is no vessel
            #   eta_anneal_from  fraction of the schedule after which eta ramps to 0, so the
            #                    final steps that set vessel geometry stay deterministic
            e = float(eta)
            if e > 0.0 and eta_anneal_from is not None:
                frac = index / max(len(schedule) - 1, 1)
                if frac >= eta_anneal_from:
                    e *= max(0.0, 1.0 - (frac - eta_anneal_from) /
                             max(1.0 - eta_anneal_from, 1e-6))
            if e > 0.0:
                z = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
                # The mask scales ETA PER VOXEL, not the noise. Masking z directly would
                # leave sqrt(1-e^2)*eps behind wherever the mask is 0 -- variance 1-e^2
                # instead of 1, which shrinks the state and breaks the forward marginal.
                # With a per-voxel e_v the weights stay the legs of a unit right triangle
                # everywhere, so e_v = 0 recovers the deterministic path exactly.
                e_v = e * eta_mask.to(z.dtype) if eta_mask is not None else \
                    torch.full_like(z, e)
                noise = (1.0 - e_v * e_v).clamp_min(0.0).sqrt() * eps + e_v * z
            else:
                noise = eps
            x = (1.0 - m_n) * x0 + m_n * y + sd_n * noise
        return x

    @torch.inference_mode()
    def endpoint(self, projector, condition: torch.Tensor) -> torch.Tensor:
        """The bridge's starting point on its own -- the regression estimate. Logged every
        validation so 'the bridge improves on its endpoint' is a measured claim."""
        return projector(condition)


@torch.inference_mode()
def validate(model, projector, bridge: BrownianBridge, dataset: PairedLatentDataset,
             config: dict, vae, device: str) -> dict:
    spec = config["validation"]
    stats = dataset.stats[dataset.target_modality]
    cases = dataset.cases[:int(spec["cases"])]
    counts = spec["steps"]
    counts = [counts] if isinstance(counts, int) else list(counts)
    rows: dict[str, list[float]] = {}
    truth = brain = None

    for case in cases:
        target_latent, condition = dataset.full_case(case)
        condition = condition[None].to(device)
        shape = (1,) + tuple(target_latent.shape)
        modes = {f"bridge{k}": bridge.sample(model, projector, condition, shape, device, k)
                 for k in counts}
        # The endpoint alone: the bridge's own regression estimate, before any refinement.
        # Logged so "the reverse process improves on its endpoint" stays a measured claim.
        modes["endpoint"] = bridge.endpoint(projector, condition)

        source = ROOT / "Dataset" / "split_numpy" / dataset.split / "MRA" / f"{case}-MRA.npy"
        truth = np.asarray(np.load(source), np.float32)
        brain = np.asarray(np.load(mask_path_for(source, dataset.split))) > 0
        for name, sampled in modes.items():
            rows.setdefault(f"{name}_latent_mse", []).append(
                float(F.mse_loss(sampled[0].float().cpu(), target_latent)))
            if not spec.get("decode", True) or vae is None:
                continue
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                recon = decode_tiled(vae, destandardize_t(sampled, stats).to(sampled.dtype))
            recon = np.clip(recon[0, 0].float().cpu().numpy(), 0.0, 1.0)
            _, masked, psnr = masked_ssim_and_psnr(truth, recon, brain)
            dice, _ = vessel_scores(truth, recon)
            for key, value in (("masked_ssim", masked), ("psnr", psnr), ("vessel_dice", dice),
                               ("mip_ssim", mip_ssim(truth, recon)),
                               ("cldice", hard_cldice(truth, recon)),
                               ("p999", float(np.percentile(recon[brain], 99.9)))):
                rows.setdefault(f"{name}_{key}", []).append(value)
        torch.cuda.empty_cache()
    summary = {k: float(np.mean(v)) for k, v in rows.items()}
    if truth is not None:
        summary["target_p999"] = float(np.percentile(truth[brain], 99.9))
    return summary
