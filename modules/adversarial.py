"""Discriminators and hinge-GAN helpers for the VAE and bridge fine-tuning arms."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.unet import sinusoidal_embedding


class PatchDiscriminator3D(nn.Module):
    """Fully convolutional, so it accepts every extent the sampler produces."""

    def __init__(self, base: int = 32, levels: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv3d(1, base, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
        channels = base
        for index in range(1, levels):
            nxt = min(base * (2 ** index), 256)
            layers += [nn.Conv3d(channels, nxt, 4, 2, 1),
                       nn.GroupNorm(8, nxt),
                       nn.LeakyReLU(0.2, inplace=True)]
            channels = nxt
        layers += [nn.Conv3d(channels, 1, 3, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def soft_mip(x: torch.Tensor, axis: int, tau: float = 0.1,
             gumbel: bool = True) -> torch.Tensor:
    """Differentiable MIP. A hard `amax` routes gradient to ONE voxel per projection ray;
    this spreads it across the plausible maxima, which is the point of Yu et al.'s Gumbel
    formulation. Theirs accumulates over slice prefixes; this is the single-projection
    variant, which keeps the gradient behaviour at a fraction of the cost."""
    # Scale FIRST, then add noise. Writing softmax((x + G)/tau) -- the paper's notation
    # read literally -- divides signal and noise together, so their ratio never changes:
    # with x in [0,1] and Gumbel noise of std 1.28 the noise wins the ranking at every
    # temperature and the "soft max" returns a near-random voxel. Verified by the unit
    # test: soft_mip_loss(x, x) returned 2.30 instead of ~0.
    logits = x / max(tau, 1e-6)
    if gumbel:
        u = torch.rand_like(x)
        logits = logits + (-torch.log(-torch.log(u + 1e-20) + 1e-20))
    weights = torch.softmax(logits, dim=axis)
    return (weights * x).sum(dim=axis)


class MIPDiscriminator2D(nn.Module):
    """2-D PatchGAN over maximum-intensity projections of the decoded crop.

    Two deliberate choices, both from REVIEW_2026-09-02 section 8: the discriminator is
    2-D on MIPs rather than 3-D on latents, because MIP is where vessel sharpness and
    continuity are actually judged and a 2-D net is far cheaper; and it is CONDITIONED ON
    THE TIMESTEP, because the previous latent discriminator saw x0_pred at every noise
    level without knowing which, so real/fake was separable from blur alone."""

    def __init__(self, base: int = 32, levels: int = 3, time_dim: int = 128) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(1, base, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        channels = base
        for index in range(1, levels):
            nxt = min(base * 2 ** index, 256)
            layers += [nn.Conv2d(channels, nxt, 4, 2, 1),
                       nn.GroupNorm(min(8, nxt), nxt), nn.LeakyReLU(0.2, True)]
            channels = nxt
        self.features = nn.Sequential(*layers)
        self.head = nn.Conv2d(channels, 1, 3, padding=1)
        self.time_dim = int(time_dim)
        # Miyato projection conditioning: <phi(t), pooled features> added to the logit
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, channels), nn.SiLU(),
                                      nn.Linear(channels, channels))
        nn.init.zeros_(self.time_mlp[-1].weight)
        nn.init.zeros_(self.time_mlp[-1].bias)

    def forward(self, image: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        h = self.features(image)
        logit = self.head(h)
        embedding = self.time_mlp(sinusoidal_embedding(timesteps, self.time_dim))
        projection = (h.mean(dim=(2, 3)) * embedding).sum(dim=1)
        return logit + projection[:, None, None, None]


def hinge_d_loss(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    return 0.5 * (F.relu(1.0 - real).mean() + F.relu(1.0 + fake).mean())


def hinge_g_loss(fake: torch.Tensor) -> torch.Tensor:
    return -fake.mean()


def adaptive_weight(rec_loss: torch.Tensor, adv_loss: torch.Tensor,
                    last_layer: torch.Tensor, maximum: float = 1e4) -> torch.Tensor:
    """VQGAN's lambda. A fixed adversarial weight silently becomes dominant as the
    discriminator sharpens; balancing gradient norms at the decoder's output layer keeps
    the adversarial pull at a constant fraction of the reconstruction pull."""
    try:
        g_rec = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]
        g_adv = torch.autograd.grad(adv_loss, last_layer, retain_graph=True)[0]
    except RuntimeError:
        return torch.tensor(0.0, device=last_layer.device)
    return (g_rec.float().norm() / (g_adv.float().norm() + 1e-4)).clamp(0.0, maximum).detach()
