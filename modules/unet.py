"""Conditional 3D UNet with AdaGN conditioning (the bridge's denoiser) and its building blocks."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def Normalize(channels: int, num_groups: int = 32) -> nn.GroupNorm:
    groups = min(int(num_groups), int(channels))
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(num_groups=groups, num_channels=channels, eps=1e-6)


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(half, 1)
    )
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class Downsample3D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample3D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class CondEncoder3D(nn.Module):
    """Multi-scale features from the source latents, one map per UNet level."""

    def __init__(self, in_channels: int, base: int, multipliers: list[int]) -> None:
        super().__init__()
        self.stem = nn.Conv3d(in_channels, base, 3, padding=1)
        self.stages = nn.ModuleList()
        self.downs = nn.ModuleList()
        current = base
        for index, mult in enumerate(multipliers):
            width = base * mult
            self.stages.append(nn.Sequential(
                Normalize(current), nn.SiLU(), nn.Conv3d(current, width, 3, padding=1),
                Normalize(width), nn.SiLU(), nn.Conv3d(width, width, 3, padding=1)))
            current = width
            self.downs.append(nn.Identity() if index == len(multipliers) - 1
                              else Downsample3D(current))
        self.widths = [base * m for m in multipliers]

    def forward(self, condition: torch.Tensor) -> list[torch.Tensor]:
        h = self.stem(condition)
        out = []
        for stage, down in zip(self.stages, self.downs):
            h = stage(h)
            out.append(h)
            h = down(h)
        return out


class AdaGNResBlock3D(nn.Module):
    """Residual block whose second normalisation is modulated by BOTH the timestep (a
    vector) and the conditioning feature (a spatial map). Both projections are
    zero-initialised, so at step 0 the block is exactly a plain residual block and the
    conditioning cannot destabilise a fresh network."""

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, cond_ch: int,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = Normalize(in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        groups = Normalize(out_ch).num_groups
        self.norm2 = nn.GroupNorm(groups, out_ch, eps=1e-6, affine=False)
        self.dropout = nn.Dropout(float(dropout))
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, 2 * out_ch)
        self.cond_proj = nn.Conv3d(cond_ch, 2 * out_ch, 1)
        nn.init.zeros_(self.emb_proj.weight); nn.init.zeros_(self.emb_proj.bias)
        nn.init.zeros_(self.cond_proj.weight); nn.init.zeros_(self.cond_proj.bias)
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor,
                cond: torch.Tensor | None) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb_proj(F.silu(emb)).chunk(2, dim=1)
        scale = scale[:, :, None, None, None]
        shift = shift[:, :, None, None, None]
        if cond is not None:
            if cond.shape[-3:] != h.shape[-3:]:
                cond = F.interpolate(cond, size=h.shape[-3:], mode="nearest")
            c_scale, c_shift = self.cond_proj(cond).chunk(2, dim=1)
            scale = scale + c_scale
            shift = shift + c_shift
        h = self.norm2(h) * (1.0 + scale) + shift
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class SelfAttention3D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = Normalize(channels)
        self.qkv = nn.Conv3d(channels, channels * 3, 1)
        self.proj = nn.Conv3d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, c, d * h * w).unbind(1)
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                             v.transpose(1, 2))
        return x + self.proj(out.transpose(1, 2).reshape(b, c, d, h, w))


class AdaGNUNet3D(nn.Module):
    """v2 UNet. Conditioning enters through AdaGN at every level, not as an input concat,
    and a learned null embedding provides the unconditional branch that CFG needs."""

    def __init__(self, target_channels: int, source_channels: int, base: int,
                 multipliers: list[int], blocks: int, attention_levels: list[int],
                 dropout: float, time_embed_dim: int, cond_base: int) -> None:
        super().__init__()
        self.base = int(base)
        self.target_channels = int(target_channels)
        self.time_mlp = nn.Sequential(nn.Linear(base, time_embed_dim), nn.SiLU(),
                                      nn.Linear(time_embed_dim, time_embed_dim))
        self.cond_encoder = CondEncoder3D(source_channels, cond_base, multipliers)
        # learned null conditioning: what the model sees when the condition is dropped
        self.null_cond = nn.Parameter(torch.zeros(1, source_channels, 1, 1, 1))

        channels = [base * m for m in multipliers]
        cond_widths = self.cond_encoder.widths
        self.conv_in = nn.Conv3d(target_channels, channels[0], 3, padding=1)

        self.down, self.downsample, self.down_attn = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        skips = [channels[0]]
        current = channels[0]
        for level, width in enumerate(channels):
            stage, attn = nn.ModuleList(), nn.ModuleList()
            for _ in range(blocks):
                stage.append(AdaGNResBlock3D(current, width, time_embed_dim,
                                             cond_widths[level], dropout))
                attn.append(SelfAttention3D(width) if level in attention_levels else nn.Identity())
                current = width
                skips.append(current)
            self.down.append(stage); self.down_attn.append(attn)
            last = level == len(channels) - 1
            self.downsample.append(nn.Identity() if last else Downsample3D(current))
            if not last:
                skips.append(current)

        self.mid1 = AdaGNResBlock3D(current, current, time_embed_dim, cond_widths[-1], dropout)
        self.mid_attn = SelfAttention3D(current)
        self.mid2 = AdaGNResBlock3D(current, current, time_embed_dim, cond_widths[-1], dropout)

        self.up, self.upsample, self.up_attn = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        self.up_levels = []
        for level in reversed(range(len(channels))):
            width = channels[level]
            stage, attn = nn.ModuleList(), nn.ModuleList()
            for _ in range(blocks + 1):
                stage.append(AdaGNResBlock3D(current + skips.pop(), width, time_embed_dim,
                                             cond_widths[level], dropout))
                attn.append(SelfAttention3D(width) if level in attention_levels else nn.Identity())
                current = width
            self.up.append(stage); self.up_attn.append(attn)
            self.up_levels.append(level)
            self.upsample.append(Upsample3D(current) if level > 0 else nn.Identity())

        self.norm_out = Normalize(current)
        self.conv_out = nn.Conv3d(current, target_channels, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight); nn.init.zeros_(self.conv_out.bias)

    def forward(self, noisy: torch.Tensor, timesteps: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        emb = self.time_mlp(sinusoidal_embedding(timesteps, self.base))
        features = self.cond_encoder(condition)
        h = self.conv_in(noisy)
        skips = [h]
        for level, (stage, attn, down) in enumerate(zip(self.down, self.down_attn, self.downsample)):
            for block, at in zip(stage, attn):
                h = block(h, emb, features[level])
                h = at(h) if not isinstance(at, nn.Identity) else h
                skips.append(h)
            if not isinstance(down, nn.Identity):
                h = down(h)
                skips.append(h)
        h = self.mid2(self.mid_attn(self.mid1(h, emb, features[-1])), emb, features[-1])
        for stage, attn, up, level in zip(self.up, self.up_attn, self.upsample, self.up_levels):
            for block, at in zip(stage, attn):
                skip = skips.pop()
                if skip.shape[-3:] != h.shape[-3:]:
                    h = F.interpolate(h, size=skip.shape[-3:], mode="nearest")
                h = block(torch.cat([h, skip], dim=1), emb, features[level])
                h = at(h) if not isinstance(at, nn.Identity) else h
            h = up(h)
        return self.conv_out(F.silu(self.norm_out(h)))

    def null_condition(self, like: torch.Tensor) -> torch.Tensor:
        return self.null_cond.expand(like.shape[0], -1, *like.shape[2:]).to(like.dtype)


def build_model(config: dict, target_channels: int, source_channels: int, device: str):
    m = config["model"]
    return AdaGNUNet3D(target_channels, source_channels, int(m["base_channels"]),
                       list(m["channel_multipliers"]), int(m["blocks_per_level"]),
                       list(m["attention_levels"]), float(m["dropout"]),
                       int(m["time_embed_dim"]), int(m["cond_base_channels"])).to(device)
