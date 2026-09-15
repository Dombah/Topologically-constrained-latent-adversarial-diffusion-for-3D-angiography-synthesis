"""MRA autoencoder (VAEv4 / v5): architecture, tiled encode/decode, crop dataset, losses,
evaluation and checkpoint I/O. Training loops live in train_vae_mra.py and finetune_vae_mra.py."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from modules.paths import ROOT, DATA_ROOT, volume_paths, mask_path_for
from modules.metrics import (masked_ssim_and_psnr, vessel_scores, mip_ssim, hard_cldice,
                             latent_diagnostics)


CACHE_ROOT = DATA_ROOT / "cache" / "vaev4"


RUN_ROOT = ROOT / "checkpoints" / "vaev4_mra"


TARGET_SHAPE = (512, 576, 96)


BASE_CHANNELS = 12


CHANNEL_MULTIPLIERS = (1, 2, 4)


NUM_RES_BLOCKS = 1


DOWNSAMPLE_FACTOR = 4


# (patch size, batch size, sampling probability). Batch sizes hold peak VRAM near 7 GB;
# throughput is ~25 Mvoxel/s regardless of shape, so cost is proportional to voxels.
SIZE_SPECS = (
    ((96, 96, 96), 4, 0.40),
    ((64, 64, 64), 8, 0.15),
    ((128, 128, 64), 3, 0.20),
    ((160, 160, 96), 1, 0.15),
    ((192, 192, 96), 1, 0.10),
)


# Crop placement: vessel-centred / anywhere-in-brain / anywhere at all. The last bucket is
# what teaches the encoder that a mostly empty field of view is normal.
P_VESSEL, P_BRAIN = 0.55, 0.30


LR = 2e-4


WEIGHT_DECAY = 1e-2


GRAD_CLIP = 1.0


STEPS_PER_EPOCH = 1000


WARMUP_STEPS = 300


DECAY_START_EPOCH = 0      # cosine decay to 0 spans [DECAY_START_EPOCH, --epochs]; 0 = whole run.


W_L1 = 1.0


W_VESSEL_L1 = 4.0          # multiplies the vessel-band emphasis inside the L1 term


W_MIP = 0.15               # 3-view MIP L1 on the patch


W_BACKGROUND = 0.5         # false positives outside the brain mask


W_PERCEPTUAL = 0.05        # 2.5-D VGG16 (weights already cached locally)


W_EQUIVARIANCE = 0.25      # EQ-VAE / Skorokhodov rescale+rotate consistency


P_EQUIVARIANCE = 0.5


EQ_SCALES = (0.5, 0.625, 0.75)


W_SCALE = 0.1              # pins per-channel latent std to 1 (MAISI's criterion)


W_TAIL = 0.02              # penalty on |mu| beyond TAIL_SIGMA (absolute, since std is pinned)


TAIL_SIGMA = 4.0


KL_INIT = 1e-5


KL_WARMUP_EPOCHS = 4


KL_TARGET_STD = 1.0        # MAISI: keep the latent std in [0.9, 1.1]


KL_STD_TOLERANCE = 0.1


KL_ADJUST_RATE = 1.2


KL_BOUNDS = (1e-7, 3e-2)


VAL_EVERY = 10


VAL_CASES_PER_SCANNER = 4


FINAL_CASES = 24


FINAL_SKELETON_CASES = 8


LATENT_STATS_CASES = 60    # train volumes used for the exported per-channel statistics


# Acceptance gate (REVIEW_2026-09-02.md section 7, P3). Measured in the deployment regime.
ACCEPT = {  # keys must match what evaluate() emits: latent diagnostics carry the regime prefix
    "tiled_masked_ssim": (0.975, "ge"),
    "tiled_vessel_dice": (0.88, "ge"),
    "tiled_cldice": (0.85, "ge"),
    "extent_gap_masked_ssim": (0.01, "le"),
    "tiled_latent_kurtosis": (4.0, "le"),
    "tiled_latent_max_abs_sigma": (8.0, "le"),
    "tiled_latent_high_freq_fraction": (0.50, "le"),
}


# v3 measured on the VALIDATION split, so it is comparable with the numbers below. (An
# earlier version of this table quoted v3's test-split figures, 0.9793/0.887, against v4's
# validation figures, which flattered v3 by ~0.014 SSIM and ~0.028 Dice.) clDice is the one
# entry still taken from the test split, marked with * in the printout.
V3_REFERENCE = {
    "tiled_masked_ssim": 0.9662, "tiled_vessel_dice": 0.870, "tiled_cldice": 0.868,
    "extent_gap_masked_ssim": 0.0635, "tiled_latent_kurtosis": 8.41,
    "tiled_latent_max_abs_sigma": 19.2, "tiled_latent_high_freq_fraction": 0.655,
}


V3_REFERENCE_FROM_TEST_SPLIT = {"tiled_cldice"}


class ChannelRMSNorm3d(nn.Module):
    """RMS normalisation across channels at each voxel.

    The point of v4: statistics are taken over the channel axis only, so they are identical
    whether the input is a 64^3 patch or a 512x576x96 volume. GroupNorm with one or two
    channels per group, as in v2/v3, instead normalises over the whole spatial extent, which
    is why v3 behaves differently on patches and on volumes.
    """

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The reduction accumulates in fp32 but nothing full-size is upcast: on a whole
        # 512x576x96 volume an fp32 copy of every activation costs seconds and gigabytes.
        variance = (x * x).mean(dim=1, keepdim=True, dtype=torch.float32)
        scale = torch.rsqrt(variance + self.eps).to(x.dtype)
        shape = (1, -1) + (1,) * (x.dim() - 2)
        return x * scale * self.weight.view(shape).to(x.dtype) + self.bias.view(shape).to(x.dtype)


def make_norm(channels: int, kind: str) -> nn.Module:
    if kind == "channel_rms":
        return ChannelRMSNorm3d(channels)
    if kind == "group":  # v3 behaviour, kept for a controlled comparison
        groups = min(32, channels)
        while channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels, eps=1e-6)
    raise ValueError(f"unknown norm: {kind}")


class ResidualBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: str) -> None:
        super().__init__()
        self.norm1 = make_norm(in_ch, norm)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm2 = make_norm(out_ch, norm)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


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


class Encoder3D(nn.Module):
    def __init__(self, base: int, mults: tuple[int, ...], latent: int, blocks: int, norm: str) -> None:
        super().__init__()
        channels = [base * m for m in mults]
        self.conv_in = nn.Conv3d(1, base, 3, padding=1)
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        in_ch = base
        for level, out_ch in enumerate(channels):
            stage = nn.ModuleList([ResidualBlock3D(in_ch if i == 0 else out_ch, out_ch, norm) for i in range(blocks)])
            self.stages.append(stage)
            in_ch = out_ch
            self.downsamples.append(Downsample3D(out_ch) if level < len(channels) - 1 else nn.Identity())
        self.mid = nn.Sequential(ResidualBlock3D(in_ch, in_ch, norm), ResidualBlock3D(in_ch, in_ch, norm))
        self.norm_out = make_norm(in_ch, norm)
        self.conv_out = nn.Conv3d(in_ch, 2 * latent, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv_in(x)
        for stage, down in zip(self.stages, self.downsamples):
            for block in stage:
                h = block(h)
            h = down(h)
        h = self.mid(h)
        h = self.conv_out(F.silu(self.norm_out(h)))
        mu, logvar = h.chunk(2, dim=1)
        return mu, logvar.clamp(-10.0, 5.0)


class Decoder3D(nn.Module):
    def __init__(self, base: int, mults: tuple[int, ...], latent: int, blocks: int, norm: str) -> None:
        super().__init__()
        channels = [base * m for m in mults]
        self.conv_in = nn.Conv3d(latent, channels[-1], 3, padding=1)
        self.mid = nn.Sequential(ResidualBlock3D(channels[-1], channels[-1], norm),
                                 ResidualBlock3D(channels[-1], channels[-1], norm))
        self.stages = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        in_ch = channels[-1]
        for level in reversed(range(len(channels))):
            out_ch = channels[level]
            stage = nn.ModuleList([ResidualBlock3D(in_ch if i == 0 else out_ch, out_ch, norm) for i in range(blocks + 1)])
            self.stages.append(stage)
            in_ch = out_ch
            self.upsamples.append(Upsample3D(out_ch) if level > 0 else nn.Identity())
        self.norm_out = make_norm(in_ch, norm)
        self.conv_out = nn.Conv3d(in_ch, 1, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.mid(self.conv_in(z))
        for stage, up in zip(self.stages, self.upsamples):
            for block in stage:
                h = block(h)
            h = up(h)
        return torch.sigmoid(self.conv_out(F.silu(self.norm_out(h))))


class VAEv4(nn.Module):
    def __init__(self, latent_channels: int = 8, base: int = BASE_CHANNELS,
                 mults: tuple[int, ...] = CHANNEL_MULTIPLIERS, blocks: int = NUM_RES_BLOCKS,
                 norm: str = "channel_rms") -> None:
        super().__init__()
        self.config = {"latent_channels": int(latent_channels), "base_channels": int(base),
                       "channel_multipliers": list(mults), "num_res_blocks": int(blocks),
                       "norm": str(norm), "downsample_factor": DOWNSAMPLE_FACTOR}
        self.latent_channels = int(latent_channels)
        self.encoder = Encoder3D(base, mults, latent_channels, blocks, norm)
        self.decoder = Decoder3D(base, mults, latent_channels, blocks, norm)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decode(z), mu, logvar, z


def build_coarse_cache(image_path: Path, split: str, block: int = 8) -> dict:
    """Per-volume sampling aids, computed once: a block-max intensity map for vessel-biased
    crops, a block brain-occupancy map, and the intensity percentiles the loss uses."""
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_ROOT / f"{image_path.stem}_b{block}.npz"
    if cache_file.exists():
        with np.load(cache_file) as data:
            return {k: data[k] for k in data.files}
    volume = np.asarray(np.load(image_path), dtype=np.float32)
    mask = np.asarray(np.load(mask_path_for(image_path, split)), dtype=np.float32)
    shape = tuple(s // block for s in volume.shape)
    trimmed = volume[: shape[0] * block, : shape[1] * block, : shape[2] * block]
    trimmed_mask = mask[: shape[0] * block, : shape[1] * block, : shape[2] * block]
    blocks = trimmed.reshape(shape[0], block, shape[1], block, shape[2], block)
    mask_blocks = trimmed_mask.reshape(shape[0], block, shape[1], block, shape[2], block)
    record = {
        "vessel": blocks.max(axis=(1, 3, 5)).astype(np.float32),
        "brain": mask_blocks.mean(axis=(1, 3, 5)).astype(np.float32),
        "p99": np.array(np.percentile(volume, 99.0), dtype=np.float32),
        "p999": np.array(np.percentile(volume, 99.9), dtype=np.float32),
        "block": np.array(block, dtype=np.int32),
    }
    np.savez(cache_file, **record)
    return record


class MRACropDataset(Dataset):
    """Random crops at several extents. Indices arrive as (volume_index, size_index) tuples
    from MultiExtentBatchSampler, so every item in a batch shares one crop size."""

    def __init__(self, split: str, size_specs=SIZE_SPECS, seed: int = 0) -> None:
        self.split = split
        self.paths = volume_paths(split)
        if not self.paths:
            raise FileNotFoundError(f"no MRA volumes under {DATA_ROOT / split / 'MRA'}")
        self.sizes = [tuple(spec[0]) for spec in size_specs]
        self.coarse = [build_coarse_cache(path, split) for path in self.paths]
        self.seed = int(seed)
        self._volumes: dict[int, np.ndarray] = {}
        self._masks: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.paths)

    def __getstate__(self) -> dict:
        # memmaps must not be pickled into dataloader workers: numpy would materialise them
        state = self.__dict__.copy()
        state["_volumes"], state["_masks"] = {}, {}
        return state

    def _memmap(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        if index not in self._volumes:  # one memmap per worker, opened lazily
            self._volumes[index] = np.load(self.paths[index], mmap_mode="r")
            self._masks[index] = np.load(mask_path_for(self.paths[index], self.split), mmap_mode="r")
        return self._volumes[index], self._masks[index]

    def _crop_start(self, index: int, size: tuple[int, int, int], rng: np.random.Generator) -> tuple[int, ...]:
        coarse = self.coarse[index]
        block = int(coarse["block"])
        shape = self._memmap(index)[0].shape
        limits = [max(int(shape[axis]) - int(size[axis]), 0) for axis in range(3)]
        draw = rng.random()
        if draw < P_VESSEL:
            weights = np.maximum(coarse["vessel"] - float(coarse["p99"]), 0.0).ravel()
            total = float(weights.sum())
            field = weights if total > 0 else coarse["brain"].ravel()
        elif draw < P_VESSEL + P_BRAIN:
            field = coarse["brain"].ravel()
        else:
            return tuple(int(rng.integers(0, limit + 1)) if limit > 0 else 0 for limit in limits)
        total = float(field.sum())
        if total <= 0:
            return tuple(int(rng.integers(0, limit + 1)) if limit > 0 else 0 for limit in limits)
        flat = int(rng.choice(field.size, p=field / total))
        centre = np.unravel_index(flat, coarse["vessel"].shape)
        start = []
        for axis in range(3):
            jitter = int(rng.integers(-size[axis] // 4, size[axis] // 4 + 1))
            value = int(centre[axis]) * block + block // 2 - size[axis] // 2 + jitter
            start.append(int(np.clip(value, 0, limits[axis])))
        return tuple(start)

    def __getitem__(self, key) -> dict:
        index, size_index = key
        size = self.sizes[int(size_index)]
        # torch.initial_seed() differs per worker and per epoch, so crops do not repeat
        # across workers the way a shared Generator copied into each worker would.
        rng = np.random.default_rng((torch.initial_seed() + 7919 * int(index) + 104729 * int(size_index)) % (1 << 63))
        volume, mask = self._memmap(int(index))
        start = self._crop_start(int(index), size, rng)
        slices = tuple(slice(start[axis], start[axis] + size[axis]) for axis in range(3))
        patch = np.asarray(volume[slices], dtype=np.float32)
        patch_mask = np.asarray(mask[slices], dtype=np.float32)
        if patch.shape != tuple(size):  # volumes shorter than the crop on some axis
            pad = [(0, size[axis] - patch.shape[axis]) for axis in range(3)]
            patch = np.pad(patch, pad)
            patch_mask = np.pad(patch_mask, pad)
        if rng.random() < 0.5:  # left-right flip
            patch, patch_mask = patch[::-1].copy(), patch_mask[::-1].copy()
        coarse = self.coarse[int(index)]
        return {
            "image": torch.from_numpy(patch)[None],
            "mask": torch.from_numpy(patch_mask)[None],
            "p99": torch.tensor(float(coarse["p99"]), dtype=torch.float32),
            "p999": torch.tensor(float(coarse["p999"]), dtype=torch.float32),
        }


class MultiExtentBatchSampler(torch.utils.data.Sampler):
    """Yields batches of (volume_index, size_index); one crop size per batch, batch size
    chosen so peak VRAM stays near 7 GB whatever the extent."""

    def __init__(self, n_volumes: int, size_specs=SIZE_SPECS, steps: int = STEPS_PER_EPOCH, seed: int = 0) -> None:
        self.n_volumes = int(n_volumes)
        self.specs = list(size_specs)
        self.steps = int(steps)
        self.probs = np.array([spec[2] for spec in self.specs], dtype=np.float64)
        self.probs /= self.probs.sum()
        self.epoch = 0
        self.seed = int(seed)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.steps

    def __iter__(self):
        rng = np.random.default_rng(self.seed + 9973 * self.epoch)
        for _ in range(self.steps):
            size_index = int(rng.choice(len(self.specs), p=self.probs))
            batch_size = int(self.specs[size_index][1])
            volumes = rng.integers(0, self.n_volumes, size=batch_size)
            yield [(int(v), size_index) for v in volumes]


class Perceptual25D(nn.Module):
    """2.5-D VGG16 feature loss. Moon et al. 2025 found ImageNet VGG beats 3-D MedicalNet
    features for this kind of medical reconstruction; MONAI's 3-D LDM does the same thing
    with is_fake_3d=True. Weights come from the local torch hub cache."""

    LAYERS = (3, 8, 15)  # relu1_2, relu2_2, relu3_3

    def __init__(self, planes_per_axis: int = 3, max_side: int = 160) -> None:
        super().__init__()
        from torchvision.models import vgg16
        features = vgg16(weights="IMAGENET1K_V1").features[: max(self.LAYERS) + 1]
        features.eval()
        for parameter in features.parameters():
            parameter.requires_grad_(False)
        self.features = features
        self.planes_per_axis = int(planes_per_axis)
        self.max_side = int(max_side)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _planes(self, volume: torch.Tensor, picks: list[tuple[int, int]]) -> torch.Tensor:
        out = []
        for axis, index in picks:
            plane = volume.select(dim=2 + axis, index=index)  # [B, 1, A, B]
            out.append(plane)
        planes = torch.cat([p for p in out if p.shape[-2:] == out[0].shape[-2:]], dim=0) if len(
            {tuple(p.shape[-2:]) for p in out}) == 1 else None
        if planes is None:  # mixed plane shapes: resize each to a common square
            side = min(self.max_side, max(max(p.shape[-2:]) for p in out))
            planes = torch.cat([F.interpolate(p, size=(side, side), mode="bilinear", align_corners=False) for p in out], dim=0)
        elif max(planes.shape[-2:]) > self.max_side:
            planes = F.interpolate(planes, size=(self.max_side, self.max_side), mode="bilinear", align_corners=False)
        return planes

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        picks: list[tuple[int, int]] = []
        for axis in range(3):
            extent = pred.shape[2 + axis]
            for _ in range(self.planes_per_axis):
                picks.append((axis, random.randrange(extent)))
        pred_planes = self._planes(pred, picks)
        with torch.no_grad():
            target_planes = self._planes(target, picks)
        loss = pred.new_zeros(())
        p = ((pred_planes.repeat(1, 3, 1, 1)) - self.mean) / self.std
        t = ((target_planes.repeat(1, 3, 1, 1)) - self.mean) / self.std
        for index, layer in enumerate(self.features):
            p = layer(p)
            with torch.no_grad():
                t = layer(t)
            if index in self.LAYERS:
                loss = loss + F.l1_loss(p, t)
        return loss / len(self.LAYERS)


def vessel_weight_map(target: torch.Tensor, p99: torch.Tensor, p999: torch.Tensor) -> torch.Tensor:
    """1 inside the parenchyma, rising to 1 + W_VESSEL_L1 across the per-volume vessel band."""
    lo = p99.float().view(-1, 1, 1, 1, 1)
    hi = torch.clamp(p999.float().view(-1, 1, 1, 1, 1), min=lo + 1e-4)
    ramp = ((target - lo) / (hi - lo)).clamp(0.0, 1.0)
    return 1.0 + W_VESSEL_L1 * ramp


def mip_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.stack([F.l1_loss(pred.amax(dim=axis), target.amax(dim=axis)) for axis in (2, 3, 4)]).mean()


def transform_pair(z: torch.Tensor, x: torch.Tensor, scale: float, rotations: int) -> tuple[torch.Tensor, torch.Tensor]:
    """EQ-VAE: the same rescale+rotation applied in latent space and in image space."""
    latent_size = tuple(max(4, int(round(dim * scale))) for dim in z.shape[-3:])
    image_size = tuple(dim * DOWNSAMPLE_FACTOR for dim in latent_size)
    z_t = F.interpolate(z, size=latent_size, mode="trilinear", align_corners=False)
    x_t = F.interpolate(x, size=image_size, mode="trilinear", align_corners=False)
    if rotations and z_t.shape[-3] == z_t.shape[-2]:
        z_t = torch.rot90(z_t, rotations, dims=(2, 3))
        x_t = torch.rot90(x_t, rotations, dims=(2, 3))
    return z_t, x_t


def hann_weight(shape: tuple[int, ...], device, dtype) -> torch.Tensor:
    windows = [torch.hann_window(dim, periodic=False, device=device, dtype=dtype).clamp_min(1e-3) for dim in shape]
    return windows[0][:, None, None] * windows[1][None, :, None] * windows[2][None, None, :]


def tile_starts(extent: int, tile: int, overlap: int) -> list[int]:
    if tile >= extent:
        return [0]
    stride = max(1, tile - overlap)
    starts = list(range(0, extent - tile + 1, stride))
    if starts[-1] != extent - tile:
        starts.append(extent - tile)
    return starts


@torch.inference_mode()
def encode_tiled(model: VAEv4, volume: torch.Tensor, tile: int = 96, overlap: int = 32) -> torch.Tensor:
    factor = DOWNSAMPLE_FACTOR
    spatial = volume.shape[-3:]
    accumulator = weights = None
    for x0 in tile_starts(spatial[0], tile, overlap):
        for y0 in tile_starts(spatial[1], tile, overlap):
            for z0 in tile_starts(spatial[2], min(tile, spatial[2]), overlap):
                zt = min(tile, spatial[2])
                patch = volume[..., x0:x0 + tile, y0:y0 + tile, z0:z0 + zt]
                mu, _ = model.encode(patch)
                if accumulator is None:
                    latent_shape = tuple(dim // factor for dim in spatial)
                    accumulator = torch.zeros((volume.shape[0], mu.shape[1], *latent_shape),
                                              device=volume.device, dtype=torch.float32)
                    weights = torch.zeros_like(accumulator)
                blend = hann_weight(mu.shape[-3:], mu.device, torch.float32)[None, None]
                region = (..., slice(x0 // factor, x0 // factor + mu.shape[-3]),
                          slice(y0 // factor, y0 // factor + mu.shape[-2]),
                          slice(z0 // factor, z0 // factor + mu.shape[-1]))
                accumulator[region] += mu.float() * blend
                weights[region] += blend
    return accumulator / weights.clamp_min(1e-8)


@torch.inference_mode()
def decode_tiled(model: VAEv4, latent: torch.Tensor, tile: int = 24, overlap: int = 8) -> torch.Tensor:
    factor = DOWNSAMPLE_FACTOR
    spatial = latent.shape[-3:]
    accumulator = weights = None
    for x0 in tile_starts(spatial[0], tile, overlap):
        for y0 in tile_starts(spatial[1], tile, overlap):
            for z0 in tile_starts(spatial[2], min(tile, spatial[2]), overlap):
                zt = min(tile, spatial[2])
                recon = model.decode(latent[..., x0:x0 + tile, y0:y0 + tile, z0:z0 + zt])
                if accumulator is None:
                    image_shape = tuple(dim * factor for dim in spatial)
                    accumulator = torch.zeros((latent.shape[0], 1, *image_shape),
                                              device=latent.device, dtype=torch.float32)
                    weights = torch.zeros_like(accumulator)
                blend = hann_weight(recon.shape[-3:], recon.device, torch.float32)[None, None]
                region = (..., slice(x0 * factor, x0 * factor + recon.shape[-3]),
                          slice(y0 * factor, y0 * factor + recon.shape[-2]),
                          slice(z0 * factor, z0 * factor + recon.shape[-1]))
                accumulator[region] += recon.float() * blend
                weights[region] += blend
    return accumulator / weights.clamp_min(1e-8)


@torch.inference_mode()
def reconstruct(model: VAEv4, volume: torch.Tensor, regime: str) -> tuple[np.ndarray, np.ndarray]:
    # fp16 rather than the bf16 used for training: 10 mantissa bits instead of 7 keeps the
    # metric within ~1e-4 of a full fp32 pass, which matters against the acceptance gate.
    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=volume.is_cuda):
        if regime == "tiled":
            latent = encode_tiled(model, volume)
            recon = decode_tiled(model, latent)
        elif regime == "direct":
            latent, _ = model.encode(volume)
            recon = model.decode(latent)
        else:
            raise ValueError(regime)
    return recon[0, 0].float().cpu().numpy(), latent[0].float().cpu().numpy()


def evaluate(model: VAEv4, cases: list[Path], split: str, device: str, regimes=("tiled", "direct"),
             skeleton_cases: int = 0) -> dict:
    model.eval()
    rows: dict[str, list[dict]] = {regime: [] for regime in regimes}
    latents: dict[str, list[np.ndarray]] = {regime: [] for regime in regimes}
    for index, path in enumerate(cases):
        target = np.asarray(np.load(path), dtype=np.float32)
        brain = np.asarray(np.load(mask_path_for(path, split))) > 0
        volume = torch.from_numpy(target)[None, None].to(device)
        for regime in regimes:
            try:
                recon, latent = reconstruct(model, volume, regime)
            except RuntimeError as error:
                if "out of memory" not in str(error).lower():
                    raise
                torch.cuda.empty_cache()
                continue
            ssim, masked, psnr = masked_ssim_and_psnr(target, recon, brain)
            dice, vessel_l1 = vessel_scores(target, recon)
            row = {"ssim": ssim, "masked_ssim": masked, "psnr": psnr, "vessel_dice": dice,
                   "vessel_l1": vessel_l1, "mip_ssim": mip_ssim(target, recon)}
            if index < skeleton_cases:
                row["cldice"] = hard_cldice(target, recon)
            rows[regime].append(row)
            latents[regime].append(latent)
        del volume
        torch.cuda.empty_cache()
    summary: dict[str, float] = {}
    for regime in regimes:
        if not rows[regime]:
            continue
        keys = {key for row in rows[regime] for key in row}
        for key in keys:
            values = [row[key] for row in rows[regime] if key in row]
            summary[f"{regime}_{key}"] = float(np.mean(values))
        for key, value in latent_diagnostics(latents[regime]).items():
            summary[f"{regime}_{key}"] = value
    if "tiled_masked_ssim" in summary and "direct_masked_ssim" in summary:
        summary["extent_gap_masked_ssim"] = abs(summary["tiled_masked_ssim"] - summary["direct_masked_ssim"])
        if len(latents["tiled"]) == len(latents["direct"]):
            both = [np.corrcoef(a.ravel(), b.ravel())[0, 1] for a, b in zip(latents["tiled"], latents["direct"])]
            summary["extent_latent_correlation"] = float(np.mean(both))
    model.train()
    return summary


def selection_score(summary: dict) -> float:
    """Vessel-aware, measured in the deployment regime. The review's central complaint about
    v3 model selection was that it was 80 % global-window SSIM and blind to vessels."""
    return (0.45 * summary.get("tiled_vessel_dice", 0.0)
            + 0.30 * summary.get("tiled_masked_ssim", 0.0)
            + 0.25 * summary.get("tiled_mip_ssim", 0.0))


def save_checkpoint(path: Path, model: VAEv4, optimizer, epoch: int, best: float, kl_weight: float,
                    extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
                "epoch": int(epoch), "best_metric": float(best), "kl_weight": float(kl_weight),
                "model_config": model.config, "extra": extra or {}}, path)


def load_vaev4(path: Path, device: str = "cuda") -> VAEv4:
    """Rebuild a trained VAEv4 from a checkpoint (used by the LDM stage)."""
    blob = torch.load(path, map_location=device, weights_only=False)
    config = blob["model_config"]
    model = VAEv4(latent_channels=config["latent_channels"], base=config["base_channels"],
                  mults=tuple(config["channel_multipliers"]), blocks=config["num_res_blocks"],
                  norm=config["norm"]).to(device)
    model.load_state_dict(blob["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def export_latent_channel_stats(model: VAEv4, device: str, run_dir: Path, cases: int) -> dict:
    """Per-channel mean/std over training volumes, encoded exactly as the LDM will encode
    them. The LDM stage standardises with these instead of v3's single global scalar."""
    paths = volume_paths("train")[:cases]
    sums = None
    count = 0
    for path in paths:
        volume = torch.from_numpy(np.asarray(np.load(path), dtype=np.float32))[None, None].to(device)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=volume.is_cuda):
            latent = encode_tiled(model, volume)[0].float().cpu().numpy()
        flat = latent.reshape(latent.shape[0], -1)
        if sums is None:
            sums = np.zeros(flat.shape[0], dtype=np.float64)
            squares = np.zeros(flat.shape[0], dtype=np.float64)
        sums += flat.sum(axis=1)
        squares += (flat.astype(np.float64) ** 2).sum(axis=1)
        count += flat.shape[1]
        del volume
        torch.cuda.empty_cache()
    mean = sums / max(count, 1)
    std = np.sqrt(np.maximum(squares / max(count, 1) - mean ** 2, 1e-12))
    stats = {"encode_mode": "tiled_hann_96_overlap32", "cases": len(paths),
             "per_channel_mean": mean.tolist(), "per_channel_std": std.tolist(),
             "global_mean": float(mean.mean()), "global_std": float(std.mean())}
    (run_dir / "latent_channel_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def acceptance_report(summary: dict) -> tuple[bool, list[dict]]:
    rows = []
    passed = True
    for key, (threshold, direction) in ACCEPT.items():
        value = summary.get(key, float("nan"))
        if math.isnan(value):
            ok = False
        else:
            ok = value >= threshold if direction == "ge" else value <= threshold
        passed = passed and ok
        rows.append({"metric": key, "value": value, "threshold": threshold,
                     "direction": direction, "v3": V3_REFERENCE.get(key), "pass": bool(ok)})
    return passed, rows


PERTURB_SIGMAS = (0.1, 0.25, 0.5)   # latent noise, in units of per-channel std. Beyond


@torch.inference_mode()
def evaluate_perturbed(model: VAEv4, cases: list[Path], split: str, device: str,
                       sigmas=PERTURB_SIGMAS) -> dict:
    """Decode mu + noise at several sigmas (units of per-channel std) and score the vessel
    tree. The diffusion sampler never hands the decoder an exact latent, so this is closer
    to the decoder's real operating point than clean reconstruction is."""
    model.eval()
    out: dict[str, float] = {}
    per_sigma: dict[float, list[tuple[float, float]]] = {s: [] for s in sigmas}
    generator = torch.Generator(device=device).manual_seed(1234)
    for path in cases:
        target = np.asarray(np.load(path), dtype=np.float32)
        volume = torch.from_numpy(target)[None, None].to(device)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            latent = encode_tiled(model, volume)
        channel_std = latent.float().transpose(0, 1).reshape(latent.shape[1], -1).std(dim=1)
        for sigma in sigmas:
            noise = torch.randn(latent.shape, device=device, generator=generator, dtype=torch.float32)
            perturbed = latent.float() + noise * sigma * channel_std.view(1, -1, 1, 1, 1)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                recon = decode_tiled(model, perturbed.to(latent.dtype))
            recon = np.clip(recon[0, 0].float().cpu().numpy(), 0.0, 1.0)
            dice, _ = vessel_scores(target, recon)
            per_sigma[sigma].append((dice, hard_cldice(target, recon)))
        del volume, latent
        torch.cuda.empty_cache()
    for sigma, rows in per_sigma.items():
        out[f"perturbed_{sigma}_vessel_dice"] = float(np.mean([r[0] for r in rows]))
        out[f"perturbed_{sigma}_cldice"] = float(np.mean([r[1] for r in rows]))
    return out
