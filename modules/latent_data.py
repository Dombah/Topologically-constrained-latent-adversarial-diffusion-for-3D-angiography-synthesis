"""Paired latent dataset (T1/T2/PD condition -> MRA target), standardisation and cropping."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from modules.paths import ROOT


class PairedLatentDataset(Dataset):
    """Random crops of (source latents, target latent), standardised per channel."""

    def __init__(self, config: dict, split: str, seed: int = 0) -> None:
        data = config["data"]
        self.target_root = ROOT / data["target_root"]
        self.cond_root = ROOT / data["conditioning_root"]
        self.target_modality = data["target_modality"]
        self.sources = list(data["source_modalities"])
        self.crop = tuple(data["latent_crop"])
        self.frangi_root = ROOT / data["frangi_root"] if data.get("frangi_root") else None
        self.split = split
        self.seed = int(seed)

        target_files = {p.stem.rsplit("-", 1)[0]: p
                        for p in (self.target_root / self.target_modality / split).glob("*.npy")}
        self.cases: list[str] = []
        self.paths: dict[str, dict[str, Path]] = {}
        for case, target_path in sorted(target_files.items()):
            entry = {self.target_modality: target_path}
            for modality in self.sources:
                candidate = self.cond_root / modality / split / f"{case}-{modality}.npy"
                if not candidate.exists():
                    entry = None
                    break
                entry[modality] = candidate
            if entry:
                if self.frangi_root is not None:
                    frangi = self.frangi_root / split / f"{case}.npy"
                    entry["frangi"] = frangi if frangi.exists() else None
                self.cases.append(case)
                self.paths[case] = entry
        if not self.cases:
            raise SystemExit(
                f"no complete cases for split '{split}'. Looked for "
                f"{self.target_root / self.target_modality / split} and "
                f"{self.cond_root / '<T1|T2|PD>' / split}")

        self.stats = load_latent_stats(config)
        self.cache: dict[str, np.ndarray] = {}
        if data.get("preload_to_ram", True):
            dtype = np.float16 if data.get("cache_dtype", "float16") == "float16" else np.float32
            started = time.time()
            for case in self.cases:
                for modality, path in self.paths[case].items():
                    if path is not None:
                        self.cache[str(path)] = np.asarray(np.load(path), dtype=dtype)
            gigabytes = sum(a.nbytes for a in self.cache.values()) / 1e9
            print(f"  {split}: {len(self.cases)} cases, preloaded {gigabytes:.2f} GB "
                  f"in {time.time() - started:.1f}s", flush=True)
        else:
            print(f"  {split}: {len(self.cases)} cases (lazy mmap)", flush=True)

    def __getstate__(self):
        """Windows spawns DataLoader workers, which PICKLES this object. The RAM cache is
        ~16.5 GB for the train split, so three workers would try to materialise ~50 GB and
        the run dies. Workers get the object without the cache and fall back to mmap
        (112 ms/item against an 85 ms cached read) -- still far inside a 245 ms GPU step
        once several workers overlap."""
        state = self.__dict__.copy()
        state["cache"] = {}
        return state

    def __len__(self) -> int:
        return len(self.cases)

    def _load(self, path: Path) -> np.ndarray:
        key = str(path)
        if key in self.cache:
            return self.cache[key]
        return np.load(path, mmap_mode="r")

    def full_case(self, case: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Whole standardised latents for validation (no crop)."""
        target = standardize_np(np.asarray(self._load(self.paths[case][self.target_modality]),
                                           np.float32), self.stats[self.target_modality])
        sources = [standardize_np(np.asarray(self._load(self.paths[case][m]), np.float32),
                                  self.stats[m]) for m in self.sources]
        return torch.from_numpy(target), torch.from_numpy(np.concatenate(sources, axis=0))

    def __getitem__(self, index: int):
        case = self.cases[index % len(self.cases)]
        # reseed per item so workers do not share a crop sequence
        rng = np.random.default_rng(torch.initial_seed() % (2 ** 31) + 7919 * index)
        target = np.asarray(self._load(self.paths[case][self.target_modality]), np.float32)
        starts = [int(rng.integers(0, max(full - want, 0) + 1))
                  for full, want in zip(target.shape[1:], self.crop)]
        sl = (slice(None),) + tuple(slice(s, s + c) for s, c in zip(starts, self.crop))
        target = standardize_np(target[sl], self.stats[self.target_modality])
        sources = []
        for modality in self.sources:
            arr = np.asarray(self._load(self.paths[case][modality]), np.float32)[sl]
            sources.append(standardize_np(arr, self.stats[modality]))
        # Frangi vesselness, already stored at latent resolution, cropped identically.
        # It weights the x0-space losses toward vessels; zeros where a map is missing.
        frangi_path = self.paths[case].get("frangi")
        if frangi_path is None:
            frangi = np.zeros((1,) + self.crop, np.float32)
        else:
            frangi = np.asarray(self._load(frangi_path), np.float32)[sl[0:1] + sl[1:]]
            frangi = frangi[:1]
        return {"target": torch.from_numpy(target),
                "source": torch.from_numpy(np.concatenate(sources, axis=0)),
                "frangi": torch.from_numpy(np.ascontiguousarray(frangi))}


def load_latent_stats(config: dict) -> dict:
    """Per-channel stats for every modality. The target's come from export_latents.py; the
    conditioning ones are measured here if the multimodal export did not write them."""
    data = config["data"]
    stats: dict[str, dict] = {}
    target_stats = ROOT / data["target_root"] / "latent_stats.json"
    if not target_stats.exists():
        raise SystemExit(f"missing {target_stats} -- run export_latents.py first")
    stats[data["target_modality"]] = json.loads(target_stats.read_text())[data["target_modality"]]

    cache = ROOT / data["conditioning_root"] / "latent_stats_per_channel.json"
    if cache.exists():
        stats.update(json.loads(cache.read_text()))
        return stats
    measured = {}
    for modality in data["source_modalities"]:
        files = sorted((ROOT / data["conditioning_root"] / modality / "train").glob("*.npy"))
        if not files:
            raise SystemExit(f"no train latents for {modality}")
        total = total_sq = None
        count = 0
        for path in files:
            flat = np.asarray(np.load(path), np.float64).reshape(np.load(path, mmap_mode="r").shape[0], -1)
            total = flat.sum(1) if total is None else total + flat.sum(1)
            total_sq = (flat ** 2).sum(1) if total_sq is None else total_sq + (flat ** 2).sum(1)
            count += flat.shape[1]
        mean = total / count
        std = np.sqrt(np.maximum(total_sq / count - mean ** 2, 1e-12))
        measured[modality] = {"per_channel_mean": mean.tolist(), "per_channel_std": std.tolist()}
        print(f"  measured {modality} channel stats over {len(files)} train volumes")
    cache.write_text(json.dumps(measured, indent=2))
    stats.update(measured)
    return stats


def standardize_np(latent: np.ndarray, stats: dict) -> np.ndarray:
    mean = np.asarray(stats["per_channel_mean"], np.float32).reshape(-1, 1, 1, 1)
    std = np.asarray(stats["per_channel_std"], np.float32).reshape(-1, 1, 1, 1)
    return (latent.astype(np.float32) - mean) / std


def destandardize_t(latent: torch.Tensor, stats: dict) -> torch.Tensor:
    shape = (1, -1, 1, 1, 1) if latent.dim() == 5 else (-1, 1, 1, 1)
    mean = torch.as_tensor(stats["per_channel_mean"], dtype=torch.float32,
                           device=latent.device).view(shape)
    std = torch.as_tensor(stats["per_channel_std"], dtype=torch.float32,
                          device=latent.device).view(shape)
    return latent.float() * std + mean


def crop_latent(*tensors: torch.Tensor, size: tuple[int, int, int],
                generator=None) -> list[torch.Tensor]:
    """The same random crop applied to every tensor, so pred and target stay aligned."""
    shape = tensors[0].shape[-3:]
    starts = [int(torch.randint(0, max(full - want, 0) + 1, (1,), generator=generator).item())
              for full, want in zip(shape, size)]
    sl = (slice(None), slice(None)) + tuple(slice(s, s + c) for s, c in zip(starts, size))
    return [t[sl] for t in tensors]
