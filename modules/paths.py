"""Repository paths and case helpers shared by every script."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "Dataset" / "split_numpy"

# New training runs land here by default, so rerunning a step never touches a thesis result.
REBUILD_ROOT = ROOT / "checkpoints" / "rebuild"

# The artefacts the thesis numbers come from. Training and export refuse to write into them.
PROTECTED = [
    ROOT / "checkpoints" / "vaev2_multimodal" / "kl5e5_from_epoch10",
    ROOT / "checkpoints" / "vaev4_mra" / "v4_c16_fixed",
    ROOT / "checkpoints" / "vaev5_mra",
    ROOT / "checkpoints" / "ldm_bbdm",
    ROOT / "checkpoints" / "ldm_bbdm_vessel",
    ROOT / "latents",
    ROOT / "runs" / "table53_test",
]


def refuse_protected(path, what: str = "output") -> Path:
    """Exit if `path` is, or lies inside, a thesis artefact; return it resolved otherwise."""
    path = Path(path).resolve()
    for protected in PROTECTED:
        protected = protected.resolve()
        if path == protected or protected in path.parents:
            raise SystemExit(f"refusing to write {what} into {path}: it holds a thesis result "
                             f"({protected.relative_to(ROOT)}). Pass another output folder.")
    return path


def volume_paths(split: str) -> list[Path]:
    return sorted((DATA_ROOT / split / "MRA").glob("*.npy"))


def mask_path_for(image_path: Path, split: str) -> Path:
    return DATA_ROOT / split / "masks" / "MRA" / image_path.name.replace("-MRA.npy", "-MRA_mask.npy")


def scanner_of(path: Path) -> str:
    return path.stem.split("-")[1]


def pick_validation_cases(split: str, per_scanner: int) -> list[Path]:
    paths = volume_paths(split)
    chosen: list[Path] = []
    for scanner in ("Guys", "HH", "IOP"):
        chosen += [path for path in paths if scanner_of(path) == scanner][:per_scanner]
    return chosen


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
