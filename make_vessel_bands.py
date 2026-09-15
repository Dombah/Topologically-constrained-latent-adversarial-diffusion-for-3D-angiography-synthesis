"""Per-case vessel band for the clDice loss: whole-volume (p98, p99.5) of the target MRA.

    uv run make_vessel_bands.py                   # -> Dataset/vessel_bands.json (train + val)

WHY THESE PERCENTILES, ON THIS SUPPORT
--------------------------------------
vessel_scores() thresholds at np.percentile(target, 99.0) over the WHOLE volume, background
included. The finetune loss was using a FIXED band of (0.3935, 0.99) -- the training set's
BRAIN-MASKED (p99, p99.9). Different support, so the two disagreed badly: 64 % of the voxels
the metric scores as vessel got exactly zero weight in the loss, and the soft mask carried
only 15 % of the hard mask's mass.

Measured over 25 training volumes, soft-mask mass / hard-mask count:
    fixed (0.3935, 0.99)          0.15
    per-case (p99,   p99.9)       0.24
    per-case (p98,   p99.5)       1.03     <- this
    per-case centred tau +- d/4   2.28
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from modules.paths import DATA_ROOT, ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA_ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "Dataset" / "vessel_bands.json")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--limit", type=int, default=0, help="cases per split, 0 = all")
    args = parser.parse_args()

    out = {}
    for split in args.splits:
        files = sorted((args.data / split / "MRA").glob("*-MRA.npy"))
        if args.limit:
            files = files[:args.limit]
        for i, f in enumerate(files, 1):
            v = np.asarray(np.load(f, mmap_mode="r"), np.float32).ravel()
            out[f.name.replace("-MRA.npy", "")] = [float(np.percentile(v, 98.0)),
                                                   float(np.percentile(v, 99.5))]
            if i % 50 == 0 or i == len(files):
                print(f"  {split} {i}/{len(files)}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    lo = np.array([v[0] for v in out.values()])
    hi = np.array([v[1] for v in out.values()])
    print(f"wrote {args.out}  ({len(out)} cases)")
    print(f"  p98   mean {lo.mean():.4f} sd {lo.std():.4f}   p99.5 mean {hi.mean():.4f} sd {hi.std():.4f}")


if __name__ == "__main__":
    main()
