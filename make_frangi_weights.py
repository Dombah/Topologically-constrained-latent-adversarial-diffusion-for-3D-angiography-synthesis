"""Frangi vesselness weight maps at latent resolution, for the bridge's vessel-weighted losses.

    uv run make_frangi_weights.py
    uv run make_frangi_weights.py --max-cases 2 --output-root checkpoints/_smoke/frangi   # quick check

Reads the MRA volumes and brain masks from Dataset/split_numpy and writes one
(1, 128, 144, 24) weight map per case to Dataset/frangi_old/{train,val}. The losses use them
as 1 + alpha * w. Test needs no weights. Parameters are in configs/frangi.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from modules.frangi import generate_frangi_vessel_weight_cache
from modules.paths import ROOT

PARAMETERS = ("sigmas", "black_ridges", "robust_percentile", "frangi_downsample", "floor_percentile",
              "frangi_gamma", "latent_renormalize", "latent_gamma", "post_min_value", "post_renormalize")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "frangi.json")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    output_root = args.output_root or ROOT / cfg["output_root"]
    summary = generate_frangi_vessel_weight_cache(
        dataset_root=ROOT / cfg["dataset_root"], latent_root=ROOT / cfg["latent_root"],
        output_root=output_root, splits=list(cfg["splits"]), max_cases=args.max_cases,
        overwrite=args.overwrite, show_progress=True,
        **{k: (tuple(cfg[k]) if k == "sigmas" else cfg[k]) for k in PARAMETERS})

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "frangi_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (output_root / "frangi_summary.json").write_text(json.dumps(summary, indent=2, default=str),
                                                     encoding="utf-8")
    print(f"wrote {output_root}")


if __name__ == "__main__":
    main()
