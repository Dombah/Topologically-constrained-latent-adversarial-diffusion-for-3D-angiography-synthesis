"""Split the cropped IXI cases into train / val / test and convert them to .npy.

    uv run split_dataset.py
    uv run split_dataset.py --cropped Dataset/Cropped --splits Dataset/splits --out Dataset/split_numpy

Patient-level split, stratified by site (Guys / HH / IOP), 80 / 10 / 10 with seed 42, which
gives 455 / 57 / 56 on the 568 complete cases. Dataset/splits/manifests/*.csv records which
case went where: keep it, it is what defines the test set.

Output: <out>/{train,val,test}/{T1,T2,PD,MRA}/<case>-<M>.npy (float16) and
<out>/{split}/masks/{M}/ (uint8).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from modules.paths import ROOT
from modules.split import convert_ixi_split_to_numpy, split_ixi_dataset

MODALITIES = ("T1", "T2", "PD", "MRA")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cropped", type=Path, default=ROOT / "Dataset" / "Cropped")
    parser.add_argument("--splits", type=Path, default=ROOT / "Dataset" / "splits")
    parser.add_argument("--out", type=Path, default=ROOT / "Dataset" / "split_numpy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest-only", action="store_true",
                        help="write the split manifests and stop (no .npy conversion)")
    args = parser.parse_args()

    split_ixi_dataset(
        image_folders={m: str(args.cropped / m) for m in MODALITIES},
        mask_folders={m: str(args.cropped / "masks" / m) for m in MODALITIES},
        output_base_dir=str(args.splits), file_action="manifest", seed=args.seed)
    if args.manifest_only:
        return
    convert_ixi_split_to_numpy(split_base_dir=str(args.splits), output_dir=str(args.out),
                               image_dtype=np.float16, mask_dtype=np.uint8, output_format="npy")


if __name__ == "__main__":
    main()
