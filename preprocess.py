"""Preprocess the raw IXI NIfTI files into the cropped, normalised volumes the models train on.

    uv run preprocess.py                                   # every step, in order
    uv run preprocess.py --from normalize                  # resume at one step
    uv run preprocess.py --only skullstrip                 # a single step

Input: Dataset/Quadruplets/{T1,T2,PD,MRA}/<IXI id>-<modality>.nii.gz (see configs/preprocess.json).

Steps (each skips files it has already written, so an interrupted run can be restarted):

    quadruplets  keep only cases with all four modalities   (DELETES the others from raw)
    reorient     reorient every volume to LPS               (in place)
    skullstrip   SynthStrip on all four modalities          -> SkullStripped/{M}, SkullStripped/masks/{M}
    resample     MRA to 0.4x0.4x0.8 mm, then T1/T2/PD and
                 all masks onto each patient's MRA grid     -> Resampled/{M}, Resampled/{M}/masks
    rename       drop the "_stripped" suffix SynthStrip added
    normalize    masked percentile normalisation to [0, 1]  -> Normalized/{M}, Normalized/metadata
    crop         crop / pad to 512x576x96                   -> Cropped/{M}, Cropped/masks/{M}

Then run split_dataset.py. The functions are the ones processing.ipynb called; this script only
fixes their order and arguments.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from modules.paths import ROOT

STEPS = ["quadruplets", "reorient", "skullstrip", "resample", "rename", "normalize", "crop"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "preprocess.json")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--from", dest="start", choices=STEPS, default=STEPS[0])
    group.add_argument("--only", choices=STEPS, default=None)
    args = parser.parse_args()

    import SimpleITK as sitk
    from modules import preprocessing as P

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    mods = tuple(cfg["modalities"])
    ext = tuple(cfg["image_extensions"])
    folder = {k: ROOT / v for k, v in cfg["folders"].items()}
    steps = [args.only] if args.only else STEPS[STEPS.index(args.start):]

    for step in steps:
        print(f"\n=== {step} ===", flush=True)
        if step == "quadruplets":
            P.isolate_quadruplets(str(folder["raw"]), inplace=True, modalities=mods,
                                  output_folder="Quadruplets", image_extensions=ext)

        elif step == "reorient":
            P.reorient_dataset_to_lps(str(folder["raw"]), inplace=True, modalities=mods,
                                      output_folder="LPS", target_orientation="LPS",
                                      image_extensions=ext)

        elif step == "skullstrip":
            s = cfg["synthstrip"]
            P.run_synthstrip_on_modalities(
                image_folder=str(folder["raw"]), output_folder=str(folder["skullstripped"]),
                modalities=mods, max_patients=int(s["max_patients"]), image_extensions=ext,
                model_path=str(ROOT / s["model_path"]), use_gpu=bool(s["use_gpu"]),
                overwrite=False)

        elif step == "resample":
            stripped, resampled = folder["skullstripped"], folder["resampled"]
            P.resample_folder_to_spacing(
                str(stripped / "MRA"), target_spacing=tuple(cfg["resample"]["target_spacing"]),
                image_extensions=ext, output_folder=str(resampled / "MRA"), overwrite=False,
                output_pixel_type=sitk.sitkFloat32)
            for m in mods:
                if m != "MRA":
                    P.resample_folder_to_reference(
                        input_folder=str(stripped / m), reference_folder_or_path=str(resampled / "MRA"),
                        output_folder=str(resampled / m), input_modality=m, reference_modality="MRA",
                        transform_folder=None, interpolator="linear")
                P.resample_mask_folder_to_reference(
                    mask_folder=str(stripped / "masks" / m),
                    reference_folder_or_path=str(resampled / "MRA"),
                    output_mask_folder=str(resampled / m / "masks"), mask_modality=m,
                    reference_modality="MRA", transform_folder=None)

        elif step == "rename":
            for m in mods:
                for path in sorted((folder["resampled"] / m).glob("*_stripped*")):
                    if path.is_file():
                        target = path.with_name(path.name.replace("_stripped", ""))
                        print(f"  {path.name} -> {target.name}")
                        path.rename(target)

        elif step == "normalize":
            n, out = cfg["normalize"], folder["normalized"]
            P.normalize_training_modalities(
                image_folders={m: str(folder["resampled"] / m) for m in mods},
                mask_folders={m: str(folder["resampled"] / m / "masks") for m in mods},
                output_root=str(out), modalities=mods,
                lower_percentile=float(n["lower_percentile"]),
                upper_percentile=float(n["upper_percentile"]),
                mra_output_range=tuple(n["mra_output_range"]),
                anatomical_metadata_path=str(out / "metadata" / "t1_t2_pd_normalization.csv"),
                mra_metadata_path=str(out / "metadata" / "mra_normalization.csv"),
                overwrite=False)

        elif step == "crop":
            P.crop_or_pad_folders_to_size(
                input_folders={m: str(folder["normalized"] / m) for m in mods},
                mask_folders={m: str(folder["normalized"] / m / "masks") for m in mods},
                output_folder=str(folder["cropped"]),
                target_size=tuple(cfg["crop"]["target_size"]), overwrite=False)

    print("\ndone. Next: uv run split_dataset.py", flush=True)


if __name__ == "__main__":
    main()
