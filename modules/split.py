"""Site-stratified train/val/test split of the preprocessed IXI cases and conversion to .npy."""
from __future__ import annotations

import csv
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Mapping

import numpy as np
import SimpleITK as sitk


DEFAULT_SEED = 42


DEFAULT_MODALITIES = ("T1", "T2", "PD", "MRA")


DEFAULT_IMAGE_EXTENSIONS = (".nii.gz", ".nii", ".npy", ".npz")


def _has_known_extension(filename: str, image_extensions=DEFAULT_IMAGE_EXTENSIONS) -> bool:
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension.lower()) for extension in image_extensions)


def _strip_known_extension(filename: str, image_extensions=DEFAULT_IMAGE_EXTENSIONS) -> str:
    filename_lower = filename.lower()
    for extension in sorted(image_extensions, key=len, reverse=True):
        if filename_lower.endswith(extension.lower()):
            return filename[: -len(extension)]
    return Path(filename).stem


def _patient_stem_from_path(path, modality, image_extensions=DEFAULT_IMAGE_EXTENSIONS):
    base = _strip_known_extension(Path(path).name, image_extensions)
    if base.lower().endswith("_mask"):
        base = base[:-5]

    token = f"-{modality}".lower()
    index = base.lower().rfind(token)
    if index < 0:
        return None
    return base[:index]


def _scanner_from_patient(patient_id: str) -> str:
    parts = str(patient_id).split("-")
    if len(parts) >= 2:
        return parts[1]
    return "UNKNOWN"


def _collect_modality_files(
    folder,
    modality,
    recursive=False,
    image_extensions=DEFAULT_IMAGE_EXTENSIONS,
):
    folder = Path(folder)
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    files = defaultdict(list)
    ignored = []

    for path in sorted(iterator):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            ignored.append(path)
            continue
        if not _has_known_extension(path.name, image_extensions):
            ignored.append(path)
            continue
        patient = _patient_stem_from_path(
            path,
            modality=modality,
            image_extensions=image_extensions,
        )
        if patient is None:
            ignored.append(path)
            continue
        files[patient].append(path)

    return files, ignored


def _folder_mapping(folders, modalities=DEFAULT_MODALITIES):
    if folders is None:
        return None
    if isinstance(folders, Mapping):
        return {str(modality).upper(): Path(folder) for modality, folder in folders.items()}

    folders = [Path(folder) for folder in folders]
    if len(folders) != len(modalities):
        raise ValueError("Folder list length must match modalities length")
    return {
        str(modality).upper(): folder
        for modality, folder in zip(modalities, folders)
    }


def _relative_or_absolute(path):
    return str(Path(path).resolve())


def _safe_link_or_copy(source, destination, file_action="manifest", overwrite=False):
    source = Path(source)
    destination = Path(destination)

    if file_action == "manifest":
        return False

    if destination.exists() or destination.is_symlink():
        if not overwrite:
            return False
        if destination.is_dir():
            raise IsADirectoryError(f"Refusing to replace directory: {destination}")
        destination.unlink()

    destination.parent.mkdir(parents=True, exist_ok=True)

    if file_action == "copy":
        shutil.copy2(source, destination)
    elif file_action == "symlink":
        os.symlink(source.resolve(), destination)
    elif file_action == "hardlink":
        os.link(source, destination)
    else:
        raise ValueError("file_action must be one of: manifest, copy, symlink, hardlink")
    return True


def _split_one_group(items, train_ratio, val_ratio, test_ratio, rng):
    items = list(items)
    rng.shuffle(items)
    n_items = len(items)
    if n_items == 0:
        return [], [], []

    train_count = int(round(n_items * train_ratio))
    val_count = int(round(n_items * val_ratio))
    test_count = n_items - train_count - val_count

    if test_count < 0:
        val_count = max(0, val_count + test_count)
        test_count = 0

    if n_items >= 3:
        counts = [train_count, val_count, test_count]
        for index in range(3):
            if counts[index] == 0:
                largest = int(np.argmax(counts))
                if counts[largest] > 1:
                    counts[largest] -= 1
                    counts[index] += 1
        train_count, val_count, test_count = counts

    train = items[:train_count]
    val = items[train_count:train_count + val_count]
    test = items[train_count + val_count:train_count + val_count + test_count]
    return train, val, test


def _split_cases_by_scanner(cases, train_ratio, val_ratio, test_ratio, seed):
    by_scanner = defaultdict(list)
    for patient_id, case in sorted(cases.items()):
        by_scanner[case["scanner"]].append(patient_id)

    rng = np.random.default_rng(int(seed))
    split_patients = {"train": [], "val": [], "test": []}
    scanner_summary = {}

    for scanner, patient_ids in sorted(by_scanner.items()):
        train, val, test = _split_one_group(
            patient_ids,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            rng=rng,
        )
        split_patients["train"].extend(train)
        split_patients["val"].extend(val)
        split_patients["test"].extend(test)
        scanner_summary[scanner] = {
            "total": len(patient_ids),
            "train": len(train),
            "val": len(val),
            "test": len(test),
        }

    for split_name in split_patients:
        split_patients[split_name].sort()
    return split_patients, scanner_summary


def collect_complete_ixi_cases(
    image_folders,
    mask_folders=None,
    modalities=DEFAULT_MODALITIES,
    recursive=False,
    image_extensions=DEFAULT_IMAGE_EXTENSIONS,
) -> dict:
    """
    Collect complete same-patient cases from modality image folders and optional mask folders.

    Returns a dict keyed by patient id:
        {patient: {"scanner": ..., "images": {modality: path}, "masks": {modality: path}}}
    """
    modalities = tuple(str(modality).upper() for modality in modalities)
    image_folders = _folder_mapping(image_folders, modalities=modalities)
    mask_folders = _folder_mapping(mask_folders, modalities=modalities)

    if image_folders is None:
        raise ValueError("image_folders is required")
    missing_modalities = [modality for modality in modalities if modality not in image_folders]
    if missing_modalities:
        raise ValueError(f"Missing image folders for modalities: {missing_modalities}")

    image_files = {}
    mask_files = {}
    ignored = {"images": {}, "masks": {}}

    for modality in modalities:
        if not image_folders[modality].is_dir():
            raise NotADirectoryError(f"Image folder does not exist for {modality}: {image_folders[modality]}")
        image_files[modality], ignored["images"][modality] = _collect_modality_files(
            image_folders[modality],
            modality=modality,
            recursive=recursive,
            image_extensions=image_extensions,
        )

        if mask_folders is not None:
            if modality not in mask_folders:
                raise ValueError(f"Missing mask folder for modality {modality}")
            if not mask_folders[modality].is_dir():
                raise NotADirectoryError(f"Mask folder does not exist for {modality}: {mask_folders[modality]}")
            mask_files[modality], ignored["masks"][modality] = _collect_modality_files(
                mask_folders[modality],
                modality=modality,
                recursive=recursive,
                image_extensions=image_extensions,
            )

    patient_sets = [set(image_files[modality]) for modality in modalities]
    if mask_folders is not None:
        patient_sets.extend(set(mask_files[modality]) for modality in modalities)
    complete_patients = sorted(set.intersection(*patient_sets)) if patient_sets else []

    cases = {}
    for patient in complete_patients:
        cases[patient] = {
            "patient_id": patient,
            "scanner": _scanner_from_patient(patient),
            "images": {
                modality: image_files[modality][patient][0]
                for modality in modalities
            },
            "masks": {
                modality: mask_files[modality][patient][0]
                for modality in modalities
            } if mask_folders is not None else {},
        }

    all_image_patients = sorted(set.union(*(set(image_files[m]) for m in modalities)))
    incomplete_patients = [
        patient for patient in all_image_patients
        if patient not in complete_patients
    ]

    return {
        "cases": cases,
        "modalities": modalities,
        "ignored": {
            "images": {modality: [str(path) for path in ignored["images"][modality]] for modality in modalities},
            "masks": {
                modality: [str(path) for path in ignored["masks"].get(modality, [])]
                for modality in modalities
            },
        },
        "incomplete_patients": incomplete_patients,
    }


def _manifest_fieldnames(modalities, include_masks=True):
    fields = ["split", "patient_id", "scanner"]
    fields.extend(f"image_{modality}" for modality in modalities)
    if include_masks:
        fields.extend(f"mask_{modality}" for modality in modalities)
    return fields


def _case_manifest_row(split_name, patient_id, case, modalities, include_masks=True):
    row = {
        "split": split_name,
        "patient_id": patient_id,
        "scanner": case["scanner"],
    }
    for modality in modalities:
        row[f"image_{modality}"] = _relative_or_absolute(case["images"][modality])
    if include_masks:
        for modality in modalities:
            row[f"mask_{modality}"] = _relative_or_absolute(case["masks"][modality])
    return row


def _write_manifest_csv(path, rows, modalities, include_masks=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=_manifest_fieldnames(modalities, include_masks=include_masks),
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def split_ixi_dataset(
    image_folders,
    mask_folders=None,
    output_base_dir="Dataset/splits_current",
    modalities=DEFAULT_MODALITIES,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=DEFAULT_SEED,
    file_action="manifest",
    recursive=False,
    image_extensions=(".nii.gz", ".nii"),
    overwrite=False,
    preview_limit=10,
    print_stats=True,
) -> dict:
    """
    Split already-processed IXI images into train/val/test with scanner-balanced proportions.

    This function does not preprocess images. It collects complete same-patient
    examples across the provided modality folders and optional mask folders,
    writes split manifests, and optionally copies/links files into split folders.

    file_action:
        "manifest": write CSV/JSON manifests only.
        "copy": copy images/masks into output_base_dir/train|val|test/.
        "symlink": create symlinks into output_base_dir/train|val|test/.
        "hardlink": create hardlinks into output_base_dir/train|val|test/.
    """
    ratio_sum = float(train_ratio) + float(val_ratio) + float(test_ratio)
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {ratio_sum}")
    if file_action not in {"manifest", "copy", "symlink", "hardlink"}:
        raise ValueError("file_action must be one of: manifest, copy, symlink, hardlink")

    modalities = tuple(str(modality).upper() for modality in modalities)
    output_path = Path(output_base_dir)
    collected = collect_complete_ixi_cases(
        image_folders=image_folders,
        mask_folders=mask_folders,
        modalities=modalities,
        recursive=recursive,
        image_extensions=image_extensions,
    )
    cases = collected["cases"]
    if not cases:
        raise ValueError("No complete cases were found")

    split_patients, scanner_summary = _split_cases_by_scanner(
        cases,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    manifest_rows = []
    split_rows = {}
    include_masks = mask_folders is not None
    linked_or_copied = 0
    skipped_existing = 0

    for split_name in ("train", "val", "test"):
        rows = []
        for patient_id in split_patients[split_name]:
            case = cases[patient_id]
            row = _case_manifest_row(
                split_name,
                patient_id,
                case,
                modalities=modalities,
                include_masks=include_masks,
            )
            rows.append(row)
            manifest_rows.append(row)

            if file_action != "manifest":
                for modality in modalities:
                    source = case["images"][modality]
                    destination = output_path / split_name / modality / source.name
                    changed = _safe_link_or_copy(source, destination, file_action=file_action, overwrite=overwrite)
                    linked_or_copied += int(changed)
                    skipped_existing += int(not changed and destination.exists())

                if include_masks:
                    for modality in modalities:
                        source = case["masks"][modality]
                        destination = output_path / split_name / "masks" / modality / source.name
                        changed = _safe_link_or_copy(source, destination, file_action=file_action, overwrite=overwrite)
                        linked_or_copied += int(changed)
                        skipped_existing += int(not changed and destination.exists())

        split_rows[split_name] = rows

    manifest_dir = output_path / "manifests"
    _write_manifest_csv(manifest_dir / "all.csv", manifest_rows, modalities, include_masks=include_masks)
    for split_name, rows in split_rows.items():
        _write_manifest_csv(manifest_dir / f"{split_name}.csv", rows, modalities, include_masks=include_masks)

    stats = {
        "seed": seed,
        "modalities": modalities,
        "file_action": file_action,
        "total_cases": len(cases),
        "splits": {
            split_name: {
                "cases": len(split_patients[split_name]),
                "patients": split_patients[split_name],
                "scanner_distribution": dict(
                    sorted(
                        {
                            scanner: sum(cases[p]["scanner"] == scanner for p in split_patients[split_name])
                            for scanner in {case["scanner"] for case in cases.values()}
                        }.items()
                    )
                ),
            }
            for split_name in ("train", "val", "test")
        },
        "scanner_summary": scanner_summary,
        "manifests": {
            "all": str(manifest_dir / "all.csv"),
            "train": str(manifest_dir / "train.csv"),
            "val": str(manifest_dir / "val.csv"),
            "test": str(manifest_dir / "test.csv"),
        },
        "linked_or_copied_files": linked_or_copied,
        "skipped_existing_files": skipped_existing,
        "incomplete_patients": collected["incomplete_patients"],
        "ignored_files": collected["ignored"],
    }
    _write_json(manifest_dir / "split_stats.json", stats)

    if print_stats:
        print("IXI split finished")
        print(f"Output: {output_path}")
        print(f"Seed: {seed}")
        print(f"File action: {file_action}")
        print(f"Complete cases: {len(cases)}")
        print(f"Incomplete image patients: {len(collected['incomplete_patients'])}")
        print("\nScanner-balanced split:")
        for scanner, counts in sorted(scanner_summary.items()):
            print(
                f"  {scanner}: total={counts['total']}, "
                f"train={counts['train']}, val={counts['val']}, test={counts['test']}"
            )
        print("\nSplit totals:")
        for split_name in ("train", "val", "test"):
            split_stats = stats["splits"][split_name]
            print(f"  {split_name}: {split_stats['cases']} cases {split_stats['scanner_distribution']}")
        print(f"\nManifest folder: {manifest_dir}")
        if file_action != "manifest":
            print(f"Files written: {linked_or_copied}")
            print(f"Skipped existing files: {skipped_existing}")
        if collected["incomplete_patients"]:
            print(f"Incomplete patient preview: {collected['incomplete_patients'][:preview_limit]}")

    return stats


def _read_manifest(path):
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _load_volume_array(path, image_dtype=np.float16, array_order="xyz"):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        loaded = np.load(path, allow_pickle=False)
        key = "array" if "array" in loaded.files else loaded.files[0]
        array = loaded[key]
    else:
        image = sitk.ReadImage(str(path))
        array = sitk.GetArrayFromImage(image)
        if array_order == "xyz":
            array = array.transpose(2, 1, 0)
        elif array_order == "zyx":
            pass
        else:
            raise ValueError("array_order must be 'xyz' or 'zyx'")
    return np.asarray(array, dtype=image_dtype)


def _save_array(path, array, output_format="npy"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "npy":
        np.save(path, array, allow_pickle=False)
    elif output_format == "npz":
        np.savez_compressed(path, array=array)
    else:
        raise ValueError("output_format must be 'npy' or 'npz'")


def _output_array_path(output_root, split_name, kind, modality, source_path, output_format):
    source_path = Path(source_path)
    base = _strip_known_extension(source_path.name, DEFAULT_IMAGE_EXTENSIONS)
    extension = ".npy" if output_format == "npy" else ".npz"
    if kind == "image":
        return Path(output_root) / split_name / modality / f"{base}{extension}"
    return Path(output_root) / split_name / "masks" / modality / f"{base}{extension}"


def convert_ixi_split_to_numpy(
    split_base_dir,
    output_dir,
    splits=("train", "val", "test"),
    modalities=DEFAULT_MODALITIES,
    image_dtype=np.float16,
    mask_dtype=np.uint8,
    output_format="npz",
    array_order="xyz",
    overwrite=False,
    max_cases=None,
    print_stats=True,
) -> dict:
    """
    Convert split manifest files to training arrays.

    Images are saved as float16 by default. Masks are saved as uint8 by default.
    The default array_order="xyz" matches the VAEv2 volume convention in this
    project. The default output_format="npz" uses compression and is usually
    smaller than `.nii.gz` or raw `.npy`, at the cost of slower loading.
    """
    split_base_dir = Path(split_base_dir)
    output_dir = Path(output_dir)
    modalities = tuple(str(modality).upper() for modality in modalities)
    if output_format not in {"npy", "npz"}:
        raise ValueError("output_format must be 'npy' or 'npz'")

    manifest_dir = split_base_dir / "manifests"
    if not manifest_dir.is_dir():
        raise NotADirectoryError(f"Manifest folder not found: {manifest_dir}")

    image_dtype = np.dtype(image_dtype)
    mask_dtype = np.dtype(mask_dtype)
    summary = {
        "split_base_dir": str(split_base_dir),
        "output_dir": str(output_dir),
        "output_format": output_format,
        "image_dtype": str(image_dtype),
        "mask_dtype": str(mask_dtype),
        "array_order": array_order,
        "splits": {},
        "failed": [],
    }

    converted_manifest_rows = []
    converted_fieldnames = ["split", "patient_id", "scanner"]
    converted_fieldnames.extend(f"image_{modality}" for modality in modalities)
    converted_fieldnames.extend(f"mask_{modality}" for modality in modalities)

    for split_name in splits:
        rows = _read_manifest(manifest_dir / f"{split_name}.csv")
        if max_cases is not None:
            rows = rows[: int(max_cases)]

        converted_images = 0
        converted_masks = 0
        skipped_existing = 0
        split_failures = []

        for row in rows:
            converted_row = {
                "split": split_name,
                "patient_id": row["patient_id"],
                "scanner": row["scanner"],
            }
            for modality in modalities:
                source = Path(row[f"image_{modality}"])
                destination = _output_array_path(output_dir, split_name, "image", modality, source, output_format)
                converted_row[f"image_{modality}"] = str(destination)
                if destination.exists() and not overwrite:
                    skipped_existing += 1
                else:
                    try:
                        array = _load_volume_array(source, image_dtype=image_dtype, array_order=array_order)
                        _save_array(destination, array, output_format=output_format)
                        converted_images += 1
                    except Exception as error:
                        split_failures.append({"file": str(source), "error": repr(error)})

                mask_key = f"mask_{modality}"
                if mask_key in row and row[mask_key]:
                    source = Path(row[mask_key])
                    destination = _output_array_path(output_dir, split_name, "mask", modality, source, output_format)
                    converted_row[mask_key] = str(destination)
                    if destination.exists() and not overwrite:
                        skipped_existing += 1
                    else:
                        try:
                            array = _load_volume_array(source, image_dtype=mask_dtype, array_order=array_order)
                            array = (array > 0).astype(mask_dtype, copy=False)
                            _save_array(destination, array, output_format=output_format)
                            converted_masks += 1
                        except Exception as error:
                            split_failures.append({"file": str(source), "error": repr(error)})
                else:
                    converted_row[mask_key] = ""

            converted_manifest_rows.append(converted_row)

        summary["splits"][split_name] = {
            "cases": len(rows),
            "converted_images": converted_images,
            "converted_masks": converted_masks,
            "skipped_existing": skipped_existing,
            "failed": len(split_failures),
        }
        summary["failed"].extend(split_failures)

    converted_manifest_dir = output_dir / "manifests"
    converted_manifest_dir.mkdir(parents=True, exist_ok=True)
    with (converted_manifest_dir / "all.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=converted_fieldnames)
        writer.writeheader()
        writer.writerows(converted_manifest_rows)

    for split_name in splits:
        rows = [row for row in converted_manifest_rows if row["split"] == split_name]
        with (converted_manifest_dir / f"{split_name}.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=converted_fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    _write_json(converted_manifest_dir / "conversion_stats.json", summary)

    if print_stats:
        print("Split array conversion finished")
        print(f"Input split folder: {split_base_dir}")
        print(f"Output array folder: {output_dir}")
        print(f"Images dtype: {image_dtype}")
        print(f"Masks dtype: {mask_dtype}")
        print(f"Output format: {output_format}")
        for split_name, split_stats in summary["splits"].items():
            print(
                f"  {split_name}: cases={split_stats['cases']}, "
                f"images={split_stats['converted_images']}, "
                f"masks={split_stats['converted_masks']}, "
                f"skipped={split_stats['skipped_existing']}, "
                f"failed={split_stats['failed']}"
            )
        if summary["failed"]:
            print(f"Failure preview: {summary['failed'][:10]}")

    return summary
