"""NIfTI preprocessing for IXI: complete quadruplets, LPS reorientation, SynthStrip skull
stripping, resampling onto the MRA grid, masked percentile normalisation, crop/pad.

Driven by preprocess.py. Functions are unchanged from the original notebook pipeline;
inspection and plotting helpers were dropped."""
# Every function imports what it needs locally (kept verbatim from the original module).


def _has_known_extension(filename, image_extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension.lower()) for extension in image_extensions)


def _strip_known_extension(filename, image_extensions):
    from pathlib import Path

    filename_lower = filename.lower()
    for extension in sorted(image_extensions, key=len, reverse=True):
        if filename_lower.endswith(extension.lower()):
            return filename[: -len(extension)]
    return Path(filename).stem


def _extract_patient_stem(file_path, modality, image_extensions):
    base_name = _strip_known_extension(file_path.name, image_extensions)
    modality_suffix = f"-{modality}"

    if not base_name.lower().endswith(modality_suffix.lower()):
        return None

    return base_name[: -len(modality_suffix)]


def _collect_modality_files(modality_dir, modality, image_extensions):
    from collections import defaultdict

    patient_files = defaultdict(list)
    ignored_files = []

    for file_path in sorted(modality_dir.iterdir()):
        if not file_path.is_file():
            continue

        if not _has_known_extension(file_path.name, image_extensions):
            ignored_files.append(file_path)
            continue

        patient_stem = _extract_patient_stem(file_path, modality, image_extensions)
        if patient_stem is None:
            ignored_files.append(file_path)
            continue

        patient_files[patient_stem].append(file_path)

    return patient_files, ignored_files


def _print_isolate_quadruplets_stats(summary, preview_limit):
    print("Quadruplet isolation finished")
    print(f"Main folder: {summary['main_folder']}")
    print(f"In-place: {summary['inplace']}")
    print(f"Output folder: {summary['output_folder']}")
    print(f"Complete patients: {summary['n_complete_patients']}")
    print(f"Incomplete patients: {summary['n_incomplete_patients']}")
    print(f"Patients by modality: {summary['patients_by_modality']}")
    print(f"Files by modality: {summary['files_by_modality']}")
    print(f"Copied files: {len(summary['copied_files'])}")
    print(f"Removed files: {len(summary['removed_files'])}")
    print(f"Ignored files by modality: {summary['ignored_files_by_modality']}")

    if preview_limit and summary["incomplete_patients"]:
        preview = summary["incomplete_patients"][:preview_limit]
        print(f"Incomplete patient preview ({len(preview)}): {preview}")


def isolate_quadruplets(
    main_folder,
    inplace=False,
    modalities=("T1", "T2", "PD", "MRA"),
    output_folder="Quadruplets",
    image_extensions=(".nii.gz", ".nii"),
    preview_limit=10,
):
    """
    Keep only patients that have matching T1, T2, PD, and MRA scans.

    Expected input structure:
        main_folder/
            T1/
            T2/
            PD/
            MRA/

    Expected filenames:
        <patient-stem>-<modality>.nii or <patient-stem>-<modality>.nii.gz
    """
    from pathlib import Path
    import shutil

    main_path = Path(main_folder)

    if not main_path.exists():
        raise FileNotFoundError(f"Main folder does not exist: {main_path}")
    if not main_path.is_dir():
        raise NotADirectoryError(f"Main folder is not a directory: {main_path}")

    modality_dirs = {modality: main_path / modality for modality in modalities}
    missing_dirs = [str(path) for path in modality_dirs.values() if not path.is_dir()]
    if missing_dirs:
        raise FileNotFoundError("Missing modality folders: " + ", ".join(missing_dirs))

    files_by_modality = {}
    ignored_by_modality = {}

    for modality, modality_dir in modality_dirs.items():
        files_by_patient, ignored_files = _collect_modality_files(
            modality_dir,
            modality,
            image_extensions=image_extensions,
        )
        files_by_modality[modality] = files_by_patient
        ignored_by_modality[modality] = ignored_files

    patient_sets = [set(files_by_modality[modality]) for modality in modalities]
    complete_patients = sorted(set.intersection(*patient_sets)) if patient_sets else []
    all_parsed_patients = sorted(set.union(*patient_sets)) if patient_sets else []
    complete_patient_set = set(complete_patients)
    incomplete_patients = [
        patient for patient in all_parsed_patients if patient not in complete_patient_set
    ]

    copied_files = []
    removed_files = []
    output_path = None

    if inplace:
        for modality in modalities:
            for patient_stem, file_paths in files_by_modality[modality].items():
                if patient_stem in complete_patient_set:
                    continue

                for file_path in file_paths:
                    file_path.unlink()
                    removed_files.append(str(file_path))
    else:
        output_path = main_path / output_folder
        for modality in modalities:
            destination_dir = output_path / modality
            destination_dir.mkdir(parents=True, exist_ok=True)

            for patient_stem in complete_patients:
                for source_path in files_by_modality[modality][patient_stem]:
                    destination_path = destination_dir / source_path.name
                    shutil.copy2(source_path, destination_path)
                    copied_files.append(str(destination_path))

    summary = {
        "main_folder": str(main_path),
        "inplace": inplace,
        "modalities": tuple(modalities),
        "output_folder": str(output_path) if output_path is not None else None,
        "n_complete_patients": len(complete_patients),
        "complete_patients": complete_patients,
        "n_incomplete_patients": len(incomplete_patients),
        "incomplete_patients": incomplete_patients,
        "patients_by_modality": {
            modality: len(files_by_modality[modality]) for modality in modalities
        },
        "files_by_modality": {
            modality: sum(len(paths) for paths in files_by_modality[modality].values())
            for modality in modalities
        },
        "ignored_files_by_modality": {
            modality: len(ignored_by_modality[modality]) for modality in modalities
        },
        "copied_files": copied_files,
        "removed_files": removed_files,
    }
    _print_isolate_quadruplets_stats(summary, preview_limit=preview_limit)


def _nifti_extension(path):
    from pathlib import Path

    path = Path(path)
    name_lower = path.name.lower()
    if name_lower.endswith(".nii.gz"):
        return ".nii.gz"
    if name_lower.endswith(".nii"):
        return ".nii"
    return path.suffix


def _save_nifti_safely(image, destination_path):
    from pathlib import Path
    import os
    import tempfile

    import nibabel as nib

    destination_path = Path(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    file_descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.",
        suffix=_nifti_extension(destination_path),
        dir=destination_path.parent,
    )
    os.close(file_descriptor)
    temp_path = Path(temp_name)

    try:
        nib.save(image, str(temp_path))
        temp_path.replace(destination_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _iter_modality_images(main_path, modalities, recursive, image_extensions):
    modality_dirs = {modality: main_path / modality for modality in modalities}
    missing_dirs = [str(path) for path in modality_dirs.values() if not path.is_dir()]
    if missing_dirs:
        raise FileNotFoundError("Missing modality folders: " + ", ".join(missing_dirs))

    for modality, modality_dir in modality_dirs.items():
        iterator = modality_dir.rglob("*") if recursive else modality_dir.iterdir()
        for file_path in sorted(iterator):
            if file_path.is_file() and _has_known_extension(file_path.name, image_extensions):
                yield modality, file_path


def _load_reorient_nii():
    import importlib.util
    from pathlib import Path

    script_path = Path(__file__).resolve().parents[1] / "Dataset" / "scripts" / "reorient_nii.py"
    if not script_path.is_file():
        raise FileNotFoundError(f"Could not find reorient_nii.py: {script_path}")

    spec = importlib.util.spec_from_file_location("dataset_reorient_nii", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.reorient_nii


def _print_reorient_dataset_stats(summary, preview_limit):
    print("LPS reorientation finished")
    print(f"Main folder: {summary['main_folder']}")
    print(f"In-place: {summary['inplace']}")
    print(f"Target orientation: {summary['target_orientation']}")
    print(f"Output folder: {summary['output_folder']}")
    print(f"Scanned files: {summary['n_scanned_files']}")
    print(f"Original orientation counts: {summary['orientation_counts']}")
    print(f"Already target orientation: {summary['n_already_target']}")
    print(f"Reoriented files: {summary['n_reoriented']}")
    print(f"Copied files: {summary['n_copied']}")
    print(f"Skipped existing files: {summary['n_skipped_existing']}")
    print(f"Failed files: {summary['n_failed']}")

    if preview_limit and summary["failed_files"]:
        preview = summary["failed_files"][:preview_limit]
        print(f"Failed file preview ({len(preview)}): {preview}")


def reorient_dataset_to_lps(
    main_folder,
    inplace=True,
    modalities=("T1", "T2", "PD", "MRA"),
    output_folder="LPS",
    target_orientation="LPS",
    recursive=False,
    overwrite=True,
    image_extensions=(".nii.gz", ".nii"),
    verbose=False,
    preview_limit=10,
):
    """
    Reorient all NIfTI images in the requested modality folders to LPS+ orientation.
    """
    from pathlib import Path
    import contextlib
    import io
    import shutil

    import nibabel as nib

    reorient_nii = _load_reorient_nii()
    main_path = Path(main_folder)
    if not main_path.is_dir():
        raise NotADirectoryError(f"Main folder is not a directory: {main_path}")

    output_path = None
    if not inplace:
        output_path = Path(output_folder)
        if not output_path.is_absolute():
            output_path = main_path / output_path
        if output_path.resolve() == main_path.resolve():
            raise ValueError("output_folder cannot be the same as main_folder")

    summary = {
        "main_folder": str(main_path),
        "inplace": inplace,
        "target_orientation": target_orientation,
        "output_folder": str(output_path) if output_path is not None else None,
        "orientation_counts": {},
        "already_target_files": [],
        "reoriented_files": [],
        "copied_files": [],
        "skipped_existing_files": [],
        "failed_files": [],
    }

    for _, source_path in _iter_modality_images(
        main_path,
        modalities=modalities,
        recursive=recursive,
        image_extensions=image_extensions,
    ):
        try:
            image = nib.load(str(source_path))
            orientation = "".join(nib.aff2axcodes(image.affine))
            summary["orientation_counts"][orientation] = (
                summary["orientation_counts"].get(orientation, 0) + 1
            )

            destination_path = source_path
            if output_path is not None:
                destination_path = output_path / source_path.relative_to(main_path)
                if destination_path.exists() and not overwrite:
                    summary["skipped_existing_files"].append(str(destination_path))
                    continue

            if orientation == target_orientation:
                summary["already_target_files"].append(str(source_path))
                if output_path is not None:
                    destination_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_path, destination_path)
                    summary["copied_files"].append(str(destination_path))
                continue

            if verbose:
                reoriented = reorient_nii(image, targ_aff=target_orientation)
            else:
                with contextlib.redirect_stdout(io.StringIO()):
                    reoriented = reorient_nii(image, targ_aff=target_orientation)

            _save_nifti_safely(reoriented, destination_path)
            summary["reoriented_files"].append(str(destination_path))
        except Exception as error:
            summary["failed_files"].append({"file": str(source_path), "error": repr(error)})

    summary["n_scanned_files"] = sum(summary["orientation_counts"].values())
    summary["n_already_target"] = len(summary["already_target_files"])
    summary["n_reoriented"] = len(summary["reoriented_files"])
    summary["n_copied"] = len(summary["copied_files"])
    summary["n_skipped_existing"] = len(summary["skipped_existing_files"])
    summary["n_failed"] = len(summary["failed_files"])
    _print_reorient_dataset_stats(summary, preview_limit=preview_limit)


def _output_stem(source_path):
    stem = source_path.name
    for extension in (".nii.gz", ".nii"):
        if stem.lower().endswith(extension):
            return stem[: -len(extension)]
    return source_path.stem


def _select_complete_patients(image_folder, modalities, image_extensions, max_patients):
    from pathlib import Path

    image_path = Path(image_folder)
    modality_dirs = {modality: image_path / modality for modality in modalities}
    missing_dirs = [str(path) for path in modality_dirs.values() if not path.is_dir()]
    if missing_dirs:
        raise FileNotFoundError("Missing modality folders: " + ", ".join(missing_dirs))

    files_by_modality = {}
    for modality, modality_dir in modality_dirs.items():
        files_by_patient, _ = _collect_modality_files(
            modality_dir,
            modality,
            image_extensions=image_extensions,
        )
        files_by_modality[modality] = files_by_patient

    patient_sets = [set(files_by_modality[modality]) for modality in modalities]
    complete_patients = sorted(set.intersection(*patient_sets)) if patient_sets else []
    if max_patients is not None:
        complete_patients = complete_patients[:max_patients]

    return complete_patients, files_by_modality


def _synthstrip_base_command(synthstrip_command):
    import sys

    if synthstrip_command is None:
        return [sys.executable, "-m", "nipreps.synthstrip"]
    if isinstance(synthstrip_command, str):
        return [synthstrip_command]
    return list(synthstrip_command)


def _print_synthstrip_stats(summary, preview_limit):
    print("SynthStrip skull stripping finished")
    print(f"Image folder: {summary['image_folder']}")
    print(f"Output folder: {summary['output_folder']}")
    print(f"Mask folder: {summary['mask_folder']}")
    print(f"Modalities: {summary['modalities']}")
    print(f"Selected patients: {summary['selected_patients']}")
    print(f"Processed files: {summary['n_processed']}")
    print(f"Skipped existing files: {summary['n_skipped_existing']}")
    print(f"Failed files: {summary['n_failed']}")

    if preview_limit and summary["failed_files"]:
        preview = summary["failed_files"][:preview_limit]
        print(f"Failed file preview ({len(preview)}): {preview}")


def run_synthstrip_on_modalities(
    image_folder,
    output_folder,
    modalities=("T1", "T2", "PD"),
    max_patients=4,
    image_extensions=(".nii.gz", ".nii"),
    model_path="Dataset/models/synthstrip.1.pt",
    mask_folder=None,
    synthstrip_command=None,
    use_gpu=False,
    num_threads=None,
    border=1,
    overwrite=False,
    preview_limit=10,
):
    """
    Run SynthStrip on selected modality folders and write skull-stripped images.

    The input folder is expected to contain one subfolder per modality. Outputs are
    written to output_folder/<modality>/, while masks are written to
    output_folder/masks/<modality>/ unless mask_folder is provided.
    """
    from pathlib import Path
    import subprocess

    image_path = Path(image_folder)
    output_path = Path(output_folder)
    model_path = Path(model_path)

    if not image_path.is_dir():
        raise NotADirectoryError(f"Image folder is not a directory: {image_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"SynthStrip model file does not exist: {model_path}")

    mask_path = Path(mask_folder) if mask_folder is not None else output_path / "masks"
    selected_patients, files_by_modality = _select_complete_patients(
        image_path,
        modalities=modalities,
        image_extensions=image_extensions,
        max_patients=max_patients,
    )

    summary = {
        "image_folder": str(image_path),
        "output_folder": str(output_path),
        "mask_folder": str(mask_path),
        "modalities": tuple(modalities),
        "selected_patients": selected_patients,
        "processed_files": [],
        "skipped_existing_files": [],
        "failed_files": [],
    }

    if not selected_patients:
        _print_synthstrip_stats(
            {
                **summary,
                "n_processed": 0,
                "n_skipped_existing": 0,
                "n_failed": 0,
            },
            preview_limit=preview_limit,
        )
        return

    base_command = _synthstrip_base_command(synthstrip_command)
    total = len(selected_patients) * len(modalities)
    index = 0

    for patient_stem in selected_patients:
        for modality in modalities:
            index += 1
            source_files = files_by_modality[modality][patient_stem]
            source_path = source_files[0]
            if len(source_files) > 1:
                print(
                    f"[{index}/{total}] Warning: multiple {modality} files for "
                    f"{patient_stem}; using {source_path.name}"
                )

            destination_dir = output_path / modality
            destination_mask_dir = mask_path / modality
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination_mask_dir.mkdir(parents=True, exist_ok=True)

            stem = _output_stem(source_path)
            stripped_path = destination_dir / f"{stem}_stripped.nii.gz"
            mask_output_path = destination_mask_dir / f"{stem}_mask.nii.gz"

            if stripped_path.exists() and mask_output_path.exists() and not overwrite:
                print(f"[{index}/{total}] Skipping {source_path.name} (already done)")
                summary["skipped_existing_files"].append(str(stripped_path))
                continue

            command = [
                *base_command,
                "-i",
                str(source_path),
                "-o",
                str(stripped_path),
                "-m",
                str(mask_output_path),
                "--model",
                str(model_path),
                "-b",
                str(border),
            ]
            if use_gpu:
                command.append("-g")
            if num_threads is not None:
                command.extend(["-n", str(num_threads)])

            print(f"[{index}/{total}] Processing {source_path.name}")
            try:
                subprocess.run(command, check=True)
                summary["processed_files"].append(str(stripped_path))
            except subprocess.CalledProcessError as error:
                summary["failed_files"].append(
                    {
                        "file": str(source_path),
                        "error": repr(error),
                    }
                )

    summary["n_processed"] = len(summary["processed_files"])
    summary["n_skipped_existing"] = len(summary["skipped_existing_files"])
    summary["n_failed"] = len(summary["failed_files"])
    _print_synthstrip_stats(summary, preview_limit=preview_limit)


def _sitk_image(input_image):
    import SimpleITK as sitk

    if isinstance(input_image, sitk.Image):
        return input_image
    return sitk.ReadImage(str(input_image))


def _sitk_interpolator(interpolator, is_label=False):
    import SimpleITK as sitk

    if interpolator is None:
        return sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    if isinstance(interpolator, int):
        return interpolator

    name = str(interpolator).lower()
    options = {
        "nearest": sitk.sitkNearestNeighbor,
        "nearestneighbor": sitk.sitkNearestNeighbor,
        "nn": sitk.sitkNearestNeighbor,
        "label": sitk.sitkNearestNeighbor,
        "linear": sitk.sitkLinear,
        "bspline": sitk.sitkBSpline,
        "spline": sitk.sitkBSpline,
        "gaussian": sitk.sitkGaussian,
    }
    if name not in options:
        raise ValueError(f"Unknown interpolator: {interpolator}")
    return options[name]


def _sitk_transform(transform=None, transform_path=None):
    import SimpleITK as sitk

    if transform is not None and transform_path is not None:
        raise ValueError("Use either transform or transform_path, not both")
    if transform is not None:
        return transform
    if transform_path is not None:
        return sitk.ReadTransform(str(transform_path))
    return sitk.Transform(3, sitk.sitkIdentity)


def _sitk_target_size_for_spacing(image, target_spacing):
    size = image.GetSize()
    spacing = image.GetSpacing()
    return tuple(
        max(1, int(round(size[axis] * spacing[axis] / target_spacing[axis])))
        for axis in range(3)
    )


def _write_sitk_image_safely(image, destination_path):
    from pathlib import Path
    import os
    import tempfile

    import SimpleITK as sitk

    destination_path = Path(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    file_descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.",
        suffix=_nifti_extension(destination_path),
        dir=destination_path.parent,
    )
    os.close(file_descriptor)
    temp_path = Path(temp_name)

    try:
        sitk.WriteImage(image, str(temp_path))
        temp_path.replace(destination_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _copy_sitk_metadata(source_image, array):
    import SimpleITK as sitk

    output = sitk.GetImageFromArray(array)
    output.CopyInformation(source_image)
    return output


def _print_image_grid(prefix, image):
    print(
        f"{prefix}: size={tuple(image.GetSize())}, "
        f"spacing={tuple(round(float(v), 6) for v in image.GetSpacing())}"
    )


def resample_image_to_spacing(
    input_image_path,
    output_image_path,
    target_spacing=(0.5, 0.5, 0.8),
    interpolator="linear",
    default_value=0.0,
    output_pixel_type=None,
    overwrite=False,
    print_stats=True,
):
    """
    Resample one image to a target voxel spacing.

    Spacing is in SimpleITK/NIfTI axis order: (x, y, z). For your next step,
    use target_spacing=(0.5, 0.5, 0.8).
    """
    from pathlib import Path
    import SimpleITK as sitk

    output_path = Path(output_image_path)
    if output_path.exists() and not overwrite:
        if print_stats:
            print(f"Skipping existing resampled image: {output_path}")
        return

    image = _sitk_image(input_image_path)
    target_spacing = tuple(float(value) for value in target_spacing)
    target_size = _sitk_target_size_for_spacing(image, target_spacing)

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(target_size)
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetDefaultPixelValue(float(default_value))
    resampler.SetInterpolator(_sitk_interpolator(interpolator))
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetOutputPixelType(output_pixel_type or image.GetPixelID())

    resampled = resampler.Execute(image)
    _write_sitk_image_safely(resampled, output_path)

    if print_stats:
        print("Resampled image")
        print(f"Input: {input_image_path}")
        print(f"Output: {output_path}")
        _print_image_grid("Before", image)
        _print_image_grid("After", resampled)


def resample_folder_to_spacing(
    input_folder,
    output_folder,
    target_spacing=(0.5, 0.5, 0.8),
    recursive=False,
    image_extensions=(".nii.gz", ".nii"),
    interpolator="linear",
    default_value=0.0,
    overwrite=False,
    max_images=None,
    print_stats=True,
    output_pixel_type=None,
):
    """Resample every NIfTI image in a folder to the target spacing."""
    from pathlib import Path

    input_path = Path(input_folder)
    output_path = Path(output_folder)
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input folder does not exist: {input_path}")

    iterator = input_path.rglob("*") if recursive else input_path.iterdir()
    files = [
        path for path in sorted(iterator)
        if path.is_file() and _has_known_extension(path.name, image_extensions)
    ]
    if max_images is not None:
        files = files[:max_images]

    processed = 0
    skipped = 0
    failed = []
    for file_path in files:
        destination_path = output_path / file_path.relative_to(input_path)
        if destination_path.exists() and not overwrite:
            skipped += 1
            continue
        try:
            resample_image_to_spacing(
                file_path,
                destination_path,
                target_spacing=target_spacing,
                interpolator=interpolator,
                default_value=default_value,
                overwrite=True,
                print_stats=False,
                output_pixel_type=output_pixel_type,
            )
            processed += 1
        except Exception as error:
            failed.append({"file": str(file_path), "error": repr(error)})

    if print_stats:
        print("Folder spacing resampling finished")
        print(f"Input folder: {input_path}")
        print(f"Output folder: {output_path}")
        print(f"Target spacing: {target_spacing}")
        print(f"Processed files: {processed}")
        print(f"Skipped existing files: {skipped}")
        print(f"Failed files: {len(failed)}")
        if failed:
            print(f"Failed file preview: {failed[:10]}")


def _extract_patient_stem_flexible(file_path, modality=None, image_extensions=(".nii.gz", ".nii")):
    from pathlib import Path

    base_name = _strip_known_extension(Path(file_path).name, image_extensions)
    if modality is None:
        return base_name

    token = f"-{modality}".lower()
    index = base_name.lower().rfind(token)
    if index < 0:
        return None
    return base_name[:index]


def _extract_patient_stem_any_modality(
    file_path,
    modalities=("T1", "T2", "PD", "MRA"),
    image_extensions=(".nii.gz", ".nii"),
):
    from pathlib import Path

    for modality in modalities:
        patient = _extract_patient_stem_flexible(
            file_path,
            modality=modality,
            image_extensions=image_extensions,
        )
        if patient is not None:
            return patient

    return _strip_known_extension(Path(file_path).name, image_extensions)


def _collect_files_by_patient_stem(
    folder,
    modality=None,
    recursive=False,
    image_extensions=(".nii.gz", ".nii"),
):
    from collections import defaultdict
    from pathlib import Path

    folder = Path(folder)
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    files_by_patient = defaultdict(list)
    ignored_files = []

    for path in sorted(iterator):
        if not path.is_file():
            continue
        if not _has_known_extension(path.name, image_extensions):
            ignored_files.append(path)
            continue
        patient = _extract_patient_stem_flexible(
            path,
            modality=modality,
            image_extensions=image_extensions,
        )
        if patient is None:
            ignored_files.append(path)
            continue
        files_by_patient[patient].append(path)

    return files_by_patient, ignored_files


def _collect_files_by_patient_any_modality(
    folder,
    modalities=("T1", "T2", "PD", "MRA"),
    recursive=False,
    image_extensions=(".nii.gz", ".nii"),
):
    from collections import defaultdict
    from pathlib import Path

    folder = Path(folder)
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    files_by_patient = defaultdict(list)
    ignored_files = []

    for path in sorted(iterator):
        if not path.is_file():
            continue
        if not _has_known_extension(path.name, image_extensions):
            ignored_files.append(path)
            continue

        patient = None
        for modality in modalities:
            patient = _extract_patient_stem_flexible(
                path,
                modality=modality,
                image_extensions=image_extensions,
            )
            if patient is not None:
                break

        if patient is None:
            ignored_files.append(path)
            continue
        files_by_patient[patient].append(path)

    return files_by_patient, ignored_files


def _collect_transforms_by_patient_stem(
    transform_folder,
    modality=None,
    recursive=False,
    transform_extensions=(".tfm", ".h5", ".mat", ".txt"),
):
    from collections import defaultdict
    from pathlib import Path

    transform_folder = Path(transform_folder)
    iterator = transform_folder.rglob("*") if recursive else transform_folder.iterdir()
    transforms_by_patient = defaultdict(list)
    ignored_files = []

    for path in sorted(iterator):
        if not path.is_file():
            continue
        if not _has_known_extension(path.name, transform_extensions):
            ignored_files.append(path)
            continue
        patient = _extract_patient_stem_flexible(
            path,
            modality=modality,
            image_extensions=transform_extensions,
        )
        if patient is None:
            patient = _strip_known_extension(path.name, transform_extensions)
        transforms_by_patient[patient].append(path)

    return transforms_by_patient, ignored_files


def _resolve_transform_map(transform_folder, transform_map, input_modality, recursive, transform_extensions):
    from pathlib import Path

    if transform_folder is not None and transform_map is not None:
        raise ValueError("Use either transform_folder or transform_map, not both")
    if transform_map is not None:
        return {str(key): Path(value) for key, value in transform_map.items()}, []
    if transform_folder is None:
        return {}, []

    transforms_by_patient, ignored = _collect_transforms_by_patient_stem(
        transform_folder,
        modality=input_modality,
        recursive=recursive,
        transform_extensions=transform_extensions,
    )
    return {patient: paths[0] for patient, paths in transforms_by_patient.items()}, ignored


def _single_reference_or_map(reference_folder_or_path, reference_modality, recursive, image_extensions):
    from pathlib import Path

    reference_path = Path(reference_folder_or_path)
    if reference_path.is_file():
        return reference_path, {}, []
    if not reference_path.is_dir():
        raise FileNotFoundError(f"Reference path does not exist: {reference_path}")
    references_by_patient, ignored = _collect_files_by_patient_stem(
        reference_path,
        modality=reference_modality,
        recursive=recursive,
        image_extensions=image_extensions,
    )
    return None, {patient: paths[0] for patient, paths in references_by_patient.items()}, ignored


def _print_resample_to_reference_folder_stats(summary, preview_limit=10):
    print("Folder reference-grid resampling finished")
    print(f"Input folder: {summary['input_folder']}")
    print(f"Reference: {summary['reference']}")
    print(f"Output folder: {summary['output_folder']}")
    print(f"Input modality: {summary['input_modality']}")
    print(f"Reference modality: {summary['reference_modality']}")
    print(f"Processed files: {summary['n_processed']}")
    print(f"Skipped existing files: {summary['n_skipped_existing']}")
    print(f"Failed files: {summary['n_failed']}")
    print(f"Ignored input files: {summary['n_ignored_input_files']}")
    print(f"Ignored reference files: {summary['n_ignored_reference_files']}")
    print(f"Ignored transform files: {summary['n_ignored_transform_files']}")
    if preview_limit and summary["failed_files"]:
        print(f"Failed file preview: {summary['failed_files'][:preview_limit]}")


def resample_image_to_reference(
    input_image_path,
    reference_image_path,
    output_image_path,
    transform=None,
    transform_path=None,
    interpolator="linear",
    is_label=False,
    default_value=0.0,
    output_pixel_type=None,
    overwrite=False,
    print_stats=True,
):
    """
    Resample an image into the exact grid of a reference image.

    If you pass a registration transform, it must be the transform expected by
    SimpleITK Resample: output/reference physical points mapped into the input
    image physical space.
    """
    from pathlib import Path
    import SimpleITK as sitk

    output_path = Path(output_image_path)
    if output_path.exists() and not overwrite:
        if print_stats:
            print(f"Skipping existing reference-resampled image: {output_path}")
        return

    image = _sitk_image(input_image_path)
    reference = _sitk_image(reference_image_path)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetTransform(_sitk_transform(transform=transform, transform_path=transform_path))
    resampler.SetDefaultPixelValue(float(default_value))
    resampler.SetInterpolator(_sitk_interpolator(interpolator, is_label=is_label))
    if output_pixel_type is not None:
        resampler.SetOutputPixelType(output_pixel_type)
    elif is_label:
        resampler.SetOutputPixelType(sitk.sitkUInt8)
    else:
        resampler.SetOutputPixelType(image.GetPixelID())

    resampled = resampler.Execute(image)
    if is_label:
        resampled = sitk.Cast(resampled > 0, sitk.sitkUInt8)
    _write_sitk_image_safely(resampled, output_path)

    if print_stats:
        print("Resampled image to reference grid")
        print(f"Input: {input_image_path}")
        print(f"Reference: {reference_image_path}")
        print(f"Output: {output_path}")
        _print_image_grid("Input grid", image)
        _print_image_grid("Reference grid", reference)
        _print_image_grid("Output grid", resampled)


def resample_mask_to_reference(
    mask_path,
    reference_image_path,
    output_mask_path,
    transform=None,
    transform_path=None,
    foreground_threshold=0.5,
    overwrite=False,
    print_stats=True,
):
    """Transform a SynStrip mask into the exact grid of a reference image."""
    from pathlib import Path
    import SimpleITK as sitk

    output_path = Path(output_mask_path)
    if output_path.exists() and not overwrite:
        if print_stats:
            print(f"Skipping existing reference-resampled mask: {output_path}")
        return

    mask = _sitk_image(mask_path)
    binary_mask = sitk.Cast(mask > foreground_threshold, sitk.sitkUInt8)
    temporary_input = output_path.parent / f".{output_path.stem}_binary_input.nii.gz"
    _write_sitk_image_safely(binary_mask, temporary_input)
    try:
        resample_image_to_reference(
            temporary_input,
            reference_image_path,
            output_path,
            transform=transform,
            transform_path=transform_path,
            interpolator="nearest",
            is_label=True,
            default_value=0,
            overwrite=True,
            print_stats=False,
        )
    finally:
        temporary_input.unlink(missing_ok=True)

    if print_stats:
        result = sitk.ReadImage(str(output_path))
        print("SynStrip mask transformed to reference grid")
        print(f"Mask: {mask_path}")
        print(f"Reference: {reference_image_path}")
        print(f"Output: {output_path}")
        _print_image_grid("Output mask grid", result)


def resample_folder_to_reference(
    input_folder,
    reference_folder_or_path,
    output_folder,
    input_modality=None,
    reference_modality="MRA",
    transform_folder=None,
    transform_map=None,
    transform_extensions=(".tfm", ".h5", ".mat", ".txt"),
    recursive=False,
    image_extensions=(".nii.gz", ".nii"),
    interpolator="linear",
    default_value=0.0,
    output_pixel_type=None,
    overwrite=False,
    max_images=None,
    require_transform=False,
    preview_limit=10,
    print_stats=True,
):
    """
    Resample a whole folder of images into each patient's reference image grid.

    Files are matched by patient stem. For example:
        IXI002-Guys-0828-T1_stripped.nii.gz -> IXI002-Guys-0828-MRA.nii.gz

    reference_folder_or_path can be either a reference folder or one reference
    image path that should be used for every input image.
    """
    from pathlib import Path

    input_path = Path(input_folder)
    output_path = Path(output_folder)
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input folder does not exist: {input_path}")

    input_files_by_patient, ignored_input = _collect_files_by_patient_stem(
        input_path,
        modality=input_modality,
        recursive=recursive,
        image_extensions=image_extensions,
    )
    single_reference, references_by_patient, ignored_reference = _single_reference_or_map(
        reference_folder_or_path,
        reference_modality=reference_modality,
        recursive=recursive,
        image_extensions=image_extensions,
    )
    transforms_by_patient, ignored_transform = _resolve_transform_map(
        transform_folder=transform_folder,
        transform_map=transform_map,
        input_modality=input_modality,
        recursive=recursive,
        transform_extensions=transform_extensions,
    )

    pairs = []
    for patient, source_paths in input_files_by_patient.items():
        for source_path in source_paths:
            pairs.append((patient, source_path))
    pairs.sort(key=lambda item: str(item[1]))
    if max_images is not None:
        pairs = pairs[:max_images]

    processed = 0
    skipped = 0
    failed = []

    for patient, source_path in pairs:
        reference_path = single_reference or references_by_patient.get(patient)
        if reference_path is None:
            failed.append({"file": str(source_path), "error": f"No reference found for patient {patient}"})
            continue

        transform_path = transforms_by_patient.get(patient)
        if require_transform and transform_path is None:
            failed.append({"file": str(source_path), "error": f"No transform found for patient {patient}"})
            continue

        destination_path = output_path / source_path.relative_to(input_path)
        if destination_path.exists() and not overwrite:
            skipped += 1
            continue

        try:
            resample_image_to_reference(
                input_image_path=source_path,
                reference_image_path=reference_path,
                output_image_path=destination_path,
                transform_path=transform_path,
                interpolator=interpolator,
                default_value=default_value,
                output_pixel_type=output_pixel_type,
                overwrite=True,
                print_stats=False,
            )
            processed += 1
        except Exception as error:
            failed.append({"file": str(source_path), "error": repr(error)})

    summary = {
        "input_folder": str(input_path),
        "reference": str(reference_folder_or_path),
        "output_folder": str(output_path),
        "input_modality": input_modality,
        "reference_modality": reference_modality,
        "n_processed": processed,
        "n_skipped_existing": skipped,
        "n_failed": len(failed),
        "failed_files": failed,
        "n_ignored_input_files": len(ignored_input),
        "n_ignored_reference_files": len(ignored_reference),
        "n_ignored_transform_files": len(ignored_transform),
    }
    if print_stats:
        _print_resample_to_reference_folder_stats(summary, preview_limit=preview_limit)


def resample_mask_folder_to_reference(
    mask_folder,
    reference_folder_or_path,
    output_mask_folder,
    mask_modality=None,
    reference_modality="MRA",
    transform_folder=None,
    transform_map=None,
    transform_extensions=(".tfm", ".h5", ".mat", ".txt"),
    recursive=False,
    image_extensions=(".nii.gz", ".nii"),
    foreground_threshold=0.5,
    overwrite=False,
    max_images=None,
    require_transform=False,
    preview_limit=10,
    print_stats=True,
):
    """
    Resample a whole folder of binary masks into each patient's reference grid.

    Uses nearest-neighbor interpolation and writes uint8 binary masks.
    """
    from pathlib import Path

    mask_path = Path(mask_folder)
    output_path = Path(output_mask_folder)
    if not mask_path.is_dir():
        raise NotADirectoryError(f"Mask folder does not exist: {mask_path}")

    masks_by_patient, ignored_masks = _collect_files_by_patient_stem(
        mask_path,
        modality=mask_modality,
        recursive=recursive,
        image_extensions=image_extensions,
    )
    single_reference, references_by_patient, ignored_reference = _single_reference_or_map(
        reference_folder_or_path,
        reference_modality=reference_modality,
        recursive=recursive,
        image_extensions=image_extensions,
    )
    transforms_by_patient, ignored_transform = _resolve_transform_map(
        transform_folder=transform_folder,
        transform_map=transform_map,
        input_modality=mask_modality,
        recursive=recursive,
        transform_extensions=transform_extensions,
    )

    pairs = []
    for patient, source_paths in masks_by_patient.items():
        for source_path in source_paths:
            pairs.append((patient, source_path))
    pairs.sort(key=lambda item: str(item[1]))
    if max_images is not None:
        pairs = pairs[:max_images]

    processed = 0
    skipped = 0
    failed = []

    for patient, source_path in pairs:
        reference_path = single_reference or references_by_patient.get(patient)
        if reference_path is None:
            failed.append({"file": str(source_path), "error": f"No reference found for patient {patient}"})
            continue

        transform_path = transforms_by_patient.get(patient)
        if require_transform and transform_path is None:
            failed.append({"file": str(source_path), "error": f"No transform found for patient {patient}"})
            continue

        destination_path = output_path / source_path.relative_to(mask_path)
        if destination_path.exists() and not overwrite:
            skipped += 1
            continue

        try:
            resample_mask_to_reference(
                mask_path=source_path,
                reference_image_path=reference_path,
                output_mask_path=destination_path,
                transform_path=transform_path,
                foreground_threshold=foreground_threshold,
                overwrite=True,
                print_stats=False,
            )
            processed += 1
        except Exception as error:
            failed.append({"file": str(source_path), "error": repr(error)})

    summary = {
        "input_folder": str(mask_path),
        "reference": str(reference_folder_or_path),
        "output_folder": str(output_path),
        "input_modality": mask_modality,
        "reference_modality": reference_modality,
        "n_processed": processed,
        "n_skipped_existing": skipped,
        "n_failed": len(failed),
        "failed_files": failed,
        "n_ignored_input_files": len(ignored_masks),
        "n_ignored_reference_files": len(ignored_reference),
        "n_ignored_transform_files": len(ignored_transform),
    }
    if print_stats:
        _print_resample_to_reference_folder_stats(summary, preview_limit=preview_limit)


def _mask_center_physical(mask_image):
    import SimpleITK as sitk

    binary_mask = sitk.Cast(mask_image > 0, sitk.sitkUInt8)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(binary_mask)
    if not stats.HasLabel(1):
        raise ValueError("Mask is empty; cannot compute crop center")

    bbox = stats.GetBoundingBox(1)
    center_index = (
        bbox[0] + (bbox[3] - 1) / 2.0,
        bbox[1] + (bbox[4] - 1) / 2.0,
        bbox[2] + (bbox[5] - 1) / 2.0,
    )
    return mask_image.TransformContinuousIndexToPhysicalPoint(center_index)


def _crop_center_index(image, center="image", center_mask_path=None):
    if center_mask_path is not None:
        mask = _sitk_image(center_mask_path)
        center_physical = _mask_center_physical(mask)
        return image.TransformPhysicalPointToContinuousIndex(center_physical)

    if center is None or center == "image":
        size = image.GetSize()
        return tuple((size[axis] - 1) / 2.0 for axis in range(3))

    if len(center) != 3:
        raise ValueError(f"center must have 3 values, got {center}")
    return tuple(float(value) for value in center)


def _crop_or_pad_start_index(image, target_size, center="image", center_mask_path=None, z_crop_pad_mode="center"):
    source_size = tuple(int(value) for value in image.GetSize())
    center_index = _crop_center_index(image, center=center, center_mask_path=center_mask_path)
    start_index = [
        int(round(center_index[axis] - target_size[axis] / 2.0))
        for axis in range(3)
    ]

    mode = str(z_crop_pad_mode).lower()
    if mode in ("center", "centre"):
        pass
    elif mode in ("front", "first", "low", "lower", "start"):
        # Align the high-z end of the source and output. For 100 -> 96 this
        # crops source slices 0..3; for 92 -> 96 this pads output slices 0..3.
        start_index[2] = source_size[2] - int(target_size[2])
    else:
        raise ValueError("z_crop_pad_mode must be 'center' or 'front'")

    return tuple(start_index), tuple(center_index)


def crop_or_pad_to_size(
    input_image_path,
    output_image_path,
    target_size=(512, 512, 128),
    center="image",
    center_mask_path=None,
    z_crop_pad_mode="center",
    is_label=False,
    pad_value=0.0,
    interpolator=None,
    overwrite=False,
    print_stats=True,
):
    """
    Crop or pad one image to a fixed voxel size while preserving its spacing,
    direction, and physical coordinate system.

    target_size is in SimpleITK/NIfTI axis order: (x, y, z).

    z_crop_pad_mode="center" keeps the previous behavior. z_crop_pad_mode="front"
    center-crops/pads x/y but applies all z crop/pad at the low-z side:
    100 -> 96 removes the first 4 z slices, and 92 -> 96 pads the first 4
    output z slices.
    """
    from pathlib import Path
    import SimpleITK as sitk

    output_path = Path(output_image_path)
    if output_path.exists() and not overwrite:
        if print_stats:
            print(f"Skipping existing cropped/padded image: {output_path}")
        return

    image = _sitk_image(input_image_path)
    target_size = tuple(int(value) for value in target_size)
    start_index, center_index = _crop_or_pad_start_index(
        image,
        target_size,
        center=center,
        center_mask_path=center_mask_path,
        z_crop_pad_mode=z_crop_pad_mode,
    )
    output_origin = image.TransformIndexToPhysicalPoint(start_index)

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(target_size)
    resampler.SetOutputSpacing(image.GetSpacing())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(output_origin)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetDefaultPixelValue(float(pad_value))
    resampler.SetInterpolator(_sitk_interpolator(interpolator, is_label=is_label))
    resampler.SetOutputPixelType(sitk.sitkUInt8 if is_label else image.GetPixelID())

    output = resampler.Execute(image)
    if is_label:
        output = sitk.Cast(output > 0, sitk.sitkUInt8)
    _write_sitk_image_safely(output, output_path)

    if print_stats:
        print("Crop/pad finished")
        print(f"Input: {input_image_path}")
        print(f"Output: {output_path}")
        print(f"Target size: {target_size}")
        print(f"Z crop/pad mode: {z_crop_pad_mode}")
        print(f"Center index: {tuple(round(float(v), 3) for v in center_index)}")
        print(f"Start index: {start_index}")
        _print_image_grid("Before", image)
        _print_image_grid("After", output)


def _folder_items_with_modalities(input_folders, modalities=None):
    from pathlib import Path

    if isinstance(input_folders, dict):
        return [(str(modality), Path(folder)) for modality, folder in input_folders.items()]

    folders = [Path(folder) for folder in input_folders]
    if modalities is None:
        return [(folder.name, folder) for folder in folders]

    if len(modalities) != len(folders):
        raise ValueError("modalities must have the same length as input_folders")
    return [(str(modality), folder) for modality, folder in zip(modalities, folders)]


def crop_or_pad_folders_to_size(
    input_folders,
    output_folder,
    target_size=(576, 576, 96),
    modalities=None,
    mask_folders=None,
    output_mask_folder=None,
    recursive=False,
    image_extensions=(".nii.gz", ".nii"),
    center="image",
    z_crop_pad_mode="front",
    is_label=False,
    pad_value=0.0,
    interpolator=None,
    overwrite=False,
    max_images=None,
    preview_limit=10,
    print_stats=True,
):
    """
    Crop/pad one or more modality folders into output_folder/<modality>/.

    input_folders can be a list of folders or a dict of modality -> folder. If
    a list is used, modality names are inferred from folder names unless
    modalities is supplied.

    If mask_folders is provided, matching masks are cropped/padded as labels
    with the same target size and z rule. They are written to
    output_folder/masks/<modality>/ by default, or to
    output_mask_folder/<modality>/ when output_mask_folder is supplied.

    By default this uses target_size=(576, 576, 96), center-crops/pads x/y, and
    applies all z crop/pad at the first/low-z slices. That means 100 -> 96
    removes the first 4 z slices, and 92 -> 96 pads the first 4 output z slices.
    Existing output files are skipped when overwrite=False.
    """
    from pathlib import Path

    output_path = Path(output_folder)
    target_size = tuple(int(value) for value in target_size)
    items = _folder_items_with_modalities(input_folders, modalities=modalities)
    mask_items = None
    mask_output_path = None
    if mask_folders is not None:
        if isinstance(mask_folders, dict):
            mask_items = {str(modality): Path(folder) for modality, folder in mask_folders.items()}
        else:
            mask_folder_list = [Path(folder) for folder in mask_folders]
            if len(mask_folder_list) != len(items):
                raise ValueError("mask_folders must have the same length as input_folders")
            mask_items = {
                str(modality): mask_folder
                for (modality, _), mask_folder in zip(items, mask_folder_list)
            }
        mask_output_path = Path(output_mask_folder) if output_mask_folder is not None else output_path / "masks"

    summary = {
        "output_folder": str(output_path),
        "output_mask_folder": None if mask_output_path is None else str(mask_output_path),
        "target_size": target_size,
        "z_crop_pad_mode": z_crop_pad_mode,
        "processed": {},
        "skipped_existing": {},
        "failed": {},
        "ignored_files": {},
        "processed_masks": {},
        "skipped_existing_masks": {},
        "failed_masks": {},
        "ignored_mask_files": {},
    }

    for modality, folder in items:
        if not folder.is_dir():
            raise NotADirectoryError(f"{modality} folder does not exist: {folder}")

        iterator = folder.rglob("*") if recursive else folder.iterdir()
        all_paths = sorted(iterator)
        files = [
            path for path in all_paths
            if path.is_file() and _has_known_extension(path.name, image_extensions)
        ]
        ignored_files = [
            path for path in all_paths
            if path.is_file() and not _has_known_extension(path.name, image_extensions)
        ]
        if max_images is not None:
            files = files[:max_images]

        modality_output = output_path / modality
        processed = 0
        skipped = 0
        failed = []
        mask_folder = None
        masks_by_patient = None
        ignored_mask_files = []
        processed_masks = 0
        skipped_masks = 0
        failed_masks = []

        if mask_items is not None:
            if modality not in mask_items:
                raise ValueError(f"No mask folder provided for modality {modality}")
            mask_folder = mask_items[modality]
            if not mask_folder.is_dir():
                raise NotADirectoryError(f"{modality} mask folder does not exist: {mask_folder}")
            masks_by_patient, ignored_mask_files = _collect_files_by_patient_any_modality(
                mask_folder,
                modalities=("T1", "T2", "PD", "MRA"),
                recursive=recursive,
                image_extensions=image_extensions,
            )

        for source_path in files:
            destination_path = modality_output / source_path.relative_to(folder)
            if destination_path.exists() and not overwrite:
                skipped += 1
            else:
                try:
                    crop_or_pad_to_size(
                        input_image_path=source_path,
                        output_image_path=destination_path,
                        target_size=target_size,
                        center=center,
                        z_crop_pad_mode=z_crop_pad_mode,
                        is_label=is_label,
                        pad_value=pad_value,
                        interpolator=interpolator,
                        overwrite=overwrite,
                        print_stats=False,
                    )
                    processed += 1
                except Exception as error:
                    failed.append({"file": str(source_path), "error": repr(error)})

            if masks_by_patient is not None:
                patient = _extract_patient_stem_any_modality(
                    source_path,
                    image_extensions=image_extensions,
                )
                mask_candidates = masks_by_patient.get(patient)
                if not mask_candidates:
                    failed_masks.append(
                        {"file": str(source_path), "error": f"No mask found for patient {patient}"}
                    )
                    continue

                mask_source_path = mask_candidates[0]
                mask_destination_path = (mask_output_path / modality) / mask_source_path.relative_to(mask_folder)
                if mask_destination_path.exists() and not overwrite:
                    skipped_masks += 1
                    continue

                try:
                    crop_or_pad_to_size(
                        input_image_path=mask_source_path,
                        output_image_path=mask_destination_path,
                        target_size=target_size,
                        center=center,
                        z_crop_pad_mode=z_crop_pad_mode,
                        is_label=True,
                        pad_value=0.0,
                        interpolator="nearest",
                        overwrite=overwrite,
                        print_stats=False,
                    )
                    processed_masks += 1
                except Exception as error:
                    failed_masks.append({"file": str(mask_source_path), "error": repr(error)})

        summary["processed"][modality] = processed
        summary["skipped_existing"][modality] = skipped
        summary["failed"][modality] = failed
        summary["ignored_files"][modality] = len(ignored_files)
        summary["processed_masks"][modality] = processed_masks
        summary["skipped_existing_masks"][modality] = skipped_masks
        summary["failed_masks"][modality] = failed_masks
        summary["ignored_mask_files"][modality] = len(ignored_mask_files)

    if print_stats:
        print("Folder crop/pad finished")
        print(f"Output folder: {output_path}")
        if mask_output_path is not None:
            print(f"Output mask folder: {mask_output_path}")
        print(f"Target size: {target_size}")
        print(f"Z crop/pad mode: {z_crop_pad_mode}")
        print(f"Max images per folder: {max_images}")
        for modality, folder in items:
            print(f"\n{modality}")
            print(f"Input folder: {folder}")
            print(f"Output subfolder: {output_path / modality}")
            print(f"Processed images: {summary['processed'][modality]}")
            print(f"Skipped existing images: {summary['skipped_existing'][modality]}")
            print(f"Failed images: {len(summary['failed'][modality])}")
            print(f"Ignored files: {summary['ignored_files'][modality]}")
            if summary["failed"][modality]:
                print(f"Failed preview: {summary['failed'][modality][:preview_limit]}")
            if mask_items is not None:
                print(f"Input mask folder: {mask_items[modality]}")
                print(f"Output mask subfolder: {mask_output_path / modality}")
                print(f"Processed masks: {summary['processed_masks'][modality]}")
                print(f"Skipped existing masks: {summary['skipped_existing_masks'][modality]}")
                print(f"Failed masks: {len(summary['failed_masks'][modality])}")
                print(f"Ignored mask files: {summary['ignored_mask_files'][modality]}")
                if summary["failed_masks"][modality]:
                    print(f"Failed mask preview: {summary['failed_masks'][modality][:preview_limit]}")

    return summary


def _load_image_and_mask_arrays(image_path, mask_path):
    import numpy as np
    import SimpleITK as sitk

    image = _sitk_image(image_path)
    mask = _sitk_image(mask_path)
    if image.GetSize() != mask.GetSize():
        raise ValueError(f"Mask size {mask.GetSize()} does not match image size {image.GetSize()}")
    if image.GetDimension() != 3 or mask.GetDimension() != 3:
        raise ValueError("Image and mask must both be 3D")

    array = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)
    mask_array = sitk.GetArrayFromImage(mask) > 0
    if not np.any(mask_array):
        raise ValueError(f"Mask is empty: {mask_path}")
    return image, array, mask_array


def _normalization_metadata_fieldnames():
    return [
        "patient_id",
        "modality",
        "method",
        "input_path",
        "mask_path",
        "output_path",
        "lower_percentile",
        "upper_percentile",
        "clip_low_value",
        "clip_high_value",
        "zscore_mean",
        "zscore_std",
        "scale_output_min",
        "scale_output_max",
        "outside_value",
        "inside_voxel_count",
        "total_voxel_count",
        "mask_foreground_fraction",
        "inside_before_min",
        "inside_before_max",
        "inside_before_mean",
        "inside_before_std",
        "inside_after_min",
        "inside_after_max",
        "inside_after_mean",
        "inside_after_std",
        "size_x",
        "size_y",
        "size_z",
        "spacing_x",
        "spacing_y",
        "spacing_z",
    ]


def _normalization_metadata_key(record):
    return (
        str(record.get("patient_id", "")),
        str(record.get("modality", "")),
        str(record.get("output_path", "")),
    )


def _write_normalization_metadata_csv(metadata_path, records, replace_existing=True):
    from pathlib import Path
    import csv

    if metadata_path is None or not records:
        return

    metadata_path = Path(metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = _normalization_metadata_fieldnames()
    rows = []
    if metadata_path.exists() and replace_existing:
        with metadata_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            replacement_keys = {_normalization_metadata_key(record) for record in records}
            rows = [
                row for row in reader
                if _normalization_metadata_key(row) not in replacement_keys
            ]

    for record in records:
        rows.append({field: record.get(field, "") for field in fieldnames})

    with metadata_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_training_normalization_metadata_record(
    *,
    patient_id,
    modality,
    method,
    input_image_path,
    mask_path,
    output_image_path,
    image,
    array,
    mask_array,
    normalized,
    lower_percentile,
    upper_percentile,
    clip_low,
    clip_high,
    zscore_mean,
    zscore_std,
    scale_output_min,
    scale_output_max,
    outside_value,
):
    import numpy as np

    inside_before = array[mask_array]
    inside_after = normalized[mask_array]
    size = tuple(image.GetSize())
    spacing = tuple(float(value) for value in image.GetSpacing())
    inside_voxel_count = int(np.count_nonzero(mask_array))
    total_voxel_count = int(mask_array.size)

    return {
        "patient_id": "" if patient_id is None else str(patient_id),
        "modality": str(modality),
        "method": str(method),
        "input_path": str(input_image_path),
        "mask_path": str(mask_path),
        "output_path": str(output_image_path),
        "lower_percentile": float(lower_percentile),
        "upper_percentile": float(upper_percentile),
        "clip_low_value": float(clip_low),
        "clip_high_value": float(clip_high),
        "zscore_mean": "" if zscore_mean is None else float(zscore_mean),
        "zscore_std": "" if zscore_std is None else float(zscore_std),
        "scale_output_min": "" if scale_output_min is None else float(scale_output_min),
        "scale_output_max": "" if scale_output_max is None else float(scale_output_max),
        "outside_value": float(outside_value),
        "inside_voxel_count": inside_voxel_count,
        "total_voxel_count": total_voxel_count,
        "mask_foreground_fraction": inside_voxel_count / max(total_voxel_count, 1),
        "inside_before_min": float(np.min(inside_before)),
        "inside_before_max": float(np.max(inside_before)),
        "inside_before_mean": float(np.mean(inside_before)),
        "inside_before_std": float(np.std(inside_before)),
        "inside_after_min": float(np.min(inside_after)),
        "inside_after_max": float(np.max(inside_after)),
        "inside_after_mean": float(np.mean(inside_after)),
        "inside_after_std": float(np.std(inside_after)),
        "size_x": int(size[0]),
        "size_y": int(size[1]),
        "size_z": int(size[2]),
        "spacing_x": float(spacing[0]),
        "spacing_y": float(spacing[1]),
        "spacing_z": float(spacing[2]),
    }


def normalize_modality_image_for_training(
    input_image_path,
    mask_path,
    output_image_path,
    modality,
    lower_percentile=1.0,
    upper_percentile=99.0,
    mra_output_range=(0.0, 1.0),
    outside_value=0.0,
    eps=1e-8,
    overwrite=False,
    patient_id=None,
    metadata_path=None,
    metadata_records=None,
    print_stats=True,
):
    """
    Normalize one image using the training rules for this project.

    T1/T2/PD:
        percentile clip using inside-mask values, z-score per subject using
        clipped inside-mask values, then set outside-mask voxels to 0.

    MRA:
        percentile clip using inside-mask values, scale to mra_output_range,
        then set outside-mask voxels to 0.

    If metadata_path or metadata_records is provided, the clip and scaling
    statistics are stored so normalized outputs can be audited or inverted
    later. Existing output images are left untouched when overwrite=False, but
    metadata is still computed from the input image and mask.
    """
    from pathlib import Path

    import numpy as np

    output_path = Path(output_image_path)
    output_exists = output_path.exists()
    has_metadata_target = metadata_path is not None or metadata_records is not None
    if output_exists and not overwrite and not has_metadata_target:
        if print_stats:
            print(f"Skipping existing normalized image: {output_path}")
        return

    modality = str(modality).upper()
    if modality not in {"T1", "T2", "PD", "MRA"}:
        raise ValueError("modality must be one of: T1, T2, PD, MRA")

    image, array, mask_array = _load_image_and_mask_arrays(input_image_path, mask_path)
    inside_values = array[mask_array]
    clip_low = float(np.percentile(inside_values, lower_percentile))
    clip_high = float(np.percentile(inside_values, upper_percentile))

    if clip_high <= clip_low:
        clipped = array.astype(np.float32, copy=True)
    else:
        clipped = np.clip(array, clip_low, clip_high).astype(np.float32, copy=False)

    if modality in {"T1", "T2", "PD"}:
        clipped_inside = clipped[mask_array]
        mean = float(np.mean(clipped_inside))
        std = float(np.std(clipped_inside))
        normalized = (clipped - mean) / max(std, eps)
        method = "clip+zscore"
        center_stat = mean
        scale_stat = std
        zscore_mean = mean
        zscore_std = std
        scale_output_min = None
        scale_output_max = None
    else:
        out_low, out_high = float(mra_output_range[0]), float(mra_output_range[1])
        if clip_high > clip_low:
            normalized = (clipped - clip_low) / (clip_high - clip_low)
            normalized = normalized * (out_high - out_low) + out_low
        else:
            normalized = np.zeros_like(clipped, dtype=np.float32) + out_low
        method = f"clip+scale{tuple(mra_output_range)}"
        center_stat = clip_low
        scale_stat = clip_high
        zscore_mean = None
        zscore_std = None
        scale_output_min = out_low
        scale_output_max = out_high

    normalized = normalized.astype(np.float32, copy=False)
    normalized[~mask_array] = float(outside_value)
    if not output_exists or overwrite:
        output = _copy_sitk_metadata(image, normalized)
        _write_sitk_image_safely(output, output_path)

    metadata_record = _build_training_normalization_metadata_record(
        patient_id=patient_id,
        modality=modality,
        method=method,
        input_image_path=input_image_path,
        mask_path=mask_path,
        output_image_path=output_path,
        image=image,
        array=array,
        mask_array=mask_array,
        normalized=normalized,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        clip_low=clip_low,
        clip_high=clip_high,
        zscore_mean=zscore_mean,
        zscore_std=zscore_std,
        scale_output_min=scale_output_min,
        scale_output_max=scale_output_max,
        outside_value=outside_value,
    )
    if metadata_records is not None:
        metadata_records.append(metadata_record)
    if metadata_path is not None:
        _write_normalization_metadata_csv(metadata_path, [metadata_record])

    if print_stats:
        inside_after = normalized[mask_array]
        print("Training normalization finished")
        print(f"Input: {input_image_path}")
        print(f"Mask: {mask_path}")
        print(f"Output: {output_path}")
        if output_exists and not overwrite:
            print("Output image already existed; image was left unchanged and metadata was updated.")
        print(f"Modality: {modality}")
        print(f"Method: {method}")
        print(f"Clip percentiles: {lower_percentile} / {upper_percentile}")
        print(f"Clip values: {clip_low:.6g} / {clip_high:.6g}")
        print(f"Center/scale stat: {center_stat:.6g} / {scale_stat:.6g}")
        print(f"Inside-mask after min/max: {float(np.min(inside_after)):.6g} / {float(np.max(inside_after)):.6g}")
        print(f"Outside-mask value: {outside_value}")
        if metadata_path is not None:
            print(f"Metadata file: {metadata_path}")


def normalize_modality_folder_for_training(
    input_folder,
    mask_folder,
    output_folder,
    modality,
    mask_modality=None,
    lower_percentile=1.0,
    upper_percentile=99.0,
    mra_output_range=(0.0, 1.0),
    outside_value=0.0,
    recursive=False,
    image_extensions=(".nii.gz", ".nii"),
    overwrite=False,
    max_images=None,
    metadata_path=None,
    preview_limit=10,
    print_stats=True,
):
    """
    Normalize one modality folder using project training rules.

    For T1/T2/PD this does clip-inside-mask + z-score per subject. For MRA
    this does clip-inside-mask + scaling to mra_output_range. In both cases,
    outside-mask voxels are set to outside_value after normalization.

    metadata_path writes one CSV row per processed subject/modality. Existing
    rows with the same patient, modality, and output path are replaced.
    """
    from pathlib import Path

    input_path = Path(input_folder)
    mask_path = Path(mask_folder)
    output_path = Path(output_folder)
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input folder does not exist: {input_path}")
    if not mask_path.is_dir():
        raise NotADirectoryError(f"Mask folder does not exist: {mask_path}")

    modality = str(modality).upper()
    images_by_patient, ignored_images = _collect_files_by_patient_stem(
        input_path,
        modality=modality,
        recursive=recursive,
        image_extensions=image_extensions,
    )

    if mask_modality is None:
        masks_by_patient, ignored_masks = _collect_files_by_patient_any_modality(
            mask_path,
            modalities=(modality, "T1", "T2", "PD", "MRA"),
            recursive=recursive,
            image_extensions=image_extensions,
        )
    else:
        masks_by_patient, ignored_masks = _collect_files_by_patient_stem(
            mask_path,
            modality=mask_modality,
            recursive=recursive,
            image_extensions=image_extensions,
        )

    pairs = []
    for patient, image_paths in images_by_patient.items():
        for image_path in image_paths:
            pairs.append((patient, image_path))
    pairs.sort(key=lambda item: str(item[1]))
    if max_images is not None:
        pairs = pairs[:max_images]

    processed = 0
    skipped = 0
    failed = []
    metadata_records = []
    for patient, image_path in pairs:
        mask_candidates = masks_by_patient.get(patient)
        if not mask_candidates:
            failed.append({"file": str(image_path), "error": f"No mask found for patient {patient}"})
            continue

        destination_path = output_path / image_path.relative_to(input_path)
        destination_exists = destination_path.exists()
        if destination_exists and not overwrite and metadata_path is None:
            skipped += 1
            continue

        try:
            normalize_modality_image_for_training(
                input_image_path=image_path,
                mask_path=mask_candidates[0],
                output_image_path=destination_path,
                modality=modality,
                lower_percentile=lower_percentile,
                upper_percentile=upper_percentile,
                mra_output_range=mra_output_range,
                outside_value=outside_value,
                overwrite=overwrite,
                patient_id=patient,
                metadata_records=metadata_records,
                print_stats=False,
            )
            if destination_exists and not overwrite:
                skipped += 1
            else:
                processed += 1
        except Exception as error:
            failed.append({"file": str(image_path), "error": repr(error)})

    _write_normalization_metadata_csv(metadata_path, metadata_records)

    if print_stats:
        print("Training normalization folder finished")
        print(f"Input folder: {input_path}")
        print(f"Mask folder: {mask_path}")
        print(f"Output folder: {output_path}")
        print(f"Modality: {modality}")
        print(f"Processed images: {processed}")
        print(f"Skipped existing images: {skipped}")
        print(f"Failed images: {len(failed)}")
        print(f"Ignored input files: {len(ignored_images)}")
        print(f"Ignored mask files: {len(ignored_masks)}")
        if metadata_path is not None:
            print(f"Metadata file: {metadata_path}")
            print(f"Metadata rows written: {len(metadata_records)}")
        if failed:
            print(f"Failed image preview: {failed[:preview_limit]}")


def normalize_training_modalities(
    image_folders,
    mask_folders,
    output_folders=None,
    output_root=None,
    modalities=("T1", "T2", "PD", "MRA"),
    lower_percentile=1.0,
    upper_percentile=99.0,
    mra_output_range=(0.0, 1.0),
    outside_value=0.0,
    recursive=False,
    image_extensions=(".nii.gz", ".nii"),
    overwrite=False,
    max_images=None,
    anatomical_metadata_path=None,
    mra_metadata_path=None,
    print_stats=True,
):
    """
    Normalize T1/T2/PD/MRA folders with the project-specific rules.

    image_folders and mask_folders are dictionaries keyed by modality. Use
    output_folders for explicit output paths, or output_root to write one
    subfolder per modality.

    anatomical_metadata_path stores T1/T2/PD rows together. mra_metadata_path
    stores MRA rows separately.
    """
    from pathlib import Path

    if output_folders is None and output_root is None:
        raise ValueError("Provide output_folders or output_root")

    for modality in modalities:
        modality = str(modality).upper()
        if output_folders is not None:
            output_folder = output_folders[modality]
        else:
            output_folder = Path(output_root) / modality

        if modality in {"T1", "T2", "PD"}:
            metadata_path = anatomical_metadata_path
        elif modality == "MRA":
            metadata_path = mra_metadata_path
        else:
            metadata_path = None

        normalize_modality_folder_for_training(
            input_folder=image_folders[modality],
            mask_folder=mask_folders[modality],
            output_folder=output_folder,
            modality=modality,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
            mra_output_range=mra_output_range,
            outside_value=outside_value,
            recursive=recursive,
            image_extensions=image_extensions,
            overwrite=overwrite,
            max_images=max_images,
            metadata_path=metadata_path,
            print_stats=print_stats,
        )
