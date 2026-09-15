"""Preprocessing and volume I/O helpers (modules/preprocessing.py, modules/vae_multimodal.py).

Run:  uv run pytest tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sitk = pytest.importorskip("SimpleITK")

from modules import preprocessing as pre  # noqa: E402
from modules import vae_multimodal as mm  # noqa: E402


def write_nii(path: Path, array: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> Path:
    """Write an [x, y, z]-ordered numpy array as NIfTI (SimpleITK reads back as [z, y, x])."""
    img = sitk.GetImageFromArray(np.ascontiguousarray(array.transpose(2, 1, 0)))
    img.SetSpacing(tuple(float(s) for s in spacing))
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(path))
    return path


def read_nii(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).transpose(2, 1, 0)


# =============================================================================
# modules/VAEv2.py - reachable surface
# =============================================================================
class TestVolumeToNumpy:
    def test_reads_npy(self, tmp_path):
        arr = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
        np.save(tmp_path / "v.npy", arr)
        assert np.array_equal(mm._volume_to_numpy(tmp_path / "v.npy"), arr)

    def test_reads_npz_first_key(self, tmp_path):
        arr = np.ones((2, 2, 2), np.float32)
        np.savez(tmp_path / "v.npz", array=arr)
        assert np.array_equal(mm._volume_to_numpy(tmp_path / "v.npz"), arr)

    def test_reads_nifti(self, tmp_path):
        arr = np.random.rand(4, 5, 6).astype(np.float32)
        write_nii(tmp_path / "v.nii.gz", arr)
        got = mm._volume_to_numpy(tmp_path / "v.nii.gz")
        assert got.shape == arr.shape
        assert np.allclose(got, arr, atol=1e-5)

    def test_casts_to_requested_dtype(self, tmp_path):
        np.save(tmp_path / "v.npy", np.ones((2, 2, 2), np.float64))
        assert mm._volume_to_numpy(tmp_path / "v.npy", dtype=np.float32).dtype == np.float32


class TestCollectModalityFiles:
    def test_groups_images_and_masks_by_patient(self, tmp_path):
        for name in ("sub-01-T1.nii.gz", "sub-01-T1_mask.nii.gz",
                     "sub-02-T1.nii.gz", "sub-02-T2.nii.gz"):
            (tmp_path / name).write_bytes(b"")
        found = mm._collect_modality_files(tmp_path, "T1")
        assert set(found) == {"sub-01", "sub-02"}

    def test_ignores_other_modalities(self, tmp_path):
        (tmp_path / "sub-01-T2.nii.gz").write_bytes(b"")
        assert mm._collect_modality_files(tmp_path, "T1") == {}

    def test_npy_requires_explicit_extension(self, tmp_path):
        """Default image_extensions is NIfTI-only; VAEv3 passes .npy/.npz explicitly."""
        (tmp_path / "sub-01-MRA.npy").write_bytes(b"")
        assert mm._collect_modality_files(tmp_path, "MRA") == {}
        found = mm._collect_modality_files(
            tmp_path, "MRA", image_extensions=(".npy", ".npz", ".nii.gz", ".nii"))
        assert set(found) == {"sub-01"}


class TestCropOrPadToSize:
    def test_pads_up_to_target(self, tmp_path, capsys):
        src = write_nii(tmp_path / "in.nii.gz", np.ones((8, 8, 8), np.float32))
        out = tmp_path / "out.nii.gz"
        pre.crop_or_pad_to_size(src, out, target_size=(16, 16, 16), print_stats=False)
        capsys.readouterr()
        assert read_nii(out).shape == (16, 16, 16)

    def test_crops_down_to_target(self, tmp_path, capsys):
        src = write_nii(tmp_path / "in.nii.gz", np.ones((16, 16, 16), np.float32))
        out = tmp_path / "out.nii.gz"
        pre.crop_or_pad_to_size(src, out, target_size=(8, 8, 8), print_stats=False)
        capsys.readouterr()
        assert read_nii(out).shape == (8, 8, 8)

    def test_preserves_spacing(self, tmp_path, capsys):
        src = write_nii(tmp_path / "in.nii.gz", np.ones((8, 8, 8), np.float32), spacing=(0.5, 0.5, 2.0))
        out = tmp_path / "out.nii.gz"
        pre.crop_or_pad_to_size(src, out, target_size=(16, 16, 16), print_stats=False)
        capsys.readouterr()
        assert sitk.ReadImage(str(out)).GetSpacing() == pytest.approx((0.5, 0.5, 2.0))

    def test_pad_value_is_applied(self, tmp_path, capsys):
        src = write_nii(tmp_path / "in.nii.gz", np.ones((8, 8, 8), np.float32))
        out = tmp_path / "out.nii.gz"
        pre.crop_or_pad_to_size(src, out, target_size=(16, 16, 16), pad_value=0.0, print_stats=False)
        capsys.readouterr()
        arr = read_nii(out)
        assert arr.min() == 0.0 and arr.max() == 1.0
        assert int((arr > 0.5).sum()) == 8 ** 3, "original voxel count must be preserved"

    def test_identity_when_already_target_size(self, tmp_path, capsys):
        arr = np.random.rand(8, 8, 8).astype(np.float32)
        src = write_nii(tmp_path / "in.nii.gz", arr)
        out = tmp_path / "out.nii.gz"
        pre.crop_or_pad_to_size(src, out, target_size=(8, 8, 8), print_stats=False)
        capsys.readouterr()
        assert np.allclose(read_nii(out), arr, atol=1e-5)


