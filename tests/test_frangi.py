"""Frangi vessel-weight generation (modules/frangi.py): helpers, response on a synthetic
tube, degenerate inputs and the end-to-end cache writer.

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

from modules import frangi as fv  # noqa: E402

DATASET_ROOT = ROOT / "Dataset/split_numpy"


class TestLatentFileHelpers:
    @pytest.mark.parametrize("stem,modality,expected", [
        ("IXI016-Guys-0697-MRA", "MRA", "IXI016-Guys-0697"),
        ("IXI016-Guys-0697-T1", "T1", "IXI016-Guys-0697"),
    ])
    def test_latent_patient_id(self, stem, modality, expected):
        assert fv.latent_patient_id(Path(f"{stem}.npy"), modality) == expected

    def test_collect_latent_files_missing_dir_returns_empty(self, tmp_path):
        assert fv.collect_latent_files(tmp_path, "val", "MRA") == {}


class TestFrangiHelpers:
    @pytest.mark.parametrize("stem,expected", [
        ("IXI016-Guys-0697-MRA", "IXI016-Guys-0697"),
        ("IXI016-Guys-0697-MRA_mask", "IXI016-Guys-0697"),
        ("IXI016-Guys-0697-MRA-mask", "IXI016-Guys-0697"),
        ("IXI016-Guys-0697", "IXI016-Guys-0697"),
    ])
    def test_mra_patient_id(self, stem, expected):
        assert fv._mra_patient_id(f"{stem}.npy") == expected

    def test_as_volume3d_accepts_both_ranks(self):
        assert fv._as_volume3d(np.zeros((4, 4, 4)), name="x").shape == (4, 4, 4)
        assert fv._as_volume3d(np.zeros((1, 4, 4, 4)), name="x").shape == (4, 4, 4)

    def test_as_volume3d_rejects_multichannel(self):
        with pytest.raises(ValueError, match=r"\[D, H, W\]"):
            fv._as_volume3d(np.zeros((2, 4, 4, 4)), name="img")

    def test_as_volume3d_scrubs_nonfinite(self):
        src = np.array([[[np.nan, np.inf], [-np.inf, 2.0]]], np.float32)
        out = fv._as_volume3d(src, name="x")
        assert np.isfinite(out).all()
        assert out.tolist() == [[[0.0, 0.0], [0.0, 2.0]]]

    def test_resize_volume3d_modes(self):
        src = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
        assert fv._resize_volume3d(src, (2, 2, 2), mode="nearest").shape == (2, 2, 2)
        assert fv._resize_volume3d(src, (8, 8, 8), mode="trilinear").shape == (8, 8, 8)

    @pytest.mark.parametrize("value,expected", [
        (2, (2, 2, 2)),
        ([1, 2, 3], (1, 2, 3)),
        ((4, 4, 1), (4, 4, 1)),
    ])
    def test_as_downsample_tuple(self, value, expected):
        assert fv._as_downsample_tuple(value) == expected

    def test_as_downsample_tuple_rejects_bad_length(self):
        with pytest.raises(ValueError, match="length-3 sequence"):
            fv._as_downsample_tuple([2, 2])

    def test_as_downsample_tuple_rejects_zero_or_negative(self):
        with pytest.raises(ValueError, match=">= 1"):
            fv._as_downsample_tuple([1, 0, 2])

    @pytest.mark.parametrize("shape,factors,expected", [
        ((8, 8, 8), (2, 2, 2), (4, 4, 4)),
        ((9, 9, 9), (2, 2, 2), (4, 4, 4)),      # round-half-even
        ((3, 3, 3), (10, 10, 10), (1, 1, 1)),   # never collapses below 1
        ((8, 8, 8), (1, 1, 1), (8, 8, 8)),
    ])
    def test_downsample_shape(self, shape, factors, expected):
        assert fv._downsample_shape(shape, factors) == expected

    def test_masked_values_selects_mask_region(self):
        arr = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
        mask = np.zeros((2, 2, 2), np.float32)
        mask[0, 0, 0] = 1.0
        mask[1, 1, 1] = 1.0
        assert sorted(fv._masked_values(arr, mask).tolist()) == [0.0, 7.0]

    def test_masked_values_ignores_empty_mask(self):
        arr = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
        assert fv._masked_values(arr, np.zeros((2, 2, 2), np.float32)).size == 8

    def test_masked_values_positive_only_and_finite(self):
        arr = np.array([-1.0, 0.0, 2.0, np.nan, np.inf], np.float32)
        assert fv._masked_values(arr, None, positive_only=True).tolist() == [2.0]

    def test_suppress_weak_responses_subtracts_floor(self):
        w = np.array([0.0, 1.0, 2.0, 3.0, 4.0], np.float32)
        out = fv._suppress_weak_responses(w, None, floor_percentile=50.0)
        assert out.min() == 0.0
        assert out.max() < w.max()
        assert (out >= 0).all()

    def test_suppress_weak_responses_disabled(self):
        w = np.array([0.0, 1.0, 2.0], np.float32)
        assert np.array_equal(fv._suppress_weak_responses(w, None, floor_percentile=None), w)

    def test_suppress_weak_responses_clips_negatives(self):
        out = fv._suppress_weak_responses(np.array([-5.0, 1.0], np.float32), None, floor_percentile=None)
        assert out.min() == 0.0

    def test_normalize_weight_map_scales_to_unit_range(self):
        w = np.linspace(0, 10, 100, dtype=np.float32)
        out = fv._normalize_weight_map(w, None, robust_percentile=99.0)
        assert out.min() >= 0.0 and out.max() <= 1.0
        assert out.max() == pytest.approx(1.0, abs=1e-5)

    def test_normalize_weight_map_gamma_darkens(self):
        w = np.linspace(0, 10, 100, dtype=np.float32)
        base = fv._normalize_weight_map(w, None, robust_percentile=99.0, gamma=1.0)
        gamma2 = fv._normalize_weight_map(w, None, robust_percentile=99.0, gamma=2.0)
        assert (gamma2 <= base + 1e-6).all() and gamma2.sum() < base.sum()

    def test_normalize_weight_map_all_zero_input(self):
        out = fv._normalize_weight_map(np.zeros((4, 4, 4), np.float32), None, robust_percentile=99.0)
        assert out.shape == (4, 4, 4) and out.max() == 0.0

    def test_postprocess_applies_min_value_threshold(self):
        w = np.array([0.001, 0.05, 0.5, 1.0], np.float32)
        out = fv._postprocess_weight_map(w, None, min_value=0.02, renormalize=False, robust_percentile=99.0)
        assert out[0] == 0.0 and out[1] == pytest.approx(0.05)

    def test_postprocess_renormalize_rescales_to_one(self):
        w = np.array([0.0, 0.1, 0.2, 0.3], np.float32)
        out = fv._postprocess_weight_map(w, None, min_value=None, renormalize=True, robust_percentile=100.0)
        assert out.max() == pytest.approx(1.0, abs=1e-5)

    def test_postprocess_respects_mask(self):
        w = np.ones((2, 2, 2), np.float32)
        mask = np.zeros((2, 2, 2), np.float32)
        mask[0] = 1.0
        out = fv._postprocess_weight_map(w, mask, min_value=None, renormalize=False, robust_percentile=99.0)
        assert out[1].max() == 0.0 and out[0].min() == 1.0


def make_tube(shape=(48, 48, 48), radius=2.0, seed=0):
    """A bright vessel-like cylinder along z, Gaussian cross-section, on noisy soft tissue.

    Deliberately continuous-valued: a *binary* phantom collapses to all zeros inside
    `_normalize_image_for_frangi` (see TestFrangiDegenerateInput below).
    """
    rng = np.random.default_rng(seed)
    vol = rng.uniform(0.05, 0.25, shape).astype(np.float32)
    cy, cx = shape[0] / 2, shape[1] / 2
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    profile = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * radius ** 2)).astype(np.float32)
    vol = np.clip(vol + profile[:, :, None], 0.0, None)
    tube = np.broadcast_to(profile[:, :, None] > 0.5, shape)
    return vol, tube


class TestFrangiResponse:
    def test_responds_on_tube_not_background(self):
        pytest.importorskip("skimage")
        vol, tube = make_tube()
        w = fv._frangi_weight(vol, None, sigmas=(1.0, 2.0), black_ridges=False,
                              robust_percentile=99.5, floor_percentile=None, gamma=1.0)
        assert w.shape == vol.shape
        assert w.min() >= 0.0 and w.max() <= 1.0
        assert w[tube].mean() > 10 * w[~tube].mean(), "Frangi response is not concentrated on the vessel"

    def test_black_ridges_flag_inverts_preference(self):
        pytest.importorskip("skimage")
        vol, tube = make_tube()
        kw = dict(sigmas=(1.0, 2.0), robust_percentile=99.5, floor_percentile=None, gamma=1.0)
        bright = fv._frangi_weight(vol, None, black_ridges=False, **kw)
        dark = fv._frangi_weight(vol, None, black_ridges=True, **kw)
        assert bright[tube].mean() > dark[tube].mean()

    def test_mask_zeroes_response_outside(self):
        pytest.importorskip("skimage")
        vol, tube = make_tube()
        mask = np.zeros(vol.shape, np.float32)
        mask[:24] = 1.0
        w = fv._frangi_weight(vol, mask, sigmas=(1.0, 2.0), black_ridges=False,
                              robust_percentile=99.5, floor_percentile=None, gamma=1.0)
        assert w[24:].max() == 0.0, "response must be confined to the mask"


class TestFrangiDegenerateInput:
    """A constant-valued volume yields an all-zero weight map, with no error or warning.

    Not reachable from real MRA (continuous intensities), but it is the same silent-zero
    failure mode that makes every Frangi-weighted loss collapse to a plain unweighted loss.
    Documented here so a future change to the guard is a deliberate one.
    """

    def test_binary_volume_normalizes_to_zero(self):
        vol = np.zeros((16, 16, 16), np.float32)
        vol[6:10, 6:10, :] = 1.0
        assert fv._normalize_image_for_frangi(vol, None).max() == 0.0

    def test_constant_volume_gives_zero_weights_silently(self):
        pytest.importorskip("skimage")
        w = fv._frangi_weight(np.ones((16, 16, 16), np.float32), None, sigmas=(1.0,),
                              black_ridges=False, robust_percentile=99.5,
                              floor_percentile=None, gamma=1.0)
        assert w.max() == 0.0


class TestFrangiCacheEndToEnd:
    def test_generates_latent_resolution_weights(self, tmp_path, capsys):
        pytest.importorskip("skimage")
        split = "val"
        img_dir = tmp_path / "dataset" / split / "MRA"
        mask_dir = tmp_path / "dataset" / split / "masks" / "MRA"
        lat_dir = tmp_path / "latents" / "MRA" / split
        for d in (img_dir, mask_dir, lat_dir):
            d.mkdir(parents=True)

        vol, _ = make_tube(shape=(32, 32, 32))
        for pid in ("caseA", "caseB"):
            np.save(img_dir / f"{pid}-MRA.npy", vol)
            np.save(mask_dir / f"{pid}-MRA_mask.npy", np.ones((32, 32, 32), np.uint8))
            np.save(lat_dir / f"{pid}-MRA.npy", np.zeros((16, 8, 8, 8), np.float16))

        summary = fv.generate_frangi_vessel_weight_cache(
            dataset_root=tmp_path / "dataset",
            latent_root=tmp_path / "latents",
            output_root=tmp_path / "out",
            splits=(split,),
            sigmas=(1.0, 2.0),
            frangi_downsample=1,
            floor_percentile=None,
            show_progress=False,
        )
        capsys.readouterr()

        written = sorted((tmp_path / "out" / split).glob("*.npy"))
        assert len(written) == 2, summary
        for p in written:
            w = np.load(p)
            assert w.shape == (1, 8, 8, 8), "weights must be saved at latent resolution"
            assert w.dtype == np.float16, "cache is stored half-precision on purpose"
            assert np.isfinite(w).all()
            assert w.min() >= 0.0 and w.max() <= 1.0
            assert w.max() > 0.0, "all-zero vessel weights would silently disable the Frangi losses"

    def test_raises_when_images_and_latents_do_not_pair(self, tmp_path):
        (tmp_path / "dataset" / "val" / "MRA").mkdir(parents=True)
        (tmp_path / "latents" / "MRA" / "val").mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="No paired MRA images and latents"):
            fv.generate_frangi_vessel_weight_cache(
                dataset_root=tmp_path / "dataset",
                latent_root=tmp_path / "latents",
                output_root=tmp_path / "out",
                splits=("val",),
                show_progress=False,
            )


@pytest.mark.skipif(not (DATASET_ROOT / "val" / "MRA").exists(), reason="dataset absent")
class TestFrangiRealTree:
    def test_image_and_mask_discovery_align(self):
        imgs = fv._collect_mra_image_files(DATASET_ROOT, "val")
        masks = fv._collect_mra_mask_files(DATASET_ROOT, "val")
        assert len(imgs) == 57 and len(masks) == 57
        assert set(imgs) == set(masks), "image and mask patient ids must match"

    def test_missing_split_returns_empty(self):
        assert fv._collect_mra_image_files(DATASET_ROOT, "nonexistent") == {}
