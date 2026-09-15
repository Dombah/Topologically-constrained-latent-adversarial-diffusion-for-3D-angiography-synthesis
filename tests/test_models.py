"""Models, losses and metrics used by the thesis pipeline, on small synthetic tensors (CPU).

Run:  uv run pytest tests -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.bridge import BridgeProjector, BrownianBridge  # noqa: E402
from modules.losses import frangi_weighted_l1  # noqa: E402
from modules.metrics import betti, vessel_scores  # noqa: E402
from modules.paths import PROTECTED, ROOT as PATHS_ROOT, refuse_protected  # noqa: E402
from modules.topology import (soft_cldice_loss, soft_dice_loss,  # noqa: E402
                              vessel_probability_banded)
from modules.unet import build_model, sinusoidal_embedding  # noqa: E402
from modules.vae_mra import VAEv4, decode_tiled, encode_tiled  # noqa: E402
from modules.vae_multimodal import (_encode_deterministic, build_multimodal_vae,  # noqa: E402
                                    load_vaev2_checkpoint)

torch.manual_seed(0)


def tiny_bridge_config() -> dict:
    config = json.loads((ROOT / "configs" / "bbdm.json").read_text(encoding="utf-8"))
    config["model"].update({"base_channels": 8, "channel_multipliers": [1, 2], "blocks_per_level": 1,
                            "attention_levels": [1], "time_embed_dim": 16, "cond_base_channels": 8})
    return config


class TestPaths:
    def test_root_is_repository(self):
        assert PATHS_ROOT == ROOT

    @pytest.mark.parametrize("path", PROTECTED)
    def test_thesis_artefacts_are_refused(self, path):
        with pytest.raises(SystemExit):
            refuse_protected(path / "sub" / "best.pt")

    def test_rebuild_folder_is_allowed(self):
        assert refuse_protected(ROOT / "checkpoints" / "rebuild" / "bbdm").name == "bbdm"

    def test_sibling_with_shared_prefix_is_allowed(self):
        refuse_protected(ROOT / "checkpoints" / "ldm_bbdm_perc")


class TestUNetAndBridge:
    def test_sinusoidal_embedding_shape(self):
        assert sinusoidal_embedding(torch.tensor([0, 10, 999]), 16).shape == (3, 16)

    def test_unet_forward_shape(self):
        model = build_model(tiny_bridge_config(), target_channels=4, source_channels=6, device="cpu")
        x = torch.randn(1, 4, 8, 8, 8)
        out = model(x, torch.tensor([10]), torch.randn(1, 6, 8, 8, 8))
        assert out.shape == x.shape

    def test_bridge_endpoints(self):
        bridge = BrownianBridge(steps=100, max_variance=1.0, device="cpu")
        assert float(bridge.m[0]) == pytest.approx(0.0, abs=1e-6)
        assert float(bridge.m[-1]) == pytest.approx(1.0, abs=1e-6)

    def test_add_noise_and_to_x0_are_consistent(self):
        bridge = BrownianBridge(steps=100, max_variance=1.0, device="cpu")
        x0, y = torch.randn(2, 4, 6, 6, 6), torch.randn(2, 4, 6, 6, 6)
        t = torch.tensor([10, 70])
        x_t, target = bridge.add_noise(x0, y, t, torch.randn_like(x0))
        assert torch.allclose(bridge.to_x0(x_t, target), x0, atol=1e-5)

    def test_projector_maps_channels(self):
        projector = BridgeProjector(6, 4, base=8, blocks=1)
        assert projector(torch.randn(1, 6, 8, 8, 8)).shape == (1, 4, 8, 8, 8)


class TestTopologyAndLosses:
    def test_cldice_and_dice_are_zero_for_identical_masks(self):
        mask = torch.zeros(1, 1, 16, 16, 16)
        mask[..., 8, 8, 2:14] = 1.0
        assert float(soft_cldice_loss(mask, mask)) == pytest.approx(0.0, abs=0.05)
        assert float(soft_dice_loss(mask, mask)) == pytest.approx(0.0, abs=0.05)

    def test_dice_penalises_thickening(self):
        thin = torch.zeros(1, 1, 16, 16, 16)
        thin[..., 7:9, 7:9, 2:14] = 1.0
        thick = torch.zeros_like(thin)
        thick[..., 4:12, 4:12, 2:14] = 1.0
        assert float(soft_dice_loss(thick, thin)) > float(soft_dice_loss(thin, thin)) + 0.3

    def test_banded_probability_is_a_ramp(self):
        volume = torch.linspace(0, 1, 11).view(1, 1, 11, 1, 1)
        p = vessel_probability_banded(volume, torch.tensor([[0.2, 0.6]]))
        assert float(p.min()) == 0.0 and float(p.max()) == 1.0
        assert float(p[0, 0, 4, 0, 0]) == pytest.approx(0.5, abs=1e-5)

    def test_frangi_l1_reduces_to_l1_without_vessels(self):
        a, b = torch.rand(1, 1, 4, 4, 4), torch.rand(1, 1, 4, 4, 4)
        assert float(frangi_weighted_l1(a, b, torch.zeros_like(a), 8.0)) == pytest.approx(
            float((a - b).abs().mean()), rel=1e-6)


class TestMetrics:
    def test_betti_of_ring_and_blob(self):
        ring = np.zeros((40, 40, 9), bool)
        yy, xx = np.mgrid[:40, :40]
        radius = np.hypot(yy - 20, xx - 20)
        ring[(radius > 10) & (radius < 14), 3:6] = True
        b0, b1, _ = betti(ring)
        assert (b0, b1) == (1, 1)
        blob = np.zeros((20, 20, 20), bool)
        blob[5:15, 5:15, 5:15] = True
        assert betti(blob)[:2] == (1, 0)

    def test_vessel_dice_is_one_for_identical_volumes(self):
        vol = np.random.default_rng(0).random((32, 32, 32)).astype(np.float32)
        assert vessel_scores(vol, vol)[0] == pytest.approx(1.0)


class TestVAEs:
    def test_mra_vae_tiled_round_trip_shapes(self):
        model = VAEv4(latent_channels=16).eval()
        volume = torch.rand(1, 1, 96, 96, 48)
        with torch.no_grad():
            latent = encode_tiled(model, volume)
            assert latent.shape == (1, 16, 24, 24, 12)
            assert decode_tiled(model, latent).shape == volume.shape

    def test_multimodal_vae_matches_thesis_architecture(self):
        config = json.loads((ROOT / "configs" / "vae_multimodal.json").read_text(encoding="utf-8"))
        model = build_multimodal_vae(config)
        assert model.modalities == ("T1", "T2", "PD")
        assert sum(p.numel() for p in model.parameters()) == pytest.approx(3.538e6, rel=1e-3)

    def test_multimodal_encoding_is_deterministic(self):
        config = json.loads((ROOT / "configs" / "vae_multimodal.json").read_text(encoding="utf-8"))
        model = build_multimodal_vae(config).eval()
        x = torch.rand(1, 1, 32, 32, 16)
        mask = torch.tensor([[1.0, 0.0, 0.0]])
        with torch.no_grad():
            a, _ = _encode_deterministic(model, x, mask)
            b, _ = _encode_deterministic(model, x, mask)
        assert torch.equal(a, b)

    def test_load_vaev2_checkpoint_round_trip(self, tmp_path):
        config = json.loads((ROOT / "configs" / "vae_multimodal.json").read_text(encoding="utf-8"))
        model, fresh = build_multimodal_vae(config), build_multimodal_vae(config)
        torch.save({"model_state_dict": model.state_dict(), "epoch": 3}, tmp_path / "m.pt")
        blob = load_vaev2_checkpoint(tmp_path / "m.pt", fresh)
        assert blob["epoch"] == 3
        for (k, v), w in zip(model.state_dict().items(), fresh.state_dict().values()):
            assert torch.equal(v, w), k


THESIS = {
    "multimodal": ROOT / "checkpoints/vaev2_multimodal/kl5e5_from_epoch10/best.pt",
    "vae_v5": ROOT / "checkpoints/vaev5_mra/v5_cldice/best.pt",
    "bridge": ROOT / "checkpoints/ldm_bbdm/best.pt",
    "delivered": ROOT / "checkpoints/ldm_bbdm_vessel/ft_cldice/best.pt",
}


@pytest.mark.real_data
@pytest.mark.skipif(not all(p.exists() for p in THESIS.values()), reason="thesis checkpoints absent")
class TestThesisCheckpointsLoad:
    def test_multimodal_vae_loads_strict(self):
        config = json.loads((ROOT / "configs" / "vae_multimodal.json").read_text(encoding="utf-8"))
        model = build_multimodal_vae(config)
        model.load_state_dict(torch.load(THESIS["multimodal"], map_location="cpu",
                                         weights_only=False)["model_state_dict"])

    def test_mra_vae_loads(self):
        from modules.vae_mra import load_vaev4
        load_vaev4(THESIS["vae_v5"], "cpu")

    @pytest.mark.parametrize("name", ["bridge", "delivered"])
    def test_bridge_checkpoints_load_strict(self, name):
        blob = torch.load(THESIS[name], map_location="cpu", weights_only=False)
        config = blob["config"]
        model = build_model(config, target_channels=16, source_channels=24, device="cpu")
        model.load_state_dict(blob["ema"])
        projector = BridgeProjector(24, 16, base=int(config["model"]["projector_base"]),
                                    blocks=int(config["model"]["projector_blocks"]))
        projector.load_state_dict(blob["ema_projector"])
