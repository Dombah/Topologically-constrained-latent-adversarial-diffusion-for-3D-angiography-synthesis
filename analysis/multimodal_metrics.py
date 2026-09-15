"""SSIM i PSNR za višemodalni varijacijski autoenkoder, istim postupkom kao za MRA model.

Brojke koje su dosad bile u radu potjecale su iz vlastitog dnevnika treniranja i računate
su drukčije: bez kliznog prozora, samo globalnim momentima po volumenu, uz pretpostavljeni
raspon vrijednosti 1,0. T1, T2 i PD volumeni su z-normalizirani, pa im stvarni raspon nije
1,0 nego oko pet do sedam, zbog čega je i PSNR bio sustavno podcijenjen.

Ovdje se koristi skimage, uniformni klizni prozor 7 x 7 x 7, i izvještavaju se dvije
vrijednosti kao i za MRA model: globalni SSIM preko cijelog volumena i maskirani SSIM
unutar maske mozga, uz odrezivanje ruba od tri voksela.

PSNR se računa dvojako, da se vidi odakle razlika prema ranijim brojkama:
  - uz stvarni raspon referentnog volumena, što je ispravno za z-normalizirane podatke
  - uz raspon 1,0, kako je bilo prije

Rekonstrukcija ide determinističkim putem preko srednje vrijednosti razdiobe, dakle isto
kako su izvezeni latentni prikazi za difuzijski model.
"""
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))   # so `modules` imports when run as uv run analysis/<script>.py
from modules.vae_multimodal import (VAEv2Multimodal, _make_single_modality_mask, _volume_tensor,
                                    _encode_deterministic)

DEV = "cuda"
MODS = ("T1", "T2", "PD")
CKPT = ROOT / "checkpoints/vaev2_multimodal/kl5e5_from_epoch10/best.pt"
OUT = ROOT / "runs" / "multimodal_metrics"
OUT.mkdir(parents=True, exist_ok=True)
INNER = (slice(3, -3),) * 3

model = VAEv2Multimodal(patch_size=(192, 192, 64), modalities=MODS, in_channels=1,
                        base_channels=16, channel_multipliers=(1, 2, 4),
                        latent_channels=8, blocks_per_level=1,
                        output_activation=None).to(DEV)
blob = torch.load(CKPT, map_location="cpu", weights_only=False)
model.load_state_dict(blob["model_state_dict"])
model.eval()
print("učitano: epoha %s, zabilježeno mjerilo %.4f"
      % (blob.get("epoch"), blob.get("best_metric", float("nan"))), flush=True)
del blob

cases = sorted({p.stem.rsplit("-", 1)[0]
                for p in (ROOT / "Dataset/split_numpy/val/T1").glob("*.npy")})
print("volumena: %d  modaliteta: %d" % (len(cases), len(MODS)), flush=True)

rows = defaultdict(list)
t0 = time.time()
for i, case in enumerate(cases, 1):
    for idx, mod in enumerate(MODS):
        p = ROOT / ("Dataset/split_numpy/val/" + mod) / ("%s-%s.npy" % (case, mod))
        mp = ROOT / ("Dataset/split_numpy/val/masks/" + mod) / ("%s-%s_mask.npy" % (case, mod))
        target = np.asarray(np.load(p), np.float32)
        brain = np.asarray(np.load(mp)) > 0

        mask = _make_single_modality_mask(idx, len(MODS), DEV)
        x = _volume_tensor(target, DEV)
        spatial = tuple(int(v) for v in x.shape[-3:])
        with torch.inference_mode():
            x_work, _ = model._pad_spatial_to_factor(x)
            mu, _ = _encode_deterministic(model, x_work, mask)
            recon = model.decode(mu, modality_mask=mask)
            recon = model._crop_recon_to_spatial(recon, spatial)
        recon = recon[0, 0].float().cpu().numpy()
        del x, mu
        torch.cuda.empty_cache()

        rng = float(target.max() - target.min())
        g, smap = structural_similarity(target, recon, data_range=rng, full=True)
        sel = brain[INNER]
        m = float(smap[INNER][sel].mean()) if sel.any() else float("nan")
        mse = float(np.mean((target - recon) ** 2))
        mse_brain = float(np.mean((target[brain] - recon[brain]) ** 2)) if brain.any() else mse
        rows[mod].append({
            "ssim_global": float(g), "ssim_masked": m,
            "psnr": 10.0 * math.log10(rng ** 2 / max(mse, 1e-12)),
            "psnr_masked": 10.0 * math.log10(rng ** 2 / max(mse_brain, 1e-12)),
            "psnr_range1": 10.0 * math.log10(1.0 / max(mse, 1e-12)),
            "data_range": rng})
    if i % 10 == 0 or i == len(cases):
        print("   %d/%d  [%.1f min]" % (i, len(cases), (time.time() - t0) / 60), flush=True)

summary = {mod: {k: float(np.mean([r[k] for r in v])) for k in v[0]}
           for mod, v in rows.items()}
(OUT / "multimodal_val.json").write_text(json.dumps(
    {"checkpoint": str(CKPT.relative_to(ROOT)), "split": "val", "cases": len(cases),
     "summary": summary}, indent=2), encoding="utf-8")

print("\n%-6s %11s %11s %9s %11s %12s" % (
    "mod", "SSIM glob", "SSIM mask", "PSNR", "PSNR mask", "PSNR (R=1)"))
for mod in MODS:
    r = summary[mod]
    print("%-6s %11.4f %11.4f %9.2f %11.2f %12.2f" % (
        mod, r["ssim_global"], r["ssim_masked"], r["psnr"], r["psnr_masked"],
        r["psnr_range1"]))
print("\nprosječni raspon vrijednosti po modalitetu: " + ", ".join(
    "%s %.2f" % (m, summary[m]["data_range"]) for m in MODS))
print("zapisano ->", OUT / "multimodal_val.json", flush=True)
