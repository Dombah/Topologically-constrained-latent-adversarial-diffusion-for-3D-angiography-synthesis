"""MIP PSNR isporučenog MRA autoenkodera na cijelom validacijskom skupu.

Ponovna ocjena arma (ablation_full_val.json) dala je MIP SSIM, ali ne i MIP PSNR, a taj
broj stoji u tablici rezultata VAE modela. Ovdje se računa istim postupkom kao MIP SSIM:
projekcija najvećih intenziteta po svakoj od triju osi, PSNR uz raspon vrijednosti 1,0,
pa prosjek triju osi. MIP SSIM se računa usput kao provjera da postupak reproducira
0,9945 iz ponovne ocjene.

Rekonstrukcija ide izravnim prolazom kroz cijeli volumen, dakle u režimu u kojem
autoenkoder radi kada dekodira izlaz difuzijskog modela.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))   # so `modules` imports when run as uv run analysis/<script>.py
from modules.vae_mra import load_vaev4, reconstruct

DEV = "cuda"
CKPT = ROOT / "checkpoints/vaev5_mra/v5_cldice/best.pt"
OUT = ROOT / "runs" / "figures" / "mip_psnr_v5_cldice.json"

vae = load_vaev4(CKPT, DEV)
paths = sorted((ROOT / "Dataset/split_numpy/val/MRA").glob("*.npy"))
print("volumena: %d" % len(paths), flush=True)

ssims, psnrs = [], []
t0 = time.time()
for i, p in enumerate(paths, 1):
    truth = np.asarray(np.load(p), np.float32)
    with torch.inference_mode():
        rec, _ = reconstruct(vae, torch.from_numpy(truth)[None, None].to(DEV), "direct")
    rec = np.clip(rec, 0.0, 1.0)
    torch.cuda.empty_cache()
    s, q = [], []
    for a in range(3):
        mt, mr = truth.max(a), rec.max(a)
        s.append(structural_similarity(mt, mr, data_range=1.0))
        q.append(10.0 * math.log10(1.0 / max(float(np.mean((mt - mr) ** 2)), 1e-12)))
    ssims.append(float(np.mean(s)))
    psnrs.append(float(np.mean(q)))
    if i % 10 == 0 or i == len(paths):
        print("   %d/%d  [%.1f min]" % (i, len(paths), (time.time() - t0) / 60), flush=True)

OUT.write_text(json.dumps({"checkpoint": str(CKPT.relative_to(ROOT)), "split": "val",
                           "cases": len(paths), "regime": "direct",
                           "mip_ssim": float(np.mean(ssims)),
                           "mip_psnr": float(np.mean(psnrs))}, indent=2), encoding="utf-8")
print("\nMIP SSIM  %.4f   (ponovna ocjena: 0,9945)" % np.mean(ssims))
print("MIP PSNR  %.3f dB" % np.mean(psnrs))
print("zapisano ->", OUT, flush=True)
