"""Mjere MRA autoenkodera samo na IOP ispitanicima iz validacijskog skupa.

Ponovna ocjena na svih 57 volumena daje prosjek u kojem Guys nosi 32 volumena, HH 18, a
IOP samo 7, pa IOP, koji je vidljivo najslabiji, gotovo nestane u prosjeku. Ovdje se isti
postupak vrti samo na tih sedam volumena, za sve četiri grane, da se vidi koliko
autoenkoder gubi na tom uređaju.

Postupak je doslovno isti kao u ponovnoj ocjeni: izravna rekonstrukcija cijelog volumena,
skimage uz jednoliki prozor 7 x 7 x 7 i raspon 1,0, maskirani SSIM unutar maske mozga uz
odrezan rub od tri voksela, Dice i clDice na pragu 99. percentila cijelog volumena, te
Dice nakon dodavanja Gaussova šuma standardne devijacije 0,1 u latentni prikaz, mjerene u
jedinicama standardne devijacije kanala.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))   # so `modules` imports when run as uv run analysis/<script>.py
from modules.metrics import masked_ssim_and_psnr, mip_ssim, vessel_scores, hard_cldice
from modules.paths import mask_path_for
from modules.vae_mra import load_vaev4, encode_tiled, decode_tiled, reconstruct

DEV = "cuda"
SIGMA = 0.1
SITE = "IOP"
OUT = ROOT / "runs" / "vae_site" / ("vae_%s.json" % SITE.lower())
OUT.parent.mkdir(parents=True, exist_ok=True)

ARMS = [
    ("osnovni model", ROOT / "checkpoints/vaev4_mra/v4_c16_fixed/best.pt"),
    ("+ topolosko", ROOT / "checkpoints/vaev5_mra/v5_cldice/best.pt"),
    ("+ suparnicko", ROOT / "checkpoints/vaev5_mra/v5_adv/best.pt"),
    ("+ oboje", ROOT / "checkpoints/vaev5_mra/v5_both/best.pt"),
]

paths = [p for p in sorted((ROOT / "Dataset/split_numpy/val/MRA").glob("*.npy"))
         if p.stem.split("-")[1] == SITE]
print("volumena (%s): %d" % (SITE, len(paths)), flush=True)

volumes = []
for p in paths:
    truth = np.asarray(np.load(p), np.float32)
    volumes.append((p.stem.rsplit("-", 1)[0], truth,
                    np.asarray(np.load(mask_path_for(p, "val"))) > 0))

results, t0 = {}, time.time()
print("\n%-16s %10s %10s %8s %9s %9s %8s %9s %11s" % (
    "varijanta", "SSIM glob", "SSIM mask", "PSNR", "MIP SSIM", "MIP PSNR", "Dice",
    "clDice", "Dice sum0,1"))
for name, ckpt in ARMS:
    vae = load_vaev4(ckpt, DEV)
    generator = torch.Generator(device=DEV).manual_seed(1234)
    rows = []
    for case, truth, brain in volumes:
        with torch.inference_mode():
            volume = torch.from_numpy(truth)[None, None].to(DEV)
            recon, _ = reconstruct(vae, volume, "direct")
            with torch.amp.autocast("cuda", dtype=torch.float16):
                latent = encode_tiled(vae, volume)
            channel_std = latent.float().transpose(0, 1).reshape(latent.shape[1], -1).std(dim=1)
            noisy = latent.float() + torch.randn(
                latent.shape, device=DEV, generator=generator, dtype=torch.float32) \
                * SIGMA * channel_std.view(1, -1, 1, 1, 1)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                noisy_recon = decode_tiled(vae, noisy.to(latent.dtype))
        recon = np.clip(recon, 0.0, 1.0)
        noisy_recon = np.clip(noisy_recon[0, 0].float().cpu().numpy(), 0.0, 1.0)
        del volume, latent, noisy
        torch.cuda.empty_cache()

        glob, masked, psnr = masked_ssim_and_psnr(truth, recon, brain)
        mip_p = float(np.mean([10.0 * math.log10(1.0 / max(float(np.mean(
            (truth.max(a) - recon.max(a)) ** 2)), 1e-12)) for a in range(3)]))
        rows.append(dict(case=case, ssim_global=glob, ssim_masked=masked, psnr=psnr,
                         mip_ssim=mip_ssim(truth, recon), mip_psnr=mip_p,
                         dice=vessel_scores(truth, recon)[0],
                         cldice=hard_cldice(truth, recon),
                         dice_noise=vessel_scores(truth, noisy_recon)[0]))
    mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "case"}
    results[name] = {"checkpoint": str(ckpt.relative_to(ROOT)), "mean": mean, "rows": rows}
    print("%-16s %10.4f %10.4f %8.2f %9.4f %9.2f %8.4f %9.4f %11.4f   [%.1f min]" % (
        name, mean["ssim_global"], mean["ssim_masked"], mean["psnr"], mean["mip_ssim"],
        mean["mip_psnr"], mean["dice"], mean["cldice"], mean["dice_noise"],
        (time.time() - t0) / 60), flush=True)
    del vae
    torch.cuda.empty_cache()

OUT.write_text(json.dumps({"site": SITE, "split": "val", "cases": [c for c, _, _ in volumes],
                           "sigma": SIGMA, "arms": results}, indent=2), encoding="utf-8")
print("\npo ispitaniku, isporuceni model (+ topolosko):")
for r in results["+ topolosko"]["rows"]:
    print("   %-18s SSIM %.4f / %.4f  PSNR %.2f  Dice %.4f  clDice %.4f"
          % (r["case"], r["ssim_global"], r["ssim_masked"], r["psnr"], r["dice"], r["cldice"]))
print("\nzapisano ->", OUT, flush=True)
