"""Koliko koraka mosta zapravo treba?

Dosad su uspoređena samo dva broja koraka, 5 i 50, jer je tako stajalo u konfiguraciji
validacije. Ovo mjeri niz 1, 2, 3, 5, 10, 20, 50 i 100 na odabranom modelu, uz krajnju
točku projektora kao referentni redak (nula koraka, bez difuzije).

Slučajevi su birani uravnoteženo po ustanovama (4 po ustanovi), a ne prvih N po abecedi,
jer su se ranija mjerenja pokazala osjetljivima na to koja je ustanova zastupljena.

Uz uobičajene mjere prati se i omjer pojasa: udio voksela iznad praga u odnosu na
referentni volumen. Bez njega se ne vidi razlikuje li porast Dice mjere stvarno bolje
žile od jednostavnog zadebljavanja struktura.

Ništa se ne piše u rad; ispisuje se tablica i sprema json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))   # so `modules` imports when run as uv run analysis/<script>.py
from modules.metrics import masked_ssim_and_psnr, vessel_scores, mip_ssim, hard_cldice
from modules.paths import mask_path_for, pick_validation_cases
from modules.vae_mra import load_vaev4, decode_tiled
from modules.latent_data import PairedLatentDataset, destandardize_t
from modules.unet import build_model
from modules.bridge import BridgeProjector, BrownianBridge

DEV = "cuda"
STEPS = [1, 2, 3, 5, 10, 20, 50, 100]
PER_SITE = 4
CKPT = ROOT / "checkpoints/ldm_bbdm_vessel/ft_cldice/best.pt"
OUT = ROOT / "runs" / "step_sweep"
OUT.mkdir(parents=True, exist_ok=True)

cfg = json.loads((CKPT.parent / "config.json").read_text())
ds = PairedLatentDataset(cfg, "val", seed=42)
sc = sum(len(ds.stats[m]["per_channel_mean"]) for m in ds.sources)
tc = len(ds.stats[ds.target_modality]["per_channel_mean"])
stats = ds.stats[ds.target_modality]

ck = torch.load(CKPT, map_location=DEV, weights_only=False)
net = build_model(cfg, tc, sc, DEV); net.load_state_dict(ck["ema"]); net.eval()
proj = BridgeProjector(sc, tc, base=int(cfg["model"]["projector_base"]),
                       blocks=int(cfg["model"]["projector_blocks"])).to(DEV)
proj.load_state_dict(ck["ema_projector"]); proj.eval()
del ck
vae = load_vaev4(ROOT / cfg["vae"]["checkpoint"], DEV)
bridge = BrownianBridge(int(cfg["bridge"]["steps"]), float(cfg["bridge"]["max_variance"]), DEV)

wanted = {p.stem.rsplit("-", 1)[0] for p in pick_validation_cases("val", PER_SITE)}
cases = [c for c in ds.cases if c in wanted]
sites = {}
for c in cases:
    sites[c.split("-")[1]] = sites.get(c.split("-")[1], 0) + 1
print("slučajeva: %d  %s" % (len(cases), sites), flush=True)


def decode(lat):
    with torch.amp.autocast("cuda", dtype=torch.float16):
        r = decode_tiled(vae, destandardize_t(lat, stats).to(lat.dtype))
    return np.clip(r[0, 0].float().cpu().numpy(), 0.0, 1.0)


def score(truth, brain, recon):
    thr = float(np.percentile(truth, 99.0))
    g, m, psnr = masked_ssim_and_psnr(truth, recon, brain)
    dice, _ = vessel_scores(truth, recon)
    return dict(dice=dice, cldice=hard_cldice(truth, recon), ssim_global=g,
                ssim_masked=m, psnr=psnr, mip_ssim=mip_ssim(truth, recon),
                band=float((recon >= thr).sum() / max((truth >= thr).sum(), 1)))


rows = {str(k): [] for k in STEPS}
rows["endpoint"] = []
t0 = time.time()
for i, case in enumerate(cases, 1):
    tgt, cond = ds.full_case(case)
    cond = cond[None].to(DEV)
    shape = (1,) + tuple(tgt.shape)
    p = ROOT / "Dataset/split_numpy/val/MRA" / (case + "-MRA.npy")
    truth = np.asarray(np.load(p), np.float32)
    brain = np.asarray(np.load(mask_path_for(p, "val"))) > 0

    with torch.inference_mode():
        rows["endpoint"].append(score(truth, brain, decode(bridge.endpoint(proj, cond))))
        for k in STEPS:
            rec = decode(bridge.sample(net, proj, cond, shape, DEV, k))
            rows[str(k)].append(score(truth, brain, rec))
            torch.cuda.empty_cache()
    print("  %2d/%d  %s  [%.1f min]" % (i, len(cases), case, (time.time() - t0) / 60),
          flush=True)

summary = {k: {m: float(np.mean([r[m] for r in v])) for m in v[0]} for k, v in rows.items()}
(OUT / "sweep.json").write_text(json.dumps(
    {"checkpoint": str(CKPT), "cases": cases, "per_site": PER_SITE,
     "summary": summary, "rows": rows}, indent=2, default=float), encoding="utf-8")

print("\n%-10s %8s %8s %10s %10s %8s %9s %7s" % (
    "koraka", "Dice", "clDice", "SSIM glob", "SSIM mask", "PSNR", "MIP SSIM", "pojas"))
for k in ["endpoint"] + [str(x) for x in STEPS]:
    r = summary[k]
    print("%-10s %8.4f %8.4f %10.4f %10.4f %8.2f %9.4f %7.2f" % (
        k, r["dice"], r["cldice"], r["ssim_global"], r["ssim_masked"], r["psnr"],
        r["mip_ssim"], r["band"]))
print("\nzapisano ->", OUT / "sweep.json", flush=True)
