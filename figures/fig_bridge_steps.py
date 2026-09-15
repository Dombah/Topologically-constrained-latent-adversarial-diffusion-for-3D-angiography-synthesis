"""Slike za usporedbu broja koraka mosta: 1, 3, 5 i 50.

Tri slučaja s validacijskog skupa, po jedan iz svake ustanove, prvi po abecedi unutar
ustanove — dakle pravilo zadano unaprijed, bez biranja.

Uz svaku je ploču ispisan omjer pojasa, jer je upravo on ono što se u tablici vidi kao
monoton porast s brojem koraka: pri malom broju model podbacuje u pokrivenosti, pri
velikom zadebljava. Bez te brojke slike se lako čitaju krivo.

Sve MRA ploče jednog slučaja dijele isti prozor intenziteta, uzet iz referentnog volumena.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))   # so `modules` imports when run as uv run figures/<script>.py
from modules.paths import mask_path_for
from modules.vae_mra import load_vaev4, decode_tiled
from modules.latent_data import PairedLatentDataset, destandardize_t
from modules.unet import build_model
from modules.bridge import BridgeProjector, BrownianBridge

DEV = "cuda"
STEPS = [1, 3, 5, 50]
CKPT = ROOT / "checkpoints/ldm_bbdm_vessel/ft_cldice/best.pt"
OUT = ROOT / "runs" / "step_previews"
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

cases = []
for site in ("Guys", "HH", "IOP"):
    cases += [c for c in ds.cases if c.split("-")[1] == site][:1]
print("slučajevi:", cases, flush=True)


def decode(lat):
    with torch.amp.autocast("cuda", dtype=torch.float16):
        r = decode_tiled(vae, destandardize_t(lat, stats).to(lat.dtype))
    return np.clip(r[0, 0].float().cpu().numpy(), 0.0, 1.0)


panels = []
t0 = time.time()
for case in cases:
    tgt, cond = ds.full_case(case)
    cond = cond[None].to(DEV)
    shape = (1,) + tuple(tgt.shape)
    p = ROOT / "Dataset/split_numpy/val/MRA" / (case + "-MRA.npy")
    truth = np.asarray(np.load(p), np.float32)
    brain = np.asarray(np.load(mask_path_for(p, "val"))) > 0
    thr = float(np.percentile(truth, 99.0))
    tb = truth >= thr
    recs = {}
    with torch.inference_mode():
        for k in STEPS:
            r = decode(bridge.sample(net, proj, cond, shape, DEV, k))
            recs[k] = (r, float((r >= thr).sum() / max(tb.sum(), 1)))
            torch.cuda.empty_cache()
    panels.append((case, truth, brain, recs))
    print("  %s  [%.1f min]" % (case, (time.time() - t0) / 60), flush=True)


def render(kind, fname, title):
    cols = 1 + len(STEPS)
    fig, ax = plt.subplots(len(panels), cols, figsize=(2.65 * cols, 3.0 * len(panels)),
                           facecolor="white")
    ax = np.atleast_2d(ax)
    for i, (case, truth, brain, recs) in enumerate(panels):
        gt = truth.max(2)
        bm = brain.max(2)
        vmax = max(0.45, float(np.percentile(gt[bm], 99.5)))
        if kind == "detail":
            ys, xs = np.nonzero(gt >= np.percentile(gt[bm], 99.7))
            hy = 100
            hx = int(round(gt.shape[1] / gt.shape[0] * hy))
            cy = int(np.clip(int(ys.mean()), hy, gt.shape[0] - hy))
            cx = int(np.clip(int(xs.mean()), hx, gt.shape[1] - hx))
            sl = (slice(cy - hy, cy + hy), slice(cx - hx, cx + hx))
            vmax *= 0.75
        views = [("referentni", gt if kind == "mip" else gt[sl], None)]
        for k in STEPS:
            r, band = recs[k]
            m = r.max(2)
            views.append(("%d %s" % (k, "korak" if k == 1 else "koraka"),
                          m if kind == "mip" else m[sl], band))
        for j, (lab, img, band) in enumerate(views):
            a = ax[i, j]
            a.imshow(np.rot90(img), cmap="gray", vmin=0, vmax=vmax,
                     interpolation="antialiased" if kind == "mip" else "bicubic")
            a.set_axis_off()
            if i == 0:
                a.set_title(lab, fontsize=10.5, pad=5)
            if band is not None:
                a.text(0.5, -0.035, "pojas %.2f" % band, transform=a.transAxes,
                       ha="center", va="top", fontsize=7.6, color="#555555")
        ax[i, 0].text(-0.05, 0.5, case, rotation=90, va="center", ha="center",
                      transform=ax[i, 0].transAxes, fontsize=7.6, color="#555555")
    fig.suptitle(title, fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(OUT / fname, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("zapisano ->", OUT / fname, flush=True)


render("mip", "koraci_mip.png",
       "Broj koraka mosta — aksijalna projekcija najvećih vrijednosti")
render("detail", "koraci_detalj.png",
       "Broj koraka mosta — uvećani prikaz Willisova kruga")
