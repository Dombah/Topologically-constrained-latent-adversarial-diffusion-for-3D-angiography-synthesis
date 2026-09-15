"""Slike za 5.4: sinteza isporučenim modelom na testnom skupu, jedna slika po ispitaniku.

Ispitanici su izabrani fiksnim pravilom, bez gledanja slika: prva dva iz Guysa, prvi iz
HH-a i prvi iz IOP-a, abecednim redom unutar testnog skupa. Izgled je isti kao kod
validacijskih slika koje su već u radu: tri retka (GT, rekonstrukcija, apsolutna pogreška),
lijevo aksijalni presjek na razini velikih žila, desno MIP projekcija, sve četiri datoteke
jednakih dimenzija.

Model je isporučeni most (ldm_bbdm_vessel/ft_cldice, epoha 10), pet koraka, deterministično.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))   # so `modules` imports when run as uv run figures/<script>.py
from modules.metrics import masked_ssim_and_psnr
from modules.paths import mask_path_for
from modules.vae_mra import load_vaev4, decode_tiled
from modules.latent_data import PairedLatentDataset, destandardize_t
from modules.unet import build_model
from modules.bridge import BridgeProjector, BrownianBridge

DEV = "cuda"
SPLIT = "test"
STEPS = 5
MARGIN = 6
CASES = ["IXI022-Guys-0701", "IXI030-Guys-0708", "IXI056-HH-1327", "IXI315-IOP-0888"]
CKPT = ROOT / "checkpoints/ldm_bbdm_vessel/ft_cldice/best.pt"
OUT = ROOT / "runs" / "figures" / "test_v8"
OUT.mkdir(parents=True, exist_ok=True)


def hr(x, n=4):
    return ("%.*f" % (n, x)).replace(".", ",")


cfg = json.loads((CKPT.parent / "config.json").read_text())
ds = PairedLatentDataset(cfg, SPLIT, seed=42)
sc = sum(len(ds.stats[m]["per_channel_mean"]) for m in ds.sources)
tc = len(ds.stats[ds.target_modality]["per_channel_mean"])
stats = ds.stats[ds.target_modality]
vae = load_vaev4(ROOT / cfg["vae"]["checkpoint"], DEV)
ck = torch.load(CKPT, map_location=DEV, weights_only=False)
net = build_model(cfg, tc, sc, DEV); net.load_state_dict(ck["ema"]); net.eval()
proj = BridgeProjector(sc, tc, base=int(cfg["model"]["projector_base"]),
                       blocks=int(cfg["model"]["projector_blocks"])).to(DEV)
proj.load_state_dict(ck["ema_projector"]); proj.eval()
epoch = int(ck["epoch"])
del ck
bridge = BrownianBridge(int(cfg["bridge"]["steps"]), float(cfg["bridge"]["max_variance"]), DEV)

panels, numbers = {}, {}
for case in CASES:
    target, cond = ds.full_case(case)
    path = ROOT / "Dataset/split_numpy" / SPLIT / "MRA" / (case + "-MRA.npy")
    truth = np.asarray(np.load(path), np.float32)
    brain = np.asarray(np.load(mask_path_for(path, SPLIT))) > 0
    with torch.inference_mode():
        latent = bridge.sample(net, proj, cond[None].to(DEV), (1,) + tuple(target.shape), DEV, STEPS)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            decoded = decode_tiled(vae, destandardize_t(latent, stats).to(latent.dtype))
    synth = np.clip(decoded[0, 0].float().cpu().numpy(), 0.0, 1.0)
    torch.cuda.empty_cache()

    ssim_global, ssim_masked, psnr = masked_ssim_and_psnr(truth, synth, brain)
    z = int(np.median(np.nonzero(truth >= np.percentile(truth[brain], 99.9))[2]))
    xs, ys = np.nonzero(brain.any(2))[0], np.nonzero(brain.any((0, 2)))[0]
    box = (slice(max(int(xs.min()) - MARGIN, 0), int(xs.max()) + MARGIN),
           slice(max(int(ys.min()) - MARGIN, 0), int(ys.max()) + MARGIN))
    panels[case] = dict(t_sl=truth[..., z][box], s_sl=synth[..., z][box],
                        t_mip=truth.max(2)[box], s_mip=synth.max(2)[box],
                        b_sl=brain[..., z][box], b_mip=brain.max(2)[box])
    numbers[case] = dict(z=z, ssim_global=float(ssim_global), ssim_masked=float(ssim_masked),
                         psnr=float(psnr))
    print("  %-18s z=%d  SSIM %.4f (maska %.4f)  PSNR %.2f" % (case, z, ssim_global,
                                                              ssim_masked, psnr), flush=True)

PAD_H = max(p[k].shape[0] for p in panels.values() for k in ("t_sl", "t_mip"))
PAD_W = max(p[k].shape[1] for p in panels.values() for k in ("t_sl", "t_mip"))


def pad(image):
    top, left = (PAD_H - image.shape[0]) // 2, (PAD_W - image.shape[1]) // 2
    return np.pad(image, ((top, PAD_H - image.shape[0] - top), (left, PAD_W - image.shape[1] - left)))


def show(ax, image, vmax, title, cmap="gray", size=8.5):
    ax.imshow(np.rot90(pad(image)), cmap=cmap, vmin=0.0, vmax=vmax, interpolation="antialiased")
    ax.set_axis_off()
    ax.set_title(title, fontsize=size, pad=3)


WIDTH = 6.8
HEIGHT = 3.0 * (WIDTH * 0.49) * (PAD_W / PAD_H) + 1.35
for case in CASES:
    p, n = panels[case], numbers[case]
    v_sl = max(0.45, float(np.percentile(p["t_sl"][p["b_sl"]], 99.5)))
    v_mip = max(0.45, float(np.percentile(p["t_mip"][p["b_mip"]], 99.5)))
    e_sl, e_mip = np.abs(p["t_sl"] - p["s_sl"]), np.abs(p["t_mip"] - p["s_mip"])
    ve = max(0.15, float(np.percentile(e_mip, 99.0)))
    fig, ax = plt.subplots(3, 2, figsize=(WIDTH, HEIGHT), facecolor="white")
    show(ax[0, 0], p["t_sl"], v_sl, "%s MRA GT, presjek\nSSIM %s PSNR %s"
         % (case, hr(n["ssim_global"]), hr(n["psnr"], 2)))
    show(ax[0, 1], p["t_mip"], v_mip, "MRA GT, MIP projekcija\n")
    show(ax[1, 0], p["s_sl"], v_sl, "generirani MRA, presjek")
    show(ax[1, 1], p["s_mip"], v_mip, "generirani MRA, MIP projekcija")
    show(ax[2, 0], e_sl, ve, "apsolutna pogreška, presjek", cmap="magma")
    show(ax[2, 1], e_mip, ve, "apsolutna pogreška, MIP projekcija", cmap="magma")
    fig.tight_layout(h_pad=1.2, w_pad=0.4)
    fig.savefig(OUT / ("most_test_%s.png" % case), dpi=150, facecolor="white")
    plt.close(fig)
    print("zapisano ->", OUT / ("most_test_%s.png" % case), flush=True)

(OUT / "most_test_brojke.json").write_text(json.dumps(
    {"model": str(CKPT.relative_to(ROOT)), "epoch": epoch, "split": SPLIT, "steps": STEPS,
     "selection": "prva dva Guys, prvi HH, prvi IOP abecednim redom", "cases": numbers},
    indent=2), encoding="utf-8")
