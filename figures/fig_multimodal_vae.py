"""Slika 5.1: rekonstrukcije višemodalnog autoenkodera za T1, T2 i PD, na validacijskom skupu.

Stara je slika imala ispitanike iz testnog skupa, brojke računate drugim postupkom nego u
tablici 5.2 i oznake na engleskom. Nova koristi ista četiri validacijska ispitanika kao slika
MRA autoenkodera (dva iz Guysa, jedan iz HH-a, jedan iz IOP-a) i isti presjek po ispitaniku,
na razini velikih žila, pa se T1, T2, PD i MRA slike međusobno poklapaju.

Brojke u natpisima računate su istim postupkom kao tablica 5.2: skimage, jednoliki prozor
7 x 7 x 7, uz stvarni raspon vrijednosti z-normaliziranog volumena. Rekonstrukcija ide
determinističkim putem preko srednje vrijednosti, kao i izvoz latentnih prikaza.

Izgled je isti kao kod slike MRA autoenkodera: ime ispitanika i modalitet te ispod SSIM i
PSNR iznad lijeve sličice, oznaka rekonstrukcije iznad desne. Tri datoteke istih dimenzija.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))   # so `modules` imports when run as uv run figures/<script>.py
from modules.vae_multimodal import (VAEv2Multimodal, _make_single_modality_mask, _volume_tensor,
                                    _encode_deterministic)
from modules.paths import mask_path_for

DEV = "cuda"
MODS = ("T1", "T2", "PD")
MARGIN = 6
CASES = ["IXI016-Guys-0697", "IXI026-Guys-0696", "IXI033-HH-1259", "IXI290-IOP-0874"]
CKPT = ROOT / "checkpoints/vaev2_multimodal/kl5e5_from_epoch10/best.pt"
OUT = ROOT / "runs" / "figures"


def hr(x, n=4):
    return ("%.*f" % (n, x)).replace(".", ",")


model = VAEv2Multimodal(patch_size=(192, 192, 64), modalities=MODS, in_channels=1,
                        base_channels=16, channel_multipliers=(1, 2, 4), latent_channels=8,
                        blocks_per_level=1, output_activation=None).to(DEV)
blob = torch.load(CKPT, map_location="cpu", weights_only=False)
model.load_state_dict(blob["model_state_dict"])
model.eval()
del blob

# Geometrija iz MRA volumena: isti presjek i isti okvir kao na slici MRA autoenkodera.
geometry = {}
for case in CASES:
    path = ROOT / "Dataset/split_numpy/val/MRA" / (case + "-MRA.npy")
    mra = np.asarray(np.load(path), np.float32)
    brain = np.asarray(np.load(mask_path_for(path, "val"))) > 0
    z = int(np.median(np.nonzero(mra >= np.percentile(mra[brain], 99.9))[2]))
    xs, ys = np.nonzero(brain.any(2))[0], np.nonzero(brain.any((0, 2)))[0]
    box = (slice(max(int(xs.min()) - MARGIN, 0), int(xs.max()) + MARGIN),
           slice(max(int(ys.min()) - MARGIN, 0), int(ys.max()) + MARGIN))
    geometry[case] = (z, box)

panels, numbers = {}, {}
for idx, mod in enumerate(MODS):
    for case in CASES:
        z, box = geometry[case]
        path = ROOT / "Dataset/split_numpy/val" / mod / ("%s-%s.npy" % (case, mod))
        mpath = ROOT / "Dataset/split_numpy/val/masks" / mod / ("%s-%s_mask.npy" % (case, mod))
        target = np.asarray(np.load(path), np.float32)
        brain = np.asarray(np.load(mpath)) > 0
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

        data_range = float(target.max() - target.min())
        ssim_global = float(structural_similarity(target, recon, data_range=data_range))
        psnr = 10.0 * math.log10(data_range ** 2 / max(float(np.mean((target - recon) ** 2)), 1e-12))
        panels[(mod, case)] = (target[..., z][box], recon[..., z][box], brain[..., z][box])
        numbers.setdefault(mod, {})[case] = dict(z=z, ssim_global=ssim_global, psnr=psnr,
                                                data_range=data_range)
        print("  %-3s %-18s z=%d  SSIM %.4f  PSNR %.2f" % (mod, case, z, ssim_global, psnr),
              flush=True)

PAD_H = max(p[0].shape[0] for p in panels.values())
PAD_W = max(p[0].shape[1] for p in panels.values())


def pad(image, fill):
    top, left = (PAD_H - image.shape[0]) // 2, (PAD_W - image.shape[1]) // 2
    return np.pad(image, ((top, PAD_H - image.shape[0] - top), (left, PAD_W - image.shape[1] - left)),
                  constant_values=fill)


def show(ax, image, vmin, vmax, title, fill):
    ax.imshow(np.rot90(pad(image, fill)), cmap="gray", vmin=vmin, vmax=vmax,
              interpolation="antialiased")
    ax.set_axis_off()
    ax.set_title(title, fontsize=6.0 if "--docx" in sys.argv else 8.5, pad=2)


# Za Word: točno 1403 x 799 piksela, dimenzije starih slika u dokumentu, da se okvir ne
# razvuče i da se raspored stranica ne pomakne.
DOCX = "--docx" in sys.argv
if DOCX:
    WIDTH, HEIGHT = 1403 / 150, 799 / 150
else:
    WIDTH = 12.2
    HEIGHT = 2 * (WIDTH * 0.97 / 4) * (PAD_W / PAD_H) + 1.25        # rot90 zamijeni osi
for mod in MODS:
    fig, axes = plt.subplots(2, 4, figsize=(WIDTH, HEIGHT), facecolor="white")
    for k, case in enumerate(CASES):
        r, c = divmod(k, 2)
        gt, rec, brain = panels[(mod, case)]
        # Prozor iz referentnog presjeka unutar mozga; pozadina je nula nakon z-normalizacije
        # i ostaje siva, kao na izvorniku, a rekonstrukcija dijeli isti prozor.
        vmin, vmax = np.percentile(gt[brain], (1.0, 99.0))
        n = numbers[mod][case]
        show(axes[r, 2 * c], gt, vmin, vmax, "%s %s GT\nSSIM %s PSNR %s"
             % (case, mod, hr(n["ssim_global"]), hr(n["psnr"], 2)), fill=0.0)
        show(axes[r, 2 * c + 1], rec, vmin, vmax, "%s rekonstrukcija\n" % mod, fill=0.0)
    if DOCX:
        fig.subplots_adjust(left=0.005, right=0.995, top=1.0 - 0.34 / HEIGHT, bottom=0.005,
                            wspace=0.04, hspace=0.30)
    else:
        fig.subplots_adjust(left=0.01, right=0.99, top=1.0 - 0.55 / HEIGHT, bottom=0.01,
                            wspace=0.05, hspace=0.62 / (HEIGHT / 2))
    name = "visemodalni_%s%s.png" % (mod, "_docx" if DOCX else "")
    fig.savefig(OUT / name, dpi=150, facecolor="white")
    plt.close(fig)
    print("zapisano ->", OUT / name, flush=True)

(OUT / "visemodalni_brojke.json").write_text(json.dumps(
    {"checkpoint": str(CKPT.relative_to(ROOT)), "split": "val", "cases": numbers}, indent=2),
    encoding="utf-8")
