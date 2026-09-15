"""Uzorkovanje uz eta = 0,2 na ista četiri ispitanika kao na slikama u radu.

Raniji sweep mjerio je eta na 12 volumena i 50 koraka. Slike u radu rade s pet koraka, pa
se ovdje mjeri točno ta postavka: eta 0 i eta 0,2, isti model, isto sjeme po slučaju, pet
koraka mosta.

Uz uobičajene mjere bilježi se i tekstura: standardna devijacija visokopropusnog ostatka
(v - uniformni filtar 3) unutar maske mozga bez žilnog pojasa, izražena kao postotak iste
veličine na referentnom volumenu. Uz nju idu pojas žila (koliko voksela model gurne preko
praga u odnosu na referencu) i preciznost, jer šum najprije pokvari upravo njih.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import uniform_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))   # so `modules` imports when run as uv run figures/<script>.py
from modules.metrics import masked_ssim_and_psnr, mip_ssim, vessel_scores, hard_cldice
from modules.paths import mask_path_for
from modules.vae_mra import load_vaev4, decode_tiled
from modules.latent_data import PairedLatentDataset, destandardize_t
from modules.unet import build_model
from modules.bridge import BridgeProjector, BrownianBridge

DEV = "cuda"
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
ETAS = [0.0, 0.2]
MARGIN = 6
ZOOM = 90
CASES = ["IXI016-Guys-0697", "IXI026-Guys-0696", "IXI033-HH-1259", "IXI290-IOP-0874"]
CKPT = ROOT / "checkpoints/ldm_bbdm_vessel/ft_cldice/best.pt"
OUT = ROOT / "runs" / ("eta02_%d" % STEPS)
OUT.mkdir(parents=True, exist_ok=True)

cfg = json.loads((CKPT.parent / "config.json").read_text())
ds = PairedLatentDataset(cfg, "val", seed=42)
sc = sum(len(ds.stats[m]["per_channel_mean"]) for m in ds.sources)
tc = len(ds.stats[ds.target_modality]["per_channel_mean"])
stats = ds.stats[ds.target_modality]
vae = load_vaev4(ROOT / cfg["vae"]["checkpoint"], DEV)
ck = torch.load(CKPT, map_location=DEV, weights_only=False)
net = build_model(cfg, tc, sc, DEV); net.load_state_dict(ck["ema"]); net.eval()
proj = BridgeProjector(sc, tc, base=int(cfg["model"]["projector_base"]),
                       blocks=int(cfg["model"]["projector_blocks"])).to(DEV)
proj.load_state_dict(ck["ema_projector"]); proj.eval()
del ck
bridge = BrownianBridge(int(cfg["bridge"]["steps"]), float(cfg["bridge"]["max_variance"]), DEV)


def hr(x, n=4):
    return ("%.*f" % (n, x)).replace(".", ",")


def tex(v, mask):
    return float(np.std((v - uniform_filter(v, size=3))[mask]))


def decode(lat):
    with torch.amp.autocast("cuda", dtype=torch.float16):
        r = decode_tiled(vae, destandardize_t(lat, stats).to(lat.dtype))
    return np.clip(r[0, 0].float().cpu().numpy(), 0.0, 1.0)


rows, panels = {e: [] for e in ETAS}, []
t0 = time.time()
for case in CASES:
    tgt, cond = ds.full_case(case)
    cond = cond[None].to(DEV)
    shape = (1,) + tuple(tgt.shape)
    p = ROOT / "Dataset/split_numpy/val/MRA" / (case + "-MRA.npy")
    truth = np.asarray(np.load(p), np.float32)
    brain = np.asarray(np.load(mask_path_for(p, "val"))) > 0
    thr = float(np.percentile(truth, 99.0))
    tb = truth >= thr
    par = brain & ~tb
    T = tex(truth, par)
    z = int(np.median(np.nonzero(truth >= np.percentile(truth[brain], 99.9))[2]))

    outs = {}
    for e in ETAS:
        g = torch.Generator(device=DEV).manual_seed(1234)
        with torch.inference_mode():
            rec = decode(bridge.sample(net, proj, cond, shape, DEV, STEPS, eta=e, generator=g))
        torch.cuda.empty_cache()
        outs[e] = rec
        gl, ms, psnr = masked_ssim_and_psnr(truth, rec, brain)
        d, _ = vessel_scores(truth, rec)
        pb = rec >= thr
        rows[e].append(dict(case=case, ssim=gl, ssim_m=ms, psnr=psnr,
                            mip=mip_ssim(truth, rec), dice=d, cldice=hard_cldice(truth, rec),
                            band=float(pb.sum() / max(tb.sum(), 1)),
                            prec=float((tb & pb).sum() / max(pb.sum(), 1)),
                            texture=100.0 * tex(rec, par) / T))
        print("  %-18s eta %.1f  SSIM %.4f/%.4f  Dice %.4f  clDice %.4f  pojas %.2f  "
              "prec %.3f  tekstura %.0f%%   [%.1f min]"
              % (case, e, gl, ms, rows[e][-1]["dice"], rows[e][-1]["cldice"],
                 rows[e][-1]["band"], rows[e][-1]["prec"], rows[e][-1]["texture"],
                 (time.time() - t0) / 60), flush=True)

    xs, ys = np.nonzero(brain.any(2))[0], np.nonzero(brain.any((0, 2)))[0]
    box = (slice(max(int(xs.min()) - MARGIN, 0), int(xs.max()) + MARGIN),
           slice(max(int(ys.min()) - MARGIN, 0), int(ys.max()) + MARGIN))
    cy, cx = int(np.mean(xs)), int(np.mean(ys))
    zb = (slice(cy - ZOOM, cy + ZOOM), slice(cx - ZOOM, cx + ZOOM))
    panels.append((case, z, [truth[..., z][box]] + [outs[e][..., z][box] for e in ETAS],
                   [truth[..., z][zb]] + [outs[e][..., z][zb] for e in ETAS],
                   [truth.max(2)[box]] + [outs[e].max(2)[box] for e in ETAS],
                   float(np.percentile(truth[..., z][brain[..., z]], 99.5)),
                   float(np.percentile(truth.max(2)[brain.max(2)], 99.5))))

summary = {str(e): {k: float(np.mean([r[k] for r in rows[e]]))
                    for k in ("ssim", "ssim_m", "psnr", "mip", "dice", "cldice", "band",
                              "prec", "texture")} for e in ETAS}
(OUT / "eta02.json").write_text(json.dumps(
    {"model": str(CKPT.relative_to(ROOT)), "steps": STEPS, "cases": CASES,
     "per_case": {str(e): rows[e] for e in ETAS}, "summary": summary}, indent=2),
    encoding="utf-8")

print("\n%5s %9s %9s %8s %9s %8s %9s %7s %7s" % (
    "eta", "SSIM", "SSIM mask", "PSNR", "MIP SSIM", "Dice", "clDice", "pojas", "prec"))
for e in ETAS:
    s = summary[str(e)]
    print("%5.1f %9.4f %9.4f %8.2f %9.4f %8.4f %9.4f %7.2f %7.3f  tekstura %3.0f%%" % (
        e, s["ssim"], s["ssim_m"], s["psnr"], s["mip"], s["dice"], s["cldice"], s["band"],
        s["prec"], s["texture"]))

# -------------------------------------------------------------------- slika
COLS = ["referenca", "η = 0", "η = 0,2"]
fig = plt.figure(figsize=(13.0, 16.4), facecolor="white")
blocks = fig.subfigures(2, 2, hspace=0.01, wspace=0.01)
for k, (case, z, sl, zoom, mip, v_sl, v_mip) in enumerate(panels):
    sf = blocks[k // 2, k % 2]
    r0 = rows[0.0][k]
    r2 = rows[0.2][k]
    sf.suptitle("%s   ·   presjek z = %d\n"
                "η = 0:     Dice %s · clDice %s · pojas %s · tekstura %.0f %%\n"
                "η = 0,2:  Dice %s · clDice %s · pojas %s · tekstura %.0f %%"
                % (case, z, hr(r0["dice"]), hr(r0["cldice"]), hr(r0["band"], 2),
                   r0["texture"], hr(r2["dice"]), hr(r2["cldice"]), hr(r2["band"], 2),
                   r2["texture"]), fontsize=8.5, y=0.995)
    ax = sf.subplots(3, 3, gridspec_kw=dict(wspace=0.02, hspace=0.12, top=0.90,
                                            bottom=0.005, left=0.01, right=0.99))
    for j in range(3):
        for i, (img, vmax, lab) in enumerate((
                (sl[j], max(0.45, v_sl), COLS[j]),
                (zoom[j], max(0.45, v_sl), "uvećano, tekstura"),
                (mip[j], max(0.45, v_mip), "MIP projekcija"))):
            ax[i, j].imshow(np.rot90(img), cmap="gray", vmin=0.0, vmax=vmax,
                            interpolation="antialiased")
            ax[i, j].set_axis_off()
            ax[i, j].set_title(lab if i == 0 else (lab if j == 0 else ""), fontsize=8.5,
                               pad=3)
fig.savefig(OUT / "eta02.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("\nzapisano ->", OUT / "eta02.png", flush=True)
