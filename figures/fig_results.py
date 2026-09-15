"""Dvije slike za poglavlje 5, iz sadašnjih modela.

  slika A  rekonstrukcija MRA autoenkodera: referenca i rekonstrukcija, po slučaju
  slika B  sinteza mostom: referenca, generirano i mapa apsolutne pogreške, na presjeku
           i na MIP projekciji, sva četiri slučaja u jednoj slici

Ista četiri slučaja na obje slike, pa se može usporediti što autoenkoder vrati i što
difuzijski model doda: dva iz Guysa, jedan iz HH-a i jedan iz IOP-a. Svi su iz
validacijskog skupa, kao i sve ostale brojke u radu; testni skup ostaje nedirnut.

Uz svaki slučaj idu iste mjere koje stoje i u tablicama: SSIM globalni i unutar maske
mozga, MIP SSIM, PSNR, Dice krvnih žila, clDice, a za autoenkoder još i Dice nakon
dodavanja Gaussova šuma standardne devijacije 0,1 u latentni prikaz, istim postupkom kao
u tablici finog podešavanja (šum se mjeri u jedinicama standardne devijacije kanala).

Mjere se računaju na cijelom volumenu. Prikaz je obrezan na okvir maske mozga, jednako za
referencu i za izlaz modela, jer inače dvije trećine svake sličice zauzima prazna
pozadina. Sve sličice jednog slučaja dijele jedan interval intenziteta, uzet iz
reference, da se izlaz modela ne bi mogao posvijetliti u bolji dojam.

Ranija je verzija tih slika bila rađena starim modelom s v-predikcijom i uzorkovanjem
DDIM-om, natpisi su govorili o sagitalnim presjecima iako su presjeci aksijalni, a sva su
četiri slučaja na slici mosta bila iz Guysa.

Međurezultati se spremaju u .npz, pa se raspored može mijenjati bez ponovnog uzorkovanja:
    python figs_rezultati.py --samo-crtanje
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))   # so `modules` imports when run as uv run figures/<script>.py
DEV = "cuda"
STEPS = 5
SIGMA = 0.1
MARGIN = 6
CASES = ["IXI016-Guys-0697", "IXI026-Guys-0696", "IXI033-HH-1259", "IXI290-IOP-0874"]
CKPT = ROOT / "checkpoints/ldm_bbdm_vessel/ft_cldice/best.pt"
OUT = ROOT / "runs" / "figures"
CACHE = OUT / "figure_panels.npz"
OUT.mkdir(parents=True, exist_ok=True)
PLOT_ONLY = "--samo-crtanje" in sys.argv

M_KEYS = ("vae_ssim", "vae_ssim_m", "vae_psnr", "vae_mip", "vae_mip_psnr", "vae_dice",
          "vae_cldice", "vae_dice_noise", "syn_ssim", "syn_ssim_m", "syn_psnr", "syn_mip",
          "syn_mip_psnr", "syn_dice", "syn_cldice", "z")


def hr(x, n=4):
    return ("%.*f" % (n, x)).replace(".", ",")


# =============================================================================
# Računanje
# =============================================================================
def compute():
    import torch
    sys.path.insert(0, str(ROOT))
    from modules.metrics import masked_ssim_and_psnr, mip_ssim, vessel_scores, hard_cldice
    from modules.paths import mask_path_for
    from modules.vae_mra import load_vaev4, encode_tiled, decode_tiled, reconstruct
    from modules.latent_data import PairedLatentDataset, destandardize_t
    from modules.unet import build_model
    from modules.bridge import BridgeProjector, BrownianBridge

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
    bridge = BrownianBridge(int(cfg["bridge"]["steps"]),
                            float(cfg["bridge"]["max_variance"]), DEV)
    gen = torch.Generator(device=DEV).manual_seed(1234)

    def mip_psnr(truth, pred):
        return float(np.mean([10.0 * np.log10(1.0 / max(float(np.mean(
            (truth.max(a) - pred.max(a)) ** 2)), 1e-12)) for a in range(3)]))

    panels, metrics = {}, {}
    t0 = time.time()
    for case in CASES:
        tgt, cond = ds.full_case(case)
        p = ROOT / "Dataset/split_numpy/val/MRA" / (case + "-MRA.npy")
        truth = np.asarray(np.load(p), np.float32)
        brain = np.asarray(np.load(mask_path_for(p, "val"))) > 0

        with torch.inference_mode():
            vol = torch.from_numpy(truth)[None, None].to(DEV)
            rec, _ = reconstruct(vae, vol, "direct")

            # Dice uz šum 0,1: isti postupak kao evaluate_perturbed -- latentni prikaz iz
            # pločičnog kodiranja, šum u jedinicama standardne devijacije po kanalu.
            with torch.amp.autocast("cuda", dtype=torch.float16):
                lat = encode_tiled(vae, vol)
            std = lat.float().transpose(0, 1).reshape(lat.shape[1], -1).std(dim=1)
            noisy = lat.float() + torch.randn(lat.shape, device=DEV, generator=gen,
                                              dtype=torch.float32) * SIGMA * std.view(1, -1, 1, 1, 1)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                rec_n = decode_tiled(vae, noisy.to(lat.dtype))
            rec_n = np.clip(rec_n[0, 0].float().cpu().numpy(), 0.0, 1.0)
            del vol, lat, noisy
            torch.cuda.empty_cache()

            sampled = bridge.sample(net, proj, cond[None].to(DEV), (1,) + tuple(tgt.shape),
                                    DEV, STEPS)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                syn = decode_tiled(vae, destandardize_t(sampled, stats).to(sampled.dtype))
            syn = np.clip(syn[0, 0].float().cpu().numpy(), 0.0, 1.0)
            torch.cuda.empty_cache()

        rec = np.clip(rec, 0.0, 1.0)
        g_v, m_v, psnr_v = masked_ssim_and_psnr(truth, rec, brain)
        g_s, m_s, psnr_s = masked_ssim_and_psnr(truth, syn, brain)
        m = dict(vae_ssim=g_v, vae_ssim_m=m_v, vae_psnr=psnr_v,
                 vae_mip=mip_ssim(truth, rec), vae_mip_psnr=mip_psnr(truth, rec),
                 vae_dice=vessel_scores(truth, rec)[0], vae_cldice=hard_cldice(truth, rec),
                 vae_dice_noise=vessel_scores(truth, rec_n)[0],
                 syn_ssim=g_s, syn_ssim_m=m_s, syn_psnr=psnr_s,
                 syn_mip=mip_ssim(truth, syn), syn_mip_psnr=mip_psnr(truth, syn),
                 syn_dice=vessel_scores(truth, syn)[0], syn_cldice=hard_cldice(truth, syn))

        # presjek na kojem su vidljive velike žile: medijan dubine najsvjetljijih voksela
        z = int(np.median(np.nonzero(truth >= np.percentile(truth[brain], 99.9))[2]))
        m["z"] = float(z)
        xs, ys = np.nonzero(brain.any(2))[0], np.nonzero(brain.any((0, 2)))[0]
        box = (slice(max(int(xs.min()) - MARGIN, 0), int(xs.max()) + MARGIN),
               slice(max(int(ys.min()) - MARGIN, 0), int(ys.max()) + MARGIN))

        panels[case] = np.stack([truth[..., z][box], rec[..., z][box], syn[..., z][box],
                                 truth.max(2)[box], rec.max(2)[box], syn.max(2)[box],
                                 brain[..., z][box].astype(np.float32),
                                 brain.max(2)[box].astype(np.float32)])
        metrics[case] = np.array([m[k] for k in M_KEYS], np.float64)
        print("  %-18s  VAE: SSIM %.4f/%.4f  MIP %.4f  Dice %.4f  sum0,1 %.4f  |  most: "
              "SSIM %.4f  Dice %.4f  clDice %.4f   [%.1f min]"
              % (case, m["vae_ssim"], m["vae_ssim_m"], m["vae_mip"], m["vae_dice"],
                 m["vae_dice_noise"], m["syn_ssim_m"], m["syn_dice"], m["syn_cldice"],
                 (time.time() - t0) / 60), flush=True)

    np.savez_compressed(CACHE, **{"p_" + c: panels[c] for c in CASES},
                        **{"m_" + c: metrics[c] for c in CASES})
    (OUT / "figure_cases.json").write_text(json.dumps(
        {"model": str(CKPT.relative_to(ROOT)), "vae": cfg["vae"]["checkpoint"],
         "split": "val", "bridge_steps": STEPS, "noise_sigma": SIGMA,
         "cases": {c: {k: round(float(v), 4) for k, v in zip(M_KEYS, metrics[c])}
                   for c in CASES}}, indent=2), encoding="utf-8")
    return panels, metrics


if PLOT_ONLY:
    blob = np.load(CACHE)
    panels = {c: blob["p_" + c] for c in CASES}
    metrics = {c: blob["m_" + c] for c in CASES}
else:
    panels, metrics = compute()

D = {c: dict(zip(M_KEYS, metrics[c])) for c in CASES}
for c in CASES:
    t_sl, v_sl, s_sl, t_mip, v_mip, s_mip, b_sl, b_mip = panels[c]
    D[c].update(t_sl=t_sl, v_sl=v_sl, s_sl=s_sl, t_mip=t_mip, v_mip=v_mip, s_mip=s_mip,
                w_sl=max(0.45, float(np.percentile(t_sl[b_sl > 0], 99.5))),
                w_mip=max(0.45, float(np.percentile(t_mip[b_mip > 0], 99.5))))


def show(ax, img, vmax, title, cmap="gray", size=8.5):
    ax.imshow(np.rot90(img), cmap=cmap, vmin=0.0, vmax=vmax, interpolation="antialiased")
    ax.set_axis_off()
    ax.set_title(title, fontsize=size, pad=3)


# =============================================================================
# Slika A: autoenkoder
# =============================================================================
fig = plt.figure(figsize=(12.2, 7.6), facecolor="white")
blocks = fig.subfigures(2, 2, hspace=0.01, wspace=0.01)
for k, c in enumerate(CASES):
    d = D[c]
    sf = blocks[k // 2, k % 2]
    # Natpisi stoje iznad pojedine sličice, kao na ranijoj verziji slike: lijevo ime
    # ispitanika i mjere, desno samo oznaka. Prazni reci na kraju desnog natpisa dižu ga
    # na visinu prvog retka lijevoga, jer se natpis sidri donjim rubom.
    ax = sf.subplots(1, 2, gridspec_kw=dict(wspace=0.02, top=0.86, bottom=0.01,
                                            left=0.01, right=0.99))
    show(ax[0], d["t_sl"], d["w_sl"],
         "%s MRA GT\nSSIM %s PSNR %s" % (c, hr(d["vae_ssim"]), hr(d["vae_psnr"], 2)),
         size=8.5)
    show(ax[1], d["v_sl"], d["w_sl"], "MRA rekonstrukcija\n", size=8.5)
fig.savefig(OUT / "vae_rekonstrukcija.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("zapisano ->", OUT / "vae_rekonstrukcija.png", flush=True)

# =============================================================================
# Slika B: sinteza mostom
# =============================================================================
fig = plt.figure(figsize=(12.2, 15.2), facecolor="white")
blocks = fig.subfigures(2, 2, hspace=0.01, wspace=0.01)
for k, c in enumerate(CASES):
    d = D[c]
    sf = blocks[k // 2, k % 2]
    ax = sf.subplots(3, 2, gridspec_kw=dict(wspace=0.02, hspace=0.13, top=0.90,
                                            bottom=0.005, left=0.01, right=0.99))
    e_sl, e_mip = np.abs(d["t_sl"] - d["s_sl"]), np.abs(d["t_mip"] - d["s_mip"])
    ve = max(0.15, float(np.percentile(e_mip, 99.0)))
    show(ax[0, 0], d["t_sl"], d["w_sl"],
         "%s MRA GT, presjek\nSSIM %s PSNR %s"
         % (c, hr(d["syn_ssim"]), hr(d["syn_psnr"], 2)), size=8.5)
    show(ax[0, 1], d["t_mip"], d["w_mip"], "MRA GT, MIP projekcija\n")
    show(ax[1, 0], d["s_sl"], d["w_sl"], "MRA rekonstrukcija, presjek")
    show(ax[1, 1], d["s_mip"], d["w_mip"], "MRA rekonstrukcija, MIP projekcija")
    show(ax[2, 0], e_sl, ve, "apsolutna pogreška, presjek", cmap="magma")
    show(ax[2, 1], e_mip, ve, "apsolutna pogreška, MIP", cmap="magma")
fig.savefig(OUT / "most_sinteza.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("zapisano ->", OUT / "most_sinteza.png", flush=True)

# =============================================================================
# Slika B, pacijent po pacijent: ista podjela, svaki ispitanik u svojoj datoteci
#
# Okvir mozga nije jednak kod svih ispitanika, pa bi slike izašle različitih dimenzija, a
# anatomija bi na njima bila u različitom mjerilu. Sve se sličice zato nadopunjuju
# pozadinom do zajedničke veličine, platno se računa iz tog zajedničkog oblika, a zapis ide
# bez naknadnog obrezivanja, pa sve četiri datoteke izlaze jednake do piksela.
# =============================================================================
PANELS = ("t_sl", "s_sl", "t_mip", "s_mip")
PAD_H = max(D[c][k].shape[0] for c in CASES for k in PANELS)
PAD_W = max(D[c][k].shape[1] for c in CASES for k in PANELS)


def pad(image):
    top = (PAD_H - image.shape[0]) // 2
    left = (PAD_W - image.shape[1]) // 2
    return np.pad(image, ((top, PAD_H - image.shape[0] - top),
                          (left, PAD_W - image.shape[1] - left)))


WIDTH = 6.8
HEIGHT = 3.0 * (WIDTH * 0.49) * (PAD_W / PAD_H) + 1.35   # rot90 zamijeni osi
for c in CASES:
    d = D[c]
    fig, ax = plt.subplots(3, 2, figsize=(WIDTH, HEIGHT), facecolor="white")
    e_sl, e_mip = np.abs(d["t_sl"] - d["s_sl"]), np.abs(d["t_mip"] - d["s_mip"])
    ve = max(0.15, float(np.percentile(e_mip, 99.0)))
    show(ax[0, 0], pad(d["t_sl"]), d["w_sl"],
         "%s MRA GT, presjek\nSSIM %s PSNR %s"
         % (c, hr(d["syn_ssim"]), hr(d["syn_psnr"], 2)), size=8.5)
    show(ax[0, 1], pad(d["t_mip"]), d["w_mip"], "MRA GT, MIP projekcija\n")
    show(ax[1, 0], pad(d["s_sl"]), d["w_sl"], "MRA rekonstrukcija, presjek")
    show(ax[1, 1], pad(d["s_mip"]), d["w_mip"], "MRA rekonstrukcija, MIP projekcija")
    show(ax[2, 0], pad(e_sl), ve, "apsolutna pogreška, presjek", cmap="magma")
    show(ax[2, 1], pad(e_mip), ve, "apsolutna pogreška, MIP projekcija", cmap="magma")
    fig.tight_layout(h_pad=1.2, w_pad=0.4)
    fig.savefig(OUT / ("most_%s.png" % c), dpi=150, facecolor="white")
    plt.close(fig)
    print("zapisano ->", OUT / ("most_%s.png" % c), flush=True)
