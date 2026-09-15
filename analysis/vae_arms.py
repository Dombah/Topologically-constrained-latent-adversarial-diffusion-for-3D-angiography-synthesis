"""Re-score the finished MRA VAE arms on the FULL validation split (57 volumes).

Two changes from the shipped ablation:

1. All 57 validation volumes instead of the 23 that pick_validation_cases("val", 8)
   selects. Differences between arms are ~0.002 vessel Dice; n=23 is thin for that.

2. SSIM reported under BOTH parameterisations, because they answer different questions:

     default    skimage's uniform 7-window with sample covariance -- what every number
                recorded in this project so far used, kept so history stays comparable
     canonical  gaussian_weights=True, sigma=1.5, use_sample_covariance=False -- the
                Wang et al. 2004 formulation that papers report, which skimage documents
                as reproducing the original MATLAB implementation

   and each of those globally AND inside the brain mask. Global SSIM on these volumes is
   inflated because ~67 % of each is empty background reproduced near-perfectly; the
   masked figure is the one that reflects reconstruction quality where it matters. Both
   are kept so the thesis can report the comparable number and the honest one.

Nothing is retrained and nothing touches the test split. Results go to a NEW file so the
23-case record stays intact.
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
sys.path.insert(0, str(ROOT))
import modules.vae_mra as T   # evaluate() looks these metric functions up here
from modules.paths import volume_paths, pick_validation_cases
from modules.vae_mra import load_vaev4, evaluate
from modules.vae_mra import evaluate_perturbed

DEV = "cuda"
OUT = ROOT / "runs" / "vae_arms" / "ablation_full_val.json"   # thesis copy: checkpoints/vaev5_mra/
OUT.parent.mkdir(parents=True, exist_ok=True)
SKELETON_ALL = True   # clDice on every volume, not just the first few
CANON = dict(gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
INNER = (slice(3, -3),) * 3

ARMS = [
    ("v4 (none)", ROOT / "checkpoints/vaev4_mra/v4_c16_fixed/best.pt"),
    ("cldice",    ROOT / "checkpoints/vaev5_mra/v5_cldice/best.pt"),
    ("both",      ROOT / "checkpoints/vaev5_mra/v5_both/best.pt"),
    ("adv",       ROOT / "checkpoints/vaev5_mra/v5_adv/best.pt"),
]

_extra = []
_orig_ssim = T.masked_ssim_and_psnr
_orig_mip = T.mip_ssim


def _patched_ssim(target, recon, brain):
    """Returns the default tuple unchanged so evaluate() behaves as before, and stashes
    the canonical parameterisation alongside it."""
    recon_c = np.clip(recon, 0.0, 1.0).astype(np.float32)
    target_c = target.astype(np.float32)
    g_def, map_def = structural_similarity(target_c, recon_c, data_range=1.0, full=True)
    g_can, map_can = structural_similarity(target_c, recon_c, data_range=1.0, full=True,
                                           **CANON)
    sel = brain[INNER]
    m_def = float(map_def[INNER][sel].mean()) if sel.any() else float("nan")
    m_can = float(map_can[INNER][sel].mean()) if sel.any() else float("nan")
    mse = float(np.mean((target_c - recon_c) ** 2))
    psnr = 10.0 * math.log10(1.0 / max(mse, 1e-12))
    _extra.append({"ssim_global_default": float(g_def), "ssim_masked_default": m_def,
                   "ssim_global_canonical": float(g_can), "ssim_masked_canonical": m_can})
    return float(g_def), m_def, psnr


def _patched_mip(target, recon):
    recon_c = np.clip(recon, 0.0, 1.0)
    d = float(np.mean([structural_similarity(target.max(a), recon_c.max(a), data_range=1.0)
                       for a in range(3)]))
    c = float(np.mean([structural_similarity(target.max(a), recon_c.max(a), data_range=1.0,
                                             **CANON) for a in range(3)]))
    if _extra:
        _extra[-1]["mip_ssim_default"] = d
        _extra[-1]["mip_ssim_canonical"] = c
    return d


T.masked_ssim_and_psnr = _patched_ssim
T.mip_ssim = _patched_mip

cases = volume_paths("val")
perturb = pick_validation_cases("val", 2)[:6]
print("validation volumes: %d   perturbation cases: %d" % (len(cases), len(perturb)),
      flush=True)

results = {}
for name, ckpt in ARMS:
    if not ckpt.exists():
        print("SKIP %s -- no checkpoint at %s" % (name, ckpt), flush=True)
        continue
    t0 = time.time()
    _extra.clear()
    model = load_vaev4(ckpt, DEV)
    # clDice is skipped past skeleton_cases, so the default of 8 would average it over
    # a different (and site-skewed) sample than every other metric in the same table.
    summary = evaluate(model, cases, "val", DEV,
                       skeleton_cases=len(cases) if SKELETON_ALL else 8)
    summary.update(evaluate_perturbed(model, perturb, "val", DEV))

    # evaluate() loops regimes ("tiled", "direct") inside the case loop, so calls
    # alternate. Verified by count before being trusted.
    n = len(cases)
    if len(_extra) == 2 * n:
        for i, regime in enumerate(("tiled", "direct")):
            sl = _extra[i::2]
            for k in sl[0]:
                summary["%s_%s" % (regime, k)] = float(np.mean(
                    [r[k] for r in sl if k in r]))
    else:
        print("  WARNING: %d ssim calls for %d cases; pooling instead of splitting regimes"
              % (len(_extra), n), flush=True)
        for k in _extra[0]:
            summary["pooled_%s" % k] = float(np.mean([r[k] for r in _extra if k in r]))

    summary["arm"] = name
    summary["validation_cases"] = len(cases)
    summary["perturb_cases"] = len(perturb)
    summary["checkpoint"] = str(ckpt)
    results[name] = summary
    del model
    torch.cuda.empty_cache()
    print("%-12s %.1f min | PSNR %.3f | SSIM glob %.4f canon %.4f | masked %.4f canon "
          "%.4f | Dice %.4f | clDice %.4f" % (
              name, (time.time() - t0) / 60, summary["direct_psnr"],
              summary.get("direct_ssim_global_default", float("nan")),
              summary.get("direct_ssim_global_canonical", float("nan")),
              summary.get("direct_ssim_masked_default", float("nan")),
              summary.get("direct_ssim_masked_canonical", float("nan")),
              summary["direct_vessel_dice"], summary["direct_cldice"]), flush=True)
    OUT.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")

T.masked_ssim_and_psnr = _orig_ssim
T.mip_ssim = _orig_mip
print("\nwrote", OUT, flush=True)
