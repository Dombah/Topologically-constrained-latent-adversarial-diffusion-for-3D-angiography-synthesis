"""MedicalNet FID za trenutni MRA varijacijski autoenkoder, na validacijskom skupu.

Postojeća bilježnica zna učitati samo VAEv3, pa se ovdje ponavlja isti mjerni postupak, ali
s v4/v5 arhitekturom. Sve što utječe na vrijednost preuzeto je doslovno da brojka ostane
usporediva s ranije objavljenom: MedicalNet resnet10 (23 skupa podataka), značajke iz
layer4 uz globalno sažimanje, ulaz preuzorkovan na 96 x 224 x 224 trolinearno, z-normalizacija
po prednjem planu izvornog volumena uz odrezivanje na 6 sigma, te Frechetova udaljenost sa
skupljanjem kovarijance 0,1.

Dvije namjerne razlike od ranijeg izračuna, obje navedene u ispisu:
  - samo validacijski skup, a ne svih 568 volumena, pa testni skup ostaje nedirnut
  - rekonstrukcija izravnim prolazom kroz cijeli volumen, jer su i sve ostale v5 brojke
    u radu mjerene tako
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))   # so `modules` imports when run as uv run analysis/<script>.py
from modules.paths import volume_paths
from modules.vae_mra import load_vaev4, reconstruct

DEV = torch.device("cuda")
MN_ROOT = ROOT / "third_party" / "MedicalNet"
MN_WEIGHT = MN_ROOT / "resnet_10_23dataset.pth"
MN_INPUT = (96, 224, 224)
DEPTH, SHORTCUT = 10, "B"
CLIP, ALPHA = 6.0, 0.10
CKPT = ROOT / "checkpoints/vaev5_mra/v5_cldice/best.pt"
OUT = ROOT / "runs" / "medicalnet_fid" / "v5_cldice_val"
OUT.mkdir(parents=True, exist_ok=True)


class Layer4GAP(nn.Module):
    def __init__(self, m):
        super().__init__()
        for name in ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3",
                     "layer4"):
            setattr(self, name, getattr(m, name))

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return F.adaptive_avg_pool3d(x, 1).flatten(1)


def load_extractor():
    if not hasattr(torch.nn.init, "kaiming_normal"):
        torch.nn.init.kaiming_normal = torch.nn.init.kaiming_normal_
    spec = importlib.util.spec_from_file_location("mn_resnet", MN_ROOT / "models" / "resnet.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = getattr(mod, "resnet%d" % DEPTH)(
        sample_input_D=MN_INPUT[0], sample_input_H=MN_INPUT[1], sample_input_W=MN_INPUT[2],
        num_seg_classes=2, shortcut_type=SHORTCUT, no_cuda=False)
    blob = torch.load(MN_WEIGHT, map_location="cpu")
    src = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
    tgt = model.state_dict()
    matched = {}
    for k, v in src.items():
        k = str(k)
        for p in ("module.", "model."):
            if k.startswith(p):
                k = k[len(p):]
        if k in tgt and tuple(v.shape) == tuple(tgt[k].shape):
            matched[k] = v
    need = {k for k in tgt if k.startswith(("conv1.", "bn1.", "layer1.", "layer2.",
                                            "layer3.", "layer4."))
            and not k.endswith("num_batches_tracked")}
    missing = sorted(need - set(matched))
    if missing:
        raise SystemExit("nepotpuno učitane težine: %d nedostaje" % len(missing))
    model.load_state_dict({**tgt, **matched}, strict=True)
    ex = Layer4GAP(model).to(DEV).eval()
    for p in ex.parameters():
        p.requires_grad_(False)
    return ex


def prepare_pair(real_xyz, recon_xyz):
    """Preuzorkovanje i normalizacija, doslovno kao u ranijem izračunu."""
    pair = np.stack([real_xyz, recon_xyz], 0).astype(np.float32)
    dhw = torch.from_numpy(pair.transpose(0, 3, 2, 1)).unsqueeze(1).to(DEV)
    resized = F.interpolate(dhw, size=MN_INPUT, mode="trilinear", align_corners=False)
    mask = torch.from_numpy((real_xyz > 0.0).astype(np.float32))
    mask = mask.permute(2, 1, 0)[None, None].to(DEV)
    sup = F.interpolate(mask, size=MN_INPUT, mode="nearest") > 0.5
    if int(sup.sum()) < 128:
        raise RuntimeError("prazan prednji plan nakon preuzorkovanja")
    out = []
    for i in range(2):
        v = resized[i, 0]
        vals = v[sup[0, 0]]
        out.append(((v - vals.mean()) / vals.std(unbiased=False).clamp_min(1e-6))
                   .clamp(-CLIP, CLIP))
    return torch.stack(out, 0).unsqueeze(1)


def shrink(cov, alpha=ALPHA):
    scale = float(np.trace(cov) / cov.shape[0])
    return (1 - alpha) * cov + alpha * scale * np.eye(cov.shape[0])


def trace_sqrt(a, b):
    a = (a + a.T) * 0.5
    b = (b + b.T) * 0.5
    wa, va = np.linalg.eigh(a)
    sa = (va * np.sqrt(np.clip(wa, 0, None))[None, :]) @ va.T
    mid = sa @ b @ sa
    mid = (mid + mid.T) * 0.5
    w = np.linalg.eigvalsh(mid)
    tol = max(1e-10, 1e-8 * float(np.max(np.abs(w))))
    if float(w.min()) < -tol:
        raise RuntimeError("negativna svojstvena vrijednost: %g" % w.min())
    return float(np.sqrt(np.clip(w, 0, None)).sum())


def fid(real, gen, alpha=None):
    mu_r, mu_g = real.mean(0), gen.mean(0)
    cr = np.cov(real, rowvar=False, ddof=1)
    cg = np.cov(gen, rowvar=False, ddof=1)
    if alpha is not None:
        cr, cg = shrink(cr, alpha), shrink(cg, alpha)
    return max(0.0, float(np.square(mu_r - mu_g).sum())
               + float(np.trace(cr) + np.trace(cg) - 2.0 * trace_sqrt(cr, cg)))


ex = load_extractor()
vae = load_vaev4(CKPT, DEV)
paths = volume_paths("val")
print("MedicalNet resnet%d  |  volumena: %d  |  model: %s"
      % (DEPTH, len(paths), CKPT.relative_to(ROOT)), flush=True)

R, G = [], []
t0 = time.time()
for i, p in enumerate(paths, 1):
    real = np.asarray(np.load(p), np.float32)
    vol = torch.from_numpy(real)[None, None].to(DEV)
    with torch.inference_mode():
        recon, _ = reconstruct(vae, vol, "direct")
        batch = prepare_pair(real, np.clip(recon, 0.0, 1.0))
        feats = ex(batch)
    R.append(feats[0].float().cpu().numpy())
    G.append(feats[1].float().cpu().numpy())
    del vol
    torch.cuda.empty_cache()
    if i % 10 == 0 or i == len(paths):
        print("   %d/%d  [%.1f min]" % (i, len(paths), (time.time() - t0) / 60), flush=True)

R, G = np.stack(R).astype(np.float64), np.stack(G).astype(np.float64)
res = {"model": str(CKPT.relative_to(ROOT)), "split": "val", "cases": len(paths),
       "feature_dim": int(R.shape[1]),
       "fid_empirical": fid(R, G), "fid_shrinkage": fid(R, G, ALPHA),
       "paired_feature_l2": float(np.mean(np.linalg.norm(R - G, axis=1)
                                          / np.maximum(np.linalg.norm(R, axis=1), 1e-8))),
       "recon_mode": "direct", "shrinkage_alpha": ALPHA}
(OUT / "fid.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
print("\nFID (empirijski)  %.6f" % res["fid_empirical"])
print("FID (sa skupljanjem) %.6f" % res["fid_shrinkage"])
print("relativna razlika značajki %.5f" % res["paired_feature_l2"])
print("zapisano ->", OUT / "fid.json", flush=True)
