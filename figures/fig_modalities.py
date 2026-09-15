"""Slika 3.1: T1, T2 i PD isti presjek istog ispitanika iz skupa IXI.

Stara slika 3.1 bila je preuzeta s interneta i prikazivala je FLAIR, kojega u skupu IXI nema.
Nova je izrađena iz podataka rada: validacijski ispitanik IXI016-Guys-0697 (isti kao na
slikama 5.1 i 5.2), aksijalni presjek z = 55 na razini lateralnih ventrikula. Podaci su nakon
predobrade, dakle bez lubanje i poravnati s MRA volumenom.

Raspored je isti kao na staroj slici, bez natpisa na samoj slici: tri presjeka s lijeva na
desno na crnoj pozadini. Omjer stranica jednak je staroj slici (858 x 344), da se okvir u
dokumentu ne razvuče; slika se sprema u dvostrukoj razlučivosti.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))   # so `modules` imports when run as uv run figures/<script>.py
CASE, Z = "IXI016-Guys-0697", 55
OUT = ROOT / "runs" / "figures" / "slika31_T1_T2_PD.png"
W, H = 1716, 688                      # 2 x 858 x 344
OUTER, GAP, MARGIN = 10, 14, 8

slices = []
for mod in ("T1", "T2", "PD"):
    vol = np.asarray(np.load(ROOT / "Dataset/split_numpy/val" / mod / ("%s-%s.npy" % (CASE, mod))),
                     np.float32)
    mask = np.asarray(np.load(ROOT / "Dataset/split_numpy/val/masks" / mod /
                              ("%s-%s_mask.npy" % (CASE, mod)))) > 0
    v, m = vol[..., Z], mask[..., Z]
    lo, hi = np.percentile(v[m], (0.5, 99.5))
    img = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
    img[~m] = 0.0                     # pozadina crna, kao na izvornoj slici
    slices.append((np.rot90(img), np.rot90(m)))

# zajednički okvir oko mozga, da sva tri presjeka budu jednako velika i poravnata
union = np.any([m for _, m in slices], axis=0)
rows, cols = np.nonzero(union.any(1))[0], np.nonzero(union.any(0))[0]
r0, r1 = max(rows.min() - MARGIN, 0), min(rows.max() + MARGIN + 1, union.shape[0])
c0, c1 = max(cols.min() - MARGIN, 0), min(cols.max() + MARGIN + 1, union.shape[1])

cell_w, cell_h = (W - 2 * OUTER - 2 * GAP) // 3, H - 2 * OUTER
canvas = Image.new("L", (W, H), 0)
for k, (img, _) in enumerate(slices):
    crop = Image.fromarray((img[r0:r1, c0:c1] * 255).astype(np.uint8))
    scale = min(cell_w / crop.width, cell_h / crop.height)
    crop = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.LANCZOS)
    x = OUTER + k * (cell_w + GAP) + (cell_w - crop.width) // 2
    y = OUTER + (cell_h - crop.height) // 2
    canvas.paste(crop, (x, y))
canvas.convert("RGB").save(OUT)
print("zapisano ->", OUT, canvas.size)
