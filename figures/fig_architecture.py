"""Architecture figure, drawn to journal conventions.

  (a) end-to-end pipeline
  (b) U-Net internals, with the attention levels marked

No equations and no schedule inset -- those belong in the running text. Conventions:
hairline strokes at one weight, square corners, greyscale structure with a single
functional accent, trapezoids for encoder/decoder, isometric slabs for tensors, shapes
annotated beside arrows rather than crammed inside boxes. Every label is placed clear of
every arrow; routes are chosen so no leader line crosses text.

Numbers read off the implementation (train_bbdm.py, train_ldm_v2.py, the run config):
conditioning 3 x 8 = 24 ch, target 16 ch, both 128 x 144 x 24; U-Net input 16 ch with no
concatenation; 48/96/192/192, attention at the two deepest levels; 46.27 M + 0.40 M par.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / "runs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#1a1a1a"
GREY = "#707070"
ACC = "#2f5f8f"
FILL = "#f0f0f0"
FILL2 = "#fbfbfb"
DASH = (0, (3.5, 2.0))
LW = 0.85

plt.rcParams.update({"font.family": "DejaVu Sans",
                     "mathtext.fontset": "stix",
                     "figure.dpi": 150})

fig = plt.figure(figsize=(13.2, 6.6), facecolor="white")
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.66], hspace=0.10,
                      left=0.015, right=0.985, top=0.965, bottom=0.035)
ax = fig.add_subplot(gs[0])
ax.set_xlim(0, 100)
ax.set_ylim(0, 38)
ax.set_axis_off()


# --------------------------------------------------------------------------- primitives
def rect(a, x, y, w, h, label=None, sub=None, fc="white", ec=INK, ls="solid",
         fs=8.0, fs_sub=6.4, lw=LW, z=3):
    a.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                          linewidth=lw, linestyle=ls, zorder=z))
    if label and sub:
        a.text(x + w / 2, y + h * 0.63, label, ha="center", va="center",
               fontsize=fs, color=INK, zorder=z + 1, linespacing=1.35)
        a.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center",
               fontsize=fs_sub, color=GREY, zorder=z + 1, linespacing=1.45)
    elif label:
        a.text(x + w / 2, y + h / 2, label, ha="center", va="center",
               fontsize=fs, color=INK, zorder=z + 1, linespacing=1.4)


def trap(a, x, y, w, h, taper=0.32, flip=False, label=None, sub=None, fc=FILL,
         ls="solid"):
    d = h * taper / 2
    pts = ([(x, y), (x + w, y + d), (x + w, y + h - d), (x, y + h)] if not flip
           else [(x, y + d), (x + w, y), (x + w, y + h), (x, y + h - d)])
    a.add_patch(Polygon(pts, closed=True, facecolor=fc, edgecolor=INK,
                        linewidth=LW, linestyle=ls, zorder=3))
    if label:
        a.text(x + w / 2, y + h * (0.61 if sub else 0.5), label, ha="center",
               va="center", fontsize=7.4, color=INK, zorder=4, linespacing=1.35)
    if sub:
        a.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center",
               fontsize=6.2, color=GREY, zorder=4)


def slab(a, x, y, w, h, d=1.4, ec=INK, ls="solid"):
    a.add_patch(Polygon([(x, y + h), (x + d, y + h + d), (x + w + d, y + h + d),
                         (x + w, y + h)], closed=True, facecolor="#e9e9e9",
                        edgecolor=ec, linewidth=LW * 0.8, linestyle=ls, zorder=3))
    a.add_patch(Polygon([(x + w, y), (x + w + d, y + d), (x + w + d, y + h + d),
                         (x + w, y + h)], closed=True, facecolor="#dedede",
                        edgecolor=ec, linewidth=LW * 0.8, linestyle=ls, zorder=3))
    a.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor=ec,
                          linewidth=LW, linestyle=ls, zorder=4))


def arr(a, x1, y1, x2, y2, ec=INK, ls="solid", lw=LW, head=7.0, z=2):
    a.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                mutation_scale=head, linewidth=lw, color=ec,
                                linestyle=ls, shrinkA=1.0, shrinkB=1.0, zorder=z))


def line(a, x1, y1, x2, y2, ec=INK, ls="solid", lw=LW, z=2):
    a.plot([x1, x2], [y1, y2], color=ec, linestyle=ls, lw=lw, zorder=z,
           solid_capstyle="butt")


def shp(a, x, y, t, fs=6.2, color=GREY, ha="center", va="center"):
    a.text(x, y, t, ha=ha, va=va, fontsize=fs, color=color, zorder=6)


# ================================================================= (a) pipeline
ax.text(0.0, 37.6, "(a)", ha="left", va="top", fontsize=9.5, weight="bold", color=INK)

for lab, yy in (("T1", 28.6), ("T2", 23.6), ("PD", 18.6)):
    rect(ax, 1.5, yy, 6.0, 4.2, lab, fs=8.2)
shp(ax, 4.5, 16.8, "512×576×96")
for yy in (30.7, 25.7, 20.7):
    arr(ax, 7.8, yy, 11.3, 25.8)

rect(ax, 1.5, 5.4, 6.0, 4.2, "MRA", fs=8.2, ls=DASH)
shp(ax, 4.5, 3.6, "512×576×96")
arr(ax, 7.8, 7.5, 11.3, 7.5, ls=DASH)

trap(ax, 11.5, 19.2, 8.2, 13.2, label="višemodalni\nVAE enkoder", sub="zamrznut")
trap(ax, 11.5, 3.4, 8.2, 8.2, label="MRA VAE\nenkoder", sub="zamrznut", ls=DASH)
arr(ax, 19.9, 25.8, 23.3, 25.8)
arr(ax, 19.9, 7.5, 23.3, 7.5, ls=DASH)

slab(ax, 23.5, 22.0, 6.0, 7.4)
ax.text(26.5, 25.7, "$c$", ha="center", va="center", fontsize=9.5, zorder=6, color=INK)
shp(ax, 26.9, 20.0, "24 × 128×144×24")
slab(ax, 23.5, 4.3, 6.0, 6.2)
ax.text(26.5, 7.4, "$x_0$", ha="center", va="center", fontsize=9.5, zorder=6, color=INK)
shp(ax, 26.9, 2.3, "16 × 128×144×24")

# conditioning branches
arr(ax, 31.3, 27.4, 34.3, 30.2)
rect(ax, 34.5, 27.4, 8.6, 6.6, "enkoder\nuvjeta", fs=7.4, fc=FILL2)
arr(ax, 31.3, 23.8, 34.3, 20.6)
rect(ax, 34.5, 13.6, 8.6, 7.0, "$P(c)$", sub="24→16 kan.\n0,40 M", fs=9.0,
     ec=ACC, fc=FILL2)
arr(ax, 43.1, 17.1, 46.2, 17.1, ec=ACC)
slab(ax, 46.4, 14.2, 3.8, 5.8, ec=ACC)
ax.text(48.3, 17.1, "$y$", ha="center", va="center", fontsize=9.5, color=ACC, zorder=6)
arr(ax, 51.6, 17.1, 55.3, 17.1, ec=ACC)

# AdaGN: a short clear route into the top of the U-Net, label set above its own line
line(ax, 43.1, 30.7, 65.5, 30.7, ec=GREY)
arr(ax, 65.5, 30.7, 65.5, 22.4, ec=GREY)
ax.text(54.0, 31.4, "AdaGN (po razini)", ha="center", va="bottom", fontsize=6.8,
        color=GREY, style="italic")

# x0 enters from below; the run is well clear of the shape captions above it
line(ax, 31.3, 7.5, 60.0, 7.5, ec=INK, ls=DASH)
arr(ax, 60.0, 7.5, 60.0, 11.3, ec=INK, ls=DASH)

rect(ax, 55.5, 11.5, 20.0, 10.8, "3D U-Net",
     sub="AdaGN uvjetovanje  ·  46,27 M par.\nulaz 16 kanala, bez ulančavanja\ndetalj u (b)",
     fs=9.2, fs_sub=6.6, lw=1.05)

# decode
arr(ax, 75.5, 16.9, 78.3, 16.9)
slab(ax, 78.5, 13.9, 4.0, 6.0)
ax.text(80.5, 16.9, r"$\hat{x}_0$", ha="center", va="center", fontsize=9.5,
        color=INK, zorder=6)
arr(ax, 83.9, 16.9, 86.0, 16.9)
trap(ax, 86.2, 12.2, 7.2, 9.4, flip=True, label="MRA VAE\ndekoder", sub="zamrznut")
arr(ax, 93.4, 16.9, 95.0, 16.9)
slab(ax, 95.2, 14.0, 3.2, 5.8)
shp(ax, 96.2, 11.6, "512×576×96")
shp(ax, 96.2, 23.0, "sintetizirani\nMRA", fs=6.8, color=INK)

# legend, parked in empty space under the decode chain
ax.add_patch(Rectangle((78.0, 6.6), 3.4, 1.7, fc=FILL, ec=INK, lw=LW))
ax.text(82.4, 7.4, "zamrznuto", fontsize=6.5, color=GREY, va="center")
line(ax, 78.0, 4.6, 81.4, 4.6)
ax.text(82.4, 4.6, "treniranje i uzorkovanje", fontsize=6.5, color=GREY, va="center")
line(ax, 78.0, 2.6, 81.4, 2.6, ls=DASH)
ax.text(82.4, 2.6, "samo pri treniranju", fontsize=6.5, color=GREY, va="center")

# ================================================================= (b) U-Net
axu = fig.add_subplot(gs[1])
axu.set_xlim(0, 100)
axu.set_ylim(0, 100)
axu.set_axis_off()
axu.text(0.0, 100.0, "(b)", ha="left", va="top", fontsize=9.5, weight="bold", color=INK)

BW = 9.0
SHAPES = ["128×144×24", "64×72×12", "32×36×6", "16×18×3"]
CH = [48, 96, 192, 192]
ATT = [False, False, True, True]
TOP = [78.0, 64.0, 52.0, 42.0]
HT = [28.0, 21.0, 15.0, 10.0]
ENC = [6.0, 20.0, 34.0, 48.0]
DEC = {2: 60.0, 1: 74.0, 0: 88.0}


def mid(k):
    return TOP[k] - HT[k] / 2


def block(x, k, tag=True):
    rect(axu, x, TOP[k] - HT[k], BW, HT[k], fc=FILL2)
    axu.text(x + BW / 2, mid(k), str(CH[k]), ha="center", va="center",
             fontsize=7.2, color=INK, rotation=90, zorder=5)
    if ATT[k]:
        axu.scatter([x + BW / 2], [TOP[k] - HT[k] - 4.2], s=14, marker="*",
                    color=ACC, zorder=6)
    if tag:
        axu.text(x + BW / 2, TOP[k] + 9.0, SHAPES[k], ha="center", va="center",
                 fontsize=6.2, color=GREY)


for k, x in enumerate(ENC):
    block(x, k, tag=True)
for k, x in DEC.items():
    block(x, k, tag=False)

for k in range(3):
    arr(axu, ENC[k] + BW, mid(k), ENC[k + 1], mid(k + 1))
arr(axu, ENC[3] + BW, mid(3), DEC[2], mid(2))
arr(axu, DEC[2] + BW, mid(2), DEC[1], mid(1))
arr(axu, DEC[1] + BW, mid(1), DEC[0], mid(0))

for k in (0, 1, 2):                      # skips routed above each level, under its tag
    yy = TOP[k] + 4.2
    line(axu, ENC[k] + BW / 2, TOP[k], ENC[k] + BW / 2, yy, ec=GREY, ls=DASH,
         lw=0.6, z=1)
    line(axu, ENC[k] + BW / 2, yy, DEC[k] + BW / 2, yy, ec=GREY, ls=DASH, lw=0.6, z=1)
    arr(axu, DEC[k] + BW / 2, yy, DEC[k] + BW / 2, TOP[k], ec=GREY, ls=DASH,
        lw=0.6, head=5.5, z=1)

arr(axu, 1.0, mid(0), 5.5, mid(0))
arr(axu, DEC[0] + BW, mid(0), 99.0, mid(0))
axu.text(0.6, mid(0) + 6.0, "$x_t$", fontsize=7.8, color=INK, va="center")
axu.text(99.4, mid(0) + 6.0, "$x_t - x_0$", fontsize=7.8, color=INK, va="center",
         ha="right")

axu.scatter([34.0], [5.0], s=14, marker="*", color=ACC)
axu.text(36.5, 5.0, "samopažnja", fontsize=6.8, color=ACC, va="center")
line(axu, 56.0, 5.0, 62.0, 5.0, ec=GREY, ls=DASH, lw=0.6)
axu.text(63.5, 5.0, "veze preskakivanja", fontsize=6.8, color=GREY, va="center")

fig.savefig(OUT / "arhitektura.pdf", bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "arhitektura.png", dpi=400, bbox_inches="tight", facecolor="white")

CAPTION = (
    "Slika 4.1. Arhitektura predloženog modela. (a) Cjelokupni tijek: T1, T2 i PD "
    "propuštaju se kroz zamrznuti višemodalni varijacijski autoenkoder u uvjetni latentni "
    "prikaz c (24 kanala), a MRA kroz vlastiti autoenkoder u ciljni latentni prikaz x0 "
    "(16 kanala). Isprekidane linije označavaju put koji postoji samo pri treniranju — "
    "pri uzorkovanju se MRA enkoder ne poziva. Projektor P(c) preslikava uvjet u ciljni "
    "latentni prostor i daje krajnju točku mosta y, pa uzorkovanje počinje od te procjene, "
    "a ne od šuma. (b) Unutarnja struktura 3D U-Net mreže; uvjet ulazi AdaGN modulacijom "
    "na svakoj razini, a ne ulančavanjem na ulazu.")
(OUT / "arhitektura_potpis.txt").write_text(CAPTION, encoding="utf-8")

print("wrote", OUT / "arhitektura.pdf")
print("wrote", OUT / "arhitektura.png")
print("wrote", OUT / "arhitektura_potpis.txt")
