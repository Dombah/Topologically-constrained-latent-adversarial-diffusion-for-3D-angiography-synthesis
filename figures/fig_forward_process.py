"""Methods figure: standard diffusion vs the Brownian bridge forward process.

Both schedules are the ones the project actually trains with, reproduced from the source:
  cosine ᾱ_t        -- train_ldm_v2.GuidedDiffusion.__init__
  m_t, δ_t          -- train_bbdm.BrownianBridge.__init__   (δ_t = 2s(m_t − m_t²), s = 1)

The panels draw the MARGINAL q(x_t | ·) they are labelled with: mean ± 1σ and ± 2σ. Two
inputs are shown per panel, because the whole difference is what happens to them at t = T --
standard diffusion sends every input to the same N(0, I); the bridge sends each to its own
endpoint y = P(cond).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "runs" / "figures"; OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE = "#2a78d6", "#eb6834"          # validated categorical slots 1 and 2
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dedddaa0"
STEPS = 1000
CASE_A, CASE_B = (1.05, 0.45), (-0.65, -0.30)     # (x_0, y) for two example cases

# --- the two schedules, verbatim from the training code -----------------------------
tt = np.linspace(0, STEPS, STEPS + 1) / STEPS
f = np.cos((tt + 0.008) / 1.008 * np.pi / 2) ** 2
abar = np.clip((f / f[0])[1:], 1e-8, 0.9999)          # ᾱ_t, cosine
sqrt_a, sd_diff = np.sqrt(abar), np.sqrt(1.0 - abar)

m = np.arange(STEPS) / (STEPS - 1)                     # m_t, linear
delta = 2.0 * (m - m * m)                              # δ_t, s = 1
sd_bridge = np.sqrt(delta)
u = np.linspace(0, 1, STEPS)

# --- figure ---------------------------------------------------------------------------
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
                     "font.family": "DejaVu Sans", "figure.dpi": 160})
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10.6, 3.4),
                                    gridspec_kw={"width_ratios": [1, 1, 0.9], "wspace": 0.30})
for ax in (ax1, ax2, ax3):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#b8b7b3")
    ax.tick_params(colors=INK2, length=3)
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_xlim(0, 1); ax.set_xlabel("t / T", color=INK2)


def envelope(ax, mean, sd, color, primary):
    """±2σ then ±1σ, so the corridor reads as a distribution rather than a line."""
    ax.fill_between(u, mean - 2 * sd, mean + 2 * sd, color=color,
                    alpha=0.13 if primary else 0.07, lw=0)
    ax.fill_between(u, mean - sd, mean + sd, color=color,
                    alpha=0.22 if primary else 0.10, lw=0)
    ax.plot(u, mean, color=color, lw=2.1 if primary else 1.3,
            ls="-" if primary else (0, (4, 2.2)), alpha=1.0 if primary else 0.85)


# (a) standard diffusion -- both inputs land on the same N(0, I)
for (x0, _), primary in ((CASE_A, True), (CASE_B, False)):
    envelope(ax1, sqrt_a * x0, sd_diff, BLUE, primary)
for x0, _ in (CASE_A, CASE_B):
    ax1.plot([0.0], [x0], "o", color=BLUE, ms=5.5, zorder=5)
ax1.plot([1.0], [0.0], "o", color=BLUE, ms=7, zorder=6)
ax1.set_ylim(-2.55, 2.55)
ax1.set_ylabel("latent value", color=INK2)
ax1.set_title("(a)  Standard diffusion   $q(x_t\\,|\\,x_0)$", color=INK, loc="left", pad=9)
ax1.text(0.035, CASE_A[0] + 0.22, "$x_0$", color=INK, fontsize=10)
ax1.text(0.035, CASE_B[0] - 0.52, "$x_0'$", color=INK, fontsize=10)
ax1.annotate("$\\mathcal{N}(0, I)$\nboth inputs, one terminal", (1.0, 0.0), xytext=(0.30, -2.10),
             color=INK, fontsize=8.5, ha="left",
             arrowprops=dict(arrowstyle="->", color=INK2, lw=0.9,
                             connectionstyle="arc3,rad=-0.3"))
ax1.text(0.5, 2.18, "variance climbs to 1 — the signal is destroyed",
         color=INK2, fontsize=8, ha="center", style="italic")

# (b) brownian bridge -- each input lands on its own endpoint
for (x0, y), primary in ((CASE_A, True), (CASE_B, False)):
    envelope(ax2, (1 - m) * x0 + m * y, sd_bridge, ORANGE, primary)
for x0, y in (CASE_A, CASE_B):
    ax2.plot([0.0, 1.0], [x0, y], "o", color=ORANGE, ms=5.5, zorder=5)
ax2.set_ylim(-2.55, 2.55)
ax2.set_title("(b)  Brownian bridge   $q(x_t\\,|\\,x_0, y)$", color=INK, loc="left", pad=9)
ax2.text(0.035, CASE_A[0] + 0.22, "$x_0$", color=INK, fontsize=10)
ax2.text(0.035, CASE_B[0] - 0.52, "$x_0'$", color=INK, fontsize=10)
ax2.annotate("$y = P(\\mathrm{cond})$", (1.0, CASE_A[1]), xytext=(0.60, 1.62),
             color=INK, fontsize=9, ha="left",
             arrowprops=dict(arrowstyle="->", color=INK2, lw=0.9,
                             connectionstyle="arc3,rad=0.25"))
ax2.annotate("$y'$", (1.0, CASE_B[1]), xytext=(0.86, -1.30), color=INK, fontsize=9,
             arrowprops=dict(arrowstyle="->", color=INK2, lw=0.9,
                             connectionstyle="arc3,rad=-0.25"))
ax2.text(0.5, 2.18, "pinned at both ends — each input keeps its own",
         color=INK2, fontsize=8, ha="center", style="italic")

# (c) the variance schedules -- the whole difference in one line each
ax3.plot(u, 1.0 - abar, color=BLUE, lw=2.1)
ax3.plot(u, delta, color=ORANGE, lw=2.1)
ax3.set_ylim(-0.035, 1.10); ax3.set_ylabel("forward variance", color=INK2)
ax3.set_title("(c)  Variance schedule", color=INK, loc="left", pad=9)
ax3.text(0.055, 0.86, "$1-\\bar{\\alpha}_t$", color=BLUE, fontsize=10.5, fontweight="bold")
ax3.text(0.055, 0.30, "$\\delta_t$", color=ORANGE, fontsize=10.5, fontweight="bold")
ax3.text(0.30, 0.545, "$2s(m_t-m_t^2)$", color=ORANGE, fontsize=8.5)
ax3.plot([1.0], [0.0], "o", color=ORANGE, ms=6.5, zorder=5)
ax3.plot([1.0], [1.0 - abar[-1]], "o", color=BLUE, ms=6.5, zorder=5)
ax3.annotate("$\\delta_T = 0$", (1.0, 0.0), xytext=(0.60, 0.14), color=INK2, fontsize=9,
             arrowprops=dict(arrowstyle="->", color=INK2, lw=0.9))

fig.subplots_adjust(left=0.062, right=0.985, top=0.87, bottom=0.155)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"forward_process.{ext}", bbox_inches="tight",
                facecolor="white", dpi=(200 if ext == "png" else None))
print("wrote", OUT / "forward_process.pdf", "and .png")
print(f"check  1-abar[T] = {1-abar[-1]:.4f} (expect ~1)   delta[T] = {delta[-1]:.6f} (expect 0)")
print(f"       delta.max = {delta.max():.4f} at m={m[delta.argmax()]:.3f} (expect 0.5 at 0.5)")
print(f"       bridge sd at both ends = {sd_bridge[0]:.2e}, {sd_bridge[-1]:.2e} (expect 0, 0)")
