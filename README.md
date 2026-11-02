# Topologically constrained latent adversarial diffusion for 3D angiography synthesis

**Generating 3D brain angiography from routine MRI.**

MSc thesis · FERIT Osijek, 2026 · graded 5/5 · *"Topološki ograničena latentna suparnička difuzija za sintezu 3D angiografije"*
<!-- Add the thesis PDF link here once it's online. -->

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![tests](https://img.shields.io/badge/tests-94%20pytest-0A9EDC?style=flat&logo=pytest&logoColor=white)
![uv](https://img.shields.io/badge/env-uv-DE5FE9?style=flat)

<!-- Hero image: run `uv run figures/fig_test_synthesis.py`, copy one of the
     runs/figures/most_test_<case>.png files to docs/test_synthesis.png, commit it, and uncomment.
<p align="center">
  <img src="docs/test_synthesis.png" width="90%" alt="Ground-truth vs. synthesised TOF-MRA, maximum intensity projections">
</p>
-->

**The problem.** Time-of-flight MR angiography (TOF-MRA) shows the brain's blood vessels. It is a separate scan, though, and many patients never get one. This project **synthesises a full 3D TOF-MRA volume from the T1, T2 and PD scans** a patient already has.

**Why it's hard.** Vessels fill a tiny fraction of the volume and are often thinner than a voxel. A model can score well on image metrics while breaking vessels apart or dropping them altogether. That is why this project measures vessel *topology* as well as pixel similarity.

## 🧠 How it works

```
T1/T2/PD ─► multimodal VAE ─► 24-ch condition ─┬─► bridge projector ─► Brownian bridge, 5 steps ─► 16-ch MRA latent ─► MRA VAE decoder ─► TOF-MRA
                                               │
MRA ──────► MRA VAE (v5, clDice) ─► 16-ch target (training only)
```

1. **Two 3D VAEs** compress the inputs and the target. The multimodal VAE encodes T1/T2/PD into a 24-channel condition. The MRA VAE, fine-tuned with clDice so the latents keep vessels intact, encodes TOF-MRA into a 16-channel latent. Diffusion then runs on a grid with 64× fewer voxels than the original scan.
2. **A Brownian-bridge diffusion model (BBDM)** travels from a projected estimate of the target to the target itself, instead of starting from pure noise. Because it starts from an informed point, **5 sampling steps** are enough. The 3D U-Net receives the condition through AdaGN modulation at every level.
3. **Topology-aware fine-tuning** adds clDice, which rewards connected vessel centrelines, together with soft Dice and Frangi vesselness weighting. Adversarial losses were tested as well and are reported in the ablation below.

## 📊 Results

All results are on the **held-out test set**: 56 subjects from 3 hospitals, evaluated once. Bridge sampling uses 5 steps.

| Model | Masked SSIM | MIP SSIM | Vessel Dice | clDice | Vessel precision | Vessel volume ratio |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| *MRA VAE reconstruction (upper bound)* | *0.991* | *0.994* | *0.948* | *0.933* | *0.959* | *0.98* |
| Projector only, no diffusion | 0.498 | 0.591 | 0.034 | 0.027 | 0.630 | 0.09 |
| Brownian bridge (base) | 0.541 | 0.698 | 0.248 | 0.160 | 0.416 | 0.63 |
| + clDice only | 0.539 | 0.694 | 0.278 | **0.296** | 0.247 | 1.63 |
| **+ clDice, soft Dice, vessel weights (delivered)** | **0.541** | **0.697** | **0.270** | 0.207 | **0.360** | **0.87** |

*Vessel volume ratio is the predicted vessel volume divided by the true vessel volume. A value of 1.0 means vessels have the right thickness.*

**What the numbers show:**
- **Diffusion is what creates the vessels.** The projector alone produces an image of roughly the right overall shape but almost no vessels (Dice 0.03). The bridge recovers them (Dice 0.25).
- **Topology fine-tuning improves connectivity without blurring.** The delivered model raises clDice by **29%** and vessel Dice by **9%** over the base bridge, and keeps SSIM unchanged.
- **A higher score isn't always better.** clDice on its own gives the best connectivity number by *thickening* vessels: they come out at 1.63× their true volume and precision drops to 0.25. The delivered model was chosen because its vessels have close to the right thickness (0.87×).
- **The bottleneck is the mapping, not the autoencoder.** The MRA VAE reconstructs vessels almost perfectly (Dice 0.95). The hard part is predicting where the vessels are from scans that barely show them.

**Known limitations:**
- Thin peripheral vessels can still break or come out thicker than they should.
- Absolute vessel overlap is still modest (Dice 0.27). The model gets the major vessels, not the full tree.
- Results vary between the three hospitals. Per-site numbers are in `runs/table53_test/summary.json`.

## ✨ Engineering highlights
- **Full pipeline.** Raw IXI NIfTI files go through reorientation, SynthStrip skull-stripping, resampling and a site-stratified 455/57/56 split. After that come the two VAEs, the latent bridge, and evaluation that includes Betti numbers and per-site breakdowns.
- **Reproducible.** Every table and figure in the thesis is regenerated by a single script in `analysis/` or `figures/`. Dependencies are locked with `uv`.
- **Safe to experiment.** The training scripts refuse to overwrite the checkpoints and test results that the thesis reports. Every trainer has a `--smoke` mode, and the test split is used exactly once.
- **Tested.** 94 pytest tests cover the models, losses, metrics, preprocessing and the protection of thesis artefacts.

Step-by-step settings, expected numbers, checks and troubleshooting for every training stage are in **[Instructions.md](Instructions.md)**. The rest of this file is the map and the command list.

---

## 1. Setup

Requirements: Python 3.12, an NVIDIA GPU with at least 16 GB (developed on an RTX 5080), and
[uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

This installs the exact versions in `uv.lock`, the environment the thesis results came from
(torch 2.10 + CUDA 13.0 from the PyTorch index). **Run every command from the repository
root.**

External weights, not in git:

| file | source | location |
|---|---|---|
| `synthstrip.1.pt` | [SynthStrip](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/) | `third_party/synthstrip/` |
| `resnet_10_23dataset.pth` and `models/resnet.py` | [Tencent/MedicalNet](https://github.com/Tencent/MedicalNet) | `third_party/MedicalNet/` (FID only) |

## 2. Layout

```
preprocess.py             raw IXI NIfTI -> Dataset/Cropped
split_dataset.py          -> Dataset/splits (manifest) + Dataset/split_numpy/{train,val,test}
train_vae_multimodal.py   T1/T2/PD autoencoder (the condition)
train_vae_mra.py          MRA autoencoder v4
finetune_vae_mra.py       v4 -> v5 arms: cldice / adv / both
export_latents.py         --model mra | multimodal
make_frangi_weights.py    Frangi vessel weights at latent resolution
make_vessel_bands.py      per-case vessel band for the clDice loss
train_bbdm.py             Brownian-bridge diffusion model
finetune_bbdm.py          arms: cldice_alone / cldice / adv_alone / both / delivered
evaluate.py               Tables 5.3 and 5.4 (bridge, all arms, Betti numbers, per site)
evaluate_vae.py           MRA VAE acceptance metrics

modules/      library code, never run directly
  paths.py            repository paths, case helpers, protection of thesis artefacts
  preprocessing.py    reorientation, SynthStrip, resampling, normalisation, cropping
  split.py            site-stratified split and .npy conversion
  frangi.py           Frangi vesselness weights
  vae_multimodal.py   multimodal VAE: model, patch dataset, loss, training loop, encoding
  vae_mra.py          MRA VAE v4/v5: model, tiling, crop dataset, losses, evaluation
  latent_data.py      paired latent dataset and per-channel standardisation
  unet.py             conditional AdaGN 3D UNet
  bridge.py           Brownian bridge, projector, validation
  losses.py           vessel-weighted latent losses
  topology.py         soft skeleton, clDice, soft Dice, vessel probability
  adversarial.py      3D patch and MIP discriminators, hinge losses, adaptive weight
  metrics.py          masked SSIM/PSNR, MIP SSIM, vessel Dice, clDice, Betti numbers
configs/      preprocess.json, vae_multimodal.json, frangi.json, bbdm.json
analysis/     the other thesis numbers (Tables 5.1/5.2, FID, IOP, bridge steps)
figures/      one script per thesis figure
tests/        pytest
```

## 3. Thesis artefacts are protected

These hold the results the thesis reports. Every training and export script refuses to write
into them:

| path | what |
|---|---|
| `checkpoints/vaev2_multimodal/kl5e5_from_epoch10/best.pt` | multimodal VAE (epoch 75) |
| `checkpoints/vaev4_mra/v4_c16_fixed/best.pt` | MRA VAE v4 |
| `checkpoints/vaev5_mra/v5_{cldice,adv,both}/best.pt` | MRA VAE v5 arms; `v5_cldice` is used everywhere |
| `checkpoints/ldm_bbdm/best.pt` | base bridge (epoch 170) and its arms `ft_*`, `both_ep10_assessed.pt` |
| `checkpoints/ldm_bbdm_vessel/ft_cldice/best.pt` | **the delivered model** |
| `latents/{T1,T2,PD}` | conditioning latents, a **fixed input**  |
| `latents_v5_cldice/` | MRA target latents (the export reproduces them bit-for-bit) |
| `runs/table53_test/` | the test-set results of Tables 5.3 and 5.4 |

## 4. Pipeline

Run `--smoke` before any long run; smoke runs write only to `checkpoints/_smoke`. Times are
for one RTX 5080.

### 4.1 Data

1. Download IXI T1, T2, PD and MRA into `Dataset/Quadruplets/{T1,T2,PD,MRA}/`.
   **Keep an untouched copy elsewhere**: the first two steps delete incomplete cases and
   reorient files in place.
2. Preprocess and split:

```bash
uv run preprocess.py
uv run split_dataset.py
```

### 4.2 Multimodal VAE (condition)

About 23 h.

```bash
uv run train_vae_multimodal.py --smoke
uv run train_vae_multimodal.py
uv run export_latents.py --model multimodal --checkpoint checkpoints/rebuild/vae_multimodal/multimodal_rebuild/best.pt
```

### 4.3 MRA VAE (target)

About 9 h for v4, plus 30 epochs per v5 arm.

```bash
uv run train_vae_mra.py --smoke
uv run train_vae_mra.py
uv run finetune_vae_mra.py --checkpoint checkpoints/rebuild/vae_mra/v4_c16/best.pt --arms cldice adv both -n 30
uv run export_latents.py --checkpoint checkpoints/rebuild/vae_mra_v5/v5_cldice/best.pt --out latents_rebuild_v5
```

### 4.4 Loss inputs

```bash
uv run make_frangi_weights.py
uv run make_vessel_bands.py
```

### 4.5 Brownian bridge

About 15 h. The thesis model peaked at epoch 170.

```bash
uv run train_bbdm.py --sanity 5 --scratch
uv run train_bbdm.py
```

`configs/bbdm.json` reads the thesis latents and VAE. For a full rebuild, point
`data.target_root`, `data.conditioning_root` and `vae.checkpoint` at your own outputs.

### 4.6 Fine-tuning arms

40 epochs each, about 2.5 h per arm.

```bash
uv run finetune_bbdm.py --arms cldice --smoke --scratch
uv run finetune_bbdm.py --arms cldice_alone cldice adv_alone both -n 40
uv run finetune_bbdm.py --arms delivered -n 40
```

By default the arms start from the thesis base bridge; pass `--checkpoint` for your own.

### 4.7 Evaluation

Validation first; the test split is used once:

```bash
uv run evaluate.py --split val --cases 4 --models vae isporuceni
uv run evaluate.py --split test --out runs/table53_test_rerun
uv run evaluate_vae.py --split val
```

The other thesis numbers and all figures come from scripts without arguments:

```bash
uv run analysis/vae_arms.py            # Table 5.1: VAE arms on all 57 validation volumes
uv run analysis/multimodal_metrics.py  # Table 5.2: multimodal VAE SSIM / PSNR per modality
uv run analysis/vae_mip_psnr.py        # MIP PSNR of the MRA VAE
uv run analysis/vae_site_iop.py        # MRA VAE on the IOP cases only
uv run analysis/vae_fid.py             # MedicalNet FID, validation split
uv run analysis/bridge_steps.py        # number of bridge steps
uv run figures/fig_modalities.py       # Figure 3.1   (all figures -> runs/figures, runs/step_previews)
uv run figures/fig_architecture.py
uv run figures/fig_forward_process.py
uv run figures/fig_multimodal_vae.py   # Figure 5.1
uv run figures/fig_results.py
uv run figures/fig_test_synthesis.py   # Section 5.4, test split
uv run figures/fig_bridge_steps.py
uv run figures/fig_eta.py
```

## 5. Tests

```bash
uv run pytest
```

94 tests cover the models, losses, metrics, Frangi weights, preprocessing and the protection
of thesis folders, plus `--help` for every script. Tests that need the local thesis
checkpoints skip themselves when those are absent.
