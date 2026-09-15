# Topologically constrained latent adversarial diffusion for 3D angiography synthesis

Synthesis of 3D TOF-MRA from T1, T2 and PD MRI (IXI dataset, 568 cases, 455 / 57 / 56 split).
A Brownian-bridge diffusion model works in the latent spaces of two autoencoders and is then
fine-tuned with topology (clDice + soft Dice) and adversarial losses.

```
T1/T2/PD ─► multimodal VAE ─► 24-ch condition ─┬─► bridge projector ─► Brownian bridge, 5 steps ─► 16-ch MRA latent ─► MRA VAE decoder ─► TOF-MRA
                                               │
MRA ──────► MRA VAE (v5, clDice) ─► 16-ch target (training only)
```

Step-by-step settings, expected numbers, checks and troubleshooting for every training stage
are in **[Instructions.md](Instructions.md)**. This file is the map and the command list.

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
