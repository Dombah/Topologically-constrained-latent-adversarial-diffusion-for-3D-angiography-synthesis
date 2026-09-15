# Instructions: preprocessing, training and fine-tuning

This is the practical guide for rebuilding the models in this repository from raw IXI data.
`README.md` gives the layout and the full command list. This file covers the long-running
steps: what each one does, its settings, and how to check that it worked.

Timings are for one RTX 5080 (16 GB) with 32 GB RAM. Run everything from the repository root
with `uv run`.

## Rules

1. **Smoke first.** Every training script has `--smoke`; `train_bbdm.py` also has
   `--sanity N`. Let it finish before starting a run that takes hours. It exercises data
   loading, forward and backward passes, validation and checkpoint writing. Smoke runs write
   only to `checkpoints/_smoke`.
2. **Order matters.** A model trained on latents is tied to the exact VAE checkpoint that
   produced them. If a VAE changes, re-export its latents and retrain everything after it.
3. **Do not look at the test split** until the final evaluation. Model selection uses
   validation only.
4. **Runs resume.** Trainers write `latest.pt`; rerunning the same command continues from it.
   `--scratch` starts over.
5. **Thesis results are protected.** The checkpoints, latents and test results listed in
   `README.md` §3 cannot be written by any script. New runs go to `checkpoints/rebuild/`.
6. **Report brain-masked SSIM** (skimage, 7×7×7 window), not the global single-window SSIM,
   which is about 0.007 optimistic and nearly blind to vessels.

## Order of work

| # | step | command | default output | time |
|---|---|---|---|---|
| 1 | preprocessing | `preprocess.py` | `Dataset/Cropped/` | not recorded (SynthStrip dominates) |
| 2 | split | `split_dataset.py` | `Dataset/splits/`, `Dataset/split_numpy/` | not recorded |
| 3 | multimodal VAE | `train_vae_multimodal.py` | `checkpoints/rebuild/vae_multimodal/multimodal_rebuild/` | ~23 h |
| 4 | conditioning latents | `export_latents.py --model multimodal` | `latents_multimodal/` | not recorded |
| 5 | MRA VAE v4 | `train_vae_mra.py` | `checkpoints/rebuild/vae_mra/v4_c16/` | ~9 h |
| 6 | MRA VAE v5 arms | `finetune_vae_mra.py` | `checkpoints/rebuild/vae_mra_v5/v5_<arm>/` | not recorded |
| 7 | target latents | `export_latents.py` | `latents_v5_cldice/` (or `--out`) | not recorded |
| 8 | Frangi weights, vessel bands | `make_frangi_weights.py`, `make_vessel_bands.py` | `Dataset/frangi_old/`, `Dataset/vessel_bands.json` | ~1.5 h for Frangi (about 10 s per case on CPU) |
| 9 | Brownian bridge | `train_bbdm.py` | `checkpoints/rebuild/bbdm/` | ~15 h |
| 10 | bridge fine-tuning | `finetune_bbdm.py` | `checkpoints/rebuild/bbdm/ft_<arm>/` | ~2.5 h per arm |

Dependencies between steps:
- Steps 3–4 and 5–7 are independent.
- Step 8 needs `Dataset/split_numpy` and a latent folder for the case list and grid.
- Step 9 needs 4, 7 and 8.

Every step's inputs default to the thesis artefacts, so a single step can be rerun without
the ones before it.

---

## 1. Preprocessing

**Input.** The IXI T1, T2, PD and MRA NIfTI files from <https://brain-development.org/ixi-dataset/>:

```
Dataset/Quadruplets/T1/IXI002-Guys-0828-T1.nii.gz
Dataset/Quadruplets/T2/...   Dataset/Quadruplets/PD/...   Dataset/Quadruplets/MRA/...
```

**Keep an untouched copy of the download elsewhere.** The first two steps work in place: they
delete the files of incomplete cases and overwrite volumes that are not in LPS. The SynthStrip
model goes in `third_party/synthstrip/synthstrip.1.pt`.

```bash
uv run preprocess.py                    # all steps
uv run preprocess.py --from normalize   # resume at a step
uv run preprocess.py --only skullstrip  # one step
```

| step | what it does | settings (`configs/preprocess.json`) |
|---|---|---|
| `quadruplets` | keeps cases with all four modalities (568) | |
| `reorient` | reorients every volume to LPS | |
| `skullstrip` | SynthStrip on all four modalities | GPU, `Dataset/SkullStripped/{M}`, masks in `SkullStripped/masks/{M}` |
| `resample` | MRA to 0.4 × 0.4 × 0.8 mm; T1/T2/PD (linear) and masks (nearest) onto each patient's MRA grid | `target_spacing` |
| `rename` | removes the `_stripped` suffix SynthStrip adds | |
| `normalize` | per volume, inside the brain mask, 0.5–99.9 percentile to [0, 1] | `lower_percentile`, `upper_percentile` |
| `crop` | crop or pad to 512 × 576 × 96 | `target_size` |

Every step skips files it has already written, so an interrupted run can be restarted.

**Check:**
- `Dataset/Cropped/{T1,T2,PD,MRA}` each hold 568 volumes of 512 × 576 × 96.
- `Dataset/Normalized/metadata/*.csv` has one row per volume. A failed skull strip shows up
  as an outlier.
- Open two or three cases per site (Guys, HH, IOP). Vessels should be bright in the MRA, with
  no skull left.

## 2. Split

```bash
uv run split_dataset.py
```

- Patient-level split, stratified by site, seed 42: **455 train / 57 val / 56 test**.
- `Dataset/splits/manifests/*.csv` defines which case is in which split. Keep it.
- Output: `Dataset/split_numpy/{split}/{T1,T2,PD,MRA}/<case>-<M>.npy` (float16) and
  `{split}/masks/{M}` (uint8).
- `--manifest-only` writes the manifests without converting.

**Check:** the script reproduces the thesis assignment exactly; this was verified against the
existing manifest. The site ratio is about 55 / 32 / 13 % Guys / HH / IOP in every split.

---

## 3. Multimodal VAE (condition)

Three modality-specific encoders and decoders share one 8-channel latent space at 4×
downsampling. Every patch comes with a one-hot modality mask. The bridge stacks the three
latents into a 24-channel condition.

```bash
uv run train_vae_multimodal.py --smoke
uv run train_vae_multimodal.py
```

The original training notebook no longer exists. `configs/vae_multimodal.json` reproduces
the thesis checkpoint `kl5e5_from_epoch10/best.pt`. Its values were recovered from the
checkpoint's optimiser and scheduler state and from `training.jsonl`; the patch size was
confirmed by the author.

| setting | value |
|---|---|
| model | base 16 channels, multipliers (1, 2, 4), 1 block per level, 8 latent channels, no output activation, 3.54 M parameters |
| data | 96³ patches, 8 per patient (6 brain / 2 background), 455 patients, i.e. 3640 steps per epoch; volumes used as normalised in preprocessing |
| optimiser | AdamW, lr 1e-4, weight decay 0.01, batch 1, grad clip 1.0, AMP |
| schedule | cosine annealing over 100 epochs |
| loss | L1 reconstruction + KL + 0.1 × latent-alignment term |
| KL weight | linear from 1e-6 to **5e-5** over epochs 5–55, then constant (matches every logged value) |
| validation | every 5 epochs on the validation volumes; best by SSIM |
| thesis result | best at **epoch 75**: val SSIM 0.946, PSNR 32.6 |

**Check:**
- `active_channels` stays 8; no channel has collapsed.
- Validation SSIM climbs past ~0.94 per modality.
- `validation_previews/` shows sharp grey–white matter boundaries.

### Conditioning latents: fixed input

The thesis bridge was trained on `latents/{T1,T2,PD}`. Those files cannot be regenerated
exactly: the exporting notebook is gone, and no checkpoint or encoding setting on disk
reproduces them. The closest is the thesis checkpoint (correlation 0.976, max |diff| 0.27).

- **Keep `latents/` and treat it as input data.** Scripts refuse to write into it.
- For a from-scratch rebuild, export equivalent latents:

```bash
uv run export_latents.py --model multimodal --checkpoint checkpoints/rebuild/vae_multimodal/multimodal_rebuild/best.pt
```

This writes deterministic means, (8, 128, 144, 24) float16, to
`latents_multimodal/{T1,T2,PD}/{split}/`, plus `latent_stats_per_channel.json` computed from
train only. Then set `data.conditioning_root` in `configs/bbdm.json` to `latents_multimodal`
and retrain the bridge.

---

## 4. MRA VAE

### 4.1 Base model v4

```bash
uv run train_vae_mra.py --smoke
uv run train_vae_mra.py                 # run name v4_c16, 16 latent channels
```

| setting | value |
|---|---|
| model | 16 latent channels, ChannelRMSNorm (not GroupNorm), 1.09 M parameters |
| crops | mixed extents per step: 96³ ×4 (40 %), 64³ ×8 (15 %), 128²×64 ×3 (20 %), 160²×96 (15 %), 192²×96 (10 %) |
| optimiser | lr 2e-4, 1000 steps per epoch, 300 warm-up steps, cosine to 0 from epoch 71, 120 epochs, 9 h budget |
| loss weights | L1 1.0, vessel L1 4.0, MIP 0.15, background 0.5, perceptual 0.02, equivariance 0.25, tail 0.02, scale 0.1 |
| KL | 1e-7, fixed |

Mixed crop extents make the model behave the same on patches and on whole volumes; the
earlier v3 model failed exactly there.

**Acceptance.** The run ends by writing `acceptance.json`, and all 7 criteria must pass.
Thesis model: masked SSIM 0.982, vessel Dice 0.929, clDice 0.939, extent gap 1e-4, latent
kurtosis −0.82, max 4.4 σ, high-frequency power 9.8 %. Do not continue if a criterion fails.
`uv run evaluate_vae.py --checkpoint <run>/best.pt` rescores any checkpoint.

### 4.2 Fine-tuning arms v5

```bash
uv run finetune_vae_mra.py --arms cldice --smoke
uv run finetune_vae_mra.py --checkpoint checkpoints/rebuild/vae_mra/v4_c16/best.pt --arms cldice adv both -n 30
```

| arm | adds | used for |
|---|---|---|
| `cldice` | soft clDice on the decoded vessel band | **the bridge's target latents** |
| `adv` | 3D patch discriminator (hinge), delayed start | Table 5.1 |
| `both` | both terms | Table 5.1 |

- Every arm uses lr 2e-5, discriminator lr 2e-4 and 30 epochs, starting from v4 `best.pt`
  (default: the thesis v4).
- The v4 loss weights stay unchanged.
- **Check** `arm_results.json` and `ablation.json`: the `cldice` arm must keep masked SSIM
  within about 0.001 of v4 and raise clDice.
- `analysis/vae_arms.py` rescores the arms on all 57 validation volumes.

### 4.3 Target latents

```bash
uv run export_latents.py --checkpoint checkpoints/rebuild/vae_mra_v5/v5_cldice/best.pt --out latents_rebuild_v5
```

- Encoding is whole-volume (`--regime direct`), stored as float16.
- `latent_stats.json` holds the per-channel mean and std from train only. The bridge
  standardises with it; channel means are not zero (−0.4 to +1.0).
- The script decodes a few cases back and prints a PASS/FAIL verdict.
- With the default arguments it reproduces `latents_v5_cldice` bit-for-bit.

---

## 5. Inputs for the bridge losses

```bash
uv run make_frangi_weights.py
uv run make_vessel_bands.py
```

**Frangi weights** (`Dataset/frangi_old/{train,val}`, settings in `configs/frangi.json`):
- Computed at 2× downsampling: σ = 1, 2, 3, bright ridges, robust percentile 99.7.
- Suppressed below the 60th percentile, then γ 1.15.
- Resized to the latent grid (1, 128, 144, 24) with γ 1.1, floor 0.02, renormalised.
- The bridge uses them as `1 + 8·w` on its x0-space losses. Test needs none.
- Existing files are skipped unless `--overwrite`.
- Regenerating reproduces the thesis maps bit-for-bit.

**Vessel bands** (`Dataset/vessel_bands.json`):
- Per case, the whole-volume (p98, p99.5) of the MRA: the support the Dice metric uses.
- clDice turns this band into a soft vessel mask. Its mass matches the hard metric mask
  (ratio 1.03); the fixed brain-masked band used before covered only 15 %.
- 512 entries (train + val), reproduced exactly.

---

## 6. Brownian-bridge diffusion model

```bash
uv run train_bbdm.py --sanity 5 --scratch
uv run train_bbdm.py
```

`configs/bbdm.json` links the stages. By default it reads the thesis inputs:

```json
"data": { "target_root": "latents_v5_cldice", "conditioning_root": "latents",
          "frangi_root": "Dataset/frangi_old", "latent_crop": [128, 144, 24], "preload_to_ram": false },
"vae":  { "checkpoint": "checkpoints/vaev5_mra/v5_cldice/best.pt" },
"output": { "run_dir": "checkpoints/rebuild/bbdm" }
```

For a full rebuild, point the three input entries at your own exports.

| setting | value |
|---|---|
| model | AdaGN UNet: base 48, multipliers (1, 2, 4, 4), 2 blocks per level, attention at levels 2–3; conditioning encoder (base 48) injects scale and shift per level; bridge projector (base 64, 3 blocks) gives the start point; 46 M parameters |
| process | Brownian bridge from the projector output to the MRA latent; sampling with 5 bridge steps |
| losses | bridge MSE 1.0, endpoint L1 0.5, x0 smooth-L1 1.0, Frangi L1 1.5, Frangi MIP 0.25, vessel gradient 0.08; auxiliary terms only while m_t ≤ 0.8 |
| training | full 128 × 144 × 24 latents, batch 1, 455 steps per epoch, AdamW lr 1e-4, weight decay 0.01, 500 warm-up steps, EMA 0.999, bfloat16 (float16 underflows at t = 999), 3 workers, mmap loading |
| budget | 15 h cap. The thesis model peaked at **epoch 170**; about 200 epochs is enough |

**Watch `validation.jsonl`:**
- **Vessel Dice** (bridge-5 and bridge-50) should pass about 0.2 by epoch 100. The thesis
  base model reached 0.230 at bridge-50.
- **`endpoint_vessel_dice`** is the projector alone and stays near 0.05. The script warns if
  the reverse process stops beating it, which would mean the model has become a regressor.
- **Performance:** about 8.4 GB peak memory, about 0.24 s per step.

---

## 7. Fine-tuning the bridge

Every arm loads the base bridge (default `checkpoints/ldm_bbdm/best.pt`, epoch 170), freezes
the projector and trains 40 epochs.

```bash
uv run finetune_bbdm.py --arms cldice --smoke --scratch
uv run finetune_bbdm.py --arms cldice_alone cldice adv_alone both -n 40
uv run finetune_bbdm.py --arms delivered -n 40
```

| arm | added to the bridge loss | output (thesis copy) | Table 5.3 row |
|---|---|---|---|
| `cldice_alone` | clDice (α = 1.0), weight 0.25 | `rebuild/bbdm/ft_cldice_alone` (`ldm_bbdm/ft_cldice_alone`) | + clDice |
| `cldice` | 0.5 clDice + 0.5 soft Dice | `rebuild/bbdm/ft_cldice` (`ldm_bbdm/ft_cldice`) | + clDice i meki Dice |
| `adv_alone` | MIP PatchGAN hinge loss, weight 0.25 × adaptive λ | `rebuild/bbdm/ft_adv_alone` (`ldm_bbdm/ft_adv_alone`) | + suparnički gubitak |
| `both` | `cldice` + adversarial | `rebuild/bbdm/ft_both` (`ldm_bbdm/both_ep10_assessed.pt`, epoch 10) | + clDice i suparnički |
| `delivered` | `cldice` + Frangi L1 4.0 + vessel gradient 0.2 | `rebuild/bbdm_vessel/ft_cldice` (`ldm_bbdm_vessel/ft_cldice`) | **isporučeni model** |

All five arms reproduce the original training losses exactly; this was checked against the
pre-cleanup script.

Common settings:
- lr 2e-5; discriminator lr 2e-4, starting after 500 steps; soft-MIP temperature 0.1.
- Every 2nd step, the topology loss decodes a 128 × 144 × 4 latent slab through the frozen
  MRA VAE, using each case's band from `vessel_bands.json`.
- Validation and a snapshot every 10 epochs; `best.pt` by validation vessel Dice.
- `config.json` in each arm folder records α and the loss overrides.

Always look at the MIP previews, not only at Dice. An arm that raises Dice by thickening
vessels shows a band ratio above 1 and falling precision; clDice alone did exactly that
(band 2.77×).

---

## Troubleshooting

| symptom | cause and fix |
|---|---|
| epochs suddenly 5–7× slower, GPU at low power while showing 100 % | RAM paging of a latent preload. Keep `preload_to_ram: false` (mmap) |
| DataLoader workers fail to start on Windows | the dataset is pickled into each worker; the bridge scripts fall back to 0 workers. Close memory-heavy apps |
| "non-finite loss … skipping" | the step is skipped and training continues. If it repeats, resume from `latest.pt` with a lower lr |
| CUDA OOM during VAE validation or export | set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; use `--regime tiled` for the MRA export; the multimodal export falls back to tiling by itself |
| bridge metrics far worse than expected | `vae.checkpoint` is not the checkpoint the target latents were exported from, or `conditioning_root` points at latents from a different multimodal VAE |
| "refusing to write … it holds a thesis result" | intended. Pass another `--run-dir` / `--out` |
| "N cases have no vessel band" | regenerate `Dataset/vessel_bands.json` after a new split |
