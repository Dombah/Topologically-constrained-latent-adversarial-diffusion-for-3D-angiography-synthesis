# Repository cleanup record

The repository was reorganised on 14–15 September 2026 from notebooks and experiment scripts
into `.py` entry scripts plus a `modules/` library.

- **Rollback:** the git tag `pre-cleanup` and a code zip, both kept locally (not published).
- **Where to look:** `README.md` describes the result; `Instructions.md` covers training.

## Status

| part | state |
|---|---|
| git repository, backup tag, code zip | done |
| scratchpad-only thesis scripts rescued into `analysis/`, `figures/` | done |
| shared code moved into `modules/` (verbatim extraction, no rewrites) | done |
| notebooks ported to `.py` | done |
| superseded scripts, notebooks, old modules, tools, agent scaffolding deleted from git | done |
| thesis artefacts protected; new runs go to `checkpoints/rebuild/` | done |
| tests rewritten (94 pass), `pyproject.toml` + `uv.lock` pinned to the installed environment | done |
| deleting data and checkpoints not needed by the thesis (§4) | done 2026-09-15: 228 GB freed; tests and an 8-row evaluation rerun afterwards |

## 1. What moved where

| before | after |
|---|---|
| `processing.ipynb` | `preprocess.py`, `split_dataset.py`, `make_frangi_weights.py` + `modules/preprocessing.py`, `modules/split.py`, `modules/frangi.py` |
| `split_ixi_dataset.py` | `modules/split.py` (minus 4 dead inspection helpers) |
| `modules/mri_preprocessing.py` | `modules/preprocessing.py` (52 reachable functions of 74) |
| `modules/frangi_vessels.py` + part of old `latent_data.py` | `modules/frangi.py` |
| `modules/VAEv2.py` (multimodal part) + `modules/utils.py` (export) | `modules/vae_multimodal.py`; trainer rebuilt as `train_vae_multimodal.py` |
| `train_vaev4.py` | `modules/vae_mra.py`, `modules/metrics.py`, `modules/paths.py` + `train_vae_mra.py` |
| `finetune_vaev4.py` | `finetune_vae_mra.py` + `modules/topology.py`, `modules/adversarial.py` |
| `validate_vaev4.py` | `evaluate_vae.py` |
| `export_latents.py` | same name; adds `--model multimodal`; defaults are now v5_cldice |
| `train_ldm.py` (library part) | `modules/latent_data.py`, `modules/losses.py` |
| `train_ldm_v2.py` (library part) + `modules/LDM3D.py` | `modules/unet.py` |
| `finetune_ldm.py` (library part) | `modules/adversarial.py`, `modules/latent_data.py`, `modules/topology.py` |
| `train_bbdm.py` | `modules/bridge.py` + `train_bbdm.py` |
| `finetune_bbdm.py` | trimmed to the thesis arms `cldice_alone`, `cldice`, `adv_alone`, `both`, `delivered` |
| `tools/eval_table53.py` | `evaluate.py` |
| scratchpad `make_bands.py` | `make_vessel_bands.py` |
| scratchpad `mm_metrics`, `fid_v5`, `mip_psnr_v5`, `rescore_vae`, `vae_iop`, `step_sweep` | `analysis/multimodal_metrics.py`, `vae_fid.py`, `vae_mip_psnr.py`, `vae_arms.py`, `vae_site_iop.py`, `bridge_steps.py` |
| scratchpad `figs_*`, `fig_arch2`, `fig_forward`, `step_previews`, `eta02` | `figures/fig_*.py` |
| `ldm/config_bbdm_mmap.json` | `configs/bbdm.json` (output → `checkpoints/rebuild/bbdm`) |

**Deleted** (still in the `pre-cleanup` tag):
- the four notebooks;
- `train_ldm.py`, `train_ldm_v2.py`, `finetune_ldm.py`, `train_cascade.py`, `finetune_adv.py`, `evaluate_runs.py`, `run_overnight.py`;
- the `.cmd` / `.sh` chains;
- `tools/`, `ldm/`;
- `modules/{LDM3D,VAEv2,mri_preprocessing,frangi_vessels,mra_decoder,utils}.py`;
- `CLEANUP_PLAN.md`, `REVIEW_2026-09-02.md`, `.github/`, `.agents/`;
- `tests/test_notebook_losses.py`.

Also dropped: the fine-tuning arms `perc`, `percdice`, `advslice`, `slicedice` and `moments`
(not in the thesis), and the `--noc` night rows of the evaluation.

## 2. How each part was checked against the old code

| check | result |
|---|---|
| `evaluate.py` vs `tools/eval_table53.py`, val, 4 cases, rows vae / projektor / osnovni / cldice_dice / isporuceni | all 198 numbers and `rows.jsonl` **bit-identical** |
| `train_bbdm.py --smoke` old vs new | identical |
| `finetune_bbdm.py` smoke for `cldice`, `adv_alone`, `both`, `cldice_alone` (= old `--alpha 1.0`), `delivered` (= old `--loss frangi_l1=4.0 --loss vessel_gradient=0.2`) | identical losses. This check caught a bug introduced during trimming: the clDice fraction variable was shadowed by `frangi_alpha`. It was fixed before commit |
| `finetune_vae_mra.py --smoke` old vs new | identical, including the perturbation ablation table |
| `export_latents.py` (MRA, defaults) | reproduces `latents_v5_cldice` bit-for-bit |
| `make_frangi_weights.py`, 2 train + 2 val cases | bit-identical to `Dataset/frangi_old`, using `latents_v5_cldice` for the grid |
| `make_vessel_bands.py`, 6 cases | identical to `Dataset/vessel_bands.json` |
| `split_dataset.py` on the 568 case names | identical train/val/test assignment |
| `train_vae_multimodal.py` recipe | KL ramp reproduces every `kl_weight` in `training.jsonl`; the model loads the thesis checkpoint `strict=True` |
| `preprocess.py` | **not run.** The raw IXI files are no longer on disk. Functions are unchanged and called with the notebook's arguments |
| `train_vae_mra.py --smoke` | runs; not comparable because the smoke test is unseeded. The model path is covered by the bit-identical `vae` evaluation row |

**Fixed input: `latents/{T1,T2,PD}`.** These cannot be regenerated. None of the tested
encodings of the multimodal checkpoints reproduces them:
- direct encoding;
- tiled encoding with 96³, 192×192×64 and 196×196×64 tiles, overlap 0/4/16, uniform and Hann
  blending;
- the checkpoints `best.pt` and epochs 70–90.

The closest reaches correlation 0.976. The 16-channel attention VAE is ruled out because the
latents have 8 channels. They are kept as input data and are write-protected.

## 3. Things found and fixed along the way

- **Smoke runs overwrote real results.** `--smoke` in the old `train_bbdm.py` and
  `finetune_vaev4.py` wrote into the real run folders and could overwrite thesis checkpoints.
  All smoke runs now go to `checkpoints/_smoke`, and `modules/paths.PROTECTED` blocks writes
  into thesis artefacts.
- **Wrong script defaults.** `train_vaev4.py` defaulted to 8 channels (the thesis model has
  16) and `export_latents.py` to v4 / `latents_v4`. Both defaults are corrected.
- **Missing script.** `Dataset/vessel_bands.json`, a required fine-tuning input, was produced
  by a script that existed only in a temporary folder. It is now `make_vessel_bands.py`.

## 4. Data cleanup (done 2026-09-15)

Disk use went from 1.1 TB to 859 GB. Remaining: Dataset 181 GB, latents 12 GB, latents_v5_cldice
7.5 GB, checkpoints 1.3 GB, runs 62 MB. Training logs (`*.jsonl`, `config.json`) were kept next
to every kept checkpoint; only `.pt` snapshots were removed. `runs/eta02*` and `runs/step_previews`
were also kept (figure outputs).

### Keep

| path | size |
|---|---|
| `Dataset/split_numpy/{train,val,test,manifests}` | ~180 GB |
| `Dataset/splits`, `Dataset/frangi_old`, `Dataset/vessel_bands.json` | 0.4 GB |
| `latents/{T1,T2,PD}` + `latents/latent_stats_per_channel.json` (fixed input) | 11.2 GB |
| `latents_v5_cldice/` | 7.5 GB |
| `checkpoints/vaev2_multimodal/kl5e5_from_epoch10/best.pt` + logs | 41 MB |
| `checkpoints/vaev4_mra/v4_c16_fixed/{best.pt,config.json,acceptance.json}` | 13 MB |
| `checkpoints/vaev5_mra/v5_{cldice,adv,both}/best.pt` + json, `ablation_full_val.json` | 53 MB |
| `checkpoints/ldm_bbdm/{best.pt,config.json}`, `ft_cldice_alone`, `ft_cldice`, `ft_adv_alone` (`best.pt` + json), `both_ep10_assessed.pt` | 0.9 GB |
| `checkpoints/ldm_bbdm_vessel/ft_cldice/{best.pt,config.json,*.jsonl}` | 178 MB |
| `runs/table53_test`, `runs/multimodal_metrics`, `runs/medicalnet_fid/v5_cldice_val`, `runs/step_sweep`, `runs/figures`, `runs/vae_site` | <0.1 GB |
| `third_party/MedicalNet/{models/resnet.py,LICENSE,resnet_10_23dataset.pth}`, `third_party/synthstrip/` | 85 MB |

### Delete

| path | size | reason |
|---|---|---|
| `Dataset/split_numpy/cache` | 88.9 GB | patch caches, rebuilt automatically |
| `checkpoints/ldm_direct_mra` | 83.1 GB | old v-prediction LDM on v3 latents |
| `checkpoints/ldm_v5_cldice`, `ldm_v5_cldice_v2` | 10.8 GB | v-prediction LDM v1/v2, not in the thesis |
| `checkpoints/ldm_bbdm/{epoch_*,latest}.pt`, `ft_both/`, `ft_*/{epoch_*,latest}.pt` | ~5.5 GB | snapshots (the table uses `both_ep10_assessed.pt`) |
| `checkpoints/ldm_bbdm_vessel/{ft_advslice,ft_moments}`, `ft_cldice/{epoch_*,latest}.pt` | ~5.2 GB | dropped arms, snapshots |
| `checkpoints/ldm_bbdm_vessel_level`, `ldm_bbdm_perc`, `ldm_cascade` | 4.5 GB | side experiments |
| `checkpoints/VAE`, `vaev2_mra`, `vaev3_mra` | 3.3 GB | superseded VAEs |
| `checkpoints/vaev2_multimodal/multimodal_attention_from_scratch`, `kl5e5_from_epoch10/epoch_*.pt` | 1.4 GB | unused variant, snapshots |
| `checkpoints/vaev4_mra/v4_c8_channelrms`, `v4_c16_fixed/{epoch_*,latest}.pt`, `vaev5_mra/*/{final,latest}.pt` | ~0.2 GB | |
| `checkpoints/test_previews`, `checkpoints/*.log`, `checkpoints/_smoke` | small | |
| `latents/MRA`, `latents/MRA_first` | 15 GB | v3 latents |
| `latents_stage1` | 6.75 GB | cascade experiment |
| `Dataset/frangi_small`, `Dataset/models`, `Dataset/scripts` | 0.45 GB | unused variant; SynthStrip model copied to `third_party/`; inspection scripts |
| `runs/*` except the folders kept above | ~0.1 GB | diagnostics |
| `third_party/MedicalNet/{images,toy_data,datasets,*.yml,train.py,test*.py,setting.py,model.py}` | 41 MB | |
