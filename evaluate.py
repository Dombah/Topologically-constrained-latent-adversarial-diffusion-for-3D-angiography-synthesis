"""Brojke za tablicu 5.3: sve grane mosta na cijelom skupu, s Betti brojevima.

    uv run evaluate.py --split val --cases 2 --models vae isporuceni     # brza provjera
    uv run evaluate.py --split test --out runs/table53_test_rerun         # konačna tablica, 56 volumena

runs/table53_test (rezultat iz rada) je zaštićen, pa ponovni izračun na testnom skupu treba --out.

Redci tablice (MODELS niže):
    vae           rekonstrukcija pravog latentnog prikaza kroz MRA VAE -- gornja granica,
                  jer difuzijski model radi u istom latentnom prostoru
    projektor     krajnja točka mosta dekodirana bez ijednog koraka difuzije -- koliko most
                  dodaje; projektor je u svim granama zamrznut, pa je isti za sve
    osnovni       osnovni most, epoha 170
    cldice_sam    + clDice bez mekog Dicea (grana koja je zadebljavala žile)
    cldice_dice   + clDice i meki Dice
    suparnicki    + suparnički gubitak (MIP diskriminator)
    oboje         + clDice i suparnički gubitak, epoha 10
    isporuceni    + clDice i meki Dice uz pojačane žilne težine -- model iz rada

Mjere, svaka po volumenu pa prosjek (i po ustanovi):
    PSNR, SSIM globalni i maskirani   isti postupak kao tablica 5.1 (skimage, prozor 7x7x7)
    MIP SSIM                          prosjek triju osi
    Dice, clDice, pojas, preciznost   prag na 99. percentilu referentnog volumena
    b0, b1                            Betti brojevi maske žila na pragu 2 x medijan
                                      intenziteta referentnog volumena unutar maske mozga

Zašto taj prag za Betti brojeve: na 99. percentilu većina komponenti nije žila nego šum
referentnog volumena (1462 od 1482 ima srednji intenzitet ispod dvostrukog medijana), pa bi
b0 mjerio šum, a ne vaskularno stablo. Dvostruki medijan izdvaja same žile. Isti prag,
izračunat iz reference, primjenjuje se i na generirani volumen, kao i za Dice.

Betti brojevi računaju se na maski očišćenoj od komponenti manjih od 20 voksela, uz
26-susjedstvo za žile i 6-susjedstvo za pozadinu (uobičajen par u digitalnoj topologiji):
    b0 = broj povezanih komponenti
    b2 = broj zatvorenih šupljina (komponente pozadine koje ne dodiruju rub volumena)
    b1 = b0 + b2 - Eulerova karakteristika     (broj neovisnih petlji)

Konačna tablica računa se na testnom skupu (--split test); izbor modela napravljen je prije
toga na validacijskom. Retci se dopisuju u rows.jsonl čim su izračunati, pa se prekinuta
vožnja nastavlja od mjesta gdje je stala.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from modules.bridge import BridgeProjector, BrownianBridge
from modules.latent_data import PairedLatentDataset, destandardize_t
from modules.metrics import betti, hard_cldice, masked_ssim_and_psnr, mip_ssim, vessel_scores
from modules.paths import refuse_protected, ROOT, mask_path_for
from modules.unet import build_model
from modules.vae_mra import decode_tiled, load_vaev4, reconstruct

STEPS = 5

MODELS = [
    ("vae", "MRA VAE rekonstrukcija", None),
    ("projektor", "projektor, bez difuzije", "checkpoints/ldm_bbdm/best.pt"),
    ("osnovni", "osnovni model", "checkpoints/ldm_bbdm/best.pt"),
    ("cldice_sam", "+ clDice", "checkpoints/ldm_bbdm/ft_cldice_alone/best.pt"),
    ("cldice_dice", "+ clDice i meki Dice", "checkpoints/ldm_bbdm/ft_cldice/best.pt"),
    ("suparnicki", "+ suparnički gubitak", "checkpoints/ldm_bbdm/ft_adv_alone/best.pt"),
    ("oboje", "+ clDice i suparnički", "checkpoints/ldm_bbdm/both_ep10_assessed.pt"),
    ("isporuceni", "+ clDice i meki Dice, žilne težine", "checkpoints/ldm_bbdm_vessel/ft_cldice/best.pt"),
]


def measure(model: str, case: str, prediction: np.ndarray, split: str) -> dict:
    path = ROOT / "Dataset/split_numpy" / split / "MRA" / (case + "-MRA.npy")
    truth = np.asarray(np.load(path), np.float32)
    brain = np.asarray(np.load(mask_path_for(path, split))) > 0
    prediction = np.clip(prediction.astype(np.float32), 0.0, 1.0)

    ssim_global, ssim_masked, psnr = masked_ssim_and_psnr(truth, prediction, brain)
    threshold = float(np.percentile(truth, 99.0))
    true_band, pred_band = truth >= threshold, prediction >= threshold
    vessel = 2.0 * float(np.median(truth[brain]))
    b0_true, b1_true, _ = betti(truth >= vessel)
    b0_pred, b1_pred, _ = betti(prediction >= vessel)
    return dict(
        model=model, case=case, site=case.split("-")[1],
        psnr=psnr, ssim_global=ssim_global, ssim_masked=ssim_masked,
        mip_ssim=mip_ssim(truth, prediction),
        dice=vessel_scores(truth, prediction)[0], cldice=hard_cldice(truth, prediction),
        band=float(pred_band.sum() / max(true_band.sum(), 1)),
        precision=float((true_band & pred_band).sum() / max(pred_band.sum(), 1)),
        vessel_threshold=vessel, b0_true=b0_true, b1_true=b1_true,
        b0=b0_pred, b1=b1_pred,
        b0_error=abs(b0_pred - b0_true), b1_error=abs(b1_pred - b1_true))


# =============================================================================
# Generiranje na GPU-u
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--cases", type=int, default=0, help="0 = cijeli validacijski skup")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--out", type=Path, default=None,
                        help="zadano runs/table53_<split>")
    args = parser.parse_args()
    if args.out is None:
        args.out = ROOT / "runs" / ("table53_" + args.split)

    import torch

    catalogue = MODELS
    if args.models:
        catalogue = [m for m in catalogue if m[0] in args.models]
    refuse_protected(args.out, "evaluation results")
    args.out.mkdir(parents=True, exist_ok=True)
    rows_path = args.out / "rows.jsonl"
    done = set()
    if rows_path.exists():
        for line in open(rows_path, encoding="utf-8"):
            row = json.loads(line)
            done.add((row["model"], row["case"]))

    device = "cuda"
    config = json.loads((ROOT / "checkpoints/ldm_bbdm_vessel/ft_cldice/config.json").read_text())
    dataset = PairedLatentDataset(config, args.split, seed=42)
    cases = sorted(dataset.cases)
    if args.cases:
        cases = cases[:args.cases]
    source_ch = sum(len(dataset.stats[m]["per_channel_mean"]) for m in dataset.sources)
    target_ch = len(dataset.stats[dataset.target_modality]["per_channel_mean"])
    stats = dataset.stats[dataset.target_modality]
    vae = load_vaev4(ROOT / config["vae"]["checkpoint"], device)
    bridge = BrownianBridge(int(config["bridge"]["steps"]),
                            float(config["bridge"]["max_variance"]), device)
    print("volumena: %d  modela: %d  radnika: %d" % (len(cases), len(catalogue), args.workers),
          flush=True)

    def decode(latent):
        with torch.amp.autocast("cuda", dtype=torch.float16):
            volume = decode_tiled(vae, destandardize_t(latent, stats).to(latent.dtype))
        return volume[0, 0].float().cpu().numpy().astype(np.float16)

    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool, \
            open(rows_path, "a", encoding="utf-8") as sink:
        for key, label, checkpoint in catalogue:
            todo = [c for c in cases if (key, c) not in done]
            if not todo:
                print("  %-12s već izračunat" % key, flush=True)
                continue
            model = projector = None
            if checkpoint is not None:
                ckpt = ROOT / checkpoint
                local = json.loads((ckpt.parent / "config.json").read_text())
                blob = torch.load(ckpt, map_location=device, weights_only=False)
                projector = BridgeProjector(source_ch, target_ch,
                                            base=int(local["model"]["projector_base"]),
                                            blocks=int(local["model"]["projector_blocks"])).to(device)
                projector.load_state_dict(blob["ema_projector"]); projector.eval()
                if key != "projektor":
                    model = build_model(local, target_ch, source_ch, device)
                    model.load_state_dict(blob["ema"]); model.eval()
                del blob

            futures = []
            for case in todo:
                target, cond = dataset.full_case(case)
                with torch.inference_mode():
                    if key == "vae":
                        path = ROOT / "Dataset/split_numpy" / args.split / "MRA" / (case + "-MRA.npy")
                        truth = torch.from_numpy(np.asarray(np.load(path), np.float32))[None, None]
                        prediction, _ = reconstruct(vae, truth.to(device), "direct")
                        prediction = np.clip(prediction, 0.0, 1.0).astype(np.float16)
                    elif key == "projektor":
                        prediction = decode(projector(cond[None].to(device)))
                    else:
                        prediction = decode(bridge.sample(model, projector, cond[None].to(device),
                                                          (1,) + tuple(target.shape), device, STEPS))
                torch.cuda.empty_cache()
                futures.append(pool.submit(measure, key, case, prediction, args.split))
                # Ne gomilati više od dvostrukog broja radnika predviđanja u memoriji.
                while sum(not f.done() for f in futures) >= 2 * args.workers:
                    time.sleep(0.2)
            for future in futures:
                row = future.result()
                row["label"] = label
                sink.write(json.dumps(row) + "\n")
                sink.flush()
            del model, projector
            torch.cuda.empty_cache()
            print("  %-12s gotovo, %d volumena  [%.1f min]"
                  % (key, len(todo), (time.time() - started) / 60), flush=True)

    summarise(rows_path, [m[0] for m in catalogue], cases, args.out)


def summarise(rows_path: Path, keys: list[str], cases: list[str], out: Path) -> None:
    rows = [json.loads(line) for line in open(rows_path, encoding="utf-8")]
    wanted = set(cases)
    metrics = ("psnr", "ssim_global", "ssim_masked", "mip_ssim", "dice", "cldice", "band",
               "precision", "b0", "b1", "b0_error", "b1_error")
    summary = {}
    for key in keys:
        mine = [r for r in rows if r["model"] == key and r["case"] in wanted]
        if not mine:
            continue
        entry = {"label": mine[0]["label"], "n": len(mine),
                 "mean": {m: float(np.mean([r[m] for r in mine])) for m in metrics},
                 "by_site": {}}
        for site in ("Guys", "HH", "IOP"):
            part = [r for r in mine if r["site"] == site]
            if part:
                entry["by_site"][site] = {"n": len(part), **{m: float(np.mean([r[m] for r in part]))
                                                            for m in metrics}}
        summary[key] = entry
    reference = [r for r in rows if r["model"] == keys[0] and r["case"] in wanted]
    if reference:
        summary["referenca"] = {"b0": float(np.mean([r["b0_true"] for r in reference])),
                                "b1": float(np.mean([r["b1_true"] for r in reference])),
                                "n": len(reference)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                      encoding="utf-8")

    print("\n%-36s %4s %7s %7s %7s %7s %7s %7s %6s %6s %6s %6s %6s %6s" % (
        "model", "n", "PSNR", "SSIMg", "SSIMm", "MIPss", "Dice", "clDice", "pojas", "prec",
        "b0", "b1", "|db0|", "|db1|"))
    if "referenca" in summary:
        r = summary["referenca"]
        print("%-36s %4d %7s %7s %7s %7s %7s %7s %6s %6s %6.1f %6.1f" % (
            "referentni volumen", r["n"], "", "", "", "", "", "", "", "", r["b0"], r["b1"]))
    for key in keys:
        if key not in summary:
            continue
        m = summary[key]["mean"]
        print("%-36s %4d %7.2f %7.4f %7.4f %7.4f %7.4f %7.4f %6.2f %6.3f %6.1f %6.1f %6.1f %6.1f" % (
            summary[key]["label"], summary[key]["n"], m["psnr"], m["ssim_global"],
            m["ssim_masked"], m["mip_ssim"], m["dice"], m["cldice"], m["band"], m["precision"],
            m["b0"], m["b1"], m["b0_error"], m["b1_error"]))
    print("\nzapisano ->", out / "summary.json", flush=True)


if __name__ == "__main__":
    main()
