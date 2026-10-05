"""Run the colour-invariance benchmark for a trained checkpoint or a zero-shot baseline.

  python -m sareeid.evaluate --manifest work/manifest.csv --ckpt runs/atto/best.pt
  python -m sareeid.evaluate --manifest work/manifest.csv --zero-shot convnext_atto.d2_in1k [--gray]

The verification threshold is calibrated on the *val* split (EER point) and then applied
unchanged to *test*, so the reported accuracy is not tuned on test data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .benchmark import FAMILIES, build_specs, embed_specs
from .data import load_manifest
from .metrics import identification, verification
from .model import load_checkpoint, zero_shot_model
from . import recolor as rc


def run_benchmark(model, items, device, gray=False, seed=0, threshold=None, workers=2,
                  batch_size=128, tta=False, size=224) -> dict:
    B = build_specs(items, seed=seed, size=size)
    S = B["specs"]
    Z = embed_specs(model, S, device, batch_size=batch_size, workers=workers, gray=gray, tta=tta, size=size)
    lab = lambda idx: [S[i]["label"] for i in idx]
    g = B["gallery"]
    res: dict = {"n_designs": B["n_designs"]}

    # P1 colorway, per family and pooled (train families vs held-out families)
    per_fam = {f: identification(Z[B["p1"][f]], lab(B["p1"][f]), Z[g], lab(g)) for f in FAMILIES}
    res["P1_per_family"] = per_fam
    for name, fams in (("all", FAMILIES), ("train_families", rc.TRAIN_FAMILIES),
                       ("heldout_families", rc.HELDOUT_FAMILIES)):
        q = [i for f in fams for i in B["p1"][f]]
        res[f"P1_{name}"] = identification(Z[q], lab(q), Z[g], lab(g))
    # P2 colour trap
    t = B["trap"]
    res["P2_trap"] = identification(Z[t], lab(t), Z[g], lab(g), donor_lab=[S[i]["donor_label"] for i in t])
    # P3 shared palette
    p3 = [identification(Z[q], lab(q), Z[gg], lab(gg)) for gg, q in B["shared"]]
    res["P3_shared_palette"] = {k: float(np.mean([r[k] for r in p3])) for k in p3[0]}
    # P4 real re-photos
    if B["rephoto"]:
        r = B["rephoto"]
        res["P4_rephoto"] = identification(Z[r], lab(r), Z[g], lab(g))

    # verification
    P = B["pairs"]
    cos = lambda pairs: (Z[[a for a, _ in pairs]] * Z[[b for _, b in pairs]]).sum(1).numpy()
    sg, sr, ss = cos(P["genuine"]), cos(P["impostor_random"]), cos(P["impostor_samepal"])
    res["verif_all"] = verification(np.r_[sg, sr, ss], np.r_[np.ones_like(sg), np.zeros_like(sr), np.zeros_like(ss)],
                                    threshold)
    res["verif_samepalette_only"] = verification(np.r_[sg, ss], np.r_[np.ones_like(sg), np.zeros_like(ss)], threshold)
    res["score_stats"] = {k: [float(v.mean()), float(v.std())] for k, v in
                          (("genuine", sg), ("impostor_random", sr), ("impostor_samepal", ss))}
    return res


def headline(res: dict) -> dict:
    """The handful of numbers worth putting in a table."""
    return {
        "P1 R@1": res["P1_all"]["R@1"], "P1 mAP": res["P1_all"]["mAP"],
        "P1 R@1 held-out fam": res["P1_heldout_families"]["R@1"],
        "P2 R@1": res["P2_trap"]["R@1"], "P2 Trap@1 (lower=better)": res["P2_trap"]["Trap@1"],
        "P3 R@1": res["P3_shared_palette"]["R@1"],
        **({"P4 R@1": res["P4_rephoto"]["R@1"]} if "P4_rephoto" in res else {}),
        "Verif AUC": res["verif_all"]["AUC"], "Verif EER": res["verif_all"]["EER"],
        "TAR@FAR=1%": res["verif_all"]["TAR@FAR=0.01"],
        "Same-palette AUC": res["verif_samepalette_only"]["AUC"],
        **({"Acc@val-thr": res["verif_all"]["acc@thr"]} if "acc@thr" in res["verif_all"] else {}),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ckpt")
    ap.add_argument("--zero-shot", help="timm backbone name for a no-training baseline")
    ap.add_argument("--gray", action="store_true", help="feed grayscale (zero-shot ablation)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", required=True, help="output json path")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.ckpt:
        model, ck = load_checkpoint(a.ckpt)
        gray = ck.get("gray", False) or a.gray
        name = Path(a.ckpt).parent.name
    else:
        model, gray, name = zero_shot_model(a.zero_shot), a.gray, f"zeroshot_{a.zero_shot}" + ("_gray" if a.gray else "")
    model.to(device)
    kw = dict(gray=gray, seed=a.seed, workers=a.workers, tta=a.tta)
    val = run_benchmark(model, load_manifest(a.manifest, "val"), device, **kw)
    thr = val["verif_all"]["thr_at_EER"]
    test = run_benchmark(model, load_manifest(a.manifest, a.split), device, threshold=thr, **kw)
    out = {"model": name, "split": a.split, "val_threshold": thr, "headline": headline(test), "full": test}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out["headline"], indent=2))


if __name__ == "__main__":
    main()
