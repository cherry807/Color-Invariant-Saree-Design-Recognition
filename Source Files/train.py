"""Train the colour-invariant design embedder.

  python -m sareeid.train --manifest work/manifest.csv --out runs/atto --epochs 30

Each batch = P designs x K views. Every view gets an independent crop, a recolor (palette
remap / palette transfer from a shared bank / HSV / channel shuffle / gray) and capture
artefacts. Multi-positive SupCon pulls views of the same design together and pushes all
other designs apart, including designs that were given the very same palette.
Model selection uses the val-split benchmark (mean of P1, P2, P3 Recall@1).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import MultiViewTrainSet, load_manifest
from .evaluate import headline, run_benchmark
from .losses import supcon_loss
from .model import SareeEmbedder


def collate(batch):
    x = torch.cat([b[0] for b in batch])
    y = torch.tensor([b[1] for b in batch]).repeat_interleave(batch[0][0].shape[0])
    return x, y


def selection_score(res):
    return (res["P1_all"]["R@1"] + res["P2_trap"]["R@1"] + res["P3_shared_palette"]["R@1"]) / 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backbone", default="convnext_atto.d2_in1k")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--P", type=int, default=48, help="designs per batch")
    ap.add_argument("--views", type=int, default=4, help="views per design")
    ap.add_argument("--lr", type=float, default=3e-4, help="head lr; backbone gets lr*backbone_mult")
    ap.add_argument("--backbone-mult", type=float, default=0.3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--warmup", type=int, default=1, help="warmup epochs")
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--gray", action="store_true", help="ablation: train on grayscale input")
    ap.add_argument("--bank", type=int, default=24, help="shared palette bank size; 0 = ablation without sharing")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0, help="debug: cap steps per epoch")
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(a), indent=2))

    train_items, val_items = load_manifest(a.manifest, "train"), load_manifest(a.manifest, "val")
    ds = MultiViewTrainSet(train_items, size=a.size, views=a.views, bank_size=a.bank, seed=a.seed, gray=a.gray)
    P = min(a.P, len(ds))
    # persistent_workers=False so the per-epoch palette bank (set_epoch) reaches the workers
    dl = DataLoader(ds, batch_size=P, shuffle=True, drop_last=True, num_workers=a.workers,
                    collate_fn=collate, pin_memory=device.type == "cuda", persistent_workers=False)

    model = SareeEmbedder(a.backbone, a.dim, pretrained=True).to(device)
    bb = list(model.backbone.parameters())
    rest = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": bb, "lr": a.lr * a.backbone_mult}, {"params": rest, "lr": a.lr}],
                            weight_decay=a.wd)
    steps_per_epoch = min(len(dl), a.max_steps) if a.max_steps else len(dl)
    total, warm = a.epochs * steps_per_epoch, a.warmup * steps_per_epoch
    base = [g["lr"] for g in opt.param_groups]
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    hist, best, step = [], -1.0, 0
    for ep in range(a.epochs):
        ds.set_epoch(ep)
        model.train()
        t0, losses = time.time(), []
        for it, (x, y) in enumerate(dl):
            if a.max_steps and it >= a.max_steps:
                break
            f = step / max(1, warm) if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
            for g, b in zip(opt.param_groups, base):
                g["lr"] = b * f
            x, y = x.to(device, non_blocking=True), y.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                z = model(x)
            loss = supcon_loss(z, y, a.temperature)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            losses.append(loss.item()); step += 1
        rec = {"epoch": ep + 1, "loss": float(np.mean(losses)), "time_s": round(time.time() - t0, 1),
               "gem_p": model.pool.p.item()}
        if (ep + 1) % a.eval_every == 0 or ep + 1 == a.epochs:
            res = run_benchmark(model, val_items, device, gray=a.gray, workers=a.workers, size=a.size)
            rec["val"] = headline(res)
            rec["val_score"] = selection_score(res)
            if rec["val_score"] > best:
                best = rec["val_score"]
                torch.save({"state_dict": model.state_dict(), "backbone": a.backbone, "dim": a.dim,
                            "gray": a.gray, "size": a.size, "epoch": ep + 1,
                            "val_threshold": res["verif_all"]["thr_at_EER"], "val": rec["val"]}, out / "best.pt")
                rec["saved"] = True
        hist.append(rec)
        (out / "history.json").write_text(json.dumps(hist, indent=2))
        print(json.dumps(rec))
    print(f"[train] best val score {best:.4f} -> {out / 'best.pt'}")


if __name__ == "__main__":
    main()
