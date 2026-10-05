from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve


def identification(q: torch.Tensor, q_lab, g: torch.Tensor, g_lab, ks=(1, 5, 10), donor_lab=None) -> dict:
    """Rank the gallery for every query. Returns Recall@k (CMC), mAP and optionally Trap@1."""
    q_lab, g_lab = torch.as_tensor(q_lab), torch.as_tensor(g_lab)
    sim = q @ g.T
    order = sim.argsort(dim=1, descending=True)
    hits = (g_lab[order] == q_lab[:, None])  # (Q, G) bool in rank order
    out = {f"R@{k}": hits[:, :k].any(1).float().mean().item() for k in ks if k <= g.shape[0]}
    cum = hits.float().cumsum(1)
    ranks = torch.arange(1, hits.shape[1] + 1).float()
    ap = ((cum / ranks) * hits).sum(1) / hits.sum(1).clamp(min=1)
    out["mAP"] = ap.mean().item()
    if donor_lab is not None:
        out["Trap@1"] = (g_lab[order[:, 0]] == torch.as_tensor(donor_lab)).float().mean().item()
    out["n_queries"] = int(q.shape[0])
    return out


def verification(scores: np.ndarray, same: np.ndarray, threshold: float | None = None) -> dict:
    scores, same = np.asarray(scores, float), np.asarray(same, bool)
    fpr, tpr, thr = roc_curve(same, scores)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    out = {"AUC": float(roc_auc_score(same, scores)), "EER": float((fpr[i] + fnr[i]) / 2),
           "thr_at_EER": float(thr[i])}
    for far in (1e-2, 1e-3):
        ok = fpr <= far
        out[f"TAR@FAR={far:g}"] = float(tpr[ok].max()) if ok.any() else 0.0
    if threshold is not None:  # threshold calibrated on the validation split
        pred = scores >= threshold
        out["threshold"] = float(threshold)
        out["acc@thr"] = float((pred == same).mean())
        out["TAR@thr"] = float(pred[same].mean())
        out["FAR@thr"] = float(pred[~same].mean())
    out["n_genuine"], out["n_impostor"] = int(same.sum()), int((~same).sum())
    return out
