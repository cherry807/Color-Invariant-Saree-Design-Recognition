"""Synthetic colorway generation.

A saree "design" is the spatial layout of its motifs; a "colorway" is the palette the
loom fills that layout with. We simulate new colorways by segmenting the image into its
dominant color regions (k-means in CIELAB) and re-painting each region, while keeping the
per-pixel residual (weave texture, shading, folds) so the result still looks like fabric.

Generators are split into two families:
  TRAIN_FAMILIES  - used as training augmentation
  HELDOUT_FAMILIES - used ONLY in evaluation, to check that invariance generalises beyond
                     the exact augmentations the model was trained on.

All functions take / return uint8 RGB arrays (H, W, 3) and an np.random.Generator.
"""
from __future__ import annotations

import cv2
import numpy as np

Palette = np.ndarray  # (k, 3) float32 Lab (OpenCV 8-bit scale: L,a,b in 0..255)


def to_lab(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)


def from_lab(lab: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def kmeans_lab(lab: np.ndarray, k: int, rng: np.random.Generator, iters: int = 8,
               sample: int = 4096) -> tuple[Palette, np.ndarray]:
    """Tiny k-means on a pixel subsample, then full-image assignment. Returns
    centroids sorted by lightness and an (H, W) label map."""
    px = lab.reshape(-1, 3)
    s = px[rng.choice(len(px), min(sample, len(px)), replace=False)]
    c = s[rng.choice(len(s), k, replace=False)].copy()
    for _ in range(iters):
        a = ((s[:, None, :] - c[None]) ** 2).sum(-1).argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                c[j] = s[m].mean(0)
    order = np.argsort(c[:, 0])
    c = c[order]
    labels = ((px[:, None, :] - c[None]) ** 2).sum(-1).argmin(1)
    return c, labels.reshape(lab.shape[:2])


def extract_palette(img: np.ndarray, k: int = 6, seed: int = 0) -> Palette:
    return kmeans_lab(to_lab(img), k, np.random.default_rng(seed))[0]


def random_palette(k: int, rng: np.random.Generator, min_dist: float = 35.0) -> Palette:
    """Random Lab palette whose colors stay mutually distinguishable, so a recolor never
    merges two motif regions into one (which would genuinely change the design)."""
    cols: list[np.ndarray] = []
    for _ in range(200 * k):
        c = np.array([rng.uniform(25, 235), 128 + rng.normal(0, 40), 128 + rng.normal(0, 40)],
                     np.float32)
        if all(np.linalg.norm(c - o) >= min_dist for o in cols):
            cols.append(c)
            if len(cols) == k:
                break
    while len(cols) < k:  # fall back to spreading lightness
        cols.append(np.array([255 * len(cols) / k, 128, 128], np.float32))
    return np.stack(cols)


def _fit_palette(src: Palette, tgt: Palette) -> Palette:
    """Resample a lightness-sorted target palette to len(src) entries (rank matching)."""
    tgt = tgt[np.argsort(tgt[:, 0])]
    idx = np.round(np.linspace(0, len(tgt) - 1, len(src))).astype(int)
    return tgt[idx]


def palette_remap(img: np.ndarray, rng: np.random.Generator, target: Palette | None = None,
                  k: int | None = None, mode: str = "random", shuffle_prob: float = 0.3) -> np.ndarray:
    """Repaint each k-means color region.
    mode='random'  : fresh random palette
    mode='permute' : shuffle the image's own colors between regions (gold-on-red -> red-on-gold)
    mode='transfer': map regions to `target` by lightness rank (palette of another saree)
    """
    lab = to_lab(img)
    k = k or int(rng.integers(3, 8))
    cent, _ = kmeans_lab(lab, k, rng)
    cent = merge_close(cent)  # shading of one dye must not become two colours
    k = len(cent)
    if mode == "transfer" and target is not None:
        new = _fit_palette(cent, target)
        if rng.random() < shuffle_prob:  # also allow non-rank-preserving assignments
            new = new[rng.permutation(k)]
    elif mode == "permute":
        new = cent[rng.permutation(k)]
    else:
        new = random_palette(k, rng)
    # soft assignment: pixels between two dyes blend smoothly instead of snapping (no blotches)
    w = soft_assign(lab, cent)
    resid = lab - w @ cent
    resid[..., 1:] *= rng.uniform(0.3, 1.0)  # keep texture in L, damp old chroma
    return from_lab(w @ new + resid)


def _lab_dist2(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Squared Lab distance with lightness down-weighted: folds/shadows change L, not the dye."""
    d = x[..., None, :] - c
    return 0.35 * d[..., 0] ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2


def merge_close(cent: Palette, thr: float = 22.0) -> Palette:
    cent = [c.copy() for c in cent]
    merged = True
    while merged and len(cent) > 2:
        merged = False
        C = np.stack(cent)
        D = np.sqrt(_lab_dist2(C, C)) + np.eye(len(C)) * 1e9
        i, j = np.unravel_index(D.argmin(), D.shape)
        if D[i, j] < thr:
            cent[i] = (C[i] + C[j]) / 2
            cent.pop(j)
            merged = True
    C = np.stack(cent)
    return C[np.argsort(C[:, 0])]


def soft_assign(lab: np.ndarray, cent: Palette, tau: float = 60.0) -> np.ndarray:
    d = _lab_dist2(lab, cent)
    d -= d.min(-1, keepdims=True)
    w = np.exp(-d / tau)
    return w / w.sum(-1, keepdims=True)


def hsv_shift(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(0, 180)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.4, 1.6), 0, 255)
    hsv[..., 2] = 255 * (hsv[..., 2] / 255) ** rng.uniform(0.6, 1.6)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def channel_shuffle_invert(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = img[..., rng.permutation(3)].copy()
    inv = rng.random(3) < 0.35
    out[..., inv] = 255 - out[..., inv]
    return out


def gray(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if rng.random() < 0.3:
        g = 255 - g
    return np.repeat(g[..., None], 3, -1)


# ---- held-out (evaluation-only) generators -------------------------------------------

def channel_curves(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random monotone tone curve per RGB channel (direction random) - a 'dye lot' change."""
    out = np.empty_like(img)
    xs = np.linspace(0, 255, 6)
    for ch in range(3):
        ys = np.sort(rng.uniform(0, 255, 6))
        ys[0], ys[-1] = rng.uniform(0, 60), rng.uniform(195, 255)
        if rng.random() < 0.5:
            ys = ys[::-1]
        lut = np.interp(np.arange(256), xs, ys).astype(np.uint8)
        out[..., ch] = lut[img[..., ch]]
    return out


def reinhard_transfer(img: np.ndarray, rng: np.random.Generator,
                      ref: np.ndarray | None = None) -> np.ndarray:
    """Global Lab mean/std transfer (Reinhard et al. 2001) to a reference or random target."""
    lab = to_lab(img)
    mu, sd = lab.reshape(-1, 3).mean(0), lab.reshape(-1, 3).std(0) + 1e-3
    if ref is not None:
        r = to_lab(ref).reshape(-1, 3)
        tmu, tsd = r.mean(0), r.std(0) + 1e-3
    else:
        tmu = np.array([rng.uniform(70, 190), 128 + rng.normal(0, 30), 128 + rng.normal(0, 30)])
        tsd = sd * rng.uniform(0.6, 1.5, 3)
    return from_lab((lab - mu) / sd * tsd + tmu)


TRAIN_FAMILIES = ("palette_random", "palette_permute", "palette_transfer", "hsv", "channel", "gray")
HELDOUT_FAMILIES = ("curves", "reinhard")


def apply_family(img: np.ndarray, family: str, rng: np.random.Generator,
                 target: Palette | None = None, ref: np.ndarray | None = None) -> np.ndarray:
    if family == "identity":
        return img
    if family == "palette_random":
        return palette_remap(img, rng, mode="random")
    if family == "palette_permute":
        return palette_remap(img, rng, mode="permute")
    if family == "palette_transfer":
        return palette_remap(img, rng, target=target, mode="transfer" if target is not None else "random")
    if family == "hsv":
        return hsv_shift(img, rng)
    if family == "channel":
        return channel_shuffle_invert(img, rng)
    if family == "gray":
        return gray(img, rng)
    if family == "curves":
        return channel_curves(img, rng)
    if family == "reinhard":
        return reinhard_transfer(img, rng, ref)
    raise ValueError(family)


TRAIN_WEIGHTS = {"identity": 0.08, "palette_random": 0.25, "palette_permute": 0.12,
                 "palette_transfer": 0.30, "hsv": 0.12, "channel": 0.08, "gray": 0.05}


def random_train_recolor(img: np.ndarray, rng: np.random.Generator,
                         target: Palette | None = None) -> np.ndarray:
    fams, w = zip(*TRAIN_WEIGHTS.items())
    fam = fams[rng.choice(len(fams), p=np.array(w) / sum(w))]
    return apply_family(img, fam, rng, target=target)
