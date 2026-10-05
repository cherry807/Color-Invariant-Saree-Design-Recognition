"""Data preparation and datasets.

prepare(): scan raw roots -> cache resized copies -> group near-duplicates -> split by group.
The split is done on *groups*, so two photos of the same physical saree can never end up on
both sides of the train/test boundary (the most common leakage bug in retrieval benchmarks).
"""
from __future__ import annotations

import csv
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from . import recolor as rc

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def read_rgb(path: str | os.PathLike) -> np.ndarray | None:
    buf = np.fromfile(str(path), np.uint8)  # np.fromfile handles non-ASCII Windows paths
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_short(img: np.ndarray, short: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = short / min(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def phash(img: np.ndarray) -> list[int]:
    """64-bit DCT perceptual hashes on grayscale for all 8 rotations/flips (index 0 = as-is).
    Training treats rotated/flipped copies as the same design, so a rotated copy of a test image
    sitting in train would leak - grouping therefore matches against every orientation."""
    g0 = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (32, 32), interpolation=cv2.INTER_AREA)
    hs = []
    for g in (g0, g0[:, ::-1]):
        for r in range(4):
            d = cv2.dct(np.ascontiguousarray(np.rot90(g, r)).astype(np.float32))[:8, :8].flatten()
            hs.append(int("".join("1" if b else "0" for b in d > np.median(d[1:])), 2))
    return hs


_POP8 = np.array([bin(i).count("1") for i in range(256)], np.uint8)


def _popcount(x: np.ndarray) -> np.ndarray:
    return _POP8[x.view(np.uint8).reshape(*x.shape, 8)].sum(-1)


def source_stem(path: str) -> str:
    """Roboflow exports save several augmented copies of one photo as `<name>_jpg.rf.<hash>.jpg`;
    everything before `.rf.` identifies the source photo."""
    name = Path(path).name
    return name.split(".rf.")[0] if ".rf." in name else name


def _group_by_hash(hashes: list[list[int]], thr: int, keys: list[str] | None = None) -> list[int]:
    """Union-find over pairs whose min Hamming distance (over the 8 orientations) <= thr,
    plus pairs sharing the same source key (e.g. Roboflow augmentations of one photo)."""
    n = len(hashes)
    H = np.array(hashes, dtype=np.uint64)  # (n, 8)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n - 1):
        dist = _popcount(np.bitwise_xor(H[i + 1:, :1], H[i][None, :])).min(1)
        for j in np.nonzero(dist <= thr)[0]:
            a, b = find(i), find(i + 1 + int(j))
            if a != b:
                parent[b] = a
    if keys is not None:
        first: dict[str, int] = {}
        for i, k in enumerate(keys):
            if k in first:
                a, b = find(first[k]), find(i)
                if a != b:
                    parent[b] = a
            else:
                first[k] = i
    roots = [find(i) for i in range(n)]
    remap = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    return [remap[r] for r in roots]


def prepare(roots: list[str], out_dir: str, short: int = 320, min_side: int = 96,
            hash_thr: int = 6, split=(0.7, 0.1, 0.2), seed: int = 0) -> str:
    """Build cache + manifest.csv (columns: path, source, orig, group, split)."""
    out = Path(out_dir)
    (out / "cache").mkdir(parents=True, exist_ok=True)
    rows, seen_md5 = [], set()
    n_bad = n_small = n_dup = 0
    for root in roots:
        root_p = Path(root)
        # kagglehub caches datasets as .../<slug>/versions/<n>: name the source by the slug
        src_name = root_p.parent.parent.name if root_p.name.isdigit() and root_p.parent.name == "versions" else root_p.name
        files = sorted(p for p in root_p.rglob("*") if p.suffix.lower() in IMG_EXT and p.is_file())
        print(f"[prepare] {root} ({src_name}): {len(files)} image files")
        for n, p in enumerate(files, 1):
            if n % 500 == 0:
                print(f"  {n}/{len(files)}", flush=True)
            raw = p.read_bytes()  # read once: network drives (Colab/Drive) are slow
            md5 = hashlib.md5(raw).hexdigest()
            if md5 in seen_md5:  # byte-identical file listed twice
                n_dup += 1
                continue
            bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR) if raw else None
            img = None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if img is None:
                n_bad += 1
                continue
            if min(img.shape[:2]) < min_side:
                n_small += 1
                continue
            seen_md5.add(md5)
            img = resize_short(img, short)
            cp = out / "cache" / f"{md5}.jpg"
            if not cp.exists():
                cv2.imwrite(str(cp), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            rows.append({"path": str(cp), "source": src_name, "orig": str(p), "hash": phash(img)})
    keys = [r["source"] + "/" + source_stem(r["orig"]) if ".rf." in r["orig"] else r["orig"] for r in rows]
    if len(rows) < 10:
        raise RuntimeError(f"only {len(rows)} usable images found under {roots} "
                           f"({n_bad} unreadable, {n_small} too small) - check the dataset paths")
    groups = _group_by_hash([r["hash"] for r in rows], hash_thr, keys)
    rng = np.random.default_rng(seed)
    ug = rng.permutation(max(groups) + 1)
    n_tr, n_va = int(split[0] * len(ug)), int(split[1] * len(ug))
    split_of = {g: ("train" if i < n_tr else "val" if i < n_tr + n_va else "test") for i, g in enumerate(ug)}
    for r, g in zip(rows, groups):
        r["group"], r["split"] = g, split_of[g]
        del r["hash"]
    mpath = out / "manifest.csv"
    with open(mpath, "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "source", "orig", "group", "split"])
        w.writeheader()
        w.writerows(rows)
    multi = sum(1 for g in set(groups) if groups.count(g) > 1)
    print(f"[prepare] kept {len(rows)} images in {len(set(groups))} groups "
          f"({multi} groups with >1 photo) | dropped: {n_bad} unreadable, {n_small} too small, "
          f"{n_dup} byte-duplicates")
    for s in ("train", "val", "test"):
        print(f"  {s}: {sum(r['split'] == s for r in rows)} images")
    return str(mpath)


@dataclass
class Item:
    path: str
    group: int
    split: str
    source: str


def load_manifest(path: str, split: str | None = None) -> list[Item]:
    with open(path, encoding="utf8") as f:
        items = [Item(r["path"], int(r["group"]), r["split"], r["source"]) for r in csv.DictReader(f)]
    return [i for i in items if split is None or i.split == split]


# ---- geometric / photometric augmentation (numpy, applied to uint8 RGB) --------------

def random_resized_crop(img, size, rng, scale=(0.35, 1.0), ratio=(0.75, 1.333)):
    h, w = img.shape[:2]
    for _ in range(10):
        area = h * w * rng.uniform(*scale)
        ar = np.exp(rng.uniform(np.log(ratio[0]), np.log(ratio[1])))
        cw, ch = int(round(np.sqrt(area * ar))), int(round(np.sqrt(area / ar)))
        if 0 < cw <= w and 0 < ch <= h:
            x, y = rng.integers(0, w - cw + 1), rng.integers(0, h - ch + 1)
            return cv2.resize(img[y:y + ch, x:x + cw], (size, size), interpolation=cv2.INTER_AREA)
    return center_crop(img, size)


def center_crop(img, size):
    img = resize_short(img, size) if min(img.shape[:2]) > size else cv2.resize(
        img, None, fx=size / min(img.shape[:2]), fy=size / min(img.shape[:2]))
    h, w = img.shape[:2]
    y, x = (h - size) // 2, (w - size) // 2
    return np.ascontiguousarray(img[y:y + size, x:x + size])


def geo_photo_aug(img, rng, strength=1.0):
    """Orientation-free fabric augmentation + capture artefacts (blur, noise, JPEG, light)."""
    if rng.random() < 0.5:
        img = img[:, ::-1]
    img = np.rot90(img, int(rng.integers(0, 4)))
    if rng.random() < 0.5 * strength:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-20, 20), rng.uniform(1.0, 1.25))
        img = cv2.warpAffine(np.ascontiguousarray(img), M, (w, h), borderMode=cv2.BORDER_REFLECT)
    img = np.ascontiguousarray(img)
    if rng.random() < 0.3 * strength:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.5))
    if rng.random() < 0.3 * strength:  # uneven illumination (shadow / fold)
        h, w = img.shape[:2]
        gx = np.linspace(rng.uniform(0.6, 1.0), rng.uniform(1.0, 1.3), w)[None, :, None]
        img = np.clip(img * gx, 0, 255).astype(np.uint8)
    if rng.random() < 0.2 * strength:
        img = np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.3 * strength:
        _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(40, 90))])
        img = cv2.imdecode(enc, cv2.IMREAD_UNCHANGED)
    return img


def to_tensor(img: np.ndarray, gray: bool = False) -> torch.Tensor:
    if gray:
        img = np.repeat(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[..., None], 3, -1)
    x = (img.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


class ImageCache:
    """Per-worker LRU-free memo of decoded cached images (the cache is already small)."""

    def __init__(self):
        self.mem: dict[str, np.ndarray] = {}

    def __call__(self, path):
        if path not in self.mem:
            self.mem[path] = read_rgb(path)
        return self.mem[path]


class MultiViewTrainSet(Dataset):
    """Returns K independently recolored + augmented views of one image (one design).

    Palette sharing: recolor targets are drawn from a small *palette bank* that is shared by
    the whole batch, so the batch is full of different designs wearing identical palettes.
    In the contrastive loss those are negatives - colour alone can no longer separate them.
    """

    def __init__(self, items: list[Item], size=224, views=4, bank_size=24, seed=0, gray=False):
        self.items, self.size, self.views, self.gray = items, size, views, gray
        self.load = ImageCache()
        rng = np.random.default_rng(seed)
        print(f"[train] extracting palettes for {len(items)} images ...")
        self.palettes = [rc.extract_palette(read_rgb(it.path), k=6, seed=i) for i, it in enumerate(items)]
        self.bank_size = bank_size
        self.set_epoch(0)
        self._rng = rng
        # one label per group: multiple photos of the same saree are positives too
        uniq = {g: i for i, g in enumerate(dict.fromkeys(it.group for it in items))}
        self.labels = [uniq[it.group] for it in items]

    def set_epoch(self, epoch: int):
        if self.bank_size == 0:  # ablation: independent per-sample targets, no sharing
            self.bank = None
            return
        rng = np.random.default_rng(10_000 + epoch)
        n_real = self.bank_size // 2
        real = [self.palettes[i] for i in rng.choice(len(self.palettes), n_real)]
        self.bank = real + [rc.random_palette(6, rng) for _ in range(self.bank_size - n_real)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rng = np.random.default_rng()
        img = self.load(self.items[idx].path)
        views = []
        for _ in range(self.views):
            v = random_resized_crop(img, self.size, rng)
            if self.bank is not None:
                target = self.bank[rng.integers(len(self.bank))]
            elif rng.random() < 0.5:
                target = self.palettes[rng.integers(len(self.palettes))]
            else:
                target = rc.random_palette(6, rng)
            v = rc.random_train_recolor(v, rng, target=target)
            v = geo_photo_aug(v, rng)
            views.append(to_tensor(v, self.gray))
        return torch.stack(views), self.labels[idx]
