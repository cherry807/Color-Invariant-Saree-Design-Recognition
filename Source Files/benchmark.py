"""Deterministic colour-invariance benchmark built from a held-out split.

Every benchmark image is described by a small spec and rendered on the fly with a fixed seed,
so the benchmark is exactly reproducible, never stored on disk, and identical for every model.

Protocols (gallery always = one clean, centre-cropped image per held-out design):
  P1 colorway  - queries = each design re-coloured by EVERY generator family (6 train + 2
                 held-out), random crop/rotation/blur/JPEG. Measures R@1/R@5/mAP per family.
  P2 trap      - query = design d painted in the palette of another design d' (the 'donor').
                 A colour-biased model retrieves d'. We report R@1 and Trap@1 (top-1 == donor).
  P3 shared    - gallery = all designs in palette A, queries = all designs in palette B. Colour
                 carries zero information; only the motif can identify. Averaged over R draws.
  P4 rephoto   - real second photos of the same saree (near-duplicate groups), if any.
Verification pairs:
  genuine            (clean d, P1 query of d)            - cross-palette, same motif
  impostor_random    (clean d, P1 query of d')           - different motif
  impostor_samepal   (d in palette A, d' in palette A)   - different motif, IDENTICAL palette
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from . import recolor as rc
from .data import Item, ImageCache, center_crop, geo_photo_aug, random_resized_crop, read_rgb, to_tensor

FAMILIES = rc.TRAIN_FAMILIES + rc.HELDOUT_FAMILIES


def build_specs(items: list[Item], seed: int = 0, shared_rounds: int = 3,
                pairs_per_design: int = 2, size: int = 224) -> dict:
    rng = np.random.default_rng(seed)
    # one representative per group = the design; remaining group members = real re-photos
    reps, extra = {}, []
    for it in items:
        (extra.append(it) if it.group in reps else reps.__setitem__(it.group, it))
    designs = list(reps.values())
    n = len(designs)
    assert n >= 3, "need at least 3 held-out designs"
    paths = [d.path for d in designs]
    S = []  # all specs; roles reference indices into S

    def add(**kw):
        S.append(kw)
        return len(S) - 1

    def other(i):
        j = int(rng.integers(n - 1))
        return j + (j >= i)

    gallery = [add(op="orig", path=paths[i], label=i) for i in range(n)]
    p1 = {f: [] for f in FAMILIES}
    for i in range(n):
        for f in FAMILIES:
            d = other(i)
            p1[f].append(add(op="colorway", path=paths[i], label=i, family=f, donor=paths[d],
                             donor_label=d, seed=int(rng.integers(2**31))))
    trap = []
    for i in range(n):
        d = other(i)
        trap.append(add(op="colorway", path=paths[i], label=i, family="palette_transfer",
                        donor=paths[d], donor_label=d, seed=int(rng.integers(2**31)), strict=True))
    shared = []
    for r in range(shared_rounds):
        pa, pb = int(rng.integers(2**31)), int(rng.integers(2**31))
        g = [add(op="palette", path=paths[i], label=i, pal_seed=pa, seed=int(rng.integers(2**31)), aug=False)
             for i in range(n)]
        q = [add(op="palette", path=paths[i], label=i, pal_seed=pb, seed=int(rng.integers(2**31)), aug=True)
             for i in range(n)]
        shared.append((g, q))
    rephoto = [add(op="orig", path=it.path, label=[d.group for d in designs].index(it.group))
               for it in extra]

    # verification pairs (indices into S)
    gen, imp_r, imp_s = [], [], []
    allq = [(f, k) for f in FAMILIES for k in range(n)]
    for i in range(n):
        for _ in range(pairs_per_design):
            f = FAMILIES[int(rng.integers(len(FAMILIES)))]
            gen.append((gallery[i], p1[f][i]))
            j = other(i)
            imp_r.append((gallery[i], p1[f][j]))
            g, _ = shared[int(rng.integers(shared_rounds))]
            imp_s.append((g[i], g[other(i)]))
    return dict(specs=S, n_designs=n, gallery=gallery, p1=p1, trap=trap, shared=shared,
                rephoto=rephoto, pairs=dict(genuine=gen, impostor_random=imp_r, impostor_samepal=imp_s),
                size=size, n_queries_p1=len(allq))


class SpecDataset(Dataset):
    def __init__(self, specs: list[dict], size: int = 224, gray: bool = False):
        self.specs, self.size, self.gray = specs, size, gray
        self.load = ImageCache()
        self.pal_cache: dict = {}

    def __len__(self):
        return len(self.specs)

    def render(self, s: dict) -> np.ndarray:
        img = self.load(s["path"])
        if s["op"] == "orig":
            return center_crop(img, self.size)
        rng = np.random.default_rng(s["seed"])
        if s["op"] == "palette":
            v = center_crop(img, self.size)
            pal = rc.random_palette(6, np.random.default_rng(s["pal_seed"]))
            v = rc.palette_remap(v, rng, target=pal, k=6, mode="transfer", shuffle_prob=0.0)
            return geo_photo_aug(random_resized_crop(v, self.size, rng, scale=(0.6, 1.0)), rng, 0.7) \
                if s["aug"] else v
        # op == colorway
        v = random_resized_crop(img, self.size, rng, scale=(0.5, 1.0))
        donor = self.load(s["donor"])
        if s["donor"] not in self.pal_cache:
            self.pal_cache[s["donor"]] = rc.extract_palette(donor, k=6, seed=0)
        if s.get("strict"):
            v = rc.palette_remap(v, rng, target=self.pal_cache[s["donor"]], k=6, mode="transfer", shuffle_prob=0.0)
        else:
            v = rc.apply_family(v, s["family"], rng, target=self.pal_cache[s["donor"]],
                                ref=center_crop(donor, self.size))
        return geo_photo_aug(v, rng, 0.7)

    def __getitem__(self, i):
        return to_tensor(self.render(self.specs[i]), self.gray), i


@torch.no_grad()
def embed_specs(model, specs, device, batch_size=128, workers=2, gray=False, tta=False, size=224):
    ds = SpecDataset(specs, size=size, gray=gray)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=workers,
                                     persistent_workers=False)
    out = None
    model.eval()
    for x, idx in dl:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            z = model(x).float()
            if tta:
                z = torch.nn.functional.normalize(z + model(x.flip(-1)).float(), dim=-1)
        if out is None:
            out = torch.empty(len(ds), z.shape[1])
        out[idx] = z.cpu()
    return out
