"""Inference CLI: build a gallery index, identify a query, verify a pair.

  python -m sareeid.infer index    --ckpt best.pt --images gallery_dir --out gallery.pt
  python -m sareeid.infer identify --ckpt best.pt --gallery gallery.pt --image q.jpg --topk 5
  python -m sareeid.infer verify   --ckpt best.pt --image a.jpg --image2 b.jpg

Pre-processing: decode -> RGB -> resize short side to `size` -> centre crop -> ImageNet norm.
Post-processing: L2-normalised embedding, (optional) h-flip TTA; gallery images are grouped by
design label (sub-folder name, else file stem) and a design scores the MAX over its colourways,
so adding more colourways of a design to the gallery can only help it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import IMG_EXT, center_crop, read_rgb, to_tensor
from .model import load_checkpoint


class SareeMatcher:
    def __init__(self, ckpt: str, device: str | None = None, tta: bool = True):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, ck = load_checkpoint(ckpt)
        self.model.to(self.device)
        self.size, self.gray, self.tta = ck.get("size", 224), ck.get("gray", False), tta
        self.threshold = ck.get("val_threshold", 0.5)

    @torch.no_grad()
    def embed(self, paths: list[str], batch_size: int = 64) -> torch.Tensor:
        out = []
        for i in range(0, len(paths), batch_size):
            x = torch.stack([to_tensor(center_crop(read_rgb(p), self.size), self.gray)
                             for p in paths[i:i + batch_size]]).to(self.device)
            z = self.model(x)
            if self.tta:
                z = F.normalize(z + self.model(x.flip(-1)), dim=-1)
            out.append(z.cpu())
        return torch.cat(out)

    def build_index(self, folder: str) -> dict:
        root = Path(folder)
        paths = sorted(str(p) for p in root.rglob("*") if p.suffix.lower() in IMG_EXT)
        labels = [Path(p).parent.name if Path(p).parent != root else Path(p).stem for p in paths]
        return {"emb": self.embed(paths).half(), "paths": paths, "labels": labels}

    def identify(self, image: str, index: dict, topk: int = 5) -> list[tuple[str, float, str]]:
        q = self.embed([image])[0]
        sims = index["emb"].float() @ q
        best: dict[str, tuple[float, str]] = {}
        for s, lab, p in zip(sims.tolist(), index["labels"], index["paths"]):
            if lab not in best or s > best[lab][0]:
                best[lab] = (s, p)
        ranked = sorted(best.items(), key=lambda kv: -kv[1][0])[:topk]
        return [(lab, s, p) for lab, (s, p) in ranked]

    def verify(self, a: str, b: str) -> tuple[bool, float]:
        z = self.embed([a, b])
        s = float(z[0] @ z[1])
        return s >= self.threshold, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["index", "identify", "verify"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--images")
    ap.add_argument("--gallery")
    ap.add_argument("--image")
    ap.add_argument("--image2")
    ap.add_argument("--out", default="gallery.pt")
    ap.add_argument("--topk", type=int, default=5)
    a = ap.parse_args()
    m = SareeMatcher(a.ckpt)
    if a.cmd == "index":
        idx = m.build_index(a.images)
        torch.save(idx, a.out)
        print(f"indexed {len(idx['paths'])} images / {len(set(idx['labels']))} designs -> {a.out}")
    elif a.cmd == "identify":
        for rank, (lab, s, p) in enumerate(m.identify(a.image, torch.load(a.gallery), a.topk), 1):
            print(f"{rank}. {lab}  sim={s:.3f}  ({p})")
    else:
        same, s = m.verify(a.image, a.image2)
        print(f"{'SAME' if same else 'DIFFERENT'} design  sim={s:.3f}  threshold={m.threshold:.3f}")


if __name__ == "__main__":
    main()
