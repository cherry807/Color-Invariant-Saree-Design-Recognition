"""python -m sareeid.prepare --roots <dir> [<dir> ...] --out work"""
import argparse

from .data import prepare

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True, help="folders scanned recursively for images")
    ap.add_argument("--out", default="work")
    ap.add_argument("--short", type=int, default=320, help="cache resolution (short side)")
    ap.add_argument("--hash-thr", type=int, default=6, help="pHash Hamming radius for near-duplicate grouping")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    prepare(a.roots, a.out, short=a.short, hash_thr=a.hash_thr, seed=a.seed)
