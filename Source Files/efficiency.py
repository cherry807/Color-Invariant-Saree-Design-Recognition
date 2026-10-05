"""Params / FLOPs / latency / embedding size.

  python -m sareeid.efficiency --backbone convnext_atto.d2_in1k [--ckpt best.pt] --out eff.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

from .model import SareeEmbedder, load_checkpoint


@torch.no_grad()
def latency_ms(model, x, n=50, warm=10):
    # fp16 on GPU via autocast (same as eval); GeM deliberately pools in fp32
    amp = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=x.is_cuda)
    with amp:
        for _ in range(warm):
            model(x)
    if x.is_cuda:
        torch.cuda.synchronize()
    t = time.perf_counter()
    with amp:
        for _ in range(n):
            model(x)
    if x.is_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


def report(model: SareeEmbedder, size: int = 224) -> dict:
    model = model.eval()
    x = torch.randn(1, 3, size, size)
    with FlopCounterMode(display=False) as fc:
        model(x)
    out = {
        "backbone": model.backbone_name,
        "params_M": sum(p.numel() for p in model.parameters()) / 1e6,
        # FlopCounterMode counts 1 MAC = 2 FLOPs; papers/timm usually quote GMACs as "GFLOPs"
        "GFLOPs_per_image": fc.get_total_flops() / 1e9,
        "GMACs_per_image": fc.get_total_flops() / 2e9,
        "embedding_dim": model.dim or model.backbone.num_features,
        "input": f"{size}x{size}",
        "cpu_threads": torch.get_num_threads(),
        "cpu_latency_ms_bs1": latency_ms(model, x),
    }
    out["embedding_bytes_fp16"] = out["embedding_dim"] * 2
    if torch.cuda.is_available():
        m = model.cuda()
        out["gpu"] = torch.cuda.get_device_name(0)
        out["gpu_latency_ms_bs1_fp16"] = latency_ms(m, x.cuda())
        xb = torch.randn(64, 3, size, size, device="cuda")
        out["gpu_throughput_img_s_bs64_fp16"] = 64 / (latency_ms(m, xb, n=20) / 1000)
        model.cpu()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="convnext_atto.d2_in1k")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--ckpt")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--out")
    a = ap.parse_args()
    m = load_checkpoint(a.ckpt)[0] if a.ckpt else SareeEmbedder(a.backbone, a.dim, pretrained=False)
    r = report(m, a.size)
    print(json.dumps(r, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(r, indent=2))
