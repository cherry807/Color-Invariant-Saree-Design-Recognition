"""Embedding network: pretrained timm backbone -> GeM pooling -> linear projection -> L2 norm."""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeM(nn.Module):
    """Generalised-mean pooling (Radenovic et al. 2018). p=1 is avg, p->inf is max; learnable p
    lets the net emphasise sparse, salient motif activations over plain background."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p))
        self.eps = eps

    def forward(self, x):
        # fp32 so x**p cannot overflow under fp16 autocast
        return x.float().clamp(min=self.eps).pow(self.p).mean((-2, -1)).pow(1.0 / self.p)


class SareeEmbedder(nn.Module):
    def __init__(self, backbone: str = "convnext_atto.d2_in1k", dim: int = 256, pretrained: bool = True):
        super().__init__()
        self.backbone_name, self.dim = backbone, dim
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="")
        c = self.backbone.num_features
        self.pool = GeM()
        self.head = nn.Sequential(nn.Linear(c, dim), nn.BatchNorm1d(dim)) if dim else nn.Identity()

    def features(self, x):
        f = self.backbone(x)
        if f.ndim == 3:  # ViT tokens (B, N, C) -> drop prefix tokens, average patches
            f = f[:, getattr(self.backbone, "num_prefix_tokens", 1):].mean(1)
            return f
        return self.pool(f)

    def forward(self, x):
        return F.normalize(self.head(self.features(x)), dim=-1)


def load_checkpoint(path: str, map_location="cpu") -> tuple[SareeEmbedder, dict]:
    ck = torch.load(path, map_location=map_location, weights_only=False)
    m = SareeEmbedder(ck["backbone"], ck["dim"], pretrained=False)
    m.load_state_dict(ck["state_dict"])
    return m.eval(), ck


def zero_shot_model(backbone: str) -> SareeEmbedder:
    """Pretrained backbone + GeM, no projection - the 'no training' baseline."""
    return SareeEmbedder(backbone, dim=0, pretrained=True).eval()
