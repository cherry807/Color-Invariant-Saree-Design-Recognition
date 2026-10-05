import torch
import torch.nn.functional as F


def supcon_loss(z: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Multi-positive supervised contrastive loss (Khosla et al. 2020).

    z: (N, D) L2-normalised embeddings, labels: (N,) design ids. Every other view of the same
    design is a positive; every view of any other design (including the ones rendered in the
    *same* palette, thanks to the shared palette bank) is a negative.
    """
    z = F.normalize(z.float(), dim=-1)
    sim = z @ z.T / temperature
    n = z.shape[0]
    self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
    pos = (labels[:, None] == labels[None, :]) & ~self_mask
    sim = sim.masked_fill(self_mask, float("-inf"))
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    n_pos = pos.sum(1)
    loss = -(log_prob.masked_fill(~pos, 0).sum(1)) / n_pos.clamp(min=1)
    return loss[n_pos > 0].mean()
