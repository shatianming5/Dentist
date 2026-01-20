from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def cross_entropy_per_sample(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    weight: torch.Tensor | None,
    label_smoothing: float,
) -> torch.Tensor:
    try:
        return F.cross_entropy(
            logits,
            labels,
            weight=weight,
            reduction="none",
            label_smoothing=max(0.0, float(label_smoothing)),
        )
    except TypeError:
        return F.cross_entropy(logits, labels, weight=weight, reduction="none")


def supervised_contrastive_loss(z: torch.Tensor, labels: torch.Tensor, *, temperature: float) -> torch.Tensor:
    if z.ndim != 2:
        raise ValueError(f"Expected z (N,D), got {tuple(z.shape)}")
    if labels.ndim != 1 or labels.shape[0] != z.shape[0]:
        raise ValueError(f"Expected labels (N,), got {tuple(labels.shape)} for z {tuple(z.shape)}")

    t = float(temperature)
    if not math.isfinite(t) or t <= 0:
        raise ValueError(f"Invalid temperature: {temperature}")

    z = F.normalize(z, dim=1)
    sim = torch.matmul(z, z.t()) / t  # (N,N)
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()

    n = int(z.shape[0])
    eye = torch.eye(n, device=z.device, dtype=torch.bool)
    labels2 = labels.view(-1, 1)
    pos = (labels2 == labels2.t()) & (~eye)
    exp = torch.exp(sim) * (~eye)
    log_prob = sim - torch.log(exp.sum(dim=1, keepdim=True).clamp_min(1e-12))

    pos_count = pos.sum(dim=1)
    valid = pos_count > 0
    if not torch.any(valid):
        return z.new_zeros(())
    mean_log_prob_pos = (log_prob * pos.float()).sum(dim=1) / pos_count.clamp_min(1)
    return -mean_log_prob_pos[valid].mean()


def coral_loss_between_groups(z: torch.Tensor, domains: torch.Tensor, *, g0: int = 0, g1: int = 1) -> torch.Tensor:
    if z.ndim != 2:
        raise ValueError(f"Expected z (B,D), got {tuple(z.shape)}")
    if domains.ndim != 1 or domains.shape[0] != z.shape[0]:
        raise ValueError(f"Expected domains (B,), got {tuple(domains.shape)} for z {tuple(z.shape)}")
    m0 = domains == int(g0)
    m1 = domains == int(g1)
    if int(m0.sum()) < 2 or int(m1.sum()) < 2:
        return z.new_zeros(())
    z0 = z[m0]
    z1 = z[m1]
    z0 = z0 - z0.mean(dim=0, keepdim=True)
    z1 = z1 - z1.mean(dim=0, keepdim=True)
    c0 = (z0.t() @ z0) / float(max(1, int(z0.shape[0]) - 1))
    c1 = (z1.t() @ z1) / float(max(1, int(z1.shape[0]) - 1))
    d = float(z.shape[1])
    return ((c0 - c1) ** 2).mean() / max(1.0, d)

