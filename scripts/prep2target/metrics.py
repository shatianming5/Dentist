from __future__ import annotations

from typing import Any

import torch


def chamfer_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.ndim != 3 or target.ndim != 3 or pred.shape[-1] != 3 or target.shape[-1] != 3:
        raise ValueError(f"Invalid shapes: pred={tuple(pred.shape)} target={tuple(target.shape)}")
    d = torch.cdist(pred, target, p=2.0)
    return d.min(dim=2).values.mean() + d.min(dim=1).values.mean()


def margin_loss(pred: torch.Tensor, margin_pts: torch.Tensor) -> torch.Tensor:
    if pred.ndim != 3 or margin_pts.ndim != 3:
        raise ValueError(f"Invalid shapes: pred={tuple(pred.shape)} margin={tuple(margin_pts.shape)}")
    d = torch.cdist(margin_pts, pred, p=2.0)  # (B, K, N)
    return d.min(dim=2).values.mean()


def margin_loss_per_sample(pred: torch.Tensor, margin_pts: torch.Tensor) -> torch.Tensor:
    if pred.ndim != 3 or margin_pts.ndim != 3:
        raise ValueError(f"Invalid shapes: pred={tuple(pred.shape)} margin={tuple(margin_pts.shape)}")
    d = torch.cdist(margin_pts, pred, p=2.0)  # (B, K, N)
    return d.min(dim=2).values.mean(dim=1)


def occlusion_penalty(raw_pred: torch.Tensor, opp_pts: torch.Tensor, clearance: float) -> torch.Tensor:
    if raw_pred.ndim != 3 or opp_pts.ndim != 3:
        raise ValueError(f"Invalid shapes: raw_pred={tuple(raw_pred.shape)} opp={tuple(opp_pts.shape)}")
    d = torch.cdist(raw_pred, opp_pts, p=2.0)  # (B, N, M)
    min_d = d.min(dim=2).values  # (B, N)
    pen = torch.relu(float(clearance) - min_d)  # (B, N)
    return pen.mean()


def occlusion_penalty_per_sample(raw_pred: torch.Tensor, opp_pts: torch.Tensor, clearance: float) -> torch.Tensor:
    if raw_pred.ndim != 3 or opp_pts.ndim != 3:
        raise ValueError(f"Invalid shapes: raw_pred={tuple(raw_pred.shape)} opp={tuple(opp_pts.shape)}")
    d = torch.cdist(raw_pred, opp_pts, p=2.0)  # (B, N, M)
    min_d = d.min(dim=2).values  # (B, N)
    pen = torch.relu(float(clearance) - min_d)  # (B, N)
    return pen.mean(dim=1)


def occlusion_penalty_with_stats(
    raw_pred: torch.Tensor, opp_pts: torch.Tensor, clearance: float
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    pen_mean = occlusion_penalty(raw_pred, opp_pts, clearance)
    d = torch.cdist(raw_pred, opp_pts, p=2.0)  # (B, N, M)
    min_d = d.min(dim=2).values  # (B, N)
    pen = torch.relu(float(clearance) - min_d)  # (B, N)
    stats: dict[str, Any] = {
        "pen_mean": float(pen.mean().item()),
        "pen_max": float(pen.max().item()),
        "pen_p95": float(torch.quantile(pen.flatten(), 0.95).item()),
    }
    return pen_mean, {str(k): float(v) for k, v in stats.items()}, min_d

