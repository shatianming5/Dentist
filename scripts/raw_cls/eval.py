from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .metrics import confusion_matrix, metrics_from_confusion


def tooth_position_to_group_id(tp: Any) -> int:
    s = str(tp or "")
    if s == "前磨牙":
        return 0
    if s == "磨牙":
        return 1
    return 2


def domain_group_ids(metas: list[dict[str, Any]], *, group_key: str) -> list[int]:
    key = str(group_key or "").strip()
    if not key or key == "none":
        return [0 for _ in metas]
    if key == "tooth_position":
        return [tooth_position_to_group_id(m.get("tooth_position")) for m in metas]
    raise ValueError(f"Unknown domain group_key: {group_key!r} (supported: tooth_position)")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    *,
    tta: int = 0,
    tta_seed: int = 0,
    domain_group_key: str = "",
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[list[float]] = []
    metas: list[dict[str, Any]] = []

    # TTA rotates geometry. For point-feature inputs (xyz + normals/curvature/radius),
    # only rotate xyz (+ normals if present). Use dataset-provided slices when available.
    ds = getattr(loader, "dataset", None)
    while ds is not None and not hasattr(ds, "_slice_xyz") and hasattr(ds, "dataset"):
        ds = getattr(ds, "dataset")
    slice_xyz = getattr(ds, "_slice_xyz", None)
    slice_normals = getattr(ds, "_slice_normals", None)
    if not isinstance(slice_xyz, slice):
        slice_xyz = slice(0, 3)
    if not isinstance(slice_normals, slice):
        slice_normals = None

    tta_n = int(tta or 0)
    gen: torch.Generator | None = None
    if tta_n > 1:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(tta_seed))
    two_pi = float(2.0 * math.pi)

    for points, extra, labels, meta in loader:
        points = points.to(device=device, dtype=torch.float32, non_blocking=True)
        extra = extra.to(device=device, dtype=torch.float32, non_blocking=True)
        labels = labels.to(device=device, dtype=torch.long, non_blocking=True)
        domains: torch.Tensor | None = None
        if str(domain_group_key or "").strip():
            gids = domain_group_ids(list(meta), group_key=str(domain_group_key))
            domains = torch.as_tensor(gids, device=device, dtype=torch.long)
        if tta_n > 1:
            b = int(points.shape[0])
            probs_accum = torch.zeros((b, int(num_classes)), device=device, dtype=torch.float32)
            for _ in range(tta_n):
                angles = torch.rand((b,), generator=gen, device=device, dtype=torch.float32) * two_pi
                c = torch.cos(angles)
                s = torch.sin(angles)
                rot = torch.zeros((b, 3, 3), device=device, dtype=torch.float32)
                rot[:, 0, 0] = c
                rot[:, 0, 1] = -s
                rot[:, 1, 0] = s
                rot[:, 1, 1] = c
                rot[:, 2, 2] = 1.0
                pts = points
                if points.ndim == 3 and int(points.shape[2]) >= 3:
                    pts = points.clone()
                    xyz = pts[:, :, slice_xyz]
                    if xyz.shape[-1] == 3:
                        pts[:, :, slice_xyz] = torch.matmul(xyz, rot.transpose(1, 2))
                    if slice_normals is not None:
                        nrm = pts[:, :, slice_normals]
                        if nrm.shape[-1] == 3:
                            pts[:, :, slice_normals] = torch.matmul(nrm, rot.transpose(1, 2))
                logits = model(pts, extra, domains=domains)
                probs_accum += F.softmax(logits, dim=1)
            probs = probs_accum / float(tta_n)
        else:
            logits = model(points, extra, domains=domains)
            probs = F.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)

        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(pred.detach().cpu().tolist())
        y_prob.extend(probs.detach().cpu().tolist())
        metas.extend(list(meta))

    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)
    cm = confusion_matrix(y_true_np, y_pred_np, num_classes=num_classes)
    rows: list[dict[str, Any]] = []
    for t, p, prob, m in zip(y_true, y_pred, y_prob, metas, strict=True):
        rows.append({**m, "y_true": int(t), "y_pred": int(p), "probs": [float(x) for x in prob]})
    return {"y_true": y_true_np, "y_pred": y_pred_np}, cm, rows


def save_confusion_csv(cm: np.ndarray, labels_by_id: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true/pred", *labels_by_id])
        for i, lab in enumerate(labels_by_id):
            w.writerow([lab, *cm[i].tolist()])


def save_errors_csv(
    rows: list[dict[str, Any]],
    labels_by_id: list[str],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["case_key", "split", "label", "y_true_label", "y_pred_label", "sample_npz", "p_pred"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            if int(r["y_true"]) == int(r["y_pred"]):
                continue
            probs = r.get("probs") or []
            y_pred = int(r["y_pred"])
            p_pred = float(probs[y_pred]) if 0 <= y_pred < len(probs) else 0.0
            w.writerow(
                {
                    "case_key": r.get("case_key"),
                    "split": r.get("split"),
                    "label": r.get("label"),
                    "y_true_label": labels_by_id[int(r["y_true"])],
                    "y_pred_label": labels_by_id[int(r["y_pred"])],
                    "sample_npz": r.get("sample_npz"),
                    "p_pred": f"{p_pred:.6f}",
                }
            )


def subset_rows(rows: list[dict[str, Any]], *, key: str, value: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        if str(r.get(key, "")) == value:
            out.append(r)
    return out


def metrics_for_rows(rows: list[dict[str, Any]], labels_by_id: list[str]) -> dict[str, Any]:
    if not rows:
        return {"total": 0}
    y_true = np.asarray([int(r["y_true"]) for r in rows], dtype=np.int64)
    y_pred = np.asarray([int(r["y_pred"]) for r in rows], dtype=np.int64)
    cm = confusion_matrix(y_true, y_pred, num_classes=len(labels_by_id))
    out = metrics_from_confusion(cm, labels_by_id)
    out["confusion"] = cm.tolist()
    return out


def calibration_basic(rows: list[dict[str, Any]], num_classes: int, *, n_bins: int = 15) -> dict[str, Any]:
    if not rows:
        return {"total": 0}
    probs = np.asarray([r.get("probs") or [] for r in rows], dtype=np.float64)
    y_true = np.asarray([int(r.get("y_true", 0)) for r in rows], dtype=np.int64)
    if probs.ndim != 2 or probs.shape[0] != y_true.shape[0] or probs.shape[1] != int(num_classes):
        return {"total": int(y_true.shape[0]), "error": "invalid probs shape"}

    eps = 1e-12
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == y_true).astype(np.float64)

    # ECE with equally spaced confidence bins.
    n_bins_i = max(2, int(n_bins))
    bins = np.linspace(0.0, 1.0, num=n_bins_i + 1)
    ece = 0.0
    for i in range(n_bins_i):
        lo = bins[i]
        hi = bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if not np.any(mask):
            continue
        w = float(np.mean(mask))
        bin_acc = float(np.mean(acc[mask]))
        bin_conf = float(np.mean(conf[mask]))
        ece += w * abs(bin_acc - bin_conf)

    p_true = probs[np.arange(y_true.shape[0]), y_true]
    nll = float(-np.mean(np.log(np.clip(p_true, eps, 1.0))))
    onehot = np.zeros_like(probs, dtype=np.float64)
    onehot[np.arange(y_true.shape[0]), y_true] = 1.0
    brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))

    return {
        "total": int(y_true.shape[0]),
        "ece": float(ece),
        "nll": float(nll),
        "brier": float(brier),
        "mean_conf": float(np.mean(conf)),
        "accuracy": float(np.mean(acc)),
    }

