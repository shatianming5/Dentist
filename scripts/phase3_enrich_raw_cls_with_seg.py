#!/usr/bin/env python3
"""Enrich raw_cls data with per-point segmentation probabilities.

For each case, determines which k-fold this case belongs to, loads the
segmentation model trained with that fold as the held-out test fold
(so the seg model never saw this case during training), and generates
per-point restoration probability.

Also transfers ground-truth segmentation labels from raw_seg data via
nearest-neighbor matching (oracle baseline).

Usage:
    cd scripts && python phase3_enrich_raw_cls_with_seg.py \
        --cls-root ../processed/raw_cls/v13_main4 \
        --seg-root ../processed/raw_seg/v1 \
        --kfold ../metadata/splits_raw_case_kfold.json \
        --seg-run-root ../runs/research_segcls_full/seg_pointnet \
        --out ../processed/raw_cls/v13_main4/seg_overlay \
        --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from _lib.device import normalize_device
from _lib.io import read_json, read_jsonl, write_json
from _lib.time import utc_now_iso
from phase3_train_raw_seg import PointNetSeg


def safe_case_name(case_key: str) -> str:
    s = str(case_key).replace("\\", "/").strip("/")
    s = s.replace("/", "__").replace(" ", "_")
    s = re.sub(r"[^0-9A-Za-z_\\-.\u4e00-\u9fff]+", "_", s)
    return s


def build_case_to_fold(kfold_path: Path) -> dict[str, int]:
    """Map each case_key to its fold index."""
    kfold = read_json(kfold_path)
    if "case_to_fold" in kfold:
        return {str(k): int(v) for k, v in kfold["case_to_fold"].items()}
    # Fallback: rebuild from folds dict
    folds = kfold.get("folds", {})
    out: dict[str, int] = {}
    for fold_idx, cases in folds.items():
        for case_key in cases:
            out[str(case_key)] = int(fold_idx)
    return out


@torch.no_grad()
def predict_seg_prob_ensemble(
    models: list[torch.nn.Module],
    points_np: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Average restoration probability across multiple seg models."""
    pts = torch.from_numpy(np.asarray(points_np, dtype=np.float32)).unsqueeze(0).to(device)
    probs = []
    for m in models:
        logits = m(pts)  # (1, 2, N)
        prob = torch.softmax(logits, dim=1)[0, 1]  # (N,)
        probs.append(prob)
    avg = torch.stack(probs, dim=0).mean(dim=0)
    return avg.cpu().numpy().astype(np.float32)


def transfer_seg_gt_nn(
    cls_points: np.ndarray,
    seg_points: np.ndarray,
    seg_labels: np.ndarray,
) -> np.ndarray:
    """Transfer GT seg labels from raw_seg to raw_cls points via nearest neighbor."""
    from scipy.spatial import cKDTree
    tree = cKDTree(seg_points)
    _, idx = tree.query(cls_points, k=1)
    return seg_labels[idx].astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description="Enrich raw_cls data with segmentation features.")
    ap.add_argument("--cls-root", type=Path, default=Path("../processed/raw_cls/v13_main4"))
    ap.add_argument("--seg-root", type=Path, default=Path("../processed/raw_seg/v1"))
    ap.add_argument("--kfold", type=Path, default=Path("../metadata/splits_raw_case_kfold.json"))
    ap.add_argument("--seg-run-root", type=Path, default=Path("../runs/research_segcls_full/seg_pointnet"))
    ap.add_argument("--seeds", type=str, default="1337,2020,2021")
    ap.add_argument("--out", type=Path, default=Path("../processed/raw_cls/v13_main4/seg_overlay"))
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seg-model-name", type=str, default="pointnet_seg")
    args = ap.parse_args()

    cls_root = args.cls_root.resolve()
    seg_root = args.seg_root.resolve()
    kfold_path = args.kfold.resolve()
    seg_run_root = args.seg_run_root.resolve()
    out_root = args.out.resolve()
    device_str = normalize_device(str(args.device))
    device = torch.device(device_str)
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    print(f"[enrich] cls_root = {cls_root}")
    print(f"[enrich] seg_root = {seg_root}")
    print(f"[enrich] kfold    = {kfold_path}")
    print(f"[enrich] seg_runs = {seg_run_root}")
    print(f"[enrich] seeds    = {seeds}")
    print(f"[enrich] device   = {device_str}")
    print(f"[enrich] out      = {out_root}")

    # Build case→fold mapping
    case_to_fold = build_case_to_fold(kfold_path)

    # Load cls index
    cls_rows = read_jsonl(cls_root / "index.jsonl")
    print(f"[enrich] {len(cls_rows)} cls samples")

    # Load seg index for GT transfer
    seg_rows = read_jsonl(seg_root / "index.jsonl")
    seg_by_case: dict[str, dict[str, Any]] = {}
    for r in seg_rows:
        seg_by_case[str(r["case_key"])] = r
    print(f"[enrich] {len(seg_rows)} seg samples")

    # Determine which folds we need
    folds_needed = sorted(set(case_to_fold.values()))
    print(f"[enrich] folds needed: {folds_needed}")

    # Load seg models per fold (ensemble across seeds)
    fold_models: dict[int, list[torch.nn.Module]] = {}
    for fold_idx in folds_needed:
        models = []
        for seed in seeds:
            run_name = f"seg_pointnet_fold{fold_idx}_seed{seed}"
            ckpt_path = seg_run_root / run_name / "ckpt_best.pt"
            if not ckpt_path.exists():
                print(f"[warn] missing checkpoint: {ckpt_path}")
                continue
            model = PointNetSeg(num_classes=2)
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
            state = ckpt.get("model", ckpt)
            model.load_state_dict(state, strict=True)
            model.to(device)
            model.eval()
            models.append(model)
        fold_models[fold_idx] = models
        print(f"[enrich] fold {fold_idx}: loaded {len(models)} seg models")

    # Process each case
    out_root.mkdir(parents=True, exist_ok=True)
    results = []
    missing_seg_gt = 0

    for i, row in enumerate(cls_rows, start=1):
        case_key = str(row["case_key"])
        cls_rel = str(row["sample_npz"])
        fold_idx = case_to_fold.get(case_key)
        if fold_idx is None:
            print(f"[warn] case {case_key} not in kfold splits, skipping")
            continue

        # Load cls points
        cls_npz = cls_root / cls_rel
        with np.load(cls_npz) as z:
            cls_points = np.asarray(z["points"], dtype=np.float32)

        n_pts = cls_points.shape[0]

        # Compute seg_prob using fold's ensemble
        models = fold_models.get(fold_idx, [])
        if models:
            seg_prob = predict_seg_prob_ensemble(models, cls_points, device)
        else:
            seg_prob = np.full(n_pts, 0.5, dtype=np.float32)
            print(f"[warn] no models for fold {fold_idx}, using 0.5 for {case_key}")

        # Transfer GT seg labels via NN
        seg_gt = np.zeros(n_pts, dtype=np.float32)
        seg_row = seg_by_case.get(case_key)
        if seg_row is not None:
            seg_npz = seg_root / str(seg_row["sample_npz"])
            with np.load(seg_npz) as z:
                seg_points = np.asarray(z["points"], dtype=np.float32)
                seg_labels = np.asarray(z["labels"], dtype=np.int64)
            seg_gt = transfer_seg_gt_nn(cls_points, seg_points, seg_labels)
        else:
            missing_seg_gt += 1

        # Save overlay
        out_path = out_root / cls_rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(out_path),
            seg_prob=seg_prob,
            seg_gt=seg_gt,
        )

        prob_mean = float(np.mean(seg_prob))
        gt_frac = float(np.mean(seg_gt))
        results.append({
            "case_key": case_key,
            "fold": fold_idx,
            "n_points": n_pts,
            "seg_prob_mean": round(prob_mean, 4),
            "seg_gt_frac": round(gt_frac, 4),
        })

        if i % 10 == 0 or i == len(cls_rows):
            print(f"[enrich] {i}/{len(cls_rows)} done")

    # Save summary
    summary = {
        "generated_at": utc_now_iso(),
        "cls_root": str(cls_root),
        "seg_root": str(seg_root),
        "seg_run_root": str(seg_run_root),
        "seeds": seeds,
        "device": device_str,
        "n_cases": len(results),
        "missing_seg_gt": missing_seg_gt,
        "results": results,
    }
    write_json(out_root / "enrichment_summary.json", summary)

    # Print stats
    prob_means = [r["seg_prob_mean"] for r in results]
    gt_fracs = [r["seg_gt_frac"] for r in results]
    print(f"\n[enrich] DONE: {len(results)} cases enriched")
    print(f"[enrich] seg_prob mean: {np.mean(prob_means):.4f} ± {np.std(prob_means):.4f}")
    print(f"[enrich] seg_gt  frac: {np.mean(gt_fracs):.4f} ± {np.std(gt_fracs):.4f}")
    print(f"[enrich] missing GT:   {missing_seg_gt}")
    print(f"[enrich] output:       {out_root}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
