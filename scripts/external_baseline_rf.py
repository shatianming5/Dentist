#!/usr/bin/env python3
"""External baseline: Random Forest on hand-crafted per-case features.

Uses geometry statistics (curvature, surface area, point density, etc.)
and optionally segmentation mask statistics as input features for a
simple Random Forest classifier. Same 5-fold protocol as the deep models.
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore", category=FutureWarning)


def extract_geometry_features(points: np.ndarray, *, k: int = 20) -> dict[str, float]:
    """Extract hand-crafted geometry features from a point cloud."""
    n = points.shape[0]
    if n < 3:
        return {k: 0.0 for k in [
            "n_points", "bbox_vol", "bbox_diag", "spread_xyz_0", "spread_xyz_1", "spread_xyz_2",
            "mean_nn_dist", "std_nn_dist", "mean_curvature", "std_curvature",
            "pca_ratio_01", "pca_ratio_02",
        ]}

    # BBox features
    pmin, pmax = points.min(0), points.max(0)
    bbox_extent = pmax - pmin
    bbox_vol = float(np.prod(bbox_extent + 1e-12))
    bbox_diag = float(np.linalg.norm(bbox_extent))

    # Spread (std per axis)
    spread = points.std(0)

    # NN distances
    tree = cKDTree(points)
    kk = min(k, n - 1)
    dd, _ = tree.query(points, k=kk + 1)
    nn_dists = dd[:, 1:]  # exclude self
    mean_nn = float(nn_dists.mean())
    std_nn = float(nn_dists.std())

    # Local PCA curvature estimate
    _, ii = tree.query(points, k=min(k, n))
    curvatures = []
    for i in range(min(n, 500)):
        neighbors = points[ii[i]]
        if neighbors.shape[0] < 3:
            continue
        cov = np.cov(neighbors.T)
        try:
            evals = np.linalg.eigvalsh(cov)
            evals = np.sort(np.abs(evals))
            curv = evals[0] / (evals.sum() + 1e-12)
            curvatures.append(curv)
        except Exception:
            continue
    mean_curv = float(np.mean(curvatures)) if curvatures else 0.0
    std_curv = float(np.std(curvatures)) if curvatures else 0.0

    # Global PCA
    cov_global = np.cov(points.T)
    try:
        evals_g = np.sort(np.abs(np.linalg.eigvalsh(cov_global)))[::-1]
        pca_01 = evals_g[0] / (evals_g[1] + 1e-12)
        pca_02 = evals_g[0] / (evals_g[2] + 1e-12)
    except Exception:
        pca_01 = pca_02 = 1.0

    return {
        "n_points": float(n),
        "bbox_vol": bbox_vol,
        "bbox_diag": bbox_diag,
        "spread_xyz_0": float(spread[0]),
        "spread_xyz_1": float(spread[1]),
        "spread_xyz_2": float(spread[2]),
        "mean_nn_dist": mean_nn,
        "std_nn_dist": std_nn,
        "mean_curvature": mean_curv,
        "std_curvature": std_curv,
        "pca_ratio_01": pca_01,
        "pca_ratio_02": pca_02,
    }


def extract_seg_features(seg_prob: np.ndarray, seg_gt: np.ndarray) -> dict[str, float]:
    """Extract segmentation-related statistics."""
    return {
        "seg_prob_mean": float(np.mean(seg_prob)),
        "seg_prob_std": float(np.std(seg_prob)),
        "seg_prob_p90": float(np.quantile(seg_prob, 0.9)),
        "seg_prob_p10": float(np.quantile(seg_prob, 0.1)),
        "seg_prob_frac_above_05": float(np.mean(seg_prob > 0.5)),
        "seg_gt_frac": float(np.mean(seg_gt)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls-root", type=Path, default=Path("processed/raw_cls/v13_main4"))
    ap.add_argument("--seg-overlay", type=Path, default=Path("processed/raw_cls/v13_main4/seg_overlay"))
    ap.add_argument("--kfold", type=Path, default=Path("metadata/splits_raw_case_kfold.json"))
    ap.add_argument("--out", type=Path, default=Path("paper_tables/external_baseline_rf.md"))
    ap.add_argument("--seeds", type=str, default="1337,2020,2021")
    ap.add_argument("--use-seg", action="store_true", help="Include seg features (predicted).")
    ap.add_argument("--use-seg-gt", action="store_true", help="Include oracle seg features.")
    args = ap.parse_args()

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, confusion_matrix

    cls_root = args.cls_root.resolve()
    seg_overlay = args.seg_overlay.resolve()
    kfold_path = args.kfold.resolve()
    seeds = [int(s) for s in args.seeds.split(",")]

    # Load data
    with open(cls_root / "index.jsonl") as f:
        rows = [json.loads(l) for l in f]
    with open(cls_root / "label_map.json") as f:
        label_map = json.load(f)
    with open(kfold_path) as f:
        kfold = json.load(f)

    case_to_fold = kfold.get("case_to_fold", {})
    if not case_to_fold:
        for fold_str, cases in kfold.get("folds", {}).items():
            for ck in cases:
                case_to_fold[str(ck)] = int(fold_str)

    # Extract features for all cases
    print(f"Extracting features from {len(rows)} cases...")
    features_all = []
    labels_all = []
    folds_all = []
    case_keys_all = []

    for i, row in enumerate(rows):
        case_key = str(row["case_key"])
        label = str(row["label"])
        fold = case_to_fold.get(case_key, -1)
        rel = str(row["sample_npz"])

        with np.load(cls_root / rel) as z:
            points = np.asarray(z["points"], dtype=np.float32)

        geom = extract_geometry_features(points)
        feat_dict = dict(geom)

        if args.use_seg or args.use_seg_gt:
            overlay_path = seg_overlay / rel
            if overlay_path.exists():
                with np.load(overlay_path) as ov:
                    seg_prob = np.asarray(ov.get("seg_prob", np.full(points.shape[0], 0.5)), dtype=np.float32)
                    seg_gt = np.asarray(ov.get("seg_gt", np.zeros(points.shape[0])), dtype=np.float32)
                seg_feats = extract_seg_features(seg_prob, seg_gt)
                if args.use_seg:
                    for k, v in seg_feats.items():
                        if not k.startswith("seg_gt"):
                            feat_dict[k] = v
                if args.use_seg_gt:
                    feat_dict["seg_gt_frac"] = seg_feats["seg_gt_frac"]

        features_all.append(feat_dict)
        labels_all.append(label_map[label])
        folds_all.append(fold)
        case_keys_all.append(case_key)

    feat_names = sorted(features_all[0].keys())
    X = np.array([[f[k] for k in feat_names] for f in features_all], dtype=np.float32)
    y = np.array(labels_all, dtype=np.int64)
    folds = np.array(folds_all, dtype=np.int64)

    print(f"Features: {feat_names}")
    print(f"X shape: {X.shape}, labels: {Counter(y.tolist())}")

    n_folds = int(kfold.get("k", 5))
    all_results = []

    for seed in seeds:
        for test_fold in range(n_folds):
            val_fold = (test_fold + 1) % n_folds
            train_mask = (folds != test_fold) & (folds != val_fold)
            test_mask = folds == test_fold

            X_train, y_train = X[train_mask], y[train_mask]
            X_test, y_test = X[test_mask], y[test_mask]

            rf = RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            )
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            per_class_f1 = f1_score(y_test, y_pred, average=None, zero_division=0)
            cm = confusion_matrix(y_test, y_pred, labels=list(range(len(label_map))))

            all_results.append({
                "seed": seed,
                "fold": test_fold,
                "acc": acc,
                "macro_f1": macro_f1,
                "bal_acc": bal_acc,
                "per_class_f1": per_class_f1.tolist(),
                "confusion_matrix": cm.tolist(),
            })
            print(f"  fold={test_fold} seed={seed}: acc={acc:.4f} macro_f1={macro_f1:.4f} bal_acc={bal_acc:.4f}")

    # Aggregate
    accs = [r["acc"] for r in all_results]
    f1s = [r["macro_f1"] for r in all_results]
    bals = [r["bal_acc"] for r in all_results]

    id_to_label = {v: k for k, v in label_map.items()}
    per_class = np.array([r["per_class_f1"] for r in all_results])

    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# External Baseline: Random Forest",
        "",
        f"- features: {', '.join(feat_names)}",
        f"- use_seg_prob: {args.use_seg}",
        f"- use_seg_gt: {args.use_seg_gt}",
        f"- seeds: {seeds}",
        f"- folds: {n_folds}",
        f"- runs: {len(all_results)}",
        "",
        "## Aggregate Results",
        "",
        f"| Metric | Mean ± Std |",
        f"|--------|-----------|",
        f"| Test Accuracy | {np.mean(accs):.4f} ± {np.std(accs):.4f} |",
        f"| Test Macro-F1 | {np.mean(f1s):.4f} ± {np.std(f1s):.4f} |",
        f"| Test Balanced Acc | {np.mean(bals):.4f} ± {np.std(bals):.4f} |",
        "",
        "## Per-Class F1",
        "",
        "| Class | Mean ± Std |",
        "|-------|-----------|",
    ]
    for c in range(per_class.shape[1]):
        label_name = id_to_label.get(c, str(c))
        lines.append(f"| {label_name} | {per_class[:, c].mean():.4f} ± {per_class[:, c].std():.4f} |")

    lines.extend(["", "## Per-Fold Results", "", "| Fold | Seed | Acc | Macro-F1 | Bal Acc |", "|------|------|-----|----------|---------|"])
    for r in all_results:
        lines.append(f"| {r['fold']} | {r['seed']} | {r['acc']:.4f} | {r['macro_f1']:.4f} | {r['bal_acc']:.4f} |")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nResults written to {out}")
    print(f"Macro-F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
