#!/usr/bin/env python3
"""RF per-point segmentation baseline.

Uses hand-crafted per-point features (normals, curvature, height, local density)
with sklearn RandomForestClassifier. Matches the protocol of DL methods:
5-fold cross-validation, multiple seeds.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).parent))


def extract_per_point_features(points: np.ndarray, k: int = 20) -> np.ndarray:
    """Extract hand-crafted features for each point.

    Features per point (11 dims):
      - xyz coordinates (3)
      - height (z normalized) (1)
      - local normal estimate (3)
      - mean curvature proxy (1)
      - local point density (1)
      - mean NN distance (1)
      - std NN distance (1)
    """
    N = len(points)
    tree = cKDTree(points)
    kk = min(k, N - 1)
    dists, indices = tree.query(points, k=kk + 1)
    nn_dists = dists[:, 1:]  # exclude self
    nn_idx = indices[:, 1:]

    # Normalize xyz to [0,1]
    pmin, pmax = points.min(0), points.max(0)
    prange = pmax - pmin + 1e-8
    xyz_norm = (points - pmin) / prange

    # Height feature (z-coordinate normalized)
    height = xyz_norm[:, 2:3]

    # Local normals via PCA of k-NN
    normals = np.zeros((N, 3), dtype=np.float32)
    curvature = np.zeros((N, 1), dtype=np.float32)
    for i in range(N):
        neighbors = points[nn_idx[i]]
        cov = np.cov(neighbors.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normals[i] = eigvecs[:, 0]  # smallest eigenvalue = normal direction
        curvature[i] = eigvals[0] / (eigvals.sum() + 1e-8)

    # Local density (inverse mean NN distance)
    mean_nn = nn_dists.mean(axis=1, keepdims=True)
    std_nn = nn_dists.std(axis=1, keepdims=True)
    density = 1.0 / (mean_nn + 1e-8)

    features = np.hstack([
        xyz_norm,        # 3
        height,          # 1
        normals,         # 3
        curvature,       # 1
        density,         # 1
        mean_nn,         # 1
        std_nn,          # 1
    ])  # total: 11

    return features.astype(np.float32)


def run_rf_seg(data_root: Path, kfold_path: Path, seed: int, fold: int,
               n_estimators: int = 100) -> dict:
    """Run RF segmentation for one seed and fold."""
    from _lib.io import read_json, read_jsonl

    label_map = read_json(data_root / "label_map.json")
    num_classes = len(label_map)
    index_rows = read_jsonl(data_root / "index.jsonl")

    kfold_obj = read_json(kfold_path)
    k = int(kfold_obj["k"])
    c2f = kfold_obj["case_to_fold"]

    test_fold = fold
    val_fold = (fold + 1) % k

    train_X, train_y = [], []
    test_X, test_y = [], []

    for row in index_rows:
        f = int(c2f.get(row["case_key"], -1))
        npz_path = data_root / row["sample_npz"]
        data = np.load(str(npz_path))
        points = data["points"].astype(np.float32)
        labels = data["labels"].astype(np.int64)

        feats = extract_per_point_features(points)

        if f == test_fold:
            test_X.append(feats)
            test_y.append(labels)
        elif f != val_fold:
            train_X.append(feats)
            train_y.append(labels)

    train_X = np.vstack(train_X)
    train_y = np.concatenate(train_y)
    test_X = np.vstack(test_X)
    test_y = np.concatenate(test_y)

    print(f"  [s{seed} f{fold}] train={len(train_X):,} test={len(test_X):,}", flush=True)

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=20,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=seed,
    )
    clf.fit(train_X, train_y)
    pred = clf.predict(test_X)

    # Compute per-case metrics
    acc = float(np.mean(pred == test_y))
    ious = []
    for c in range(num_classes):
        tp = int(np.sum((pred == c) & (test_y == c)))
        fp = int(np.sum((pred == c) & (test_y != c)))
        fn = int(np.sum((pred != c) & (test_y == c)))
        ious.append(tp / max(tp + fp + fn, 1))

    return {
        "accuracy": acc,
        "mean_iou": float(np.mean(ious)),
        "per_class_iou": ious,
    }


def main():
    ap = argparse.ArgumentParser(description="RF per-point segmentation baseline")
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--kfold", type=Path, default=Path("metadata/splits_raw_case_kfold.json"))
    ap.add_argument("--seeds", type=str, default="1337,2020,2021,42,7")
    ap.add_argument("--run-root", type=Path, default=None,
                    help="Save individual results.json per seed/fold")
    ap.add_argument("--output", type=Path, default=None,
                    help="Save aggregated results JSON")
    ap.add_argument("--n-estimators", type=int, default=100)
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    kfold_path = Path(args.kfold).resolve()
    seeds = [int(s) for s in args.seeds.split(",")]

    from _lib.io import read_json
    kfold_obj = read_json(kfold_path)
    n_folds = int(kfold_obj["k"])

    all_results = []
    for seed in seeds:
        fold_mious = []
        for fold in range(n_folds):
            print(f"Running seed={seed} fold={fold}...", flush=True)
            metrics = run_rf_seg(data_root, kfold_path, seed, fold, args.n_estimators)
            fold_mious.append(metrics["mean_iou"])

            # Save individual result
            if args.run_root:
                run_dir = args.run_root / f"rf_seg_s{seed}_fold{fold}"
                run_dir.mkdir(parents=True, exist_ok=True)
                result = {
                    "train_config": {
                        "model": "random_forest",
                        "n_estimators": args.n_estimators,
                        "seed": seed,
                        "fold": fold,
                    },
                    "test_metrics": metrics,
                }
                with open(run_dir / "results.json", "w") as f:
                    json.dump(result, f, indent=2)

            print(f"  mIoU={metrics['mean_iou']:.4f} IoU={metrics['per_class_iou']}")

        entry = {
            "seed": seed,
            "fold_mious": fold_mious,
            "mean": float(np.mean(fold_mious)),
        }
        all_results.append(entry)
        print(f"  seed={seed} mean_mIoU={entry['mean']:.4f}")

    # Summary
    all_mious = [m for e in all_results for m in e["fold_mious"]]
    print(f"\nOverall: mIoU={np.mean(all_mious):.4f}±{np.std(all_mious):.4f} (n={len(all_mious)})")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
