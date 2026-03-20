#!/usr/bin/env python3
"""DINOv3-based point cloud segmentation pipeline.

Steps:
1. Render each point cloud from multiple views as 2D images
2. Extract DINOv3 dense features from each view
3. Back-project 2D patch features to 3D points
4. Train an MLP segmentation head on per-point DINOv3 features
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Step 1: Multi-view rendering (no OpenGL needed, pure numpy projection)
# ---------------------------------------------------------------------------

def get_view_matrices(n_views: int = 6):
    """Return camera rotation matrices for n views around the object."""
    views = []
    # 4 side views (0, 90, 180, 270 degrees around Z)
    for i in range(4):
        angle = i * math.pi / 2
        c, s = math.cos(angle), math.sin(angle)
        # Rotate around Z then tilt 30 degrees
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        tilt = math.radians(30)
        ct, st = math.cos(tilt), math.sin(tilt)
        Rx = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]], dtype=np.float32)
        views.append(Rx @ Rz)
    # Top view
    views.append(np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32))
    # Bottom view
    views.append(np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32))
    return views[:n_views]


def render_depth_image(points: np.ndarray, R: np.ndarray, img_size: int = 512):
    """Render point cloud as a depth image from a given view.

    Returns:
        img: (H, W, 3) float32 image with depth-based coloring
        pixel_to_point: (H, W) int32, maps each pixel to original point index (-1 if empty)
    """
    # Transform points
    pts = (R @ points.T).T  # (N, 3)

    # Orthographic projection: use X, Y as image coords, Z as depth
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    # Normalize to [0, img_size-1]
    margin = 0.05
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_range = max(x_max - x_min, 1e-6) * (1 + 2 * margin)
    y_range = max(y_max - y_min, 1e-6) * (1 + 2 * margin)
    scale = max(x_range, y_range)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2

    px = ((x - cx) / scale + 0.5) * (img_size - 1)
    py = ((y - cy) / scale + 0.5) * (img_size - 1)

    px = np.clip(px, 0, img_size - 1).astype(np.int32)
    py = np.clip(py, 0, img_size - 1).astype(np.int32)

    # Z-buffer: keep closest point per pixel
    z_norm = (z - z.min()) / max(z.max() - z.min(), 1e-6)  # 0=close, 1=far

    depth_buf = np.full((img_size, img_size), np.inf, dtype=np.float32)
    pixel_to_point = np.full((img_size, img_size), -1, dtype=np.int32)

    # Sort by depth (far to near) so near points overwrite far
    order = np.argsort(-z)  # far first
    for idx in order:
        r, c = py[idx], px[idx]
        if z[idx] < depth_buf[r, c]:
            depth_buf[r, c] = z[idx]
            pixel_to_point[r, c] = idx

    # Create colored image (depth-based grayscale + normal-based coloring)
    img = np.zeros((img_size, img_size, 3), dtype=np.float32)
    mask = pixel_to_point >= 0
    if mask.any():
        valid_z = depth_buf[mask]
        z_vis = 1.0 - (valid_z - valid_z.min()) / max(valid_z.max() - valid_z.min(), 1e-6)
        img[mask, 0] = z_vis * 0.8 + 0.2
        img[mask, 1] = z_vis * 0.6 + 0.2
        img[mask, 2] = z_vis * 0.4 + 0.2

    return img, pixel_to_point, (px, py)


# ---------------------------------------------------------------------------
# Step 2: DINOv3 Feature Extraction
# ---------------------------------------------------------------------------

def load_dinov3(device: str = "cuda"):
    """Load DINOv3 ViT-S/16 backbone."""
    import timm
    model = timm.create_model('vit_small_patch16_dinov3', pretrained=True, num_classes=0)
    model.eval()
    model.to(device)
    return model


def extract_dense_features(model, img: np.ndarray, device: str = "cuda", img_size: int = 512):
    """Extract dense patch features from a rendered image.

    Args:
        img: (H, W, 3) float32 image
    Returns:
        features: (n_patches_h, n_patches_w, D) feature map
    """
    # Normalize with ImageNet stats
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    x = (img - mean) / std
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model.forward_features(x)  # (1, N+1+reg, D)

    # Remove CLS token (first) and register tokens
    patch_size = 16
    n_patches = img_size // patch_size  # 32
    n_patch_tokens = n_patches * n_patches  # 1024

    # feat has shape (1, 1+n_patch_tokens+n_register, D)
    # The patch tokens start after CLS token
    # DINOv3 has 4 register tokens appended after patch tokens
    patch_feat = feat[0, 1:1+n_patch_tokens, :]  # (1024, D)
    patch_feat = patch_feat.reshape(n_patches, n_patches, -1)  # (32, 32, D)

    return patch_feat.cpu().numpy()


# ---------------------------------------------------------------------------
# Step 3: Back-project features to 3D points
# ---------------------------------------------------------------------------

def backproject_features(points: np.ndarray, views: list, model, device: str = "cuda",
                         img_size: int = 512):
    """Extract DINOv3 features for each 3D point via multi-view rendering.

    Returns:
        point_features: (N, D) per-point feature vectors
    """
    patch_size = 16
    n_patches = img_size // patch_size
    N = len(points)
    D = 384  # DINOv3 ViT-S feature dim

    feat_sum = np.zeros((N, D), dtype=np.float32)
    feat_count = np.zeros(N, dtype=np.float32)

    for R in views:
        # Render
        img, pixel_to_point, (px, py) = render_depth_image(points, R, img_size)

        # Extract features
        patch_feat = extract_dense_features(model, img, device, img_size)  # (32, 32, D)

        # For each point, find its patch and get the feature
        for i in range(N):
            # Find which pixel this point maps to
            pi_x, pi_y = px[i], py[i]
            # Map pixel to patch
            patch_r = min(pi_y // patch_size, n_patches - 1)
            patch_c = min(pi_x // patch_size, n_patches - 1)

            # Check if this point is visible (closest in its pixel)
            if pixel_to_point[pi_y, pi_x] == i:
                feat_sum[i] += patch_feat[patch_r, patch_c]
                feat_count[i] += 1

    # For points not visible in any view, use nearest visible point's feature
    visible = feat_count > 0
    if visible.any() and not visible.all():
        from scipy.spatial import cKDTree
        tree = cKDTree(points[visible])
        _, nn_idx = tree.query(points[~visible])
        visible_indices = np.where(visible)[0]
        feat_sum[~visible] = feat_sum[visible_indices[nn_idx]]
        feat_count[~visible] = 1

    # Average features
    feat_count = np.maximum(feat_count, 1)
    point_features = feat_sum / feat_count[:, None]

    return point_features


def backproject_features_fast(points: np.ndarray, views: list, model, device: str = "cuda",
                              img_size: int = 512):
    """Vectorized version - much faster than per-point loop."""
    patch_size = 16
    n_patches = img_size // patch_size
    N = len(points)
    D = 384

    feat_sum = np.zeros((N, D), dtype=np.float32)
    feat_count = np.zeros(N, dtype=np.float32)

    for R in views:
        img, pixel_to_point, (px, py) = render_depth_image(points, R, img_size)
        patch_feat = extract_dense_features(model, img, device, img_size)  # (n_p, n_p, D)

        # Vectorized: map each point to its patch
        patch_r = np.minimum(py // patch_size, n_patches - 1)
        patch_c = np.minimum(px // patch_size, n_patches - 1)

        # Check visibility: point is visible if it's the closest at its pixel
        visible_mask = pixel_to_point[py, px] == np.arange(N)

        # Get features for visible points
        vis_idx = np.where(visible_mask)[0]
        if len(vis_idx) > 0:
            feat_sum[vis_idx] += patch_feat[patch_r[vis_idx], patch_c[vis_idx]]
            feat_count[vis_idx] += 1

    # Handle invisible points via nearest neighbor
    visible = feat_count > 0
    if visible.any() and not visible.all():
        from scipy.spatial import cKDTree
        tree = cKDTree(points[visible])
        _, nn_idx = tree.query(points[~visible])
        visible_indices = np.where(visible)[0]
        feat_sum[~visible] = feat_sum[visible_indices[nn_idx]]
        feat_count[~visible] = 1

    feat_count = np.maximum(feat_count, 1)
    return feat_sum / feat_count[:, None]


# ---------------------------------------------------------------------------
# Step 4: Feature cache - precompute all features
# ---------------------------------------------------------------------------

def precompute_features(data_root: Path, output_dir: Path, device: str = "cuda",
                        n_views: int = 6, img_size: int = 512):
    """Precompute DINOv3 features for all samples."""
    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = data_root / "index.jsonl"
    with open(index_path) as f:
        rows = [json.loads(line) for line in f]

    print(f"Loading DINOv3 ViT-S/16...", flush=True)
    model = load_dinov3(device)
    views = get_view_matrices(n_views)

    print(f"Processing {len(rows)} samples with {n_views} views...", flush=True)

    for i, row in enumerate(rows):
        npz_path = data_root / row["sample_npz"]
        out_path = output_dir / (Path(row["sample_npz"]).stem + "_dinov3.npz")

        if out_path.exists():
            print(f"[{i+1}/{len(rows)}] SKIP {out_path.name}", flush=True)
            continue

        data = np.load(str(npz_path))
        points = data["points"].astype(np.float32)
        labels = data["labels"]

        t0 = time.time()
        feats = backproject_features_fast(points, views, model, device, img_size)
        dt = time.time() - t0

        np.savez_compressed(str(out_path), features=feats, labels=labels)
        print(f"[{i+1}/{len(rows)}] {out_path.name} feat={feats.shape} dt={dt:.1f}s", flush=True)

    print(f"[DONE] Features saved to {output_dir}", flush=True)


# ---------------------------------------------------------------------------
# Step 5: MLP Segmentation Head + Training
# ---------------------------------------------------------------------------

class DINOv3SegDataset(Dataset):
    def __init__(self, feat_dir: Path, data_root: Path, rows: list):
        self.feat_dir = Path(feat_dir)
        self.data_root = Path(data_root)
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        feat_path = self.feat_dir / (Path(row["sample_npz"]).stem + "_dinov3.npz")
        data = np.load(str(feat_path))
        features = torch.from_numpy(data["features"].astype(np.float32))  # (N, 384)
        labels = torch.from_numpy(data["labels"].astype(np.int64))        # (N,)
        return {"features": features, "labels": labels, "case_key": row["case_key"]}


class MLPSegHead(nn.Module):
    """Simple MLP segmentation head on frozen DINOv3 features."""
    def __init__(self, in_dim: int = 384, num_classes: int = 2, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x):
        # x: (B, N, D) -> reshape to (B*N, D), apply MLP, reshape back
        B, N, D = x.shape
        out = self.net(x.reshape(B * N, D))  # (B*N, C)
        return out.reshape(B, N, -1).permute(0, 2, 1)  # (B, C, N)


def compute_metrics(pred, gt, num_classes=2):
    acc = float(np.mean(pred == gt))
    ious = []
    for c in range(num_classes):
        tp = int(np.sum((pred == c) & (gt == c)))
        fp = int(np.sum((pred == c) & (gt != c)))
        fn = int(np.sum((pred != c) & (gt == c)))
        ious.append(tp / max(tp + fp + fn, 1))
    return {"accuracy": acc, "mean_iou": float(np.mean(ious)), "per_class_iou": ious}


def train_mlp_head(feat_dir: Path, data_root: Path, kfold_path: Path,
                   fold: int, seed: int, run_root: Path,
                   epochs: int = 200, lr: float = 1e-3, device: str = "cuda"):
    """Train MLP segmentation head on DINOv3 features."""
    from _lib.seed import set_seed
    from _lib.io import read_json, read_jsonl, write_json
    from _lib.time import utc_now_iso

    set_seed(seed)
    dev = torch.device(device)

    label_map = read_json(data_root / "label_map.json")
    num_classes = len(label_map)
    index_rows = read_jsonl(data_root / "index.jsonl")

    # K-fold split
    kfold_obj = read_json(kfold_path)
    k = int(kfold_obj["k"])
    test_fold = fold
    val_fold = (fold + 1) % k
    c2f = kfold_obj["case_to_fold"]
    for row in index_rows:
        f = int(c2f.get(row["case_key"], -1))
        if f == test_fold:
            row["split"] = "test"
        elif f == val_fold:
            row["split"] = "val"
        else:
            row["split"] = "train"

    train_rows = [r for r in index_rows if r["split"] == "train"]
    val_rows = [r for r in index_rows if r["split"] == "val"]
    test_rows = [r for r in index_rows if r["split"] == "test"]
    print(f"[data] train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")

    train_ds = DINOv3SegDataset(feat_dir, data_root, train_rows)
    val_ds = DINOv3SegDataset(feat_dir, data_root, val_rows)
    test_ds = DINOv3SegDataset(feat_dir, data_root, test_rows)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

    model = MLPSegHead(in_dim=384, num_classes=num_classes, hidden=256, dropout=0.3).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    exp_name = f"dinov3_mlp_s{seed}_fold{fold}"
    run_dir = run_root / exp_name
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "generated_at": utc_now_iso(),
        "model": "dinov3_mlp",
        "backbone": "vit_small_patch16_dinov3",
        "num_classes": num_classes,
        "n_points": 8192,
        "seed": seed,
        "fold": fold,
        "epochs": epochs,
        "lr": lr,
    }
    write_json(run_dir / "train_config.json", cfg)

    best_val_iou = -1.0
    patience_counter = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, total_correct, total_pts = 0.0, 0, 0
        for batch in train_loader:
            feat = batch["features"].to(dev)
            lbl = batch["labels"].to(dev)
            logits = model(feat)  # (B, C, N)
            loss = F.cross_entropy(logits, lbl)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * feat.shape[0]
            total_correct += int((logits.argmax(1) == lbl).sum().item())
            total_pts += int(lbl.numel())
        scheduler.step()

        # Validate
        model.eval()
        val_preds, val_labels = [], []
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                feat = batch["features"].to(dev)
                lbl = batch["labels"].to(dev)
                logits = model(feat)
                val_loss += F.cross_entropy(logits, lbl).item() * feat.shape[0]
                val_preds.append(logits.argmax(1).cpu().numpy())
                val_labels.append(lbl.cpu().numpy())

        val_preds = np.concatenate(val_preds).ravel()
        val_labels_np = np.concatenate(val_labels).ravel()
        val_m = compute_metrics(val_preds, val_labels_np, num_classes)
        val_iou = val_m["mean_iou"]

        improved = val_iou > best_val_iou + 1e-6
        if improved:
            best_val_iou = val_iou
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        dt = time.time() - t0
        mark = "*" if improved else ""
        if epoch % 10 == 0 or epoch <= 5 or improved:
            print(f"[{epoch:03d}/{epochs}] trn_loss={total_loss/max(len(train_ds),1):.4f} "
                  f"val_mIoU={val_iou:.4f} best={best_val_iou:.4f} "
                  f"lr={optimizer.param_groups[0]['lr']:.6f} dt={dt:.1f}s {mark}", flush=True)

        if patience_counter >= 30:
            print(f"[early stop] no improvement for 30 epochs")
            break

    # Test
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            feat = batch["features"].to(dev)
            lbl = batch["labels"].to(dev)
            logits = model(feat)
            test_preds.append(logits.argmax(1).cpu().numpy())
            test_labels.append(lbl.cpu().numpy())

    test_preds = np.concatenate(test_preds).ravel()
    test_labels_np = np.concatenate(test_labels).ravel()
    test_m = compute_metrics(test_preds, test_labels_np, num_classes)

    print(f"\n[TEST] acc={test_m['accuracy']:.4f} mIoU={test_m['mean_iou']:.4f} "
          f"IoU_bg={test_m['per_class_iou'][0]:.4f} IoU_res={test_m['per_class_iou'][1]:.4f}")

    results = {
        "train_config": cfg,
        "best_val_miou": best_val_iou,
        "test_metrics": test_m,
    }
    write_json(run_dir / "results.json", results)
    print(f"[done] {run_dir}/results.json")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    # Precompute features
    p1 = sub.add_parser("extract", help="Extract DINOv3 features for all samples")
    p1.add_argument("--data-root", type=Path, default=Path("processed/raw_seg/v1"))
    p1.add_argument("--output-dir", type=Path, default=Path("processed/raw_seg/v1/dinov3_features"))
    p1.add_argument("--device", type=str, default="cuda")
    p1.add_argument("--n-views", type=int, default=6)
    p1.add_argument("--img-size", type=int, default=512)

    # Train MLP head
    p2 = sub.add_parser("train", help="Train MLP segmentation head")
    p2.add_argument("--feat-dir", type=Path, default=Path("processed/raw_seg/v1/dinov3_features"))
    p2.add_argument("--data-root", type=Path, default=Path("processed/raw_seg/v1"))
    p2.add_argument("--kfold", type=Path, default=Path("metadata/splits_raw_case_kfold.json"))
    p2.add_argument("--fold", type=int, required=True)
    p2.add_argument("--seed", type=int, default=1337)
    p2.add_argument("--run-root", type=Path, default=Path("runs/raw_seg_dinov3"))
    p2.add_argument("--epochs", type=int, default=200)
    p2.add_argument("--lr", type=float, default=1e-3)
    p2.add_argument("--device", type=str, default="cuda")

    args = ap.parse_args()

    if args.cmd == "extract":
        precompute_features(args.data_root.resolve(), args.output_dir.resolve(),
                           args.device, args.n_views, args.img_size)
    elif args.cmd == "train":
        os.chdir(Path(__file__).parent)
        train_mlp_head(args.feat_dir.resolve(), args.data_root.resolve(),
                      args.kfold.resolve(), args.fold, args.seed,
                      args.run_root.resolve(), args.epochs, args.lr, args.device)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
