#!/usr/bin/env python3
"""MV-ViT-ft (ft4): End-to-end fine-tuning of ViT-S/16 backbone + MLP seg head.

Architecture:
  1. Pre-render 6 orthographic views per point cloud (cached, CPU)
  2. Pre-compute point→patch mapping per view (cached)
  3. Forward: ViT backbone (last 4 blocks unfrozen) → gather per-point features → MLP head
  4. Differential LRs: backbone 5e-6, head 1e-3

Reproduces the ft4 runs in runs/dinov3_finetune/ft4/ and ft4_balanced/.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Add project root so _lib imports work
sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------------------
# Multi-view rendering (from dinov3_seg_pipeline.py)
# ---------------------------------------------------------------------------

def get_view_matrices(n_views: int = 6):
    views = []
    for i in range(4):
        angle = i * math.pi / 2
        c, s = math.cos(angle), math.sin(angle)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        tilt = math.radians(30)
        ct, st = math.cos(tilt), math.sin(tilt)
        Rx = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]], dtype=np.float32)
        views.append(Rx @ Rz)
    views.append(np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32))
    views.append(np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32))
    return views[:n_views]


def render_and_map(points: np.ndarray, R: np.ndarray, img_size: int = 512, patch_size: int = 16):
    """Render a view and return the image + point-to-patch mapping.

    Returns:
        img: (3, H, W) float32 normalized image tensor
        point_patch_idx: (N,) int32 — flattened patch index for each point (-1 if not visible)
    """
    N = len(points)
    pts = (R @ points.T).T
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    margin = 0.05
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    scale = max(max(x_max - x_min, 1e-6) * (1 + 2 * margin),
                max(y_max - y_min, 1e-6) * (1 + 2 * margin))
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2

    px = ((x - cx) / scale + 0.5) * (img_size - 1)
    py = ((y - cy) / scale + 0.5) * (img_size - 1)
    px = np.clip(px, 0, img_size - 1).astype(np.int32)
    py = np.clip(py, 0, img_size - 1).astype(np.int32)

    # Z-buffer
    depth_buf = np.full((img_size, img_size), np.inf, dtype=np.float32)
    pixel_to_point = np.full((img_size, img_size), -1, dtype=np.int32)
    order = np.argsort(-z)
    for idx in order:
        r, c = py[idx], px[idx]
        if z[idx] < depth_buf[r, c]:
            depth_buf[r, c] = z[idx]
            pixel_to_point[r, c] = idx

    # Create depth-colored image
    img = np.zeros((img_size, img_size, 3), dtype=np.float32)
    mask = pixel_to_point >= 0
    if mask.any():
        valid_z = depth_buf[mask]
        z_vis = 1.0 - (valid_z - valid_z.min()) / max(valid_z.max() - valid_z.min(), 1e-6)
        img[mask, 0] = z_vis * 0.8 + 0.2
        img[mask, 1] = z_vis * 0.6 + 0.2
        img[mask, 2] = z_vis * 0.4 + 0.2

    # Normalize with ImageNet stats
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)  # (3, H, W)

    # Point-to-patch mapping
    n_patches = img_size // patch_size
    patch_r = np.minimum(py // patch_size, n_patches - 1)
    patch_c = np.minimum(px // patch_size, n_patches - 1)
    visible_mask = pixel_to_point[py, px] == np.arange(N)
    point_patch_idx = np.full(N, -1, dtype=np.int32)
    point_patch_idx[visible_mask] = patch_r[visible_mask] * n_patches + patch_c[visible_mask]

    return img.astype(np.float32), point_patch_idx


# ---------------------------------------------------------------------------
# Dataset: pre-renders views and caches mappings
# ---------------------------------------------------------------------------

class MVViTSegDataset(Dataset):
    def __init__(self, data_root: Path, rows: list, n_views: int = 6,
                 img_size: int = 512, cache_dir: Path | None = None):
        self.data_root = Path(data_root)
        self.rows = rows
        self.views = get_view_matrices(n_views)
        self.n_views = n_views
        self.img_size = img_size
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        npz_path = self.data_root / row["sample_npz"]
        stem = Path(row["sample_npz"]).stem

        # Try cache first
        if self.cache_dir:
            cache_path = self.cache_dir / f"{stem}_views.npz"
            if cache_path.exists():
                cached = np.load(str(cache_path))
                images = torch.from_numpy(cached["images"])
                mappings = torch.from_numpy(cached["mappings"])
                labels = torch.from_numpy(cached["labels"])
                return {"images": images, "mappings": mappings, "labels": labels}

        data = np.load(str(npz_path))
        points = data["points"].astype(np.float32)
        labels = data["labels"].astype(np.int64)

        images = []
        mappings = []
        for R in self.views:
            img, pmap = render_and_map(points, R, self.img_size)
            images.append(img)
            mappings.append(pmap)

        images = np.stack(images)    # (V, 3, H, W)
        mappings = np.stack(mappings) # (V, N)

        # Cache for next epoch
        if self.cache_dir:
            np.savez_compressed(str(cache_path), images=images, mappings=mappings, labels=labels)

        return {
            "images": torch.from_numpy(images),
            "mappings": torch.from_numpy(mappings.astype(np.int64)),
            "labels": torch.from_numpy(labels),
        }


# ---------------------------------------------------------------------------
# Model: ViT backbone + MLP head with differentiable feature gathering
# ---------------------------------------------------------------------------

class MVViTFT4(nn.Module):
    """Multi-view ViT with fine-tuned backbone + MLP segmentation head."""

    def __init__(self, num_classes: int = 2, unfreeze_blocks: int = 4,
                 hidden: int = 256, dropout: float = 0.3, img_size: int = 512):
        super().__init__()
        import timm

        # Load ViT-S/16 backbone with DINOv3 pretrained weights
        # Must match the frozen pipeline's backbone for consistency
        self.backbone = timm.create_model(
            'vit_small_patch16_dinov3', pretrained=True, num_classes=0,
            img_size=img_size)

        self.feat_dim = 384  # ViT-S feature dimension
        self.patch_size = 16
        self.n_patches = img_size // self.patch_size
        self.n_patch_tokens = self.n_patches * self.n_patches

        # Freeze all, then unfreeze last N blocks
        for p in self.backbone.parameters():
            p.requires_grad = False

        blocks = self.backbone.blocks
        n_blocks = len(blocks)
        for i in range(max(0, n_blocks - unfreeze_blocks), n_blocks):
            for p in blocks[i].parameters():
                p.requires_grad = True
        # Also unfreeze norm layer
        if hasattr(self.backbone, 'norm'):
            for p in self.backbone.norm.parameters():
                p.requires_grad = True

        # MLP segmentation head (matches frozen pipeline)
        self.head = nn.Sequential(
            nn.Linear(self.feat_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes),
        )

    def extract_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract patch features from batch of images.

        Args:
            images: (B*V, 3, H, W)
        Returns:
            patch_features: (B*V, n_patches*n_patches, D)
        """
        feat = self.backbone.forward_features(images)  # (B*V, 1+P+reg, D)
        patch_feat = feat[:, 1:1+self.n_patch_tokens, :]  # (B*V, P, D)
        return patch_feat

    def gather_point_features(self, patch_features: torch.Tensor,
                              mappings: torch.Tensor) -> torch.Tensor:
        """Gather per-point features from multi-view patch features.

        Args:
            patch_features: (B, V, P, D) — patch features per view
            mappings: (B, V, N) int64 — point-to-patch index (-1 = invisible)
        Returns:
            point_features: (B, N, D) — averaged per-point features
        """
        B, V, P, D = patch_features.shape
        N = mappings.shape[2]

        # Replace -1 with 0 for safe gathering (will be masked out)
        valid = mappings >= 0  # (B, V, N)
        safe_idx = mappings.clamp(min=0)  # (B, V, N)

        # Expand index for gathering: (B, V, N, D)
        idx_expanded = safe_idx.unsqueeze(-1).expand(B, V, N, D)

        # Gather features: for each point, get its patch feature from each view
        gathered = torch.gather(patch_features, 2, idx_expanded)  # (B, V, N, D)

        # Mask invisible points and average across views
        valid_f = valid.unsqueeze(-1).float()  # (B, V, N, 1)
        feat_sum = (gathered * valid_f).sum(dim=1)  # (B, N, D)
        count = valid_f.sum(dim=1).clamp(min=1)  # (B, N, 1)

        return feat_sum / count  # (B, N, D)

    def forward(self, images: torch.Tensor, mappings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, V, 3, H, W)
            mappings: (B, V, N) int64
        Returns:
            logits: (B, C, N)
        """
        B, V = images.shape[:2]
        N = mappings.shape[2]

        # Process all views through backbone
        imgs_flat = images.reshape(B * V, *images.shape[2:])  # (B*V, 3, H, W)
        patch_feat = self.extract_patch_features(imgs_flat)  # (B*V, P, D)
        patch_feat = patch_feat.reshape(B, V, -1, self.feat_dim)  # (B, V, P, D)

        # Gather per-point features
        point_feat = self.gather_point_features(patch_feat, mappings)  # (B, N, D)

        # MLP head: need (B*N, D) for BatchNorm
        out = self.head(point_feat.reshape(B * N, self.feat_dim))  # (B*N, C)
        return out.reshape(B, N, -1).permute(0, 2, 1)  # (B, C, N)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pred, gt, num_classes=2):
    acc = float(np.mean(pred == gt))
    ious = []
    for c in range(num_classes):
        tp = int(np.sum((pred == c) & (gt == c)))
        fp = int(np.sum((pred == c) & (gt != c)))
        fn = int(np.sum((pred != c) & (gt == c)))
        ious.append(tp / max(tp + fp + fn, 1))
    return {"accuracy": acc, "mean_iou": float(np.mean(ious)), "per_class_iou": ious}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    from _lib.seed import set_seed
    from _lib.io import read_json, read_jsonl, write_json

    set_seed(args.seed)
    device = torch.device(args.device)

    data_root = Path(args.data_root).resolve()
    kfold_path = Path(args.kfold).resolve()
    run_root = Path(args.run_root).resolve()

    label_map = read_json(data_root / "label_map.json")
    num_classes = len(label_map)
    index_rows = read_jsonl(data_root / "index.jsonl")

    # K-fold split
    kfold_obj = read_json(kfold_path)
    k = int(kfold_obj["k"])
    test_fold = args.fold
    val_fold = (args.fold + 1) % k
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

    # Determine cache directory
    cache_dir = data_root / "mvvit_view_cache"

    train_ds = MVViTSegDataset(data_root, train_rows, cache_dir=cache_dir)
    val_ds = MVViTSegDataset(data_root, val_rows, cache_dir=cache_dir)
    test_ds = MVViTSegDataset(data_root, test_rows, cache_dir=cache_dir)

    # batch_size=1 due to large image sizes (6 views × 512×512)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    # Model
    model = MVViTFT4(
        num_classes=num_classes,
        unfreeze_blocks=args.unfreeze_blocks,
        hidden=256, dropout=0.3,
        img_size=args.img_size,
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] total={total_params:,} trainable={trainable_params:,} "
          f"({100*trainable_params/total_params:.1f}%)")

    # Optimizer with differential learning rates
    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and 'head' not in n]
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and 'head' in n]

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": head_params, "lr": args.lr_head},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7)

    # Loss: standard cross-entropy (no class weights, matching original pipeline)
    criterion_weight = None

    # Experiment directory
    exp_name = args.exp_name or f"dinov3_ft{args.unfreeze_blocks}_s{args.seed}_fold{args.fold}"
    run_dir = run_root / exp_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if (run_dir / "results.json").exists():
        print(f"[skip] {run_dir}/results.json already exists")
        return

    best_val_iou = -1.0
    patience_counter = 0
    best_state = None
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, total_correct, total_pts = 0.0, 0, 0

        for batch in train_loader:
            images = batch["images"].to(device)   # (B, V, 3, H, W)
            mappings = batch["mappings"].to(device) # (B, V, N)
            labels = batch["labels"].to(device)     # (B, N)

            logits = model(images, mappings)  # (B, C, N)
            loss = F.cross_entropy(logits, labels, weight=criterion_weight)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * images.shape[0]
            total_correct += int((logits.argmax(1) == labels).sum().item())
            total_pts += int(labels.numel())

        scheduler.step()

        # Validate
        model.eval()
        val_preds, val_labels_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["images"].to(device)
                mappings = batch["mappings"].to(device)
                labels = batch["labels"].to(device)
                logits = model(images, mappings)
                val_preds.append(logits.argmax(1).cpu().numpy())
                val_labels_list.append(labels.cpu().numpy())

        val_preds = np.concatenate(val_preds).ravel()
        val_labels_np = np.concatenate(val_labels_list).ravel()
        val_m = compute_metrics(val_preds, val_labels_np, num_classes)
        val_iou = val_m["mean_iou"]

        improved = val_iou > best_val_iou + 1e-6
        if improved:
            best_val_iou = val_iou
            patience_counter = 0
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        dt = time.time() - t0
        mark = "*" if improved else ""
        if epoch % 5 == 0 or epoch <= 3 or improved:
            print(f"[{epoch:03d}/{args.epochs}] loss={total_loss/max(len(train_ds),1):.4f} "
                  f"val_mIoU={val_iou:.4f} best={best_val_iou:.4f} "
                  f"lr_bb={optimizer.param_groups[0]['lr']:.2e} "
                  f"lr_hd={optimizer.param_groups[1]['lr']:.2e} "
                  f"dt={dt:.1f}s {mark}", flush=True)

        if patience_counter >= args.patience:
            print(f"[early stop] no improvement for {args.patience} epochs at epoch {epoch}")
            break

    # Test
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    test_preds, test_labels_list = [], []
    with torch.no_grad():
        for batch in test_loader:
            images = batch["images"].to(device)
            mappings = batch["mappings"].to(device)
            labels = batch["labels"].to(device)
            logits = model(images, mappings)
            test_preds.append(logits.argmax(1).cpu().numpy())
            test_labels_list.append(labels.cpu().numpy())

    test_preds = np.concatenate(test_preds).ravel()
    test_labels_np = np.concatenate(test_labels_list).ravel()
    test_m = compute_metrics(test_preds, test_labels_np, num_classes)

    print(f"\n[TEST] acc={test_m['accuracy']:.4f} mIoU={test_m['mean_iou']:.4f} "
          f"IoU_bg={test_m['per_class_iou'][0]:.4f} IoU_res={test_m['per_class_iou'][1]:.4f}")

    # Save results (matching existing format)
    config = {
        "model": f"dinov3_ft{args.unfreeze_blocks}",
        "unfreeze_blocks": args.unfreeze_blocks,
        "seed": args.seed,
        "fold": args.fold,
        "lr_backbone": args.lr_backbone,
        "lr_head": args.lr_head,
    }
    if args.protocol:
        config["protocol"] = args.protocol
    config["epoch_stopped"] = best_epoch

    results = {
        "train_config": config,
        "best_val_miou": best_val_iou,
        "test_metrics": test_m,
    }
    write_json(run_dir / "results.json", results)
    print(f"[done] {run_dir}/results.json")


def main():
    ap = argparse.ArgumentParser(description="MV-ViT-ft4 segmentation training")
    ap.add_argument("--data-root", type=Path, required=True,
                    help="processed/raw_seg/v1 (balanced) or v2_natural")
    ap.add_argument("--kfold", type=Path, default=Path("metadata/splits_raw_case_kfold.json"))
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--run-root", type=Path, required=True,
                    help="e.g. runs/dinov3_finetune/ft4")
    ap.add_argument("--exp-name", type=str, default=None,
                    help="Override experiment dir name")
    ap.add_argument("--protocol", type=str, default=None,
                    help="Tag: 'balanced' or 'natural'")
    ap.add_argument("--unfreeze-blocks", type=int, default=4)
    ap.add_argument("--lr-backbone", type=float, default=5e-6)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
