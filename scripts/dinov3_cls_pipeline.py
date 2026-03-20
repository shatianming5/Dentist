#!/usr/bin/env python3
"""DINOv3-based point cloud classification pipeline.

Renders point clouds as multi-view images, extracts DINOv3 global features,
trains MLP classifier.
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
# Multi-view rendering (same as seg pipeline)
# ---------------------------------------------------------------------------

def get_view_matrices(n_views=6):
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


def render_depth_image(points, R, img_size=512):
    pts = (R @ points.T).T
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    margin = 0.05
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    scale = max(max(x_max - x_min, 1e-6), max(y_max - y_min, 1e-6)) * (1 + 2 * margin)
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    px = ((x - cx) / scale + 0.5) * (img_size - 1)
    py = ((y - cy) / scale + 0.5) * (img_size - 1)
    px = np.clip(px, 0, img_size - 1).astype(np.int32)
    py = np.clip(py, 0, img_size - 1).astype(np.int32)
    z_norm = (z - z.min()) / max(z.max() - z.min(), 1e-6)
    depth_buf = np.full((img_size, img_size), np.inf, dtype=np.float32)
    pixel_to_point = np.full((img_size, img_size), -1, dtype=np.int32)
    order = np.argsort(-z)
    for idx in order:
        r, c = py[idx], px[idx]
        if z[idx] < depth_buf[r, c]:
            depth_buf[r, c] = z[idx]
            pixel_to_point[r, c] = idx
    img = np.zeros((img_size, img_size, 3), dtype=np.float32)
    mask = pixel_to_point >= 0
    if mask.any():
        valid_z = depth_buf[mask]
        z_vis = 1.0 - (valid_z - valid_z.min()) / max(valid_z.max() - valid_z.min(), 1e-6)
        img[mask, 0] = z_vis * 0.8 + 0.2
        img[mask, 1] = z_vis * 0.6 + 0.2
        img[mask, 2] = z_vis * 0.4 + 0.2
    return img


# ---------------------------------------------------------------------------
# DINOv3 global feature extraction
# ---------------------------------------------------------------------------

def load_dinov3(device="cuda"):
    import timm
    model = timm.create_model('vit_small_patch16_dinov3', pretrained=True, num_classes=0)
    model.eval()
    model.to(device)
    return model


def extract_global_features(model, imgs, device="cuda"):
    """Extract global features from multiple rendered images and aggregate.

    Args:
        imgs: list of (H, W, 3) float32 images
    Returns:
        global_feat: (D,) aggregated feature vector
    """
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    all_feats = []
    for img in imgs:
        x = (img - mean) / std
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.forward_features(x)  # (1, N+1+reg, D)
        # Use CLS token (index 0) as global feature
        cls_feat = feat[0, 0, :].cpu().numpy()
        # Also use mean of patch tokens
        patch_size = 16
        n_patches = img.shape[0] // patch_size
        n_patch_tokens = n_patches * n_patches
        patch_mean = feat[0, 1:1+n_patch_tokens, :].mean(dim=0).cpu().numpy()
        # Concatenate CLS + mean_patches
        all_feats.append(np.concatenate([cls_feat, patch_mean]))

    # Average over views
    return np.mean(all_feats, axis=0).astype(np.float32)


def precompute_cls_features(data_root, output_dir, device="cuda", n_views=6, img_size=512):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(data_root)

    with open(data_root / "index.jsonl") as f:
        rows = [json.loads(line) for line in f]

    print("Loading DINOv3 ViT-S/16...", flush=True)
    model = load_dinov3(device)
    views = get_view_matrices(n_views)

    print(f"Processing {len(rows)} samples with {n_views} views...", flush=True)

    for i, row in enumerate(rows):
        npz_path = data_root / row["sample_npz"]
        # Create safe filename
        safe_name = row["case_key"].replace("/", "__").replace(" ", "_") + "_dinov3.npz"
        out_path = output_dir / safe_name

        if out_path.exists():
            print(f"[{i+1}/{len(rows)}] SKIP {safe_name}", flush=True)
            continue

        data = np.load(str(npz_path))
        points = data["points"].astype(np.float32)

        t0 = time.time()
        imgs = [render_depth_image(points, R, img_size) for R in views]
        feat = extract_global_features(model, imgs, device)
        dt = time.time() - t0

        np.savez_compressed(str(out_path), features=feat, label=row["label"])
        print(f"[{i+1}/{len(rows)}] {safe_name} feat={feat.shape} dt={dt:.1f}s", flush=True)

    print(f"[DONE] Features saved to {output_dir}", flush=True)


# ---------------------------------------------------------------------------
# Classification Dataset + MLP Head
# ---------------------------------------------------------------------------

class DINOv3ClsDataset(Dataset):
    def __init__(self, feat_dir, rows, label_to_id):
        self.feat_dir = Path(feat_dir)
        self.rows = rows
        self.label_to_id = label_to_id

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        safe_name = row["case_key"].replace("/", "__").replace(" ", "_") + "_dinov3.npz"
        data = np.load(str(self.feat_dir / safe_name))
        feat = torch.from_numpy(data["features"].astype(np.float32))
        label = self.label_to_id[row["label"]]
        return {"features": feat, "label": label, "case_key": row["case_key"]}


class MLPClassifier(nn.Module):
    def __init__(self, in_dim=768, num_classes=4, hidden=256, dropout=0.3):
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
        return self.net(x)


def train_cls_head(feat_dir, data_root, kfold_path, fold, seed, run_root,
                   epochs=200, lr=1e-3, device="cuda"):
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from _lib.seed import set_seed
    from _lib.io import read_json, read_jsonl, write_json
    from _lib.time import utc_now_iso

    set_seed(seed)
    dev = torch.device(device)
    data_root = Path(data_root)
    feat_dir = Path(feat_dir)

    label_map = read_json(data_root / "label_map.json")
    label_to_id = {str(k): int(v) for k, v in label_map.items()}
    num_classes = len(label_to_id)
    index_rows = read_jsonl(data_root / "index.jsonl")

    # K-fold split
    kfold_obj = read_json(Path(kfold_path))
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

    # Feature dim: CLS(384) + mean_patches(384) = 768
    in_dim = 768
    train_ds = DINOv3ClsDataset(feat_dir, train_rows, label_to_id)
    val_ds = DINOv3ClsDataset(feat_dir, val_rows, label_to_id)
    test_ds = DINOv3ClsDataset(feat_dir, test_rows, label_to_id)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    model = MLPClassifier(in_dim=in_dim, num_classes=num_classes, hidden=256, dropout=0.3).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    exp_name = f"dinov3_cls_s{seed}_fold{fold}"
    run_dir = Path(run_root) / exp_name
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "generated_at": utc_now_iso(),
        "model": "dinov3_cls_mlp",
        "backbone": "vit_small_patch16_dinov3",
        "num_classes": num_classes,
        "seed": seed, "fold": fold, "epochs": epochs, "lr": lr,
    }
    write_json(run_dir / "train_config.json", cfg)

    best_val_acc = -1.0
    patience_counter = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for batch in train_loader:
            feat = batch["features"].to(dev)
            lbl = torch.tensor(batch["label"]).to(dev)
            logits = model(feat)
            loss = F.cross_entropy(logits, lbl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * feat.shape[0]
            correct += int((logits.argmax(1) == lbl).sum().item())
            total += feat.shape[0]
        scheduler.step()

        # Val
        model.eval()
        val_correct, val_total = 0, 0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                feat = batch["features"].to(dev)
                lbl = torch.tensor(batch["label"]).to(dev)
                logits = model(feat)
                val_preds.extend(logits.argmax(1).cpu().tolist())
                val_labels.extend(lbl.cpu().tolist())
                val_correct += int((logits.argmax(1) == lbl).sum().item())
                val_total += feat.shape[0]

        val_acc = val_correct / max(val_total, 1)
        # Macro F1
        from sklearn.metrics import f1_score
        val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)

        improved = val_f1 > best_val_acc + 1e-6
        if improved:
            best_val_acc = val_f1
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        dt = time.time() - t0
        mark = "*" if improved else ""
        if epoch % 10 == 0 or epoch <= 5 or improved:
            print(f"[{epoch:03d}/{epochs}] trn_loss={total_loss/max(total,1):.4f} "
                  f"val_acc={val_acc:.4f} val_f1={val_f1:.4f} best_f1={best_val_acc:.4f} "
                  f"dt={dt:.1f}s {mark}", flush=True)

        if patience_counter >= 30:
            print(f"[early stop] no improvement for 30 epochs")
            break

    # Test
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    test_preds, test_labels_list = [], []
    with torch.no_grad():
        for batch in test_loader:
            feat = batch["features"].to(dev)
            lbl = torch.tensor(batch["label"]).to(dev)
            logits = model(feat)
            test_preds.extend(logits.argmax(1).cpu().tolist())
            test_labels_list.extend(lbl.cpu().tolist())

    from sklearn.metrics import accuracy_score, f1_score, classification_report
    test_acc = accuracy_score(test_labels_list, test_preds)
    test_f1 = f1_score(test_labels_list, test_preds, average='macro', zero_division=0)

    id_to_label = {v: k for k, v in label_to_id.items()}
    target_names = [id_to_label[i] for i in range(num_classes)]
    report = classification_report(test_labels_list, test_preds, labels=list(range(num_classes)), target_names=target_names, zero_division=0)

    print(f"\n[TEST] acc={test_acc:.4f} macro_f1={test_f1:.4f}")
    print(report)

    results = {
        "train_config": cfg,
        "best_val_f1": best_val_acc,
        "test_metrics": {"accuracy": test_acc, "macro_f1": test_f1},
    }
    write_json(run_dir / "results.json", results)
    print(f"[done] {run_dir}/results.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    p1 = sub.add_parser("extract")
    p1.add_argument("--data-root", type=Path, default=Path("processed/raw_cls/v13_main4"))
    p1.add_argument("--output-dir", type=Path, default=Path("processed/raw_cls/v13_main4/dinov3_features"))
    p1.add_argument("--device", type=str, default="cuda")
    p1.add_argument("--n-views", type=int, default=6)
    p1.add_argument("--img-size", type=int, default=512)

    p2 = sub.add_parser("train")
    p2.add_argument("--feat-dir", type=Path, default=Path("processed/raw_cls/v13_main4/dinov3_features"))
    p2.add_argument("--data-root", type=Path, default=Path("processed/raw_cls/v13_main4"))
    p2.add_argument("--kfold", type=Path, default=Path("metadata/splits_raw_case_kfold.json"))
    p2.add_argument("--fold", type=int, required=True)
    p2.add_argument("--seed", type=int, default=1337)
    p2.add_argument("--run-root", type=Path, default=Path("runs/raw_cls_dinov3"))
    p2.add_argument("--epochs", type=int, default=200)
    p2.add_argument("--lr", type=float, default=1e-3)
    p2.add_argument("--device", type=str, default="cuda")

    args = ap.parse_args()
    if args.cmd == "extract":
        precompute_cls_features(args.data_root.resolve(), args.output_dir.resolve(),
                               args.device, args.n_views, args.img_size)
    elif args.cmd == "train":
        os.chdir(Path(__file__).parent)
        train_cls_head(args.feat_dir.resolve(), args.data_root.resolve(),
                      args.kfold.resolve(), args.fold, args.seed,
                      args.run_root.resolve(), args.epochs, args.lr, args.device)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
