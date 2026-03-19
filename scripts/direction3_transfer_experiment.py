#!/usr/bin/env python3
"""Direction 3: Transfer Learning — Teeth3DS pretrain → Restoration segmentation.

Strategy:
1. Train PointNet classifier on Teeth3DS (14K teeth, 32-class FDI)
2. Extract encoder weights (feat.*)
3. Initialize PointNetSeg with pretrained encoder
4. Fine-tune on restoration segmentation (79 cases, 5-fold CV)
5. Compare with training from scratch

This script handles step 3-5. Step 1-2 uses existing phase2_train_teeth3ds_fdi_cls.py.
"""
import sys
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from collections import Counter

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))
from _lib.seed import set_seed
from _lib.device import normalize_device
from _lib.io import read_json, read_jsonl, write_json
from _lib.time import utc_now_iso

ROOT = Path(__file__).resolve().parents[1]


# --- Models (same as phase3_train_raw_seg.py) ---

class PointNetSeg(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Sequential(nn.Conv1d(3, 64, 1, bias=False), nn.BatchNorm1d(64), nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1, bias=False), nn.BatchNorm1d(128), nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 256, 1, bias=False), nn.BatchNorm1d(256), nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.Conv1d(256, 512, 1, bias=False), nn.BatchNorm1d(512), nn.ReLU(True))
        self.seg_head = nn.Sequential(
            nn.Conv1d(128 + 512, 256, 1, bias=False), nn.BatchNorm1d(256), nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Conv1d(256, 128, 1, bias=False), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Conv1d(128, num_classes, 1),
        )

    def forward(self, points):
        x = points.transpose(1, 2).contiguous()
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        g = torch.max(x4, dim=2, keepdim=True).values.expand_as(x4)
        feat = torch.cat([x2, g], dim=1)
        return self.seg_head(feat)


# --- Dataset ---

class RawSegDataset(Dataset):
    def __init__(self, data_root, index_rows, n_points, augment=False):
        self.data_root = Path(data_root)
        self.rows = list(index_rows)
        self.n_points = n_points
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        npz_path = self.data_root / row["sample_npz"]
        data = np.load(str(npz_path))
        pts = data["points"].astype(np.float32)
        lbl = data["labels"].astype(np.int64)
        if len(pts) >= self.n_points:
            choice = np.random.choice(len(pts), self.n_points, replace=False)
        else:
            choice = np.random.choice(len(pts), self.n_points, replace=True)
        pts, lbl = pts[choice], lbl[choice]
        if self.augment:
            theta = np.random.uniform(0, 2 * np.pi)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
            pts = pts @ R.T
            pts *= np.random.uniform(0.8, 1.2)
            pts += np.random.normal(0, 0.01, pts.shape).astype(np.float32)
        return {"points": torch.from_numpy(pts), "labels": torch.from_numpy(lbl),
                "case_key": row["case_key"]}


def compute_metrics(pred, gt, num_classes=2):
    ious = []
    for c in range(num_classes):
        tp = ((pred == c) & (gt == c)).sum()
        fp = ((pred == c) & (gt != c)).sum()
        fn = ((pred != c) & (gt == c)).sum()
        iou = tp / (tp + fp + fn + 1e-10)
        ious.append(float(iou))
    return {"miou": float(np.mean(ious)), "ious": ious, "acc": float((pred == gt).mean())}


def load_pretrained_encoder(model, ckpt_path):
    """Load pretrained PointNet encoder (conv1-conv4) from classification checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state") or ckpt.get("model") or ckpt

    # Map classification feat.* to segmentation conv1-conv4
    # Classification PointNet: feat = Sequential(Conv1d, BN, ReLU, Conv1d, BN, ReLU, ...)
    # feat.0=Conv1d(3,64), feat.1=BN(64), feat.2=ReLU (no params)
    # feat.3=Conv1d(64,128), feat.4=BN(128), feat.5=ReLU
    # feat.6=Conv1d(128,256), feat.7=BN(256), feat.8=ReLU
    # feat.9=Conv1d(256,512), feat.10=BN(512), feat.11=ReLU
    # 
    # Segmentation PointNetSeg: conv1 = Sequential(Conv1d(3,64), BN, ReLU)
    # conv1.0=Conv1d, conv1.1=BN, conv1.2=ReLU
    # conv2.0=Conv1d(64,128), conv2.1=BN, conv2.2=ReLU ... etc.
    
    cls_to_seg = {}
    # feat.{0,1} -> conv1.{0,1}, feat.{3,4} -> conv2.{0,1}, etc.
    feat_conv_indices = [(0, 1), (3, 4), (6, 7), (9, 10)]  # (conv_idx, bn_idx) in feat.*
    for layer_i, (conv_idx, bn_idx) in enumerate(feat_conv_indices):
        seg_prefix = f"conv{layer_i+1}"
        # Conv1d weight
        for suffix in [".weight", ".bias"]:
            src = f"feat.{conv_idx}{suffix}"
            dst = f"{seg_prefix}.0{suffix}"
            if src in state:
                cls_to_seg[dst] = state[src]
        # BatchNorm
        for suffix in [".weight", ".bias", ".running_mean", ".running_var", ".num_batches_tracked"]:
            src = f"feat.{bn_idx}{suffix}"
            dst = f"{seg_prefix}.1{suffix}"
            if src in state:
                cls_to_seg[dst] = state[src]

    # Load matched weights
    model_state = model.state_dict()
    loaded = 0
    for k, v in cls_to_seg.items():
        if k in model_state and model_state[k].shape == v.shape:
            model_state[k] = v
            loaded += 1

    model.load_state_dict(model_state)
    print(f"[transfer] Loaded {loaded} tensors from pretrained encoder")
    return loaded


def train_one_fold(data_root, kfold_obj, test_fold, pretrained_ckpt, args, device):
    """Train one fold, return test metrics."""
    k = kfold_obj["k"]
    val_fold = (test_fold + 1) % k
    c2f = kfold_obj["case_to_fold"]
    
    index_rows = read_jsonl(data_root / "index.jsonl")
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
    
    # Class weights
    train_labels = []
    for r in train_rows:
        d = np.load(str(data_root / r["sample_npz"]))
        train_labels.extend(d["labels"].tolist())
    counts = Counter(train_labels)
    total = sum(counts.values())
    weights = torch.tensor([total / (len(counts) * counts.get(c, 1)) for c in range(2)], dtype=torch.float32).to(device)
    
    model = PointNetSeg(num_classes=2, dropout=args.dropout).to(device)
    
    if pretrained_ckpt:
        load_pretrained_encoder(model, pretrained_ckpt)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    train_ds = RawSegDataset(data_root, train_rows, args.n_points, augment=True)
    val_ds = RawSegDataset(data_root, val_rows, args.n_points, augment=False)
    test_ds = RawSegDataset(data_root, test_rows, args.n_points, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    best_val_miou = 0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            pts = batch["points"].to(device)
            lbl = batch["labels"].to(device)
            out = model(pts)
            loss = criterion(out, lbl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        # Validate
        model.eval()
        all_pred, all_gt = [], []
        with torch.no_grad():
            for batch in val_loader:
                pts = batch["points"].to(device)
                lbl = batch["labels"]
                out = model(pts)
                pred = out.argmax(dim=1).cpu().numpy()
                all_pred.append(pred.flatten())
                all_gt.append(lbl.numpy().flatten())
        
        all_pred = np.concatenate(all_pred)
        all_gt = np.concatenate(all_gt)
        val_m = compute_metrics(all_pred, all_gt)
        
        if val_m["miou"] > best_val_miou:
            best_val_miou = val_m["miou"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break
    
    # Test with best model
    model.load_state_dict(best_state)
    model.eval()
    all_pred, all_gt, case_metrics = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            pts = batch["points"].to(device)
            lbl = batch["labels"]
            out = model(pts)
            pred = out.argmax(dim=1).cpu().numpy()
            for i in range(len(pred)):
                m = compute_metrics(pred[i], lbl[i].numpy())
                case_metrics.append(m)
            all_pred.append(pred.flatten())
            all_gt.append(lbl.numpy().flatten())
    
    all_pred = np.concatenate(all_pred)
    all_gt = np.concatenate(all_gt)
    test_m = compute_metrics(all_pred, all_gt)
    
    print(f"  Fold {test_fold}: val_miou={best_val_miou:.4f}, test_miou={test_m['miou']:.4f}, "
          f"epoch={epoch+1}, n_train={len(train_rows)}, n_test={len(test_rows)}")
    
    return {
        "fold": test_fold,
        "best_val_miou": best_val_miou,
        "test_miou": test_m["miou"],
        "test_acc": test_m["acc"],
        "test_ious": test_m["ious"],
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "epochs_trained": epoch + 1,
        "per_case_miou": [cm["miou"] for cm in case_metrics],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=ROOT / "processed/raw_seg/v2_natural")
    ap.add_argument("--kfold", type=Path, default=ROOT / "metadata/splits_raw_case_kfold.json")
    ap.add_argument("--pretrained-ckpt", type=str, default="")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--n-points", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--tag", type=str, default="scratch")
    args = ap.parse_args()
    
    set_seed(args.seed)
    device = torch.device(normalize_device(args.device))
    data_root = args.data_root.resolve()
    kfold_obj = read_json(args.kfold.resolve())
    k = kfold_obj["k"]
    
    mode = "pretrained" if args.pretrained_ckpt else "scratch"
    print(f"[D3] Transfer experiment: mode={mode}, data={data_root.name}, device={device}")
    if args.pretrained_ckpt:
        print(f"[D3] Pretrained checkpoint: {args.pretrained_ckpt}")
    
    fold_results = []
    for fold in range(k):
        set_seed(args.seed)
        r = train_one_fold(data_root, kfold_obj, fold, 
                          args.pretrained_ckpt if args.pretrained_ckpt else None,
                          args, device)
        fold_results.append(r)
    
    # Aggregate
    test_mious = [r["test_miou"] for r in fold_results]
    mean_miou = np.mean(test_mious)
    std_miou = np.std(test_mious)
    
    print(f"\n[D3] {mode.upper()}: mean_mIoU = {mean_miou:.4f} ± {std_miou:.4f}")
    print(f"     Per-fold: {[f'{m:.4f}' for m in test_mious]}")
    
    # Save results
    out_dir = ROOT / "runs" / "direction3_transfer"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    result = {
        "mode": mode,
        "pretrained_ckpt": args.pretrained_ckpt or None,
        "data_root": str(data_root),
        "mean_miou": float(mean_miou),
        "std_miou": float(std_miou),
        "fold_results": fold_results,
        "config": {
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "n_points": args.n_points,
            "seed": args.seed,
        },
        "timestamp": utc_now_iso(),
    }
    
    out_path = out_dir / f"result_{args.tag}.json"
    write_json(out_path, result)
    print(f"[D3] Results saved to {out_path}")
    return mean_miou


if __name__ == "__main__":
    main()
