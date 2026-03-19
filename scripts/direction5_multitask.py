#!/usr/bin/env python3
"""Direction 5: Multi-task model — shared encoder, dual heads (seg + cls).

Joint training on segmentation + classification simultaneously.
The encoder sees more supervision signal → potentially better features.
"""
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).parent))
from _lib.seed import set_seed
from _lib.device import normalize_device
from _lib.io import read_json, read_jsonl, write_json
from _lib.time import utc_now_iso

ROOT = Path(__file__).resolve().parents[1]


class PointNetMultiTask(nn.Module):
    """PointNet with shared encoder → segmentation head + classification head."""
    def __init__(self, num_seg_classes=2, num_cls_classes=4, dropout=0.3):
        super().__init__()
        # Shared encoder
        self.conv1 = nn.Sequential(nn.Conv1d(3, 64, 1, bias=False), nn.BatchNorm1d(64), nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1, bias=False), nn.BatchNorm1d(128), nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 256, 1, bias=False), nn.BatchNorm1d(256), nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.Conv1d(256, 512, 1, bias=False), nn.BatchNorm1d(512), nn.ReLU(True))
        
        # Segmentation head: per-point(128) + global(512) = 640
        self.seg_head = nn.Sequential(
            nn.Conv1d(128 + 512, 256, 1, bias=False), nn.BatchNorm1d(256), nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Conv1d(256, 128, 1, bias=False), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Conv1d(128, num_seg_classes, 1),
        )
        
        # Classification head: global(512)
        self.cls_head = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(True), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(True), nn.Dropout(dropout),
            nn.Linear(128, num_cls_classes),
        )
    
    def forward(self, points):
        x = points.transpose(1, 2).contiguous()
        x1 = self.conv1(x)       # (B, 64, N)
        x2 = self.conv2(x1)      # (B, 128, N)
        x3 = self.conv3(x2)      # (B, 256, N)
        x4 = self.conv4(x3)      # (B, 512, N)
        
        # Global feature
        g = torch.max(x4, dim=2, keepdim=True).values  # (B, 512, 1)
        
        # Seg: concat per-point + global
        g_expand = g.expand_as(x4)    # (B, 512, N)
        seg_feat = torch.cat([x2, g_expand], dim=1)  # (B, 640, N)
        seg_out = self.seg_head(seg_feat)  # (B, 2, N)
        
        # Cls: global pooled
        cls_out = self.cls_head(g.squeeze(2))  # (B, 4)
        
        return seg_out, cls_out


class MultiTaskDataset(Dataset):
    def __init__(self, data_root, rows, n_points, label_map, augment=False):
        self.data_root = Path(data_root)
        self.rows = rows
        self.n_points = n_points
        self.label_map = label_map
        self.augment = augment
    
    def __len__(self):
        return len(self.rows)
    
    def __getitem__(self, idx):
        row = self.rows[idx]
        data = np.load(str(self.data_root / row["sample_npz"]))
        pts = data["points"].astype(np.float32)
        seg_labels = data["labels"].astype(np.int64)
        cls_label = self.label_map[row["label"]]
        
        if len(pts) >= self.n_points:
            choice = np.random.choice(len(pts), self.n_points, replace=False)
        else:
            choice = np.random.choice(len(pts), self.n_points, replace=True)
        pts, seg_labels = pts[choice], seg_labels[choice]
        
        if self.augment:
            theta = np.random.uniform(0, 2 * np.pi)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
            pts = pts @ R.T
            pts *= np.random.uniform(0.8, 1.2)
            pts += np.random.normal(0, 0.01, pts.shape).astype(np.float32)
        
        return (torch.from_numpy(pts), torch.from_numpy(seg_labels), 
                torch.tensor(cls_label, dtype=torch.long), row["case_key"])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=ROOT / "processed/raw_seg/v1")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--n-points", type=int, default=4096)
    ap.add_argument("--cls-weight", type=float, default=1.0, help="Weight for cls loss vs seg loss")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--tag", type=str, default="multitask")
    args = ap.parse_args()
    
    set_seed(args.seed)
    device = torch.device(normalize_device(args.device))
    data_root = args.data_root.resolve()
    
    label_map = {"充填": 0, "全冠": 1, "桩核冠": 2, "高嵌体": 3}
    label_names = ["充填", "全冠", "桩核冠", "高嵌体"]
    
    kfold = read_json(ROOT / "metadata/splits_raw_case_kfold.json")
    k = kfold["k"]
    c2f = kfold["case_to_fold"]
    index_rows = read_jsonl(data_root / "index.jsonl")
    
    print(f"[D5] Multi-task: data={data_root.name}, cls_weight={args.cls_weight}, device={device}")
    
    fold_seg_mious, fold_cls_accs, fold_cls_f1s = [], [], []
    
    for fold in range(k):
        set_seed(args.seed)
        val_fold = (fold + 1) % k
        
        train_rows, val_rows, test_rows = [], [], []
        for row in index_rows:
            f = int(c2f.get(row["case_key"], -1))
            if f == fold: test_rows.append(row)
            elif f == val_fold: val_rows.append(row)
            else: train_rows.append(row)
        
        # Seg class weights
        seg_counts = Counter()
        for r in train_rows:
            d = np.load(str(data_root / r["sample_npz"]))
            seg_counts.update(d["labels"].tolist())
        seg_total = sum(seg_counts.values())
        seg_weights = torch.tensor([seg_total / (2 * seg_counts.get(c, 1)) for c in range(2)]).float().to(device)
        
        # Cls class weights
        cls_counts = Counter(label_map[r["label"]] for r in train_rows)
        cls_total = sum(cls_counts.values())
        cls_weights = torch.tensor([cls_total / (4 * cls_counts.get(c, 1)) for c in range(4)]).float().to(device)
        
        model = PointNetMultiTask().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        seg_criterion = nn.CrossEntropyLoss(weight=seg_weights)
        cls_criterion = nn.CrossEntropyLoss(weight=cls_weights)
        
        train_ds = MultiTaskDataset(data_root, train_rows, args.n_points, label_map, augment=True)
        val_ds = MultiTaskDataset(data_root, val_rows, args.n_points, label_map, augment=False)
        test_ds = MultiTaskDataset(data_root, test_rows, args.n_points, label_map, augment=False)
        
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2)
        
        best_val_combined = 0
        patience_counter = 0
        best_state = None
        
        for epoch in range(args.epochs):
            model.train()
            for pts, seg_lbl, cls_lbl, _ in train_loader:
                pts = pts.to(device)
                seg_lbl = seg_lbl.to(device)
                cls_lbl = cls_lbl.to(device)
                seg_out, cls_out = model(pts)
                seg_loss = seg_criterion(seg_out, seg_lbl)
                cls_loss = cls_criterion(cls_out, cls_lbl)
                loss = seg_loss + args.cls_weight * cls_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            
            # Val
            model.eval()
            seg_preds, seg_gts, cls_preds, cls_gts = [], [], [], []
            with torch.no_grad():
                for pts, seg_lbl, cls_lbl, _ in val_loader:
                    seg_out, cls_out = model(pts.to(device))
                    seg_preds.append(seg_out.argmax(dim=1).cpu().numpy().flatten())
                    seg_gts.append(seg_lbl.numpy().flatten())
                    cls_preds.extend(cls_out.argmax(dim=1).cpu().tolist())
                    cls_gts.extend(cls_lbl.tolist())
            
            seg_preds = np.concatenate(seg_preds)
            seg_gts = np.concatenate(seg_gts)
            seg_iou = float(np.mean([
                ((seg_preds == c) & (seg_gts == c)).sum() / 
                max(((seg_preds == c) | (seg_gts == c)).sum(), 1)
                for c in range(2)
            ]))
            cls_acc = accuracy_score(cls_gts, cls_preds)
            combined = seg_iou + cls_acc
            
            if combined > best_val_combined:
                best_val_combined = combined
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    break
        
        # Test
        model.load_state_dict(best_state)
        model.eval()
        seg_preds, seg_gts, cls_preds, cls_gts = [], [], [], []
        with torch.no_grad():
            for pts, seg_lbl, cls_lbl, _ in test_loader:
                seg_out, cls_out = model(pts.to(device))
                seg_preds.append(seg_out.argmax(dim=1).cpu().numpy().flatten())
                seg_gts.append(seg_lbl.numpy().flatten())
                cls_preds.extend(cls_out.argmax(dim=1).cpu().tolist())
                cls_gts.extend(cls_lbl.tolist())
        
        seg_preds = np.concatenate(seg_preds)
        seg_gts = np.concatenate(seg_gts)
        seg_miou = float(np.mean([
            ((seg_preds == c) & (seg_gts == c)).sum() / max(((seg_preds == c) | (seg_gts == c)).sum(), 1)
            for c in range(2)
        ]))
        cls_acc = accuracy_score(cls_gts, cls_preds)
        cls_f1 = f1_score(cls_gts, cls_preds, average='macro', zero_division=0)
        
        fold_seg_mious.append(seg_miou)
        fold_cls_accs.append(cls_acc)
        fold_cls_f1s.append(cls_f1)
        
        print(f"  Fold {fold}: seg_mIoU={seg_miou:.4f}, cls_acc={cls_acc:.3f}, cls_F1={cls_f1:.3f}, epoch={epoch+1}")
    
    print(f"\n[D5] MULTITASK ({data_root.name}):")
    print(f"  Seg mIoU: {np.mean(fold_seg_mious):.4f} ± {np.std(fold_seg_mious):.4f}")
    print(f"  Cls Acc:  {np.mean(fold_cls_accs):.3f} ± {np.std(fold_cls_accs):.3f}")
    print(f"  Cls F1:   {np.mean(fold_cls_f1s):.3f} ± {np.std(fold_cls_f1s):.3f}")
    
    result = {
        "mode": args.tag,
        "data_root": str(data_root),
        "cls_weight": args.cls_weight,
        "seg_miou": {"mean": float(np.mean(fold_seg_mious)), "std": float(np.std(fold_seg_mious)), "per_fold": fold_seg_mious},
        "cls_acc": {"mean": float(np.mean(fold_cls_accs)), "std": float(np.std(fold_cls_accs)), "per_fold": fold_cls_accs},
        "cls_f1": {"mean": float(np.mean(fold_cls_f1s)), "std": float(np.std(fold_cls_f1s)), "per_fold": fold_cls_f1s},
        "timestamp": utc_now_iso(),
    }
    
    out_dir = ROOT / "runs" / "direction5_multitask"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / f"result_{args.tag}.json", result)
    print(f"Saved to {out_dir / f'result_{args.tag}.json'}")


if __name__ == "__main__":
    main()
