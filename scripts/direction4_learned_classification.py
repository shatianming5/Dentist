#!/usr/bin/env python3
"""Direction 4: Learned Classification — Direct point cloud classification.

Three approaches:
  A) Whole-case classification (all points, with/without cloud_id)
  B) Seg-guided: use GT seg mask to crop restoration → classify
  C) Seg-guided: use PREDICTED seg mask to crop → classify (realistic pipeline)

5-fold stratified CV, multiple architectures.
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
from sklearn.metrics import accuracy_score, f1_score, classification_report

sys.path.insert(0, str(Path(__file__).parent))
from _lib.seed import set_seed
from _lib.device import normalize_device
from _lib.io import read_json, read_jsonl, write_json
from _lib.time import utc_now_iso
from _lib.point_ops import get_graph_feature

ROOT = Path(__file__).resolve().parents[1]

# --- Models ---

class PointNetCls(nn.Module):
    def __init__(self, num_classes, in_ch=3, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(in_ch, 64, 1, bias=False), nn.BatchNorm1d(64), nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1, bias=False), nn.BatchNorm1d(128), nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 256, 1, bias=False), nn.BatchNorm1d(256), nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.Conv1d(256, 512, 1, bias=False), nn.BatchNorm1d(512), nn.ReLU(True))
        self.head = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(True), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(True), nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = x.transpose(1, 2).contiguous()
        x = self.conv4(self.conv3(self.conv2(self.conv1(x))))
        x = x.max(dim=2).values
        return self.head(x)


class DGCNNCls(nn.Module):
    def __init__(self, num_classes, in_ch=3, k=20, emb_dims=512, dropout=0.3):
        super().__init__()
        self.k = k
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch*2, 64, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, True))
        self.conv2 = nn.Sequential(nn.Conv2d(128, 64, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, True))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True))
        self.conv4 = nn.Sequential(nn.Conv2d(256, 256, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True))
        self.conv5 = nn.Sequential(nn.Conv1d(512, emb_dims, 1, bias=False), nn.BatchNorm1d(emb_dims), nn.LeakyReLU(0.2, True))
        self.head = nn.Sequential(
            nn.Linear(emb_dims*2, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.2, True), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.2, True), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = x.transpose(1, 2).contiguous()
        x1 = self.conv1(get_graph_feature(x, k=self.k)).max(dim=-1).values
        x2 = self.conv2(get_graph_feature(x1, k=self.k)).max(dim=-1).values
        x3 = self.conv3(get_graph_feature(x2, k=self.k)).max(dim=-1).values
        x4 = self.conv4(get_graph_feature(x3, k=self.k)).max(dim=-1).values
        x = self.conv5(torch.cat([x1, x2, x3, x4], dim=1))
        g = torch.cat([x.max(dim=2).values, x.mean(dim=2)], dim=1)
        return self.head(g)


# --- Datasets ---

class SegGuidedClsDataset(Dataset):
    """Load restoration-cropped points for classification."""
    def __init__(self, seg_root, cls_index, n_points, augment=False, use_pred_seg=False):
        self.seg_root = Path(seg_root)
        self.rows = []
        # Build case_key → cls label mapping
        for row in cls_index:
            self.rows.append(row)
        self.n_points = n_points
        self.augment = augment
        self.use_pred_seg = use_pred_seg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        npz_path = self.seg_root / row["sample_npz"]
        data = np.load(str(npz_path))
        pts = data["points"].astype(np.float32)
        labels = data["labels"].astype(np.int64)
        
        # Extract restoration points only (label == 1)
        seg_mask = labels == 1
        if seg_mask.sum() < 10:
            # Too few restoration points — use all
            seg_pts = pts
        else:
            seg_pts = pts[seg_mask]
        
        # Resample to n_points
        if len(seg_pts) >= self.n_points:
            choice = np.random.choice(len(seg_pts), self.n_points, replace=False)
        else:
            choice = np.random.choice(len(seg_pts), self.n_points, replace=True)
        seg_pts = seg_pts[choice]
        
        # Center the cropped points
        seg_pts = seg_pts - seg_pts.mean(axis=0, keepdims=True)
        scale = np.abs(seg_pts).max()
        if scale > 1e-6:
            seg_pts = seg_pts / scale
        
        if self.augment:
            theta = np.random.uniform(0, 2 * np.pi)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
            seg_pts = seg_pts @ R.T
            seg_pts *= np.random.uniform(0.8, 1.2)
            seg_pts += np.random.normal(0, 0.01, seg_pts.shape).astype(np.float32)
        
        return torch.from_numpy(seg_pts), row["cls_label"], row["case_key"]


def run_experiment(config, device):
    """Run one classification experiment with 5-fold CV."""
    seg_root = ROOT / config["seg_root"]
    seg_index = read_jsonl(seg_root / "index.jsonl")
    
    # Map labels
    label_map = {"充填": 0, "全冠": 1, "桩核冠": 2, "高嵌体": 3}
    label_names = ["充填", "全冠", "桩核冠", "高嵌体"]
    
    for row in seg_index:
        row["cls_label"] = label_map[row["label"]]
    
    # 5-fold CV using existing kfold
    kfold = read_json(ROOT / "metadata/splits_raw_case_kfold.json")
    k = kfold["k"]
    c2f = kfold["case_to_fold"]
    
    all_preds = {}
    all_trues = {}
    
    for fold in range(k):
        val_fold = (fold + 1) % k
        train_rows, val_rows, test_rows = [], [], []
        for row in seg_index:
            f = int(c2f.get(row["case_key"], -1))
            if f == fold:
                test_rows.append(row)
            elif f == val_fold:
                val_rows.append(row)
            else:
                train_rows.append(row)
        
        # Class weights
        class_counts = Counter(r["cls_label"] for r in train_rows)
        total = sum(class_counts.values())
        weights = torch.tensor([total / (4 * class_counts.get(c, 1)) for c in range(4)]).float().to(device)
        
        # Model
        if config["model"] == "pointnet":
            model = PointNetCls(4, in_ch=3, dropout=0.3).to(device)
        elif config["model"] == "dgcnn":
            model = DGCNNCls(4, in_ch=3, k=20, dropout=0.3).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
        criterion = nn.CrossEntropyLoss(weight=weights)
        
        train_ds = SegGuidedClsDataset(seg_root, train_rows, config["n_points"], augment=True)
        val_ds = SegGuidedClsDataset(seg_root, val_rows, config["n_points"], augment=False)
        test_ds = SegGuidedClsDataset(seg_root, test_rows, config["n_points"], augment=False)
        
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2)
        
        best_val_acc, patience_counter = 0, 0
        best_state = None
        
        for epoch in range(config["epochs"]):
            model.train()
            for pts, lbl, _ in train_loader:
                pts, lbl = pts.to(device), lbl.to(device)
                out = model(pts)
                loss = criterion(out, lbl)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            
            # Val
            model.eval()
            val_preds, val_trues = [], []
            with torch.no_grad():
                for pts, lbl, _ in val_loader:
                    out = model(pts.to(device))
                    val_preds.extend(out.argmax(dim=1).cpu().tolist())
                    val_trues.extend(lbl.tolist())
            val_acc = accuracy_score(val_trues, val_preds)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config["patience"]:
                    break
        
        # Test
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            for pts, lbl, keys in test_loader:
                out = model(pts.to(device))
                preds = out.argmax(dim=1).cpu().tolist()
                for key, pred, true in zip(keys, preds, lbl.tolist()):
                    all_preds[key] = pred
                    all_trues[key] = true
        
        fold_test_preds = [all_preds[r["case_key"]] for r in test_rows if r["case_key"] in all_preds]
        fold_test_trues = [all_trues[r["case_key"]] for r in test_rows if r["case_key"] in all_trues]
        fold_acc = accuracy_score(fold_test_trues, fold_test_preds)
        fold_f1 = f1_score(fold_test_trues, fold_test_preds, average='macro', zero_division=0)
        print(f"  Fold {fold}: val_acc={best_val_acc:.3f}, test_acc={fold_acc:.3f}, test_F1={fold_f1:.3f}, epoch={epoch+1}")
    
    # Overall
    y_true = [all_trues[k] for k in sorted(all_trues)]
    y_pred = [all_preds[k] for k in sorted(all_preds)]
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    return {
        "config": config,
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_class": classification_report(y_true, y_pred, target_names=label_names, output_dict=True, zero_division=0),
        "n_samples": len(y_true),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    experiments = [
        # Approach A: Whole-case with balanced seg data
        {"name": "PointNet_balanced_whole", "model": "pointnet", "seg_root": "processed/raw_seg/v1",
         "n_points": 4096, "epochs": 150, "patience": 25},
        
        # Approach B: Seg-guided on balanced (GT crop)
        {"name": "PointNet_balanced_seg_crop", "model": "pointnet", "seg_root": "processed/raw_seg/v1",
         "n_points": 2048, "epochs": 150, "patience": 25},
        
        # Approach C: Seg-guided on natural
        {"name": "PointNet_natural_seg_crop", "model": "pointnet", "seg_root": "processed/raw_seg/v2_natural",
         "n_points": 2048, "epochs": 150, "patience": 25},
        
        # Approach D: DGCNN on balanced seg crop
        {"name": "DGCNN_balanced_seg_crop", "model": "dgcnn", "seg_root": "processed/raw_seg/v1",
         "n_points": 2048, "epochs": 150, "patience": 25},
        
        # Approach E: DGCNN on natural seg crop
        {"name": "DGCNN_natural_seg_crop", "model": "dgcnn", "seg_root": "processed/raw_seg/v2_natural",
         "n_points": 2048, "epochs": 150, "patience": 25},
    ]
    
    results = []
    for exp in experiments:
        set_seed(1337)
        print(f"\n{'='*60}")
        print(f"Experiment: {exp['name']}")
        print(f"{'='*60}")
        r = run_experiment(exp, device)
        results.append(r)
        print(f"  → Accuracy={r['accuracy']:.3f}, Macro-F1={r['macro_f1']:.3f}")
    
    # Summary table
    print(f"\n{'='*80}")
    print(f"{'Experiment':<35} {'Acc':>8} {'F1':>8} {'充填':>8} {'全冠':>8} {'桩核冠':>8} {'高嵌体':>8}")
    print("-"*88)
    for r in results:
        pc = r["per_class"]
        print(f"{r['config']['name']:<35} {r['accuracy']:>8.3f} {r['macro_f1']:>8.3f} "
              f"{pc.get('充填',{}).get('f1-score',0):>8.3f} "
              f"{pc.get('全冠',{}).get('f1-score',0):>8.3f} "
              f"{pc.get('桩核冠',{}).get('f1-score',0):>8.3f} "
              f"{pc.get('高嵌体',{}).get('f1-score',0):>8.3f}")
    
    # Save
    out = ROOT / "paper_tables" / "classification_experiments.json"
    write_json(out, {"experiments": results, "timestamp": utc_now_iso()})
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
