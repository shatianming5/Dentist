#!/usr/bin/env python3
"""Direction 7: Domain Adaptation — balanced pretrain → natural fine-tune.

Compare:
  A) Train from scratch on natural
  B) Pretrain on balanced → fine-tune on natural  
  C) Pretrain on balanced → freeze encoder → train head on natural
"""
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
from _lib.seed import set_seed
from _lib.device import normalize_device
from _lib.io import read_json, read_jsonl, write_json
from _lib.time import utc_now_iso

ROOT = Path(__file__).resolve().parents[1]


class PointNetSeg(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(3, 64, 1, bias=False), nn.BatchNorm1d(64), nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1, bias=False), nn.BatchNorm1d(128), nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 256, 1, bias=False), nn.BatchNorm1d(256), nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.Conv1d(256, 512, 1, bias=False), nn.BatchNorm1d(512), nn.ReLU(True))
        self.seg_head = nn.Sequential(
            nn.Conv1d(640, 256, 1, bias=False), nn.BatchNorm1d(256), nn.ReLU(True), nn.Dropout(dropout),
            nn.Conv1d(256, 128, 1, bias=False), nn.BatchNorm1d(128), nn.ReLU(True), nn.Dropout(dropout),
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

    def freeze_encoder(self):
        for m in [self.conv1, self.conv2, self.conv3, self.conv4]:
            for p in m.parameters():
                p.requires_grad = False
    
    def unfreeze_encoder(self):
        for m in [self.conv1, self.conv2, self.conv3, self.conv4]:
            for p in m.parameters():
                p.requires_grad = True


class RawSegDataset(Dataset):
    def __init__(self, data_root, rows, n_points, augment=False):
        self.data_root = Path(data_root)
        self.rows = rows
        self.n_points = n_points
        self.augment = augment
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        row = self.rows[idx]
        data = np.load(str(self.data_root / row["sample_npz"]))
        pts = data["points"].astype(np.float32)
        lbl = data["labels"].astype(np.int64)
        choice = np.random.choice(len(pts), self.n_points, replace=len(pts) < self.n_points)
        pts, lbl = pts[choice], lbl[choice]
        if self.augment:
            theta = np.random.uniform(0, 2*np.pi)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float32)
            pts = pts @ R.T * np.random.uniform(0.8,1.2)
            pts += np.random.normal(0, 0.01, pts.shape).astype(np.float32)
        return {"points": torch.from_numpy(pts), "labels": torch.from_numpy(lbl)}


def compute_miou(pred, gt, nc=2):
    ious = []
    for c in range(nc):
        tp = ((pred==c)&(gt==c)).sum()
        un = ((pred==c)|(gt==c)).sum()
        ious.append(float(tp)/max(float(un),1))
    return float(np.mean(ious))


def train_seg(model, data_root, kfold, fold, device, epochs=100, patience=20, n_points=8192, lr=1e-3):
    k = kfold["k"]
    val_fold = (fold+1) % k
    c2f = kfold["case_to_fold"]
    rows = read_jsonl(data_root / "index.jsonl")
    for r in rows:
        f = int(c2f.get(r["case_key"], -1))
        r["split"] = "test" if f==fold else ("val" if f==val_fold else "train")
    
    train_rows = [r for r in rows if r["split"]=="train"]
    val_rows = [r for r in rows if r["split"]=="val"]
    test_rows = [r for r in rows if r["split"]=="test"]
    
    # Class weights
    seg_counts = Counter()
    for r in train_rows:
        seg_counts.update(np.load(str(data_root / r["sample_npz"]))["labels"].tolist())
    total = sum(seg_counts.values())
    weights = torch.tensor([total/(2*seg_counts.get(c,1)) for c in range(2)]).float().to(device)
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    train_loader = DataLoader(RawSegDataset(data_root, train_rows, n_points, augment=True), batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(RawSegDataset(data_root, val_rows, n_points), batch_size=16, num_workers=2)
    test_loader = DataLoader(RawSegDataset(data_root, test_rows, n_points), batch_size=16, num_workers=2)
    
    best_val, patience_ctr, best_state = 0, 0, None
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            out = model(batch["points"].to(device))
            loss = criterion(out, batch["labels"].to(device))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        scheduler.step()
        
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch["points"].to(device))
                preds.append(out.argmax(1).cpu().numpy().flatten())
                gts.append(batch["labels"].numpy().flatten())
        val_miou = compute_miou(np.concatenate(preds), np.concatenate(gts))
        if val_miou > best_val:
            best_val = val_miou
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience: break
    
    model.load_state_dict(best_state)
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch["points"].to(device))
            preds.append(out.argmax(1).cpu().numpy().flatten())
            gts.append(batch["labels"].numpy().flatten())
    test_miou = compute_miou(np.concatenate(preds), np.concatenate(gts))
    return best_val, test_miou, epoch+1, best_state


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bal_root = ROOT / "processed/raw_seg/v1"
    nat_root = ROOT / "processed/raw_seg/v2_natural"
    kfold = read_json(ROOT / "metadata/splits_raw_case_kfold.json")
    k = kfold["k"]
    
    results = {}
    
    # --- Condition A: Scratch on natural ---
    print("=== Condition A: Scratch on natural ===")
    a_mious = []
    for fold in range(k):
        set_seed(1337)
        model = PointNetSeg().to(device)
        val, test, ep, _ = train_seg(model, nat_root, kfold, fold, device)
        a_mious.append(test)
        print(f"  Fold {fold}: val={val:.4f}, test={test:.4f}, epoch={ep}")
    print(f"  Mean: {np.mean(a_mious):.4f} ± {np.std(a_mious):.4f}")
    results["scratch_natural"] = {"mean": float(np.mean(a_mious)), "std": float(np.std(a_mious)), "folds": a_mious}
    
    # --- Condition B: Balanced pretrain → natural fine-tune (full) ---
    print("\n=== Condition B: Balanced pretrain → natural fine-tune ===")
    b_mious = []
    for fold in range(k):
        set_seed(1337)
        model = PointNetSeg().to(device)
        # Phase 1: pretrain on balanced
        print(f"  Fold {fold} Phase 1 (balanced pretrain)...", end=" ", flush=True)
        _, _, ep1, state1 = train_seg(model, bal_root, kfold, fold, device, epochs=80, patience=15)
        print(f"epoch={ep1}", flush=True)
        # Phase 2: fine-tune on natural
        model.load_state_dict(state1)
        model = model.to(device)
        print(f"  Fold {fold} Phase 2 (natural fine-tune)...", end=" ", flush=True)
        val, test, ep2, _ = train_seg(model, nat_root, kfold, fold, device, epochs=80, patience=15, lr=5e-4)
        print(f"epoch={ep2}, test={test:.4f}", flush=True)
        b_mious.append(test)
    print(f"  Mean: {np.mean(b_mious):.4f} ± {np.std(b_mious):.4f}")
    results["balanced_pretrain_finetune"] = {"mean": float(np.mean(b_mious)), "std": float(np.std(b_mious)), "folds": b_mious}
    
    # --- Condition C: Balanced pretrain → freeze encoder → train head on natural ---
    print("\n=== Condition C: Balanced pretrain → freeze encoder → natural ===")
    c_mious = []
    for fold in range(k):
        set_seed(1337)
        model = PointNetSeg().to(device)
        print(f"  Fold {fold} Phase 1 (balanced pretrain)...", end=" ", flush=True)
        _, _, ep1, state1 = train_seg(model, bal_root, kfold, fold, device, epochs=80, patience=15)
        print(f"epoch={ep1}", flush=True)
        model.load_state_dict(state1)
        model = model.to(device)
        model.freeze_encoder()
        print(f"  Fold {fold} Phase 2 (frozen encoder, natural)...", end=" ", flush=True)
        val, test, ep2, _ = train_seg(model, nat_root, kfold, fold, device, epochs=80, patience=15)
        print(f"epoch={ep2}, test={test:.4f}", flush=True)
        c_mious.append(test)
    print(f"  Mean: {np.mean(c_mious):.4f} ± {np.std(c_mious):.4f}")
    results["balanced_pretrain_frozen"] = {"mean": float(np.mean(c_mious)), "std": float(np.std(c_mious)), "folds": c_mious}
    
    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"{'Condition':<40} {'mIoU':>12}")
    print(f"{'-'*55}")
    for name, r in results.items():
        print(f"{name:<40} {r['mean']:.4f} ± {r['std']:.4f}")
    
    # Paired tests
    from scipy import stats
    for name in ["balanced_pretrain_finetune", "balanced_pretrain_frozen"]:
        diff = np.array(results[name]["folds"]) - np.array(results["scratch_natural"]["folds"])
        t, p = stats.ttest_rel(results[name]["folds"], results["scratch_natural"]["folds"])
        print(f"\n{name} vs scratch: diff={np.mean(diff):+.4f}, t={t:.3f}, p={p:.4f}")
    
    out_dir = ROOT / "runs" / "direction7_domain_adaptation"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "results.json", {**results, "timestamp": utc_now_iso()})
    print(f"\nSaved to {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
