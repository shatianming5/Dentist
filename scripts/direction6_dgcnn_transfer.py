#!/usr/bin/env python3
"""D6.2: Transfer DGCNN encoder (conv1-4) from Teeth3DS pretrain → restoration segmentation.
Compare with scratch baseline using paired 5-fold CV + bootstrap test."""

from __future__ import annotations
import sys, os, json, copy, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from scipy import stats

from phase3_train_raw_seg import DGCNNv2Seg, RawSegDataset, get_graph_feature, read_jsonl

# ─── Config ───
PRETRAIN_CKPT = "runs/pretrain/teeth3ds_fdi_dgcnn_seed1337/ckpt_best.pt"
DATA_ROOT = "processed/raw_seg/v2_natural"
OUT_DIR = "runs/direction6_dgcnn_transfer"
N_POINTS = 8192
N_FOLDS = 5
SEED = 1337
EPOCHS = 120
PATIENCE = 25
LR = 1e-3
BATCH = 8
N_BOOTSTRAP = 10000


def load_pretrained_encoder(seg_model: DGCNNv2Seg, ckpt_path: str) -> int:
    """Transfer conv1-conv4 from DGCNNv2Classifier → DGCNNv2Seg."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cls_sd = ckpt["model_state"]
    
    transferred = 0
    seg_sd = seg_model.state_dict()
    for i in range(1, 5):  # conv1 through conv4
        for suffix in ["0.weight", "1.weight", "1.bias", "1.running_mean", "1.running_var", "1.num_batches_tracked"]:
            cls_key = f"feat.conv{i}.{suffix}"
            seg_key = f"conv{i}.{suffix}"
            if cls_key in cls_sd and seg_key in seg_sd:
                if cls_sd[cls_key].shape == seg_sd[seg_key].shape:
                    seg_sd[seg_key] = cls_sd[cls_key]
                    transferred += 1
    
    seg_model.load_state_dict(seg_sd)
    return transferred


def train_one_fold(model, train_loader, val_loader, device, epochs, patience, lr):
    """Train and return best val mIoU."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_miou = 0.0
    best_state = None
    wait = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            pts = batch["points"].to(device)
            lbl = batch["labels"].to(device)
            logits = model(pts)
            loss = criterion(logits, lbl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        # Validate
        model.eval()
        ious = []
        with torch.no_grad():
            for batch in val_loader:
                pts = batch["points"].to(device)
                lbl = batch["labels"].to(device)
                pred = model(pts).argmax(1)
                for c in range(2):
                    inter = ((pred == c) & (lbl == c)).sum().item()
                    union = ((pred == c) | (lbl == c)).sum().item()
                    if union > 0:
                        ious.append(inter / union)
        
        miou = np.mean(ious) if ious else 0.0
        if miou > best_miou:
            best_miou = miou
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    
    model.load_state_dict(best_state)
    return best_miou


def evaluate(model, loader, device):
    """Get per-sample mIoU for bootstrap test."""
    model.eval()
    sample_mious = []
    with torch.no_grad():
        for batch in loader:
            pts = batch["points"].to(device)
            lbl = batch["labels"].to(device)
            pred = model(pts).argmax(1)
            for i in range(pts.shape[0]):
                ious = []
                for c in range(2):
                    inter = ((pred[i] == c) & (lbl[i] == c)).sum().item()
                    union = ((pred[i] | lbl[i]) if c == 1 else ((pred[i] == c) | (lbl[i] == c))).sum().item()
                    # Correct IoU
                    inter = ((pred[i] == c) & (lbl[i] == c)).sum().item()
                    union = ((pred[i] == c) | (lbl[i] == c)).sum().item()
                    if union > 0:
                        ious.append(inter / union)
                sample_mious.append(np.mean(ious) if ious else 0.0)
    return np.array(sample_mious)


def paired_bootstrap_test(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 42):
    """Two-sided paired bootstrap test: H0: mean(a) = mean(b)."""
    rng = np.random.RandomState(seed)
    obs_diff = np.mean(b) - np.mean(a)
    n = len(a)
    diffs = b - a
    count = 0
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_diff = np.mean(diffs[idx])
        if abs(boot_diff) >= abs(obs_diff):
            count += 1
    return obs_diff, count / n_boot


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    from pathlib import Path
    data_root = Path(DATA_ROOT)
    index_rows = read_jsonl(data_root / "index.jsonl")
    labels = [r.get("label", "unknown") for r in index_rows]
    labels = np.array(labels)
    
    print(f"Dataset: {len(index_rows)} samples, device: {device}")
    print(f"Pretrain ckpt: {PRETRAIN_CKPT}")
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    results = {"scratch": [], "pretrained": []}
    all_scratch_mious = []
    all_pretrain_mious = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(index_rows)), labels)):
        print(f"\n{'='*60}")
        print(f"Fold {fold}")
        print(f"{'='*60}")
        
        train_rows = [index_rows[i] for i in train_idx]
        val_rows = [index_rows[i] for i in val_idx]
        
        train_ds = RawSegDataset(data_root, train_rows, N_POINTS, augment=True)
        val_ds = RawSegDataset(data_root, val_rows, N_POINTS, augment=False)
        
        train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, 
                                  num_workers=4, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False,
                                num_workers=4, pin_memory=True)
        
        # --- Scratch ---
        torch.manual_seed(SEED + fold)
        model_scratch = DGCNNv2Seg(num_classes=2, k=20, emb_dims=512).to(device)
        miou_scratch = train_one_fold(model_scratch, train_loader, val_loader, device, EPOCHS, PATIENCE, LR)
        scratch_per_sample = evaluate(model_scratch, val_loader, device)
        results["scratch"].append(miou_scratch)
        all_scratch_mious.append(scratch_per_sample)
        print(f"  Scratch mIoU: {miou_scratch:.4f}")
        
        # --- Pretrained ---
        torch.manual_seed(SEED + fold)
        model_pre = DGCNNv2Seg(num_classes=2, k=20, emb_dims=512).to(device)
        n_transferred = load_pretrained_encoder(model_pre, PRETRAIN_CKPT)
        print(f"  Transferred {n_transferred} tensors from DGCNN pretrain")
        miou_pre = train_one_fold(model_pre, train_loader, val_loader, device, EPOCHS, PATIENCE, LR)
        pre_per_sample = evaluate(model_pre, val_loader, device)
        results["pretrained"].append(miou_pre)
        all_pretrain_mious.append(pre_per_sample)
        print(f"  Pretrained mIoU: {miou_pre:.4f}")
        print(f"  Δ = {miou_pre - miou_scratch:+.4f}")
    
    # Aggregate
    s_mean = np.mean(results["scratch"])
    p_mean = np.mean(results["pretrained"])
    delta = p_mean - s_mean
    
    # Paired t-test on fold means
    t_stat, t_pval = stats.ttest_rel(results["pretrained"], results["scratch"])
    
    # Bootstrap on per-sample
    all_s = np.concatenate(all_scratch_mious)
    all_p = np.concatenate(all_pretrain_mious)
    boot_diff, boot_pval = paired_bootstrap_test(all_s, all_p, N_BOOTSTRAP)
    
    print(f"\n{'='*60}")
    print(f"DGCNN Transfer Results (D6.2)")
    print(f"{'='*60}")
    print(f"Scratch:    {s_mean:.4f} ({results['scratch']})")
    print(f"Pretrained: {p_mean:.4f} ({results['pretrained']})")
    print(f"Δ mIoU:     {delta:+.4f}")
    print(f"Paired t:   t={t_stat:.3f}, p={t_pval:.4f}")
    print(f"Bootstrap:  Δ={boot_diff:+.4f}, p={boot_pval:.4f}")
    
    # Save
    out = {
        "experiment": "D6.2_DGCNN_transfer",
        "description": "Transfer DGCNN conv1-4 from Teeth3DS FDI classification to restoration segmentation",
        "pretrain_source": PRETRAIN_CKPT,
        "pretrain_val_acc": 0.7236,
        "pretrain_val_f1": 0.6535,
        "n_transferred_tensors": n_transferred,
        "scratch": {"fold_mious": results["scratch"], "mean": s_mean},
        "pretrained": {"fold_mious": results["pretrained"], "mean": p_mean},
        "delta_miou": delta,
        "paired_t": {"t_stat": t_stat, "p_value": t_pval},
        "bootstrap": {"observed_diff": boot_diff, "p_value": boot_pval, "n_bootstrap": N_BOOTSTRAP},
        "conclusion": "significant" if boot_pval < 0.05 else "not_significant",
    }
    
    out_path = os.path.join(OUT_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
