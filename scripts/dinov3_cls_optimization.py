#!/usr/bin/env python3
"""DINOv3 classification optimization experiments.

Strategies:
1. Seg-guided: use dense per-point DINOv3 features, mask to restoration, pool → classify
2. Enriched pooling: CLS + mean + max instead of CLS only
3. Binary classification: Filling (direct) vs Indirect (crown+post-core+onlay)
4. Prototypical network (few-shot friendly)
5. Stronger regularization + class-balanced loss
"""

from __future__ import annotations
import sys, os, json, copy, math
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from pathlib import Path

# ─── Config ───
CLS_FEAT_DIR = Path("processed/raw_cls/v13_main4/dinov3_features")
SEG_FEAT_DIR = Path("processed/raw_seg/v1/dinov3_features")
DATA_ROOT = Path("processed/raw_seg/v2_natural")
OUT_DIR = Path("runs/dinov3_cls_optimization")
N_FOLDS = 5
SEED = 1337
EPOCHS = 300
PATIENCE = 50
BATCH = 16


# ──────────────────────────────────────────────
# Datasets
# ──────────────────────────────────────────────

class GlobalFeatDataset(Dataset):
    """Load precomputed global (768,) features."""
    def __init__(self, feat_dir, case_keys, labels, label_to_id):
        self.feat_dir = Path(feat_dir)
        self.case_keys = case_keys
        self.labels = labels
        self.label_to_id = label_to_id

    def __len__(self):
        return len(self.case_keys)

    def __getitem__(self, idx):
        key = self.case_keys[idx]
        safe = key.replace("/", "__").replace(" ", "_") + "_dinov3.npz"
        d = np.load(str(self.feat_dir / safe))
        feat = torch.from_numpy(d["features"].astype(np.float32))
        label = self.label_to_id[self.labels[idx]]
        return {"features": feat, "label": label}


class DenseFeatDataset(Dataset):
    """Load dense per-point (8192, 384) features + seg labels for seg-guided cls."""
    def __init__(self, feat_dir, case_keys, labels, label_to_id):
        self.feat_dir = Path(feat_dir)
        self.case_keys = case_keys
        self.labels = labels
        self.label_to_id = label_to_id

    def __len__(self):
        return len(self.case_keys)

    def __getitem__(self, idx):
        key = self.case_keys[idx]
        safe = key.replace("/", "__").replace(" ", "_") + "_dinov3.npz"
        path = self.feat_dir / safe
        if not path.exists():
            # Try alternative naming
            safe2 = key.replace("/", "__") + "_dinov3.npz"
            path = self.feat_dir / safe2
        d = np.load(str(path))
        feats = torch.from_numpy(d["features"].astype(np.float32))  # (8192, 384)
        seg_labels = torch.from_numpy(d["labels"].astype(np.int64))  # (8192,)
        cls_label = self.label_to_id[self.labels[idx]]
        return {"features": feats, "seg_labels": seg_labels, "label": cls_label}


# ──────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────

class MLPHead(nn.Module):
    """Standard MLP classifier."""
    def __init__(self, in_dim, num_classes, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class SegGuidedClassifier(nn.Module):
    """Use dense per-point features, mask to restoration, pool, classify."""
    def __init__(self, feat_dim=384, num_classes=4, hidden=256, dropout=0.3):
        super().__init__()
        # Pool: mean + max + std of restoration points → 3 * feat_dim
        self.head = MLPHead(feat_dim * 3, num_classes, hidden, dropout)

    def forward(self, features, seg_labels):
        # features: (B, N, D), seg_labels: (B, N)
        B, N, D = features.shape
        outputs = []
        for i in range(B):
            mask = seg_labels[i] == 1  # restoration points
            if mask.sum() < 5:
                # Fallback: use all points
                mask = torch.ones(N, dtype=torch.bool, device=features.device)
            pts = features[i, mask]  # (M, D)
            mean_f = pts.mean(0)
            max_f = pts.max(0).values
            std_f = pts.std(0) if pts.shape[0] > 1 else torch.zeros(D, device=features.device)
            outputs.append(torch.cat([mean_f, max_f, std_f]))
        pooled = torch.stack(outputs)  # (B, 3D)
        return self.head(pooled)


class PrototypicalClassifier:
    """Few-shot prototypical network: compute class centroids, classify by distance."""
    def __init__(self):
        self.prototypes = None

    def fit(self, features, labels, num_classes):
        self.prototypes = []
        for c in range(num_classes):
            mask = labels == c
            if mask.sum() > 0:
                self.prototypes.append(features[mask].mean(0))
            else:
                self.prototypes.append(torch.zeros_like(features[0]))
        self.prototypes = torch.stack(self.prototypes)

    def predict(self, features):
        # Cosine similarity
        feats_norm = F.normalize(features, dim=1)
        proto_norm = F.normalize(self.prototypes, dim=1)
        sim = feats_norm @ proto_norm.T
        return sim.argmax(1)


# ──────────────────────────────────────────────
# Training utilities
# ──────────────────────────────────────────────

def get_class_weights(labels, num_classes):
    """Inverse frequency class weights."""
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts[counts == 0] = 1
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.FloatTensor(weights)


def train_mlp(model, train_loader, val_loader, device, epochs, patience, lr,
              class_weights=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            feat = batch["features"].to(device)
            lbl = batch["label"].to(device)
            if "seg_labels" in batch:
                seg = batch["seg_labels"].to(device)
                logits = model(feat, seg)
            else:
                logits = model(feat)
            loss = criterion(logits, lbl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Val
        model.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                feat = batch["features"].to(device)
                lbl = batch["label"]
                if "seg_labels" in batch:
                    seg = batch["seg_labels"].to(device)
                    logits = model(feat, seg)
                else:
                    logits = model(feat)
                pred = logits.argmax(1).cpu()
                all_pred.extend(pred.tolist())
                all_true.extend(lbl.tolist())
        f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return best_f1


def evaluate(model, loader, device, id_to_label):
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in loader:
            feat = batch["features"].to(device)
            lbl = batch["label"]
            if "seg_labels" in batch:
                seg = batch["seg_labels"].to(device)
                logits = model(feat, seg)
            else:
                logits = model(feat)
            pred = logits.argmax(1).cpu()
            all_pred.extend(pred.tolist())
            all_true.extend(lbl.tolist())
    acc = accuracy_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)
    return acc, f1, all_pred, all_true


# ──────────────────────────────────────────────
# Experiments
# ──────────────────────────────────────────────

def load_manifest():
    """Load case keys and labels."""
    from _lib.io import read_jsonl
    # Use v1 index for cls (same 79 cases)
    index_path = Path("processed/raw_seg/v1/index.jsonl")
    rows = read_jsonl(index_path)
    case_keys = [r["case_key"] for r in rows]
    labels = [r.get("label", "unknown") for r in rows]
    return case_keys, labels


def run_experiment(name, dataset_cls, feat_dir, model_fn, case_keys, labels,
                   label_to_id, id_to_label, num_classes, device):
    """Run 5-fold CV experiment."""
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"{'='*60}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    label_arr = np.array(labels)
    label_ids = np.array([label_to_id[l] for l in labels])

    fold_accs, fold_f1s = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(labels)), labels)):
        torch.manual_seed(SEED + fold)
        np.random.seed(SEED + fold)

        train_keys = [case_keys[i] for i in train_idx]
        val_keys = [case_keys[i] for i in val_idx]
        train_labels = [labels[i] for i in train_idx]
        val_labels = [labels[i] for i in val_idx]

        train_ds = dataset_cls(feat_dir, train_keys, train_labels, label_to_id)
        val_ds = dataset_cls(feat_dir, val_keys, val_labels, label_to_id)

        train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False)

        # Class weights
        train_label_ids = [label_to_id[l] for l in train_labels]
        cw = get_class_weights(np.array(train_label_ids), num_classes)

        model = model_fn().to(device)
        train_mlp(model, train_loader, val_loader, device, EPOCHS, PATIENCE, 
                  lr=5e-4, class_weights=cw)
        acc, f1, preds, trues = evaluate(model, val_loader, device, id_to_label)
        fold_accs.append(acc)
        fold_f1s.append(f1)
        print(f"  Fold {fold}: Acc={acc:.3f}, F1={f1:.3f}")

    mean_acc = np.mean(fold_accs)
    mean_f1 = np.mean(fold_f1s)
    print(f"  → Mean Acc={mean_acc:.4f}, Mean F1={mean_f1:.4f}")
    return {"name": name, "fold_accs": fold_accs, "fold_f1s": fold_f1s,
            "mean_acc": float(mean_acc), "mean_f1": float(mean_f1)}


def run_prototypical(case_keys, labels, label_to_id, num_classes, device):
    """Prototypical network on DINOv3 global features."""
    print(f"\n{'='*60}")
    print(f"Experiment: Prototypical Network (DINOv3 global)")
    print(f"{'='*60}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_accs, fold_f1s = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(labels)), labels)):
        # Load features
        train_feats, train_labels = [], []
        val_feats, val_labels = [], []

        for idx in train_idx:
            safe = case_keys[idx].replace("/", "__").replace(" ", "_") + "_dinov3.npz"
            d = np.load(str(CLS_FEAT_DIR / safe))
            train_feats.append(d["features"])
            train_labels.append(label_to_id[labels[idx]])

        for idx in val_idx:
            safe = case_keys[idx].replace("/", "__").replace(" ", "_") + "_dinov3.npz"
            d = np.load(str(CLS_FEAT_DIR / safe))
            val_feats.append(d["features"])
            val_labels.append(label_to_id[labels[idx]])

        train_feats = torch.FloatTensor(np.array(train_feats)).to(device)
        train_labels_t = torch.LongTensor(train_labels).to(device)
        val_feats = torch.FloatTensor(np.array(val_feats)).to(device)
        val_labels_arr = np.array(val_labels)

        proto = PrototypicalClassifier()
        proto.fit(train_feats, train_labels_t, num_classes)
        preds = proto.predict(val_feats).cpu().numpy()

        acc = accuracy_score(val_labels_arr, preds)
        f1 = f1_score(val_labels_arr, preds, average="macro", zero_division=0)
        fold_accs.append(acc)
        fold_f1s.append(f1)
        print(f"  Fold {fold}: Acc={acc:.3f}, F1={f1:.3f}")

    mean_acc = np.mean(fold_accs)
    mean_f1 = np.mean(fold_f1s)
    print(f"  → Mean Acc={mean_acc:.4f}, Mean F1={mean_f1:.4f}")
    return {"name": "Prototypical (cosine)", "fold_accs": fold_accs, "fold_f1s": fold_f1s,
            "mean_acc": float(mean_acc), "mean_f1": float(mean_f1)}


def run_binary(case_keys, labels, label_to_id_bin, device):
    """Binary: Filling vs Indirect."""
    print(f"\n{'='*60}")
    print(f"Experiment: Binary (Filling vs Indirect)")
    print(f"{'='*60}")

    bin_labels = ["direct" if l == "充填" else "indirect" for l in labels]
    bin_l2id = {"direct": 0, "indirect": 1}
    bin_id2l = {0: "direct", 1: "indirect"}

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_accs, fold_f1s = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(bin_labels)), bin_labels)):
        torch.manual_seed(SEED + fold)
        train_keys = [case_keys[i] for i in train_idx]
        val_keys = [case_keys[i] for i in val_idx]
        train_bl = [bin_labels[i] for i in train_idx]
        val_bl = [bin_labels[i] for i in val_idx]

        train_ds = GlobalFeatDataset(CLS_FEAT_DIR, train_keys, train_bl, bin_l2id)
        val_ds = GlobalFeatDataset(CLS_FEAT_DIR, val_keys, val_bl, bin_l2id)
        train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False)

        cw = get_class_weights(np.array([bin_l2id[l] for l in train_bl]), 2)
        model = MLPHead(768, 2, hidden=128, dropout=0.4).to(device)
        train_mlp(model, train_loader, val_loader, device, EPOCHS, PATIENCE,
                  lr=5e-4, class_weights=cw)
        acc, f1, _, _ = evaluate(model, val_loader, device, bin_id2l)
        fold_accs.append(acc)
        fold_f1s.append(f1)
        print(f"  Fold {fold}: Acc={acc:.3f}, F1={f1:.3f}")

    mean_acc = np.mean(fold_accs)
    mean_f1 = np.mean(fold_f1s)
    print(f"  → Mean Acc={mean_acc:.4f}, Mean F1={mean_f1:.4f}")
    return {"name": "Binary (direct vs indirect)", "fold_accs": fold_accs, "fold_f1s": fold_f1s,
            "mean_acc": float(mean_acc), "mean_f1": float(mean_f1),
            "class_split": "Filling(12) vs Indirect(67)"}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    case_keys, labels = load_manifest()
    label_set = sorted(set(labels))
    label_to_id = {l: i for i, l in enumerate(label_set)}
    id_to_label = {i: l for l, i in label_to_id.items()}
    num_classes = len(label_set)

    print(f"Cases: {len(case_keys)}, Classes: {num_classes}")
    print(f"Label map: {label_to_id}")
    print(f"Distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    all_results = []

    # Exp 1: Baseline (same as existing — CLS token + MLP, no class weights)
    r1 = run_experiment(
        "Baseline (CLS token + MLP)",
        GlobalFeatDataset, CLS_FEAT_DIR,
        lambda: MLPHead(768, num_classes, hidden=256, dropout=0.3),
        case_keys, labels, label_to_id, id_to_label, num_classes, device
    )
    all_results.append(r1)

    # Exp 2: Stronger regularization + class-balanced + GELU + AdamW
    r2 = run_experiment(
        "CLS + balanced loss + strong reg",
        GlobalFeatDataset, CLS_FEAT_DIR,
        lambda: MLPHead(768, num_classes, hidden=128, dropout=0.5),
        case_keys, labels, label_to_id, id_to_label, num_classes, device
    )
    all_results.append(r2)

    # Exp 3: Seg-guided classification (dense features + restoration mask)
    r3 = run_experiment(
        "Seg-guided (dense feat + mask)",
        DenseFeatDataset, SEG_FEAT_DIR,
        lambda: SegGuidedClassifier(384, num_classes, hidden=256, dropout=0.3),
        case_keys, labels, label_to_id, id_to_label, num_classes, device
    )
    all_results.append(r3)

    # Exp 4: Seg-guided + strong regularization
    r4 = run_experiment(
        "Seg-guided + strong reg",
        DenseFeatDataset, SEG_FEAT_DIR,
        lambda: SegGuidedClassifier(384, num_classes, hidden=128, dropout=0.5),
        case_keys, labels, label_to_id, id_to_label, num_classes, device
    )
    all_results.append(r4)

    # Exp 5: Prototypical network
    r5 = run_prototypical(case_keys, labels, label_to_id, num_classes, device)
    all_results.append(r5)

    # Exp 6: Binary classification
    r6 = run_binary(case_keys, labels, None, device)
    all_results.append(r6)

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY — DINOv3 Classification Optimization")
    print(f"{'='*70}")
    print(f"{'Experiment':<40} {'Acc':>8} {'F1':>8}")
    print("-" * 58)
    for r in all_results:
        print(f"{r['name']:<40} {r['mean_acc']:>8.4f} {r['mean_f1']:>8.4f}")
    print(f"\n{'Previous best (PointNet whole)':<40} {'0.329':>8} {'0.279':>8}")
    print(f"{'Random baseline':<40} {'0.250':>8} {'0.250':>8}")

    # Save
    out_path = OUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump({"experiments": all_results}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
