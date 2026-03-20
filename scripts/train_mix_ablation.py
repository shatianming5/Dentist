#!/usr/bin/env python3
"""Mixing ratio ablation for DGCNN segmentation.

Varies the fraction of natural-protocol data mixed into balanced training.
Ratios: 0% (balanced-only), 25%, 50%, 75%, 100% (full mixing).
Evaluates on both balanced and natural test sets.
"""
from __future__ import annotations
import argparse, json, math, os, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib.io import read_json, read_jsonl, write_json
from _lib.point_ops import get_graph_feature
from _lib.seed import set_seed
from _lib.device import normalize_device

# ── Reuse DGCNNv2Seg from main training script ───────────────────────────
from phase3_train_raw_seg import (
    DGCNNv2Seg, RawSegDataset, compute_seg_metrics, focal_loss
)


def train_one_epoch(model, loader, optimizer, device, class_weights):
    model.train()
    total_loss = 0
    n = 0
    for batch in loader:
        pts = batch["points"].to(device)
        lbl = batch["labels"].to(device)
        logits = model(pts)
        loss = F.cross_entropy(logits, lbl, weight=class_weights)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * pts.size(0)
        n += pts.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    n = 0
    for batch in loader:
        pts = batch["points"].to(device)
        lbl = batch["labels"].to(device)
        logits = model(pts)
        loss = F.cross_entropy(logits, lbl)
        total_loss += loss.item() * pts.size(0)
        n += pts.size(0)
        all_preds.append(logits.argmax(1).cpu().numpy().ravel())
        all_labels.append(lbl.cpu().numpy().ravel())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    metrics = compute_seg_metrics(preds, labels, num_classes)
    metrics["loss"] = total_loss / max(n, 1)
    return metrics


def subsample_dataset(dataset, ratio, rng):
    """Subsample a dataset to keep `ratio` fraction of samples."""
    n = len(dataset)
    k = max(1, int(round(n * ratio)))
    indices = rng.choice(n, k, replace=False).tolist()
    return Subset(dataset, indices)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=float, required=True,
                    help="Fraction of natural data to mix (0.0=bal-only, 1.0=full mix)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-points", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bal-root", type=Path, default=Path("processed/raw_seg/v1"))
    ap.add_argument("--nat-root", type=Path, default=Path("processed/raw_seg/v2_natural"))
    ap.add_argument("--run-dir", type=Path, default=None)
    args = ap.parse_args()

    if args.run_dir is None:
        tag = f"ratio{int(args.ratio*100)}"
        args.run_dir = Path(f"runs/mix_ablation/dgcnn_{tag}_s{args.seed}_f{args.fold}")
    args.run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = normalize_device(args.device)
    num_classes = 2

    # Load indices
    bal_rows = list(read_jsonl(args.bal_root / "index.jsonl"))
    nat_rows = list(read_jsonl(args.nat_root / "index.jsonl"))

    # Get fold split using case_to_fold mapping
    kfold_obj = read_json(Path("metadata/splits_raw_case_kfold.json"))
    k = int(kfold_obj["k"])
    test_fold = args.fold
    val_fold = (test_fold + 1) % k
    c2f = kfold_obj["case_to_fold"]

    def assign_splits(rows):
        for row in rows:
            f = int(c2f.get(row["case_key"], -1))
            if f == test_fold:
                row["split"] = "test"
            elif f == val_fold:
                row["split"] = "val"
            else:
                row["split"] = "train"
        return rows

    bal_rows = assign_splits(bal_rows)
    nat_rows = assign_splits(nat_rows)

    bal_train = [r for r in bal_rows if r["split"] == "train"]
    bal_val = [r for r in bal_rows if r["split"] == "val"]
    bal_test = [r for r in bal_rows if r["split"] == "test"]
    nat_train = [r for r in nat_rows if r["split"] == "train"]
    nat_test = [r for r in nat_rows if r["split"] == "test"]

    # Build training dataset with mixing ratio
    bal_train_ds = RawSegDataset(args.bal_root, bal_train, args.n_points, augment=True)

    if args.ratio > 0:
        nat_train_ds = RawSegDataset(args.nat_root, nat_train, args.n_points, augment=True)
        if args.ratio < 1.0:
            rng = np.random.RandomState(args.seed)
            nat_train_ds = subsample_dataset(nat_train_ds, args.ratio, rng)
        train_ds = ConcatDataset([bal_train_ds, nat_train_ds])
    else:
        train_ds = bal_train_ds

    val_ds = RawSegDataset(args.bal_root, bal_val, args.n_points, augment=False)
    nat_test_ds = RawSegDataset(args.nat_root, nat_test, args.n_points, augment=False)
    bal_test_ds = RawSegDataset(args.bal_root, bal_test, args.n_points, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    nat_test_loader = DataLoader(nat_test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=4, pin_memory=True)
    bal_test_loader = DataLoader(bal_test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=4, pin_memory=True)

    print(f"[config] ratio={args.ratio}, seed={args.seed}, fold={args.fold}")
    print(f"[data] train={len(train_ds)} (bal={len(bal_train_ds)}"
          f"{f'+nat_sub={len(nat_train_ds) if isinstance(nat_train_ds, Subset) else len(nat_train_ds)}' if args.ratio > 0 else ''})")
    print(f"[data] val={len(val_ds)}, test_nat={len(nat_test_ds)}, test_bal={len(bal_test_ds)}")

    # Class weights from training labels
    all_labels = []
    for i in range(len(bal_train_ds)):
        all_labels.append(bal_train_ds[i]["labels"].numpy())
    all_labels = np.concatenate(all_labels)
    counts = np.bincount(all_labels, minlength=num_classes).astype(float)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    model = DGCNNv2Seg(num_classes=num_classes, dropout=0.3, k=20, emb_dims=512).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_miou = -1
    best_state = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, class_weights)
        val_metrics = evaluate(model, val_loader, device, num_classes)
        scheduler.step()
        elapsed = time.time() - t0

        val_miou = val_metrics["mean_iou"]
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  ep {epoch:3d} | loss={train_loss:.4f} val_miou={val_miou:.4f} "
                  f"best={best_val_miou:.4f} pat={patience_counter} [{elapsed:.1f}s]")

        if patience_counter >= args.patience:
            print(f"  Early stop at epoch {epoch}")
            break

    # Evaluate best model
    model.load_state_dict(best_state)
    model.to(device)
    nat_metrics = evaluate(model, nat_test_loader, device, num_classes)
    bal_metrics = evaluate(model, bal_test_loader, device, num_classes)

    results = {
        "ratio": args.ratio,
        "seed": args.seed,
        "fold": args.fold,
        "best_epoch": epoch - patience_counter,
        "best_val_miou": best_val_miou,
        "test_nat_miou": nat_metrics["mean_iou"],
        "test_bal_miou": bal_metrics["mean_iou"],
        "train_size": len(train_ds),
    }
    write_json(args.run_dir / "results.json", results)
    print(f"\n[done] ratio={args.ratio} nat={nat_metrics['mean_iou']:.4f} "
          f"bal={bal_metrics['mean_iou']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
