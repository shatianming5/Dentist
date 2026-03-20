#!/usr/bin/env python3
"""Protocol-Adversarial Feature Alignment (PAFA) for DGCNN segmentation.

Trains DGCNN on mixed-protocol data with an adversarial protocol discriminator
via gradient reversal layer, forcing the encoder to learn protocol-invariant features.

Comparisons:
  - mixing:      standard mixed-protocol training (baseline)
  - mixing+PAFA: mixed-protocol training with adversarial protocol alignment
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import DataLoader, Dataset, ConcatDataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib.device import normalize_device
from _lib.io import read_json, read_jsonl, write_json
from _lib.point_ops import get_graph_feature
from _lib.seed import set_seed


# ---------------------------------------------------------------------------
# Gradient Reversal Layer
# ---------------------------------------------------------------------------

class GradReverse(Function):
    """Gradient reversal layer: forward passes through, backward flips sign."""
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


# ---------------------------------------------------------------------------
# DGCNN Segmentation (same as phase3_train_raw_seg.py)
# ---------------------------------------------------------------------------

class DGCNNv2Seg(nn.Module):
    def __init__(self, num_classes: int = 2, dropout: float = 0.3,
                 k: int = 20, emb_dims: int = 512) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.k = int(k)
        self.emb_dims = int(emb_dims)
        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, 1, bias=False), nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, 1, bias=False), nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, 1, bias=False), nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, 1, bias=False), nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True))
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, int(emb_dims), 1, bias=False), nn.BatchNorm1d(int(emb_dims)),
            nn.LeakyReLU(0.2, True))
        seg_in = 512 + int(emb_dims)
        self.seg_head = nn.Sequential(
            nn.Conv1d(seg_in, 256, 1, bias=False), nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, True), nn.Dropout(float(dropout)),
            nn.Conv1d(256, 128, 1, bias=False), nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, True), nn.Dropout(float(dropout)),
            nn.Conv1d(128, int(num_classes), 1),
        )

    def forward(self, points: torch.Tensor, return_features: bool = False):
        x = points.transpose(1, 2).contiguous()  # (B, 3, N)
        N = x.shape[2]
        x1 = self.conv1(get_graph_feature(x, k=self.k)).max(dim=-1).values
        x2 = self.conv2(get_graph_feature(x1, k=self.k)).max(dim=-1).values
        x3 = self.conv3(get_graph_feature(x2, k=self.k)).max(dim=-1).values
        x4 = self.conv4(get_graph_feature(x3, k=self.k)).max(dim=-1).values
        x_cat = torch.cat([x1, x2, x3, x4], dim=1)  # (B, 512, N)
        x5 = self.conv5(x_cat)  # (B, emb_dims, N)
        g = torch.max(x5, dim=2, keepdim=True).values  # (B, emb_dims, 1)
        g_expand = g.expand(-1, -1, N)
        feat = torch.cat([x_cat, g_expand], dim=1)  # (B, 512+emb_dims, N)
        logits = self.seg_head(feat)  # (B, C, N)
        if return_features:
            return logits, g.squeeze(2)  # logits + global feature (B, emb_dims)
        return logits


# ---------------------------------------------------------------------------
# Protocol Discriminator
# ---------------------------------------------------------------------------

class ProtocolDiscriminator(nn.Module):
    """Lightweight MLP that classifies global features → protocol label."""
    def __init__(self, in_dim: int = 512, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(hidden, 64),
            nn.ReLU(True),
            nn.Linear(64, 2),  # 2 protocols: balanced(0) vs natural(1)
        )

    def forward(self, feat):
        return self.net(feat)


# ---------------------------------------------------------------------------
# Dataset with protocol labels
# ---------------------------------------------------------------------------

class MixedProtocolDataset(Dataset):
    """Loads point clouds with both segmentation labels and protocol labels."""
    def __init__(self, data_root: Path, index_rows: list[dict], n_points: int,
                 protocol_label: int, augment: bool = False):
        self.data_root = Path(data_root)
        self.rows = list(index_rows)
        self.n_points = int(n_points)
        self.protocol_label = int(protocol_label)  # 0=balanced, 1=natural
        self.augment = bool(augment)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        data = np.load(str(self.data_root / row["sample_npz"]))
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
            pts = pts * np.random.uniform(0.8, 1.2)
            pts = pts + np.random.normal(0, 0.01, pts.shape).astype(np.float32)
        return {
            "points": torch.from_numpy(pts),
            "labels": torch.from_numpy(lbl),
            "protocol": torch.tensor(self.protocol_label, dtype=torch.long),
            "case_key": row["case_key"],
        }


# ---------------------------------------------------------------------------
# Lambda schedule (progressive ramp-up from DANN paper)
# ---------------------------------------------------------------------------

def lambda_schedule(epoch: int, max_epochs: int, gamma: float = 10.0) -> float:
    """DANN-style progressive lambda: 0 → 1 over training."""
    p = epoch / max_epochs
    return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

def focal_loss(logits, targets, alpha=None, gamma=2.0):
    ce = F.cross_entropy(logits, targets, weight=alpha, reduction='none')
    pt = torch.exp(-ce)
    return (((1 - pt) ** gamma) * ce).mean()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_seg_metrics(pred, gt, num_classes):
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

def train_one_epoch_pafa(model, discriminator, loader, optimizer_seg, optimizer_disc,
                         device, class_weights, lambd, use_focal=True):
    model.train()
    discriminator.train()
    total_seg_loss = 0.0
    total_disc_loss = 0.0
    total_correct = 0
    total_points = 0
    disc_correct = 0
    disc_total = 0

    for batch in loader:
        pts = batch["points"].to(device)
        lbl = batch["labels"].to(device)
        proto = batch["protocol"].to(device)  # (B,)

        # Forward: get segmentation logits + global features
        logits, global_feat = model(pts, return_features=True)

        # Segmentation loss
        if use_focal:
            seg_loss = focal_loss(logits, lbl, alpha=class_weights)
        else:
            seg_loss = F.cross_entropy(logits, lbl, weight=class_weights)

        # Adversarial loss: gradient reversal on global features
        feat_reversed = grad_reverse(global_feat, lambd)
        disc_logits = discriminator(feat_reversed)
        disc_loss = F.cross_entropy(disc_logits, proto)

        # Combined loss
        total_loss = seg_loss + disc_loss

        optimizer_seg.zero_grad()
        optimizer_disc.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
        optimizer_seg.step()
        optimizer_disc.step()

        B = pts.shape[0]
        total_seg_loss += seg_loss.item() * B
        total_disc_loss += disc_loss.item() * B
        total_correct += int((logits.argmax(dim=1) == lbl).sum().item())
        total_points += int(lbl.numel())
        disc_correct += int((disc_logits.argmax(dim=1) == proto).sum().item())
        disc_total += B

    n = max(len(loader.dataset), 1)
    return {
        "seg_loss": total_seg_loss / n,
        "disc_loss": total_disc_loss / n,
        "seg_acc": total_correct / max(total_points, 1),
        "disc_acc": disc_correct / max(disc_total, 1),
    }


def train_one_epoch_mixing(model, loader, optimizer, device, class_weights, use_focal=True):
    """Standard mixing training (no adversarial loss)."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_points = 0
    for batch in loader:
        pts = batch["points"].to(device)
        lbl = batch["labels"].to(device)
        logits = model(pts)
        if use_focal:
            loss = focal_loss(logits, lbl, alpha=class_weights)
        else:
            loss = F.cross_entropy(logits, lbl, weight=class_weights)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * pts.shape[0]
        total_correct += int((logits.argmax(dim=1) == lbl).sum().item())
        total_points += int(lbl.numel())
    n = max(len(loader.dataset), 1)
    return {"loss": total_loss / n, "accuracy": total_correct / max(total_points, 1)}


@torch.no_grad()
def evaluate(model, loader, device, num_classes, class_weights=None):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    for batch in loader:
        pts = batch["points"].to(device)
        lbl = batch["labels"].to(device)
        out = model(pts)
        logits = out[0] if isinstance(out, tuple) else out
        loss = F.cross_entropy(logits, lbl, weight=class_weights)
        total_loss += loss.item() * pts.shape[0]
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(lbl.cpu().numpy())
    preds = np.concatenate(all_preds).ravel()
    labels = np.concatenate(all_labels).ravel()
    n = max(len(loader.dataset), 1)
    metrics = compute_seg_metrics(preds, labels, num_classes)
    metrics["loss"] = total_loss / n
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="PAFA: Protocol-Adversarial Feature Alignment")
    ap.add_argument("--mode", choices=["mixing", "pafa"], required=True)
    ap.add_argument("--bal-root", type=Path, default=Path("processed/raw_seg/v1_binary"))
    ap.add_argument("--nat-root", type=Path, default=Path("processed/raw_seg/v2_natural"))
    ap.add_argument("--kfold", type=Path, default=Path("metadata/splits_raw_case_kfold.json"))
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--disc-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--n-points", type=int, default=8192)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--emb-dims", type=int, default=512)
    ap.add_argument("--lambda-max", type=float, default=1.0,
                    help="Max adversarial lambda (ramped up from 0)")
    ap.add_argument("--run-root", type=Path, default=Path("runs/pafa"))
    ap.add_argument("--exp-name", type=str, default="")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(normalize_device(args.device))
    num_classes = 2  # binary segmentation

    # Load k-fold splits
    kfold_obj = read_json(args.kfold.resolve())
    k = int(kfold_obj["k"])
    test_fold = args.fold
    val_fold = (test_fold + 1) % k
    c2f = kfold_obj["case_to_fold"]

    def assign_splits(index_rows):
        for row in index_rows:
            f = int(c2f.get(row["case_key"], -1))
            if f == test_fold:
                row["split"] = "test"
            elif f == val_fold:
                row["split"] = "val"
            else:
                row["split"] = "train"
        return index_rows

    # Load balanced and natural data
    bal_rows = assign_splits(read_jsonl(args.bal_root.resolve() / "index.jsonl"))
    nat_rows = assign_splits(read_jsonl(args.nat_root.resolve() / "index.jsonl"))

    bal_train = [r for r in bal_rows if r["split"] == "train"]
    bal_val   = [r for r in bal_rows if r["split"] == "val"]
    bal_test  = [r for r in bal_rows if r["split"] == "test"]
    nat_train = [r for r in nat_rows if r["split"] == "train"]
    nat_val   = [r for r in nat_rows if r["split"] == "val"]
    nat_test  = [r for r in nat_rows if r["split"] == "test"]

    print(f"[data] bal: train={len(bal_train)} val={len(bal_val)} test={len(bal_test)}")
    print(f"[data] nat: train={len(nat_train)} val={len(nat_val)} test={len(nat_test)}")

    # Create mixed datasets with protocol labels
    bal_train_ds = MixedProtocolDataset(args.bal_root.resolve(), bal_train, args.n_points,
                                        protocol_label=0, augment=True)
    nat_train_ds = MixedProtocolDataset(args.nat_root.resolve(), nat_train, args.n_points,
                                        protocol_label=1, augment=True)
    mixed_train_ds = ConcatDataset([bal_train_ds, nat_train_ds])

    # Evaluation on natural-protocol test set (the challenging case)
    nat_test_ds = MixedProtocolDataset(args.nat_root.resolve(), nat_test, args.n_points,
                                       protocol_label=1, augment=False)
    bal_test_ds = MixedProtocolDataset(args.bal_root.resolve(), bal_test, args.n_points,
                                       protocol_label=0, augment=False)
    # Val on natural (we care about natural performance)
    nat_val_ds = MixedProtocolDataset(args.nat_root.resolve(), nat_val, args.n_points,
                                      protocol_label=1, augment=False)

    train_loader = DataLoader(mixed_train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=False)
    nat_val_loader = DataLoader(nat_val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=2, pin_memory=True)
    nat_test_loader = DataLoader(nat_test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=2, pin_memory=True)
    bal_test_loader = DataLoader(bal_test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=2, pin_memory=True)

    # Compute class weights from mixed training data
    label_counts = Counter()
    for ds in [bal_train_ds, nat_train_ds]:
        for row in ds.rows:
            d = np.load(str(ds.data_root / row["sample_npz"]))
            unique, counts = np.unique(d["labels"], return_counts=True)
            for u, c in zip(unique, counts):
                label_counts[int(u)] += int(c)
    total = sum(label_counts.values())
    w = [total / max(label_counts.get(c, 1) * num_classes, 1) for c in range(num_classes)]
    class_weights = torch.tensor(w, dtype=torch.float32, device=device)
    print(f"[weights] {class_weights.tolist()}")

    # Model
    model = DGCNNv2Seg(num_classes=num_classes, dropout=args.dropout,
                       k=args.k, emb_dims=args.emb_dims).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] DGCNN params={param_count:,} device={device}")

    # Run directory
    exp_name = args.exp_name or f"dgcnn_{args.mode}_s{args.seed}_f{args.fold}"
    run_dir = args.run_root / exp_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "pafa":
        discriminator = ProtocolDiscriminator(in_dim=args.emb_dims, hidden=128).to(device)
        disc_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
        print(f"[discriminator] params={disc_params:,}")

        optimizer_seg = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                          weight_decay=args.weight_decay)
        optimizer_disc = torch.optim.AdamW(discriminator.parameters(), lr=args.disc_lr,
                                           weight_decay=args.weight_decay)
        scheduler_seg = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_seg,
                                                                     T_max=args.epochs, eta_min=1e-6)
        scheduler_disc = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_disc,
                                                                      T_max=args.epochs, eta_min=1e-6)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                T_max=args.epochs, eta_min=1e-6)

    best_val_miou = -1.0
    best_epoch = -1
    patience_counter = 0
    history = []

    for epoch in range(args.epochs):
        t0 = time.time()

        if args.mode == "pafa":
            lambd = lambda_schedule(epoch, args.epochs) * args.lambda_max
            train_metrics = train_one_epoch_pafa(
                model, discriminator, train_loader,
                optimizer_seg, optimizer_disc, device,
                class_weights, lambd, use_focal=True)
            scheduler_seg.step()
            scheduler_disc.step()
            train_str = (f"seg_loss={train_metrics['seg_loss']:.4f} "
                        f"disc_loss={train_metrics['disc_loss']:.4f} "
                        f"disc_acc={train_metrics['disc_acc']:.3f} "
                        f"λ={lambd:.3f}")
        else:
            train_metrics = train_one_epoch_mixing(
                model, train_loader, optimizer, device,
                class_weights, use_focal=True)
            scheduler.step()
            train_str = f"loss={train_metrics['loss']:.4f}"
            lambd = 0.0

        val_metrics = evaluate(model, nat_val_loader, device, num_classes, class_weights)
        dt = time.time() - t0

        record = {
            "epoch": epoch,
            "lambd": lambd,
            "train": train_metrics,
            "val_nat": val_metrics,
            "time_s": dt,
        }
        history.append(record)

        val_miou = val_metrics["mean_iou"]
        improved = val_miou > best_val_miou
        if improved:
            best_val_miou = val_miou
            best_epoch = epoch
            patience_counter = 0
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "val_miou": val_miou},
                       run_dir / "ckpt_best.pt")
        else:
            patience_counter += 1

        marker = " *" if improved else ""
        print(f"[{epoch:3d}/{args.epochs}] {train_str} | "
              f"val_nat_miou={val_miou:.4f}{marker} | {dt:.1f}s", flush=True)

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch} (patience={args.patience})")
            break

    # Load best model and evaluate on test sets
    ckpt = torch.load(run_dir / "ckpt_best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])

    test_nat = evaluate(model, nat_test_loader, device, num_classes, class_weights)
    test_bal = evaluate(model, bal_test_loader, device, num_classes, class_weights)

    results = {
        "mode": args.mode,
        "seed": args.seed,
        "fold": args.fold,
        "best_epoch": best_epoch,
        "best_val_miou": best_val_miou,
        "test_nat_miou": test_nat["mean_iou"],
        "test_nat_metrics": test_nat,
        "test_bal_miou": test_bal["mean_iou"],
        "test_bal_metrics": test_bal,
        "lambda_max": args.lambda_max,
        "epochs_trained": len(history),
    }

    write_json(run_dir / "results.json", results)
    write_json(run_dir / "history.json", history)

    print(f"\n[DONE] {args.mode} seed={args.seed} fold={args.fold}")
    print(f"  best_epoch={best_epoch}  val_nat_miou={best_val_miou:.4f}")
    print(f"  test_nat_miou={test_nat['mean_iou']:.4f}  test_bal_miou={test_bal['mean_iou']:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
