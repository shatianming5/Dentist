"""
Pre-train DGCNN on Teeth3DS tooth/gingiva binary segmentation.
Then supports fine-tuning on our restoration data.
"""
import sys, os, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from phase3_train_raw_seg import DGCNNv2Seg, RawSegDataset, compute_seg_metrics

class Teeth3DSDataset(Dataset):
    def __init__(self, data_root, case_keys, n_points=8192, augment=False):
        self.data_root = Path(data_root)
        index = []
        with open(self.data_root / "index.jsonl") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line.strip())
                    if row["case_key"] in case_keys:
                        index.append(row)
        self.rows = index
        self.n_points = n_points
        self.augment = augment
    
    def __len__(self):
        return len(self.rows)
    
    def __getitem__(self, idx):
        row = self.rows[idx]
        d = np.load(str(self.data_root / row["sample_npz"]))
        pts = d["points"].astype(np.float32)
        lbl = d["labels"].astype(np.int64)
        
        if len(pts) >= self.n_points:
            choice = np.random.choice(len(pts), self.n_points, replace=False)
        else:
            choice = np.random.choice(len(pts), self.n_points, replace=True)
        pts, lbl = pts[choice], lbl[choice]
        
        if self.augment:
            # Random rotation around Z
            theta = np.random.uniform(0, 2 * np.pi)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
            pts = pts @ R.T
            pts = pts * np.random.uniform(0.8, 1.2)
            pts += np.random.normal(0, 0.01, pts.shape).astype(np.float32)
        
        return {"points": torch.from_numpy(pts), "labels": torch.from_numpy(lbl),
                "case_key": row["case_key"]}


def train_epoch(model, loader, optimizer, device, class_weights=None):
    model.train()
    total_loss, n = 0, 0
    for batch in loader:
        pts = batch["points"].to(device)
        lbl = batch["labels"].to(device)
        logits = model(pts)
        loss = F.cross_entropy(logits, lbl, weight=class_weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * pts.shape[0]
        n += pts.shape[0]
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device, class_weights=None):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    for batch in loader:
        pts = batch["points"].to(device)
        lbl = batch["labels"].to(device)
        logits = model(pts)
        loss = F.cross_entropy(logits, lbl, weight=class_weights)
        total_loss += loss.item() * pts.shape[0]
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_labels.append(lbl.cpu().numpy())
    preds = np.concatenate(all_preds).ravel()
    labels = np.concatenate(all_labels).ravel()
    metrics = compute_seg_metrics(preds, labels, 2)
    metrics["loss"] = total_loss / max(len(preds) // 8192, 1)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pretrain", "finetune"], required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--pretrained_ckpt", type=str, default=None)
    parser.add_argument("--finetune_data", type=str, default=None,
                        help="Path to restoration data root for fine-tuning")
    parser.add_argument("--finetune_split", type=str, default=None)
    parser.add_argument("--finetune_fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    model = DGCNNv2Seg(num_classes=2, dropout=0.3, k=20, emb_dims=512).to(device)
    
    if args.mode == "pretrain":
        data_root = Path("processed/teeth3ds_binary/v1")
        with open(data_root / "split_pretrain.json") as f:
            split = json.load(f)
        
        train_ds = Teeth3DSDataset(data_root, set(split["train"]), augment=True)
        val_ds = Teeth3DSDataset(data_root, set(split["val"]), augment=False)
        
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        best_miou = 0
        patience_ctr = 0
        
        print(f"Pre-training DGCNN on Teeth3DS: {len(train_ds)} train, {len(val_ds)} val")
        
        for epoch in range(args.epochs):
            t0 = time.time()
            train_loss = train_epoch(model, train_loader, optimizer, device)
            val_metrics = evaluate(model, val_loader, device)
            scheduler.step()
            elapsed = time.time() - t0
            
            if val_metrics["mean_iou"] > best_miou:
                best_miou = val_metrics["mean_iou"]
                torch.save({"model": model.state_dict(), "epoch": epoch,
                           "best_val_miou": best_miou}, str(out_dir / "ckpt_best.pt"))
                patience_ctr = 0
                marker = " *"
            else:
                patience_ctr += 1
                marker = ""
            
            if epoch % 5 == 0 or marker:
                print(f"  E{epoch:3d} train_loss={train_loss:.4f} val_miou={val_metrics['mean_iou']:.4f} "
                      f"val_loss={val_metrics['loss']:.4f} ({elapsed:.1f}s){marker}")
            
            if patience_ctr >= 15:
                print(f"  Early stopping at epoch {epoch}")
                break
        
        print(f"\nBest val mIoU: {best_miou:.4f}")
        # Save final config
        with open(out_dir / "pretrain_config.json", "w") as f:
            json.dump({"mode": "pretrain", "dataset": "teeth3ds", "epochs": epoch+1,
                       "best_miou": best_miou, "n_train": len(train_ds), "n_val": len(val_ds)}, f, indent=2)
    
    elif args.mode == "finetune":
        # Load pre-trained weights (if available)
        if args.pretrained_ckpt and os.path.exists(args.pretrained_ckpt):
            ckpt = torch.load(args.pretrained_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            print(f"Loaded pretrained weights from {args.pretrained_ckpt} (miou={ckpt.get('best_val_miou', '?')})")
        else:
            print("Training from scratch (no pretrained weights)")
        
        # Load our restoration data
        data_root = Path(args.finetune_data)
        with open(args.finetune_split) as f:
            splits = json.load(f)
        
        index = []
        with open(data_root / "index.jsonl") as f:
            for line in f:
                if line.strip(): index.append(json.loads(line.strip()))
        
        fold_val_cases = set(splits["folds"][str(args.finetune_fold)])
        all_cases = set(r["case_key"] for r in index)
        train_cases = all_cases - fold_val_cases
        
        train_rows = [r for r in index if r["case_key"] in train_cases]
        val_rows = [r for r in index if r["case_key"] in fold_val_cases]
        
        train_ds = RawSegDataset(data_root, train_rows, 8192, augment=True)
        val_ds = RawSegDataset(data_root, val_rows, 8192, augment=False)
        
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        best_miou = 0
        patience_ctr = 0
        
        print(f"Fine-tuning on {len(train_ds)} train, {len(val_ds)} val (fold {args.finetune_fold})")
        
        for epoch in range(100):
            train_loss = train_epoch(model, train_loader, optimizer, device)
            val_metrics = evaluate(model, val_loader, device)
            
            if val_metrics["mean_iou"] > best_miou:
                best_miou = val_metrics["mean_iou"]
                torch.save({"model": model.state_dict(), "best_val_miou": best_miou},
                          str(out_dir / "ckpt_best.pt"))
                patience_ctr = 0
                marker = " *"
            else:
                patience_ctr += 1
                marker = ""
            
            if epoch % 10 == 0 or marker:
                print(f"  E{epoch:3d} train_loss={train_loss:.4f} val_miou={val_metrics['mean_iou']:.4f}{marker}")
            
            if patience_ctr >= 20:
                print(f"  Early stopping at epoch {epoch}")
                break
        
        print(f"\nBest val mIoU: {best_miou:.4f}")
        with open(out_dir / "finetune_results.json", "w") as f:
            json.dump({"best_miou": best_miou, "pretrained": args.pretrained_ckpt is not None,
                       "fold": args.finetune_fold, "seed": args.seed,
                       "protocol": "balanced" if "v1" in str(data_root) else "natural"}, f, indent=2)


if __name__ == "__main__":
    main()
