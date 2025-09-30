#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import Iterable, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from train_simclr_pointnet import PointNetEncoder, sample_points


def read_list(p: str) -> List[str]:
    with open(p, 'r', encoding='utf-8') as f:
        return [ln.strip() for ln in f if ln.strip()]


class NPZClsDataset(Dataset):
    def __init__(self, root: str, split_list: str, num_points: int):
        self.root = root
        self.files = read_list(split_list)
        # filter samples with valid label
        self.files = [f for f in self.files if self._label(self._path(f)) >= 0]
        self.num_points = num_points

    def _path(self, rel: str) -> str:
        return rel if os.path.isabs(rel) else os.path.join(self.root, rel)

    def _label(self, path: str) -> int:
        d = np.load(path)
        return int(d['label']) if 'label' in d else -1

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i: int):
        rel = self.files[i]
        p = self._path(rel)
        d = np.load(p)
        pts = d['points'].astype(np.float32)
        pts = sample_points(pts, self.num_points)
        y = int(d['label'])
        return torch.from_numpy(pts), torch.tensor(y, dtype=torch.long)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += y.numel()
    return correct / max(1, total)


def main(argv: Iterable[str]) -> int:
    ap = argparse.ArgumentParser(description='Linear probe (Upper vs Lower) with frozen PointNet encoder')
    ap.add_argument('--root', default='data_npz')
    ap.add_argument('--train_list', default='splits/train.txt')
    ap.add_argument('--val_list', default='splits/val.txt')
    ap.add_argument('--points', type=int, default=2048)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--ckpt', default='outputs/simclr_pointnet/ckpt_best.pth')
    ap.add_argument('--out', default='outputs/linear_probe')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args(list(argv))

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)

    # datasets
    train_ds = NPZClsDataset(args.root, args.train_list, args.points)
    val_ds = NPZClsDataset(args.root, args.val_list, args.points)
    # fallback: if no labeled samples in val, split a small chunk from train
    if len(val_ds) == 0 and len(train_ds) > 1:
        k = max(1, len(train_ds)//5)
        idx = list(range(len(train_ds)))
        val_idx = set(idx[:k])
        class Subset(Dataset):
            def __init__(self, base, keep):
                self.base = base; self.keep = list(keep)
            def __len__(self): return len(self.keep)
            def __getitem__(self,i): return self.base[self.keep[i]]
        val_ds = Subset(train_ds, val_idx)
        train_ds = Subset(train_ds, [i for i in idx if i not in val_idx])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f'train samples: {len(train_ds)} | val samples: {len(val_ds)}')

    # model
    encoder = PointNetEncoder(out_dim=256).to(device)
    # load pretrained if exists
    if os.path.isfile(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location='cpu')
        state = ckpt.get('encoder', ckpt)
        missing, unexpected = encoder.load_state_dict(state, strict=False)
        print(f'loaded pretrained: missing={len(missing)} unexpected={len(unexpected)}')
    # freeze encoder
    for p in encoder.parameters():
        p.requires_grad = False

    clf = nn.Linear(256, 2).to(device)

    class Model(nn.Module):
        def __init__(self, enc, head):
            super().__init__()
            self.enc = enc
            self.head = head
        def forward(self, x):
            f = self.enc(x)
            return self.head(f)

    model = Model(encoder, clf).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best = 0.0
    for ep in range(1, args.epochs+1):
        model.train()
        losses = []
        for x,y in train_loader:
            x = x.to(device).float(); y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward(); opt.step()
            losses.append(loss.item())
        acc = evaluate(model, val_loader, device)
        print(f'Epoch {ep}: loss={np.mean(losses):.4f} val_acc={acc:.4f}')
        if acc > best:
            best = acc
            torch.save({'encoder': encoder.state_dict(), 'head': clf.state_dict()}, os.path.join(args.out, 'ckpt_best.pth'))
    print(f'Done. best_val_acc={best:.4f}')
    return 0


if __name__ == '__main__':
    import sys
    raise SystemExit(main(sys.argv[1:]))
