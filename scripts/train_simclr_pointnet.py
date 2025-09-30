#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from typing import Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def read_list(p: str) -> List[str]:
    with open(p, 'r', encoding='utf-8') as f:
        return [ln.strip() for ln in f if ln.strip()]


def augment(pc: np.ndarray) -> np.ndarray:
    pts = pc.copy()
    # jitter
    pts += np.random.normal(0, 0.01, size=pts.shape).astype(np.float32)
    # scale
    s = np.random.uniform(0.9, 1.1)
    pts *= s
    # rotate random around z
    ang = np.random.uniform(-np.pi, np.pi)
    c, s = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    pts = (Rz @ pts.T).T
    # small 3D rotation
    ax = np.random.uniform(-np.pi/18, np.pi/18)
    ay = np.random.uniform(-np.pi/18, np.pi/18)
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    pts = (Ry @ (Rx @ pts.T)).T
    return pts.astype(np.float32)


def sample_points(pts: np.ndarray, n: int) -> np.ndarray:
    m = pts.shape[0]
    if m >= n:
        idx = np.random.choice(m, n, replace=False)
    else:
        pad = np.random.choice(m, n - m, replace=True)
        idx = np.concatenate([np.arange(m), pad])
    return pts[idx]


class NPZDataset(Dataset):
    def __init__(self, root: str, split_list: str, num_points: int, two_views: bool = True):
        self.root = root
        self.files = read_list(split_list)
        self.num_points = num_points
        self.two_views = two_views

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i: int):
        rel = self.files[i]
        p = rel if os.path.isabs(rel) else os.path.join(self.root, rel)
        d = np.load(p)
        pts = d['points'].astype(np.float32)
        pts = sample_points(pts, self.num_points)
        if self.two_views:
            v1 = augment(pts)
            v2 = augment(pts)
            return torch.from_numpy(v1), torch.from_numpy(v2)
        else:
            return torch.from_numpy(pts)


class PointNetEncoder(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(True),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU(True),
        )
        self.fc = nn.Sequential(
            nn.Linear(256, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(True),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,N,3] -> [B,3,N]
        x = x.transpose(1, 2)
        x = self.mlp1(x)
        x = torch.max(x, dim=2)[0]  # [B,256]
        x = self.fc(x)               # [B,out_dim]
        return x


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(True),
            nn.Linear(in_dim, proj_dim)
        )

    def forward(self, x):
        return self.net(x)


def info_nce(z1: torch.Tensor, z2: torch.Tensor, temp: float = 0.07) -> torch.Tensor:
    # z1,z2: [B,D]
    z1 = F.normalize(z1, dim=1, eps=1e-6)
    z2 = F.normalize(z2, dim=1, eps=1e-6)
    B = z1.size(0)
    reps = torch.cat([z1, z2], dim=0)  # [2B,D]
    sim = reps @ reps.t()               # [2B,2B]
    mask = torch.eye(2*B, dtype=torch.bool, device=reps.device)
    sim = sim / temp
    sim = sim - 1e9 * mask
    targets = torch.cat([torch.arange(B, 2*B), torch.arange(0, B)]).to(reps.device)
    loss = F.cross_entropy(sim, targets)
    return loss


def main(argv: Iterable[str]) -> int:
    ap = argparse.ArgumentParser(description='SimCLR pretraining on point clouds (PointNet)')
    ap.add_argument('--root', default='data_npz', help='NPZ root')
    ap.add_argument('--train_list', default='splits/train.txt')
    ap.add_argument('--val_list', default='splits/val.txt')
    ap.add_argument('--points', type=int, default=2048)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--out', default='outputs/simclr_pointnet')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args(list(argv))

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)

    train_ds = NPZDataset(args.root, args.train_list, args.points, two_views=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    encoder = PointNetEncoder(out_dim=256).to(device)
    proj = ProjectionHead(256, 128).to(device)
    params = list(encoder.parameters()) + list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    best = 1e9
    for epoch in range(1, args.epochs+1):
        encoder.train(); proj.train()
        t0 = time.time()
        losses: List[float] = []
        for v1, v2 in train_loader:
            v1 = v1.to(device).float()
            v2 = v2.to(device).float()
            f1 = encoder(v1)
            f2 = encoder(v2)
            z1 = proj(f1)
            z2 = proj(f2)
            loss = info_nce(z1, z2, temp=0.07)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        avg = float(np.mean(losses)) if losses else 0.0
        dt = time.time() - t0
        print(f'Epoch {epoch}: loss={avg:.4f} time={dt:.1f}s')
        torch.save({'encoder': encoder.state_dict(), 'proj': proj.state_dict()}, os.path.join(args.out, 'ckpt_last.pth'))
        if avg < best:
            best = avg
            torch.save({'encoder': encoder.state_dict(), 'proj': proj.state_dict()}, os.path.join(args.out, 'ckpt_best.pth'))
    print('Done')
    return 0


if __name__ == '__main__':
    import sys
    raise SystemExit(main(sys.argv[1:]))
