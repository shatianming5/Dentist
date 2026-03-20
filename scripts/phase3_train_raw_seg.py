#!/usr/bin/env python3
"""Phase 3: raw point-cloud segmentation (binary: background vs restoration)."""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import time
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from _lib.device import normalize_device
from _lib.env import get_env_info
from _lib.git import get_git_info
from _lib.io import read_json, read_jsonl, write_json
from _lib.point_ops import get_graph_feature
from _lib.seed import set_seed
from _lib.time import utc_now_iso
from pointmlp import PointMLPBlock, knn_indices

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class PointNetSeg(nn.Module):
    """PointNet segmentation: per-point + global feature concatenation."""
    def __init__(self, num_classes: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        # Shared MLP per-point feature extraction
        self.conv1 = nn.Sequential(nn.Conv1d(3, 64, 1, bias=False), nn.BatchNorm1d(64), nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1, bias=False), nn.BatchNorm1d(128), nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 256, 1, bias=False), nn.BatchNorm1d(256), nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.Conv1d(256, 512, 1, bias=False), nn.BatchNorm1d(512), nn.ReLU(True))
        # Segmentation head: per-point(128) + global(512) = 640
        self.seg_head = nn.Sequential(
            nn.Conv1d(128 + 512, 256, 1, bias=False), nn.BatchNorm1d(256), nn.ReLU(True),
            nn.Dropout(p=float(dropout)),
            nn.Conv1d(256, 128, 1, bias=False), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Dropout(p=float(dropout)),
            nn.Conv1d(128, int(num_classes), 1),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        # points: (B, N, 3)
        x = points.transpose(1, 2).contiguous()  # (B, 3, N)
        x1 = self.conv1(x)    # (B, 64, N)
        x2 = self.conv2(x1)   # (B, 128, N) -- local features
        x3 = self.conv3(x2)   # (B, 256, N)
        x4 = self.conv4(x3)   # (B, 512, N)
        g = torch.max(x4, dim=2, keepdim=True).values  # (B, 512, 1)
        g = g.expand(-1, -1, x.shape[2])               # (B, 512, N)
        feat = torch.cat([x2, g], dim=1)                # (B, 640, N)
        return self.seg_head(feat)                       # (B, C, N)


class DGCNNv2Seg(nn.Module):
    """DGCNN segmentation with edge convolution and per-point output."""
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
        # seg head: per-point(64+64+128+256=512) + global(emb_dims) tiled
        seg_in = 512 + int(emb_dims)
        self.seg_head = nn.Sequential(
            nn.Conv1d(seg_in, 256, 1, bias=False), nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, True), nn.Dropout(float(dropout)),
            nn.Conv1d(256, 128, 1, bias=False), nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, True), nn.Dropout(float(dropout)),
            nn.Conv1d(128, int(num_classes), 1),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        x = points.transpose(1, 2).contiguous()  # (B, 3, N)
        N = x.shape[2]
        x1 = self.conv1(get_graph_feature(x, k=self.k)).max(dim=-1).values  # (B,64,N)
        x2 = self.conv2(get_graph_feature(x1, k=self.k)).max(dim=-1).values # (B,64,N)
        x3 = self.conv3(get_graph_feature(x2, k=self.k)).max(dim=-1).values # (B,128,N)
        x4 = self.conv4(get_graph_feature(x3, k=self.k)).max(dim=-1).values # (B,256,N)
        x_cat = torch.cat([x1, x2, x3, x4], dim=1)  # (B, 512, N)
        x5 = self.conv5(x_cat)  # (B, emb_dims, N)
        g = torch.max(x5, dim=2, keepdim=True).values.expand(-1, -1, N)  # (B, emb_dims, N)
        feat = torch.cat([x_cat, g], dim=1)  # (B, 512+emb_dims, N)
        return self.seg_head(feat)  # (B, C, N)


class PointNet2Seg(nn.Module):
    """PointNet++ segmentation with set abstraction + feature propagation."""
    def __init__(self, num_classes: int = 2, dropout: float = 0.3,
                 sa1_npoint: int = 1024, sa1_nsample: int = 32,
                 sa2_npoint: int = 256, sa2_nsample: int = 64) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        from pointnet2 import PointNet2SetAbstraction, square_distance
        # Encoder (downsampling)
        self.sa1 = PointNet2SetAbstraction(npoint=sa1_npoint, nsample=sa1_nsample,
                                            in_channel=3, mlp=(64, 64, 128), group_all=False)
        self.sa2 = PointNet2SetAbstraction(npoint=sa2_npoint, nsample=sa2_nsample,
                                            in_channel=3 + 128, mlp=(128, 128, 256), group_all=False)
        self.sa3 = PointNet2SetAbstraction(npoint=0, nsample=0,
                                            in_channel=3 + 256, mlp=(256, 512, 1024), group_all=True)
        # Decoder (feature propagation via interpolation)
        self.fp3 = _FPModule(in_channel=1024 + 256, mlp=[256, 256])
        self.fp2 = _FPModule(in_channel=256 + 128, mlp=[256, 128])
        self.fp1 = _FPModule(in_channel=128, mlp=[128, 128, 128])
        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv1d(128, 128, 1, bias=False), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Dropout(float(dropout)),
            nn.Conv1d(128, int(num_classes), 1),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        # points: (B, N, 3)
        from pointnet2 import square_distance
        xyz = points  # (B, N, 3)
        l0_xyz, l0_points = xyz, None
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Propagate back
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        # l0_points: (B, N, 128)
        x = l0_points.transpose(1, 2).contiguous()  # (B, 128, N)
        return self.seg_head(x)  # (B, C, N)


class _FPModule(nn.Module):
    """Feature propagation module for PointNet++ segmentation."""
    def __init__(self, in_channel: int, mlp: list[int]) -> None:
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last = in_channel
        for out_ch in mlp:
            self.mlp_convs.append(nn.Conv1d(last, out_ch, 1, bias=False))
            self.mlp_bns.append(nn.BatchNorm1d(out_ch))
            last = out_ch

    def forward(self, xyz1: torch.Tensor, xyz2: torch.Tensor,
                points1: torch.Tensor | None, points2: torch.Tensor) -> torch.Tensor:
        """Interpolate features from xyz2 to xyz1 using 3-NN inverse-distance weighting."""
        from pointnet2 import square_distance
        B, N, _ = xyz1.shape
        _, S, D2 = points2.shape

        if S == 1:
            interpolated = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)  # (B, N, S)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # 3 nearest
            dist_recip = 1.0 / (dists + 1e-8)
            norm = dist_recip.sum(dim=-1, keepdim=True)
            weight = dist_recip / norm  # (B, N, 3)
            # Memory-efficient gather: index into points2 per batch
            idx_flat = idx.reshape(B, -1)  # (B, N*3)
            gathered = torch.gather(points2, 1, idx_flat.unsqueeze(-1).expand(-1, -1, D2))  # (B, N*3, D2)
            gathered = gathered.reshape(B, N, 3, D2)  # (B, N, 3, D2)
            interpolated = (gathered * weight.unsqueeze(-1)).sum(dim=2)  # (B, N, D2)

        if points1 is not None:
            new_points = torch.cat([points1, interpolated], dim=-1)
        else:
            new_points = interpolated

        # MLP
        x = new_points.transpose(1, 2).contiguous()  # (B, C, N)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            x = F.relu(bn(conv(x)), inplace=True)
        return x.transpose(1, 2).contiguous()  # (B, N, C_out)


class PointTransformerSeg(nn.Module):
    """Lightweight Point Transformer for segmentation."""
    def __init__(self, num_classes: int = 2, dropout: float = 0.3,
                 dim: int = 96, depth: int = 4, k: int = 16, ffn_mult: float = 2.0) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.k = int(k)
        self.dim = int(dim)
        self.embed = nn.Sequential(
            nn.Conv1d(3, int(dim), 1, bias=False), nn.BatchNorm1d(int(dim)), nn.ReLU(True))
        self.blocks = nn.ModuleList()
        for _ in range(int(depth)):
            self.blocks.append(_PTBlock(int(dim), int(k), float(ffn_mult)))
        self.seg_head = nn.Sequential(
            nn.Conv1d(int(dim), 128, 1, bias=False), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Dropout(float(dropout)),
            nn.Conv1d(128, int(num_classes), 1),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        x = points.transpose(1, 2).contiguous()  # (B, 3, N)
        x = self.embed(x)  # (B, dim, N)
        for blk in self.blocks:
            x = blk(x, points)
        return self.seg_head(x)  # (B, C, N)


class _PTBlock(nn.Module):
    """Point Transformer block with kNN attention."""
    def __init__(self, dim: int, k: int, ffn_mult: float) -> None:
        super().__init__()
        self.k = int(k)
        self.q = nn.Conv1d(dim, dim, 1, bias=False)
        self.k_proj = nn.Conv1d(dim, dim, 1, bias=False)
        self.v = nn.Conv1d(dim, dim, 1, bias=False)
        self.pos_enc = nn.Sequential(
            nn.Conv2d(3, dim, 1, bias=False), nn.BatchNorm2d(dim), nn.ReLU(True),
            nn.Conv2d(dim, dim, 1, bias=False))
        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False), nn.BatchNorm2d(dim), nn.ReLU(True),
            nn.Conv2d(dim, dim, 1, bias=False))
        self.norm1 = nn.BatchNorm1d(dim)
        hid = int(dim * ffn_mult)
        self.ffn = nn.Sequential(
            nn.Conv1d(dim, hid, 1, bias=False), nn.BatchNorm1d(hid), nn.ReLU(True),
            nn.Conv1d(hid, dim, 1, bias=False))
        self.norm2 = nn.BatchNorm1d(dim)

    def forward(self, x: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        # x: (B, D, N), points: (B, N, 3)
        B, D, N = x.shape
        xyz = points.transpose(1, 2).contiguous()  # (B, 3, N)
        # kNN
        with torch.no_grad():
            dist = torch.cdist(points, points)  # (B, N, N)
            _, idx = dist.topk(self.k, dim=-1, largest=False)  # (B, N, k)
        idx_flat = idx.reshape(B, -1)  # (B, N*k)
        # gather
        q = self.q(x)   # (B, D, N)
        k_ = self.k_proj(x)
        v_ = self.v(x)
        # expand idx for gathering
        idx_e = idx_flat.unsqueeze(1).expand(-1, D, -1)  # (B, D, N*k)
        k_nn = k_.gather(2, idx_e).reshape(B, D, N, self.k)  # (B, D, N, k)
        v_nn = v_.gather(2, idx_e).reshape(B, D, N, self.k)
        # positional encoding
        xyz_e = idx_flat.unsqueeze(1).expand(-1, 3, -1)
        xyz_nn = xyz.gather(2, xyz_e).reshape(B, 3, N, self.k)
        pos_diff = xyz.unsqueeze(3) - xyz_nn  # (B, 3, N, k)
        pe = self.pos_enc(pos_diff)  # (B, D, N, k)
        # attention
        attn = self.attn_mlp(q.unsqueeze(3) - k_nn + pe)  # (B, D, N, k)
        attn = torch.softmax(attn, dim=-1)
        out = (attn * (v_nn + pe)).sum(dim=-1)  # (B, D, N)
        x = self.norm1(x + out)
        x = self.norm2(x + self.ffn(x))
        return x


class PointMLPSeg(nn.Module):
    """PointMLP segmentation: residual MLP blocks with geometric kNN message passing."""
    def __init__(self, num_classes: int = 2, dropout: float = 0.3,
                 dim: int = 64, depth: int = 4, k: int = 16, ffn_mult: float = 2.0) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.k = int(k)
        self.dim = int(dim)
        self.depth = int(depth)
        # Stem: project xyz to feature dim
        self.stem = nn.Sequential(
            nn.Linear(3, int(dim), bias=False),
            nn.LayerNorm(int(dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(dim), int(dim), bias=False),
            nn.LayerNorm(int(dim)),
            nn.ReLU(inplace=True),
        )
        # Encoder blocks
        self.blocks = nn.ModuleList([
            PointMLPBlock(dim=int(dim), k=int(k), ffn_mult=float(ffn_mult), dropout=float(dropout))
            for _ in range(int(depth))
        ])
        self.norm = nn.LayerNorm(int(dim))
        # Segmentation head: early features (dim) + late features (dim) + global (dim)
        seg_in = int(dim) * 3
        self.seg_head = nn.Sequential(
            nn.Conv1d(seg_in, 128, 1, bias=False), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Dropout(float(dropout)),
            nn.Conv1d(128, 64, 1, bias=False), nn.BatchNorm1d(64), nn.ReLU(True),
            nn.Dropout(float(dropout)),
            nn.Conv1d(64, int(num_classes), 1),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        # points: (B, N, 3)
        B, N, _ = points.shape
        xyz = points.contiguous()
        idx = knn_indices(xyz, self.k)
        x = self.stem(points)  # (B, N, dim)
        # Save early features after first block
        mid = self.depth // 2
        early_feat = x
        for i, blk in enumerate(self.blocks):
            x = blk(xyz, x, idx=idx)
            if i == mid - 1:
                early_feat = x
        x = self.norm(x)  # (B, N, dim)
        # Global feature tiled to each point
        g = torch.max(x, dim=1, keepdim=True).values.expand(-1, N, -1)  # (B, N, dim)
        # Concat: early + late + global
        feat = torch.cat([early_feat, x, g], dim=-1)  # (B, N, dim*3)
        feat = feat.transpose(1, 2).contiguous()  # (B, dim*3, N)
        return self.seg_head(feat)  # (B, C, N)


class CurveGrouping(nn.Module):
    """Walk-based curve grouping: greedily extend curves through kNN graph."""
    def __init__(self, k: int = 16, curve_len: int = 4):
        super().__init__()
        self.k = k
        self.curve_len = curve_len

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """Return curve indices (B, N, curve_len) via greedy random walk."""
        B, N, _ = xyz.shape
        with torch.no_grad():
            dist = torch.cdist(xyz, xyz)
            _, knn_idx = dist.topk(self.k, dim=-1, largest=False)  # (B, N, k)
        curves = [torch.arange(N, device=xyz.device).unsqueeze(0).expand(B, -1)]  # start: self
        current = curves[0]  # (B, N)
        for _ in range(self.curve_len - 1):
            # For each point, pick a random neighbor from kNN
            rand_k = torch.randint(1, self.k, (B, N), device=xyz.device)  # skip self (index 0)
            batch_idx = torch.arange(B, device=xyz.device).unsqueeze(1).expand(-1, N)
            point_idx = current
            next_pt = knn_idx[batch_idx, point_idx, rand_k]  # (B, N)
            curves.append(next_pt)
            current = next_pt
        return torch.stack(curves, dim=-1)  # (B, N, curve_len)


class CurveAggBlock(nn.Module):
    """Curve-based feature aggregation block."""
    def __init__(self, dim: int, curve_len: int = 4, k: int = 16, dropout: float = 0.3):
        super().__init__()
        self.grouping = CurveGrouping(k=k, curve_len=curve_len)
        self.curve_mlp = nn.Sequential(
            nn.Conv2d(dim + 3, dim, 1, bias=False), nn.BatchNorm2d(dim), nn.ReLU(True),
            nn.Conv2d(dim, dim, 1, bias=False), nn.BatchNorm2d(dim), nn.ReLU(True),
        )
        self.norm = nn.BatchNorm1d(dim)
        hid = dim * 2
        self.ffn = nn.Sequential(
            nn.Conv1d(dim, hid, 1, bias=False), nn.BatchNorm1d(hid), nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Conv1d(hid, dim, 1, bias=False),
        )
        self.norm2 = nn.BatchNorm1d(dim)

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """xyz: (B,N,3), feat: (B,D,N) -> (B,D,N)"""
        B, D, N = feat.shape
        curves = self.grouping(xyz)  # (B, N, L)
        L = curves.shape[-1]
        # Gather curve features
        curves_flat = curves.reshape(B, -1)  # (B, N*L)
        idx_e = curves_flat.unsqueeze(1).expand(-1, D, -1)
        curve_feat = feat.gather(2, idx_e).reshape(B, D, N, L)  # (B, D, N, L)
        # Gather curve xyz for positional info
        xyz_t = xyz.transpose(1, 2)  # (B, 3, N)
        idx_xyz = curves_flat.unsqueeze(1).expand(-1, 3, -1)
        curve_xyz = xyz_t.gather(2, idx_xyz).reshape(B, 3, N, L)
        rel_xyz = curve_xyz - xyz_t.unsqueeze(3)  # (B, 3, N, L)
        # Combine
        combined = torch.cat([curve_feat, rel_xyz], dim=1)  # (B, D+3, N, L)
        agg = self.curve_mlp(combined).max(dim=-1).values  # (B, D, N)
        feat = self.norm(feat + agg)
        feat = self.norm2(feat + self.ffn(feat))
        return feat


class CurveNetSeg(nn.Module):
    """CurveNet-inspired segmentation: curve-based point cloud learning."""
    def __init__(self, num_classes: int = 2, dropout: float = 0.3,
                 dim: int = 64, depth: int = 4, k: int = 16, curve_len: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(3, dim, 1, bias=False), nn.BatchNorm1d(dim), nn.ReLU(True),
            nn.Conv1d(dim, dim, 1, bias=False), nn.BatchNorm1d(dim), nn.ReLU(True),
        )
        self.blocks = nn.ModuleList([
            CurveAggBlock(dim=dim, curve_len=curve_len, k=k, dropout=dropout)
            for _ in range(depth)
        ])
        seg_in = dim * 3  # early + late + global
        self.seg_head = nn.Sequential(
            nn.Conv1d(seg_in, 128, 1, bias=False), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Conv1d(128, 64, 1, bias=False), nn.BatchNorm1d(64), nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Conv1d(64, num_classes, 1),
        )
        self.depth = depth

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        B, N, _ = points.shape
        xyz = points.contiguous()
        x = self.stem(points.transpose(1, 2))  # (B, D, N)
        mid = self.depth // 2
        early = x
        for i, blk in enumerate(self.blocks):
            x = blk(xyz, x)
            if i == mid - 1:
                early = x
        g = torch.max(x, dim=2, keepdim=True).values.expand_as(x)
        feat = torch.cat([early, x, g], dim=1)  # (B, 3D, N)
        return self.seg_head(feat)  # (B, C, N)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RawSegDataset(Dataset):
    def __init__(self, data_root: Path, index_rows: list[dict], n_points: int,
                 augment: bool = False) -> None:
        self.data_root = Path(data_root)
        self.rows = list(index_rows)
        self.n_points = int(n_points)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        npz_path = self.data_root / row["sample_npz"]
        data = np.load(str(npz_path))
        pts = data["points"].astype(np.float32)   # (M, 3)
        lbl = data["labels"].astype(np.int64)      # (M,)
        # Resample to n_points
        if len(pts) >= self.n_points:
            choice = np.random.choice(len(pts), self.n_points, replace=False)
        else:
            choice = np.random.choice(len(pts), self.n_points, replace=True)
        pts = pts[choice]
        lbl = lbl[choice]
        if self.augment:
            pts = self._augment(pts)
        return {"points": torch.from_numpy(pts), "labels": torch.from_numpy(lbl),
                "case_key": row["case_key"]}

    @staticmethod
    def _augment(pts: np.ndarray) -> np.ndarray:
        # Random rotation around Z
        theta = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        pts = pts @ R.T
        # Random scale
        scale = np.random.uniform(0.8, 1.2)
        pts = pts * scale
        # Jitter
        pts = pts + np.random.normal(0, 0.01, pts.shape).astype(np.float32)
        return pts


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_seg_metrics(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> dict:
    """Compute IoU, accuracy, precision, recall, F1 per class."""
    acc = float(np.mean(pred == gt))
    ious, precisions, recalls, f1s = [], [], [], []
    for c in range(num_classes):
        tp = int(np.sum((pred == c) & (gt == c)))
        fp = int(np.sum((pred == c) & (gt != c)))
        fn = int(np.sum((pred != c) & (gt == c)))
        iou = tp / max(tp + fp + fn, 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        ious.append(iou)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    return {
        "accuracy": acc,
        "mean_iou": float(np.mean(ious)),
        "per_class_iou": ious,
        "per_class_precision": precisions,
        "per_class_recall": recalls,
        "per_class_f1": f1s,
    }


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

def focal_loss(logits: torch.Tensor, targets: torch.Tensor,
               alpha: torch.Tensor | None = None, gamma: float = 2.0) -> torch.Tensor:
    """Focal loss for segmentation: reduces contribution of easy examples."""
    ce = F.cross_entropy(logits, targets, weight=alpha, reduction='none')
    pt = torch.exp(-ce)
    loss = ((1 - pt) ** gamma) * ce
    return loss.mean()


# ---------------------------------------------------------------------------
# Train / Eval
# ---------------------------------------------------------------------------

def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    device: torch.device, class_weights: torch.Tensor | None = None,
                    use_focal: bool = False, focal_gamma: float = 2.0) -> dict:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_points = 0
    for batch in loader:
        pts = batch["points"].to(device)       # (B, N, 3)
        lbl = batch["labels"].to(device)        # (B, N)
        logits = model(pts)                      # (B, C, N)
        if use_focal:
            loss = focal_loss(logits, lbl, alpha=class_weights, gamma=focal_gamma)
        else:
            loss = F.cross_entropy(logits, lbl, weight=class_weights)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * pts.shape[0]
        pred = logits.argmax(dim=1)
        total_correct += int((pred == lbl).sum().item())
        total_points += int(lbl.numel())
    n = max(len(loader.dataset), 1)
    return {"loss": total_loss / n, "accuracy": total_correct / max(total_points, 1)}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             num_classes: int, class_weights: torch.Tensor | None = None) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    for batch in loader:
        pts = batch["points"].to(device)
        lbl = batch["labels"].to(device)
        logits = model(pts)
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

def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 3: raw point-cloud segmentation.")
    ap.add_argument("--data-root", type=Path, default=Path("processed/raw_seg/v1"))
    ap.add_argument("--run-root", type=Path, default=Path("runs/raw_seg"))
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--model", choices=["pointnet_seg", "dgcnn", "dgcnn_v2", "point_transformer", "pointnet2", "pointmlp_seg", "curvenet_seg"], default="pointnet_seg")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--n-points", type=int, default=8192)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--dgcnn-k", type=int, default=20)
    ap.add_argument("--dgcnn-emb-dims", type=int, default=512)
    ap.add_argument("--pt-dim", type=int, default=96)
    ap.add_argument("--pt-depth", type=int, default=4)
    ap.add_argument("--pt-k", type=int, default=16)
    ap.add_argument("--pt-ffn-mult", type=float, default=2.0)
    ap.add_argument("--pmlp-dim", type=int, default=64)
    ap.add_argument("--pmlp-depth", type=int, default=4)
    ap.add_argument("--pmlp-k", type=int, default=16)
    ap.add_argument("--pmlp-ffn-mult", type=float, default=2.0)
    ap.add_argument("--kfold", type=Path, default=None)
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--val-fold", type=int, default=-1)
    ap.add_argument("--exp-name", type=str, default="")
    ap.add_argument("--no-save-ckpt", action="store_true", help="Do not save checkpoints.")
    ap.add_argument("--focal-loss", action="store_true", help="Use focal loss instead of CE.")
    ap.add_argument("--focal-gamma", type=float, default=2.0, help="Focal loss gamma.")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(normalize_device(args.device))
    data_root = args.data_root.resolve()
    label_map = read_json(data_root / "label_map.json")
    num_classes = len(label_map)
    index_rows = read_jsonl(data_root / "index.jsonl")

    # K-fold split
    if args.kfold is not None:
        kfold_obj = read_json(args.kfold.resolve())
        k = int(kfold_obj["k"])
        test_fold = int(args.fold)
        val_fold = int(args.val_fold) if args.val_fold >= 0 else (test_fold + 1) % k
        c2f = kfold_obj["case_to_fold"]
        for row in index_rows:
            ck = row["case_key"]
            f = int(c2f.get(ck, -1))
            if f == test_fold:
                row["split"] = "test"
            elif f == val_fold:
                row["split"] = "val"
            else:
                row["split"] = "train"

    train_rows = [r for r in index_rows if r.get("split") == "train"]
    val_rows   = [r for r in index_rows if r.get("split") == "val"]
    test_rows  = [r for r in index_rows if r.get("split") == "test"]
    print(f"[data] train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}", flush=True)

    # Build model
    if args.model == "pointnet_seg":
        model = PointNetSeg(num_classes=num_classes, dropout=args.dropout)
    elif args.model in ("dgcnn", "dgcnn_v2"):
        model = DGCNNv2Seg(num_classes=num_classes, dropout=args.dropout,
                           k=args.dgcnn_k, emb_dims=args.dgcnn_emb_dims)
    elif args.model == "point_transformer":
        model = PointTransformerSeg(num_classes=num_classes, dropout=args.dropout,
                                    dim=args.pt_dim, depth=args.pt_depth,
                                    k=args.pt_k, ffn_mult=args.pt_ffn_mult)
    elif args.model == "pointnet2":
        model = PointNet2Seg(num_classes=num_classes, dropout=args.dropout)
    elif args.model == "pointmlp_seg":
        model = PointMLPSeg(num_classes=num_classes, dropout=args.dropout,
                            dim=args.pmlp_dim, depth=args.pmlp_depth,
                            k=args.pmlp_k, ffn_mult=args.pmlp_ffn_mult)
    elif args.model == "curvenet_seg":
        model = CurveNetSeg(num_classes=num_classes, dropout=args.dropout,
                            dim=args.pmlp_dim, depth=args.pmlp_depth,
                            k=args.pmlp_k, curve_len=4)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    model = model.to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {args.model}  params={param_count:,}  device={device}", flush=True)

    # Datasets
    train_ds = RawSegDataset(data_root, train_rows, args.n_points, augment=True)
    val_ds   = RawSegDataset(data_root, val_rows, args.n_points, augment=False)
    test_ds  = RawSegDataset(data_root, test_rows, args.n_points, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # Class weights (inverse frequency from training labels)
    class_weights = None
    train_label_counts = Counter()
    for row in train_rows:
        d = np.load(str(data_root / row["sample_npz"]))
        unique, counts = np.unique(d["labels"], return_counts=True)
        for u, c in zip(unique, counts):
            train_label_counts[int(u)] += int(c)
    if train_label_counts:
        total = sum(train_label_counts.values())
        w = [total / max(train_label_counts.get(c, 1) * num_classes, 1) for c in range(num_classes)]
        class_weights = torch.tensor(w, dtype=torch.float32, device=device)
        print(f"[weights] {class_weights.tolist()}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Run directory
    fold_str = f"_fold{args.fold}" if args.fold >= 0 else ""
    exp_name = args.exp_name or f"{args.model}_s{args.seed}{fold_str}"
    run_dir = Path(args.run_root) / exp_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save train config
    cfg = {
        "generated_at": utc_now_iso(),
        "model": args.model,
        "num_classes": num_classes,
        "n_points": args.n_points,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "model_dgcnn_k": args.dgcnn_k,
        "model_dgcnn_emb_dims": args.dgcnn_emb_dims,
        "model_pt_dim": args.pt_dim,
        "model_pt_depth": args.pt_depth,
        "model_pt_k": args.pt_k,
        "model_pt_ffn_mult": args.pt_ffn_mult,
        "kfold": str(args.kfold) if args.kfold else "",
        "fold": args.fold,
        "data_root": str(data_root),
        "param_count": param_count,
    }
    write_json(run_dir / "train_config.json", cfg)

    # Training loop
    best_val_iou = -1.0
    patience_counter = 0
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, class_weights,
                                        use_focal=args.focal_loss, focal_gamma=args.focal_gamma)
        val_metrics = evaluate(model, val_loader, device, num_classes, class_weights)
        scheduler.step()
        dt = time.time() - t0

        val_iou = val_metrics["mean_iou"]
        improved = val_iou > best_val_iou + 1e-6
        if improved:
            best_val_iou = val_iou
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        row = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 5),
            "train_acc": round(train_metrics["accuracy"], 4),
            "val_loss": round(val_metrics["loss"], 5),
            "val_acc": round(val_metrics["accuracy"], 4),
            "val_miou": round(val_metrics["mean_iou"], 4),
            "val_iou_bg": round(val_metrics["per_class_iou"][0], 4),
            "val_iou_res": round(val_metrics["per_class_iou"][1], 4),
            "best_val_miou": round(best_val_iou, 4),
            "lr": round(optimizer.param_groups[0]["lr"], 7),
            "dt": round(dt, 1),
        }
        history.append(row)
        mark = "*" if improved else ""
        print(f"[{epoch:03d}/{args.epochs}] trn_loss={row['train_loss']:.4f} val_mIoU={row['val_miou']:.4f} "
              f"val_acc={row['val_acc']:.4f} best={row['best_val_miou']:.4f} "
              f"lr={row['lr']:.6f} dt={row['dt']:.1f}s {mark}", flush=True)

        if patience_counter >= args.patience:
            print(f"[early stop] no improvement for {args.patience} epochs", flush=True)
            break

    # Save best model
    if best_state is not None and not args.no_save_ckpt:
        ckpt_path = run_dir / "ckpt_best.pt"
        torch.save({"model": best_state, "best_val_miou": best_val_iou}, str(ckpt_path))
        print(f"[saved] {ckpt_path}", flush=True)

    # Final test evaluation
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device, num_classes, class_weights)
    print(f"\n[TEST] acc={test_metrics['accuracy']:.4f} mIoU={test_metrics['mean_iou']:.4f} "
          f"IoU_bg={test_metrics['per_class_iou'][0]:.4f} IoU_res={test_metrics['per_class_iou'][1]:.4f}", flush=True)
    print(f"  Precision: bg={test_metrics['per_class_precision'][0]:.4f} res={test_metrics['per_class_precision'][1]:.4f}")
    print(f"  Recall:    bg={test_metrics['per_class_recall'][0]:.4f} res={test_metrics['per_class_recall'][1]:.4f}")
    print(f"  F1:        bg={test_metrics['per_class_f1'][0]:.4f} res={test_metrics['per_class_f1'][1]:.4f}")

    # Save results
    results = {
        "train_config": cfg,
        "best_val_miou": best_val_iou,
        "test_metrics": test_metrics,
        "history": history,
        "env": get_env_info(),
    }
    write_json(run_dir / "results.json", results)
    print(f"\n[done] results saved to {run_dir}/results.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
