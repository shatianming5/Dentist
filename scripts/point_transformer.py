from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from pointnet2 import index_points, knn_point


def knn_indices(xyz: torch.Tensor, k: int) -> torch.Tensor:
    """Return kNN indices for each point in xyz.

    xyz: (B, N, 3)
    returns: (B, N, k) indices into N
    """
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError(f"Expected xyz (B,N,3), got {tuple(xyz.shape)}")
    b, n, _ = xyz.shape
    kk = int(k)
    if kk <= 0 or int(n) <= 1:
        return torch.zeros((b, int(n), 0), dtype=torch.long, device=xyz.device)
    # knn_point includes self as the nearest neighbor when new_xyz == xyz; drop it.
    idx = knn_point(int(kk) + 1, xyz, xyz)  # (B,N,kk+1)
    if idx.shape[-1] <= 1:
        return torch.zeros((b, int(n), 0), dtype=torch.long, device=xyz.device)
    idx = idx[:, :, 1:]
    if idx.shape[-1] > kk:
        idx = idx[:, :, :kk]
    return idx


class PointTransformerBlock(nn.Module):
    def __init__(self, *, dim: int, k: int, ffn_mult: float = 2.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.dim = int(dim)
        self.k = int(k)
        if self.dim <= 0:
            raise ValueError("dim must be >0")
        if self.k <= 0:
            raise ValueError("k must be >0")
        hidden = max(1, int(round(float(ffn_mult) * float(self.dim))))

        self.ln1 = nn.LayerNorm(self.dim)
        self.ln2 = nn.LayerNorm(self.dim)

        self.q = nn.Linear(self.dim, self.dim, bias=False)
        self.kv = nn.Linear(self.dim, self.dim * 2, bias=False)

        self.pos_mlp = nn.Sequential(
            nn.Linear(3, self.dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim, self.dim, bias=False),
        )
        self.attn_mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim, self.dim, bias=False),
        )
        self.proj = nn.Sequential(
            nn.Linear(self.dim, self.dim, bias=False),
            nn.Dropout(p=float(dropout)),
        )

        self.ffn = nn.Sequential(
            nn.Linear(self.dim, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(hidden, self.dim, bias=False),
            nn.Dropout(p=float(dropout)),
        )

    def forward(self, xyz: torch.Tensor, feats: torch.Tensor, *, idx: torch.Tensor | None = None) -> torch.Tensor:
        if xyz.ndim != 3 or xyz.shape[-1] != 3:
            raise ValueError(f"Expected xyz (B,N,3), got {tuple(xyz.shape)}")
        if feats.ndim != 3 or feats.shape[-1] != int(self.dim):
            raise ValueError(f"Expected feats (B,N,{int(self.dim)}), got {tuple(feats.shape)}")
        b, n, _ = xyz.shape
        _ = b
        _ = n
        if idx is None:
            idx = knn_indices(xyz, int(self.k))

        x = self.ln1(feats)
        neigh_xyz = index_points(xyz, idx)  # (B,N,K,3)
        neigh_x = index_points(x, idx)  # (B,N,K,C)

        rel = neigh_xyz - xyz[:, :, None, :]  # (B,N,K,3)
        pos = self.pos_mlp(rel)  # (B,N,K,C)

        q = self.q(x).unsqueeze(2)  # (B,N,1,C)
        kv = self.kv(neigh_x)  # (B,N,K,2C)
        k, v = torch.split(kv, int(self.dim), dim=-1)  # (B,N,K,C)

        attn = self.attn_mlp(q - k + pos)  # (B,N,K,C)
        attn = torch.softmax(attn, dim=2)
        out = torch.sum(attn * (v + pos), dim=2)  # (B,N,C)
        feats = feats + self.proj(out)
        feats = feats + self.ffn(self.ln2(feats))
        return feats


@dataclass(frozen=True)
class PointTransformerParams:
    dim: int = 64
    depth: int = 4
    k: int = 16
    ffn_mult: float = 2.0


class PointTransformerEncoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        params: PointTransformerParams,
        dropout: float,
    ) -> None:
        super().__init__()
        in_ch = int(in_channels)
        if in_ch < 3:
            raise ValueError("in_channels must be >=3")
        p = params
        dim = int(p.dim)
        if dim <= 0:
            raise ValueError("params.dim must be >0")
        depth = int(p.depth)
        if depth <= 0:
            raise ValueError("params.depth must be >0")

        self.in_channels = int(in_ch)
        self.dim = int(dim)
        self.depth = int(depth)
        self.k = int(p.k)

        self.stem = nn.Sequential(
            nn.Linear(int(in_ch), int(dim), bias=False),
            nn.LayerNorm(int(dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(dim), int(dim), bias=False),
            nn.LayerNorm(int(dim)),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList(
            [
                PointTransformerBlock(dim=int(dim), k=int(p.k), ffn_mult=float(p.ffn_mult), dropout=float(dropout))
                for _ in range(int(depth))
            ]
        )
        self.norm = nn.LayerNorm(int(dim))

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3 or points.shape[-1] != int(self.in_channels):
            raise ValueError(f"Expected points (B,N,{int(self.in_channels)}), got {tuple(points.shape)}")
        xyz = points[:, :, :3].contiguous()
        idx = knn_indices(xyz, int(self.k))

        x = self.stem(points)
        for blk in self.blocks:
            x = blk(xyz, x, idx=idx)
        x = self.norm(x)

        x_mean = x.mean(dim=1)
        x_max = torch.max(x, dim=1).values
        return torch.cat([x_mean, x_max], dim=1)


class PointTransformerClassifier(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        dropout: float,
        extra_dim: int = 0,
        in_channels: int = 3,
        params: PointTransformerParams | None = None,
    ) -> None:
        super().__init__()
        p = params or PointTransformerParams()
        self.in_channels = int(in_channels)
        self.feat = PointTransformerEncoder(in_channels=int(in_channels), params=p, dropout=float(dropout))
        in_dim = int(p.dim) * 2 + int(extra_dim)
        self.feature_dim = int(in_dim)
        self.head = nn.Sequential(
            nn.Linear(in_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(256, int(num_classes)),
        )

    def forward_features(
        self,
        points: torch.Tensor,
        extra: torch.Tensor | None = None,
        domains: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = domains
        if points.ndim != 3 or points.shape[-1] != int(self.in_channels):
            raise ValueError(f"Expected points (B,N,{int(self.in_channels)}), got {tuple(points.shape)}")
        x = self.feat(points)
        if extra is not None and extra.numel() > 0:
            x = torch.cat([x, extra], dim=1)
        return x

    def forward_logits_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)

    def forward(
        self,
        points: torch.Tensor,
        extra: torch.Tensor | None = None,
        domains: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_logits_from_features(self.forward_features(points, extra, domains=domains))

