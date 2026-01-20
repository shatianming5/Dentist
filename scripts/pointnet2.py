from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Pairwise squared distances.

    src: (B, N, 3)
    dst: (B, M, 3)
    returns: (B, N, M)
    """
    if src.ndim != 3 or dst.ndim != 3 or src.shape[0] != dst.shape[0] or src.shape[2] != 3 or dst.shape[2] != 3:
        raise ValueError(f"Expected src/dst (B,*,3), got src={tuple(src.shape)} dst={tuple(dst.shape)}")
    b, n, _ = src.shape
    _b2, m, _ = dst.shape
    src2 = torch.sum(src * src, dim=-1, keepdim=True)  # (B,N,1)
    dst2 = torch.sum(dst * dst, dim=-1, keepdim=True).transpose(2, 1)  # (B,1,M)
    dist = src2 - 2.0 * torch.matmul(src, dst.transpose(2, 1)) + dst2  # (B,N,M)
    return dist


@torch.no_grad()
def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Farthest point sampling indices.

    xyz: (B, N, 3)
    returns: (B, npoint)
    """
    if xyz.ndim != 3 or xyz.shape[2] != 3:
        raise ValueError(f"Expected xyz (B,N,3), got {tuple(xyz.shape)}")
    b, n, _ = xyz.shape
    npoint0 = max(1, min(int(npoint), int(n)))
    centroids = torch.zeros((b, npoint0), dtype=torch.long, device=xyz.device)
    distance = torch.full((b, n), float("inf"), device=xyz.device, dtype=xyz.dtype)
    farthest = torch.zeros((b,), dtype=torch.long, device=xyz.device)
    batch_indices = torch.arange(b, dtype=torch.long, device=xyz.device)
    for i in range(npoint0):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(b, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1).indices
    return centroids


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather points by idx.

    points: (B, N, C)
    idx: (B, S) or (B, S, K)
    returns: (B, S, C) or (B, S, K, C)
    """
    if points.ndim != 3:
        raise ValueError(f"Expected points (B,N,C), got {tuple(points.shape)}")
    if idx.ndim not in {2, 3}:
        raise ValueError(f"Expected idx (B,S) or (B,S,K), got {tuple(idx.shape)}")
    b = int(points.shape[0])
    batch_indices = torch.arange(b, dtype=torch.long, device=points.device)
    if idx.ndim == 2:
        return points[batch_indices[:, None], idx, :]
    return points[batch_indices[:, None, None], idx, :]


@torch.no_grad()
def knn_point(k: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """kNN search for each query in new_xyz over xyz.

    xyz: (B, N, 3)
    new_xyz: (B, S, 3)
    returns: idx (B, S, k)
    """
    if xyz.ndim != 3 or new_xyz.ndim != 3 or xyz.shape[0] != new_xyz.shape[0] or xyz.shape[2] != 3 or new_xyz.shape[2] != 3:
        raise ValueError(f"Expected xyz/new_xyz (B,*,3), got xyz={tuple(xyz.shape)} new_xyz={tuple(new_xyz.shape)}")
    b, n, _ = xyz.shape
    _b2, s, _ = new_xyz.shape
    if n <= 1:
        return torch.zeros((b, s, 0), dtype=torch.long, device=xyz.device)
    kk = max(1, min(int(k), int(n)))
    xyz_dist = xyz
    new_xyz_dist = new_xyz
    if xyz.device.type == "cuda" and xyz.dtype == torch.float32:
        xyz_dist = xyz.to(dtype=torch.float16)
        new_xyz_dist = new_xyz.to(dtype=torch.float16)
    dist = square_distance(new_xyz_dist, xyz_dist)  # (B,S,N)
    idx = dist.topk(k=kk, dim=-1, largest=False).indices
    return idx.to(dtype=torch.long)


def sample_and_group(
    npoint: int,
    nsample: int,
    xyz: torch.Tensor,
    points: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FPS + kNN grouping."""
    if xyz.ndim != 3 or xyz.shape[2] != 3:
        raise ValueError(f"Expected xyz (B,N,3), got {tuple(xyz.shape)}")
    b, n, _ = xyz.shape
    npoint0 = max(1, min(int(npoint), int(n)))
    fps_idx = farthest_point_sample(xyz, npoint0)  # (B, npoint)
    new_xyz = index_points(xyz, fps_idx)  # (B, npoint, 3)
    idx = knn_point(int(nsample), xyz, new_xyz)  # (B, npoint, nsample)
    grouped_xyz = index_points(xyz, idx)  # (B, npoint, nsample, 3)
    grouped_xyz_norm = grouped_xyz - new_xyz[:, :, None, :]
    if points is None:
        new_points = grouped_xyz_norm
    else:
        grouped_points = index_points(points, idx)  # (B, npoint, nsample, D)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    return new_xyz, new_points


def sample_and_group_all(
    xyz: torch.Tensor,
    points: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if xyz.ndim != 3 or xyz.shape[2] != 3:
        raise ValueError(f"Expected xyz (B,N,3), got {tuple(xyz.shape)}")
    new_xyz = xyz.mean(dim=1, keepdim=True)  # (B,1,3)
    grouped_xyz = xyz[:, None, :, :]  # (B,1,N,3)
    grouped_xyz_norm = grouped_xyz - new_xyz[:, :, None, :]
    if points is None:
        new_points = grouped_xyz_norm
    else:
        new_points = torch.cat([grouped_xyz_norm, points[:, None, :, :]], dim=-1)
    return new_xyz, new_points


class PointNet2SetAbstraction(nn.Module):
    def __init__(
        self,
        *,
        npoint: int,
        nsample: int,
        in_channel: int,
        mlp: Sequence[int],
        group_all: bool,
    ) -> None:
        super().__init__()
        self.npoint = int(npoint)
        self.nsample = int(nsample)
        self.group_all = bool(group_all)

        last = int(in_channel)
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        for out_ch in mlp:
            oc = int(out_ch)
            self.mlp_convs.append(nn.Conv2d(last, oc, kernel_size=1, bias=False))
            self.mlp_bns.append(nn.BatchNorm2d(oc))
            last = oc

    def forward(
        self,
        xyz: torch.Tensor,
        points: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.nsample, xyz, points)
        # new_points: (B, S, K, C) -> (B, C, K, S)
        x = new_points.permute(0, 3, 2, 1).contiguous()
        for conv, bn in zip(self.mlp_convs, self.mlp_bns, strict=True):
            x = F.relu(bn(conv(x)), inplace=True)
        # Max pool over K.
        x = torch.max(x, dim=2).values  # (B, C_out, S)
        x = x.transpose(1, 2).contiguous()  # (B, S, C_out)
        return new_xyz, x


class PointNet2Encoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        sa1_npoint: int = 512,
        sa1_nsample: int = 32,
        sa2_npoint: int = 128,
        sa2_nsample: int = 64,
    ) -> None:
        super().__init__()
        in_ch = int(in_channels)
        if in_ch < 3:
            raise ValueError("in_channels must be >=3")
        d = int(in_ch - 3)
        self.sa1 = PointNet2SetAbstraction(
            npoint=int(sa1_npoint),
            nsample=int(sa1_nsample),
            in_channel=3 + d,
            mlp=(64, 64, 128),
            group_all=False,
        )
        self.sa2 = PointNet2SetAbstraction(
            npoint=int(sa2_npoint),
            nsample=int(sa2_nsample),
            in_channel=3 + 128,
            mlp=(128, 128, 256),
            group_all=False,
        )
        self.sa3 = PointNet2SetAbstraction(
            npoint=0,
            nsample=0,
            in_channel=3 + 256,
            mlp=(256, 512, 1024),
            group_all=True,
        )

    def forward(self, xyz: torch.Tensor, points: torch.Tensor | None) -> torch.Tensor:
        l1_xyz, l1_points = self.sa1(xyz, points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        _l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        return l3_points.squeeze(1)


class PointNet2Classifier(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        dropout: float,
        extra_dim: int = 0,
        in_channels: int = 3,
        sa1_npoint: int = 512,
        sa1_nsample: int = 32,
        sa2_npoint: int = 128,
        sa2_nsample: int = 64,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.feat = PointNet2Encoder(
            in_channels=int(in_channels),
            sa1_npoint=int(sa1_npoint),
            sa1_nsample=int(sa1_nsample),
            sa2_npoint=int(sa2_npoint),
            sa2_nsample=int(sa2_nsample),
        )
        in_dim = 1024 + int(extra_dim)
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
        xyz = points[:, :, :3].contiguous()
        feats = points[:, :, 3:].contiguous() if int(self.in_channels) > 3 else None
        x = self.feat(xyz, feats)
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

