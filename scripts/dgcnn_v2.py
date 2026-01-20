from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from _lib.point_ops import get_graph_feature, knn


@dataclass(frozen=True)
class DGCNNv2Params:
    k: int = 20
    emb_dims: int = 1024


class DGCNNv2Encoder(nn.Module):
    def __init__(self, *, in_channels: int, params: DGCNNv2Params) -> None:
        super().__init__()
        in_ch = int(in_channels)
        if in_ch < 3:
            raise ValueError("in_channels must be >=3")
        p = params
        self.in_channels = int(in_ch)
        self.k = int(p.k)
        self.emb_dims = int(p.emb_dims)
        if self.k <= 0:
            raise ValueError("params.k must be >0")
        if self.emb_dims <= 0:
            raise ValueError("params.emb_dims must be >0")

        self.conv1 = nn.Sequential(
            nn.Conv2d(2 * int(in_ch), 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, int(self.emb_dims), kernel_size=1, bias=False),
            nn.BatchNorm1d(int(self.emb_dims)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3 or points.shape[-1] != int(self.in_channels):
            raise ValueError(f"Expected points (B,N,{int(self.in_channels)}), got {tuple(points.shape)}")
        x = points.transpose(1, 2).contiguous()  # (B, C, N)

        x1 = self.conv1(get_graph_feature(x, k=int(self.k))).max(dim=-1).values  # (B, 64, N)
        x2 = self.conv2(get_graph_feature(x1, k=int(self.k))).max(dim=-1).values  # (B, 64, N)
        x3 = self.conv3(get_graph_feature(x2, k=int(self.k))).max(dim=-1).values  # (B, 128, N)
        x4 = self.conv4(get_graph_feature(x3, k=int(self.k))).max(dim=-1).values  # (B, 256, N)

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)  # (B, 512, N)
        feat = self.conv5(x_cat)  # (B, emb_dims, N)
        max_pool = torch.max(feat, dim=2).values  # (B, emb_dims)
        avg_pool = torch.mean(feat, dim=2)  # (B, emb_dims)
        return torch.cat((max_pool, avg_pool), dim=1)  # (B, 2*emb_dims)


class DGCNNv2Classifier(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        dropout: float,
        extra_dim: int = 0,
        in_channels: int = 3,
        params: DGCNNv2Params | None = None,
    ) -> None:
        super().__init__()
        p = params or DGCNNv2Params()
        self.in_channels = int(in_channels)
        self.feat = DGCNNv2Encoder(in_channels=int(in_channels), params=p)
        in_dim = 2 * int(p.emb_dims) + int(extra_dim)
        self.feature_dim = int(in_dim)
        self.head = nn.Sequential(
            nn.Linear(in_dim, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
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
