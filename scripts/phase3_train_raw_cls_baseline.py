#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import time
import zlib
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from _lib.device import normalize_device
from _lib.env import get_env_info
from _lib.git import get_git_info
from _lib.hash import sha1_file
from _lib.io import read_json, read_jsonl, write_json, write_jsonl
from _lib.point_ops import get_graph_feature
from _lib.seed import set_seed
from _lib.time import utc_now_iso
from dgcnn_v2 import DGCNNv2Classifier, DGCNNv2Params
from pointnet2 import PointNet2Classifier
from point_transformer import PointTransformerClassifier, PointTransformerParams
from pointmlp import PointMLPClassifier, PointMLPParams

from raw_cls.augment import augment_points_torch
from raw_cls.dataset import RawClsDataset, atomic_write_npz
from raw_cls.eval import (
    calibration_basic,
    domain_group_ids,
    evaluate,
    metrics_for_rows,
    save_confusion_csv,
    save_errors_csv,
    subset_rows,
)
from raw_cls.features import build_point_features_from_xyz, parse_point_features, point_feature_dim
from raw_cls.losses import coral_loss_between_groups, cross_entropy_per_sample, supervised_contrastive_loss
from raw_cls.metrics import metrics_from_confusion
from raw_cls.preprocess_np import apply_input_preprocess_np


@dataclass(frozen=True)
class TrainConfig:
    generated_at: str
    seed: int
    device: str
    deterministic: bool
    cudnn_benchmark: bool
    data_root: str
    out_dir: str
    exp_name: str
    model: str
    num_classes: int
    n_points: int
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    lr_scheduler: str
    warmup_epochs: int
    min_lr: float
    dropout: float
    grad_clip: float
    log_every: int
    aug_rotate_z: bool
    aug_scale: float
    aug_jitter_sigma: float
    aug_jitter_clip: float
    aug_dropout_ratio: float
    num_workers: int
    patience: int
    early_stop_min_delta: float
    save_best_metric: str
    extra_features: list[str]
    extra_mean: list[float]
    extra_std: list[float]
    balanced_sampler: bool
    label_smoothing: float
    ce_weighting: str
    init_feat: str
    freeze_feat_epochs: int
    dgcnn_k: int
    pointnet2_sa1_npoint: int
    pointnet2_sa1_nsample: int
    pointnet2_sa2_npoint: int
    pointnet2_sa2_nsample: int
    pt_dim: int
    pt_depth: int
    pt_k: int
    pt_ffn_mult: float
    pmlp_dim: int
    pmlp_depth: int
    pmlp_k: int
    pmlp_ffn_mult: float
    tta: int
    kfold_path: str
    kfold_k: int
    kfold_test_fold: int
    kfold_val_fold: int
    source_train: list[str]
    source_test: list[str]
    source_val_ratio: float
    source_split_seed: int
    input_normalize: str
    input_pca_align: bool
    input_pca_align_globalz: bool
    point_features: list[str]
    feature_cache_dir: str
    feature_k: int
    calibration_bins: int
    tooth_position_dropout: float
    supcon_weight: float
    supcon_temperature: float
    supcon_proj_dim: int
    domain_method: str
    domain_group_key: str
    groupdro_eta: float
    coral_weight: float
    coral_proj_dim: int


class PointNetClassifier(nn.Module):
    def __init__(self, num_classes: int, dropout: float, extra_dim: int = 0, in_channels: int = 3) -> None:
        super().__init__()
        in_dim = 512 + int(extra_dim)
        self.feature_dim = int(in_dim)
        self.in_channels = int(in_channels)
        self.feat = nn.Sequential(
            nn.Conv1d(int(in_channels), 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(in_dim, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(128, num_classes),
        )

    def forward_features(
        self,
        points: torch.Tensor,
        extra: torch.Tensor | None = None,
        domains: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = domains
        x = points.transpose(1, 2).contiguous()  # (B,C,N)
        x = self.feat(x)  # (B,512,N)
        x = torch.max(x, dim=2).values  # (B,512)
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


class PointNetCloudSetClassifier(nn.Module):
    """PointNet with cloud-aware (multi-cloud) pooling.

    Assumes points include a scalar `cloud_id` channel (scaled to [0,0.9] as in `cloud_id/10`).
    The model:
      1) extracts per-point features with a shared PointNet stem (excluding cloud_id from stem input),
      2) pools features per cloud_id bucket (max + mean),
      3) aggregates the set of clouds via mean + max across clouds.
    """

    def __init__(
        self,
        num_classes: int,
        dropout: float,
        *,
        extra_dim: int = 0,
        in_channels: int = 3,
        cloud_id_index: int,
        max_clouds: int = 10,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.cloud_id_index = int(cloud_id_index)
        self.max_clouds = int(max_clouds)
        if self.in_channels <= 0:
            raise ValueError("in_channels must be >0")
        if self.max_clouds <= 0:
            raise ValueError("max_clouds must be >0")

        self.feat = nn.Sequential(
            nn.Conv1d(int(in_channels), 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )

        # Per-cloud stats: concat(max, mean) over points -> 1024 dims; project to 512.
        self.cloud_embed = nn.Sequential(
            nn.Linear(512 * 2, 512, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
        )

        in_dim = 512 * 2 + int(extra_dim)
        self.feature_dim = int(in_dim)
        self.head = nn.Sequential(
            nn.Linear(in_dim, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(128, int(num_classes)),
        )

    def forward_features(
        self,
        points: torch.Tensor,
        extra: torch.Tensor | None = None,
        domains: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = domains
        if points.ndim != 3 or points.shape[-1] < int(self.cloud_id_index) + 1:
            raise ValueError(
                f"Expected points (B,N,C>=cloud_id_index+1), got {tuple(points.shape)} (cloud_id_index={int(self.cloud_id_index)})"
            )
        if points.shape[-1] < 4:
            raise ValueError("PointNetCloudSetClassifier expects point_features to include scalar cloud_id (C>=4).")

        # Separate cloud_id from stem input to avoid treating it as geometry.
        cid_raw = points[:, :, int(self.cloud_id_index)]
        if int(self.cloud_id_index) == 0:
            x_in = points[:, :, 1:]
        elif int(self.cloud_id_index) == int(points.shape[2]) - 1:
            x_in = points[:, :, : int(self.cloud_id_index)]
        else:
            x_in = torch.cat([points[:, :, : int(self.cloud_id_index)], points[:, :, int(self.cloud_id_index) + 1 :]], dim=2)

        if int(x_in.shape[2]) != int(self.in_channels):
            raise ValueError(f"Stem input channel mismatch: got C={int(x_in.shape[2])} expected in_channels={int(self.in_channels)}")

        x = x_in.transpose(1, 2).contiguous()  # (B,C,N)
        x = self.feat(x)  # (B,512,N)
        x_pts = x.transpose(1, 2).contiguous()  # (B,N,512)

        cid = torch.clamp((cid_raw * 10.0).round().to(dtype=torch.long), 0, int(self.max_clouds) - 1)  # (B,N)
        idx = cid.unsqueeze(-1).expand(-1, -1, int(x_pts.shape[2]))  # (B,N,512)

        # Sum / count for mean.
        cloud_sum = x_pts.new_zeros((int(x_pts.shape[0]), int(self.max_clouds), int(x_pts.shape[2])))
        cloud_sum = cloud_sum.scatter_add(1, idx, x_pts)
        cloud_count = x_pts.new_zeros((int(x_pts.shape[0]), int(self.max_clouds), 1))
        cloud_count = cloud_count.scatter_add(1, cid.unsqueeze(-1), x_pts.new_ones((int(x_pts.shape[0]), int(x_pts.shape[1]), 1)))
        present = cloud_count.squeeze(-1) > 0
        cloud_mean = cloud_sum / cloud_count.clamp_min(1.0)

        # Max via scatter_reduce.
        neg_inf = torch.finfo(x_pts.dtype).min
        cloud_max = x_pts.new_full((int(x_pts.shape[0]), int(self.max_clouds), int(x_pts.shape[2])), neg_inf)
        cloud_max = cloud_max.scatter_reduce(1, idx, x_pts, reduce="amax", include_self=True)
        cloud_max = torch.where(present.unsqueeze(-1), cloud_max, cloud_max.new_zeros(()).expand_as(cloud_max))

        cloud_stats = torch.cat([cloud_max, cloud_mean], dim=2)  # (B,K,1024)
        cloud_emb = self.cloud_embed(cloud_stats.reshape(-1, int(cloud_stats.shape[2]))).reshape(
            int(cloud_stats.shape[0]), int(cloud_stats.shape[1]), -1
        )  # (B,K,512)
        cloud_emb = cloud_emb * present.to(dtype=cloud_emb.dtype).unsqueeze(-1)

        # Set aggregation over clouds.
        present_f = present.to(dtype=cloud_emb.dtype)
        denom = present_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_cloud = (cloud_emb * present_f.unsqueeze(-1)).sum(dim=1) / denom  # (B,512)
        cloud_emb_masked = cloud_emb.masked_fill(~present.unsqueeze(-1), float("-inf"))
        max_cloud = cloud_emb_masked.max(dim=1).values
        max_cloud = torch.where(torch.isfinite(max_cloud), max_cloud, max_cloud.new_zeros(()).expand_as(max_cloud))

        feats = torch.cat([max_cloud, mean_cloud], dim=1)
        if extra is not None and extra.numel() > 0:
            feats = torch.cat([feats, extra], dim=1)
        return feats

    def forward_logits_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)

    def forward(
        self,
        points: torch.Tensor,
        extra: torch.Tensor | None = None,
        domains: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_logits_from_features(self.forward_features(points, extra, domains=domains))


class PointNetCloudAttnClassifier(nn.Module):
    """PointNet with cloud-aware pooling + attention over clouds.

    Same assumptions as `PointNetCloudSetClassifier`:
      - points include a scalar `cloud_id` channel (scaled to [0,0.9] as in `cloud_id/10`),
      - cloud_id is excluded from the PointNet stem input and only used to group points into clouds.

    Differences:
      - pools a per-cloud embedding, then aggregates clouds via max + attention-weighted sum
        (instead of max + mean).
      - injects simple per-cloud geometry meta (centroid + sampled fraction) into the cloud embedding.
    """

    def __init__(
        self,
        num_classes: int,
        dropout: float,
        *,
        extra_dim: int = 0,
        in_channels: int = 3,
        cloud_id_index: int,
        max_clouds: int = 10,
        attn_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.cloud_id_index = int(cloud_id_index)
        self.max_clouds = int(max_clouds)
        if self.in_channels <= 0:
            raise ValueError("in_channels must be >0")
        if self.max_clouds <= 0:
            raise ValueError("max_clouds must be >0")

        self.feat = nn.Sequential(
            nn.Conv1d(int(in_channels), 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )

        # Per-cloud stats: concat(max, mean) over points -> 1024 dims.
        # Add simple per-cloud meta: centroid(xyz)=3 and sampled fraction=1.
        cloud_meta_dim = 4
        self.cloud_embed = nn.Sequential(
            nn.Linear(512 * 2 + int(cloud_meta_dim), 512, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
        )

        self.cloud_attn = nn.Sequential(
            nn.Linear(512, int(attn_hidden), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(attn_hidden), 1, bias=False),
        )

        in_dim = 512 * 2 + int(extra_dim)
        self.feature_dim = int(in_dim)
        self.head = nn.Sequential(
            nn.Linear(in_dim, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(128, int(num_classes)),
        )

    def forward_features(
        self,
        points: torch.Tensor,
        extra: torch.Tensor | None = None,
        domains: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = domains
        if points.ndim != 3 or points.shape[-1] < max(4, int(self.cloud_id_index) + 1):
            raise ValueError(
                f"Expected points (B,N,C>=4), got {tuple(points.shape)} (cloud_id_index={int(self.cloud_id_index)})"
            )
        if points.shape[-1] < 4:
            raise ValueError("PointNetCloudAttnClassifier expects point_features to include scalar cloud_id (C>=4).")

        # Separate cloud_id from stem input to avoid treating it as geometry.
        cid_raw = points[:, :, int(self.cloud_id_index)]
        if int(self.cloud_id_index) == 0:
            x_in = points[:, :, 1:]
        elif int(self.cloud_id_index) == int(points.shape[2]) - 1:
            x_in = points[:, :, : int(self.cloud_id_index)]
        else:
            x_in = torch.cat([points[:, :, : int(self.cloud_id_index)], points[:, :, int(self.cloud_id_index) + 1 :]], dim=2)

        if int(x_in.shape[2]) != int(self.in_channels):
            raise ValueError(f"Stem input channel mismatch: got C={int(x_in.shape[2])} expected in_channels={int(self.in_channels)}")

        x = x_in.transpose(1, 2).contiguous()  # (B,C,N)
        x = self.feat(x)  # (B,512,N)
        x_pts = x.transpose(1, 2).contiguous()  # (B,N,512)

        cid = torch.clamp((cid_raw * 10.0).round().to(dtype=torch.long), 0, int(self.max_clouds) - 1)  # (B,N)
        idx = cid.unsqueeze(-1).expand(-1, -1, int(x_pts.shape[2]))  # (B,N,512)

        # Sum / count for mean.
        cloud_sum = x_pts.new_zeros((int(x_pts.shape[0]), int(self.max_clouds), int(x_pts.shape[2])))
        cloud_sum = cloud_sum.scatter_add(1, idx, x_pts)
        cloud_count = x_pts.new_zeros((int(x_pts.shape[0]), int(self.max_clouds), 1))
        cloud_count = cloud_count.scatter_add(1, cid.unsqueeze(-1), x_pts.new_ones((int(x_pts.shape[0]), int(x_pts.shape[1]), 1)))
        present = cloud_count.squeeze(-1) > 0
        cloud_mean = cloud_sum / cloud_count.clamp_min(1.0)

        # Max via scatter_reduce.
        neg_inf = torch.finfo(x_pts.dtype).min
        cloud_max = x_pts.new_full((int(x_pts.shape[0]), int(self.max_clouds), int(x_pts.shape[2])), neg_inf)
        cloud_max = cloud_max.scatter_reduce(1, idx, x_pts, reduce="amax", include_self=True)
        cloud_max = torch.where(present.unsqueeze(-1), cloud_max, cloud_max.new_zeros(()).expand_as(cloud_max))

        # Per-cloud centroid in normalized xyz space (use the first 3 channels which are xyz by convention in this repo).
        xyz = points[:, :, :3]
        idx_xyz = cid.unsqueeze(-1).expand(-1, -1, 3)
        cloud_xyz_sum = xyz.new_zeros((int(xyz.shape[0]), int(self.max_clouds), 3))
        cloud_xyz_sum = cloud_xyz_sum.scatter_add(1, idx_xyz, xyz)
        cloud_centroid = cloud_xyz_sum / cloud_count.clamp_min(1.0)
        cloud_frac = (cloud_count.squeeze(-1) / float(max(1, int(points.shape[1])))).unsqueeze(-1)

        cloud_stats = torch.cat([cloud_max, cloud_mean, cloud_centroid, cloud_frac], dim=2)  # (B,K,1028)
        cloud_emb = self.cloud_embed(cloud_stats.reshape(-1, int(cloud_stats.shape[2]))).reshape(
            int(cloud_stats.shape[0]), int(cloud_stats.shape[1]), -1
        )  # (B,K,512)
        cloud_emb = cloud_emb * present.to(dtype=cloud_emb.dtype).unsqueeze(-1)

        # Set aggregation over clouds.
        cloud_emb_masked = cloud_emb.masked_fill(~present.unsqueeze(-1), float("-inf"))
        max_cloud = cloud_emb_masked.max(dim=1).values
        max_cloud = torch.where(torch.isfinite(max_cloud), max_cloud, max_cloud.new_zeros(()).expand_as(max_cloud))

        attn_logits = self.cloud_attn(cloud_emb).squeeze(-1)  # (B,K)
        attn_logits = attn_logits.masked_fill(~present, float("-inf"))
        attn_w = torch.softmax(attn_logits, dim=1)  # (B,K)
        attn_w = torch.where(torch.isfinite(attn_w), attn_w, attn_w.new_zeros(()).expand_as(attn_w))
        attn_pool = torch.sum(cloud_emb * attn_w.unsqueeze(-1), dim=1)  # (B,512)

        feats = torch.cat([max_cloud, attn_pool], dim=1)
        if extra is not None and extra.numel() > 0:
            feats = torch.cat([feats, extra], dim=1)
        return feats

    def forward_logits_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)

    def forward(
        self,
        points: torch.Tensor,
        extra: torch.Tensor | None = None,
        domains: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_logits_from_features(self.forward_features(points, extra, domains=domains))


class PointNetCloudMILClassifier(nn.Module):
    """PointNet with cloud-aware pooling + MIL aggregation over clouds.

    - points include a scalar `cloud_id` channel (scaled to [0,0.9] as in `cloud_id/10`),
    - cloud_id is excluded from the PointNet stem input and only used to group points into clouds.

    Strategy:
      - pool per-cloud embeddings (max+mean over points + centroid + fraction),
      - compute per-cloud logits,
      - aggregate logits via logsumexp over clouds (class-wise "pick clouds").
    """

    def __init__(
        self,
        num_classes: int,
        dropout: float,
        *,
        extra_dim: int = 0,
        in_channels: int = 3,
        cloud_id_index: int,
        max_clouds: int = 10,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.cloud_id_index = int(cloud_id_index)
        self.max_clouds = int(max_clouds)
        self.num_classes = int(num_classes)
        if self.in_channels <= 0:
            raise ValueError("in_channels must be >0")
        if self.max_clouds <= 0:
            raise ValueError("max_clouds must be >0")
        if self.num_classes <= 1:
            raise ValueError("num_classes must be >1")

        self.feature_dim = int(self.num_classes)

        self.feat = nn.Sequential(
            nn.Conv1d(int(in_channels), 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )

        cloud_meta_dim = 4  # centroid(xyz)=3 + sampled fraction=1
        self.cloud_embed = nn.Sequential(
            nn.Linear(512 * 2 + int(cloud_meta_dim), 512, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
        )

        self.cloud_head = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(256, int(num_classes)),
        )

        self.extra_head: nn.Module | None = None
        if int(extra_dim) > 0:
            self.extra_head = nn.Sequential(
                nn.Linear(int(extra_dim), 64, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout(p=float(dropout)),
                nn.Linear(64, int(num_classes)),
            )

    def forward_features(
        self,
        points: torch.Tensor,
        extra: torch.Tensor | None = None,
        domains: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = domains
        if points.ndim != 3 or points.shape[-1] < max(4, int(self.cloud_id_index) + 1):
            raise ValueError(
                f"Expected points (B,N,C>=4), got {tuple(points.shape)} (cloud_id_index={int(self.cloud_id_index)})"
            )

        cid_raw = points[:, :, int(self.cloud_id_index)]
        if int(self.cloud_id_index) == 0:
            x_in = points[:, :, 1:]
        elif int(self.cloud_id_index) == int(points.shape[2]) - 1:
            x_in = points[:, :, : int(self.cloud_id_index)]
        else:
            x_in = torch.cat([points[:, :, : int(self.cloud_id_index)], points[:, :, int(self.cloud_id_index) + 1 :]], dim=2)

        if int(x_in.shape[2]) != int(self.in_channels):
            raise ValueError(f"Stem input channel mismatch: got C={int(x_in.shape[2])} expected in_channels={int(self.in_channels)}")

        x = x_in.transpose(1, 2).contiguous()  # (B,C,N)
        x = self.feat(x)  # (B,512,N)
        x_pts = x.transpose(1, 2).contiguous()  # (B,N,512)

        cid = torch.clamp((cid_raw * 10.0).round().to(dtype=torch.long), 0, int(self.max_clouds) - 1)  # (B,N)
        idx = cid.unsqueeze(-1).expand(-1, -1, int(x_pts.shape[2]))  # (B,N,512)

        cloud_sum = x_pts.new_zeros((int(x_pts.shape[0]), int(self.max_clouds), int(x_pts.shape[2])))
        cloud_sum = cloud_sum.scatter_add(1, idx, x_pts)
        cloud_count = x_pts.new_zeros((int(x_pts.shape[0]), int(self.max_clouds), 1))
        cloud_count = cloud_count.scatter_add(1, cid.unsqueeze(-1), x_pts.new_ones((int(x_pts.shape[0]), int(x_pts.shape[1]), 1)))
        present = cloud_count.squeeze(-1) > 0
        cloud_mean = cloud_sum / cloud_count.clamp_min(1.0)

        neg_inf = torch.finfo(x_pts.dtype).min
        cloud_max = x_pts.new_full((int(x_pts.shape[0]), int(self.max_clouds), int(x_pts.shape[2])), neg_inf)
        cloud_max = cloud_max.scatter_reduce(1, idx, x_pts, reduce="amax", include_self=True)
        cloud_max = torch.where(present.unsqueeze(-1), cloud_max, cloud_max.new_zeros(()).expand_as(cloud_max))

        xyz = points[:, :, :3]
        idx_xyz = cid.unsqueeze(-1).expand(-1, -1, 3)
        cloud_xyz_sum = xyz.new_zeros((int(xyz.shape[0]), int(self.max_clouds), 3))
        cloud_xyz_sum = cloud_xyz_sum.scatter_add(1, idx_xyz, xyz)
        cloud_centroid = cloud_xyz_sum / cloud_count.clamp_min(1.0)
        cloud_frac = (cloud_count.squeeze(-1) / float(max(1, int(points.shape[1])))).unsqueeze(-1)

        cloud_stats = torch.cat([cloud_max, cloud_mean, cloud_centroid, cloud_frac], dim=2)  # (B,K,1028)
        cloud_emb = self.cloud_embed(cloud_stats.reshape(-1, int(cloud_stats.shape[2]))).reshape(
            int(cloud_stats.shape[0]), int(cloud_stats.shape[1]), -1
        )  # (B,K,512)
        cloud_emb = cloud_emb * present.to(dtype=cloud_emb.dtype).unsqueeze(-1)

        cloud_logits = self.cloud_head(cloud_emb.reshape(-1, int(cloud_emb.shape[2]))).reshape(
            int(cloud_emb.shape[0]), int(cloud_emb.shape[1]), int(self.num_classes)
        )  # (B,K,C)
        cloud_logits = cloud_logits.masked_fill(~present.unsqueeze(-1), float("-inf"))
        logits = torch.logsumexp(cloud_logits, dim=1)  # (B,C)
        logits = torch.where(torch.isfinite(logits), logits, logits.new_zeros(()).expand_as(logits))

        if extra is not None and extra.numel() > 0 and self.extra_head is not None:
            logits = logits + self.extra_head(extra)
        return logits

    def forward_logits_from_features(self, features: torch.Tensor) -> torch.Tensor:
        # Features are logits for this MIL head.
        return features

    def forward(
        self,
        points: torch.Tensor,
        extra: torch.Tensor | None = None,
        domains: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_logits_from_features(self.forward_features(points, extra, domains=domains))


class DSBatchNorm1d(nn.Module):
    def __init__(self, num_features: int, num_domains: int) -> None:
        super().__init__()
        n = int(num_domains)
        if n <= 0:
            raise ValueError("num_domains must be >0")
        self.bns = nn.ModuleList([nn.BatchNorm1d(int(num_features)) for _ in range(n)])

    def forward(self, x: torch.Tensor, domains: torch.Tensor) -> torch.Tensor:
        if domains.ndim != 1 or domains.shape[0] != x.shape[0]:
            raise ValueError(f"domains must be (B,), got {tuple(domains.shape)} vs x {tuple(x.shape)}")
        out = torch.empty_like(x)
        for d, bn in enumerate(self.bns):
            mask = domains == int(d)
            if torch.any(mask):
                out[mask] = bn(x[mask])
        return out


class PointNetDSBNClassifier(nn.Module):
    def __init__(self, num_classes: int, dropout: float, *, extra_dim: int = 0, in_channels: int = 3, num_domains: int = 3) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_domains = int(num_domains)
        in_dim = 512 + int(extra_dim)
        self.feature_dim = int(in_dim)

        self.conv1 = nn.Conv1d(int(in_channels), 64, 1, bias=False)
        self.bn1 = DSBatchNorm1d(64, num_domains=self.num_domains)
        self.conv2 = nn.Conv1d(64, 128, 1, bias=False)
        self.bn2 = DSBatchNorm1d(128, num_domains=self.num_domains)
        self.conv3 = nn.Conv1d(128, 256, 1, bias=False)
        self.bn3 = DSBatchNorm1d(256, num_domains=self.num_domains)
        self.conv4 = nn.Conv1d(256, 512, 1, bias=False)
        self.bn4 = DSBatchNorm1d(512, num_domains=self.num_domains)

        self.head = nn.Sequential(
            nn.Linear(in_dim, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(128, int(num_classes)),
        )

    def forward_features(
        self,
        points: torch.Tensor,
        extra: torch.Tensor | None = None,
        domains: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if points.ndim != 3 or points.shape[-1] != int(self.in_channels):
            raise ValueError(f"Expected points (B,N,{int(self.in_channels)}), got {tuple(points.shape)}")
        if domains is None:
            domains = points.new_zeros((points.shape[0],), dtype=torch.long)
        domains = domains.to(device=points.device, dtype=torch.long)

        x = points.transpose(1, 2).contiguous()  # (B,C,N)
        x = F.relu(self.bn1(self.conv1(x), domains), inplace=True)
        x = F.relu(self.bn2(self.conv2(x), domains), inplace=True)
        x = F.relu(self.bn3(self.conv3(x), domains), inplace=True)
        x = F.relu(self.bn4(self.conv4(x), domains), inplace=True)
        x = torch.max(x, dim=2).values  # (B,512)
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


class PointNetPosMoEClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        dropout: float,
        *,
        extra_dim: int,
        missing_index: int,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        if int(extra_dim) <= 0:
            raise ValueError("PointNetPosMoEClassifier requires extra_dim>0.")
        self.in_channels = int(in_channels)
        self.extra_dim = int(extra_dim)
        self.missing_index = int(missing_index)
        self.feature_dim = 512 + int(extra_dim)

        self.feat = nn.Sequential(
            nn.Conv1d(int(in_channels), 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )

        self.head_geo = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(128, int(num_classes)),
        )
        self.head_pos = nn.Sequential(
            nn.Linear(512 + int(extra_dim), 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(128, int(num_classes)),
        )

    def _pooled(self, points: torch.Tensor) -> torch.Tensor:
        x = points.transpose(1, 2).contiguous()
        x = self.feat(x)
        return torch.max(x, dim=2).values

    def forward_features(
        self,
        points: torch.Tensor,
        extra: torch.Tensor | None = None,
        domains: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = domains
        x = self._pooled(points)
        if extra is None or extra.numel() <= 0:
            raise ValueError("PointNetPosMoEClassifier expects non-empty extra features.")
        return torch.cat([x, extra], dim=1)

    def forward_logits_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.head_pos(features)

    def forward(
        self,
        points: torch.Tensor,
        extra: torch.Tensor | None = None,
        domains: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = domains
        if extra is None or extra.numel() <= 0:
            raise ValueError("PointNetPosMoEClassifier expects non-empty extra features.")
        if int(self.missing_index) < 0 or int(self.missing_index) >= int(extra.shape[1]):
            raise ValueError(f"missing_index out of range: {self.missing_index} (extra_dim={int(extra.shape[1])})")
        pooled = self._pooled(points)
        logits_geo = self.head_geo(pooled)
        logits_pos = self.head_pos(torch.cat([pooled, extra], dim=1))
        missing = extra[:, int(self.missing_index)] >= 0.5
        if not torch.any(missing):
            return logits_pos
        out = logits_pos.clone()
        out[missing] = logits_geo[missing]
        return out


class DGCNNClassifier(nn.Module):
    def __init__(self, num_classes: int, dropout: float, k: int = 20, extra_dim: int = 0, in_channels: int = 3) -> None:
        super().__init__()
        self.k = int(k)
        self.in_channels = int(in_channels)
        self.conv1 = nn.Sequential(
            nn.Conv2d(2 * int(in_channels), 64, kernel_size=1, bias=False),
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
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        in_dim = 1024 * 2 + int(extra_dim)
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
            nn.Linear(256, num_classes),
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
        x = points.transpose(1, 2).contiguous()  # (B, C, N)

        x1 = self.conv1(get_graph_feature(x, k=self.k)).max(dim=-1).values  # (B, 64, N)
        x2 = self.conv2(get_graph_feature(x1, k=self.k)).max(dim=-1).values  # (B, 64, N)
        x3 = self.conv3(get_graph_feature(x2, k=self.k)).max(dim=-1).values  # (B, 128, N)
        x4 = self.conv4(get_graph_feature(x3, k=self.k)).max(dim=-1).values  # (B, 256, N)

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)  # (B, 512, N)
        feat = self.conv5(x_cat)  # (B, 1024, N)
        max_pool = torch.max(feat, dim=2).values  # (B, 1024)
        avg_pool = torch.mean(feat, dim=2)  # (B, 1024)
        g = torch.cat((max_pool, avg_pool), dim=1)  # (B, 2048)
        if extra is not None and extra.numel() > 0:
            g = torch.cat([g, extra], dim=1)
        return g

    def forward_logits_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)

    def forward(
        self,
        points: torch.Tensor,
        extra: torch.Tensor | None = None,
        domains: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_logits_from_features(self.forward_features(points, extra, domains=domains))


class MetaMLPClassifier(nn.Module):
    def __init__(self, num_classes: int, dropout: float, extra_dim: int) -> None:
        super().__init__()
        d = int(extra_dim)
        if d <= 0:
            raise ValueError("MetaMLPClassifier requires extra_dim>0 (set --extra-features).")
        self.feature_dim = int(d)
        self.net = nn.Sequential(
            nn.Linear(d, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(64, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(64, int(num_classes)),
        )

    def forward_features(
        self,
        points: torch.Tensor,
        extra: torch.Tensor | None = None,
        domains: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = points
        _ = domains
        if extra is None or extra.numel() <= 0:
            raise ValueError("MetaMLPClassifier expects non-empty extra features.")
        return extra

    def forward_logits_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)

    def forward(
        self,
        points: torch.Tensor,
        extra: torch.Tensor | None = None,
        domains: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_logits_from_features(self.forward_features(points, extra, domains=domains))


class GeomMLPClassifier(nn.Module):
    """Traditional geometric-feature baseline.

    Computes simple global statistics from the point cloud and classifies with an MLP.
    """

    def __init__(self, num_classes: int, dropout: float, extra_dim: int = 0) -> None:
        super().__init__()
        self.extra_dim = int(extra_dim)
        base_dim = 14  # centroid(3) + std_xyz(3) + bbox_size(3) + eigvals(3) + (mean_norm,std_norm)(2)
        in_dim = base_dim + self.extra_dim
        self.feature_dim = int(in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(64, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(64, int(num_classes)),
        )

    def forward_features(
        self,
        points: torch.Tensor,
        extra: torch.Tensor | None = None,
        domains: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = domains
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError(f"Expected points (B,N,3), got {tuple(points.shape)}")
        x = points
        centroid = x.mean(dim=1)  # (B,3)
        xc = x - centroid[:, None, :]
        std_xyz = xc.std(dim=1)  # (B,3)
        mn = x.min(dim=1).values
        mx = x.max(dim=1).values
        bbox = mx - mn  # (B,3)

        cov = torch.bmm(xc.transpose(1, 2), xc) / float(max(1, x.shape[1]))  # (B,3,3)
        try:
            eig = torch.linalg.eigvalsh(cov)  # (B,3), sorted
        except Exception:
            eig = torch.zeros((x.shape[0], 3), device=x.device, dtype=x.dtype)

        norms = torch.linalg.norm(xc, dim=2)  # (B,N)
        mean_norm = norms.mean(dim=1, keepdim=True)
        std_norm = norms.std(dim=1, keepdim=True)

        feats = torch.cat([centroid, std_xyz, bbox, eig, mean_norm, std_norm], dim=1)
        if extra is not None and extra.numel() > 0:
            feats = torch.cat([feats, extra], dim=1)
        return feats

    def forward_logits_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)

    def forward(
        self,
        points: torch.Tensor,
        extra: torch.Tensor | None = None,
        domains: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_logits_from_features(self.forward_features(points, extra, domains=domains))


class CloudGeomMLPClassifier(nn.Module):
    """Cloud-aware geometric baseline.

    Assumes points include a discrete `cloud_id` channel (scaled to [0,0.9] as in `cloud_id/10`).
    Computes GeomMLP-style statistics per cloud_id bucket and concatenates them.
    """

    def __init__(
        self,
        num_classes: int,
        dropout: float,
        *,
        extra_dim: int = 0,
        cloud_id_index: int,
        max_clouds: int = 10,
    ) -> None:
        super().__init__()
        self.extra_dim = int(extra_dim)
        self.cloud_id_index = int(cloud_id_index)
        self.max_clouds = int(max_clouds)
        if self.max_clouds <= 0:
            raise ValueError("max_clouds must be >0")

        base_dim = 14  # centroid(3) + std_xyz(3) + bbox_size(3) + eigvals(3) + (mean_norm,std_norm)(2)
        in_dim = (base_dim * self.max_clouds) + self.extra_dim
        self.feature_dim = int(in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(128, int(num_classes)),
        )

    def forward_features(
        self,
        points: torch.Tensor,
        extra: torch.Tensor | None = None,
        domains: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = domains
        if points.ndim != 3 or points.shape[-1] < max(4, int(self.cloud_id_index) + 1):
            raise ValueError(f"Expected points (B,N,C>=4), got {tuple(points.shape)} (cloud_id_index={self.cloud_id_index})")

        xyz = points[:, :, :3]
        cid_raw = points[:, :, int(self.cloud_id_index)]
        cid = torch.clamp((cid_raw * 10.0).round().to(dtype=torch.long), 0, int(self.max_clouds) - 1)

        b = int(points.shape[0])
        feats_per_cloud: list[torch.Tensor] = []
        for u in range(int(self.max_clouds)):
            mask = cid == int(u)  # (B,N)
            cnt = mask.sum(dim=1, keepdim=True)  # (B,1)
            present = cnt > 0
            cnt_f = cnt.clamp(min=1).to(dtype=xyz.dtype)
            m3 = mask.to(dtype=xyz.dtype).unsqueeze(-1)  # (B,N,1)

            centroid = (xyz * m3).sum(dim=1) / cnt_f  # (B,3)
            diff = (xyz - centroid[:, None, :]) * m3
            std_xyz = torch.sqrt((diff * diff).sum(dim=1) / cnt_f + 1e-12)

            # bbox (masked min/max); missing clouds -> zeros
            min_xyz = xyz.masked_fill(~mask.unsqueeze(-1), float("inf")).amin(dim=1)
            max_xyz = xyz.masked_fill(~mask.unsqueeze(-1), float("-inf")).amax(dim=1)
            bbox = max_xyz - min_xyz
            bbox = torch.where(present.expand(-1, bbox.shape[1]), bbox, torch.zeros_like(bbox))

            cov = torch.bmm(diff.transpose(1, 2), diff) / cnt_f.view(b, 1, 1)
            try:
                eig = torch.linalg.eigvalsh(cov)  # (B,3)
            except Exception:
                eig = torch.zeros((b, 3), device=xyz.device, dtype=xyz.dtype)

            dnorm = torch.linalg.norm(xyz - centroid[:, None, :], dim=2)  # (B,N)
            dmask = mask.to(dtype=xyz.dtype)
            mean_norm = (dnorm * dmask).sum(dim=1, keepdim=True) / cnt_f
            var_norm = ((dnorm - mean_norm) ** 2 * dmask).sum(dim=1, keepdim=True) / cnt_f
            std_norm = torch.sqrt(var_norm + 1e-12)

            feats_per_cloud.append(torch.cat([centroid, std_xyz, bbox, eig, mean_norm, std_norm], dim=1))

        feats = torch.cat(feats_per_cloud, dim=1)
        if extra is not None and extra.numel() > 0:
            feats = torch.cat([feats, extra], dim=1)
        return feats

    def forward_logits_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)

    def forward(
        self,
        points: torch.Tensor,
        extra: torch.Tensor | None = None,
        domains: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_logits_from_features(self.forward_features(points, extra, domains=domains))


def build_loaders(
    data_root: Path,
    cfg: TrainConfig,
    label_to_id: dict[str, int],
    *,
    split_override: dict[str, str] | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, Counter[int]]:
    index_path = data_root / "index.jsonl"
    rows = read_jsonl(index_path)
    load_points = str(cfg.model).lower() in {
        "pointnet",
        "pointnet2",
        "pointnet_cloudattn",
        "pointnet_cloudset",
        "pointnet_cloudmil",
        "dgcnn",
        "geom_mlp",
        "cloud_geom_mlp",
        "pointnet_dsbn",
        "pointnet_pos_moe",
    }

    def split_of(r: dict[str, Any]) -> str:
        if split_override is None:
            return str(r.get("split") or "")
        return str(split_override.get(str(r.get("case_key") or ""), "unknown"))

    if split_override is not None:
        # Keep metadata consistent (so downstream analysis sees the actual split used).
        rows = [{**r, "split": split_of(r)} for r in rows]

    train_rows = [r for r in rows if split_of(r) == "train"]
    val_rows = [r for r in rows if split_of(r) == "val"]
    test_rows = [r for r in rows if split_of(r) == "test"]

    train_counts: Counter[int] = Counter()
    for r in train_rows:
        train_counts[label_to_id[str(r["label"])]] += 1

    # Train-set feature normalization stats.
    feat_names = list(cfg.extra_features or [])
    if feat_names:
        feats = []
        for r in train_rows:
            tmp_ds = RawClsDataset(
                rows=[r],
                data_root=data_root,
                label_to_id=label_to_id,
                n_points=cfg.n_points,
                seed=cfg.seed,
                train=False,
                aug_rotate_z=False,
                aug_scale=0.0,
                aug_jitter_sigma=0.0,
                aug_jitter_clip=0.0,
                aug_dropout_ratio=0.0,
                load_points=False,
                extra_features=feat_names,
            )
            feats.append(tmp_ds._extract_extra(r))
        mat = np.stack(feats, axis=0).astype(np.float32, copy=False)
        mean = mat.mean(axis=0)
        std = mat.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32, copy=False)
    else:
        mean = np.zeros((0,), dtype=np.float32)
        std = np.ones((0,), dtype=np.float32)

    ds_train = RawClsDataset(
        rows=train_rows,
        data_root=data_root,
        label_to_id=label_to_id,
        n_points=cfg.n_points,
        seed=cfg.seed,
        train=True,
        aug_rotate_z=cfg.aug_rotate_z,
        aug_scale=cfg.aug_scale,
        aug_jitter_sigma=cfg.aug_jitter_sigma,
        aug_jitter_clip=cfg.aug_jitter_clip,
        aug_dropout_ratio=cfg.aug_dropout_ratio,
        load_points=load_points,
        extra_features=feat_names,
        extra_mean=mean,
        extra_std=std,
        point_features=list(cfg.point_features or ["xyz"])
        if str(cfg.model).lower()
        in {
            "pointnet",
            "pointnet2",
            "pointnet_cloudattn",
            "pointnet_cloudset",
            "pointnet_cloudmil",
            "dgcnn",
            "pointnet_dsbn",
            "pointnet_pos_moe",
            "cloud_geom_mlp",
        }
        else ["xyz"],
        feature_cache_dir=Path(cfg.feature_cache_dir)
        if str(cfg.feature_cache_dir or "").strip()
        and str(cfg.model).lower()
        in {
            "pointnet",
            "pointnet2",
            "pointnet_cloudattn",
            "pointnet_cloudset",
            "pointnet_cloudmil",
            "dgcnn",
            "pointnet_dsbn",
            "pointnet_pos_moe",
            "cloud_geom_mlp",
        }
        else None,
        feature_k=int(cfg.feature_k),
        tooth_position_dropout=float(cfg.tooth_position_dropout),
        input_normalize=str(cfg.input_normalize or "none"),
        input_pca_align=bool(cfg.input_pca_align),
        input_pca_align_globalz=bool(cfg.input_pca_align_globalz),
    )
    ds_val = RawClsDataset(
        rows=val_rows,
        data_root=data_root,
        label_to_id=label_to_id,
        n_points=cfg.n_points,
        seed=cfg.seed,
        train=False,
        aug_rotate_z=False,
        aug_scale=0.0,
        aug_jitter_sigma=0.0,
        aug_jitter_clip=0.0,
        aug_dropout_ratio=0.0,
        load_points=load_points,
        extra_features=feat_names,
        extra_mean=mean,
        extra_std=std,
        point_features=list(cfg.point_features or ["xyz"])
        if str(cfg.model).lower()
        in {
            "pointnet",
            "pointnet2",
            "pointnet_cloudattn",
            "pointnet_cloudset",
            "pointnet_cloudmil",
            "dgcnn",
            "pointnet_dsbn",
            "pointnet_pos_moe",
            "cloud_geom_mlp",
        }
        else ["xyz"],
        feature_cache_dir=Path(cfg.feature_cache_dir)
        if str(cfg.feature_cache_dir or "").strip()
        and str(cfg.model).lower()
        in {
            "pointnet",
            "pointnet2",
            "pointnet_cloudattn",
            "pointnet_cloudset",
            "pointnet_cloudmil",
            "dgcnn",
            "pointnet_dsbn",
            "pointnet_pos_moe",
            "cloud_geom_mlp",
        }
        else None,
        feature_k=int(cfg.feature_k),
        tooth_position_dropout=0.0,
        input_normalize=str(cfg.input_normalize or "none"),
        input_pca_align=bool(cfg.input_pca_align),
        input_pca_align_globalz=bool(cfg.input_pca_align_globalz),
    )
    ds_test = RawClsDataset(
        rows=test_rows,
        data_root=data_root,
        label_to_id=label_to_id,
        n_points=cfg.n_points,
        seed=cfg.seed,
        train=False,
        aug_rotate_z=False,
        aug_scale=0.0,
        aug_jitter_sigma=0.0,
        aug_jitter_clip=0.0,
        aug_dropout_ratio=0.0,
        load_points=load_points,
        extra_features=feat_names,
        extra_mean=mean,
        extra_std=std,
        point_features=list(cfg.point_features or ["xyz"])
        if str(cfg.model).lower()
        in {
            "pointnet",
            "pointnet2",
            "pointnet_cloudattn",
            "pointnet_cloudset",
            "pointnet_cloudmil",
            "dgcnn",
            "pointnet_dsbn",
            "pointnet_pos_moe",
            "cloud_geom_mlp",
        }
        else ["xyz"],
        feature_cache_dir=Path(cfg.feature_cache_dir)
        if str(cfg.feature_cache_dir or "").strip()
        and str(cfg.model).lower()
        in {
            "pointnet",
            "pointnet2",
            "pointnet_cloudattn",
            "pointnet_cloudset",
            "pointnet_cloudmil",
            "dgcnn",
            "pointnet_dsbn",
            "pointnet_pos_moe",
            "cloud_geom_mlp",
        }
        else None,
        feature_k=int(cfg.feature_k),
        tooth_position_dropout=0.0,
        input_normalize=str(cfg.input_normalize or "none"),
        input_pca_align=bool(cfg.input_pca_align),
        input_pca_align_globalz=bool(cfg.input_pca_align_globalz),
    )

    def collate(
        batch: list[tuple[torch.Tensor, torch.Tensor, int, dict[str, Any]]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
        pts, extras, ys, metas = zip(*batch, strict=True)
        extra_batch = torch.stack(list(extras), dim=0) if extras and extras[0].numel() else torch.zeros((len(batch), 0))
        return (
            torch.stack(list(pts), dim=0),
            extra_batch,
            torch.tensor(list(ys), dtype=torch.long),
            list(metas),
        )

    pin = cfg.device == "cuda"
    sampler = None
    shuffle = True
    if bool(cfg.balanced_sampler) and len(train_rows) > 0:
        # Inverse-frequency sampling by label (case-level).
        weights = []
        for r in train_rows:
            class_id = int(label_to_id[str(r["label"])])
            cnt = float(train_counts.get(class_id, 1))
            weights.append(1.0 / max(1.0, cnt))
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
        )
        shuffle = False

    train_drop_last = False
    if int(cfg.batch_size) > 1 and (len(train_rows) % int(cfg.batch_size)) == 1:
        # Avoid BatchNorm crash on a last batch of size 1.
        train_drop_last = True
        print(
            f"[loader] train_drop_last=True (avoid batch_size=1): n_train={len(train_rows)} batch_size={int(cfg.batch_size)}",
            flush=True,
        )

    train_loader = DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        drop_last=train_drop_last,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=max(1, cfg.batch_size),
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        drop_last=False,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        ds_test,
        batch_size=max(1, cfg.batch_size),
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        drop_last=False,
        collate_fn=collate,
    )
    return train_loader, val_loader, test_loader, train_counts


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Phase 3: raw classification baselines (PointNet/PointNet2/DGCNN/DGCNNv2/PointTransformer)."
    )
    ap.add_argument("--data-root", type=Path, default=Path("processed/raw_cls/v1"))
    ap.add_argument("--run-root", type=Path, default=Path("runs/raw_cls_baseline"))
    ap.add_argument("--exp-name", type=str, default="")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", type=str, default="auto", help="auto|cuda|cpu")
    ap.add_argument("--deterministic", action="store_true", default=True, help="Enable deterministic cuDNN (default: true).")
    ap.add_argument("--no-deterministic", action="store_false", dest="deterministic", help="Disable deterministic cuDNN.")
    ap.add_argument("--cudnn-benchmark", action="store_true", default=False, help="Enable cuDNN benchmark (only if deterministic=false).")
    ap.add_argument("--no-cudnn-benchmark", action="store_false", dest="cudnn_benchmark", help="Disable cuDNN benchmark.")
    ap.add_argument(
        "--model",
        choices=[
            "pointnet",
            "pointnet2",
            "point_transformer",
            "pointmlp",
            "pointnet_cloudattn",
            "pointnet_cloudmil",
            "pointnet_cloudset",
            "dgcnn",
            "dgcnn_v2",
            "meta_mlp",
            "geom_mlp",
            "cloud_geom_mlp",
            "pointnet_dsbn",
            "pointnet_pos_moe",
        ],
        default="pointnet",
    )
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=15, help="Early stopping patience on val macro_f1_present")
    ap.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1e-6,
        help="Minimum absolute improvement in val score to reset early-stopping patience (default: 1e-6).",
    )
    ap.add_argument(
        "--save-best-metric",
        type=str,
        default="macro_f1_present",
        help="Metric to pick best checkpoint / early-stop on: accuracy|macro_f1_present|balanced_accuracy_present|macro_f1_all",
    )
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--lr-scheduler", type=str, default="", help="none|cosine (optional).")
    ap.add_argument("--warmup-epochs", type=int, default=0, help="Warmup epochs for lr-scheduler (default: 0).")
    ap.add_argument("--min-lr", type=float, default=0.0, help="Min lr for cosine schedule (default: 0).")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--grad-clip", type=float, default=0.0, help="Optional grad norm clipping (0=disable).")
    ap.add_argument("--log-every", type=int, default=0, help="Optional step logging frequency (0=epoch-only).")
    ap.add_argument("--n-points", type=int, default=4096)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--balanced-sampler", action="store_true", help="Use inverse-frequency sampling on train split.")
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument(
        "--ce-weighting",
        type=str,
        default="auto",
        help="Cross-entropy class weighting: auto|none|inverse_freq (auto: inverse_freq only when balanced_sampler is off).",
    )
    ap.add_argument("--calibration-bins", type=int, default=15, help="Bins for ECE/rel diagram style metrics.")
    ap.add_argument("--tooth-position-dropout", type=float, default=0.0, help="Train-time dropout for tooth_position (simulates missing).")
    ap.add_argument("--supcon-weight", type=float, default=0.0, help="Supervised contrastive loss weight (0=disable).")
    ap.add_argument("--supcon-temp", type=float, default=0.07, help="SupCon temperature.")
    ap.add_argument("--supcon-proj-dim", type=int, default=128, help="SupCon projection dim.")
    ap.add_argument("--supcon-aug-rotate-z", type=str, default="", help="Override SupCon view augmentation rotate_z (true/false).")
    ap.add_argument("--supcon-aug-scale", type=float, default=None, help="Override SupCon view augmentation scale (s in [1-s,1+s]).")
    ap.add_argument("--supcon-aug-jitter-sigma1", type=float, default=None, help="SupCon view1 jitter sigma override.")
    ap.add_argument("--supcon-aug-jitter-sigma2", type=float, default=None, help="SupCon view2 jitter sigma override.")
    ap.add_argument("--supcon-aug-jitter-clip1", type=float, default=None, help="SupCon view1 jitter clip override.")
    ap.add_argument("--supcon-aug-jitter-clip2", type=float, default=None, help="SupCon view2 jitter clip override.")
    ap.add_argument("--supcon-aug-dropout-ratio1", type=float, default=None, help="SupCon view1 dropout ratio override.")
    ap.add_argument("--supcon-aug-dropout-ratio2", type=float, default=None, help="SupCon view2 dropout ratio override.")
    ap.add_argument("--init-feat", type=Path, default=None, help="Optional: init encoder feat from a checkpoint.")
    ap.add_argument("--freeze-feat-epochs", type=int, default=0, help="Freeze encoder feat for first K epochs.")
    ap.add_argument("--dgcnn-k", type=int, default=20, help="kNN size for DGCNN (only when --model=dgcnn|dgcnn_v2).")
    ap.add_argument("--pointnet2-sa1-npoint", type=int, default=512, help="PointNet2 SA1 npoint (only when --model=pointnet2).")
    ap.add_argument("--pointnet2-sa1-nsample", type=int, default=32, help="PointNet2 SA1 nsample (only when --model=pointnet2).")
    ap.add_argument("--pointnet2-sa2-npoint", type=int, default=128, help="PointNet2 SA2 npoint (only when --model=pointnet2).")
    ap.add_argument("--pointnet2-sa2-nsample", type=int, default=64, help="PointNet2 SA2 nsample (only when --model=pointnet2).")
    ap.add_argument("--pt-dim", type=int, default=96, help="PointTransformer embedding dim (only when --model=point_transformer).")
    ap.add_argument("--pt-depth", type=int, default=4, help="PointTransformer depth (only when --model=point_transformer).")
    ap.add_argument("--pt-k", type=int, default=16, help="PointTransformer kNN size (only when --model=point_transformer).")
    ap.add_argument("--pt-ffn-mult", type=float, default=2.0, help="PointTransformer FFN hidden multiplier (only when --model=point_transformer).")
    ap.add_argument("--pmlp-dim", type=int, default=128, help="PointMLP embedding dim (only when --model=pointmlp).")
    ap.add_argument("--pmlp-depth", type=int, default=6, help="PointMLP depth (only when --model=pointmlp).")
    ap.add_argument("--pmlp-k", type=int, default=16, help="PointMLP kNN size (only when --model=pointmlp).")
    ap.add_argument("--pmlp-ffn-mult", type=float, default=2.0, help="PointMLP FFN hidden multiplier (only when --model=pointmlp).")
    ap.add_argument("--tta", type=int, default=0, help="Test-time augmentation passes for final val/test eval (0=disable).")
    ap.add_argument("--domain-method", type=str, default="", help="baseline|groupdro|coral|dsbn|pos_moe (optional).")
    ap.add_argument("--domain-group-key", type=str, default="tooth_position", help="Grouping key for domain methods (default: tooth_position).")
    ap.add_argument("--groupdro-eta", type=float, default=0.1, help="GroupDRO eta (only when --domain-method=groupdro).")
    ap.add_argument("--coral-weight", type=float, default=0.0, help="CORAL loss weight (only when --domain-method=coral).")
    ap.add_argument("--coral-proj-dim", type=int, default=128, help="CORAL projection dim.")
    ap.add_argument("--kfold", type=Path, default=None, help="Optional: K-fold split file (metadata/*.json).")
    ap.add_argument("--fold", type=int, default=-1, help="Test fold index when using --kfold.")
    ap.add_argument("--val-fold", type=int, default=-1, help="Val fold index when using --kfold (default: fold+1).")
    ap.add_argument("--source-train", type=str, default="", help="Optional: override split by `source` field (train domain).")
    ap.add_argument("--source-test", type=str, default="", help="Optional: override split by `source` field (test domain).")
    ap.add_argument("--source-val-ratio", type=float, default=0.1, help="Val ratio within source-train (default: 0.1).")
    ap.add_argument(
        "--source-split-seed",
        type=int,
        default=0,
        help="RNG seed for deterministic source train/val split (0=use --seed).",
    )
    ap.add_argument("--aug-rotate-z", action="store_true", default=True)
    ap.add_argument("--no-aug-rotate-z", action="store_false", dest="aug_rotate_z")
    ap.add_argument("--aug-scale", type=float, default=0.2, help="Uniform scale in [1-s, 1+s]")
    ap.add_argument("--aug-jitter-sigma", type=float, default=0.01)
    ap.add_argument("--aug-jitter-clip", type=float, default=0.05)
    ap.add_argument("--aug-dropout-ratio", type=float, default=0.1)
    ap.add_argument(
        "--extra-features",
        type=str,
        default="",
        help="Comma-separated: scale,log_scale,points,log_points,objects_used (default: none).",
    )
    ap.add_argument(
        "--point-features",
        type=str,
        default="xyz",
        help="Comma-separated: xyz,normals,curvature,radius (default: xyz).",
    )
    ap.add_argument(
        "--input-normalize",
        type=str,
        default="none",
        help="Optional input normalization applied at load time: none|max_norm|bbox_diag (default: none).",
    )
    ap.add_argument("--input-pca-align", action="store_true", help="Optional PCA alignment applied at load time (default: off).")
    ap.add_argument(
        "--input-pca-align-globalz",
        action="store_true",
        help="When PCA is enabled, reorder axes to align with global Z/X for more stable orientation (default: off).",
    )
    ap.add_argument(
        "--feature-cache-dir",
        type=Path,
        default=None,
        help="Optional: cache directory for derived point features (npz).",
    )
    ap.add_argument("--feature-k", type=int, default=30, help="kNN size for normals/curvature/radius (default: 30).")
    ap.add_argument("--precompute-features", action="store_true", help="Precompute feature cache before training.")
    args = ap.parse_args()
    t0 = time.time()

    def _parse_bool(s: str, *, default: bool) -> bool:
        t = str(s or "").strip().lower()
        if not t:
            return bool(default)
        if t in {"1", "true", "yes", "y", "on"}:
            return True
        if t in {"0", "false", "no", "n", "off"}:
            return False
        return bool(default)

    extra_features = [s.strip() for s in str(args.extra_features or "").split(",") if s.strip()]
    allowed_extra = {
        "scale",
        "log_scale",
        "points",
        "log_points",
        "objects_used",
        "tooth_position_premolar",
        "tooth_position_molar",
        "tooth_position_missing",
    }
    for name in extra_features:
        if name not in allowed_extra:
            raise SystemExit(f"Unknown --extra-features item: {name}")

    try:
        point_features = parse_point_features(str(args.point_features))
    except Exception as e:
        raise SystemExit(f"Invalid --point-features: {e}") from e
    input_normalize = str(args.input_normalize or "").strip().lower() or "none"
    if input_normalize not in {"none", "off", "max_norm", "bbox_diag"}:
        raise SystemExit(f"Invalid --input-normalize: {args.input_normalize!r} (allowed: none,max_norm,bbox_diag)")
    args.input_normalize = input_normalize
    feature_k = int(args.feature_k)
    feature_cache_dir = ""
    if args.feature_cache_dir is not None:
        feature_cache_dir = str(args.feature_cache_dir.expanduser().resolve())

    data_root = args.data_root.resolve()
    if not data_root.exists():
        raise SystemExit(f"Missing data_root: {data_root}")
    label_map_path = data_root / "label_map.json"
    if not label_map_path.exists():
        raise SystemExit(f"Missing label_map.json: {label_map_path}")

    label_to_id = {str(k): int(v) for k, v in read_json(label_map_path).items()}
    labels_by_id = [None] * (max(label_to_id.values()) + 1 if label_to_id else 0)
    for lab, i in label_to_id.items():
        labels_by_id[int(i)] = lab
    if any(x is None for x in labels_by_id):
        raise SystemExit("label_map.json must map to a contiguous 0..C-1 space")
    labels_by_id = [str(x) for x in labels_by_id]

    init_feat_path = ""
    init_feat_sha1 = ""
    init_feat_bytes = 0
    if args.init_feat is not None:
        init_feat_file = args.init_feat.expanduser().resolve()
        if not init_feat_file.is_file():
            raise SystemExit(f"Missing init checkpoint: {init_feat_file}")
        init_feat_path = str(init_feat_file)
        init_feat_bytes = int(init_feat_file.stat().st_size)
        init_feat_sha1 = sha1_file(init_feat_file)
        print(
            f"[init-feat] path={init_feat_file} bytes={init_feat_bytes} sha1={init_feat_sha1}",
            flush=True,
        )

    split_override: dict[str, str] | None = None
    kfold_path = ""
    kfold_k = 0
    kfold_test_fold = -1
    kfold_val_fold = -1
    if args.kfold is not None:
        kfold_file = args.kfold.expanduser().resolve()
        if not kfold_file.is_file():
            raise SystemExit(f"Missing kfold file: {kfold_file}")
        obj = read_json(kfold_file)
        kfold_k = int(obj.get("k") or 0)
        if kfold_k <= 1:
            raise SystemExit(f"Invalid k in kfold file: {kfold_k}")
        kfold_test_fold = int(args.fold)
        if kfold_test_fold < 0 or kfold_test_fold >= kfold_k:
            raise SystemExit(f"--fold must be in [0,{kfold_k-1}] when using --kfold")
        kfold_val_fold = int(args.val_fold) if int(args.val_fold) >= 0 else (kfold_test_fold + 1) % kfold_k
        if kfold_val_fold < 0 or kfold_val_fold >= kfold_k:
            raise SystemExit(f"--val-fold must be in [0,{kfold_k-1}] when using --kfold")

        case_to_fold = obj.get("case_to_fold") or {}
        if not case_to_fold:
            folds = obj.get("folds") or {}
            for fold_str, items in folds.items():
                try:
                    fid = int(fold_str)
                except Exception:
                    continue
                for ck in items:
                    case_to_fold[str(ck)] = fid

        split_override = {}
        for case_key, fid in case_to_fold.items():
            f = int(fid)
            if f == kfold_test_fold:
                split_override[str(case_key)] = "test"
            elif f == kfold_val_fold:
                split_override[str(case_key)] = "val"
            else:
                split_override[str(case_key)] = "train"
        kfold_path = str(kfold_file)

    def _parse_csv(text: str) -> list[str]:
        return [s.strip() for s in str(text or "").split(",") if s.strip()]

    source_train = _parse_csv(str(args.source_train))
    source_test = _parse_csv(str(args.source_test))
    source_val_ratio = float(args.source_val_ratio)
    source_split_seed = int(args.source_split_seed) if int(args.source_split_seed) != 0 else int(args.seed)

    if source_train or source_test:
        if split_override is not None:
            raise SystemExit("--source-train/--source-test cannot be combined with --kfold.")
        if not source_train or not source_test:
            raise SystemExit("--source-train and --source-test must both be set when using source override.")
        if not (0.0 < source_val_ratio < 0.5):
            raise SystemExit("--source-val-ratio must be in (0,0.5).")

        index_rows = read_jsonl(data_root / "index.jsonl")
        split_override = {}

        for r in index_rows:
            ck = str(r.get("case_key") or "")
            src = str(r.get("source") or "")
            split0 = str(r.get("split") or "unknown")
            if not ck:
                continue
            in_train = src in source_train
            in_test = src in source_test
            if in_train and in_test:
                # In-domain evaluation: respect the repo-provided split (train/val/test).
                split_override[ck] = split0 if split0 in {"train", "val", "test"} else "unknown"
                continue
            if in_train:
                # Train domain: only use train/val from that domain.
                split_override[ck] = split0 if split0 in {"train", "val"} else "unknown"
                continue
            if in_test:
                # Test domain: only evaluate on the official test split to avoid leakage.
                split_override[ck] = "test" if split0 == "test" else "unknown"
                continue
            split_override[ck] = "unknown"

        counts = Counter(split_override.values())
        if counts.get("val", 0) < 3:
            # Fallback: if the filtered source has too few official val samples, create a
            # deterministic val split within the source-train domain.
            def _is_val(case_key: str) -> bool:
                h = zlib.crc32(f"{case_key}|{source_split_seed}".encode("utf-8")) & 0xFFFFFFFF
                u = h / float(2**32)
                return u < float(source_val_ratio)

            for r in index_rows:
                ck = str(r.get("case_key") or "")
                src = str(r.get("source") or "")
                split0 = str(r.get("split") or "unknown")
                if not ck:
                    continue
                if src in source_train and split0 == "train":
                    split_override[ck] = "val" if _is_val(ck) else "train"

        counts = Counter(split_override.values())
        if counts.get("train", 0) < 10:
            raise SystemExit(f"source override produced too few train samples: {counts}")
        if counts.get("val", 0) < 3:
            raise SystemExit(f"source override produced too few val samples: {counts}")
        if counts.get("test", 0) < 3:
            raise SystemExit(f"source override produced too few test samples: {counts}")
        print(
            f"[source_override] train_sources={source_train} test_sources={source_test} "
            f"val_ratio={source_val_ratio:g} split_seed={source_split_seed} counts={dict(counts)}",
            flush=True,
        )

    model_name = str(args.model).strip().lower()
    if model_name in {
        "pointnet",
        "pointnet_cloudset",
        "pointnet_dsbn",
        "pointnet_pos_moe",
        "dgcnn",
        "dgcnn_v2",
        "geom_mlp",
        "cloud_geom_mlp",
    } and int(args.n_points) <= 0:
        raise SystemExit("--n-points must be >0 for pointnet/dgcnn/geom_mlp variants.")
    if model_name == "meta_mlp" and not extra_features:
        raise SystemExit("--extra-features is required for --model meta_mlp.")
    default_name = f"{model_name}_n{args.n_points}_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if split_override is not None:
        default_name = f"{default_name}_k{kfold_k}_fold{kfold_test_fold}"
    exp_name = args.exp_name.strip() or default_name
    out_dir = (args.run_root.resolve() / exp_name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    index_rows_all = read_jsonl(data_root / "index.jsonl")
    if "scale" in extra_features:
        scales = []
        for r in index_rows_all:
            if r.get("split") == "unknown":
                continue
            try:
                scales.append(float(r.get("scale") or 0.0))
            except Exception:
                continue
        if scales:
            s_min = float(min(scales))
            s_max = float(max(scales))
            s_mean = float(sum(scales) / len(scales))
            print(f"[scale] n={len(scales)} min={s_min:.6g} mean={s_mean:.6g} max={s_max:.6g}", flush=True)

    device_str = normalize_device(args.device)
    device = torch.device(device_str)
    deterministic = bool(args.deterministic)
    cudnn_benchmark = bool(args.cudnn_benchmark)
    if deterministic and cudnn_benchmark:
        cudnn_benchmark = False
    set_seed(int(args.seed), deterministic=deterministic, cudnn_benchmark=cudnn_benchmark)
    if device_str == "cuda":
        torch.cuda.reset_peak_memory_stats()

    lr_scheduler = str(args.lr_scheduler or "").strip().lower() or "none"
    if lr_scheduler in {"", "none"}:
        lr_scheduler = "none"
    if lr_scheduler not in {"none", "cosine"}:
        raise SystemExit(f"Unknown --lr-scheduler: {lr_scheduler!r} (supported: none, cosine)")
    warmup_epochs = max(0, int(args.warmup_epochs))
    min_lr = max(0.0, float(args.min_lr))

    ce_weighting = str(args.ce_weighting or "").strip().lower() or "auto"
    if ce_weighting not in {"auto", "none", "inverse_freq"}:
        raise SystemExit(f"Invalid --ce-weighting: {args.ce_weighting!r} (allowed: auto, none, inverse_freq)")
    args.ce_weighting = ce_weighting

    cfg = TrainConfig(
        generated_at=utc_now_iso(),
        seed=int(args.seed),
        device=device_str,
        deterministic=bool(deterministic),
        cudnn_benchmark=bool(cudnn_benchmark),
        data_root=str(data_root),
        out_dir=str(out_dir),
        exp_name=exp_name,
        model=model_name,
        num_classes=len(labels_by_id),
        n_points=int(args.n_points),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        lr_scheduler=str(lr_scheduler),
        warmup_epochs=int(warmup_epochs),
        min_lr=float(min_lr),
        dropout=float(args.dropout),
        grad_clip=float(args.grad_clip),
        log_every=max(0, int(args.log_every)),
        aug_rotate_z=bool(args.aug_rotate_z),
        aug_scale=float(args.aug_scale),
        aug_jitter_sigma=float(args.aug_jitter_sigma),
        aug_jitter_clip=float(args.aug_jitter_clip),
        aug_dropout_ratio=float(args.aug_dropout_ratio),
        num_workers=int(args.num_workers),
        patience=int(args.patience),
        early_stop_min_delta=float(args.early_stop_min_delta),
        save_best_metric=str(args.save_best_metric or "macro_f1_present"),
        extra_features=extra_features,
        extra_mean=[],
        extra_std=[],
        balanced_sampler=bool(args.balanced_sampler),
        label_smoothing=float(args.label_smoothing),
        ce_weighting=str(args.ce_weighting),
        init_feat=init_feat_path,
        freeze_feat_epochs=int(args.freeze_feat_epochs),
        dgcnn_k=int(args.dgcnn_k),
        pointnet2_sa1_npoint=int(args.pointnet2_sa1_npoint),
        pointnet2_sa1_nsample=int(args.pointnet2_sa1_nsample),
        pointnet2_sa2_npoint=int(args.pointnet2_sa2_npoint),
        pointnet2_sa2_nsample=int(args.pointnet2_sa2_nsample),
        pt_dim=int(args.pt_dim),
        pt_depth=int(args.pt_depth),
        pt_k=int(args.pt_k),
        pt_ffn_mult=float(args.pt_ffn_mult),
        pmlp_dim=int(args.pmlp_dim),
        pmlp_depth=int(args.pmlp_depth),
        pmlp_k=int(args.pmlp_k),
        pmlp_ffn_mult=float(args.pmlp_ffn_mult),
        tta=max(0, int(args.tta)),
        kfold_path=str(kfold_path),
        kfold_k=int(kfold_k),
        kfold_test_fold=int(kfold_test_fold),
        kfold_val_fold=int(kfold_val_fold),
        source_train=list(source_train),
        source_test=list(source_test),
        source_val_ratio=float(source_val_ratio) if (source_train or source_test) else 0.0,
        source_split_seed=int(source_split_seed) if (source_train or source_test) else 0,
        input_normalize=str(args.input_normalize or "none"),
        input_pca_align=bool(args.input_pca_align),
        input_pca_align_globalz=bool(args.input_pca_align_globalz),
        point_features=list(point_features)
        if model_name
        in {
            "pointnet",
            "pointnet2",
            "point_transformer",
            "pointmlp",
            "pointnet_cloudattn",
            "pointnet_cloudmil",
            "pointnet_cloudset",
            "pointnet_dsbn",
            "pointnet_pos_moe",
            "dgcnn",
            "cloud_geom_mlp",
        }
        else ["xyz"],
        feature_cache_dir=str(feature_cache_dir)
        if model_name
        in {
            "pointnet",
            "pointnet2",
            "point_transformer",
            "pointmlp",
            "pointnet_cloudattn",
            "pointnet_cloudmil",
            "pointnet_cloudset",
            "pointnet_dsbn",
            "pointnet_pos_moe",
            "dgcnn",
            "cloud_geom_mlp",
        }
        else "",
        feature_k=int(feature_k),
        calibration_bins=int(args.calibration_bins),
        tooth_position_dropout=float(args.tooth_position_dropout),
        supcon_weight=float(args.supcon_weight),
        supcon_temperature=float(args.supcon_temp),
        supcon_proj_dim=int(args.supcon_proj_dim),
        domain_method=str(args.domain_method or ""),
        domain_group_key=str(args.domain_group_key or ""),
        groupdro_eta=float(args.groupdro_eta),
        coral_weight=float(args.coral_weight),
        coral_proj_dim=int(args.coral_proj_dim),
    )

    feature_cache_total = 0
    feature_cache_hits = 0
    feature_cache_misses = 0
    feature_cache_hit_rate: float | None = None
    if bool(args.precompute_features) and model_name in {
        "pointnet",
        "pointnet2",
        "point_transformer",
        "pointmlp",
        "pointnet_cloudattn",
        "pointnet_cloudmil",
        "pointnet_cloudset",
        "pointnet_dsbn",
        "pointnet_pos_moe",
        "dgcnn",
    } and point_features != ["xyz"]:
        in_norm = str(args.input_normalize or "").strip().lower() or "none"
        pca0 = bool(args.input_pca_align)
        pca_gz0 = bool(args.input_pca_align_globalz)
        if not feature_cache_dir:
            raise SystemExit("--precompute-features requires --feature-cache-dir")
        cache_root = Path(feature_cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        rows_all = read_jsonl(data_root / "index.jsonl")
        hits = 0
        misses = 0
        pf_str = ",".join(point_features)
        dim_expected = point_feature_dim(point_features)
        for rr in rows_all:
            rel0 = str(rr.get("sample_npz") or "")
            if not rel0:
                continue
            cache_path = (cache_root / rel0).resolve()
            ok = False
            if cache_path.exists():
                try:
                    with np.load(cache_path) as z:
                        feat0 = np.asarray(z["feat"], dtype=np.float32)
                        pf0 = str(z.get("point_features", "")).strip()
                        k0 = int(np.asarray(z.get("k", 0)).reshape(-1)[0]) if "k" in z else 0
                        n0 = int(np.asarray(z.get("n_points", 0)).reshape(-1)[0]) if "n_points" in z else 0
                        in_norm0 = str(np.asarray(z.get("input_normalize", "")).reshape(-1)[0]) if "input_normalize" in z else ""
                        pca1 = bool(int(np.asarray(z.get("input_pca_align", 0)).reshape(-1)[0])) if "input_pca_align" in z else False
                        pca_gz1 = bool(int(np.asarray(z.get("input_pca_align_globalz", 0)).reshape(-1)[0])) if "input_pca_align_globalz" in z else False
                    ok = (
                        feat0.ndim == 2
                        and feat0.shape[0] == int(args.n_points)
                        and feat0.shape[1] == int(dim_expected)
                        and pf0 == pf_str
                        and k0 == int(feature_k)
                        and n0 == int(args.n_points)
                        and str(in_norm0).strip().lower() == in_norm
                        and bool(pca1) == pca0
                        and bool(pca_gz1) == pca_gz0
                    )
                except Exception:
                    ok = False
            if ok:
                hits += 1
                continue

            npz_path = data_root / rel0
            want_rgb0 = "rgb" in point_features
            want_cid0 = any(pf in point_features for pf in {"cloud_id", "cloud_id_onehot"})
            with np.load(npz_path) as data:
                pts0 = np.asarray(data["points"], dtype=np.float32)
                rgb0 = np.asarray(data["rgb"]) if want_rgb0 and "rgb" in data.files else None
                cid0 = np.asarray(data["cloud_id"]) if want_cid0 and "cloud_id" in data.files else None
            if pts0.ndim != 2 or pts0.shape[1] != 3:
                raise SystemExit(f"Invalid points shape {pts0.shape} in {npz_path}")
            if want_rgb0 and rgb0 is None:
                raise SystemExit(f"Missing `rgb` in {npz_path} (point_features={point_features})")
            if rgb0 is not None and (rgb0.ndim != 2 or rgb0.shape[0] != pts0.shape[0] or rgb0.shape[1] != 3):
                raise SystemExit(f"Invalid rgb shape {None if rgb0 is None else rgb0.shape} in {npz_path} for points {pts0.shape}")
            if want_cid0 and cid0 is None:
                raise SystemExit(f"Missing `cloud_id` in {npz_path} (point_features={point_features})")
            if cid0 is not None and int(np.asarray(cid0).reshape(-1).shape[0]) != int(pts0.shape[0]):
                raise SystemExit(f"Invalid cloud_id shape {None if cid0 is None else np.asarray(cid0).reshape(-1).shape} in {npz_path} for points {pts0.shape}")
            if in_norm not in {"none", "off"} or pca0:
                pts0 = apply_input_preprocess_np(
                    pts0,
                    input_normalize=in_norm,
                    pca_align=pca0,
                    pca_align_globalz=pca_gz0,
                )
            if int(args.n_points) > 0 and pts0.shape[0] != int(args.n_points):
                # Deterministic fallback sampling.
                h = zlib.crc32(str(rr.get("case_key") or rel0).encode("utf-8")) & 0xFFFFFFFF
                tmp_rng = np.random.default_rng(int(h))
                if pts0.shape[0] > int(args.n_points):
                    sel = tmp_rng.choice(pts0.shape[0], size=int(args.n_points), replace=False)
                else:
                    sel = tmp_rng.choice(pts0.shape[0], size=int(args.n_points), replace=True)
                pts0 = pts0[sel]
                if rgb0 is not None:
                    rgb0 = rgb0[sel]
                if cid0 is not None:
                    cid0 = np.asarray(cid0).reshape(-1)[sel]

            feat0 = build_point_features_from_xyz(
                pts0,
                point_features=point_features,
                k=int(feature_k),
                device=device if device.type == "cuda" else torch.device("cpu"),
                rgb_u8_np=rgb0,
                cloud_id_np=cid0,
            )
            atomic_write_npz(
                cache_path,
                arrays={
                    "feat": feat0,
                    "point_features": np.asarray(pf_str),
                    "k": np.asarray(int(feature_k), dtype=np.int32),
                    "n_points": np.asarray(int(args.n_points), dtype=np.int32),
                    "input_normalize": np.asarray(str(in_norm)),
                    "input_pca_align": np.asarray(int(pca0), dtype=np.int32),
                    "input_pca_align_globalz": np.asarray(int(pca_gz0), dtype=np.int32),
                },
            )
            misses += 1
            if (hits + misses) % 50 == 0:
                print(f"[precompute] {hits+misses}/{len(rows_all)} cached (hits={hits}, misses={misses})")
        total = hits + misses
        hr = hits / total if total > 0 else 0.0
        print(f"[precompute] done: total={total} hits={hits} misses={misses} hit_rate={hr:.3f} cache_dir={cache_root}")
        feature_cache_total = int(total)
        feature_cache_hits = int(hits)
        feature_cache_misses = int(misses)
        feature_cache_hit_rate = float(hr)

    train_loader, val_loader, test_loader, train_counts = build_loaders(
        data_root,
        cfg,
        label_to_id,
        split_override=split_override,
    )

    # Update config with feature normalization stats inferred from train split.
    ds0: RawClsDataset = train_loader.dataset  # type: ignore[assignment]
    mean = ds0.extra_mean if ds0.extra_mean is not None else np.zeros((0,), dtype=np.float32)
    std = ds0.extra_std if ds0.extra_std is not None else np.ones((0,), dtype=np.float32)
    cfg = TrainConfig(**{**asdict(cfg), "extra_mean": [float(x) for x in mean.tolist()], "extra_std": [float(x) for x in std.tolist()]})
    write_json(out_dir / "config.json", asdict(cfg))
    write_json(out_dir / "env.json", {"env": get_env_info(), "git": get_git_info(Path.cwd())})
    print(
        "[optim] "
        f"lr={float(cfg.lr)} wd={float(cfg.weight_decay)} "
        f"scheduler={str(cfg.lr_scheduler)} warmup_epochs={int(cfg.warmup_epochs)} min_lr={float(cfg.min_lr)} "
        f"grad_clip={float(cfg.grad_clip)} log_every={int(cfg.log_every)}",
        flush=True,
    )
    print(
        "[runtime] "
        f"deterministic={bool(cfg.deterministic)} cudnn_benchmark={bool(cfg.cudnn_benchmark)}",
        flush=True,
    )

    # Class weights from train split.
    #
    # NOTE: When using `--balanced-sampler`, we already rebalance minibatches by label.
    # Applying inverse-frequency weights *again* in the loss can over-correct and
    # severely hurt recall on the majority class. For journal reporting we prefer
    # one rebalancing mechanism at a time:
    #   - balanced sampler ON  -> unweighted CE (weight=None)
    #   - balanced sampler OFF -> inverse-frequency weighted CE
    loss_weight: torch.Tensor | None = None
    ce_w = str(cfg.ce_weighting or "").strip().lower() or "auto"
    if ce_w not in {"auto", "none", "inverse_freq"}:
        raise SystemExit(f"Invalid ce_weighting in config: {cfg.ce_weighting!r}")
    use_inv_freq = False
    if ce_w == "inverse_freq":
        use_inv_freq = True
    elif ce_w == "none":
        use_inv_freq = False
    else:
        use_inv_freq = not bool(cfg.balanced_sampler)

    if bool(use_inv_freq):
        weights = torch.ones(cfg.num_classes, dtype=torch.float32)
        total_train = sum(train_counts.values()) if train_counts else 0
        for class_id in range(cfg.num_classes):
            cnt = int(train_counts.get(class_id, 0))
            if cnt <= 0 or total_train <= 0:
                weights[class_id] = 0.0
            else:
                weights[class_id] = float(total_train) / float(cfg.num_classes * cnt)
        loss_weight = weights.to(device)

    if cfg.model == "pointnet":
        in_ch = point_feature_dim(list(cfg.point_features or ["xyz"]))
        model = PointNetClassifier(
            num_classes=cfg.num_classes,
            dropout=cfg.dropout,
            extra_dim=len(cfg.extra_features),
            in_channels=in_ch,
        )
    elif cfg.model == "pointnet2":
        in_ch = point_feature_dim(list(cfg.point_features or ["xyz"]))
        model = PointNet2Classifier(
            num_classes=int(cfg.num_classes),
            dropout=float(cfg.dropout),
            extra_dim=len(cfg.extra_features),
            in_channels=int(in_ch),
            sa1_npoint=int(cfg.pointnet2_sa1_npoint),
            sa1_nsample=int(cfg.pointnet2_sa1_nsample),
            sa2_npoint=int(cfg.pointnet2_sa2_npoint),
            sa2_nsample=int(cfg.pointnet2_sa2_nsample),
        )
    elif cfg.model == "point_transformer":
        in_ch = point_feature_dim(list(cfg.point_features or ["xyz"]))
        params = PointTransformerParams(
            dim=int(cfg.pt_dim),
            depth=int(cfg.pt_depth),
            k=int(cfg.pt_k),
            ffn_mult=float(cfg.pt_ffn_mult),
        )
        model = PointTransformerClassifier(
            num_classes=int(cfg.num_classes),
            dropout=float(cfg.dropout),
            extra_dim=len(cfg.extra_features),
            in_channels=int(in_ch),
            params=params,
        )
    elif cfg.model == "pointmlp":
        in_ch = point_feature_dim(list(cfg.point_features or ["xyz"]))
        params = PointMLPParams(
            dim=int(cfg.pmlp_dim),
            depth=int(cfg.pmlp_depth),
            k=int(cfg.pmlp_k),
            ffn_mult=float(cfg.pmlp_ffn_mult),
        )
        model = PointMLPClassifier(
            num_classes=int(cfg.num_classes),
            dropout=float(cfg.dropout),
            extra_dim=len(cfg.extra_features),
            in_channels=int(in_ch),
            params=params,
        )
    elif cfg.model == "pointnet_cloudset":
        pf = list(cfg.point_features or ["xyz"])
        if "cloud_id" not in pf:
            raise SystemExit("pointnet_cloudset requires point_features to include `cloud_id`.")
        if "cloud_id_onehot" in pf:
            raise SystemExit("pointnet_cloudset only supports scalar cloud_id (not onehot).")
        # Locate the cloud_id channel offset and derive stem input channels (excluding cloud_id).
        off = 0
        cid_idx: int | None = None
        for name in pf:
            if name == "xyz":
                off += 3
            elif name == "normals":
                off += 3
            elif name == "curvature":
                off += 1
            elif name == "radius":
                off += 1
            elif name == "rgb":
                off += 3
            elif name == "cloud_id":
                cid_idx = off
                off += 1
            elif name == "cloud_id_onehot":
                raise SystemExit("pointnet_cloudset only supports scalar cloud_id (not onehot).")
            else:
                raise SystemExit(f"Unknown point feature: {name}")
        if cid_idx is None:
            raise SystemExit("pointnet_cloudset internal error: cloud_id index not found.")
        total_ch = point_feature_dim(pf)
        stem_ch = int(total_ch) - 1
        if stem_ch <= 0:
            raise SystemExit(f"Invalid stem channels for pointnet_cloudset: total={total_ch} stem={stem_ch}")
        model = PointNetCloudSetClassifier(
            num_classes=cfg.num_classes,
            dropout=cfg.dropout,
            extra_dim=len(cfg.extra_features),
            in_channels=int(stem_ch),
            cloud_id_index=int(cid_idx),
            max_clouds=10,
        )
    elif cfg.model == "pointnet_cloudattn":
        pf = list(cfg.point_features or ["xyz"])
        if "cloud_id" not in pf:
            raise SystemExit("pointnet_cloudattn requires point_features to include `cloud_id`.")
        if "cloud_id_onehot" in pf:
            raise SystemExit("pointnet_cloudattn only supports scalar cloud_id (not onehot).")
        # Locate the cloud_id channel offset and derive stem input channels (excluding cloud_id).
        off = 0
        cid_idx: int | None = None
        for name in pf:
            if name == "xyz":
                off += 3
            elif name == "normals":
                off += 3
            elif name == "curvature":
                off += 1
            elif name == "radius":
                off += 1
            elif name == "rgb":
                off += 3
            elif name == "cloud_id":
                cid_idx = off
                off += 1
            elif name == "cloud_id_onehot":
                raise SystemExit("pointnet_cloudattn only supports scalar cloud_id (not onehot).")
            else:
                raise SystemExit(f"Unknown point feature: {name}")
        if cid_idx is None:
            raise SystemExit("pointnet_cloudattn internal error: cloud_id index not found.")
        total_ch = point_feature_dim(pf)
        stem_ch = int(total_ch) - 1
        if stem_ch <= 0:
            raise SystemExit(f"Invalid stem channels for pointnet_cloudattn: total={total_ch} stem={stem_ch}")
        model = PointNetCloudAttnClassifier(
            num_classes=cfg.num_classes,
            dropout=cfg.dropout,
            extra_dim=len(cfg.extra_features),
            in_channels=int(stem_ch),
            cloud_id_index=int(cid_idx),
            max_clouds=10,
        )
    elif cfg.model == "pointnet_cloudmil":
        pf = list(cfg.point_features or ["xyz"])
        if "cloud_id" not in pf:
            raise SystemExit("pointnet_cloudmil requires point_features to include `cloud_id`.")
        if "cloud_id_onehot" in pf:
            raise SystemExit("pointnet_cloudmil only supports scalar cloud_id (not onehot).")
        off = 0
        cid_idx: int | None = None
        for name in pf:
            if name == "xyz":
                off += 3
            elif name == "normals":
                off += 3
            elif name == "curvature":
                off += 1
            elif name == "radius":
                off += 1
            elif name == "rgb":
                off += 3
            elif name == "cloud_id":
                cid_idx = off
                off += 1
            elif name == "cloud_id_onehot":
                raise SystemExit("pointnet_cloudmil only supports scalar cloud_id (not onehot).")
            else:
                raise SystemExit(f"Unknown point feature: {name}")
        if cid_idx is None:
            raise SystemExit("pointnet_cloudmil internal error: cloud_id index not found.")
        total_ch = point_feature_dim(pf)
        stem_ch = int(total_ch) - 1
        if stem_ch <= 0:
            raise SystemExit(f"Invalid stem channels for pointnet_cloudmil: total={total_ch} stem={stem_ch}")
        model = PointNetCloudMILClassifier(
            num_classes=cfg.num_classes,
            dropout=cfg.dropout,
            extra_dim=len(cfg.extra_features),
            in_channels=int(stem_ch),
            cloud_id_index=int(cid_idx),
            max_clouds=10,
        )
    elif cfg.model == "pointnet_dsbn":
        in_ch = point_feature_dim(list(cfg.point_features or ["xyz"]))
        model = PointNetDSBNClassifier(
            num_classes=cfg.num_classes,
            dropout=cfg.dropout,
            extra_dim=len(cfg.extra_features),
            in_channels=in_ch,
            num_domains=3,
        )
    elif cfg.model == "pointnet_pos_moe":
        in_ch = point_feature_dim(list(cfg.point_features or ["xyz"]))
        feat_names = list(cfg.extra_features or [])
        if "tooth_position_missing" not in feat_names:
            raise SystemExit("pointnet_pos_moe requires --extra-features to include tooth_position_missing.")
        missing_idx = feat_names.index("tooth_position_missing")
        model = PointNetPosMoEClassifier(
            num_classes=cfg.num_classes,
            dropout=cfg.dropout,
            extra_dim=len(feat_names),
            missing_index=int(missing_idx),
            in_channels=in_ch,
        )
    elif cfg.model == "dgcnn":
        in_ch = point_feature_dim(list(cfg.point_features or ["xyz"]))
        model = DGCNNClassifier(
            num_classes=cfg.num_classes,
            dropout=cfg.dropout,
            k=int(cfg.dgcnn_k),
            extra_dim=len(cfg.extra_features),
            in_channels=in_ch,
        )
    elif cfg.model == "dgcnn_v2":
        in_ch = point_feature_dim(list(cfg.point_features or ["xyz"]))
        model = DGCNNv2Classifier(
            num_classes=cfg.num_classes,
            dropout=cfg.dropout,
            extra_dim=len(cfg.extra_features),
            in_channels=in_ch,
            params=DGCNNv2Params(k=int(cfg.dgcnn_k)),
        )
    elif cfg.model == "meta_mlp":
        model = MetaMLPClassifier(num_classes=cfg.num_classes, dropout=cfg.dropout, extra_dim=len(cfg.extra_features))
    elif cfg.model == "geom_mlp":
        model = GeomMLPClassifier(num_classes=cfg.num_classes, dropout=cfg.dropout, extra_dim=len(cfg.extra_features))
    elif cfg.model == "cloud_geom_mlp":
        pf = list(cfg.point_features or ["xyz"])
        if "cloud_id" not in pf:
            raise SystemExit("cloud_geom_mlp requires point_features to include `cloud_id`.")
        # Compute feature offset to locate the cloud_id scalar channel.
        off = 0
        cid_idx: int | None = None
        for name in pf:
            if name == "xyz":
                off += 3
            elif name == "normals":
                off += 3
            elif name == "curvature":
                off += 1
            elif name == "radius":
                off += 1
            elif name == "rgb":
                off += 3
            elif name == "cloud_id":
                cid_idx = off
                off += 1
            elif name == "cloud_id_onehot":
                raise SystemExit("cloud_geom_mlp only supports scalar cloud_id (not onehot).")
            else:
                raise SystemExit(f"Unknown point feature: {name}")
        if cid_idx is None:
            raise SystemExit("cloud_geom_mlp internal error: cloud_id index not found.")
        model = CloudGeomMLPClassifier(
            num_classes=cfg.num_classes,
            dropout=cfg.dropout,
            extra_dim=len(cfg.extra_features),
            cloud_id_index=int(cid_idx),
            max_clouds=10,
        )
    else:
        raise SystemExit(f"Unknown model: {cfg.model}")
    model = model.to(device)

    if (
        cfg.model
        in {"pointnet", "pointnet2", "point_transformer", "pointmlp", "pointnet_cloudset", "pointnet_pos_moe", "dgcnn_v2"}
        and cfg.init_feat
    ):
        ckpt = torch.load(Path(cfg.init_feat), map_location="cpu")
        state = ckpt.get("model_state") or ckpt.get("model") or ckpt
        feat_state = {k[len("feat.") :]: v for k, v in state.items() if isinstance(k, str) and k.startswith("feat.")}
        if not feat_state:
            raise SystemExit(f"No 'feat.*' keys found in init checkpoint: {cfg.init_feat}")
        if not hasattr(model, "feat"):
            raise SystemExit(f"--init-feat requires model to have `.feat` (model={cfg.model})")
        feat_module = getattr(model, "feat")
        if not isinstance(feat_module, nn.Module):
            raise SystemExit(f"--init-feat requires model.feat to be an nn.Module (model={cfg.model})")
        target = feat_module.state_dict()
        # Allow safe init when point feature dim changes (e.g. xyz -> xyz+normals+curv+radius).
        # Only adapt the first conv weight by padding/truncation; keep other layers strict.
        def _adapt_first_conv_weight(
            state_in: dict[str, torch.Tensor],
            state_tgt: dict[str, torch.Tensor],
            key: str,
        ) -> dict[str, torch.Tensor]:
            if key not in state_in or key not in state_tgt:
                return state_in
            w_src = state_in[key]
            w_tgt = state_tgt[key]
            if not (isinstance(w_src, torch.Tensor) and isinstance(w_tgt, torch.Tensor)):
                return state_in
            if tuple(w_src.shape) == tuple(w_tgt.shape):
                return state_in
            if w_src.ndim != w_tgt.ndim or w_src.shape[0] != w_tgt.shape[0] or tuple(w_src.shape[2:]) != tuple(w_tgt.shape[2:]):
                raise SystemExit(
                    f"init_feat first conv weight shape mismatch for {key}: src={tuple(w_src.shape)} tgt={tuple(w_tgt.shape)}"
                )
            new_w = w_tgt.clone()
            c = int(min(w_src.shape[1], w_tgt.shape[1]))
            new_w[:, :c, ...] = w_src[:, :c, ...]
            if int(w_tgt.shape[1]) > int(w_src.shape[1]) and int(w_src.shape[1]) > 0:
                fill = w_src.mean(dim=1, keepdim=True)
                reps = [1] * int(w_src.ndim)
                reps[1] = int(w_tgt.shape[1]) - c
                new_w[:, c:, ...] = fill.repeat(*reps)
            state_out = dict(state_in)
            state_out[key] = new_w
            print(
                f"[init-feat] adapted {key}: src={tuple(w_src.shape)} tgt={tuple(w_tgt.shape)} copy={c} fill=mean",
                flush=True,
            )
            return state_out

        # Known "first conv" keys for different backbones.
        for w_key in ("0.weight", "stem.0.weight", "sa1.mlp_convs.0.weight", "conv1.0.weight"):
            feat_state = _adapt_first_conv_weight(feat_state, target, w_key)
        feat_module.load_state_dict(feat_state, strict=True)

    domain_method = str(cfg.domain_method or "").strip().lower() or "baseline"
    domain_group_key = str(cfg.domain_group_key or "").strip() or "tooth_position"
    if domain_method not in {"baseline", "groupdro", "coral", "dsbn", "pos_moe"}:
        raise SystemExit(f"Unknown --domain-method: {domain_method!r}")
    if domain_method == "dsbn" and cfg.model != "pointnet_dsbn":
        raise SystemExit("--domain-method=dsbn requires --model=pointnet_dsbn.")
    if domain_method == "pos_moe" and cfg.model != "pointnet_pos_moe":
        raise SystemExit("--domain-method=pos_moe requires --model=pointnet_pos_moe.")
    if float(cfg.supcon_weight) > 0 and domain_method != "baseline":
        raise SystemExit("SupCon cannot be combined with domain_method != baseline in this script.")

    print(f"[domain] method={domain_method} group_key={domain_group_key}", flush=True)
    supcon_aug_rotate_z = bool(cfg.aug_rotate_z)
    if str(args.supcon_aug_rotate_z or "").strip():
        supcon_aug_rotate_z = _parse_bool(str(args.supcon_aug_rotate_z), default=supcon_aug_rotate_z)
    supcon_aug_scale = float(cfg.aug_scale) if args.supcon_aug_scale is None else float(args.supcon_aug_scale)
    supcon_aug_jitter_sigma1 = float(cfg.aug_jitter_sigma) if args.supcon_aug_jitter_sigma1 is None else float(args.supcon_aug_jitter_sigma1)
    supcon_aug_jitter_sigma2 = float(cfg.aug_jitter_sigma) if args.supcon_aug_jitter_sigma2 is None else float(args.supcon_aug_jitter_sigma2)
    supcon_aug_jitter_clip1 = float(cfg.aug_jitter_clip) if args.supcon_aug_jitter_clip1 is None else float(args.supcon_aug_jitter_clip1)
    supcon_aug_jitter_clip2 = float(cfg.aug_jitter_clip) if args.supcon_aug_jitter_clip2 is None else float(args.supcon_aug_jitter_clip2)
    supcon_aug_dropout_ratio1 = float(cfg.aug_dropout_ratio) if args.supcon_aug_dropout_ratio1 is None else float(args.supcon_aug_dropout_ratio1)
    supcon_aug_dropout_ratio2 = float(cfg.aug_dropout_ratio) if args.supcon_aug_dropout_ratio2 is None else float(args.supcon_aug_dropout_ratio2)
    if float(cfg.supcon_weight) > 0:
        print(
            "[supcon] "
            f"weight={float(cfg.supcon_weight)} temp={float(cfg.supcon_temperature)} proj_dim={int(cfg.supcon_proj_dim)}; "
            f"rotate_z={bool(supcon_aug_rotate_z)} scale={float(supcon_aug_scale)}; "
            f"view1(jitter_sigma={float(supcon_aug_jitter_sigma1)} jitter_clip={float(supcon_aug_jitter_clip1)} dropout_ratio={float(supcon_aug_dropout_ratio1)}), "
            f"view2(jitter_sigma={float(supcon_aug_jitter_sigma2)} jitter_clip={float(supcon_aug_jitter_clip2)} dropout_ratio={float(supcon_aug_dropout_ratio2)})",
            flush=True,
        )

    # SupCon view augmentations operate on the point tensor directly. Use dataset-provided
    # feature slices so we can correctly transform xyz/normals/radius without assuming an order.
    ds_sup = getattr(train_loader, "dataset", None)
    while ds_sup is not None and not hasattr(ds_sup, "_slice_xyz") and hasattr(ds_sup, "dataset"):
        ds_sup = getattr(ds_sup, "dataset")
    supcon_slice_xyz = getattr(ds_sup, "_slice_xyz", None)
    if not isinstance(supcon_slice_xyz, slice):
        supcon_slice_xyz = slice(0, 3)
    supcon_slice_normals = getattr(ds_sup, "_slice_normals", None)
    if not isinstance(supcon_slice_normals, slice):
        supcon_slice_normals = None
    supcon_idx_curv = getattr(ds_sup, "_idx_curv", None)
    if not isinstance(supcon_idx_curv, int):
        supcon_idx_curv = None
    supcon_idx_radius = getattr(ds_sup, "_idx_radius", None)
    if not isinstance(supcon_idx_radius, int):
        supcon_idx_radius = None

    feature_dim = int(getattr(model, "feature_dim", 0) or 0)
    if feature_dim <= 0:
        raise SystemExit("Internal error: model.feature_dim missing/invalid.")

    supcon_proj: nn.Module | None = None
    if float(cfg.supcon_weight) > 0:
        supcon_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, int(cfg.supcon_proj_dim), bias=False),
        ).to(device)

    coral_w = float(cfg.coral_weight) if domain_method == "coral" else 0.0
    coral_proj: nn.Module | None = None
    if coral_w > 0:
        coral_dim = int(cfg.coral_proj_dim)
        if coral_dim <= 0:
            raise SystemExit("--coral-proj-dim must be >0.")
        coral_proj = nn.Linear(feature_dim, coral_dim, bias=False).to(device)

    opt_params = list(model.parameters())
    if supcon_proj is not None:
        opt_params += list(supcon_proj.parameters())
    if coral_proj is not None:
        opt_params += list(coral_proj.parameters())
    opt = torch.optim.AdamW(opt_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    try:
        loss_fn: Any = nn.CrossEntropyLoss(weight=loss_weight, label_smoothing=max(0.0, float(cfg.label_smoothing)))
    except TypeError:
        loss_fn = nn.CrossEntropyLoss(weight=loss_weight)

    best_val = -1.0
    best_epoch = -1
    best_path = out_dir / "model_best.pt"
    history_path = out_dir / "history.jsonl"
    epochs_no_improve = 0
    groupdro_w: torch.Tensor | None = None
    if domain_method == "groupdro":
        groupdro_w = torch.ones((3,), device=device, dtype=torch.float32)
        groupdro_w = groupdro_w / groupdro_w.sum()

    with history_path.open("w", encoding="utf-8") as hist_f:
        for epoch in range(1, cfg.epochs + 1):
            # Manual LR scheduling (keeps config-driven behavior explicit and reproducible).
            base_lr = float(cfg.lr)
            cur_lr = base_lr
            if str(cfg.lr_scheduler or "").strip().lower() == "cosine":
                min_lr_eff = float(min(float(cfg.min_lr), base_lr))
                warm = int(max(0, int(cfg.warmup_epochs)))
                if warm > 0 and epoch <= warm:
                    cur_lr = base_lr * (float(epoch) / float(max(1, warm)))
                else:
                    denom = max(1, int(cfg.epochs) - warm)
                    t = float(epoch - warm) / float(denom)
                    t = min(1.0, max(0.0, t))
                    cur_lr = min_lr_eff + (base_lr - min_lr_eff) * 0.5 * (1.0 + math.cos(math.pi * t))
            for pg in opt.param_groups:
                pg["lr"] = float(cur_lr)

            if cfg.model == "pointnet":
                cast_model: PointNetClassifier = model  # type: ignore[assignment]
                freeze = cfg.freeze_feat_epochs > 0 and epoch <= int(cfg.freeze_feat_epochs)
                for p in cast_model.feat.parameters():
                    p.requires_grad = not freeze
            elif cfg.model == "pointnet2":
                cast_model: PointNet2Classifier = model  # type: ignore[assignment]
                freeze = cfg.freeze_feat_epochs > 0 and epoch <= int(cfg.freeze_feat_epochs)
                for p in cast_model.feat.parameters():
                    p.requires_grad = not freeze

            model.train()
            loss_sum = 0.0
            ce_sum = 0.0
            sup_sum = 0.0
            coral_sum = 0.0
            n_seen = 0
            n_correct = 0
            tp_valid_seen = 0
            tp_dropped_seen = 0
            tp_missing_seen = 0
            domain_counts = [0, 0, 0]  # premolar/molar/missing when group_key=tooth_position
            for step_i, (points, extra, labels, meta) in enumerate(train_loader, start=1):
                points = points.to(device=device, dtype=torch.float32, non_blocking=True)
                extra = extra.to(device=device, dtype=torch.float32, non_blocking=True)
                labels = labels.to(device=device, dtype=torch.long, non_blocking=True)
                domains = torch.as_tensor(domain_group_ids(list(meta), group_key=domain_group_key), device=device, dtype=torch.long)

                opt.zero_grad(set_to_none=True)

                bsz = int(labels.shape[0])
                tp_valid_seen += sum(1 for m in meta if (m.get("tooth_position_raw") or "") in {"前磨牙", "磨牙"})
                tp_dropped_seen += sum(1 for m in meta if bool(m.get("tooth_position_dropped") or False))
                tp_missing_seen += sum(1 for m in meta if not str(m.get("tooth_position") or ""))
                try:
                    dc = torch.bincount(domains.detach().cpu(), minlength=3)
                    domain_counts[0] += int(dc[0].item())
                    domain_counts[1] += int(dc[1].item())
                    domain_counts[2] += int(dc[2].item())
                except Exception:
                    pass

                ce: torch.Tensor
                sup: torch.Tensor | None = None
                coral_term: torch.Tensor | None = None

                if float(cfg.supcon_weight) > 0:
                    assert supcon_proj is not None
                    v1 = augment_points_torch(
                        points,
                        rotate_z=bool(supcon_aug_rotate_z),
                        scale=float(supcon_aug_scale),
                        jitter_sigma=float(supcon_aug_jitter_sigma1),
                        jitter_clip=float(supcon_aug_jitter_clip1),
                        dropout_ratio=float(supcon_aug_dropout_ratio1),
                        gen=None,
                        slice_xyz=supcon_slice_xyz,
                        slice_normals=supcon_slice_normals,
                        idx_curv=supcon_idx_curv,
                        idx_radius=supcon_idx_radius,
                    )
                    v2 = augment_points_torch(
                        points,
                        rotate_z=bool(supcon_aug_rotate_z),
                        scale=float(supcon_aug_scale),
                        jitter_sigma=float(supcon_aug_jitter_sigma2),
                        jitter_clip=float(supcon_aug_jitter_clip2),
                        dropout_ratio=float(supcon_aug_dropout_ratio2),
                        gen=None,
                        slice_xyz=supcon_slice_xyz,
                        slice_normals=supcon_slice_normals,
                        idx_curv=supcon_idx_curv,
                        idx_radius=supcon_idx_radius,
                    )
                    f1 = model.forward_features(v1, extra, domains=domains)  # type: ignore[attr-defined]
                    f2 = model.forward_features(v2, extra, domains=domains)  # type: ignore[attr-defined]
                    logits1 = model.forward_logits_from_features(f1)  # type: ignore[attr-defined]
                    logits2 = model.forward_logits_from_features(f2)  # type: ignore[attr-defined]
                    ce = 0.5 * (loss_fn(logits1, labels) + loss_fn(logits2, labels))
                    z = torch.cat([supcon_proj(f1), supcon_proj(f2)], dim=0)
                    y2 = torch.cat([labels, labels], dim=0)
                    sup = supervised_contrastive_loss(z, y2, temperature=float(cfg.supcon_temperature))
                    loss = ce + float(cfg.supcon_weight) * sup
                    logits = 0.5 * (logits1 + logits2)
                else:
                    logits = model(points, extra, domains=domains)  # type: ignore[call-arg]
                    if domain_method == "groupdro":
                        assert groupdro_w is not None
                        ce_per = cross_entropy_per_sample(
                            logits,
                            labels,
                            weight=loss_weight,
                            label_smoothing=float(cfg.label_smoothing),
                        )
                        present: list[int] = []
                        group_losses: list[torch.Tensor] = []
                        for g in range(int(groupdro_w.shape[0])):
                            mask = domains == int(g)
                            if torch.any(mask):
                                gl = ce_per[mask].mean()
                                group_losses.append(gl)
                                present.append(int(g))
                                groupdro_w[int(g)] = groupdro_w[int(g)] * torch.exp(float(cfg.groupdro_eta) * gl.detach())
                        groupdro_w = groupdro_w / groupdro_w.sum()
                        if present:
                            idx = torch.as_tensor(present, device=device, dtype=torch.long)
                            w_present = groupdro_w[idx]
                            w_present = w_present / w_present.sum()
                            ce = (torch.stack(group_losses, dim=0) * w_present).sum()
                        else:
                            ce = ce_per.mean()
                        loss = ce
                    else:
                        ce = loss_fn(logits, labels)
                        loss = ce

                    if coral_w > 0 and coral_proj is not None:
                        feats = model.forward_features(points, extra, domains=domains)  # type: ignore[attr-defined]
                        z = coral_proj(feats)
                        coral_term = coral_loss_between_groups(z, domains, g0=0, g1=1)
                        loss = loss + coral_w * coral_term

                loss.backward()
                if float(cfg.grad_clip) > 0:
                    torch.nn.utils.clip_grad_norm_(opt_params, max_norm=float(cfg.grad_clip))
                opt.step()

                loss_sum += float(loss.detach().cpu().item()) * bsz
                ce_sum += float(ce.detach().cpu().item()) * bsz
                if sup is not None:
                    sup_sum += float(sup.detach().cpu().item()) * bsz
                if coral_term is not None:
                    coral_sum += float(coral_term.detach().cpu().item()) * bsz
                n_seen += bsz
                pred = torch.argmax(logits, dim=1)
                n_correct += int((pred == labels).sum().detach().cpu().item())

                if int(cfg.log_every) > 0 and (step_i % int(cfg.log_every) == 0):
                    print(
                        f"[step] epoch={epoch:03d} step={step_i:04d} "
                        f"loss={float(loss.detach().cpu().item()):.4f} lr={float(cur_lr):.6g}",
                        flush=True,
                    )

            train_loss = loss_sum / max(1, n_seen)
            train_loss_ce = ce_sum / max(1, n_seen)
            train_loss_supcon = sup_sum / max(1, n_seen)
            train_loss_coral = coral_sum / max(1, n_seen)
            train_acc = n_correct / max(1, n_seen)
            tp_drop_ratio = float(tp_dropped_seen) / float(max(1, tp_valid_seen))
            tp_missing_ratio = float(tp_missing_seen) / float(max(1, n_seen))

            _, cm_val, _rows_val = evaluate(
                model,
                val_loader,
                device,
                num_classes=cfg.num_classes,
                domain_group_key=domain_group_key,
            )
            val_metrics = metrics_from_confusion(cm_val, labels_by_id)
            metric_name = str(cfg.save_best_metric or "macro_f1_present").strip()
            if metric_name not in val_metrics:
                metric_name = "macro_f1_present"
            val_score = float(val_metrics.get(metric_name) or 0.0)

            row = {
                "epoch": epoch,
                "lr": float(cur_lr),
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_loss_ce": train_loss_ce,
                "train_loss_supcon": train_loss_supcon,
                "train_loss_coral": train_loss_coral,
                "groupdro_weights": (groupdro_w.detach().cpu().tolist() if groupdro_w is not None else None),
                "domain_counts": {"premolar": domain_counts[0], "molar": domain_counts[1], "missing": domain_counts[2]},
                "tooth_position_valid": int(tp_valid_seen),
                "tooth_position_dropped": int(tp_dropped_seen),
                "tooth_position_dropped_ratio": float(tp_drop_ratio),
                "tooth_position_missing": int(tp_missing_seen),
                "tooth_position_missing_ratio": float(tp_missing_ratio),
                "val_score_metric": metric_name,
                "val_score": float(val_score),
                "val": val_metrics,
            }
            hist_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            hist_f.flush()

            min_delta = max(0.0, float(cfg.early_stop_min_delta))
            improved = val_score > best_val + max(1e-12, min_delta)
            if improved:
                best_val = val_score
                best_epoch = epoch
                epochs_no_improve = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": opt.state_dict(),
                        "labels_by_id": labels_by_id,
                        "config": asdict(cfg),
                        "val_metrics": val_metrics,
                    },
                    best_path,
                )
                write_json(out_dir / "best_val_metrics.json", val_metrics)
                save_confusion_csv(cm_val, labels_by_id, out_dir / "confusion_val.csv")
            else:
                epochs_no_improve += 1

            extra_bits: list[str] = []
            if float(cfg.supcon_weight) > 0:
                extra_bits.append(f"ce={train_loss_ce:.4f} sup={train_loss_supcon:.4f}")
            if coral_w > 0:
                extra_bits.append(f"coral={train_loss_coral:.4f}")
            if groupdro_w is not None:
                w0, w1, w2 = [float(x) for x in groupdro_w.detach().cpu().tolist()]
                extra_bits.append(f"w=[{w0:.3f},{w1:.3f},{w2:.3f}]")
            if float(cfg.tooth_position_dropout) > 0 or domain_method == "pos_moe":
                extra_bits.append(f"tp_drop={tp_drop_ratio:.3f} tp_missing={tp_missing_ratio:.3f}")

            extra_str = (" " + " ".join(extra_bits)) if extra_bits else ""
            print(
                f"[epoch {epoch:03d}] train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
                f"val_{metric_name}={val_score:.3f} {'*' if improved else ''}{extra_str}"
            )

            if cfg.patience > 0 and epochs_no_improve >= cfg.patience:
                print(f"[early-stop] no improvement for {cfg.patience} epochs (best_epoch={best_epoch})")
                break

    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])

    _pred_val, cm_val, rows_val = evaluate(
        model,
        val_loader,
        device,
        num_classes=cfg.num_classes,
        tta=int(cfg.tta),
        tta_seed=int(cfg.seed) + 123,
        domain_group_key=domain_group_key,
    )
    _pred_test, cm_test, rows_test = evaluate(
        model,
        test_loader,
        device,
        num_classes=cfg.num_classes,
        tta=int(cfg.tta),
        tta_seed=int(cfg.seed) + 456,
        domain_group_key=domain_group_key,
    )

    val_metrics = metrics_from_confusion(cm_val, labels_by_id)
    test_metrics = metrics_from_confusion(cm_test, labels_by_id)

    save_confusion_csv(cm_val, labels_by_id, out_dir / "confusion_val_final.csv")
    save_confusion_csv(cm_test, labels_by_id, out_dir / "confusion_test.csv")
    save_errors_csv(rows_test, labels_by_id, out_dir / "errors_test.csv")
    write_jsonl(out_dir / "preds_val.jsonl", rows_val)
    write_jsonl(out_dir / "preds_test.jsonl", rows_test)

    by_source = {}
    for src in sorted({str(r.get("source") or "") for r in rows_test}):
        by_source[src or "(missing)"] = metrics_for_rows(
            [r for r in rows_test if str(r.get("source") or "") == src],
            labels_by_id,
        )

    by_tooth_pos = {}
    for tp in sorted({str(r.get("tooth_position") or "") for r in rows_test}):
        by_tooth_pos[tp or "(missing)"] = metrics_for_rows(
            [r for r in rows_test if str(r.get("tooth_position") or "") == tp],
            labels_by_id,
        )

    cal_val = calibration_basic(rows_val, cfg.num_classes, n_bins=int(cfg.calibration_bins))
    cal_test = calibration_basic(rows_test, cfg.num_classes, n_bins=int(cfg.calibration_bins))
    cal_by_source = {}
    for src in sorted({str(r.get("source") or "") for r in rows_test}):
        cal_by_source[src or "(missing)"] = calibration_basic(
            [r for r in rows_test if str(r.get("source") or "") == src],
            cfg.num_classes,
            n_bins=int(cfg.calibration_bins),
        )

    wall_time_sec = float(max(0.0, time.time() - float(t0)))
    runtime: dict[str, Any] = {
        "wall_time_sec": wall_time_sec,
        "wall_time_hr": wall_time_sec / 3600.0,
    }
    if device_str == "cuda":
        try:
            runtime.update(
                {
                    "cuda_device_name": str(torch.cuda.get_device_name(torch.cuda.current_device())),
                    "cuda_peak_allocated_mb": float(torch.cuda.max_memory_allocated() / (1024.0**2)),
                    "cuda_peak_reserved_mb": float(torch.cuda.max_memory_reserved() / (1024.0**2)),
                }
            )
        except Exception:
            pass

    summary = {
        "generated_at": utc_now_iso(),
        "best_epoch": best_epoch,
        "val": val_metrics,
        "val_calibration": cal_val,
        "test": test_metrics,
        "test_by_source": by_source,
        "test_by_tooth_position": by_tooth_pos,
        "test_calibration": cal_test,
        "test_by_source_calibration": cal_by_source,
        "runtime": runtime,
        "sanity": {
            "balanced_sampler": bool(cfg.balanced_sampler),
            "loss_weighted": bool(loss_weight is not None),
            "point_features": list(cfg.point_features or []),
            "feature_k": int(cfg.feature_k),
            "feature_cache_dir": str(cfg.feature_cache_dir or ""),
            "feature_cache_total": int(feature_cache_total),
            "feature_cache_hits": int(feature_cache_hits),
            "feature_cache_misses": int(feature_cache_misses),
            "feature_cache_hit_rate": (float(feature_cache_hit_rate) if feature_cache_hit_rate is not None else None),
            "calibration_bins": int(cfg.calibration_bins),
            "tooth_position_dropout": float(cfg.tooth_position_dropout),
            "source_train": list(cfg.source_train or []),
            "source_test": list(cfg.source_test or []),
            "source_val_ratio": float(cfg.source_val_ratio),
            "source_split_seed": int(cfg.source_split_seed),
            "init_feat_path": str(init_feat_path or ""),
            "init_feat_sha1": str(init_feat_sha1 or ""),
            "init_feat_bytes": int(init_feat_bytes),
            "lr_scheduler": str(cfg.lr_scheduler),
            "warmup_epochs": int(cfg.warmup_epochs),
            "min_lr": float(cfg.min_lr),
            "grad_clip": float(cfg.grad_clip),
            "log_every": int(cfg.log_every),
            "deterministic": bool(cfg.deterministic),
            "cudnn_benchmark": bool(cfg.cudnn_benchmark),
            "domain_method": str(domain_method),
            "domain_group_key": str(domain_group_key),
            "supcon_weight": float(cfg.supcon_weight),
            "coral_weight": float(coral_w),
            "groupdro_eta": float(cfg.groupdro_eta),
            "groupdro_weights_end": (groupdro_w.detach().cpu().tolist() if groupdro_w is not None else None),
        },
        "notes": [
            "macro_f1_present averages classes with support>0 in that split.",
            "If a class has zero support in a split, it is excluded from macro_f1_present/balanced_accuracy_present.",
        ],
    }
    write_json(out_dir / "metrics.json", summary)

    print(f"[OK] run_dir: {out_dir}")
    print(f"[OK] wrote: {out_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
