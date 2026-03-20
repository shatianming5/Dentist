#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from _lib.device import normalize_device
from _lib.env import get_env_info
from _lib.git import get_git_info
from _lib.io import read_json, read_jsonl, write_json, write_jsonl
from _lib.seed import set_seed
from _lib.time import utc_now_iso
from phase2_build_teeth3ds_teeth import normalize_points, pca_align
from raw_cls.eval import calibration_basic, metrics_for_rows, save_confusion_csv, save_errors_csv
from raw_cls.metrics import confusion_matrix, metrics_from_confusion
from raw_cls_temperature_scaling import calibration_basic as calibration_basic_probs, fit_temperature, temp_scale_probs
from phase3_train_raw_seg import DGCNNv2Seg, PointNetSeg, PointTransformerSeg, compute_seg_metrics


CLS_LABEL_ORDER = ["充填", "全冠", "桩核冠", "高嵌体"]


@dataclass(frozen=True)
class JointConfig:
    generated_at: str
    seed: int
    device: str
    data_root: str
    out_dir: str
    exp_name: str
    num_classes: int
    seg_num_classes: int
    n_points: int
    batch_size: int
    epochs: int
    patience: int
    lr: float
    weight_decay: float
    dropout: float
    num_workers: int
    balanced_sampler: bool
    label_smoothing: float
    aug_rotate_z: bool
    aug_scale: float
    aug_jitter_sigma: float
    aug_jitter_clip: float
    tta: int
    seg_loss_weight: float
    pooling_mode: str
    cls_train_mask: str
    cls_mask_mix_epochs: int
    cls_topk_ratio: float
    selection_seg_weight: float
    selection_calibration_weight: float
    selection_calibration_metric: str
    calibration_bins: int
    aux_gt_cls_weight: float
    consistency_weight: float
    consistency_temp: float
    feature_consistency_weight: float
    init_feat: str
    init_feat_loaded_keys: int
    seg_teacher_model: str
    seg_teacher_ckpt: str
    seg_teacher_weight: float
    seg_teacher_temp: float
    teeth3ds_teacher_ckpt: str
    teeth3ds_teacher_weight: float
    teeth3ds_teacher_points: int
    kfold_path: str
    kfold_k: int
    kfold_test_fold: int
    kfold_val_fold: int


def build_cls_label_map(rows: list[dict[str, Any]]) -> dict[str, int]:
    labels = sorted({str(r.get("label") or "").strip() for r in rows if str(r.get("label") or "").strip()})
    order = [lab for lab in CLS_LABEL_ORDER if lab in labels]
    rest = [lab for lab in labels if lab not in order]
    order.extend(rest)
    return {lab: i for i, lab in enumerate(order)}


def apply_kfold(index_rows: list[dict[str, Any]], kfold_path: Path | None, fold: int, val_fold: int) -> tuple[int, int]:
    if kfold_path is None:
        return 0, -1
    kfold_obj = read_json(kfold_path.resolve())
    k = int(kfold_obj["k"])
    test_fold = int(fold)
    val_fold_i = int(val_fold) if int(val_fold) >= 0 else (test_fold + 1) % k
    case_to_fold = kfold_obj["case_to_fold"]
    for row in index_rows:
        ck = str(row.get("case_key") or "")
        f = int(case_to_fold.get(ck, -1))
        if f == test_fold:
            row["split"] = "test"
        elif f == val_fold_i:
            row["split"] = "val"
        else:
            row["split"] = "train"
    return k, val_fold_i


class RawSegClsDataset(Dataset):
    def __init__(
        self,
        *,
        rows: list[dict[str, Any]],
        data_root: Path,
        cls_label_to_id: dict[str, int],
        n_points: int,
        train: bool,
        aug_rotate_z: bool,
        aug_scale: float,
        aug_jitter_sigma: float,
        aug_jitter_clip: float,
    ) -> None:
        self.rows = list(rows)
        self.data_root = Path(data_root)
        self.cls_label_to_id = dict(cls_label_to_id)
        self.n_points = int(n_points)
        self.train = bool(train)
        self.aug_rotate_z = bool(aug_rotate_z)
        self.aug_scale = float(aug_scale)
        self.aug_jitter_sigma = float(aug_jitter_sigma)
        self.aug_jitter_clip = float(aug_jitter_clip)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[int(idx)]
        npz_path = self.data_root / str(row["sample_npz"])
        with np.load(npz_path) as data:
            pts = np.asarray(data["points"], dtype=np.float32)
            seg_labels = np.asarray(data["labels"], dtype=np.int64)
        n_total = int(pts.shape[0])
        replace = n_total < self.n_points
        choice = np.random.choice(n_total, self.n_points, replace=replace).astype(np.int64, copy=False)
        pts = pts[choice]
        seg_labels = seg_labels[choice]
        if self.train:
            pts = self._augment(pts)
        cls_label = self.cls_label_to_id[str(row["label"])]
        return {
            "points": torch.from_numpy(pts),
            "seg_labels": torch.from_numpy(seg_labels),
            "cls_label": torch.tensor(int(cls_label), dtype=torch.long),
            "case_key": str(row.get("case_key") or ""),
            "split": str(row.get("split") or ""),
            "label": str(row.get("label") or ""),
            "source": str(row.get("source") or ""),
            "tooth_position": str(row.get("tooth_position") or ""),
            "sample_npz": str(row.get("sample_npz") or ""),
        }

    def _augment(self, pts: np.ndarray) -> np.ndarray:
        out = np.asarray(pts, dtype=np.float32).copy()
        if self.aug_rotate_z:
            theta = float(np.random.uniform(0.0, 2.0 * math.pi))
            c, s = math.cos(theta), math.sin(theta)
            rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
            out = out @ rot.T
        if self.aug_scale > 0:
            scale = float(np.random.uniform(1.0 - self.aug_scale, 1.0 + self.aug_scale))
            out = out * np.float32(scale)
        if self.aug_jitter_sigma > 0:
            noise = np.random.normal(0.0, self.aug_jitter_sigma, size=out.shape).astype(np.float32)
            clip = float(max(0.0, self.aug_jitter_clip))
            if clip > 0:
                noise = np.clip(noise, -clip, clip)
            out = out + noise
        return out.astype(np.float32, copy=False)


class PointNetSegCls(nn.Module):
    def __init__(
        self,
        num_classes: int,
        *,
        seg_num_classes: int = 2,
        dropout: float = 0.1,
        cls_topk_ratio: float = 0.5,
        pooling_mode: str = "topk",
    ) -> None:
        super().__init__()
        self.cls_topk_ratio = float(cls_topk_ratio)
        self.pooling_mode = str(pooling_mode)
        self.conv1 = nn.Sequential(nn.Conv1d(3, 64, 1, bias=False), nn.BatchNorm1d(64), nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1, bias=False), nn.BatchNorm1d(128), nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 256, 1, bias=False), nn.BatchNorm1d(256), nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.Conv1d(256, 512, 1, bias=False), nn.BatchNorm1d(512), nn.ReLU(True))
        self.seg_head = nn.Sequential(
            nn.Conv1d(128 + 512, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(float(dropout)),
            nn.Conv1d(256, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(float(dropout)),
            nn.Conv1d(128, int(seg_num_classes), 1),
        )
        cls_in_dim = 512 * 3
        if self.pooling_mode == "hybrid":
            cls_in_dim = 512 * 6 + 128 * 2 + 4
        if self.pooling_mode == "factorized":
            self.restore_proj = self._make_branch_mlp(512 * 2, float(dropout))
            self.context_proj = self._make_branch_mlp(512 * 2, float(dropout))
            self.global_proj = self._make_branch_mlp(512, float(dropout))
            self.restore_head = nn.Linear(128, int(num_classes))
            self.context_head = nn.Linear(128, int(num_classes))
            self.global_head = nn.Linear(128, int(num_classes))
            self.gate_head = nn.Sequential(
                nn.Linear(4, 32, bias=False),
                nn.BatchNorm1d(32),
                nn.ReLU(True),
                nn.Linear(32, 3),
            )
        else:
            self.cls_head = nn.Sequential(
                nn.Linear(int(cls_in_dim), 256, bias=False),
                nn.BatchNorm1d(256),
                nn.ReLU(True),
                nn.Dropout(float(dropout)),
                nn.Linear(256, 128, bias=False),
                nn.BatchNorm1d(128),
                nn.ReLU(True),
                nn.Dropout(float(dropout)),
                nn.Linear(128, int(num_classes)),
            )

    def _make_branch_mlp(self, in_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(int(in_dim), 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(float(dropout)),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
        )

    def _masked_mean(self, feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        return (feat * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(eps)

    def _masked_max(self, feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = mask > 0.5
        masked = feat.masked_fill(~valid.expand_as(feat), torch.finfo(feat.dtype).min)
        pooled = masked.max(dim=2).values
        empty = (~valid).all(dim=2).squeeze(1)
        if bool(empty.any()):
            pooled[empty] = feat.max(dim=2).values[empty]
        return pooled

    def _topk_mask(self, restore_prob: torch.Tensor) -> torch.Tensor:
        score = restore_prob.squeeze(1)
        n_points = int(score.shape[1])
        k = max(1, int(round(float(self.cls_topk_ratio) * n_points)))
        topk_idx = torch.topk(score, k=k, dim=1, largest=True, sorted=False).indices
        mask = torch.zeros_like(score)
        mask.scatter_(1, topk_idx, 1.0)
        return mask.unsqueeze(1)

    def _mask_stats(self, hard_mask: torch.Tensor, soft_mask: torch.Tensor) -> torch.Tensor:
        hard_ratio = hard_mask.mean(dim=2)
        soft_mean = soft_mask.mean(dim=2)
        soft_max = soft_mask.max(dim=2).values
        soft_std = soft_mask.std(dim=2, unbiased=False)
        return torch.cat([hard_ratio, soft_mean, soft_max, soft_std], dim=1)

    def forward(
        self,
        points: torch.Tensor,
        *,
        cls_mask: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        x = points.transpose(1, 2).contiguous()
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        g = torch.max(x4, dim=2, keepdim=True).values
        g_expand = g.expand(-1, -1, x.shape[2])
        seg_feat = torch.cat([x2, g_expand], dim=1)
        seg_logits = self.seg_head(seg_feat)
        restore_prob = torch.softmax(seg_logits, dim=1)[:, 1:2, :]
        if cls_mask is None:
            cls_mask = self._topk_mask(restore_prob)
        else:
            if cls_mask.ndim == 2:
                cls_mask = cls_mask.unsqueeze(1)
            cls_mask = cls_mask.to(dtype=x4.dtype)
        global_feat = g.squeeze(2)
        aux: dict[str, torch.Tensor] = {}
        if self.pooling_mode == "hybrid":
            soft_mask = restore_prob
            context_mask = (1.0 - restore_prob).clamp_min(0.0)
            restore_hard_max = self._masked_max(x4, cls_mask)
            restore_hard_mean = self._masked_mean(x4, cls_mask)
            restore_soft_mean = self._masked_mean(x4, soft_mask)
            context_mean = self._masked_mean(x4, context_mask)
            context_max = self._masked_max(x4, (cls_mask <= 0.5).to(dtype=x4.dtype))
            restore_local_mean = self._masked_mean(x2, cls_mask)
            restore_local_max = self._masked_max(x2, cls_mask)
            mask_stats = self._mask_stats(cls_mask, soft_mask)
            cls_feat = torch.cat(
                [
                    restore_hard_max,
                    restore_hard_mean,
                    restore_soft_mean,
                    context_mean,
                    context_max,
                    global_feat,
                    restore_local_mean,
                    restore_local_max,
                    mask_stats,
                ],
                dim=1,
            )
            cls_logits = self.cls_head(cls_feat)
            aux = {
                "cls_feat": cls_feat,
                "mask_stats": mask_stats,
                "restore_prob": restore_prob,
                "restore_max": restore_hard_max,
                "restore_mean": restore_hard_mean,
            }
        elif self.pooling_mode == "factorized":
            soft_mask = restore_prob
            context_hard_mask = (cls_mask <= 0.5).to(dtype=x4.dtype)
            context_soft_mask = (1.0 - soft_mask).clamp_min(0.0)
            mask_stats = self._mask_stats(cls_mask, soft_mask)
            restore_feat = torch.cat(
                [
                    self._masked_max(x4, cls_mask),
                    self._masked_mean(x4, soft_mask),
                ],
                dim=1,
            )
            context_feat = torch.cat(
                [
                    self._masked_max(x4, context_hard_mask),
                    self._masked_mean(x4, context_soft_mask),
                ],
                dim=1,
            )
            restore_emb = self.restore_proj(restore_feat)
            context_emb = self.context_proj(context_feat)
            global_emb = self.global_proj(global_feat)
            gate = torch.softmax(self.gate_head(mask_stats), dim=1)
            restore_logits = self.restore_head(restore_emb)
            context_logits = self.context_head(context_emb)
            global_logits = self.global_head(global_emb)
            cls_logits = (
                gate[:, 0:1] * restore_logits
                + gate[:, 1:2] * context_logits
                + gate[:, 2:3] * global_logits
            )
            aux = {
                "cls_feat": torch.cat([restore_emb, context_emb, global_emb, gate], dim=1),
                "mask_stats": mask_stats,
                "restore_prob": restore_prob,
                "restore_max": self._masked_max(x4, cls_mask),
                "restore_mean": self._masked_mean(x4, soft_mask),
            }
        else:
            restore_max = self._masked_max(x4, cls_mask)
            restore_mean = self._masked_mean(x4, cls_mask)
            cls_feat = torch.cat([restore_max, restore_mean, global_feat], dim=1)
            cls_logits = self.cls_head(cls_feat)
            aux = {
                "cls_feat": cls_feat,
                "mask_stats": self._mask_stats(cls_mask, restore_prob),
                "restore_prob": restore_prob,
                "restore_max": restore_max,
                "restore_mean": restore_mean,
            }
        if return_aux:
            return cls_logits, seg_logits, aux
        return cls_logits, seg_logits


class Teeth3DSPointNetTeacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv1d(3, 64, 1, bias=False),
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

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        x = points.transpose(1, 2).contiguous()
        x = self.feat(x)
        return torch.max(x, dim=2).values


def load_pointnet_shared_feat_init(model: PointNetSegCls, init_path: Path) -> dict[str, Any]:
    ckpt = torch.load(init_path, map_location="cpu")
    state = ckpt.get("model_state") or ckpt.get("model") or ckpt
    if not isinstance(state, dict):
        raise SystemExit(f"Invalid init checkpoint (expected dict-like state): {init_path}")

    prefix_map = {
        "feat.0.": "conv1.0.",
        "feat.1.": "conv1.1.",
        "feat.3.": "conv2.0.",
        "feat.4.": "conv2.1.",
        "feat.6.": "conv3.0.",
        "feat.7.": "conv3.1.",
        "feat.9.": "conv4.0.",
        "feat.10.": "conv4.1.",
    }
    target_state = model.state_dict()
    target_backbone_keys = {k for k in target_state if k.startswith(("conv1.", "conv2.", "conv3.", "conv4."))}
    remapped: dict[str, torch.Tensor] = {}
    source_mode = "direct"
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            continue
        mapped_key = ""
        if key.startswith(("conv1.", "conv2.", "conv3.", "conv4.")):
            mapped_key = key
        else:
            for src_prefix, tgt_prefix in prefix_map.items():
                if key.startswith(src_prefix):
                    mapped_key = tgt_prefix + key[len(src_prefix) :]
                    source_mode = "feat-prefix"
                    break
        if not mapped_key or mapped_key not in target_backbone_keys:
            continue
        tgt_value = target_state[mapped_key]
        if tuple(value.shape) != tuple(tgt_value.shape):
            raise SystemExit(
                f"init_feat shape mismatch for {mapped_key}: src={tuple(value.shape)} tgt={tuple(tgt_value.shape)}"
            )
        remapped[mapped_key] = value
    missing_backbone = sorted(target_backbone_keys - set(remapped))
    if missing_backbone:
        preview = ", ".join(missing_backbone[:8])
        more = "" if len(missing_backbone) <= 8 else f" (+{len(missing_backbone) - 8} more)"
        raise SystemExit(f"init_feat missing backbone keys from {init_path}: {preview}{more}")
    incompatible = model.load_state_dict(remapped, strict=False)
    if incompatible.unexpected_keys:
        raise SystemExit(f"Unexpected keys when loading init_feat: {incompatible.unexpected_keys}")
    loaded_keys = len(remapped)
    print(
        f"[init-feat] loaded pointnet shared backbone from {init_path} mode={source_mode} keys={loaded_keys}",
        flush=True,
    )
    return {
        "path": str(init_path),
        "loaded_keys": int(loaded_keys),
        "source_mode": str(source_mode),
    }


def load_teeth3ds_pointnet_teacher(ckpt_path: Path, device: torch.device) -> Teeth3DSPointNetTeacher:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state") or ckpt.get("model") or ckpt
    if not isinstance(state, dict):
        raise SystemExit(f"Invalid teacher checkpoint (expected dict-like state): {ckpt_path}")
    feat_state = {k[len("feat.") :]: v for k, v in state.items() if isinstance(k, str) and k.startswith("feat.")}
    if not feat_state:
        raise SystemExit(f"No 'feat.*' keys found in teacher checkpoint: {ckpt_path}")
    teacher = Teeth3DSPointNetTeacher().to(device)
    teacher.feat.load_state_dict(feat_state, strict=True)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"[teeth3ds-teacher] loaded frozen PointNet teacher from {ckpt_path}", flush=True)
    return teacher


@torch.no_grad()
def build_teeth3ds_teacher_crops(points: torch.Tensor, gt_mask: torch.Tensor, *, n_points: int) -> torch.Tensor:
    pts_np = points.detach().cpu().numpy().astype(np.float32, copy=False)
    mask_np = (gt_mask.detach().cpu().numpy() > 0.5)
    crops: list[np.ndarray] = []
    out_n = int(max(1, n_points))
    for i in range(int(points.shape[0])):
        pts_i = pts_np[i]
        mask_i = mask_np[i]
        center = pts_i[mask_i].mean(axis=0, dtype=np.float64).astype(np.float32) if bool(mask_i.any()) else pts_i.mean(axis=0, dtype=np.float64).astype(np.float32)
        k = min(out_n, int(pts_i.shape[0]))
        if k >= int(pts_i.shape[0]):
            local = pts_i
        else:
            dist = np.sum((pts_i - center[None, :]) ** 2, axis=1)
            idx = np.argpartition(dist, kth=k - 1)[:k]
            local = pts_i[idx]
        local, _centroid, _scale = normalize_points(local, "bbox_diag")
        local, _r = pca_align(local, align_globalz=False)
        if int(local.shape[0]) < out_n:
            rep = np.resize(np.arange(int(local.shape[0]), dtype=np.int64), out_n)
            local = local[rep]
        elif int(local.shape[0]) > out_n:
            local = local[:out_n]
        crops.append(local.astype(np.float32, copy=False))
    return torch.from_numpy(np.stack(crops, axis=0))


def rotate_points_batch(points: torch.Tensor, *, generator: torch.Generator | None) -> torch.Tensor:
    b = int(points.shape[0])
    angles = torch.rand((b,), generator=generator, device=points.device, dtype=points.dtype) * float(2.0 * math.pi)
    c = torch.cos(angles)
    s = torch.sin(angles)
    rot = torch.zeros((b, 3, 3), device=points.device, dtype=points.dtype)
    rot[:, 0, 0] = c
    rot[:, 0, 1] = -s
    rot[:, 1, 0] = s
    rot[:, 1, 1] = c
    rot[:, 2, 2] = 1.0
    return torch.matmul(points, rot.transpose(1, 2))


def batch_meta_rows(batch: dict[str, Any]) -> list[dict[str, Any]]:
    total = int(batch["cls_label"].shape[0])
    rows: list[dict[str, Any]] = []
    for i in range(total):
        rows.append(
            {
                "case_key": str(batch["case_key"][i]),
                "split": str(batch["split"][i]),
                "label": str(batch["label"][i]),
                "source": str(batch["source"][i]),
                "tooth_position": str(batch["tooth_position"][i]),
                "sample_npz": str(batch["sample_npz"][i]),
            }
        )
    return rows


def make_weighted_sampler(rows: list[dict[str, Any]], cls_label_to_id: dict[str, int]) -> WeightedRandomSampler | None:
    if not rows:
        return None
    counts = Counter(int(cls_label_to_id[str(r["label"])]) for r in rows)
    weights = [1.0 / max(1, counts[int(cls_label_to_id[str(r["label"])])]) for r in rows]
    return WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)


def apply_probs_to_rows(rows: list[dict[str, Any]], probs: np.ndarray) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if probs.ndim != 2 or probs.shape[0] != len(rows):
        raise ValueError(f"Shape mismatch between rows ({len(rows)}) and probs {tuple(probs.shape)}")
    for row, prob in zip(rows, probs.tolist(), strict=True):
        prob_f = [float(x) for x in prob]
        new_row = dict(row)
        new_row["probs"] = prob_f
        new_row["y_pred"] = int(np.argmax(prob_f))
        out.append(new_row)
    return out


def make_seg_teacher_model(name: str) -> nn.Module:
    nm = str(name).strip().lower()
    if nm == "pointnet_seg":
        return PointNetSeg(num_classes=2)
    if nm == "dgcnn_v2":
        return DGCNNv2Seg(num_classes=2)
    if nm == "point_transformer":
        return PointTransformerSeg(num_classes=2)
    raise SystemExit(f"Unsupported --seg-teacher-model: {name}")


def load_seg_teacher(ckpt_path: Path, model_name: str, device: torch.device) -> nn.Module:
    ckpt_obj = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt_obj.get("model", ckpt_obj)
    if not isinstance(state, dict):
        raise SystemExit(f"Invalid seg teacher checkpoint: {ckpt_path}")
    model = make_seg_teacher_model(model_name).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[seg-teacher] loaded frozen {model_name} teacher from {ckpt_path}", flush=True)
    return model


@torch.no_grad()
def evaluate_joint(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    num_classes: int,
    seg_num_classes: int,
    tta: int,
    tta_seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[list[float]] = []
    metas: list[dict[str, Any]] = []
    seg_preds: list[np.ndarray] = []
    seg_gts: list[np.ndarray] = []
    gen: torch.Generator | None = None
    if int(tta) > 1:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(tta_seed))

    for batch in loader:
        points = batch["points"].to(device=device, dtype=torch.float32, non_blocking=True)
        seg_labels = batch["seg_labels"].to(device=device, dtype=torch.long, non_blocking=True)
        cls_labels = batch["cls_label"].to(device=device, dtype=torch.long, non_blocking=True)
        meta_rows = batch_meta_rows(batch)

        if int(tta) > 1:
            cls_prob_accum = torch.zeros((int(points.shape[0]), int(num_classes)), device=device, dtype=torch.float32)
            seg_prob_accum = torch.zeros((int(points.shape[0]), int(seg_num_classes), int(points.shape[1])), device=device, dtype=torch.float32)
            for _ in range(int(tta)):
                pts_aug = rotate_points_batch(points, generator=gen)
                cls_logits, seg_logits = model(pts_aug)
                cls_prob_accum += torch.softmax(cls_logits, dim=1)
                seg_prob_accum += torch.softmax(seg_logits, dim=1)
            cls_probs = cls_prob_accum / float(int(tta))
            seg_probs = seg_prob_accum / float(int(tta))
        else:
            cls_logits, seg_logits = model(points)
            cls_probs = torch.softmax(cls_logits, dim=1)
            seg_probs = torch.softmax(seg_logits, dim=1)

        cls_pred = torch.argmax(cls_probs, dim=1)
        seg_pred = torch.argmax(seg_probs, dim=1)

        y_true.extend(cls_labels.detach().cpu().tolist())
        y_pred.extend(cls_pred.detach().cpu().tolist())
        y_prob.extend(cls_probs.detach().cpu().tolist())
        metas.extend(meta_rows)
        seg_preds.append(seg_pred.detach().cpu().numpy())
        seg_gts.append(seg_labels.detach().cpu().numpy())

    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)
    cm = confusion_matrix(y_true_np, y_pred_np, num_classes=int(num_classes))
    rows: list[dict[str, Any]] = []
    for t, p, prob, meta in zip(y_true, y_pred, y_prob, metas, strict=True):
        rows.append({**meta, "y_true": int(t), "y_pred": int(p), "probs": [float(x) for x in prob]})
    seg_pred_np = np.concatenate(seg_preds, axis=0).reshape(-1)
    seg_gt_np = np.concatenate(seg_gts, axis=0).reshape(-1)
    seg_metrics = compute_seg_metrics(seg_pred_np, seg_gt_np, int(seg_num_classes))
    return cm, rows, seg_metrics


def main() -> int:
    ap = argparse.ArgumentParser(description="Joint raw segmentation + classification on full-case point clouds.")
    ap.add_argument("--data-root", type=Path, default=Path("processed/raw_seg/v1"))
    ap.add_argument("--run-root", type=Path, default=Path("runs/research_segcls_joint"))
    ap.add_argument("--exp-name", type=str, default="")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--n-points", type=int, default=8192)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--balanced-sampler", action="store_true")
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--aug-rotate-z", action="store_true", default=True)
    ap.add_argument("--no-aug-rotate-z", action="store_false", dest="aug_rotate_z")
    ap.add_argument("--aug-scale", type=float, default=0.1)
    ap.add_argument("--aug-jitter-sigma", type=float, default=0.01)
    ap.add_argument("--aug-jitter-clip", type=float, default=0.03)
    ap.add_argument("--tta", type=int, default=0)
    ap.add_argument("--seg-loss-weight", type=float, default=0.5)
    ap.add_argument("--pooling-mode", type=str, choices=["topk", "hybrid", "factorized"], default="topk")
    ap.add_argument("--cls-train-mask", type=str, choices=["gt", "pred", "mix"], default="gt")
    ap.add_argument("--cls-mask-mix-epochs", type=int, default=20)
    ap.add_argument("--cls-topk-ratio", type=float, default=0.5)
    ap.add_argument("--selection-seg-weight", type=float, default=0.0)
    ap.add_argument("--selection-calibration-weight", type=float, default=0.0)
    ap.add_argument("--selection-calibration-metric", type=str, choices=["ece", "nll", "brier"], default="ece")
    ap.add_argument("--calibration-bins", type=int, default=15)
    ap.add_argument("--aux-gt-cls-weight", type=float, default=0.0)
    ap.add_argument("--consistency-weight", type=float, default=0.0)
    ap.add_argument("--consistency-temp", type=float, default=1.0)
    ap.add_argument("--feature-consistency-weight", type=float, default=0.0)
    ap.add_argument("--init-feat", type=Path, default=None)
    ap.add_argument("--seg-teacher-model", type=str, choices=["pointnet_seg", "dgcnn_v2", "point_transformer"], default="dgcnn_v2")
    ap.add_argument("--seg-teacher-ckpt", type=Path, default=None)
    ap.add_argument("--seg-teacher-weight", type=float, default=0.0)
    ap.add_argument("--seg-teacher-temp", type=float, default=1.0)
    ap.add_argument("--teeth3ds-teacher-ckpt", type=Path, default=None)
    ap.add_argument("--teeth3ds-teacher-weight", type=float, default=0.0)
    ap.add_argument("--teeth3ds-teacher-points", type=int, default=1024)
    ap.add_argument("--kfold", type=Path, default=None)
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--val-fold", type=int, default=-1)
    args = ap.parse_args()

    set_seed(int(args.seed))
    device_str = normalize_device(str(args.device))
    device = torch.device(device_str)
    data_root = args.data_root.resolve()
    index_rows = read_jsonl(data_root / "index.jsonl")
    if not index_rows:
        raise SystemExit(f"Empty index under {data_root}")

    cls_label_to_id = build_cls_label_map(index_rows)
    labels_by_id = [lab for lab, _ in sorted(cls_label_to_id.items(), key=lambda kv: kv[1])]
    seg_label_map = read_json(data_root / "label_map.json")
    seg_num_classes = int(len(seg_label_map))
    kfold_k, val_fold = apply_kfold(index_rows, args.kfold if args.kfold is not None else None, int(args.fold), int(args.val_fold))

    train_rows = [r for r in index_rows if str(r.get("split") or "") == "train"]
    val_rows = [r for r in index_rows if str(r.get("split") or "") == "val"]
    test_rows = [r for r in index_rows if str(r.get("split") or "") == "test"]
    print(f"[data] train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}", flush=True)

    train_ds = RawSegClsDataset(
        rows=train_rows,
        data_root=data_root,
        cls_label_to_id=cls_label_to_id,
        n_points=int(args.n_points),
        train=True,
        aug_rotate_z=bool(args.aug_rotate_z),
        aug_scale=float(args.aug_scale),
        aug_jitter_sigma=float(args.aug_jitter_sigma),
        aug_jitter_clip=float(args.aug_jitter_clip),
    )
    eval_kwargs = {
        "data_root": data_root,
        "cls_label_to_id": cls_label_to_id,
        "n_points": int(args.n_points),
        "train": False,
        "aug_rotate_z": False,
        "aug_scale": 0.0,
        "aug_jitter_sigma": 0.0,
        "aug_jitter_clip": 0.0,
    }
    val_ds = RawSegClsDataset(rows=val_rows, **eval_kwargs)
    test_ds = RawSegClsDataset(rows=test_rows, **eval_kwargs)

    sampler = make_weighted_sampler(train_rows, cls_label_to_id) if bool(args.balanced_sampler) else None
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers), pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers), pin_memory=True)

    cls_counts = Counter(int(cls_label_to_id[str(r["label"])]) for r in train_rows)
    total_cls = sum(cls_counts.values())
    cls_weight = torch.tensor(
        [total_cls / max(1, cls_counts.get(i, 1) * len(cls_label_to_id)) for i in range(len(cls_label_to_id))],
        dtype=torch.float32,
        device=device,
    )
    seg_counts: Counter[int] = Counter()
    for row in train_rows:
        with np.load(data_root / str(row["sample_npz"])) as z:
            seg = np.asarray(z["labels"], dtype=np.int64)
        uniq, cnt = np.unique(seg, return_counts=True)
        for u, c in zip(uniq.tolist(), cnt.tolist(), strict=True):
            seg_counts[int(u)] += int(c)
    total_seg = sum(seg_counts.values())
    seg_weight = torch.tensor(
        [total_seg / max(1, seg_counts.get(i, 1) * seg_num_classes) for i in range(seg_num_classes)],
        dtype=torch.float32,
        device=device,
    )
    print(f"[weights] cls={cls_weight.tolist()} seg={seg_weight.tolist()}", flush=True)

    model = PointNetSegCls(
        num_classes=len(cls_label_to_id),
        seg_num_classes=seg_num_classes,
        dropout=float(args.dropout),
        cls_topk_ratio=float(args.cls_topk_ratio),
        pooling_mode=str(args.pooling_mode),
    ).to(device)
    init_info = {"path": "", "loaded_keys": 0, "source_mode": ""}
    if args.init_feat is not None:
        init_feat_path = args.init_feat.expanduser().resolve()
        if not init_feat_path.is_file():
            raise SystemExit(f"Missing init checkpoint: {init_feat_path}")
        init_info = load_pointnet_shared_feat_init(model, init_feat_path)
    seg_teacher = None
    seg_teacher_info = {"model": "", "path": "", "weight": 0.0, "temp": 1.0}
    if float(args.seg_teacher_weight) > 0.0:
        if args.seg_teacher_ckpt is None:
            raise SystemExit("--seg-teacher-weight > 0 requires --seg-teacher-ckpt")
        seg_teacher_ckpt = args.seg_teacher_ckpt.expanduser().resolve()
        if not seg_teacher_ckpt.is_file():
            raise SystemExit(f"Missing seg teacher checkpoint: {seg_teacher_ckpt}")
        seg_teacher = load_seg_teacher(seg_teacher_ckpt, str(args.seg_teacher_model), device)
        seg_teacher_info = {
            "model": str(args.seg_teacher_model),
            "path": str(seg_teacher_ckpt),
            "weight": float(args.seg_teacher_weight),
            "temp": float(args.seg_teacher_temp),
        }
    teeth3ds_teacher = None
    teeth3ds_teacher_info = {"path": "", "weight": 0.0, "points": 0}
    if float(args.teeth3ds_teacher_weight) > 0.0:
        if args.teeth3ds_teacher_ckpt is None:
            raise SystemExit("--teeth3ds-teacher-weight > 0 requires --teeth3ds-teacher-ckpt")
        teacher_ckpt_path = args.teeth3ds_teacher_ckpt.expanduser().resolve()
        if not teacher_ckpt_path.is_file():
            raise SystemExit(f"Missing Teeth3DS teacher checkpoint: {teacher_ckpt_path}")
        teeth3ds_teacher = load_teeth3ds_pointnet_teacher(teacher_ckpt_path, device)
        teeth3ds_teacher_info = {
            "path": str(teacher_ckpt_path),
            "weight": float(args.teeth3ds_teacher_weight),
            "points": int(args.teeth3ds_teacher_points),
        }
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(args.epochs), eta_min=1e-6)

    fold_str = f"_fold{int(args.fold)}" if int(args.fold) >= 0 else ""
    exp_name = str(args.exp_name).strip() or f"pointnet_segcls_s{int(args.seed)}{fold_str}"
    out_dir = (args.run_root.resolve() / exp_name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = JointConfig(
        generated_at=utc_now_iso(),
        seed=int(args.seed),
        device=device_str,
        data_root=str(data_root),
        out_dir=str(out_dir),
        exp_name=exp_name,
        num_classes=len(cls_label_to_id),
        seg_num_classes=seg_num_classes,
        n_points=int(args.n_points),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        patience=int(args.patience),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        dropout=float(args.dropout),
        num_workers=int(args.num_workers),
        balanced_sampler=bool(args.balanced_sampler),
        label_smoothing=float(args.label_smoothing),
        aug_rotate_z=bool(args.aug_rotate_z),
        aug_scale=float(args.aug_scale),
        aug_jitter_sigma=float(args.aug_jitter_sigma),
        aug_jitter_clip=float(args.aug_jitter_clip),
        tta=int(args.tta),
        seg_loss_weight=float(args.seg_loss_weight),
        pooling_mode=str(args.pooling_mode),
        cls_train_mask=str(args.cls_train_mask),
        cls_mask_mix_epochs=int(args.cls_mask_mix_epochs),
        cls_topk_ratio=float(args.cls_topk_ratio),
        selection_seg_weight=float(args.selection_seg_weight),
        selection_calibration_weight=float(args.selection_calibration_weight),
        selection_calibration_metric=str(args.selection_calibration_metric),
        calibration_bins=int(args.calibration_bins),
        aux_gt_cls_weight=float(args.aux_gt_cls_weight),
        consistency_weight=float(args.consistency_weight),
        consistency_temp=float(args.consistency_temp),
        feature_consistency_weight=float(args.feature_consistency_weight),
        init_feat=str(init_info["path"]),
        init_feat_loaded_keys=int(init_info["loaded_keys"]),
        seg_teacher_model=str(seg_teacher_info["model"]),
        seg_teacher_ckpt=str(seg_teacher_info["path"]),
        seg_teacher_weight=float(seg_teacher_info["weight"]),
        seg_teacher_temp=float(seg_teacher_info["temp"]),
        teeth3ds_teacher_ckpt=str(teeth3ds_teacher_info["path"]),
        teeth3ds_teacher_weight=float(teeth3ds_teacher_info["weight"]),
        teeth3ds_teacher_points=int(teeth3ds_teacher_info["points"]),
        kfold_path=str(args.kfold.resolve()) if args.kfold is not None else "",
        kfold_k=int(kfold_k),
        kfold_test_fold=int(args.fold),
        kfold_val_fold=int(val_fold),
    )
    write_json(out_dir / "config.json", asdict(cfg))

    best_val = -1.0
    best_val_seg = -1.0
    best_val_calibration = float("inf")
    best_select_score = -1.0e18
    best_epoch = 0
    epochs_no_improve = 0
    best_path = out_dir / "model_best.pt"
    history_path = out_dir / "history.jsonl"
    t0 = time.time()

    with history_path.open("w", encoding="utf-8") as hist_f:
        for epoch in range(1, int(args.epochs) + 1):
            model.train()
            loss_sum = 0.0
            cls_loss_sum = 0.0
            cls_aux_loss_sum = 0.0
            consistency_loss_sum = 0.0
            feature_consistency_loss_sum = 0.0
            seg_teacher_loss_sum = 0.0
            teeth3ds_teacher_loss_sum = 0.0
            seg_loss_sum = 0.0
            cls_correct = 0
            cls_seen = 0

            for batch in train_loader:
                points = batch["points"].to(device=device, dtype=torch.float32, non_blocking=True)
                seg_labels = batch["seg_labels"].to(device=device, dtype=torch.long, non_blocking=True)
                cls_labels = batch["cls_label"].to(device=device, dtype=torch.long, non_blocking=True)
                gt_cls_mask = (seg_labels == 1).to(dtype=torch.float32)

                optimizer.zero_grad(set_to_none=True)
                cls_mask: torch.Tensor | None = None
                train_mask_mode = str(args.cls_train_mask)
                if train_mask_mode == "gt":
                    cls_mask = gt_cls_mask
                elif train_mask_mode == "mix":
                    mix_epochs = max(1, int(args.cls_mask_mix_epochs))
                    gt_prob = max(0.0, 1.0 - float(epoch - 1) / float(mix_epochs))
                    if float(np.random.rand()) < gt_prob:
                        cls_mask = gt_cls_mask
                need_main_aux = float(args.feature_consistency_weight) > 0.0 or teeth3ds_teacher is not None
                main_out = model(points, cls_mask=cls_mask, return_aux=need_main_aux)
                if need_main_aux:
                    cls_logits, seg_logits, main_aux = main_out
                else:
                    cls_logits, seg_logits = main_out
                    main_aux = {}
                cls_loss = F.cross_entropy(
                    cls_logits,
                    cls_labels,
                    weight=(None if bool(args.balanced_sampler) else cls_weight),
                    label_smoothing=float(args.label_smoothing),
                )
                cls_aux_loss = torch.zeros((), device=device, dtype=torch.float32)
                consistency_loss = torch.zeros((), device=device, dtype=torch.float32)
                feature_consistency_loss = torch.zeros((), device=device, dtype=torch.float32)
                seg_teacher_loss = torch.zeros((), device=device, dtype=torch.float32)
                teeth3ds_teacher_loss = torch.zeros((), device=device, dtype=torch.float32)
                if (
                    float(args.aux_gt_cls_weight) > 0.0
                    or float(args.consistency_weight) > 0.0
                    or float(args.feature_consistency_weight) > 0.0
                ):
                    teacher_out = model(points, cls_mask=gt_cls_mask, return_aux=(float(args.feature_consistency_weight) > 0.0))
                    if float(args.feature_consistency_weight) > 0.0:
                        teacher_logits, _, teacher_aux = teacher_out
                    else:
                        teacher_logits, _ = teacher_out
                        teacher_aux = {}
                    if float(args.aux_gt_cls_weight) > 0.0:
                        cls_aux_loss = F.cross_entropy(
                            teacher_logits,
                            cls_labels,
                            weight=(None if bool(args.balanced_sampler) else cls_weight),
                            label_smoothing=float(args.label_smoothing),
                        )
                    if float(args.consistency_weight) > 0.0:
                        temp = max(1.0e-6, float(args.consistency_temp))
                        consistency_loss = F.kl_div(
                            F.log_softmax(cls_logits / temp, dim=1),
                            F.softmax(teacher_logits.detach() / temp, dim=1),
                            reduction="batchmean",
                        ) * (temp * temp)
                    if float(args.feature_consistency_weight) > 0.0:
                        feature_consistency_loss = (
                            1.0
                            - F.cosine_similarity(
                                main_aux["cls_feat"],
                                teacher_aux["cls_feat"].detach(),
                                dim=1,
                                eps=1.0e-6,
                            )
                        ).mean()
                if teeth3ds_teacher is not None:
                    if "restore_max" not in main_aux:
                        raise SystemExit("Internal error: Teeth3DS teacher distillation requires main_aux['restore_max'].")
                    teacher_crops = build_teeth3ds_teacher_crops(
                        points,
                        gt_cls_mask,
                        n_points=int(args.teeth3ds_teacher_points),
                    ).to(device=device, dtype=torch.float32, non_blocking=True)
                    with torch.no_grad():
                        teacher_feat = teeth3ds_teacher(teacher_crops)
                    teeth3ds_teacher_loss = (
                        1.0
                        - F.cosine_similarity(
                            main_aux["restore_max"],
                            teacher_feat.detach(),
                            dim=1,
                            eps=1.0e-6,
                        )
                    ).mean()
                if seg_teacher is not None:
                    with torch.no_grad():
                        seg_teacher_logits = seg_teacher(points)
                    seg_temp = max(1.0e-6, float(args.seg_teacher_temp))
                    seg_teacher_loss = (
                        F.kl_div(
                            F.log_softmax(seg_logits / seg_temp, dim=1),
                            F.softmax(seg_teacher_logits.detach() / seg_temp, dim=1),
                            reduction="none",
                        ).sum(dim=1)
                    ).mean() * (seg_temp * seg_temp)
                seg_loss = F.cross_entropy(seg_logits, seg_labels, weight=seg_weight)
                loss = (
                    cls_loss
                    + float(args.aux_gt_cls_weight) * cls_aux_loss
                    + float(args.consistency_weight) * consistency_loss
                    + float(args.feature_consistency_weight) * feature_consistency_loss
                    + float(args.seg_teacher_weight) * seg_teacher_loss
                    + float(args.teeth3ds_teacher_weight) * teeth3ds_teacher_loss
                    + float(args.seg_loss_weight) * seg_loss
                )
                loss.backward()
                optimizer.step()

                bsz = int(points.shape[0])
                loss_sum += float(loss.detach().cpu().item()) * bsz
                cls_loss_sum += float(cls_loss.detach().cpu().item()) * bsz
                cls_aux_loss_sum += float(cls_aux_loss.detach().cpu().item()) * bsz
                consistency_loss_sum += float(consistency_loss.detach().cpu().item()) * bsz
                feature_consistency_loss_sum += float(feature_consistency_loss.detach().cpu().item()) * bsz
                seg_teacher_loss_sum += float(seg_teacher_loss.detach().cpu().item()) * bsz
                teeth3ds_teacher_loss_sum += float(teeth3ds_teacher_loss.detach().cpu().item()) * bsz
                seg_loss_sum += float(seg_loss.detach().cpu().item()) * bsz
                cls_seen += bsz
                cls_pred = torch.argmax(cls_logits, dim=1)
                cls_correct += int((cls_pred == cls_labels).sum().detach().cpu().item())

            scheduler.step()
            train_loss = loss_sum / max(1, cls_seen)
            train_cls_loss = cls_loss_sum / max(1, cls_seen)
            train_cls_aux_loss = cls_aux_loss_sum / max(1, cls_seen)
            train_consistency_loss = consistency_loss_sum / max(1, cls_seen)
            train_feature_consistency_loss = feature_consistency_loss_sum / max(1, cls_seen)
            train_seg_teacher_loss = seg_teacher_loss_sum / max(1, cls_seen)
            train_teeth3ds_teacher_loss = teeth3ds_teacher_loss_sum / max(1, cls_seen)
            train_seg_loss = seg_loss_sum / max(1, cls_seen)
            train_acc = cls_correct / max(1, cls_seen)

            cm_val, rows_val, seg_val_metrics = evaluate_joint(
                model,
                val_loader,
                device,
                num_classes=len(cls_label_to_id),
                seg_num_classes=seg_num_classes,
                tta=0,
                tta_seed=int(args.seed) + 100,
            )
            val_metrics = metrics_from_confusion(cm_val, labels_by_id)
            val_score = float(val_metrics.get("macro_f1_present") or 0.0)
            val_seg_score = float(seg_val_metrics.get("mean_iou") or 0.0)
            val_calibration = calibration_basic(rows_val, len(cls_label_to_id), n_bins=int(args.calibration_bins))
            val_cal_metric_name = str(args.selection_calibration_metric)
            val_cal_metric = float(val_calibration.get(val_cal_metric_name) or 0.0)
            select_score = (
                val_score
                + float(args.selection_seg_weight) * val_seg_score
                - float(args.selection_calibration_weight) * val_cal_metric
            )
            improved = (select_score > best_select_score + 1e-6) or (
                abs(select_score - best_select_score) <= 1e-6
                and (
                    (val_score > best_val + 1e-6)
                    or (
                        abs(val_score - best_val) <= 1e-6
                        and (
                            (val_seg_score > best_val_seg + 1e-6)
                            or (
                                abs(val_seg_score - best_val_seg) <= 1e-6
                                and val_cal_metric < best_val_calibration - 1e-6
                            )
                        )
                    )
                )
            )
            row = {
                "epoch": epoch,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "train_loss": float(train_loss),
                "train_cls_loss": float(train_cls_loss),
                "train_cls_aux_loss": float(train_cls_aux_loss),
                "train_consistency_loss": float(train_consistency_loss),
                "train_feature_consistency_loss": float(train_feature_consistency_loss),
                "train_seg_teacher_loss": float(train_seg_teacher_loss),
                "train_teeth3ds_teacher_loss": float(train_teeth3ds_teacher_loss),
                "train_seg_loss": float(train_seg_loss),
                "train_acc": float(train_acc),
                "train_mask_mode": str(args.cls_train_mask),
                "train_gt_mask_prob": (
                    float(max(0.0, 1.0 - float(epoch - 1) / float(max(1, int(args.cls_mask_mix_epochs)))))
                    if str(args.cls_train_mask) == "mix"
                    else (1.0 if str(args.cls_train_mask) == "gt" else 0.0)
                ),
                "val_score": float(val_score),
                "val_select_score": float(select_score),
                "val_calibration": val_calibration,
                "val_selection_calibration_metric": val_cal_metric_name,
                "val_selection_calibration_value": float(val_cal_metric),
                "val": val_metrics,
                "val_seg": seg_val_metrics,
            }
            hist_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            hist_f.flush()

            if improved:
                best_select_score = select_score
                best_val = val_score
                best_val_seg = val_seg_score
                best_val_calibration = val_cal_metric
                best_epoch = epoch
                epochs_no_improve = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "config": asdict(cfg),
                        "val_metrics": val_metrics,
                        "val_seg_metrics": seg_val_metrics,
                        "val_calibration": val_calibration,
                        "val_select_score": float(select_score),
                    },
                    best_path,
                )
                write_json(
                    out_dir / "best_val_metrics.json",
                    {
                        "val": val_metrics,
                        "val_seg": seg_val_metrics,
                        "val_calibration": val_calibration,
                        "selection": {
                            "score": float(select_score),
                            "metric": val_cal_metric_name,
                            "value": float(val_cal_metric),
                        },
                    },
                )
                save_confusion_csv(cm_val, labels_by_id, out_dir / "confusion_val.csv")
            else:
                epochs_no_improve += 1

            print(
                f"[epoch {epoch:03d}] train_loss={train_loss:.4f} cls={train_cls_loss:.4f} "
                f"aux={train_cls_aux_loss:.4f} cons={train_consistency_loss:.4f} "
                f"feat={train_feature_consistency_loss:.4f} segt={train_seg_teacher_loss:.4f} "
                f"t3d={train_teeth3ds_teacher_loss:.4f} seg={train_seg_loss:.4f} "
                f"train_acc={train_acc:.3f} val_macro_f1_present={val_score:.3f} val_seg_miou={seg_val_metrics['mean_iou']:.3f} "
                f"val_{val_cal_metric_name}={val_cal_metric:.4f} val_select={select_score:.3f} {'*' if improved else ''}",
                flush=True,
            )
            if int(args.patience) > 0 and epochs_no_improve >= int(args.patience):
                print(f"[early-stop] no improvement for {int(args.patience)} epochs (best_epoch={best_epoch})", flush=True)
                break

    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])

    cm_val, rows_val, seg_val_metrics = evaluate_joint(
        model,
        val_loader,
        device,
        num_classes=len(cls_label_to_id),
        seg_num_classes=seg_num_classes,
        tta=int(args.tta),
        tta_seed=int(args.seed) + 123,
    )
    cm_test, rows_test, seg_test_metrics = evaluate_joint(
        model,
        test_loader,
        device,
        num_classes=len(cls_label_to_id),
        seg_num_classes=seg_num_classes,
        tta=int(args.tta),
        tta_seed=int(args.seed) + 456,
    )
    val_metrics = metrics_from_confusion(cm_val, labels_by_id)
    test_metrics = metrics_from_confusion(cm_test, labels_by_id)

    save_confusion_csv(cm_val, labels_by_id, out_dir / "confusion_val_final.csv")
    save_confusion_csv(cm_test, labels_by_id, out_dir / "confusion_test.csv")
    save_errors_csv(rows_test, labels_by_id, out_dir / "errors_test.csv")
    write_jsonl(out_dir / "preds_val.jsonl", rows_val)
    write_jsonl(out_dir / "preds_test.jsonl", rows_test)

    by_source: dict[str, Any] = {}
    for src in sorted({str(r.get("source") or "") for r in rows_test}):
        by_source[src or "(missing)"] = metrics_for_rows([r for r in rows_test if str(r.get("source") or "") == src], labels_by_id)

    by_tooth_pos: dict[str, Any] = {}
    for tp in sorted({str(r.get("tooth_position") or "") for r in rows_test}):
        by_tooth_pos[tp or "(missing)"] = metrics_for_rows([r for r in rows_test if str(r.get("tooth_position") or "") == tp], labels_by_id)

    cal_val = calibration_basic(rows_val, len(cls_label_to_id), n_bins=int(args.calibration_bins))
    cal_test = calibration_basic(rows_test, len(cls_label_to_id), n_bins=int(args.calibration_bins))
    cal_by_source: dict[str, Any] = {}
    for src in sorted({str(r.get("source") or "") for r in rows_test}):
        cal_by_source[src or "(missing)"] = calibration_basic(
            [r for r in rows_test if str(r.get("source") or "") == src],
            len(cls_label_to_id),
            n_bins=int(args.calibration_bins),
        )

    probs_val = np.asarray([r.get("probs") or [] for r in rows_val], dtype=np.float64)
    y_val = np.asarray([int(r.get("y_true", 0)) for r in rows_val], dtype=np.int64)
    probs_test = np.asarray([r.get("probs") or [] for r in rows_test], dtype=np.float64)
    y_test = np.asarray([int(r.get("y_true", 0)) for r in rows_test], dtype=np.int64)
    temp_T, temp_fit = fit_temperature(probs_val, y_val)
    probs_val_cal = temp_scale_probs(probs_val, T=float(temp_T))
    probs_test_cal = temp_scale_probs(probs_test, T=float(temp_T))
    rows_val_cal = apply_probs_to_rows(rows_val, probs_val_cal)
    rows_test_cal = apply_probs_to_rows(rows_test, probs_test_cal)
    cal_val_temp = calibration_basic_probs(probs_val_cal, y_val, n_bins=int(args.calibration_bins))
    cal_test_temp = calibration_basic_probs(probs_test_cal, y_test, n_bins=int(args.calibration_bins))
    test_metrics_temp = metrics_for_rows(rows_test_cal, labels_by_id)
    val_metrics_temp = metrics_for_rows(rows_val_cal, labels_by_id)
    cal_by_source_temp: dict[str, Any] = {}
    for src in sorted({str(r.get("source") or "") for r in rows_test_cal}):
        cal_by_source_temp[src or "(missing)"] = calibration_basic_probs(
            np.asarray([r.get("probs") or [] for r in rows_test_cal if str(r.get("source") or "") == src], dtype=np.float64),
            np.asarray([int(r.get("y_true", 0)) for r in rows_test_cal if str(r.get("source") or "") == src], dtype=np.int64),
            n_bins=int(args.calibration_bins),
        )
    calib_json = {
        "temperature": float(temp_T),
        "bins": int(args.calibration_bins),
        "fit": temp_fit,
        "val": {"before": cal_val, "after": cal_val_temp},
        "test": {"before": cal_test, "after": cal_test_temp},
        "test_metrics_after_temperature": test_metrics_temp,
        "val_metrics_after_temperature": val_metrics_temp,
    }
    write_json(out_dir / "calib.json", calib_json)
    write_jsonl(out_dir / "preds_val_calibrated.jsonl", rows_val_cal)
    write_jsonl(out_dir / "preds_test_calibrated.jsonl", rows_test_cal)

    wall_time_sec = float(max(0.0, time.time() - t0))
    runtime: dict[str, Any] = {
        "wall_time_sec": wall_time_sec,
        "wall_time_hr": wall_time_sec / 3600.0,
    }
    if device_str == "cuda":
        try:
            runtime.update(
                {
                    "cuda_device_name": str(torch.cuda.get_device_name(torch.cuda.current_device())),
                    "cuda_peak_allocated_mb": float(torch.cuda.max_memory_allocated() / (1024.0 ** 2)),
                    "cuda_peak_reserved_mb": float(torch.cuda.max_memory_reserved() / (1024.0 ** 2)),
                }
            )
        except Exception:
            pass

    notes = [
        f"Classification pooling mode: {str(args.pooling_mode)}.",
        (
            f"Training cls mask mode: {str(args.cls_train_mask)}"
            f" (mix_epochs={int(args.cls_mask_mix_epochs)}); inference uses top-k ratio={float(args.cls_topk_ratio):.3f}."
        ),
        (
            f"GT-mask auxiliary classification weight={float(args.aux_gt_cls_weight):.3f}, "
            f"consistency weight={float(args.consistency_weight):.3f}, "
            f"temperature={float(args.consistency_temp):.3f}."
        ),
        f"Feature-consistency weight on predicted-vs-GT pooled evidence={float(args.feature_consistency_weight):.3f}.",
        (
            f"Checkpoint selection score: val_macro_f1_present + {float(args.selection_seg_weight):.3f} * val_seg_mIoU"
            f" - {float(args.selection_calibration_weight):.3f} * val_{str(args.selection_calibration_metric)}."
        ),
        (
            f"Temperature scaling fitted on validation probabilities with bins={int(args.calibration_bins)}; "
            f"T={float(temp_T):.4f}."
        ),
        "test_seg metrics are aggregated over all sampled test points after optional TTA averaging.",
    ]
    if str(args.pooling_mode) == "factorized":
        notes.insert(1, "factorized mode uses restoration/context/global experts with mask-stat gating.")
    if str(init_info["path"]):
        notes.insert(
            1,
            (
                f"Initialized shared PointNet backbone from {str(init_info['path'])} "
                f"(mode={str(init_info['source_mode'])}, keys={int(init_info['loaded_keys'])})."
            ),
        )
    if str(seg_teacher_info["path"]):
        notes.insert(
            1,
            (
                f"Applied frozen segmentation teacher {str(seg_teacher_info['model'])} from {str(seg_teacher_info['path'])} "
                f"with weight={float(seg_teacher_info['weight']):.3f} and temperature={float(seg_teacher_info['temp']):.3f}."
            ),
        )
    if str(teeth3ds_teacher_info["path"]):
        notes.insert(
            1,
            (
                f"Applied frozen Teeth3DS PointNet teacher distillation from {str(teeth3ds_teacher_info['path'])} "
                f"with weight={float(teeth3ds_teacher_info['weight']):.3f} on GT-centroid local crops "
                f"(n_points={int(teeth3ds_teacher_info['points'])})."
            ),
        )

    summary = {
        "generated_at": utc_now_iso(),
        "best_epoch": int(best_epoch),
        "val": val_metrics,
        "val_seg": seg_val_metrics,
        "val_calibration": cal_val,
        "val_temperature_scaled": {
            "metrics": val_metrics_temp,
            "calibration": cal_val_temp,
        },
        "test": test_metrics,
        "test_seg": seg_test_metrics,
        "test_by_source": by_source,
        "test_by_tooth_position": by_tooth_pos,
        "test_calibration": cal_test,
        "test_temperature_scaled": {
            "temperature": float(temp_T),
            "metrics": test_metrics_temp,
            "calibration": cal_test_temp,
            "fit": temp_fit,
        },
        "test_by_source_calibration": cal_by_source,
        "test_by_source_temperature_scaled_calibration": cal_by_source_temp,
        "runtime": runtime,
        "env": get_env_info(),
        "git": get_git_info(Path(".")),
        "notes": notes,
    }
    write_json(out_dir / "metrics.json", summary)
    print(f"[OK] run_dir: {out_dir}", flush=True)
    print(f"[OK] wrote: {out_dir / 'metrics.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
