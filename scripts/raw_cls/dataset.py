from __future__ import annotations

import math
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import build_point_features_from_xyz, point_feature_dim
from .preprocess_np import apply_input_preprocess_np, jitter, random_point_dropout, rotate_z


def atomic_write_npz(path: Path, *, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("wb") as f:
            np.savez_compressed(f, **arrays)
        tmp.replace(path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


class RawClsDataset(Dataset[tuple[torch.Tensor, int, dict[str, Any]]]):
    def __init__(
        self,
        *,
        rows: list[dict[str, Any]],
        data_root: Path,
        label_to_id: dict[str, int],
        n_points: int,
        seed: int,
        train: bool,
        aug_rotate_z: bool,
        aug_scale: float,
        aug_jitter_sigma: float,
        aug_jitter_clip: float,
        aug_dropout_ratio: float,
        load_points: bool = True,
        extra_features: list[str] | None = None,
        extra_mean: np.ndarray | None = None,
        extra_std: np.ndarray | None = None,
        point_features: list[str] | None = None,
        feature_cache_dir: Path | None = None,
        feature_k: int = 30,
        tooth_position_dropout: float = 0.0,
        input_normalize: str = "none",
        input_pca_align: bool = False,
        input_pca_align_globalz: bool = False,
    ) -> None:
        self.rows = rows
        self.data_root = data_root
        self.label_to_id = label_to_id
        self.n_points = int(n_points)
        self.seed = int(seed)
        self.train = bool(train)
        self.aug_rotate_z = bool(aug_rotate_z)
        self.aug_scale = float(aug_scale)
        self.aug_jitter_sigma = float(aug_jitter_sigma)
        self.aug_jitter_clip = float(aug_jitter_clip)
        self.aug_dropout_ratio = float(aug_dropout_ratio)
        self.load_points = bool(load_points)
        self.rng = np.random.default_rng(int(seed) + (0 if train else 10_000))
        self.extra_features = list(extra_features or [])
        self.extra_mean = extra_mean.astype(np.float32, copy=False) if extra_mean is not None else None
        self.extra_std = extra_std.astype(np.float32, copy=False) if extra_std is not None else None
        self.point_features = list(point_features or ["xyz"])
        self.feature_cache_dir = feature_cache_dir
        self.feature_k = int(feature_k)
        self.tooth_position_dropout = float(tooth_position_dropout)
        self.input_normalize = str(input_normalize or "").strip().lower() or "none"
        self.input_pca_align = bool(input_pca_align)
        self.input_pca_align_globalz = bool(input_pca_align_globalz)

        # Feature slices for augmentation transforms.
        self._slice_xyz: slice | None = None
        self._slice_normals: slice | None = None
        self._idx_curv: int | None = None
        self._idx_radius: int | None = None
        off = 0
        for name in self.point_features:
            if name == "xyz":
                self._slice_xyz = slice(off, off + 3)
                off += 3
            elif name == "normals":
                self._slice_normals = slice(off, off + 3)
                off += 3
            elif name == "curvature":
                self._idx_curv = off
                off += 1
            elif name == "radius":
                self._idx_radius = off
                off += 1
            elif name == "rgb":
                off += 3
            elif name == "cloud_id":
                off += 1
            elif name == "cloud_id_onehot":
                off += 10
            else:
                raise ValueError(f"Unknown point feature: {name}")

    def __len__(self) -> int:  # noqa: D401
        return len(self.rows)

    def _det_seed(self, s: str) -> int:
        return int(zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF)

    def _subsample_indices(
        self,
        n_total: int,
        *,
        r: dict[str, Any],
        cloud_id: np.ndarray | None,
        deterministic: bool,
    ) -> np.ndarray:
        if self.n_points <= 0 or int(n_total) == int(self.n_points):
            return np.arange(int(n_total), dtype=np.int64)

        if bool(deterministic):
            case_key = str(r.get("case_key") or "")
            rel = str(r.get("sample_npz") or "")
            seed = self._det_seed(f"rawcls|subsample|seed={self.seed}|case={case_key}|rel={rel}|n={n_total}|keep={self.n_points}")
            rng = np.random.default_rng(int(seed))
        else:
            rng = self.rng

        replace = int(n_total) < int(self.n_points)
        if cloud_id is None:
            return rng.choice(int(n_total), size=int(self.n_points), replace=replace).astype(np.int64, copy=False)

        cid = np.asarray(cloud_id).reshape(-1).astype(np.int16, copy=False)
        if int(cid.shape[0]) != int(n_total):
            raise ValueError(f"Invalid cloud_id shape {tuple(cid.shape)} for n_total={n_total}")

        unique = sorted({int(x) for x in np.unique(cid).tolist()})
        if not unique:
            return rng.choice(int(n_total), size=int(self.n_points), replace=replace).astype(np.int64, copy=False)

        k = int(len(unique))
        base = int(self.n_points) // k
        rem = int(self.n_points) - (base * k)
        counts = {u: int(np.sum(cid == int(u))) for u in unique}
        order = sorted(unique, key=lambda u: (-counts[u], int(u)))
        quotas = {u: base for u in unique}
        for j in range(rem):
            quotas[order[j]] += 1

        sel_parts: list[np.ndarray] = []
        for u in unique:
            idx_u = np.nonzero(cid == int(u))[0]
            if idx_u.size == 0:
                continue
            need = int(quotas[u])
            rep = int(idx_u.size) < need
            chosen = rng.choice(idx_u, size=need, replace=rep).astype(np.int64, copy=False)
            sel_parts.append(chosen)

        if not sel_parts:
            return rng.choice(int(n_total), size=int(self.n_points), replace=replace).astype(np.int64, copy=False)

        sel = np.concatenate(sel_parts, axis=0)
        if int(sel.shape[0]) != int(self.n_points):
            sel = rng.choice(sel, size=int(self.n_points), replace=int(sel.shape[0]) < int(self.n_points)).astype(np.int64, copy=False)
        # Mix cloud blocks (stable across train/eval by using same rng).
        return sel[rng.permutation(int(sel.shape[0]))]

    def _extract_extra(self, r: dict[str, Any]) -> np.ndarray:
        if not self.extra_features:
            return np.zeros((0,), dtype=np.float32)
        feats: list[float] = []
        tp = str(r.get("tooth_position") or "")
        tp_valid = tp in {"前磨牙", "磨牙"}
        for name in self.extra_features:
            if name == "scale":
                feats.append(float(r.get("scale") or 1.0))
            elif name == "log_scale":
                feats.append(float(math.log(max(1e-8, float(r.get("scale") or 1.0)))))
            elif name == "points":
                feats.append(float(r.get("n_points_after_cap") or 0))
            elif name == "log_points":
                feats.append(float(math.log1p(float(r.get("n_points_after_cap") or 0))))
            elif name == "objects_used":
                feats.append(float(r.get("n_objects_used") or 0))
            elif name == "tooth_position_premolar":
                feats.append(1.0 if tp == "前磨牙" else 0.0)
            elif name == "tooth_position_molar":
                feats.append(1.0 if tp == "磨牙" else 0.0)
            elif name == "tooth_position_missing":
                feats.append(0.0 if tp_valid else 1.0)
            else:
                raise ValueError(f"Unknown extra feature: {name}")
        x = np.asarray(feats, dtype=np.float32)
        if self.extra_mean is not None and self.extra_std is not None and x.size:
            x = (x - self.extra_mean) / self.extra_std
        return x.astype(np.float32, copy=False)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int, dict[str, Any]]:
        r = self.rows[int(idx)]
        rel = r["sample_npz"]
        if self.load_points:
            want_feats = self.point_features != ["xyz"]
            cache_path = None
            feat: np.ndarray | None = None
            pf_str = ",".join(self.point_features)
            dim_expected = point_feature_dim(self.point_features)
            in_norm = str(self.input_normalize or "").strip().lower() or "none"
            pca0 = bool(self.input_pca_align)
            pca_gz0 = bool(self.input_pca_align_globalz)

            target_points = int(r.get("target_points") or 0)
            can_use_cache = int(target_points) == int(self.n_points) and int(self.n_points) > 0

            if want_feats and self.feature_cache_dir is not None and can_use_cache:
                cache_path = (self.feature_cache_dir / str(rel)).resolve()
                if cache_path.exists():
                    try:
                        with np.load(cache_path) as z:
                            feat = np.asarray(z["feat"], dtype=np.float32)
                            pf0 = str(z.get("point_features", "")).strip()
                            k0 = int(np.asarray(z.get("k", 0)).reshape(-1)[0]) if "k" in z else 0
                            n0 = int(np.asarray(z.get("n_points", 0)).reshape(-1)[0]) if "n_points" in z else 0
                            in_norm0 = str(np.asarray(z.get("input_normalize", "")).reshape(-1)[0]) if "input_normalize" in z else ""
                            pca1 = bool(int(np.asarray(z.get("input_pca_align", 0)).reshape(-1)[0])) if "input_pca_align" in z else False
                            pca_gz1 = (
                                bool(int(np.asarray(z.get("input_pca_align_globalz", 0)).reshape(-1)[0]))
                                if "input_pca_align_globalz" in z
                                else False
                            )
                        if (
                            feat.ndim != 2
                            or feat.shape[0] != self.n_points
                            or feat.shape[1] != dim_expected
                            or pf0 != pf_str
                            or k0 != int(self.feature_k)
                            or n0 != int(self.n_points)
                            or str(in_norm0).strip().lower() != in_norm
                            or bool(pca1) != pca0
                            or bool(pca_gz1) != pca_gz0
                        ):
                            feat = None
                    except Exception:
                        feat = None

            if want_feats and feat is None:
                npz_path = self.data_root / str(rel)
                with np.load(npz_path) as data:
                    pts_xyz = np.asarray(data["points"], dtype=np.float32)
                    rgb_u8 = np.asarray(data["rgb"]) if "rgb" in self.point_features and "rgb" in data.files else None
                    cloud_id = (
                        np.asarray(data["cloud_id"])
                        if any(pf in self.point_features for pf in {"cloud_id", "cloud_id_onehot"}) and "cloud_id" in data.files
                        else None
                    )
                if pts_xyz.ndim != 2 or pts_xyz.shape[1] != 3:
                    raise ValueError(f"Invalid points shape {pts_xyz.shape} in {npz_path}")
                if "rgb" in self.point_features and rgb_u8 is None:
                    raise ValueError(f"Missing `rgb` in {npz_path} (point_features={self.point_features})")
                if rgb_u8 is not None and (rgb_u8.ndim != 2 or rgb_u8.shape[0] != pts_xyz.shape[0] or rgb_u8.shape[1] != 3):
                    raise ValueError(f"Invalid rgb shape {rgb_u8.shape} in {npz_path} for points {pts_xyz.shape}")
                if any(pf in self.point_features for pf in {"cloud_id", "cloud_id_onehot"}) and cloud_id is None:
                    raise ValueError(f"Missing `cloud_id` in {npz_path} (point_features={self.point_features})")
                if cloud_id is not None:
                    cid = np.asarray(cloud_id).reshape(-1)
                    if cid.shape[0] != pts_xyz.shape[0]:
                        raise ValueError(f"Invalid cloud_id shape {tuple(cid.shape)} in {npz_path} for points {pts_xyz.shape}")
                    cloud_id = cid
                if in_norm not in {"none", "off"} or pca0:
                    pts_xyz = apply_input_preprocess_np(
                        pts_xyz,
                        input_normalize=in_norm,
                        pca_align=pca0,
                        pca_align_globalz=pca_gz0,
                    )
                if self.n_points > 0 and pts_xyz.shape[0] != self.n_points:
                    sel = self._subsample_indices(
                        int(pts_xyz.shape[0]),
                        r=r,
                        cloud_id=cloud_id,
                        deterministic=not bool(self.train),
                    )
                    pts_xyz = pts_xyz[sel]
                    if rgb_u8 is not None:
                        rgb_u8 = rgb_u8[sel]
                    if cloud_id is not None:
                        cloud_id = cloud_id[sel]
                feat = build_point_features_from_xyz(
                    pts_xyz,
                    point_features=self.point_features,
                    k=int(self.feature_k),
                    device=torch.device("cpu"),
                    rgb_u8_np=rgb_u8,
                    cloud_id_np=cloud_id,
                )
                if cache_path is not None and can_use_cache:
                    atomic_write_npz(
                        cache_path,
                        arrays={
                            "feat": feat,
                            "point_features": np.asarray(pf_str),
                            "k": np.asarray(int(self.feature_k), dtype=np.int32),
                            "n_points": np.asarray(int(self.n_points), dtype=np.int32),
                            "input_normalize": np.asarray(str(in_norm)),
                            "input_pca_align": np.asarray(int(pca0), dtype=np.int32),
                            "input_pca_align_globalz": np.asarray(int(pca_gz0), dtype=np.int32),
                        },
                    )

            if want_feats:
                assert feat is not None
                pts_feat = feat.copy()
                if self.train:
                    if self.aug_rotate_z and self._slice_xyz is not None:
                        angle = float(self.rng.uniform(0.0, 2.0 * math.pi))
                        pts_feat[:, self._slice_xyz] = rotate_z(pts_feat[:, self._slice_xyz], angle)
                        if self._slice_normals is not None:
                            pts_feat[:, self._slice_normals] = rotate_z(pts_feat[:, self._slice_normals], angle)

                    if self.aug_scale and self.aug_scale > 0 and self._slice_xyz is not None:
                        lo = 1.0 - self.aug_scale
                        hi = 1.0 + self.aug_scale
                        scale = float(self.rng.uniform(lo, hi))
                        pts_feat[:, self._slice_xyz] = (pts_feat[:, self._slice_xyz] * scale).astype(np.float32, copy=False)
                        if self._idx_radius is not None:
                            pts_feat[:, self._idx_radius] = (pts_feat[:, self._idx_radius] * scale).astype(np.float32, copy=False)

                    if self._slice_xyz is not None:
                        pts_feat[:, self._slice_xyz] = jitter(
                            pts_feat[:, self._slice_xyz],
                            self.rng,
                            sigma=self.aug_jitter_sigma,
                            clip=self.aug_jitter_clip,
                        )

                    if self.aug_dropout_ratio and self.aug_dropout_ratio > 0:
                        dropout_ratio = float(self.rng.random()) * float(self.aug_dropout_ratio)
                        if dropout_ratio > 0:
                            n = pts_feat.shape[0]
                            drop_idx = self.rng.random(n) < dropout_ratio
                            if np.any(drop_idx):
                                # Preserve non-geometry channels (e.g. cloud_id) to avoid corrupting cloud grouping.
                                if self._slice_xyz is not None:
                                    pts_feat[drop_idx, self._slice_xyz] = pts_feat[0, self._slice_xyz]
                                    if self._slice_normals is not None:
                                        pts_feat[drop_idx, self._slice_normals] = pts_feat[0, self._slice_normals]
                                    if self._idx_curv is not None:
                                        pts_feat[drop_idx, self._idx_curv] = pts_feat[0, self._idx_curv]
                                    if self._idx_radius is not None:
                                        pts_feat[drop_idx, self._idx_radius] = pts_feat[0, self._idx_radius]
                                else:
                                    pts_feat[drop_idx] = pts_feat[0]
                pts = pts_feat
            else:
                npz_path = self.data_root / str(rel)
                with np.load(npz_path) as data:
                    pts = np.asarray(data["points"], dtype=np.float32)
                if pts.ndim != 2 or pts.shape[1] != 3:
                    raise ValueError(f"Invalid points shape {pts.shape} in {npz_path}")
                if in_norm not in {"none", "off"} or pca0:
                    pts = apply_input_preprocess_np(
                        pts,
                        input_normalize=in_norm,
                        pca_align=pca0,
                        pca_align_globalz=pca_gz0,
                    )

                if self.n_points > 0 and pts.shape[0] != self.n_points:
                    # Should already be fixed in Phase 1, but keep safe.
                    sel = self._subsample_indices(
                        int(pts.shape[0]),
                        r=r,
                        cloud_id=None,
                        deterministic=not bool(self.train),
                    )
                    pts = pts[sel]

                if self.train:
                    if self.aug_rotate_z:
                        angle = float(self.rng.uniform(0.0, 2.0 * math.pi))
                        pts = rotate_z(pts, angle)

                    if self.aug_scale and self.aug_scale > 0:
                        lo = 1.0 - self.aug_scale
                        hi = 1.0 + self.aug_scale
                        scale = float(self.rng.uniform(lo, hi))
                        pts = (pts * scale).astype(np.float32, copy=False)

                    pts = jitter(pts, self.rng, sigma=self.aug_jitter_sigma, clip=self.aug_jitter_clip)
                    pts = random_point_dropout(pts, self.rng, max_dropout_ratio=self.aug_dropout_ratio)
        else:
            n_dummy = max(1, int(self.n_points))
            pts = np.zeros((n_dummy, 3), dtype=np.float32)

        y = self.label_to_id[str(r["label"])]
        tp_raw = r.get("tooth_position")
        tp = str(tp_raw or "")
        tp_valid = tp in {"前磨牙", "磨牙"}
        drop_tp = False
        if self.train and tp_valid and self.tooth_position_dropout > 0:
            drop_tp = bool(self.rng.random() < float(self.tooth_position_dropout))
        tp_used: str | None = tp if (tp_valid and not drop_tp) else None
        r_eff = r if tp_used == r.get("tooth_position") else {**r, "tooth_position": tp_used}
        extra = self._extract_extra(r_eff)
        meta = {
            "case_key": r_eff.get("case_key"),
            "split": r_eff.get("split"),
            "source": r_eff.get("source"),
            "tooth_position": r_eff.get("tooth_position"),
            "tooth_position_raw": (tp if tp_valid else None),
            "tooth_position_dropped": bool(drop_tp),
            "label": r_eff.get("label"),
            "label_raw": r_eff.get("label_raw"),
            "n_objects_used": r_eff.get("n_objects_used"),
            "n_points_after_cap": r_eff.get("n_points_after_cap"),
            "scale": r_eff.get("scale"),
            "sample_npz": rel,
        }
        return torch.from_numpy(pts), torch.from_numpy(extra), int(y), meta


# Backwards-friendly alias (older code used the private name).
_atomic_write_npz = atomic_write_npz
