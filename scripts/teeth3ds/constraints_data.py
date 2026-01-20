from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from _lib.hash import stable_int_seed
from teeth3ds.obj_io import parse_obj_vertices


def opp_jaw(jaw: str) -> str:
    return "lower" if str(jaw) == "upper" else "upper"


def opp_fdi(fdi: int) -> int | None:
    """Map a tooth FDI label to its opposing jaw FDI label."""

    x = int(fdi)
    if 11 <= x <= 18:
        return x + 30  # 11 -> 41
    if 21 <= x <= 28:
        return x + 10  # 21 -> 31
    if 31 <= x <= 38:
        return x - 10  # 31 -> 21
    if 41 <= x <= 48:
        return x - 30  # 41 -> 11
    return None


def split_case_key(case_key: str) -> tuple[str, str]:
    case_id, jaw = str(case_key).rsplit("_", 1)
    return case_id, jaw


class OpposingPointsCache:
    """Cache opposing jaw/tooth point samples for occlusion constraints."""

    def __init__(
        self,
        *,
        teeth3ds_dir: Path,
        n_points: int,
        seed: int,
        mode: str = "jaw",
        min_points: int = 200,
        cache_dir: Path | None = None,
        write_cache: bool = False,
    ) -> None:
        self.teeth3ds_dir = teeth3ds_dir
        self.n_points = int(n_points)
        self.seed = int(seed)
        self.mode = str(mode).strip().lower()
        if self.mode not in {"jaw", "tooth"}:
            raise ValueError(f"Unknown occlusion mode: {mode} (allowed: jaw, tooth)")
        self.min_points = int(min_points)
        self.cache_dir = cache_dir
        self.write_cache = bool(write_cache)

        self._cache: dict[str, np.ndarray | None] = {}
        self._jaw_verts_cache: dict[tuple[str, str], np.ndarray] = {}
        self._jaw_labels_cache: dict[tuple[str, str], np.ndarray | None] = {}

    def _sample_n(self, pts: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
        if pts.shape[0] == n:
            return pts
        replace = pts.shape[0] < n
        idx = rng.choice(pts.shape[0], size=int(n), replace=replace)
        return pts[idx]

    def _load_jaw_verts_and_labels(self, case_id: str, jaw: str) -> tuple[np.ndarray, np.ndarray | None] | None:
        jaw_key = (str(case_id), str(jaw))
        verts = self._jaw_verts_cache.get(jaw_key)
        labels = self._jaw_labels_cache.get(jaw_key)
        if verts is not None and labels is not None:
            return verts, labels

        obj_path = self.teeth3ds_dir / jaw / case_id / f"{case_id}_{jaw}.obj"
        if not obj_path.exists():
            self._jaw_labels_cache[jaw_key] = None
            return None

        json_path = self.teeth3ds_dir / jaw / case_id / f"{case_id}_{jaw}.json"
        if not json_path.exists():
            self._jaw_labels_cache[jaw_key] = None
            return None

        try:
            verts = parse_obj_vertices(obj_path, stop_at_faces=True)
            seg = json.loads(json_path.read_text(encoding="utf-8"))
            labels_arr = np.asarray(seg.get("labels", []), dtype=np.int32)
        except Exception:
            self._jaw_labels_cache[jaw_key] = None
            return None

        if labels_arr.shape[0] != verts.shape[0]:
            self._jaw_labels_cache[jaw_key] = None
            return None

        self._jaw_verts_cache[jaw_key] = verts
        self._jaw_labels_cache[jaw_key] = labels_arr
        return verts, labels_arr

    def get(self, case_id: str, jaw: str, *, fdi: int | None = None) -> np.ndarray | None:
        if self.mode == "jaw":
            key = f"{case_id}_{jaw}"
        else:
            if fdi is None:
                return None
            key = f"{case_id}_{jaw}_fdi{int(fdi)}"
        if key in self._cache:
            return self._cache[key]

        if self.cache_dir is not None:
            cache_path = self.cache_dir / f"{key}.npz"
            if cache_path.exists():
                try:
                    with np.load(cache_path) as data:
                        pts = np.asarray(data["points"], dtype=np.float32)
                    if pts.ndim == 2 and pts.shape[1] == 3:
                        self._cache[key] = pts.astype(np.float32, copy=False)
                        return self._cache[key]
                except Exception:
                    pass

        rng = np.random.default_rng(stable_int_seed(key, self.seed))
        try:
            if self.mode == "jaw":
                obj_path = self.teeth3ds_dir / jaw / case_id / f"{case_id}_{jaw}.obj"
                if not obj_path.exists():
                    self._cache[key] = None
                    return None
                verts = parse_obj_vertices(obj_path, stop_at_faces=True)
                sampled = self._sample_n(verts, self.n_points, rng).astype(np.float32, copy=False)
            else:
                loaded = self._load_jaw_verts_and_labels(str(case_id), str(jaw))
                if loaded is None:
                    self._cache[key] = None
                    return None
                verts, labels_arr = loaded
                if labels_arr is None:
                    self._cache[key] = None
                    return None

                mask = labels_arr == int(fdi)
                pts = verts[mask]
                if int(pts.shape[0]) < int(self.min_points):
                    self._cache[key] = None
                    return None
                sampled = self._sample_n(pts, self.n_points, rng).astype(np.float32, copy=False)
        except Exception:
            self._cache[key] = None
            return None

        self._cache[key] = sampled
        if self.cache_dir is not None and self.write_cache:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(self.cache_dir / f"{key}.npz", points=sampled)
            except Exception:
                pass
        return sampled


@dataclass(frozen=True)
class ConstraintsSample:
    prep: torch.Tensor
    tgt: torch.Tensor
    margin: torch.Tensor
    centroid: torch.Tensor
    scale: torch.Tensor
    r: torch.Tensor
    case_id: str
    jaw: str
    instance_id: int
    fdi: int
    sample_npz: str
    cut_q: float
    cut_thr: float
    cut_n: torch.Tensor


class Teeth3DSConstraintsPrepTargetDataset(Dataset[ConstraintsSample]):
    def __init__(
        self,
        *,
        rows: list[dict[str, Any]],
        data_root: Path,
        n_points: int,
        seed: int,
        cut_mode: str,
        cut_q_min: float,
        cut_q_max: float,
        margin_band: float,
        margin_points: int,
        deterministic: bool,
        rng_offset: int,
    ) -> None:
        self.rows = rows
        self.data_root = data_root
        self.n_points = int(n_points)
        self.seed = int(seed)
        self.cut_mode = str(cut_mode).strip().lower()
        if self.cut_mode not in {"z", "plane"}:
            raise ValueError(f"Unknown cut_mode: {cut_mode} (allowed: z, plane)")
        self.cut_q_min = float(cut_q_min)
        self.cut_q_max = float(cut_q_max)
        self.margin_band = float(margin_band)
        self.margin_points = int(margin_points)
        self.deterministic = bool(deterministic)
        self.rng = np.random.default_rng(int(seed) + int(rng_offset))

    def __len__(self) -> int:  # noqa: D401
        return len(self.rows)

    def _sample_n(self, pts: np.ndarray, n: int) -> np.ndarray:
        if pts.shape[0] == n:
            return pts
        replace = pts.shape[0] < n
        idx = self.rng.choice(pts.shape[0], size=int(n), replace=replace)
        return pts[idx]

    def _cut_q(self, meta: dict[str, Any]) -> float:
        if not self.deterministic:
            q = float(self.rng.uniform(self.cut_q_min, self.cut_q_max))
            return float(np.clip(q, 0.05, 0.95))
        key = f"{meta.get('case_key')}__{meta.get('jaw')}__{meta.get('instance_id')}__{meta.get('fdi')}"
        rng = np.random.default_rng(stable_int_seed(key, self.seed))
        q = float(rng.uniform(self.cut_q_min, self.cut_q_max))
        return float(np.clip(q, 0.05, 0.95))

    def _cut_normal(self, meta: dict[str, Any]) -> np.ndarray:
        if self.cut_mode == "z":
            return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        key = f"{meta.get('case_key')}__{meta.get('jaw')}__{meta.get('instance_id')}__{meta.get('fdi')}__plane"
        rng = self.rng if not self.deterministic else np.random.default_rng(stable_int_seed(key, self.seed))
        v = rng.normal(size=(3,)).astype(np.float32)
        n = float(np.linalg.norm(v))
        if not np.isfinite(n) or n < 1e-6:
            return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        return (v / n).astype(np.float32, copy=False)

    def __getitem__(self, idx: int) -> ConstraintsSample:
        r = self.rows[int(idx)]
        npz_path = self.data_root / str(r["sample_npz"])
        with np.load(npz_path) as data:
            tgt = np.asarray(data["points"], dtype=np.float32)
        if tgt.ndim != 2 or tgt.shape[1] != 3:
            raise ValueError(f"Invalid points shape {tgt.shape} in {npz_path}")

        if self.n_points > 0 and tgt.shape[0] != self.n_points:
            tgt = self._sample_n(tgt, self.n_points)

        q = self._cut_q(r)
        nrm = self._cut_normal(r)
        proj = tgt @ nrm
        thr = float(np.quantile(proj, q))

        kept = tgt[proj <= thr]
        if kept.shape[0] < max(16, int(0.2 * self.n_points)):
            kept = tgt
        prep = self._sample_n(kept, self.n_points)

        band = max(1e-6, float(self.margin_band))
        margin_candidates = tgt[np.abs(proj - thr) <= band]
        if margin_candidates.shape[0] < 4:
            margin_candidates = kept
        margin_pts = self._sample_n(margin_candidates, self.margin_points)

        case_id, jaw = split_case_key(str(r.get("case_key") or ""))
        instance_id = int(r.get("instance_id") or 0)
        fdi = int(r.get("fdi") or 0)
        sample_npz = str(r.get("sample_npz") or "")

        centroid_np = r.get("centroid")
        centroid = (
            np.asarray(centroid_np, dtype=np.float32).reshape(3) if centroid_np is not None else np.zeros((3,), dtype=np.float32)
        )
        r_np = r.get("R")
        rmat = np.asarray(r_np, dtype=np.float32).reshape(3, 3) if r_np is not None else np.eye(3, dtype=np.float32)
        scale = float(r.get("scale") or 1.0)

        return ConstraintsSample(
            prep=torch.from_numpy(prep),
            tgt=torch.from_numpy(tgt),
            margin=torch.from_numpy(margin_pts),
            centroid=torch.from_numpy(centroid),
            scale=torch.tensor(scale, dtype=torch.float32),
            r=torch.from_numpy(rmat),
            case_id=str(case_id),
            jaw=str(jaw),
            instance_id=int(instance_id),
            fdi=int(fdi),
            sample_npz=sample_npz,
            cut_q=float(q),
            cut_thr=float(thr),
            cut_n=torch.from_numpy(np.asarray(nrm, dtype=np.float32)),
        )


def collate_constraints(batch: list[ConstraintsSample]) -> dict[str, Any]:
    return {
        "prep": torch.stack([b.prep for b in batch], dim=0),
        "tgt": torch.stack([b.tgt for b in batch], dim=0),
        "margin": torch.stack([b.margin for b in batch], dim=0),
        "centroid": torch.stack([b.centroid for b in batch], dim=0),
        "scale": torch.stack([b.scale for b in batch], dim=0),
        "R": torch.stack([b.r for b in batch], dim=0),
        "case_id": [b.case_id for b in batch],
        "jaw": [b.jaw for b in batch],
        "instance_id": [int(b.instance_id) for b in batch],
        "fdi": [int(b.fdi) for b in batch],
        "sample_npz": [b.sample_npz for b in batch],
        "cut_q": torch.tensor([float(b.cut_q) for b in batch], dtype=torch.float32),
        "cut_thr": torch.tensor([float(b.cut_thr) for b in batch], dtype=torch.float32),
        "cut_n": torch.stack([b.cut_n for b in batch], dim=0),
    }

