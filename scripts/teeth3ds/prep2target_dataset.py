from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from _lib.hash import stable_int_seed


class Teeth3DSPrepTargetDataset(Dataset[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]]):
    def __init__(
        self,
        *,
        rows: list[dict[str, Any]],
        data_root: Path,
        n_points: int,
        seed: int,
        train: bool,
        cut_mode: str,
        cut_q_min: float,
        cut_q_max: float,
    ) -> None:
        self.rows = rows
        self.data_root = data_root
        self.n_points = int(n_points)
        self.seed = int(seed)
        self.train = bool(train)
        self.cut_mode = str(cut_mode).strip().lower()
        if self.cut_mode not in {"z", "plane"}:
            raise ValueError(f"Unknown cut_mode: {cut_mode} (allowed: z, plane)")
        self.cut_q_min = float(cut_q_min)
        self.cut_q_max = float(cut_q_max)
        self.rng = np.random.default_rng(int(seed) + (0 if train else 10_000))

    def __len__(self) -> int:  # noqa: D401
        return len(self.rows)

    def _sample_n(self, pts: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
        if pts.shape[0] == n:
            return pts
        replace = pts.shape[0] < n
        idx = rng.choice(pts.shape[0], size=int(n), replace=replace)
        return pts[idx]

    def _cut_normal(self, meta: dict[str, Any]) -> np.ndarray:
        if self.cut_mode == "z":
            return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        key = f"{meta.get('case_key')}__{meta.get('jaw')}__{meta.get('instance_id')}__{meta.get('fdi')}__plane"
        rng = self.rng if self.train else np.random.default_rng(stable_int_seed(key, self.seed))
        v = rng.normal(size=(3,)).astype(np.float32)
        n = float(np.linalg.norm(v))
        if not np.isfinite(n) or n < 1e-6:
            return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        return (v / n).astype(np.float32, copy=False)

    def _make_prep(self, tgt: np.ndarray, meta: dict[str, Any]) -> tuple[np.ndarray, float, np.ndarray, float]:
        nrm = self._cut_normal(meta)
        proj = tgt @ nrm
        if self.train:
            q = float(self.rng.uniform(self.cut_q_min, self.cut_q_max))
        else:
            key = f"{meta.get('case_key')}__{meta.get('jaw')}__{meta.get('instance_id')}__{meta.get('fdi')}"
            rng = np.random.default_rng(stable_int_seed(key, self.seed))
            q = float(rng.uniform(self.cut_q_min, self.cut_q_max))

        q = float(np.clip(q, 0.05, 0.95))
        thr = float(np.quantile(proj, q))
        kept = tgt[proj <= thr]
        if kept.shape[0] < max(16, int(0.2 * self.n_points)):
            kept = tgt
        prep = self._sample_n(kept, self.n_points, self.rng)
        return prep, q, nrm, thr

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        r = self.rows[int(idx)]
        npz_path = self.data_root / str(r["sample_npz"])
        with np.load(npz_path) as data:
            tgt = np.asarray(data["points"], dtype=np.float32)
        if tgt.ndim != 2 or tgt.shape[1] != 3:
            raise ValueError(f"Invalid points shape {tgt.shape} in {npz_path}")

        if self.n_points > 0 and tgt.shape[0] != self.n_points:
            tgt = self._sample_n(tgt, self.n_points, self.rng)

        prep, q, nrm, thr = self._make_prep(tgt, r)
        meta = dict(r)
        meta["cut_q"] = float(q)
        meta["cut_mode"] = self.cut_mode
        meta["cut_n"] = np.asarray(nrm, dtype=np.float32)
        meta["cut_thr"] = float(thr)
        return torch.from_numpy(prep), torch.from_numpy(tgt), meta

