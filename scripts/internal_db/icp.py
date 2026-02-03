from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .heatmap_ops import pca_initial_transform
from .mesh_io import OptionalDependencyError


def _require(name: str) -> Any:
    try:
        return __import__(name)
    except Exception as e:  # noqa: BLE001
        raise OptionalDependencyError(
            f"Missing optional dependency '{name}'. Install via: pip install -r configs/env/requirements_vis.txt"
        ) from e


@dataclass(frozen=True)
class ICPResult:
    matrix_4x4: np.ndarray
    aligned: np.ndarray
    cost: float


def icp_align(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    max_iterations: int = 50,
    threshold: float = 1e-5,
    use_pca_init: bool = True,
    initial: np.ndarray | None = None,
) -> ICPResult:
    """Rigid ICP align source -> target.

    Notes:
      - trimesh ICP needs a rough initial transform; we use PCA by default.
      - Returns a 4x4 transform and aligned points.
    """
    trimesh = _require("trimesh")

    src = np.asarray(source_points, dtype=np.float64)
    tgt = np.asarray(target_points, dtype=np.float64)
    if src.ndim != 2 or src.shape[1] != 3 or src.shape[0] < 3:
        raise ValueError("source_points must be (N,3) with N>=3")
    if tgt.ndim != 2 or tgt.shape[1] != 3 or tgt.shape[0] < 3:
        raise ValueError("target_points must be (M,3) with M>=3")

    init = None
    if initial is not None:
        init = np.asarray(initial, dtype=np.float64)
        if init.shape != (4, 4):
            raise ValueError("initial must be 4x4")
    elif bool(use_pca_init):
        init = pca_initial_transform(src.astype(np.float32), tgt.astype(np.float32))

    mat, aligned, cost = trimesh.registration.icp(
        src,
        tgt,
        initial=init,
        threshold=float(threshold),
        max_iterations=int(max_iterations),
    )
    return ICPResult(matrix_4x4=np.asarray(mat, dtype=np.float64), aligned=np.asarray(aligned, dtype=np.float32), cost=float(cost))

