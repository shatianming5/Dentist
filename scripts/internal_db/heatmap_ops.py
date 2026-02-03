from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .mesh_io import OptionalDependencyError, pca_align


def _require(name: str) -> Any:
    try:
        return __import__(name)
    except Exception as e:  # noqa: BLE001
        raise OptionalDependencyError(
            f"Missing optional dependency '{name}'. Install via: pip install -r configs/env/requirements_vis.txt"
        ) from e


def normalize01(x: np.ndarray) -> np.ndarray:
    xx = np.asarray(x, dtype=np.float32).reshape(-1)
    if xx.size == 0:
        return xx
    lo = float(np.nanmin(xx))
    hi = float(np.nanmax(xx))
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        return np.zeros_like(xx, dtype=np.float32)
    out = (xx - lo) / (hi - lo)
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    return out.astype(np.float32, copy=False)


def polydata_faces_to_triangles(poly: Any) -> np.ndarray:
    faces = np.asarray(poly.faces)
    if faces.size == 0:
        return np.zeros((0, 3), dtype=np.int64)
    if faces.size % 4 != 0:
        raise ValueError("Expected triangle faces encoded as [3,i,j,k] blocks")
    f = faces.reshape(-1, 4)
    if not np.all(f[:, 0] == 3):
        raise ValueError("Non-triangle faces found; call triangulate() before exporting")
    return f[:, 1:4].astype(np.int64, copy=False)


def sample_points_on_polydata(poly: Any, *, n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Sample points uniformly on surface. Returns (points, face_index)."""
    trimesh = _require("trimesh")
    tri = polydata_faces_to_triangles(poly)
    tm = trimesh.Trimesh(vertices=np.asarray(poly.points, dtype=np.float64), faces=tri, process=False)
    if int(n) <= 0:
        raise ValueError("n must be >0")
    rng = np.random.default_rng(int(seed))
    pts, face_idx = trimesh.sample.sample_surface(tm, int(n), seed=rng)
    return np.asarray(pts, dtype=np.float32), np.asarray(face_idx, dtype=np.int64)


def map_point_scalar_to_vertices(
    *,
    vertices: np.ndarray,
    points: np.ndarray,
    scalar: np.ndarray,
    mode: str = "knn_mean",
    k: int = 8,
) -> np.ndarray:
    """Map scalar values defined on `points` onto mesh `vertices`."""
    from scipy.spatial import cKDTree

    v = np.asarray(vertices, dtype=np.float32)
    p = np.asarray(points, dtype=np.float32)
    s = np.asarray(scalar, dtype=np.float32).reshape(-1)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"vertices must be (V,3), got {tuple(v.shape)}")
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"points must be (N,3), got {tuple(p.shape)}")
    if s.shape[0] != p.shape[0]:
        raise ValueError(f"scalar length must match points: scalar={s.shape[0]} points={p.shape[0]}")

    m = str(mode).strip().lower()
    tree = cKDTree(p.astype(np.float64, copy=False))
    if m == "nearest":
        _, idx = tree.query(v.astype(np.float64, copy=False), k=1)
        return s[np.asarray(idx, dtype=np.int64)].astype(np.float32, copy=False)
    if m == "knn_mean":
        kk = max(1, int(k))
        _, idx = tree.query(v.astype(np.float64, copy=False), k=kk)
        idx = np.asarray(idx, dtype=np.int64)
        if idx.ndim == 1:
            return s[idx].astype(np.float32, copy=False)
        return np.mean(s[idx], axis=1).astype(np.float32, copy=False)
    raise ValueError(f"Unknown map mode: {mode!r} (supported: nearest, knn_mean)")


def smooth_vertex_scalar_knn(
    vertices: np.ndarray,
    scalar: np.ndarray,
    *,
    k: int = 12,
    iters: int = 1,
    alpha: float = 0.6,
) -> np.ndarray:
    """Smooth a per-vertex scalar by iterated kNN averaging.

    This is purely for visualization (reduce speckle/noise). It does not
    change mesh geometry.
    """
    from scipy.spatial import cKDTree

    v = np.asarray(vertices, dtype=np.float32)
    s0 = np.asarray(scalar, dtype=np.float32).reshape(-1)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"vertices must be (V,3), got {tuple(v.shape)}")
    if s0.shape[0] != v.shape[0]:
        raise ValueError(f"scalar length must match vertices: scalar={s0.shape[0]} vertices={v.shape[0]}")

    n = int(v.shape[0])
    kk = int(k)
    if n <= 0 or int(iters) <= 0 or kk <= 1:
        return s0.astype(np.float32, copy=False)
    kk = min(kk, n)

    a = float(alpha)
    a = max(0.0, min(1.0, a))
    if a <= 1e-9:
        return s0.astype(np.float32, copy=False)

    tree = cKDTree(v.astype(np.float64, copy=False))
    _dist, nn = tree.query(v.astype(np.float64, copy=False), k=kk)
    nn = np.asarray(nn, dtype=np.int64)
    if nn.ndim == 1:
        nn = nn[:, None]

    s = s0.astype(np.float32, copy=True)
    for _ in range(int(iters)):
        mean = np.mean(s[nn], axis=1).astype(np.float32, copy=False)
        s = (1.0 - a) * s + a * mean
    return s.astype(np.float32, copy=False)


def pca_initial_transform(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    """Compute a rough rigid init transform (4x4) using PCA."""
    src = np.asarray(source_points, dtype=np.float32)
    tgt = np.asarray(target_points, dtype=np.float32)
    if src.ndim != 2 or src.shape[1] != 3 or src.shape[0] < 3:
        raise ValueError("source_points must be (N,3) with N>=3")
    if tgt.ndim != 2 or tgt.shape[1] != 3 or tgt.shape[0] < 3:
        raise ValueError("target_points must be (M,3) with M>=3")

    R_s, t_s = pca_align(src)
    R_t, t_t = pca_align(tgt)
    rot = (R_s @ R_t.T).astype(np.float64)
    trans = (-t_s.astype(np.float64) @ rot + t_t.astype(np.float64)).astype(np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rot
    T[:3, 3] = trans
    return T.astype(np.float64, copy=False)


def compute_custom_scalar_on_vertices(poly: Any, *, kind: str) -> np.ndarray:
    """Compute a simple scalar field on mesh vertices."""
    k = str(kind).strip().lower()
    pts = np.asarray(poly.points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("poly.points must be (V,3)")
    if k in {"z", "height"}:
        return normalize01(pts[:, 2])
    if k in {"y"}:
        return normalize01(pts[:, 1])
    if k in {"x"}:
        return normalize01(pts[:, 0])
    if k in {"curvature", "curvature_mean", "mean_curvature"}:
        curv = np.asarray(poly.curvature(curv_type="mean"), dtype=np.float32)
        curv = np.nan_to_num(curv, nan=0.0, posinf=0.0, neginf=0.0)
        curv = np.abs(curv)
        return normalize01(curv)
    if k in {"curvature_gaussian", "gaussian_curvature"}:
        curv = np.asarray(poly.curvature(curv_type="gaussian"), dtype=np.float32)
        curv = np.nan_to_num(curv, nan=0.0, posinf=0.0, neginf=0.0)
        curv = np.abs(curv)
        return normalize01(curv)
    raise ValueError(
        f"Unknown scalar kind: {kind!r} (supported: z,x,y,curvature_mean,curvature_gaussian)"
    )


@dataclass(frozen=True)
class HeatField:
    name: str
    points: np.ndarray  # (N,3)
    heat01: np.ndarray  # (N,)
