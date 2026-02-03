from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class OptionalDependencyError(RuntimeError):
    pass


def _require(name: str) -> Any:
    try:
        return __import__(name)
    except Exception as e:  # noqa: BLE001
        raise OptionalDependencyError(
            f"Missing optional dependency '{name}'. Install via: pip install -r configs/env/requirements_vis.txt"
        ) from e


@dataclass(frozen=True)
class MeshPrepResult:
    mesh: Any  # pyvista.PolyData
    transform_4x4: np.ndarray  # applied transform (world -> viewer)


def load_stl_as_polydata(path: Path) -> Any:
    pv = _require("pyvista")
    trimesh = _require("trimesh")

    m = trimesh.load_mesh(str(path), force="mesh")
    if m is None:
        raise ValueError(f"Failed to load mesh: {path}")
    if hasattr(trimesh, "Scene") and isinstance(m, trimesh.Scene):
        geoms = []
        for g in m.dump().values():
            geoms.append(g)
        if not geoms:
            raise ValueError(f"Empty scene: {path}")
        m = trimesh.util.concatenate(geoms)

    vertices = np.asarray(m.vertices, dtype=np.float32)
    faces = np.asarray(m.faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Expected triangular faces, got {faces.shape} from {path}")
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]).ravel()
    poly = pv.PolyData(vertices, faces_pv)
    return poly


def pca_align(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (R, t) such that aligned = (points - t) @ R.

    - R is a right-handed orthonormal basis
    - Sign is stabilized by extending positive range
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be Nx3")
    t = points.mean(axis=0, dtype=np.float64)
    x = points.astype(np.float64) - t
    cov = (x.T @ x) / max(1, x.shape[0])
    w, v = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    R = v[:, order]

    # Stabilize signs: prefer axis direction with larger positive extent.
    for i in range(3):
        proj = x @ R[:, i]
        if abs(proj.min()) > abs(proj.max()):
            R[:, i] *= -1.0

    # Ensure right-handed.
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1.0
    return R.astype(np.float32), t.astype(np.float32)


def prepare_polydata(
    poly: Any,
    *,
    align: str = "pca",
    decimate: float = 0.6,
    smooth_iters: int = 15,
) -> MeshPrepResult:
    pv = _require("pyvista")

    if poly is None:
        raise ValueError("poly is None")
    poly = poly.triangulate()
    poly = poly.clean(inplace=False)

    # Optional decimation (target_reduction in [0, 1)).
    dec = float(decimate)
    if dec > 0:
        dec = min(max(dec, 0.0), 0.95)
        try:
            poly = poly.decimate_pro(dec, preserve_topology=True)
        except Exception:
            # Fallback if preserve_topology not supported
            poly = poly.decimate_pro(dec)

    # Optional smoothing.
    if smooth_iters and int(smooth_iters) > 0:
        iters = int(smooth_iters)
        iters = max(0, min(iters, 200))
        poly = poly.smooth(n_iter=iters, relaxation_factor=0.01, feature_smoothing=False, boundary_smoothing=True)

    poly = poly.compute_normals(
        cell_normals=True,
        point_normals=True,
        consistent_normals=True,
        auto_orient_normals=True,
        inplace=False,
    )

    pts = np.asarray(poly.points, dtype=np.float32)
    T = np.eye(4, dtype=np.float32)

    mode = str(align).strip().lower()
    if mode == "none":
        return MeshPrepResult(mesh=poly, transform_4x4=T)
    if mode == "center":
        center = pts.mean(axis=0, dtype=np.float64).astype(np.float32)
        poly.points = (pts - center).astype(np.float32)
        T[:3, 3] = -center
        return MeshPrepResult(mesh=poly, transform_4x4=T)
    if mode == "pca":
        R, t = pca_align(pts)
        aligned = (pts.astype(np.float32) - t) @ R
        poly.points = aligned.astype(np.float32)
        T[:3, :3] = R
        T[:3, 3] = -t @ R
        return MeshPrepResult(mesh=poly, transform_4x4=T)
    raise ValueError(f"Unknown align mode: {align!r} (supported: none, center, pca)")


def compute_case_alignment(
    meshes: list[Any],
    *,
    mode: str,
    max_points: int = 200_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a single alignment transform for multiple meshes.

    Returns (R, t) such that aligned = (points - t) @ R.
    """
    if not meshes:
        raise ValueError("meshes must be non-empty")

    pts_list = [np.asarray(m.points, dtype=np.float32) for m in meshes]
    total = int(sum(p.shape[0] for p in pts_list))
    if total <= 0:
        raise ValueError("meshes have no points")

    pts_all: np.ndarray
    if total > int(max_points) and int(max_points) > 0:
        rng = np.random.default_rng(0)
        sampled: list[np.ndarray] = []
        for pts in pts_list:
            n = int(pts.shape[0])
            k = int(max(1, round(int(max_points) * n / total)))
            if k >= n:
                sampled.append(pts)
                continue
            idx = rng.choice(n, size=k, replace=False)
            sampled.append(pts[idx])
        pts_all = np.concatenate(sampled, axis=0)
    else:
        pts_all = np.concatenate(pts_list, axis=0)

    m = str(mode).strip().lower()
    if m == "none":
        return np.eye(3, dtype=np.float32), np.zeros((3,), dtype=np.float32)
    if m == "center":
        t = pts_all.mean(axis=0, dtype=np.float64).astype(np.float32)
        return np.eye(3, dtype=np.float32), t
    if m == "pca":
        R, t = pca_align(pts_all)
        return R.astype(np.float32), t.astype(np.float32)
    raise ValueError(f"Unknown mode: {mode!r} (supported: none, center, pca)")


def apply_alignment(mesh: Any, *, R: np.ndarray, t: np.ndarray) -> Any:
    """Apply (R, t) in-place: points <- (points - t) @ R."""
    pts = np.asarray(mesh.points, dtype=np.float32)
    mesh.points = ((pts - t.astype(np.float32)) @ R.astype(np.float32)).astype(np.float32)
    return mesh


def infer_camera(center: np.ndarray, radius: float, *, view: str) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    """Return (pos, focal, up) for pyvista camera_position."""
    c = center.astype(float).tolist()
    r = float(radius)
    d = max(1e-3, 2.5 * r)
    v = str(view).strip().lower()
    if v == "front":
        pos = (c[0], c[1] - d, c[2])
        up = (0.0, 0.0, 1.0)
    elif v == "back":
        pos = (c[0], c[1] + d, c[2])
        up = (0.0, 0.0, 1.0)
    elif v == "left":
        pos = (c[0] - d, c[1], c[2])
        up = (0.0, 0.0, 1.0)
    elif v == "right":
        pos = (c[0] + d, c[1], c[2])
        up = (0.0, 0.0, 1.0)
    elif v == "top":
        pos = (c[0], c[1], c[2] + d)
        up = (0.0, 1.0, 0.0)
    elif v == "bottom":
        pos = (c[0], c[1], c[2] - d)
        up = (0.0, 1.0, 0.0)
    else:
        raise ValueError(f"Unknown view: {view!r} (supported: front, back, left, right, top, bottom)")
    return (pos, tuple(c), up)


def bounds_center_radius(bounds: tuple[float, float, float, float, float, float]) -> tuple[np.ndarray, float]:
    xmin, xmax, ymin, ymax, zmin, zmax = [float(x) for x in bounds]
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    cz = 0.5 * (zmin + zmax)
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    radius = 0.5 * math.sqrt(max(1e-12, dx * dx + dy * dy + dz * dz))
    return np.asarray([cx, cy, cz], dtype=np.float32), float(radius)
