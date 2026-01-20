from __future__ import annotations

import math

import numpy as np


def rotate_z(points: np.ndarray, angle: float) -> np.ndarray:
    c = float(math.cos(angle))
    s = float(math.sin(angle))
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return points @ rot.T


def jitter(points: np.ndarray, rng: np.random.Generator, sigma: float, clip: float) -> np.ndarray:
    if sigma <= 0:
        return points
    noise = rng.normal(loc=0.0, scale=sigma, size=points.shape).astype(np.float32)
    if clip > 0:
        noise = np.clip(noise, -clip, clip)
    return points + noise


def random_point_dropout(points: np.ndarray, rng: np.random.Generator, max_dropout_ratio: float) -> np.ndarray:
    if max_dropout_ratio <= 0:
        return points
    dropout_ratio = float(rng.random()) * float(max_dropout_ratio)
    if dropout_ratio <= 0:
        return points
    n = points.shape[0]
    drop_idx = rng.random(n) < dropout_ratio
    if not np.any(drop_idx):
        return points
    points2 = points.copy()
    points2[drop_idx] = points2[0]
    return points2


def _safe_scale_div(points: np.ndarray, scale: float) -> np.ndarray:
    s = float(scale)
    if not math.isfinite(s) or s <= 0:
        s = 1.0
    return (points / s).astype(np.float32, copy=False)


def normalize_points_np(points: np.ndarray, mode: str) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected points (N,3), got {tuple(pts.shape)}")
    m = str(mode or "").strip().lower() or "none"
    if m in {"none", "off"}:
        return pts
    if pts.shape[0] <= 0:
        return pts
    if m == "max_norm":
        norms = np.linalg.norm(pts.astype(np.float64), axis=1)
        scale = float(np.max(norms)) if norms.size else 1.0
        return _safe_scale_div(pts, scale)
    if m == "bbox_diag":
        mn = pts.min(axis=0)
        mx = pts.max(axis=0)
        scale = float(np.linalg.norm((mx - mn).astype(np.float64)))
        return _safe_scale_div(pts, scale)
    raise ValueError(f"Unknown normalize mode: {mode}")


def pca_align_np(points: np.ndarray, *, align_globalz: bool = False) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected points (N,3), got {tuple(pts.shape)}")
    if pts.shape[0] <= 1:
        return pts

    x = pts.astype(np.float64, copy=False)
    cov = (x.T @ x) / max(1, x.shape[0])
    w, v = np.linalg.eigh(cov)  # ascending eigenvalues
    order = np.argsort(w)[::-1]
    r = v[:, order].astype(np.float32, copy=False)  # columns are axes
    r = r.copy()

    if bool(align_globalz):
        jz = int(np.argmax(np.abs(r[2, :])))
        remaining = [i for i in range(3) if i != jz]
        jx = remaining[int(np.argmax(np.abs(r[0, remaining])))]
        jy = [i for i in remaining if i != jx][0]
        r = r[:, [jx, jy, jz]]
        if float(r[2, 2]) < 0:
            r[:, 2] = -r[:, 2]
        if float(r[0, 0]) < 0:
            r[:, 0] = -r[:, 0]
    else:
        for i in range(3):
            axis = r[:, i]
            j = int(np.argmax(np.abs(axis)))
            if axis[j] < 0:
                r[:, i] = -axis

    if float(np.linalg.det(r.astype(np.float64))) < 0:
        r[:, 1] = -r[:, 1]

    return (pts @ r).astype(np.float32, copy=False)


def apply_input_preprocess_np(
    points: np.ndarray,
    *,
    input_normalize: str,
    pca_align: bool,
    pca_align_globalz: bool,
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    in_norm = str(input_normalize or "").strip().lower() or "none"
    do_pca = bool(pca_align)
    if in_norm not in {"none", "off"} or do_pca:
        centroid = pts.mean(axis=0, dtype=np.float64).astype(np.float32)
        pts = (pts - centroid).astype(np.float32, copy=False)
    pts = normalize_points_np(pts, in_norm)
    if do_pca:
        pts = pca_align_np(pts, align_globalz=bool(pca_align_globalz))
    return pts.astype(np.float32, copy=False)

