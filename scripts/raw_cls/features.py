from __future__ import annotations

import numpy as np
import torch

from _lib.point_ops import knn


def parse_point_features(text: str) -> list[str]:
    raw = str(text or "").strip()
    if not raw:
        return ["xyz"]
    s = raw.replace(" ", "")
    if s.lower() in {"x,y,z", "x,y,z,"}:
        return ["xyz"]
    items = [t.strip().lower() for t in raw.split(",") if t.strip()]
    allowed = {"xyz", "normals", "curvature", "radius", "rgb", "cloud_id", "cloud_id_onehot"}
    out: list[str] = []
    for t in items:
        if t not in allowed:
            raise ValueError(f"Unknown point feature: {t} (allowed: {sorted(allowed)})")
        if t not in out:
            out.append(t)
    if "xyz" not in out:
        raise ValueError("point_features must include `xyz` (needed for geometry/augmentations)")
    return out


def point_feature_dim(point_features: list[str]) -> int:
    dim = 0
    for name in point_features:
        if name == "xyz":
            dim += 3
        elif name == "normals":
            dim += 3
        elif name == "curvature":
            dim += 1
        elif name == "radius":
            dim += 1
        elif name == "rgb":
            dim += 3
        elif name == "cloud_id":
            dim += 1
        elif name == "cloud_id_onehot":
            dim += 10
    return int(dim)


@torch.no_grad()
def compute_normals_curv_radius(xyz: torch.Tensor, *, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate per-point normals/curvature/radius from xyz with local PCA."""
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Expected xyz shape (N,3), got {tuple(xyz.shape)}")
    n = int(xyz.shape[0])
    if n <= 1:
        z3 = xyz.new_zeros((n, 3))
        z1 = xyz.new_zeros((n,))
        return z3, z1, z1
    kk = max(1, min(int(k), n - 1))

    x = xyz.t().unsqueeze(0).contiguous()  # (1,3,N)
    idx = knn(x, k=kk)[0]  # (N,k)
    neighbors = xyz[idx.reshape(-1)].reshape(n, kk, 3)  # (N,k,3)

    # Radius: mean neighbor distance to center point.
    radius = torch.linalg.norm(neighbors - xyz[:, None, :], dim=2).mean(dim=1)  # (N,)

    # Local PCA on neighbors.
    mu = neighbors.mean(dim=1, keepdim=True)  # (N,1,3)
    xc = neighbors - mu  # (N,k,3)
    cov = torch.bmm(xc.transpose(1, 2), xc) / float(max(1, kk))  # (N,3,3)
    try:
        evals, evecs = torch.linalg.eigh(cov)  # evals asc
    except Exception:
        evals = xyz.new_zeros((n, 3))
        evecs = xyz.new_zeros((n, 3, 3))

    normals = evecs[:, :, 0]
    normals = normals / (torch.linalg.norm(normals, dim=1, keepdim=True) + 1e-12)
    # Normal sign is ambiguous for PCA; orient consistently so the signal is usable.
    centroid = xyz.mean(dim=0, keepdim=True)
    dot = (normals * (xyz - centroid)).sum(dim=1)
    normals = torch.where(dot[:, None] < 0, -normals, normals)
    curv = evals[:, 0] / (evals.sum(dim=1) + 1e-12)
    return normals, curv, radius


@torch.no_grad()
def build_point_features_from_xyz(
    xyz_np: np.ndarray,
    *,
    point_features: list[str],
    k: int,
    device: torch.device,
    rgb_u8_np: np.ndarray | None = None,
    cloud_id_np: np.ndarray | None = None,
) -> np.ndarray:
    xyz = torch.from_numpy(np.asarray(xyz_np, dtype=np.float32)).to(device)
    rgb: torch.Tensor | None = None
    if "rgb" in point_features:
        if rgb_u8_np is None:
            raise ValueError("point_features includes rgb but rgb_u8_np is missing")
        rgb_arr = np.asarray(rgb_u8_np)
        if rgb_arr.ndim != 2 or rgb_arr.shape[1] != 3 or rgb_arr.shape[0] != xyz.shape[0]:
            raise ValueError(f"Invalid rgb shape {tuple(rgb_arr.shape)} for xyz {tuple(xyz.shape)}")
        if rgb_arr.dtype == np.uint8:
            rgb_f = rgb_arr.astype(np.float32) / 255.0
        else:
            rgb_f = rgb_arr.astype(np.float32, copy=False)
            # Accept either [0,1] or [0,255]-like floats.
            if float(np.max(rgb_f, initial=0.0)) > 1.5:
                rgb_f = rgb_f / 255.0
        rgb = torch.from_numpy(rgb_f).to(device)
    cloud_id: torch.Tensor | None = None
    if "cloud_id" in point_features or "cloud_id_onehot" in point_features:
        if cloud_id_np is None:
            raise ValueError("point_features includes cloud_id/cloud_id_onehot but cloud_id_np is missing")
        cid_arr = np.asarray(cloud_id_np).reshape(-1)
        if cid_arr.shape[0] != xyz.shape[0]:
            raise ValueError(f"Invalid cloud_id shape {tuple(cid_arr.shape)} for xyz {tuple(xyz.shape)}")
        if "cloud_id_onehot" in point_features:
            cid_i = cid_arr.astype(np.int64, copy=False)
            cid_i = np.clip(cid_i, 0, 9)
            onehot = np.zeros((cid_i.shape[0], 10), dtype=np.float32)
            onehot[np.arange(cid_i.shape[0]), cid_i] = 1.0
            cloud_id = torch.from_numpy(onehot).to(device)
        else:
            cid_f = cid_arr.astype(np.float32, copy=False) / 10.0
            cloud_id = torch.from_numpy(cid_f[:, None]).to(device)
    need_local = any(n in {"normals", "curvature", "radius"} for n in point_features)
    normals = curv = radius = None
    if need_local:
        normals, curv, radius = compute_normals_curv_radius(xyz, k=int(k))

    outs: list[torch.Tensor] = []
    for name in point_features:
        if name == "xyz":
            outs.append(xyz)
        elif name == "normals":
            assert normals is not None
            outs.append(normals)
        elif name == "curvature":
            assert curv is not None
            outs.append(curv[:, None])
        elif name == "radius":
            assert radius is not None
            outs.append(radius[:, None])
        elif name == "rgb":
            assert rgb is not None
            outs.append(rgb)
        elif name == "cloud_id":
            assert cloud_id is not None
            outs.append(cloud_id)
        elif name == "cloud_id_onehot":
            assert cloud_id is not None
            outs.append(cloud_id)
        else:
            raise ValueError(f"Unknown point feature: {name}")
    feat = torch.cat(outs, dim=1)
    return feat.detach().to("cpu").numpy().astype(np.float32, copy=False)

