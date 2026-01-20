from __future__ import annotations

import math

import torch


def augment_points_torch(
    points: torch.Tensor,
    *,
    rotate_z: bool,
    scale: float,
    jitter_sigma: float,
    jitter_clip: float,
    dropout_ratio: float,
    gen: torch.Generator | None = None,
    slice_xyz: slice | None = None,
    slice_normals: slice | None = None,
    idx_curv: int | None = None,
    idx_radius: int | None = None,
) -> torch.Tensor:
    if points.ndim != 3 or points.shape[-1] < 3:
        raise ValueError(f"Expected points (B,N,C>=3), got {tuple(points.shape)}")
    x = points.clone()
    b = int(x.shape[0])
    slice_xyz0 = slice_xyz if isinstance(slice_xyz, slice) else slice(0, 3)
    slice_normals0 = slice_normals if isinstance(slice_normals, slice) else None
    idx_curv0 = int(idx_curv) if isinstance(idx_curv, int) else None
    idx_radius0 = int(idx_radius) if isinstance(idx_radius, int) else None
    if rotate_z:
        two_pi = float(2.0 * math.pi)
        angles = torch.rand((b,), generator=gen, device=x.device, dtype=x.dtype) * two_pi
        c = torch.cos(angles)
        s = torch.sin(angles)
        rot = torch.zeros((b, 3, 3), device=x.device, dtype=x.dtype)
        rot[:, 0, 0] = c
        rot[:, 0, 1] = -s
        rot[:, 1, 0] = s
        rot[:, 1, 1] = c
        rot[:, 2, 2] = 1.0
        xyz = x[:, :, slice_xyz0]
        if xyz.shape[-1] != 3:
            raise ValueError(f"Expected xyz slice (B,N,3), got {tuple(xyz.shape)}")
        x[:, :, slice_xyz0] = torch.matmul(xyz, rot.transpose(1, 2))
        if slice_normals0 is not None:
            nrm = x[:, :, slice_normals0]
            if nrm.shape[-1] != 3:
                raise ValueError(f"Expected normals slice (B,N,3), got {tuple(nrm.shape)}")
            x[:, :, slice_normals0] = torch.matmul(nrm, rot.transpose(1, 2))

    s0 = float(scale)
    if s0 and s0 > 0:
        lo = 1.0 - s0
        hi = 1.0 + s0
        sc = torch.empty((b, 1, 1), device=x.device, dtype=x.dtype).uniform_(lo, hi, generator=gen)
        x[:, :, slice_xyz0] = x[:, :, slice_xyz0] * sc
        if idx_radius0 is not None and idx_radius0 < int(x.shape[2]):
            scale_vec = sc[:, 0, 0]
            x[:, :, idx_radius0] = x[:, :, idx_radius0] * scale_vec[:, None]

    sig = float(jitter_sigma)
    if sig and sig > 0:
        noise = torch.randn(x[:, :, slice_xyz0].shape, generator=gen, device=x.device, dtype=x.dtype) * sig
        clip = float(jitter_clip)
        if clip and clip > 0:
            noise = noise.clamp(-clip, clip)
        x[:, :, slice_xyz0] = x[:, :, slice_xyz0] + noise

    dr = float(dropout_ratio)
    if dr and dr > 0:
        # Per-sample random dropout in [0, dr].
        probs = torch.rand((b,), generator=gen, device=x.device, dtype=x.dtype) * dr
        keep = torch.rand((b, x.shape[1]), generator=gen, device=x.device, dtype=x.dtype) >= probs[:, None]
        keep = keep.to(dtype=torch.bool)
        x2 = x.clone()
        for i in range(b):
            if not torch.any(keep[i]):
                continue
            # Preserve non-geometry channels (e.g. cloud_id) so dropout does not change cloud assignment.
            ref_xyz = x2[i, 0, slice_xyz0].clone()
            x2[i, ~keep[i], slice_xyz0] = ref_xyz
            if slice_normals0 is not None:
                ref_nrm = x2[i, 0, slice_normals0].clone()
                x2[i, ~keep[i], slice_normals0] = ref_nrm
            if idx_curv0 is not None and idx_curv0 < int(x2.shape[2]):
                ref_curv = x2[i, 0, idx_curv0].clone()
                x2[i, ~keep[i], idx_curv0] = ref_curv
            if idx_radius0 is not None and idx_radius0 < int(x2.shape[2]):
                ref_rad = x2[i, 0, idx_radius0].clone()
                x2[i, ~keep[i], idx_radius0] = ref_rad
        x = x2

    return x

