from __future__ import annotations

from pathlib import Path

import numpy as np


def parse_obj_vertices(obj_path: Path, *, stop_at_faces: bool = True) -> np.ndarray:
    """Parse only vertex XYZ from a Teeth3DS-style OBJ.

    Notes:
    - We intentionally ignore faces/uv/normals, and stop early when faces start
      (to avoid scanning huge files unnecessarily).
    - OBJ vertex lines may contain extra fields (e.g., per-vertex color); we
      only read the first 3 floats.
    """

    verts: list[np.ndarray] = []
    seen_v = False
    with obj_path.open("rb") as f:
        for line in f:
            if line.startswith(b"v "):
                seen_v = True
                xyz = np.fromstring(line[2:], sep=" ", count=3, dtype=np.float32)
                if xyz.shape[0] == 3 and np.isfinite(xyz).all():
                    verts.append(xyz)
                continue
            if stop_at_faces and seen_v and line.startswith(b"f "):
                break
    if not verts:
        raise ValueError(f"no vertices parsed from: {obj_path}")
    return np.vstack(verts).astype(np.float32, copy=False)

