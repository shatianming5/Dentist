from __future__ import annotations

from pathlib import Path

import numpy as np


def write_ply_xyz(path: Path, points: np.ndarray) -> None:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {int(pts.shape[0])}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]
    for x, y, z in pts:
        lines.append(f"{float(x):.6f} {float(y):.6f} {float(z):.6f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

