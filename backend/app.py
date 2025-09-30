from __future__ import annotations

import os
from typing import List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware


DATA_DIR = os.path.join(os.getcwd(), "data")


def _ensure_data_dir() -> None:
    if not os.path.isdir(DATA_DIR):
        raise RuntimeError(f"Data folder not found: {DATA_DIR}")


def _list_files_with_ext(folder: str, ext: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    out = []
    for name in os.listdir(folder):
        if name.lower().endswith(ext.lower()):
            out.append(name)
    return sorted(out)


def _load_points_bin(path: str, peek: int) -> Tuple[List[Tuple[float, float, float]], Optional[List[Tuple[int, int, int]]]]:
    # Import here to avoid hard dependency if script is not needed
    import sys

    scripts_dir = os.path.join(os.getcwd(), "scripts")
    if scripts_dir not in sys.path:
        sys.path.append(scripts_dir)
    from ccbin_v2_reader import load_points  # type: ignore

    return load_points(path, peek=peek)


app = FastAPI(title="Dentist 3D Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/files/bin")
def list_bin_files():
    _ensure_data_dir()
    return {"files": _list_files_with_ext(DATA_DIR, ".bin")}


@app.get("/points/bin/{filename}")
def get_points_from_bin(
    filename: str,
    peek: int = Query(100, ge=1, le=5000, description="Number of points to preview"),
):
    _ensure_data_dir()
    path = os.path.join(DATA_DIR, filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        pts, cols = _load_points_bin(path, peek=peek)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load points: {e}")

    # shape JSON
    items = []
    for i, (x, y, z) in enumerate(pts):
        item = {"x": float(x), "y": float(y), "z": float(z)}
        if cols and i < len(cols):
            r, g, b = cols[i]
            item["rgb"] = {"r": int(r), "g": int(g), "b": int(b)}
        items.append(item)

    return {"file": filename, "count": len(pts), "points": items}


# Optional: list PLY outputs if user converted
@app.get("/files/ply")
def list_ply_files():
    folder = os.path.join(os.getcwd(), "data_ply")
    return {"files": _list_files_with_ext(folder, ".ply")}

