from __future__ import annotations

import os
from typing import List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import subprocess
import sys
import uuid
from threading import Lock



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


# ---------------------------- Pipeline jobs ---------------------------- #

class ExportNPZReq(BaseModel):
    src: str = "data"
    dst: str = "data_npz"
    points: int = 2048
    limit: int = 0


class PretrainSimCLRReq(BaseModel):
    root: str = "data_npz"
    train_list: str = "splits/train.txt"
    val_list: str = "splits/val.txt"
    points: int = 2048
    epochs: int = 20
    batch_size: int = 16
    lr: float = 1e-3
    out: str = "outputs/simclr_pointnet"


class LinearProbeReq(BaseModel):
    root: str = "data_npz"
    train_list: str = "splits/train.txt"
    val_list: str = "splits/val.txt"
    points: int = 2048
    epochs: int = 20
    batch_size: int = 16
    ckpt: str = "outputs/simclr_pointnet/ckpt_best.pth"
    out: str = "outputs/linear_probe"


_JOBS: dict[str, subprocess.Popen] = {}
_JLOCK = Lock()


def _spawn(cmd: list[str], log_path: str | None = None) -> str:
    kwargs = {}
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        f = open(log_path, "w", buffering=1)
        kwargs.update(dict(stdout=f, stderr=subprocess.STDOUT))
    proc = subprocess.Popen(cmd, **kwargs)
    jid = str(uuid.uuid4())
    with _JLOCK:
        _JOBS[jid] = proc
    return jid


@app.post("/pipeline/export_npz")
def pipeline_export_npz(req: ExportNPZReq):
    _ensure_data_dir()
    cmd = [sys.executable, "scripts/export_points_npz.py", "--src", req.src, "--dst", req.dst, "--points", str(req.points)]
    if req.limit:
        cmd += ["--limit", str(req.limit)]
    jid = _spawn(cmd, log_path=os.path.join("outputs", "logs", "export_npz.log"))
    return {"job_id": jid, "cmd": cmd}


@app.post("/pipeline/pretrain_simclr")
def pipeline_pretrain_simclr(req: PretrainSimCLRReq):
    cmd = [
        sys.executable,
        "scripts/train_simclr_pointnet.py",
        "--root", req.root,
        "--train_list", req.train_list,
        "--val_list", req.val_list,
        "--points", str(req.points),
        "--epochs", str(req.epochs),
        "--batch_size", str(req.batch_size),
        "--lr", str(req.lr),
        "--out", req.out,
    ]
    jid = _spawn(cmd, log_path=os.path.join("outputs", "logs", "pretrain_simclr.log"))
    return {"job_id": jid, "cmd": cmd}


@app.post("/pipeline/linear_probe")
def pipeline_linear_probe(req: LinearProbeReq):
    cmd = [
        sys.executable,
        "scripts/linear_probe_pointnet.py",
        "--root", req.root,
        "--train_list", req.train_list,
        "--val_list", req.val_list,
        "--points", str(req.points),
        "--epochs", str(req.epochs),
        "--batch_size", str(req.batch_size),
        "--ckpt", req.ckpt,
        "--out", req.out,
    ]
    jid = _spawn(cmd, log_path=os.path.join("outputs", "logs", "linear_probe.log"))
    return {"job_id": jid, "cmd": cmd}


@app.get("/pipeline/status/{job_id}")
def pipeline_status(job_id: str):
    with _JLOCK:
        proc = _JOBS.get(job_id)
    if not proc:
        raise HTTPException(status_code=404, detail="Job not found")
    code = proc.poll()
    return {"running": code is None, "returncode": code}
