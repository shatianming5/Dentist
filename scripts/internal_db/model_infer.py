from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .mesh_io import OptionalDependencyError


def _require(name: str) -> Any:
    try:
        return __import__(name)
    except Exception as e:  # noqa: BLE001
        raise OptionalDependencyError(
            f"Missing optional dependency '{name}'. Install via: pip install -r configs/env/requirements_vis.txt"
        ) from e


def _read_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    return obj if isinstance(obj, dict) else {}


def _softmax_np(logits: np.ndarray, axis: int) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float32)
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return (ex / np.sum(ex, axis=axis, keepdims=True)).astype(np.float32, copy=False)


@dataclass(frozen=True)
class RawSegRun:
    run_dir: Path
    model_name: str
    num_classes: int
    n_points: int
    dropout: float
    dgcnn_k: int
    dgcnn_emb_dims: int
    pt_dim: int
    pt_depth: int
    pt_k: int
    pt_ffn_mult: float


def load_raw_seg_run(run_dir: Path) -> RawSegRun:
    cfg_path = (run_dir / "train_config.json").resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing train_config.json: {cfg_path}")
    cfg = _read_json(cfg_path)
    return RawSegRun(
        run_dir=run_dir,
        model_name=str(cfg.get("model") or ""),
        num_classes=int(cfg.get("num_classes") or 0),
        n_points=int(cfg.get("n_points") or 0),
        dropout=float(cfg.get("dropout") or 0.0),
        dgcnn_k=int(cfg.get("model_dgcnn_k") or 20),
        dgcnn_emb_dims=int(cfg.get("model_dgcnn_emb_dims") or 512),
        pt_dim=int(cfg.get("model_pt_dim") or 96),
        pt_depth=int(cfg.get("model_pt_depth") or 4),
        pt_k=int(cfg.get("model_pt_k") or 16),
        pt_ffn_mult=float(cfg.get("model_pt_ffn_mult") or 2.0),
    )


def load_raw_seg_model(run_dir: Path, *, device: str = "cpu") -> tuple[Any, RawSegRun]:
    torch = _require("torch")

    run_dir = run_dir.expanduser().resolve()
    spec = load_raw_seg_run(run_dir)
    if spec.num_classes <= 1:
        raise ValueError(f"Invalid num_classes={spec.num_classes} in {run_dir}")

    try:
        import phase3_train_raw_seg as train_mod
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Failed to import phase3_train_raw_seg (run from repo root or scripts/)") from e

    name = str(spec.model_name).strip().lower()
    if name == "pointnet_seg":
        model = train_mod.PointNetSeg(num_classes=int(spec.num_classes), dropout=float(spec.dropout))
    elif name in {"dgcnn", "dgcnn_v2"}:
        model = train_mod.DGCNNv2Seg(
            num_classes=int(spec.num_classes),
            dropout=float(spec.dropout),
            k=int(spec.dgcnn_k),
            emb_dims=int(spec.dgcnn_emb_dims),
        )
    elif name in {"point_transformer", "pointtransformer"}:
        model = train_mod.PointTransformerSeg(
            num_classes=int(spec.num_classes),
            dropout=float(spec.dropout),
            dim=int(spec.pt_dim),
            depth=int(spec.pt_depth),
            k=int(spec.pt_k),
            ffn_mult=float(spec.pt_ffn_mult),
        )
    else:
        raise ValueError(f"Unsupported raw_seg model={spec.model_name!r} in {run_dir}")

    ckpt_path = (run_dir / "ckpt_best.pt").resolve()
    if not ckpt_path.is_file():
        ckpt_path = (run_dir / "model_best.pt").resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {run_dir}/ckpt_best.pt or model_best.pt")

    ckpt = torch.load(str(ckpt_path), map_location=str(device))
    state = ckpt.get("model") if isinstance(ckpt, dict) else None
    if state is None:
        state = ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    model.to(torch.device(str(device)))
    return model, spec


def infer_raw_seg_probs(model: Any, points: np.ndarray, *, device: str = "cpu") -> np.ndarray:
    torch = _require("torch")

    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be (N,3), got {tuple(pts.shape)}")

    x = torch.from_numpy(pts[None, :, :]).to(device=torch.device(str(device)), dtype=torch.float32)
    with torch.no_grad():
        logits = model(x)  # (B,C,N)
        probs = torch.softmax(logits, dim=1)
    out = probs[0].detach().cpu().numpy().astype(np.float32, copy=False)  # (C,N)
    return out


def heat_from_probs(
    probs: np.ndarray,
    *,
    mode: str,
    class_id: int | None = None,
) -> np.ndarray:
    p = np.asarray(probs, dtype=np.float32)
    if p.ndim != 2:
        raise ValueError("probs must be (C,N)")
    m = str(mode).strip().lower()
    if m == "maxprob":
        return np.max(p, axis=0).astype(np.float32, copy=False)
    if m in {"entropy", "ent"}:
        eps = 1e-8
        pp = np.clip(p, eps, 1.0)
        ent = -np.sum(pp * np.log(pp), axis=0)
        ent = ent / float(math.log(p.shape[0]))  # normalize to [0,1]
        return ent.astype(np.float32, copy=False)
    if m in {"class_prob", "prob"}:
        if class_id is None:
            raise ValueError("class_id is required for mode=class_prob")
        c = int(class_id)
        if c < 0 or c >= p.shape[0]:
            raise ValueError(f"class_id out of range: {class_id} (C={p.shape[0]})")
        return p[c].astype(np.float32, copy=False)
    raise ValueError("Unknown heat mode (supported: maxprob, entropy, class_prob)")


@dataclass(frozen=True)
class RawClsRun:
    run_dir: Path
    model_name: str
    num_classes: int
    n_points: int
    dropout: float
    extra_dim: int
    dgcnn_k: int


def load_raw_cls_run(run_dir: Path) -> RawClsRun:
    cfg_path = (run_dir / "config.json").resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing config.json: {cfg_path}")
    cfg = _read_json(cfg_path)
    extra_features = list(cfg.get("extra_features") or [])
    return RawClsRun(
        run_dir=run_dir,
        model_name=str(cfg.get("model") or ""),
        num_classes=int(cfg.get("num_classes") or 0),
        n_points=int(cfg.get("n_points") or 0),
        dropout=float(cfg.get("dropout") or 0.0),
        extra_dim=int(len(extra_features)),
        dgcnn_k=int(cfg.get("dgcnn_k") or 20),
    )


def load_raw_cls_model(run_dir: Path, *, device: str = "cpu") -> tuple[Any, RawClsRun]:
    torch = _require("torch")

    run_dir = run_dir.expanduser().resolve()
    spec = load_raw_cls_run(run_dir)
    if spec.num_classes <= 1:
        raise ValueError(f"Invalid num_classes={spec.num_classes} in {run_dir}")

    try:
        import phase3_train_raw_cls_baseline as train_mod
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Failed to import phase3_train_raw_cls_baseline (run from repo root or scripts/)") from e

    name = str(spec.model_name).strip().lower()
    if name == "pointnet":
        model = train_mod.PointNetClassifier(num_classes=int(spec.num_classes), dropout=float(spec.dropout), extra_dim=int(spec.extra_dim))
    elif name == "dgcnn":
        model = train_mod.DGCNNClassifier(
            num_classes=int(spec.num_classes),
            dropout=float(spec.dropout),
            k=int(spec.dgcnn_k),
            extra_dim=int(spec.extra_dim),
        )
    else:
        raise ValueError(f"Unsupported raw_cls model={spec.model_name!r} in {run_dir}")

    ckpt_path = (run_dir / "model_best.pt").resolve()
    if not ckpt_path.is_file():
        ckpt_path = (run_dir / "ckpt_best.pt").resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {run_dir}/model_best.pt")

    ckpt = torch.load(str(ckpt_path), map_location=str(device))
    state = None
    if isinstance(ckpt, dict):
        # raw_cls checkpoints use 'model_state' (ordered dict) in this repo.
        state = ckpt.get("model_state") or ckpt.get("model") or ckpt.get("state_dict")
    if state is None:
        state = ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    model.to(torch.device(str(device)))
    return model, spec


def raw_cls_point_saliency(
    model: Any,
    points: np.ndarray,
    *,
    device: str = "cpu",
    target_class: int | str = "pred",
    extra_dim: int = 0,
) -> tuple[np.ndarray, int]:
    """Gradient saliency per point using d(logit[target])/d(xyz)."""
    torch = _require("torch")

    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be (N,3), got {tuple(pts.shape)}")

    dev = torch.device(str(device))
    x = torch.from_numpy(pts[None, :, :]).to(device=dev, dtype=torch.float32)
    x.requires_grad_(True)

    extra: torch.Tensor | None = None
    if int(extra_dim) > 0:
        extra = torch.zeros((1, int(extra_dim)), device=dev, dtype=torch.float32)

    logits = model(x, extra, domains=None)
    if isinstance(target_class, str) and target_class == "pred":
        t = int(torch.argmax(logits, dim=1).detach().cpu().item())
    else:
        t = int(target_class)
    score = logits[0, t]
    model.zero_grad(set_to_none=True)
    if x.grad is not None:
        x.grad.zero_()
    score.backward()
    g = x.grad.detach().cpu().numpy()[0]  # (N,3)
    heat = np.linalg.norm(g.astype(np.float64), axis=1).astype(np.float32)
    return heat, t
