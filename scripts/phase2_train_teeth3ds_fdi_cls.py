#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from dgcnn_v2 import DGCNNv2Classifier, DGCNNv2Params
from pointnet2 import PointNet2Classifier
from point_transformer import PointTransformerClassifier, PointTransformerParams
from pointmlp import PointMLPClassifier, PointMLPParams


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _run_git(root: Path, args: list[str]) -> str | None:
    try:
        out = subprocess.check_output(["git", *args], cwd=str(root), stderr=subprocess.DEVNULL)
        s = out.decode("utf-8", errors="replace").strip()
        return s or None
    except Exception:
        return None


def get_git_info(root: Path) -> dict[str, Any]:
    return {
        "commit": _run_git(root, ["rev-parse", "HEAD"]),
        "dirty": _run_git(root, ["status", "--porcelain"]) not in {None, ""},
    }


def get_env_info() -> dict[str, Any]:
    return {
        "generated_at": utc_now_iso(),
        "python": sys.version.replace("\n", " "),
        "numpy": getattr(np, "__version__", "unknown"),
        "torch": getattr(torch, "__version__", "unknown"),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda": getattr(torch.version, "cuda", None),
        "cudnn": torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None,
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        rows.append(json.loads(s))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_device(device: str) -> str:
    d = device.strip().lower()
    if d in {"auto", ""}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if d in {"cuda", "cpu"}:
        if d == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return d
    raise ValueError(f"Unsupported device: {device}")


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


class Teeth3DSToothFDIDataset(Dataset[tuple[torch.Tensor, int, dict[str, Any]]]):
    def __init__(
        self,
        *,
        rows: list[dict[str, Any]],
        data_root: Path,
        n_points: int,
        seed: int,
        train: bool,
        fdi_to_id: dict[str, int],
        aug_rotate_z: bool,
        aug_scale: float,
        aug_jitter_sigma: float,
        aug_jitter_clip: float,
        aug_dropout_ratio: float,
    ) -> None:
        self.rows = rows
        self.data_root = data_root
        self.n_points = int(n_points)
        self.train = bool(train)
        self.fdi_to_id = dict(fdi_to_id)
        self.aug_rotate_z = bool(aug_rotate_z)
        self.aug_scale = float(aug_scale)
        self.aug_jitter_sigma = float(aug_jitter_sigma)
        self.aug_jitter_clip = float(aug_jitter_clip)
        self.aug_dropout_ratio = float(aug_dropout_ratio)
        self.rng = np.random.default_rng(int(seed) + (0 if train else 10_000))

    def __len__(self) -> int:  # noqa: D401
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, dict[str, Any]]:
        r = self.rows[int(idx)]
        npz_path = self.data_root / str(r["sample_npz"])
        with np.load(npz_path) as data:
            pts = np.asarray(data["points"], dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"Invalid points shape {pts.shape} in {npz_path}")

        if self.n_points > 0 and pts.shape[0] != self.n_points:
            if pts.shape[0] > self.n_points:
                sel = self.rng.choice(pts.shape[0], size=self.n_points, replace=False)
            else:
                sel = self.rng.choice(pts.shape[0], size=self.n_points, replace=True)
            pts = pts[sel]

        if self.train:
            if self.aug_rotate_z:
                angle = float(self.rng.uniform(0.0, 2.0 * math.pi))
                pts = rotate_z(pts, angle)

            if self.aug_scale > 0:
                lo = 1.0 - self.aug_scale
                hi = 1.0 + self.aug_scale
                scale = float(self.rng.uniform(lo, hi))
                pts = (pts * scale).astype(np.float32, copy=False)

            pts = jitter(pts, self.rng, sigma=self.aug_jitter_sigma, clip=self.aug_jitter_clip)
            pts = random_point_dropout(pts, self.rng, max_dropout_ratio=self.aug_dropout_ratio)

        fdi = r.get("fdi")
        y = self.fdi_to_id.get(str(fdi))
        if y is None:
            raise ValueError(f"Unknown fdi={fdi!r} (npz={npz_path})")

        meta = {
            "case_key": r.get("case_key"),
            "id_patient": r.get("id_patient"),
            "jaw": r.get("jaw"),
            "instance_id": r.get("instance_id"),
            "split": r.get("split"),
            "fdi": int(fdi) if fdi is not None else None,
            "sample_npz": str(r.get("sample_npz")),
        }
        return torch.from_numpy(pts), int(y), meta


class PointNetFDIClassifier(nn.Module):
    def __init__(self, num_classes: int, dropout: float, in_channels: int = 3) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.feat = nn.Sequential(
            nn.Conv1d(int(in_channels), 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(128, int(num_classes)),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3 or points.shape[-1] != int(self.in_channels):
            raise ValueError(f"Expected points (B,N,{int(self.in_channels)}), got {tuple(points.shape)}")
        x = points.transpose(1, 2).contiguous()
        x = self.feat(x)
        x = torch.max(x, dim=2).values
        return self.head(x)


def _limit(rows: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    if k and k > 0:
        return rows[: int(k)]
    return rows


def macro_f1(y_true: list[int], y_pred: list[int], *, num_classes: int) -> float:
    if not y_true:
        return 0.0
    tp = [0] * int(num_classes)
    fp = [0] * int(num_classes)
    fn = [0] * int(num_classes)
    for yt, yp in zip(y_true, y_pred, strict=True):
        if int(yt) == int(yp):
            tp[int(yt)] += 1
        else:
            fp[int(yp)] += 1
            fn[int(yt)] += 1
    f1s: list[float] = []
    for c in range(int(num_classes)):
        prec = tp[c] / max(1, tp[c] + fp[c])
        rec = tp[c] / max(1, tp[c] + fn[c])
        if prec + rec <= 0:
            f1s.append(0.0)
        else:
            f1s.append(2.0 * prec * rec / (prec + rec))
    return float(sum(f1s) / max(1, len(f1s)))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, *, num_classes: int) -> dict[str, float]:
    model.eval()
    sum_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []
    n = 0
    for pts, y, _meta in loader:
        pts = pts.to(device)
        y = y.to(device)
        logits = model(pts)
        loss = F.cross_entropy(logits, y).item()
        sum_loss += float(loss) * int(pts.shape[0])
        n += int(pts.shape[0])
        pred = logits.argmax(dim=1).detach().cpu().numpy().astype(np.int64).tolist()
        y_pred.extend(pred)
        y_true.extend(y.detach().cpu().numpy().astype(np.int64).tolist())
    acc = float(sum(int(a == b) for a, b in zip(y_true, y_pred, strict=True)) / max(1, len(y_true)))
    return {"loss": float(sum_loss / max(1, n)), "acc": acc, "macro_f1": macro_f1(y_true, y_pred, num_classes=num_classes)}


@dataclass(frozen=True)
class TrainConfig:
    generated_at: str
    seed: int
    device: str
    data_root: str
    out_dir: str
    exp_name: str
    n_points: int
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    dropout: float
    aug_rotate_z: bool
    aug_scale: float
    aug_jitter_sigma: float
    aug_jitter_clip: float
    aug_dropout_ratio: float
    num_workers: int
    patience: int
    balanced_sampler: bool
    include_unknown_split: bool
    limit_train: int
    limit_val: int
    limit_test: int


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Phase 2: supervised pretrain (PointNet/PointNet2/DGCNNv2/PointTransformer/PointMLP) on Teeth3DS single-tooth FDI classification."
    )
    ap.add_argument("--data-root", type=Path, default=Path("processed/teeth3ds_teeth/v1"))
    ap.add_argument("--runs-dir", type=Path, default=Path("runs/pretrain"))
    ap.add_argument("--model", choices=["pointnet", "pointnet2", "point_transformer", "pointmlp", "dgcnn_v2"], default="pointnet")
    ap.add_argument("--exp-name", type=str, default="")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--n-points", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--balanced-sampler", action="store_true")
    ap.add_argument("--aug-rotate-z", action="store_true", help="Rotate around Z (off by default; FDI is orientation-sensitive).")
    ap.add_argument("--aug-scale", type=float, default=0.02)
    ap.add_argument("--aug-jitter-sigma", type=float, default=0.005)
    ap.add_argument("--aug-jitter-clip", type=float, default=0.02)
    ap.add_argument("--aug-dropout-ratio", type=float, default=0.1)
    ap.add_argument("--pointnet2-sa1-npoint", type=int, default=512, help="PointNet2 SA1 npoint (only when --model=pointnet2).")
    ap.add_argument("--pointnet2-sa1-nsample", type=int, default=32, help="PointNet2 SA1 nsample (only when --model=pointnet2).")
    ap.add_argument("--pointnet2-sa2-npoint", type=int, default=128, help="PointNet2 SA2 npoint (only when --model=pointnet2).")
    ap.add_argument("--pointnet2-sa2-nsample", type=int, default=64, help="PointNet2 SA2 nsample (only when --model=pointnet2).")
    ap.add_argument("--pt-dim", type=int, default=96, help="PointTransformer embedding dim (only when --model=point_transformer).")
    ap.add_argument("--pt-depth", type=int, default=4, help="PointTransformer depth (only when --model=point_transformer).")
    ap.add_argument("--pt-k", type=int, default=16, help="PointTransformer kNN size (only when --model=point_transformer).")
    ap.add_argument("--pt-ffn-mult", type=float, default=2.0, help="PointTransformer FFN hidden multiplier (only when --model=point_transformer).")
    ap.add_argument("--pmlp-dim", type=int, default=128, help="PointMLP embedding dim (only when --model=pointmlp).")
    ap.add_argument("--pmlp-depth", type=int, default=6, help="PointMLP depth (only when --model=pointmlp).")
    ap.add_argument("--pmlp-k", type=int, default=16, help="PointMLP kNN size (only when --model=pointmlp).")
    ap.add_argument("--pmlp-ffn-mult", type=float, default=2.0, help="PointMLP FFN hidden multiplier (only when --model=pointmlp).")
    ap.add_argument("--dgcnn-k", type=int, default=20, help="kNN size for DGCNNv2 (only when --model=dgcnn_v2).")
    ap.add_argument("--include-unknown-split", action="store_true")
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-val", type=int, default=0)
    ap.add_argument("--limit-test", type=int, default=0)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_root = args.data_root.expanduser().resolve()
    if not data_root.exists():
        raise SystemExit(f"Missing data_root: {data_root}")

    index_path = data_root / "index.jsonl"
    fdi_values_path = data_root / "fdi_values.json"
    if not index_path.is_file():
        raise SystemExit(f"Missing index: {index_path} (run scripts/phase2_build_teeth3ds_teeth.py first)")
    if not fdi_values_path.is_file():
        raise SystemExit(f"Missing fdi_values: {fdi_values_path} (run scripts/phase2_build_teeth3ds_teeth.py first)")

    rows = read_jsonl(index_path)
    fdi_values = read_json(fdi_values_path)
    if not isinstance(fdi_values, list) or not fdi_values:
        raise SystemExit(f"Invalid fdi_values: {fdi_values_path}")
    fdi_to_id = {str(int(v)): int(i) for i, v in enumerate(fdi_values)}
    num_classes = int(len(fdi_to_id))

    train_rows = [r for r in rows if r.get("split") == "train"]
    val_rows = [r for r in rows if r.get("split") == "val"]
    test_rows = [r for r in rows if r.get("split") == "test"]
    unknown_rows = [r for r in rows if str(r.get("split") or "").strip() not in {"train", "val", "test"}]
    if args.include_unknown_split:
        train_rows = train_rows + unknown_rows

    train_rows = _limit(train_rows, int(args.limit_train))
    val_rows = _limit(val_rows, int(args.limit_val))
    test_rows = _limit(test_rows, int(args.limit_test))

    if not train_rows or not val_rows:
        raise SystemExit(f"Empty train/val splits (train={len(train_rows)}, val={len(val_rows)}) in {index_path}")

    model_name = str(args.model or "pointnet").strip() or "pointnet"
    exp_name = str(args.exp_name or "").strip() or f"teeth3ds_fdi_{model_name}"
    out_dir = (args.runs_dir.expanduser().resolve() / f"{exp_name}_seed{int(args.seed)}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(args.seed))
    device = torch.device(normalize_device(str(args.device)))
    cli = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    write_json(
        out_dir / "config.json",
        {
            "cli": cli,
            "train": asdict(
                TrainConfig(
                    generated_at=utc_now_iso(),
                    seed=int(args.seed),
                    device=str(device),
                    data_root=str(data_root),
                    out_dir=str(out_dir),
                    exp_name=str(exp_name),
                    n_points=int(args.n_points),
                    batch_size=int(args.batch_size),
                    epochs=int(args.epochs),
                    lr=float(args.lr),
                    weight_decay=float(args.weight_decay),
                    dropout=float(args.dropout),
                    aug_rotate_z=bool(args.aug_rotate_z),
                    aug_scale=float(args.aug_scale),
                    aug_jitter_sigma=float(args.aug_jitter_sigma),
                    aug_jitter_clip=float(args.aug_jitter_clip),
                    aug_dropout_ratio=float(args.aug_dropout_ratio),
                    num_workers=int(0 if str(device) == "cpu" else 4),
                    patience=int(args.patience),
                    balanced_sampler=bool(args.balanced_sampler),
                    include_unknown_split=bool(args.include_unknown_split),
                    limit_train=int(args.limit_train),
                    limit_val=int(args.limit_val),
                    limit_test=int(args.limit_test),
                )
            ),
            "git": get_git_info(root),
            "env": get_env_info(),
            "data": {
                "index": str(index_path),
                "fdi_values": str(fdi_values_path),
                "counts": {
                    "train": len(train_rows),
                    "val": len(val_rows),
                    "test": len(test_rows),
                    "unknown": len(unknown_rows),
                },
            },
            "labels": {"num_classes": num_classes, "fdi_to_id": fdi_to_id},
            "model": {"name": model_name},
        },
    )

    ds_train = Teeth3DSToothFDIDataset(
        rows=train_rows,
        data_root=data_root,
        n_points=int(args.n_points),
        seed=int(args.seed),
        train=True,
        fdi_to_id=fdi_to_id,
        aug_rotate_z=bool(args.aug_rotate_z),
        aug_scale=float(args.aug_scale),
        aug_jitter_sigma=float(args.aug_jitter_sigma),
        aug_jitter_clip=float(args.aug_jitter_clip),
        aug_dropout_ratio=float(args.aug_dropout_ratio),
    )
    ds_val = Teeth3DSToothFDIDataset(
        rows=val_rows,
        data_root=data_root,
        n_points=int(args.n_points),
        seed=int(args.seed),
        train=False,
        fdi_to_id=fdi_to_id,
        aug_rotate_z=False,
        aug_scale=0.0,
        aug_jitter_sigma=0.0,
        aug_jitter_clip=0.0,
        aug_dropout_ratio=0.0,
    )
    ds_test = Teeth3DSToothFDIDataset(
        rows=test_rows,
        data_root=data_root,
        n_points=int(args.n_points),
        seed=int(args.seed),
        train=False,
        fdi_to_id=fdi_to_id,
        aug_rotate_z=False,
        aug_scale=0.0,
        aug_jitter_sigma=0.0,
        aug_jitter_clip=0.0,
        aug_dropout_ratio=0.0,
    )

    def collate(batch: list[tuple[torch.Tensor, int, dict[str, Any]]]) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
        pts, ys, metas = zip(*batch, strict=True)
        return torch.stack(list(pts), dim=0), torch.tensor(list(ys), dtype=torch.long), list(metas)

    sampler = None
    shuffle = True
    if bool(args.balanced_sampler):
        counts = {}
        for r in train_rows:
            k = str(r.get("fdi"))
            counts[k] = int(counts.get(k, 0)) + 1
        weights = [1.0 / max(1, int(counts.get(str(r.get("fdi")), 1))) for r in train_rows]
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        shuffle = False

    dl_train = DataLoader(
        ds_train,
        batch_size=int(args.batch_size),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(0 if str(device) == "cpu" else 4),
        drop_last=True,
        collate_fn=collate,
    )
    dl_val = DataLoader(ds_val, batch_size=int(args.batch_size), shuffle=False, num_workers=int(0 if str(device) == "cpu" else 2), collate_fn=collate)
    dl_test = DataLoader(ds_test, batch_size=int(args.batch_size), shuffle=False, num_workers=int(0 if str(device) == "cpu" else 2), collate_fn=collate)

    if model_name == "pointnet":
        model = PointNetFDIClassifier(num_classes=num_classes, dropout=float(args.dropout), in_channels=3).to(device)
    elif model_name == "pointnet2":
        model = PointNet2Classifier(
            num_classes=int(num_classes),
            dropout=float(args.dropout),
            extra_dim=0,
            in_channels=3,
            sa1_npoint=int(args.pointnet2_sa1_npoint),
            sa1_nsample=int(args.pointnet2_sa1_nsample),
            sa2_npoint=int(args.pointnet2_sa2_npoint),
            sa2_nsample=int(args.pointnet2_sa2_nsample),
        ).to(device)
    elif model_name == "point_transformer":
        params = PointTransformerParams(
            dim=int(args.pt_dim),
            depth=int(args.pt_depth),
            k=int(args.pt_k),
            ffn_mult=float(args.pt_ffn_mult),
        )
        model = PointTransformerClassifier(
            num_classes=int(num_classes),
            dropout=float(args.dropout),
            extra_dim=0,
            in_channels=3,
            params=params,
        ).to(device)
    elif model_name == "pointmlp":
        params = PointMLPParams(
            dim=int(args.pmlp_dim),
            depth=int(args.pmlp_depth),
            k=int(args.pmlp_k),
            ffn_mult=float(args.pmlp_ffn_mult),
        )
        model = PointMLPClassifier(
            num_classes=int(num_classes),
            dropout=float(args.dropout),
            extra_dim=0,
            in_channels=3,
            params=params,
        ).to(device)
    elif model_name == "dgcnn_v2":
        params = DGCNNv2Params(k=int(args.dgcnn_k), emb_dims=1024)
        model = DGCNNv2Classifier(
            num_classes=int(num_classes),
            dropout=float(args.dropout),
            extra_dim=0,
            in_channels=3,
            params=params,
        ).to(device)
    else:  # pragma: no cover
        raise SystemExit(f"Unknown model: {model_name}")
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    history_path = out_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "train_macro_f1", "val_loss", "val_acc", "val_macro_f1"])

    best_val = -1.0
    best_epoch = -1
    bad_epochs = 0

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        sum_loss = 0.0
        y_true: list[int] = []
        y_pred: list[int] = []
        n = 0
        for pts, y, _meta in dl_train:
            pts = pts.to(device)
            y = y.to(device)
            logits = model(pts)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            sum_loss += float(loss.item()) * int(pts.shape[0])
            n += int(pts.shape[0])
            pred = logits.argmax(dim=1).detach().cpu().numpy().astype(np.int64).tolist()
            y_pred.extend(pred)
            y_true.extend(y.detach().cpu().numpy().astype(np.int64).tolist())

        train_loss = float(sum_loss / max(1, n))
        train_acc = float(sum(int(a == b) for a, b in zip(y_true, y_pred, strict=True)) / max(1, len(y_true)))
        train_f1 = macro_f1(y_true, y_pred, num_classes=num_classes)

        val = evaluate(model, dl_val, device, num_classes=num_classes)
        with history_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    epoch,
                    f"{train_loss:.6f}",
                    f"{train_acc:.6f}",
                    f"{train_f1:.6f}",
                    f"{val['loss']:.6f}",
                    f"{val['acc']:.6f}",
                    f"{val['macro_f1']:.6f}",
                ]
            )

        improved = float(val["acc"]) > best_val + 1e-6
        if improved:
            best_val = float(val["acc"])
            best_epoch = int(epoch)
            bad_epochs = 0
            torch.save(
                {"model_state": model.state_dict(), "epoch": int(epoch), "val_acc": float(val["acc"]), "val_macro_f1": float(val["macro_f1"])},
                out_dir / "ckpt_best.pt",
            )
        else:
            bad_epochs += 1

        print(
            f"[epoch {epoch:03d}] "
            f"train(loss={train_loss:.6f} acc={train_acc:.4f} f1={train_f1:.4f}) "
            f"val(loss={float(val['loss']):.6f} acc={float(val['acc']):.4f} f1={float(val['macro_f1']):.4f}) "
            f"best_val_acc={best_val:.4f} ({best_epoch})",
            flush=True,
        )
        if int(args.patience) > 0 and bad_epochs >= int(args.patience):
            print(f"[early-stop] no improvement for {bad_epochs} epochs", flush=True)
            break

    if (out_dir / "ckpt_best.pt").exists():
        ckpt = torch.load(out_dir / "ckpt_best.pt", map_location=device)
        model.load_state_dict(ckpt["model_state"])

    final_val = evaluate(model, dl_val, device, num_classes=num_classes)
    final_test = evaluate(model, dl_test, device, num_classes=num_classes) if len(ds_test) > 0 else {}
    write_json(
        out_dir / "metrics.json",
        {
            "best_epoch": int(best_epoch),
            "best_val_acc": float(best_val),
            "val": dict(final_val),
            "test": dict(final_test),
            "labels": {"num_classes": int(num_classes)},
        },
    )
    print(f"[OK] out_dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
