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
from torch.utils.data import DataLoader, Dataset


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def utc_compact_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


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


def chamfer_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.ndim != 3 or target.ndim != 3 or pred.shape[-1] != 3 or target.shape[-1] != 3:
        raise ValueError(f"Invalid shapes: pred={tuple(pred.shape)} target={tuple(target.shape)}")
    d = torch.cdist(pred, target, p=2.0)
    return d.min(dim=2).values.mean() + d.min(dim=1).values.mean()


def jitter(points: np.ndarray, rng: np.random.Generator, sigma: float, clip: float) -> np.ndarray:
    if sigma <= 0:
        return points
    noise = rng.normal(loc=0.0, scale=sigma, size=points.shape).astype(np.float32)
    if clip > 0:
        noise = np.clip(noise, -clip, clip)
    return points + noise


class RawPointDataset(Dataset[tuple[torch.Tensor, dict[str, Any]]]):
    def __init__(
        self,
        *,
        rows: list[dict[str, Any]],
        data_root: Path,
        n_points: int,
        seed: int,
        train: bool,
        aug_jitter_sigma: float,
        aug_jitter_clip: float,
    ) -> None:
        self.rows = rows
        self.data_root = data_root
        self.n_points = int(n_points)
        self.train = bool(train)
        self.aug_jitter_sigma = float(aug_jitter_sigma)
        self.aug_jitter_clip = float(aug_jitter_clip)
        self.rng = np.random.default_rng(int(seed) + (0 if train else 10_000))

    def __len__(self) -> int:  # noqa: D401
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, Any]]:
        r = self.rows[int(idx)]
        npz_path = self.data_root / str(r["sample_npz"])
        with np.load(npz_path) as data:
            pts = np.asarray(data["points"], dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"Invalid points shape {pts.shape} in {npz_path}")

        if self.n_points > 0 and pts.shape[0] != self.n_points:
            replace = pts.shape[0] < self.n_points
            sel = self.rng.choice(pts.shape[0], size=self.n_points, replace=replace)
            pts = pts[sel]

        if self.train:
            pts = jitter(pts, self.rng, sigma=self.aug_jitter_sigma, clip=self.aug_jitter_clip)

        meta = {
            "case_key": r.get("case_key"),
            "split": r.get("split"),
            "source": r.get("source"),
            "label": r.get("label"),
            "sample_npz": r.get("sample_npz"),
        }
        return torch.from_numpy(pts), meta


class PointNetAE(nn.Module):
    """A simple PointNet-style autoencoder.

    Encoder uses the same `feat` stack as `scripts/phase3_train_raw_cls_baseline.py` (PointNetClassifier.feat),
    so its weights can be reused to initialize the classifier via `--init-feat`.
    """

    def __init__(self, n_points: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.n_points = int(n_points)
        self.feat = nn.Sequential(
            nn.Conv1d(3, 64, 1, bias=False),
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
        self.decoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(1024, self.n_points * 3),
        )

    def encode(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError(f"Expected (B,N,3), got {tuple(points.shape)}")
        x = points.transpose(1, 2).contiguous()  # (B,3,N)
        x = self.feat(x)  # (B,512,N)
        return torch.max(x, dim=2).values  # (B,512)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        z = self.encode(points)
        out = self.decoder(z)
        return out.view(points.shape[0], self.n_points, 3)


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
    aug_jitter_sigma: float
    aug_jitter_clip: float
    num_workers: int
    patience: int
    include_unknown_split: bool
    limit_train: int
    limit_val: int


def _limit(rows: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    if k and k > 0:
        return rows[: int(k)]
    return rows


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    n = 0
    for pts, _meta in loader:
        pts = pts.to(device)
        pred = model(pts)
        loss = chamfer_l2(pred, pts).item()
        total += float(loss) * int(pts.shape[0])
        n += int(pts.shape[0])
    return total / max(1, n)


@torch.no_grad()
def save_preview(model: nn.Module, dataset: RawPointDataset, device: torch.device, out_dir: Path, *, n_samples: int) -> None:
    model.eval()
    metas: list[dict[str, Any]] = []
    gts: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    for i in range(min(int(n_samples), len(dataset))):
        pts, meta = dataset[i]
        x = pts.unsqueeze(0).to(device)
        y = model(x).squeeze(0).cpu().numpy().astype(np.float32, copy=False)
        metas.append(meta)
        gts.append(pts.numpy().astype(np.float32, copy=False))
        preds.append(y)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "recon_preview.npz", gt=np.stack(gts, axis=0), pred=np.stack(preds, axis=0))
    (out_dir / "recon_preview.jsonl").write_text(
        "\n".join(json.dumps(m, ensure_ascii=False) for m in metas) + "\n", encoding="utf-8"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 2: pretrain PointNet feat on raw point clouds via autoencoding.")
    ap.add_argument("--data-root", type=Path, default=Path("processed/raw_cls/v6"))
    ap.add_argument("--run-root", type=Path, default=Path("runs/raw_ae"))
    ap.add_argument("--exp-name", type=str, default="")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--n-points", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--aug-jitter-sigma", type=float, default=0.01)
    ap.add_argument("--aug-jitter-clip", type=float, default=0.05)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--include-unknown-split", action="store_true")
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-val", type=int, default=0)
    ap.add_argument("--preview-samples", type=int, default=16)
    args = ap.parse_args()

    data_root = args.data_root.resolve()
    index_path = data_root / "index.jsonl"
    if not index_path.exists():
        raise SystemExit(f"Missing index.jsonl: {index_path}")

    rows = read_jsonl(index_path)
    train_rows = [r for r in rows if r.get("split") == "train"]
    if args.include_unknown_split:
        train_rows = train_rows + [r for r in rows if r.get("split") == "unknown"]
    val_rows = [r for r in rows if r.get("split") == "val"]
    train_rows = _limit(train_rows, int(args.limit_train))
    val_rows = _limit(val_rows, int(args.limit_val))

    device_str = normalize_device(str(args.device))
    device = torch.device(device_str)
    set_seed(int(args.seed))

    exp_name = str(args.exp_name).strip() or f"rawAE_n{int(args.n_points)}_seed{int(args.seed)}_{utc_compact_ts()}"
    out_dir = (args.run_root.resolve() / exp_name).resolve()
    out_dir.mkdir(parents=True, exist_ok=False)

    cfg = TrainConfig(
        generated_at=utc_now_iso(),
        seed=int(args.seed),
        device=device_str,
        data_root=str(data_root),
        out_dir=str(out_dir),
        exp_name=exp_name,
        n_points=int(args.n_points),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        dropout=float(args.dropout),
        aug_jitter_sigma=float(args.aug_jitter_sigma),
        aug_jitter_clip=float(args.aug_jitter_clip),
        num_workers=int(args.num_workers),
        patience=int(args.patience),
        include_unknown_split=bool(args.include_unknown_split),
        limit_train=int(args.limit_train),
        limit_val=int(args.limit_val),
    )
    write_json(out_dir / "config.json", asdict(cfg))
    write_json(out_dir / "env.json", {"env": get_env_info(), "git": get_git_info(Path.cwd())})

    ds_train = RawPointDataset(
        rows=train_rows,
        data_root=data_root,
        n_points=cfg.n_points,
        seed=cfg.seed,
        train=True,
        aug_jitter_sigma=cfg.aug_jitter_sigma,
        aug_jitter_clip=cfg.aug_jitter_clip,
    )
    ds_val = RawPointDataset(
        rows=val_rows,
        data_root=data_root,
        n_points=cfg.n_points,
        seed=cfg.seed,
        train=False,
        aug_jitter_sigma=0.0,
        aug_jitter_clip=0.0,
    )

    def collate(
        batch: list[tuple[torch.Tensor, dict[str, Any]]],
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        pts, metas = zip(*batch, strict=True)
        return torch.stack(list(pts), dim=0), list(metas)

    dl_train = DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
        collate_fn=collate,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate,
    )

    model = PointNetAE(n_points=cfg.n_points, dropout=cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history_path = out_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss"])

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        sum_loss = 0.0
        n = 0
        for pts, _meta in dl_train:
            pts = pts.to(device)
            pred = model(pts)
            loss = chamfer_l2(pred, pts)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sum_loss += float(loss.item()) * int(pts.shape[0])
            n += int(pts.shape[0])
        train_loss = sum_loss / max(1, n)
        val_loss = evaluate(model, dl_val, device)

        with history_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}"])

        improved = val_loss < best_val - 1e-6
        if improved:
            best_val = float(val_loss)
            best_epoch = epoch
            bad_epochs = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, out_dir / "ckpt_best.pt")
        else:
            bad_epochs += 1

        print(f"[epoch {epoch:03d}] train={train_loss:.6f} val={val_loss:.6f} best={best_val:.6f} ({best_epoch})", flush=True)
        if cfg.patience > 0 and bad_epochs >= cfg.patience:
            print(f"[early-stop] no improvement for {bad_epochs} epochs", flush=True)
            break

    if (out_dir / "ckpt_best.pt").exists():
        ckpt = torch.load(out_dir / "ckpt_best.pt", map_location=device)
        model.load_state_dict(ckpt["model_state"])

    final_val = evaluate(model, dl_val, device)
    save_preview(model, ds_val, device, out_dir / "previews", n_samples=int(args.preview_samples))
    write_json(
        out_dir / "metrics.json",
        {"best_epoch": int(best_epoch), "best_val_loss": float(best_val), "val_loss": float(final_val)},
    )
    print(f"[OK] out_dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
