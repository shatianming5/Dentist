#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
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
    d = torch.cdist(pred, target, p=2.0)  # (B, N, M)
    min_pred = d.min(dim=2).values
    min_tgt = d.min(dim=1).values
    return min_pred.mean() + min_tgt.mean()


class Teeth3DSToothDataset(Dataset[tuple[torch.Tensor, dict[str, Any]]]):
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

    def _maybe_jitter(self, pts: np.ndarray) -> np.ndarray:
        if not self.train or self.aug_jitter_sigma <= 0:
            return pts
        noise = self.rng.normal(loc=0.0, scale=self.aug_jitter_sigma, size=pts.shape).astype(np.float32)
        if self.aug_jitter_clip > 0:
            noise = np.clip(noise, -self.aug_jitter_clip, self.aug_jitter_clip)
        return (pts + noise).astype(np.float32, copy=False)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, Any]]:
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

        pts = self._maybe_jitter(pts)
        return torch.from_numpy(pts), r


class PointNetEncoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        ld = int(latent_dim)
        self.net = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, ld, 1),
            nn.BatchNorm1d(ld),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != 3:
            raise ValueError(f"Expected (B,N,3), got {tuple(x.shape)}")
        x = x.transpose(1, 2)  # (B,3,N)
        feat = self.net(x)  # (B,ld,N)
        z = feat.max(dim=2).values  # (B,ld)
        return z


class MLPDecoder(nn.Module):
    def __init__(self, latent_dim: int, n_points: int) -> None:
        super().__init__()
        ld = int(latent_dim)
        n = int(n_points)
        self.n_points = n
        self.net = nn.Sequential(
            nn.Linear(ld, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, n * 3),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2:
            raise ValueError(f"Expected (B,ld), got {tuple(z.shape)}")
        x = self.net(z)
        return x.view(z.shape[0], self.n_points, 3)


class PointAE(nn.Module):
    def __init__(self, latent_dim: int, n_points: int) -> None:
        super().__init__()
        self.encoder = PointNetEncoder(latent_dim=int(latent_dim))
        self.decoder = MLPDecoder(latent_dim=int(latent_dim), n_points=int(n_points))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


@dataclass(frozen=True)
class TrainConfig:
    generated_at: str
    seed: int
    device: str
    data_root: str
    out_dir: str
    exp_name: str
    n_points: int
    latent_dim: int
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    aug_jitter_sigma: float
    aug_jitter_clip: float
    num_workers: int
    patience: int
    include_unknown_split: bool
    limit_train: int
    limit_val: int
    limit_test: int


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
def save_preview(
    model: nn.Module,
    dataset: Teeth3DSToothDataset,
    device: torch.device,
    out_dir: Path,
    *,
    n_samples: int,
) -> None:
    model.eval()
    metas: list[dict[str, Any]] = []
    gts: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    for i in range(min(int(n_samples), len(dataset))):
        pts, meta = dataset[i]
        x = pts.unsqueeze(0).to(device)
        y = model(x).squeeze(0).cpu().numpy().astype(np.float32, copy=False)
        metas.append(
            {
                "case_key": meta.get("case_key"),
                "jaw": meta.get("jaw"),
                "instance_id": meta.get("instance_id"),
                "fdi": meta.get("fdi"),
                "sample_npz": meta.get("sample_npz"),
            }
        )
        gts.append(pts.numpy().astype(np.float32, copy=False))
        preds.append(y)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "recon_preview.npz", gt=np.stack(gts, axis=0), pred=np.stack(preds, axis=0))
    (out_dir / "recon_preview.jsonl").write_text(
        "\n".join(json.dumps(m, ensure_ascii=False) for m in metas) + "\n", encoding="utf-8"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 2: train a point cloud AE on Teeth3DS single-tooth dataset.")
    ap.add_argument("--root", type=Path, default=Path("."), help="Repo root (default: .)")
    ap.add_argument("--data-root", type=Path, default=Path("processed/teeth3ds_teeth/v1"))
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--n-points", type=int, default=1024)
    ap.add_argument("--latent-dim", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--aug-jitter-sigma", type=float, default=0.0)
    ap.add_argument("--aug-jitter-clip", type=float, default=0.02)
    ap.add_argument(
        "--include-unknown-split",
        action="store_true",
        help="Include rows with split=unknown into training set (val/test still fixed).",
    )
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-val", type=int, default=0)
    ap.add_argument("--limit-test", type=int, default=0)
    ap.add_argument("--runs-dir", type=Path, default=Path("runs/teeth3ds_ae"))
    ap.add_argument("--exp-name", type=str, default="")
    ap.add_argument("--preview-samples", type=int, default=16)
    args = ap.parse_args()

    root = args.root.resolve()
    data_root = (root / args.data_root).resolve()
    index_path = data_root / "index.jsonl"
    if not index_path.exists():
        raise SystemExit(f"Missing index: {index_path} (run scripts/phase2_build_teeth3ds_teeth.py first)")

    rows = read_jsonl(index_path)
    rows_known = [r for r in rows if r.get("split") in {"train", "val", "test"}]
    rows_unknown = [r for r in rows if r.get("split") == "unknown"]

    train_rows = [r for r in rows_known if r.get("split") == "train"]
    if args.include_unknown_split:
        train_rows = train_rows + rows_unknown
    val_rows = [r for r in rows_known if r.get("split") == "val"]
    test_rows = [r for r in rows_known if r.get("split") == "test"]

    train_rows = _limit(train_rows, int(args.limit_train))
    val_rows = _limit(val_rows, int(args.limit_val))
    test_rows = _limit(test_rows, int(args.limit_test))

    device_str = normalize_device(str(args.device))
    device = torch.device(device_str)
    set_seed(int(args.seed))

    exp_name = str(args.exp_name).strip() or f"ae_n{int(args.n_points)}_z{int(args.latent_dim)}_seed{int(args.seed)}_{utc_compact_ts()}"
    out_dir = (root / args.runs_dir / exp_name).resolve()
    out_dir.mkdir(parents=True, exist_ok=False)

    cfg = TrainConfig(
        generated_at=utc_now_iso(),
        seed=int(args.seed),
        device=device_str,
        data_root=str(data_root),
        out_dir=str(out_dir),
        exp_name=exp_name,
        n_points=int(args.n_points),
        latent_dim=int(args.latent_dim),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        aug_jitter_sigma=float(args.aug_jitter_sigma),
        aug_jitter_clip=float(args.aug_jitter_clip),
        num_workers=int(args.num_workers),
        patience=int(args.patience),
        include_unknown_split=bool(args.include_unknown_split),
        limit_train=int(args.limit_train),
        limit_val=int(args.limit_val),
        limit_test=int(args.limit_test),
    )
    write_json(out_dir / "config.json", asdict(cfg))

    ds_train = Teeth3DSToothDataset(
        rows=train_rows,
        data_root=data_root,
        n_points=cfg.n_points,
        seed=cfg.seed,
        train=True,
        aug_jitter_sigma=cfg.aug_jitter_sigma,
        aug_jitter_clip=cfg.aug_jitter_clip,
    )
    ds_val = Teeth3DSToothDataset(
        rows=val_rows,
        data_root=data_root,
        n_points=cfg.n_points,
        seed=cfg.seed,
        train=False,
        aug_jitter_sigma=0.0,
        aug_jitter_clip=0.0,
    )
    ds_test = Teeth3DSToothDataset(
        rows=test_rows,
        data_root=data_root,
        n_points=cfg.n_points,
        seed=cfg.seed,
        train=False,
        aug_jitter_sigma=0.0,
        aug_jitter_clip=0.0,
    )

    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    dl_test = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = PointAE(latent_dim=cfg.latent_dim, n_points=cfg.n_points).to(device)
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
        total = 0.0
        n = 0
        for pts, _meta in dl_train:
            pts = pts.to(device)
            pred = model(pts)
            loss = chamfer_l2(pred, pts)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item()) * int(pts.shape[0])
            n += int(pts.shape[0])
        train_loss = total / max(1, n)
        val_loss = evaluate(model, dl_val, device)

        with history_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}"])

        improved = val_loss < best_val - 1e-6
        if improved:
            best_val = val_loss
            best_epoch = epoch
            bad_epochs = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, out_dir / "ckpt_best.pt")
        else:
            bad_epochs += 1

        print(
            f"[epoch {epoch:03d}] train={train_loss:.6f} val={val_loss:.6f} best={best_val:.6f} (epoch {best_epoch})",
            flush=True,
        )

        if cfg.patience > 0 and bad_epochs >= cfg.patience:
            print(f"[early-stop] no improvement for {bad_epochs} epochs", flush=True)
            break

    if (out_dir / "ckpt_best.pt").exists():
        ckpt = torch.load(out_dir / "ckpt_best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])

    test_loss = evaluate(model, dl_test, device)
    torch.save({"model": model.state_dict(), "epoch": best_epoch, "val_loss": best_val, "test_loss": test_loss}, out_dir / "ckpt_final.pt")

    save_preview(
        model,
        ds_val,
        device,
        out_dir / "previews",
        n_samples=int(args.preview_samples),
    )

    metrics = {"best_epoch": int(best_epoch), "best_val": float(best_val), "test_loss": float(test_loss)}
    write_json(out_dir / "metrics.json", metrics)
    print(f"[OK] out_dir: {out_dir}")
    print(f"[OK] test_loss: {test_loss:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

