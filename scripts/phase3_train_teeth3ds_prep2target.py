#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from _lib.device import normalize_device
from _lib.io import read_jsonl, write_json, write_jsonl
from _lib.seed import set_seed
from _lib.time import utc_compact_ts, utc_now_iso
from prep2target.metrics import chamfer_l2
from prep2target.models import Prep2TargetNet
from teeth3ds.prep2target_dataset import Teeth3DSPrepTargetDataset

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
    num_workers: int
    patience: int
    include_unknown_split: bool
    cut_mode: str
    cut_q_min: float
    cut_q_max: float
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
    for prep, tgt, _meta in loader:
        prep = prep.to(device)
        tgt = tgt.to(device)
        pred = model(prep)
        loss = chamfer_l2(pred, tgt).item()
        total += float(loss) * int(prep.shape[0])
        n += int(prep.shape[0])
    return total / max(1, n)


@torch.no_grad()
def save_preview(
    model: nn.Module,
    dataset: Teeth3DSPrepTargetDataset,
    device: torch.device,
    out_dir: Path,
    *,
    n_samples: int,
) -> None:
    model.eval()
    metas: list[dict[str, Any]] = []
    preps: list[np.ndarray] = []
    gts: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    for i in range(min(int(n_samples), len(dataset))):
        prep, tgt, meta = dataset[i]
        x = prep.unsqueeze(0).to(device)
        y = model(x).squeeze(0).cpu().numpy().astype(np.float32, copy=False)
        metas.append(
            {
                "case_key": meta.get("case_key"),
                "jaw": meta.get("jaw"),
                "instance_id": meta.get("instance_id"),
                "fdi": meta.get("fdi"),
                "sample_npz": meta.get("sample_npz"),
                "cut_q": meta.get("cut_q"),
            }
        )
        preps.append(prep.numpy().astype(np.float32, copy=False))
        gts.append(tgt.numpy().astype(np.float32, copy=False))
        preds.append(y)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "prep2target_preview.npz",
        prep=np.stack(preps, axis=0),
        gt=np.stack(gts, axis=0),
        pred=np.stack(preds, axis=0),
    )
    write_jsonl(out_dir / "prep2target_preview.jsonl", metas)


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 3: synthetic prep->target baseline on Teeth3DS teeth.")
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--data-root", type=Path, default=Path("processed/teeth3ds_teeth/v1"))
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--n-points", type=int, default=512)
    ap.add_argument("--latent-dim", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument(
        "--include-unknown-split",
        action="store_true",
        help="Include rows with split=unknown into training set (val/test still fixed).",
    )
    ap.add_argument(
        "--cut-mode",
        choices=["z", "plane"],
        default="plane",
        help="How to synthesize prep from target: z-quantile cut (z) or random plane cut (plane).",
    )
    ap.add_argument("--cut-q-min", type=float, default=0.65)
    ap.add_argument("--cut-q-max", type=float, default=0.75)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-val", type=int, default=0)
    ap.add_argument("--limit-test", type=int, default=0)
    ap.add_argument("--runs-dir", type=Path, default=Path("runs/teeth3ds_prep2target"))
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

    exp_name = (
        str(args.exp_name).strip()
        or f"p2t_{str(args.cut_mode)}_n{int(args.n_points)}_z{int(args.latent_dim)}_seed{int(args.seed)}_{utc_compact_ts()}"
    )
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
        num_workers=int(args.num_workers),
        patience=int(args.patience),
        include_unknown_split=bool(args.include_unknown_split),
        cut_mode=str(args.cut_mode),
        cut_q_min=float(args.cut_q_min),
        cut_q_max=float(args.cut_q_max),
        limit_train=int(args.limit_train),
        limit_val=int(args.limit_val),
        limit_test=int(args.limit_test),
    )
    write_json(out_dir / "config.json", asdict(cfg))

    ds_train = Teeth3DSPrepTargetDataset(
        rows=train_rows,
        data_root=data_root,
        n_points=cfg.n_points,
        seed=cfg.seed,
        train=True,
        cut_mode=cfg.cut_mode,
        cut_q_min=cfg.cut_q_min,
        cut_q_max=cfg.cut_q_max,
    )
    ds_val = Teeth3DSPrepTargetDataset(
        rows=val_rows,
        data_root=data_root,
        n_points=cfg.n_points,
        seed=cfg.seed,
        train=False,
        cut_mode=cfg.cut_mode,
        cut_q_min=cfg.cut_q_min,
        cut_q_max=cfg.cut_q_max,
    )
    ds_test = Teeth3DSPrepTargetDataset(
        rows=test_rows,
        data_root=data_root,
        n_points=cfg.n_points,
        seed=cfg.seed,
        train=False,
        cut_mode=cfg.cut_mode,
        cut_q_min=cfg.cut_q_min,
        cut_q_max=cfg.cut_q_max,
    )

    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    dl_test = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = Prep2TargetNet(latent_dim=cfg.latent_dim, n_points=cfg.n_points).to(device)
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
        for prep, tgt, _meta in dl_train:
            prep = prep.to(device)
            tgt = tgt.to(device)
            pred = model(prep)
            loss = chamfer_l2(pred, tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item()) * int(prep.shape[0])
            n += int(prep.shape[0])
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
