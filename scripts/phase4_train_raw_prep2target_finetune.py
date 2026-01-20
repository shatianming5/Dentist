#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from _lib.device import normalize_device
from _lib.io import read_jsonl, write_json, write_jsonl
from _lib.seed import set_seed
from _lib.time import utc_compact_ts, utc_now_iso
from prep2target.dataset_raw import RawPrep2TargetDataset
from prep2target.metrics import (
    chamfer_l2,
    margin_loss,
    margin_loss_per_sample,
    occlusion_penalty,
    occlusion_penalty_per_sample,
)
from prep2target.models import ConstraintAuxHead, Prep2TargetLabelNet, Prep2TargetNet, forward_pred_and_latent
from prep2target.preview import write_ply_xyz
from prep2target.stats import pearsonr, spearmanr

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
    lambda_margin: float
    lambda_occlusion: float
    occlusion_clearance: float
    aux_weight_margin: float
    aux_weight_occlusion: float
    aux_hidden_dim: int
    init_ckpt: str
    cond_label: bool
    label_to_id: dict[str, int]


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    lambda_margin: float,
    lambda_occlusion: float,
    occlusion_clearance: float,
    aux_weight_margin: float,
    aux_weight_occlusion: float,
    aux_head: nn.Module | None,
    cond_label: bool,
) -> dict[str, float]:
    model.eval()
    sum_total = 0.0
    sum_chamfer = 0.0
    sum_margin = 0.0
    sum_occ = 0.0
    sum_aux_m_mse = 0.0
    sum_aux_o_mse = 0.0
    sum_aux_m_mae = 0.0
    sum_aux_o_mae = 0.0
    aux_pred_m: list[float] = []
    aux_gt_m: list[float] = []
    aux_pred_o: list[float] = []
    aux_gt_o: list[float] = []
    n = 0
    for prep, tgt, margin, opp, centroid, scale, rmat, label_id, _meta in loader:
        prep = prep.to(device)
        tgt = tgt.to(device)
        margin = margin.to(device)
        opp = opp.to(device)
        centroid = centroid.to(device)
        scale = scale.to(device).view(-1, 1, 1)
        rmat = rmat.to(device)
        label_id = label_id.to(device)

        pred, z = forward_pred_and_latent(model, prep, label_id, cond_label=cond_label)
        c = chamfer_l2(pred, tgt)
        m = margin_loss(pred, margin)
        raw_pred = (pred @ rmat.transpose(1, 2)) * scale + centroid[:, None, :]
        o = occlusion_penalty(raw_pred, opp, clearance=occlusion_clearance)

        total = c + float(lambda_margin) * m + float(lambda_occlusion) * o
        use_aux = aux_head is not None and (float(aux_weight_margin) > 0 or float(aux_weight_occlusion) > 0)
        if use_aux:
            gt_m = margin_loss_per_sample(tgt, margin)
            raw_tgt = (tgt @ rmat.transpose(1, 2)) * scale + centroid[:, None, :]
            gt_o = occlusion_penalty_per_sample(raw_tgt, opp, clearance=occlusion_clearance)
            pred_aux = aux_head(z)
            pred_m = pred_aux[:, 0]
            pred_o = pred_aux[:, 1]

            bsz = int(prep.shape[0])
            if float(aux_weight_margin) > 0:
                mse_m = F.mse_loss(pred_m, gt_m)
                mae_m = torch.mean(torch.abs(pred_m - gt_m))
                total = total + float(aux_weight_margin) * mse_m
                sum_aux_m_mse += float(mse_m.item()) * bsz
                sum_aux_m_mae += float(mae_m.item()) * bsz
            if float(aux_weight_occlusion) > 0:
                mse_o = F.mse_loss(pred_o, gt_o)
                mae_o = torch.mean(torch.abs(pred_o - gt_o))
                total = total + float(aux_weight_occlusion) * mse_o
                sum_aux_o_mse += float(mse_o.item()) * bsz
                sum_aux_o_mae += float(mae_o.item()) * bsz

            aux_pred_m.extend(pred_m.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1).tolist())
            aux_gt_m.extend(gt_m.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1).tolist())
            aux_pred_o.extend(pred_o.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1).tolist())
            aux_gt_o.extend(gt_o.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1).tolist())

        bsz = int(prep.shape[0])
        sum_total += float(total.item()) * bsz
        sum_chamfer += float(c.item()) * bsz
        sum_margin += float(m.item()) * bsz
        sum_occ += float(o.item()) * bsz
        n += bsz

    denom = max(1, n)
    use_aux_out = aux_head is not None and (float(aux_weight_margin) > 0 or float(aux_weight_occlusion) > 0)
    pear_m = pearsonr(np.asarray(aux_pred_m), np.asarray(aux_gt_m)) if use_aux_out else None
    spear_m = spearmanr(np.asarray(aux_pred_m), np.asarray(aux_gt_m)) if use_aux_out else None
    pear_o = pearsonr(np.asarray(aux_pred_o), np.asarray(aux_gt_o)) if use_aux_out else None
    spear_o = spearmanr(np.asarray(aux_pred_o), np.asarray(aux_gt_o)) if use_aux_out else None
    return {
        "total": sum_total / denom,
        "chamfer": sum_chamfer / denom,
        "margin": sum_margin / denom,
        "occlusion": sum_occ / denom,
        "aux_margin_mse": (sum_aux_m_mse / denom) if use_aux_out else 0.0,
        "aux_margin_mae": (sum_aux_m_mae / denom) if use_aux_out else 0.0,
        "aux_margin_pearson": float(pear_m) if pear_m is not None else 0.0,
        "aux_margin_spearman": float(spear_m) if spear_m is not None else 0.0,
        "aux_occlusion_mse": (sum_aux_o_mse / denom) if use_aux_out else 0.0,
        "aux_occlusion_mae": (sum_aux_o_mae / denom) if use_aux_out else 0.0,
        "aux_occlusion_pearson": float(pear_o) if pear_o is not None else 0.0,
        "aux_occlusion_spearman": float(spear_o) if spear_o is not None else 0.0,
    }


@torch.no_grad()
def save_preview(model: nn.Module, dataset: RawPrep2TargetDataset, device: torch.device, out_dir: Path, n: int) -> None:
    model.eval()
    metas: list[dict[str, Any]] = []
    preps: list[np.ndarray] = []
    gts: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    for i in range(min(int(n), len(dataset))):
        prep, tgt, _margin, _opp, _centroid, _scale, _rmat, label_id, meta = dataset[i]
        x = prep.unsqueeze(0).to(device)
        if hasattr(model, "label_emb"):
            y = model(x, label_id.to(device).view(1)).squeeze(0).cpu().numpy().astype(np.float32, copy=False)
        else:
            y = model(x).squeeze(0).cpu().numpy().astype(np.float32, copy=False)
        preps.append(prep.numpy().astype(np.float32, copy=False))
        gts.append(tgt.numpy().astype(np.float32, copy=False))
        preds.append(y)
        metas.append(
            {
                "case_key": meta.get("case_key"),
                "label": meta.get("label"),
                "sample_npz": meta.get("sample_npz"),
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "preview.npz", prep=np.stack(preps), gt=np.stack(gts), pred=np.stack(preds))
    write_jsonl(out_dir / "preview.jsonl", metas)


@torch.no_grad()
def save_test_previews(model: nn.Module, dataset: RawPrep2TargetDataset, device: torch.device, out_dir: Path, n: int) -> None:
    model.eval()
    metas: list[dict[str, Any]] = []
    preps: list[np.ndarray] = []
    gts: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    preps_raw: list[np.ndarray] = []
    gts_raw: list[np.ndarray] = []
    preds_raw: list[np.ndarray] = []

    for i in range(min(int(n), len(dataset))):
        prep, tgt, _margin, _opp, centroid, scale, rmat, label_id, meta = dataset[i]
        x = prep.unsqueeze(0).to(device)
        if hasattr(model, "label_emb"):
            pred = model(x, label_id.to(device).view(1)).squeeze(0).cpu().numpy().astype(np.float32, copy=False)
        else:
            pred = model(x).squeeze(0).cpu().numpy().astype(np.float32, copy=False)

        prep_np = prep.numpy().astype(np.float32, copy=False)
        tgt_np = tgt.numpy().astype(np.float32, copy=False)
        centroid_np = centroid.numpy().astype(np.float32, copy=False).reshape(1, 3)
        rmat_np = rmat.numpy().astype(np.float32, copy=False).reshape(3, 3)
        scale_f = float(scale.item())

        def to_raw(pts: np.ndarray) -> np.ndarray:
            p = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
            return (p @ rmat_np.T) * scale_f + centroid_np

        preps.append(prep_np)
        gts.append(tgt_np)
        preds.append(pred)
        preps_raw.append(to_raw(prep_np))
        gts_raw.append(to_raw(tgt_np))
        preds_raw.append(to_raw(pred))
        metas.append(
            {
                "i": int(i),
                "case_key": meta.get("case_key"),
                "label": meta.get("label"),
                "sample_npz": meta.get("sample_npz"),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    if preds:
        np.save(out_dir / "pred_target.npy", np.stack(preds, axis=0))
        np.save(out_dir / "gt_target.npy", np.stack(gts, axis=0))
        np.save(out_dir / "prep.npy", np.stack(preps, axis=0))

    if preds_raw:
        np.save(out_dir / "pred_target_raw.npy", np.stack(preds_raw, axis=0))
        np.save(out_dir / "gt_target_raw.npy", np.stack(gts_raw, axis=0))
        np.save(out_dir / "prep_raw.npy", np.stack(preps_raw, axis=0))
        write_ply_xyz(out_dir / "pred_target.ply", preds_raw[0])
        write_ply_xyz(out_dir / "gt_target.ply", gts_raw[0])
        write_ply_xyz(out_dir / "prep.ply", preps_raw[0])

    write_jsonl(out_dir / "preview.jsonl", metas)


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 4: fine-tune prep->target on raw_prep2target dataset (with opposing jaw + synthetic margin).")
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--data-root", type=Path, default=Path("processed/raw_prep2target/v1"))
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--n-points", type=int, default=512)
    ap.add_argument("--latent-dim", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--lambda-margin", type=float, default=0.1)
    ap.add_argument("--lambda-occlusion", type=float, default=0.1)
    ap.add_argument("--occlusion-clearance", type=float, default=0.5)
    ap.add_argument("--aux-weight-margin", type=float, default=0.0, help="Aux head MSE weight to predict GT margin proxy (0 disables).")
    ap.add_argument("--aux-weight-occlusion", type=float, default=0.0, help="Aux head MSE weight to predict GT occlusion proxy (0 disables).")
    ap.add_argument("--aux-hidden-dim", type=int, default=128, help="Aux head hidden dim (only used when aux enabled).")
    ap.add_argument("--init-ckpt", type=str, default="", help="Optional checkpoint path to initialize weights.")
    ap.add_argument("--cond-label", action="store_true", help="Condition on raw restoration label via an embedding added to latent.")
    ap.add_argument("--runs-dir", type=Path, default=Path("runs/raw_prep2target_finetune"))
    ap.add_argument("--exp-name", type=str, default="")
    ap.add_argument("--preview-samples", type=int, default=16)
    args = ap.parse_args()

    root = args.root.resolve()
    data_root = (root / args.data_root).resolve()
    index_path = data_root / "index.jsonl"
    if not index_path.exists():
        raise SystemExit(f"Missing index: {index_path} (run scripts/phase4_build_raw_prep2target.py first)")

    rows = read_jsonl(index_path)
    train_rows = [r for r in rows if r.get("split") == "train"]
    val_rows = [r for r in rows if r.get("split") == "val"]
    test_rows = [r for r in rows if r.get("split") == "test"]

    label_set = sorted({str(r.get("label") or "") for r in train_rows})
    label_to_id = {name: i for i, name in enumerate(label_set)}
    if "" not in label_to_id:
        label_to_id[""] = len(label_to_id)

    device_str = normalize_device(str(args.device))
    device = torch.device(device_str)
    set_seed(int(args.seed))

    exp_name = str(args.exp_name).strip() or f"raw_p2t_n{int(args.n_points)}_z{int(args.latent_dim)}_{utc_compact_ts()}"
    out_dir = (root / args.runs_dir / exp_name).resolve()
    # Allow deterministic run directory layouts (e.g., seed-based paths) when driven
    # by a higher-level runner. If a completed run exists, skip to avoid clobbering.
    metrics_path = out_dir / "metrics.json"
    skip_train = out_dir.exists() and metrics_path.exists()
    if skip_train:
        print(f"[SKIP] metrics exists: {metrics_path}", flush=True)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    init_ckpt_str = str(args.init_ckpt or "").strip()
    init_ckpt_path = Path(init_ckpt_str).expanduser() if init_ckpt_str else None
    if init_ckpt_path and not init_ckpt_path.is_absolute():
        init_ckpt_path = (root / init_ckpt_path).resolve()
    init_ckpt = str(init_ckpt_path) if init_ckpt_path else ""

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
        lambda_margin=float(args.lambda_margin),
        lambda_occlusion=float(args.lambda_occlusion),
        occlusion_clearance=float(args.occlusion_clearance),
        aux_weight_margin=float(args.aux_weight_margin),
        aux_weight_occlusion=float(args.aux_weight_occlusion),
        aux_hidden_dim=int(args.aux_hidden_dim),
        init_ckpt=init_ckpt,
        cond_label=bool(args.cond_label),
        label_to_id=label_to_id,
    )
    if not skip_train:
        write_json(out_dir / "config.json", asdict(cfg))

    ds_train = RawPrep2TargetDataset(rows=train_rows, data_root=data_root, label_to_id=label_to_id)
    ds_val = RawPrep2TargetDataset(rows=val_rows, data_root=data_root, label_to_id=label_to_id)
    ds_test = RawPrep2TargetDataset(rows=test_rows, data_root=data_root, label_to_id=label_to_id)

    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    dl_test = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    if cfg.cond_label:
        model = Prep2TargetLabelNet(latent_dim=cfg.latent_dim, n_points=cfg.n_points, num_labels=len(label_to_id)).to(device)
    else:
        model = Prep2TargetNet(latent_dim=cfg.latent_dim, n_points=cfg.n_points).to(device)
    use_aux = float(cfg.aux_weight_margin) > 0 or float(cfg.aux_weight_occlusion) > 0
    if use_aux:
        model.aux_head = ConstraintAuxHead(latent_dim=cfg.latent_dim, hidden_dim=cfg.aux_hidden_dim).to(device)
    if cfg.init_ckpt:
        ckpt_path = Path(cfg.init_ckpt)
        if not ckpt_path.is_file():
            raise SystemExit(f"init_ckpt must be a file: {ckpt_path}")
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"], strict=False)

    if skip_train:
        # Refresh previews without re-training (keeps README artifacts stable across code updates).
        ckpt_best = out_dir / "ckpt_best.pt"
        ckpt_final = out_dir / "ckpt_final.pt"
        load_path = ckpt_best if ckpt_best.is_file() else ckpt_final if ckpt_final.is_file() else None
        if load_path is not None:
            ckpt = torch.load(load_path, map_location=device, weights_only=False)
            if isinstance(ckpt, dict) and "model" in ckpt:
                model.load_state_dict(ckpt["model"], strict=False)
        need_test_preview = not (out_dir / "previews" / "test" / "pred_target.npy").exists()
        need_val_preview = not (out_dir / "previews" / "preview.npz").exists()
        if need_val_preview:
            save_preview(model, ds_val, device, out_dir / "previews", n=int(args.preview_samples))
        if need_test_preview:
            save_test_previews(model, ds_test, device, out_dir / "previews" / "test", n=int(args.preview_samples))
        return 0

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    aux_head = getattr(model, "aux_head", None) if use_aux else None

    history_path = out_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "epoch",
                "train_total",
                "train_chamfer",
                "train_margin",
                "train_occlusion",
                "train_aux_margin_mse",
                "train_aux_margin_mae",
                "train_aux_occlusion_mse",
                "train_aux_occlusion_mae",
                "val_total",
                "val_chamfer",
                "val_margin",
                "val_occlusion",
                "val_aux_margin_mse",
                "val_aux_margin_mae",
                "val_aux_occlusion_mse",
                "val_aux_occlusion_mae",
            ]
        )

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        sum_total = 0.0
        sum_c = 0.0
        sum_m = 0.0
        sum_o = 0.0
        sum_aux_m_mse = 0.0
        sum_aux_m_mae = 0.0
        sum_aux_o_mse = 0.0
        sum_aux_o_mae = 0.0
        n = 0

        for prep, tgt, margin, opp, centroid, scale, rmat, label_id, _meta in dl_train:
            prep = prep.to(device)
            tgt = tgt.to(device)
            margin = margin.to(device)
            opp = opp.to(device)
            centroid = centroid.to(device)
            scale = scale.to(device).view(-1, 1, 1)
            rmat = rmat.to(device)
            label_id = label_id.to(device)

            pred, z = forward_pred_and_latent(model, prep, label_id, cond_label=cfg.cond_label)
            c = chamfer_l2(pred, tgt)
            m = margin_loss(pred, margin)
            raw_pred = (pred @ rmat.transpose(1, 2)) * scale + centroid[:, None, :]
            o = occlusion_penalty(raw_pred, opp, clearance=cfg.occlusion_clearance)

            total = c + float(cfg.lambda_margin) * m + float(cfg.lambda_occlusion) * o
            bsz = int(prep.shape[0])
            if aux_head is not None:
                gt_m = margin_loss_per_sample(tgt, margin)
                raw_tgt = (tgt @ rmat.transpose(1, 2)) * scale + centroid[:, None, :]
                gt_o = occlusion_penalty_per_sample(raw_tgt, opp, clearance=cfg.occlusion_clearance)
                pred_aux = aux_head(z)
                pred_m = pred_aux[:, 0]
                pred_o = pred_aux[:, 1]

                if float(cfg.aux_weight_margin) > 0:
                    mse_m = F.mse_loss(pred_m, gt_m)
                    mae_m = torch.mean(torch.abs(pred_m - gt_m))
                    total = total + float(cfg.aux_weight_margin) * mse_m
                    sum_aux_m_mse += float(mse_m.item()) * bsz
                    sum_aux_m_mae += float(mae_m.item()) * bsz
                if float(cfg.aux_weight_occlusion) > 0:
                    mse_o = F.mse_loss(pred_o, gt_o)
                    mae_o = torch.mean(torch.abs(pred_o - gt_o))
                    total = total + float(cfg.aux_weight_occlusion) * mse_o
                    sum_aux_o_mse += float(mse_o.item()) * bsz
                    sum_aux_o_mae += float(mae_o.item()) * bsz

            opt.zero_grad(set_to_none=True)
            total.backward()
            opt.step()

            sum_total += float(total.item()) * bsz
            sum_c += float(c.item()) * bsz
            sum_m += float(m.item()) * bsz
            sum_o += float(o.item()) * bsz
            n += bsz

        denom = max(1, n)
        train_metrics = {
            "total": sum_total / denom,
            "chamfer": sum_c / denom,
            "margin": sum_m / denom,
            "occlusion": sum_o / denom,
            "aux_margin_mse": sum_aux_m_mse / denom,
            "aux_margin_mae": sum_aux_m_mae / denom,
            "aux_occlusion_mse": sum_aux_o_mse / denom,
            "aux_occlusion_mae": sum_aux_o_mae / denom,
        }

        val_metrics = evaluate(
            model,
            dl_val,
            device,
            lambda_margin=cfg.lambda_margin,
            lambda_occlusion=cfg.lambda_occlusion,
            occlusion_clearance=cfg.occlusion_clearance,
            aux_weight_margin=cfg.aux_weight_margin,
            aux_weight_occlusion=cfg.aux_weight_occlusion,
            aux_head=aux_head,
            cond_label=cfg.cond_label,
        )

        with history_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    epoch,
                    f"{train_metrics['total']:.6f}",
                    f"{train_metrics['chamfer']:.6f}",
                    f"{train_metrics['margin']:.6f}",
                    f"{train_metrics['occlusion']:.6f}",
                    f"{train_metrics['aux_margin_mse']:.6f}",
                    f"{train_metrics['aux_margin_mae']:.6f}",
                    f"{train_metrics['aux_occlusion_mse']:.6f}",
                    f"{train_metrics['aux_occlusion_mae']:.6f}",
                    f"{val_metrics['total']:.6f}",
                    f"{val_metrics['chamfer']:.6f}",
                    f"{val_metrics['margin']:.6f}",
                    f"{val_metrics['occlusion']:.6f}",
                    f"{val_metrics['aux_margin_mse']:.6f}",
                    f"{val_metrics['aux_margin_mae']:.6f}",
                    f"{val_metrics['aux_occlusion_mse']:.6f}",
                    f"{val_metrics['aux_occlusion_mae']:.6f}",
                ]
            )

        improved = val_metrics["total"] < best_val - 1e-6
        if improved:
            best_val = float(val_metrics["total"])
            best_epoch = epoch
            bad_epochs = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_metrics": val_metrics}, out_dir / "ckpt_best.pt")
        else:
            bad_epochs += 1

        print(
            f"[epoch {epoch:03d}] train_total={train_metrics['total']:.6f} val_total={val_metrics['total']:.6f} best={best_val:.6f} (epoch {best_epoch})",
            flush=True,
        )

        if cfg.patience > 0 and bad_epochs >= cfg.patience:
            print(f"[early-stop] no improvement for {bad_epochs} epochs", flush=True)
            break

    if (out_dir / "ckpt_best.pt").exists():
        ckpt = torch.load(out_dir / "ckpt_best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])

    test_metrics = evaluate(
        model,
        dl_test,
        device,
        lambda_margin=cfg.lambda_margin,
        lambda_occlusion=cfg.lambda_occlusion,
        occlusion_clearance=cfg.occlusion_clearance,
        aux_weight_margin=cfg.aux_weight_margin,
        aux_weight_occlusion=cfg.aux_weight_occlusion,
        aux_head=aux_head,
        cond_label=cfg.cond_label,
    )
    torch.save({"model": model.state_dict(), "epoch": best_epoch, "best_val_total": best_val, "test_metrics": test_metrics}, out_dir / "ckpt_final.pt")
    save_preview(model, ds_val, device, out_dir / "previews", n=int(args.preview_samples))
    save_test_previews(model, ds_test, device, out_dir / "previews" / "test", n=int(args.preview_samples))

    metrics: dict[str, Any] = {"best_epoch": int(best_epoch), "best_val_total": float(best_val), "test": test_metrics}
    if aux_head is not None:
        metrics["aux"] = {
            "weights": {"margin": float(cfg.aux_weight_margin), "occlusion": float(cfg.aux_weight_occlusion)},
            "hidden_dim": int(cfg.aux_hidden_dim),
            "test": {
                "margin": {
                    "mse": float(test_metrics.get("aux_margin_mse") or 0.0),
                    "mae": float(test_metrics.get("aux_margin_mae") or 0.0),
                    "pearson": float(test_metrics.get("aux_margin_pearson") or 0.0),
                    "spearman": float(test_metrics.get("aux_margin_spearman") or 0.0),
                },
                "occlusion": {
                    "mse": float(test_metrics.get("aux_occlusion_mse") or 0.0),
                    "mae": float(test_metrics.get("aux_occlusion_mae") or 0.0),
                    "pearson": float(test_metrics.get("aux_occlusion_pearson") or 0.0),
                    "spearman": float(test_metrics.get("aux_occlusion_spearman") or 0.0),
                },
            },
        }
    write_json(out_dir / "metrics.json", metrics)
    print(f"[OK] out_dir: {out_dir}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
