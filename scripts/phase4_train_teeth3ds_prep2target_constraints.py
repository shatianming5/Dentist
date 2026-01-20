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
from _lib.env import get_env_info
from _lib.git import get_git_info
from _lib.io import read_jsonl, write_json, write_jsonl
from _lib.seed import set_seed
from _lib.time import utc_compact_ts, utc_now_iso
from prep2target.metrics import chamfer_l2, margin_loss, occlusion_penalty_with_stats
from prep2target.models import Prep2TargetNet
from teeth3ds.constraints_data import (
    OpposingPointsCache,
    Teeth3DSConstraintsPrepTargetDataset,
    collate_constraints,
    opp_fdi,
    opp_jaw,
    split_case_key,
)


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
    margin_band: float
    margin_points: int
    lambda_margin: float
    lambda_occlusion: float
    occlusion_mode: str
    occlusion_clearance: float
    occlusion_points: int
    occlusion_min_points: int
    occlusion_max_center_dist_mult: float
    teeth3ds_dir: str
    limit_train: int
    limit_val: int
    limit_test: int


def _limit(rows: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    if k and k > 0:
        return rows[: int(k)]
    return rows

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    lambda_margin: float,
    lambda_occlusion: float,
    occlusion_clearance: float,
    jaw_cache: OpposingPointsCache,
    occlusion_max_center_dist_mult: float,
) -> dict[str, Any]:
    model.eval()
    sum_chamfer = 0.0
    sum_margin = 0.0
    sum_occ = 0.0
    sum_total = 0.0
    n = 0
    occ_samples = 0
    occ_skipped = 0
    occ_points = 0
    occ_contact = 0
    occ_min_d_sum = 0.0
    min_d_samples: list[np.ndarray] = []
    rng = np.random.default_rng(12345)

    for batch in loader:
        prep = batch["prep"].to(device)
        tgt = batch["tgt"].to(device)
        margin_pts = batch["margin"].to(device)

        pred = model(prep)
        c = chamfer_l2(pred, tgt)
        m = margin_loss(pred, margin_pts) if lambda_margin > 0 else torch.tensor(0.0, device=device)

        o = torch.tensor(0.0, device=device)
        if lambda_occlusion > 0:
            case_ids = batch["case_id"]
            jaws = batch["jaw"]
            fdis = batch["fdi"]
            cent = batch["centroid"].to(device)  # (B,3)
            scale_flat = batch["scale"].to(device).view(-1)  # (B,)
            scale = scale_flat.view(-1, 1, 1)  # (B,1,1)
            r = batch["R"].to(device)  # (B,3,3)
            raw_pred = (pred @ r.transpose(1, 2)) * scale + cent[:, None, :]

            sel_idx: list[int] = []
            opp_list: list[torch.Tensor] = []

            for i, (case_id, jaw, fdi) in enumerate(zip(case_ids, jaws, fdis, strict=True)):
                opp_jaw_name = opp_jaw(str(jaw))
                opp_fdi_id = opp_fdi(int(fdi)) if jaw_cache.mode == "tooth" else None
                opp = jaw_cache.get(str(case_id), opp_jaw_name, fdi=opp_fdi_id)
                if opp is None:
                    continue
                sel_idx.append(i)
                opp_list.append(torch.from_numpy(opp))
            if sel_idx:
                opp_pts = torch.stack(opp_list, dim=0).to(device)
                sel_t = torch.tensor(sel_idx, device=device)
                raw_sel = raw_pred[sel_t]
                cent_sel = cent[sel_t]
                scale_sel = scale_flat[sel_t]

                # Skip obviously misaligned opposing samples (heuristic): opposing centroid too far from target centroid.
                if float(occlusion_max_center_dist_mult) > 0:
                    opp_cent = opp_pts.mean(dim=1)  # (Bsel,3)
                    dist = torch.linalg.norm(opp_cent - cent_sel, dim=1)  # (Bsel,)
                    thr = float(occlusion_max_center_dist_mult) * scale_sel
                    ok = dist <= thr
                    if ok.any():
                        raw_sel = raw_sel[ok]
                        opp_pts = opp_pts[ok]
                    else:
                        raw_sel = raw_sel[:0]
                        opp_pts = opp_pts[:0]
                    skipped = int((~ok).sum().item())
                    occ_skipped += skipped

                if raw_sel.numel() > 0:
                    o, _stats, min_d = occlusion_penalty_with_stats(raw_sel, opp_pts, occlusion_clearance)
                    occ_samples += int(raw_sel.shape[0])
                    occ_points += int(min_d.numel())
                    occ_contact += int((min_d < float(occlusion_clearance)).sum().item())
                    occ_min_d_sum += float(min_d.sum().item())

                    flat = min_d.detach().cpu().numpy().reshape(-1)
                    if flat.size:
                        k = min(4096, flat.size)
                        if flat.size > k:
                            sel = rng.choice(flat.size, size=k, replace=False)
                            flat = flat[sel]
                        min_d_samples.append(flat.astype(np.float32, copy=False))

        total = c + float(lambda_margin) * m + float(lambda_occlusion) * o
        bsz = int(prep.shape[0])
        sum_chamfer += float(c.item()) * bsz
        sum_margin += float(m.item()) * bsz
        sum_occ += float(o.item()) * bsz
        sum_total += float(total.item()) * bsz
        n += bsz

    denom = max(1, n)
    out: dict[str, Any] = {
        "chamfer": sum_chamfer / denom,
        "margin": sum_margin / denom,
        "occlusion": sum_occ / denom,
        "total": sum_total / denom,
        "occlusion_samples": int(occ_samples),
        "occlusion_skipped": int(occ_skipped),
        "occlusion_points": int(occ_points),
    }
    if occ_points > 0 and float(occlusion_clearance) > 0:
        out["occlusion_contact_ratio"] = float(occ_contact) / float(occ_points)
        out["occlusion_min_d_mean"] = float(occ_min_d_sum) / float(occ_points)
    else:
        out["occlusion_contact_ratio"] = 0.0
        out["occlusion_min_d_mean"] = 0.0

    if min_d_samples:
        all_s = np.concatenate(min_d_samples, axis=0)
        if all_s.size > 200_000:
            idx = rng.choice(all_s.size, size=200_000, replace=False)
            all_s = all_s[idx]
        out["occlusion_min_d_p05"] = float(np.quantile(all_s, 0.05))
        out["occlusion_min_d_p50"] = float(np.quantile(all_s, 0.50))
        out["occlusion_min_d_p95"] = float(np.quantile(all_s, 0.95))
    else:
        out["occlusion_min_d_p05"] = 0.0
        out["occlusion_min_d_p50"] = 0.0
        out["occlusion_min_d_p95"] = 0.0
    return out


@torch.no_grad()
def save_preview(
    model: nn.Module,
    dataset: Teeth3DSConstraintsPrepTargetDataset,
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
        sample = dataset[i]
        x = sample.prep.unsqueeze(0).to(device)
        y = model(x).squeeze(0).cpu().numpy().astype(np.float32, copy=False)
        metas.append(
            {
                "case_id": sample.case_id,
                "jaw": sample.jaw,
                "instance_id": int(sample.instance_id),
                "fdi": int(sample.fdi),
                "sample_npz": sample.sample_npz,
                "cut_q": float(sample.cut_q),
                "cut_mode": dataset.cut_mode,
                "cut_thr": float(sample.cut_thr),
                "cut_n": sample.cut_n.tolist(),
            }
        )
        preps.append(sample.prep.numpy().astype(np.float32, copy=False))
        gts.append(sample.tgt.numpy().astype(np.float32, copy=False))
        preds.append(y)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "preview.npz",
        prep=np.stack(preps, axis=0),
        gt=np.stack(gts, axis=0),
        pred=np.stack(preds, axis=0),
    )
    write_jsonl(out_dir / "preview.jsonl", metas)


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 4: prep->target with minimal functional constraints (margin/occlusion).")
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--data-root", type=Path, default=Path("processed/teeth3ds_teeth/v1"))
    ap.add_argument("--teeth3ds-dir", type=Path, default=Path("data/teeth3ds"), help="Needed for opposing jaw occlusion.")
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
    ap.add_argument("--include-unknown-split", action="store_true")
    ap.add_argument(
        "--cut-mode",
        choices=["z", "plane"],
        default="plane",
        help="How to synthesize prep from target: z-quantile cut (z) or random plane cut (plane).",
    )
    ap.add_argument("--cut-q-min", type=float, default=0.65)
    ap.add_argument("--cut-q-max", type=float, default=0.75)
    ap.add_argument("--margin-band", type=float, default=0.02)
    ap.add_argument("--margin-points", type=int, default=64)
    ap.add_argument("--lambda-margin", type=float, default=0.0)
    ap.add_argument("--lambda-occlusion", type=float, default=0.0)
    ap.add_argument(
        "--occlusion-mode",
        choices=["jaw", "tooth"],
        default="tooth",
        help="Occlusion target: opposing jaw points (jaw) or opposing tooth points by FDI label (tooth).",
    )
    ap.add_argument("--occlusion-clearance", type=float, default=0.0)
    ap.add_argument("--occlusion-points", type=int, default=2048)
    ap.add_argument("--occlusion-min-points", type=int, default=200, help="Min raw points required for opposing points.")
    ap.add_argument(
        "--occlusion-max-center-dist-mult",
        type=float,
        default=10.0,
        help="Skip occlusion pairs when ||opp_centroid - target_centroid|| > mult * target_scale (0=disable).",
    )
    ap.add_argument(
        "--opposing-cache-dir",
        type=Path,
        default=None,
        help="Optional: directory to read/write cached opposing (jaw/tooth) point samples (npz).",
    )
    ap.add_argument(
        "--write-opposing-cache",
        action="store_true",
        help="Write opposing cache files to --opposing-cache-dir when missing.",
    )
    ap.add_argument(
        "--preload-opposing-cache",
        action="store_true",
        help="Preload opposing cache keys into memory (only recommended for occlusion-mode=jaw).",
    )
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-val", type=int, default=0)
    ap.add_argument("--limit-test", type=int, default=0)
    ap.add_argument("--runs-dir", type=Path, default=Path("runs/teeth3ds_prep2target_constraints"))
    ap.add_argument("--exp-name", type=str, default="")
    ap.add_argument("--preview-samples", type=int, default=16)
    args = ap.parse_args()

    root = args.root.resolve()
    data_root = (root / args.data_root).resolve()
    index_path = data_root / "index.jsonl"
    if not index_path.exists():
        raise SystemExit(f"Missing index: {index_path} (run scripts/phase2_build_teeth3ds_teeth.py first)")

    teeth3ds_dir = (root / args.teeth3ds_dir).resolve()
    if args.opposing_cache_dir is None:
        opposing_cache_dir = (data_root / "opposing_cache").resolve()
    else:
        p = args.opposing_cache_dir
        if not p.is_absolute():
            p = root / p
        opposing_cache_dir = p.resolve()

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
        or f"p2tC_{str(args.cut_mode)}_{str(args.occlusion_mode)}_n{int(args.n_points)}_seed{int(args.seed)}_{utc_compact_ts()}"
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
        margin_band=float(args.margin_band),
        margin_points=int(args.margin_points),
        lambda_margin=float(args.lambda_margin),
        lambda_occlusion=float(args.lambda_occlusion),
        occlusion_mode=str(args.occlusion_mode),
        occlusion_clearance=float(args.occlusion_clearance),
        occlusion_points=int(args.occlusion_points),
        occlusion_min_points=int(args.occlusion_min_points),
        occlusion_max_center_dist_mult=float(args.occlusion_max_center_dist_mult),
        teeth3ds_dir=str(teeth3ds_dir),
        limit_train=int(args.limit_train),
        limit_val=int(args.limit_val),
        limit_test=int(args.limit_test),
    )
    write_json(out_dir / "config.json", asdict(cfg))
    write_json(out_dir / "env.json", {"env": get_env_info(), "git": get_git_info(Path.cwd())})

    ds_train = Teeth3DSConstraintsPrepTargetDataset(
        rows=train_rows,
        data_root=data_root,
        n_points=cfg.n_points,
        seed=cfg.seed,
        cut_mode=cfg.cut_mode,
        cut_q_min=cfg.cut_q_min,
        cut_q_max=cfg.cut_q_max,
        margin_band=cfg.margin_band,
        margin_points=cfg.margin_points,
        deterministic=False,
        rng_offset=0,
    )
    ds_val = Teeth3DSConstraintsPrepTargetDataset(
        rows=val_rows,
        data_root=data_root,
        n_points=cfg.n_points,
        seed=cfg.seed,
        cut_mode=cfg.cut_mode,
        cut_q_min=cfg.cut_q_min,
        cut_q_max=cfg.cut_q_max,
        margin_band=cfg.margin_band,
        margin_points=cfg.margin_points,
        deterministic=True,
        rng_offset=10_000,
    )
    ds_test = Teeth3DSConstraintsPrepTargetDataset(
        rows=test_rows,
        data_root=data_root,
        n_points=cfg.n_points,
        seed=cfg.seed,
        cut_mode=cfg.cut_mode,
        cut_q_min=cfg.cut_q_min,
        cut_q_max=cfg.cut_q_max,
        margin_band=cfg.margin_band,
        margin_points=cfg.margin_points,
        deterministic=True,
        rng_offset=10_000,
    )

    dl_train = DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
        collate_fn=collate_constraints,
    )
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_constraints)
    dl_test = DataLoader(
        ds_test, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate_constraints
    )

    jaw_cache = OpposingPointsCache(
        teeth3ds_dir=teeth3ds_dir,
        n_points=cfg.occlusion_points,
        seed=cfg.seed,
        mode=cfg.occlusion_mode,
        min_points=cfg.occlusion_min_points,
        cache_dir=opposing_cache_dir,
        write_cache=bool(args.write_opposing_cache),
    )
    if cfg.lambda_occlusion > 0 and bool(args.preload_opposing_cache):
        if jaw_cache.mode != "jaw":
            print("[preload] skip: preload is only recommended for --occlusion-mode=jaw", flush=True)
        else:
            needed: set[tuple[str, str]] = set()
            for r in rows_known:
                case_id, jaw = split_case_key(str(r["case_key"]))
                needed.add((case_id, opp_jaw(str(jaw))))
            if args.include_unknown_split:
                for r in rows_unknown:
                    case_id, jaw = split_case_key(str(r["case_key"]))
                    needed.add((case_id, opp_jaw(str(jaw))))
            print(f"[preload] opposing jaw keys: {len(needed)}", flush=True)
            for i, (case_id, jaw) in enumerate(sorted(needed), start=1):
                jaw_cache.get(case_id, jaw)
                if i % 200 == 0:
                    print(f"[preload] {i}/{len(needed)}", flush=True)

    model = Prep2TargetNet(latent_dim=cfg.latent_dim, n_points=cfg.n_points).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

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
                "val_total",
                "val_chamfer",
                "val_margin",
                "val_occlusion",
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
        n = 0

        for batch in dl_train:
            prep = batch["prep"].to(device)
            tgt = batch["tgt"].to(device)
            margin_pts = batch["margin"].to(device)

            pred = model(prep)
            c = chamfer_l2(pred, tgt)
            m = margin_loss(pred, margin_pts) if cfg.lambda_margin > 0 else torch.tensor(0.0, device=device)

            o = torch.tensor(0.0, device=device)
            if cfg.lambda_occlusion > 0:
                case_ids = batch["case_id"]
                jaws = batch["jaw"]
                fdis = batch["fdi"]

                cent = batch["centroid"].to(device)
                scale_flat = batch["scale"].to(device).view(-1)
                scale = scale_flat.view(-1, 1, 1)
                r = batch["R"].to(device)
                raw_pred = (pred @ r.transpose(1, 2)) * scale + cent[:, None, :]

                sel_idx: list[int] = []
                opp_list: list[torch.Tensor] = []
                for i, (case_id, jaw, fdi) in enumerate(zip(case_ids, jaws, fdis, strict=True)):
                    opp_jaw_name = opp_jaw(str(jaw))
                    opp_fdi_id = opp_fdi(int(fdi)) if jaw_cache.mode == "tooth" else None
                    opp = jaw_cache.get(str(case_id), opp_jaw_name, fdi=opp_fdi_id)
                    if opp is None:
                        continue
                    sel_idx.append(i)
                    opp_list.append(torch.from_numpy(opp))
                if sel_idx:
                    opp_pts = torch.stack(opp_list, dim=0).to(device)
                    sel_t = torch.tensor(sel_idx, device=device)
                    raw_sel = raw_pred[sel_t]
                    cent_sel = cent[sel_t]
                    scale_sel = scale_flat[sel_t]

                    if float(cfg.occlusion_max_center_dist_mult) > 0:
                        opp_cent = opp_pts.mean(dim=1)
                        dist = torch.linalg.norm(opp_cent - cent_sel, dim=1)
                        thr = float(cfg.occlusion_max_center_dist_mult) * scale_sel
                        ok = dist <= thr
                        if ok.any():
                            raw_sel = raw_sel[ok]
                            opp_pts = opp_pts[ok]
                        else:
                            raw_sel = raw_sel[:0]
                            opp_pts = opp_pts[:0]

                    if raw_sel.numel() > 0:
                        o, _stats, _min_d = occlusion_penalty_with_stats(raw_sel, opp_pts, cfg.occlusion_clearance)

            total = c + float(cfg.lambda_margin) * m + float(cfg.lambda_occlusion) * o
            opt.zero_grad(set_to_none=True)
            total.backward()
            opt.step()

            bsz = int(prep.shape[0])
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
        }

        val_metrics = evaluate(
            model,
            dl_val,
            device,
            lambda_margin=cfg.lambda_margin,
            lambda_occlusion=cfg.lambda_occlusion,
            occlusion_clearance=cfg.occlusion_clearance,
            jaw_cache=jaw_cache,
            occlusion_max_center_dist_mult=cfg.occlusion_max_center_dist_mult,
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
                    f"{val_metrics['total']:.6f}",
                    f"{val_metrics['chamfer']:.6f}",
                    f"{val_metrics['margin']:.6f}",
                    f"{val_metrics['occlusion']:.6f}",
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
        ckpt = torch.load(out_dir / "ckpt_best.pt", map_location=device)
        model.load_state_dict(ckpt["model"])

    val_metrics_best = evaluate(
        model,
        dl_val,
        device,
        lambda_margin=cfg.lambda_margin,
        lambda_occlusion=cfg.lambda_occlusion,
        occlusion_clearance=cfg.occlusion_clearance,
        jaw_cache=jaw_cache,
        occlusion_max_center_dist_mult=cfg.occlusion_max_center_dist_mult,
    )
    test_metrics = evaluate(
        model,
        dl_test,
        device,
        lambda_margin=cfg.lambda_margin,
        lambda_occlusion=cfg.lambda_occlusion,
        occlusion_clearance=cfg.occlusion_clearance,
        jaw_cache=jaw_cache,
        occlusion_max_center_dist_mult=cfg.occlusion_max_center_dist_mult,
    )
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": best_epoch,
            "best_val_total": best_val,
            "val_metrics": val_metrics_best,
            "test_metrics": test_metrics,
        },
        out_dir / "ckpt_final.pt",
    )

    save_preview(model, ds_val, device, out_dir / "previews", n_samples=int(args.preview_samples))

    metrics = {
        "best_epoch": int(best_epoch),
        "best_val_total": float(best_val),
        "val": val_metrics_best,
        "test": test_metrics,
    }
    write_json(out_dir / "metrics.json", metrics)
    print(f"[OK] out_dir: {out_dir}")
    print(f"[OK] test_total: {test_metrics['total']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
