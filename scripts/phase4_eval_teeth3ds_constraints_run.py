#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from _lib.device import normalize_device
from _lib.io import read_json, read_jsonl, write_json
from _lib.time import utc_now_iso
from prep2target.metrics import chamfer_l2, margin_loss
from prep2target.models import Prep2TargetNet
from teeth3ds.constraints_data import (
    OpposingPointsCache,
    Teeth3DSConstraintsPrepTargetDataset,
    collate_constraints,
    opp_fdi,
    opp_jaw,
)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    clearance: float,
    occlusion_mode: str,
    occlusion_max_center_dist_mult: float,
    opposing: OpposingPointsCache | None,
) -> dict[str, Any]:
    model.eval()
    sum_chamfer = 0.0
    sum_margin = 0.0
    pen_values: list[np.ndarray] = []
    min_d_samples: list[np.ndarray] = []
    min_d_sum = 0.0
    min_d_n = 0
    n = 0
    occ_n = 0

    occ_mode = str(occlusion_mode).strip().lower()
    if occ_mode not in {"jaw", "tooth"}:
        raise ValueError(f"Unknown occlusion_mode: {occlusion_mode} (allowed: jaw, tooth)")

    for batch in loader:
        prep = batch["prep"].to(device)
        tgt = batch["tgt"].to(device)
        margin_pts = batch["margin"].to(device)
        cent = batch["centroid"].to(device)
        scale_flat = batch["scale"].to(device).view(-1)
        scale = scale_flat.view(-1, 1, 1)
        r = batch["R"].to(device)

        pred = model(prep)
        c = chamfer_l2(pred, tgt)
        m = margin_loss(pred, margin_pts)

        sum_chamfer += float(c.item()) * int(prep.shape[0])
        sum_margin += float(m.item()) * int(prep.shape[0])

        if opposing is not None and float(clearance) > 0:
            raw_pred = (pred @ r.transpose(1, 2)) * scale + cent[:, None, :]
            opp_list: list[np.ndarray] = []
            sel_idx: list[int] = []
            for i, (case_id, jaw, fdi) in enumerate(zip(batch["case_id"], batch["jaw"], batch["fdi"], strict=True)):
                opp_jaw_name = opp_jaw(str(jaw))
                if occ_mode == "jaw":
                    pts = opposing.get(str(case_id), opp_jaw_name)
                else:
                    opp_fdi_id = opp_fdi(int(fdi))
                    if opp_fdi_id is None:
                        continue
                    pts = opposing.get(str(case_id), opp_jaw_name, fdi=int(opp_fdi_id))
                if pts is None:
                    continue
                opp_list.append(pts)
                sel_idx.append(i)

            if sel_idx:
                opp = torch.from_numpy(np.stack(opp_list, axis=0)).to(device)
                sel_t = torch.tensor(sel_idx, device=device)
                raw_sel = raw_pred[sel_t]

                if float(occlusion_max_center_dist_mult) > 0:
                    opp_cent = opp.mean(dim=1)
                    cent_sel = cent[sel_t]
                    scale_sel = scale_flat[sel_t]
                    dist = torch.linalg.norm(opp_cent - cent_sel, dim=1)
                    ok = dist <= float(occlusion_max_center_dist_mult) * scale_sel
                    if ok.any():
                        raw_sel = raw_sel[ok]
                        opp = opp[ok]
                    else:
                        raw_sel = raw_sel[:0]
                        opp = opp[:0]

                if raw_sel.numel() > 0:
                    d = torch.cdist(raw_sel, opp, p=2.0)
                    min_d = d.min(dim=2).values
                    pen = torch.relu(float(clearance) - min_d)
                    pen_values.append(pen.detach().cpu().numpy().reshape(-1))
                    md = min_d.detach().cpu().numpy().reshape(-1)
                    min_d_samples.append(md.astype(np.float32, copy=False))
                    min_d_sum += float(md.sum())
                    min_d_n += int(md.size)
                    occ_n += int(raw_sel.shape[0])

        n += int(prep.shape[0])

    denom = max(1, n)
    out: dict[str, Any] = {
        "count": int(n),
        "chamfer": sum_chamfer / denom,
        "margin": sum_margin / denom,
        "occlusion_count": int(occ_n),
    }
    if pen_values:
        pen_all = np.concatenate(pen_values, axis=0)
        out["occlusion_pen_mean"] = float(np.mean(pen_all))
        out["occlusion_pen_max"] = float(np.max(pen_all))
        out["occlusion_pen_p95"] = float(np.quantile(pen_all, 0.95))
        out["occlusion_contact_ratio"] = float(np.mean(pen_all > 0))
    else:
        out["occlusion_pen_mean"] = 0.0
        out["occlusion_pen_max"] = 0.0
        out["occlusion_pen_p95"] = 0.0
        out["occlusion_contact_ratio"] = 0.0

    if min_d_samples and min_d_n > 0:
        all_md = np.concatenate(min_d_samples, axis=0)
        if all_md.size > 200_000:
            rng = np.random.default_rng(123)
            idx = rng.choice(all_md.size, size=200_000, replace=False)
            all_md = all_md[idx]
        out["occlusion_min_d_mean"] = float(min_d_sum) / float(min_d_n)
        out["occlusion_min_d_p05"] = float(np.quantile(all_md, 0.05))
        out["occlusion_min_d_p50"] = float(np.quantile(all_md, 0.50))
        out["occlusion_min_d_p95"] = float(np.quantile(all_md, 0.95))
    else:
        out["occlusion_min_d_mean"] = 0.0
        out["occlusion_min_d_p05"] = 0.0
        out["occlusion_min_d_p50"] = 0.0
        out["occlusion_min_d_p95"] = 0.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate a constraints run with fixed metrics (chamfer/margin/occlusion).")
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("processed/teeth3ds_teeth/v1"))
    ap.add_argument("--teeth3ds-dir", type=Path, default=Path("data/teeth3ds"))
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cut-mode", type=str, default=None, help="Override cut_mode: z|plane (default: from run config).")
    ap.add_argument("--cut-q-min", type=float, default=None)
    ap.add_argument("--cut-q-max", type=float, default=None)
    ap.add_argument("--margin-band", type=float, default=None)
    ap.add_argument("--margin-points", type=int, default=None)
    ap.add_argument("--deterministic", action="store_true", help="Use deterministic cut_q per sample.")
    ap.add_argument(
        "--occlusion-mode", type=str, default=None, help="Override occlusion_mode: jaw|tooth (default: from run config)."
    )
    ap.add_argument("--occlusion-clearance", type=float, default=None)
    ap.add_argument("--occlusion-max-center-dist-mult", type=float, default=None)
    ap.add_argument("--opposing-cache-dir", type=Path, default=Path("processed/teeth3ds_teeth/v1/opposing_cache"))
    ap.add_argument("--write-opposing-cache", action="store_true", help="Write missing opposing cache npz files.")
    ap.add_argument("--no-occlusion", action="store_true")
    ap.add_argument("--ckpt", choices=["best", "final"], default="best")
    args = ap.parse_args()

    root = args.root.resolve()
    run_dir = (root / args.run_dir).resolve()
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        raise SystemExit(f"Missing config.json in run dir: {run_dir}")
    cfg = read_json(cfg_path)

    data_root = (root / args.data_root).resolve()
    index_path = data_root / "index.jsonl"
    rows = read_jsonl(index_path)
    rows = [r for r in rows if r.get("split") == args.split]
    if int(args.limit) > 0:
        rows = rows[: int(args.limit)]

    n_points = int(cfg.get("n_points") or 256)
    latent_dim = int(cfg.get("latent_dim") or 256)
    seed = int(cfg.get("seed") or 1337)
    occlusion_points = int(cfg.get("occlusion_points") or 2048)
    occlusion_min_points = int(cfg.get("occlusion_min_points") or 200)

    teeth3ds_dir_cfg = str(cfg.get("teeth3ds_dir") or "").strip()
    if teeth3ds_dir_cfg and Path(teeth3ds_dir_cfg).exists():
        teeth3ds_dir = Path(teeth3ds_dir_cfg).resolve()
    else:
        teeth3ds_dir = (root / args.teeth3ds_dir).resolve()
    if not teeth3ds_dir.exists():
        raise SystemExit(f"Missing teeth3ds_dir: {teeth3ds_dir} (set --teeth3ds-dir or fix run config.json)")

    cut_mode = str(args.cut_mode).strip().lower() if args.cut_mode else str(cfg.get("cut_mode") or "z").strip().lower()
    if cut_mode not in {"z", "plane"}:
        raise SystemExit(f"Unknown cut_mode: {cut_mode} (expected z|plane)")
    cut_q_min = float(args.cut_q_min if args.cut_q_min is not None else (cfg.get("cut_q_min") or 0.65))
    cut_q_max = float(args.cut_q_max if args.cut_q_max is not None else (cfg.get("cut_q_max") or 0.75))
    margin_band = float(args.margin_band if args.margin_band is not None else (cfg.get("margin_band") or 0.02))
    margin_points = int(args.margin_points if args.margin_points is not None else (cfg.get("margin_points") or 64))

    occlusion_mode = (
        str(args.occlusion_mode).strip().lower() if args.occlusion_mode else str(cfg.get("occlusion_mode") or "jaw").strip().lower()
    )
    if occlusion_mode not in {"jaw", "tooth"}:
        raise SystemExit(f"Unknown occlusion_mode: {occlusion_mode} (expected jaw|tooth)")
    clearance = float(args.occlusion_clearance if args.occlusion_clearance is not None else (cfg.get("occlusion_clearance") or 0.5))
    occlusion_max_center_dist_mult = float(
        args.occlusion_max_center_dist_mult
        if args.occlusion_max_center_dist_mult is not None
        else (cfg.get("occlusion_max_center_dist_mult") or 0.0)
    )

    device_str = normalize_device(str(args.device))
    device = torch.device(device_str)

    ds = Teeth3DSConstraintsPrepTargetDataset(
        rows=rows,
        data_root=data_root,
        n_points=n_points,
        seed=seed,
        cut_mode=cut_mode,
        cut_q_min=cut_q_min,
        cut_q_max=cut_q_max,
        margin_band=margin_band,
        margin_points=margin_points,
        deterministic=bool(args.deterministic),
        rng_offset=10_000,
    )
    dl = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=collate_constraints,
    )

    model = Prep2TargetNet(latent_dim=latent_dim, n_points=n_points).to(device)
    ckpt_path = run_dir / ("ckpt_best.pt" if args.ckpt == "best" else "ckpt_final.pt")
    if not ckpt_path.exists():
        ckpt_path = run_dir / "ckpt_final.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    opposing: OpposingPointsCache | None = None
    cache_dir = (root / args.opposing_cache_dir).resolve()
    if not args.no_occlusion and float(clearance) > 0:
        opposing = OpposingPointsCache(
            teeth3ds_dir=teeth3ds_dir,
            n_points=occlusion_points,
            seed=seed,
            mode=str(occlusion_mode),
            min_points=occlusion_min_points,
            cache_dir=cache_dir,
            write_cache=bool(args.write_opposing_cache),
        )

    out = evaluate(
        model,
        dl,
        device,
        clearance=float(clearance),
        occlusion_mode=str(occlusion_mode),
        occlusion_max_center_dist_mult=float(occlusion_max_center_dist_mult),
        opposing=opposing,
    )
    result = {
        "generated_at": utc_now_iso(),
        "run_dir": str(run_dir),
        "split": str(args.split),
        "n_points": n_points,
        "latent_dim": latent_dim,
        "cut_mode": str(cut_mode),
        "cut_q_min": cut_q_min,
        "cut_q_max": cut_q_max,
        "margin_band": margin_band,
        "margin_points": margin_points,
        "occlusion_mode": str(occlusion_mode),
        "occlusion_clearance": float(clearance),
        "occlusion_max_center_dist_mult": float(occlusion_max_center_dist_mult),
        "occlusion_enabled": bool(opposing is not None and not args.no_occlusion and float(clearance) > 0),
        "metrics": out,
    }

    out_path = run_dir / f"eval_{args.split}.json"
    write_json(out_path, result)
    print(f"[OK] wrote: {out_path}")
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
