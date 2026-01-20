#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def read_json_maybe(path: Path) -> dict[str, Any] | None:
    try:
        if not path.exists():
            return None
        return read_json(path)
    except Exception:
        return None


def fmt(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        if not math.isfinite(x):
            return ""
        return f"{x:.6f}"
    return str(x)


@dataclass(frozen=True)
class Row:
    exp: str
    out_dir: str
    cut_mode: str
    occlusion_mode: str
    n_points: int
    latent_dim: int
    cut_q_min: float
    cut_q_max: float
    lambda_margin: float
    lambda_occlusion: float
    occlusion_clearance: float
    occlusion_points: int
    occlusion_max_center_dist_mult: float
    best_epoch: int
    best_val_total: float
    eval_val_total: float | None
    eval_val_chamfer: float | None
    eval_val_margin: float | None
    eval_val_occlusion_pen_mean: float | None
    eval_val_occlusion_contact_ratio: float | None
    eval_val_occlusion_min_d_p05: float | None
    eval_val_occlusion_min_d_p50: float | None
    eval_val_occlusion_min_d_p95: float | None
    eval_test_total: float | None
    eval_test_chamfer: float | None
    eval_test_margin: float | None
    eval_test_occlusion_pen_mean: float | None
    eval_test_occlusion_contact_ratio: float | None
    eval_test_occlusion_min_d_p05: float | None
    eval_test_occlusion_min_d_p50: float | None
    eval_test_occlusion_min_d_p95: float | None


def get_float(d: dict[str, Any] | None, key: str) -> float | None:
    if not d:
        return None
    v = d.get(key)
    if v is None:
        return None
    try:
        x = float(v)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return x


def eval_total(m: dict[str, Any] | None, *, lambda_margin: float, lambda_occlusion: float) -> float | None:
    if not m:
        return None
    chamfer = get_float(m, "chamfer")
    margin = get_float(m, "margin")
    occ_pen = get_float(m, "occlusion_pen_mean")
    if chamfer is None or margin is None or occ_pen is None:
        return None
    return float(chamfer + float(lambda_margin) * margin + float(lambda_occlusion) * occ_pen)


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize teeth3ds_prep2target_constraints runs into CSV/MD.")
    ap.add_argument("--runs-dir", type=Path, default=Path("runs/teeth3ds_prep2target_constraints"))
    ap.add_argument("--out-prefix", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=50, help="Max rows to show in the Markdown table.")
    args = ap.parse_args()

    runs_dir = args.runs_dir.resolve()
    if not runs_dir.exists():
        raise SystemExit(f"Missing runs dir: {runs_dir}")

    if args.out_prefix is None:
        out_prefix = runs_dir / f"summary_{utc_now_compact()}"
    else:
        out_prefix = args.out_prefix
        if not out_prefix.is_absolute():
            out_prefix = (Path.cwd() / out_prefix).resolve()
        else:
            out_prefix = out_prefix.resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    rows: list[Row] = []
    for d in sorted([p for p in runs_dir.iterdir() if p.is_dir()]):
        cfg_path = d / "config.json"
        met_path = d / "metrics.json"
        if not cfg_path.exists():
            continue
        try:
            cfg = read_json(cfg_path)
            met = read_json_maybe(met_path) or {}

            cut_mode = str(cfg.get("cut_mode") or "z").strip().lower()
            occlusion_mode = str(cfg.get("occlusion_mode") or "jaw").strip().lower()
            if cut_mode not in {"z", "plane"}:
                cut_mode = "z"
            if occlusion_mode not in {"jaw", "tooth"}:
                occlusion_mode = "jaw"

            lambda_margin = float(cfg.get("lambda_margin") or 0.0)
            lambda_occlusion = float(cfg.get("lambda_occlusion") or 0.0)

            eval_val = read_json_maybe(d / "eval_val.json")
            eval_test = read_json_maybe(d / "eval_test.json")
            eval_val_m = (eval_val or {}).get("metrics") if isinstance((eval_val or {}).get("metrics"), dict) else None
            eval_test_m = (eval_test or {}).get("metrics") if isinstance((eval_test or {}).get("metrics"), dict) else None

            rows.append(
                Row(
                    exp=str(cfg.get("exp_name") or d.name),
                    out_dir=str(d),
                    cut_mode=cut_mode,
                    occlusion_mode=occlusion_mode,
                    n_points=int(cfg.get("n_points") or 0),
                    latent_dim=int(cfg.get("latent_dim") or 0),
                    cut_q_min=float(cfg.get("cut_q_min") or 0.0),
                    cut_q_max=float(cfg.get("cut_q_max") or 0.0),
                    lambda_margin=lambda_margin,
                    lambda_occlusion=lambda_occlusion,
                    occlusion_clearance=float(cfg.get("occlusion_clearance") or 0.0),
                    occlusion_points=int(cfg.get("occlusion_points") or 0),
                    occlusion_max_center_dist_mult=float(cfg.get("occlusion_max_center_dist_mult") or 0.0),
                    best_epoch=int(met.get("best_epoch") or 0),
                    best_val_total=float(met.get("best_val_total") or float("inf")),
                    eval_val_total=eval_total(eval_val_m, lambda_margin=lambda_margin, lambda_occlusion=lambda_occlusion),
                    eval_val_chamfer=get_float(eval_val_m, "chamfer"),
                    eval_val_margin=get_float(eval_val_m, "margin"),
                    eval_val_occlusion_pen_mean=get_float(eval_val_m, "occlusion_pen_mean"),
                    eval_val_occlusion_contact_ratio=get_float(eval_val_m, "occlusion_contact_ratio"),
                    eval_val_occlusion_min_d_p05=get_float(eval_val_m, "occlusion_min_d_p05"),
                    eval_val_occlusion_min_d_p50=get_float(eval_val_m, "occlusion_min_d_p50"),
                    eval_val_occlusion_min_d_p95=get_float(eval_val_m, "occlusion_min_d_p95"),
                    eval_test_total=eval_total(eval_test_m, lambda_margin=lambda_margin, lambda_occlusion=lambda_occlusion),
                    eval_test_chamfer=get_float(eval_test_m, "chamfer"),
                    eval_test_margin=get_float(eval_test_m, "margin"),
                    eval_test_occlusion_pen_mean=get_float(eval_test_m, "occlusion_pen_mean"),
                    eval_test_occlusion_contact_ratio=get_float(eval_test_m, "occlusion_contact_ratio"),
                    eval_test_occlusion_min_d_p05=get_float(eval_test_m, "occlusion_min_d_p05"),
                    eval_test_occlusion_min_d_p50=get_float(eval_test_m, "occlusion_min_d_p50"),
                    eval_test_occlusion_min_d_p95=get_float(eval_test_m, "occlusion_min_d_p95"),
                )
            )
        except Exception:
            continue

    def sort_key(r: Row) -> tuple[float, float]:
        primary = r.eval_val_total if r.eval_val_total is not None else r.best_val_total
        secondary = r.eval_test_total if r.eval_test_total is not None else float("inf")
        return (float(primary), float(secondary))

    rows.sort(key=sort_key)

    csv_path = Path(str(out_prefix) + ".csv")
    md_path = Path(str(out_prefix) + ".md")

    header = [
        "exp",
        "cut_mode",
        "occlusion_mode",
        "n_points",
        "latent_dim",
        "cut_q_min",
        "cut_q_max",
        "lambda_margin",
        "lambda_occlusion",
        "occlusion_clearance",
        "occlusion_points",
        "occlusion_max_center_dist_mult",
        "best_epoch",
        "best_val_total",
        "eval_val_total",
        "eval_val_chamfer",
        "eval_val_margin",
        "eval_val_occlusion_pen_mean",
        "eval_val_occlusion_contact_ratio",
        "eval_val_occlusion_min_d_p05",
        "eval_val_occlusion_min_d_p50",
        "eval_val_occlusion_min_d_p95",
        "eval_test_total",
        "eval_test_chamfer",
        "eval_test_margin",
        "eval_test_occlusion_pen_mean",
        "eval_test_occlusion_contact_ratio",
        "eval_test_occlusion_min_d_p05",
        "eval_test_occlusion_min_d_p50",
        "eval_test_occlusion_min_d_p95",
        "out_dir",
    ]
    lines = [",".join(header)]
    for r in rows:
        lines.append(
            ",".join(
                [
                    r.exp,
                    r.cut_mode,
                    r.occlusion_mode,
                    str(r.n_points),
                    str(r.latent_dim),
                    fmt(r.cut_q_min),
                    fmt(r.cut_q_max),
                    fmt(r.lambda_margin),
                    fmt(r.lambda_occlusion),
                    fmt(r.occlusion_clearance),
                    str(r.occlusion_points),
                    fmt(r.occlusion_max_center_dist_mult),
                    str(r.best_epoch),
                    fmt(r.best_val_total),
                    fmt(r.eval_val_total),
                    fmt(r.eval_val_chamfer),
                    fmt(r.eval_val_margin),
                    fmt(r.eval_val_occlusion_pen_mean),
                    fmt(r.eval_val_occlusion_contact_ratio),
                    fmt(r.eval_val_occlusion_min_d_p05),
                    fmt(r.eval_val_occlusion_min_d_p50),
                    fmt(r.eval_val_occlusion_min_d_p95),
                    fmt(r.eval_test_total),
                    fmt(r.eval_test_chamfer),
                    fmt(r.eval_test_margin),
                    fmt(r.eval_test_occlusion_pen_mean),
                    fmt(r.eval_test_occlusion_contact_ratio),
                    fmt(r.eval_test_occlusion_min_d_p05),
                    fmt(r.eval_test_occlusion_min_d_p50),
                    fmt(r.eval_test_occlusion_min_d_p95),
                    r.out_dir,
                ]
            )
        )
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    md_lines: list[str] = []
    md_lines.append(f"# teeth3ds_prep2target_constraints runs summary ({runs_dir.name})")
    md_lines.append("")
    md_lines.append(f"- total_runs: {len(rows)}")
    md_lines.append(f"- generated_at: {datetime.now(timezone.utc).isoformat().replace('+00:00','Z')}")
    md_lines.append("")
    md_lines.append("| eval_val_total | eval_test_total | chamfer(val) | margin(val) | occ_contact(val) | min_d_p05(val) | λ_margin | λ_occ | occ_mode | cut_mode | exp |")
    md_lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|")
    for r in rows[: max(1, int(args.limit))]:
        md_lines.append(
            "| "
            + " | ".join(
                [
                    fmt(r.eval_val_total),
                    fmt(r.eval_test_total),
                    fmt(r.eval_val_chamfer),
                    fmt(r.eval_val_margin),
                    fmt(r.eval_val_occlusion_contact_ratio),
                    fmt(r.eval_val_occlusion_min_d_p05),
                    fmt(r.lambda_margin),
                    fmt(r.lambda_occlusion),
                    r.occlusion_mode,
                    r.cut_mode,
                    r.exp,
                ]
            )
            + " |"
        )
    md_lines.append("")
    md_lines.append("## Notes")
    md_lines.append("- Per-run files: config.json / metrics.json / eval_{val,test}.json / history.csv / previews/ .")
    md_lines.append("- eval_* metrics are preferred for cross-run comparisons (fixed evaluation settings).")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[OK] wrote: {csv_path}")
    print(f"[OK] wrote: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
