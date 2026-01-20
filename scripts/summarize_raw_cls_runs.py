#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.6f}"
    return str(x)


@dataclass(frozen=True)
class Row:
    exp: str
    model: str
    n_points: int
    seed: int
    kfold_k: int
    test_fold: int
    val_fold: int
    val_acc: float
    val_macro_f1: float
    test_acc: float
    test_macro_f1: float
    out_dir: str


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize runs/raw_cls_baseline into CSV/MD.")
    ap.add_argument("--runs-dir", type=Path, default=Path("runs/raw_cls_baseline"))
    ap.add_argument("--out-prefix", type=Path, default=None)
    args = ap.parse_args()

    runs_dir = args.runs_dir.resolve()
    if not runs_dir.exists():
        raise SystemExit(f"Missing runs dir: {runs_dir}")

    if args.out_prefix is None:
        out_prefix = runs_dir / f"summary_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    else:
        out_prefix = args.out_prefix
        if not out_prefix.is_absolute():
            out_prefix = runs_dir / out_prefix
    out_prefix = out_prefix.resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    rows: list[Row] = []
    for d in sorted([p for p in runs_dir.iterdir() if p.is_dir()]):
        cfg_path = d / "config.json"
        met_path = d / "metrics.json"
        if not cfg_path.exists() or not met_path.exists():
            continue
        try:
            cfg = read_json(cfg_path)
            met = read_json(met_path)
            val = met.get("val") or {}
            test = met.get("test") or {}
            rows.append(
                Row(
                    exp=str(cfg.get("exp_name") or d.name),
                    model=str(cfg.get("model") or "unknown"),
                    n_points=int(cfg.get("n_points") or 0),
                    seed=int(cfg.get("seed") or 0),
                    kfold_k=int(cfg.get("kfold_k") or 0),
                    test_fold=int(cfg.get("kfold_test_fold") or -1),
                    val_fold=int(cfg.get("kfold_val_fold") or -1),
                    val_acc=float(val.get("accuracy") or 0.0),
                    val_macro_f1=float(val.get("macro_f1_present") or 0.0),
                    test_acc=float(test.get("accuracy") or 0.0),
                    test_macro_f1=float(test.get("macro_f1_present") or 0.0),
                    out_dir=str(d),
                )
            )
        except Exception:
            continue

    rows.sort(key=lambda r: (-r.test_macro_f1, -r.test_acc))

    csv_path = Path(str(out_prefix) + ".csv")
    md_path = Path(str(out_prefix) + ".md")

    header = [
        "exp",
        "model",
        "n_points",
        "seed",
        "kfold_k",
        "test_fold",
        "val_fold",
        "val_acc",
        "val_macro_f1",
        "test_acc",
        "test_macro_f1",
        "out_dir",
    ]
    lines = [",".join(header)]
    for r in rows:
        lines.append(
            ",".join(
                [
                    r.exp,
                    r.model,
                    str(r.n_points),
                    str(r.seed),
                    str(r.kfold_k),
                    str(r.test_fold),
                    str(r.val_fold),
                    fmt(r.val_acc),
                    fmt(r.val_macro_f1),
                    fmt(r.test_acc),
                    fmt(r.test_macro_f1),
                    r.out_dir,
                ]
            )
        )
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    md_lines: list[str] = []
    md_lines.append(f"# raw_cls runs summary ({runs_dir.name})")
    md_lines.append("")
    md_lines.append(f"- total_runs: {len(rows)}")
    md_lines.append(f"- generated_at: {datetime.now(timezone.utc).isoformat().replace('+00:00','Z')}")
    md_lines.append("")
    md_lines.append("| test_macro_f1 | test_acc | val_macro_f1 | val_acc | model | n_points | seed | kfold | exp |")
    md_lines.append("|---:|---:|---:|---:|---|---:|---:|---:|---|")
    for r in rows[:50]:
        md_lines.append(
            "| "
            + " | ".join(
                [
                    fmt(r.test_macro_f1),
                    fmt(r.test_acc),
                    fmt(r.val_macro_f1),
                    fmt(r.val_acc),
                    r.model,
                    str(r.n_points),
                    str(r.seed),
                    str(r.kfold_k),
                    r.exp,
                ]
            )
            + " |"
        )
    md_lines.append("")
    md_lines.append("## Notes")
    md_lines.append("- Full details per run are in each run dir: config.json / metrics.json / model_best.pt / preds_*.jsonl / errors_*.csv .")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[OK] wrote: {csv_path}")
    print(f"[OK] wrote: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
