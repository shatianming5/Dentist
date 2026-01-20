#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    if len(xs) == 1:
        return float(xs[0]), 0.0
    return float(statistics.mean(xs)), float(statistics.stdev(xs))


def fmt_ms(m: float, s: float) -> str:
    return f"{m:.4f}±{s:.4f}"


def join_sources(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, list):
        return ",".join(str(x) for x in v if str(x).strip())
    return str(v)


@dataclass(frozen=True)
class Run:
    exp: str
    out_dir: str
    data_tag: str
    model: str
    n_points: int
    seed: int
    balanced: bool
    label_smoothing: float
    extra_features: str
    tta: int
    source_train: str
    source_test: str
    source_val_ratio: float
    test_acc: float
    test_f1_present: float
    test_f1_all: float
    test_bal_acc: float
    test_ece: float
    test_nll: float
    test_brier: float


def main() -> int:
    ap = argparse.ArgumentParser(description="Make a journal-style summary table for raw_cls domain-shift runs (mean±std over seeds).")
    ap.add_argument("--runs-dir", type=Path, default=Path("runs/raw_cls_domain_shift"))
    ap.add_argument("--out", type=Path, default=Path("paper_tables/raw_cls_domain_shift_table.md"))
    ap.add_argument("--data-tag", type=str, default="", help="Only include runs whose dataset tag matches (e.g. v13_main4).")
    ap.add_argument("--tta", type=int, default=8, help="Only include runs with this tta (default: 8). Use -1 for all.")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    runs_dir = args.runs_dir.resolve()
    if not runs_dir.exists():
        raise SystemExit(f"Missing runs dir: {runs_dir}")

    runs: list[Run] = []
    for d in sorted([p for p in runs_dir.iterdir() if p.is_dir()]):
        cfg_path = d / "config.json"
        met_path = d / "metrics.json"
        if not cfg_path.exists() or not met_path.exists():
            continue
        try:
            cfg = read_json(cfg_path)
            met = read_json(met_path)
            test = met.get("test") or {}
            cal = met.get("test_calibration") or {}
            data_root = str(cfg.get("data_root") or "")
            data_tag = Path(data_root).name if data_root else ""
            if str(args.data_tag).strip() and data_tag != str(args.data_tag).strip():
                continue
            tta = int(cfg.get("tta") or 0)
            if int(args.tta) >= 0 and int(tta) != int(args.tta):
                continue
            runs.append(
                Run(
                    exp=str(cfg.get("exp_name") or d.name),
                    out_dir=str(d),
                    data_tag=data_tag,
                    model=str(cfg.get("model") or "unknown"),
                    n_points=int(cfg.get("n_points") or 0),
                    seed=int(cfg.get("seed") or 0),
                    balanced=bool(cfg.get("balanced_sampler") or False),
                    label_smoothing=float(cfg.get("label_smoothing") or 0.0),
                    extra_features=",".join(cfg.get("extra_features") or []),
                    tta=tta,
                    source_train=join_sources(cfg.get("source_train") or ""),
                    source_test=join_sources(cfg.get("source_test") or ""),
                    source_val_ratio=float(cfg.get("source_val_ratio") or 0.0),
                    test_acc=float(test.get("accuracy") or 0.0),
                    test_f1_present=float(test.get("macro_f1_present") or 0.0),
                    test_f1_all=float(test.get("macro_f1_all") or 0.0),
                    test_bal_acc=float(test.get("balanced_accuracy_present") or 0.0),
                    test_ece=float(cal.get("ece") or 0.0),
                    test_nll=float(cal.get("nll") or 0.0),
                    test_brier=float(cal.get("brier") or 0.0),
                )
            )
        except Exception:
            continue

    groups: dict[tuple[Any, ...], list[Run]] = {}
    for r in runs:
        key = (
            r.data_tag,
            r.model.lower(),
            r.n_points,
            bool(r.balanced),
            round(float(r.label_smoothing), 6),
            r.extra_features,
            int(r.tta),
            r.source_train,
            r.source_test,
            round(float(r.source_val_ratio), 6),
        )
        groups.setdefault(key, []).append(r)

    rows: list[tuple[float, float, int, tuple[Any, ...]]] = []
    for key, items in groups.items():
        f1s = [float(x.test_f1_present) for x in items]
        accs = [float(x.test_acc) for x in items]
        m_f1, _ = mean_std(f1s)
        m_acc, _ = mean_std(accs)
        rows.append((m_f1, m_acc, len(items), key))
    rows.sort(key=lambda t: (-t[0], -t[1], -t[2]))

    if int(args.limit) > 0:
        rows = rows[: int(args.limit)]

    out_lines: list[str] = []
    out_lines.append("# raw_cls domain-shift baselines (paper summary)")
    out_lines.append("")
    out_lines.append(f"- generated_at: {utc_now_iso()}")
    out_lines.append(f"- runs_dir: `{runs_dir}`")
    out_lines.append(f"- total_runs: {len(runs)}")
    out_lines.append(f"- total_groups: {len(groups)}")
    out_lines.append("")
    out_lines.append(
        "| test_macro_f1_present (mean±std) | test_macro_f1_all (mean±std) | bal_acc_present (mean±std) | test_acc (mean±std) | ece (mean±std) | n | model | n_points | tta | train_source | test_source | val_ratio | dataset |"
    )
    out_lines.append("|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---|---:|---|")

    for _m_f1, _m_acc, n, key in rows:
        (
            data_tag,
            model,
            n_points,
            balanced,
            label_smoothing,
            extra_features,
            tta,
            source_train,
            source_test,
            source_val_ratio,
        ) = key
        items = groups[key]
        f1p_m, f1p_s = mean_std([float(x.test_f1_present) for x in items])
        f1a_m, f1a_s = mean_std([float(x.test_f1_all) for x in items])
        bal_m, bal_s = mean_std([float(x.test_bal_acc) for x in items])
        acc_m, acc_s = mean_std([float(x.test_acc) for x in items])
        ece_m, ece_s = mean_std([float(x.test_ece) for x in items])

        out_lines.append(
            "| "
            + " | ".join(
                [
                    fmt_ms(f1p_m, f1p_s),
                    fmt_ms(f1a_m, f1a_s),
                    fmt_ms(bal_m, bal_s),
                    fmt_ms(acc_m, acc_s),
                    fmt_ms(ece_m, ece_s),
                    str(int(n)),
                    str(model),
                    str(int(n_points)),
                    str(int(tta)),
                    str(source_train or "(missing)"),
                    str(source_test or "(missing)"),
                    f"{float(source_val_ratio):.3f}",
                    str(data_tag),
                ]
            )
            + " |"
        )

    out_lines.append("")
    out_lines.append("Notes:")
    out_lines.append("- `n` counts seeds (each run is one seed under a fixed train_source→test_source split).")
    out_lines.append("- Full per-run details live under each run dir (config.json/metrics.json/preds_test.jsonl).")
    out_lines.append("")

    out_path = args.out.resolve() if args.out.is_absolute() else (Path.cwd() / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
