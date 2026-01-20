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


def parse_time_ts(iso: str | None, *, fallback_path: Path) -> float:
    s = str(iso or "").strip()
    if s:
        try:
            # Accept trailing Z (UTC).
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            return float(dt.timestamp())
        except Exception:
            pass
    try:
        return float(fallback_path.stat().st_mtime)
    except Exception:
        return 0.0


@dataclass(frozen=True)
class Run:
    exp: str
    out_dir: str
    generated_at: str
    sort_ts: float
    data_tag: str
    model: str
    n_points: int
    seed: int
    kfold_k: int
    test_fold: int
    balanced: bool
    label_smoothing: float
    extra_features: str
    tta: int
    val_acc: float
    val_f1: float
    test_acc: float
    test_f1: float


def main() -> int:
    ap = argparse.ArgumentParser(description="Make a journal-style summary table for raw_cls runs (mean±std).")
    ap.add_argument("--runs-dir", type=Path, default=Path("runs/raw_cls_baseline"))
    ap.add_argument("--out", type=Path, default=Path("paper_tables/raw_cls_table.md"))
    ap.add_argument("--data-tag", type=str, default="", help="Only include runs whose dataset tag matches (e.g. v13_main4).")
    ap.add_argument("--kfold-only", action="store_true", help="Only include k-fold runs (kfold_k>0).")
    ap.add_argument("--limit", type=int, default=0, help="Only include first N runs after sorting (0=all).")
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
            val = met.get("val") or {}
            test = met.get("test") or {}
            data_root = str(cfg.get("data_root") or "")
            data_tag = Path(data_root).name if data_root else ""
            if str(args.data_tag).strip() and data_tag != str(args.data_tag).strip():
                continue
            gen_at = str(cfg.get("generated_at") or "")
            ts = parse_time_ts(gen_at, fallback_path=cfg_path)
            runs.append(
                Run(
                    exp=str(cfg.get("exp_name") or d.name),
                    out_dir=str(d),
                    generated_at=gen_at,
                    sort_ts=float(ts),
                    data_tag=data_tag,
                    model=str(cfg.get("model") or "unknown"),
                    n_points=int(cfg.get("n_points") or 0),
                    seed=int(cfg.get("seed") or 0),
                    kfold_k=int(cfg.get("kfold_k") or 0),
                    test_fold=int(cfg.get("kfold_test_fold") or -1),
                    balanced=bool(cfg.get("balanced_sampler") or False),
                    label_smoothing=float(cfg.get("label_smoothing") or 0.0),
                    extra_features=",".join(cfg.get("extra_features") or []),
                    tta=int(cfg.get("tta") or 0),
                    val_acc=float(val.get("accuracy") or 0.0),
                    val_f1=float(val.get("macro_f1_present") or 0.0),
                    test_acc=float(test.get("accuracy") or 0.0),
                    test_f1=float(test.get("macro_f1_present") or 0.0),
                )
            )
        except Exception:
            continue

    # Group by "paper configuration".
    # For k-fold runs, deduplicate by (seed, test_fold) and keep the latest run
    # to avoid table contamination when old exp_name/tag variants coexist.
    groups: dict[tuple[Any, ...], dict[tuple[int, int], Run]] = {}
    dedup_dropped: list[tuple[Run, Run]] = []  # (kept, dropped)
    for r in runs:
        if bool(args.kfold_only) and int(r.kfold_k) <= 0:
            continue
        key = (
            r.data_tag,
            r.model,
            r.n_points,
            r.kfold_k,
            r.balanced,
            round(float(r.label_smoothing), 6),
            r.extra_features,
            int(r.tta),
        )
        # Non-kfold runs: keep all (keyed by unique out_dir).
        if int(r.kfold_k) <= 0:
            subkey = (str(r.out_dir), 0)
            groups.setdefault(key, {})[subkey] = r
            continue

        # K-fold runs: dedup by seed+fold.
        subkey = (int(r.seed), int(r.test_fold))
        m = groups.setdefault(key, {})
        prev = m.get(subkey)
        if prev is None:
            m[subkey] = r
            continue
        # Prefer newer by generated_at (or mtime fallback); tie-break by out_dir.
        keep, drop = (r, prev) if (float(r.sort_ts), str(r.out_dir)) > (float(prev.sort_ts), str(prev.out_dir)) else (prev, r)
        if keep is not prev:
            m[subkey] = keep
        dedup_dropped.append((keep, drop))

    rows: list[tuple[float, float, int, tuple[Any, ...]]] = []
    for key, items in groups.items():
        items_list = list(items.values())
        f1s = [float(x.test_f1) for x in items_list]
        accs = [float(x.test_acc) for x in items_list]
        m_f1, _s_f1 = mean_std(f1s)
        m_acc, _s_acc = mean_std(accs)
        rows.append((m_f1, m_acc, len(items_list), key))
    rows.sort(key=lambda t: (-t[0], -t[1], -t[2]))

    if int(args.limit) > 0:
        rows = rows[: int(args.limit)]

    out_lines: list[str] = []
    out_lines.append("# raw_cls baselines (paper summary)")
    out_lines.append("")
    out_lines.append(f"- generated_at: {utc_now_iso()}")
    out_lines.append(f"- runs_dir: `{runs_dir}`")
    out_lines.append(f"- total_runs: {len(runs)}")
    out_lines.append(f"- total_groups: {len(groups)}")
    out_lines.append(f"- dedup_dropped: {len(dedup_dropped)} (k-fold duplicates)")
    out_lines.append("")
    out_lines.append("| test_macro_f1 (mean±std) | test_acc (mean±std) | n | model | n_points | kfold_k | balanced | label_smoothing | extra_features | tta | dataset |")
    out_lines.append("|---:|---:|---:|---|---:|---:|---|---:|---|---:|---|")
    for _m_f1, _m_acc, n, key in rows:
        data_tag, model, n_points, kfold_k, balanced, label_smoothing, extra_features, tta = key
        items_list = list(groups[key].values())
        f1s = [float(x.test_f1) for x in items_list]
        accs = [float(x.test_acc) for x in items_list]
        m_f1, s_f1 = mean_std(f1s)
        m_acc, s_acc = mean_std(accs)
        out_lines.append(
            "| "
            + " | ".join(
                [
                    fmt_ms(m_f1, s_f1),
                    fmt_ms(m_acc, s_acc),
                    str(n),
                    str(model),
                    str(int(n_points)),
                    str(int(kfold_k)),
                    "yes" if bool(balanced) else "no",
                    f"{float(label_smoothing):.3f}",
                    str(extra_features or "(none)"),
                    str(int(tta)),
                    str(data_tag),
                ]
            )
            + " |"
        )
    out_lines.append("")
    out_lines.append("Notes:")
    out_lines.append("- `n` counts runs; for k-fold it is typically `seeds × folds`.")
    if dedup_dropped:
        out_lines.append("- k-fold dedup: keep the latest run per (seed,test_fold); older duplicates are dropped.")
        # Show up to 5 duplicates for auditability.
        for keep, drop in dedup_dropped[:5]:
            out_lines.append(
                f"  - dropped duplicate: keep=`{Path(keep.out_dir).name}` (t={keep.generated_at or 'mtime'}) "
                f"drop=`{Path(drop.out_dir).name}` (t={drop.generated_at or 'mtime'})"
            )
    out_lines.append("- For journal reporting, consider adding bootstrap CIs from `preds_test.jsonl` (see `scripts/raw_cls_bootstrap_ci.py`).")
    out_lines.append("")

    out_path = args.out.resolve() if args.out.is_absolute() else (Path.cwd() / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
