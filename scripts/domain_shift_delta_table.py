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


def parse_domain_tag(tag: str) -> tuple[str, str] | None:
    s = str(tag or "")
    if not s.startswith("A2B_") or "_to_" not in s:
        return None
    rest = s[len("A2B_") :]
    train, test = rest.split("_to_", 1)
    train = train.strip()
    test = test.strip()
    if not train or not test:
        return None
    return train, test


@dataclass(frozen=True)
class Run:
    train: str
    test: str
    exp: str
    model: str
    seed: int
    acc: float
    macro_f1: float
    ece: float


def load_run(root: Path, metrics_path: Path) -> Run | None:
    rel = metrics_path.relative_to(root)
    parts = rel.parts
    if len(parts) < 5:
        return None
    domain_tag, exp, model, fold_part, seed_part = parts[:5]
    _ = fold_part
    if not seed_part.startswith("seed="):
        return None
    try:
        seed = int(seed_part.split("=", 1)[1])
    except Exception:
        return None

    tt = parse_domain_tag(str(domain_tag))
    if tt is None:
        return None
    train, test = tt

    d = read_json(metrics_path)
    test_m = d.get("test") or {}
    cal = d.get("test_calibration") or {}
    try:
        acc = float(test_m.get("accuracy") or 0.0)
        macro_f1 = float(test_m.get("macro_f1_present") or 0.0)
        ece = float(cal.get("ece") or 0.0)
    except Exception:
        return None

    return Run(train=train, test=test, exp=str(exp), model=str(model), seed=int(seed), acc=acc, macro_f1=macro_f1, ece=ece)


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute domain-shift deltas vs in-domain baselines (README DoD 7.8).")
    ap.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs/domain_shift/v13_main4"),
        help="Root that contains A2B_* directories.",
    )
    ap.add_argument("--out", type=Path, default=Path("paper_tables/domain_shift_delta.md"))
    args = ap.parse_args()

    root = args.runs_root.expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Missing runs root: {root}")

    runs: list[Run] = []
    for mp in sorted(root.rglob("metrics.json")):
        r = load_run(root, mp)
        if r is not None:
            runs.append(r)

    # Baseline-only (as required by README 7.8 delta definition).
    runs = [r for r in runs if r.exp == "baseline"]

    # Index in-domain baselines: B->B for each model+seed.
    in_domain: dict[tuple[str, str, int], Run] = {}
    cross: dict[tuple[str, str, str, int], Run] = {}
    for r in runs:
        if r.train == r.test:
            in_domain[(r.test, r.model, r.seed)] = r
        else:
            cross[(r.train, r.test, r.model, r.seed)] = r

    # Collect cross-domain directions.
    directions = sorted({(r.train, r.test) for r in runs if r.train != r.test})
    models = sorted({r.model for r in runs})

    out_lines: list[str] = []
    out_lines.append("# Domain-shift delta report (baseline)")
    out_lines.append("")
    out_lines.append(f"- generated_at: {utc_now_iso()}")
    out_lines.append(f"- runs_root: `{root}`")
    out_lines.append(f"- baseline_runs: {len(runs)}")
    out_lines.append("")
    out_lines.append(
        "| direction | model | n | acc_in (mean±std) | acc_cross (mean±std) | Δacc (cross-in) | macro_f1_in (mean±std) | macro_f1_cross (mean±std) | Δmacro_f1 | ece_in (mean±std) | ece_cross (mean±std) | Δece |"
    )
    out_lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for train, test in directions:
        for model in models:
            cross_by_seed = {seed: r for (a, b, m, seed), r in cross.items() if a == train and b == test and m == model}
            in_by_seed = {seed: r for (dom, m, seed), r in in_domain.items() if dom == test and m == model}
            common_seeds = sorted(set(cross_by_seed) & set(in_by_seed))
            if not common_seeds:
                continue

            acc_in = [in_by_seed[s].acc for s in common_seeds]
            acc_cross = [cross_by_seed[s].acc for s in common_seeds]
            acc_delta = [cross_by_seed[s].acc - in_by_seed[s].acc for s in common_seeds]

            f1_in = [in_by_seed[s].macro_f1 for s in common_seeds]
            f1_cross = [cross_by_seed[s].macro_f1 for s in common_seeds]
            f1_delta = [cross_by_seed[s].macro_f1 - in_by_seed[s].macro_f1 for s in common_seeds]

            ece_in = [in_by_seed[s].ece for s in common_seeds]
            ece_cross = [cross_by_seed[s].ece for s in common_seeds]
            ece_delta = [cross_by_seed[s].ece - in_by_seed[s].ece for s in common_seeds]

            acc_in_m, acc_in_s = mean_std(acc_in)
            acc_cross_m, acc_cross_s = mean_std(acc_cross)
            acc_d_m, acc_d_s = mean_std(acc_delta)

            f1_in_m, f1_in_s = mean_std(f1_in)
            f1_cross_m, f1_cross_s = mean_std(f1_cross)
            f1_d_m, f1_d_s = mean_std(f1_delta)

            ece_in_m, ece_in_s = mean_std(ece_in)
            ece_cross_m, ece_cross_s = mean_std(ece_cross)
            ece_d_m, ece_d_s = mean_std(ece_delta)

            out_lines.append(
                "| "
                + " | ".join(
                    [
                        f"{train}→{test}",
                        str(model),
                        str(len(common_seeds)),
                        fmt_ms(acc_in_m, acc_in_s),
                        fmt_ms(acc_cross_m, acc_cross_s),
                        fmt_ms(acc_d_m, acc_d_s),
                        fmt_ms(f1_in_m, f1_in_s),
                        fmt_ms(f1_cross_m, f1_cross_s),
                        fmt_ms(f1_d_m, f1_d_s),
                        fmt_ms(ece_in_m, ece_in_s),
                        fmt_ms(ece_cross_m, ece_cross_s),
                        fmt_ms(ece_d_m, ece_d_s),
                    ]
                )
                + " |"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

