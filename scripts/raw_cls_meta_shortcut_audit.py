#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def _quantile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    if q <= 0:
        return float(min(xs))
    if q >= 1:
        return float(max(xs))
    ys = sorted(xs)
    pos = (len(ys) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ys[lo])
    w = pos - lo
    return float((1 - w) * ys[lo] + w * ys[hi])


def _mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    if len(xs) == 1:
        return float(xs[0]), 0.0
    return float(statistics.mean(xs)), float(statistics.stdev(xs))


def auc_roc(scores: list[float], labels: list[int]) -> float:
    """Compute AUC (ROC) using rank statistic (Mann–Whitney U), handling ties."""
    if len(scores) != len(labels):
        raise ValueError("scores/labels length mismatch")
    n = len(scores)
    if n == 0:
        return 0.0
    n_pos = sum(1 for y in labels if int(y) == 1)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0

    # Sort by score (ascending). For ties, assign average rank.
    order = sorted(range(n), key=lambda i: float(scores[i]))
    ranks = [0.0] * n
    i = 0
    rank = 1
    while i < n:
        j = i
        s0 = float(scores[order[i]])
        while j < n and float(scores[order[j]]) == s0:
            j += 1
        avg_rank = (rank + (rank + (j - i) - 1)) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        rank += j - i
        i = j

    sum_pos_ranks = sum(ranks[i] for i in range(n) if int(labels[i]) == 1)
    u = sum_pos_ranks - (n_pos * (n_pos + 1)) / 2.0
    return float(u / (n_pos * n_neg))


@dataclass(frozen=True)
class Row:
    source: str
    split: str
    scale: float
    n_points_after_cap: float
    n_objects_used: float


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Audit whether simple meta features can discriminate sources (shortcut risk audit)."
    )
    ap.add_argument("--data-root", type=Path, default=Path("processed/raw_cls/v13_main4"))
    ap.add_argument("--pos-source", type=str, default="专家标注", help="Positive class for AUC (default: 专家标注).")
    ap.add_argument("--out", type=Path, default=Path("paper_tables/raw_cls_meta_shortcut_audit.md"))
    args = ap.parse_args()

    data_root = args.data_root.expanduser().resolve()
    index_path = data_root / "index.jsonl"
    if not index_path.is_file():
        raise SystemExit(f"Missing index.jsonl: {index_path}")

    pos_source = str(args.pos_source).strip()
    rows_raw = read_jsonl(index_path)
    rows: list[Row] = []
    for r in rows_raw:
        rows.append(
            Row(
                source=str(r.get("source") or ""),
                split=str(r.get("split") or ""),
                scale=float(r.get("scale") or 0.0),
                n_points_after_cap=float(r.get("n_points_after_cap") or 0.0),
                n_objects_used=float(r.get("n_objects_used") or 0.0),
            )
        )

    sources = sorted({r.source for r in rows if r.source})
    splits = ["train", "val", "test", "unknown"]

    def _by_source(xs: list[Row], fn) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {s: [] for s in sources}
        for rr in xs:
            if rr.source in out:
                out[rr.source].append(float(fn(rr)))
        return out

    def _emit_stats(title: str, values_by_source: dict[str, list[float]]) -> list[str]:
        lines: list[str] = []
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| source | n | mean±std | p05 | p50 | p95 |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for s in sources:
            xs = values_by_source.get(s, [])
            m, sd = _mean_std(xs)
            lines.append(
                "| "
                + " | ".join(
                    [
                        s or "(missing)",
                        str(len(xs)),
                        f"{m:.4g}±{sd:.4g}",
                        f"{_quantile(xs, 0.05):.4g}",
                        f"{_quantile(xs, 0.50):.4g}",
                        f"{_quantile(xs, 0.95):.4g}",
                    ]
                )
                + " |"
            )
        lines.append("")
        return lines

    lines: list[str] = []
    lines.append("# raw_cls meta shortcut audit (source separability)")
    lines.append("")
    lines.append(f"- data_root: `{data_root}`")
    lines.append(f"- rows: {len(rows)}")
    lines.append(f"- sources: {sources}")
    lines.append(f"- pos_source (AUC=1): `{pos_source}`")
    lines.append("")

    # Overall stats.
    lines += _emit_stats("scale", _by_source(rows, lambda r: r.scale))
    lines += _emit_stats("log1p(n_points_after_cap)", _by_source(rows, lambda r: math.log1p(r.n_points_after_cap)))
    lines += _emit_stats("n_objects_used", _by_source(rows, lambda r: r.n_objects_used))

    # AUCs (overall + per split).
    def _auc_report(subset: list[Row], *, title: str) -> None:
        scores_scale = [r.scale for r in subset]
        scores_lpts = [math.log1p(r.n_points_after_cap) for r in subset]
        scores_ou = [r.n_objects_used for r in subset]
        labels = [1 if r.source == pos_source else 0 for r in subset]
        lines.append(f"## AUC (source separability) — {title}")
        lines.append("")
        lines.append("| feature | AUC |")
        lines.append("|---|---:|")
        lines.append(f"| scale | {auc_roc(scores_scale, labels):.4f} |")
        lines.append(f"| log1p(n_points_after_cap) | {auc_roc(scores_lpts, labels):.4f} |")
        lines.append(f"| n_objects_used | {auc_roc(scores_ou, labels):.4f} |")
        lines.append("")

    _auc_report(rows, title="all splits")
    for sp in splits:
        subset = [r for r in rows if r.split == sp]
        if subset:
            _auc_report(subset, title=f"split={sp}")

    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
