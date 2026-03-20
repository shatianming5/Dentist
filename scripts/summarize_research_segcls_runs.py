#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUN_RE = re.compile(r"_fold(?P<fold>\d+)_seed(?P<seed>\d+)")
PAIRWISE_CLS_VARIANTS = [
    ("gtseg_pointnet", "all_pointnet"),
    ("joint_pointnet", "all_pointnet"),
    ("joint_pointnet", "gtseg_pointnet"),
]
CLS_METRICS = [
    ("test_accuracy", "acc"),
    ("test_macro_f1_present", "macro_f1"),
    ("test_balanced_accuracy_present", "bal_acc"),
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_run_identity(name: str) -> dict[str, Any]:
    match = RUN_RE.search(name)
    if not match:
        return {"fold": None, "seed": None, "pair_key": name}
    fold = int(match.group("fold"))
    seed = int(match.group("seed"))
    return {
        "fold": fold,
        "seed": seed,
        "pair_key": f"fold{fold}_seed{seed}",
    }


def fmt_mean_std(vals: list[float]) -> str:
    if not vals:
        return ""
    if len(vals) == 1:
        return f"{vals[0]:.4f}"
    return f"{statistics.mean(vals):.4f}±{statistics.pstdev(vals):.4f}"


def fmt_ci(mean_val: float, ci: tuple[float, float] | None) -> str:
    if ci is None:
        return f"{mean_val:.4f}"
    return f"{mean_val:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]"


def bootstrap_mean_ci(
    vals: list[float],
    *,
    n_boot: int = 10000,
    seed: int = 1337,
) -> tuple[float, float] | None:
    if not vals:
        return None
    if len(vals) == 1:
        return (float(vals[0]), float(vals[0]))
    rng = random.Random(int(seed))
    n = len(vals)
    means: list[float] = []
    for _ in range(int(n_boot)):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        means.append(float(statistics.mean(sample)))
    means.sort()
    lo = means[int(0.025 * (len(means) - 1))]
    hi = means[int(0.975 * (len(means) - 1))]
    return float(lo), float(hi)


def collect_variant_runs(variant_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cls_rows: list[dict[str, Any]] = []
    seg_rows: list[dict[str, Any]] = []
    for d in sorted([p for p in variant_dir.iterdir() if p.is_dir()]):
        ident = parse_run_identity(d.name)
        met = d / "metrics.json"
        res = d / "results.json"
        if met.exists():
            obj = read_json(met)
            test = obj.get("test") or {}
            row = {
                "run": d.name,
                **ident,
                "test_accuracy": float(test.get("accuracy") or 0.0),
                "test_macro_f1_present": float(test.get("macro_f1_present") or 0.0),
                "test_balanced_accuracy_present": float(test.get("balanced_accuracy_present") or 0.0),
                "test_ece": float((obj.get("test_calibration") or {}).get("ece") or 0.0),
            }
            if "test_seg" in obj:
                row["test_seg_mean_iou"] = float((obj.get("test_seg") or {}).get("mean_iou") or 0.0)
            cls_rows.append(row)
        elif res.exists():
            obj = read_json(res)
            tm = obj.get("test_metrics") or {}
            seg_rows.append(
                {
                    "run": d.name,
                    **ident,
                    "test_accuracy": float(tm.get("accuracy") or 0.0),
                    "test_mean_iou": float(tm.get("mean_iou") or 0.0),
                }
            )
    return cls_rows, seg_rows


def aligned_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(r["pair_key"]): r for r in rows}


def pairwise_compare(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    *,
    metric: str,
) -> dict[str, Any]:
    by_key_a = aligned_rows(rows_a)
    by_key_b = aligned_rows(rows_b)
    keys = sorted(set(by_key_a) & set(by_key_b))
    deltas = [float(by_key_a[k][metric]) - float(by_key_b[k][metric]) for k in keys]
    wins = sum(1 for x in deltas if x > 1e-12)
    losses = sum(1 for x in deltas if x < -1e-12)
    ties = len(deltas) - wins - losses
    mean_delta = float(statistics.mean(deltas)) if deltas else 0.0
    return {
        "n_pairs": len(keys),
        "keys": keys,
        "mean_delta": mean_delta,
        "ci95": bootstrap_mean_ci(deltas, seed=1337 + len(metric) + len(keys)),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "deltas": deltas,
    }


def fold_means(rows: list[dict[str, Any]], *, metric: str) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        fold = row.get("fold")
        if fold is None:
            continue
        grouped[int(fold)].append(float(row[metric]))
    return {fold: float(statistics.mean(vals)) for fold, vals in sorted(grouped.items()) if vals}


def top_runs(rows: list[dict[str, Any]], *, metric: str, k: int = 3) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: float(row.get(metric) or 0.0), reverse=True)
    out: list[dict[str, Any]] = []
    for row in ranked[: int(k)]:
        item = {
            "run": str(row["run"]),
            "fold": row.get("fold"),
            "seed": row.get("seed"),
            metric: float(row.get(metric) or 0.0),
        }
        if "test_accuracy" in row:
            item["test_accuracy"] = float(row.get("test_accuracy") or 0.0)
        if "test_seg_mean_iou" in row:
            item["test_seg_mean_iou"] = float(row.get("test_seg_mean_iou") or 0.0)
        out.append(item)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize and analyze full research seg+cls runs.")
    ap.add_argument("--root", type=Path, default=Path("runs/research_segcls_full"))
    ap.add_argument("--out-prefix", type=Path, default=Path("paper_tables/research_segcls_full_summary"))
    args = ap.parse_args()

    root = args.root.resolve()
    out_prefix = args.out_prefix.resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    summary_json: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "root": str(root),
        "variants": {},
        "pairwise": {},
        "per_fold": {},
        "best_runs": {},
    }
    md_lines = [
        f"# research seg+cls full summary ({root.name})",
        "",
        f"- generated_at: {summary_json['generated_at']}",
        "",
        "| variant | n | test_acc | test_macro_f1 | test_bal_acc | test_ece | test_seg_miou |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    variant_cls_rows: dict[str, list[dict[str, Any]]] = {}
    variant_seg_rows: dict[str, list[dict[str, Any]]] = {}

    for variant_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        cls_rows, seg_rows = collect_variant_runs(variant_dir)
        variant_name = variant_dir.name
        variant_cls_rows[variant_name] = cls_rows
        variant_seg_rows[variant_name] = seg_rows
        metrics = {
            "n": len(cls_rows) if cls_rows else len(seg_rows),
            "runs": cls_rows if cls_rows else seg_rows,
            "test_accuracy": [float(r["test_accuracy"]) for r in cls_rows if "test_accuracy" in r],
            "test_macro_f1_present": [float(r["test_macro_f1_present"]) for r in cls_rows if "test_macro_f1_present" in r],
            "test_balanced_accuracy_present": [float(r["test_balanced_accuracy_present"]) for r in cls_rows if "test_balanced_accuracy_present" in r],
            "test_ece": [float(r["test_ece"]) for r in cls_rows if "test_ece" in r],
            "test_seg_mean_iou": [float(r["test_seg_mean_iou"]) for r in cls_rows if "test_seg_mean_iou" in r],
            "seg_only_test_accuracy": [float(r["test_accuracy"]) for r in seg_rows],
            "seg_only_test_mean_iou": [float(r["test_mean_iou"]) for r in seg_rows],
        }
        summary_json["variants"][variant_name] = metrics

        n = metrics["n"]
        acc = fmt_mean_std(metrics.get("test_accuracy") or metrics.get("seg_only_test_accuracy") or [])
        f1 = fmt_mean_std(metrics.get("test_macro_f1_present") or [])
        bal = fmt_mean_std(metrics.get("test_balanced_accuracy_present") or [])
        ece = fmt_mean_std(metrics.get("test_ece") or [])
        miou = fmt_mean_std(metrics.get("test_seg_mean_iou") or metrics.get("seg_only_test_mean_iou") or [])
        md_lines.append(f"| {variant_name} | {n} | {acc} | {f1} | {bal} | {ece} | {miou} |")

    md_lines.extend(["", "## Pairwise Comparisons", ""])
    md_lines.extend(
        [
            "| comparison | n_pairs | delta_acc | delta_macro_f1 | delta_bal_acc | wins/ties/losses |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for a_name, b_name in PAIRWISE_CLS_VARIANTS:
        rows_a = variant_cls_rows.get(a_name) or []
        rows_b = variant_cls_rows.get(b_name) or []
        if not rows_a or not rows_b:
            continue
        pair_key = f"{a_name}__minus__{b_name}"
        pair_obj: dict[str, Any] = {}
        display: dict[str, str] = {}
        n_pairs = 0
        wins = ties = losses = 0
        for metric, short_name in CLS_METRICS:
            comp = pairwise_compare(rows_a, rows_b, metric=metric)
            pair_obj[short_name] = comp
            n_pairs = int(comp["n_pairs"])
            wins, ties, losses = int(comp["wins"]), int(comp["ties"]), int(comp["losses"])
            display[short_name] = fmt_ci(float(comp["mean_delta"]), comp["ci95"])
        summary_json["pairwise"][pair_key] = pair_obj
        md_lines.append(
            f"| {a_name} - {b_name} | {n_pairs} | {display['acc']} | {display['macro_f1']} | {display['bal_acc']} | {wins}/{ties}/{losses} |"
        )

    md_lines.extend(["", "## Per-Fold Means", ""])
    fold_ids = sorted(
        {
            int(row["fold"])
            for rows in variant_cls_rows.values()
            for row in rows
            if row.get("fold") is not None
        }
    )
    if fold_ids:
        header = "| fold | all_pointnet macro-F1 | gtseg_pointnet macro-F1 | joint_pointnet macro-F1 |"
        sep = "|---:|---:|---:|---:|"
        md_lines.extend([header, sep])
        for variant_name in ["all_pointnet", "gtseg_pointnet", "joint_pointnet"]:
            summary_json["per_fold"].setdefault(variant_name, {})
            summary_json["per_fold"][variant_name]["macro_f1"] = fold_means(
                variant_cls_rows.get(variant_name) or [],
                metric="test_macro_f1_present",
            )
            summary_json["per_fold"][variant_name]["accuracy"] = fold_means(
                variant_cls_rows.get(variant_name) or [],
                metric="test_accuracy",
            )
        for fold in fold_ids:
            all_f1 = summary_json["per_fold"].get("all_pointnet", {}).get("macro_f1", {}).get(fold)
            gt_f1 = summary_json["per_fold"].get("gtseg_pointnet", {}).get("macro_f1", {}).get(fold)
            joint_f1 = summary_json["per_fold"].get("joint_pointnet", {}).get("macro_f1", {}).get(fold)
            md_lines.append(
                f"| {fold} | {'' if all_f1 is None else f'{all_f1:.4f}'} | {'' if gt_f1 is None else f'{gt_f1:.4f}'} | {'' if joint_f1 is None else f'{joint_f1:.4f}'} |"
            )

    md_lines.extend(["", "## Best Runs", ""])
    for variant_name in ["all_pointnet", "gtseg_pointnet", "joint_pointnet"]:
        rows = variant_cls_rows.get(variant_name) or []
        if not rows:
            continue
        best = top_runs(rows, metric="test_macro_f1_present", k=3)
        summary_json["best_runs"][variant_name] = best
        md_lines.append(f"### {variant_name}")
        md_lines.append("")
        for item in best:
            seg_part = ""
            if "test_seg_mean_iou" in item:
                seg_part = f", seg_mIoU={item['test_seg_mean_iou']:.4f}"
            md_lines.append(
                f"- `{item['run']}`: macro-F1={item['test_macro_f1_present']:.4f}, acc={item['test_accuracy']:.4f}{seg_part}"
            )
        md_lines.append("")

    seg_rows = variant_seg_rows.get("seg_pointnet") or []
    if seg_rows:
        best = top_runs(seg_rows, metric="test_mean_iou", k=3)
        summary_json["best_runs"]["seg_pointnet"] = best
        md_lines.append("### seg_pointnet")
        md_lines.append("")
        for item in best:
            md_lines.append(
                f"- `{item['run']}`: mIoU={item['test_mean_iou']:.4f}, acc={item['test_accuracy']:.4f}"
            )
        md_lines.append("")

    json_path = Path(str(out_prefix) + ".json")
    md_path = Path(str(out_prefix) + ".md")
    json_path.write_text(json.dumps(summary_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote: {json_path}")
    print(f"[OK] wrote: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
