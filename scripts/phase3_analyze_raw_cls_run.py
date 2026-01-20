#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        rows.append(json.loads(s))
    return rows


@dataclass(frozen=True)
class Row:
    case_key: str
    source: str
    y_true: int
    y_pred: int
    probs: list[float]


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze a raw_cls baseline run (preds_test.jsonl).")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--topk", type=int, default=15, help="Top-K most confident wrong predictions to show.")
    ap.add_argument(
        "--exclude-labels",
        type=str,
        default="",
        help="Comma-separated labels to exclude from analysis metrics (optional).",
    )
    ap.add_argument("--out", type=Path, default=None, help="Optional output Markdown file (default: stdout).")
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    preds_path = run_dir / "preds_test.jsonl"
    metrics_path = run_dir / "metrics.json"
    if not preds_path.exists():
        raise SystemExit(f"Missing: {preds_path}")
    if not metrics_path.exists():
        raise SystemExit(f"Missing: {metrics_path}")

    metrics = read_json(metrics_path)
    labels_by_id: list[str] = []
    cfg = read_json(run_dir / "config.json")
    # labels_by_id is stored in model checkpoint, but we can infer from metrics keys order.
    per_class = (metrics.get("test") or {}).get("per_class") or {}
    labels_by_id = list(per_class.keys())
    if not labels_by_id:
        raise SystemExit("Cannot infer labels_by_id from metrics.json")

    exclude_labels = {s.strip() for s in str(args.exclude_labels or "").split(",") if s.strip()}

    raw = read_jsonl(preds_path)
    rows: list[Row] = []
    for r in raw:
        rows.append(
            Row(
                case_key=str(r.get("case_key") or ""),
                source=str(r.get("source") or ""),
                y_true=int(r.get("y_true")),
                y_pred=int(r.get("y_pred")),
                probs=[float(x) for x in (r.get("probs") or [])],
            )
        )

    wrong: list[dict[str, Any]] = []
    confusion_pairs: Counter[tuple[str, str]] = Counter()
    for r in rows:
        if r.y_true == r.y_pred:
            continue
        y_true_lab = labels_by_id[r.y_true] if 0 <= r.y_true < len(labels_by_id) else str(r.y_true)
        y_pred_lab = labels_by_id[r.y_pred] if 0 <= r.y_pred < len(labels_by_id) else str(r.y_pred)
        p_pred = r.probs[r.y_pred] if 0 <= r.y_pred < len(r.probs) else 0.0
        p_true = r.probs[r.y_true] if 0 <= r.y_true < len(r.probs) else 0.0
        wrong.append(
            {
                "case_key": r.case_key,
                "source": r.source or "(missing)",
                "true": y_true_lab,
                "pred": y_pred_lab,
                "p_pred": float(p_pred),
                "p_true": float(p_true),
            }
        )
        confusion_pairs[(y_true_lab, y_pred_lab)] += 1

    wrong.sort(key=lambda x: float(x["p_pred"]), reverse=True)

    lines: list[str] = []
    lines.append(f"# raw_cls run 分析：`{run_dir.name}`")
    lines.append("")
    lines.append("## 配置")
    lines.append(f"- exp_name: `{cfg.get('exp_name')}`")
    lines.append(f"- data_root: `{cfg.get('data_root')}`")
    lines.append(f"- n_points: {cfg.get('n_points')}")
    lines.append("")

    lines.append("## 总体指标（test）")
    test_m = metrics.get("test") or {}
    lines.append(f"- accuracy: {test_m.get('accuracy')}")
    lines.append(f"- macro_f1_all: {test_m.get('macro_f1_all')}")
    lines.append(f"- balanced_accuracy_present: {test_m.get('balanced_accuracy_present')}")
    lines.append("")

    if exclude_labels:
        keep_ids = [i for i, lab in enumerate(labels_by_id) if lab not in exclude_labels]
        keep_labels = [labels_by_id[i] for i in keep_ids]
        old_to_new = {old: new for new, old in enumerate(keep_ids)}

        y_true_new: list[int] = []
        y_pred_new: list[int] = []
        for r in rows:
            if 0 <= r.y_true < len(labels_by_id) and labels_by_id[r.y_true] in exclude_labels:
                continue
            if r.y_true not in old_to_new:
                continue
            y_true_new.append(old_to_new[r.y_true])
            y_pred_new.append(old_to_new.get(r.y_pred, -1))

        total = len(y_true_new)
        cm = [[0 for _ in range(len(keep_labels) + 1)] for _ in range(len(keep_labels))]
        correct = 0
        for t, p in zip(y_true_new, y_pred_new, strict=True):
            if p == -1:
                cm[t][-1] += 1
            else:
                cm[t][p] += 1
                if t == p:
                    correct += 1
        acc = correct / total if total else 0.0

        # macro-F1 over kept labels, treating excluded predictions as wrong.
        f1s: list[float] = []
        for i in range(len(keep_labels)):
            tp = cm[i][i]
            fp = sum(cm[r][i] for r in range(len(keep_labels))) - tp
            fn = sum(cm[i]) - tp
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
            f1s.append(float(f1))
        macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

        lines.append(f"## 排除标签后的指标（exclude_labels={sorted(exclude_labels)}）")
        lines.append(f"- labels: {keep_labels} (+ other)")
        lines.append(f"- total: {total}")
        lines.append(f"- accuracy: {acc}")
        lines.append(f"- macro_f1: {macro_f1}")
        lines.append("")

    lines.append("## 主要混淆（test，按次数）")
    for (t, p), n in confusion_pairs.most_common(10):
        lines.append(f"- {t} → {p}: {n}")
    lines.append("")

    lines.append(f"## 最自信的错分 Top-{int(args.topk)}")
    for item in wrong[: int(args.topk)]:
        lines.append(
            f"- {item['case_key']} ({item['source']}): {item['true']} → {item['pred']} "
            f"(p_pred={item['p_pred']:.3f}, p_true={item['p_true']:.3f})"
        )
    lines.append("")

    out = "\n".join(lines) + "\n"
    if args.out is not None:
        out_path = (run_dir / args.out).resolve() if not args.out.is_absolute() else args.out.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out, encoding="utf-8")
        print(f"[OK] wrote: {out_path}")
    else:
        print(out, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
