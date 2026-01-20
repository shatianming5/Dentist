#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


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


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist(), strict=True):
        if 0 <= int(t) < num_classes and 0 <= int(p) < num_classes:
            cm[int(t), int(p)] += 1
    return cm


def metrics_from_confusion(cm: np.ndarray) -> dict[str, float]:
    num_classes = int(cm.shape[0])
    total = int(cm.sum())
    correct = int(np.trace(cm))
    acc = float(correct / total) if total > 0 else 0.0

    f1s_all: list[float] = []
    f1s_present: list[float] = []
    recalls_present: list[float] = []
    for i in range(num_classes):
        tp = float(cm[i, i])
        fp = float(cm[:, i].sum() - cm[i, i])
        fn = float(cm[i, :].sum() - cm[i, i])
        support = float(cm[i, :].sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1s_all.append(float(f1))
        if support > 0:
            f1s_present.append(float(f1))
            recalls_present.append(float(recall))

    macro_f1_all = float(np.mean(f1s_all)) if f1s_all else 0.0
    macro_f1_present = float(np.mean(f1s_present)) if f1s_present else 0.0
    bal_acc_present = float(np.mean(recalls_present)) if recalls_present else 0.0
    return {
        "accuracy": float(acc),
        "macro_f1_all": float(macro_f1_all),
        "macro_f1_present": float(macro_f1_present),
        "balanced_accuracy_present": float(bal_acc_present),
    }


@dataclass(frozen=True)
class CI:
    mean: float
    lo: float
    hi: float


def percentile_ci(xs: np.ndarray, alpha: float = 0.05) -> CI:
    xs2 = np.asarray(xs, dtype=np.float64)
    mean = float(xs2.mean()) if xs2.size else 0.0
    if xs2.size <= 1:
        return CI(mean=mean, lo=mean, hi=mean)
    lo = float(np.quantile(xs2, alpha / 2.0))
    hi = float(np.quantile(xs2, 1.0 - alpha / 2.0))
    return CI(mean=mean, lo=lo, hi=hi)


def bootstrap_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    num_classes: int,
    n_boot: int,
    seed: int,
) -> dict[str, CI]:
    n = int(y_true.shape[0])
    rng = np.random.default_rng(int(seed))
    accs: list[float] = []
    f1p: list[float] = []
    f1a: list[float] = []
    bacc: list[float] = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n, endpoint=False)
        cm = confusion_matrix(y_true[idx], y_pred[idx], num_classes=num_classes)
        m = metrics_from_confusion(cm)
        accs.append(float(m["accuracy"]))
        f1a.append(float(m["macro_f1_all"]))
        f1p.append(float(m["macro_f1_present"]))
        bacc.append(float(m["balanced_accuracy_present"]))
    return {
        "accuracy": percentile_ci(np.asarray(accs)),
        "macro_f1_all": percentile_ci(np.asarray(f1a)),
        "macro_f1_present": percentile_ci(np.asarray(f1p)),
        "balanced_accuracy_present": percentile_ci(np.asarray(bacc)),
    }


def paired_bootstrap_diff(
    keys: list[str],
    y_true: dict[str, int],
    y_pred_a: dict[str, int],
    y_pred_b: dict[str, int],
    *,
    num_classes: int,
    n_boot: int,
    seed: int,
    metric: str,
) -> dict[str, Any]:
    if metric not in {"accuracy", "macro_f1_all", "macro_f1_present", "balanced_accuracy_present"}:
        raise ValueError(f"Unsupported metric: {metric}")
    n = len(keys)
    rng = random.Random(int(seed))
    diffs: list[float] = []
    for _ in range(int(n_boot)):
        sample = [keys[rng.randrange(n)] for _ in range(n)]
        yt = np.asarray([y_true[k] for k in sample], dtype=np.int64)
        ya = np.asarray([y_pred_a[k] for k in sample], dtype=np.int64)
        yb = np.asarray([y_pred_b[k] for k in sample], dtype=np.int64)
        ma = metrics_from_confusion(confusion_matrix(yt, ya, num_classes=num_classes))
        mb = metrics_from_confusion(confusion_matrix(yt, yb, num_classes=num_classes))
        diffs.append(float(mb[metric] - ma[metric]))

    xs = np.asarray(diffs, dtype=np.float64)
    ci = percentile_ci(xs)
    p = float(2.0 * min(np.mean(xs <= 0), np.mean(xs >= 0))) if xs.size else 1.0
    return {
        "metric": metric,
        "n": int(n),
        "diff_mean": float(ci.mean),
        "diff_ci95": [float(ci.lo), float(ci.hi)],
        "p_two_sided": p,
    }


def infer_num_classes(run_dir: Path) -> int:
    met = read_json(run_dir / "metrics.json")
    test = (met.get("test") or {}).get("per_class") or {}
    if test:
        return int(len(list(test.keys())))
    preds_path = run_dir / "preds_test.jsonl"
    rows = read_jsonl(preds_path)
    mx = 0
    for r in rows:
        mx = max(mx, int(r.get("y_true", 0)), int(r.get("y_pred", 0)))
    return int(mx + 1)


def load_preds(run_dir: Path, split: str) -> list[dict[str, Any]]:
    p = run_dir / f"preds_{split}.jsonl"
    if not p.exists():
        raise SystemExit(f"Missing preds file: {p}")
    return read_jsonl(p)


def main() -> int:
    ap = argparse.ArgumentParser(description="Bootstrap confidence intervals for raw_cls runs (and paired comparison).")
    ap.add_argument("--run-dir", type=Path, required=True, help="A run dir containing preds_{split}.jsonl and metrics.json.")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--n-bootstrap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")
    ap.add_argument("--compare", type=Path, default=None, help="Optional: second run dir to compare against (paired bootstrap).")
    ap.add_argument(
        "--metric",
        type=str,
        default="macro_f1_present",
        help="Metric for paired diff: accuracy|macro_f1_all|macro_f1_present|balanced_accuracy_present",
    )
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    if not (run_dir / "metrics.json").exists():
        raise SystemExit(f"Missing metrics.json: {run_dir}")

    split = str(args.split)
    rows = load_preds(run_dir, split)
    num_classes = infer_num_classes(run_dir)

    y_true = np.asarray([int(r["y_true"]) for r in rows], dtype=np.int64)
    y_pred = np.asarray([int(r["y_pred"]) for r in rows], dtype=np.int64)
    base = metrics_from_confusion(confusion_matrix(y_true, y_pred, num_classes=num_classes))
    ci = bootstrap_metrics(
        y_true,
        y_pred,
        num_classes=num_classes,
        n_boot=int(args.n_bootstrap),
        seed=int(args.seed),
    )

    out: dict[str, Any] = {
        "run_dir": str(run_dir),
        "split": split,
        "n": int(y_true.shape[0]),
        "num_classes": int(num_classes),
        "point_estimate": base,
        "ci95": {k: {"mean": v.mean, "lo": v.lo, "hi": v.hi} for k, v in ci.items()},
    }

    if args.compare is not None:
        run_b = args.compare.resolve()
        rows_b = load_preds(run_b, split)
        # Align by case_key (fallback to index if missing).
        a_map = {str(r.get("case_key") or i): r for i, r in enumerate(rows)}
        b_map = {str(r.get("case_key") or i): r for i, r in enumerate(rows_b)}
        keys = sorted(set(a_map.keys()) & set(b_map.keys()))
        if not keys:
            raise SystemExit("No overlapping keys between runs (need matching case_key values).")
        y_true_k: dict[str, int] = {}
        y_pred_a: dict[str, int] = {}
        y_pred_b: dict[str, int] = {}
        for k0 in keys:
            ya = int(a_map[k0]["y_true"])
            yb = int(b_map[k0]["y_true"])
            if ya != yb:
                raise SystemExit(f"Mismatched y_true for key={k0}: {ya} vs {yb}")
            y_true_k[k0] = ya
            y_pred_a[k0] = int(a_map[k0]["y_pred"])
            y_pred_b[k0] = int(b_map[k0]["y_pred"])
        out["paired_diff"] = paired_bootstrap_diff(
            keys,
            y_true_k,
            y_pred_a,
            y_pred_b,
            num_classes=num_classes,
            n_boot=int(args.n_bootstrap),
            seed=int(args.seed),
            metric=str(args.metric),
        )
        out["compare_run_dir"] = str(run_b)

    if args.out is not None:
        out_path = args.out.resolve() if args.out.is_absolute() else (Path.cwd() / args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[OK] wrote: {out_path}")
    else:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

