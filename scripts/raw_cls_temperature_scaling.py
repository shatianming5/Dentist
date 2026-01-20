#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        rows.append(json.loads(s))
    return rows


def calibration_basic(probs: np.ndarray, y_true: np.ndarray, *, n_bins: int = 15) -> dict[str, Any]:
    if probs.ndim != 2:
        return {"total": int(y_true.shape[0]), "error": "invalid probs shape"}
    if y_true.ndim != 1 or y_true.shape[0] != probs.shape[0]:
        return {"total": int(y_true.shape[0]), "error": "invalid y_true shape"}

    eps = 1e-12
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == y_true).astype(np.float64)

    n_bins_i = max(2, int(n_bins))
    bins = np.linspace(0.0, 1.0, num=n_bins_i + 1)
    ece = 0.0
    for i in range(n_bins_i):
        lo = bins[i]
        hi = bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if not np.any(mask):
            continue
        w = float(np.mean(mask))
        bin_acc = float(np.mean(acc[mask]))
        bin_conf = float(np.mean(conf[mask]))
        ece += w * abs(bin_acc - bin_conf)

    p_true = probs[np.arange(y_true.shape[0]), y_true]
    nll = float(-np.mean(np.log(np.clip(p_true, eps, 1.0))))
    onehot = np.zeros_like(probs, dtype=np.float64)
    onehot[np.arange(y_true.shape[0]), y_true] = 1.0
    brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))

    return {
        "total": int(y_true.shape[0]),
        "ece": float(ece),
        "nll": float(nll),
        "brier": float(brier),
        "mean_conf": float(np.mean(conf)),
        "accuracy": float(np.mean(acc)),
    }


def temp_scale_probs(probs: np.ndarray, *, T: float) -> np.ndarray:
    eps = 1e-12
    tt = float(T)
    if not math.isfinite(tt) or tt <= 0:
        raise ValueError(f"Invalid temperature: {T}")
    logp = np.log(np.clip(probs, eps, 1.0)) / tt
    # Stable softmax.
    logp = logp - logp.max(axis=1, keepdims=True)
    exp = np.exp(logp)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), eps, None)


def nll_of_T(probs: np.ndarray, y_true: np.ndarray, *, T: float) -> float:
    eps = 1e-12
    p = temp_scale_probs(probs, T=T)
    p_true = p[np.arange(y_true.shape[0]), y_true]
    return float(-np.mean(np.log(np.clip(p_true, eps, 1.0))))


def fit_temperature(probs: np.ndarray, y_true: np.ndarray) -> tuple[float, dict[str, Any]]:
    # Log-spaced grid search then local refine around best.
    candidates = np.logspace(-2, 1, num=200, base=10.0).astype(np.float64)
    losses = np.array([nll_of_T(probs, y_true, T=float(t)) for t in candidates], dtype=np.float64)
    best_i = int(np.argmin(losses))
    best_T = float(candidates[best_i])

    # Refine in a small window around best (in log-space).
    lo = max(0, best_i - 5)
    hi = min(len(candidates) - 1, best_i + 5)
    t_lo = float(candidates[lo])
    t_hi = float(candidates[hi])
    refine = np.logspace(math.log10(t_lo), math.log10(t_hi), num=200, base=10.0).astype(np.float64)
    refine_losses = np.array([nll_of_T(probs, y_true, T=float(t)) for t in refine], dtype=np.float64)
    best2_i = int(np.argmin(refine_losses))
    best2_T = float(refine[best2_i])

    info = {
        "grid": {"num": int(candidates.shape[0]), "T_min": float(candidates.min()), "T_max": float(candidates.max())},
        "best_grid": {"T": float(best_T), "nll": float(losses[best_i])},
        "refine": {"num": int(refine.shape[0]), "T_min": float(refine.min()), "T_max": float(refine.max())},
        "best_refine": {"T": float(best2_T), "nll": float(refine_losses[best2_i])},
    }
    return best2_T, info


@dataclass(frozen=True)
class BinStats:
    lo: float
    hi: float
    count: int
    mean_conf: float
    mean_acc: float


def reliability_bins(probs: np.ndarray, y_true: np.ndarray, *, n_bins: int) -> list[BinStats]:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == y_true).astype(np.float64)

    n_bins_i = max(2, int(n_bins))
    edges = np.linspace(0.0, 1.0, num=n_bins_i + 1)
    out: list[BinStats] = []
    for i in range(n_bins_i):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        cnt = int(np.sum(mask))
        if cnt <= 0:
            out.append(BinStats(lo=lo, hi=hi, count=0, mean_conf=0.0, mean_acc=0.0))
            continue
        out.append(
            BinStats(
                lo=lo,
                hi=hi,
                count=cnt,
                mean_conf=float(np.mean(conf[mask])),
                mean_acc=float(np.mean(acc[mask])),
            )
        )
    return out


def plot_reliability(bins: list[BinStats], *, title: str, out_png: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise SystemExit("Missing matplotlib; install via `cd configs/env && pip install -r requirements.txt`.") from e

    xs = [0.5 * (b.lo + b.hi) for b in bins]
    acc = [b.mean_acc for b in bins]
    conf = [b.mean_conf for b in bins]
    w = [(b.hi - b.lo) * 0.9 for b in bins]

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(5, 5), dpi=160)
    ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="ideal")
    ax.bar(xs, acc, width=w, alpha=0.6, label="accuracy")
    ax.plot(xs, conf, color="C1", marker="o", linewidth=1, label="confidence")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy / confidence")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Temperature scaling for raw_cls runs (fits on val, evaluates on test).")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--bins", type=int, default=15)
    ap.add_argument("--out", type=Path, default=None, help="Output calib.json (default: <run>/calib.json)")
    ap.add_argument("--plot", type=Path, default=None, help="Output reliability.png (default: <run>/reliability.png)")
    args = ap.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.exists():
        raise SystemExit(f"Missing run_dir: {run_dir}")
    preds_val = run_dir / "preds_val.jsonl"
    preds_test = run_dir / "preds_test.jsonl"
    if not preds_val.is_file():
        raise SystemExit(f"Missing preds_val.jsonl: {preds_val}")
    if not preds_test.is_file():
        raise SystemExit(f"Missing preds_test.jsonl: {preds_test}")

    rows_val = read_jsonl(preds_val)
    rows_test = read_jsonl(preds_test)
    probs_val = np.asarray([r.get("probs") or [] for r in rows_val], dtype=np.float64)
    y_val = np.asarray([int(r.get("y_true", 0)) for r in rows_val], dtype=np.int64)
    probs_test = np.asarray([r.get("probs") or [] for r in rows_test], dtype=np.float64)
    y_test = np.asarray([int(r.get("y_true", 0)) for r in rows_test], dtype=np.int64)
    if probs_val.ndim != 2 or probs_val.shape[0] != y_val.shape[0]:
        raise SystemExit(f"Invalid val probs shape: {probs_val.shape} vs y {y_val.shape}")
    if probs_test.ndim != 2 or probs_test.shape[0] != y_test.shape[0]:
        raise SystemExit(f"Invalid test probs shape: {probs_test.shape} vs y {y_test.shape}")
    if probs_val.shape[1] != probs_test.shape[1]:
        raise SystemExit(f"Val/test num_classes mismatch: {probs_val.shape[1]} vs {probs_test.shape[1]}")

    T, fit_info = fit_temperature(probs_val, y_val)
    p_val_cal = temp_scale_probs(probs_val, T=T)
    p_test_cal = temp_scale_probs(probs_test, T=T)

    out_json = args.out.expanduser().resolve() if args.out is not None else (run_dir / "calib.json")
    out_png = args.plot.expanduser().resolve() if args.plot is not None else (run_dir / "reliability.png")

    before_val = calibration_basic(probs_val, y_val, n_bins=int(args.bins))
    after_val = calibration_basic(p_val_cal, y_val, n_bins=int(args.bins))
    before_test = calibration_basic(probs_test, y_test, n_bins=int(args.bins))
    after_test = calibration_basic(p_test_cal, y_test, n_bins=int(args.bins))

    bins_after_test = reliability_bins(p_test_cal, y_test, n_bins=int(args.bins))
    plot_reliability(bins_after_test, title=f"Reliability (T={T:.3g})", out_png=out_png)

    obj = {
        "temperature": float(T),
        "bins": int(args.bins),
        "fit": fit_info,
        "val": {"before": before_val, "after": after_val},
        "test": {"before": before_test, "after": after_test},
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[OK] wrote: {out_json}")
    print(f"[OK] wrote: {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
