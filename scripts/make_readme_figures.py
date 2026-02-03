#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _plot_confusion_matrix(
    cm: np.ndarray,
    *,
    labels: list[str],
    title: str,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    # Ensure Chinese glyphs render in common Linux environments.
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    cm = np.asarray(cm, dtype=np.float64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"confusion_matrix must be square, got {cm.shape}")
    n = int(cm.shape[0])
    if len(labels) != n:
        raise ValueError(f"labels must have length {n}, got {len(labels)}")

    # Row-normalized for color, but annotate with both count and percent.
    row_sum = cm.sum(axis=1, keepdims=True)
    denom = np.where(row_sum > 0, row_sum, 1.0)
    cm_norm = cm / denom

    fig = plt.figure(figsize=(9.5, 8.0), dpi=140)
    ax = fig.add_subplot(111)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)

    for i in range(n):
        for j in range(n):
            count = int(cm[i, j])
            pct = float(cm_norm[i, j]) * 100.0
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{count}\n{pct:.1f}%", ha="center", va="center", fontsize=9, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized (%)", rotation=90)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0", "25", "50", "75", "100"])

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate small, README-friendly figures from paper_tables/*.json")
    ap.add_argument(
        "--raw-cls-eval",
        type=Path,
        default=Path("paper_tables/raw_cls_ensemble_eval_mean_v18_best.json"),
        help="Path to raw_cls ensemble eval json (mean v18 best).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("assets/readme"),
        help="Output directory for README figures.",
    )
    ap.add_argument(
        "--raw-cls-labels",
        type=str,
        default="充填,全冠,桩核冠,高嵌体",
        help="Comma-separated class labels (must match confusion_matrix order).",
    )
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    _ensure_dir(out_dir)

    # raw_cls confusion matrix
    raw_path = args.raw_cls_eval
    d = _read_json(raw_path)
    overall = d.get("overall") or {}
    cm = np.asarray(overall.get("confusion_matrix"), dtype=np.float64)
    labels = [s.strip() for s in str(args.raw_cls_labels).split(",") if s.strip()]

    acc = float(overall.get("accuracy", float("nan")))
    macro_f1 = float(overall.get("macro_f1", float("nan")))
    bal_acc = float(overall.get("balanced_acc", float("nan")))
    n = int(overall.get("n", 0) or 0)
    title = f"raw_cls v18 mean ensemble — n={n} | acc={acc:.4f} | macro_f1={macro_f1:.4f} | bal_acc={bal_acc:.4f}"

    out_cm = out_dir / "paper_tables" / "raw_cls_cm_v18_mean.png"
    _plot_confusion_matrix(cm, labels=labels, title=title, out_path=out_cm)

    print(f"[OK] Wrote: {out_cm}")


if __name__ == "__main__":
    main()
