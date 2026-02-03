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


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


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


def _compute_reliability_bins(
    conf: np.ndarray,
    correct: np.ndarray,
    *,
    n_bins: int,
) -> dict[str, np.ndarray]:
    conf = np.asarray(conf, dtype=np.float64).reshape(-1)
    correct = np.asarray(correct, dtype=np.float64).reshape(-1)
    if conf.shape[0] != correct.shape[0]:
        raise ValueError("conf and correct must have same length")
    if conf.size == 0:
        raise ValueError("empty predictions")
    nb = int(n_bins)
    nb = max(2, min(nb, 50))

    edges = np.linspace(0.0, 1.0, nb + 1, dtype=np.float64)
    # Place 1.0 into the last bin by using right=True and clipping.
    bin_idx = np.digitize(conf, edges, right=True) - 1
    bin_idx = np.clip(bin_idx, 0, nb - 1)

    bin_count = np.zeros((nb,), dtype=np.int64)
    bin_acc = np.zeros((nb,), dtype=np.float64)
    bin_conf = np.zeros((nb,), dtype=np.float64)
    for b in range(nb):
        mask = bin_idx == b
        c = int(mask.sum())
        bin_count[b] = c
        if c <= 0:
            bin_acc[b] = np.nan
            bin_conf[b] = np.nan
            continue
        bin_acc[b] = float(np.mean(correct[mask]))
        bin_conf[b] = float(np.mean(conf[mask]))

    total = int(conf.shape[0])
    weights = bin_count.astype(np.float64) / max(1.0, float(total))
    ece = float(np.nansum(weights * np.abs(bin_acc - bin_conf)))
    return {
        "edges": edges,
        "count": bin_count,
        "acc": bin_acc,
        "conf": bin_conf,
        "ece": np.asarray([ece], dtype=np.float64),
    }


def _plot_reliability_diagram(
    bins: dict[str, np.ndarray],
    *,
    title: str,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    edges = np.asarray(bins["edges"], dtype=np.float64)
    acc = np.asarray(bins["acc"], dtype=np.float64)
    conf = np.asarray(bins["conf"], dtype=np.float64)
    cnt = np.asarray(bins["count"], dtype=np.int64)

    nb = int(cnt.shape[0])
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = (edges[1] - edges[0]) * 0.9

    fig = plt.figure(figsize=(9.5, 7.2), dpi=140)
    ax = fig.add_subplot(111)

    # Reliability bars = accuracy per bin.
    ax.bar(centers, np.nan_to_num(acc, nan=0.0), width=width, color="#2563eb", alpha=0.75, label="Accuracy (per bin)")
    # Perfect calibration.
    ax.plot([0, 1], [0, 1], linestyle="--", color="#111827", linewidth=1.2, label="Perfect calibration")
    # Confidence curve.
    ax.plot(centers, np.nan_to_num(conf, nan=np.nan), marker="o", color="#f97316", linewidth=1.2, label="Mean confidence")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.5)

    # Annotate counts on top of bars for non-empty bins.
    for x, y, n in zip(centers.tolist(), acc.tolist(), cnt.tolist(), strict=False):
        if not np.isfinite(y) or n <= 0:
            continue
        ax.text(float(x), float(y) + 0.02, f"n={n}", ha="center", va="bottom", fontsize=8, color="#111827")

    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _load_probs_from_runs(
    *,
    runs_root: Path,
    exp: str,
    model: str,
    fold: int,
    seed: int,
    split: str = "test",
) -> dict[str, dict]:
    p = runs_root / exp / model / f"fold={int(fold)}" / f"seed={int(seed)}" / f"preds_{split}.jsonl"
    if not p.exists():
        raise FileNotFoundError(str(p))
    rows = _read_jsonl(p)
    out: dict[str, dict] = {}
    for r in rows:
        case_key = str(r.get("case_key", "")).strip()
        if not case_key:
            continue
        probs = r.get("probs")
        y_true = r.get("y_true")
        if probs is None or y_true is None:
            continue
        out[case_key] = {
            "y_true": int(y_true),
            "probs": np.asarray(probs, dtype=np.float64),
        }
    return out


def _build_mean_prob_ensemble_predictions(eval_json: dict, *, runs_root_override: Path | None) -> tuple[np.ndarray, np.ndarray]:
    runs_root = Path(str(eval_json.get("runs_root", ""))).expanduser()
    if runs_root_override is not None:
        runs_root = runs_root_override.expanduser()
    runs_root = runs_root.resolve()

    members = eval_json.get("members") or []
    seeds = eval_json.get("seeds") or []
    folds = eval_json.get("folds") or []
    if not members or not seeds or not folds:
        raise ValueError("eval json missing members/seeds/folds")

    # Aggregate probs across all (member, seed) for each case_key (across folds).
    probs_sum: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    y_true_by_key: dict[str, int] = {}

    missing_files = 0
    for mem in members:
        exp = str(mem.get("exp", "")).strip()
        model = str(mem.get("model", "")).strip()
        if not exp or not model:
            continue
        for fold in folds:
            for seed in seeds:
                try:
                    rows = _load_probs_from_runs(
                        runs_root=runs_root,
                        exp=exp,
                        model=model,
                        fold=int(fold),
                        seed=int(seed),
                        split="test",
                    )
                except FileNotFoundError:
                    missing_files += 1
                    continue
                for k, v in rows.items():
                    p = v["probs"]
                    yt = int(v["y_true"])
                    if k not in probs_sum:
                        probs_sum[k] = p.astype(np.float64, copy=True)
                        counts[k] = 1
                        y_true_by_key[k] = yt
                    else:
                        probs_sum[k] += p
                        counts[k] += 1
                        if y_true_by_key.get(k) != yt:
                            raise ValueError(f"y_true mismatch for {k}: {y_true_by_key.get(k)} vs {yt}")

    if missing_files:
        print(f"[warn] Missing preds_test.jsonl files: {missing_files} (will compute with available runs)")
    if not probs_sum:
        raise FileNotFoundError(f"No preds_test.jsonl found under runs_root={runs_root}")

    # Final arrays.
    keys = sorted(probs_sum.keys())
    probs = np.stack([(probs_sum[k] / float(max(1, counts[k]))) for k in keys], axis=0)
    y_true = np.asarray([y_true_by_key[k] for k in keys], dtype=np.int64)
    return probs, y_true


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate small, README-friendly figures from paper_tables/*.json")
    ap.add_argument(
        "--raw-cls-eval",
        type=Path,
        default=Path("paper_tables/raw_cls_ensemble_eval_mean_v18_best.json"),
        help="Path to raw_cls ensemble eval json (mean v18 best).",
    )
    ap.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="Optional override for runs root (if paper_tables json uses an absolute path).",
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
    ap.add_argument("--calibration-bins", type=int, default=15, help="Number of bins for reliability diagram.")
    ap.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Skip generating reliability diagram (still generates confusion matrix).",
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

    if bool(args.skip_calibration):
        return

    # Reliability diagram (requires access to runs/*/preds_test.jsonl referenced by the eval json).
    try:
        probs, y_true = _build_mean_prob_ensemble_predictions(d, runs_root_override=args.runs_root)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] Skip reliability diagram: {e}")
        return

    conf = np.max(probs, axis=1)
    y_pred = np.argmax(probs, axis=1)
    correct = (y_pred == y_true).astype(np.float64)
    bins = _compute_reliability_bins(conf, correct, n_bins=int(args.calibration_bins))
    ece = float(bins["ece"][0])
    title_rel = f"Calibration (reliability) — raw_cls v18 mean ensemble — n={int(y_true.shape[0])} | ECE≈{ece:.4f}"
    out_rel = out_dir / "paper_tables" / "raw_cls_calibration_v18_mean.png"
    _plot_reliability_diagram(bins, title=title_rel, out_path=out_rel)
    print(f"[OK] Wrote: {out_rel}")


if __name__ == "__main__":
    main()
