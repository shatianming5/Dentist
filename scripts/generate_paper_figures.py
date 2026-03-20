#!/usr/bin/env python3
"""Generate paper figures: mixing barplot, PAFA scatter, bump chart."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

OUT = Path("assets")
OUT.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=1.2)


# ── Figure A: Protocol Mixing Barplot ──────────────────────────────────────
def fig_mixing_barplot():
    bench = json.loads(Path("paper_tables/full_benchmark_n25_all.json").read_text())
    
    methods_keys = [
        ("DGCNN", "DGCNN"),
        ("PointNet++", "PointNet++"),
        ("PointNet", "PointNet"),
        ("MV-ViT-ft", "MV-ViT (ft)"),
        ("RF", "Random\nForest"),
        ("DINOv2-MV", "DINOv2"),
        ("PT", "Point\nTransformer"),
    ]
    
    labels = [nice for _, nice in methods_keys]
    bal_vals = [bench[f"{k} balanced"] for k, _ in methods_keys]
    nat_vals = [bench[f"{k} natural"] for k, _ in methods_keys]
    bal_only = [np.mean(v) for v in bal_vals]
    nat_only = [np.mean(v) for v in nat_vals]
    bal_err = [np.std(v) for v in bal_vals]
    nat_err = [np.std(v) for v in nat_vals]
    
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(methods_keys))
    w = 0.35
    
    bars1 = ax.bar(x - w/2, bal_only, w, yerr=bal_err, capsize=3,
                   label="Balanced Protocol", color="#4C72B0", edgecolor="white", error_kw=dict(lw=1))
    bars2 = ax.bar(x + w/2, nat_only, w, yerr=nat_err, capsize=3,
                   label="Natural Protocol", color="#DD8452", edgecolor="white", error_kw=dict(lw=1))
    
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.008,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=9)
    
    ax.set_ylabel("Mean IoU", fontsize=13)
    ax.set_title("Protocol-Induced Performance Gap Across 7 Methods", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0.4, 1.05)
    ax.legend(fontsize=11, loc="upper right")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)
    
    # Annotate max gap
    gaps = [b - n for b, n in zip(bal_only, nat_only)]
    max_idx = int(np.argmax(gaps))
    gap = gaps[max_idx]
    mid_y = (bal_only[max_idx] + nat_only[max_idx]) / 2
    ax.annotate(f"Max gap: {gap:.3f}",
                xy=(max_idx + 0.25, mid_y), fontsize=10, color="red",
                fontweight="bold",
                arrowprops=dict(arrowstyle="<->", color="red", lw=1.5),
                xytext=(max_idx + 0.8, mid_y))
    
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"figure_mixing_barplot.{ext}", dpi=300, bbox_inches="tight")
    plt.close()
    print("[OK] figure_mixing_barplot")


# ── Figure B: PAFA vs Mixing Paired Scatter ────────────────────────────────
def fig_pafa_scatter():
    data = json.loads(Path("paper_tables/pafa_aligned_results.json").read_text())
    
    mix_nat = [r["nat"] for r in data["per_run"]["mixing"]]
    pafa_nat = [r["nat"] for r in data["per_run"]["pafa"]]
    mix_bal = [r["bal"] for r in data["per_run"]["mixing"]]
    pafa_bal = [r["bal"] for r in data["per_run"]["pafa"]]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    
    # Natural protocol
    ax = axes[0]
    ax.scatter(mix_nat, pafa_nat, c="#E24A33", s=60, alpha=0.8, edgecolors="white", zorder=3)
    lims = [min(min(mix_nat), min(pafa_nat)) - 0.02, max(max(mix_nat), max(pafa_nat)) + 0.02]
    ax.plot(lims, lims, "k--", alpha=0.4, label="y = x (no effect)")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Mixing mIoU (baseline)", fontsize=12)
    ax.set_ylabel("PAFA mIoU", fontsize=12)
    ax.set_title("Natural Protocol\n(Δ = −0.046, p = 0.001)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    # Count below diagonal
    below = sum(1 for m, p in zip(mix_nat, pafa_nat) if p < m)
    ax.text(0.05, 0.95, f"{below}/15 below diagonal\n(PAFA hurts)",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffcccc", alpha=0.8))
    
    # Balanced protocol
    ax = axes[1]
    ax.scatter(mix_bal, pafa_bal, c="#348ABD", s=60, alpha=0.8, edgecolors="white", zorder=3)
    lims = [min(min(mix_bal), min(pafa_bal)) - 0.02, max(max(mix_bal), max(pafa_bal)) + 0.02]
    ax.plot(lims, lims, "k--", alpha=0.4, label="y = x (no effect)")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Mixing mIoU (baseline)", fontsize=12)
    ax.set_ylabel("PAFA mIoU", fontsize=12)
    ax.set_title("Balanced Protocol\n(Δ = −0.007, p = 0.010)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    below = sum(1 for m, p in zip(mix_bal, pafa_bal) if p < m)
    ax.text(0.05, 0.95, f"{below}/15 below diagonal",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#cce5ff", alpha=0.8))
    
    fig.suptitle("PAFA vs. Mixing Baseline: Paired Comparison (15 seed×fold pairs)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"figure_pafa_scatter.{ext}", dpi=300, bbox_inches="tight")
    plt.close()
    print("[OK] figure_pafa_scatter")


# ── Figure C: Method Ranking Bump Chart ────────────────────────────────────
def fig_bump_chart():
    bench = json.loads(Path("paper_tables/full_benchmark_n25_all.json").read_text())
    
    methods_keys = [
        ("DGCNN", "DGCNN"),
        ("PointNet++", "PointNet++"),
        ("PointNet", "PointNet"),
        ("MV-ViT-ft", "MV-ViT (ft)"),
        ("RF", "Random Forest"),
        ("DINOv2-MV", "DINOv2-MV"),
        ("PT", "Point Transformer"),
    ]
    
    bal_scores = {nice: np.mean(bench[f"{k} balanced"]) for k, nice in methods_keys}
    nat_scores = {nice: np.mean(bench[f"{k} natural"]) for k, nice in methods_keys}
    
    bal_ranked = sorted(bal_scores.items(), key=lambda x: -x[1])
    nat_ranked = sorted(nat_scores.items(), key=lambda x: -x[1])
    
    bal_rank = {m: i+1 for i, (m, _) in enumerate(bal_ranked)}
    nat_rank = {m: i+1 for i, (m, _) in enumerate(nat_ranked)}
    
    common = [nice for _, nice in methods_keys]
    
    colors = sns.color_palette("husl", len(common))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for i, m in enumerate(common):
        br = bal_rank[m]
        nr = nat_rank[m]
        color = colors[i]
        ax.plot([0, 1], [br, nr], "o-", color=color, linewidth=2.5, markersize=10,
                label=m, zorder=3)
        # Labels
        ax.text(-0.08, br, f"{br}. {m}", fontsize=10,
                va="center", ha="right", color=color, fontweight="bold")
        ax.text(1.08, nr, f"{nr}. {m}", fontsize=10,
                va="center", ha="left", color=color, fontweight="bold")
    
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(len(common) + 0.5, 0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Balanced Protocol", "Natural Protocol"], fontsize=13, fontweight="bold")
    ax.set_ylabel("Rank (1 = best)", fontsize=12)
    ax.set_title("Method Ranking Shift Between Protocols", fontsize=14, fontweight="bold")
    ax.set_yticks(range(1, len(common) + 1))
    ax.grid(axis="y", alpha=0.3)
    ax.grid(axis="x", alpha=0)
    
    # Highlight rank changes > 2
    for m in common:
        delta = abs(nat_rank[m] - bal_rank[m])
        if delta >= 2:
            mid_x = 0.5
            mid_y = (bal_rank[m] + nat_rank[m]) / 2
            ax.text(mid_x, mid_y, f"Δ{delta}", fontsize=9, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"figure_bump_chart.{ext}", dpi=300, bbox_inches="tight")
    plt.close()
    print("[OK] figure_bump_chart")


if __name__ == "__main__":
    fig_mixing_barplot()
    fig_pafa_scatter()
    fig_bump_chart()
    print("\nAll figures generated in assets/")
