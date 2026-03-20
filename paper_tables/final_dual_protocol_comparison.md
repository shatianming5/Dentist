# Table 2: Dual-Protocol Segmentation Benchmark

## Main Results (5-fold × 3-seed = 15 runs per cell)

### Primary Architecture Comparison

| Model | Balanced mIoU | Natural mIoU | Δ (absolute) | Δ (relative) |
|-------|:---:|:---:|:---:|:---:|
| **DGCNN** | **0.957 ± 0.044** | **0.690 ± 0.045** | −0.267 | −27.9% |
| RF | 0.906 ± 0.030 | 0.546 ± 0.021 | −0.361 | −39.8% |
| PointNet | 0.832 ± 0.088 | 0.662 ± 0.030 | −0.170 | −20.5% |

All pairwise differences significant (Wilcoxon p < 0.05).

### Point Transformer Stability Analysis (Reported Separately)

PT exhibits extreme seed-dependent bimodality and is reported separately from the main comparison:

| Condition | Convergence Rate | Converged mIoU | Failed mIoU | All-Run Median |
|-----------|:---:|:---:|:---:|:---:|
| Balanced (lr=5e-5) | 5/15 (33%) | 0.902 ± 0.047 | 0.423 ± 0.112 | 0.512 |
| Natural (lr=5e-5) | 2/15 (13%) | 0.610 ± 0.005 | 0.547 ± 0.037 | 0.551 |

- Convergence is perfectly seed-determined: seed 1337 converges across all folds; seeds 2020, 2021 fail.
- LR sweep: 1e-3 → 0.375, 5e-4 → 0.417, 1e-4 → 0.333, 5e-5 → 0.942 (seed-dependent).
- When converged, PT approaches DGCNN (0.902 vs 0.957), but the 33% convergence rate makes it unsuitable for reliable deployment without extensive hyperparameter search.

## Ranking Change Across Protocols

| Rank | Balanced | Natural |
|------|----------|---------|
| 1 | DGCNN (0.957) | DGCNN (0.690) |
| 2 | RF (0.906) | PointNet (0.662) |
| 3 | PointNet (0.832) | RF (0.546) |

**Key finding**: RF drops from #2 to tied-last under natural ratio. PointNet is most imbalance-robust (−20.5% relative drop vs RF's −39.8%).

## Pairwise Wilcoxon Signed-Rank Tests

### Balanced Protocol
| Comparison | p-value | Sig. |
|------------|:---:|:---:|
| DGCNN vs RF | 0.015 | * |
| DGCNN vs PointNet | <0.001 | *** |
| RF vs PointNet | 0.002 | ** |

### Natural Protocol
| Comparison | p-value | Sig. |
|------------|:---:|:---:|
| DGCNN vs PointNet | 0.041 | * |
| DGCNN vs RF | <0.001 | *** |
| PointNet vs RF | <0.001 | *** |

## Boundary IoU (ε = 0.02 unit-sphere radius, all 4 models)

| Model | Balanced | Natural |
|-------|:---:|:---:|
| DGCNN | 0.724 ± 0.095 | 0.324 ± 0.044 |
| PointNet | 0.384 ± 0.059 | 0.317 ± 0.061 |
| PT (tuned) | 0.374 ± 0.038 | 0.311 ± 0.055 |
| RF | 0.349 ± 0.035 | 0.321 ± 0.102 |

Finding: DGCNN boundary accuracy is significantly higher under balanced (0.724 vs 0.35–0.38 for others), but all models converge to ~0.31–0.32 under natural ratio — near-random boundary segmentation.

## Protocol Gap Analysis

| Model | Gap | Rel. Drop | Interpretation |
|-------|:---:|:---:|---|
| RF | 0.361 | 39.8% | Geometry-only features most affected by class imbalance |
| DGCNN | 0.267 | 27.9% | Strong learned features still degrade substantially |
| PointNet | 0.170 | 20.5% | Most robust; global features less sensitive to local ratio |

## Ablations (DGCNN Natural)
| Variant | mIoU | Δ vs baseline |
|---------|:---:|:---:|
| CE + inv-freq weights (baseline) | 0.690 | — |
| Focal loss (γ=2) | 0.634 | −0.056 |
| 32k points (4× more) | 0.668 | −0.022 |
| Cross-eval (train bal → test nat) | 0.443 | −0.247 |

## Natural-Ratio Per-Case Analysis (DGCNN, 75 evaluated)
- Correlation: restoration ratio vs mIoU r=0.312, p=0.006
- Median per-case mIoU: 0.790
- Cases >0.8 mIoU: 37/75 (49%)
- Cases >0.9 mIoU: 7/75 (9%)
- Worst failures: cases with extreme class imbalance (model predicts all-background)

## Related Work Positioning

Existing dental 3D segmentation benchmarks (Teeth3DS [Ben-Hamadou et al., 2023], OSTeethSeg 2023 challenge, DentalPointNet [Tian et al., 2019]) target **tooth-level instance segmentation** — separating individual teeth from gingiva. This is a fundamentally different task from **restoration-level binary segmentation**, which distinguishes restored tooth structure from natural tooth material within a single dental arch. No prior benchmark addresses restoration segmentation from intraoral 3D scans. The dual-protocol evaluation is motivated by the specific imbalance structure of restoration segmentation (~19% minority class), which is more extreme than typical tooth segmentation (~50% teeth vs gingiva) and creates a domain where balanced sampling substantially inflates performance metrics and changes architecture rankings.

## Effect Sizes (Cohen's d, paired)

| Comparison | Balanced d | Natural d | Reversal |
|------------|:---:|:---:|---|
| PointNet vs RF | −0.73 | +3.25 | RF wins → PointNet wins |
| DGCNN vs RF | +0.83 | +3.18 | Gap increases 4× |
| DGCNN vs PointNet | +2.06 | +0.63 | Gap shrinks 3× |

## Boundary IoU Trivial Baseline

All-background predictor boundary mIoU ≈ 0.25 (BG IoU ≈ 0.5 at boundary, RES IoU = 0).
Models at 0.31–0.32 under natural ratio are only marginally above trivial (+0.07).
Under balanced ratio, DGCNN achieves 0.724 (substantially above trivial).

## Data Augmentation

All deep learning models use: random Z-axis rotation, uniform scale [0.8, 1.2], Gaussian jitter (σ=0.01).
RF uses hand-crafted geometric features without augmentation.

---

## Addendum: Complete Clinical Validation Metrics (Round 12-13)

### Table 6: Per-Point Clinical Metrics (Natural Protocol, DGCNN)
| Metric | Value |
|--------|-------|
| Sensitivity | 0.743 |
| Specificity | 0.933 |
| PPV | 0.729 |
| NPV | 0.937 |
| AUC-ROC | 0.938 |
| ECE | 0.023 |
| Brier score | 0.074 |

### Table 7: Uncertainty-Aware Selective Prediction
| Accept % | n | mIoU | Min mIoU | Screening | 
|----------|---|------|----------|-----------|
| 100% | 75 | 0.732 | 0.291 | None |
| 80% | 60 | 0.785 | 0.315 | Uncertainty-frac |
| 70% | 52 | 0.804 | 0.315 | Uncertainty-frac |
| 50% | 37 | 0.804 | 0.315 | Uncertainty-frac |

Confidence–mIoU r=0.598 (p<0.0001)

### Table 8: Inter-Model Consistency
| Comparison | κ (mean ± std) | Range |
|-----------|---------------|-------|
| DGCNN vs PointNet | 0.671 ± 0.256 | [−0.069, 0.945] |
| DGCNN seed₁ vs seed₂ | 0.695 ± 0.251 | [−0.210, 0.953] |
| DGCNN vs GT | 0.672 | — |
| PointNet vs GT | 0.606 | — |

### Table 9: Per-Type Results with Bootstrap 95% CIs
| Type | n | Natural mIoU | 95% CI | ≥0.7 rate | 95% CI |
|------|---|-------------|--------|-----------|--------|
| 充填 | 11 | 0.846 | [0.818, 0.876] | 100% | — |
| 全冠 | 13 | 0.800 | [0.675, 0.850] | 84.6% | [53.8%, 100%] |
| 桩核冠 | 10 | 0.696 | [0.562, 0.803] | 60.0% | [30.0%, 90.0%] |
| 高嵌体 | 41 | 0.689 | [0.635, 0.738] | 56.1% | [41.5%, 70.7%] |

### Updated Main Results (5 Models)
| Model | Balanced | Natural | Δ | Cohen's d |
|-------|----------|---------|---|-----------|
| DGCNN | 0.957 ± 0.044 | 0.690 ± 0.045 | −0.267 | 5.99 |
| PN2 | 0.950 ± 0.045 | 0.593 ± 0.102 | −0.357 | 4.53 |
| RF | 0.906 ± 0.030 | 0.546 ± 0.021 | −0.361 | 13.87 |
| PointNet | 0.832 ± 0.088 | 0.662 ± 0.030 | −0.170 | 2.59 |
| PT(tuned) | 0.583 ± 0.245 | 0.555 ± 0.041 | −0.028 | 0.16 |

Ranking flip: Balanced DGCNN≈PN2>RF>PN>PT → Natural DGCNN>PN>PN2≈RF>PT
