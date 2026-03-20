# Segmentation Architecture Comparison

## Main Results (5-fold × 3-seed, n=79 cases)

| Model | mIoU | Accuracy | BG IoU | Rest IoU | Params |
|-------|------|----------|--------|----------|--------|
| PointNet | 0.832±0.088 | 0.906±0.059 | 0.833±0.076 | 0.832±0.100 | 371,778 |
| **DGCNN** | **0.957±0.044** | **0.977±0.024** | **0.957±0.044** | **0.957±0.044** | **650,626** |
| Point Transformer | 0.375±0.033 | 0.597±0.025 | 0.196±0.052 | 0.553±0.015 | 387,682 |

## Pairwise Statistical Tests (paired bootstrap, 10k iterations)

| Comparison | Δ mIoU | 95% CI | Significant | W/L |
|------------|--------|--------|-------------|-----|
| DGCNN vs PointNet | +0.125 | [+0.102, +0.160] | **Yes** | 15/0 |
| DGCNN vs Point Transformer | +0.582 | [+0.550, +0.610] | **Yes** | 15/0 |
| PointNet vs Point Transformer | +0.457 | [+0.400, +0.502] | **Yes** | 15/0 |

## DGCNN Per-Fold Breakdown

| Fold | n_test | mIoU | Corrected mIoU |
|------|--------|------|----------------|
| 0 | 16 | 0.981±0.003 | 0.981±0.003 |
| 1 | 16 | 0.870±0.001 | 0.986±0.004 (furao2 label-swap) |
| 2 | 16 | 0.979±0.005 | 0.979±0.005 |
| 3 | 16 | 0.981±0.002 | 0.981±0.002 |
| 4 | 15 | 0.973±0.002 | 0.973±0.002 |

## Per-Case DGCNN Analysis

- 77/79 cases: mIoU > 0.97
- 1 annotation error (furao2): raw mIoU=0.002, corrected=0.989
- 1 moderate case (dxm): mIoU=0.841
- Corrected mean: 0.983 ± 0.017

## Key Findings

1. **DGCNN achieves near-perfect dental restoration segmentation** (mIoU=0.957, corrected 0.983)
2. **Architecture choice matters**: DGCNN (graph-based) >> PointNet (pointwise) >> Point Transformer (attention-based)
3. **Point Transformer fails on small datasets**: self-attention is too data-hungry for n=79
4. **Annotation error detection**: model correctly identifies label-swap in 1 case
5. **All pairwise comparisons significant** with 15/0 W/L ratio
