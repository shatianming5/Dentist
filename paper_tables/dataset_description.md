# Table 1: Dataset Description

## Dentist-79 Intraoral Scan Benchmark

| Property | Value |
|----------|-------|
| Total cases | 79 |
| Source | Clinical 3D intraoral scans (STL-derived point clouds) |
| Annotation | Per-point binary: background (tooth structure) vs. restoration |
| Cross-validation | 5-fold case-level stratified |
| Seeds per fold | 3 (1337, 2020, 2021) → 15 total runs |

## Dual Evaluation Protocols

| Protocol | Points/case | Restoration ratio | Sampling | Purpose |
|----------|-------------|-------------------|----------|---------|
| Balanced (v1) | 8192 | 50.0% (fixed 50/50) | 4096 seg + 4096 rem | Controlled architecture comparison |
| Natural (v2) | 8192 | 19.3% ± 6.1% | Uniform random | Deployment-realistic evaluation |

## Natural-Ratio Distribution

| Statistic | Value |
|-----------|-------|
| Mean restoration ratio | 0.193 |
| Std restoration ratio | 0.061 |
| Min restoration ratio | 0.003 |
| Max restoration ratio | 0.364 |
| Median restoration ratio | 0.202 |

## Annotation Quality

- 78/79 cases verified via DGCNN per-case analysis
- 1 labeling error detected (furao2: swapped seg/rem labels) — corrected in natural protocol
- Corrected DGCNN balanced mIoU: 0.983 ± 0.017 (excluding furao2)

## Normalization & Augmentation

- Point clouds centered at centroid, scaled to unit sphere
- Training augmentation (DL models): random Z-axis rotation, uniform scaling [0.8, 1.2×], Gaussian jitter (σ = 0.01)
- RF uses hand-crafted geometric features without augmentation
