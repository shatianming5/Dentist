# External Baseline: Random Forest

- features: bbox_diag, bbox_vol, mean_curvature, mean_nn_dist, n_points, pca_ratio_01, pca_ratio_02, seg_gt_frac, seg_prob_frac_above_05, seg_prob_mean, seg_prob_p10, seg_prob_p90, seg_prob_std, spread_xyz_0, spread_xyz_1, spread_xyz_2, std_curvature, std_nn_dist
- use_seg_prob: True
- use_seg_gt: True
- seeds: [1337, 2020, 2021]
- folds: 5
- runs: 15

## Aggregate Results

| Metric | Mean ± Std |
|--------|-----------|
| Test Accuracy | 0.3522 ± 0.1033 |
| Test Macro-F1 | 0.2060 ± 0.0975 |
| Test Balanced Acc | 0.2739 ± 0.1397 |

## Per-Class F1

| Class | Mean ± Std |
|-------|-----------|
| 充填 | 0.1067 ± 0.1769 |
| 全冠 | 0.1111 ± 0.2250 |
| 桩核冠 | 0.1398 ± 0.1720 |
| 高嵌体 | 0.4666 ± 0.1468 |

## Per-Fold Results

| Fold | Seed | Acc | Macro-F1 | Bal Acc |
|------|------|-----|----------|---------|
| 0 | 1337 | 0.4375 | 0.3389 | 0.5139 |
| 1 | 1337 | 0.2500 | 0.1548 | 0.2000 |
| 2 | 1337 | 0.3750 | 0.1364 | 0.1429 |
| 3 | 1337 | 0.3125 | 0.1250 | 0.2083 |
| 4 | 1337 | 0.4667 | 0.2750 | 0.3393 |
| 0 | 2020 | 0.3750 | 0.3010 | 0.4861 |
| 1 | 2020 | 0.1875 | 0.1294 | 0.1500 |
| 2 | 2020 | 0.3125 | 0.1190 | 0.1190 |
| 3 | 2020 | 0.3125 | 0.1250 | 0.2083 |
| 4 | 2020 | 0.5333 | 0.3829 | 0.4018 |
| 0 | 2021 | 0.3750 | 0.3010 | 0.4861 |
| 1 | 2021 | 0.1875 | 0.1250 | 0.1500 |
| 2 | 2021 | 0.3125 | 0.1190 | 0.1190 |
| 3 | 2021 | 0.3125 | 0.1250 | 0.2083 |
| 4 | 2021 | 0.5333 | 0.3333 | 0.3750 |