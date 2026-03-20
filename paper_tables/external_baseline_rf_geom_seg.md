# External Baseline: Random Forest

- features: bbox_diag, bbox_vol, mean_curvature, mean_nn_dist, n_points, pca_ratio_01, pca_ratio_02, seg_prob_frac_above_05, seg_prob_mean, seg_prob_p10, seg_prob_p90, seg_prob_std, spread_xyz_0, spread_xyz_1, spread_xyz_2, std_curvature, std_nn_dist
- use_seg_prob: True
- use_seg_gt: False
- seeds: [1337, 2020, 2021]
- folds: 5
- runs: 15

## Aggregate Results

| Metric | Mean ± Std |
|--------|-----------|
| Test Accuracy | 0.3564 ± 0.1102 |
| Test Macro-F1 | 0.1944 ± 0.0726 |
| Test Balanced Acc | 0.2636 ± 0.1207 |

## Per-Class F1

| Class | Mean ± Std |
|-------|-----------|
| 充填 | 0.1022 ± 0.1702 |
| 全冠 | 0.0444 ± 0.1663 |
| 桩核冠 | 0.1601 ± 0.1738 |
| 高嵌体 | 0.4707 ± 0.1805 |

## Per-Fold Results

| Fold | Seed | Acc | Macro-F1 | Bal Acc |
|------|------|-----|----------|---------|
| 0 | 1337 | 0.3750 | 0.3010 | 0.4861 |
| 1 | 1337 | 0.1875 | 0.1288 | 0.1500 |
| 2 | 1337 | 0.4375 | 0.1522 | 0.1667 |
| 3 | 1337 | 0.3125 | 0.1250 | 0.2083 |
| 4 | 1337 | 0.5333 | 0.3333 | 0.3750 |
| 0 | 2020 | 0.3125 | 0.2652 | 0.4583 |
| 1 | 2020 | 0.1875 | 0.1294 | 0.1500 |
| 2 | 2020 | 0.3750 | 0.1364 | 0.1429 |
| 3 | 2020 | 0.3125 | 0.1250 | 0.2083 |
| 4 | 2020 | 0.5333 | 0.2750 | 0.3125 |
| 0 | 2021 | 0.3125 | 0.2652 | 0.4583 |
| 1 | 2021 | 0.1875 | 0.1218 | 0.1500 |
| 2 | 2021 | 0.4375 | 0.1591 | 0.1667 |
| 3 | 2021 | 0.3750 | 0.2316 | 0.2708 |
| 4 | 2021 | 0.4667 | 0.1667 | 0.2500 |