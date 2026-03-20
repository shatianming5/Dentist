# research seg+cls full summary (research_segcls_full)

- generated_at: 2026-03-17T07:50:48.379213Z

| variant | n | test_acc | test_macro_f1 | test_bal_acc | test_ece | test_seg_miou |
|---|---:|---:|---:|---:|---:|---:|
| all_pointnet | 15 | 0.2239±0.0886 | 0.1413±0.0598 | 0.2096±0.0997 | 0.3197±0.1153 |  |
| gtseg_pointnet | 15 | 0.3092±0.1108 | 0.2803±0.0861 | 0.4102±0.1703 | 0.3155±0.0476 |  |
| joint_pointnet | 15 | 0.2411±0.1106 | 0.1899±0.1158 | 0.2496±0.1633 | 0.3657±0.1357 | 0.7603±0.0938 |
| seg_pointnet | 15 | 0.9060±0.0592 |  |  |  | 0.8323±0.0877 |

## Pairwise Comparisons

| comparison | n_pairs | delta_acc | delta_macro_f1 | delta_bal_acc | wins/ties/losses |
|---|---:|---:|---:|---:|---:|
| gtseg_pointnet - all_pointnet | 15 | 0.0853 [0.0100, 0.1506] | 0.1390 [0.0826, 0.1933] | 0.2006 [0.0931, 0.3065] | 13/0/2 |
| joint_pointnet - all_pointnet | 15 | 0.0172 [-0.0533, 0.0922] | 0.0486 [-0.0151, 0.1269] | 0.0399 [-0.0384, 0.1197] | 7/0/8 |
| joint_pointnet - gtseg_pointnet | 15 | -0.0681 [-0.1356, 0.0031] | -0.0904 [-0.1661, -0.0019] | -0.1606 [-0.2664, -0.0700] | 3/0/12 |

## Per-Fold Means

| fold | all_pointnet macro-F1 | gtseg_pointnet macro-F1 | joint_pointnet macro-F1 |
|---:|---:|---:|---:|
| 0 | 0.1687 | 0.2402 | 0.1391 |
| 1 | 0.1270 | 0.3044 | 0.1487 |
| 2 | 0.1365 | 0.2591 | 0.2329 |
| 3 | 0.0853 | 0.2893 | 0.2687 |
| 4 | 0.1889 | 0.3086 | 0.1602 |

## Best Runs

### all_pointnet

- `all_pointnet_fold0_seed2020`: macro-F1=0.2566, acc=0.3750
- `all_pointnet_fold4_seed1337`: macro-F1=0.2404, acc=0.2667
- `all_pointnet_fold2_seed2020`: macro-F1=0.2000, acc=0.3750

### gtseg_pointnet

- `gtseg_pointnet_fold1_seed2021`: macro-F1=0.4333, acc=0.4375
- `gtseg_pointnet_fold0_seed2021`: macro-F1=0.3904, acc=0.3750
- `gtseg_pointnet_fold4_seed1337`: macro-F1=0.3857, acc=0.4667

### joint_pointnet

- `joint_pointnet_fold2_seed2021`: macro-F1=0.5251, acc=0.3125, seg_mIoU=0.8598
- `joint_pointnet_fold3_seed1337`: macro-F1=0.3393, acc=0.3750, seg_mIoU=0.8152
- `joint_pointnet_fold3_seed2020`: macro-F1=0.2833, acc=0.3750, seg_mIoU=0.7080

### seg_pointnet

- `seg_pointnet_fold2_seed1337`: mIoU=0.8999, acc=0.9474
- `seg_pointnet_fold2_seed2021`: mIoU=0.8937, acc=0.9439
- `seg_pointnet_fold4_seed2021`: mIoU=0.8777, acc=0.9350

