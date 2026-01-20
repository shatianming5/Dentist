# raw_cls meta shortcut audit (source separability)

- data_root: `/home/ubuntu/tiasha/dentist/processed/raw_cls/v13_main4`
- rows: 248
- sources: ['专家标注', '普通标注']
- pos_source (AUC=1): `专家标注`

## scale

| source | n | mean±std | p05 | p50 | p95 |
|---|---:|---:|---:|---:|---:|
| 专家标注 | 83 | 4.152±0.9522 | 2.602 | 4.133 | 5.62 |
| 普通标注 | 165 | 4.151±1.064 | 2.454 | 4.183 | 5.923 |

## log1p(n_points_after_cap)

| source | n | mean±std | p05 | p50 | p95 |
|---|---:|---:|---:|---:|---:|
| 专家标注 | 83 | 9.858±0.6141 | 8.866 | 9.899 | 10.79 |
| 普通标注 | 165 | 9.763±0.6746 | 8.685 | 9.975 | 10.67 |

## n_objects_used

| source | n | mean±std | p05 | p50 | p95 |
|---|---:|---:|---:|---:|---:|
| 专家标注 | 83 | 1±0 | 1 | 1 | 1 |
| 普通标注 | 165 | 1±0 | 1 | 1 | 1 |

## AUC (source separability) — all splits

| feature | AUC |
|---|---:|
| scale | 0.5004 |
| log1p(n_points_after_cap) | 0.5333 |
| n_objects_used | 0.5000 |

## AUC (source separability) — split=train

| feature | AUC |
|---|---:|
| scale | 0.4879 |
| log1p(n_points_after_cap) | 0.5187 |
| n_objects_used | 0.5000 |

## AUC (source separability) — split=val

| feature | AUC |
|---|---:|
| scale | 0.5521 |
| log1p(n_points_after_cap) | 0.5312 |
| n_objects_used | 0.5000 |

## AUC (source separability) — split=test

| feature | AUC |
|---|---:|
| scale | 0.5800 |
| log1p(n_points_after_cap) | 0.6100 |
| n_objects_used | 0.5000 |

