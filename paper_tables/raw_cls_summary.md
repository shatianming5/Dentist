# Aggregate runs (raw_cls)

- generated_at: 2026-03-16T20:31:42Z
- root: `/mnt/SSD_4TB/zechuan/Dentist/runs/raw_cls/v13_main4`
- runs: 30
- seed_filter: [1337, 2020, 2021]
- fold_filter: [0, 1, 2, 3, 4]

- groups: 2

| test_macro_f1 (mean±std) | test_acc (mean±std) | test_bal_acc (mean±std) | test_ece (mean±std) | n | dataset | domain | exp | model | point_features | extra_features | sampler | label_smoothing | tta |
|---:|---:|---:|---:|---:|---|---|---|---|---|---|---|---|---:|
| 0.1376±0.0737 | 0.2867±0.1391 | 0.2704±0.0769 | 0.1815±0.1298 | 15 | v13_main4 | - | baseline | pointnet | xyz | (none) | bal | ls=0.1 | 8 |
| 0.1364±0.0664 | 0.3344±0.1951 | 0.2464±0.0549 | 0.2607±0.1467 | 15 | v13_main4 | - | baseline | dgcnn | xyz | (none) | bal | ls=0.1 | 8 |
