# Aggregate runs (raw_cls)

- generated_at: 2026-01-19T17:25:02Z
- root: `/home/ubuntu/tiasha/dentist/runs/raw_cls/v22_main4_seg_small3_cloudid_eq`
- runs: 15
- seed_filter: [1337, 2020, 2021]
- fold_filter: [0, 1, 2, 3, 4]

- groups: 1

| test_macro_f1 (mean±std) | test_acc (mean±std) | test_bal_acc (mean±std) | test_ece (mean±std) | n | dataset | domain | exp | model | point_features | extra_features | sampler | label_smoothing | tta |
|---:|---:|---:|---:|---:|---|---|---|---|---|---|---|---|---:|
| 0.4185±0.0745 | 0.4777±0.0852 | 0.4488±0.0702 | 0.2401±0.1072 | 15 | v22_main4_seg_small3_cloudid_eq | - | teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1_seg_small3_cloudid_eq_xyzonly | pointnet | xyz | tooth_position_premolar,tooth_position_molar,tooth_position_missing | bal | ls=0 | 8 |
