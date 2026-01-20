# Aggregate runs (raw_cls)

- generated_at: 2026-01-18T03:07:47Z
- root: `/home/ubuntu/tiasha/dentist/runs/raw_cls/v17_main4_seg_all_cloudid`
- runs: 15
- seed_filter: [1337, 2020, 2021]
- fold_filter: [0, 1, 2, 3, 4]

- groups: 1

| test_macro_f1 (mean±std) | test_acc (mean±std) | test_bal_acc (mean±std) | test_ece (mean±std) | n | dataset | domain | exp | model | point_features | extra_features | sampler | label_smoothing | tta |
|---:|---:|---:|---:|---:|---|---|---|---|---|---|---|---|---:|
| 0.4393±0.0498 | 0.4711±0.0502 | 0.4537±0.0570 | 0.3251±0.0511 | 15 | v17_main4_seg_all_cloudid | - | teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_dropout0p1_seg_all_cloudid | pointnet | xyz,cloud_id | tooth_position_premolar,tooth_position_molar,tooth_position_missing | bal | ls=0 | 0 |
