# Aggregate runs (raw_cls)

- generated_at: 2026-01-18T05:03:03Z
- root: `/home/ubuntu/tiasha/dentist/runs/raw_cls/v20_main4_seg_all_cloudid_eq_pool16384`
- runs: 15
- seed_filter: [1337, 2020, 2021]
- fold_filter: [0, 1, 2, 3, 4]

- groups: 1

| test_macro_f1 (mean±std) | test_acc (mean±std) | test_bal_acc (mean±std) | test_ece (mean±std) | n | dataset | domain | exp | model | point_features | extra_features | sampler | label_smoothing | tta |
|---:|---:|---:|---:|---:|---|---|---|---|---|---|---|---|---:|
| 0.4503±0.0686 | 0.4657±0.0691 | 0.4534±0.0645 | 0.3217±0.0643 | 15 | v20_main4_seg_all_cloudid_eq_pool16384 | - | teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_dropout0p1_seg_all_cloudid_eq_pool16384 | pointnet | xyz,cloud_id | tooth_position_premolar,tooth_position_molar,tooth_position_missing | bal | ls=0 | 0 |
