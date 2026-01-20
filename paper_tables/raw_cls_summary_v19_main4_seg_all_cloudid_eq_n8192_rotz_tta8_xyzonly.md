# Aggregate runs (raw_cls)

- generated_at: 2026-01-18T09:39:18Z
- root: `/home/ubuntu/tiasha/dentist/runs/raw_cls/v19_main4_seg_all_cloudid_eq_n8192`
- runs: 30
- seed_filter: [1337, 2020, 2021]
- fold_filter: [0, 1, 2, 3, 4]

- groups: 2

| test_macro_f1 (mean±std) | test_acc (mean±std) | test_bal_acc (mean±std) | test_ece (mean±std) | n | dataset | domain | exp | model | point_features | extra_features | sampler | label_smoothing | tta |
|---:|---:|---:|---:|---:|---|---|---|---|---|---|---|---|---:|
| 0.4709±0.0932 | 0.5119±0.0759 | 0.5014±0.0841 | 0.1907±0.0635 | 15 | v19_main4_seg_all_cloudid_eq_n8192 | - | teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1_seg_all_cloudid_eq_n8192_xyzonly | pointnet | xyz | tooth_position_premolar,tooth_position_molar,tooth_position_missing | bal | ls=0 | 8 |
| 0.4361±0.0812 | 0.4656±0.0768 | 0.4540±0.0613 | 0.3292±0.0952 | 15 | v19_main4_seg_all_cloudid_eq_n8192 | - | teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_dropout0p1_seg_all_cloudid_eq_n8192 | pointnet | xyz,cloud_id | tooth_position_premolar,tooth_position_molar,tooth_position_missing | bal | ls=0 | 0 |
