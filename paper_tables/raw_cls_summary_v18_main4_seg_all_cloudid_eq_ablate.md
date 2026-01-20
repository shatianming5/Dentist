# Aggregate runs (raw_cls)

- generated_at: 2026-01-18T04:23:04Z
- root: `/home/ubuntu/tiasha/dentist/runs/raw_cls/v18_main4_seg_all_cloudid_eq`
- runs: 30
- seed_filter: [1337, 2020, 2021]
- fold_filter: [0, 1, 2, 3, 4]

- groups: 2

| test_macro_f1 (mean±std) | test_acc (mean±std) | test_bal_acc (mean±std) | test_ece (mean±std) | n | dataset | domain | exp | model | point_features | extra_features | sampler | label_smoothing | tta |
|---:|---:|---:|---:|---:|---|---|---|---|---|---|---|---|---:|
| 0.4553±0.0651 | 0.4852±0.0470 | 0.4666±0.0701 | 0.3278±0.0648 | 15 | v18_main4_seg_all_cloudid_eq | - | teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_dropout0p1_seg_all_cloudid_eq | pointnet | xyz,cloud_id | tooth_position_premolar,tooth_position_molar,tooth_position_missing | bal | ls=0 | 0 |
| 0.4049±0.0579 | 0.4523±0.0576 | 0.4263±0.0551 | 0.3225±0.0632 | 15 | v18_main4_seg_all_cloudid_eq | - | teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_dropout0p1_seg_all_cloudid_eq_meta | pointnet | xyz,cloud_id | tooth_position_premolar,tooth_position_molar,tooth_position_missing,scale,log_points,objects_used | bal | ls=0 | 0 |
