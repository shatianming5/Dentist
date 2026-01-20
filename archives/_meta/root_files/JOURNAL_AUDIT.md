# Journal audit report

- generated_at: 2026-01-16T17:43:59Z
- root: `/home/ubuntu/tiasha/dentist`
- python: `3.13.2`
- platform: `Linux-6.12.0-124.21.1.el10_1.x86_64-x86_64-with-glibc2.39`
- summary: ok=12 warn=0 fail=0

| check | status | notes |
|---|---|---|
| `teeth3ds_patient_splits` | ok | path: /home/ubuntu/tiasha/dentist/metadata/splits_teeth3ds.json<br>patients: train=369 val=40 test=491 |
| `teeth3ds_processed_index` | ok | path: /home/ubuntu/tiasha/dentist/processed/teeth3ds_teeth/v1/index.jsonl<br>rows: {'train': 9815, 'val': 1041, 'test': 13058, 'unknown': 0}<br>patients: train=369 val=40 test=491 unknown=0 |
| `raw_cls_dataset` | ok | path: /home/ubuntu/tiasha/dentist/processed/raw_cls/v13_main4<br>splits: {'train': 196, 'val': 22, 'test': 30, 'unknown': 0}<br>labels: ['充填', '全冠', '桩核冠', '高嵌体'] |
| `raw_kfold_file` | ok | path: /home/ubuntu/tiasha/dentist/metadata/splits_raw_case_kfold.json<br>k=5<br>case_to_fold=253 folds=5 |
| `phase3_raw_cls_baseline_features` | ok | path: /home/ubuntu/tiasha/dentist/scripts/phase3_train_raw_cls_baseline.py |
| `raw_ae_pretrain_script` | ok | path: /home/ubuntu/tiasha/dentist/scripts/phase2_train_raw_ae.py |
| `pca_align_globalz_option` | ok | path: /home/ubuntu/tiasha/dentist/scripts/phase2_build_teeth3ds_teeth.py |
| `prep2target_cut_mode_plane` | ok | path: /home/ubuntu/tiasha/dentist/scripts/phase3_train_teeth3ds_prep2target.py |
| `constraints_training_features` | ok | path: /home/ubuntu/tiasha/dentist/scripts/phase4_train_teeth3ds_prep2target_constraints.py |
| `constraints_eval_features` | ok | path: /home/ubuntu/tiasha/dentist/scripts/phase4_eval_teeth3ds_constraints_run.py |
| `diagnostic_scripts` | ok | ok: 4 scripts |
| `requirements_txt` | ok | path: /home/ubuntu/tiasha/dentist/requirements.txt |
