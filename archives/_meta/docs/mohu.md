# Mohu

> Backlog derived from `docs/plan.md` + current repo state. Doc spec: `~/.codex/skills/rd-loop-orchestrator/references/docs-spec.md`.

## 1. Not Implemented

- [x] M0017: README YAML 模板兼容层：旧字段名/结构应可直接跑通
  - Ref: README 3.*（YAML 模板）, P0010, C0011
  - Context: README 中展示的 YAML 字段（如 `augment.scale_min/max`、`augment.rotate_z_deg`、root-level `precompute.*`、`features.use_normals`、SupCon 的 `augment.view1/view2`）与当前实现字段存在差异；如果照抄 README 模板会 silently fallback 到默认值，造成“README 可复制但不可用”。
  - Acceptance:
    - `scripts/train.py` 解析 config 时支持 README 模板字段的映射（不破坏现有字段）。
    - 至少覆盖：augment(scale_min/scale_max/rotate_z_deg)、loss(temperature/proj_dim)、precompute(root-level)、features(use_normals/use_curvature/use_radius/use_global_scale_token/global_scale_fields)、supcon(view1/view2)。
  - Verification:
    - `python3 scripts/train.py --config configs/raw_cls/exp/readme_template_compat_smoke.yaml --fold 0 --seed 1 --set runtime.device=cpu --set runtime.num_workers=0 --set train.epochs=1 --set train.patience=1 --set train.batch_size=8 --set data.n_points=256 --set eval.tta=1`
    - 观察 `runs/_smoke/v13_main4/readme_template_compat_smoke/pointnet/fold=0/seed=1/logs.txt` 中 `--supcon-aug-scale 0.02`、`--supcon-aug-rotate-z false`、`--point-features xyz,normals,curvature,radius`、以及 `[supcon] ... rotate_z=False scale=0.02 ...`。
  - Resolution: Implemented in `scripts/train.py` (`_normalize_augment_cfg/_normalize_features_cfg/_normalize_precompute_cfg`), plus smoke config `configs/raw_cls/exp/readme_template_compat_smoke.yaml`.

- [x] M0018: README runtime/logging/scheduler 字段落地（非占位）
  - Ref: README 3.1/3.3/3.4, P0010, C0011
  - Context: `configs/common/logging.yaml` 与 `configs/common/sched_cosine.yaml` 目前是占位并被忽略；README 模板中显式引用，期刊复现会质疑配置无效。
  - Acceptance:
    - raw_cls/domain_shift：支持 `train.grad_clip`（或 `logging.log_every`），并支持 `optim.scheduler=cosine`（可选 warmup/min_lr）。
    - runtime：支持 `runtime.deterministic` / `runtime.cudnn_benchmark` 开关（默认保持可复现）。
    - 更新 `configs/common/logging.yaml` 与 `configs/common/sched_cosine.yaml` 为真实字段（不再写“ignored”）。
  - Verification:
    - `python3 scripts/train.py --config configs/raw_cls/exp/baseline.yaml --fold 0 --seed 3 --set runtime.device=cpu --set runtime.num_workers=0 --set train.epochs=2 --set train.patience=1 --set train.batch_size=8 --set data.n_points=512`
    - 观察 `runs/raw_cls/v13_main4/baseline/dgcnn/fold=0/seed=3/logs.txt` 中 `[optim] ... scheduler=cosine ... log_every=50`、`[runtime] deterministic=...`。
  - Resolution: Added runtime/logging/scheduler args in `scripts/train.py` + `scripts/phase3_train_raw_cls_baseline.py`; updated `configs/common/{logging,sched_cosine}.yaml` and included them in `configs/raw_cls/exp/baseline.yaml`.

- [x] M0019: README 5.5/7.13：prep2target multitask_constraints（aux heads）落地
  - Ref: README 5.5, 7.13, P0010, C0011
  - Context: README 明确给出 multitask_constraints 的 DoD，但当前实现只有 CD + constraints penalty，没有 aux 预测头与相关性报告。
  - Acceptance:
    - `scripts/phase4_train_raw_prep2target_finetune.py` 增加 aux head：预测（gt）margin/occlusion proxy 标量；输出 test MAE/MSE + Pearson/Spearman。
    - 新增 `configs/prep2target/exp/multitask_constraints.yaml` 并更新 `scripts/run_full_prep2target_readme.sh`（可选跑）。
    - `metrics.json` 包含 `aux` 字段（误差+相关性），且 `paper_tables/prep2target_summary.md` 可额外给出 aux 指标（或另出一张表）。
  - Verification:
    - `python3 scripts/train.py --config configs/prep2target/exp/multitask_constraints.yaml --seed 0 --set runtime.device=cpu --set runtime.num_workers=0 --set train.epochs=2 --set train.patience=1`
    - 产物：`runs/prep2target/v1/multitask_constraints/p2t/seed=0/metrics.json` 包含 `aux.{weights,test}` 与 Pearson/Spearman；test preview 在 `runs/prep2target/v1/multitask_constraints/p2t/seed=0/previews/test/`。
  - Resolution: Implemented aux head + metrics in `scripts/phase4_train_raw_prep2target_finetune.py`, wired via `scripts/train.py`, added config + README update; `scripts/run_full_prep2target_readme.sh` 支持 `RUN_MULTITASK_CONSTRAINTS=1`。

- [x] M0014: README DoD：raw_cls/domain-shift 关键训练统计需落盘（runtime/VRAM、SupCon/对齐项、pos 缺失率等）
  - Ref: README 7.1/7.4/7.5/7.8/7.9/7.10, P0010, C0011, E0011, E0012, E0101, E0102
  - Context: README 的 DoD 要求不仅要“跑通”，还要能在产物里复现关键统计（避免审稿人质疑“算法分支没真启用/没记录”）。
  - Acceptance:
    - `scripts/phase3_train_raw_cls_baseline.py` 写入 `metrics.json`：wall time + CUDA peak VRAM（如 cuda），并补齐 `train_source/test_source`（domain-shift 时）与 feature-cache hit/miss（如启用预计算）。
    - `history.jsonl` 每 epoch 至少包含：`train_loss_ce`；SupCon 时包含 `train_loss_supcon`；CORAL 时包含 `train_loss_coral`；GroupDRO 时包含 `groupdro_weights`；DSBN/pos_moe 时包含 `domain_counts` 与 position missing/drop 比例。
    - `test_by_tooth_position` 包含 `(missing)` bucket。
  - Verification:
    - `python3 scripts/train.py --config configs/raw_cls/exp/supcon.yaml --fold 0 --seed 0 --set runtime.device=cpu --set train.epochs=2 --set train.patience=1`
    - `python3 scripts/train.py --config configs/domain_shift/exp/A2B_pos_moe.yaml --train_source 普通标注 --test_source 专家标注 --fold 0 --seed 0 --set runtime.device=cpu --set train.epochs=2 --set train.patience=1`
  - Resolution: Implemented runtime+cache+source+init-feat fields in `metrics.json`, added per-epoch DoD stats to `history.jsonl`, included `(missing)` in `test_by_tooth_position`, and fixed BN last-batch=1 crash via conditional `train_drop_last` in loaders. Smoke verified with `runs/raw_cls/v13_main4/supcon/pointnet/fold=0/seed=2/` and `runs/domain_shift/v13_main4/A2B_普通标注_to_专家标注/pos_moe/pointnet_pos_moe/fold=0/seed=1/`.

- [x] M0015: prep2target 需保存 test 预测点云用于可视化（pred_target.npy/ply）
  - Ref: P0010, C0011, E0103
  - Context: 当前 `phase4_train_raw_prep2target_finetune.py` 只保存 val 的 `preview.npz`，投稿会被质疑缺少可视化证据与可复现预览产物。
  - Acceptance: 每个 run 目录下保存至少 1 个 test 样本的预测点云（`pred_target.npy` + `pred_target.ply`，允许抽样），并带 meta（case_key/sample_npz）。
  - Verification: `python3 scripts/train.py --config configs/prep2target/exp/baseline.yaml --seed 0 --set runtime.device=cpu --set train.epochs=2 --set train.patience=1`
  - Resolution: Added `save_test_previews()` + `.ply` writer and made `scripts/train.py` always call Phase4 runner so existing runs can refresh previews without retraining. Smoke verified artifacts under `runs/prep2target/v1/baseline/p2t/seed=0/previews/test/`.

- [x] M0016: domain-shift 输出跨域掉分（Δacc/Δmacro_f1/ΔECE）相对 in-domain
  - Ref: P0010, C0011, E0102
  - Context: 仅给 A->B 绝对分数不够期刊级，需额外输出“跨域掉分”以便对比 in-domain baseline。
  - Acceptance: `scripts/run_full_domain_shift_readme.sh` 在 `paper_tables/` 额外生成 `domain_shift_delta.md`（baseline: A->B 相对 B->B 的 Δ 指标，mean±std over seeds）。
  - Verification: `bash scripts/run_full_domain_shift_readme.sh` 完成后 `test -f paper_tables/domain_shift_delta.md`
  - Resolution: Implemented in `scripts/domain_shift_delta_table.py` and wired in `scripts/run_full_domain_shift_readme.sh` (generates `paper_tables/domain_shift_delta.md`).

- [x] M0013: ablation YAML + postprocess 需 smoke 验证通过
  - Ref: P0010, C0011, E0011, E0012
  - Context: configs 中新增/补齐了多个 ablation 配置与后处理（temp scaling / selective），需要至少一轮 smoke 验证确保可跑、产物齐全、不会因旧 run 跳过而漏掉后处理产物。
  - Acceptance: `python3 scripts/train.py --config configs/raw_cls/exp/selective.yaml --fold 0 --seed 0 --set runtime.device=cpu --set train.epochs=3 --set train.patience=1` 生成 `metrics.json` + `calib.json` + `reliability.png` + `selective.md`；`python3 scripts/train.py --config configs/domain_shift/exp/A2B_pos_moe.yaml --train_source 普通标注 --test_source 专家标注 --fold 0 --seed 0 --set runtime.device=cpu --set train.epochs=3 --set train.patience=1` 生成 `metrics.json`。
  - Verification: `python3 scripts/docs_audit.py --root . --out docs/DOCS_AUDIT.md`
  - Resolution: Ran both smoke commands successfully; artifacts present under `runs/raw_cls/v13_main4/selective/` and `runs/domain_shift/v13_main4/A2B_普通标注_to_专家标注/pos_moe/`.

- [x] M0011: raw_cls 增加 traditional geometric baseline（geom_mlp）并完成 k-fold×multi-seed full runs
  - Ref: P0008, C0009, E0009
  - Context: 期刊审稿常要求“传统特征/非深度”基线对照；当前主要是 PointNet/DGCNN + metadata 变体，不够完整。
  - Acceptance: `scripts/phase3_train_raw_cls_baseline.py` 支持 `--model geom_mlp`；`paper_tables/raw_cls_table_v13.md` 出现 geom_mlp 行且 `n=15`。
  - Verification: `CUDA_VISIBLE_DEVICES=0 python3 scripts/run_raw_cls_kfold.py --root . --data-root processed/raw_cls/v13_main4 --kfold metadata/splits_raw_case_kfold.json --models geom_mlp --seeds 1337,2020,2021 --folds all --device cuda --epochs 120 --patience 25 --batch-size 64 --n-points 4096 --balanced-sampler --label-smoothing 0.1 --tta 8`
  - Resolution: Added `GeomMLPClassifier` in `scripts/phase3_train_raw_cls_baseline.py`, ran k=5×seeds=3 full, and refreshed `paper_tables/raw_cls_table_v13.md` (geom_mlp n=15).

- [x] M0012: raw_cls domain-shift 实验（普通标注↔专家标注）需要落地并输出表格
  - Ref: P0009, C0010, E0010
  - Context: 目前只有 by_source 指标/校准报告；期刊更偏好 train-on-A test-on-B 的显式域泛化评估。
  - Acceptance: `scripts/phase3_train_raw_cls_baseline.py` 支持 `--source-train/--source-test` 覆写 split；完成两种方向（普通→专家、专家→普通）pointnet/dgcnn 的 ≥3 seeds runs；生成 `paper_tables/raw_cls_domain_shift_table_v13.md`。
  - Verification: `python3 scripts/run_raw_cls_domain_shift.py --root . --data-root processed/raw_cls/v13_main4 --models pointnet,dgcnn --seeds 1337,2020,2021 --train-source 普通标注 --test-source 专家标注 --device cuda --epochs 120 --patience 25 --batch-size 32 --n-points 4096 --balanced-sampler --label-smoothing 0.1 --tta 8 && python3 scripts/run_raw_cls_domain_shift.py --root . --data-root processed/raw_cls/v13_main4 --models pointnet,dgcnn --seeds 1337,2020,2021 --train-source 专家标注 --test-source 普通标注 --device cuda --epochs 120 --patience 25 --batch-size 32 --n-points 4096 --balanced-sampler --label-smoothing 0.1 --tta 8 && python3 scripts/paper_raw_cls_domain_shift_table.py --runs-dir runs/raw_cls_domain_shift --out paper_tables/raw_cls_domain_shift_table_v13.md --data-tag v13_main4`
  - Resolution: Implemented source override splitting + runner/table scripts, ran both directions for pointnet/dgcnn (seeds=3), and generated `paper_tables/raw_cls_domain_shift_table_v13.md`.

- [x] M0007: raw_cls 表格聚合需去重（避免重复 run 污染 n/均值）
  - Ref: P0005, C0006
  - Context: `runs/raw_cls_baseline/` 中可能存在同一 (seed, fold, config) 的重复目录（例如旧 exp_name/tag），会导致 `paper_tables/raw_cls_table_v13.md` 的 `n` 错误（如 16 而非 15）。
  - Acceptance: `scripts/paper_table_raw_cls.py` 对 k-fold runs 按 (seed,test_fold) 去重（保留最新/有效项），并在表格/日志中给出 duplicates 警告；输出表中 v13_main4 的 meta-feature 行 pointnet/dgcnn 均满足 `n = 5 × 3`。
  - Verification: `python3 scripts/paper_table_raw_cls.py --runs-dir runs/raw_cls_baseline --out paper_tables/raw_cls_table_v13.md --data-tag v13_main4 --kfold-only`
  - Resolution: Updated `scripts/paper_table_raw_cls.py` to dedup by (seed,test_fold) and regenerated `paper_tables/raw_cls_table_v13.md` (meta-feature rows n=15).

- [x] M0008: raw_cls k-fold merged 报告需补齐 paired bootstrap diff（关键对照显著性）
  - Ref: P0006, C0007
  - Context: 当前 merged 报告已包含 CI + by_source，但缺少“方法差异”的 paired diff/显著性输出，期刊审稿常要求 effect size。
  - Acceptance: `scripts/paper_raw_cls_kfold_merged_report.py` 输出 `comparisons`（至少包含 pointnet vs dgcnn、meta vs none 的 macro_f1 diff 的 CI/p-value），并在 markdown 中增加对应表格段落。
  - Verification: `python3 scripts/paper_raw_cls_kfold_merged_report.py --runs-dir runs/raw_cls_baseline --out-prefix paper_tables/raw_cls_kfold_merged_report_v13_main4 --data-tag v13_main4`
  - Resolution: Added hierarchical paired bootstrap comparisons to `scripts/paper_raw_cls_kfold_merged_report.py` and regenerated `paper_tables/raw_cls_kfold_merged_report_v13_main4.{md,json}`.

- [x] M0009: raw_cls 增加 meta-only baseline（仅用元特征，不读点云）
  - Ref: P0007, C0008, E0008
  - Context: meta-feature augmented baseline 可能引入 shortcut；需要用 meta-only 模型量化“元信息本身能达到的上限”，让论文论证更稳健。
  - Acceptance: `scripts/phase3_train_raw_cls_baseline.py` 支持 `--model meta_mlp`，并可通过 `scripts/run_raw_cls_kfold.py` 跑通 k=5×seeds=3；更新 `paper_tables/raw_cls_table_v13.md` 增加 meta_mlp 行（extra_features=`scale,log_points,objects_used`，`n = 5 × 3`）。
  - Verification: `CUDA_VISIBLE_DEVICES=0 python3 scripts/run_raw_cls_kfold.py --root . --data-root processed/raw_cls/v13_main4 --kfold metadata/splits_raw_case_kfold.json --models meta_mlp --seeds 1337,2020,2021 --folds all --device cuda --epochs 120 --patience 25 --batch-size 64 --n-points 0 --balanced-sampler --label-smoothing 0.1 --tta 0 --extra-features scale,log_points,objects_used`
  - Resolution: Added `MetaMLPClassifier` + `load_points=False` path in `scripts/phase3_train_raw_cls_baseline.py`, ran E0008 full, and refreshed `paper_tables/raw_cls_table_v13.md`.

- [x] M0010: 增补环境级复现材料（environment.yml）
  - Ref: C0005
  - Context: `requirements.txt` 仅提供最小依赖；期刊复现更偏好 conda 环境文件或容器文件。
  - Acceptance: 新增 `environment.yml`（建议 python/torch/numpy/pip），并在 `PAPER_PROTOCOL.md` 指出如何使用；`scripts/paper_audit.py` 与 `scripts/plan_proof.py --strict` 通过。
  - Verification: `test -f environment.yml && python3 scripts/paper_audit.py --root . --out PAPER_AUDIT.md && python3 scripts/plan_proof.py --root . --plan docs/plan.md --experiment docs/experiment.md --out docs/proof.md --strict`
  - Resolution: Added `environment.yml`, updated `PAPER_PROTOCOL.md`, and refreshed `PAPER_AUDIT.md`/`docs/proof.md`.

- [x] M0006: raw_cls 需要补齐 k-fold merged 报告（按 seed 合并全量 test + bootstrap CI + 按 source 分域）
  - Ref: P0006, C0007, E0007
  - Context: 目前主要表格按“run=fold×seed”聚合；期刊更偏好按 seed 合并全量样本后再做 CI/分域统计，避免 fold 独立性争议。
  - Acceptance: 生成 `paper_tables/raw_cls_kfold_merged_report_v13_main4.md` 与 `paper_tables/raw_cls_kfold_merged_report_v13_main4.json`，包含 pointnet/dgcnn 的 overall+by_source+CI。
  - Verification: `python3 scripts/paper_raw_cls_kfold_merged_report.py --runs-dir runs/raw_cls_baseline --out-prefix paper_tables/raw_cls_kfold_merged_report_v13_main4 --data-tag v13_main4`
  - Resolution: Added `scripts/paper_raw_cls_kfold_merged_report.py` and generated the report artifacts under `paper_tables/`.

- [x] M0005: raw_cls meta-feature augmented baseline（scale/log_points/objects_used）需要补齐 k-fold×multi-seed full runs + 更新表格
  - Ref: P0005, C0006, E0006
  - Context: 现有 point-only baseline macro-F1 偏弱；加入可审计元信息（如 `scale`、点数、objects_used）可能提升区分度，需要期刊级完整 runs 支撑。
  - Acceptance: `paper_tables/raw_cls_table_v13.md` 中 pointnet/dgcnn 的 extra_features=`scale,log_points,objects_used` 行 `n = 5 × 3`；对应 runs 目录均有 `metrics.json`。
  - Verification: `python3 scripts/paper_table_raw_cls.py --runs-dir runs/raw_cls_baseline --out paper_tables/raw_cls_table_v13.md --data-tag v13_main4 --kfold-only`
  - Resolution: Ran E0006 full (k=5 × seeds=3 for PointNet/DGCNN) and refreshed `paper_tables/raw_cls_table_v13.md`.

- [x] M0001: raw_cls 需要补齐 multi-seed k-fold full runs（否则 C0001/C0002 只是单种子）
  - Ref: P0002, C0001, C0002, E0001, E0002
  - Context: 当前仅完成 `seed=1337` 的 k=5（pointnet/dgcnn），需要 ≥3 seeds 才能给出期刊可接受的方差/CI。
  - Acceptance: `paper_tables/raw_cls_table_v13.md` 中 pointnet/dgcnn 的 k-fold 组 `n = 5 × 3`；并生成至少 1 个 CI JSON + 1 个 paired diff JSON。
  - Verification: `python3 scripts/paper_table_raw_cls.py --runs-dir runs/raw_cls_baseline --out paper_tables/raw_cls_table_v13.md --data-tag v13_main4 --kfold-only`
  - Resolution: Completed seeds=1337,2020,2021 × k=5 for PointNet/DGCNN; refreshed `paper_tables/raw_cls_table_v13.md` (n=15) + generated `paper_tables/raw_cls_paired_dgcnn_vs_pointnet_fold0_boot5000.json`.

- [x] M0002: 补齐 docs/experiment.md 台账（E####）并提供可跑的 smoke/full 命令
  - Ref: P0004, C0005
  - Context: 目前 `docs/experiment.md` 还是模板；缺少实验表格化记录与可恢复队列（.rd_queue）。
  - Acceptance: `docs/experiment.md` 至少包含 raw_cls 与 constraints 的实验条目（含 smoke/full 命令、指标、资源估计、产物路径）。
  - Verification: `python3 scripts/docs_audit.py --root . --out docs/DOCS_AUDIT.md`
  - Resolution: Filled `docs/experiment.md` with E0001–E0005 and verified via `docs/DOCS_AUDIT.md`.

- [x] M0003: proof 闸门（claims→evidence）未落地：需要基于实验产物勾选 C####/P####
  - Ref: C0001, C0002, C0003, C0004, C0005, P0001, P0002, P0003, P0004
  - Context: 当前有 `paper_tables/` 与 `*_AUDIT.md`，但 `docs/plan.md` 的 checkbox 还未按证据更新。
  - Acceptance: `docs/plan.md` 中可证明的 C####/P#### 被勾选；未证明项明确标注缺口与下一步实验（E####）。
  - Verification: `python3 scripts/plan_proof.py --root . --plan docs/plan.md --experiment docs/experiment.md --out docs/proof.md`
  - Resolution: Added `scripts/plan_proof.py` + generated `docs/proof.md`; updated `docs/plan.md` checkboxes per proof.

## 2. Ambiguities

- [x] M0004: 约束指标的单位/坐标系需要在论文协议中写清（避免被解读为临床 mm/μm）
  - Ref: P0003, C0004
  - Context: constraints 的 `min_d`/clearance 在不同 stage 可能处于 normalized space 或 raw space；若不说明，审稿会质疑临床意义。
  - Acceptance: 在 `PAPER_PROTOCOL.md` 与 `docs/experiment.md` 中明确写出指标计算空间、clearance 的单位与局限性。
  - Verification: `rg -n \"(normalized space|OBJ space|OBJ coordinate)\" PAPER_PROTOCOL.md docs/experiment.md`
  - Resolution: Added explicit metric space notes to `PAPER_PROTOCOL.md` and `docs/experiment.md` (see E0004 row).

## Resolved (optional)
