# Plan

## Goals

- Prepare this repository for a **CS/AI journal submission** with:
  - paper claims aligned to the current implementation,
  - reproducible experimental protocol (k-fold × multi-seed, CIs),
  - journal-grade reporting artifacts (tables, audits, run logs).
- Scope constraint: **do not change data/labels/splits or licensing** in this loop (data-related items are documented but deferred).
  - Data-related risks (sample size, label schema/imbalance, extraction heuristics, licensing) are tracked as out-of-scope for now.

## Claims (C####)

- [x] C0001: raw_cls 提供期刊级可复现实验：k-fold × multi-seed + mean±std 表 + bootstrap CI/paired diff。
  - Evidence: E0001, E0002
  - Proof rule: `paper_tables/raw_cls_table_v13.md` 里 pointnet/dgcnn 的主设置（extra_features=`(none)`）k-fold 组 `n = k × seeds`，且 `paper_tables/raw_cls_ci_*.json` 存在。
  - Notes: 数据小导致方差大，但“协议 + 置信区间”可被审稿接受。

- [x] C0002: raw_cls 明确报告域差异与校准：by source/tooth_position 的 accuracy/macro-F1 + ECE/NLL/Brier。
  - Evidence: E0001
  - Proof rule: 运行产物 `runs/raw_cls_baseline/*/metrics.json` 包含 `test_by_source_calibration`、`test_by_tooth_position`、`test_calibration`。
  - Notes: 该 claim 不要求域泛化方法，只要求“可审计的分域报告”。

- [x] C0003: Teeth3DS prep→target 在论文中被界定为 **synthetic proxy**（非真实临床 prep→crown），避免过度主张。
  - Evidence: E0003
  - Proof rule: `PAPER_SCOPE.md` 明确区分 implemented vs proposed，且 `plan_report.md` 顶部包含 scope 提示。
  - Notes: 这是期刊审稿常见扣分点（叙事大于贡献）的防御性改动。

- [x] C0004: 约束评估指标更敏感且可解释：输出 `occlusion_contact_ratio` + `min_d` 分位数，并在汇总表中呈现。
  - Evidence: E0004
  - Proof rule: `paper_tables/constraints_summary.md` 有 `min_d_p05/p50/p95` 列且对应 `eval_{val,test}.json` 存在。
  - Notes: 该 proxy 仍非临床金标准，但比 `occ_pen_mean` 更可区分。

- [x] C0005: 复现资料达到“期刊视角”：统一命令、环境记录、审计报告、实验台账与队列产物可回溯。
  - Evidence: E0005
  - Proof rule: `docs/experiment.md` + `.rd_queue/{logs,results,queue.json}` + `PAPER_PROTOCOL.md` + `JOURNAL_AUDIT.md` + `environment.yml` 均存在且能一键生成。
  - Notes: raw_cls multi-seed k-fold full runs 已补齐；其余 full 训练预算可按期刊需求再扩展。

- [x] C0006: raw_cls 增加 “meta-feature augmented” baseline（scale/log_points/objects_used）并给出 k-fold × multi-seed 报告。
  - Evidence: E0006
  - Proof rule: `paper_tables/raw_cls_table_v13.md` 包含 extra_features=`scale,log_points,objects_used` 的 pointnet/dgcnn 行且 `n = 5 × 3`（去重后严格等于）。
  - Notes: 该 baseline 仅使用 Phase1 已记录的元信息（不改动 labels/splits）。

- [x] C0007: raw_cls 提供 k-fold merged 报告：按 seed 合并全量 test cases，输出 bootstrap CI + 分域（source）指标/校准 + 关键对照的 paired bootstrap diff（同时包含 macro_f1_all / balanced_acc 等补充指标）。
  - Evidence: E0007
  - Proof rule: `paper_tables/raw_cls_kfold_merged_report_v13_main4.{md,json}` 存在且 json 中包含 pointnet/dgcnn 结果与 `comparisons`（macro_f1 diff 的 CI/p-value）。
  - Notes: 该报告避免把每个 fold 当作独立样本，更便于期刊解释。

- [x] C0008: raw_cls 增加 “meta-only” baseline（仅用元特征，不读点云）以审计 shortcut 风险，并给出 k-fold × multi-seed 报告。
  - Evidence: E0008
  - Proof rule: `paper_tables/raw_cls_table_v13.md` 包含 model=`meta_mlp` 且 extra_features=`scale,log_points,objects_used` 的 k-fold 组 `n = 5 × 3`，并且对应 runs 目录均有 `metrics.json`。
  - Notes: 该 baseline 用于回答“元信息是否已经足够区分”，避免论文被质疑存在 shortcut。

- [x] C0009: raw_cls 增加 “traditional geometric” baseline（geom_mlp）并入 k-fold × multi-seed 表，补齐期刊级对照。
  - Evidence: E0009
  - Proof rule: `paper_tables/raw_cls_table_v13.md` 包含 model=`geom_mlp` 且 extra_features=`(none)` 的 k-fold 组 `n = 5 × 3`。
  - Notes: 该 baseline 只用点云的统计几何特征（非深层特征），用于满足“传统特征基线”对照需求。

- [x] C0010: raw_cls 补齐 domain-shift 实验：train-on-A test-on-B（普通标注↔专家标注），并报告 multi-seed mean±std + 校准指标。
  - Evidence: E0010
  - Proof rule: `paper_tables/raw_cls_domain_shift_table_v13.md` 存在，且包含两种方向（A→B, B→A）的 pointnet/dgcnn 行（seeds≥3）。
  - Notes: 数据仍小，但该实验可以回答“域差异是否影响泛化/校准”，比单纯 by_source 报告更接近期刊期望。

- [x] C0011: 仓库提供的 ablation 配置（scale_token/supcon/temp_scaling/selective + domain_shift groupdro/coral/dsbn/pos_moe）全部可运行，并在 runs 树中产出期刊可审计的附加产物（`calib.json`/`reliability.png`/`selective.{md,json}`）。
  - Evidence: E0011
  - Proof rule: 对应 `configs/**/exp/*.yaml` 均存在；至少 1 个 smoke run 目录内包含 `metrics.json` 且（当配置启用时）包含 `calib.json`/`reliability.png`/`selective.md`。
  - Notes: 该 claim 先保证“可跑+可审计”；性能提升另起 claim（避免炼丹叙事）。

- [x] C0012: Teeth3DS 单牙 FDI 判别式预训练作为 init，可显著提升 raw_cls 主任务 PointNet（k-fold×multi-seed）。
  - Evidence: E0104（以及 E0101 baseline/supcon 对照）
  - Proof rule: `paper_tables/raw_cls_summary.md` 中 `teeth3ds_fdi_pretrain_finetune` 的 mean macro_f1/acc 高于 `baseline` 与 `supcon` 的 PointNet 行（seed=1337,2020,2021; fold=0..4）。
  - Notes: 该提升不涉及 raw 数据/标签修改；仅引入外部 Teeth3DS 的牙体形状先验（无 test/val 泄漏）。

## Plan Items (P####)

- [x] P0001: 建立 journal-safe paper scope 与协议文档（避免叙事 > 贡献）
  - Linked claims: C0003, C0005
  - Definition of done: `PAPER_SCOPE.md` / `PAPER_PROTOCOL.md` / `PAPER_AUDIT.md` 一致且通过审计。
  - Verification: `python3 scripts/paper_audit.py --root . --out PAPER_AUDIT.md`
  - Touchpoints: `PAPER_SCOPE.md`, `PAPER_PROTOCOL.md`, `plan_report.md`, `scripts/paper_audit.py`

- [x] P0002: raw_cls 期刊级实验协议落地（k-fold×multi-seed + CI + 表格）
  - Linked claims: C0001, C0002, C0005
  - Definition of done: 能跑 smoke/full；产物落在 `runs/raw_cls_baseline/` + `paper_tables/`。
  - Verification: `python3 scripts/paper_table_raw_cls.py --runs-dir runs/raw_cls_baseline --out paper_tables/raw_cls_table_v13.md --data-tag v13_main4 --kfold-only`
  - Touchpoints: `scripts/run_raw_cls_kfold.py`, `scripts/phase3_train_raw_cls_baseline.py`, `scripts/raw_cls_bootstrap_ci.py`, `scripts/paper_table_raw_cls.py`

- [x] P0003: 约束评估与汇总表升级（min_d 分位数/对颌 tooth-mode 支持）
  - Linked claims: C0004, C0005
  - Definition of done: `eval_{val,test}.json` 产出包含 `min_d` 分位数；汇总表可直接用于论文。
  - Verification: `python3 scripts/phase4_summarize_constraints_runs.py --runs-dir runs/teeth3ds_prep2target_constraints --out-prefix paper_tables/constraints_summary`
  - Touchpoints: `scripts/phase4_eval_teeth3ds_constraints_run.py`, `scripts/run_constraints_eval_suite.py`, `scripts/phase4_summarize_constraints_runs.py`

- [x] P0004: 按 docs-spec 建立 plan→mohu→experiment→proof 闭环文档与队列产物
  - Linked claims: C0005
  - Definition of done: `docs/plan.md` / `docs/mohu.md` / `docs/experiment.md` 完整；smoke 任务使用 `.rd_queue/` 可恢复。
  - Verification: `test -f docs/plan.md && test -f docs/mohu.md && test -f docs/experiment.md`
  - Touchpoints: `docs/plan.md`, `docs/mohu.md`, `docs/experiment.md`, `.rd_queue/`

- [x] P0005: raw_cls meta-feature baseline 跑通并入表（k-fold × multi-seed）
  - Linked claims: C0006, C0001
  - Definition of done: `runs/raw_cls_baseline/` 中 pointnet/dgcnn 生成 `k × seeds` runs（extra_features=`scale,log_points,objects_used`）；`paper_tables/raw_cls_table_v13.md` 更新并包含新行。
  - Verification: `python3 scripts/paper_table_raw_cls.py --runs-dir runs/raw_cls_baseline --out paper_tables/raw_cls_table_v13.md --data-tag v13_main4 --kfold-only`
  - Touchpoints: `scripts/run_raw_cls_kfold.py`, `scripts/phase3_train_raw_cls_baseline.py`, `scripts/paper_table_raw_cls.py`

- [x] P0006: raw_cls k-fold merged 报告（CI + 分域）
  - Linked claims: C0007, C0001, C0002
  - Definition of done: 生成 `paper_tables/raw_cls_kfold_merged_report_v13_main4.{md,json}` 并可一键复现。
  - Verification: `python3 scripts/paper_raw_cls_kfold_merged_report.py --runs-dir runs/raw_cls_baseline --out-prefix paper_tables/raw_cls_kfold_merged_report_v13_main4 --data-tag v13_main4`
  - Touchpoints: `scripts/paper_raw_cls_kfold_merged_report.py`

- [x] P0007: raw_cls meta-only baseline（k-fold × multi-seed）并入表
  - Linked claims: C0008, C0005
  - Definition of done: `scripts/phase3_train_raw_cls_baseline.py` 支持 `--model meta_mlp`（不加载点云）；完成 seeds=1337,2020,2021 × k=5 full runs 并更新 `paper_tables/raw_cls_table_v13.md`。
  - Verification: `python3 scripts/paper_table_raw_cls.py --runs-dir runs/raw_cls_baseline --out paper_tables/raw_cls_table_v13.md --data-tag v13_main4 --kfold-only`
  - Touchpoints: `scripts/phase3_train_raw_cls_baseline.py`, `scripts/run_raw_cls_kfold.py`, `paper_tables/raw_cls_table_v13.md`

- [x] P0008: raw_cls geom_mlp baseline（k-fold × multi-seed）并入表
  - Linked claims: C0009
  - Definition of done: `scripts/phase3_train_raw_cls_baseline.py` 支持 `--model geom_mlp`；完成 seeds=1337,2020,2021 × k=5 full runs 并更新 `paper_tables/raw_cls_table_v13.md`。
  - Verification: `CUDA_VISIBLE_DEVICES=0 python3 scripts/run_raw_cls_kfold.py --root . --data-root processed/raw_cls/v13_main4 --kfold metadata/splits_raw_case_kfold.json --models geom_mlp --seeds 1337,2020,2021 --folds all --device cuda --epochs 120 --patience 25 --batch-size 64 --n-points 4096 --balanced-sampler --label-smoothing 0.1 --tta 8`
  - Touchpoints: `scripts/phase3_train_raw_cls_baseline.py`, `scripts/run_raw_cls_kfold.py`, `paper_tables/raw_cls_table_v13.md`

- [x] P0009: raw_cls domain-shift 训练/评估协议与表格
  - Linked claims: C0010
  - Definition of done: 支持按 source 覆写 split（train/val/test），并跑通普通标注↔专家标注双向 domain shift 的 pointnet/dgcnn（≥3 seeds），输出 `paper_tables/raw_cls_domain_shift_table_v13.md`。
  - Verification: `python3 scripts/paper_raw_cls_domain_shift_table.py --runs-dir runs/raw_cls_domain_shift --out paper_tables/raw_cls_domain_shift_table_v13.md --data-tag v13_main4`
  - Touchpoints: `scripts/phase3_train_raw_cls_baseline.py`, `scripts/run_raw_cls_domain_shift.py`, `scripts/paper_raw_cls_domain_shift_table.py`

- [x] P0010: ablation YAML + postprocess 补齐并可跑
  - Linked claims: C0011
  - Definition of done: `configs/raw_cls/exp/{scale_token,supcon,pretrain_finetune,temp_scaling,selective}.yaml` 与 `configs/domain_shift/exp/{A2B_groupdro,A2B_coral,A2B_dsbn,A2B_pos_moe}.yaml` 全部存在；`scripts/train.py` 支持温度缩放与 selective 后处理；至少 1 个 smoke run 通过并生成预期附加产物。
  - Verification: `python3 scripts/train.py --config configs/raw_cls/exp/selective.yaml --fold 0 --seed 0 --set runtime.device=cpu`
  - Touchpoints: `scripts/train.py`, `scripts/phase3_train_raw_cls_baseline.py`, `scripts/raw_cls_temperature_scaling.py`, `scripts/raw_cls_selective_eval.py`, `configs/**/exp/*.yaml`

- [x] P0011: Teeth3DS FDI 预训练 → raw_cls 微调落地并出表
  - Linked claims: C0012
  - Definition of done: 新增 Teeth3DS FDI 预训练脚本 + raw_cls finetune 配置与一键脚本；完成 full（k=5×seeds=3）并更新 `paper_tables/raw_cls_summary.md`。
  - Verification: `CUDA_VISIBLE_DEVICES=0 bash scripts/run_full_rawcls_teeth3ds_fdi_pretrain_finetune.sh`
  - Touchpoints: `scripts/phase2_train_teeth3ds_fdi_cls.py`, `configs/raw_cls/exp/teeth3ds_fdi_pretrain_finetune.yaml`, `scripts/run_full_rawcls_teeth3ds_fdi_pretrain_finetune.sh`, `paper_tables/raw_cls_summary.md`

## Changelog

- 2026-01-16 Init (rd-loop-orchestrator)
- 2026-01-16 Proof update: marked C0003–C0005 + P0001/P0003/P0004 as done based on `docs/proof.md`.
- 2026-01-16 Proof update: marked C0001–C0002 + P0002 as done based on `docs/proof.md` (after E0001/E0002 full runs).
- 2026-01-16 Proof update: marked C0006 + P0005 as done (after E0006 full runs).
- 2026-01-16 Proof update: marked C0007 + P0006 as done (after E0007 report generation).
- 2026-01-16 Proof update: marked C0005 as done (after adding environment.yml).
- 2026-01-16 Proof update: marked C0006 as done (after k-fold dedup fixed; n=15 strict).
- 2026-01-16 Proof update: marked C0007 as done (after adding paired comparisons to merged report).
- 2026-01-16 Proof update: marked C0008 + P0007 as done (after E0008 meta-only full runs).
- 2026-01-16 Proof update: marked C0009–C0010 + P0008/P0009 as done (after E0009/E0010 full runs + table generation).
- 2026-01-17 Add C0011/P0010: ablation configs + postprocess runnable (verified via E0011/E0012).
- 2026-01-17 Add C0012/P0011: Teeth3DS FDI pretrain → raw_cls finetune (verified via E0104).
