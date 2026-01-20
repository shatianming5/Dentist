# Proof audit

- generated_at: 2026-01-17T15:28:50Z
- root: `/home/ubuntu/tiasha/dentist`
- plan: `docs/plan.md`
- experiment: `docs/experiment.md`
- summary: proved=12 not_proved=0 (rule: all evidence experiments must be marked Full + artifact checks pass)

## C0001
- claim: raw_cls 提供期刊级可复现实验：k-fold × multi-seed + mean±std 表 + bootstrap CI/paired diff。
- evidence: E0001, E0002
- status: PROVED
- proof_rule: `paper_tables/raw_cls_table_v13.md` 里 pointnet/dgcnn 的主设置（extra_features=`(none)`）k-fold 组 `n = k × seeds`，且 `paper_tables/raw_cls_ci_*.json` 存在。

## C0002
- claim: raw_cls 明确报告域差异与校准：by source/tooth_position 的 accuracy/macro-F1 + ECE/NLL/Brier。
- evidence: E0001
- status: PROVED
- proof_rule: 运行产物 `runs/raw_cls_baseline/*/metrics.json` 包含 `test_by_source_calibration`、`test_by_tooth_position`、`test_calibration`。
- notes:
  - ok: runs/raw_cls_baseline/paper_rawcls_v13_main4_dgcnn_n4096_k5_fold1_seed2020_bal_ls0p1_tta8/metrics.json

## C0003
- claim: Teeth3DS prep→target 在论文中被界定为 **synthetic proxy**（非真实临床 prep→crown），避免过度主张。
- evidence: E0003
- status: PROVED
- proof_rule: `PAPER_SCOPE.md` 明确区分 implemented vs proposed，且 `plan_report.md` 顶部包含 scope 提示。
- notes:
  - ok

## C0004
- claim: 约束评估指标更敏感且可解释：输出 `occlusion_contact_ratio` + `min_d` 分位数，并在汇总表中呈现。
- evidence: E0004
- status: PROVED
- proof_rule: `paper_tables/constraints_summary.md` 有 `min_d_p05/p50/p95` 列且对应 `eval_{val,test}.json` 存在。
- notes:
  - ok

## C0005
- claim: 复现资料达到“期刊视角”：统一命令、环境记录、审计报告、实验台账与队列产物可回溯。
- evidence: E0005
- status: PROVED
- proof_rule: `docs/experiment.md` + `.rd_queue/{logs,results,queue.json}` + `PAPER_PROTOCOL.md` + `JOURNAL_AUDIT.md` + `environment.yml` 均存在且能一键生成。
- notes:
  - ok (note: .rd_queue is created by tmux runner)

## C0006
- claim: raw_cls 增加 “meta-feature augmented” baseline（scale/log_points/objects_used）并给出 k-fold × multi-seed 报告。
- evidence: E0006
- status: PROVED
- proof_rule: `paper_tables/raw_cls_table_v13.md` 包含 extra_features=`scale,log_points,objects_used` 的 pointnet/dgcnn 行且 `n = 5 × 3`（去重后严格等于）。
- notes:
  - ok: found pointnet/dgcnn rows (n=15) for extra_features=scale,log_points,objects_used

## C0007
- claim: raw_cls 提供 k-fold merged 报告：按 seed 合并全量 test cases，输出 bootstrap CI + 分域（source）指标/校准 + 关键对照的 paired bootstrap diff（同时包含 macro_f1_all / balanced_acc 等补充指标）。
- evidence: E0007
- status: PROVED
- proof_rule: `paper_tables/raw_cls_kfold_merged_report_v13_main4.{md,json}` 存在且 json 中包含 pointnet/dgcnn 结果与 `comparisons`（macro_f1 diff 的 CI/p-value）。
- notes:
  - ok

## C0008
- claim: raw_cls 增加 “meta-only” baseline（仅用元特征，不读点云）以审计 shortcut 风险，并给出 k-fold × multi-seed 报告。
- evidence: E0008
- status: PROVED
- proof_rule: `paper_tables/raw_cls_table_v13.md` 包含 model=`meta_mlp` 且 extra_features=`scale,log_points,objects_used` 的 k-fold 组 `n = 5 × 3`，并且对应 runs 目录均有 `metrics.json`。
- notes:
  - ok: found meta_mlp row (n=15)

## C0009
- claim: raw_cls 增加 “traditional geometric” baseline（geom_mlp）并入 k-fold × multi-seed 表，补齐期刊级对照。
- evidence: E0009
- status: PROVED
- proof_rule: `paper_tables/raw_cls_table_v13.md` 包含 model=`geom_mlp` 且 extra_features=`(none)` 的 k-fold 组 `n = 5 × 3`。
- notes:
  - ok: found geom_mlp row (n=15)

## C0010
- claim: raw_cls 补齐 domain-shift 实验：train-on-A test-on-B（普通标注↔专家标注），并报告 multi-seed mean±std + 校准指标。
- evidence: E0010
- status: PROVED
- proof_rule: `paper_tables/raw_cls_domain_shift_table_v13.md` 存在，且包含两种方向（A→B, B→A）的 pointnet/dgcnn 行（seeds≥3）。
- notes:
  - ok: found both directions (A↔B) for pointnet/dgcnn with n>=3

## C0011
- claim: 仓库提供的 ablation 配置（scale_token/supcon/temp_scaling/selective + domain_shift groupdro/coral/dsbn/pos_moe）全部可运行，并在 runs 树中产出期刊可审计的附加产物（`calib.json`/`reliability.png`/`selective.{md,json}`）。
- evidence: E0011
- status: PROVED
- proof_rule: 对应 `configs/**/exp/*.yaml` 均存在；至少 1 个 smoke run 目录内包含 `metrics.json` 且（当配置启用时）包含 `calib.json`/`reliability.png`/`selective.md`。

## C0012
- claim: Teeth3DS 单牙 FDI 判别式预训练作为 init，可显著提升 raw_cls 主任务 PointNet（k-fold×multi-seed）。
- evidence: E0104, E0101
- status: PROVED
- proof_rule: `paper_tables/raw_cls_summary.md` 中 `teeth3ds_fdi_pretrain_finetune` 的 mean macro_f1/acc 高于 `baseline` 与 `supcon` 的 PointNet 行（seed=1337,2020,2021; fold=0..4）。
