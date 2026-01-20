# Results Index (What to Read)

本页是“从用户角度”快速判断仓库是否有产出、产出在哪里、可信度如何。

## raw_cls（修复体类型分类）

数据版本：`processed/raw_cls/v18_main4_seg_all_cloudid_eq`（4 类：充填/全冠/桩核冠/高嵌体，n=248）

- **当前 repo 内 SOTA（multi-member × multi-seed mean-prob ensemble）**
  - 主结果：`paper_tables/raw_cls_multi_ensemble_v18_pointnet_pointnet_curvrad_pointnet2.json`
  - 完整指标（含 confusion_matrix）：`paper_tables/raw_cls_ensemble_eval_mean_v18_best.json`
  - 误差/不确定样本清单（用于人工复核/噪声评估）：`paper_tables/raw_cls_error_audit_v18_sota.json`

- 单模型/消融汇总（按 dataset/exp/model 聚合）
  - `paper_tables/raw_cls_summary.md`（v13 主线汇总）
  - `paper_tables/raw_cls_summary_v18_main4_seg_all_cloudid_eq_*.md`（v18 相关消融）
  - `paper_tables/raw_cls_table_v13.md`（v13 主表：k-fold × multi-seed）

## domain_shift（普通标注 ↔ 专家标注）

- 汇总表：`paper_tables/domain_shift_summary.md`
- 跨域掉分 Δ 指标（对比 in-domain baseline）：`paper_tables/domain_shift_delta.md`

## constraints / prep2target（synthetic proxy 任务）

- 约束评估汇总：`paper_tables/constraints_summary.md`
- prep2target 汇总：`paper_tables/prep2target_summary.md`

## “期刊级可复现”证据链

- claims：`docs/plan.md`
- 实验台账（含 smoke/full 命令）：`docs/experiment.md`
- proof 审计产物：`docs/proof.md`

