# paper_tables Index

`paper_tables/` 是“用户评估/期刊复现”的主输出目录：表格、集成结果 JSON、CI/paired diff、domain-shift delta 等都在这里。

## 快速入口

- raw_cls 当前 SOTA（multi-member mean-prob ensemble）：`raw_cls_multi_ensemble_v18_pointnet_pointnet_curvrad_pointnet2.json`
- raw_cls SOTA 完整指标（含 confusion_matrix）：`raw_cls_ensemble_eval_mean_v18_best.json`
- raw_cls SOTA 误差/不确定样本清单：`raw_cls_error_audit_v18_sota.json`

## raw_cls（v13 主线：期刊级 baseline + CI）

- k-fold × multi-seed 主表：`raw_cls_table_v13.md`
- k-fold merged 报告（CI + paired diff + 分域）：`raw_cls_kfold_merged_report_v13_main4.md`
- domain-shift 表：`raw_cls_domain_shift_table_v13.md`

## raw_cls（v18/v19/v20：数据/模型消融汇总）

- v18 汇总（多份）：`raw_cls_summary_v18_main4_seg_all_cloudid_eq_*.md`
- v19 汇总：`raw_cls_summary_v19_main4_seg_all_cloudid_eq_n8192*.md`
- v20 汇总：`raw_cls_summary_v20_main4_seg_all_cloudid_eq_pool16384*.md`

## domain_shift（README DoD）

- 汇总：`domain_shift_summary.md`
- Δ 指标：`domain_shift_delta.md`

## constraints / prep2target

- constraints：`constraints_summary.md`
- prep2target：`prep2target_summary.md`

