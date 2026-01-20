# Runs Directory

`runs/` 保存所有训练/评估的落盘产物。**从用户评估角度**，这里是“可复现/可审计”的核心证据来源（而不是 stdout）。

典型目录结构：

```
runs/<task>/<dataset>/<exp>/<model>/fold=<k>/seed=<s>/
```

例如：

```
runs/raw_cls/v18_main4_seg_all_cloudid_eq/<exp>/<model>/fold=0/seed=1337/
```

每个 run 目录通常包含：

- `metrics.json`：核心指标（acc/macro_f1/bal_acc）+ 校准（ece/nll/brier）+ 分域（by_source/by_tooth_position）
- `preds_val.jsonl` / `preds_test.jsonl`：逐样本预测（用于 CI/paired diff/误差分析/集成）
- `config.yaml` / `config.json`：本次 run 的有效配置快照
- `env.json`：环境与 git 信息
- `logs.txt`：训练日志（含实际命令行）
- 可能还有：`confusion_*.csv`、`errors_test.csv`、`calib.json`、`reliability.png`、`selective.{md,json}`

建议配合以下目录阅读：

- `docs/experiment.md`：哪些 run 应该存在（smoke/full）
- `paper_tables/`：从 runs 聚合出来的表格与证据

