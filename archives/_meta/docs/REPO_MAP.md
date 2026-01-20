# Repo Map (User View)

从“用户评估/复现”的角度，这个仓库可以拆成：**数据 → 处理后数据集 → 训练/评估 → 结果汇总/表格 → 文档闭环**。

## 顶层文件（你可能会困惑的）

- `README.md`：数据与目录说明（入口导航见顶部）。
- `plan.md`：研究/工程路线图（包含大量“未来工作”，不作为论文 claim 证据来源）。
- `plan_report.md`：研究蓝图（明确包含未实现模块）。
- `PAPER_SCOPE.md`：论文叙事边界（implemented vs proposed）。
- `PAPER_PROTOCOL.md`：复现协议（环境、命令、产物约定）。
- `JOURNAL_AUDIT.md` / `PAPER_AUDIT.md`：审计/检查产物（用于投稿自检）。
- `DATASET_STATS.md` / `RAW_DATASET_STATS.md`：数据统计报告（脚本可复现生成）。

## 关键目录（按“用户会用到的频率”排序）

- `docs/`
  - `docs/INDEX.md`：项目总索引（从这里找入口）。
  - `docs/QUICKSTART.md`：最小 smoke + 复现 SOTA。
  - `docs/USER_REVIEW.md`：从用户角度对仓库可读性/可复现性的评估与验收路径。
  - `docs/plan.md` / `docs/experiment.md` / `docs/proof.md`：期刊级闭环文档（claims→evidence）。
- `configs/`：YAML 配置（`scripts/train.py` 支持 defaults 链与 `--set k=v` 覆写）。
- `scripts/`：代码入口（训练/评估/汇总几乎都在这里；索引见 `scripts/INDEX.md`）。
- `configs/INDEX.md`：配置目录的阅读方式与常见入口索引。
- `runs/`：训练输出（每次训练一个 run 目录，包含 metrics/preds/config/logs）。
- `paper_tables/`：汇总与证据（表格、集成结果 JSON、delta/CI 报告等）。

## 数据相关目录（“能跑起来”的前置条件）

- `data/`：Teeth3DS + landmarks + 官方 splits（已解压整理）。
- `raw/`：CloudCompare CCB2 bin + Excel 标签（专有/原始）。
- `converted/`：从 `raw/` 提取出来的点云对象（npz/ply）及清单/标签。
- `processed/`：训练用版本化数据集（raw_cls 等任务会从这里读）。
- `metadata/`：k-fold split、派生索引等。

## 训练/评估的“入口链”

- 统一入口：`scripts/train.py`
  - raw_cls：调用 `scripts/phase3_train_raw_cls_baseline.py`
  - domain_shift：同上（通过 train/test source override）
  - prep2target：调用 `scripts/phase4_train_raw_prep2target_finetune.py`

## 结果怎么评估（用户视角的 checklist）

1) **可运行**：至少 1 条 smoke 能在 CPU 完成并产出 `metrics.json`（见 `docs/QUICKSTART.md`）。
2) **可复现**：同一 config 重跑不会污染 n（汇总脚本按 fold/seed 去重；runs 目录结构固定）。
3) **可审计**：每个 run 目录至少含：
   - `metrics.json`（指标+校准+分域）
   - `preds_val.jsonl` / `preds_test.jsonl`（可用于 CI/paired diff/误差分析）
   - `config.yaml` / `env.json` / `logs.txt`
4) **结论有证据**：`docs/proof.md` 能把 `docs/plan.md` 的 claims 映射到 `docs/experiment.md` 的实验与 `paper_tables/` 的产物。
