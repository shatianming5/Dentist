# User Review (Readability & Reproducibility)

本页从“第一次打开仓库的人”的角度，评价这个仓库是否**可读、可跑、可复现、可审计**，并给出最快的验证路径。

## 你能从这个仓库得到什么？

- **可复现的训练/评估入口**：统一由 `scripts/train.py` 驱动（读取 `configs/`，支持 `--set k=v` 覆写）。
- **可审计的落盘证据**：每次 run 都在 `runs/` 留下 `metrics.json / preds_*.jsonl / config.yaml / logs.txt / env.json`（详见 `runs/README.md`）。
- **期刊式结果产物**：聚合表/集成结果/误差审计都落在 `paper_tables/`（先看 `paper_tables/INDEX.md`）。
- **当前 repo 内 raw_cls SOTA**（4 类修复体分类，n=248）：`overall_acc=0.6371`  
  - 主结果：`paper_tables/raw_cls_multi_ensemble_v18_pointnet_pointnet_curvrad_pointnet2.json`  
  - 完整指标（含 confusion_matrix + 校准）：`paper_tables/raw_cls_ensemble_eval_mean_v18_best.json`

## 你最容易踩坑的点（从用户角度）

1) **“研究路线图”和“已实现证据”混在一起**  
   - `plan.md` / `plan_report.md` 含大量未来工作；想找“能证明的结论”请以 `docs/plan.md` + `docs/experiment.md` + `docs/proof.md` 为准。
2) **数据目录很多，但它们是分层产物**  
   - `raw/` → `converted/` → `processed/` 是流水线，不是重复数据。入口见 `docs/INDEX.md`。
3) **跑不出结果通常是因为环境/路径/数据版本**  
   - 先跑 `docs/QUICKSTART.md` 的 CPU smoke，确认能在 `runs/` 产出 `metrics.json`。

## 5 分钟“用户验收”路径（推荐）

1) **跑 smoke**：按 `docs/QUICKSTART.md` 执行 CPU 命令，确认 `runs/.../metrics.json` 生成。
2) **确认“结果看哪里”**：打开 `docs/RESULTS.md`，能定位到对应的 `paper_tables/*.md/*.json`。
3) **确认 SOTA 证据链**：打开 `paper_tables/raw_cls_multi_ensemble_v18_pointnet_pointnet_curvrad_pointnet2.json`（score）与 `paper_tables/raw_cls_ensemble_eval_mean_v18_best.json`（完整指标）。
4) **确认可审计**：随便进入一个 `runs/raw_cls/.../fold=*/seed=*/`，检查 `preds_test.jsonl` 与 `config.yaml` 是否存在。

## 结论（用户视角）

- 优点：入口集中（`docs/INDEX.md` / `docs/QUICKSTART.md`）、证据落盘完整（`runs/` + `paper_tables/`）、结果可被脚本复现与审计。
- 不足：目录与文档较多，新用户需要先理解“产物分层”（raw→converted→processed→runs→paper_tables）；另外部分“未来工作”文档容易被误认为已实现结论。
- 建议阅读顺序：`docs/INDEX.md` → `docs/QUICKSTART.md` → `docs/RESULTS.md` → `docs/REPO_MAP.md` → `paper_tables/INDEX.md`。

