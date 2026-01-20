# Milestone 4：λ 消融（Teeth3DS 合成 prep→target + 功能约束最小版本）

## 本次跑了什么
- 模型：`scripts/phase4_train_teeth3ds_prep2target_constraints.py`
- 数据：`processed/teeth3ds_teeth/v1/index.jsonl`
- Sweep：不同 `λ_margin / λ_occlusion`（训练均为全量，无 `--limit-*`）
- 评估脚本：`scripts/phase4_eval_teeth3ds_constraints_run.py`

## 统一评估设置（用于横向对比）
- split：val
- `cut_q`：固定 0.7（deterministic）
- `margin_band=0.02`，`margin_points=64`
- `occlusion_clearance=0.5`，对颌点来自 `processed/teeth3ds_teeth/v1/opposing_cache/`

## 结果（val）
| run | λ_margin | λ_occ | chamfer | margin | occ_pen_mean | occ_contact |
|---|---:|---:|---:|---:|---:|---:|
| p2tC_n256_seed1337_20251214_073510 | 0.00 | 0.10 | 0.056806 | 0.025260 | 0.003351 | 0.0257 |
| p2tC_n256_seed1337_20251214_074825 | 0.10 | 0.10 | 0.057234 | 0.024019 | 0.003247 | 0.0250 |
| p2tC_n256_seed1337_20251214_040657 | 0.10 | 0.00 | 0.057258 | 0.024137 | 0.003330 | 0.0254 |
| p2tC_n256_seed1337_20251214_035634 | 0.00 | 0.00 | 0.057328 | 0.025741 | 0.003297 | 0.0253 |
| p2tC_n256_seed1337_20251214_044131 | 0.00 | 0.05 | 0.057475 | 0.025085 | 0.003247 | 0.0249 |
| p2tC_n256_seed1337_20251214_041713 | 0.50 | 0.00 | 0.058293 | 0.021085 | 0.003284 | 0.0252 |
| p2tC_n256_seed1337_20251214_042818 | 1.00 | 0.00 | 0.061447 | 0.020047 | 0.003190 | 0.0245 |
| p2tC_n256_seed1337_20251214_001951 | 0.50 | 0.10 | 0.104256 | 0.035616 | 0.003349 | 0.0254 |

## 建议（下一步怎么用）
- 如果目标是“尽量不伤 chamfer，同时把 margin 拉下来一点”：优先用 `λ_margin=0.1, λ_occ=0.1`（`runs/teeth3ds_prep2target_constraints/p2tC_n256_seed1337_20251214_074825:1`）。
- 如果目标是“强压 margin（允许 chamfer 变差）”：可尝试 `λ_margin=0.5` 或 `1.0`（对应 run 见上表）。
- `occ_contact`/`occ_pen_mean` 区分度不强（val 上 95% penalty 为 0），后续建议把评估指标改成 `min_d` 的分位数（或提高 `occlusion_clearance` 做更敏感的 proxy）。

## 额外：test 快速对比（固定同一评估设置）
- baseline（`λ_margin=0, λ_occ=0`）：`runs/teeth3ds_prep2target_constraints/p2tC_n256_seed1337_20251214_035634/eval_test.json:1`
- 推荐（`λ_margin=0.1, λ_occ=0.1`）：`runs/teeth3ds_prep2target_constraints/p2tC_n256_seed1337_20251214_074825/eval_test.json:1`
