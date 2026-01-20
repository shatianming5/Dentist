# configs/ Index

`configs/` 是本仓库的“实验真相来源”：所有训练/评估都应由 `scripts/train.py --config <yaml>` 驱动，避免手工参数漂移。

## 组织方式

- `configs/common/`：通用 defaults（runtime/logger/model/augment 等）
- `configs/raw_cls/`：修复体分类（dataset/data/train/exp）
- `configs/domain_shift/`：跨域设置（A→B/B→A）
- `configs/prep2target/`：synthetic proxy 任务

## 常见用法

- 运行一个配置：

```bash
python3 scripts/train.py --config configs/raw_cls/exp/baseline.yaml --fold 0 --seed 1337
```

- 临时覆写参数（不改 YAML）：

```bash
python3 scripts/train.py \
  --config configs/raw_cls/exp/baseline.yaml \
  --fold 0 --seed 0 \
  --set runtime.device=cpu \
  --set train.epochs=1
```

## raw_cls 相关（最常用）

- data 版本：`configs/raw_cls/data_*.yaml`
- 训练超参：`configs/raw_cls/train_*.yaml`
- 实验组合（入口）：`configs/raw_cls/exp/*.yaml`

常见“可审计 ablation”入口（是否提升性能不作保证，重点是**能跑+产物齐全**）：

- `scale_token`：scale 作为全局 token / meta feature
- `supcon`：小样本对比学习（SupCon）
- `temp_scaling`：温度标定（fold 内拟合，test 上评）
- `selective`：选择性分类/拒识（输出 `selective.{md,json}`）

domain_shift 常见 ablation：

- `groupdro` / `coral` / `dsbn` / `pos_moe`

建议搭配阅读：

- 配置入口与 SOTA 复现：`docs/QUICKSTART.md`
- 结果索引：`docs/RESULTS.md` / `paper_tables/INDEX.md`

