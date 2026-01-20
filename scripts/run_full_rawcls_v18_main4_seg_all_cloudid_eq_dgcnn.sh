#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SEEDS=(1337 2020 2021)
FOLDS=(0 1 2 3 4)

CFG="configs/raw_cls/exp/teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_dropout0p1_seg_all_cloudid_eq_dgcnn.yaml"
EXP="teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_dropout0p1_seg_all_cloudid_eq_dgcnn"
MODEL="dgcnn"

for fold in "${FOLDS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_dir="runs/raw_cls/v18_main4_seg_all_cloudid_eq/${EXP}/${MODEL}/fold=${fold}/seed=${seed}"
    if [[ -f "${run_dir}/metrics.json" ]]; then
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] SKIP ${run_dir} (metrics.json exists)"
      continue
    fi
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] RUN cfg=${CFG} exp=${EXP} fold=${fold} seed=${seed}"
    python3 scripts/train.py \
      --config "${CFG}" \
      --fold "${fold}" \
      --seed "${seed}" \
      --set runtime.device=cuda \
      --set "model.name=${MODEL}"
  done
done

python3 scripts/aggregate_runs.py \
  --root runs/raw_cls/v18_main4_seg_all_cloudid_eq \
  --out paper_tables/raw_cls_summary_v18_main4_seg_all_cloudid_eq_dgcnn.md \
  --seeds 1337,2020,2021 \
  --folds 0,1,2,3,4

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] DONE aggregate -> paper_tables/raw_cls_summary_v18_main4_seg_all_cloudid_eq_dgcnn.md"

