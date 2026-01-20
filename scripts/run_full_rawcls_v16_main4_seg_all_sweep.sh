#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SEEDS=(1337 2020 2021)
FOLDS=(0 1 2 3 4)
MODEL="pointnet"

run_suite() {
  local cfg="$1"
  local exp="$2"

  for fold in "${FOLDS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      local run_dir="runs/raw_cls/v16_main4_seg_all/${exp}/${MODEL}/fold=${fold}/seed=${seed}"
      if [[ -f "${run_dir}/metrics.json" ]]; then
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] SKIP ${run_dir} (metrics.json exists)"
        continue
      fi
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] RUN cfg=${cfg} exp=${exp} fold=${fold} seed=${seed}"
      python3 scripts/train.py \
        --config "${cfg}" \
        --fold "${fold}" \
        --seed "${seed}" \
        --set runtime.device=cuda \
        --set "model.name=${MODEL}"
    done
  done
}

run_suite "configs/raw_cls/exp/teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_seg_all.yaml" \
  "teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_seg_all"
run_suite "configs/raw_cls/exp/teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_dropout0p1_seg_all.yaml" \
  "teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_dropout0p1_seg_all"
run_suite "configs/raw_cls/exp/teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_dropout0_seg_all.yaml" \
  "teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_dropout0_seg_all"

python3 scripts/aggregate_runs.py \
  --root runs/raw_cls/v16_main4_seg_all \
  --out paper_tables/raw_cls_summary_v16_main4_seg_all_sweep.md \
  --seeds 1337,2020,2021 \
  --folds 0,1,2,3,4

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] DONE aggregate -> paper_tables/raw_cls_summary_v16_main4_seg_all_sweep.md"

