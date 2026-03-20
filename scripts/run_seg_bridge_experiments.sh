#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SEEDS=(1337 2020 2021)
FOLDS=(0 1 2 3 4)
MODEL="pointnet"

# --- Experiment variants ---
declare -A CONFIGS
CONFIGS[baseline_v13_cloudid]="configs/raw_cls/exp/baseline_v13_cloudid.yaml"
CONFIGS[seg_enriched_seg_prob]="configs/raw_cls/exp/seg_enriched_seg_prob.yaml"
CONFIGS[seg_enriched_seg_gt]="configs/raw_cls/exp/seg_enriched_seg_gt.yaml"
CONFIGS[seg_enriched_seg_prob_only]="configs/raw_cls/exp/seg_enriched_seg_prob_only.yaml"

RUN_ROOT="runs/raw_cls/seg_bridge"

for exp_name in baseline_v13_cloudid seg_enriched_seg_prob seg_enriched_seg_gt seg_enriched_seg_prob_only; do
  CFG="${CONFIGS[$exp_name]}"
  echo ""
  echo "========================================"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] EXPERIMENT: ${exp_name}"
  echo "========================================"
  for fold in "${FOLDS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      run_dir="${RUN_ROOT}/${exp_name}/${MODEL}/fold=${fold}/seed=${seed}"
      if [[ -f "${run_dir}/metrics.json" ]]; then
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] SKIP ${run_dir} (metrics.json exists)"
        continue
      fi
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] RUN exp=${exp_name} fold=${fold} seed=${seed}"
      python3 scripts/train.py \
        --config "${CFG}" \
        --fold "${fold}" \
        --seed "${seed}" \
        --set runtime.device=cuda \
        --set "model.name=${MODEL}" \
        2>&1 | tail -5
    done
  done
done

echo ""
echo "========================================"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] SUMMARIZE"
echo "========================================"

python3 scripts/aggregate_runs.py \
  --root "${RUN_ROOT}" \
  --out paper_tables/seg_bridge_summary.md \
  --seeds 1337,2020,2021 \
  --folds 0,1,2,3,4

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] DONE -> paper_tables/seg_bridge_summary.md"
