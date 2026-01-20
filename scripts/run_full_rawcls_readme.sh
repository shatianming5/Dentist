#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SEEDS=(1337 2020 2021)
FOLDS=(0 1 2 3 4)

run_one() {
  local cfg="$1"
  local exp="$2"
  local model="$3"
  local fold="$4"
  local seed="$5"

  local run_dir="runs/raw_cls/v13_main4/${exp}/${model}/fold=${fold}/seed=${seed}"
  if [[ -f "${run_dir}/metrics.json" ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] SKIP ${run_dir} (metrics.json exists)"
    return 0
  fi

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] RUN cfg=${cfg} exp=${exp} model=${model} fold=${fold} seed=${seed}"
  python3 scripts/train.py --config "${cfg}" --fold "${fold}" --seed "${seed}" --set runtime.device=cuda --set "model.name=${model}"
}

run_kfold_suite() {
  local cfg="$1"
  local exp="$2"
  shift 2
  local models=("$@")

  for model in "${models[@]}"; do
    for fold in "${FOLDS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        run_one "${cfg}" "${exp}" "${model}" "${fold}" "${seed}"
      done
    done
  done
}

# README ablations: full k-fold × multi-seed suite.
run_kfold_suite "configs/raw_cls/exp/baseline.yaml" "baseline" "pointnet" "dgcnn"
# DGCNN is much heavier; run ablations on PointNet by default.
run_kfold_suite "configs/raw_cls/exp/feat_normcurv.yaml" "feat_normcurv" "pointnet"
run_kfold_suite "configs/raw_cls/exp/scale_token.yaml" "scale_token" "pointnet"
run_kfold_suite "configs/raw_cls/exp/supcon.yaml" "supcon" "pointnet"
run_kfold_suite "configs/raw_cls/exp/pretrain_finetune.yaml" "pretrain_finetune" "pointnet"

python3 scripts/aggregate_runs.py --root runs/raw_cls/v13_main4 --out paper_tables/raw_cls_summary.md --seeds 1337,2020,2021 --folds 0,1,2,3,4
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] DONE aggregate -> paper_tables/raw_cls_summary.md"
