#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SEEDS=(1337 2020 2021)
FOLD=0

run_one() {
  local cfg="$1"
  local exp="$2"
  local model="$3"
  local train_source="$4"
  local test_source="$5"
  local seed="$6"

  local run_dir="runs/domain_shift/v13_main4/A2B_${train_source}_to_${test_source}/${exp}/${model}/fold=${FOLD}/seed=${seed}"
  if [[ -f "${run_dir}/metrics.json" ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] SKIP ${run_dir} (metrics.json exists)"
    return 0
  fi

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] RUN cfg=${cfg} exp=${exp} model=${model} seed=${seed} ${train_source} -> ${test_source}"
  python3 scripts/train.py \
    --config "${cfg}" \
    --fold "${FOLD}" \
    --seed "${seed}" \
    --train_source "${train_source}" \
    --test_source "${test_source}" \
    --set runtime.device=cuda \
    --set "model.name=${model}"
}

run_direction() {
  local train_source="$1"
  local test_source="$2"

  for seed in "${SEEDS[@]}"; do
    run_one "configs/domain_shift/exp/A2B_baseline.yaml" "baseline" "pointnet" "${train_source}" "${test_source}" "${seed}"
    run_one "configs/domain_shift/exp/A2B_baseline.yaml" "baseline" "dgcnn" "${train_source}" "${test_source}" "${seed}"

    run_one "configs/domain_shift/exp/A2B_groupdro.yaml" "groupdro" "pointnet" "${train_source}" "${test_source}" "${seed}"
    run_one "configs/domain_shift/exp/A2B_coral.yaml" "coral" "pointnet" "${train_source}" "${test_source}" "${seed}"
    run_one "configs/domain_shift/exp/A2B_dsbn.yaml" "dsbn" "pointnet_dsbn" "${train_source}" "${test_source}" "${seed}"
    run_one "configs/domain_shift/exp/A2B_pos_moe.yaml" "pos_moe" "pointnet_pos_moe" "${train_source}" "${test_source}" "${seed}"
  done
}

run_in_domain_baseline() {
  local source="$1"
  for seed in "${SEEDS[@]}"; do
    run_one "configs/domain_shift/exp/A2B_baseline.yaml" "baseline" "pointnet" "${source}" "${source}" "${seed}"
    run_one "configs/domain_shift/exp/A2B_baseline.yaml" "baseline" "dgcnn" "${source}" "${source}" "${seed}"
  done
}

# Both directions.
run_direction "普通标注" "专家标注"
run_direction "专家标注" "普通标注"

# In-domain baselines (needed for README DoD 7.8 deltas).
run_in_domain_baseline "普通标注"
run_in_domain_baseline "专家标注"

python3 scripts/aggregate_runs.py --root runs/domain_shift/v13_main4 --out paper_tables/domain_shift_summary.md --seeds 1337,2020,2021 --folds 0
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] DONE aggregate -> paper_tables/domain_shift_summary.md"

python3 scripts/domain_shift_delta_table.py --runs-root runs/domain_shift/v13_main4 --out paper_tables/domain_shift_delta.md
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] DONE delta -> paper_tables/domain_shift_delta.md"
