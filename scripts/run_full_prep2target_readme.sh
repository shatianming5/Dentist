#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SEEDS=(1337 2020 2021)
RUN_MULTITASK_CONSTRAINTS="${RUN_MULTITASK_CONSTRAINTS:-0}"

run_one() {
  local cfg="$1"
  local exp="$2"
  local seed="$3"

  local run_dir="runs/prep2target/v1/${exp}/p2t/seed=${seed}"
  if [[ -f "${run_dir}/metrics.json" && -f "${run_dir}/previews/test/pred_target.npy" ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] SKIP ${run_dir} (metrics.json + test preview exists)"
    return 0
  fi

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] RUN cfg=${cfg} exp=${exp} seed=${seed}"
  python3 scripts/train.py --config "${cfg}" --seed "${seed}" --set runtime.device=cuda
}

for seed in "${SEEDS[@]}"; do
  run_one "configs/prep2target/exp/baseline.yaml" "baseline" "${seed}"
  run_one "configs/prep2target/exp/constraints_margin.yaml" "constraints_margin" "${seed}"
  run_one "configs/prep2target/exp/constraints_occlusion.yaml" "constraints_occlusion" "${seed}"
  if [[ "${RUN_MULTITASK_CONSTRAINTS}" == "1" ]]; then
    run_one "configs/prep2target/exp/multitask_constraints.yaml" "multitask_constraints" "${seed}"
  fi
done

python3 scripts/aggregate_runs.py --root runs/prep2target/v1 --out paper_tables/prep2target_summary.md --seeds 1337,2020,2021
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] DONE aggregate -> paper_tables/prep2target_summary.md"
