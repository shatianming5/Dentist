#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

run_one() {
  local fold="$1"
  local seed="$2"
  local tag="$3"
  local aux="$4"
  local cons="$5"
  local segw="$6"
  local calw="$7"
  local calmetric="$8"
  local exp="joint_locc_cal_fold${fold}_seed${seed}_${tag}"
  local out="runs/research_segcls_probe/locc_calibration_joint/${exp}/metrics.json"
  if [[ -f "$out" ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] SKIP $exp"
    return 0
  fi
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] RUN $exp"
  python3 scripts/phase3_train_raw_segcls_joint.py \
    --data-root processed/raw_seg/v1 \
    --run-root runs/research_segcls_probe/locc_calibration_joint \
    --exp-name "$exp" \
    --seed "$seed" \
    --device cuda \
    --epochs 80 \
    --patience 20 \
    --batch-size 16 \
    --n-points 8192 \
    --num-workers 0 \
    --balanced-sampler \
    --label-smoothing 0.1 \
    --seg-loss-weight 0.5 \
    --pooling-mode topk \
    --cls-train-mask pred \
    --cls-topk-ratio 0.5 \
    --aux-gt-cls-weight "$aux" \
    --consistency-weight "$cons" \
    --consistency-temp 1.0 \
    --selection-seg-weight "$segw" \
    --selection-calibration-weight "$calw" \
    --selection-calibration-metric "$calmetric" \
    --calibration-bins 15 \
    --tta 4 \
    --kfold metadata/splits_raw_case_kfold.json \
    --fold "$fold"
}

# fold2 best locc baseline: aux050_cons010
run_one 2 2021 ece010 0.50 0.10 0.00 0.10 ece
run_one 2 2021 ece020 0.50 0.10 0.00 0.20 ece
run_one 2 2021 seg025_ece010 0.50 0.10 0.25 0.10 ece

# fold3 best locc baseline: aux025_cons025
run_one 3 1337 ece010 0.25 0.25 0.00 0.10 ece
run_one 3 1337 ece020 0.25 0.25 0.00 0.20 ece
run_one 3 1337 seg025_ece010 0.25 0.25 0.25 0.10 ece

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] DONE locc calibration sweep"
