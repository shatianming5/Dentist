#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

FOLDS=(2 3)
declare -A SEED_BY_FOLD=([2]=2021 [3]=1337)
CONFIGS=(
  "aux025_cons025 0.25 0.25"
  "aux050_cons010 0.50 0.10"
  "aux100_cons025 1.00 0.25"
  "aux050_cons050 0.50 0.50"
)

run_one() {
  local fold="$1"
  local seed="$2"
  local tag="$3"
  local aux="$4"
  local cons="$5"
  local exp="joint_locc_fold${fold}_seed${seed}_${tag}"
  local out="runs/research_segcls_probe/locc_joint/${exp}/metrics.json"
  if [[ -f "$out" ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] SKIP $exp"
    return 0
  fi
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] RUN $exp"
  python3 scripts/phase3_train_raw_segcls_joint.py \
    --data-root processed/raw_seg/v1 \
    --run-root runs/research_segcls_probe/locc_joint \
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
    --tta 4 \
    --kfold metadata/splits_raw_case_kfold.json \
    --fold "$fold"
}

for fold in "${FOLDS[@]}"; do
  seed="${SEED_BY_FOLD[$fold]}"
  for cfg in "${CONFIGS[@]}"; do
    read -r tag aux cons <<<"$cfg"
    run_one "$fold" "$seed" "$tag" "$aux" "$cons"
  done
done

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] DONE locc sweep"
