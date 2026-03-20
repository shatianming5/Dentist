#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

teacher_ckpt() {
  local fold="$1"
  local seed="$2"
  echo "runs/research_segcls_probe/raw_seg_teacher_dgcnn/dgcnn_v2_teacher_fold${fold}_seed${seed}/ckpt_best.pt"
}

run_one() {
  local fold="$1"
  local seed="$2"
  local tag="$3"
  local aux="$4"
  local cons="$5"
  local segtw="$6"
  local ckpt
  ckpt="$(teacher_ckpt "$fold" "$seed")"
  if [[ ! -f "$ckpt" ]]; then
    echo "Missing teacher checkpoint: $ckpt" >&2
    return 1
  fi
  local exp="joint_locc_segteacher_fold${fold}_seed${seed}_${tag}"
  local out="runs/research_segcls_probe/locc_segteacher_joint/${exp}/metrics.json"
  if [[ -f "$out" ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] SKIP $exp"
    return 0
  fi
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] RUN $exp"
  python3 scripts/phase3_train_raw_segcls_joint.py \
    --data-root processed/raw_seg/v1 \
    --run-root runs/research_segcls_probe/locc_segteacher_joint \
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
    --seg-teacher-model dgcnn_v2 \
    --seg-teacher-ckpt "$ckpt" \
    --seg-teacher-weight "$segtw" \
    --seg-teacher-temp 1.0 \
    --tta 4 \
    --kfold metadata/splits_raw_case_kfold.json \
    --fold "$fold"
}

run_one 2 2021 tseg010_v2 0.50 0.10 0.10
run_one 2 2021 tseg025_v2 0.50 0.10 0.25
run_one 3 1337 tseg010_v2 0.25 0.25 0.10
run_one 3 1337 tseg025_v2 0.25 0.25 0.25

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] DONE locc strong seg teacher sweep"
