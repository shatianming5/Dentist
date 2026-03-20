#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

train_teacher() {
  local fold="$1"
  local seed="$2"
  local exp="dgcnn_v2_teacher_fold${fold}_seed${seed}"
  local run_root="runs/research_segcls_probe/raw_seg_teacher_dgcnn"
  local ckpt="${run_root}/${exp}/ckpt_best.pt"
  if [[ -f "$ckpt" ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] SKIP teacher $exp"
    return 0
  fi
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] RUN teacher $exp"
  python3 scripts/phase3_train_raw_seg.py \
    --data-root processed/raw_seg/v1 \
    --run-root "$run_root" \
    --exp-name "$exp" \
    --seed "$seed" \
    --device cuda \
    --model dgcnn_v2 \
    --epochs 100 \
    --patience 20 \
    --batch-size 16 \
    --n-points 8192 \
    --num-workers 0 \
    --kfold metadata/splits_raw_case_kfold.json \
    --fold "$fold"
}

build_predtopk_dataset() {
  local fold="$1"
  local seed="$2"
  local exp="dgcnn_v2_teacher_fold${fold}_seed${seed}"
  local ckpt="runs/research_segcls_probe/raw_seg_teacher_dgcnn/${exp}/ckpt_best.pt"
  local out="processed/raw_cls_from_raw_seg/dgcnn_predtopk_probe_fold${fold}_seed${seed}"
  if [[ -f "${out}/build_config.json" ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] SKIP dataset fold${fold} seed${seed}"
    return 0
  fi
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] BUILD dataset fold${fold} seed${seed}"
  python3 scripts/phase3_build_raw_cls_from_raw_seg.py \
    --seg-root processed/raw_seg/v1 \
    --out "$out" \
    --mode pred_topk \
    --topk 4096 \
    --seg-model dgcnn_v2 \
    --seg-ckpt "$ckpt" \
    --device cuda
}

run_cls_probe() {
  local fold="$1"
  local seed="$2"
  local data_root="processed/raw_cls_from_raw_seg/dgcnn_predtopk_probe_fold${fold}_seed${seed}"
  local exp="predtopk_dgcnn_pointnet_fold${fold}_seed${seed}"
  local out="runs/research_segcls_probe/predtopk_dgcnn_pointnet/${exp}/metrics.json"
  if [[ -f "$out" ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] SKIP cls $exp"
    return 0
  fi
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] RUN cls $exp"
  python3 scripts/phase3_train_raw_cls_baseline.py \
    --data-root "$data_root" \
    --run-root runs/research_segcls_probe/predtopk_dgcnn_pointnet \
    --exp-name "$exp" \
    --seed "$seed" \
    --device cuda \
    --model pointnet \
    --epochs 80 \
    --patience 20 \
    --batch-size 16 \
    --n-points 4096 \
    --num-workers 0 \
    --balanced-sampler \
    --label-smoothing 0.1 \
    --tta 4 \
    --dropout 0.1 \
    --kfold metadata/splits_raw_case_kfold.json \
    --fold "$fold"
}

for spec in "2 2021" "3 1337"; do
  read -r fold seed <<<"$spec"
  train_teacher "$fold" "$seed"
  build_predtopk_dataset "$fold" "$seed"
  run_cls_probe "$fold" "$seed"
done

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] DONE stronger raw_seg teacher probes"
