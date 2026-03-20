#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

GPU="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

NEW_SEEDS=(42 7)
FOLDS=(0 1 2 3 4)

run_seg() {
  local model="$1" data_root="$2" run_root="$3" exp_name="$4" seed="$5" fold="$6"
  local extra_args="${7:-}"
  local out="${run_root}/${exp_name}/results.json"
  if [[ -f "$out" ]]; then
    echo "[$(date +%H:%M:%S)] SKIP $exp_name"
    return 0
  fi
  echo "[$(date +%H:%M:%S)] RUN $exp_name on GPU $GPU"
  python3 scripts/phase3_train_raw_seg.py \
    --data-root "$data_root" \
    --run-root "$run_root" \
    --exp-name "$exp_name" \
    --seed "$seed" \
    --device cuda \
    --model "$model" \
    --epochs 100 \
    --patience 20 \
    --batch-size 16 \
    --n-points 8192 \
    --num-workers 2 \
    --kfold metadata/splits_raw_case_kfold.json \
    --fold "$fold" \
    $extra_args
}

JOB="$2"

case "$JOB" in
  pn_bal)
    for seed in "${NEW_SEEDS[@]}"; do for fold in "${FOLDS[@]}"; do
      run_seg pointnet_seg processed/raw_seg/v1 runs/research_segcls_full/seg_pointnet "pointnet_bal_s${seed}_f${fold}" "$seed" "$fold"
    done; done ;;
  pn_nat)
    for seed in "${NEW_SEEDS[@]}"; do for fold in "${FOLDS[@]}"; do
      run_seg pointnet_seg processed/raw_seg/v2_natural runs/research_segcls_full/seg_pointnet_natural "pointnet_nat_s${seed}_f${fold}" "$seed" "$fold"
    done; done ;;
  pn2_bal)
    for seed in "${NEW_SEEDS[@]}"; do for fold in "${FOLDS[@]}"; do
      run_seg pointnet2 processed/raw_seg/v1 runs/research_segcls_full/seg_pointnet2 "pointnet2_bal_s${seed}_f${fold}" "$seed" "$fold"
    done; done ;;
  pn2_nat)
    for seed in "${NEW_SEEDS[@]}"; do for fold in "${FOLDS[@]}"; do
      run_seg pointnet2 processed/raw_seg/v2_natural runs/research_segcls_full/seg_pointnet2_natural "pointnet2_nat_s${seed}_f${fold}" "$seed" "$fold"
    done; done ;;
  dgcnn_bal)
    for seed in "${NEW_SEEDS[@]}"; do for fold in "${FOLDS[@]}"; do
      run_seg dgcnn_v2 processed/raw_seg/v1 runs/raw_seg "dgcnn_v2_s${seed}_fold${fold}" "$seed" "$fold"
    done; done ;;
  dgcnn_nat)
    for seed in "${NEW_SEEDS[@]}"; do for fold in "${FOLDS[@]}"; do
      run_seg dgcnn_v2 processed/raw_seg/v2_natural runs/research_segcls_full/seg_dgcnn_v2_natural "dgcnn_nat_s${seed}_f${fold}" "$seed" "$fold"
    done; done ;;
  pt_bal)
    for seed in "${NEW_SEEDS[@]}"; do for fold in "${FOLDS[@]}"; do
      run_seg point_transformer processed/raw_seg/v1 runs/research_segcls_full/seg_pt_tuned "pt_bal_s${seed}_f${fold}" "$seed" "$fold" "--pt-dim 96 --pt-depth 4 --pt-k 16"
    done; done ;;
  pt_nat)
    for seed in "${NEW_SEEDS[@]}"; do for fold in "${FOLDS[@]}"; do
      run_seg point_transformer processed/raw_seg/v2_natural runs/research_segcls_full/seg_pt_tuned_natural "pt_nat_s${seed}_f${fold}" "$seed" "$fold" "--pt-dim 96 --pt-depth 4 --pt-k 16"
    done; done ;;
  *)
    echo "Unknown job: $JOB"; exit 1 ;;
esac

echo "[$(date +%H:%M:%S)] GPU $GPU JOB $JOB DONE"
