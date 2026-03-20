#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SEEDS=(1337 2020 2021)
FOLDS=(0 1 2 3 4)

build_datasets() {
  python3 scripts/phase3_build_raw_cls_from_raw_seg.py \
    --seg-root processed/raw_seg/v1 \
    --out processed/raw_cls_from_raw_seg/v1_all \
    --mode all

  python3 scripts/phase3_build_raw_cls_from_raw_seg.py \
    --seg-root processed/raw_seg/v1 \
    --out processed/raw_cls_from_raw_seg/v1_gt_seg \
    --mode gt_seg
}

run_seg_baseline() {
  for fold in "${FOLDS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      local exp="seg_pointnet_fold${fold}_seed${seed}"
      local out="runs/research_segcls_full/seg_pointnet/${exp}/results.json"
      if [[ -f "$out" ]]; then
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] SKIP $exp"
        continue
      fi
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] RUN seg baseline $exp"
      python3 scripts/phase3_train_raw_seg.py \
        --data-root processed/raw_seg/v1 \
        --run-root runs/research_segcls_full/seg_pointnet \
        --exp-name "$exp" \
        --seed "$seed" \
        --device cuda \
        --model pointnet_seg \
        --epochs 40 \
        --patience 10 \
        --batch-size 16 \
        --n-points 8192 \
        --num-workers 0 \
        --kfold metadata/splits_raw_case_kfold.json \
        --fold "$fold"
    done
  done
}

run_cls_baseline() {
  local data_root="$1"
  local variant="$2"
  for fold in "${FOLDS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      local exp="${variant}_pointnet_fold${fold}_seed${seed}"
      local out="runs/research_segcls_full/${variant}_pointnet/${exp}/metrics.json"
      if [[ -f "$out" ]]; then
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] SKIP $exp"
        continue
      fi
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] RUN cls baseline $exp"
      python3 scripts/phase3_train_raw_cls_baseline.py \
        --data-root "$data_root" \
        --run-root "runs/research_segcls_full/${variant}_pointnet" \
        --exp-name "$exp" \
        --seed "$seed" \
        --device cuda \
        --model pointnet \
        --epochs 120 \
        --patience 25 \
        --batch-size 16 \
        --n-points 4096 \
        --num-workers 0 \
        --balanced-sampler \
        --label-smoothing 0.1 \
        --tta 8 \
        --dropout 0.1 \
        --kfold metadata/splits_raw_case_kfold.json \
        --fold "$fold"
    done
  done
}

run_joint_main() {
  for fold in "${FOLDS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      local exp="joint_pointnet_fold${fold}_seed${seed}"
      local out="runs/research_segcls_full/joint_pointnet/${exp}/metrics.json"
      if [[ -f "$out" ]]; then
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] SKIP $exp"
        continue
      fi
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] RUN joint main $exp"
      python3 scripts/phase3_train_raw_segcls_joint.py \
        --data-root processed/raw_seg/v1 \
        --run-root runs/research_segcls_full/joint_pointnet \
        --exp-name "$exp" \
        --seed "$seed" \
        --device cuda \
        --epochs 120 \
        --patience 25 \
        --batch-size 16 \
        --n-points 8192 \
        --num-workers 0 \
        --balanced-sampler \
        --label-smoothing 0.1 \
        --seg-loss-weight 0.5 \
        --cls-train-mask gt \
        --cls-topk-ratio 0.5 \
        --tta 8 \
        --kfold metadata/splits_raw_case_kfold.json \
        --fold "$fold"
    done
  done
}

build_datasets
run_seg_baseline
run_cls_baseline processed/raw_cls_from_raw_seg/v1_all all
run_cls_baseline processed/raw_cls_from_raw_seg/v1_gt_seg gtseg
run_joint_main
python3 scripts/summarize_research_segcls_runs.py --root runs/research_segcls_full --out-prefix paper_tables/research_segcls_full_summary
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] DONE full research seg+cls suite"
