#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

INIT_CKPT="${INIT_CKPT:-runs/pretrain/teeth3ds_fdi_pointnet_seed1337/ckpt_best.pt}"
GPU_FOLD2="${GPU_FOLD2:-0}"
GPU_FOLD3="${GPU_FOLD3:-1}"
LOG_DIR="${LOG_DIR:-runs/research_segcls_probe_logs}"

mkdir -p "$LOG_DIR"

if [[ ! -f "$INIT_CKPT" ]]; then
  echo "Missing init checkpoint: $INIT_CKPT" >&2
  exit 2
fi

run_one() {
  local gpu="$1"
  local fold="$2"
  local seed="$3"
  local tag="$4"
  local aux="$5"
  local cons="$6"
  local exp="joint_locc_teeth3dsinit_fold${fold}_seed${seed}_${tag}"
  local out="runs/research_segcls_probe/locc_teeth3ds_init/${exp}/metrics.json"
  local log="${LOG_DIR}/${exp}.log"
  if [[ -f "$out" ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] SKIP $exp"
    return 0
  fi
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] RUN $exp gpu=${gpu} log=${log}"
  CUDA_VISIBLE_DEVICES="$gpu" python3 scripts/phase3_train_raw_segcls_joint.py \
    --data-root processed/raw_seg/v1 \
    --run-root runs/research_segcls_probe/locc_teeth3ds_init \
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
    --init-feat "$INIT_CKPT" \
    --kfold metadata/splits_raw_case_kfold.json \
    --fold "$fold" \
    2>&1 | tee "$log"
}

run_one "$GPU_FOLD2" 2 2021 aux050_cons010 0.50 0.10 &
pid_fold2=$!
run_one "$GPU_FOLD3" 3 1337 aux025_cons025 0.25 0.25 &
pid_fold3=$!

wait "$pid_fold2"
wait "$pid_fold3"
