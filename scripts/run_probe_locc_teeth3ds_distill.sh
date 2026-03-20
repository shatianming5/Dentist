#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TEACHER_CKPT="${TEACHER_CKPT:-runs/pretrain/teeth3ds_fdi_pointnet_seed1337/ckpt_best.pt}"
LOG_DIR="${LOG_DIR:-runs/research_segcls_probe_logs}"
GPU_0="${GPU_0:-0}"
GPU_1="${GPU_1:-1}"
GPU_2="${GPU_2:-3}"
GPU_3="${GPU_3:-4}"

mkdir -p "$LOG_DIR"

if [[ ! -f "$TEACHER_CKPT" ]]; then
  echo "Missing teacher checkpoint: $TEACHER_CKPT" >&2
  exit 2
fi

run_one() {
  local gpu="$1"
  local fold="$2"
  local seed="$3"
  local base_tag="$4"
  local aux="$5"
  local cons="$6"
  local t3d="$7"
  local t3d_tag
  t3d_tag="$(printf 't3d%03d' "$(python3 - <<PY
w = float(${t3d})
print(int(round(w * 100)))
PY
)")"
  local exp="joint_locc_teeth3dsdistill_fold${fold}_seed${seed}_${base_tag}_${t3d_tag}"
  local out="runs/research_segcls_probe/locc_teeth3ds_distill/${exp}/metrics.json"
  local log="${LOG_DIR}/${exp}.log"
  if [[ -f "$out" ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] SKIP $exp"
    return 0
  fi
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] RUN $exp gpu=${gpu} log=${log}"
  CUDA_VISIBLE_DEVICES="$gpu" python3 scripts/phase3_train_raw_segcls_joint.py \
    --data-root processed/raw_seg/v1 \
    --run-root runs/research_segcls_probe/locc_teeth3ds_distill \
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
    --teeth3ds-teacher-ckpt "$TEACHER_CKPT" \
    --teeth3ds-teacher-weight "$t3d" \
    --teeth3ds-teacher-points 1024 \
    --kfold metadata/splits_raw_case_kfold.json \
    --fold "$fold" \
    2>&1 | tee "$log"
}

run_one "$GPU_0" 2 2021 aux050_cons010 0.50 0.10 0.10 &
pid0=$!
run_one "$GPU_1" 2 2021 aux050_cons010 0.50 0.10 0.25 &
pid1=$!
run_one "$GPU_2" 3 1337 aux025_cons025 0.25 0.25 0.10 &
pid2=$!
run_one "$GPU_3" 3 1337 aux025_cons025 0.25 0.25 0.25 &
pid3=$!

wait "$pid0"
wait "$pid1"
wait "$pid2"
wait "$pid3"
