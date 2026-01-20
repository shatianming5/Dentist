#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PRETRAIN_SEED="${PRETRAIN_SEED:-1337}"
PRETRAIN_N_POINTS="${PRETRAIN_N_POINTS:-1024}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-128}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-200}"
PRETRAIN_PATIENCE="${PRETRAIN_PATIENCE:-25}"

PRETRAIN_DIR="runs/pretrain/teeth3ds_fdi_pointnet_seed${PRETRAIN_SEED}"
PRETRAIN_CKPT="${PRETRAIN_DIR}/ckpt_best.pt"

if [[ ! -f "${PRETRAIN_CKPT}" ]]; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] PRETRAIN Teeth3DS FDI -> ${PRETRAIN_DIR}"
  python3 scripts/phase2_train_teeth3ds_fdi_cls.py \
    --device cuda \
    --seed "${PRETRAIN_SEED}" \
    --n-points "${PRETRAIN_N_POINTS}" \
    --batch-size "${PRETRAIN_BATCH_SIZE}" \
    --epochs "${PRETRAIN_EPOCHS}" \
    --patience "${PRETRAIN_PATIENCE}" \
    --balanced-sampler
else
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] SKIP PRETRAIN (exists) ${PRETRAIN_CKPT}"
fi

SEEDS=(1337 2020 2021)
FOLDS=(0 1 2 3 4)

run_suite() {
  local cfg="$1"
  local exp="$2"
  local model="pointnet"

  for fold in "${FOLDS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      local run_dir="runs/raw_cls/v13_main4/${exp}/${model}/fold=${fold}/seed=${seed}"
      if [[ -f "${run_dir}/metrics.json" ]]; then
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] SKIP ${run_dir} (metrics.json exists)"
        continue
      fi
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] RUN cfg=${cfg} exp=${exp} fold=${fold} seed=${seed}"
      python3 scripts/train.py \
        --config "${cfg}" \
        --fold "${fold}" \
        --seed "${seed}" \
        --set runtime.device=cuda \
        --set "model.name=${model}" \
        --set "train.init_feat=${PRETRAIN_CKPT}"
    done
  done
}

run_suite "configs/raw_cls/exp/teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0.yaml" "teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0"
run_suite "configs/raw_cls/exp/teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_dropout0p1.yaml" "teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_dropout0p1"
run_suite "configs/raw_cls/exp/teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_dropout0.yaml" "teeth3ds_fdi_pretrain_supcon_norot_tta0_posfeat_ls0_dropout0"

python3 scripts/aggregate_runs.py --root runs/raw_cls/v13_main4 --out paper_tables/raw_cls_summary.md --seeds 1337,2020,2021 --folds 0,1,2,3,4
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] DONE aggregate -> paper_tables/raw_cls_summary.md"

