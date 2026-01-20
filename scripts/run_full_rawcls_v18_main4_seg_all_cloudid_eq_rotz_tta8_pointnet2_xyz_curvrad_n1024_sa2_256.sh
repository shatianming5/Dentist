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

PRETRAIN_EXP="teeth3ds_fdi_pointnet2"
PRETRAIN_DIR="runs/pretrain/${PRETRAIN_EXP}_seed${PRETRAIN_SEED}"
PRETRAIN_CKPT="${PRETRAIN_DIR}/ckpt_best.pt"

if [[ ! -f "${PRETRAIN_CKPT}" ]]; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] PRETRAIN Teeth3DS FDI (${PRETRAIN_EXP}) -> ${PRETRAIN_DIR}"
  python3 scripts/phase2_train_teeth3ds_fdi_cls.py \
    --device cuda \
    --model pointnet2 \
    --exp-name "${PRETRAIN_EXP}" \
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

CFG="configs/raw_cls/exp/teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1_seg_all_cloudid_eq_xyz_curvrad_pointnet2_n1024_sa2_256.yaml"
EXP="teeth3ds_fdi_pretrain_supcon_rotz_tta8_posfeat_ls0_dropout0p1_seg_all_cloudid_eq_xyz_curvrad_pointnet2_n1024_sa2_256"
MODEL="pointnet2"

for fold in "${FOLDS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_dir="runs/raw_cls/v18_main4_seg_all_cloudid_eq/${EXP}/${MODEL}/fold=${fold}/seed=${seed}"
    if [[ -f "${run_dir}/metrics.json" ]]; then
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] SKIP ${run_dir} (metrics.json exists)"
      continue
    fi
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] RUN cfg=${CFG} exp=${EXP} fold=${fold} seed=${seed}"
    python3 scripts/train.py \
      --config "${CFG}" \
      --fold "${fold}" \
      --seed "${seed}" \
      --set runtime.device=cuda \
      --set "model.name=${MODEL}" \
      --set "train.init_feat=${PRETRAIN_CKPT}"
  done
done

python3 scripts/aggregate_runs.py \
  --root runs/raw_cls/v18_main4_seg_all_cloudid_eq \
  --out paper_tables/raw_cls_summary_v18_main4_seg_all_cloudid_eq_rotz_tta8_pointnet2_xyz_curvrad_n1024_sa2_256.md \
  --seeds 1337,2020,2021 \
  --folds 0,1,2,3,4

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] DONE aggregate -> paper_tables/raw_cls_summary_v18_main4_seg_all_cloudid_eq_rotz_tta8_pointnet2_xyz_curvrad_n1024_sa2_256.md"

