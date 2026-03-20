#!/bin/bash
# Run DGCNN fine-tuning experiment: pretrained vs scratch, under both protocols
# Requires: runs/teeth3ds_pretrain/dgcnn_pretrain/ckpt_best.pt

set -e
cd /mnt/SSD_4TB/zechuan/Dentist

PRETRAINED_CKPT="runs/teeth3ds_pretrain/dgcnn_pretrain/ckpt_best.pt"
SPLIT="metadata/splits_raw_case_kfold.json"
GPU=0

echo "=== DGCNN Fine-tuning Experiment ==="
echo "Pretrained checkpoint: $PRETRAINED_CKPT"
echo ""

for SEED in 1337 2020 2021; do
  for FOLD in 0 1 2 3 4; do
    echo "--- Seed=$SEED Fold=$FOLD ---"
    
    # 1. Pretrained → Balanced
    OUT="runs/teeth3ds_pretrain/ft_bal_pretrained_s${SEED}_f${FOLD}"
    if [ ! -f "$OUT/ckpt_best.pt" ]; then
      CUDA_VISIBLE_DEVICES=$GPU python3 scripts/pretrain_teeth3ds.py \
        --mode finetune --gpu 0 --seed $SEED \
        --pretrained_ckpt "$PRETRAINED_CKPT" \
        --finetune_data processed/raw_seg/v1 \
        --finetune_split "$SPLIT" --finetune_fold $FOLD \
        --output_dir "$OUT"
    fi
    
    # 2. Scratch → Balanced (already have these from existing experiments)
    
    # 3. Pretrained → Natural
    OUT="runs/teeth3ds_pretrain/ft_nat_pretrained_s${SEED}_f${FOLD}"
    if [ ! -f "$OUT/ckpt_best.pt" ]; then
      CUDA_VISIBLE_DEVICES=$GPU python3 scripts/pretrain_teeth3ds.py \
        --mode finetune --gpu 0 --seed $SEED \
        --pretrained_ckpt "$PRETRAINED_CKPT" \
        --finetune_data processed/raw_seg/v2_natural \
        --finetune_split "$SPLIT" --finetune_fold $FOLD \
        --output_dir "$OUT"
    fi
    
    # 4. Scratch → Natural (already have from existing experiments)
    
  done
done

echo ""
echo "=== Experiment complete ==="
