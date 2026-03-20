#!/bin/bash
# Launch MV-ViT-ft4 additional seeds (42, 7) for natural protocol
set -e
cd /mnt/SSD_4TB/zechuan/Dentist

SEEDS="42 7"
FOLDS="0 1 2 3 4"

echo "=== MV-ViT-ft4 Natural (GPU 3) ==="
for seed in $SEEDS; do
  for fold in $FOLDS; do
    exp="dinov3_ft4_s${seed}_fold${fold}"
    if [ -f "runs/dinov3_finetune/ft4/${exp}/results.json" ]; then
      echo "SKIP $exp"
      continue
    fi
    echo "RUN $exp ..."
    python3 scripts/train_mvvit_ft4.py \
      --data-root processed/raw_seg/v2_natural \
      --kfold metadata/splits_raw_case_kfold.json \
      --fold "$fold" --seed "$seed" \
      --run-root runs/dinov3_finetune/ft4 \
      --exp-name "$exp" \
      --protocol natural \
      --epochs 50 --patience 15 \
      --device cuda:3 2>&1 | grep -E "\[data\]|\[model\]|\[TEST\]|\[done\]|\[early"
    echo "  DONE $exp"
  done
done
echo "=== ALL NATURAL DONE ==="
