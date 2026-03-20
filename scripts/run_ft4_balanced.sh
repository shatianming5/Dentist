#!/bin/bash
# Launch MV-ViT-ft4 additional seeds (42, 7) for both protocols
# GPU 0: balanced protocol
# GPU 3: natural protocol
set -e
cd /mnt/SSD_4TB/zechuan/Dentist

SEEDS="42 7"
FOLDS="0 1 2 3 4"

echo "=== MV-ViT-ft4 Balanced (GPU 0) ==="
for seed in $SEEDS; do
  for fold in $FOLDS; do
    exp="dinov3_ft4_bal_s${seed}_fold${fold}"
    if [ -f "runs/dinov3_finetune/ft4_balanced/${exp}/results.json" ]; then
      echo "SKIP $exp"
      continue
    fi
    echo "RUN $exp ..."
    python3 scripts/train_mvvit_ft4.py \
      --data-root processed/raw_seg/v1 \
      --kfold metadata/splits_raw_case_kfold.json \
      --fold "$fold" --seed "$seed" \
      --run-root runs/dinov3_finetune/ft4_balanced \
      --exp-name "$exp" \
      --protocol balanced \
      --epochs 50 --patience 15 \
      --device cuda:0 2>&1 | grep -E "\[data\]|\[model\]|\[TEST\]|\[done\]|\[early"
    echo "  DONE $exp"
  done
done
echo "=== ALL BALANCED DONE ==="
