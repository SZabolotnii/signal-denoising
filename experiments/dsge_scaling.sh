#!/bin/bash
set -e
cd /Users/serhiizabolotnii/Projects/signal-denoising
source .venv/bin/activate

DS="data_generation/datasets/deep_space_polygauss_qpsk_bs1024_n400000_0310b7e7"
COMMON="--epochs 50 --partial-train 0.05 --batch-size 4096 --lr 3e-4 --nperseg 128 --seed 42 --device cpu --dsge-basis fractional --dsge-orders 3 --dsge-variant A"

i=0
total=6

for WIDTH in 16 32 64; do
  for NOISE in gaussian non_gaussian; do
    i=$((i+1))
    echo ""
    echo "########## [$i/$total] width=$WIDTH noise=$NOISE ##########"
    python train/training_hybrid.py \
      --dataset "$DS" \
      --noise-type "$NOISE" \
      --unet-width "$WIDTH" \
      $COMMON \
      2>&1 || echo "  FAILED: width=$WIDTH noise=$NOISE"
    echo "  ---- done [$i/$total] ----"
  done
done

echo ""
echo "========== DSGE SCALING COMPLETE =========="
