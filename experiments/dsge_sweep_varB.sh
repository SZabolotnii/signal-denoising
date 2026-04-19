#!/bin/bash
set -e
cd /Users/serhiizabolotnii/Projects/signal-denoising
source .venv/bin/activate

DS="data_generation/datasets/deep_space_polygauss_qpsk_bs1024_n400000_0310b7e7"
COMMON="--epochs 50 --partial-train 0.05 --batch-size 4096 --lr 3e-4 --nperseg 128 --seed 42 --device cpu"

i=12
total=24

for BASIS in fractional polynomial robust; do
  for ORDER in 2 3; do
    for NOISE in gaussian non_gaussian; do
      i=$((i+1))
      echo ""
      echo "########## [$i/$total] vB ${BASIS} S=${ORDER} ${NOISE} ##########"
      python train/training_hybrid.py \
        --dataset "$DS" \
        --noise-type "$NOISE" \
        --dsge-basis "$BASIS" \
        --dsge-orders "$ORDER" \
        --dsge-variant B \
        $COMMON \
        2>&1 || echo "  FAILED: vB ${BASIS} S=${ORDER} ${NOISE}"
      echo "  ---- done [$i/$total] ----"
    done
  done
done

echo ""
echo "========== DSGE SWEEP VARIANT B COMPLETE =========="
