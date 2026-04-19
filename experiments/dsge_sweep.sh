#!/bin/bash
# DSGE Experiment: compare Variant A vs B, different bases, S=2,3
# No Transformer — only Hybrid DSGE-UNet
# Both noise types: gaussian + non_gaussian

set -e

DATASET="data_generation/datasets/deep_space_polygauss_qpsk_bs1024_n400000_0310b7e7"
EPOCHS=50
PARTIAL=0.05
DEVICE=cpu

echo "============================================================"
echo "DSGE Sweep Experiment"
echo "Dataset: $DATASET"
echo "Epochs: $EPOCHS, Partial: $PARTIAL, Device: $DEVICE"
echo "============================================================"

# Variant A: reconstruction + residual (3 channels)
# Variant B: reconstruction + weighted basis (2+S channels)

for VARIANT in A B; do
  for BASIS in fractional polynomial robust; do
    for ORDER in 2 3; do
      # robust basis with S=2 is fine (tanh, sigmoid)
      for NOISE in gaussian non_gaussian; do
        echo ""
        echo "########## Variant=$VARIANT  Basis=$BASIS  S=$ORDER  Noise=$NOISE ##########"
        python train/training_hybrid.py \
          --dataset "$DATASET" \
          --noise-type "$NOISE" \
          --epochs "$EPOCHS" \
          --batch-size 4096 \
          --lr 3e-4 \
          --dsge-basis "$BASIS" \
          --dsge-orders "$ORDER" \
          --dsge-variant "$VARIANT" \
          --nperseg 128 \
          --seed 42 \
          --partial-train "$PARTIAL" \
          --device "$DEVICE" \
          || echo "ERROR: Variant=$VARIANT Basis=$BASIS S=$ORDER Noise=$NOISE"
      done
    done
  done
done

echo ""
echo "============================================================"
echo "DSGE Sweep Complete!"
echo "============================================================"
