#!/bin/bash
# Phase B2 — deep_space sanity check
# --------------------------------
# Goal: go/no-go gate before full B2 deep_space sweep. Validate that:
#   (1) pipeline runs to completion on deep_space SNR range (−20..0 dB);
#   (2) U-Net and ResNet learn a non-degenerate mapping;
#   (3) Hybrid-DSGE does not immediately collapse (as in FPV NG seeds 42/44);
#   (4) training time scales as expected.
#
# Scale: 10% of 400k = 40k samples, seed=42, both noise types.
# Estimated compute: ~2h (≈4× FPV 25% per-invocation / 2 noise types).
#
# If sanity passes → kick off b2_deep_space_main.sh (50% × 3 seeds, ~20h).
# If sanity fails → diagnose before committing to the full sweep.

set -e
cd /Users/serhiizabolotnii/Projects/signal-denoising
source .venv/bin/activate 2>/dev/null || true

DS="data_generation/datasets/deep_space_polygauss_qpsk_bs1024_n400000_0310b7e7"
SEED=42
NOISE_TYPES=(gaussian non_gaussian)
MODELS="unet,resnet,hybrid,wavelet"
EPOCHS=15
PARTIAL=0.10

START_TS=$(date +"%Y%m%d_%H%M%S")
LOG="experiments/results/b2_sanity_${START_TS}.log"
mkdir -p experiments/results

echo "========== Phase B2 — deep_space SANITY ==========" | tee "$LOG"
echo "Start: $(date)" | tee -a "$LOG"
echo "Dataset: $DS" | tee -a "$LOG"
echo "Models: $MODELS" | tee -a "$LOG"
echo "Seed: $SEED | Noises: ${NOISE_TYPES[*]}" | tee -a "$LOG"
echo "Epochs: $EPOCHS | Data fraction: $PARTIAL | Device: cpu" | tee -a "$LOG"
echo "==================================================" | tee -a "$LOG"

i=0
total=${#NOISE_TYPES[@]}

for noise in "${NOISE_TYPES[@]}"; do
    i=$((i+1))
    echo "" | tee -a "$LOG"
    echo "### [$i/$total] seed=$SEED noise=$noise $(date +%H:%M:%S) ###" | tee -a "$LOG"
    python train/train_all.py \
        --dataset "$DS" \
        --models "$MODELS" \
        --noise-types "$noise" \
        --epochs $EPOCHS \
        --partial-train $PARTIAL \
        --device cpu \
        --nperseg 128 \
        --seed $SEED 2>&1 | tee -a "$LOG"
    echo "### [$i/$total] done $(date +%H:%M:%S) ###" | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "========== Phase B2 sanity complete $(date) ==========" | tee -a "$LOG"
echo "Log: $LOG" | tee -a "$LOG"
