#!/bin/bash
# Phase B2 — deep_space MAIN (3 seeds × 25% data × 30 epochs)
# -----------------------------------------------------------
# Goal: publishable cross-seed statistics for the central hypothesis
# (NG-training > G-training) on deep_space (−20..0 dB) and deterministic
# confirmation that HybridDSGE collapses deterministically under low-SNR.
#
# Sanity (§12.5) on seed=42 / 10% / 15ep already showed:
#   UNet G→G=+5.74, NG→NG=+7.32 (Δ=+1.58 dB)
#   ResNet G→G=+5.36, NG→NG=+6.46 (Δ=+1.10 dB)
#   Hybrid collapses on both noise types (~0 dB)
#
# This run provides μ±σ over 3 seeds for publication tables and repeats
# Hybrid collapse across seeds to confirm it is not a seed-specific artefact.
#
# Scope: 3 seeds × 2 noise types × 4 models = 24 model-trainings.
# Expected compute: ~12-16h CPU (scaling from 6h sanity: 2.5× data × 2× ep × 3 seeds
# / 2 noise = ~15× per noise-type, but amortised over shared data load).

set -e
cd /Users/serhiizabolotnii/Projects/signal-denoising
source .venv/bin/activate 2>/dev/null || true

DS="data_generation/datasets/deep_space_polygauss_qpsk_bs1024_n400000_0310b7e7"
SEEDS=(42 43 44)
NOISE_TYPES=(gaussian non_gaussian)
MODELS="unet,resnet,hybrid,wavelet"
EPOCHS=30
PARTIAL=0.25

START_TS=$(date +"%Y%m%d_%H%M%S")
LOG="experiments/results/b2_main_${START_TS}.log"
mkdir -p experiments/results

echo "========== Phase B2 — deep_space MAIN ==========" | tee "$LOG"
echo "Start: $(date)" | tee -a "$LOG"
echo "Dataset: $DS" | tee -a "$LOG"
echo "Models: $MODELS" | tee -a "$LOG"
echo "Seeds: ${SEEDS[*]} | Noises: ${NOISE_TYPES[*]}" | tee -a "$LOG"
echo "Epochs: $EPOCHS | Data fraction: $PARTIAL | Device: cpu" | tee -a "$LOG"
echo "=================================================" | tee -a "$LOG"

total=$(( ${#SEEDS[@]} * ${#NOISE_TYPES[@]} ))
i=0

for seed in "${SEEDS[@]}"; do
    for noise in "${NOISE_TYPES[@]}"; do
        i=$((i+1))
        echo "" | tee -a "$LOG"
        echo "### [$i/$total] seed=$seed noise=$noise $(date +%H:%M:%S) ###" | tee -a "$LOG"
        python train/train_all.py \
            --dataset "$DS" \
            --models "$MODELS" \
            --noise-types "$noise" \
            --epochs $EPOCHS \
            --partial-train $PARTIAL \
            --device cpu \
            --nperseg 128 \
            --seed $seed 2>&1 | tee -a "$LOG"
        echo "### [$i/$total] done $(date +%H:%M:%S) ###" | tee -a "$LOG"
    done
done

echo "" | tee -a "$LOG"
echo "========== Phase B2 main complete $(date) ==========" | tee -a "$LOG"
echo "Log: $LOG" | tee -a "$LOG"
