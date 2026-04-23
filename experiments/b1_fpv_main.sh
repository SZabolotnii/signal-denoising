#!/bin/bash
# Phase B1 — FPV main experiment
# --------------------------------
# Goal: establish whether DSGE-UNet helps vs baseline U-Net / Transformer / Wavelet
# on FPV telemetry, with cross-seed reproducibility (3 seeds).
#
# Config rationale:
#   • data_fraction=0.25 — matches A2 so hybrid numbers are directly comparable
#     with H1/H3/H4 (same 25k training samples). Can scale later if budget allows.
#   • epochs=30 with patience=5 early stop — empirically enough for FPV models to
#     converge (per H1 which used 8 and showed stable basins already).
#   • seeds=42,43,44 — matches A1 repro seeds.
#   • models: unet, resnet, hybrid, wavelet (skip VAE — known bad; skip
#     transformer — CPU batch=128 at ~30min/epoch would push total to 100h.
#     Transformer baselines retained from H1 prior runs + can be added as
#     single-seed follow-up if time permits).
#   • Default losses: MSE for gaussian, SmoothL1(β=0.02) for non_gaussian.
#     Bimodality on non_gaussian expected for DSGE-UNet; documented in A2-H1.
#     If issue repeats at full scale, follow-up with MSE override.
#   • per-model defaults for batch_size/lr (tuned for ~8GB GPU but run fine on CPU).
#
# Estimated compute: ~16h on M3 Max CPU (6 invocations × 4 models).
# Per-invocation measured: UNet ~86m, ResNet ~45m, Hybrid ~25m, Wavelet ~5m
# → ~2.7h/inv × 6 = ~16h total. (Transformer excluded: ~16h/run on CPU.)
#
# Output: each invocation creates its own run_dir in
#   data_generation/datasets/fpv_.../runs/run_YYYYMMDD_<uuid>/
# Aggregation: downstream `train/compare_report.py --run <dir>` for each run.

set -e
cd /Users/serhiizabolotnii/Projects/signal-denoising
source .venv/bin/activate 2>/dev/null || true

DS="data_generation/datasets/fpv_telemetry_polygauss_qpsk_bs1024_n100000_953c56e8"
SEEDS=(42 43 44)
NOISE_TYPES=(gaussian non_gaussian)
MODELS="unet,resnet,hybrid,wavelet"
EPOCHS=30
PARTIAL=0.25

START_TS=$(date +"%Y%m%d_%H%M%S")
LOG="experiments/results/b1_fpv_main_${START_TS}.log"
mkdir -p experiments/results

echo "========== Phase B1 — FPV main ==========" | tee "$LOG"
echo "Start: $(date)" | tee -a "$LOG"
echo "Dataset: $DS" | tee -a "$LOG"
echo "Models: $MODELS" | tee -a "$LOG"
echo "Seeds: ${SEEDS[*]} | Noises: ${NOISE_TYPES[*]}" | tee -a "$LOG"
echo "Epochs: $EPOCHS | Data fraction: $PARTIAL | Device: cpu" | tee -a "$LOG"
echo "==========================================" | tee -a "$LOG"

i=0
total=$((${#SEEDS[@]} * ${#NOISE_TYPES[@]}))

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
echo "========== Phase B1 complete $(date) ==========" | tee -a "$LOG"
echo "Log: $LOG" | tee -a "$LOG"
