#!/bin/bash
# Phase B4 — fine-tune UNet pretrained on synthetic (B1 FPV, B2 deep_space)
# on adapted RadioML 2018 subsets. 2 scenarios × 2 noise × 2 seeds = 8 runs.
set -e
cd /Users/serhiizabolotnii/Projects/signal-denoising
source .venv/bin/activate 2>/dev/null || true

FPV_DS_SRC=data_generation/datasets/fpv_telemetry_polygauss_qpsk_bs1024_n100000_953c56e8
DS_DS_SRC=data_generation/datasets/deep_space_polygauss_qpsk_bs1024_n400000_0310b7e7
REAL_FPV=data_generation/datasets/radioml2018_bpsk_qpsk_fpv
REAL_DS=data_generation/datasets/radioml2018_bpsk_qpsk_deep_space

TS=$(date +%Y%m%d_%H%M%S)
LOG=experiments/results/b4_finetune_${TS}.log
mkdir -p experiments/results
echo "========== Phase B4 — fine-tune UNet on real RadioML ==========" | tee "$LOG"
echo "Start: $(date)" | tee -a "$LOG"
echo "Config: 10 epochs, lr=1e-4, partial=0.25, cpu, seed x2" | tee -a "$LOG"

# (scenario, source_dataset, real_dataset, [ (seed, noise, src_run_id) ... ])
# FPV: seeds 42, 43
declare -A FPV_RUNS=(
    ["42_gaussian"]="run_20260421_267176a3"
    ["42_non_gaussian"]="run_20260421_92a2e0c4"
    ["43_gaussian"]="run_20260421_fe4d376a"
    ["43_non_gaussian"]="run_20260422_d98d21b3"
)
declare -A DS_RUNS=(
    ["42_gaussian"]="run_20260423_ed46f843"
    ["42_non_gaussian"]="run_20260423_3d8957d2"
    ["43_gaussian"]="run_20260423_2b4b99cc"
    ["43_non_gaussian"]="run_20260424_20fc5edf"
)

echo "" | tee -a "$LOG"
echo "--- FPV fine-tune (4 runs) ---" | tee -a "$LOG"
i=0
for key in "${!FPV_RUNS[@]}"; do
    i=$((i+1))
    seed=${key%_*}
    noise=${key#*_}
    src="${FPV_RUNS[$key]}"
    echo "### [$i/4] FPV seed=$seed noise=$noise src=$src $(date +%H:%M:%S) ###" | tee -a "$LOG"
    python experiments/b4_finetune.py \
        --b1-run "$FPV_DS_SRC/runs/$src" \
        --real-dataset "$REAL_FPV" \
        --model unet --noise-type "$noise" \
        --seed "$seed" --epochs 10 --lr 1e-4 --partial 0.25 \
        --device cpu --nperseg 128 2>&1 | tee -a "$LOG" | tail -3
done

echo "" | tee -a "$LOG"
echo "--- deep_space fine-tune (4 runs) ---" | tee -a "$LOG"
i=0
for key in "${!DS_RUNS[@]}"; do
    i=$((i+1))
    seed=${key%_*}
    noise=${key#*_}
    src="${DS_RUNS[$key]}"
    echo "### [$i/4] deep_space seed=$seed noise=$noise src=$src $(date +%H:%M:%S) ###" | tee -a "$LOG"
    python experiments/b4_finetune.py \
        --b1-run "$DS_DS_SRC/runs/$src" \
        --real-dataset "$REAL_DS" \
        --model unet --noise-type "$noise" \
        --seed "$seed" --epochs 10 --lr 1e-4 --partial 0.25 \
        --device cpu --nperseg 128 2>&1 | tee -a "$LOG" | tail -3
done

echo "" | tee -a "$LOG"
echo "========== Phase B4 complete $(date) ==========" | tee -a "$LOG"
echo "Log: $LOG"
