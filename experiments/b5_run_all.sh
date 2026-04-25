#!/bin/bash
# Phase B5 — fine-tune UNet pretrained on synthetic, on synthetic+real-noise
# datasets. 2 scenarios × 2 noise × 2 seeds = 8 runs. macOS bash 3.2 compat.
set -e
cd /Users/serhiizabolotnii/Projects/signal-denoising
source .venv/bin/activate 2>/dev/null || true

FPV_SRC=data_generation/datasets/fpv_telemetry_polygauss_qpsk_bs1024_n100000_953c56e8
DS_SRC=data_generation/datasets/deep_space_polygauss_qpsk_bs1024_n400000_0310b7e7
B5_FPV=data_generation/datasets/fpv_realnoise_bpsk_qpsk
B5_DS=data_generation/datasets/deep_space_realnoise_bpsk_qpsk

TS=$(date +%Y%m%d_%H%M%S)
LOG=experiments/results/b5_finetune_${TS}.log
mkdir -p experiments/results
echo "========== Phase B5 — synthetic+real-noise fine-tune ==========" | tee "$LOG"
echo "Start: $(date)" | tee -a "$LOG"
echo "Config: 10 ep, lr=1e-4, partial=0.25, cpu" | tee -a "$LOG"

# Parallel arrays: scen seed noise src_run_id
SCEN=(fpv fpv fpv fpv ds ds ds ds)
SEED=(42  42  43  43  42 42 43 43)
NOISE=(gaussian non_gaussian gaussian non_gaussian gaussian non_gaussian gaussian non_gaussian)
SRC=(
  run_20260421_267176a3
  run_20260421_92a2e0c4
  run_20260421_fe4d376a
  run_20260422_d98d21b3
  run_20260423_ed46f843
  run_20260423_3d8957d2
  run_20260423_2b4b99cc
  run_20260424_20fc5edf
)

total=${#SCEN[@]}
for i in $(seq 0 $((total-1))); do
    n=$((i+1))
    s=${SCEN[$i]}
    seed=${SEED[$i]}
    noise=${NOISE[$i]}
    src=${SRC[$i]}
    if [ "$s" = "fpv" ]; then
        SRC_DS=$FPV_SRC; B5_DS_PATH=$B5_FPV
    else
        SRC_DS=$DS_SRC; B5_DS_PATH=$B5_DS
    fi
    echo "" | tee -a "$LOG"
    echo "### [$n/$total] $s seed=$seed noise=$noise src=$src $(date +%H:%M:%S) ###" | tee -a "$LOG"
    python experiments/b4_finetune.py \
        --b1-run "$SRC_DS/runs/$src" \
        --real-dataset "$B5_DS_PATH" \
        --model unet --noise-type "$noise" \
        --seed "$seed" --epochs 10 --lr 1e-4 --partial 0.25 \
        --device cpu --nperseg 128 2>&1 | tee -a "$LOG" | tail -3
    echo "### [$n/$total] done $(date +%H:%M:%S) ###" | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "========== Phase B5 complete $(date) ==========" | tee -a "$LOG"
