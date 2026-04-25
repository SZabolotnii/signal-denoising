#!/bin/bash
# Phase B5 post-eval — measure fine-tuned models on ACTUAL RadioML test set
# (not the B5 synthetic+real-noise mix). This confirms whether sim-to-real
# gap closed for real-world frames.
set -e
cd /Users/serhiizabolotnii/Projects/signal-denoising
source .venv/bin/activate 2>/dev/null || true

REAL_FPV=data_generation/datasets/radioml2018_bpsk_qpsk_fpv
REAL_DS=data_generation/datasets/radioml2018_bpsk_qpsk_deep_space
B5_FPV=data_generation/datasets/fpv_realnoise_bpsk_qpsk
B5_DS=data_generation/datasets/deep_space_realnoise_bpsk_qpsk

TS=$(date +%Y%m%d_%H%M%S)
LOG=experiments/results/b5_eval_real_${TS}.log
mkdir -p experiments/results
echo "========== Phase B5 — eval on actual RadioML test ==========" | tee "$LOG"
echo "Start: $(date)" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "--- FPV B5 fine-tuned × actual RadioML FPV test ---" | tee -a "$LOG"
for rid in $(ls -1d $B5_FPV/runs/*/  2>/dev/null | xargs -n1 basename | sort); do
    [ -d "$B5_FPV/runs/$rid" ] || continue
    [ -f "$B5_FPV/runs/$rid/b4_finetune_meta.json" ] || continue
    echo "### $rid ###" | tee -a "$LOG"
    python experiments/b3_real_sdr_zeroshot.py \
        --b1-run "$B5_FPV/runs/$rid" \
        --real-dataset "$REAL_FPV" \
        --test-noise non_gaussian 2>&1 | tee -a "$LOG" | tail -3
done

echo "" | tee -a "$LOG"
echo "--- deep_space B5 fine-tuned × actual RadioML deep_space test ---" | tee -a "$LOG"
for rid in $(ls -1d $B5_DS/runs/*/ 2>/dev/null | xargs -n1 basename | sort); do
    [ -d "$B5_DS/runs/$rid" ] || continue
    [ -f "$B5_DS/runs/$rid/b4_finetune_meta.json" ] || continue
    echo "### $rid ###" | tee -a "$LOG"
    python experiments/b3_real_sdr_zeroshot.py \
        --b1-run "$B5_DS/runs/$rid" \
        --real-dataset "$REAL_DS" \
        --test-noise non_gaussian 2>&1 | tee -a "$LOG" | tail -3
done

echo "" | tee -a "$LOG"
echo "========== Phase B5 eval complete $(date) ==========" | tee -a "$LOG"
