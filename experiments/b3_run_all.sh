#!/bin/bash
# Phase B3 — run zero-shot real-SDR eval on all B1 FPV and B2 deep_space run_dirs
# vs the RadioML 2018.01A BPSK/QPSK subsets.
set -e
cd /Users/serhiizabolotnii/Projects/signal-denoising
source .venv/bin/activate 2>/dev/null || true

FPV_DS=data_generation/datasets/fpv_telemetry_polygauss_qpsk_bs1024_n100000_953c56e8
DS_DS=data_generation/datasets/deep_space_polygauss_qpsk_bs1024_n400000_0310b7e7
REAL_FPV=data_generation/datasets/radioml2018_bpsk_qpsk_fpv
REAL_DS=data_generation/datasets/radioml2018_bpsk_qpsk_deep_space

TS=$(date +%Y%m%d_%H%M%S)
LOG=experiments/results/b3_runall_${TS}.log
mkdir -p experiments/results
echo "========== Phase B3 — zero-shot real-SDR eval ==========" | tee "$LOG"
echo "Start: $(date)" | tee -a "$LOG"

# B1 FPV × RadioML FPV subset
FPV_RUNS=(
    run_20260421_267176a3   # seed 42 G
    run_20260421_92a2e0c4   # seed 42 NG
    run_20260421_fe4d376a   # seed 43 G
    run_20260422_d98d21b3   # seed 43 NG
    run_20260422_3d92fb74   # seed 44 G
    run_20260422_1023e287   # seed 44 NG
)
echo "" | tee -a "$LOG"
echo "--- FPV B1 runs × RadioML FPV subset ---" | tee -a "$LOG"
for rid in "${FPV_RUNS[@]}"; do
    echo "### $rid ###" | tee -a "$LOG"
    python experiments/b3_real_sdr_zeroshot.py \
        --b1-run "$FPV_DS/runs/$rid" \
        --real-dataset "$REAL_FPV" \
        --test-noise non_gaussian 2>&1 | tee -a "$LOG" | tail -3
done

# B2 deep_space × RadioML deep_space subset
B2_RUNS=(
    run_20260423_ed46f843   # seed 42 G
    run_20260423_3d8957d2   # seed 42 NG
    run_20260423_2b4b99cc   # seed 43 G
    run_20260424_20fc5edf   # seed 43 NG
)
echo "" | tee -a "$LOG"
echo "--- B2 deep_space runs × RadioML deep_space subset ---" | tee -a "$LOG"
for rid in "${B2_RUNS[@]}"; do
    echo "### $rid ###" | tee -a "$LOG"
    python experiments/b3_real_sdr_zeroshot.py \
        --b1-run "$DS_DS/runs/$rid" \
        --real-dataset "$REAL_DS" \
        --test-noise non_gaussian 2>&1 | tee -a "$LOG" | tail -3
done

echo "" | tee -a "$LOG"
echo "========== Phase B3 complete $(date) ==========" | tee -a "$LOG"
echo "Log: $LOG"
