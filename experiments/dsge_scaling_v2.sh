#!/bin/bash
set -e
cd /Users/serhiizabolotnii/Projects/signal-denoising
source .venv/bin/activate

DS="data_generation/datasets/deep_space_polygauss_qpsk_bs1024_n400000_0310b7e7"
# Use the WINNING powers from DSGE sweep: [0.5, 1.5, 2.0]
# Pass via python directly since CLI doesn't support custom powers easily

i=0
total=6

for WIDTH in 16 32 64; do
  for NOISE in gaussian non_gaussian; do
    i=$((i+1))
    echo ""
    echo "########## [$i/$total] width=$WIDTH noise=$NOISE powers=[0.5,1.5,2.0] ##########"
    python -c "
from train.training_hybrid import HybridUnetTrainer
from pathlib import Path
import json

ds = Path('$DS')
with open(ds / 'dataset_config.json') as f:
    cfg = json.load(f)

result = HybridUnetTrainer(
    dataset_path=ds,
    noise_type='$NOISE',
    dsge_order=3,
    dsge_basis='fractional',
    dsge_variant='A',
    dsge_powers=[0.5, 1.5, 2.0],
    unet_width=$WIDTH,
    batch_size=4096,
    epochs=50,
    learning_rate=3e-4,
    signal_len=cfg['block_size'],
    fs=cfg['sample_rate'],
    nperseg=128,
    noverlap=96,
    random_state=42,
    data_fraction=0.05,
    device='cpu',
).train()
print(f'  -> val_SNR = {result[\"val_snr\"]:.2f} dB, params = {result.get(\"model\",\"?\")}')
" 2>&1 || echo "  FAILED: width=$WIDTH noise=$NOISE"
    echo "  ---- done [$i/$total] ----"
  done
done

echo ""
echo "========== DSGE SCALING v2 COMPLETE =========="
