#!/bin/bash
set -e
cd /Users/serhiizabolotnii/Projects/signal-denoising
source .venv/bin/activate

DS="data_generation/datasets/deep_space_polygauss_qpsk_bs1024_n400000_0310b7e7"

echo "========== DSGE Reproducibility Test =========="
echo "Config: vA fractional S=3 powers=[0.5,1.5,2.0] gaussian"
echo "10 runs with seeds 42..51"
echo "================================================"

for SEED in 42 43 44 45 46 47 48 49 50 51; do
  echo ""
  echo "########## seed=$SEED ##########"
  python -c "
import torch
torch.manual_seed($SEED)

from train.training_hybrid import HybridUnetTrainer
from pathlib import Path
import json

ds = Path('$DS')
with open(ds / 'dataset_config.json') as f:
    cfg = json.load(f)

result = HybridUnetTrainer(
    dataset_path=ds,
    noise_type='gaussian',
    dsge_order=3,
    dsge_basis='fractional',
    dsge_variant='A',
    dsge_powers=[0.5, 1.5, 2.0],
    unet_width=16,
    batch_size=4096,
    epochs=50,
    learning_rate=3e-4,
    signal_len=cfg['block_size'],
    fs=cfg['sample_rate'],
    nperseg=128,
    noverlap=96,
    random_state=$SEED,
    data_fraction=0.05,
    device='cpu',
).train()
print(f'RESULT seed=$SEED val_SNR={result[\"val_snr\"]:.4f} dB')
" 2>&1 | grep -E "RESULT|ERROR|val_SNR=|Model params"
  echo "  ---- seed=$SEED done ----"
done

echo ""
echo "========== REPRODUCIBILITY TEST COMPLETE =========="
