#!/bin/bash
set -e
cd /Users/serhiizabolotnii/Projects/signal-denoising
source .venv/bin/activate

DS="data_generation/datasets/fpv_telemetry_polygauss_qpsk_bs1024_n100000_953c56e8"
COMMON="--epochs 50 --partial-train 0.25 --device cpu --nperseg 128 --seed 42"

echo "========== FPV Experiment =========="
echo "Dataset: $DS"
echo "Models: UNet, ResNet, DSGE vA frac S=3"
echo "Data: 25% (25k samples)"
echo "===================================="

i=0
total=8

# 1. UNet (both noise types)
for NOISE in gaussian non_gaussian; do
  i=$((i+1))
  echo ""
  echo "########## [$i/$total] UNet $NOISE ##########"
  python train/train_all.py --dataset "$DS" --models unet --noise-types "$NOISE" $COMMON 2>&1 | grep -E "train=|Best model|ERROR|Done"
  echo "  ---- done [$i/$total] ----"
done

# 2. ResNet (both noise types)
for NOISE in gaussian non_gaussian; do
  i=$((i+1))
  echo ""
  echo "########## [$i/$total] ResNet $NOISE ##########"
  python train/train_all.py --dataset "$DS" --models resnet --noise-types "$NOISE" $COMMON 2>&1 | grep -E "train=|Best model|ERROR|Done"
  echo "  ---- done [$i/$total] ----"
done

# 3. DSGE vA fractional S=3 (both noise types, 3 seeds for reproducibility)
for NOISE in gaussian non_gaussian; do
  i=$((i+1))
  echo ""
  echo "########## [$i/$total] DSGE vA frac S=3 $NOISE ##########"
  python -c "
from train.training_hybrid import HybridUnetTrainer
from pathlib import Path
import json, torch
torch.manual_seed(42)

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
    unet_width=16,
    batch_size=1024,
    epochs=50,
    learning_rate=3e-4,
    signal_len=cfg['block_size'],
    fs=cfg['sample_rate'],
    nperseg=128,
    noverlap=96,
    random_state=42,
    data_fraction=0.25,
    device='cpu',
).train()
print(f'RESULT DSGE $NOISE val_SNR={result[\"val_snr\"]:.4f} dB')
" 2>&1 | grep -E "RESULT|ERROR|Model params|val_SNR="
  echo "  ---- done [$i/$total] ----"
done

# 4. Wavelet baseline (both noise types)
for NOISE in gaussian non_gaussian; do
  i=$((i+1))
  echo ""
  echo "########## [$i/$total] Wavelet $NOISE ##########"
  python train/train_all.py --dataset "$DS" --models wavelet --noise-types "$NOISE" $COMMON 2>&1 | grep -E "Best params|ERROR|Done"
  echo "  ---- done [$i/$total] ----"
done

echo ""
echo "========== FPV EXPERIMENT COMPLETE =========="
