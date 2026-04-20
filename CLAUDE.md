# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Neural network-based radio signal denoising research. Tests whether models trained on non-Gaussian (polygaussian) noise outperform models trained on standard Gaussian (AWGN) noise for real-world radio signals.

- **Signal:** Real-valued 1D, 1024 samples at 8192 Hz (~125ms blocks)
- **Scenarios:** FPV telemetry (SNR -5..+15 dB) and deep space (SNR -20..0 dB)
- **Core hypothesis:** NN trained on realistic non-Gaussian noise generalizes better than AWGN-trained NN

## Common Commands

```bash
# Generate dataset (default: FPV telemetry, 400k samples)
python data_generation/generation.py
python data_generation/generation.py --scenario deep_space --n-samples 200000

# Train all models on both noise types
python train/train_all.py --dataset data_generation/datasets/<name> --epochs 50

# Quick debug run (5% data, 2 epochs)
python train/train_all.py --dataset data_generation/datasets/<name> --partial-train 0.05 --epochs 2

# Train specific models/noise types
python train/train_all.py --dataset <path> --models unet,resnet --noise-types gaussian

# With W&B logging
python train/train_all.py --dataset <path> --wandb-project signal-denoising

# Train individual model
python train/training_uae.py --dataset <path> --noise-type non_gaussian --epochs 30

# DSGE basis sweep (13 combos via train/sweep_hybrid.py)
python train/sweep_hybrid.py --dataset <path> --noise-types non_gaussian --epochs 30

# Full DSGE ablation sweep (24 configs: Variant A/B × 3 bases × 2 orders × 2 noise types)
python experiments/run_dsge_sweep.py --dataset <path> --epochs 30

# Generate cross-evaluation report
python train/compare_report.py --run <dataset>/weights/runs/run_YYYYMMDD_<uuid>

# Batch inference on a saved run
python inference/inference_all_models.py --run <dataset>/weights/runs/run_YYYYMMDD_<uuid> --input <signals.npy>

# Evaluate dataset quality (cumulant-based analysis)
python data_generation/evaluate_dataset.py data_generation/datasets/<name>/
```

## Architecture

### Models (in `models/`)

| Model | Domain | Approach |
|-------|--------|----------|
| U-Net (`autoencoder_unet.py`) | Spectral | STFT → predict sigmoid mask → iSTFT |
| ResNet (`autoencoder_resnet.py`) | Spectral | Same STFT mask approach |
| VAE (`autoencoder_vae.py`) | Spectral | Variational autoencoder with STFT |
| Transformer (`time_series_trasformer.py`) | Time | Raw signal → self-attention → denoised signal |
| Hybrid DSGE-UNet (`hybrid_unet.py`) | Spectral | Multi-channel input: original STFT + DSGE basis STFTs; configurable `base_channels` |
| Wavelet (`wavelet.py`) | Time-Freq | Classical grid search (CPU-only baseline) |

**Spectral models** all follow: signal → STFT(nperseg=128, 75% overlap) → model predicts mask ∈ [0,1] → mask × noisy_spectrum → iSTFT. Uses `torch.stft`/`torch.istft` on GPU.

**DSGE layer** (`dsge_layer.py`): Nonlinear basis expansion (fractional, polynomial, trigonometric, robust) fitted via Tikhonov-regularized least squares (`tikhonov_lambda=0.01`). Provides two channel strategies for the Hybrid U-Net:
- **Variant A:** [reconstruction, residual] — 2 extra channels
- **Variant B:** [reconstruction, weighted basis components] — outperforms A in experiments

`dsge_layer.fit(clean, noisy)` is called once on training data; state serialized to `.npz` alongside `model_best.pth`.

### Training Pipeline (`train/`)

Each model has a `Trainer` class with a uniform interface:
- `__init__()`: loads data, creates model/optimizer/scheduler
- `train() → dict`: training loop, returns results with metrics
- `denoise_numpy(noisy: np.ndarray) → np.ndarray`: batch inference `[N, 1024] → [N, 1024]`

**train_all.py** orchestrates: for each noise_type × model → train → save best weights (by val_SNR) → evaluate per-SNR.

**Key training details:**
- Loss: MSE for Gaussian, SmoothL1(β=0.02) for non-Gaussian (see `train/losses.py`)
- Optimizer: Adam with ReduceLROnPlateau (patience=3, factor=0.5)
- Early stopping: patience=5
- Data split: 50% train / 25% val / 25% test
- Model order: Transformer first (largest VRAM consumer, avoids fragmentation)

### Per-model defaults (tuned for ~8GB GPU)

| Model | Batch Size | Learning Rate |
|-------|-----------|---------------|
| Transformer | 128 | 1e-3 (stalls at 1e-4) |
| U-Net | 1024 | 1e-3 |
| VAE | 8192 | 6e-4 |
| ResNet | 2048 | 6e-4 |
| Hybrid | 4096 | 3e-4 |
| Wavelet | 512 | N/A (grid search) |

### Evaluation (`train/compare_report.py`, `train/snr_curve.py`)

Cross-evaluates all trained models on both Gaussian and non-Gaussian test sets. Outputs:
- Markdown reports (English + Ukrainian)
- CSV/JSON data tables
- Figures: SNR heatmap, per-model curves, DSGE scatter, example denoising

Metrics in `metrics.py`: MSE, MAE, RMSE, SNR (dB). SNR = 10·log₁₀(Σclean² / Σnoise²).

### Inference (`inference/`)

Per-model inference wrappers (`inference_uae.py`, `inference_resnet.py`, etc.) and a unified `inference_all_models.py`. `inference_hybrid.py` handles DSGE state loading from `.npz` before inference.

### Data Generation (`data_generation/generation.py`)

Generates synthetic QPSK/BPSK signals with configurable noise. Output per dataset:
- `train/` and `test/`: `clean_signals.npy`, `gaussian_signals.npy`, `non_gaussian_signals.npy`
- `dataset_config.json`: block_size, sample_rate, scenario params
- Test sets have per-SNR files at ~10 discrete SNR levels

### Output Structure

```
<dataset>/weights/runs/run_YYYYMMDD_<uuid>/
├── <Model>_<noise_type>/
│   ├── model_best.pth (or best_params.json for wavelet)
│   └── figures/
├── comparison_report_<ts>.md
├── comparison_data_<ts>.csv
└── figures/
```

## Key Dependencies

PyTorch 2.6, NumPy, SciPy (STFT), PyWavelets, matplotlib, wandb (optional). Use `requirements-cuda.txt` for NVIDIA GPU setups (pins CUDA 12.4 packages).

## Environment Setup

```bash
pip install -r requirements.txt          # CPU / MPS
pip install -r requirements-cuda.txt     # NVIDIA GPU

# Create .env manually (.env.template was removed):
echo "WANDB_API_KEY=your_key_here" > .env
```

## Experiments

`experiments/` contains reproducibility bash scripts (`dsge_*.sh`, `fpv_experiment.sh`) and `run_dsge_sweep.py` for systematic DSGE ablations. Research findings are tracked in `experiments/RESEARCH_STATUS_*.md`.

There is no formal test suite (`tests/` is empty); correctness is validated through the experiment pipeline and compare_report outputs.

## Device Support

`train/device_utils.py` auto-detects CUDA → MPS (Apple Silicon) → CPU. MPS fallback is active on M-series Macs with PyTorch 2.6.
