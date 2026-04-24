# B2 deep_space main — cross-seed aggregate (2 seeds: 42, 43)

Source: `experiments/b2_main.sh`, killed after [4/6] at 13:45 EEST 2026-04-24. 4 run_dirs in `data_generation/datasets/deep_space_polygauss_qpsk_bs1024_n400000_0310b7e7/runs/`.

Metrics authority: `training_report_*.json` per trainer's `denoise_numpy`. Two aggregates reported:
- **overall (testset)** — `test_metrics.SNR` (pooled SNR, low-SNR dominated).
- **overall (per-SNR mean)** — unweighted mean of 10 per-SNR-bin SNR_out values (equal weighting).

## 1. Overall SNR μ ± σ [dB] (n=2 seeds)

| Model | Metric | Gaussian | Non-Gaussian | Δ(NG−G) |
|---|---|---|---|---:|
| UnetAutoencoder | testset SNR | 3.92 ± 0.03 (n=2) | 4.95 ± 0.08 (n=2) | +1.04 |
| UnetAutoencoder | per-SNR mean | 6.47 ± 0.17 (n=2) | 7.57 ± 0.02 (n=2) | +1.10 |
| ResNetAutoencoder | testset SNR | 3.82 ± 0.02 (n=2) | 4.77 ± 0.08 (n=2) | +0.95 |
| ResNetAutoencoder | per-SNR mean | 6.28 ± 0.10 (n=2) | 7.14 ± 0.00 (n=2) | +0.86 |
| HybridDSGE_UNet | testset SNR | 0.86 ± 1.21 (n=2) | 0.00 ± 0.00 (n=2) | -0.86 |
| HybridDSGE_UNet | per-SNR mean | 0.20 ± 0.29 (n=2) | 0.00 ± 0.00 (n=2) | -0.20 |
| Wavelet | test_MSE | 1.0944 ± 0.0043 (n=2) | 3.7410 ± 0.0241 (n=2) | — |

## 2. Per-seed test SNR [dB] (raw)

| Seed | Noise | UNet (testset) | UNet (per-SNR mean) | ResNet (testset) | ResNet (per-SNR mean) | Hybrid (testset) | Hybrid (per-SNR mean) | Wavelet (test_MSE) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | G | 3.94 | 6.60 | 3.84 | 6.35 | 1.72 | 0.40 | 1.0975 |
| 42 | NG | 5.01 | 7.59 | 4.83 | 7.13 | 0.00 | 0.00 | 3.7580 |
| 43 | G | 3.90 | 6.35 | 3.81 | 6.21 | 0.00 | -0.01 | 1.0914 |
| 43 | NG | 4.90 | 7.56 | 4.72 | 7.14 | 0.00 | 0.00 | 3.7239 |

## 3. Per-SNR breakdown (μ ± σ across 2 seeds) [SNR_out dB]

### gaussian

| Model | m20dB | m17dB | m15dB | m12dB | m10dB | m7dB | m5dB | m3dB | p0dB | p3dB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UnetAutoencoder | -0.44±0.00 | +0.96±0.10 | +2.15±0.32 | +4.10±0.02 | +5.98±0.11 | +7.87±0.00 | +8.51±0.15 | +10.33±0.21 | +11.49±0.37 | +13.79±0.50 |
| ResNetAutoencoder | -0.21±0.00 | +0.91±0.30 | +1.76±0.49 | +3.85±0.03 | +5.07±0.47 | +7.45±0.10 | +8.33±0.06 | +10.31±0.10 | +11.47±0.10 | +13.86±0.18 |
| HybridDSGE_UNet | -0.12±0.17 | -4.36±6.11 | -3.44±4.83 | +0.39±0.55 | -1.22±1.72 | +0.09±0.13 | +2.11±2.98 | +1.81±2.55 | +3.20±4.52 | +3.52±4.99 |

### non_gaussian

| Model | m20dB | m17dB | m15dB | m12dB | m10dB | m7dB | m5dB | m3dB | p0dB | p3dB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UnetAutoencoder | +0.03±0.10 | +1.73±0.09 | +3.17±0.08 | +5.23±0.06 | +6.59±0.04 | +8.68±0.00 | +10.05±0.03 | +11.44±0.04 | +13.45±0.06 | +15.37±0.08 |
| ResNetAutoencoder | +0.42±0.17 | +1.64±0.10 | +2.84±0.06 | +4.86±0.02 | +6.16±0.00 | +8.10±0.02 | +9.37±0.04 | +10.69±0.08 | +12.67±0.12 | +14.62±0.12 |
| HybridDSGE_UNet | -0.00±0.00 | -0.00±0.00 | +0.00±0.00 | +0.00±0.00 | +0.00±0.00 | +0.00±0.00 | +0.00±0.00 | +0.00±0.00 | +0.00±0.00 | +0.00±0.00 |

## 4. Validation SNR (μ ± σ) [dB]

| Model | Gaussian | Non-Gaussian |
|---|---:|---:|
| UnetAutoencoder | 3.89 ± 0.08 (n=2) | 4.93 ± 0.00 (n=2) |
| ResNetAutoencoder | 3.80 ± 0.07 (n=2) | 4.74 ± 0.00 (n=2) |
| HybridDSGE_UNet | 0.84 ± 1.19 (n=2) | 0.00 ± 0.00 (n=2) |

