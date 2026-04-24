# B3 zero-shot on RadioML 2018.01A — cross-seed aggregate

Models trained on synthetic polygauss datasets (FPV B1 3 seeds, deep_space B2 2 seeds) evaluated zero-shot on real RadioML 2018 BPSK+QPSK frames at SNR levels matched to the training scenario. Metric: mean SNR_out across per-SNR bins.

## 1. Per-scenario × train_noise aggregate (mean SNR ± σ, dB)

### FPV

| Model | G-trained μ±σ | NG-trained μ±σ | Δ(NG−G) |
|---|---|---|---:|
| HybridDSGE_UNet | -0.31 ± 0.01 (n=3) | -0.04 ± 0.07 (n=3) | +0.27 |
| ResNetAutoencoder | -0.48 ± 0.23 (n=3) | -0.50 ± 0.14 (n=3) | -0.01 |
| UnetAutoencoder | -0.46 ± 0.10 (n=3) | -0.24 ± 0.10 (n=3) | +0.22 |
| Wavelet | -1.59 ± 0.00 (n=3) | -1.59 ± 0.00 (n=3) | +0.00 |

### deep_space

| Model | G-trained μ±σ | NG-trained μ±σ | Δ(NG−G) |
|---|---|---|---:|
| HybridDSGE_UNet | -0.39 ± 0.25 (n=2) | -0.02 ± 0.00 (n=2) | +0.38 |
| ResNetAutoencoder | -0.76 ± 0.17 (n=2) | -0.61 ± 0.08 (n=2) | +0.15 |
| UnetAutoencoder | -0.71 ± 0.17 (n=2) | -0.75 ± 0.05 (n=2) | -0.04 |
| Wavelet | -0.49 ± 0.00 (n=2) | -0.49 ± 0.00 (n=2) | +0.00 |

## 2. Per-seed raw mean SNR (dB)

### FPV

| Seed | Train | HybridDSGE_UNet | ResNetAutoencoder | UnetAutoencoder | Wavelet |
|---|---|---:|---:|---:|---:|
| 42 | G | -0.29 | -0.68 | -0.57 | -1.59 |
| 42 | NG | -0.00 | -0.46 | -0.18 | -1.59 |
| 43 | G | -0.31 | -0.24 | -0.45 | -1.59 |
| 43 | NG | -0.00 | -0.66 | -0.18 | -1.59 |
| 44 | G | -0.32 | -0.53 | -0.36 | -1.59 |
| 44 | NG | -0.12 | -0.38 | -0.36 | -1.59 |

### deep_space

| Seed | Train | HybridDSGE_UNet | ResNetAutoencoder | UnetAutoencoder | Wavelet |
|---|---|---:|---:|---:|---:|
| 42 | G | -0.22 | -0.64 | -0.83 | -0.49 |
| 42 | NG | -0.02 | -0.66 | -0.71 | -0.49 |
| 43 | G | -0.57 | -0.88 | -0.59 | -0.49 |
| 43 | NG | -0.02 | -0.55 | -0.78 | -0.49 |

