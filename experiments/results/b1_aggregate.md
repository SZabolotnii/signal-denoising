# B1 FPV main — cross-seed aggregate (3 seeds: 42, 43, 44)
**Source:** `training_report.json` from each run (uses each Trainer's own `denoise_numpy`).
**Note:** `compare_report.py` is broken for ResNet (mask not applied) and HybridDSGE (regex skips `_vA` suffix); aggregator bypasses it.

## 1. Overall test SNR (μ ± σ) [dB]

Computed on the held-out 25%-test split via each trainer's `denoise_numpy`.

| Model | n_params | Gaussian train | Non-Gaussian train |
|---|---:|---:|---:|
| UnetAutoencoder | — | 14.22 ± 0.08 (n=3) | 15.00 ± 0.07 (n=3) |
| ResNetAutoencoder | — | 13.14 ± 0.07 (n=3) | 14.40 ± 0.03 (n=3) |
| HybridDSGE_UNet | — | 11.13 ± 0.39 (n=3) | 3.92 ± 6.79 (n=3) |
| Wavelet | — | MSE 0.0896 ± 0.0005 (n=3) | MSE 0.1544 ± 0.0010 (n=3) |

## 2. Per-seed test SNR [dB] (raw)

| Seed | Noise train | UNet | ResNet | HybridDSGE | Wavelet (test_MSE) |
|---|---|---:|---:|---:|---:|
| 42 | G | 14.23 | 13.08 | 10.72 | 0.0901 |
| 42 | NG | 14.93 | 14.40 | 0.00 | 0.1546 |
| 43 | G | 14.13 | 13.21 | 11.51 | 0.0890 |
| 43 | NG | 14.99 | 14.43 | 11.77 | 0.1532 |
| 44 | G | 14.29 | 13.13 | 11.15 | 0.0895 |
| 44 | NG | 15.07 | 14.38 | 0.00 | 0.1553 |

## 3. Per-SNR breakdown (μ ± σ across 3 seeds) [SNR_out dB]

### gaussian

| Model | m5dB | m2dB | p0dB | p3dB | p5dB | p8dB | p10dB | p12dB | p15dB | p18dB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UnetAutoencoder | 8.95±0.07 | 7.69±2.02 | 12.57±0.06 | 14.37±0.51 | 16.01±0.22 | 17.99±0.07 | 19.13±0.10 | 20.19±0.20 | 21.59±0.22 | 23.56±0.19 |
| ResNetAutoencoder | 8.03±0.12 | 6.87±2.09 | 11.46±0.13 | 13.14±0.68 | 14.82±0.31 | 16.62±0.15 | 17.82±0.14 | 19.09±0.14 | 20.92±0.10 | 22.72±0.10 |
| HybridDSGE_UNet | 5.56±0.30 | 4.55±1.41 | 9.80±0.47 | 10.31±0.80 | 12.22±0.47 | 14.53±0.52 | 15.30±0.51 | 16.03±0.52 | 16.80±0.58 | 17.38±0.69 |

### non_gaussian

| Model | m5dB | m2dB | p0dB | p3dB | p5dB | p8dB | p10dB | p12dB | p15dB | p18dB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UnetAutoencoder | 9.82±0.11 | 11.95±0.08 | 13.31±0.07 | 15.32±0.06 | 16.62±0.06 | 18.58±0.05 | 19.87±0.04 | 21.25±0.04 | 23.31±0.05 | 25.54±0.06 |
| ResNetAutoencoder | 9.26±0.09 | 11.37±0.06 | 12.71±0.06 | 14.72±0.06 | 16.03±0.06 | 17.98±0.05 | 19.25±0.04 | 20.57±0.04 | 22.50±0.04 | 24.48±0.04 |
| HybridDSGE_UNet | 1.83±3.16 | 3.02±5.24 | 3.71±6.43 | 4.38±7.59 | 4.66±8.07 | 4.98±8.63 | 5.11±8.86 | 5.24±9.08 | 5.35±9.26 | 5.43±9.40 |

## 4. Validation SNR (μ ± σ) [dB] — training-set sanity

| Model | Gaussian | Non-Gaussian |
|---|---:|---:|
| UnetAutoencoder | 14.30 ± 0.07 (n=3) | 15.07 ± 0.05 (n=3) |
| ResNetAutoencoder | 13.21 ± 0.13 (n=3) | 14.48 ± 0.05 (n=3) |
| HybridDSGE_UNet | 11.19 ± 0.43 (n=3) | 3.91 ± 6.77 (n=3) |

