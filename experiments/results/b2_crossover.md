# B2 deep_space — crossover matrix (n=2 seeds)

Cross-evaluation: each model trained on `noise_trained` is evaluated on test sets for both noise types. Source: 4 `compare_report.py` CSV outputs (see §11.8 for authoritative inference fixes).

## 1. Crossover μ±σ [dB] (n=2 seeds; 'all' = aggregate SNR_out)

| Model | Train | G→G | G→NG | NG→G | NG→NG |
|---|---|---:|---:|---:|---:|
| UnetAutoencoder | μ ± σ | +6.47 ± 0.17 | +6.67 ± 0.22 | +7.25 ± 0.02 | +7.57 ± 0.02 |
| ResNetAutoencoder | μ ± σ | +6.28 ± 0.10 | +6.42 ± 0.16 | +6.82 ± 0.00 | +7.14 ± 0.00 |
| HybridDSGE_UNet_robust_S3_vA | μ ± σ | +0.20 ± 0.29 | -0.78 ± 1.08 | +0.00 ± 0.00 | +0.00 ± 0.00 |
| Wavelet | μ ± σ | -4.45 ± 0.00 | -5.95 ± 0.00 | -4.45 ± 0.00 | -5.95 ± 0.00 |

## 2. Per-seed crossover [dB]

| Seed | Model | G→G | G→NG | NG→G | NG→NG |
|---|---|---:|---:|---:|---:|
| 42 | UnetAutoencoder | +6.60 | +6.82 | +7.26 | +7.59 |
| 42 | ResNetAutoencoder | +6.35 | +6.53 | +6.82 | +7.13 |
| 42 | HybridDSGE_UNet_robust_S3_vA | +0.40 | -1.54 | +0.00 | +0.00 |
| 42 | Wavelet | -4.45 | -5.95 | -4.45 | -5.95 |
| 43 | UnetAutoencoder | +6.35 | +6.52 | +7.23 | +7.56 |
| 43 | ResNetAutoencoder | +6.21 | +6.31 | +6.83 | +7.14 |
| 43 | HybridDSGE_UNet_robust_S3_vA | -0.01 | -0.02 | +0.00 | +0.00 |
| 43 | Wavelet | -4.45 | -5.95 | -4.45 | -5.95 |

## 3. Key deltas for central hypothesis

**Hypothesis:** NG-training generalizes better on NG test (NG→NG > G→NG), and maintains or loses minimally on G test (NG→G ≥ G→G − ε).

| Model | Δ_NG_test = NG→NG − G→NG | Δ_G_test = NG→G − G→G |
|---|---:|---:|
| UnetAutoencoder | +0.90 | +0.77 |
| ResNetAutoencoder | +0.71 | +0.54 |
| HybridDSGE_UNet_robust_S3_vA | +0.78 | -0.20 |
| Wavelet | +0.00 | +0.00 |

