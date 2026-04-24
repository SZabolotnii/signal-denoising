# B3 zero-shot real-SDR eval — 20260424_204855

- Source models: `data_generation/datasets/deep_space_polygauss_qpsk_bs1024_n400000_0310b7e7/runs/run_20260423_2b4b99cc`
- Real dataset:  `data_generation/datasets/radioml2018_bpsk_qpsk_deep_space` (noise=non_gaussian)

## Per-model per-SNR SNR (dB)

| Model | m20dB | m16dB | m12dB | m8dB | m4dB | 0dB | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| HybridDSGE_UNet_robust_S3_vA_gaussian | -0.23 | -0.23 | -0.24 | -0.23 | -0.22 | -0.16 | -0.22 |
| ResNetAutoencoder_gaussian | -0.44 | -0.45 | -0.47 | -0.49 | -0.73 | -1.23 | -0.64 |
| UnetAutoencoder_gaussian | -0.70 | -0.71 | -0.72 | -0.74 | -0.90 | -1.21 | -0.83 |
| Wavelet_gaussian | -0.26 | -0.26 | -0.30 | -0.31 | -0.57 | -1.23 | -0.49 |
