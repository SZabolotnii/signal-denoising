# B3 zero-shot real-SDR eval — 20260424_204859

- Source models: `data_generation/datasets/deep_space_polygauss_qpsk_bs1024_n400000_0310b7e7/runs/run_20260424_20fc5edf`
- Real dataset:  `data_generation/datasets/radioml2018_bpsk_qpsk_deep_space` (noise=non_gaussian)

## Per-model per-SNR SNR (dB)

| Model | m20dB | m16dB | m12dB | m8dB | m4dB | 0dB | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| HybridDSGE_UNet_robust_S3_vA_non_gaussian | -0.02 | -0.01 | -0.02 | -0.01 | -0.01 | -0.02 | -0.02 |
| ResNetAutoencoder_non_gaussian | -0.39 | -0.40 | -0.41 | -0.43 | -0.63 | -1.05 | -0.55 |
| UnetAutoencoder_non_gaussian | -0.59 | -0.61 | -0.63 | -0.65 | -0.89 | -1.33 | -0.78 |
| Wavelet_non_gaussian | -0.26 | -0.26 | -0.30 | -0.31 | -0.57 | -1.23 | -0.49 |
