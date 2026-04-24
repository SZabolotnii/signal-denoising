# B3 zero-shot real-SDR eval — 20260424_204832

- Source models: `data_generation/datasets/fpv_telemetry_polygauss_qpsk_bs1024_n100000_953c56e8/runs/run_20260421_fe4d376a`
- Real dataset:  `data_generation/datasets/radioml2018_bpsk_qpsk_fpv` (noise=non_gaussian)

## Per-model per-SNR SNR (dB)

| Model | 0dB | 4dB | 8dB | 12dB | 16dB | 18dB | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| HybridDSGE_UNet_robust_S3_vA_gaussian | -0.41 | -0.47 | -0.36 | -0.25 | -0.18 | -0.19 | -0.31 |
| ResNetAutoencoder_gaussian | -0.07 | -0.14 | -0.34 | -0.34 | -0.26 | -0.28 | -0.24 |
| UnetAutoencoder_gaussian | -0.20 | -0.48 | -0.64 | -0.54 | -0.39 | -0.42 | -0.45 |
| Wavelet_gaussian | -2.13 | -2.32 | -1.78 | -1.35 | -0.96 | -1.01 | -1.59 |
