# B5 fine-tuned eval on real radioml2018_bpsk_qpsk_deep_space — 20260425_181247

- Source B5 dataset: `data_generation/datasets/deep_space_realnoise_bpsk_qpsk`
- Real test dir: `data_generation/datasets/radioml2018_bpsk_qpsk_deep_space`
- Test noise: non_gaussian

## Per-run SNR_out (dB) on actual RadioML test

| Run | m20dB | m16dB | m12dB | m8dB | m4dB | 0dB | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| run_20260425_5345d88d_UnetAutoencoder_non_gaussian | -0.33 | -0.33 | -0.36 | -0.38 | -0.61 | -1.14 | -0.53 |
| run_20260425_aa1a55fa_UnetAutoencoder_gaussian | -0.36 | -0.37 | -0.38 | -0.39 | -0.46 | -0.67 | -0.44 |
| run_20260425_b61c86e1_UnetAutoencoder_non_gaussian | -0.37 | -0.38 | -0.39 | -0.41 | -0.59 | -1.00 | -0.52 |
| run_20260425_b8c754df_UnetAutoencoder_gaussian | -0.32 | -0.32 | -0.33 | -0.35 | -0.48 | -0.79 | -0.43 |
