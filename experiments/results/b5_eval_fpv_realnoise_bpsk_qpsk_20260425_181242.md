# B5 fine-tuned eval on real radioml2018_bpsk_qpsk_fpv — 20260425_181242

- Source B5 dataset: `data_generation/datasets/fpv_realnoise_bpsk_qpsk`
- Real test dir: `data_generation/datasets/radioml2018_bpsk_qpsk_fpv`
- Test noise: non_gaussian

## Per-run SNR_out (dB) on actual RadioML test

| Run | 0dB | 4dB | 8dB | 12dB | 16dB | 18dB | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| run_20260425_1a40f044_UnetAutoencoder_non_gaussian | -0.01 | -0.02 | -0.08 | -0.09 | -0.07 | -0.08 | -0.06 |
| run_20260425_32431ed3_UnetAutoencoder_non_gaussian | -0.01 | -0.11 | -0.23 | -0.22 | -0.17 | -0.19 | -0.16 |
| run_20260425_71d66651_UnetAutoencoder_gaussian | -0.35 | -0.46 | -0.43 | -0.32 | -0.23 | -0.25 | -0.34 |
| run_20260425_942b8fad_UnetAutoencoder_gaussian | -0.41 | -0.57 | -0.51 | -0.39 | -0.28 | -0.30 | -0.41 |
