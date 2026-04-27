# B5 fine-tune vs B3 zero-shot — UNet on real RadioML 2018

Same UNet architecture, same B1/B2 pretrained checkpoints. B3: zero-shot eval. B5: pretrained → fine-tuned 10 ep on (synthetic_clean + real_noise_injection) at lr=1e-4, partial=0.25.

## 1. Mean SNR_out (dB) on actual RadioML test, μ±σ

| Scenario | Train noise | B3 zero-shot | B5 fine-tune | Δ improvement |
|---|---|---:|---:|---:|
| FPV | gaussian | -0.46 ± 0.10 (n=3) | -0.38 ± 0.05 (n=2) | **+0.08** |
| FPV | non_gaussian | -0.24 ± 0.10 (n=3) | -0.11 ± 0.07 (n=2) | **+0.13** |
| deep_space | gaussian | -0.71 ± 0.17 (n=2) | -0.44 ± 0.00 (n=2) | **+0.27** |
| deep_space | non_gaussian | -0.75 ± 0.05 (n=2) | -0.52 ± 0.00 (n=2) | **+0.22** |

## 2. Per-seed B5 raw

| Scenario | Train | Seed | Mean SNR (dB) |
|---|---|---|---:|
| FPV | gaussian | None | -0.411 |
| FPV | gaussian | None | -0.340 |
| FPV | non_gaussian | None | -0.156 |
| FPV | non_gaussian | None | -0.058 |
| deep_space | gaussian | 42 | -0.438 |
| deep_space | gaussian | 42 | -0.433 |
| deep_space | non_gaussian | 42 | -0.525 |
| deep_space | non_gaussian | 42 | -0.523 |

