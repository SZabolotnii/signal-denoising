# Real SDR Dataset Survey for DSGE Denoising Evaluation

**Created:** 2026-04-20
**Phase:** 0 of research plan (`RESEARCH_PLAN_20260420.md`)
**Goal:** Identify ≥1 public real-world SDR dataset compatible with our denoising pipeline to answer Q9 (generalization from synthetic to real signals).

---

## Pipeline compatibility requirements

The adapter must produce a directory matching the layout consumed by `train/train_all.py` and `data_generation/evaluate_dataset.py`:

```
data_generation/datasets/<name>/
├── dataset_config.json         # uid, scenario, block_size, sample_rate, snr_range,
│                               # noise_types, test_snr_points, samples_per_snr, num_train
├── train/
│   ├── clean_signals.npy          # (N_train, 1024) float32
│   ├── gaussian_signals.npy       # clean + AWGN at random SNR in range (optional)
│   ├── non_gaussian_signals.npy   # clean + real-measured noise (primary)
│   ├── non_gaussian_noise_only.npy  # non_gaussian_signals - clean_signals
│   └── snr_values.npy             # (N_train,) float32, dB per sample
└── test/
    ├── test_<snr>dB_clean.npy                 # (samples_per_snr, 1024)
    ├── test_<snr>dB_non_gaussian.npy
    └── test_<snr>dB_non_gaussian_noise_only.npy
```

**Constraints:**
- Signals: real-valued 1D, 1024 samples, 8192 Hz sample rate (≈125 ms block).
- SDR data is complex-valued IQ at high sample rates (≥1 MHz). Adapter must: (a) project to real (take `np.real`, or frequency-shift to baseband then take `real`), and (b) decimate/resample to 8192 Hz, then window to 1024-sample blocks.
- Each block needs a clean/noisy pair. Since real recordings do not provide clean references, we use one of the strategies in "RadioML caveat" below.

---

## Candidate datasets

### Tier 1 — primary candidates

| Dataset | Size | Signal types | SNR range | Format | Source |
|---------|------|--------------|-----------|--------|--------|
| **RadioML 2018.01A** (DeepSig) | ~20 GB | 24 modulations incl. BPSK/QPSK | −20…+30 dB, 26 levels | HDF5, 1024-sample complex frames | deepsig.ai/datasets |
| **MIT RF Challenge ICASSP 2024** | ~10 GB | SOI + interference (separation task) | variable | NumPy, complex baseband | github.com/RFChallenge/icassp2024rfchallenge |
| **CSPB.ML.2018R2** | ~15 GB | BPSK, QPSK + pulse shaping | variable | Custom IQ | via authors |

**Recommendation:** start with **RadioML 2018.01A** — de-facto benchmark, cited in 400+ papers, supports filtering by SNR, contains our target modulations (BPSK/QPSK).

**Status (2026-04-20):** RadioML 2018.01A download is currently unavailable from DeepSig; we defer this path until the dataset is retrievable. Adapter (`data_generation/load_radioml2018.py`) is complete and syntax-verified for later use. **DroneDetect is promoted to the primary real-SDR source for Phase B3.**

### Tier 2 — FPV / drone telemetry scenario

| Dataset | Notes | Source |
|---------|-------|--------|
| DroneDetect | BladeRF SDR + GNURadio, 2.4 GHz ISM, real UAV RF | ieee-dataport.org |
| DroneRF | 227 segments, 3 drones, real background RF | al-sad.github.io/DroneRF |
| AirID | 4× USRP B200mini on DJI M100 | genesys-lab.org/airid |

**Recommendation (FPV story):** **DroneDetect** — best match for 2.4 GHz ISM band telemetry.

### Tier 3 — noise-only (augmentation)

- **NGGAN PLC noise dataset** — real impulsive noise from commercial NB-PLC modem. Useful as a drop-in replacement for our synthetic polygauss generator.

### Tier 4 — rejected

- **RadioML 2016** — deprecated by DeepSig.
- **TorchSig** — synthetic generator, not a fixed dataset.
- **NASA DSN** — no public denoising-ready dataset.
- **MIGOU-MOD** — IoT domain mismatch.

---

## RadioML caveat: no clean reference

RadioML contains only noisy signals at known SNRs (no paired clean/noisy). Three mitigation strategies, documented in the adapter and Discussion section of the paper:

1. **(a) High-SNR as proxy clean.** Use frames at SNR ≥ +18 dB as a "clean reference" and low-SNR (≤ 0 dB) frames as noisy inputs. Same modulation + symbol rate grouping.
2. **(b) SNR improvement reformulation.** Train to map (noisy, SNR=X dB) → (less-noisy, SNR=Y>X dB). Metric: output SNR − input SNR.
3. **(c) Document as fundamental limitation.** Any zero-shot evaluation on real recordings lacks a ground-truth reference. This is inherent to real SDR data — discussed honestly in the paper.

For Phase 0 feasibility, **strategy (a)** is the initial choice (simplest, cleanest pairs). Fallback to (b) if paired frames from the same underlying symbol stream cannot be recovered.

---

## Phase 0 acceptance criteria

- [ ] `data_generation/load_radioml2018.py` produces a cached subset directory that loads without errors.
- [ ] `python data_generation/evaluate_dataset.py <path>` runs to completion on the subset.
- [ ] Noise statistics (cumulants γ₃, γ₄) of the adapted real-noise-only signals documented for comparison against synthetic polygauss.
- [ ] (Stretch) `data_generation/load_dronedetect.py` smoke-tested.

Downstream Phase B3 uses the subset to validate models trained on FPV (B1) and deep_space (B2) via zero-shot inference.
