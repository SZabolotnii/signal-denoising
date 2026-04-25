# Real-data experiments — Phase B3+ status

**Created:** 2026-04-25
**Author:** Сергій Заболотній
**Scope:** експерименти з реальними SDR-даними (RadioML 2018.01A, та потенційно
DroneDetect / OTA recordings). Окремо від `RESEARCH_STATUS_20260419.md` бо
methodology, datasets, та paper-positioning суттєво інші.

---

## §1. Завершені експерименти

### §1.1 B3 — zero-shot eval (2026-04-24, ✅ done)

Pretrained synthetic models (10 run_dirs: 6 FPV B1 + 4 deep_space B2) на
adapted RadioML 2018 subsets (BPSK+QPSK).

**Результат:** uniform NEGATIVE SNR (−0.02..−1.59 dB), domain gap **8-15 dB**.
Hybrid NG identity collapse → optimal zero-shot ("do no harm"). UNet FPV NG
показав partial transfer (+0.22 dB Δ vs G).

Артефакти: `experiments/results/b3_aggregate.{md,json}`,
`experiments/results/b3_zeroshot_run_*.{md,json}` (10 шт).
Деталі — `RESEARCH_STATUS_20260419.md` §15.

### §1.2 B4 smoke — fine-tune (2026-04-24, ❌ methodologically flawed)

Спроба fine-tune'ити UNet на RadioML train split. Smoke test (3 ep, 5% data,
seed=42, NG, FPV) показав **degradation** з −0.18 dB (zero-shot) → **−1.31 dB**
після fine-tune.

**Причина:** RadioML "clean reference" — high-SNR proxy з **різним symbol
stream** ніж noisy frame. Fine-tune вчить модель hallucinate unrelated signal,
руйнуючи навчені priors.

**Висновок:** Vanilla supervised fine-tune на RadioML pairs **infeasible**
для denoising задачі. RadioML 2018 у його raw form придатний тільки для:
1. Zero-shot eval (B3 — OK).
2. Noise-only extraction для injection (B5 — see §2).
3. Modulation classification (out of scope).

Артефакти: `experiments/b4_finetune.py`, `experiments/b4_run_all.sh` (не
запущено), failed run у
`data_generation/datasets/radioml2018_bpsk_qpsk_fpv/runs/run_20260424_232538f3_*`.

---

## §2. Plan: B5 — real-noise injection training (2026-04-25, ⏳ planned)

### §2.1 Motivation

B3 показав sim-to-real gap. B4 показав, що paired fine-tune infeasible.
**Hybrid strategy:** train на synthetic clean signals + real noise samples
(extracted residual з RadioML high-SNR proxy mechanism), щоб закрити gap
без mismatched-symbol artefact'у.

### §2.2 Method

Use `non_gaussian_noise_only.npy` файли з adapted RadioML subsets:
- FPV: `data_generation/datasets/radioml2018_bpsk_qpsk_fpv/train/non_gaussian_noise_only.npy`
- deep_space: `data_generation/datasets/radioml2018_bpsk_qpsk_deep_space/train/non_gaussian_noise_only.npy`

These are residuals `(noisy − clean_proxy)` — contain real RF channel noise
**plus** small symbol variation (caveat). For low-SNR frames, residual is
dominated by noise → suitable proxy для real noise sample bank.

**Training pipeline:**

1. Generate (or reuse) synthetic clean signals.
2. Each batch: sample N real-noise vectors with replacement, scale to target
   SNR, add to synthetic clean → form `(clean, clean+real_noise)` pair.
3. Train UNet/ResNet/Hybrid від random init (або resume з B1/B2 checkpoint
   for faster convergence) на цьому mixed-noise dataset.
4. Eval на:
   - Synthetic test (sanity — performance не повинен fall below B1/B2).
   - Real RadioML test (B3-style zero-shot eval) — main metric.

**Expected outcome:** if real-noise injection helps, real test SNR → positive
(+1..+5 dB) vs B3 negative. If it doesn't help at все, sim-to-real gap
fundamentally about signal model differences (symbol rate, bandwidth,
oversampling), не лише noise.

### §2.3 Implementation steps

1. **Add `--real-noise-bank <path>` to `data_generation/generation.py`** OR
   write standalone `data_generation/generate_with_real_noise.py` що приймає
   synthetic clean + noise bank → emits training-ready dataset.
2. **Train UNet × 2 scenarios × 2 noise types × 2 seeds** = 8 runs.
   Resume з B1/B2 checkpoints (warm start, fewer epochs).
3. **B3-style zero-shot eval** на real RadioML test.
4. **Aggregate** з `analysis/aggregate_b5.py`.
5. **Update §3 below** з результатами.

### §2.4 Compute estimate

| Step | Time |
|---|---|
| Dataset generation (mix real noise) | ~30 min |
| Train 8 runs (warm start, 10-15 ep, 25% data) | ~10-12 h CPU |
| Eval (B3-style) on real test | ~5 min |
| Aggregation + write-up | ~30 min |
| **Total** | **~12-14 h** |

### §2.5 Production deployment recipe (paper-relevant)

Якщо B5 success → paper рекомендація для real-world:

1. На target SDR/platform — collect 5-10 хв ambient RF без active signal
   (quiet bands, between bursts).
2. Extract noise frames `[1024]`.
3. Train denoiser on synthetic_clean + scaled real_noise pairs.
4. Continuous adaptation: щотижневі re-collection + 1-2 ep fine-tune для
   tracking RF environment drift.

Це **practical deployment workflow**, не лише academic finding.

---

## §3. Результати B5 (заповнюється in-progress)

*Pending execution.*

---

## §4. Майбутні real-data tracks

- **B6:** OTA-recorded SigMF samples (gnu-radio community demos) для truly
  controlled real signal benchmark.
- **B7:** DroneDetect SigMF subset якщо disk + access дозволять — proper paired
  data via same-segment time splits.
- **B8:** Self-supervised Noise2Noise на pure RadioML noisy-only frames
  (no clean reference required).
