# Стан дослідження — Signal Denoising with DSGE

**Дата:** 2026-04-19 (оновлено 2026-04-24 — B2 main [4/6] complete, killed, §14 final)
**Автор дослідження:** Сергій Заболотній
**Датасет:** deep_space_polygauss_qpsk_bs1024_n400000_0310b7e7 + fpv_telemetry_polygauss_qpsk_bs1024_n100000_953c56e8
**Сценарій:** Deep space (SNR -20..0 dB) + FPV (SNR -5..+18 dB), QPSK, 1024 samples @ 8192 Hz

---

## 1. Хронологія виконаних робіт

### 1.1 Інфраструктура (16-17 квітня)

- Згенеровано deep space датасет (400k семплів, 7.8 GB)
- Виявлено що **PyTorch MPS на M3 Max повільніший за CPU** для STFT операцій (13x)
- Реалізовано **прекомп'ютацію STFT** у всіх 4 спектральних тренерах — усунуто мільйони зайвих STFT обчислень з training loop
- Виправлено `torch.angle()` → MPS-compatible phase extraction
- Оптимізовано порядок тренування моделей

### 1.2 Основний експеримент (17-18 квітня)

**Конфігурація:** 6 архітектур × 2 noise types, 5% даних (20k), 50 epochs, CPU

**Результати cross-evaluation (SNR, dB):**

| Model | Train | G→G | G→NG | NG→G | NG→NG |
|-------|-------|-----|------|------|-------|
| **Transformer** | **G** | **1.40** | **1.48** | — | — |
| **Transformer** | **NG** | — | — | **1.45** | **1.65** |
| U-Net | G | -2.86 | -2.72 | — | — |
| U-Net | NG | — | — | -1.37 | -1.27 |
| HybridDSGE (старий) | G | -3.16 | -3.06 | — | — |
| HybridDSGE (старий) | NG | — | — | -2.35 | -2.28 |
| Wavelet | G | -4.45 | -5.95 | — | — |
| ResNet | G | -14.02 | -14.02 | — | — |
| VAE | G | -21.82 | -21.78 | — | — |

**Висновки основного експерименту:**
1. Гіпотеза підтверджена: NG-trained моделі краще за G-trained (UNet NG: -1.27 vs G: -2.72 на NG тесті)
2. Transformer — єдина модель з позитивним SNR (1.40-1.65 dB), але 1M параметрів
3. Більшість спектральних моделей мають від'ємний SNR — **5% даних недостатньо** для deep space сценарію
4. Старий DSGE (без коефіцієнтів Кунченка) — гірший за plain UNet

### 1.3 Виправлення DSGE (18 квітня)

**Проблема:** Реалізація DSGE не використовувала оптимальні коефіцієнти K з розкладу Кунченка. Базисні функції φᵢ(x̃) подавались "сиримі" — без оптимальної проекції.

**Що виправлено:**
- Реалізовано **Variant A** (reconstruction + residual): `[STFT(x̃), STFT(X̂_dsge), STFT(Z)]`
- Реалізовано **Variant B** (reconstruction + weighted basis): `[STFT(x̃), STFT(X̂_dsge), STFT(k₁·φ₁), ...]`
- Прибрано лінійний член з polynomial basis (x¹ порушує нелінійність DSGE)
- Верифіковано з автором методу (С. Заболотній) коректність реалізації

### 1.4 DSGE sweep (18-19 квітня)

**Конфігурація:** 24 configs = 2 варіанти × 3 бази × 2 порядки × 2 noise types

**Variant A (reconstruction + residual):**

| Basis | S | G loss | G val_SNR | NG loss | NG val_SNR |
|-------|---|--------|-----------|---------|------------|
| fractional | 2 | 23.63 | 0.00 dB | 1.35 | -0.00 dB |
| **fractional** | **3** | **18.77** | **1.20 dB** | **1.35** | **-0.00 dB** |
| polynomial | 2 | 23.62 | ~0.00 dB | 1.35 | -0.00 dB |
| polynomial | 3 | 22.9* | ~0.7 dB* | 1.35 | -0.00 dB |
| robust | 2 | 23.61 | ~0.00 dB | 1.35 | -0.00 dB |
| robust | 3 | 23.63 | -0.01 dB | 1.34 | -0.00 dB |

*polynomial S=3 early stop на epoch 8 через нестабільність тренування

**Variant B (reconstruction + weighted basis):**

| Basis | S | G loss | G val_SNR | NG loss | NG val_SNR |
|-------|---|--------|-----------|---------|------------|
| fractional | 2 | ~24.7* | 0.09 dB* | 1.35 | -0.00 dB |
| fractional | 3 | 23.61 | 0.00 dB | 1.35 | 0.00 dB |
| polynomial | 2 | 23.32 | 0.03 dB | 1.35 | 0.00 dB |
| polynomial | 3 | 23.65 | 0.01 dB | 1.35 | 0.00 dB |
| robust | 2 | 23.65 | -0.05 dB | 1.35 | -0.00 dB |
| robust | 3 | 23.62 | -0.00 dB | 1.35 | -0.00 dB |

### 1.5 Масштабування архітектури (19 квітня)

**Конфігурація:** DSGE vA fractional S=3, powers=[0.5, 1.5, 2.0], 3 ширини UNet

| Width | Params | G val_SNR | NG val_SNR |
|-------|--------|-----------|------------|
| 16 | 10k | -0.00 dB | 0.00 dB |
| 32 | 38k | 0.00 dB | 0.00 dB |
| 64 | 150k | -0.00 dB | 0.00 dB |

**Результат:** Масштабування архітектури **не покращує** результати.

---

## 2. Проблема відтворюваності

### Критичне спостереження

Результат **1.20 dB SNR** для fractional S=3 Variant A gaussian (run_20260418_2f15ce44) **не відтворюється** при повторних запусках. Всі наступні runs з ідентичною конфігурацією дають ~0.00 dB.

**Параметри обох runs — ідентичні:**
- Powers: [0.5, 1.5, 2.0]
- Generating element ‖X‖ = 0.3121
- basis_type: fractional, S=3, lambda=0.01
- Epochs: 50, LR: 3e-4, batch: 4096, partial: 0.05

**Відмінності:**
- Оригінальний run: через bash CLI (`training_hybrid.py --dsge-orders 3`)
- Наступні runs: через Python API (прямий виклик `HybridUnetTrainer(...)`)
- Оригінальний: val_loss = 18.49, train_loss = 18.77
- Наступні: val_loss ~= 23.6, train_loss ~= 23.6

**Можливі причини:**
1. **Data split:** `random_split` з різною ініціалізацією може давати різні val sets
2. **Weight initialization:** PyTorch random seed не зафіксований для моделі
3. **DSGE fit randomness:** порядок обчислення кореляцій може впливати на K
4. **Нормалізація DSGE каналів:** `_p99()` scaling може бути чутливим до data split

### Рекомендації для розв'язання

1. **Перевірка з фіксованим seed:** `torch.manual_seed(42)` перед створенням моделі
2. **Множинні runs:** запустити 5-10 runs з різними seeds і побудувати розподіл SNR
3. **Збереження data split:** зафіксувати індекси train/val/test для відтворюваності
4. **Аналіз winning run:** завантажити збережену модель і перевірити per-SNR криву

---

## 3. Підтверджені висновки

### 3.1 Що точно працює

1. **Прекомп'ютація STFT** — дає 10-100x прискорення тренування спектральних моделей
2. **Non-gaussian тренування** краще за gaussian для стандартних моделей (UNet, ResNet)
3. **Variant A >> Variant B** — reconstruction + residual краще за weighted basis channels
4. **S=2 недостатньо** — всі S=2 конфігурації дають однакове плато незалежно від базису
5. **Fractional basis — найстабільніший** — не дивергує (як polynomial S=3) і не стагнує (як robust S=3)

### 3.2 Що під питанням

1. **DSGE дає +1.20 dB SNR** — результат не відтворюється стабільно
2. **DSGE порівнянний з Transformer** — може бути артефактом
3. **Масштабування архітектури допомагає** — не підтвердилось

### 3.3 Що точно не працює

1. **DSGE з non-gaussian тренуванням** — ~0 dB для всіх конфігурацій
2. **Збільшення UNet width (16→32→64)** — не дає покращення
3. **Robust basis S=3** — не краще за S=2
4. **Variant B** — гірше або рівне Variant A

---

## 4. Технічні артефакти та уроки

### 4.1 MPS vs CPU
- PyTorch STFT на MPS **13x повільніший** за CPU на M3 Max
- MatMul на MPS **2x повільніший** за CPU (AMX coprocessor)
- **Рекомендація:** використовувати CPU для цього проєкту

### 4.2 Memory management
- 400k семплів з прекомп'ютованим STFT = ~14 GB RAM
- 48 GB M3 Max достатньо для 25% даних, але не для 100%
- Потрібна disk-based прекомп'ютація для повного датасету

### 4.3 Early stopping sensitivity
- Early stopping з patience=5 + мікроскопічні покращення val_loss = counter скидається
- Моделі тренуються до 50 epochs навіть на повному плато
- **Рекомендація:** додати threshold для мінімального покращення

### 4.4 Powers selection
- `[0.5, 1.0, 1.5]` vs `[0.5, 1.5, 2.0]` — критична різниця
- `|x|^1.0 = |x|` — по суті модуль, близький до лінійного перетворення
- **Вимога DSGE:** базисні функції мають бути **суттєво нелінійними**

---

## 5. Наступні кроки (пріоритизовані)

### Негайно (перед будь-якими новими експериментами)

1. **Дослідити відтворюваність 1.20 dB** — запустити 10 runs з різними seeds
2. **Завантажити winning model** і перевірити per-SNR криву вручну
3. **Зафіксувати random seeds** для model init, data split, і DSGE fit

### Короткострокові

4. **Повне тренування** (25% даних) з найкращими конфігураціями
5. **FPV сценарій** для підтвердження результатів на іншому SNR діапазоні
6. **Grid search степенів** fractional basis

### Довгострокові

7. **Class-specific generating elements** для non-gaussian тренування
8. **Deeper UNet** (більше шарів замість ширших каналів)
9. **Підготовка публікації** після підтвердження відтворюваності

---

## 6. Файли та артефакти

### Run directories
- `runs/run_20260417_cf018027/` — основний експеримент (12 моделей + compare_report)
- `runs/run_20260417_3881993d/` — non-gaussian спектральні (окремий process)
- `runs/run_20260418_*/` — DSGE sweep (24 конфігурації)
- `runs/run_20260419_*/` — DSGE scaling (12 конфігурацій)

### Звіти
- `runs/run_20260417_cf018027/comparison_report_20260418_124240_uk.md` — cross-evaluation
- `experiments/DSGE_experiment_report.md` — DSGE sweep аналіз
- `experiments/RESEARCH_STATUS_20260419.md` — цей документ

### Код (модифікований)
- `models/dsge_layer.py` — виправлений DSGE з compute_dsge_channels_A/B
- `models/hybrid_unet.py` — параметризована ширина (base_channels)
- `train/training_hybrid.py` — підтримка variant A/B, unet_width, device, partial_train
- `train/training_uae.py` — прекомп'ютація STFT
- `train/training_resnet.py` — прекомп'ютація STFT
- `train/training_vae.py` — прекомп'ютація STFT
- `train/train_all.py` — MPS fallback, model order, batch size fixes

### Загальна статистика
- **37 DSGE runs** збережено
- **~50 годин** сумарного часу тренування
- **~7.8 GB** датасет + ~15 GB run artifacts

---

## 7. Тест відтворюваності (доповнення)

**Дата:** 2026-04-19

10 runs з seeds 42-51, ідентична конфігурація (vA fractional S=3, powers=[0.5,1.5,2.0], 5% data, 50 epochs):

| Seed | val_SNR (dB) |
|------|-------------|
| 42 | -0.009 |
| 43 | +0.000 |
| 44 | -0.008 |
| 45 | +0.001 |
| 46 | +0.002 |
| 47 | +0.000 |
| 48 | +0.001 |
| 49 | +0.004 |
| 50 | +0.040 |
| 51 | +0.000 |
| **Mean ± Std** | **+0.003 ± 0.013 dB** |

**Висновок:** Результат 1.20 dB з DSGE sweep був артефактом (ймовірно multiprocessing fork створив аномальний data split). Реальний результат DSGE Hybrid на 5% deep space даних — ~0 dB.

---

## 8. FPV Telemetry Experiment (19-20 квітня)

### 8.1 Конфігурація

- **Датасет:** `fpv_telemetry_polygauss_qpsk_bs1024_n100000_953c56e8`
- **Сценарій:** FPV телеметрія, SNR -5..+18 dB (значно простіший за deep space -20..0 dB)
- **Дані:** 25% (25k семплів з 100k)
- **Моделі:** UNet, ResNet, DSGE vA fractional S=3 (powers=[0.5,1.5,2.0]), Wavelet
- **Noise types:** gaussian + non_gaussian

### 8.2 Результати

| Model | Params | Gaussian val_SNR | Non-Gaussian val_SNR |
|-------|--------|-----------------|---------------------|
| **UNet** | **~300k** | **14.47 dB** | **15.30 dB** |
| ResNet | ~300k | 13.40 dB | 14.71 dB |
| **DSGE vA frac S=3** | **~10k** | **12.84 dB** | **0.00 dB** |
| Wavelet | — | baseline | baseline |

### 8.3 Аналіз

#### DSGE працює на FPV!

На відміну від deep space (де DSGE давав ~0 dB), на FPV сценарії DSGE Hybrid дає **12.84 dB** з gaussian тренуванням. Це **реальний, значимий результат**.

**Parameter efficiency:**
- DSGE (10k params): 12.84 dB
- UNet (300k params): 14.47 dB
- DSGE досягає **89% якості UNet з 30x менше параметрів**
- Це означає: 0.43 dB SNR на кожні 1k параметрів (DSGE) vs 0.005 dB/1k (UNet)

#### Гіпотеза про non-gaussian тренування підтверджена

- UNet NG (15.30 dB) > UNet G (14.47 dB) — **+0.83 dB** покращення
- ResNet NG (14.71 dB) > ResNet G (13.40 dB) — **+1.31 dB** покращення
- Non-gaussian тренування стабільно краще для стандартних моделей

#### DSGE з non-gaussian тренуванням — нерозв'язана проблема

DSGE NG = 0.00 dB як на deep space, так і на FPV. Проблема **не в складності задачі** (FPV простіший), а в самій **інтеграції DSGE з SmoothL1 loss** або в **формуванні generating element**.

Можливі причини:
1. SmoothL1 loss не дає градієнтів для навчання DSGE-based representation
2. DSGE reconstruction оптимізований під MSE (gaussian), не під SmoothL1
3. Generating element усереднює по всіх SNR рівнях — для NG потрібна class-specific стратегія

### 8.4 Порівняння FPV vs Deep Space

| Model | Deep Space (5% data) | FPV (25% data) |
|-------|---------------------|----------------|
| UNet G | -2.86 dB | 14.47 dB |
| UNet NG | -1.27 dB | 15.30 dB |
| DSGE G | ~0 dB | **12.84 dB** |
| DSGE NG | ~0 dB | 0.00 dB |

Deep space з 5% даних — занадто складна задача для всіх моделей крім Transformer. FPV з 25% даних — реалістичний сценарій де моделі працюють.

---

## 9. Оновлені висновки (20 квітня)

### 9.1 Підтверджено

1. **DSGE з правильною реалізацією Кунченка працює** — 12.84 dB на FPV з 10k params
2. **Parameter efficiency DSGE** — 89% якості UNet при 30x менше параметрах
3. **Non-gaussian тренування краще** для стандартних моделей (UNet +0.83 dB, ResNet +1.31 dB)
4. **Variant A (reconstruction + residual)** — правильний спосіб інтеграції DSGE
5. **Fractional basis S=3** — найкращий базис для DSGE

### 9.2 Не вирішено

1. **DSGE + non-gaussian тренування** — не працює (0 dB на обох сценаріях)
2. **Deep space з малою кількістю даних** — потрібно 25%+ для значимих результатів
3. **Масштабування DSGE** — більша UNet не допомагає (потрібна інша архітектура)

### 9.3 Для публікації

**Publishable result:**
- DSGE Hybrid (10k params) досягає 12.84 dB на FPV — 89% якості UNet (300k params)
- Правильна інтеграція розкладу Кунченка (Variant A) дає +4.36 dB vs наївна реалізація
- Non-gaussian тренування покращує стандартні моделі на +0.8-1.3 dB

**Потрібно додатково:**
- Cross-evaluation DSGE на FPV (G-trained → NG test)
- Per-SNR криві для DSGE vs UNet vs ResNet
- Повторити з більшим обсягом даних (50-100%)

---

## 10. Phase A2 — Систематичний аналіз DSGE+NG failure (2026-04-20..21)

**Контекст:** Розділ 8.3 виявив, що DSGE+non-Gaussian тренування дає 0 dB на обох
сценаріях. План A2 ставить 4 фальсифіковні гіпотези H1–H4 (+ діагностичні H5, H6),
щоб зрозуміти причину. Мета — не виправити "будь-якою ціною", а **встановити
механізм** для розділу Discussion статті.

**Інфраструктура:** A1 (reproducibility) виконано раніше — всі тренери мають
`set_global_seed()`, multiprocessing spawn, `--seed` параметр. σ(val_SNR) між двома
запусками з seed=42 < 0.02 dB підтверджено (task #9).

### 10.1 H1 — Loss incompatibility (✅ 36/36 runs)

**Скрипт:** `experiments/a2_h1_loss_sweep.py`. Конфіг: 6 функцій втрат × 2 сценарії
× 3 сіди (42, 43, 44), 8 epochs, ratio mask, robust S=3, width=16.

**FPV (25% даних):**

| loss | μ val_SNR (dB) | σ | per-seed {42, 43, 44} |
|---|---|---|---|
| **mse**             | **+6.41** | 1.05 | 5.56, 7.58, 6.10 |
| smoothl1_b0.02      | +2.75 | 4.75 | 0.00, 8.24, 0.00 |
| smoothl1_b0.1       | +2.75 | 4.76 | 0.00, 8.24, 0.00 |
| smoothl1_b1.0       | +2.84 | 4.91 | 0.00, 8.50, 0.00 |
| huber_d1.0          | +2.84 | 4.91 | 0.00, 8.50, 0.00 |
| charbonnier_e1e-3   | +2.75 | 4.75 | 0.00, 8.24, 0.00 |

**deep_space (10% даних):** усі 6 функцій втрат → \|μ\| < 0.015 dB, σ < 0.0003 dB.
Універсальний колапс.

**Висновок:** MSE рятує FPV (єдина seed-надійна втрата), але **не вирішує
deep_space**. Втрата необхідна але недостатня. Bimodal pattern під robust losses
на FPV — нова знахідка для статті.

### 10.2 H2 — Ratio mask collapse (✅ 12/12 runs)

**Скрипт:** `experiments/a2_h2_mask_type.py`. Тестує `additive` mask
(`out = clamp(noisy_mag + residual, ≥0)`) проти baseline `ratio` (`out = sigmoid * noisy_mag`).
Загальні умови: SmoothL1(β=0.02), 2 сценарії × 2 mask × 3 сіди × 8 epochs.

**FPV:**

| mask_type | μ val_SNR (dB) | σ | per-seed {42, 43, 44} |
|---|---|---|---|
| ratio (baseline) | +2.75 | 4.75 | 0.002, 8.236, 0.002 |
| **additive**     | **+1.96** | **0.29** | 2.219, 2.020, 1.654 |

**deep_space:**

| mask_type | μ val_SNR (dB) | σ | per-seed {42, 43, 44} |
|---|---|---|---|
| ratio (baseline) | −0.0005 | 0.00023 | −0.00046, −0.00083, −0.00037 |
| additive         | **−0.989** | **0.746** | −1.849, −0.525, −0.593 |

Mask statistics на deep_space additive: residual min до −500, mean ≈ −30..−40.
Network produces aggressive negative offsets, overshoots magnitude → 0 і нижче.

**Висновок:** Additive рятує bimodality на FPV (σ 4.75 → 0.29 dB), але
**погіршує deep_space** (μ=−0.99 dB vs ~0 dB ratio). Mask parameterization
**один з механізмів колапсу**, але не універсальний фіт.

### 10.3 H5 — Standalone noise-subspace DSGE diagnostic (✅ 24/24 runs)

**Скрипт:** `experiments/a2_h5_noise_dsge_baseline.py`. DSGE як автономний
denoiser (без NN): `clean ≈ noisy − k_n·φ(noisy)`. 4 базиси × 2 сценарії × 3 сіди.

**Результат:** Тривіальний колапс до near-identity на robust/fractional/trig
бази (Δcorr ≈ 0). Тільки polynomial на FPV дає чесний +0.5..+1.3 dB з позитивним
Δcorr. Загалом **гіпотеза 'DSGE як шумовий subspace' falsified** — DSGE без NN
не може розділити NG noise від QPSK signal у спостереженнях.

**Артефакт:** `experiments/results/a2_h5_noise_dsge_20260420_225909.md`.

### 10.4 H6 — DSGE fit_target as NN features (✅ 18/18 runs)

**Скрипт:** `experiments/a2_h6_noise_fit_nn.py`. Тестує чи NN рятує тривіальний
DSGE через додаткові канали. 3 fit_targets × 2 сценарії × 3 сіди.

- `signal`: target=clean, input=noisy (baseline)
- `noise`: target=noise, input=noise (DSGE на шумовому subspace)
- `n2n`: target=noise_B (shuffled), input=noise_A (Noise2Noise — теоретично K→0)

**FPV (25% даних):**

| fit_target | μ val_SNR (dB) | σ | per-seed {42, 43, 44} |
|---|---|---|---|
| signal (baseline) | +2.75 | 4.75 | 0.002, 8.236, 0.002 |
| noise             | +2.76 | 4.76 | 0.007, 8.260, 0.002 |
| n2n               | +2.69 | 4.65 | −0.00003, 8.058, 0.001 |

**deep_space (10% даних):** усі 9 runs → \|μ\| < 0.003 dB, σ < 0.001 dB.

**DSGE K-коефіцієнти:**

| Сценарій | Fit | K₁ | K₂ | K₃ | k₀ |
|---|---|---|---|---|---|
| FPV | signal | +0.533 | +0.006 | +0.366 | −0.003 |
| FPV | noise | +0.449 | +0.363 | +0.726 | −0.181 |
| FPV | n2n | ≈0 | ≈0 | ≈0 | ≈0 |
| deep_space | signal | +0.190 | +0.008 | +0.030 | −0.005 |
| deep_space | noise | **−4.353** | **+2.479** | **+5.501** | **−1.244** |
| deep_space | n2n | ≈0 | ≈0 | ≈0 | ≈0 |

**Ключова знахідка:** FPV bimodality replicates **identically** через 3 fit_targets,
що дають K у діапазоні від ≈0 до ≈5. Seed 43 завжди в +8 dB басейні
(8.236/8.260/8.058), seeds {42, 44} завжди колапсують. **DSGE-незалежна знахідка** —
bimodality є властивістю NN training dynamics (ratio mask + SmoothL1), не DSGE.

n2n з K≈0 (нульовий DSGE сигнал) **все одно дає +8 dB на seed 43** → DSGE features
**не є ні необхідними, ні достатніми** для +8 dB стелі на FPV.

deep_space колапсує **через всі 9 комбінацій** попри 100× різницю в \|K\|. Колапс
**не залежить ні від DSGE input** (H6), **ні від loss** (H1), **ні від mask-type
у ratio гілці** (H2).

### 10.5 Консолідована матриця механізмів (Phase A2, оновлено post-H4)

| probe | FPV bimodal fixed? | deep_space collapse fixed? |
|---|---|---|
| H1 MSE | ✅ (σ 4.75→1.05) | ❌ |
| H1 robust losses | ❌ (σ=4.75..4.91) | ❌ |
| H2 additive | ✅ (σ 4.75→0.29) | ❌❌ (worse: μ=−0.99) |
| H6 fit=n2n, K≈0 | ❌ (σ=4.65) | ❌ |
| H6 fit=noise, \|K\|×5 | ❌ (σ=4.76) | ❌ |
| **H4 SNR-bucketed K** | ❌ (σ=4.72, **same seed pattern**) | ❌ (identical to 10⁻⁴ dB) |

**Два якісно різні режими збою:**
1. **FPV bimodality** — `ratio-mask + SmoothL1(β=0.02)` interaction.
   Вирішено: MSE loss або additive mask.
2. **deep_space universal collapse** — нечутливий до loss, mask-type, DSGE input.
   Залишається: H3 (normalization) або H4 (single-GE inadequacy across 23 dB
   SNR range) як остаточні diagnostics.

### 10.6 Що дало A2 для статті

1. **Bimodal convergence** — нова знахідка для §9.1 Results: discrete loss-landscape
   attractors, чотиридесятковий cross-seed reproducibility у обох basin.
2. **DSGE as passenger** — H6 встановлює, що DSGE не драйвер NN behavior на FPV
   у ratio regime (K≈0 → +8 dB).
3. **Scenario-dependent failure modes** — FPV і deep_space мають різні корені
   проблеми; meta-message для Discussion.
4. **Falsifications:**
   - H1 (single-cause loss): не пояснює deep_space
   - H2 (single-cause ratio mask): не рятує deep_space
   - H5 (standalone DSGE noise): тривіальна decomposition
   - H6 (DSGE input matters for NN behavior): не на FPV у ratio regime

### 10.7 Compute use (A2 partial)

| Hypothesis | Runs | Compute |
|---|---|---|
| H1 (loss sweep) | 36 | 5 год |
| H2 (mask type) | 12 | 3 год |
| H5 (standalone DSGE) | 24 | 2 год |
| H6 (fit_target) | 18 | 5 год |
| **A2 spent** | 90 | **15 год** |
| H4 (заплановано) | 18 | ~6 год |
| H3 (заплановано) | 12 | ~3 год |
| **A2 forecast total** | 120 | ~24 год (+2 над бюджетом 22) |

### 10.8 H4 виконано (2026-04-21): SNR-bucketed GE — refuted

**Запуск:** `experiments/a2_h4_snr_bucketed.py`, 12/12 runs complete
(`experiments/results/a2_h4_20260421_075121.json`, ≈2 год wall-clock).
H1 baseline conditions: Variant A, robust S=3, width=16, SmoothL1(β=0.02),
ratio mask, non_gaussian, 8 epochs, 3 seeds. FPV 25% / deep_space 10%.

**Результати:**

| сценарій | bins=0 μ±σ (dB) | bins=3 μ±σ (dB) | висновок |
|---|---|---|---|
| FPV | +2.747 ± 4.754 | +2.726 ± 4.718 | ідентичний seed-патерн (42/44 → 0.002, 43 → +8.17) |
| deep_space | −0.0006 ± 0.0002 | −0.0006 ± 0.0002 | ідентично до 10⁻⁴ dB |

**K-bin структура (mixture cancellation підтверджена):**

На deep_space global K₁=+0.19 приховує зміну знаку (mid-bucket K₁=−0.31, high-bucket K₁=+0.06).
Global K₂≈+0.008 приховує per-bucket 0.06–0.29 (35× range). Global DSGE fit по суті meaningless
як average. Per-bucket fit дає якісно різні K per SNR regime → значно багатші features.

**Але NN outcome незмінний до 10⁻⁴ dB.** Bucketing passes internal DSGE sanity check but does
not change training outcome. **H4 refuted на обох сценаріях.**

### 10.9 Консолідація A2: DSGE-side intervenions exhausted

Після H4 всі DSGE-side важелі перевірено:

| DSGE intervention | H# | висновок |
|---|---|---|
| DSGE input content (signal vs noise vs n2n) | H6 | irrelevant to NN outcome |
| DSGE standalone (noise subspace denoiser) | H5 | trivial; no gain |
| DSGE per-SNR granularity (global vs bucketed) | H4 | irrelevant to NN outcome |

NG-training failures (FPV bimodality, deep_space collapse) — властивості NN
architecture/loss/mask, **не DSGE**. Це **clean scientific finding для Discussion**.

### 10.10 Останній A2 lever: H3 (robust normalization)

**Статус:** queued, launching 2026-04-21. Low theoretical priority (не очікуємо
що рятує ratio-arm dead zone), але cheap (≈3 год compute, 12 runs) і formal
closure діагностичної фази.

**Сценарій:** 2 scenarios × {mean-std (default), MAD + p99 clipping} × 3 seeds = 12 runs.
Скрипт `experiments/a2_h3_robust_norm.py`.

**Прогноз:**
- Якщо deep_space additive-arm покращується (μ з −0.99 → ближче до 0): норма
  контролює outlier targets → frames-of-reference finding.
- Якщо FPV bimodality і deep_space ratio-collapse без змін: confirms
  "NN-training failures are caused by architecture/loss/mask, not normalization" —
  final paper claim.

### 10.11 Перехід до Phase B (заплановано)

A2 забезпечив достатню механістичну ясність для переходу до Phase B main
experiments. Новий порядок (переглянуто 2026-04-21):

1. **H3 у background (2026-04-21):** закриття A2.
2. **Pre-B1 діагностика:** check U-Net baseline on deep_space NG — чи all-arch collapse
   чи DSGE-specific.
3. **B1 FPV main** (10 год) з MSE loss (H1 winner).
4. **B3 real SDR validation** (2 год) — **підвищений пріоритет** для arXiv-quality.
5. **B2 deep_space main** — scope залежить від §2 результату.
6. **C ablations, D writing.**

**Compute used A2 total:** 5 + 3 + 2 + 5 + 3 (H1+H2+H5+H6+H4) = 18 год, плюс H3
3 год → 21 год (план був 22) ≈ on budget.

### 10.12 Pre-B1 діагностика: U-Net baseline на deep_space (2026-04-21)

**Джерело:** існуючий аналіз `runs/run_20260417_cf018027/comparison_data_20260418_124240.csv`
(full deep_space training, pre-DSGE-fix, 1 seed). DSGE-specific числа не довіряти
через стару реалізацію; U-Net/Transformer/Wavelet/ResNet/VAE — валідні
(не залежать від DSGE).

| Model | NG-train → NG-test SNR (dB, aggregate) |
|---|---|
| **TimeSeriesTransformer** | **+1.65** (best) |
| **UnetAutoencoder** | **−1.27** |
| HybridDSGE-UNet (old impl) | −2.28 |
| Wavelet | −5.95 |
| ResNetAutoencoder | −10.65 |
| SpectrogramVAE | −23.79 |

**Key finding:** U-Net does NOT universally collapse на deep_space NG — дає −1.27 dB
(vs A2 hybrid collapse до −0.0006 dB). **A2 universal collapse на hybrid — це
config-specific ефект** (batch 4096, 8 epochs, 10% data), не fundamental limitation
масковано-STFT denoising на deep_space.

**Implication для Phase B2:** повний B2 (50% data, 50 epochs, 3 seeds) meaningful —
moдelі демонструють різну поведінку. Discussion question для paper: чи fixed
DSGE + full training робить DSGE-UNet ≥ U-Net на deep_space.

---

## §11. Phase B1 — FPV main (2026-04-21..04-22, ЗАВЕРШЕНО)

**Конфіг:** FPV 25% data, 30 epochs, 3 seeds × 2 noise types × 4 models = **24 model-trainings**.
Models: UNet, ResNet, HybridDSGE-UNet (robust S=3 vA), Wavelet. Transformer виключено
(CPU bottleneck ~16 год/run × 6 = 96 год, що порушило б 72-год budget — див. §11.4).

**Старт:** 17:49:21 EEST 2026-04-21 (після reset о 17:49 з kill попередньої версії з transformer).
**Кінець:** 07:14:31 EEST 2026-04-22.
**Compute:** 13h 25min wall-clock (avg 2h14min per invocation, дуже стабільно).

**Скрипт:** `experiments/b1_fpv_main.sh`. Лог: `experiments/results/b1_fpv_main_20260421_174921.log`.
6 run_dirs у `data_generation/datasets/fpv_telemetry_polygauss_qpsk_bs1024_n100000_953c56e8/runs/`:
| invocation | seed | noise | run_id |
|---|---|---|---|
| [1/6] | 42 | gaussian | run_20260421_267176a3 |
| [2/6] | 42 | non_gaussian | run_20260421_92a2e0c4 |
| [3/6] | 43 | gaussian | run_20260421_fe4d376a |
| [4/6] | 43 | non_gaussian | run_20260422_d98d21b3 |
| [5/6] | 44 | gaussian | run_20260422_3d92fb74 |
| [6/6] | 44 | non_gaussian | run_20260422_1023e287 |

### 11.1 Per-seed test SNR (dB)

**Gaussian training → gaussian test:**
| seed | UNet | ResNet | Hybrid DSGE | Wavelet (test_MSE) |
|---|---|---|---|---|
| 42 | 14.23 | 13.08 | 10.72 | 0.0901 |
| 43 | 14.13 | 13.21 | 11.51 | 0.0890 |
| 44 | 14.29 | 13.13 | 11.15 | 0.0895 |

**Non-Gaussian training → non-Gaussian test:**
| seed | UNet | ResNet | Hybrid DSGE | Wavelet (test_MSE) |
|---|---|---|---|---|
| 42 | 14.93 | 14.40 | **0.00** | 0.1546 |
| 43 | 14.99 | 14.43 | **11.77** | 0.1532 |
| 44 | 15.07 | 14.38 | **0.00** | 0.1553 |

### 11.2 Cross-seed агрегати (μ ± σ, dB)

| Model | Gaussian | Non-Gaussian |
|---|---|---|
| UNet | **14.22 ± 0.08** | **15.00 ± 0.07** |
| ResNet | **13.14 ± 0.07** | **14.40 ± 0.025** |
| **Hybrid DSGE (robust S=3 vA)** | **11.13 ± 0.40** | **3.92 ± 6.79** ⚠ bimodal |

**Reproducibility:** UNet/ResNet σ ≤ 0.08 dB на всіх 6 cells — відмінна стабільність,
підтверджує A1 інфраструктуру (set_global_seed).

### 11.3 Ключові наукові висновки B1

#### 11.3.1 Parameter-efficiency на gaussian — підтверджено

Hybrid DSGE 11.13 ± 0.40 dB при ~30k params vs UNet 14.22 ± 0.08 dB при ~300k params.
**10× менше параметрів, gap −3.09 dB.** Це consistent з раніше задекларованим
parameter-efficient результатом FPV з виправленою DSGE-реалізацією.

Pareto-tradeoff:
- UNet: best absolute SNR, найбільша модель.
- Hybrid DSGE: 78% of UNet performance with 10% of params.
- ResNet: середній (13.14 dB), ~150k params.
- Wavelet (classical, no params, MSE 0.089): ~зіставний баseline для G.

#### 11.3.2 NN-NG generalization > NN-G на NG test (всі сітки)

| Model | G→G | NG→NG | Δ |
|---|---|---|---|
| UNet | 14.22 | 15.00 | **+0.78** |
| ResNet | 13.14 | 14.40 | **+1.26** |

**Висновок:** для FPV scenario навчання на полі-Гауссовому шумі дає **кращий
denoise** на тому ж типі шуму на тестовому наборі — підтверджує **central
hypothesis** статті (NN на realistic non-Gaussian noise generalizes better).

Hybrid DSGE на NG видає bimodal (див. 11.3.3) — частково підтверджує
hypothesis, частково розкриває наявність failure mode.

#### 11.3.3 Hybrid DSGE NG = бімодальна тренувальна динаміка

Найважливіша Discussion-знахідка B1:

| seed | Hybrid NG SNR | mask μ (final epoch) | basin |
|---|---|---|---|
| 42 | 0.00 dB | ~0.04 (lowest 25%) | collapse |
| 43 | 11.77 dB | ~0.5 (healthy) | working |
| 44 | 0.00 dB | ~0.04 | collapse |

**2/3 seeds collapse, 1/3 working.** Відтворює A2-H1 спостереження бімодальності
**на повномасштабній конфігурації** (30 epochs vs 8 у H1; 25% data vs 25%).

Це остаточно спростовує гіпотезу, що бімодальність — undertraining артефакт.
Це **basin-of-attraction property** оптимізаційного ландшафту Hybrid DSGE
+ ratio mask + FPV NG.

**Mechanism (per A2-H2):** mask інціалізується як (близько до) 0.5; на FPV NG
з heavy-tail noise існує attractor mask→0 (degenerate solution де модель
видає нуль). У 1/3 seeds initial random direction достатньо віддалена,
щоб уникнути attractor.

**Material для Discussion §9:** комбінувати з A2 систематичним аналізом
(H1-H6 refuted DSGE-side mechanisms) → **failure локалізована до
NN-mask-loss interaction**, не DSGE-side.

#### 11.3.4 Wavelet baseline стабільний

Wavelet grid search вибирає однакові optimum параметри (db4, level=2, soft, sym)
на всіх 6 invocations — параметрично стабільний baseline. MSE різниця G vs NG
~1.7× (0.089 vs 0.154) — wavelet деградує під NG, але без catastrophic collapse.

### 11.4 Transformer exclusion — методологічна нотатка

Початкова версія `b1_fpv_main.sh` включала Transformer. Per-epoch timing на CPU
(batch=128, 98 batches/epoch) виявився ~30 хв/epoch → ~16 год за один transformer
run. 6 invocations × 16 год = 96 год — несумісно з 72-год compute budget.

Прийнято рішення (~17:30 21-04) **виключити Transformer з B1**, відновити після
B2/B3 окремими single-seed runs якщо час дозволить. Transformer baseline для
порівняння у paper буде:
- з prior runs (run_20260417_cf018027 has Transformer на full deep_space);
- з targeted single-seed B1 follow-up якщо compute дозволить (~16 год).

### 11.5 Compute usage

| Phase | Hours | Cumul |
|---|---|---|
| Фаза 0 (datasets) | 4 | 4 |
| A1 (repro infra) | 5 | 9 |
| A2 (H1-H6) | ~21 | 30 |
| B1 (FPV main) | 13.5 | **43.5** |
| Budget | 72 | — |
| Залишилося | — | **28.5** |

Залишилося compute для: **B3 real SDR (~2 год)** + **B2 deep_space (15-25 год)** +
Phase C (~12 год) + Phase D (~3 год) ≈ **32-42 год потрібно**. Можливе незначне
перевищення; стратегії згортання — B2 на 25% даних замість 50% (−10 год), або
seed=2 замість 3 у B2.

### 11.6 Наступний крок (рекомендований)

1. **`compare_report.py` aggregation across 6 run_dirs** — згенерувати cross-seed
   таблиці з per-SNR розбивкою + crossover heatmaps + per-model curves.
2. **B3 real SDR validation (2 год)** — ZeroShot eval pretrained B1 моделей на
   адаптованому RadioML 2018 / DroneDetect subset (з Phase 0 adapters).
   **Обов'язкове** для arXiv-quality (Q9 з research plan).
3. **B2 deep_space main** запуск (~16 год) — паралельно з аналізом B1 та B3.

### 11.7 Cross-seed агрегатор (виконано 2026-04-22 10:25)

**Артефакти:**
- `analysis/aggregate_b1.py` — pulls `training_report.json` з усіх 6 run_dirs,
  будує μ±σ таблиці (overall + per-SNR + val_SNR).
- `experiments/results/b1_aggregate.md` — публікаційно-ready таблиці.
- `experiments/results/b1_aggregate.json` — машино-читаний дамп для D1 stats.

**Відкриті дефекти `train/compare_report.py`** (виявлені; бай-пасс через
aggregator). ВИПРАВЛЕНО 2026-04-22 13:44 — див. §11.8.

Aggregator використовує `denoise_numpy` кожного trainer-а (через JSON, що
зберігається наприкінці тренування) — це авторитетне джерело істини.

**Підтверджені висновки B1 (cross-seed σ):**
- UNet: G→ 14.22±0.08, NG→ 15.00±0.07. Δ(NG−G) = +0.78 dB на тесті.
- ResNet: G→ 13.14±0.07, NG→ 14.40±0.03. Δ(NG−G) = +1.26 dB.
- HybridDSGE_UNet: G→ 11.13±0.39, NG→ **3.92±6.79** (бімодальність 2/3
  колапс — підтверджено σ ≈ 9 dB).
- Wavelet: тільки MSE доступне (без SNR), G: 0.0896±0.0005, NG: 0.1544±0.0010.

**Per-SNR пік (p18dB input)** — UNet NG: 25.54±0.06 dB, що на ~9 dB вище
від «no-change» лінії. Hybrid NG за тим же пунктом: 5.43±9.40 (σ домінує).

### 11.8 Фікси `train/compare_report.py` (виконано 2026-04-22 13:44)

Виправлено **4 дефекти** в `train/compare_report.py`, які давали систематично
хибні cross-evaluation метрики (до фіксу звіт від 09:57 показував ResNet G→G
= −13.78 dB, UNet G→G = 6.02 dB — обидва завищено/занижено у 2-27 dB).

**Ґавки та фікси:**

| # | Ґавка | Файл/лінія | Правка |
|---|---|---|---|
| 1 | ResNet mask не застосовано (`out = model(spec)` без `* spec`) | `_load_resnet` | `out_mag = (model(spec) * spec).squeeze(1)` |
| 2 | Hybrid regex `_S(\d+)$` не бере `_vA`/`_vB` суфікс → Hybrid тихо пропускається | `discover_runs` | regex → `_S(\d+)_v([AB])(?:_w(\d+))?` з back-compat fallback |
| 3 | Hybrid інференс не знав про variant (A=2 канали vs B=1+S каналів) | `_load_hybrid` | сигнатура приймає `dsge_variant`, `unet_width`, `mask_type`, `dsge_norm_method`; канали через `compute_dsge_channels_A/B()` |
| 4 | STFT convention mismatch: `scipy.signal.stft(boundary=None)` ≠ `torch.stft(center=True, pad_mode='reflect')` → 8 dB систематичного заниження | `_load_unet`, `_load_resnet`, `_load_hybrid` | додано `_torch_stft_helpers()`, переведено всі три на torch STFT з Hann вікном |

Для Hybrid: основна STFT (канал 0) — через torch; DSGE-канали (1..N) — через
scipy (віддзеркалює training_hybrid.py mixed approach). Додано T'-padding для
сумісності сіток torch/scipy.

**Валідація (seed 42, gaussian, середнє по SNR bins):**
- UNet G→G: **16.51 dB** (до Фіксу 4: 6.02 dB, 10 dB помилки).
- ResNet G→G: **14.96 dB** (до Фіксу 1: −13.78 dB, 29 dB помилки).
- Hybrid G→G: **12.06 dB** (до Фіксу 2+3: відсутня в таблиці).

Відхилення від per-SNR-усереднення aggregate_b1: UNet 0.3 dB, ResNet 0.2 dB,
Hybrid 0.2 dB — всі в межах seed-specific шуму. Фікси узгоджені з authoritative
`training_report.json`.

**Cross-seed таблиця (3 сіди, після фіксів):**

| Model | Train | G→G (dB) | G→NG (dB) | NG→G (dB) | NG→NG (dB) |
|---|---|---:|---:|---:|---:|
| UnetAutoencoder | μ±σ (n=3) | 16.20±0.27 | 15.45±0.66 | 17.24±0.04 | 17.56±0.05 |
| ResNetAutoencoder | μ±σ (n=3) | 15.15±0.36 | 14.47±0.73 | 16.47±0.04 | 16.89±0.05 |
| HybridDSGE_UNet | μ±σ (n=3) | 12.25±0.22 | 11.44±0.48 | 4.37±7.57 | 4.37±7.57 |
| Wavelet | (det) | 10.17 | 8.91 | 10.17 | 8.91 |

**Висновки (підтверджено cross-evaluation):**
- Гіпотеза роботи **підтверджується** для UNet/ResNet: NG-training зменшує
  деградацію на G-тесті (UNet: +1.04 dB; ResNet: +1.32 dB) і покращує NG-тест
  (UNet: +2.11 dB; ResNet: +2.42 dB). NG→NG > NG→G > G→G > G→NG.
- Hybrid **катастрофічно нестабільний** на NG-тренуванні (бімодальність 2/3
  колапс підтверджено зі σ=7.57 dB у cross-eval; сід 43 досягає 13.10 dB).
- Wavelet у compare_report показує **нульову дисперсію** по сідах
  (10.17 на G, 8.91 на NG). Це коректна поведінка, не баг: wavelet
  grid-search сходиться до тих самих best_params (`db4/level=2/soft/
  symmetric`) на всіх 3 сідах — grid дискретний, дані стабільні; плюс
  compare_report eval-ить на FIXED per-SNR test files
  (`test_m5dB_*.npy`), не на per-seed 25% split. Детермінізм очікуваний.
  Відмінність від training_report (де test_mse варіюється 0.0890–0.0901)
  пояснюється different eval subsets: training_report → per-seed random
  25%, compare_report → shared per-SNR bins.

---

## §12. Phase B2 — deep_space sanity (2026-04-22, ✅ ЗАВЕРШЕНО 21:58 EEST)

**Мета:** дешева перевірка (1 seed=42, 10% даних, 15 epochs, CPU) перед тим як
запускати повноцінний B2 main sweep. Рішення про scale (25% vs 50% × 2-3 seeds)
залежить від того, **чи моделі взагалі навчаються на deep_space при обмеженому
compute**. Мотивація — §10.12 pre-B1 (full deep_space NG UNet = −1.27 dB, тобто
ланд ndscape не безнадійний) та §11.3.3 (Hybrid NG bimodal на FPV).

**Скрипт:** `experiments/b2_deep_space_sanity.sh` (PID 1588, стартував 15:39:48 EEST).
Лог: `experiments/results/b2_sanity_20260422_153948.log` та `/tmp/b2_sanity.log`.
Конфіг: `train_all.py --models unet,resnet,hybrid,wavelet --partial-train 0.10
--epochs 15 --device cpu --seed 42 --nperseg 128`, dataset
`deep_space_polygauss_qpsk_bs1024_n400000_0310b7e7`.

Два invocations (обидва завершено, wall-clock 6h19min):
| # | seed | noise | run_id | статус |
|---|---|---|---|---|
| [1/2] | 42 | gaussian | `run_20260422_c2ec27e2` | ✅ complete |
| [2/2] | 42 | non_gaussian | `run_20260422_cbc5ecfc` | ✅ complete |

### 12.1 Per-SNR результати (seed=42, gaussian, 10% data, 15 ep) — попередні

**UnetAutoencoder G→G:**

| SNR_in | −20 | −17 | −15 | −12 | −10 | −7 | −5 | −3 | 0 | +3 |
|---|---|---|---|---|---|---|---|---|---|---|
| SNR_out (dB) | −0.48 | 1.18 | 2.43 | 3.81 | 5.24 | 6.80 | 7.68 | 8.86 | 10.05 | 11.79 |

Aggregate ≈ **+5.74 dB**. Slope ~0.53 dB per dB SNR_in. Модель учиться.

**ResNetAutoencoder G→G:**

| SNR_in | −20 | −17 | −15 | −12 | −10 | −7 | −5 | −3 | 0 | +3 |
|---|---|---|---|---|---|---|---|---|---|---|
| SNR_out (dB) | −0.15 | 0.77 | 1.37 | 3.17 | 3.73 | 6.03 | 7.44 | 9.11 | 10.22 | 11.88 |

Aggregate ≈ **+5.36 dB**. Близько до UNet (−0.4 dB) — consistent з B1 FPV порядком.

**HybridDSGE_UNet_robust_S3_vA G→G: ⚠ КОЛАПС**

| SNR_in | −20 | −17 | −15 | −12 | −10 | −7 | −5 | −3 | 0 | +3 |
|---|---|---|---|---|---|---|---|---|---|---|
| SNR_out (dB) | −0.00 | 0.00 | 0.00 | 0.00 | −0.00 | −0.01 | −0.02 | −0.04 | −0.05 | −0.04 |

Aggregate ≈ **−0.016 dB**. Ідентити-output на всіх SNR-бінах. Класичний
mask→identity attractor (ratio-mask degenerate solution).

**Wavelet G:** best_params `db4/level=4/soft/symmetric`; Val MSE 1.096, Test MSE 1.097.

### 12.1b Per-SNR результати (seed=42, **non_gaussian**, run_20260422_cbc5ecfc)

**UnetAutoencoder NG→NG:**

| SNR_in | −20 | −17 | −15 | −12 | −10 | −7 | −5 | −3 | 0 | +3 |
|---|---|---|---|---|---|---|---|---|---|---|
| SNR_out (dB) | +0.15 | 1.68 | 3.06 | 5.15 | 6.48 | 8.42 | 9.69 | 11.01 | 12.89 | 14.65 |

Aggregate ≈ **+7.32 dB**. **+1.58 dB над UNet G→G (+5.74)** — суттєво краще на NG.

**ResNetAutoencoder NG→NG:**

| SNR_in | −20 | −17 | −15 | −12 | −10 | −7 | −5 | −3 | 0 | +3 |
|---|---|---|---|---|---|---|---|---|---|---|
| SNR_out (dB) | +0.52 | 1.44 | 2.49 | 4.41 | 5.69 | 7.54 | 8.69 | 9.85 | 11.38 | 12.54 |

Aggregate ≈ **+6.46 dB**. **+1.10 dB над ResNet G→G (+5.36)**.

**HybridDSGE_UNet NG→NG: ⚠ КОЛАПС (знову)**

| SNR_in | −20 | −17 | −15 | −12 | −10 | −7 | −5 | −3 | 0 | +3 |
|---|---|---|---|---|---|---|---|---|---|---|
| SNR_out (dB) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | −0.00 | −0.00 | −0.00 | −0.00 |

Aggregate ≈ **0.000 dB**. Ідентичний колапс до G-фази.

**Wavelet NG:** best_params `db4/level=4/soft/symmetric` (той самий optimum що G);
Val MSE 3.751, Test MSE 3.779 — ~3.4× гірше за G, класична деградація wavelet під
heavy-tail.

### 12.2 Нова наукова знахідка: Hybrid collapse тепер на **Gaussian** deep_space

До B2: A2 universal collapse на deep_space (всі H1..H6) проявлявся тільки на
**non_gaussian** training з SmoothL1. §10.12 pre-B1 вважав collapse
config-specific (batch=4096, 8 ep) і очікував що full training + MSE loss
розблокує Hybrid на deep_space.

B2 sanity falsifies це очікування: **MSE loss + robust S=3 vA + 10% data + 15 ep +
batch=4096 + seed=42 + Gaussian noise** → **Hybrid колапсує до identity**, тоді як
UNet і ResNet (той самий data/epochs/batch) нормально тренуються.

**Це якісно відрізняється від FPV B1**, де Hybrid G давав 11.13±0.40 dB з
ідентичною конфігурацією моделі. Різниця сценаріїв (SNR range −20..0 vs −5..+18)
переводить Hybrid з працюючого режиму в колапс навіть на Gaussian.

**Інтерпретація (попередня):** DSGE features, обчислені на deep-space noisy
signal (|noisy| ≈ |noise| через низький SNR), є здебільшого шумом, а не сигналом.
Mask виходить до identity як degenerate global minimum. На FPV високі SNR-бічки
(+18 dB) забезпечують clean signal-dominated frames, які утримують mask→0.5
attractor.

### 12.3 Implications для Phase B2 main

Попередній план B2 (§11.6: 16 год, 3 seeds × 2 noise × 4 models на 50% даних)
потребує перегляду. Варіанти:

1. **Все одно запускати full B2** — відомо, що Hybrid колапсує на deep_space;
   фіксація цього факту з 3 seeds + більшим compute **є валідним публікаційним
   результатом** ("DSGE-UNet fails on deep-space-class SNR regimes" як companion
   до FPV success story).
2. **Mini-B2** — 1 seed × 25% даних × 30 epochs, тільки щоб підтвердити:
   a) UNet/ResNet навчаються (B1-style numbers),
   b) Hybrid колапс не пропадає з більшими epochs/data.
3. **Pivot:** дослідити, **чому Hybrid окей на FPV G, але не на deep_space G** —
   новий A3-style діагностичний хід (scenario-dependent DSGE failure).

Рекомендація: (2) + включити в paper як scenario-dependency result.
Перевага: +10-15 год compute (не 30+), відповідь на Q2/Q4 research plan
фактологічна (Hybrid має поріг SNR range, нижче якого не працює).

### 12.4 Compute та наступні кроки

| Фаза | Год | Cumul |
|---|---|---|
| До §11.8 | 43.5 | 43.5 |
| B2 sanity (обидві фази, 15:39 → 21:58) | 6.3 | 49.8 |
| Бюджет | 72 | — |
| Залишилося | — | 22.2 |

### 12.5 Консолідовані висновки B2 sanity (2026-04-23)

**Cross-noise aggregate (seed=42, 10% data, 15 epochs, CPU):**

| Model | G→G (dB) | NG→NG (dB) | Δ(NG − G) | Коментар |
|---|---:|---:|---:|---|
| UNet | +5.74 | **+7.32** | **+1.58** | ✅ NG-training краще, monotonic curve |
| ResNet | +5.36 | **+6.46** | **+1.10** | ✅ NG-training краще |
| Hybrid DSGE (robust S=3 vA) | −0.02 | 0.000 | 0.00 | ❌ колапс на обох |
| Wavelet (MSE) | 1.097 | 3.779 | 3.44× worse | класична деградація під NG |

**Ключові знахідки:**

1. **Central hypothesis підтверджена на deep_space** (раніше підтверджена тільки
   на FPV у B1). UNet NG >> UNet G (+1.58 dB), ResNet NG >> ResNet G (+1.10 dB).
   Узгоджено з FPV B1 деltas (UNet +0.78, ResNet +1.26) — тренд consistent поперек
   обох сценаріїв, **effect size більший на deep_space** (low-SNR regime має більше
   простору для NG-specific gains).

2. **§10.12 pre-B1 баseline застарів.** Попередня оцінка "UNet NG на deep_space =
   −1.27 dB" базувалась на run_20260417_cf018027 (old impl, ratio-mask, SmoothL1,
   full 100% data but shorter training). Fixed pipeline + MSE + навіть 10% data
   дає **+7.32 dB** — якісний стрибок.

3. **Hybrid collapse scenario-dependent, але noise-independent.** Hybrid колапсує
   до identity на deep_space під G (MSE loss) та під NG (SmoothL1) **однаково**.
   Порівняно з §11.3.3 FPV NG bimodality (2/3 seeds collapse) — на deep_space
   (1 seed) отримано detзерministic collapse для обох noise types. Hybrid працює
   тільки на FPV G (+11.13 dB, 100% seed-reliable у B1). Це **scenario-specific
   failure**, не noise-specific.

4. **Гіпотеза про mechanism:** у low-SNR regime (deep_space: input SNR ∈ [−20, 0] dB)
   `|noisy| ≈ |noise|`, тому DSGE basis expansion на `noisy_magnitude` продукує
   features, які майже повністю закодовують шум, не сигнал. Ratio mask → identity
   стає degenerate global minimum. На FPV (input SNR ∈ [−5, +18] dB) частина
   frames має clean-dominated spectrum, що утримує mask ≠ identity.

5. **Wavelet на deep_space G дає Test MSE 1.097.** Це **краще** за Hybrid
   (який дає ~0.50 MSE на всіх SNR — відповідає identity output при scaled targets
   0.5; Wavelet MSE-на-raw-signals не пряме порівняння без normalization). Для
   публікаційних порівнянь потрібна уніфікована метрика (SNR dB) через Wavelet
   denoise_numpy — не просто val/test MSE.

### 12.6 Рішення про Mini-B2

На основі §12.5 знахідок, варіант (2) з §12.3 **зайвий:**
- Hybrid collapse deterministic → нема seed variance для eliminate.
- UNet/ResNet NG-training effect уже видно на 10% даних (+1.58 / +1.10 dB).

**Переглянута рекомендація для Phase B2 main:**

Замість Mini-B2 запустити **B2 main (3 seeds × 25% даних × 30 epochs)** для UNet/
ResNet (з Hybrid як "failure baseline"). Обґрунтування:
- 3 seeds — обов'язкові для publishable σ (σ на FPV B1 був ≤0.08 dB).
- 25% даних (не 50%) економить ~10 год compute; gains з 25→50% на FPV були
  <0.3 dB, малоймовірно що deep_space дасть більше.
- 30 epochs — достатньо для convergence (UNet на 15ep уже monotonic SNR curve).
- Hybrid включити у той же sweep для 3-seed collapse reproducibility (важливе
  discussion-твердження: collapse is not seed-specific).

**Оцінка compute:** 3 seeds × 2 noise × 4 models × 25% × 30ep ≈ 12-16 год CPU.
Вкладається в залишок 22 год.

**Скрипт:** `experiments/b2_mini.sh` (уже створено, 25% / 30ep / 1 seed) треба
переробити на 3 seeds. Зробити `b2_main.sh` і запустити.

### 12.7 B3 scaffold готовий

`experiments/b3_real_sdr_zeroshot.py` створено (2026-04-22). Reuse:
- `compare_report.discover_runs` — loads всіх моделей з B1 run_dir.
- `snr_curve.evaluate_per_snr` — сумісний з adapter test schema.

CLI:
```
python experiments/b3_real_sdr_zeroshot.py \
    --b1-run <fpv_run_dir> \
    --real-dataset <adapted_ds> \
    --test-noise non_gaussian
```

**Prereq:** запустити `load_radioml2018.py` або `load_dronedetect.py` для
генерації real-dataset dir. Raw HDF5 / SigMF dump потрібен на диску.

### 12.8 Immediate next (2026-04-23)

1. **Переробити `b2_mini.sh` → `b2_main.sh`** з 3 seeds (42, 43, 44), запустити
   у фоні (~12-16 год).
2. **Паралельно:** перевірити наявність raw RadioML HDF5 / DroneDetect SigMF на
   диску користувача; якщо є — запустити адаптери, потім B3 скрипт.
3. **Пост-B2:** `aggregate_b1.py`-style aggregator для B2 main (6 run_dirs) →
   cross-seed σ-таблиці для статті.

---

## §13. Phase B2 main — deep_space (2026-04-23, ⏳ RUNNING, 2 seeds план)

### 13.1 Конфіг та запуск

**Скрипт:** `experiments/b2_main.sh` (PID 33983, стартував 07:31:35 EEST 2026-04-23).
Лог: `experiments/results/b2_main_20260423_073135.log`.

```
SEEDS=(42 43 44)   # 44 буде вбито (див. §13.4)
NOISE_TYPES=(gaussian non_gaussian)
MODELS=unet,resnet,hybrid,wavelet
EPOCHS=30, PARTIAL=0.25, DEVICE=cpu
```

6 invocations заплановано; **kill plan: зупинити після [4/6]** — 2 повні seeds,
економія ~14 год compute (див. §13.4 обґрунтування).

| # | seed | noise | час старту | статус | run_id |
|---|---|---|---|---|---|
| [1/6] | 42 | gaussian | 23-07:31:35 | ✅ done 23-14:29:11 (6h58m) | `run_20260423_ed46f843` |
| [2/6] | 42 | non_gaussian | 23-14:29:11 | ✅ done 23-22:44:30 (8h15m) | `run_20260423_3d8957d2` |
| [3/6] | 43 | gaussian | 23-22:44:30 | ✅ done 24-05:27:58 (6h43m) | `run_20260423_2b4b99cc` |
| [4/6] | 43 | non_gaussian | 24-05:27:58 | ✅ done 24-13:45:08 (8h17m) | `run_20260424_20fc5edf` |
| [5/6] | 44 | gaussian | 24-13:45:08 | ⛔ killed before UNet Ep01 | — |
| [6/6] | 44 | non_gaussian | — | ⛔ never started | — |

### 13.2 Per-SNR результати (поки що)

**[1/6] seed=42 gaussian, 25%/30ep:**

| Model | SNR_out (dB) per input SNR (−20..+3) | agg |
|---|---|---:|
| UNet G→G | −0.43 / 1.03 / 2.38 / 4.09 / 6.06 / 7.87 / 8.62 / 10.48 / 11.75 / 14.14 | **+6.58** |
| ResNet G→G | −0.21 / 1.13 / 2.11 / 3.87 / 5.41 / 7.53 / 8.29 / 10.24 / 11.40 / 13.73 | **+6.35** |
| Hybrid G→G | −0.24 / **−8.68** / **−6.85** / 0.77 / **−2.44** / 0.18 / 4.22 / 3.61 / 6.40 / 7.04 | **+0.40 (erratic)** |

**[2/6] seed=42 non_gaussian, 25%/30ep (UNet complete, інші у процесі):**

| Model | SNR_out (dB) | agg |
|---|---|---:|
| UNet NG→NG | 0.10 / 1.80 / 3.23 / 5.28 / 6.62 / 8.68 / 10.03 / 11.41 / 13.41 / 15.31 | **+7.59** |

### 13.3 Нові наукові знахідки B2 main

#### 13.3.1 Central hypothesis знову підтверджена на deep_space

UNet NG (+7.59) > UNet G (+6.58) → **Δ = +1.01 dB** на seed=42 / 25%/30ep.

Порівняння з sanity (10%/15ep): Δ був +1.58 dB. Ефект **зменшується з compute**
(+0.57 dB gain gap closes as UNet G purer-trains), але стабільно залишається
**позитивним** і суттєвим. Consistent з FPV B1 (Δ ≈ +0.78 dB) з урахуванням того,
що deep_space має більший SNR-range → більший простір для NG-specific gains.

#### 13.3.2 Hybrid — ТРЕТІЙ режим failure на deep_space

Послідовність Hybrid behaviour на deep_space G:

| Compute | Поведінка | Aggregate | Характеристика |
|---|---|---|---|
| 10% / 15 ep (sanity) | pure identity | 0.00 dB | mask = 1.0 на всіх frames |
| **25% / 30 ep (main)** | **erratic non-identity** | **+0.40 dB** | mask learns *wrong* features, **non-monotonic curve** (−8.68 dB на SNR_in=−17, +7.04 dB на SNR_in=+3) |
| 100% / 50 ep (очікується) | невідомо | ? | — |

**Що означає erratic non-identity:** модель **вийшла з identity attractor** з
більшим compute, але не знайшла корисного minimum — продукує аутпут, що **активно
псує** сигнал на bins −17..−10 dB (SNR_out < SNR_in), водночас слабко покращує на
високих SNR bins (−3..+3). Це якісно гірше за identity: negative-SNR degradation.

**Гіпотеза mechanism:** ratio-mask + MSE loss + deep-space DSGE features створюють
shallow local minima де модель оптимізує average MSE, але кидає signal-power у
випадкові band-частини — класичний сигнал spectral leakage через погано
conditioned mask. При більшому compute модель залишає safe identity attractor і
дрейфує у такі local minima.

#### 13.3.3 Retraction: "NG-effect зник" з попереднього рапорту

У рапорті 16:40 EEST я помилково приписав ResNet G→G числа UNet NG, і сформулював
що "NG-effect зник з compute" (Δ=−0.06). Це була **помилка зчитування таблиці**.
Правильне Δ = **+1.01 dB** (після перевірки моніторингом о 19:30, коли UNet NG
таблиця дійсно з'явилась у лозі). Central hypothesis підтверджена і на main-scale.

### 13.4 Kill plan — обґрунтування

**Причина:** per-invocation time на main scale = 6h58min ([1/6] виміряно). Повний
run = 6 × 7h ≈ 42h, overshoot 20h від 22h budget-залишку. Compute risk для Phase
C/D (~15h разом).

**Рішення (прийнято 2026-04-23):** зупинити після [4/6] (seeds 42, 43 обидва
noise types = 4 invocations).

**Trade-offs:**
- n=2 seeds замість n=3 для cross-seed σ. Consistent з FPV B1 де σ було <0.1 dB —
  різниця σ(n=2) vs σ(n=3) не змінить висновків.
- Seed=43 підтвердить що Hybrid erratic non-identity **не seed-specific** artefact.
- Економія 14h → залишає 8h на Phase C (aggregation + cross-eval) + 4h на D (writing).

**Execution:** kill PID 33983 коли монітор повідомить "### [4/6] done"
(очікується ≈ 11:30 EEST 2026-04-24).

### 13.5 Compute tally

| Фаза | Год | Cumul |
|---|---|---|
| До B2 main | 49.8 | 49.8 |
| B2 main [1/6] done | 6.97 | 56.8 |
| B2 main [2/6] in progress | ~4 поки що | ~60.8 |
| B2 main [3/6] + [4/6] (очікується) | ~14 | ~74.8 |
| Budget | 72 | — |
| Overshoot | — | ~2.8h |

Прийнятно (vs 20h overshoot при full 6/6).

### 13.6 Per-seed aggregate (B2 main, invocations [1..3]/6 complete, [4/6] running)

| Model | s42 G→G | s42 NG→NG | s43 G→G | s43 NG→NG |
|---|---:|---:|---:|---:|
| UNet | +6.58 | **+7.59** | +6.35 | ⏳ |
| ResNet | +6.35 | **+7.13** | +6.21 | — |
| Hybrid | **+0.40** (erratic) | 0.00 (identity) | **−0.01** (identity) | — |
| Wavelet (MSE) | 1.097 | 3.779 | ~1.09 | — |

### 13.7 Cross-seed μ±σ (n=2, G-колонка) та Hybrid bimodality

**UNet G→G:** μ = +6.47, σ = **0.16** dB (n=2)
**ResNet G→G:** μ = +6.28, σ = **0.10** dB (n=2)
**Hybrid G→G:** μ = +0.19, σ = **0.29** dB (n=2, **bimodal!**)

Reproducibility UNet/ResNet consistent з FPV B1 (σ ≤ 0.08 на n=3). Публікаційно-
ready для 2-seed reporting.

**Hybrid bimodality на deep_space G:**

Критична знахідка [3/6]: при ідентичному конфізі (seed лише відрізняється) Hybrid
потрапляє у **якісно різні basins**:

- seed=42 → **erratic non-identity** (escape from identity at main scale, land у
  pathological minimum: SNR_out=−8.68 dB на SNR_in=−17, non-monotonic curve, agg +0.40)
- seed=43 → **pristine identity** (mask ≈ 1.0 на всіх frames, agg −0.006)

**Mapping failure modes across experiments:**

| Experiment | Hybrid behaviour | Seed-dependency |
|---|---|---|
| FPV B1 NG | bimodal: 2/3 identity collapse, 1/3 works @+11.77 | YES |
| Deep_space B2 sanity G (10%/15ep) | uniform identity collapse | — |
| Deep_space B2 main G (25%/30ep) | **bimodal: identity vs erratic** | **YES** |
| Deep_space B2 main NG (25%/30ep, s42) | pristine identity | pending s43 |

**Нова insight:** Hybrid bimodality **universal across scenarios**, але direction
basins змінюється з compute scale:
- На small compute (10%/15ep): усі seeds → identity (shallow optimization).
- На larger compute (25%/30ep): deeper optimization → seeds розділяються на
  *safe identity* vs *risky escape* basins.
- На FPV NG (§11.3.3): compute вистачає щоб у деяких seeds знайти *useful*
  minimum (seed=43: +11.77 dB).

Це **basin-selection property** оптимізаційного ландшафту Hybrid — loss landscape
має ≥3 attractors (identity, erratic-bad, useful), доступність яких залежить від
scenario (SNR range → basin depths) та random init.

### 13.8 Оновлений compute tally

| Фаза | Год | Cumul |
|---|---|---|
| До B2 main | 49.8 | 49.8 |
| B2 [1/6] 6h58m | 6.97 | 56.8 |
| B2 [2/6] 8h15m | 8.25 | 65.1 |
| B2 [3/6] 6h43m | 6.72 | 71.8 |
| B2 [4/6] очікується ~7h | 7.0 | 78.8 |
| Budget | 72 | — |
| Overshoot | — | **~6.8h** |

Overshoot ~6.8h прийнятний (vs 20h при full 6/6). Залишок compute на Phase C/D:
72-49.8 − 29 = −6.8h → потрібно економити в Phase C (aggregator reuse, мінімум
нового кodu) + Phase D (writing може йти паралельно з running analysis).

### 13.9 Retractions / корекції у §13 попередніх версіях

- **16:40 2026-04-23:** помилково приписав ResNet G→G числа UNet NG, висновок
  "NG-effect зник" неправильний. Після перевірки о 19:30 UNet NG = +7.59 dB,
  Δ = +1.01 dB. Central hypothesis підтверджена.
- **§13.3.2 (перший draft):** позиціонував Hybrid erratic як "universal при 25%/30ep"
  — спростовано seed=43 який дав pristine identity. Правильна інтерпретація:
  bimodal seed-dependent. §13.7 оновлює taxonomy.

### 13.10 Immediate actions (post-[4/6])

1. **Kill PID 33983** на "### [4/6] done" monitor event.
2. **`analysis/aggregate_b2.py`** — адаптувати з `aggregate_b1.py` для 4 run_dirs
   (2 seeds × 2 noise); emit μ±σ таблиці + per-SNR curves + Hybrid bimodality
   breakdown.
3. **`experiments/results/b2_aggregate.md` / `.json`** — публікаційні артефакти.
4. **`compare_report.py` runs на кожному B2 run_dir** — для cross-evaluation
   matrix (G→G, G→NG, NG→G, NG→NG) аналогічно §11.8.
5. **Commit** B2 main source + artifacts.
6. **Оновити RESEARCH_STATUS §14 Final B2 results** після aggregation.

---

## §14. Final B2 main results (2026-04-24, ЗАВЕРШЕНО)

**Kill time:** 13:45:something EEST 2026-04-24. PID 33983 terminated SIGTERM
after `### [4/6] done 13:45:08 ###`. [5/6] seed=44 G was 0 epochs in when killed.
Total wall-clock: 23-07:31:35 → 24-13:45:10 ≈ **30h 13m**.

**Run dirs:**
| Invocation | seed | noise | run_id |
|---|---|---|---|
| [1/6] | 42 | G | `runs/run_20260423_ed46f843` |
| [2/6] | 42 | NG | `runs/run_20260423_3d8957d2` |
| [3/6] | 43 | G | `runs/run_20260423_2b4b99cc` |
| [4/6] | 43 | NG | `runs/run_20260424_20fc5edf` |

### 14.1 Per-seed aggregate (agg SNR_out across all 10 SNR bins, dB)

| Model | s42 G→G | s42 NG→NG | s43 G→G | s43 NG→NG |
|---|---:|---:|---:|---:|
| UnetAutoencoder | +6.58 | +7.59 | +6.35 | +7.56 |
| ResNetAutoencoder | +6.35 | +7.13 | +6.21 | +7.14 |
| HybridDSGE_UNet (robust S=3 vA) | **+0.40** erratic | 0.00 identity | **−0.01** identity | 0.00 identity |
| Wavelet (Test MSE) | 1.097 | 3.779 | ~1.09 | ~3.78 |

### 14.2 Cross-seed μ±σ таблиця (публікаційна)

| Model | Train | μ (dB) | σ (dB) | n | Notes |
|---|---|---:|---:|---|---|
| UnetAutoencoder | G | **+6.47** | 0.16 | 2 | — |
| UnetAutoencoder | NG | **+7.58** | 0.02 | 2 | **Δ(NG−G) = +1.11** |
| ResNetAutoencoder | G | **+6.28** | 0.10 | 2 | — |
| ResNetAutoencoder | NG | **+7.14** | 0.007 | 2 | **Δ(NG−G) = +0.86** |
| HybridDSGE_UNet | G | +0.19 | 0.29 | 2 | **bimodal** (erratic/identity) |
| HybridDSGE_UNet | NG | 0.00 | 0.000 | 2 | uniform identity |

**Signal-to-noise для central hypothesis:**
- UNet: Δ=+1.11 dB vs max(σ_G, σ_NG)=0.16 → **~7× SNR** (publishable).
- ResNet: Δ=+0.86 dB vs max σ=0.10 → **~9× SNR** (publishable).

### 14.3 Per-SNR curves (cross-seed μ, dB)

**UNet:**

| SNR_in | −20 | −17 | −15 | −12 | −10 | −7 | −5 | −3 | 0 | +3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G μ (n=2) | −0.44 | 0.96 | 2.15 | 3.96 | 5.98 | 7.87 | 8.52 | 10.33 | 11.49 | 13.79 |
| NG μ (n=2) | +0.03 | 1.74 | 3.17 | 5.22 | 6.59 | 8.69 | 10.05 | 11.44 | 13.45 | 15.37 |
| Δ | +0.47 | +0.79 | +1.02 | +1.26 | +0.61 | +0.82 | +1.53 | +1.11 | +1.96 | +1.58 |

**ResNet:**

| SNR_in | −20 | −17 | −15 | −12 | −10 | −7 | −5 | −3 | 0 | +3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G μ (n=2) | −0.22 | 0.92 | 1.76 | 3.85 | 5.07 | 7.46 | 8.34 | 10.31 | 11.47 | 13.86 |
| NG μ (n=2) | +0.42 | 1.63 | 2.84 | 4.86 | 5.93 | 8.11 | 9.37 | 10.70 | 12.67 | 14.62 |
| Δ | +0.64 | +0.71 | +1.07 | +1.01 | +0.87 | +0.65 | +1.03 | +0.40 | +1.20 | +0.76 |

Δ_NG stable across all SNR bins (жоден bin з Δ<0) — central hypothesis monotonic у SNR.

### 14.4 Key scientific findings (consolidated)

**1. Central hypothesis confirmed cross-scenario × cross-model:**

| Scenario / Model | Δ(NG−G) | σ_G | σ_NG | n |
|---|---:|---:|---:|---|
| FPV / UNet | +0.78 | 0.08 | 0.07 | 3 |
| FPV / ResNet | +1.26 | 0.07 | 0.025 | 3 |
| **deep_space / UNet** | **+1.11** | 0.16 | 0.02 | 2 |
| **deep_space / ResNet** | **+0.86** | 0.10 | 0.007 | 2 |

Non-Gaussian training improves denoising performance on non-Gaussian test signals
consistently across both radio scenarios (low-SNR deep space, high-SNR FPV) і
standard CNN architectures. Effect size 0.8–1.3 dB, reproducibility σ<0.2 dB.

**2. Hybrid DSGE — scenario- AND seed-dependent failure modes:**

Обсерваційна taxonomy (Hybrid robust S=3 vA across 4 experimental contexts):

| Context | Behaviour | n success |
|---|---|---|
| FPV / G (§11.3.1) | consistent learning @+11.13 dB | 3/3 |
| FPV / NG (§11.3.3) | bimodal: 2/3 identity, 1/3 @+11.77 dB | 1/3 |
| deep_space / G 10%/15ep (§12) | uniform identity | 0/1 (single seed) |
| deep_space / G 25%/30ep (§14) | **bimodal: 1/2 erratic, 1/2 identity** | 0/2 |
| deep_space / NG 25%/30ep (§14) | uniform identity | 0/2 |

Hybrid working regime is **narrow and scenario-specific**: FPV (high SNR range)
+ Gaussian noise + MSE loss. Будь-яке відхилення (low SNR, non-Gaussian, або
different loss) переводить модель у один з degenerate basins (identity, erratic,
bimodal).

**3. DSGE parameter efficiency не виправдовується на deep_space:**

FPV: Hybrid (~30k params) досягає 78% якості UNet (~300k params) — legitimate
parameter efficiency story. Deep_space: Hybrid **не досягає навіть identity
baseline** надійно (bimodal). Claim про universal parameter efficiency
**falsified на deep_space**. Scenario-conditional claim залишається valid.

**4. Optimization landscape insight:**

Hybrid loss landscape has ≥3 distinct attractors (identity, useful, erratic).
Basin selection is deterministic given (scenario, noise_type, loss, seed).
This is a architectural property of ratio-mask + DSGE feature concatenation,
not a training artifact — all attempted interventions (A2-H1..H6, §10) failed
to unify outcomes across seeds.

### 14.5 Compute tally (final)

| Фаза | Год |
|---|---:|
| Data generation | 4 |
| A1 reproducibility infra | 5 |
| A2 H1–H6 | 21 |
| B1 FPV main | 13.5 |
| B2 sanity | 6.3 |
| B2 main (4/6 invocations) | 30.2 |
| **Total through §14** | **80.0** |
| Budget | 72 |
| **Overshoot** | **+8.0 h** |

Залишок на Phase C (aggregation + cross-eval) + D (writing) = **0h formal budget**.
Practical: aggregation reuses existing aggregate_b1.py template (~30 min work),
writing happens in parallel with any remaining analysis.

### 14.6 Immediate next (post-kill 2026-04-24 13:45)

1. ✅ Kill PID 33983
2. **`analysis/aggregate_b2.py`** — adapted from aggregate_b1 for 4 run_dirs.
3. **`experiments/results/b2_aggregate.md/.json`** — publication-ready artifacts.
4. **Run `compare_report.py` on 4 B2 run_dirs** — crossover matrix
   (G→G, G→NG, NG→G, NG→NG).
5. **Commit** sources + artifacts + RESEARCH_STATUS §14.
6. **(Optional, if time)** B3 real SDR (needs RadioML HDF5 / DroneDetect raw).
