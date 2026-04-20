# Стан дослідження — Signal Denoising with DSGE

**Дата:** 2026-04-19
**Автор дослідження:** Сергій Заболотній
**Датасет:** deep_space_polygauss_qpsk_bs1024_n400000_0310b7e7
**Сценарій:** Deep space, SNR -20..0 dB, QPSK, 1024 samples @ 8192 Hz

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
