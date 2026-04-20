# План подальших досліджень: систематична оцінка DSGE для нейромережевого знешумлення радіосигналів

## Context

**Позиціонування (уточнено 2026-04-20).** Це дослідження на **етапі верифікації концепції**: головна мета — **встановити факт**, чи застосування DSGE (розкладу в просторі Кунченка) приносить користь для задач нейромережевого знешумлення радіосигналів. Можливий результат — як позитивний ("працює в сценаріях X, Y"), так і **негативний чи змішаний** ("не дає переваги у сценаріях Z, обмеження W"). Обидва випадки є валідним науковим внеском для arXiv preprint.

**Цільова публікація.** arXiv preprint, англійською мовою. Жанр — empirical study / systematic evaluation, не methods paper з "переможним" ком месседжем.

**Сценарії (обидва входять у статтю):**
- **FPV telemetry:** SNR −5..+18 dB, 100k samples. Простіший; DSGE Hybrid (10k параметрів) уже показав 12.84 dB vs 14.47 dB U-Net (300k) → 30× parameter efficiency на gaussian training.
- **Deep space:** SNR −20..0 dB, 400k samples. Складніший; DSGE на 5% даних дає ~0 dB, Transformer домінує (+1.40 dB).
- **Real SDR data (нове, кандидати з Фази 0):** валідація висновків на реальних записах (RadioML 2018, MIT RF Challenge, DroneDetect).

**Compute budget:** <72 год сумарно на M3 Max CPU (MPS у 13× повільніше для STFT).

**Що вже зроблено (2026-04-20):**
- DSGE виправлено 2026-04-18: коректна реалізація `F·K = B` з Тіхоновською регуляризацією (`models/dsge_layer.py:163-200`).
- FPV baseline: DSGE (gaussian training) дає parameter-efficient результат (12.84 dB / 10k params).
- Deep_space baseline (5% даних): Transformer найкращий, DSGE ≈ 0 dB.
- DSGE + **non-Gaussian training** = 0 dB на обох сценаріях — **систематичне спостереження**, яке треба дослідити.
- Reproducibility тест (10 сідів) показав, що SNR gains порядку 0.01 dB у 5% режимі в межах шуму.
- Старий article draft (`feature/dsge-hybrid-unet`/DSGE/Experiment_2_Full_Report.md) писаний під застарілу версію DSGE → не використовувати як baseline для номерів, тільки як source теорії.

**Зміна дослідницького питання.** Замість "як показати, що DSGE кращий", ставимо серію конкретних фальсифіковних тверджень:

| Q | Питання | Як перевіряємо |
|---|---------|----------------|
| Q1 | Чи забезпечує коректна DSGE-декомпозиція покращення vs baseline U-Net такого ж розміру? | A/B test: UNet(10k) vs DSGE-UNet(10k) на обох synthetic сценаріях |
| Q2 | Чи зберігається parameter efficiency (FPV результат) на deep_space? | Повний deep_space експеримент з DSGE vs U-Net різних розмірів |
| Q3 | Чому DSGE + non-Gaussian training дає ~0 dB? Фундаментальне обмеження чи виправна помилка? | Систематичний аналіз 4 гіпотез H1-H4 |
| Q4 | Який базис оптимальний для signal denoising (після корекції Kunchenko fit)? | Basis sweep із 3-5 сідів |
| Q5 | Чи DSGE краще інтегрується через Variant A (reconstruction + residual) vs Variant B (weighted basis)? | Ablation study |
| Q6 | Чи масштабування DSGE-UNet (width/depth) покращує результати? | Architectural ablation |
| Q7 | Як DSGE порівнюється з класичним wavelet-denoising baseline? | Cross-comparison |
| Q8 | Чи DSGE має прийнятну latency для real-time SDR processing? | CPU benchmark, batch=1 |
| **Q9** | **Чи узагальнюються висновки з synthetic datasets на реальні SDR-записи?** | **Еval на real dataset (з Фази 0)** |

---

## Research Plan

План у 5 фаз: **Фаза 0 — survey real datasets**, потім A/B/C/D. Кожна фаза дає відповіді на підмножину Q1-Q9 та артефакти (таблиці/фігури) для статті.

---

### Фаза 0 — Survey доступних real-SDR датасетів [~4 год compute, до експериментів]

Мета: знайти ≥1 публічний датасет з реальними SDR-записами для валідації висновків (Q9). **Без цього стаття має тільки synthetic evidence**, що послаблює наукову вагу.

#### 0.1. Огляд кандидатів (завершено у pre-planning WebSearch)

Нижче ранжовані кандидати за відповідністю до нашої задачі (знешумлення QPSK/BPSK при non-Gaussian noise). Джерела в `experiments/dataset_survey.md` (створити у Фазі 0.3).

**Tier 1 — найбільш релевантні:**

| Датасет | Чому підходить | Формат | Де взяти | Розмір |
|---------|---------------|--------|----------|--------|
| **RadioML 2018.01A** (DeepSig) | 24 модуляції включно з BPSK, QPSK; SNR −20..+30 dB (26 рівнів); real + over-the-air captured; de facto benchmark у RF ML | HDF5, complex 1024-sample frames | deepsig.ai/datasets, Kaggle | ~20 GB |
| **MIT RF Challenge ICASSP 2024** | Signal separation (SOI + interference); нестаціонарний non-Gaussian контекст — точно наш use case; добре документований | NumPy, complex baseband | github.com/RFChallenge/icassp2024rfchallenge | ~10 GB |
| **CSPB.ML.2018R2** | BPSK, QPSK + pulse shaping roll-off, freq offset, variable SNR; довші фрейми (32768 IQ) | Custom, IQ | academic (через автори) | ~15 GB |

**Tier 2 — потенційні для FPV scenario (drone/telemetry):**

| Датасет | Чому підходить | Формат | Де взяти |
|---------|---------------|--------|----------|
| **DroneDetect** (IEEE DataPort) | Real UAV RF recordings через BladeRF SDR + GNURadio; 2.4 GHz ISM band (FPV telemetry) | SigMF / IQ | ieee-dataport.org |
| **DroneRF** (Mendeley/al-sad) | 227 segments, 3 drones, різні режими польоту; real background RF | MATLAB / CSV | al-sad.github.io/DroneRF |
| **AirID** (GENESYS Lab) | 4 USRP B200mini на DJI M100 UAV, over-the-air + IQ imbalance | IQ binary | genesys-lab.org/airid |

**Tier 3 — noise-only datasets (для realistic noise augmentation):**

| Датасет | Чому підходить |
|---------|---------------|
| **NGGAN PLC noise dataset** | Real impulsive noise from commercial NB-PLC modem; valid model для non-Gaussian |

**Tier 4 — не підходять напряму, але згадати:**
- RadioML 2016 (тільки synthetic, deprecated by DeepSig themselves)
- TorchSig (synthetic generator, не dataset)
- NASA DSN — немає публічного dataset для denoising
- MIGOU-MOD — IoT, інший domain

#### 0.2. Вибір цільових датасетів для статті

**Рекомендація (для схвалення юзером):**
- **Primary:** RadioML 2018.01A — універсальний, цитується в ~400+ статтях, правильний baseline для QPSK/BPSK denoising, підтримує фільтрацію по SNR.
- **Secondary (для FPV story):** DroneDetect — real drone telemetry, 2.4 GHz, подібно до нашого FPV synthetic scenario.
- **Bonus (якщо compute дозволить):** MIT RF Challenge — сильне доповнення, але задача signal separation ≠ signal denoising, тому адаптація потребує дизайн-зусиль.

#### 0.3. Data feasibility check (~4 год)

**Завдання:**
1. Завантажити 1 sample file з RadioML 2018 (~2 GB з одного SNR рівня).
2. Написати adapter `data_generation/load_radioml2018.py`:
   - Входи: HDF5 → out: (clean_signals, noisy_signals) у тому ж форматі, що `data_generation/datasets/<name>/{clean_signals.npy, gaussian_signals.npy, non_gaussian_signals.npy}`.
   - ❗Виклик: RadioML не має "clean" і "noisy" pair — тільки noisy signals з відомим SNR. Треба або:
     - (a) Використовувати high-SNR (+18..+30 dB) samples як "clean reference" + low-SNR як noisy;
     - (b) Denoise task реформулювати як "improve SNR by X dB"; використовувати input SNR як target reference.
   - Формат виходу: 1024-sample frames, compatible з існуючим pipeline.
3. Завантажити 1 sample file з DroneDetect (SigMF).
4. Написати adapter `data_generation/load_dronedetect.py`.
5. Протестувати preprocessing на 100 сигналах: STFT shapes, amplitude ranges, NaN checks.
6. Створити документ `experiments/dataset_survey.md` з результатами та рекомендаціями.

**Критерій успіху фази 0:**
- ≥1 real dataset адаптовано до нашого pipeline
- Перший preprocessed file `data_generation/datasets/radioml2018_subset/` створено
- Sanity test: `python data_generation/evaluate_dataset.py data_generation/datasets/radioml2018_subset/` не падає

**Файли:**
- `data_generation/load_radioml2018.py` (новий)
- `data_generation/load_dronedetect.py` (новий)
- `experiments/dataset_survey.md` (новий, з таблицями Tier 1-4, посиланнями, pros/cons, acceptance criteria)
- `data_generation/datasets/radioml2018_subset/` (cached subset)

---

### Фаза A — Інфраструктура та систематичний аналіз NG-failure [~22 год compute]

Мета фази: закласти фундамент для надійного експериментування та дати відповідь на Q3.

#### A1. Reproducibility infrastructure (~5 год) → Q2, Q3, Q4

**Зараз:** multiprocessing fork на macOS викликає випадкові варіації, sedy не закріплені.

**План:**
- `torch.manual_seed(seed)`, `np.random.seed(seed)`, `random.seed(seed)`, `torch.backends.cudnn.deterministic=True` у `__init__` всіх trainer-ів.
- `multiprocessing.set_start_method('spawn', force=True)` у `experiments/run_dsge_sweep.py`.
- Параметр `--seed` у всіх trainer CLI; всі sweep-скрипти ітеруют 3 сіди (42, 43, 44).
- Критерій: 2 запуски з `seed=42` → σ(val_SNR) < 0.02 dB.

**Файли:** `train/training_uae.py`, `training_resnet.py`, `training_vae.py`, `training_transformer.py`, `training_hybrid.py`, `wavelet_grid_search.py`, `train_all.py`, `experiments/run_dsge_sweep.py`.

#### A2. Систематичний аналіз DSGE + non-Gaussian training failure (~17 год) → Q3

**Фокус:** це **ключове питання цього етапу**. Мета не "виправити" а **зрозуміти**. Всі 4 гіпотези тестуємо, навіть якщо ранні не дають позитивних результатів.

**H1: Loss incompatibility (5 год).**
- SmoothL1(β=0.02) занадто мала beta → ефективно L1 для NG, градієнти дрібні.
- Тести: {MSE, SmoothL1 β=0.02, β=0.1, β=1.0, Huber δ=1.0, Charbonnier ε=1e-3}
- На FPV 25% + deep_space 10% даних, 3 сіди.
- Артефакт: таблиця Loss × Scenario × val_SNR.
- Висновок: чи SmoothL1 β=0.02 — проблема.

**H2: Ratio mask collapse (3 год).**
- Логувати distribution `sigmoid(output)` впродовж training: min/max/mean/histogram.
- Hypothesis: heavy tails NG → mask → 0.
- Alt: `additive mask` (`x_hat = noisy + residual_pred`).
- Файл: `models/hybrid_unet.py` → `mask_type ∈ {ratio, additive}`.
- Критерій успіху/спростування: зміна mask_type переключає поведінку NG training.

**H3: Per-channel normalization robustness (3 год).**
- Поточна normalization (`training_hybrid.py:292-323`) робить mean/std, чутливі до NG outliers.
- Замінити на MAD-based з clipping на 99-ом перцентилі.
- Критерій: порівняти distributions вхідних каналів до/після.

**H4: Single global generating element insufficient (6 год).**
- Теоретичний аргумент: Kunchenko space передбачає homogeneous source; усереднений clean signal за всіма SNR → subadequate GE.
- Реалізація: `fit_class_specific()` — SNR-bucketed (3-5 bins), окремі K per bin.
- При inference: blind SNR estimation → вибір bin.
- Файл: `models/dsge_layer.py` (новий метод), `train/training_hybrid.py` (параметр `--dsge-snr-bins`).
- Критерій: SNR-bucketed DSGE > single-GE DSGE на NG тесті.

**Важливо:** H1-H4 не мають бути "success or die". Будь-який результат (навіть "жодна гіпотеза не вирішує") — це **наукова знахідка**, яка попадає в Discussion.

**Артефакт A2:** новий файл `experiments/ng_failure_analysis.md` з таблицями для кожної H + коротким висновком. Використовується як draft для Results §9 у статті.

---

### Фаза B — Основні експерименти на обох synthetic сценаріях + real validation [~32 год compute]

Мета: дати відповіді на Q1, Q2, Q7, Q9.

#### B1. Повний FPV експеримент (~10 год) → Q1, Q2, Q7

**Масштаб:** 100% FPV даних (100k samples), 3 сіди.

**Моделі:** U-Net, ResNet, DSGE-UNet (варіант з Фази A2/A3 basis reconciliation), Transformer (для порівняння), Wavelet (class baseline).

**Протокол:** `experiments/fpv_main.sh` (новий, на базі існуючого `experiments/fpv_experiment.sh`):
```bash
for seed in 42 43 44; do
    python train/train_all.py \
        --dataset data_generation/datasets/fpv \
        --models unet,resnet,hybrid,transformer,wavelet \
        --noise-types gaussian,non_gaussian \
        --epochs 50 --seed $seed \
        --run-id fpv_main_seed$seed
done
```

#### B2. Deep_space експеримент (~20 год) → Q1, Q2

**Масштаб:** 50% даних (200k samples), 3 сіди. Sanity check на 10% seed 42 перед full run.

**Файл:** `experiments/deep_space_main.sh` (новий).

#### B3. Real SDR validation (~2 год) → Q9

**Масштаб:** обмежений через compute. Використовуємо:
- Pre-trained best models з B1 (FPV-trained) та B2 (deep_space-trained).
- **Zero-shot evaluation** на real dataset (RadioML 2018 subset, або DroneDetect) без дотренування.
- Якщо compute дозволить — fine-tune на 25% real dataset.

**Протокол:** `experiments/real_sdr_validation.sh` (новий):
1. Load pretrained models з B1/B2.
2. Evaluate на real dataset (з Фази 0).
3. Compute metrics (SNR improvement на всіх доступних SNR bins).
4. Generate report `run_real_sdr_validation_<ts>.md`.

**Критерій:** DSGE-UNet показує узгоджений SNR improvement pattern на real data, як на synthetic. Якщо патерн драматично інший — обговорити в Discussion.

#### B4. Per-SNR криві (cross-model) → central figures

Розширити `train/snr_curve.py` до `plot_cross_model_curves(run_dir, figure_layout='2x2_train×test')` — 4 panels × 2 scenarios + 1 real-data panel.

#### B5. Generalization matrix → table

4×4 (для кожної з моделей): train-noise × test-noise, val_SNR μ ± σ. + додаткова колонка real SDR zero-shot.

---

### Фаза C — Абляції [~12 год compute]

Мета: дати відповіді на Q4, Q5, Q6.

#### C1. Basis reconciliation (~5 год) → Q4

**Протокол:**
- Datasets: FPV 25%, deep_space 25%.
- Basis: {fractional, polynomial, robust, trigonometric} × S=3 × Variant A × 3 сіди.
- Loss: найкраща з A2 (для NG окремо).
- Файл: `experiments/basis_reconciliation.sh`.

**Артефакт:** таблиця ranking × scenario × noise. Discussion про розбіжність з Experiment_2_Full_Report.md.

#### C2. Variant A vs Variant B (~2 год) → Q5

**Протокол:**
- FPV 25%, найкращий basis з C1, Variant ∈ {A, B}, 3 сіди, 2 noise types.

#### C3. DSGE-UNet architectural scaling (~5 год) → Q6

**Протокол:**
- Dataset: FPV 100% (з Фази B1).
- Варіації: base_channels ∈ {8, 16, 32}; depth (# encoder stages) ∈ {2, 3}; DSGE order S ∈ {2, 3, 4}.
- 18 configs × 1 seed (42) → ~5 год.
- Файл: `experiments/dsge_architecture_ablation.sh`.

---

### Фаза D — Аналіз, figures, написання [~4 год compute + writing]

#### D1. Statistical significance → Q1-Q2 підтвердження (~1 год)

- Paired t-test + 95% bootstrap CI для:
  - `U-Net(G) vs DSGE-UNet(G)` на обох synthetic сценаріях + real data, на G та NG test
  - `X(G) vs X(NG)` для кожної моделі X на NG test
  - `corrected DSGE vs archived old DSGE`
- Файл: `analysis/stats.py` (новий).

#### D2. Parameter efficiency analysis → головна фігура статті

- Scatter: x=log(#params), y=val_SNR, різні маркери для моделей, кольори для сценаріїв (FPV, deep_space, real SDR), error bars (3 сіди).
- Файл: `train/compare_report.py` → `plot_parameter_efficiency()`.

#### D3. Latency benchmark (~2 год) → Q8

- Wall-clock на CPU, batch=1, 1000 blocks.
- Метрики: p50, p95, p99 latency (ms per 125-ms block).
- Файл: `inference/benchmark_latency.py` (новий).

#### D4. Write-up → arXiv preprint

**Формат:** LaTeX, arXiv-friendly.

**Структура статті:**

1. **Abstract** (200-250 слів): проблема, systematic evaluation, real+synthetic evidence, key findings.
2. **Introduction** (1 стор): background; positioning as empirical study; contributions as **findings**.
3. **Related Work** (1 стор): classical, neural, Kunchenko framework (HAR paper, PMM3), RF datasets.
4. **Theoretical Background: DSGE** (1.5 стор).
5. **Method: Hybrid DSGE-UNet** (1.5 стор).
6. **Experimental Setup** (1 стор): synthetic (FPV, deep_space) + real (RadioML/DroneDetect).
7. **Results — Main Comparison** (2-3 стор).
8. **Results — Ablations** (1-2 стор).
9. **Results — NG Training Analysis** (1-1.5 стор): центральний систематичний аналіз.
10. **Results — Real SDR Validation** (0.5-1 стор): zero-shot transfer study.
11. **Latency** (0.5 стор).
12. **Discussion** (1 стор).
13. **Conclusion** (0.3 стор).
14. **Bibliography** (30-50 refs).

**Файли:**
- Нова гілка `feature/paper-writeup`.
- `paper/main.tex`, `paper/sections/*.tex`, `paper/figures/`, `paper/tables/`, `paper/refs.bib`
- `paper/reproduce.sh`

---

## Критичні файли до редагування/створення

| Файл | Призначення | Фаза | Reuse |
|------|-------------|------|-------|
| `data_generation/load_radioml2018.py` (новий) | Adapter для RadioML 2018 | 0.3 | generation.py output format |
| `data_generation/load_dronedetect.py` (новий) | Adapter для DroneDetect (SigMF) | 0.3 | same |
| `experiments/dataset_survey.md` (новий) | Документ оглядування датасетів | 0.1-0.2 | - |
| Всі `training_*.py` + `train_all.py` | `--seed` parameter, determinism | A1 | existing |
| `experiments/run_dsge_sweep.py` | `spawn` start method | A1 | existing |
| `train/losses.py` | +Huber, Charbonnier, параметри SmoothL1 | A2/H1 | existing |
| `train/training_hybrid.py` | `--loss-fn`, `--dsge-snr-bins`, robust norm | A2/H1-H4 | existing |
| `models/dsge_layer.py` | `fit_class_specific()` | A2/H4 | existing `fit()` |
| `models/hybrid_unet.py` | `mask_type={ratio, additive}` | A2/H2 | existing `forward` |
| `experiments/fpv_main.sh` (новий) | FPV full | B1 | fpv_experiment.sh |
| `experiments/deep_space_main.sh` (новий) | deep_space main | B2 | template |
| `experiments/real_sdr_validation.sh` (новий) | Real data zero-shot eval | B3 | template |
| `train/snr_curve.py` | `plot_cross_model_curves` | B4 | existing |
| `train/compare_report.py` | Cross-seed agregation, parameter efficiency plot | B5/D2 | existing ~50 KB |
| `experiments/basis_reconciliation.sh` (новий) | Basis × S × Variant | C1 | template |
| `experiments/dsge_architecture_ablation.sh` (новий) | Width × depth × S | C3 | new |
| `analysis/stats.py` (новий) | Paired t-test, bootstrap CI | D1 | scipy/pandas |
| `inference/benchmark_latency.py` (новий) | CPU wall-clock | D3 | inference/inference_*.py |
| `experiments/ng_failure_analysis.md` (новий) | H1-H4 analysis doc | A2 | - |
| `paper/main.tex` + sections (нові, нова гілка) | Manuscript | D4 | arXiv template |

---

## Повторне використання існуючого коду

- `DSGEFeatureExtractor.fit()` (models/dsge_layer.py:163-200) — **не переписувати**.
- `DSGEFeatureExtractor.save_state()/load_state()` — для reproducibility.
- `metrics.py` (MSE/MAE/RMSE/SNR) — готові.
- `train/compare_report.py` — розширити, не переписати.
- `train/snr_curve.py` — розширити.
- `experiments/run_dsge_sweep.py`, `experiments/fpv_experiment.sh` — шаблони.
- `data_generation/evaluate_dataset.py` — cumulant-based dataset quality analysis, застосувати до real data після adapter.
- `DSGE/Experiment_2_Full_Report.md` §2 — source для Theoretical Background.
- HAR paper PDF у `DSGE/` — bibliographic reference.

---

## Verification Plan

**Per-фаза:**

- **Фаза 0 (datasets):** `python data_generation/load_radioml2018.py --snr-bins 10,14,18 --output radioml2018_subset` → не падає; `evaluate_dataset.py` показує статистики схожі до synthetic FPV.
- **A1 (infra):** 2 запуски з `--seed 42` → σ(val_SNR) < 0.02 dB.
- **A2 (NG analysis):** `experiments/ng_failure_analysis.md` з таблицями H1-H4, кожна з 3 сідами.
- **B1/B2 (main):** `comparison_report_cross_seed.md` з μ ± σ.
- **B3 (real SDR):** `run_real_sdr_validation_*.md` з SNR improvement table; consistent patterns з synthetic.
- **B4 (curves):** PDF × 2 (FPV, deep_space) + real data curve, vector.
- **B5 (heatmap):** 4×4 matrix + colour PDF.
- **C1-C3 (ablations):** JSON + markdown tables.
- **D1 (stats):** `analysis/stats.py` → p-values table.
- **D2 (efficiency):** scatter PDF з Pareto frontier.
- **D3 (latency):** p50/p95/p99 table.
- **D4 (paper):** `latexmk -pdf main.tex` без errors; arXiv upload тест.

**Final acceptance checklist:**
- [ ] Всі results tables: μ ± σ по 3 сідів на основних експериментах.
- [ ] Фігури: vector PDF, ≥ 10pt font.
- [ ] **Real SDR validation: хоча б zero-shot результат на 1 dataset** (Q9).
- [ ] Reproducibility: `paper/reproduce.sh`.
- [ ] Bibliography: HAR paper, Kunchenko works, DnCNN, Wiener, wavelet (Donoho), Huber, **RadioML dataset paper (O'Shea 2018)**, **MIT RF Challenge 2024** якщо використано.
- [ ] NG-training failure: розв'язано або чесно документовано.
- [ ] Discussion: розбіжність з старим draft, сценарій-залежна придатність DSGE.

---

## Compute budget breakdown (72 год)

| Фаза | Завдання | Орієнтовні годи |
|------|----------|----------------|
| 0 | Dataset survey + adapter development | 4 |
| **Фаза 0 total** | | **4** |
| A1 | Reproducibility infra | 5 |
| A2-H1 | Loss sweep | 5 |
| A2-H2 | Mask collapse | 3 |
| A2-H3 | Normalization robust | 3 |
| A2-H4 | Class-specific GE | 6 |
| **Фаза A total** | | **22** |
| B1 | FPV main | 10 |
| B2 | deep_space main | 20 |
| B3 | Real SDR validation | 2 |
| B4 | Per-SNR curves | 1 |
| B5 | Generalization heatmap | 0.5 |
| **Фаза B total** | | **33.5** |
| C1 | Basis reconciliation | 5 |
| C2 | Variant A/B | 2 |
| C3 | Architectural ablation | 5 |
| **Фаза C total** | | **12** |
| D1 | Stats | 0.5 |
| D2 | Efficiency figure | 0.5 |
| D3 | Latency | 2 |
| D4 | Writing | 0 (не compute) |
| **Фаза D total** | | **3** |
| **GRAND TOTAL** | | **74.5 год (перевищення 2.5)** |

**Стратегія балансування:**
- Якщо Фаза 0 зайняла >4 год → скоротити B2 до 25% даних (economia 10 год).
- Якщо B2 sanity check на 10% показує flat SNR gains → залишитися на 25% (economia 10 год).

---

## Пріоритети та ризики

### Must-have (для валідного arXiv submission)

1. **Фаза 0: ≥1 real dataset adapted** (Q9 валідація — критично для статті).
2. **A1 reproducibility**.
3. **A2 systematic NG analysis**.
4. **B1 FPV main**.
5. **B2 deep_space main**.
6. **B3 real SDR zero-shot validation**.
7. **D1 statistics**.
8. **D4 manuscript**.

### Nice-to-have

- Фаза 0: 2+ real datasets.
- B4, B5, C1, C2, C3, D2, D3.

### Optional (якщо лишається час)

- MIT RF Challenge fine-tuning.
- Real SDR fine-tuning (не тільки zero-shot).
- Additional архітектурні ablations.

### Ризики та mitigation

**Ризик 1: Real SDR dataset не має "clean reference".**
- RadioML містить тільки noisy signals at different SNRs, без explicit clean pairs.
- Mitigation (pre-планований):
  - (a) Використати highest SNR (+30 dB) як proxy clean reference; validate на lower-SNR версіях same frame.
  - (b) Реформулювати task як SNR improvement: input (SNR=X dB) → output (SNR=Y > X dB).
  - (c) Уточнити в adapter: документувати це як фундаментальне обмеження real-world evaluation у Discussion.

**Ризик 2: A2 показує, що NG failure фундаментальний.**
- Mitigation: це заплановано. Розділ 9 "Results — NG Training Analysis".

**Ризик 3: DSGE на deep_space 50% так само ~0 dB.**
- Mitigation: науковий результат — сценарій-залежна придатність. Strong meta-message.

**Ризик 4: Compute overrun (74.5 > 72 budget).**
- Pre-committed rollback:
  - Якщо after A — >30 год, B2 до 25% даних.
  - Якщо все ще тісно — C3 виключити.
  - Last resort — 2 сіди замість 3 в B1/B2.

**Ризик 5: Real SDR results суттєво відрізняються від synthetic.**
- Може підтверджувати, що synthetic моделі занадто спрощені.
- Mitigation: це сильний finding для Discussion, не weakness.

**Ризик 6: Basis reconciliation (C1) показує, що стара реалізація випадково давала кращі числа.**
- Mitigation: Discussion про corrected theory → changed empirical ranking.

**Ризик 7: FPV 100% показує, що 12.84 dB був outlier (без reproducibility).**
- Mitigation: finding про sample drift, seed artifact; переглянути наратив.

---

## Послідовність виконання (timeline)

```
Day 0-1:  Фаза 0 (dataset survey + adapter) — 4 год compute + dev
Day 2:    A1 (infra) — 5 год
Day 3-4:  A2-H1 (losses)
Day 5:    A2-H2 (mask) + A2-H3 (normalization)
Day 6-7:  A2-H4 (class-specific GE) — найбільший dev effort
Day 8:    Consolidate ng_failure_analysis.md
Day 9-11: Фаза B1 (FPV main, 3 seeds)
Day 12-17: Фаза B2 (deep_space main, 3 seeds на 50%)
Day 18:   Фаза B3 (real SDR validation)
Day 19-21: Фаза C1-C3
Day 22:   Фаза D1-D3
Day 23+:  D4 writing
```

**Контрольні точки:**
- **End of Day 1 (post-0):** checkpoint dataset survey. Якщо жоден real dataset не адаптовано — переглянути скоуп (можливо, обмежитися synthetic story з документуванням обмеження).
- **End of Day 8 (post-A):** прийняти рішення, чи вирішено NG failure.
- **End of Day 11 (post-B1):** FPV parameter efficiency re-перевірено.
- **End of Day 17 (post-B2):** всі synthetic результати зібрані.
- **End of Day 18:** real SDR checkpoint.

---

## Зв'язок з user's трьома зауваженнями

1. **FPV повернуто** → B1 (full 100%), C1-C3 (partial), D3 (latency).
2. **Мета — визначити працездатність DSGE** → 9 конкретних Q1-Q9 питань; negative/mixed результати є науковим внеском.
3. **Survey real SDR datasets** → нова Фаза 0 (4 год compute) + B3 (real validation) + Q9 у списку питань. Tier 1 кандидати: RadioML 2018.01A, MIT RF Challenge, CSPB.ML.2018R2. Tier 2 (FPV): DroneDetect, DroneRF, AirID.
