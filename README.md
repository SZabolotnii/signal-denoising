# Signal Denoising — Non-Gaussian Noise Study

## Мета дослідження

Стандартний підхід у задачах знешумлення сигналів: тренувати нейронну мережу
на даних з гауссовим шумом (AWGN). Це математично зручно, але у реальних
радіосистемах шум часто **негауссовий**: суміш різних процесів з важкими
хвостами, імпульсними сплесками, нестаціонарними характеристиками.

**Центральна гіпотеза:** нейронна мережа, навчена на негауссовому шумі,
буде краще знешумлювати реальні сигнали порівняно з мережею тієї ж архітектури,
навченою на гауссовому шумі.

**Методологія:**

```
Датасет A (AWGN)          -> Архітектура X -> NN_A -+
                                                      +-> Порівняння метрик на test set
Датасет B (Non-Gaussian)  -> Архітектура X -> NN_B -+
```

Обидві мережі мають **однакову архітектуру** і тренуються на даних однакового
обсягу при однакових SNR. Єдина змінна — тип шуму в тренувальних даних.
Якщо NN_B стабільно краще, це свідчить про цінність відповідного
моделювання шуму. Якщо різниця несуттєва, AWGN-апроксимація виправдана.

Якщо гіпотеза підтверджується, наступний крок — дослідження архітектурних
оптимізацій під негауссові шуми.

---

## Точка інтеграції в реальній системі

```
Антена -> LNA -> Mixer -> [IF фільтр] -> ADC -> DSP
  RF (МГц/ГГц)  |                               ^
            Downconversion                Тут працює NN
```

Ми працюємо з **baseband-сигналом після downconversion**: цифровим потоком
після АЦП, де реальна несуча (МГц/ГГц) вже перенесена до нуля або низької IF.
Це природна точка інтеграції для будь-якого SDR-приймача: всі наступні алгоритми
(демодуляція, декодування) отримують вже знешумлений сигнал.

**Real-valued vs I/Q:** SDR-приймач видає комплексний I/Q-сигнал.
Для цього дослідження обрано **real-valued** представлення:
гіпотеза перевіряється незалежно від форми представлення сигналу,
а real-valued зменшує кількість архітектурних змінних і дозволяє
сфокусуватись на ефекті розподілу шуму. Перехід до I/Q — природний
наступний крок після підтвердження гіпотези.

**Block-based processing:** нейронна мережа обробляє фіксовані блоки
по 256 семплів. При частоті дискретизації 8192 Гц це відповідає
латентності ~31 мс, прийнятній для більшості застосувань реального часу.
Fully convolutional архітектура дозволяє застосовувати модель до блоків
довільної довжини без переробки при деплої.

---

## Структура проекту

```
signal-denoising/
├── data_generation/
│   ├── generation.py          <- генератор синтетичних датасетів
│   ├── evaluate_dataset.py    <- оцінка якості датасету через аналіз кумулянтів
│   └── datasets/              <- згенеровані датасети (gitignored)
│       └── deep_space_..._<uid>/
│           ├── dataset_config.json
│           ├── train/         <- clean_signals.npy, gaussian_signals.npy, non_gaussian_signals.npy
│           ├── test/          <- per-SNR test files: test_m10dB_gaussian.npy, ...
│           └── weights/       <- ваги моделей (створюється автоматично)
│               ├── runs/
│               │   ├── UnetAutoencoder_gaussian/
│               │   │   ├── model_best.pth
│               │   │   └── figures/
│               │   │       ├── training_curves.png
│               │   │       └── snr_curve.png
│               │   ├── UnetAutoencoder_non_gaussian/
│               │   ├── HybridDSGE_UNet_fractional_S3_gaussian/
│               │   │   ├── model_best.pth
│               │   │   ├── dsge_state.npz
│               │   │   └── figures/
│               │   └── Wavelet_gaussian/
│               │       └── best_params.json
│               ├── figures/           <- порівняльні графіки (compare_report.py)
│               │   ├── fig1_snr_heatmap.png
│               │   ├── fig2_combined_snr_curves.png
│               │   ├── fig3_per_model_comparison.png
│               │   ├── fig4_dsge_scatter.png
│               │   └── fig5_example_denoising.png
│               ├── comparison_report_<ts>.md
│               ├── comparison_report_<ts>_uk.md
│               ├── comparison_data_<ts>.csv
│               └── training_report_<ts>.md
├── models/                    <- архітектури моделей
│   ├── autoencoder_unet.py
│   ├── autoencoder_resnet.py
│   ├── autoencoder_vae.py
│   ├── time_series_trasformer.py
│   ├── hybrid_unet.py
│   ├── dsge_layer.py
│   └── wavelet.py
├── train/                     <- скрипти тренування та аналізу
│   ├── train_all.py           <- тренування всіх моделей на обох типах шуму
│   ├── compare_report.py      <- крос-оцінка + порівняльний звіт
│   ├── sweep_hybrid.py        <- перебір архітектур HybridDSGE_UNet
│   ├── training_uae.py
│   ├── training_resnet.py
│   ├── training_vae.py
│   ├── training_transformer.py
│   ├── training_hybrid.py
│   ├── wavelet_grid_search.py
│   └── snr_curve.py           <- утиліти: per-SNR eval, криві SNR, тренувальні криві
├── inference/
│   └── inference_hybrid.py    <- inference для гібридної моделі
├── metrics.py
└── README.md
```

---

## Датасет

Синтетичний датасет моделює **цифровий baseband-сигнал після downconversion**
для двох сценаріїв:

- **deep_space**: умови дальнього космічного зв'язку, BPSK/QPSK, SNR −20..0 дБ
- **fpv_telemetry**: FPV-телеметрія / ELRS-подібні системи, CPFSK/GFSK/QPSK, SNR −5..+15 дБ

Для кожного прикладу генерується три версії: чистий сигнал, з AWGN і з негауссовим шумом.
Основний тип негауссового шуму — **нестаціонарний полігауссовий**: суміш гауссіан,
параметри якої плавно дрейфують через процеси Орнштейна-Уленбека, що імітує
реальне мінливе радіосередовище.

Детальна документація: [`data_generation/README.md`](data_generation/README.md)

```bash
# Генерація датасету (дефолт: deep_space, polygauss_nonstationary, bpsk)
python data_generation/generation.py

# Оцінка якості
python data_generation/evaluate_dataset.py data_generation/datasets/<run_folder>/
```

---

## Налаштування середовища

```bash
pip install -r requirements.txt
```

Для логування в Weights & Biases:

1. Створи акаунт на [wandb.ai](https://wandb.ai) (безкоштовно)
2. Скопіюй API ключ зі [wandb.ai/settings](https://wandb.ai/settings)
3. Додай ключ у `.env`:

```bash
cp .env.template .env
# відредагуйте .env — вставте WANDB_API_KEY=твій_ключ
```

4. Передай назву проєкту при запуску через `--wandb-project <назва>`

`.env` вже в `.gitignore` і ніколи не потрапить у репозиторій.
Якщо `.env` відсутній або `--wandb-project` не вказано — скрипти працюють
без W&B, просто не логуючи метрики.

---

## Швидкий старт

```bash
DATASET=data_generation/datasets/<назва_датасету>

# 1. Тренування всіх моделей на обох типах шуму (дефолтні параметри)
python train/train_all.py --dataset $DATASET

# 2. Порівняльний звіт із крос-оцінкою та графіками
python train/compare_report.py --dataset $DATASET
```

Готово. Звіти та графіки збережено в `<dataset>/weights/`.

---

## Тренування моделей

### `train_all.py` — тренування всіх моделей

Тренує всі (або вибрані) моделі на обох типах шуму. Кожна комбінація
(модель × тип шуму) дає окремий файл ваг.

```bash
DATASET=data_generation/datasets/<назва_датасету>

# Всі моделі, обидва типи шуму (за замовчуванням)
python train/train_all.py --dataset $DATASET

# Тільки певні моделі
python train/train_all.py --dataset $DATASET --models unet,resnet,wavelet

# Тільки один тип шуму
python train/train_all.py --dataset $DATASET --noise-types non_gaussian

# З W&B логуванням
python train/train_all.py --dataset $DATASET --wandb-project signal-denoising
```

**Аргументи:**

| Аргумент | За замовчуванням | Опис |
|---|---|---|
| `--dataset` | — | Шлях до папки датасету (обов'язковий) |
| `--noise-types` | `all` | `gaussian`, `non_gaussian` або `all` |
| `--models` | `all` | Через кому або `all`: `unet,resnet,vae,transformer,wavelet,hybrid` |
| `--epochs` | `50` | Кількість епох |
| `--batch-size` | `256` | Розмір батча |
| `--lr` | `1e-4` | Learning rate |
| `--nperseg` | `32` | Розмір вікна STFT (для спектральних моделей) |
| `--seed` | `42` | Random seed |
| `--wandb-project` | `""` | W&B project (порожньо = вимкнено) |

Після тренування в `<dataset>/weights/` з'являться:

```
weights/
├── runs/
│   ├── UnetAutoencoder_gaussian/
│   │   ├── model_best.pth
│   │   └── figures/
│   │       ├── training_curves.png   <- loss + val_SNR по епохах
│   │       └── snr_curve.png
│   ├── UnetAutoencoder_non_gaussian/
│   ├── ResNetAutoencoder_gaussian/
│   ├── ResNetAutoencoder_non_gaussian/
│   ├── SpectrogramVAE_gaussian/
│   ├── SpectrogramVAE_non_gaussian/
│   ├── TimeSeriesTransformer_gaussian/
│   ├── TimeSeriesTransformer_non_gaussian/
│   ├── HybridDSGE_UNet_fractional_S3_gaussian/
│   │   ├── model_best.pth
│   │   └── dsge_state.npz
│   ├── HybridDSGE_UNet_fractional_S3_non_gaussian/
│   ├── Wavelet_gaussian/
│   │   └── best_params.json
│   └── Wavelet_non_gaussian/
└── training_report_<timestamp>.md
```

Кожна підпапка в `runs/` перезаписується при повторному запуску — зберігається
лише найкраща модель за `val_SNR` для кожної комбінації (модель × тип шуму).

### Тренування окремих моделей

```bash
python train/training_uae.py         --dataset $DATASET --noise-type non_gaussian --epochs 30
python train/training_resnet.py      --dataset $DATASET --noise-type non_gaussian --epochs 50
python train/training_vae.py         --dataset $DATASET --noise-type non_gaussian --epochs 50
python train/training_transformer.py --dataset $DATASET --noise-type non_gaussian --epochs 50
python train/wavelet_grid_search.py  --dataset $DATASET --noise-type non_gaussian
python train/training_hybrid.py      --dataset $DATASET --noise-type non_gaussian --epochs 30
```

### HybridDSGE_UNet — перебір архітектур

Тестує всі комбінації базисних функцій і порядків полінома (13 конфігурацій):

```bash
python train/sweep_hybrid.py --dataset $DATASET --noise-types non_gaussian --epochs 30

# З W&B для порівняння в UI
python train/sweep_hybrid.py --dataset $DATASET --wandb-project signal-denoising
```

Додаткові параметри `training_hybrid.py`:

| Аргумент | За замовчуванням | Опис |
|---|---|---|
| `--dsge-order` | `3` | Кількість DSGE-каналів (S) |
| `--dsge-basis` | `fractional` | `fractional`, `polynomial`, `trigonometric`, `robust` |
| `--dsge-powers` | `0.5 1.5 2.0` | Степені/частоти через пробіл |
| `--lambda` | `0.01` | Коефіцієнт λ регуляризації Тіхонова |

---

## Порівняльний звіт

```bash
# Крос-оцінка всіх натренованих моделей + звіт
python train/compare_report.py --dataset $DATASET
```

Скрипт автоматично знаходить всі `.pth` і `.json` у `weights/`, оцінює кожну
модель на **обох** тестових наборах (gaussian і non_gaussian) та генерує:

| Файл | Опис |
|------|------|
| `comparison_report_<ts>.md` | Звіт англійською (з поясненнями) |
| `comparison_report_<ts>_uk.md` | Звіт українською |
| `comparison_data_<ts>.csv` | Плоска таблиця всіх метрик (model × train × test × SNR_in) |
| `comparison_data_<ts>.json` | Те саме у JSON |
| `figures/fig1_snr_heatmap.png` | Теплова карта SNR: всі комбінації train→test |
| `figures/fig2_combined_snr_curves.png` | Всі моделі на одному графіку (суцільний = G-train, штрих = NG-train) |
| `figures/fig3_per_model_comparison.png` | Для кожної моделі: вплив типу навчання на криву SNR |
| `figures/fig4_dsge_scatter.png` | Scatter архітектур DSGE: узагальнення по обох тестах |
| `figures/fig5_example_denoising.png` | Приклад знешумлення сигналу |

Крім того, для кожного навченого варіанту в `runs/<ModelName>_<noise_type>/figures/` зберігаються:
- `training_curves.png` — loss і val_SNR по епохах (для контролю оверфіту, не в основному звіті)
- `snr_curve.png` — крива SNR на тренувальному типі шуму

---

## Моделі

| Модель | Клас | Опис |
|--------|------|------|
| **U-Net** | `UnetAutoencoder` | STFT-маска, log-MAE + multi-res STFT loss |
| **ResNet** | `ResNetAutoencoder` | STFT autoencoder, HuberLoss |
| **VAE** | `SpectrogramVAE` | Варіаційний autoencoder на спектрограмі, HuberLoss + KL |
| **Transformer** | `TimeSeriesTransformer` | Time-domain, HuberLoss |
| **Wavelet** | `WaveletDenoising` | Порогова обробка вейвлет-коефіцієнтів, grid search |
| **HybridDSGE** | `HybridDSGE_UNet` | U-Net + DSGE-канали нелінійних ознак, HuberLoss |

Всі нейронні моделі:
- Навчаються за критерієм `max(val_SNR)` — зберігаються ваги з найкращим SNR на валідації
- Підтримують `denoise_numpy(noisy: [N,T]) → [N,T]` — уніфікований API для inference

---

## TODO

1. ~~add generation of polygaussian noise~~
2. refactor code to support variable signal lengths
3. ~~write separate classes for training and models~~
4. ~~generate big dataset~~
5. ~~train models on both noise types~~
6. ~~create comparison report with cross-evaluation~~
7. evaluate with `nperseg=128, noverlap=96` across all models

## Articles

- compare different denoising approaches (analytical, transformers, autoencoders) under Gaussian vs non-Gaussian noise
- compare architectures of autoencoders for signal denoising
