# Signal Denoising - Non-Gaussian Noise Study

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
обсягу при однакових SNR. Єдина змінна - тип шуму в тренувальних даних.
Якщо NN_B стабільно краще, це свідчить про цінність відповідного
моделювання шуму. Якщо різниця несуттєва, AWGN-апроксимація виправдана.

Якщо гіпотеза підтверджується, наступний крок - дослідження архітектурних
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
сфокусуватись на ефекті розподілу шуму. Перехід до I/Q - природний
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
│   ├── datasets/              <- згенеровані датасети (gitignored)
│   │   └── deep_space_polygauss_nonstationary_bpsk_bs256_n50000_<uid>/
│   │       ├── dataset_config.json
│   │       ├── train/         <- clean_signals.npy, gaussian_signals.npy, non_gaussian_signals.npy
│   │       ├── test/          <- per-SNR test files (test_m10dB_*.npy, ...)
│   │       └── weights/       <- trained model weights (created automatically)
│   └── README.md              <- детальна документація генерації
├── models/                    <- model architectures
├── train/                     <- training scripts
└── README.md
```

---

## Датасет

Синтетичний датасет моделює **цифровий baseband-сигнал після downconversion**
для двох сценаріїв:

- **deep_space**: умови дальнього космічного зв'язку, BPSK/QPSK, SNR -20..0 дБ
- **fpv_telemetry**: FPV-телеметрія / ELRS-подібні системи, CPFSK/GFSK/QPSK, SNR -5..+15 дБ

Для кожного прикладу генерується три версії: чистий сигнал, з AWGN і з негауссовим шумом.
Основний тип негауссового шуму - **нестаціонарний полігауссовий**: суміш гауссіан,
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

## Тренування моделей

Усі скрипти тренування приймають однакові аргументи і зберігають ваги в
`<dataset>/weights/`. Параметри сигналу (`block_size`, `sample_rate`) читаються
автоматично з `dataset_config.json`.

**Загальні аргументи:**

| Аргумент | За замовчуванням | Опис |
|---|---|---|
| `--dataset` | — | Шлях до папки датасету (обов'язковий) |
| `--noise-type` | `non_gaussian` | `gaussian` або `non_gaussian` |
| `--epochs` | залежить від моделі | Кількість епох |
| `--batch-size` | `32` | Розмір батча |
| `--lr` | `1e-4` | Learning rate |
| `--seed` | `42` | Random seed |
| `--wandb-project` | `""` | W&B project (порожньо = вимкнено) |

```bash
DATASET=data_generation/datasets/deep_space_polygauss_nonstationary_bpsk_bs256_n50000_39075e4f

# Всі моделі одразу
python train/train_all.py --dataset $DATASET --noise-type non_gaussian --models all --epochs 50

# Окремо кожна модель
python train/training_uae.py         --dataset $DATASET --noise-type non_gaussian --epochs 30
python train/training_resnet.py      --dataset $DATASET --noise-type non_gaussian --epochs 50
python train/training_vae.py         --dataset $DATASET --noise-type non_gaussian --epochs 50
python train/training_transformer.py --dataset $DATASET --noise-type non_gaussian --epochs 50
python train/wavelet_grid_search.py  --dataset $DATASET --noise-type non_gaussian
python train/training_hybrid.py      --dataset $DATASET --noise-type non_gaussian --epochs 30

# Тренування на гауссовому шумі для порівняння
python train/training_uae.py --dataset $DATASET --noise-type gaussian --epochs 30

# З W&B логуванням
python train/train_all.py --dataset $DATASET --wandb-project signal-denoising
```

Натреновані ваги зберігаються у:
```
<dataset>/weights/
├── UnetAutoencoder_non_gaussian_best.pth
├── ResNetAutoencoder_non_gaussian_best.pth
├── SpectrogramVAE_non_gaussian_best.pth
├── TimeSeriesTransformer_non_gaussian_best.pth
├── Wavelet_non_gaussian_best_params.json
├── HybridDSGE_UNet_non_gaussian_S3_best.pth
└── dsge_state_non_gaussian_S3.npz
```

### HybridDSGE_UNet

Гібридна архітектура (інтеграція DSGE + U-Net). Додаткові параметри:

| Аргумент | За замовчуванням | Опис |
|---|---|---|
| `--dsge-order` | `3` | Кількість DSGE-каналів (S) |
| `--dsge-basis` | `fractional` | Тип базисних функцій: `fractional`, `polynomial`, `trigonometric`, `robust` |
| `--dsge-powers` | `0.5 1.5 2.0` | Степені/частоти для basis (через пробіл) |
| `--lambda` | `0.01` | Коефіцієнт λ для регуляризації Тіхонова |

```bash
# Тренування гібридної моделі
python train/training_hybrid.py --dataset $DATASET --noise-type non_gaussian

# Inference / оцінка на тестовому наборі
python inference/inference_hybrid.py --dataset $DATASET --noise-type non_gaussian
```

---

## TODO

1. ~~add generation of polygaussian noise~~
2. refactor code to support variable signal lengths (currently all signals have the same length, required for autoencoder architecture or padding strategy)
3. ~~write separate classes for training and models~~
4. ~~generate big dataset~~
5. train models on the dataset
6. create comparison table with metrics
7. evaluate with `nperseg=128, noverlap=96` across all models

## Articles

- compare different denoising approaches (analytical, transformers, autoencoders) under Gaussian vs non-Gaussian noise
- compare architectures of autoencoders for signal denoising
