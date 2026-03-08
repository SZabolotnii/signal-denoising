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
│   │   └── deep_space_polygauss_nonstationary_bpsk_bs256_n50000_6d07aecc/
│   │       ├── dataset_config.json
│   │       ├── train/
│   │       └── test/
│   └── README.md              <- детальна документація генерації
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
