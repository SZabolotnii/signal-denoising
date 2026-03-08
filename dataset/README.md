# Dataset — Signal Denoising

## Базові конфігурації шуму

Для тренування і порівняння моделей використовуються два базових варіанти негауссового шуму:

### Варіант A — Полігауссовий (baseline)

Суміш гауссіан з випадковими weights/means/stds. Рівномірний спектр, негауссовий розподіл.
Моделює нелінійності підсилювачів і загальний канальний шум у deep space.

```bash
python dataset/generation.py --deep_space --polygauss
```

Результат: `dataset/deep_space_polygauss/`

---

### Варіант B — Полігауссовий + Імпульсний

Поєднання фонового негауссового шуму з рідкісними імпульсними сплесками.
Моделює реальніший канал: фоновий шум підсилювача + випадкові EM-спалахи / перешкоди від бортового обладнання.

```bash
python dataset/generation.py --deep_space --polygauss_impulse
```

Результат: `dataset/deep_space_polygauss_impulse/`

---

## Запуск генерації

З кореня репозиторію:

```bash
# Повний список опцій
python dataset/generation.py --help

# Defaults: deep_space + polygauss (еквівалентні команди)
python dataset/generation.py
python dataset/generation.py --deep_space --polygauss

# FPV-сценарій
python dataset/generation.py --fpv --polygauss
python dataset/generation.py --fpv --polygauss_impulse

# Нестаціонарний полігаус
python dataset/generation.py --deep_space --polygauss_nonstationary

# Змінити кількість прикладів
python dataset/generation.py --deep_space --polygauss --num_train 5000 --samples_per_snr 200
```

Вихідна директорія формується автоматично: `dataset/{scenario}_{noise_tag}/train/` і `test/`.

Час генерації: ~3–7 хвилин (15 000 тренувальних + 3 500 тестових прикладів).

---

## Структура файлів

```
dataset/
├── generation.py               ← єдиний скрипт генерації (параметри — в блоці __main__)
└── deep_space/
    ├── train/                  ← тренувальний датасет (варіативний SNR)
    │   ├── clean_signals.npy        (15 000 × 2048)  float32
    │   ├── gaussian_signals.npy     (15 000 × 2048)  float32
    │   ├── non_gaussian_signals.npy (15 000 × 2048)  float32
    │   └── snr_values.npy           (15 000,)         float32  ← SNR кожного прикладу в дБ
    └── test/                   ← тестовий датасет (фіксований SNR, для ROC-кривих)
        ├── test_m20dB_clean.npy          (500 × 2048)
        ├── test_m20dB_gaussian.npy       (500 × 2048)
        ├── test_m20dB_non_gaussian.npy   (500 × 2048)
        ├── test_m15dB_clean.npy
        │   ...                           ← по 3 файли на кожну SNR-точку
        └── test_p0dB_non_gaussian.npy    ← 7 SNR-точок × 3 типи = 21 файл
```

> Конвенція назв тестових файлів: `test_{SNR}_{тип}.npy`,
> де `m` = мінус, `p` = плюс: `m20dB` = −20 dB, `p0dB` = 0 dB.

**Загальний розмір:** ~400 MB.

---

## Використання файлів

### Тренування / валідація / тест моделі

Використовуються файли з `train/`. Розбивка виконується в training-скриптах (70 / 15 / 15 %):

```python
clean        = np.load("dataset/deep_space/train/clean_signals.npy")
noisy_gauss  = np.load("dataset/deep_space/train/gaussian_signals.npy")
noisy_nongauss = np.load("dataset/deep_space/train/non_gaussian_signals.npy")
snr_values   = np.load("dataset/deep_space/train/snr_values.npy")
```

Кожен рядок `i` — це один приклад: `clean[i]` — чистий сигнал, `noisy_gauss[i]` — той самий сигнал з гауссовим шумом при `snr_values[i]` дБ.

### ROC-криві (детектування)

Використовуються файли з `test/`. Для кожної SNR-точки є три масиви:

```python
snr_db = -10
tag = f"m{abs(snr_db)}dB"   # → "m10dB"

clean  = np.load(f"dataset/deep_space/test/test_{tag}_clean.npy")
gauss  = np.load(f"dataset/deep_space/test/test_{tag}_gaussian.npy")
nongauss = np.load(f"dataset/deep_space/test/test_{tag}_non_gaussian.npy")
```

Фіксований SNR дозволяє вимірювати ймовірність детектування (Pd) і хибної тривоги (Pfa) при відомому рівні шуму.

---

## Ідеї та рішення реалізовані в генераторі

### 1. Сценарний підхід

Генератор підтримує профілі `scenario`, що задають діапазон несучої, модуляції та SNR:

| Профіль | Несуча (baseband) | SNR | Модуляції | Прототип |
|---|---|---|---|---|
| `fpv_telemetry` | 300–1800 Hz | −5..+15 dB | QPSK, CPFSK, GFSK | ELRS, Crossfire |
| `deep_space` | 100–800 Hz | −20..0 dB | BPSK, QPSK | Voyager, CubeSat |

Частоти — **baseband після downconversion**: реальна несуча (ГГц) перенесена до низьких частот SDR-приймачем. `sample_rate = 8192 Hz`, `duration = 0.25 s` → **2048 відліків** на сигнал.

### 2. Реалістичні модуляції

**BPSK** — фаза ∈ {0, π}. Стандарт дальнього космосу (Voyager, New Horizons).

**QPSK** — фаза ∈ {0, π/2, π, 3π/2}. Використовується в FPV і космічних місіях.

**CPFSK** (Continuous Phase FSK) — несуча `fc`, дві частоти `fc ± h·bit_rate/2`. Фаза **неперервна** між бітами (на відміну від старого FSK де freq0/freq1 були незалежними). `h = 0.5` → MSK (мінімальне частотне зміщення).

**GFSK** — CPFSK з гаусівським фільтром на бітовому потоці (`BT = 0.4`). Стандарт Bluetooth / FPV-телеметрії. Зменшує спектральне розтікання.

**Symbol rate** пов'язаний з несучою: `symbol_rate = carrier / k`, `k ∈ [6, 12]` — гарантує 6–12 циклів несучої на символ (реалістична умова для стабільної модуляції).

### 3. Варіативний SNR з точним контролем

Рівень шуму задається через SNR в дБ, а не через фіксований `std = 1`:

```
σ_noise = sqrt( mean(signal²) / 10^(SNR/10) )
```

Масштабування через **RMS** (а не std) — коректно для шумів з ненульовим середнім (рожевий, червоний шум, полігаусівський з випадковими means). Точність: ±0.01 дБ для всіх типів шуму.

### 4. Тонке керування негауссовим шумом

Параметри `non_gaussian_noise_types` і `non_gaussian_mix_mode`:

```python
# Тільки полігауссовий (за замовчуванням)
SignalDatasetGenerator(..., non_gaussian_noise_types=["polygauss"])

# Фіксована комбінація
SignalDatasetGenerator(..., non_gaussian_noise_types=["polygauss", "impulse"],
                            non_gaussian_mix_mode="fixed")

# Випадкова підмножина при кожному прикладі
SignalDatasetGenerator(..., non_gaussian_noise_types=["impulse", "pink", "red", "polygauss"],
                            non_gaussian_mix_mode="random")
```

Доступні типи та їх спектральні характеристики:

| Тип | Спектр | Фізичний прототип |
|---|---|---|
| `polygauss` | рівний (~0 dB/dec) | стаціонарна суміш гауссіан, негауссовий канальний шум |
| `polygauss_nonstationary` | рівний, нестаціонарний | суміш гауссіан з параметрами що дрейфують через OU-процеси |
| `impulse` | рівний (широкосмуговий) | спалахи ESC-регуляторів, EM-імпульси |
| `pink` | 1/f (−1 dB/dec) | флікер-шум підсилювачів |
| `red` | 1/f² (−2 dB/dec) | броунівський шум, низькочастотні завади |

**`polygauss_nonstationary` — алгоритм:**

Суміш K ∈ [3, 5] гауссіан, параметри якої плавно дрейфують у часі:
- **Ваги** w_k(t): K траєкторій Орнштейна-Уленбека (τ ≈ 0.125 с) → softmax → Σw_k = 1
- **Дисперсії** σ_k(t): K траєкторій OU у log-просторі → exp, різні зміщення від слабкого фону до сильних викидів
- **Середні** μ_k(t): K−1 вільних траєкторій OU; остання μ_K(t) обчислюється так, щоб загальне середнє шуму залишалось нульовим: `μ_K(t) = −Σ_{k<K} w_k(t)·μ_k(t) / w_K(t)`
- **Генерація**: для кожного t обирається компонента методом рулетки, відлік береться з N(μ_{k(t)}, σ²_{k(t)}); повністю векторизовано через fancy indexing

### 5. Тестовий датасет для ROC-кривих

`generate_test_dataset(snr_values, samples_per_snr)` генерує незалежні від тренувального набори при **фіксованих** SNR-точках. Для `deep_space` обрано 7 точок з густішим семплюванням у перехідній зоні −12..−5 дБ, де алгоритми денойзингу диференціюються найбільше.
