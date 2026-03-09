# Signal Denoising — Hybrid DSGE + U-Net

Дослідження та порівняння моделей знешумлення сигналів для систем бездротового зв'язку.

## Архітектура

Гібридна модель поєднує **DSGE (Decomposition in Space with Generating Element)** з **U-Net автоенкодером**:

```
Зашумлений сигнал x̃(t)
       │
       ├── STFT(x̃)        → |Z₀|   (канал 0)
       ├── STFT(φ₁(x̃))   → |Z₁|   (канал 1)  ── DSGE
       ├── STFT(φ₂(x̃))   → |Z₂|   (канал 2)  ── канали
       └── STFT(φ₃(x̃))   → |Z₃|   (канал 3)  ──
              │
        [B, 4, F, T']
              │
         ┌────┴────┐
         │  U-Net   │  → Mask M ∈ [0, 1]
         └────┬────┘
              │
     |Ẑ| = M ⊙ |Z₀|     (Ratio Mask)
              │
     ISTFT(|Ẑ| · e^(jφ))
              │
     Очищений сигнал x̂(t)
```

Оцінка DSGE будується як:

$$\hat{x} = k_0 + \sum_{i=1}^{S} k_i \cdot \varphi_i(\tilde{x})$$

де оптимальні коефіцієнти $K$ знаходяться з системи $(F + \lambda I) \cdot K = B$ з регуляризацією Тихонова.

## Результати (Non-Gaussian шум, SNR = −10.45 dB)

| Ранг | Метод | MSE ↓ | SNR ↑ | Δ MSE vs Baseline |
|---|---|---|---|---|
| 🥇 | **Hybrid DSGE (robust)** | **0.1597** | **4.96 dB** | **−27.8%** |
| 🥈 | Hybrid DSGE (polynomial) | 0.1613 | 4.91 dB | −27.1% |
| 🥉 | Hybrid DSGE (fractional) | 0.2070 | 3.83 dB | −6.5% |
| 4 | Hybrid DSGE (trigonometric) | 0.2092 | 3.78 dB | −5.5% |
| ref | Baseline U-Net (STFT only) | 0.2213 | 3.54 dB | — |

> Детальний звіт: [`DSGE/Experiment_2_Full_Report.md`](DSGE/Experiment_2_Full_Report.md)

## Структура проєкту

```
signal-denoising/
├── models/
│   ├── autoencoder_unet.py      # Baseline U-Net (1-канальний)
│   ├── hybrid_unet.py           # Hybrid DSGE + U-Net (4-канальний)
│   └── dsge_layer.py            # DSGEFeatureExtractor
├── train/
│   ├── training_uae.py          # Тренування Baseline
│   └── training_hybrid.py       # Тренування Hybrid DSGE
├── inference/
│   └── inference_hybrid.py      # Інференс та порівняння
├── experiments/
│   ├── experiment_1_baseline.py  # Baseline vs DSGE vs Hybrid (t-test)
│   ├── experiment_2_basis.py     # Порівняння базисних функцій
│   └── experiment_3_order.py     # Оптимізація порядку S
├── tests/
│   └── test_dsge_layer.py       # 22 unit-тести
├── DSGE/
│   ├── Plan.md                  # План дослідження
│   └── Experiment_2_Full_Report.md  # Повний звіт
└── dataset/                     # Дані (не в git)
```

## Швидкий старт

```bash
# Створити віртуальне оточення
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Тренування Hybrid DSGE + U-Net (robust базис, 100 епох)
cd train/
python training_hybrid.py --dataset non_gaussian --epochs 100 --dsge-basis robust --lr 5e-5 --no-wandb

# Порівняння з Baseline
cd ../experiments/
python experiment_1_baseline.py --dataset non_gaussian
```

## Базисні функції DSGE

| Тип | Формула | Опис |
|---|---|---|
| `fractional` | $\varphi_i(x) = \text{sign}(x) \cdot \|x\|^{p_i}$ | Дробово-степеневий, зберігає знак |
| `polynomial` | $\varphi_i(x) = x^{p_i}$ | Цілі степені, чутливий до енергії |
| `trigonometric` | $\varphi_i(x) = \sin(f_i \cdot x)$ | Тригонометричний |
| `robust` | $\tanh(x), \sigma(x), \arctan(x)$ | Робастний, пригнічує імпульсний шум |