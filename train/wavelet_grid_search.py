from itertools import product

import numpy as np
import pywt

from metrics import MeanSquaredError
from models.wavelet import WaveletDenoising


def grid_search_wavelet(noisy: np.ndarray,
                        clean: np.ndarray,
                        random_state: int = 42,
                        param_grid: dict[str, list] | None = None):
    """
    Підбирає гіперпараметри класу WaveletDenoising за MSE на валідації.
    Розбиття: ~70% train, 20% val, 10% test (всередині цієї функції).
    Повертає: best_params, best_val_mse, test_mse
    """
    assert noisy.shape == clean.shape, "noisy/clean shapes must match"
    N = len(noisy)

    # розбиття на train/val/test
    rng = np.random.default_rng(random_state)
    idx = np.arange(N)
    rng.shuffle(idx)
    train_end = int(0.7 * N)
    val_end = int(0.9 * N)
    train_idx, val_idx, test_idx = idx[:train_end], idx[train_end:val_end], idx[val_end:]

    X_train, y_train = noisy[train_idx], clean[train_idx]  # train не використовується напряму, але залишимо на майбутнє
    X_val, y_val = noisy[val_idx], clean[val_idx]
    X_test, y_test = noisy[test_idx], clean[test_idx]

    # дефолтна сітка
    if param_grid is None:
        param_grid = {
            "wavelet": ["db2", "db4", "db6", "sym4", "coif1", "bior4.4"],
            "level": [1, 2, 3, 4, None],  # None -> auto (max level)
            "thresh_mode": ["soft", "hard"],
            "per_level": [True, False],
            "ext_mode": ["symmetric", "periodization"],
        }

    keys = list(param_grid.keys())
    combos = list(product(*[param_grid[k] for k in keys]))

    best_params = None
    best_val_mse = np.inf

    # один інстанс, будемо міняти параметри
    denoiser = WaveletDenoising()

    for combo in combos:
        params = dict(zip(keys, combo))

        # обмежимо рівень максимально можливим для поточного T
        T = X_val.shape[1]
        max_level = pywt.dwt_max_level(T, pywt.Wavelet(params["wavelet"]).dec_len)
        lvl = params["level"]
        if lvl is None or (isinstance(lvl, int) and lvl > max_level):
            params["level"] = max_level

        denoiser.set_params(**params)

        # валід. MSE
        mses = []
        for x_noisy, x_clean in zip(X_val, y_val):
            x_den = denoiser.denoise(x_noisy)
            mses.append(MeanSquaredError.calculate(x_clean, x_den))
        val_mse = float(np.mean(mses))

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_params = params

    # Оцінка на тесті з найкращими параметрами
    denoiser.set_params(**best_params)
    test_mses = []
    for x_noisy, x_clean in zip(X_test, y_test):
        x_den = denoiser.denoise(x_noisy)
        test_mses.append(MeanSquaredError.calculate(x_clean, x_den))
    test_mse = float(np.mean(test_mses))

    print("Best params:", best_params)
    print(f"Best VAL MSE:  {best_val_mse:.6f}")
    print(f"Final TEST MSE:{test_mse:.6f}")

    return best_params, best_val_mse, test_mse


# ─────────────────────────────────────────────────────────────────────────────
# Приклад використання
if __name__ == "__main__":
    # завантаження вашого датасету
    noisy = np.load("../data_generation/non_gaussian_signals.npy")  # або non_gaussian_signals.npy
    clean = np.load("../data_generation/clean_signals.npy")
    assert noisy.shape == clean.shape

    best_params, val_mse, test_mse = grid_search_wavelet(noisy, clean)

    # продемонструємо один кейс з найкращими параметрами
    denoiser = WaveletDenoising().set_params(**best_params)
    idx = 0
    x_noisy, x_clean = noisy[idx], clean[idx]
    x_den = denoiser.denoise(x_noisy)

    import matplotlib.pyplot as plt

    t = np.arange(len(x_noisy))
    plt.figure(figsize=(12, 5))
    plt.plot(t, x_clean, label="Clean", linewidth=2, color="black")
    plt.plot(t, x_noisy, label="Noisy", alpha=0.5, color="gray")
    plt.plot(t, x_den, label="Wavelet (tuned)", linestyle="--", linewidth=2)
    plt.title("Wavelet Denoising with tuned hyperparams (MSE)")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.ylim([-2, 2])
    plt.xlim([0, 400])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
