import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import numpy as np
import pywt

from metrics import MeanSquaredError
from models.wavelet import WaveletDenoising


GRID = {
    "wavelet":    ["db4", "sym4", "coif1"],
    "level":      [2, 3, 4],
    "thresh_mode":["soft", "hard"],
    "per_level":  [False],
    "ext_mode":   ["symmetric"],
}


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

    rng = np.random.default_rng(random_state)
    idx = np.arange(N)
    rng.shuffle(idx)
    train_end = int(0.7 * N)
    val_end = int(0.9 * N)
    train_idx, val_idx, test_idx = idx[:train_end], idx[train_end:val_end], idx[val_end:]

    X_val,  y_val  = noisy[val_idx],  clean[val_idx]
    X_test, y_test = noisy[test_idx], clean[test_idx]

    if param_grid is None:
        param_grid = GRID

    keys = list(param_grid.keys())
    combos = list(product(*[param_grid[k] for k in keys]))

    best_params = None
    best_val_mse = np.inf

    T = X_val.shape[1]
    denoiser = WaveletDenoising()

    print(f"  Grid search: {len(combos)} combos × {len(X_val)} val samples "
          f"= {len(combos) * len(X_val):,} evaluations")

    for combo in combos:
        params = dict(zip(keys, combo))

        max_level = pywt.dwt_max_level(T, pywt.Wavelet(params["wavelet"]).dec_len)
        lvl = params["level"]
        if lvl is None or (isinstance(lvl, int) and lvl > max_level):
            params["level"] = max_level

        denoiser.set_params(**params)

        mses = []
        for x_noisy, x_clean in zip(X_val, y_val):
            x_den = denoiser.denoise(x_noisy)
            mses.append(MeanSquaredError.calculate(x_clean, x_den))
        val_mse = float(np.mean(mses))

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_params = params

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
if __name__ == "__main__":
    import argparse
    import json
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parent.parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    p = argparse.ArgumentParser(description="Wavelet grid search for signal denoising")
    p.add_argument("--dataset",    required=True,
                   help="Path to dataset folder (e.g. data_generation/datasets/<name>)")
    p.add_argument("--noise-type", default="non_gaussian", choices=["gaussian", "non_gaussian"])
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--plot",       action="store_true",
                   help="Plot an example denoising result")
    args = p.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path

    with open(dataset_path / "dataset_config.json") as f:
        cfg = json.load(f)

    print(f"Dataset: {dataset_path.name}")
    print(f"Config:  block_size={cfg['block_size']}, noise_type={args.noise_type}")

    noisy = np.load(dataset_path / "train" / f"{args.noise_type}_signals.npy")
    clean = np.load(dataset_path / "train" / "clean_signals.npy")
    assert noisy.shape == clean.shape

    best_params, val_mse, test_mse = grid_search_wavelet(
        noisy, clean, random_state=args.seed,
    )

    weights_dir = dataset_path / "weights"
    weights_dir.mkdir(exist_ok=True)
    save_path = weights_dir / f"Wavelet_{args.noise_type}_best_params.json"
    with open(save_path, "w") as f:
        json.dump({"best_params": best_params, "val_mse": val_mse, "test_mse": test_mse}, f, indent=2)
    print(f"✅ Best params saved to: {save_path}")

    if args.plot:
        import matplotlib.pyplot as plt
        denoiser = WaveletDenoising().set_params(**best_params)
        x_noisy, x_clean = noisy[0], clean[0]
        x_den = denoiser.denoise(x_noisy)
        t = np.arange(len(x_noisy))
        plt.figure(figsize=(12, 5))
        plt.plot(t, x_clean, label="Clean", linewidth=2, color="black")
        plt.plot(t, x_noisy, label="Noisy", alpha=0.5, color="gray")
        plt.plot(t, x_den, label="Wavelet (tuned)", linestyle="--", linewidth=2)
        plt.title("Wavelet Denoising with tuned hyperparams (MSE)")
        plt.xlabel("Sample"); plt.ylabel("Amplitude")
        plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()
