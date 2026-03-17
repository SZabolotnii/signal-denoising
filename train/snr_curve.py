"""
Per-SNR evaluation utility.

Evaluates a denoising model on per-SNR test files and generates
SNR_out vs SNR_in performance curves.

Usage:
    from train.snr_curve import evaluate_per_snr, print_snr_table, plot_snr_curve, log_snr_curve_wandb

    results = evaluate_per_snr(trainer.denoise_numpy, dataset_path / "test", "non_gaussian")
    print_snr_table(results, model_name="UNet")
    plot_snr_curve(results, model_name="UNet", save_path=weights_dir / "snr_curve.png")
    log_snr_curve_wandb(results, model_name="UNet")
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MPL_OK = True
except Exception:
    MPL_OK = False

try:
    import wandb
    WANDB_OK = True
except Exception:
    WANDB_OK = False

from metrics import MeanSquaredError, MeanAbsoluteError, RootMeanSquaredError, SignalToNoiseRatio


def _label_to_db(label: str) -> float:
    """'m10dB' → -10.0,  '0dB' → 0.0,  'p5dB' → 5.0"""
    m = re.search(r'(\d+)dB', label, re.IGNORECASE)
    val = float(m.group(1)) if m else 0.0
    return -val if label.lower().startswith('m') else val


def evaluate_per_snr(
    denoise_fn,
    test_dir: Path,
    noise_type: str,
    batch_size: int = 64,
) -> dict:
    """
    Evaluate model on each per-SNR test file.

    Parameters
    ----------
    denoise_fn : callable
        (noisy: np.ndarray [N, T]) -> denoised: np.ndarray [N, T]
    test_dir : Path
        Directory with test_*_{noise_type}.npy and test_*_clean.npy
    noise_type : str
    batch_size : int

    Returns
    -------
    dict[str, dict]
        Sorted by SNR_in. Each entry: MSE, MAE, RMSE, SNR, snr_in_db, n_samples
    """
    test_dir = Path(test_dir)
    noisy_files = sorted(
        test_dir.glob(f"test_*_{noise_type}.npy"),
        key=lambda p: _label_to_db(p.stem.split('_')[1]),
    )

    results = {}
    for noisy_path in noisy_files:
        parts = noisy_path.stem.split('_')
        snr_label = parts[1]
        clean_path = noisy_path.parent / f"test_{snr_label}_clean.npy"
        if not clean_path.exists():
            continue

        noisy_all = np.load(noisy_path)
        clean_all = np.load(clean_path)

        chunks = []
        for start in range(0, len(noisy_all), batch_size):
            chunks.append(denoise_fn(noisy_all[start: start + batch_size]))
        denoised_all = np.concatenate(chunks, axis=0)

        results[snr_label] = {
            'MSE':       MeanSquaredError.calculate(clean_all, denoised_all),
            'MAE':       MeanAbsoluteError.calculate(clean_all, denoised_all),
            'RMSE':      RootMeanSquaredError.calculate(clean_all, denoised_all),
            'SNR':       SignalToNoiseRatio.calculate(clean_all, denoised_all),
            'snr_in_db': _label_to_db(snr_label),
            'n_samples': len(noisy_all),
        }

    return results


def print_snr_table(results: dict, model_name: str = "") -> None:
    title = f" — {model_name}" if model_name else ""
    print(f"\n=== Per-SNR metrics{title} ===")
    print(f"{'SNR_in':>8}  {'MSE':>10}  {'MAE':>10}  {'RMSE':>10}  {'SNR_out':>9}")
    print("-" * 58)
    for lbl, m in sorted(results.items(), key=lambda kv: kv[1]['snr_in_db']):
        print(f"{m['snr_in_db']:>7.0f}dB  {m['MSE']:>10.6f}  {m['MAE']:>10.6f}  "
              f"{m['RMSE']:>10.6f}  {m['SNR']:>8.2f} dB")


def plot_snr_curve(
    results: dict,
    model_name: str = "",
    save_path: Path = None,
):
    """Plot SNR_out vs SNR_in and optionally save."""
    if not MPL_OK:
        return None
    lbls = sorted(results, key=lambda l: results[l]['snr_in_db'])
    xs = [results[l]['snr_in_db'] for l in lbls]
    ys = [results[l]['SNR'] for l in lbls]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(xs, ys, 'o-', lw=2, label=model_name or "Model")
    lim = [min(xs) - 2, max(xs) + 2]
    ax.plot(lim, lim, 'k--', lw=1, alpha=0.5, label="No change (SNR_out = SNR_in)")
    ax.fill_between(lim, lim, [max(ys) + 5] * 2,
                    alpha=0.06, color='green', label="Improvement zone")
    ax.set_xlabel("SNR input (dB)")
    ax.set_ylabel("SNR output (dB)")
    ax.set_title(f"SNR curve — {model_name}" if model_name else "SNR curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120)
        print(f"  SNR curve → {save_path}")

    plt.close(fig)
    return fig


def save_training_curves(
    train_losses: list,
    val_snrs: list,
    save_path: Path,
    model_name: str = "",
    noise_type: str = "",
) -> None:
    """Save training loss + val SNR curves as PNG (overfitting monitor, not in main report)."""
    if not MPL_OK:
        return
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, train_losses, lw=2)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Train loss")
    ax1.set_title("Training loss")
    ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, val_snrs, lw=2, color='tab:orange')
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Val SNR (dB)")
    ax2.set_title("Validation SNR")
    ax2.grid(True, alpha=0.3)
    fig.suptitle(f"{model_name} | {noise_type}", fontsize=11)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"  Training curves → {save_path}")


def log_snr_curve_wandb(results: dict, model_name: str = "") -> None:
    """Log per-SNR metrics and line chart to active W&B run."""
    if not (WANDB_OK and hasattr(wandb, 'run') and wandb.run):
        return

    lbls = sorted(results, key=lambda l: results[l]['snr_in_db'])
    xs = [results[l]['snr_in_db'] for l in lbls]
    ys = [results[l]['SNR'] for l in lbls]

    table = wandb.Table(
        columns=["snr_in_db", "snr_out_db"],
        data=list(zip(xs, ys)),
    )
    tag = (model_name or "model").replace(" ", "_")
    per_snr_flat = {}
    for lbl in lbls:
        per_snr_flat[f"test_per_snr/{lbl}/snr_out"] = results[lbl]['SNR']
        per_snr_flat[f"test_per_snr/{lbl}/mse"]     = results[lbl]['MSE']
    wandb.log({
        f"charts/{tag}_snr_curve": wandb.plot.line(
            table, "snr_in_db", "snr_out_db",
            title=f"SNR curve — {model_name}",
        ),
        **per_snr_flat,
    })
