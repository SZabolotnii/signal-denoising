"""
Інференс та порівняння: baseline U-Net (UAE) vs Hybrid DSGE + U-Net.

Запуск:
    cd inference/
    python inference_hybrid.py                         # non_gaussian
    python inference_hybrid.py --dataset gaussian
    python inference_hybrid.py --sample-index 5        # конкретний сигнал
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import stft, istft
from scipy.stats import ttest_rel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.autoencoder_unet import UnetAutoencoder
from models.hybrid_unet import HybridDSGE_UNet
from models.dsge_layer import DSGEFeatureExtractor
from metrics import MeanSquaredError, MeanAbsoluteError, RootMeanSquaredError, SignalToNoiseRatio

# ──────────────────────────────────────────────────────
#  Конфігурація (дефолтна, перекривається через CLI)
# ──────────────────────────────────────────────────────

DEFAULT = dict(
    dataset_type='non_gaussian',
    signal_len=2144,
    fs=1024,
    nperseg=128,
    noverlap=96,
    random_state=42,
    dsge_order=3,
    dsge_basis='fractional',
    sample_index=0,
)


# ──────────────────────────────────────────────────────
#  Допоміжні функції
# ──────────────────────────────────────────────────────

def compute_metrics(true: np.ndarray, pred: np.ndarray) -> dict:
    return {
        'MSE':  MeanSquaredError.calculate(true, pred),
        'MAE':  MeanAbsoluteError.calculate(true, pred),
        'RMSE': RootMeanSquaredError.calculate(true, pred),
        'SNR':  SignalToNoiseRatio.calculate(true, pred),
    }


def signal_to_stft(signal: np.ndarray, fs: int, nperseg: int, noverlap: int):
    """Повертає (mag, phase)."""
    _, _, Zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return np.abs(Zxx), np.angle(Zxx)


def istft_reconstruct(mag: np.ndarray, phase: np.ndarray,
                       fs: int, nperseg: int, noverlap: int, signal_len: int) -> np.ndarray:
    Zxx = mag * np.exp(1j * phase)
    _, rec = istft(Zxx, fs=fs, nperseg=nperseg, noverlap=noverlap)
    if len(rec) >= signal_len:
        return rec[:signal_len]
    return np.pad(rec, (0, signal_len - len(rec)))


def batch_to_4ch(signals: np.ndarray, dsge: DSGEFeatureExtractor,
                 fs: int, nperseg: int, noverlap: int) -> torch.Tensor:
    """[N, T] → [N, 1+S, F, T'] тензор з нормалізованими DSGE-каналами."""
    dsge_bufs = [[] for _ in range(dsge.S)]
    stft_bufs = []

    for s in signals:
        mag, _ = signal_to_stft(s, fs, nperseg, noverlap)
        stft_bufs.append(mag)
        dsge_mags = dsge.compute_dsge_spectrograms(s)  # [S, F, T']
        for i in range(dsge.S):
            dsge_bufs[i].append(dsge_mags[i])

    stft_stack = np.stack(stft_bufs)            # [N, F, T']
    stft_ref_max = stft_stack.max() + 1e-8
    channels = [stft_stack]
    for i in range(dsge.S):
        dsge_ch = np.stack(dsge_bufs[i])        # [N, F, T']
        dsge_max = dsge_ch.max() + 1e-8
        channels.append(dsge_ch * (stft_ref_max / dsge_max))

    x4 = np.stack(channels, axis=1)             # [N, 1+S, F, T']
    return torch.tensor(x4, dtype=torch.float32)


# ──────────────────────────────────────────────────────
#  Деноїзинг батчу — baseline (1-канальний)
# ──────────────────────────────────────────────────────

def denoise_baseline(model: UnetAutoencoder, signals: np.ndarray,
                     device, fs, nperseg, noverlap, signal_len) -> np.ndarray:
    model.eval()
    rec_all = []
    with torch.no_grad():
        for s in signals:
            mag, phase = signal_to_stft(s, fs, nperseg, noverlap)
            inp = torch.tensor(mag, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            out_mag = model(inp).squeeze().cpu().numpy()
            rec_all.append(istft_reconstruct(out_mag, phase, fs, nperseg, noverlap, signal_len))
    return np.stack(rec_all)


# ──────────────────────────────────────────────────────
#  Деноїзинг батчу — hybrid (4-канальний)
# ──────────────────────────────────────────────────────

def denoise_hybrid(model: HybridDSGE_UNet, signals: np.ndarray, dsge: DSGEFeatureExtractor,
                   device, fs, nperseg, noverlap, signal_len) -> np.ndarray:
    model.eval()
    x4 = batch_to_4ch(signals, dsge, fs, nperseg, noverlap).to(device)
    phases = [signal_to_stft(s, fs, nperseg, noverlap)[1] for s in signals]
    rec_all = []
    with torch.no_grad():
        out_masks = model(x4).squeeze(1).cpu().numpy()  # [N, F, T']
        noisy_mags = x4[:, 0, :, :].cpu().numpy()
        out_mags = out_masks * noisy_mags
    for mag, phase in zip(out_mags, phases):
        rec_all.append(istft_reconstruct(mag, phase, fs, nperseg, noverlap, signal_len))
    return np.stack(rec_all)


# ──────────────────────────────────────────────────────
#  Форматування таблиці результатів
# ──────────────────────────────────────────────────────

def print_results_table(baseline_m: dict, hybrid_m: dict, p_mse: float, p_snr: float):
    print('\n' + '=' * 62)
    print(f"{'Metric':<10} {'Baseline UAE':>14} {'Hybrid DSGE':>14} {'Δ':>8}")
    print('-' * 62)
    for k in ['MSE', 'MAE', 'RMSE']:
        b, h = baseline_m[k], hybrid_m[k]
        delta = h - b
        sign = '▼' if delta < 0 else '▲'
        print(f"{k:<10} {b:>14.6f} {h:>14.6f} {sign}{abs(delta):>7.6f}")
    k = 'SNR'
    b, h = baseline_m[k], hybrid_m[k]
    delta = h - b
    sign = '▲' if delta > 0 else '▼'
    print(f"{k:<10} {b:>13.2f}dB {h:>13.2f}dB {sign}{abs(delta):>6.2f}dB")
    print('-' * 62)
    print(f"Statistical significance (paired t-test):")
    print(f"  MSE p-value = {p_mse:.4f} {'✅ significant' if p_mse < 0.05 else '❌ not significant'}")
    print(f"  SNR p-value = {p_snr:.4f} {'✅ significant' if p_snr < 0.05 else '❌ not significant'}")
    print('=' * 62)


# ──────────────────────────────────────────────────────
#  Візуалізація одного сигналу
# ──────────────────────────────────────────────────────

def visualize_sample(clean, noisy, baseline_rec, hybrid_rec, signal_len, fs,
                     dataset_type, sample_index):
    t = np.arange(signal_len) / fs
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(t, clean,        color='black',   lw=2,   label='а) Оригінальний')
    ax.plot(t, noisy,        color='gray',    alpha=.5, label='б) Зашумлений')
    ax.plot(t, baseline_rec, color='royalblue', lw=1.5, ls='--',
            label=f'в) Baseline UAE (MSE={MeanSquaredError.calculate(clean, baseline_rec):.4f})')
    ax.plot(t, hybrid_rec,   color='tomato',   lw=1.5,
            label=f'г) Hybrid DSGE (MSE={MeanSquaredError.calculate(clean, hybrid_rec):.4f})')
    ax.set_xlabel('Час (с)')
    ax.set_ylabel('Амплітуда')
    ax.set_title(f'Порівняння знешумлення — {dataset_type} (зразок #{sample_index})')
    ax.set_xlim([0.15, 0.30])
    ax.set_ylim([-3, 3])
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f'comparison_{dataset_type}_s{sample_index}.png', dpi=150)
    print(f'[Saved] comparison_{dataset_type}_s{sample_index}.png')
    plt.show()


# ──────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────

def main(cfg: dict):
    root = os.path.join(os.path.dirname(__file__), '..')
    weights_dir = os.path.join(root, 'weights')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dt = cfg['dataset_type']
    S  = cfg['dsge_order']

    # ── Завантаження даних ──────────────────────────────
    noisy_all = np.load(os.path.join(root, 'dataset', f'{dt}_signals.npy'))
    clean_all = np.load(os.path.join(root, 'dataset', 'clean_signals.npy'))
    N = len(noisy_all)

    np.random.seed(cfg['random_state'])
    idx = np.arange(N)
    np.random.shuffle(idx)
    test_idx = idx[int(0.7 * N):]

    X_test = noisy_all[test_idx]
    y_test = clean_all[test_idx]

    # ── STFT-форма для моделей ──────────────────────────
    _, _, Zxx = stft(clean_all[0], fs=cfg['fs'], nperseg=cfg['nperseg'], noverlap=cfg['noverlap'])
    input_shape = np.abs(Zxx).shape

    # ── Baseline U-Net ──────────────────────────────────
    baseline_path = os.path.join(weights_dir, f'UnetAutoencoder_{dt}_best.pth')
    if not os.path.exists(baseline_path):
        print(f'[Warning] Baseline weights not found: {baseline_path}')
        print('  → Run train/training_uae.py first, or check the weights/ directory.')
        baseline_model = None
    else:
        baseline_model = UnetAutoencoder(input_shape=input_shape)
        baseline_model.load_state_dict(torch.load(baseline_path, map_location=device))
        baseline_model.to(device)
        print(f'[Loaded] Baseline: {baseline_path}')

    # ── Hybrid U-Net ────────────────────────────────────
    hybrid_path = os.path.join(weights_dir, f'HybridDSGE_UNet_{dt}_S{S}_best.pth')
    dsge_path   = os.path.join(weights_dir, f'dsge_state_{dt}_S{S}.npz')
    if not os.path.exists(hybrid_path) or not os.path.exists(dsge_path):
        print(f'[Warning] Hybrid weights/state not found:\n  {hybrid_path}\n  {dsge_path}')
        print('  → Run train/training_hybrid.py first.')
        hybrid_model = None
        dsge = None
    else:
        hybrid_model = HybridDSGE_UNet(input_shape=input_shape, dsge_order=S)
        hybrid_model.load_state_dict(torch.load(hybrid_path, map_location=device))
        hybrid_model.to(device)
        dsge = DSGEFeatureExtractor.load_state(
            dsge_path,
            basis_type=cfg['dsge_basis'],
            stft_params={'nperseg': cfg['nperseg'], 'noverlap': cfg['noverlap'], 'fs': cfg['fs']},
        )
        print(f'[Loaded] Hybrid: {hybrid_path}')
        print(f'[Loaded] DSGE:   {dsge_path}')

    if baseline_model is None or hybrid_model is None:
        return

    # ── Деноїзинг тест-сету ─────────────────────────────
    print(f'\nDenoising {len(X_test)} test signals…')
    baseline_rec = denoise_baseline(baseline_model, X_test, device,
                                    cfg['fs'], cfg['nperseg'], cfg['noverlap'], cfg['signal_len'])
    hybrid_rec   = denoise_hybrid(hybrid_model, X_test, dsge, device,
                                  cfg['fs'], cfg['nperseg'], cfg['noverlap'], cfg['signal_len'])

    # ── Метрики ─────────────────────────────────────────
    b_metrics = compute_metrics(y_test, baseline_rec)
    h_metrics = compute_metrics(y_test, hybrid_rec)

    # Per-sample MSE і SNR для t-test
    per_mse_b = np.array([MeanSquaredError.calculate(y_test[i], baseline_rec[i]) for i in range(len(y_test))])
    per_mse_h = np.array([MeanSquaredError.calculate(y_test[i], hybrid_rec[i])   for i in range(len(y_test))])
    per_snr_b = np.array([SignalToNoiseRatio.calculate(y_test[i], baseline_rec[i]) for i in range(len(y_test))])
    per_snr_h = np.array([SignalToNoiseRatio.calculate(y_test[i], hybrid_rec[i])   for i in range(len(y_test))])

    _, p_mse = ttest_rel(per_mse_b, per_mse_h)
    _, p_snr = ttest_rel(per_snr_b, per_snr_h)

    print_results_table(b_metrics, h_metrics, p_mse, p_snr)

    # ── Візуалізація зразка ──────────────────────────────
    si = cfg['sample_index']
    visualize_sample(
        y_test[si], X_test[si],
        baseline_rec[si], hybrid_rec[si],
        cfg['signal_len'], cfg['fs'], dt, si,
    )


# ──────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Compare Baseline UAE vs Hybrid DSGE')
    p.add_argument('--dataset',       type=str, default=DEFAULT['dataset_type'],
                   choices=['gaussian', 'non_gaussian'])
    p.add_argument('--dsge-order',    type=int, default=DEFAULT['dsge_order'])
    p.add_argument('--dsge-basis',    type=str, default=DEFAULT['dsge_basis'],
                   choices=['fractional', 'polynomial', 'trigonometric', 'robust'])
    p.add_argument('--sample-index',  type=int, default=DEFAULT['sample_index'])
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg = {**DEFAULT, 'dataset_type': args.dataset,
           'dsge_order': args.dsge_order, 'dsge_basis': args.dsge_basis,
           'sample_index': args.sample_index}
    main(cfg)
