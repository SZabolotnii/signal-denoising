"""
Експеримент 1: MVP-валідація гіпотези DSGE.

Порівнює три підходи на тест-сеті:
    1. Baseline U-Net (1-канальний STFT вхід)
    2. DSGE-only  (пряме аналітичне наближення Y = k₀ + KΦ(x̃))
    3. Hybrid U-Net DSGE (4-канальний вхід: STFT + 3 DSGE)

Використовує вже натреновані ваги з weights/ директорії.
Виводить таблицю MSE/SNR + t-test для статистичної значимості.

Запуск:
    cd experiments/
    python experiment_1_baseline.py               # non_gaussian
    python experiment_1_baseline.py --dataset gaussian
"""

import argparse
import os
import sys

import numpy as np
import torch
from scipy.signal import stft, istft
from scipy.stats import ttest_rel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.autoencoder_unet import UnetAutoencoder
from models.hybrid_unet import HybridDSGE_UNet
from models.dsge_layer import DSGEFeatureExtractor
from metrics import MeanSquaredError, SignalToNoiseRatio, MeanAbsoluteError, RootMeanSquaredError


# ──────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────

def stft_mag_phase(s, fs, nperseg, noverlap):
    _, _, Zxx = stft(s, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return np.abs(Zxx), np.angle(Zxx)


def istft_rec(mag, phase, fs, nperseg, noverlap, L):
    Zxx = mag * np.exp(1j * phase)
    _, r = istft(Zxx, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return r[:L] if len(r) >= L else np.pad(r, (0, L - len(r)))


def metrics(true, pred):
    return {
        'MSE':  MeanSquaredError.calculate(true, pred),
        'MAE':  MeanAbsoluteError.calculate(true, pred),
        'RMSE': RootMeanSquaredError.calculate(true, pred),
        'SNR':  SignalToNoiseRatio.calculate(true, pred),
    }


def per_sample_mse(true, pred):
    return np.array([MeanSquaredError.calculate(true[i], pred[i]) for i in range(len(true))])


def per_sample_snr(true, pred):
    return np.array([SignalToNoiseRatio.calculate(true[i], pred[i]) for i in range(len(true))])


# ──────────────────────────────────────────────────────
#  Деноїзинг методів
# ──────────────────────────────────────────────────────

def run_baseline(model, X, device, cfg) -> np.ndarray:
    """Baseline U-Net (1 канал)."""
    model.eval()
    recs = []
    with torch.no_grad():
        for s in X:
            mag, phase = stft_mag_phase(s, cfg['fs'], cfg['nperseg'], cfg['noverlap'])
            inp = torch.tensor(mag, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            out = model(inp).squeeze().cpu().numpy()
            recs.append(istft_rec(out, phase, cfg['fs'], cfg['nperseg'], cfg['noverlap'], cfg['signal_len']))
    return np.stack(recs)


def run_dsge_only(dsge: DSGEFeatureExtractor, X, cfg) -> np.ndarray:
    """Пряме DSGE-наближення (без U-Net)."""
    return np.stack([dsge.reconstruct(s) for s in X])


def run_hybrid(model, dsge, X, device, cfg) -> np.ndarray:
    """Hybrid DSGE + U-Net (4 канали)."""
    model.eval()
    recs = []
    with torch.no_grad():
        for s in X:
            mag, phase = stft_mag_phase(s, cfg['fs'], cfg['nperseg'], cfg['noverlap'])
            dsge_mags = dsge.compute_dsge_spectrograms(s)  # [S, F, T']
            x4 = np.concatenate([mag[np.newaxis], dsge_mags], axis=0)  # [1+S, F, T']
            inp = torch.tensor(x4, dtype=torch.float32).unsqueeze(0).to(device)
            out_mask = model(inp).squeeze().cpu().numpy()
            out_mag = out_mask * mag
            recs.append(istft_rec(out_mag, phase, cfg['fs'], cfg['nperseg'], cfg['noverlap'], cfg['signal_len']))
    return np.stack(recs)


# ──────────────────────────────────────────────────────
#  Таблиця результатів
# ──────────────────────────────────────────────────────

def print_table(results: dict, p_values: dict):
    """results = {'Method': {MSE, MAE, RMSE, SNR}}"""
    cols = list(results.keys())
    print('\n' + '═' * 72)
    print(f"{'Metric':<10}", end='')
    for c in cols:
        print(f"  {c:>18}", end='')
    print()
    print('─' * 72)
    for k in ['MSE', 'MAE', 'RMSE']:
        print(f"{k:<10}", end='')
        for c in cols:
            print(f"  {results[c][k]:>18.6f}", end='')
        print()
    print(f"{'SNR':<10}", end='')
    for c in cols:
        print(f"  {results[c]['SNR']:>17.2f}dB", end='')
    print()
    print('─' * 72)
    print('\nStatistical significance (paired t-test vs Baseline):')
    for name, (p_mse, p_snr) in p_values.items():
        sig_mse = '✅' if p_mse < 0.05 else '❌'
        sig_snr = '✅' if p_snr < 0.05 else '❌'
        print(f"  {name:<22} MSE p={p_mse:.4f} {sig_mse}   SNR p={p_snr:.4f} {sig_snr}")
    print('═' * 72)


# ──────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────

def main(cfg):
    root = os.path.join(os.path.dirname(__file__), '..')
    weights = os.path.join(root, 'weights')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dt, S = cfg['dataset_type'], cfg['dsge_order']
    stft_prm = {'fs': cfg['fs'], 'nperseg': cfg['nperseg'], 'noverlap': cfg['noverlap']}

    # Дані
    noisy_all = np.load(os.path.join(root, 'dataset', f'{dt}_signals.npy'))
    clean_all = np.load(os.path.join(root, 'dataset', 'clean_signals.npy'))
    N = len(noisy_all)
    np.random.seed(cfg['random_state'])
    idx = np.random.permutation(N)
    test_idx = idx[int(0.7 * N):]
    X_test = noisy_all[test_idx]
    y_test = clean_all[test_idx]
    print(f'Test size: {len(X_test)} signals  [{dt}]')

    # Input shape
    _, _, Zxx = stft(clean_all[0], **stft_prm)
    input_shape = np.abs(Zxx).shape

    # ── Baseline ─────────────────────────────────────────
    b_path = os.path.join(weights, f'UnetAutoencoder_{dt}_best.pth')
    assert os.path.exists(b_path), f'Baseline weights missing: {b_path}\nRun train/training_uae.py first.'
    baseline = UnetAutoencoder(input_shape=input_shape)
    baseline.load_state_dict(torch.load(b_path, map_location=device))
    baseline.to(device)

    # ── Hybrid ────────────────────────────────────────────
    h_path = os.path.join(weights, f'HybridDSGE_UNet_{dt}_S{S}_best.pth')
    d_path = os.path.join(weights, f'dsge_state_{dt}_S{S}.npz')
    assert os.path.exists(h_path), f'Hybrid weights missing: {h_path}\nRun train/training_hybrid.py first.'
    assert os.path.exists(d_path), f'DSGE state missing: {d_path}'
    hybrid = HybridDSGE_UNet(input_shape=input_shape, dsge_order=S)
    hybrid.load_state_dict(torch.load(h_path, map_location=device))
    hybrid.to(device)
    dsge = DSGEFeatureExtractor.load_state(d_path, basis_type='fractional', stft_params=stft_prm)
    dsge.check_generating_element_norm()

    # ── Запуск ────────────────────────────────────────────
    print('\nRunning baseline…')
    rec_base  = run_baseline(baseline, X_test, device, cfg)
    print('Running DSGE-only…')
    rec_dsge  = run_dsge_only(dsge, X_test, cfg)
    print('Running hybrid…')
    rec_hybrid = run_hybrid(hybrid, dsge, X_test, device, cfg)

    # ── Метрики ──────────────────────────────────────────
    results = {
        'Baseline UAE': metrics(y_test, rec_base),
        'DSGE-only':    metrics(y_test, rec_dsge),
        'Hybrid DSGE':  metrics(y_test, rec_hybrid),
    }

    bmse = per_sample_mse(y_test, rec_base)
    bsnr = per_sample_snr(y_test, rec_base)
    p_values = {}
    for name, rec in [('DSGE-only', rec_dsge), ('Hybrid DSGE', rec_hybrid)]:
        _, pm = ttest_rel(bmse, per_sample_mse(y_test, rec))
        _, ps = ttest_rel(bsnr, per_sample_snr(y_test, rec))
        p_values[name] = (pm, ps)

    print_table(results, p_values)

    # Висновок
    hybrid_mse = results['Hybrid DSGE']['MSE']
    base_mse   = results['Baseline UAE']['MSE']
    improvement = (base_mse - hybrid_mse) / base_mse * 100
    print(f'\n→ Hybrid vs Baseline MSE improvement: {improvement:+.2f}%')
    p_m = p_values['Hybrid DSGE'][0]
    if improvement > 0 and p_m < 0.05:
        print('✅ DSGE гіпотеза підтверджена: значуще покращення MSE (p < 0.05).')
    elif improvement > 0:
        print('⚠️  MSE покращилась, але статистично незначуще (p ≥ 0.05).')
    else:
        print('❌ DSGE не покращило MSE. Дивись Plan.md §5.2 (мітігація).')


# ──────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',    type=str, default='non_gaussian', choices=['gaussian', 'non_gaussian'])
    p.add_argument('--dsge-order', type=int, default=3)
    args = p.parse_args()
    cfg = dict(
        dataset_type=args.dataset, dsge_order=args.dsge_order,
        signal_len=2144, fs=1024, nperseg=128, noverlap=96, random_state=42,
    )
    main(cfg)
