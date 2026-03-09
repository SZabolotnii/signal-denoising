"""
Експеримент 3: Оптимізація порядку апроксимації S (DSGE order).

Grid search: S ∈ {2, 3, 4, 5, 6}.
Для кожного S навчає модель і оцінює MSE/SNR на тест-сеті.

Запуск:
    cd experiments/
    python experiment_3_order.py --dataset non_gaussian --epochs 20
"""

import argparse
import os
import sys
import subprocess

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from metrics import MeanSquaredError, SignalToNoiseRatio
from models.hybrid_unet import HybridDSGE_UNet
from models.dsge_layer import DSGEFeatureExtractor
from scipy.signal import stft, istft
import torch


def train_for_order(S: int, dataset: str, epochs: int) -> bool:
    """Навчає модель з dsge_order=S."""
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), '..', 'train', 'training_hybrid.py'),
        '--dataset',    dataset,
        '--epochs',     str(epochs),
        '--dsge-order', str(S),
        '--no-wandb',
    ]
    print(f'\n[Exp3] Training S={S}…')
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode == 0


def evaluate_order(S: int, dataset: str, cfg: dict) -> dict:
    """Запускає тестовий набір для заданого S і повертає метрики."""
    root = os.path.join(os.path.dirname(__file__), '..')
    weights = os.path.join(root, 'weights')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fs, nperseg, noverlap = cfg['fs'], cfg['nperseg'], cfg['noverlap']
    signal_len = cfg['signal_len']

    h_path = os.path.join(weights, f'HybridDSGE_UNet_{dataset}_S{S}_best.pth')
    d_path = os.path.join(weights, f'dsge_state_{dataset}_S{S}.npz')
    if not os.path.exists(h_path) or not os.path.exists(d_path):
        print(f'[Exp3] Missing weights for S={S}, skipping eval.')
        return {}

    noisy_all = np.load(os.path.join(root, 'dataset', f'{dataset}_signals.npy'))
    clean_all = np.load(os.path.join(root, 'dataset', 'clean_signals.npy'))
    N = len(noisy_all)
    np.random.seed(42)
    idx = np.random.permutation(N)
    test_idx = idx[int(0.7 * N):]
    X_test = noisy_all[test_idx]
    y_test = clean_all[test_idx]

    _, _, Zxx0 = stft(clean_all[0], fs=fs, nperseg=nperseg, noverlap=noverlap)
    input_shape = np.abs(Zxx0).shape

    model = HybridDSGE_UNet(input_shape=input_shape, dsge_order=S)
    model.load_state_dict(torch.load(h_path, map_location=device))
    model.to(device).eval()

    dsge = DSGEFeatureExtractor.load_state(d_path, basis_type='fractional',
                                           stft_params={'fs': fs, 'nperseg': nperseg, 'noverlap': noverlap})

    recs = []
    with torch.no_grad():
        for s in X_test:
            _, _, Zxx = stft(s, fs=fs, nperseg=nperseg, noverlap=noverlap)
            mag, phase = np.abs(Zxx), np.angle(Zxx)
            d_mags = dsge.compute_dsge_spectrograms(s)
            x4 = np.concatenate([mag[np.newaxis], d_mags], axis=0)
            inp = torch.tensor(x4, dtype=torch.float32).unsqueeze(0).to(device)
            out = model(inp).squeeze().cpu().numpy()
            _, r = istft(out * np.exp(1j * phase), fs=fs, nperseg=nperseg, noverlap=noverlap)
            r = r[:signal_len] if len(r) >= signal_len else np.pad(r, (0, signal_len - len(r)))
            recs.append(r)
    recs = np.stack(recs)

    return {
        'MSE': MeanSquaredError.calculate(y_test, recs),
        'SNR': SignalToNoiseRatio.calculate(y_test, recs),
        'params': model.param_count(),
    }


def plot_results(orders, mse_vals, snr_vals, dataset: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(orders, mse_vals, 'o-', color='royalblue', lw=2)
    ax1.set_xlabel('DSGE Order S')
    ax1.set_ylabel('MSE')
    ax1.set_title(f'MSE vs S ({dataset})')
    ax1.grid(True)

    ax2.plot(orders, snr_vals, 's-', color='tomato', lw=2)
    ax2.set_xlabel('DSGE Order S')
    ax2.set_ylabel('SNR (dB)')
    ax2.set_title(f'SNR vs S ({dataset})')
    ax2.grid(True)

    plt.tight_layout()
    fname = f'exp3_order_search_{dataset}.png'
    plt.savefig(fname, dpi=150)
    print(f'[Saved] {fname}')
    plt.show()


def main(cfg: dict):
    orders = [2, 3, 4, 5, 6]
    all_results = {}

    for S in orders:
        ok = train_for_order(S, cfg['dataset'], cfg['epochs'])
        if ok:
            res = evaluate_order(S, cfg['dataset'], cfg)
            all_results[S] = res
        else:
            all_results[S] = {}

    print('\n\n' + '═' * 50)
    print('EXPERIMENT 3 — ORDER SEARCH RESULTS')
    print('═' * 50)
    print(f"{'S':>4} {'MSE':>12} {'SNR (dB)':>12} {'#Params':>12}")
    print('─' * 50)
    mse_vals, snr_vals, valid_orders = [], [], []
    for S in orders:
        r = all_results.get(S, {})
        if r:
            print(f"{S:>4} {r['MSE']:>12.6f} {r['SNR']:>12.2f} {r['params']:>12,}")
            mse_vals.append(r['MSE'])
            snr_vals.append(r['SNR'])
            valid_orders.append(S)
        else:
            print(f"{S:>4} {'N/A':>12} {'N/A':>12} {'N/A':>12}")
    print('═' * 50)

    if valid_orders:
        best_s = valid_orders[int(np.argmin(mse_vals))]
        print(f'\n→ Optimal S = {best_s} (MSE = {min(mse_vals):.6f})')
        plot_results(valid_orders, mse_vals, snr_vals, cfg['dataset'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Experiment 3: DSGE order grid search')
    p.add_argument('--dataset', type=str, default='non_gaussian', choices=['gaussian', 'non_gaussian'])
    p.add_argument('--epochs',  type=int, default=20)
    args = p.parse_args()
    main({
        'dataset': args.dataset, 'epochs': args.epochs,
        'fs': 1024, 'nperseg': 128, 'noverlap': 96, 'signal_len': 2144,
    })
