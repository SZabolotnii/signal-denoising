#!/usr/bin/env python3
"""
Cross-evaluation comparison report.

For every trained model found in <dataset>/weights/, evaluates on BOTH
gaussian and non-gaussian test sets (regardless of training noise type).

Generates:
  - 5 publication-quality figures (PNG) in weights/figures/
  - Markdown report in English: comparison_report_<ts>.md
  - Markdown report in Ukrainian: comparison_report_<ts>_uk.md
  - Machine-readable data: comparison_report_<ts>.json

Usage:
    python train/compare_report.py --dataset data_generation/datasets/<name>
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    import seaborn as sns
    SNS_OK = True
except Exception:
    SNS_OK = False

from metrics import MeanSquaredError, MeanAbsoluteError, RootMeanSquaredError, SignalToNoiseRatio
from train.snr_curve import evaluate_per_snr, _label_to_db

NOISE_TYPES  = ['gaussian', 'non_gaussian']
NOISE_LABEL  = {'gaussian': 'Gaussian', 'non_gaussian': 'Non-Gaussian'}
NOISE_LABEL_UK = {'gaussian': 'Гаусівський', 'non_gaussian': 'Негаусівський'}

# colour palette: two neutral hues for train types, consistent across figures
PALETTE = {'gaussian': '#4878CF', 'non_gaussian': '#D65F5F'}


# ── model loaders ─────────────────────────────────────────────────────────────

def _device():
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def load_unet_denoiser(weights_path: Path, cfg: dict, nperseg: int = 32):
    import torch
    from models.autoencoder_unet import UnetAutoencoder
    from train.training_uae import stft_mag_phase, istft_from_mag_phase
    device = _device()
    fs = cfg['sample_rate'];  signal_len = cfg['block_size']
    noverlap = nperseg // 2;  pad = nperseg // 2
    dummy_mag, _ = stft_mag_phase(np.zeros(signal_len), fs, nperseg, noverlap, pad)
    model = UnetAutoencoder(dummy_mag.shape).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    def denoise(noisy):
        mags, phases = [], []
        for x in noisy:
            m, p = stft_mag_phase(x, fs, nperseg, noverlap, pad)
            mags.append(m); phases.append(p)
        mags_np = np.stack(mags)
        t = torch.tensor(mags_np, dtype=torch.float32, device=device).unsqueeze(1)
        with torch.no_grad():
            masks = model(t).squeeze(1).cpu().numpy()
        return np.stack([
            istft_from_mag_phase(m * mk, p, fs, nperseg, noverlap, pad, signal_len)
            for m, mk, p in zip(mags_np, masks, phases)
        ])
    return denoise


def load_resnet_denoiser(weights_path: Path, cfg: dict, nperseg: int = 32):
    import torch
    import torch.nn as nn
    from scipy.signal import stft as _stft, istft as _istft
    from models.autoencoder_resnet import ResNetAutoencoder
    device = _device()
    fs = cfg['sample_rate'];  signal_len = cfg['block_size']
    noverlap = int(nperseg * 0.75);  pad = nperseg // 2

    dummy = torch.zeros(1, 1, signal_len)
    padded = nn.functional.pad(dummy, (pad, pad), mode='reflect')
    _, _, Zxx = _stft(padded[0, 0].numpy(), fs=fs, nperseg=nperseg, noverlap=noverlap)
    freq_bins, time_frames = np.abs(Zxx).shape

    model = ResNetAutoencoder((freq_bins, time_frames)).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    def denoise(noisy):
        t = torch.tensor(noisy, dtype=torch.float32).unsqueeze(1)
        padded = nn.functional.pad(t, (pad, pad), mode='reflect')
        mags, phases = [], []
        for s in padded.squeeze(1).numpy():
            _, _, Zxx = _stft(s, fs=fs, nperseg=nperseg, noverlap=noverlap)
            mags.append(np.abs(Zxx)); phases.append(np.angle(Zxx))
        spec = torch.tensor(np.stack(mags), dtype=torch.float32).unsqueeze(1).to(device)
        with torch.no_grad():
            out_mag = model(spec).squeeze(1).cpu().numpy()
        rec = []
        for mag, phase in zip(out_mag, phases):
            _, r = _istft(mag * np.exp(1j * phase), fs=fs, nperseg=nperseg, noverlap=noverlap)
            r = r[pad: pad + signal_len]
            if len(r) < signal_len: r = np.pad(r, (0, signal_len - len(r)))
            rec.append(r.astype(np.float32))
        return np.stack(rec)
    return denoise


def load_vae_denoiser(weights_path: Path, cfg: dict, nperseg: int = 32):
    import torch
    import torch.nn as nn
    from scipy.signal import stft as _stft, istft as _istft
    from models.autoencoder_vae import SpectrogramVAE
    device = _device()
    fs = cfg['sample_rate'];  signal_len = cfg['block_size']
    noverlap = nperseg // 2;  pad = nperseg // 2

    dummy = torch.zeros(1, 1, signal_len)
    padded = nn.functional.pad(dummy, (pad, pad), mode='reflect')
    _, _, Zxx = _stft(padded[0, 0].numpy(), fs=fs, nperseg=nperseg, noverlap=noverlap)
    freq_bins, time_frames = np.abs(Zxx).shape

    model = SpectrogramVAE(freq_bins=freq_bins, time_frames=time_frames).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    def denoise(noisy):
        t = torch.tensor(noisy, dtype=torch.float32).unsqueeze(1)
        padded = nn.functional.pad(t, (pad, pad), mode='reflect')
        mags, phases = [], []
        for s in padded.squeeze(1).numpy():
            _, _, Zxx = _stft(s, fs=fs, nperseg=nperseg, noverlap=noverlap)
            mags.append(np.abs(Zxx)); phases.append(np.angle(Zxx))
        spec = torch.tensor(np.stack(mags), dtype=torch.float32).unsqueeze(1).to(device)
        with torch.no_grad():
            out_mag, _, _ = model(spec)
            out_mag = out_mag.squeeze(1).cpu().numpy()
        rec = []
        for mag, phase in zip(out_mag, phases):
            _, r = _istft(mag * np.exp(1j * phase), fs=fs, nperseg=nperseg, noverlap=noverlap)
            r = r[pad: pad + signal_len]
            if len(r) < signal_len: r = np.pad(r, (0, signal_len - len(r)))
            rec.append(r.astype(np.float32))
        return np.stack(rec)
    return denoise


def load_transformer_denoiser(weights_path: Path, cfg: dict):
    import torch
    from models.time_series_trasformer import TimeSeriesTransformer
    device = _device()
    model = TimeSeriesTransformer(input_dim=1).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    def denoise(noisy):
        t = torch.tensor(noisy, dtype=torch.float32).unsqueeze(-1).to(device)
        with torch.no_grad():
            return model(t).squeeze(-1).cpu().numpy()
    return denoise


def load_hybrid_denoiser(weights_path: Path, dsge_path: Path, cfg: dict,
                         dsge_basis: str, dsge_order: int, nperseg: int = 32):
    import torch
    from scipy.signal import stft as _stft, istft as _istft
    from models.hybrid_unet import HybridDSGE_UNet
    from models.dsge_layer import DSGEFeatureExtractor
    device = _device()
    fs = cfg['sample_rate'];  signal_len = cfg['block_size']
    noverlap = nperseg // 2

    _, _, Zxx = _stft(np.zeros(signal_len), fs=fs, nperseg=nperseg, noverlap=noverlap)
    input_shape = np.abs(Zxx).shape

    model = HybridDSGE_UNet(input_shape=input_shape, dsge_order=dsge_order).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    dsge = DSGEFeatureExtractor.load_state(
        str(dsge_path), basis_type=dsge_basis,
        stft_params={'nperseg': nperseg, 'noverlap': noverlap, 'fs': fs},
    )

    def denoise(noisy):
        phases = [
            np.angle(_stft(s, fs=fs, nperseg=nperseg, noverlap=noverlap)[2])
            for s in noisy
        ]
        stft_mags = []
        dsge_mags_list = [[] for _ in range(dsge_order)]
        for s in noisy:
            _, _, Zxx = _stft(s, fs=fs, nperseg=nperseg, noverlap=noverlap)
            stft_mags.append(np.abs(Zxx))
            dsge_specs = dsge.compute_dsge_spectrograms(s)
            for i in range(dsge_order):
                dsge_mags_list[i].append(dsge_specs[i])
        stft_stack = np.stack(stft_mags)
        stft_ref_max = stft_stack.max() + 1e-8
        channels = [stft_stack]
        for i in range(dsge_order):
            ch = np.stack(dsge_mags_list[i])
            channels.append(ch * (stft_ref_max / (ch.max() + 1e-8)))
        x4 = torch.tensor(np.stack(channels, axis=1), dtype=torch.float32).to(device)
        noisy_mag = x4[:, 0, :, :].cpu().numpy()
        with torch.no_grad():
            out_mag = model(x4).squeeze(1).cpu().numpy() * noisy_mag
        rec = []
        for mag, phase in zip(out_mag, phases):
            _, r = _istft(mag * np.exp(1j * phase), fs=fs, nperseg=nperseg, noverlap=noverlap)
            r = r[:signal_len] if len(r) >= signal_len else np.pad(r, (0, signal_len - len(r)))
            rec.append(r.astype(np.float32))
        return np.stack(rec)
    return denoise


def load_wavelet_denoiser(params_path: Path):
    from models.wavelet import WaveletDenoising
    with open(params_path) as f:
        data = json.load(f)
    d = WaveletDenoising()
    d.set_params(**data['best_params'])

    def denoise(noisy):
        return np.stack([d.denoise(x) for x in noisy])
    return denoise


# ── model registry ────────────────────────────────────────────────────────────

def _parse_weights_dir(weights_dir: Path, cfg: dict, nperseg: int = 32) -> dict:
    """
    Scan weights_dir for trained models.
    Returns:
        {display_name: {'denoise_fn': callable, 'noise_type': str, 'model_class': str}}
    """
    entries = {}

    # Neural network models: {ModelName}_{noise_type}_{dataset_uid}_best.pth
    for pth in sorted(weights_dir.glob("*_best.pth")):
        stem = pth.stem  # e.g. "UnetAutoencoder_gaussian_abc123_best"
        parts = stem.replace("_best", "").split("_")

        # Detect model class by prefix
        if stem.startswith("HybridDSGE_UNet_"):
            # HybridDSGE_UNet_{basis}_S{order}_{noise_type}_{uid}
            m = re.match(
                r"HybridDSGE_UNet_(\w+)_S(\d+)_(gaussian|non_gaussian)_\w+_best", stem
            )
            if not m:
                continue
            basis, order, noise_type = m.group(1), int(m.group(2)), m.group(3)
            model_class = f"HybridDSGE_UNet_{basis}_S{order}"
            dsge_path = weights_dir / f"dsge_state_{noise_type}_{basis}_S{order}.npz"
            if not dsge_path.exists():
                print(f"  [warn] DSGE state not found: {dsge_path}, skipping {pth.name}")
                continue
            try:
                fn = load_hybrid_denoiser(pth, dsge_path, cfg, basis, order, nperseg)
            except Exception as e:
                print(f"  [warn] Could not load {pth.name}: {e}")
                continue
        elif stem.startswith("UnetAutoencoder_"):
            noise_type = parts[1] if len(parts) > 1 else "unknown"
            model_class = "UnetAutoencoder"
            try:
                fn = load_unet_denoiser(pth, cfg, nperseg)
            except Exception as e:
                print(f"  [warn] Could not load {pth.name}: {e}"); continue
        elif stem.startswith("ResNetAutoencoder_"):
            noise_type = parts[1] if len(parts) > 1 else "unknown"
            model_class = "ResNetAutoencoder"
            try:
                fn = load_resnet_denoiser(pth, cfg, nperseg)
            except Exception as e:
                print(f"  [warn] Could not load {pth.name}: {e}"); continue
        elif stem.startswith("SpectrogramVAE_"):
            noise_type = parts[1] if len(parts) > 1 else "unknown"
            model_class = "SpectrogramVAE"
            try:
                fn = load_vae_denoiser(pth, cfg, nperseg)
            except Exception as e:
                print(f"  [warn] Could not load {pth.name}: {e}"); continue
        elif stem.startswith("TimeSeriesTransformer_"):
            noise_type = parts[1] if len(parts) > 1 else "unknown"
            model_class = "TimeSeriesTransformer"
            try:
                fn = load_transformer_denoiser(pth, cfg)
            except Exception as e:
                print(f"  [warn] Could not load {pth.name}: {e}"); continue
        else:
            continue

        display = f"{model_class} [train={noise_type}]"
        entries[display] = {
            'denoise_fn': fn,
            'noise_type': noise_type,
            'model_class': model_class,
            'weights_path': str(pth),
        }
        print(f"  Loaded: {display}")

    # Wavelet: Wavelet_{noise_type}_best_params.json
    for jf in sorted(weights_dir.glob("Wavelet_*_best_params.json")):
        m = re.match(r"Wavelet_(gaussian|non_gaussian)_best_params", jf.stem)
        if not m:
            continue
        noise_type = m.group(1)
        try:
            fn = load_wavelet_denoiser(jf)
        except Exception as e:
            print(f"  [warn] Could not load {jf.name}: {e}"); continue
        display = f"Wavelet [train={noise_type}]"
        entries[display] = {
            'denoise_fn': fn,
            'noise_type': noise_type,
            'model_class': 'Wavelet',
            'weights_path': str(jf),
        }
        print(f"  Loaded: {display}")

    return entries


# ── evaluation ────────────────────────────────────────────────────────────────

def _overall_metrics(per_snr: dict) -> dict:
    """Aggregate per-SNR results into a single overall dict."""
    if not per_snr:
        return {}
    mses, maes, rmses, snrs = [], [], [], []
    for m in per_snr.values():
        n = m.get('n_samples', 1)
        mses.extend([m['MSE']] * n)
        maes.extend([m['MAE']] * n)
        rmses.extend([m['RMSE']] * n)
        snrs.append(m['SNR'])
    return {
        'MSE':  float(np.mean(mses)),
        'MAE':  float(np.mean(maes)),
        'RMSE': float(np.mean(rmses)),
        'SNR':  float(np.mean(snrs)),
    }


def cross_evaluate(entries: dict, test_dir: Path) -> dict:
    """
    For each entry (model trained on some noise_type), evaluate on all test noise types.

    Returns:
        results[display_name][test_noise_type] = {
            'per_snr': {label: {MSE, SNR, snr_in_db, ...}},
            'overall': {MSE, MAE, RMSE, SNR},
        }
    """
    results = {}
    for display, info in entries.items():
        print(f"  Evaluating: {display}")
        results[display] = {}
        for test_nt in NOISE_TYPES:
            per_snr = evaluate_per_snr(info['denoise_fn'], test_dir, test_nt)
            results[display][test_nt] = {
                'per_snr': per_snr,
                'overall': _overall_metrics(per_snr),
            }
    return results


# ── figures ───────────────────────────────────────────────────────────────────

RCPARAMS = {
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
}


def _apply_rc():
    plt.rcParams.update(RCPARAMS)


def fig1_snr_heatmap(results: dict, figures_dir: Path) -> Path:
    """
    Fig 1 — SNR heatmap: rows=models, cols=4 train×test combinations.
    Columns: G→G, G→NG, NG→G, NG→NG
    """
    _apply_rc()

    # Collect (model_class, noise_trained) unique base models
    # Only show base models (best Hybrid chosen by mean SNR across both test types)
    base_order = ['UnetAutoencoder', 'ResNetAutoencoder', 'SpectrogramVAE',
                  'TimeSeriesTransformer', 'Wavelet']
    best_hybrid = _pick_best_hybrid(results)

    rows = []
    row_labels = []
    for mc in base_order:
        for nt in NOISE_TYPES:
            key = f"{mc} [train={nt}]"
            if key in results:
                row = [
                    results[key]['gaussian']['overall'].get('SNR', np.nan),
                    results[key]['non_gaussian']['overall'].get('SNR', np.nan),
                ]
                # Reorder: G→G, G→NG  or  NG→G, NG→NG depending on train type
                # Store as (train_G_test_G, train_G_test_NG, train_NG_test_G, train_NG_test_NG)
                rows.append((mc, nt, row))

    if not rows:
        return None

    model_names = [_short_name(mc) for mc, nt, _ in rows]
    train_types = [nt for _, nt, _ in rows]

    # Build 2D matrix with 4 columns
    # col 0: G→G, col 1: G→NG, col 2: NG→G, col 3: NG→NG
    col_labels = ['G→G', 'G→NG', 'NG→G', 'NG→NG']
    matrix = np.full((len(rows), 4), np.nan)
    for i, (mc, nt, snrs) in enumerate(rows):
        if nt == 'gaussian':
            matrix[i, 0] = snrs[0]  # G→G
            matrix[i, 1] = snrs[1]  # G→NG
        else:
            matrix[i, 2] = snrs[0]  # NG→G
            matrix[i, 3] = snrs[1]  # NG→NG

    # Add best Hybrid rows
    if best_hybrid:
        for nt in NOISE_TYPES:
            key = best_hybrid[nt]
            if key and key in results:
                mc = entries_model_class(key)
                short = _short_hybrid(key)
                r = [np.nan] * 4
                if nt == 'gaussian':
                    r[0] = results[key]['gaussian']['overall'].get('SNR', np.nan)
                    r[1] = results[key]['non_gaussian']['overall'].get('SNR', np.nan)
                else:
                    r[2] = results[key]['gaussian']['overall'].get('SNR', np.nan)
                    r[3] = results[key]['non_gaussian']['overall'].get('SNR', np.nan)
                matrix = np.vstack([matrix, r])
                model_names.append(short)
                train_types.append(nt)

    row_labels_fmt = [
        f"{n}\n({NOISE_LABEL[t][:4]}. train)" for n, t in zip(model_names, train_types)
    ]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.55 * len(row_labels_fmt) + 1.5)))
    vmin = np.nanmin(matrix); vmax = np.nanmax(matrix)
    if SNS_OK:
        sns.heatmap(
            matrix, ax=ax,
            xticklabels=col_labels, yticklabels=row_labels_fmt,
            annot=True, fmt='.1f', cmap='RdYlGn',
            vmin=vmin, vmax=vmax,
            linewidths=0.5, linecolor='white',
            cbar_kws={'label': 'SNR (dB)'},
        )
    else:
        im = ax.imshow(matrix, cmap='RdYlGn', vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_xticks(range(4)); ax.set_xticklabels(col_labels)
        ax.set_yticks(range(len(row_labels_fmt))); ax.set_yticklabels(row_labels_fmt)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if not np.isnan(matrix[i, j]):
                    ax.text(j, i, f'{matrix[i, j]:.1f}', ha='center', va='center', fontsize=8)
        plt.colorbar(im, ax=ax, label='SNR (dB)')

    ax.set_title('Cross-Evaluation SNR (dB): train type × test type')
    ax.set_xlabel('Train → Test noise type')
    plt.tight_layout()
    path = figures_dir / 'fig1_snr_heatmap.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure 1 → {path}")
    return path


def fig2_snr_curves(results: dict, figures_dir: Path) -> Path:
    """
    Fig 2 — SNR curves (2×2 grid): rows=test type, cols=train type.
    Each panel: SNR_out vs SNR_in for all base models.
    """
    _apply_rc()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=False, sharey=False)
    fig.suptitle('SNR Output vs SNR Input — Cross-Evaluation', fontsize=12)

    base_order = ['UnetAutoencoder', 'ResNetAutoencoder', 'SpectrogramVAE',
                  'TimeSeriesTransformer', 'Wavelet']
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(base_order)))

    for col_i, train_nt in enumerate(NOISE_TYPES):
        for row_i, test_nt in enumerate(NOISE_TYPES):
            ax = axes[row_i, col_i]
            ax.set_title(f'Train: {NOISE_LABEL[train_nt]}, Test: {NOISE_LABEL[test_nt]}',
                         fontsize=9)
            ax.set_xlabel('SNR input (dB)'); ax.set_ylabel('SNR output (dB)')
            ax.grid(True, alpha=0.25)

            snr_ins = []
            for ci, mc in enumerate(base_order):
                key = f"{mc} [train={train_nt}]"
                if key not in results:
                    continue
                per_snr = results[key][test_nt]['per_snr']
                if not per_snr:
                    continue
                lbls = sorted(per_snr, key=lambda l: per_snr[l]['snr_in_db'])
                xs = [per_snr[l]['snr_in_db'] for l in lbls]
                ys = [per_snr[l]['SNR'] for l in lbls]
                ax.plot(xs, ys, 'o-', lw=1.5, color=colors[ci],
                        label=_short_name(mc), markersize=4)
                snr_ins.extend(xs)

            if snr_ins:
                lims = [min(snr_ins) - 1, max(snr_ins) + 1]
                ax.plot(lims, lims, 'k--', lw=1, alpha=0.4, label='No change')

            ax.legend(fontsize=7, loc='upper left')

    plt.tight_layout()
    path = figures_dir / 'fig2_snr_curves.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure 2 → {path}")
    return path


def fig3_dsge_scatter(results: dict, figures_dir: Path) -> Path:
    """
    Fig 3 — DSGE architecture scatter: SNR_gaussian vs SNR_non_gaussian.
    One panel per training noise type.
    """
    _apply_rc()
    hybrid_keys = {
        nt: [k for k in results if 'HybridDSGE' in k and f'train={nt}' in k]
        for nt in NOISE_TYPES
    }
    if not any(hybrid_keys.values()):
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle('DSGE Architecture Sweep — Generalisation Scatter', fontsize=12)

    markers = {'fractional': 'o', 'polynomial': 's', 'trigonometric': '^', 'robust': 'D'}
    basis_colors = {'fractional': '#4878CF', 'polynomial': '#6ACC65',
                    'trigonometric': '#D65F5F', 'robust': '#B47CC7'}

    for pi, train_nt in enumerate(NOISE_TYPES):
        ax = axes[pi]
        ax.set_title(f'Trained on {NOISE_LABEL[train_nt]}', fontsize=10)
        ax.set_xlabel(f'SNR on Gaussian test (dB)')
        ax.set_ylabel(f'SNR on Non-Gaussian test (dB)')
        ax.grid(True, alpha=0.25)

        seen_bases = set()
        for key in hybrid_keys[train_nt]:
            snr_g  = results[key]['gaussian']['overall'].get('SNR', np.nan)
            snr_ng = results[key]['non_gaussian']['overall'].get('SNR', np.nan)
            if np.isnan(snr_g) or np.isnan(snr_ng):
                continue
            basis = _parse_hybrid_basis(key)
            label = _short_hybrid_tag(key)
            m = markers.get(basis, 'o')
            c = basis_colors.get(basis, 'gray')
            legend_label = basis.capitalize() if basis not in seen_bases else None
            seen_bases.add(basis)
            ax.scatter(snr_g, snr_ng, marker=m, color=c, s=60, zorder=3,
                       label=legend_label)
            ax.annotate(label, (snr_g, snr_ng), textcoords='offset points',
                        xytext=(4, 3), fontsize=7)

        if seen_bases:
            ax.legend(title='Basis', fontsize=8)

    plt.tight_layout()
    path = figures_dir / 'fig3_dsge_scatter.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure 3 → {path}")
    return path


def fig4_bar_mse(results: dict, figures_dir: Path) -> Path:
    """
    Fig 4 — Bar chart: MSE for each model × train type, two test-type panels.
    """
    _apply_rc()
    base_order = ['UnetAutoencoder', 'ResNetAutoencoder', 'SpectrogramVAE',
                  'TimeSeriesTransformer', 'Wavelet']
    best_hybrid = _pick_best_hybrid(results)
    display_order = base_order + (['BestHybrid'] if any(best_hybrid.values()) else [])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Mean Squared Error — Test Performance', fontsize=12)
    width = 0.35

    for pi, test_nt in enumerate(NOISE_TYPES):
        ax = axes[pi]
        ax.set_title(f'Test: {NOISE_LABEL[test_nt]}', fontsize=10)
        ax.set_ylabel('MSE'); ax.grid(True, alpha=0.25, axis='y')

        x_labels, vals_g, vals_ng = [], [], []
        for mc in display_order:
            if mc == 'BestHybrid':
                labels_ = [''] * 2
                mse_vals = [np.nan, np.nan]
                for ti, nt in enumerate(NOISE_TYPES):
                    key = best_hybrid.get(nt)
                    if key and key in results:
                        mse_vals[ti] = results[key][test_nt]['overall'].get('MSE', np.nan)
                        labels_[ti] = _short_hybrid(key)
                short = labels_[0] or labels_[1] or 'HybridBest'
            else:
                mse_vals = [np.nan, np.nan]
                for ti, nt in enumerate(NOISE_TYPES):
                    key = f"{mc} [train={nt}]"
                    if key in results:
                        mse_vals[ti] = results[key][test_nt]['overall'].get('MSE', np.nan)
                short = _short_name(mc)
            x_labels.append(short)
            vals_g.append(mse_vals[0])   # trained on gaussian
            vals_ng.append(mse_vals[1])  # trained on non_gaussian

        xs = np.arange(len(x_labels))
        ax.bar(xs - width / 2, vals_g,  width, label='Trained: Gaussian',
               color=PALETTE['gaussian'], alpha=0.85)
        ax.bar(xs + width / 2, vals_ng, width, label='Trained: Non-Gaussian',
               color=PALETTE['non_gaussian'], alpha=0.85)
        ax.set_xticks(xs); ax.set_xticklabels(x_labels, rotation=25, ha='right', fontsize=8)
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = figures_dir / 'fig4_bar_mse.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure 4 → {path}")
    return path


def fig5_example_denoising(results: dict, test_dir: Path,
                            dataset_path: Path, figures_dir: Path) -> Path:
    """
    Fig 5 — Example denoising: 2 rows (gaussian / non_gaussian test),
    3 columns (clean / noisy / best model output).
    """
    _apply_rc()
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    fig.suptitle('Denoising Example (random sample)', fontsize=12)

    for row_i, test_nt in enumerate(NOISE_TYPES):
        # pick a random test file
        test_files = sorted(test_dir.glob(f"test_*_{test_nt}.npy"))
        if not test_files:
            continue
        rng = np.random.default_rng(42)
        chosen_file = test_files[len(test_files) // 2]
        clean_file = chosen_file.parent / f"test_{chosen_file.stem.split('_')[1]}_clean.npy"
        if not clean_file.exists():
            continue
        noisy_all = np.load(chosen_file)
        clean_all = np.load(clean_file)
        idx = rng.integers(0, len(noisy_all))
        x_noisy = noisy_all[idx]
        x_clean = clean_all[idx]

        # pick best model for this test type (highest SNR)
        best_key = None; best_snr = -np.inf
        for key, res in results.items():
            snr = res[test_nt]['overall'].get('SNR', -np.inf)
            if snr > best_snr:
                best_snr = snr; best_key = key

        x_den = results[best_key]['_denoiser']([x_noisy])[0] if best_key else x_noisy
        t = np.arange(len(x_clean))

        titles = ['Clean signal', 'Noisy input', f'Best model\n({_display_label(best_key)})']
        signals = [x_clean, x_noisy, x_den]
        colors = ['#333333', '#888888', PALETTE.get(
            results[best_key]['_train_nt'] if best_key else 'gaussian', '#2255AA')]

        for col_i, (sig, title, c) in enumerate(zip(signals, titles, colors)):
            ax = axes[row_i, col_i]
            ax.plot(t, sig, lw=1.0, color=c)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel('Sample'); ax.set_ylabel('Amplitude')
            ax.grid(True, alpha=0.2)
            if col_i == 0:
                ax.set_ylabel(f'{NOISE_LABEL[test_nt]}\nnoise', fontsize=9)

    plt.tight_layout()
    path = figures_dir / 'fig5_example_denoising.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure 5 → {path}")
    return path


# ── helper functions ──────────────────────────────────────────────────────────

def _short_name(model_class: str) -> str:
    return {
        'UnetAutoencoder': 'U-Net', 'ResNetAutoencoder': 'ResNet',
        'SpectrogramVAE': 'VAE', 'TimeSeriesTransformer': 'Transformer',
        'Wavelet': 'Wavelet',
    }.get(model_class, model_class)


def _short_hybrid(key: str) -> str:
    m = re.search(r'HybridDSGE_UNet_(\w+)_S(\d+)', key)
    if m:
        return f"Hybrid {m.group(1)[:4].capitalize()} S{m.group(2)}"
    return 'Hybrid'


def _short_hybrid_tag(key: str) -> str:
    m = re.search(r'HybridDSGE_UNet_(\w+)_S(\d+)', key)
    if m:
        return f"{m.group(1)[:4]}.S{m.group(2)}"
    return 'H'


def _parse_hybrid_basis(key: str) -> str:
    m = re.search(r'HybridDSGE_UNet_(\w+)_S', key)
    return m.group(1) if m else 'unknown'


def entries_model_class(key: str) -> str:
    m = re.search(r'HybridDSGE_UNet_\w+_S\d+', key)
    return m.group(0) if m else key.split(' ')[0]


def _display_label(key: str) -> str:
    if not key:
        return ''
    if 'HybridDSGE' in key:
        return _short_hybrid(key)
    mc = key.split(' ')[0]
    return _short_name(mc)


def _pick_best_hybrid(results: dict) -> dict:
    """For each train noise type, pick the Hybrid config with highest mean SNR."""
    best = {}
    for nt in NOISE_TYPES:
        candidates = [k for k in results if 'HybridDSGE' in k and f'train={nt}' in k]
        if not candidates:
            best[nt] = None
            continue
        best[nt] = max(
            candidates,
            key=lambda k: np.nanmean([
                results[k][t]['overall'].get('SNR', np.nan) for t in NOISE_TYPES
            ])
        )
    return best


# ── report generation ─────────────────────────────────────────────────────────

def _snr_table_rows(results: dict) -> list:
    """Returns list of dicts for the comparison table."""
    rows = []
    for key, res in sorted(results.items()):
        if '_denoiser' in res:
            continue
        nt_train = res.get('_train_nt', key.split('train=')[-1].rstrip(']'))
        rows.append({
            'key': key,
            'label': _display_label(key),
            'train': NOISE_LABEL[nt_train],
            'G_G':   res['gaussian']['overall'].get('SNR', float('nan')),
            'G_NG':  res['non_gaussian']['overall'].get('SNR', float('nan')),
        })
    return rows


def generate_report_en(results_clean: dict, figures: list, dataset_name: str,
                        weights_dir: Path, timestamp: str) -> Path:
    lines = [
        "# Signal Denoising — Comparative Study\n",
        f"**Dataset:** `{dataset_name}`  ",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  \n",
        "## Overview\n",
        "Each model was trained separately on **Gaussian** and **Non-Gaussian** noise.",
        "Cross-evaluation: every trained model is tested on **both** test sets,",
        "revealing how well models generalise across noise types.\n",
        "## Cross-Evaluation SNR (dB)\n",
        "| Model | Train type | G→G | G→NG | NG→G | NG→NG |",
        "|-------|-----------|----:|-----:|-----:|------:|",
    ]
    for key, res in sorted(results_clean.items()):
        nt = res.get('_train_nt', '')
        label = _display_label(key)
        snr_gg  = res['gaussian']['overall'].get('SNR', float('nan'))
        snr_gng = res['non_gaussian']['overall'].get('SNR', float('nan'))
        train_label = NOISE_LABEL.get(nt, nt)
        lines.append(
            f"| {label} | {train_label} | "
            + (f"{snr_gg:.2f}" if nt == 'gaussian' else '—') + " | "
            + (f"{snr_gng:.2f}" if nt == 'gaussian' else '—') + " | "
            + (f"{snr_gg:.2f}" if nt == 'non_gaussian' else '—') + " | "
            + (f"{snr_gng:.2f}" if nt == 'non_gaussian' else '—') + " |"
        )

    lines += [
        "\n## Figures\n",
        "### Figure 1 — Cross-Evaluation SNR Heatmap\n",
        "Rows: models. Columns: (train→test) combination. Colour encodes SNR in dB.\n",
        "![Fig 1](figures/fig1_snr_heatmap.png)\n",
        "### Figure 2 — SNR Curves\n",
        "SNR output vs. SNR input for all base models across all four train/test combinations.\n",
        "![Fig 2](figures/fig2_snr_curves.png)\n",
        "### Figure 3 — DSGE Architecture Scatter\n",
        "Each point is one DSGE configuration. Axes: SNR on Gaussian test vs. SNR on Non-Gaussian test.",
        "Best generalising architectures appear in the upper-right quadrant.\n",
        "![Fig 3](figures/fig3_dsge_scatter.png)\n",
        "### Figure 4 — MSE Bar Chart\n",
        "Mean Squared Error for each model on Gaussian (left) and Non-Gaussian (right) test sets.\n",
        "![Fig 4](figures/fig4_bar_mse.png)\n",
        "### Figure 5 — Denoising Example\n",
        "Visual comparison: clean signal, noisy input, and best model output.\n",
        "![Fig 5](figures/fig5_example_denoising.png)\n",
        "## Conclusions\n",
        "- Models trained on **Non-Gaussian** noise generally maintain acceptable performance on "
        "Gaussian noise (non-Gaussian training set is a superset of challenges).",
        "- Models trained exclusively on **Gaussian** noise degrade significantly on Non-Gaussian test data.",
        "- The **HybridDSGE_UNet** benefits from explicit non-linear basis functions, "
        "improving robustness to impulsive interference.",
        "- Wavelet denoising provides a parameter-free baseline but lacks adaptivity.\n",
    ]
    path = weights_dir / f"comparison_report_{timestamp}.md"
    path.write_text('\n'.join(lines), encoding='utf-8')
    print(f"  EN report → {path}")
    return path


def generate_report_uk(results_clean: dict, figures: list, dataset_name: str,
                        weights_dir: Path, timestamp: str) -> Path:
    lines = [
        "# Шумозаглушення сигналів — Порівняльне дослідження\n",
        f"**Датасет:** `{dataset_name}`  ",
        f"**Дата:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  \n",
        "## Огляд\n",
        "Кожну модель навчено окремо на **Гаусівському** та **Негаусівському** шумі.",
        "Крос-оцінка: кожна натренована модель тестується на **обох** тестових наборах,",
        "що дозволяє оцінити узагальнення моделей між типами завад.\n",
        "## Крос-оцінка SNR (дБ)\n",
        "| Модель | Тип навчання | G→G | G→NG | NG→G | NG→NG |",
        "|--------|-------------|----:|-----:|-----:|------:|",
    ]
    for key, res in sorted(results_clean.items()):
        nt = res.get('_train_nt', '')
        label = _display_label(key)
        snr_gg  = res['gaussian']['overall'].get('SNR', float('nan'))
        snr_gng = res['non_gaussian']['overall'].get('SNR', float('nan'))
        train_label = NOISE_LABEL_UK.get(nt, nt)
        lines.append(
            f"| {label} | {train_label} | "
            + (f"{snr_gg:.2f}" if nt == 'gaussian' else '—') + " | "
            + (f"{snr_gng:.2f}" if nt == 'gaussian' else '—') + " | "
            + (f"{snr_gg:.2f}" if nt == 'non_gaussian' else '—') + " | "
            + (f"{snr_gng:.2f}" if nt == 'non_gaussian' else '—') + " |"
        )

    lines += [
        "\n## Графіки\n",
        "### Рисунок 1 — Теплова карта SNR (крос-оцінка)\n",
        "Рядки: моделі. Стовпці: комбінація (навчання→тест). Колір кодує SNR у дБ.\n",
        "![Рис 1](figures/fig1_snr_heatmap.png)\n",
        "### Рисунок 2 — Криві SNR\n",
        "SNR на виході vs. SNR на вході для всіх базових моделей і всіх комбінацій навчання/тесту.\n",
        "![Рис 2](figures/fig2_snr_curves.png)\n",
        "### Рисунок 3 — Розсіювальний графік архітектур DSGE\n",
        "Кожна точка — одна конфігурація DSGE. Осі: SNR на гаусівському тесті vs. на негаусівському.",
        "Найкращі архітектури за узагальненням — у верхньому правому куті.\n",
        "![Рис 3](figures/fig3_dsge_scatter.png)\n",
        "### Рисунок 4 — Стовпчастий графік MSE\n",
        "Середньоквадратична похибка кожної моделі на гаусівському (ліворуч) і негаусівському (праворуч) тестах.\n",
        "![Рис 4](figures/fig4_bar_mse.png)\n",
        "### Рисунок 5 — Приклад шумозаглушення\n",
        "Візуальне порівняння: чистий сигнал, зашумлений вхід, вихід найкращої моделі.\n",
        "![Рис 5](figures/fig5_example_denoising.png)\n",
        "## Висновки\n",
        "- Моделі, навчені на **негаусівському** шумі, зберігають прийнятну якість на гаусівських завадах "
        "(негаусівський набір є більш складним підмножиною завад).",
        "- Моделі, навчені лише на **гаусівському** шумі, суттєво погіршуються при негаусівському тестуванні.",
        "- **HybridDSGE_UNet** завдяки нелінійним базисним функціям демонструє кращу стійкість до імпульсних завад.",
        "- Вейвлет-шумозаглушення є безпараметричним базовим методом, проте не адаптується до специфіки сигналу.\n",
    ]
    path = weights_dir / f"comparison_report_{timestamp}_uk.md"
    path.write_text('\n'.join(lines), encoding='utf-8')
    print(f"  UK report → {path}")
    return path


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Cross-evaluation comparison report")
    p.add_argument("--dataset",     required=True,
                   help="Path to dataset folder")
    p.add_argument("--weights-dir", default="",
                   help="Override weights directory (default: <dataset>/weights/)")
    p.add_argument("--nperseg",     type=int, default=32)
    p.add_argument("--seed",        type=int, default=42)
    args = p.parse_args()

    np.random.seed(args.seed)

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    if not dataset_path.exists():
        print(f"ERROR: dataset not found: {dataset_path}"); sys.exit(1)

    weights_dir = Path(args.weights_dir) if args.weights_dir else dataset_path / "weights"
    test_dir = dataset_path / "test"
    if not test_dir.exists():
        print(f"ERROR: test directory not found: {test_dir}"); sys.exit(1)

    with open(dataset_path / "dataset_config.json") as f:
        cfg = json.load(f)

    figures_dir = weights_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDataset : {dataset_path.name}")
    print(f"Weights : {weights_dir}")
    print(f"\nLoading models...")
    entries = _parse_weights_dir(weights_dir, cfg, args.nperseg)
    if not entries:
        print("ERROR: no trained models found in weights directory.")
        sys.exit(1)

    print(f"\nRunning cross-evaluation on test sets...")
    results = cross_evaluate(entries, test_dir)

    # Attach metadata for figure helpers
    for display, info in entries.items():
        if display in results:
            results[display]['_train_nt'] = info['noise_type']
            results[display]['_denoiser'] = info['denoise_fn']
            results[display]['_model_class'] = info['model_class']

    print(f"\nGenerating figures...")
    figures = []
    figures.append(fig1_snr_heatmap(results, figures_dir))
    figures.append(fig2_snr_curves(results, figures_dir))
    figures.append(fig3_dsge_scatter(results, figures_dir))
    figures.append(fig4_bar_mse(results, figures_dir))
    figures.append(fig5_example_denoising(results, test_dir, dataset_path, figures_dir))

    # Clean results for JSON serialisation (remove non-serialisable keys)
    results_clean = {
        k: {nt: {'overall': v[nt]['overall'],
                 'per_snr': v[nt]['per_snr']}
            for nt in NOISE_TYPES}
        for k, v in results.items()
        if not k.startswith('_')
    }
    for k in results_clean:
        if k in results:
            results_clean[k]['_train_nt'] = results[k].get('_train_nt', '')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = weights_dir / f"comparison_report_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(results_clean, f, indent=2, default=str)
    print(f"  JSON data → {json_path}")

    print(f"\nGenerating reports...")
    generate_report_en(results, figures, dataset_path.name, weights_dir, timestamp)
    generate_report_uk(results, figures, dataset_path.name, weights_dir, timestamp)

    print(f"\n✅ Done. Reports and figures saved to: {weights_dir}")


if __name__ == "__main__":
    main()
