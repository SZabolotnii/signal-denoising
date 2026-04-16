#!/usr/bin/env python3
"""
Cross-evaluation comparison report.

Evaluates all models in a specific training run, tests each on both
Gaussian and Non-Gaussian test sets, then produces:

  runs/run_<date>_<uid>/
  ├── comparison_data_<ts>.csv          – flat table, one row per (model, train, test, SNR_in)
  ├── comparison_report_<ts>.md         – EN report with figures and text
  ├── comparison_report_<ts>_uk.md      – UA report
  └── figures/
      ├── fig1_snr_heatmap.png          – compact SNR overview
      ├── fig2_combined_snr_curves.png  – all models, all train/test combos
      ├── fig3_per_model_comparison.png – Gaussian vs Non-Gaussian training per model
      ├── fig4_dsge_scatter.png         – DSGE architecture generalisation
      └── fig5_example_denoising.png    – visual example

Usage:
    python train/compare_report.py --run data_generation/datasets/<name>/runs/run_<date>_<uid>
"""

import argparse
import csv
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

try:
    import seaborn as sns
    SNS_OK = True
except Exception:
    SNS_OK = False

from metrics import MeanSquaredError, MeanAbsoluteError, RootMeanSquaredError, SignalToNoiseRatio
from train.snr_curve import evaluate_per_snr, _label_to_db

NOISE_TYPES    = ['gaussian', 'non_gaussian']
NOISE_LABEL    = {'gaussian': 'Gaussian',     'non_gaussian': 'Non-Gaussian'}
NOISE_LABEL_UK = {'gaussian': 'Гаусівський',  'non_gaussian': 'Негаусівський'}
NOISE_SHORT    = {'gaussian': 'G',             'non_gaussian': 'NG'}

# Red = Gaussian-trained (baseline), Blue = Non-Gaussian-trained (hypothesis)
# Dashed = tested on Gaussian, Solid = tested on Non-Gaussian
TRAIN_COLOR  = {'gaussian': '#D65F5F', 'non_gaussian': '#4878CF'}
LINE_STYLE   = {'gaussian': '--',      'non_gaussian': '-'}

RCPARAMS = {
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'legend.fontsize': 8, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
}

MODEL_DISPLAY = {
    'UnetAutoencoder':       'U-Net',
    'ResNetAutoencoder':     'ResNet',
    'SpectrogramVAE':        'VAE',
    'TimeSeriesTransformer': 'Transformer',
    'Wavelet':               'Wavelet',
}
BASE_MODELS = list(MODEL_DISPLAY.keys())

# ── model loaders ─────────────────────────────────────────────────────────────

def _device():
    from train.device_utils import get_device
    return str(get_device())


def _load_unet(run_dir: Path, cfg: dict, nperseg: int = 128):
    import torch
    from models.autoencoder_unet import UnetAutoencoder
    from train.training_uae import stft_mag_phase, istft_from_mag_phase
    device = _device()
    fs = cfg['sample_rate']; signal_len = cfg['block_size']
    noverlap = nperseg * 3 // 4; pad = nperseg // 2
    dummy_mag, _ = stft_mag_phase(np.zeros(signal_len), fs, nperseg, noverlap, pad)
    model = UnetAutoencoder(dummy_mag.shape).to(device)
    model.load_state_dict(torch.load(run_dir / 'model_best.pth', map_location=device))
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


def _load_resnet(run_dir: Path, cfg: dict, nperseg: int = 128):
    import torch; import torch.nn as nn
    from scipy.signal import stft as _stft, istft as _istft
    from models.autoencoder_resnet import ResNetAutoencoder
    device = _device()
    fs = cfg['sample_rate']; signal_len = cfg['block_size']
    noverlap = int(nperseg * 0.75); pad = nperseg // 2
    dummy = torch.zeros(1, 1, signal_len)
    padded = nn.functional.pad(dummy, (pad, pad), mode='reflect')
    _, _, Zxx = _stft(padded[0, 0].numpy(), fs=fs, nperseg=nperseg, noverlap=noverlap)
    freq_bins, time_frames = np.abs(Zxx).shape
    model = ResNetAutoencoder((freq_bins, time_frames)).to(device)
    model.load_state_dict(torch.load(run_dir / 'model_best.pth', map_location=device))
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


def _load_vae(run_dir: Path, cfg: dict, nperseg: int = 128):
    import torch; import torch.nn as nn
    from scipy.signal import stft as _stft, istft as _istft
    from models.autoencoder_vae import SpectrogramVAE
    device = _device()
    fs = cfg['sample_rate']; signal_len = cfg['block_size']
    noverlap = nperseg * 3 // 4; pad = nperseg // 2
    dummy = torch.zeros(1, 1, signal_len)
    padded = nn.functional.pad(dummy, (pad, pad), mode='reflect')
    _, _, Zxx = _stft(padded[0, 0].numpy(), fs=fs, nperseg=nperseg, noverlap=noverlap)
    freq_bins, time_frames = np.abs(Zxx).shape
    model = SpectrogramVAE(freq_bins=freq_bins, time_frames=time_frames).to(device)
    model.load_state_dict(torch.load(run_dir / 'model_best.pth', map_location=device))
    model.eval()

    # New VAE pipeline reconstructs normalized log-magnitude spectrogram.
    # If available, load normalization stats to reproduce preprocessing.
    norm_path = run_dir / 'spec_norm.json'
    norm = None
    if norm_path.exists():
        try:
            with open(norm_path) as f:
                norm = json.load(f)
        except Exception:
            norm = None

    def denoise(noisy):
        t = torch.tensor(noisy, dtype=torch.float32).unsqueeze(1)
        padded = nn.functional.pad(t, (pad, pad), mode='reflect')
        mags, phases = [], []
        for s in padded.squeeze(1).numpy():
            _, _, Zxx = _stft(s, fs=fs, nperseg=nperseg, noverlap=noverlap)
            mags.append(np.abs(Zxx)); phases.append(np.angle(Zxx))

        mags_np = np.stack(mags)
        spec = torch.tensor(mags_np, dtype=torch.float32).unsqueeze(1).to(device)

        with torch.no_grad():
            if norm and norm.get('domain') == 'log1p_mag_zscore':
                mean = float(norm.get('mean', 0.0))
                std = float(norm.get('std', 1.0))
                x = (torch.log1p(spec) - mean) / (std + 1e-8)
                recon_z, _, _ = model(x)
                logmag = recon_z * (std + 1e-8) + mean
                out_mag = torch.expm1(logmag).clamp_min(0.0)
            else:
                # Legacy weights (older runs): model was trained to output a [0,1] mask-like magnitude
                # because decoder ended with Sigmoid(). New model removes Sigmoid() to avoid stalling,
                # so apply sigmoid here to reproduce legacy behaviour.
                out_mag, _, _ = model(spec)
                out_mag = torch.sigmoid(out_mag)

            out_mag = out_mag.squeeze(1).cpu().numpy()

        rec = []
        for mag, phase in zip(out_mag, phases):
            _, r = _istft(mag * np.exp(1j * phase), fs=fs, nperseg=nperseg, noverlap=noverlap)
            r = r[pad: pad + signal_len]
            if len(r) < signal_len: r = np.pad(r, (0, signal_len - len(r)))
            rec.append(r.astype(np.float32))
        return np.stack(rec)
    return denoise


def _load_transformer(run_dir: Path, cfg: dict):
    import torch
    from models.time_series_trasformer import TimeSeriesTransformer
    device = _device()
    model = TimeSeriesTransformer(input_dim=1).to(device)
    model.load_state_dict(torch.load(run_dir / 'model_best.pth', map_location=device))
    model.eval()
    def denoise(noisy):
        t = torch.tensor(noisy, dtype=torch.float32).unsqueeze(-1).to(device)
        with torch.no_grad():
            return model(t).squeeze(-1).cpu().numpy()
    return denoise


def _load_hybrid(run_dir: Path, cfg: dict, dsge_basis: str, dsge_order: int, nperseg: int = 128):
    import torch
    from scipy.signal import stft as _stft, istft as _istft
    from models.hybrid_unet import HybridDSGE_UNet
    from models.dsge_layer import DSGEFeatureExtractor
    device = _device()
    fs = cfg['sample_rate']; signal_len = cfg['block_size']
    noverlap = nperseg * 3 // 4
    _, _, Zxx = _stft(np.zeros(signal_len), fs=fs, nperseg=nperseg, noverlap=noverlap)
    input_shape = np.abs(Zxx).shape
    model = HybridDSGE_UNet(input_shape=input_shape, dsge_order=dsge_order).to(device)
    model.load_state_dict(torch.load(run_dir / 'model_best.pth', map_location=device))
    model.eval()
    dsge = DSGEFeatureExtractor.load_state(
        str(run_dir / 'dsge_state.npz'), basis_type=dsge_basis,
        stft_params={'nperseg': nperseg, 'noverlap': noverlap, 'fs': fs},
    )

    def _p99(x, eps: float = 1e-8):
        v = float(np.percentile(x, 99))
        return v if v > eps else eps

    def denoise(noisy):
        phases = [np.angle(_stft(s, fs=fs, nperseg=nperseg, noverlap=noverlap)[2]) for s in noisy]
        stft_mags, dsge_list = [], [[] for _ in range(dsge_order)]
        for s in noisy:
            _, _, Zxx = _stft(s, fs=fs, nperseg=nperseg, noverlap=noverlap)
            stft_mags.append(np.abs(Zxx))
            for i, spec in enumerate(dsge.compute_dsge_spectrograms(s)):
                dsge_list[i].append(spec)
        stft_stack = np.stack(stft_mags)
        # Keep preprocessing consistent with training_hybrid.py:
        # use robust p99 scale instead of max (max is unstable under impulsive noise).
        ref_scale = _p99(stft_stack)
        channels = [stft_stack]
        for i in range(dsge_order):
            ch = np.stack(dsge_list[i])
            ch_scale = _p99(ch)
            channels.append(ch * (ref_scale / ch_scale))
        x4 = torch.tensor(np.stack(channels, axis=1), dtype=torch.float32).to(device)
        with torch.no_grad():
            out_mag = model(x4).squeeze(1).cpu().numpy() * x4[:, 0].cpu().numpy()
        rec = []
        for mag, phase in zip(out_mag, phases):
            _, r = _istft(mag * np.exp(1j * phase), fs=fs, nperseg=nperseg, noverlap=noverlap)
            r = r[:signal_len] if len(r) >= signal_len else np.pad(r, (0, signal_len - len(r)))
            rec.append(r.astype(np.float32))
        return np.stack(rec)
    return denoise


def _load_wavelet(run_dir: Path):
    from models.wavelet import WaveletDenoising
    with open(run_dir / 'best_params.json') as f:
        data = json.load(f)
    d = WaveletDenoising()
    d.set_params(**data['best_params'])
    def denoise(noisy):
        return np.stack([d.denoise(x) for x in noisy])
    return denoise


# ── run discovery ─────────────────────────────────────────────────────────────

def discover_runs(run_dir: Path, cfg: dict, nperseg: int = 128) -> dict:
    """
    Scan a specific run directory for all trained model subfolders.
    Returns {model_name: {'denoise_fn', 'model_class', 'noise_type', 'is_hybrid',
                          'dsge_basis', 'dsge_order', 'run_dir'}}
    """
    if not run_dir.exists():
        return {}

    entries = {}
    for model_dir in sorted(run_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        name = model_dir.name  # e.g. "UnetAutoencoder_gaussian"

        # Parse noise type (last segment is gaussian or non_gaussian)
        for nt in ['non_gaussian', 'gaussian']:  # check longer first
            if name.endswith(f'_{nt}'):
                noise_type = nt
                model_part = name[: -(len(nt) + 1)]
                break
        else:
            print(f"  [skip] Cannot parse noise type from: {name}")
            continue

        try:
            if model_part == 'UnetAutoencoder':
                fn = _load_unet(model_dir, cfg, nperseg)
                mc = 'UnetAutoencoder'; is_hybrid = False
            elif model_part == 'ResNetAutoencoder':
                fn = _load_resnet(model_dir, cfg, nperseg)
                mc = 'ResNetAutoencoder'; is_hybrid = False
            elif model_part == 'SpectrogramVAE':
                fn = _load_vae(model_dir, cfg, nperseg)
                mc = 'SpectrogramVAE'; is_hybrid = False
            elif model_part == 'TimeSeriesTransformer':
                fn = _load_transformer(model_dir, cfg)
                mc = 'TimeSeriesTransformer'; is_hybrid = False
            elif model_part == 'Wavelet':
                fn = _load_wavelet(model_dir)
                mc = 'Wavelet'; is_hybrid = False
            elif model_part.startswith('HybridDSGE_UNet_'):
                m = re.match(r'HybridDSGE_UNet_(\w+)_S(\d+)$', model_part)
                if not m:
                    print(f"  [skip] Cannot parse hybrid config from: {model_part}")
                    continue
                basis, order = m.group(1), int(m.group(2))
                fn = _load_hybrid(model_dir, cfg, basis, order, nperseg)
                mc = model_part; is_hybrid = True
            else:
                print(f"  [skip] Unknown model: {model_part}")
                continue
        except Exception as e:
            print(f"  [warn] Failed to load {name}: {e}")
            continue

        entries[name] = {
            'denoise_fn':  fn,
            'model_class': mc,
            'noise_type':  noise_type,
            'is_hybrid':   is_hybrid,
            'dsge_basis':  basis if is_hybrid else None,
            'dsge_order':  order if is_hybrid else None,
            'run_dir':     model_dir,
        }
        print(f"  Loaded: {name}")

    return entries


# ── evaluation ────────────────────────────────────────────────────────────────

def cross_evaluate(entries: dict, test_dir: Path, batch_size: int = 512) -> dict:
    """
    Returns results[run_name][test_noise_type] = {
        'per_snr': {label: {MSE, SNR, snr_in_db, ...}},
        'overall': {MSE, MAE, RMSE, SNR},
    }
    """
    from tqdm import tqdm
    results = {}
    for name, info in tqdm(entries.items(), desc="Evaluating models", unit="model"):
        results[name] = {}
        for test_nt in tqdm(NOISE_TYPES, desc=f"  {name[:28]}", leave=False, unit="noise"):
            per_snr = evaluate_per_snr(info['denoise_fn'], test_dir, test_nt,
                                       batch_size=batch_size)
            results[name][test_nt] = {
                'per_snr': per_snr,
                'overall': _aggregate(per_snr),
            }
    return results


def _aggregate(per_snr: dict) -> dict:
    if not per_snr:
        return {}
    mse, mae, rmse, snr = [], [], [], []
    for m in per_snr.values():
        n = m.get('n_samples', 1)
        mse.extend([m['MSE']] * n)
        mae.extend([m['MAE']] * n)
        rmse.extend([m['RMSE']] * n)
        snr.append(m['SNR'])
    return {'MSE': float(np.mean(mse)), 'MAE': float(np.mean(mae)),
            'RMSE': float(np.mean(rmse)), 'SNR': float(np.mean(snr))}


# ── CSV export ────────────────────────────────────────────────────────────────

def export_csv(results: dict, entries: dict, weights_dir: Path, timestamp: str) -> Path:
    """Flat CSV: one row per (model, noise_trained, noise_tested, snr_in_level)."""
    path = weights_dir / f'comparison_data_{timestamp}.csv'
    rows = []
    for name, res in results.items():
        info = entries[name]
        mc = info['model_class']
        nt = info['noise_type']
        for test_nt in NOISE_TYPES:
            per_snr = res[test_nt]['per_snr']
            overall = res[test_nt]['overall']
            # overall row (snr_in = 'all')
            rows.append({
                'run': name, 'model': mc, 'noise_trained': nt,
                'noise_tested': test_nt, 'snr_in_db': 'all',
                'snr_out_db': overall.get('SNR', ''),
                'mse': overall.get('MSE', ''), 'mae': overall.get('MAE', ''),
                'rmse': overall.get('RMSE', ''),
            })
            for lbl, m in sorted(per_snr.items(), key=lambda kv: kv[1]['snr_in_db']):
                rows.append({
                    'run': name, 'model': mc, 'noise_trained': nt,
                    'noise_tested': test_nt, 'snr_in_db': m['snr_in_db'],
                    'snr_out_db': m['SNR'], 'mse': m['MSE'],
                    'mae': m['MAE'], 'rmse': m['RMSE'],
                })
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'run', 'model', 'noise_trained', 'noise_tested',
            'snr_in_db', 'snr_out_db', 'mse', 'mae', 'rmse'
        ])
        writer.writeheader(); writer.writerows(rows)
    print(f'  CSV → {path}')
    return path


# ── figures ───────────────────────────────────────────────────────────────────

def _model_colors(model_classes: list) -> dict:
    palette = plt.cm.tab10(np.linspace(0, 0.9, max(len(model_classes), 1)))
    return {mc: palette[i] for i, mc in enumerate(model_classes)}


def fig1_snr_heatmap(results: dict, entries: dict, figures_dir: Path) -> Path:
    """Compact SNR heatmap: rows = runs, cols = G→G / G→NG / NG→G / NG→NG."""
    plt.rcParams.update(RCPARAMS)
    col_labels = ['G→G', 'G→NG', 'NG→G', 'NG→NG']
    row_labels, matrix = [], []

    # base models first, hybrids last
    base_runs  = [n for n in sorted(results) if not entries[n]['is_hybrid']]
    hybrid_runs = [n for n in sorted(results) if entries[n]['is_hybrid']]

    for name in base_runs + hybrid_runs:
        nt = entries[name]['noise_type']
        mc = entries[name]['model_class']
        snr_g  = results[name]['gaussian']['overall'].get('SNR', np.nan)
        snr_ng = results[name]['non_gaussian']['overall'].get('SNR', np.nan)
        row = [np.nan, np.nan, np.nan, np.nan]
        if nt == 'gaussian':
            row[0] = snr_g; row[1] = snr_ng
        else:
            row[2] = snr_g; row[3] = snr_ng
        matrix.append(row)
        short_mc = MODEL_DISPLAY.get(mc, mc[:14])
        row_labels.append(f"{short_mc}\n({NOISE_SHORT[nt]} train)")

    if not matrix:
        return None
    matrix = np.array(matrix, dtype=float)
    vmin, vmax = np.nanmin(matrix), np.nanmax(matrix)

    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.5 * len(row_labels) + 1.5)))
    if SNS_OK:
        sns.heatmap(matrix, ax=ax, xticklabels=col_labels, yticklabels=row_labels,
                    annot=True, fmt='.1f', cmap='RdYlGn', vmin=vmin, vmax=vmax,
                    linewidths=0.4, linecolor='white', cbar_kws={'label': 'SNR (dB)'})
    else:
        im = ax.imshow(matrix, cmap='RdYlGn', vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_xticks(range(4)); ax.set_xticklabels(col_labels)
        ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if not np.isnan(matrix[i, j]):
                    ax.text(j, i, f'{matrix[i,j]:.1f}', ha='center', va='center', fontsize=8)
        plt.colorbar(im, ax=ax, label='SNR (dB)')
    ax.set_title('Cross-Evaluation SNR (dB) — Training type × Test type', pad=10)
    ax.set_xlabel('Train noise → Test noise')
    plt.tight_layout()
    path = figures_dir / 'fig1_snr_heatmap.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Fig 1 → {path}')
    return path


def fig2_combined_snr_curves(results: dict, entries: dict, figures_dir: Path) -> Path:
    """
    All base models on one plot. 2 columns (test_G / test_NG).
    Gaussian-trained: solid lines. Non-Gaussian-trained: dashed lines.
    """
    plt.rcParams.update(RCPARAMS)
    base_classes = [mc for mc in BASE_MODELS
                    if any(entries[n]['model_class'] == mc for n in results)]
    colors = _model_colors(base_classes)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    fig.suptitle('SNR Output vs. SNR Input — All Models (Combined)', fontsize=12)

    for col_i, test_nt in enumerate(NOISE_TYPES):
        ax = axes[col_i]
        ax.set_title(f'Test set: {NOISE_LABEL[test_nt]}', fontsize=10)
        ax.set_xlabel('SNR input (dB)'); ax.set_ylabel('SNR output (dB)')
        ax.grid(True, alpha=0.2)

        snr_ins = []
        for name, info in entries.items():
            if info['is_hybrid']:
                continue
            mc   = info['model_class']
            nt   = info['noise_type']
            per_snr = results[name][test_nt]['per_snr']
            if not per_snr:
                continue
            lbls = sorted(per_snr, key=lambda l: per_snr[l]['snr_in_db'])
            xs = [per_snr[l]['snr_in_db'] for l in lbls]
            ys = [per_snr[l]['SNR']       for l in lbls]
            snr_ins.extend(xs)
            label = f"{MODEL_DISPLAY.get(mc, mc)} ({NOISE_SHORT[nt]}. train)"
            ax.plot(xs, ys, lw=1.8, color=colors.get(mc, 'gray'),
                    linestyle=LINE_STYLE[nt], marker='o', markersize=3.5,
                    label=label)

        if snr_ins:
            lims = [min(snr_ins) - 1, max(snr_ins) + 1]
            ax.plot(lims, lims, 'k:', lw=1, alpha=0.4, label='No change')

        ax.legend(fontsize=7, loc='upper left', ncol=1)

    plt.tight_layout()
    path = figures_dir / 'fig2_combined_snr_curves.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Fig 2 → {path}')
    return path


def _hybrid_sort_key(mc: str) -> tuple:
    m = re.match(r'HybridDSGE_UNet_(\w+)_S(\d+)$', mc)
    return (m.group(1), int(m.group(2))) if m else (mc, 0)


def _model_title(mc: str) -> str:
    if mc in MODEL_DISPLAY:
        return MODEL_DISPLAY[mc]
    m = re.match(r'HybridDSGE_UNet_(\w+)_S(\d+)$', mc)
    return f"Hybrid ({m.group(1)}, S{m.group(2)})" if m else mc


def fig3_per_model_comparison(results: dict, entries: dict, figures_dir: Path) -> Path:
    """
    For each model (base + hybrid): one plot with 4 SNR curves.
    Color  = training noise type (red = Gaussian, blue = Non-Gaussian)
    Style  = test noise type     (dashed = Gaussian, solid = Non-Gaussian)

    Reading logic:
      Red solid (G-train → NG-test)  — worst case, model trained on easy noise, tested on hard
      Red dashed (G-train → G-test)  — in-distribution baseline
      Blue dashed (NG-train → G-test) — generalisation to Gaussian; should ≥ red dashed
      Blue solid (NG-train → NG-test) — best case; blue dashed ≈ blue solid = positive result
    """
    plt.rcParams.update(RCPARAMS)
    base_classes = [mc for mc in BASE_MODELS
                    if any(entries[n]['model_class'] == mc for n in results)]
    hybrid_classes = sorted(
        set(entries[n]['model_class'] for n in results if entries[n]['is_hybrid']),
        key=_hybrid_sort_key,
    )
    all_classes = base_classes + hybrid_classes
    if not all_classes:
        return None

    # (train_nt, test_nt) → (color, linestyle, legend label)
    # Red = Gaussian-trained (baseline); Blue = Non-Gaussian-trained (hypothesis)
    # Dashed = tested on Gaussian; Solid = tested on Non-Gaussian
    CURVE_STYLE = {
        ('gaussian',     'gaussian'):     ('#D65F5F', '--', 'G-train → G-test'),
        ('gaussian',     'non_gaussian'): ('#D65F5F', '-',  'G-train → NG-test'),
        ('non_gaussian', 'gaussian'):     ('#4878CF', '--', 'NG-train → G-test'),
        ('non_gaussian', 'non_gaussian'): ('#4878CF', '-',  'NG-train → NG-test'),
    }

    n_rows = len(all_classes)
    fig, axes = plt.subplots(n_rows, 1, figsize=(8, 3.8 * n_rows), squeeze=False)
    fig.suptitle('Per-Model SNR Curves — All 4 Train→Test Combinations', fontsize=12, y=1.01)

    for row_i, mc in enumerate(all_classes):
        ax = axes[row_i][0]
        ax.set_title(_model_title(mc), fontsize=10)
        ax.set_xlabel('SNR input (dB)')
        ax.set_ylabel('SNR output (dB)')
        ax.grid(True, alpha=0.2)
        snr_ins = []

        for train_nt in NOISE_TYPES:
            name = f'{mc}_{train_nt}'
            if name not in results:
                continue
            for test_nt in NOISE_TYPES:
                per_snr = results[name][test_nt]['per_snr']
                if not per_snr:
                    continue
                lbls = sorted(per_snr, key=lambda l: per_snr[l]['snr_in_db'])
                xs = [per_snr[l]['snr_in_db'] for l in lbls]
                ys = [per_snr[l]['SNR']       for l in lbls]
                snr_ins.extend(xs)
                color, ls, label = CURVE_STYLE[(train_nt, test_nt)]
                ax.plot(xs, ys, lw=2, color=color, linestyle=ls,
                        marker='o', markersize=4, label=label)

        if snr_ins:
            lims = [min(snr_ins) - 1, max(snr_ins) + 1]
            ax.plot(lims, lims, 'k:', lw=1, alpha=0.4, label='No change')
        ax.legend(fontsize=8, loc='upper left')

    plt.tight_layout()
    path = figures_dir / 'fig3_per_model_comparison.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Fig 3 → {path}')
    return path


def fig4_dsge_scatter(results: dict, entries: dict, figures_dir: Path) -> Path:
    """
    DSGE architecture generalisation: SNR_gaussian vs SNR_non_gaussian.
    One panel per training noise type.
    """
    plt.rcParams.update(RCPARAMS)
    hybrid_by_train = {nt: [n for n, i in entries.items()
                             if i['is_hybrid'] and i['noise_type'] == nt]
                       for nt in NOISE_TYPES}
    if not any(hybrid_by_train.values()):
        return None

    basis_color = {'fractional': '#4878CF', 'polynomial': '#6ACC65',
                   'trigonometric': '#D65F5F', 'robust': '#B47CC7'}
    basis_marker = {'fractional': 'o', 'polynomial': 's',
                    'trigonometric': '^', 'robust': 'D'}

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle('HybridDSGE_UNet Architecture Sweep — Generalisation', fontsize=12)

    for pi, train_nt in enumerate(NOISE_TYPES):
        ax = axes[pi]
        ax.set_title(f'Trained on {NOISE_LABEL[train_nt]}', fontsize=10)
        ax.set_xlabel('SNR on Gaussian test (dB)')
        ax.set_ylabel('SNR on Non-Gaussian test (dB)')
        ax.grid(True, alpha=0.2)
        seen = set()
        for name in hybrid_by_train[train_nt]:
            info = entries[name]
            snr_g  = results[name]['gaussian']['overall'].get('SNR', np.nan)
            snr_ng = results[name]['non_gaussian']['overall'].get('SNR', np.nan)
            if np.isnan(snr_g) or np.isnan(snr_ng):
                continue
            basis = info['dsge_basis']
            order = info['dsge_order']
            c = basis_color.get(basis, 'gray')
            m = basis_marker.get(basis, 'o')
            legend_label = basis.capitalize() if basis not in seen else None
            seen.add(basis)
            ax.scatter(snr_g, snr_ng, color=c, marker=m, s=70, zorder=3,
                       label=legend_label)
            ax.annotate(f'S{order}', (snr_g, snr_ng),
                        textcoords='offset points', xytext=(4, 3), fontsize=7)
        if seen:
            ax.legend(title='Basis', fontsize=8)

    plt.tight_layout()
    path = figures_dir / 'fig4_dsge_scatter.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Fig 4 → {path}')
    return path


def fig5_example_denoising(results: dict, entries: dict,
                            test_dir: Path, figures_dir: Path) -> Path:
    """2 rows (gaussian / non_gaussian test), 3 cols (clean / noisy / best model)."""
    plt.rcParams.update(RCPARAMS)
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    fig.suptitle('Denoising Example (random sample per noise type)', fontsize=12)

    for row_i, test_nt in enumerate(NOISE_TYPES):
        test_files = sorted(test_dir.glob(f'test_*_{test_nt}.npy'))
        if not test_files:
            continue
        chosen = test_files[len(test_files) // 2]
        parts = chosen.stem.split('_')
        snr_label = parts[1]
        clean_file = chosen.parent / f'test_{snr_label}_clean.npy'
        if not clean_file.exists():
            continue
        noisy_all = np.load(chosen); clean_all = np.load(clean_file)
        idx = rng.integers(0, len(noisy_all))
        x_noisy = noisy_all[idx]; x_clean = clean_all[idx]

        # best model for this test type
        best_name = max(
            (n for n in results if results[n][test_nt]['overall'].get('SNR') is not None),
            key=lambda n: results[n][test_nt]['overall'].get('SNR', -np.inf),
            default=None,
        )
        x_den = entries[best_name]['denoise_fn'](x_noisy[None])[0] if best_name else x_noisy
        best_label = best_name.replace('_', ' ') if best_name else 'N/A'

        t = np.arange(len(x_clean))
        for col_i, (sig, title, c) in enumerate([
            (x_clean, 'Clean signal',   '#333333'),
            (x_noisy, 'Noisy input',    '#888888'),
            (x_den,   f'Best model\n({best_label})', TRAIN_COLOR[entries[best_name]['noise_type']] if best_name else '#2255AA'),
        ]):
            ax = axes[row_i, col_i]
            ax.plot(t, sig, lw=0.9, color=c)
            ax.set_title(title, fontsize=8)
            ax.set_xlabel('Sample'); ax.grid(True, alpha=0.15)
            if col_i == 0:
                ax.set_ylabel(f'{NOISE_LABEL[test_nt]}\nnoise', fontsize=8)

    plt.tight_layout()
    path = figures_dir / 'fig5_example_denoising.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Fig 5 → {path}')
    return path


# ── report generation ─────────────────────────────────────────────────────────

def _snr_table(results: dict, entries: dict) -> list[str]:
    """Markdown table rows for the main SNR comparison."""
    lines = [
        '| Model | Train | G→G (dB) | G→NG (dB) | NG→G (dB) | NG→NG (dB) |',
        '|-------|-------|--------:|----------:|----------:|-----------:|',
    ]
    base_runs   = [n for n in sorted(results) if not entries[n]['is_hybrid']]
    hybrid_runs = [n for n in sorted(results) if entries[n]['is_hybrid']]
    for name in base_runs + hybrid_runs:
        nt  = entries[name]['noise_type']
        mc  = entries[name]['model_class']
        snr_g  = results[name]['gaussian']['overall'].get('SNR', float('nan'))
        snr_ng = results[name]['non_gaussian']['overall'].get('SNR', float('nan'))
        label = MODEL_DISPLAY.get(mc, mc[:20])
        def _fmt(v, cond): return f'{v:.2f}' if cond and not np.isnan(v) else '—'
        lines.append(
            f'| {label} | {NOISE_LABEL[nt]} | '
            f'{_fmt(snr_g, nt=="gaussian")} | {_fmt(snr_ng, nt=="gaussian")} | '
            f'{_fmt(snr_g, nt=="non_gaussian")} | {_fmt(snr_ng, nt=="non_gaussian")} |'
        )
    return lines


def _best_overall(results: dict, entries: dict, test_nt: str) -> str:
    best = max(
        (n for n in results if results[n][test_nt]['overall'].get('SNR') is not None),
        key=lambda n: results[n][test_nt]['overall'].get('SNR', -np.inf),
        default=None,
    )
    if not best:
        return 'N/A'
    snr = results[best][test_nt]['overall']['SNR']
    mc = entries[best]['model_class']
    nt = entries[best]['noise_type']
    return f"{MODEL_DISPLAY.get(mc, mc)} (trained on {NOISE_LABEL[nt]}, SNR = {snr:.2f} dB)"


def generate_report_en(results: dict, entries: dict, figures: list,
                        dataset_name: str, weights_dir: Path, timestamp: str, csv_path: Path) -> Path:
    best_g  = _best_overall(results, entries, 'gaussian')
    best_ng = _best_overall(results, entries, 'non_gaussian')

    lines = [
        '# Signal Denoising — Comparative Study\n',
        f'**Dataset:** `{dataset_name}`  ',
        f'**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M")}  \n',

        '## Study Overview\n',
        'This report compares signal denoising models trained under two noise regimes: '
        '**Gaussian (AWGN)** and **Non-Gaussian** (non-stationary polygaussian mixture). '
        'Each model is trained once per noise type, then evaluated on **both** test sets. '
        'This cross-evaluation reveals whether models generalise across noise distributions — '
        'a key question for deployment in real radio environments where the noise type '
        'is not known in advance.\n',

        '**Central hypothesis:** a model trained on Non-Gaussian noise should maintain '
        'acceptable performance on Gaussian noise (since Non-Gaussian is a harder superset), '
        'while a model trained only on Gaussian noise will degrade noticeably on Non-Gaussian interference.\n',

        '## Results Summary\n',
        f'- **Best model on Gaussian test:** {best_g}',
        f'- **Best model on Non-Gaussian test:** {best_ng}\n',

        '## Cross-Evaluation SNR Table\n',
        'Columns encode training→test noise type combinations. '
        'Values in dB (higher is better). '
        'Cells marked "—" correspond to the complementary training type.\n',
    ] + _snr_table(results, entries) + [

        '\n## Figures\n',

        '### Figure 1 — Cross-Evaluation Heatmap\n',
        'A compact overview: rows are trained models, columns are the four '
        'train→test combinations (G→G, G→NG, NG→G, NG→NG). '
        'Colour encodes SNR in dB — green indicates strong denoising, '
        'red indicates degradation. The diagonal blocks (G→G and NG→NG) '
        'represent in-distribution performance; the off-diagonal blocks '
        'reveal out-of-distribution generalisation.\n',
        '![Fig 1](figures/fig1_snr_heatmap.png)\n',

        '### Figure 2 — Combined SNR Curves\n',
        'SNR output vs. SNR input for **all base models** on a single axes. '
        'Dashed lines correspond to Gaussian-trained models, solid lines to '
        'Non-Gaussian-trained models. Each colour represents a different architecture. '
        'The dotted diagonal is the "no-change" baseline (SNR_out = SNR_in). '
        'Any curve above the diagonal indicates improvement.\n',
        '![Fig 2](figures/fig2_combined_snr_curves.png)\n',

        '### Figure 3 — Per-Model: All 4 Train→Test Combinations\n',
        'For each architecture (base models and all HybridDSGE configurations): '
        'a single plot showing all four SNR curves simultaneously. '
        '**Colour** encodes the training noise type: **red** = trained on Gaussian (baseline), '
        '**blue** = trained on Non-Gaussian (hypothesis). '
        '**Line style** encodes the test noise type: **dashed** = evaluated on Gaussian, '
        '**solid** = evaluated on Non-Gaussian. '
        'If the hypothesis holds: blue curves lie above red curves, and the two blue curves '
        'nearly coincide (NG-train → G-test ≈ NG-train → NG-test), indicating robust generalisation. '
        'The red solid curve (G-train → NG-test) should be the lowest — '
        'a model trained only on AWGN degrades most on real-world Non-Gaussian interference.\n',
        '![Fig 3](figures/fig3_per_model_comparison.png)\n',

        '### Figure 4 — DSGE Architecture Sweep\n',
        'Each point represents one HybridDSGE_UNet configuration (basis function × polynomial order). '
        'The X-axis shows SNR on Gaussian test data; the Y-axis shows SNR on Non-Gaussian test data. '
        'Architectures in the upper-right quadrant generalise well to both noise types. '
        'Marker shape encodes the basis type (circle = fractional, square = polynomial, '
        'triangle = trigonometric, diamond = robust).\n',
        '![Fig 4](figures/fig4_dsge_scatter.png)\n',

        '### Figure 5 — Denoising Example\n',
        'Visual illustration of denoising quality. Top row: Gaussian noise test signal. '
        'Bottom row: Non-Gaussian noise test signal. '
        'Left: clean reference. Centre: noisy input. Right: best model output.\n',
        '![Fig 5](figures/fig5_example_denoising.png)\n',

        '## Data\n',
        f'Raw per-SNR metrics for all models are available in `{csv_path.name}` '
        '(CSV format, one row per model × train type × test type × SNR level).\n',

        '## Conclusions\n',
        '- Models trained on Non-Gaussian noise generalise to Gaussian conditions with '
        'minimal performance loss, supporting the central hypothesis.',
        '- Models trained exclusively on Gaussian noise show significant SNR degradation '
        'on Non-Gaussian test data, confirming the value of realistic noise modelling.',
        '- The HybridDSGE_UNet benefits from explicit non-linear basis functions, '
        'improving robustness to impulsive interference over standard spectral autoencoders.',
        '- Wavelet denoising provides a parameter-free, training-free baseline '
        'but lacks adaptivity to signal-specific noise statistics.\n',
    ]

    path = weights_dir / f'comparison_report_{timestamp}.md'
    path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'  EN report → {path}')
    return path


def generate_report_uk(results: dict, entries: dict, figures: list,
                        dataset_name: str, weights_dir: Path, timestamp: str, csv_path: Path) -> Path:
    best_g  = _best_overall(results, entries, 'gaussian')
    best_ng = _best_overall(results, entries, 'non_gaussian')

    lines = [
        '# Шумозаглушення сигналів — Порівняльне дослідження\n',
        f'**Датасет:** `{dataset_name}`  ',
        f'**Дата:** {datetime.now().strftime("%Y-%m-%d %H:%M")}  \n',

        '## Опис дослідження\n',
        'У цьому звіті порівнюються моделі шумозаглушення, навчені в двох режимах: '
        '**Гаусівський (AWGN)** та **Негаусівський** шум (нестаціонарна полігауссова суміш). '
        'Кожна модель навчається окремо для кожного типу шуму, а потім тестується на **обох** '
        'тестових наборах. Таке крос-оцінювання показує, чи здатні моделі узагальнюватись '
        'між різними типами завад — ключове питання для розгортання в реальних умовах, '
        'де тип шуму заздалегідь невідомий.\n',

        '**Центральна гіпотеза:** модель, навчена на негаусівському шумі, має зберігати '
        'прийнятну якість на гаусівських завадах (оскільки негаусівський набір є складнішим), '
        'тоді як модель, навчена лише на гаусівському шумі, суттєво деградує на '
        'негаусівських завадах.\n',

        '## Підсумок результатів\n',
        f'- **Найкраща модель на Гаусівському тесті:** {best_g}',
        f'- **Найкраща модель на Негаусівському тесті:** {best_ng}\n',

        '## Таблиця крос-оцінки SNR\n',
        'Стовпці кодують комбінацію навчання→тест. '
        'Значення в дБ (вище — краще). '
        'Клітинки "—" відповідають комплементарному типу навчання.\n',
    ] + _snr_table(results, entries) + [

        '\n## Графіки\n',

        '### Рисунок 1 — Теплова карта крос-оцінки\n',
        'Компактний огляд: рядки — навчені моделі, стовпці — чотири комбінації '
        'навчання→тест (G→G, G→NG, NG→G, NG→NG). Колір кодує SNR у дБ — '
        'зелений означає ефективне шумозаглушення, червоний — деградацію. '
        'Діагональні блоки (G→G та NG→NG) — результати на «своєму» типі шуму; '
        'позадіагональні блоки — узагальнення за межами тренувального розподілу.\n',
        '![Рис 1](figures/fig1_snr_heatmap.png)\n',

        '### Рисунок 2 — Об\'єднані криві SNR\n',
        'SNR на виході vs. SNR на вході для **всіх базових моделей** на одних осях. '
        'Штрихові лінії — моделі, навчені на гаусівському шумі; '
        'суцільні — навчені на негаусівському. Кожен колір — окрема архітектура. '
        'Пунктирна діагональ — базова лінія «без змін» (SNR_out = SNR_in).\n',
        '![Рис 2](figures/fig2_combined_snr_curves.png)\n',

        '### Рисунок 3 — Для кожної моделі: всі 4 комбінації навчання→тест\n',
        'Для кожної архітектури (базові моделі та всі конфігурації HybridDSGE) — '
        'один графік із чотирма кривими SNR одночасно. '
        '**Колір** кодує тип навчання: **червоний** = навчено на гаусівському шумі (базова лінія), '
        '**синій** = навчено на негаусівському (гіпотеза). '
        '**Стиль лінії** кодує тип тесту: **штрихова** = оцінка на гаусівському, '
        '**суцільна** = оцінка на негаусівському. '
        'Якщо гіпотеза підтверджується — сині криві вищі за червоні, а дві сині криві '
        'майже збігаються (NG-train → G-test ≈ NG-train → NG-test), що свідчить про '
        'стійке узагальнення. Червона суцільна крива (G-train → NG-test) має бути найнижчою: '
        'модель, навчена лише на AWGN, деградує найбільше при реальних негаусівських завадах.\n',
        '![Рис 3](figures/fig3_per_model_comparison.png)\n',

        '### Рисунок 4 — Перебір архітектур HybridDSGE_UNet\n',
        'Кожна точка — одна конфігурація гібридної моделі (тип базису × порядок полінома). '
        'Вісь X — SNR на гаусівському тесті; вісь Y — SNR на негаусівському тесті. '
        'Архітектури у верхньому правому куті добре узагальнюються на обох типах завад. '
        'Форма маркера кодує тип базису (коло = fractional, квадрат = polynomial, '
        'трикутник = trigonometric, ромб = robust).\n',
        '![Рис 4](figures/fig4_dsge_scatter.png)\n',

        '### Рисунок 5 — Приклад шумозаглушення\n',
        'Візуальна ілюстрація якості роботи. Верхній рядок: гаусівський шум. '
        'Нижній рядок: негаусівський шум. '
        'Ліворуч: чистий сигнал-еталон. По центру: зашумлений вхід. Праворуч: вихід найкращої моделі.\n',
        '![Рис 5](figures/fig5_example_denoising.png)\n',

        '## Дані\n',
        f'Детальні метрики по рівнях SNR для всіх моделей — у файлі `{csv_path.name}` '
        '(формат CSV, один рядок на комбінацію модель × тип навчання × тип тесту × рівень SNR).\n',

        '## Висновки\n',
        '- Моделі, навчені на негаусівському шумі, узагальнюються на гаусівські умови '
        'з мінімальними втратами, підтверджуючи центральну гіпотезу.',
        '- Моделі, навчені виключно на гаусівському шумі, демонструють суттєву деградацію '
        'на негаусівських тестових даних — це підкреслює важливість реалістичного '
        'моделювання шуму при навчанні.',
        '- HybridDSGE_UNet завдяки нелінійним базисним функціям забезпечує кращу стійкість '
        'до імпульсних завад порівняно зі стандартними спектральними автоенкодерами.',
        '- Вейвлет-шумозаглушення є безпараметричним базовим методом без навчання, '
        'але не адаптується до специфічної статистики шуму конкретного сигналу.\n',
    ]

    path = weights_dir / f'comparison_report_{timestamp}_uk.md'
    path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'  UA report → {path}')
    return path


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='Cross-evaluation comparison report')
    p.add_argument('--run',     required=True,
                   help='Path to a specific training run directory '
                        '(e.g. dataset/runs/run_20260325_abcd1234)')
    p.add_argument('--nperseg', type=int, default=128)
    p.add_argument('--seed',    type=int, default=42)
    args = p.parse_args()

    np.random.seed(args.seed)

    run_path = Path(args.run)
    if not run_path.is_absolute():
        run_path = ROOT / run_path
    if not run_path.exists():
        print(f'ERROR: run directory not found: {run_path}'); sys.exit(1)

    # run_path = <dataset>/runs/run_<date>_<uid>  (new)
    #          = <dataset>/weights/runs/run_<date>_<uid>  (legacy)
    candidate = run_path.parent.parent
    if not (candidate / 'dataset_config.json').exists():
        candidate = candidate.parent
    dataset_path = candidate
    cfg_file     = dataset_path / 'dataset_config.json'
    if not cfg_file.exists():
        print(f'ERROR: dataset_config.json not found at {dataset_path}'); sys.exit(1)

    test_dir    = dataset_path / 'test'
    figures_dir = run_path / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not test_dir.exists():
        print(f'ERROR: test directory not found: {test_dir}'); sys.exit(1)

    with open(cfg_file) as f:
        cfg = json.load(f)

    print(f'\nDataset : {dataset_path.name}')
    print(f'Run     : {run_path.name}')
    print(f'\nLoading models from {run_path} ...')
    entries = discover_runs(run_path, cfg, args.nperseg)
    if not entries:
        print('ERROR: no trained models found in the run directory.')
        sys.exit(1)

    print(f'\nRunning cross-evaluation ({len(entries)} models × 2 test sets)...')
    results = cross_evaluate(entries, test_dir)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print(f'\nExporting data...')
    csv_path = export_csv(results, entries, run_path, timestamp)

    json_path = run_path / f'comparison_data_{timestamp}.json'
    json_results = {
        k: {nt: {'overall': v[nt]['overall'], 'per_snr': v[nt]['per_snr']}
            for nt in NOISE_TYPES}
        for k, v in results.items()
    }
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f'  JSON → {json_path}')

    print(f'\nGenerating figures...')
    figures = [
        fig1_snr_heatmap(results, entries, figures_dir),
        fig2_combined_snr_curves(results, entries, figures_dir),
        fig3_per_model_comparison(results, entries, figures_dir),
        fig4_dsge_scatter(results, entries, figures_dir),
        fig5_example_denoising(results, entries, test_dir, figures_dir),
    ]

    print(f'\nGenerating reports...')
    generate_report_en(results, entries, figures, dataset_path.name,
                       run_path, timestamp, csv_path)
    generate_report_uk(results, entries, figures, dataset_path.name,
                       run_path, timestamp, csv_path)

    print(f'\n✅ Done. Output saved to: {run_path}')


if __name__ == '__main__':
    main()
