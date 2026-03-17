#!/usr/bin/env python3
"""
Inference script for HybridDSGE_UNet.

Evaluates the trained HybridDSGE_UNet model on per-SNR test files and
visualises one example signal.

Usage:
    python inference/inference_hybrid.py \
        --dataset data_generation/datasets/deep_space_polygauss_nonstationary_bpsk_bs256_n50000_39075e4f \
        --noise-type non_gaussian \
        --dsge-order 3 \
        --sample-index 0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import stft, istft

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.hybrid_unet import HybridDSGE_UNet
from models.dsge_layer import DSGEFeatureExtractor
from metrics import MeanSquaredError, MeanAbsoluteError, RootMeanSquaredError, SignalToNoiseRatio


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MSE":  MeanSquaredError.calculate(y_true, y_pred),
        "MAE":  MeanAbsoluteError.calculate(y_true, y_pred),
        "RMSE": RootMeanSquaredError.calculate(y_true, y_pred),
        "SNR":  SignalToNoiseRatio.calculate(y_true, y_pred),
    }


def build_4ch_input(
    signal_batch: np.ndarray,
    dsge: DSGEFeatureExtractor,
    fs: int,
    nperseg: int,
    noverlap: int,
    dsge_order: int,
    device: torch.device,
) -> torch.Tensor:
    """[N, T] → (B, 1+S, F, T') with per-channel DSGE normalisation."""
    stft_mags = []
    dsge_mags_list = [[] for _ in range(dsge_order)]

    for s in signal_batch:
        _, _, Zxx = stft(s, fs=fs, nperseg=nperseg, noverlap=noverlap)
        stft_mags.append(np.abs(Zxx))
        dsge_specs = dsge.compute_dsge_spectrograms(s)  # [S, F, T']
        for i in range(dsge_order):
            dsge_mags_list[i].append(dsge_specs[i])

    stft_stack = np.stack(stft_mags)  # [B, F, T']
    stft_ref_max = stft_stack.max() + 1e-8
    channels = [stft_stack]

    for i in range(dsge_order):
        ch = np.stack(dsge_mags_list[i])
        ch_max = ch.max() + 1e-8
        channels.append(ch * (stft_ref_max / ch_max))

    return torch.tensor(
        np.stack(channels, axis=1), dtype=torch.float32
    ).to(device)


def denoise_signals(
    noisy: np.ndarray,
    model: HybridDSGE_UNet,
    dsge: DSGEFeatureExtractor,
    fs: int,
    nperseg: int,
    noverlap: int,
    dsge_order: int,
    signal_len: int,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Denoise a batch of signals: STFT → mask → ISTFT."""
    model.eval()
    reconstructed = []

    for start in range(0, len(noisy), batch_size):
        batch = noisy[start: start + batch_size]

        phases = []
        for s in batch:
            _, _, Zxx = stft(s, fs=fs, nperseg=nperseg, noverlap=noverlap)
            phases.append(np.angle(Zxx))

        x4 = build_4ch_input(batch, dsge, fs, nperseg, noverlap, dsge_order, device)
        noisy_mag = x4[:, 0, :, :].cpu().numpy()

        with torch.no_grad():
            out_mag = model(x4).squeeze(1).cpu().numpy() * noisy_mag

        for mag, phase in zip(out_mag, phases):
            _, r = istft(mag * np.exp(1j * phase), fs=fs,
                         nperseg=nperseg, noverlap=noverlap)
            r = r[:signal_len] if len(r) >= signal_len \
                else np.pad(r, (0, signal_len - len(r)))
            reconstructed.append(r.astype(np.float32))

    return np.stack(reconstructed)


def parse_args():
    p = argparse.ArgumentParser(description="Inference for HybridDSGE_UNet")
    p.add_argument("--dataset", required=True,
                   help="Path to dataset folder")
    p.add_argument("--noise-type", default="non_gaussian",
                   choices=["gaussian", "non_gaussian"])
    p.add_argument("--dsge-order",  type=int, default=3)
    p.add_argument("--nperseg",     type=int, default=None,
                   help="STFT window (default: read from config or 32)")
    p.add_argument("--sample-index", type=int, default=0,
                   help="Index within the first SNR test file to visualise")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip the visualisation plot")
    return p.parse_args()


def main():
    args = parse_args()

    dataset_dir = Path(args.dataset)
    if not dataset_dir.is_absolute():
        dataset_dir = ROOT / dataset_dir
    if not dataset_dir.exists():
        print(f"ERROR: dataset not found: {dataset_dir}")
        sys.exit(1)

    with open(dataset_dir / "dataset_config.json") as f:
        cfg = json.load(f)

    signal_len = cfg["block_size"]
    fs         = cfg["sample_rate"]
    nperseg    = args.nperseg or 32
    noverlap   = nperseg // 2

    weights_dir = dataset_dir / "weights"
    model_path  = weights_dir / f"HybridDSGE_UNet_{args.noise_type}_S{args.dsge_order}_best.pth"
    dsge_path   = weights_dir / f"dsge_state_{args.noise_type}_S{args.dsge_order}.npz"

    if not model_path.exists():
        print(f"ERROR: weights not found: {model_path}")
        print("Run training first:")
        print(f"  python train/training_hybrid.py --dataset {args.dataset} "
              f"--noise-type {args.noise_type} --dsge-order {args.dsge_order}")
        sys.exit(1)

    if not dsge_path.exists():
        print(f"ERROR: DSGE state not found: {dsge_path}")
        sys.exit(1)

    print(f"Dataset   : {dataset_dir.name}")
    print(f"Config    : block_size={signal_len}, sample_rate={fs}")
    print(f"Model     : {model_path.name}")
    print(f"DSGE state: {dsge_path.name}")

    # ── Load DSGE state ────────────────────────────────────────
    dsge = DSGEFeatureExtractor.load_state(
        str(dsge_path),
        basis_type="fractional",
        stft_params={"nperseg": nperseg, "noverlap": noverlap, "fs": fs},
    )
    print(f"DSGE      : {dsge}")

    # ── Load model ─────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device    : {device}\n")

    # Determine input_shape from one signal
    test_snr_files = sorted((dataset_dir / "test").glob(f"test_*_{args.noise_type}.npy"))
    if not test_snr_files:
        print(f"ERROR: no test files matching test_*_{args.noise_type}.npy in {dataset_dir/'test'}")
        sys.exit(1)

    probe_signal = np.load(test_snr_files[0])[0]
    _, _, Zxx = stft(probe_signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    input_shape = np.abs(Zxx).shape

    model = HybridDSGE_UNet(input_shape=input_shape, dsge_order=args.dsge_order).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model params: {model.param_count():,}")

    # ── Evaluate per SNR ───────────────────────────────────────
    print("\n=== Per-SNR metrics ===")
    header = f"{'SNR label':<12} {'MSE':>10} {'MAE':>10} {'RMSE':>10} {'SNR_out':>10}"
    print(header)
    print("-" * len(header))

    first_snr_label = None
    first_noisy = None
    first_clean = None
    first_denoised = None

    for noisy_path in test_snr_files:
        # derive SNR label: test_m10dB_non_gaussian.npy → m10dB
        stem_parts = noisy_path.stem.split("_")
        snr_label = stem_parts[1]

        clean_path = noisy_path.parent / f"test_{snr_label}_clean.npy"
        if not clean_path.exists():
            continue

        noisy_arr = np.load(noisy_path)
        clean_arr = np.load(clean_path)

        denoised = denoise_signals(
            noisy_arr, model, dsge, fs, nperseg, noverlap,
            args.dsge_order, signal_len, device,
        )
        m = compute_metrics(clean_arr, denoised)
        print(f"{snr_label:<12} {m['MSE']:>10.6f} {m['MAE']:>10.6f} "
              f"{m['RMSE']:>10.6f} {m['SNR']:>10.2f} dB")

        if first_snr_label is None:
            first_snr_label = snr_label
            first_noisy    = noisy_arr
            first_clean    = clean_arr
            first_denoised = denoised

    # ── Visualisation ──────────────────────────────────────────
    if not args.no_plot and first_noisy is not None:
        idx = min(args.sample_index, len(first_noisy) - 1)
        t = np.arange(signal_len) / fs

        plt.figure(figsize=(12, 5))
        plt.plot(t, first_clean[idx],    label="Clean",    linewidth=2)
        plt.plot(t, first_noisy[idx],    label="Noisy",    alpha=0.5)
        plt.plot(t, first_denoised[idx], label="Denoised", linestyle="--", linewidth=2)
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.title(f"HybridDSGE_UNet | {args.noise_type} | SNR={first_snr_label} | sample #{idx}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
