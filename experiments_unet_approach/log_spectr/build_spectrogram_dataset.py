#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build spectrogram dataset for U-Net mask prediction:
- Loads clean and noisy time-domain datasets
- Computes STFT with reflect padding
- Saves log-magnitude inputs (X = log1p(|STFT(noisy)|)),
        log-magnitude targets (Y = log1p(|STFT(clean)|)),
        ideal ratio mask (mask = clip(|C|/(|N|+eps), 0..1)),
        and phases for TEST set (for later ISTFT)
- Uses a fixed 70/15/15 split with seed for reproducibility
"""

import os
import json
import argparse
import numpy as np
from scipy.signal import stft

# -----------------------
# Defaults (your config)
# -----------------------
DEFAULTS = dict(
    dataset_type="non_gaussian",     # or "non_gaussian"
    signal_len=2144,
    fs=1024,
    nperseg=128,
    noverlap=96,
    random_state=42,
    input_dir="../../dataset",
    out_dir="../../spectro_dataset",
    window="hann",
    eps=1e-8,
)

# -----------------------
# STFT helpers
# -----------------------
def stft_mag_phase_1d(x, fs, nperseg, noverlap, window="hann"):
    """
    STFT with reflect padding so that model/inference is consistent.
    Returns (mag, phase).
    """
    pad = nperseg // 2
    x_pad = np.pad(x, pad, mode="reflect")
    # boundary=None & padded=False ← максимально контрольована форма
    f, t, Zxx = stft(
        x_pad,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        window=window,
        boundary=None,
        padded=False,
    )
    mag = np.abs(Zxx).astype(np.float32)
    phase = np.angle(Zxx).astype(np.float32)
    return mag, phase


def build_for_arrays(noisy_arr, clean_arr, fs, nperseg, noverlap, window, eps):
    """
    Takes time-domain arrays [N, T] and returns:
      X_log [N, F, T'], Y_log [N, F, T'], mask [N, F, T'], (phases [N, F, T'])
    """
    assert noisy_arr.shape == clean_arr.shape, "noisy/clean shapes must match"
    N, T = noisy_arr.shape

    X_list, Y_list, M_list, PH_list = [], [], [], []

    for i in range(N):
        noisy = noisy_arr[i]
        clean = clean_arr[i]

        noisy_mag, noisy_phase = stft_mag_phase_1d(noisy, fs, nperseg, noverlap, window)
        clean_mag, _          = stft_mag_phase_1d(clean, fs, nperseg, noverlap, window)

        # Targets & inputs
        X_log = np.log1p(noisy_mag)         # model input
        Y_log = np.log1p(clean_mag)         # reconstruction/log target
        mask  = np.clip(clean_mag / (noisy_mag + eps), 0.0, 1.0)  # IRM mask target

        X_list.append(X_log)
        Y_list.append(Y_log)
        M_list.append(mask)
        PH_list.append(noisy_phase)

    X = np.stack(X_list, axis=0)   # [N, F, T']
    Y = np.stack(Y_list, axis=0)   # [N, F, T']
    M = np.stack(M_list, axis=0)   # [N, F, T']
    PH = np.stack(PH_list, axis=0) # [N, F, T']
    return X, Y, M, PH


def split_indices(N, random_state=42):
    """70/15/15 split with a fixed seed."""
    rng = np.random.default_rng(random_state)
    idx = np.arange(N)
    rng.shuffle(idx)
    n_val = int(0.15 * N)
    n_test = int(0.15 * N)
    n_train = N - n_val - n_test
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]
    test_idx  = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


def save_split(out_dir, split_name, X, Y, M, phase=None):
    """Saves X, Y, M (and optionally phase) for a split."""
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{split_name}_X.npy"), X)
    np.save(os.path.join(out_dir, f"{split_name}_Y.npy"), Y)
    np.save(os.path.join(out_dir, f"{split_name}_mask.npy"), M)
    if phase is not None:
        np.save(os.path.join(out_dir, f"{split_name}_phase.npy"), phase)


# -----------------------
# Main builder
# -----------------------
def main(args):
    # Paths
    noisy_path = os.path.join(args.input_dir, f"{args.dataset_type}_signals.npy")
    clean_path = os.path.join(args.input_dir, "clean_signals.npy")
    out_root = os.path.join(args.out_dir, args.dataset_type)
    os.makedirs(out_root, exist_ok=True)

    # Load
    noisy = np.load(noisy_path)  # [N, T]
    clean = np.load(clean_path)  # [N, T]
    assert noisy.shape == clean.shape, "noisy/clean datasets must match"
    assert noisy.shape[1] == args.signal_len and clean.shape[1] == args.signal_len, \
        f"Signal length mismatch: expected {args.signal_len}, got {noisy.shape[1]}"

    N = noisy.shape[0]
    print(f"[INFO] Loaded {N} samples, T={args.signal_len}")

    # Build spectrogram tensors
    print("[INFO] Computing spectrograms (this may take a moment)...")
    X, Y, M, PH = build_for_arrays(
        noisy_arr=noisy,
        clean_arr=clean,
        fs=args.fs,
        nperseg=args.nperseg,
        noverlap=args.noverlap,
        window=args.window,
        eps=args.eps,
    )
    # X/Y/M/PH shapes: [N, F, T']
    F, Tprime = X.shape[1], X.shape[2]
    print(f"[INFO] Spectrogram shape: (F={F}, T'={Tprime})")

    # Split (seeded, like your previous pipeline)
    tr_idx, va_idx, te_idx = split_indices(N, args.random_state)

    # Train/Val/Test splits
    train_X, train_Y, train_M = X[tr_idx], Y[tr_idx], M[tr_idx]
    val_X,   val_Y,   val_M   = X[va_idx], Y[va_idx], M[va_idx]
    test_X,  test_Y,  test_M  = X[te_idx], Y[te_idx], M[te_idx]
    test_PH               = PH[te_idx]     # phases saved only for test

    # Save
    print(f"[INFO] Saving arrays to: {out_root}")
    save_split(out_root, "train", train_X, train_Y, train_M, phase=None)
    save_split(out_root, "val",   val_X,   val_Y,   val_M,   phase=None)
    save_split(out_root, "test",  test_X,  test_Y,  test_M,  phase=test_PH)

    # Save metadata (for training/inference scripts)
    meta = {
        "dataset_type": args.dataset_type,
        "num_samples": int(N),
        "signal_len": int(args.signal_len),
        "fs": int(args.fs),
        "nperseg": int(args.nperseg),
        "noverlap": int(args.noverlap),
        "window": args.window,
        "pad": int(args.nperseg // 2),
        "eps": args.eps,
        "shapes": {
            "train": {"X": list(train_X.shape), "Y": list(train_Y.shape), "mask": list(train_M.shape)},
            "val":   {"X": list(val_X.shape),   "Y": list(val_Y.shape),   "mask": list(val_M.shape)},
            "test":  {"X": list(test_X.shape),  "Y": list(test_Y.shape),  "mask": list(test_M.shape), "phase": list(test_PH.shape)},
        },
        "split_indices": {
            "train_idx": tr_idx.tolist(),
            "val_idx": va_idx.tolist(),
            "test_idx": te_idx.tolist(),
        }
    }
    with open(os.path.join(out_root, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("[INFO] Done. Metadata saved to meta.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build spectrogram dataset for U-Net mask training")
    p.add_argument("--dataset_type", type=str, default=DEFAULTS["dataset_type"],
                   choices=["gaussian", "non_gaussian"], help="Which noisy set to use")
    p.add_argument("--signal_len", type=int, default=DEFAULTS["signal_len"])
    p.add_argument("--fs", type=int, default=DEFAULTS["fs"])
    p.add_argument("--nperseg", type=int, default=DEFAULTS["nperseg"])
    p.add_argument("--noverlap", type=int, default=DEFAULTS["noverlap"])
    p.add_argument("--random_state", type=int, default=DEFAULTS["random_state"])
    p.add_argument("--input_dir", type=str, default=DEFAULTS["input_dir"])
    p.add_argument("--out_dir", type=str, default=DEFAULTS["out_dir"])
    p.add_argument("--window", type=str, default=DEFAULTS["window"])
    p.add_argument("--eps", type=float, default=DEFAULTS["eps"])
    args = p.parse_args()
    main(args)
