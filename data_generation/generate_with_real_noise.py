#!/usr/bin/env python3
"""Phase B5 dataset builder — synthetic clean signals + real-noise injection.

Reuses synthetic clean signals from an existing pipeline-format dataset and
mixes them with real-noise samples drawn from `non_gaussian_noise_only.npy`
of an adapted RadioML subset. Produces a new dataset following the standard
pipeline schema (train/, test/, dataset_config.json).

Why: B3 zero-shot showed sim-to-real domain gap of 8-15 dB. B4 paired
fine-tune on RadioML pairs is methodologically flawed (mismatched symbol
streams). B5 trains on (synthetic_clean, synthetic_clean + real_noise)
pairs — preserves correct supervision while exposing the network to real
RF channel noise statistics.

Usage:
    python data_generation/generate_with_real_noise.py \\
        --synthetic data_generation/datasets/fpv_telemetry_polygauss_qpsk_bs1024_n100000_953c56e8 \\
        --noise-bank data_generation/datasets/radioml2018_bpsk_qpsk_fpv \\
        --output fpv_realnoise_bpsk_qpsk \\
        --target-snr-range -5 18
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _scale_noise_to_snr(clean: np.ndarray, noise: np.ndarray, target_snr_db: float) -> np.ndarray:
    """Scale `noise` so that `clean + scaled_noise` has SNR = target_snr_db.

    SNR_dB = 10 * log10(P_signal / P_noise)
    scale = sqrt(P_signal / (10^(SNR/10) * P_noise))
    """
    p_signal = float(np.mean(clean ** 2))
    p_noise = float(np.mean(noise ** 2))
    if p_noise <= 1e-12:
        return noise
    target_p_noise = p_signal / (10 ** (target_snr_db / 10.0))
    scale = float(np.sqrt(target_p_noise / p_noise))
    return (noise * scale).astype(np.float32)


def _mix_real_noise(
    clean: np.ndarray,            # [N, L]
    noise_bank: np.ndarray,       # [M, L]
    snr_range_db: tuple[float, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each clean frame, sample one real-noise frame, scale to a random
    SNR in `snr_range_db`, and return (noisy, noise_added, snr_per_sample).
    """
    N, L = clean.shape
    M = noise_bank.shape[0]
    snrs = rng.uniform(snr_range_db[0], snr_range_db[1], size=N).astype(np.float32)
    idx = rng.integers(0, M, size=N)

    noisy = np.empty((N, L), dtype=np.float32)
    added = np.empty((N, L), dtype=np.float32)
    for i in range(N):
        ns = _scale_noise_to_snr(clean[i], noise_bank[idx[i]], snrs[i])
        noisy[i] = clean[i] + ns
        added[i] = ns
    return noisy, added, snrs


def _build_test_per_snr(
    clean_test: np.ndarray,
    noise_bank: np.ndarray,
    test_snr_points: list[int],
    samples_per_snr: int,
    rng: np.random.Generator,
) -> dict:
    """Returns {snr: (clean_subset, noisy, noise_only)}."""
    out = {}
    n_clean = clean_test.shape[0]
    for snr in test_snr_points:
        n_take = min(samples_per_snr, n_clean)
        pick = rng.choice(n_clean, size=n_take, replace=False)
        clean_sub = clean_test[pick]
        noisy, added, _ = _mix_real_noise(clean_sub, noise_bank, (snr, snr), rng)
        out[snr] = (clean_sub, noisy, added)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--synthetic", required=True, type=Path,
                    help="Source synthetic dataset dir (provides clean signals).")
    ap.add_argument("--noise-bank", required=True, type=Path,
                    help="Adapted RadioML subset dir (provides "
                         "non_gaussian_noise_only.npy).")
    ap.add_argument("--output", required=True,
                    help="Output dataset name (under data_generation/datasets/).")
    ap.add_argument("--target-snr-range", nargs=2, type=float, default=[-5.0, 18.0],
                    help="Train SNR range (dB)")
    ap.add_argument("--test-snr-points", nargs="+", type=int,
                    default=[-5, 0, 5, 10, 15, 18],
                    help="Per-SNR test bins (dB)")
    ap.add_argument("--samples-per-snr", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"[b5] Loading synthetic clean signals from {args.synthetic}")
    clean_train = np.load(args.synthetic / "train" / "clean_signals.npy")

    # Test clean: union of per-SNR test files (they share the underlying clean
    # set in the synthetic pipeline; pull one for diversity).
    test_clean_files = sorted((args.synthetic / "test").glob("test_*_clean.npy"))
    if not test_clean_files:
        raise FileNotFoundError(f"No test_*_clean.npy in {args.synthetic}/test")
    clean_test = np.concatenate([np.load(f) for f in test_clean_files], axis=0)
    print(f"      train clean: {clean_train.shape}, test clean: {clean_test.shape}")

    print(f"[b5] Loading real-noise bank from {args.noise_bank}")
    noise_bank = np.load(args.noise_bank / "train" / "non_gaussian_noise_only.npy")
    print(f"      noise bank: {noise_bank.shape}")
    if noise_bank.shape[1] != clean_train.shape[1]:
        raise ValueError(f"Block size mismatch: clean={clean_train.shape[1]}, "
                         f"noise={noise_bank.shape[1]}")

    print(f"[b5] Mixing train pairs at SNR ∈ {args.target_snr_range}")
    noisy_tr, added_tr, snr_tr = _mix_real_noise(
        clean_train, noise_bank, tuple(args.target_snr_range), rng)

    out_dir = ROOT / "data_generation" / "datasets" / args.output
    train_dir = out_dir / "train"
    test_dir = out_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    print(f"[b5] Saving train to {train_dir}")
    np.save(train_dir / "clean_signals.npy", clean_train.astype(np.float32))
    np.save(train_dir / "non_gaussian_signals.npy", noisy_tr)
    np.save(train_dir / "non_gaussian_noise_only.npy", added_tr)
    np.save(train_dir / "gaussian_signals.npy", noisy_tr)  # alias for pipeline
    np.save(train_dir / "snr_values.npy", snr_tr)

    print(f"[b5] Building per-SNR test bins: {args.test_snr_points}")
    per_snr = _build_test_per_snr(
        clean_test, noise_bank, args.test_snr_points,
        args.samples_per_snr, rng)
    for snr, (clean_sub, noisy, added) in per_snr.items():
        tag = f"{snr}dB".replace("-", "m")
        np.save(test_dir / f"test_{tag}_clean.npy", clean_sub.astype(np.float32))
        np.save(test_dir / f"test_{tag}_non_gaussian.npy", noisy)
        np.save(test_dir / f"test_{tag}_non_gaussian_noise_only.npy", added)
        np.save(test_dir / f"test_{tag}_gaussian.npy", noisy)
    print(f"      test bins: {list(per_snr.keys())}, "
          f"{args.samples_per_snr} samples each")

    cfg = {
        "uid": uuid.uuid4().hex[:8],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scenario": "synthetic_clean_with_real_noise_injection",
        "synthetic_source": str(args.synthetic.name),
        "noise_bank_source": str(args.noise_bank.name),
        "block_size": int(clean_train.shape[1]),
        "sample_rate": 8192,
        "modulation_type": "synthetic_qpsk",
        "noise_types": ["real_radioml_residual"],
        "snr_range": list(args.target_snr_range),
        "test_snr_points": args.test_snr_points,
        "samples_per_snr": args.samples_per_snr,
        "num_train": int(clean_train.shape[0]),
        "seed": args.seed,
        "notes": (
            "Phase B5: synthetic clean signals from `synthetic_source` paired "
            "with real-noise samples drawn from `noise_bank_source` "
            "(non_gaussian_noise_only.npy). Preserves correct (clean, noisy) "
            "supervision while exposing model to real RF channel noise "
            "statistics. See experiments/REAL_DATA_STATUS.md §2."
        ),
    }
    with open(out_dir / "dataset_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[b5] Wrote {out_dir}")


if __name__ == "__main__":
    main()
