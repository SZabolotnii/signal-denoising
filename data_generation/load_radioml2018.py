"""
Adapter: RadioML 2018.01A → signal-denoising pipeline format.

Input: `GOLD_XYZ_OSC.0001_1024.hdf5` from DeepSig (deepsig.ai/datasets).
  Keys: 'X' (N, 1024, 2) complex I/Q, 'Y' (N, 24) one-hot modulation, 'Z' (N, 1) SNR dB.
  26 SNR levels (−20..+30 dB, step 2), 24 modulations, 4096 frames per (mod, SNR).

Output: directory matching `data_generation/datasets/<name>/` schema —
  train/{clean_signals,non_gaussian_signals,non_gaussian_noise_only,snr_values}.npy
  test/test_<snr>dB_{clean,non_gaussian,non_gaussian_noise_only}.npy
  dataset_config.json

Clean-reference strategy (documented in experiments/dataset_survey.md):
  (a) HIGH-SNR PROXY — use frames at SNR ≥ +28 dB as "clean reference"; pair them
      with low-SNR frames from the same modulation. The noise component is the
      algebraic difference; note that the underlying symbol streams differ between
      frames, so the "noise" also contains inter-frame symbol variation. This is a
      known limitation of using RadioML for paired denoising and is discussed in
      the paper. For Phase B3 zero-shot evaluation, this limitation is avoided
      by evaluating only on input→output consistency (no clean reference required).

Real → real projection:
  RadioML frames are complex baseband (I/Q). We take `np.real(I + j*Q)` which
  reduces to the I-channel. This halves the usable information but matches the
  real-valued pipeline contract (block_size=1024, sample_rate=8192 Hz nominal).

Usage:
  python data_generation/load_radioml2018.py \\
      --hdf5 /path/to/GOLD_XYZ_OSC.0001_1024.hdf5 \\
      --modulations BPSK QPSK \\
      --clean-snr-min 28 \\
      --noisy-snr-max 0 \\
      --output radioml2018_bpsk_qpsk
"""
from __future__ import annotations

import argparse
import json
import os
import uuid
from datetime import datetime

import numpy as np

# RadioML 2018.01A modulation label order (DeepSig spec).
RADIOML_MODULATIONS: tuple[str, ...] = (
    "OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK", "16PSK", "32PSK",
    "16APSK", "32APSK", "64APSK", "128APSK",
    "16QAM", "32QAM", "64QAM", "128QAM", "256QAM",
    "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC",
    "FM", "GMSK", "OQPSK",
)

RADIOML_SNR_LEVELS = tuple(range(-20, 32, 2))  # −20..+30 dB step 2


def load_hdf5(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load RadioML 2018.01A HDF5. Returns (X, mod_idx, snr_db)."""
    import h5py  # deferred import — optional dep.

    with h5py.File(path, "r") as f:
        X = f["X"][:]            # (N, 1024, 2)
        Y = f["Y"][:]            # (N, 24)
        Z = f["Z"][:].squeeze()  # (N,)

    mod_idx = Y.argmax(axis=1).astype(np.int32)
    snr_db = Z.astype(np.float32)
    return X, mod_idx, snr_db


def complex_to_real(X: np.ndarray) -> np.ndarray:
    """Project (N, 1024, 2) I/Q frames to (N, 1024) real via I-channel."""
    return X[..., 0].astype(np.float32)


def select_modulations(
    mod_idx: np.ndarray,
    mod_names: list[str],
) -> np.ndarray:
    """Return boolean mask for frames matching `mod_names`."""
    name_to_idx = {n: i for i, n in enumerate(RADIOML_MODULATIONS)}
    wanted = {name_to_idx[n] for n in mod_names}
    return np.isin(mod_idx, list(wanted))


def build_pairs_high_snr_proxy(
    X_real: np.ndarray,
    mod_idx: np.ndarray,
    snr_db: np.ndarray,
    clean_snr_min: float,
    noisy_snr_max: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Strategy (a): pair each low-SNR frame (≤ noisy_snr_max) with a random
    high-SNR frame (≥ clean_snr_min) from the *same modulation*.

    Returns (clean, noisy, snr_per_sample) with matched leading axis.
    """
    clean_list, noisy_list, snr_list = [], [], []
    for m in np.unique(mod_idx):
        clean_pool = np.where((mod_idx == m) & (snr_db >= clean_snr_min))[0]
        noisy_pool = np.where((mod_idx == m) & (snr_db <= noisy_snr_max))[0]
        if len(clean_pool) == 0 or len(noisy_pool) == 0:
            continue
        picks = rng.integers(0, len(clean_pool), size=len(noisy_pool))
        clean_list.append(X_real[clean_pool[picks]])
        noisy_list.append(X_real[noisy_pool])
        snr_list.append(snr_db[noisy_pool])

    clean = np.concatenate(clean_list, axis=0)
    noisy = np.concatenate(noisy_list, axis=0)
    snr = np.concatenate(snr_list, axis=0)

    # Shuffle jointly.
    perm = rng.permutation(len(clean))
    return clean[perm], noisy[perm], snr[perm]


def normalize_amplitude(clean: np.ndarray, noisy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample amplitude normalization to match synthetic pipeline scale.

    Scale each pair by `1 / max(|noisy|)` so the noisy input lies in [−1, 1].
    Uses the same scale for the clean reference to preserve the noise margin.
    """
    scales = np.maximum(np.abs(noisy).max(axis=1, keepdims=True), 1e-9)
    return clean / scales, noisy / scales


def split_train_test(
    clean: np.ndarray, noisy: np.ndarray, snr: np.ndarray,
    test_frac: float, rng: np.random.Generator,
):
    n = len(clean)
    n_test = int(n * test_frac)
    idx = rng.permutation(n)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return (
        (clean[train_idx], noisy[train_idx], snr[train_idx]),
        (clean[test_idx],  noisy[test_idx],  snr[test_idx]),
    )


def save_dataset(
    out_dir: str,
    train: tuple, test: tuple,
    test_snr_points: list[int], samples_per_snr: int,
    modulations: list[str],
    clean_snr_min: float, noisy_snr_max: float,
):
    clean_tr, noisy_tr, snr_tr = train
    clean_te, noisy_te, snr_te = test

    train_dir = os.path.join(out_dir, "train")
    test_dir = os.path.join(out_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    np.save(os.path.join(train_dir, "clean_signals.npy"), clean_tr.astype(np.float32))
    np.save(os.path.join(train_dir, "non_gaussian_signals.npy"), noisy_tr.astype(np.float32))
    np.save(
        os.path.join(train_dir, "non_gaussian_noise_only.npy"),
        (noisy_tr - clean_tr).astype(np.float32),
    )
    np.save(os.path.join(train_dir, "snr_values.npy"), snr_tr.astype(np.float32))
    # Gaussian variant left as symlink-to-non_gaussian: real captures are not AWGN.
    # Training code requires the file to exist; use noisy as placeholder to avoid
    # accidental Gaussian training on real data (user must opt in explicitly).
    np.save(os.path.join(train_dir, "gaussian_signals.npy"), noisy_tr.astype(np.float32))

    # Per-SNR test buckets.
    for snr_target in test_snr_points:
        mask = np.abs(snr_te - snr_target) < 1.0
        if mask.sum() == 0:
            continue
        take = np.where(mask)[0][:samples_per_snr]
        tag = f"{snr_target}dB".replace("-", "m")
        np.save(os.path.join(test_dir, f"test_{tag}_clean.npy"), clean_te[take].astype(np.float32))
        np.save(os.path.join(test_dir, f"test_{tag}_non_gaussian.npy"), noisy_te[take].astype(np.float32))
        np.save(
            os.path.join(test_dir, f"test_{tag}_non_gaussian_noise_only.npy"),
            (noisy_te[take] - clean_te[take]).astype(np.float32),
        )
        np.save(os.path.join(test_dir, f"test_{tag}_gaussian.npy"), noisy_te[take].astype(np.float32))

    uid = uuid.uuid4().hex[:8]
    config = {
        "uid": uid,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "folder": os.path.basename(out_dir.rstrip("/")),
        "scenario": "radioml2018_real_sdr",
        "source": "DeepSig RadioML 2018.01A",
        "modulation_type": "+".join(modulations).lower(),
        "bits_per_symbol": None,
        "block_size": 1024,
        "sample_rate": 8192,  # nominal — RadioML has no absolute rate; matches pipeline.
        "snr_range": [float(noisy_snr_max), float(noisy_snr_max)],
        "noise_types": ["real_sdr"],
        "mix_mode": "real",
        "pairing_strategy": "high_snr_proxy",
        "clean_snr_min": clean_snr_min,
        "noisy_snr_max": noisy_snr_max,
        "num_train": int(len(clean_tr)),
        "test_snr_points": list(test_snr_points),
        "samples_per_snr": samples_per_snr,
        "notes": (
            "Clean reference synthesized from high-SNR frames of the same "
            "modulation. Underlying symbol streams differ between clean and noisy "
            "frames, so the residual contains symbol variation in addition to "
            "channel noise. See experiments/dataset_survey.md §'RadioML caveat'."
        ),
    }
    with open(os.path.join(out_dir, "dataset_config.json"), "w") as f:
        json.dump(config, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--hdf5", required=True, help="Path to GOLD_XYZ_OSC.0001_1024.hdf5")
    parser.add_argument("--modulations", nargs="+", default=["BPSK", "QPSK"],
                        help=f"Subset of: {RADIOML_MODULATIONS}")
    parser.add_argument("--clean-snr-min", type=float, default=28.0,
                        help="Minimum SNR (dB) for proxy-clean frames")
    parser.add_argument("--noisy-snr-max", type=float, default=0.0,
                        help="Maximum SNR (dB) for noisy frames")
    parser.add_argument("--test-frac", type=float, default=0.25)
    parser.add_argument("--test-snr-points", nargs="+", type=int,
                        default=[-20, -15, -10, -5, 0])
    parser.add_argument("--samples-per-snr", type=int, default=500)
    parser.add_argument("--output", required=True, help="Subdirectory name under data_generation/datasets/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"[1/5] Loading HDF5: {args.hdf5}")
    X, mod_idx, snr_db = load_hdf5(args.hdf5)
    print(f"      Loaded {len(X):,} frames.")

    print(f"[2/5] Selecting modulations: {args.modulations}")
    mask = select_modulations(mod_idx, args.modulations)
    X, mod_idx, snr_db = X[mask], mod_idx[mask], snr_db[mask]
    print(f"      {len(X):,} frames after filtering.")

    X_real = complex_to_real(X)

    print(f"[3/5] Pairing (clean≥{args.clean_snr_min} dB, noisy≤{args.noisy_snr_max} dB)")
    clean, noisy, snr = build_pairs_high_snr_proxy(
        X_real, mod_idx, snr_db,
        clean_snr_min=args.clean_snr_min,
        noisy_snr_max=args.noisy_snr_max,
        rng=rng,
    )
    clean, noisy = normalize_amplitude(clean, noisy)
    print(f"      {len(clean):,} paired frames.")

    print(f"[4/5] Train/test split (test_frac={args.test_frac})")
    train, test = split_train_test(clean, noisy, snr, args.test_frac, rng)
    print(f"      train: {len(train[0]):,}, test: {len(test[0]):,}")

    out_dir = os.path.join(os.path.dirname(__file__), "datasets", args.output)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[5/5] Saving to {out_dir}")
    save_dataset(
        out_dir, train, test,
        test_snr_points=args.test_snr_points,
        samples_per_snr=args.samples_per_snr,
        modulations=args.modulations,
        clean_snr_min=args.clean_snr_min,
        noisy_snr_max=args.noisy_snr_max,
    )
    print("      Done.")


if __name__ == "__main__":
    main()
