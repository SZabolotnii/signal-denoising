"""
Adapter: DroneDetect (IEEE DataPort) → signal-denoising pipeline format.

Source: BladeRF SDR + GNURadio recordings of UAV controllers/video at 2.4 GHz
ISM band. Files are distributed as SigMF pairs:
  <name>.sigmf-data   raw interleaved int16 I/Q (or cf32)
  <name>.sigmf-meta   JSON metadata (sample_rate, dtype, captures, annotations)

Dataset has two natural classes per recording:
  - "signal" segments: drone controller active (annotated via `annotations` in meta).
  - "noise" segments: ambient RF (no drone), or gaps between bursts.

Clean-reference strategy for denoising pairs:
  - Extract high-confidence "signal" segments as proxy clean (narrowband,
    receiver at short distance → high SNR). Optionally band-pass filter around
    the annotated frequency to suppress out-of-band interference.
  - Extract "noise" segments (or long-distance low-SNR signal segments) as
    noisy inputs. Add the clean component back at a controlled SNR to form
    paired (clean, noisy) training data — this is the only way to get a
    ground-truth pair from DroneDetect, at the cost of some synthesis on top
    of real noise. Document this in the paper.

Real → real projection:
  Magnitude `|I + jQ|` or I-channel (`np.real`). We use `np.real` for
  consistency with `load_radioml2018.py`.

Resampling:
  DroneDetect sample rates (10–30 MHz) must be decimated to 8192 Hz for the
  pipeline. Uses `scipy.signal.decimate` with Chebyshev filter. This heavily
  narrowbands the signal — acceptable for the residual-noise-stat test,
  questionable for full denoising performance. Documented as a limitation.

Usage:
  python data_generation/load_dronedetect.py \\
      --sigmf-dir /path/to/dronedetect_recordings/ \\
      --output dronedetect_subset \\
      --target-snr-range -5 15
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import uuid
from datetime import datetime

import numpy as np


SIGMF_DTYPES = {
    "ci16_le": np.int16,
    "ci16":    np.int16,
    "cf32_le": np.float32,
    "cf32":    np.float32,
    "ci8":     np.int8,
}


def load_sigmf(meta_path: str) -> tuple[np.ndarray, dict]:
    """Load a single SigMF recording. Returns (iq_complex_array, meta_dict)."""
    with open(meta_path) as f:
        meta = json.load(f)

    global_ = meta["global"]
    dtype_tag = global_["core:datatype"].lower()
    base = SIGMF_DTYPES.get(dtype_tag)
    if base is None:
        raise ValueError(f"Unsupported SigMF dtype: {dtype_tag!r}")

    data_path = meta_path.replace(".sigmf-meta", ".sigmf-data")
    raw = np.fromfile(data_path, dtype=base)
    # Interleaved I, Q → complex.
    iq = raw.astype(np.float32).reshape(-1, 2)
    iq_complex = iq[:, 0] + 1j * iq[:, 1]
    if base in (np.int16, np.int8):
        iq_complex /= np.iinfo(base).max
    return iq_complex, meta


def resample_to_target(iq: np.ndarray, src_rate: float, tgt_rate: int = 8192) -> np.ndarray:
    """Decimate IQ to target rate. Preserves complex dtype."""
    from scipy.signal import decimate
    ratio = src_rate / tgt_rate
    if ratio < 1.0:
        raise ValueError(f"Source rate {src_rate} below target {tgt_rate}")
    q = int(round(ratio))
    if q == 1:
        return iq
    # `decimate` caps at q=13 per call; cascade if needed.
    out = iq
    while q > 13:
        out = decimate(out, 13, ftype="iir", zero_phase=True)
        q //= 13
    if q > 1:
        out = decimate(out, q, ftype="iir", zero_phase=True)
    return out


def segment_into_blocks(signal: np.ndarray, block_size: int = 1024) -> np.ndarray:
    """Chop 1D array into (N, block_size) without overlap, discard remainder."""
    n = (len(signal) // block_size) * block_size
    return signal[:n].reshape(-1, block_size)


def split_signal_noise_segments(
    iq: np.ndarray, meta: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a SigMF recording into (clean-proxy blocks, noise-only blocks) using
    `annotations`. Annotations with `core:label == "signal"` mark drone bursts;
    the complement is treated as ambient noise.
    """
    n_total = len(iq)
    mask_signal = np.zeros(n_total, dtype=bool)
    for ann in meta.get("annotations", []):
        start = ann["core:sample_start"]
        count = ann.get("core:sample_count", n_total - start)
        label = ann.get("core:label", "").lower()
        if "signal" in label or "drone" in label:
            mask_signal[start : start + count] = True

    signal_iq = iq[mask_signal]
    noise_iq = iq[~mask_signal]
    return signal_iq, noise_iq


def compose_pairs(
    clean_blocks: np.ndarray, noise_blocks: np.ndarray,
    target_snr_range: tuple[float, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Form paired (clean, noisy, snr_db) triples: take a clean block, additively
    mix with a random real-noise block scaled to hit a target SNR sampled
    uniformly from `target_snr_range`.
    """
    n = min(len(clean_blocks), len(noise_blocks))
    clean = clean_blocks[:n]
    noise = noise_blocks[rng.permutation(len(noise_blocks))[:n]]

    snr_db = rng.uniform(*target_snr_range, size=n).astype(np.float32)
    p_clean = np.mean(clean ** 2, axis=1, keepdims=True) + 1e-12
    p_noise = np.mean(noise ** 2, axis=1, keepdims=True) + 1e-12
    target_p_noise = p_clean / (10 ** (snr_db[:, None] / 10))
    scale = np.sqrt(target_p_noise / p_noise)

    noisy = clean + scale * noise
    return clean.astype(np.float32), noisy.astype(np.float32), snr_db


def save_dataset(
    out_dir: str,
    clean: np.ndarray, noisy: np.ndarray, snr: np.ndarray,
    test_frac: float, test_snr_points: list[int], samples_per_snr: int,
    rng: np.random.Generator,
    src_rate: float, target_snr_range: tuple[float, float],
):
    n = len(clean)
    n_test = int(n * test_frac)
    perm = rng.permutation(n)
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    train_dir = os.path.join(out_dir, "train")
    test_dir = os.path.join(out_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Per-sample amplitude normalization.
    scales = np.maximum(np.abs(noisy).max(axis=1, keepdims=True), 1e-9)
    clean_n = clean / scales
    noisy_n = noisy / scales

    np.save(os.path.join(train_dir, "clean_signals.npy"), clean_n[train_idx])
    np.save(os.path.join(train_dir, "non_gaussian_signals.npy"), noisy_n[train_idx])
    np.save(
        os.path.join(train_dir, "non_gaussian_noise_only.npy"),
        (noisy_n[train_idx] - clean_n[train_idx]),
    )
    np.save(os.path.join(train_dir, "snr_values.npy"), snr[train_idx])
    np.save(os.path.join(train_dir, "gaussian_signals.npy"), noisy_n[train_idx])

    for snr_target in test_snr_points:
        sel = test_idx[np.abs(snr[test_idx] - snr_target) < 1.0]
        if len(sel) == 0:
            continue
        sel = sel[:samples_per_snr]
        tag = f"{snr_target}dB".replace("-", "m")
        np.save(os.path.join(test_dir, f"test_{tag}_clean.npy"), clean_n[sel])
        np.save(os.path.join(test_dir, f"test_{tag}_non_gaussian.npy"), noisy_n[sel])
        np.save(
            os.path.join(test_dir, f"test_{tag}_non_gaussian_noise_only.npy"),
            noisy_n[sel] - clean_n[sel],
        )
        np.save(os.path.join(test_dir, f"test_{tag}_gaussian.npy"), noisy_n[sel])

    config = {
        "uid": uuid.uuid4().hex[:8],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "folder": os.path.basename(out_dir.rstrip("/")),
        "scenario": "dronedetect_real_sdr",
        "source": "DroneDetect (IEEE DataPort) — BladeRF SDR, 2.4 GHz ISM",
        "modulation_type": "drone_telemetry_mixed",
        "bits_per_symbol": None,
        "block_size": 1024,
        "sample_rate": 8192,
        "src_sample_rate": float(src_rate),
        "snr_range": list(target_snr_range),
        "noise_types": ["real_sdr_ambient"],
        "mix_mode": "synthetic_on_real_noise",
        "pairing_strategy": "clean_proxy_plus_real_noise",
        "num_train": int(len(train_idx)),
        "test_snr_points": list(test_snr_points),
        "samples_per_snr": samples_per_snr,
        "notes": (
            "Clean = signal-annotated SigMF segments (high-SNR bursts); noise = "
            "ambient-RF segments. Pairs synthesized by additive mixing at target "
            "SNR — 'non_gaussian' here means real-measured noise additively "
            "combined with real clean proxy. Documented in dataset_survey.md."
        ),
    }
    with open(os.path.join(out_dir, "dataset_config.json"), "w") as f:
        json.dump(config, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--sigmf-dir", required=True,
                        help="Directory containing .sigmf-meta + .sigmf-data pairs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-snr-range", nargs=2, type=float, default=[-5.0, 15.0])
    parser.add_argument("--test-frac", type=float, default=0.25)
    parser.add_argument("--test-snr-points", nargs="+", type=int,
                        default=[-5, 0, 5, 10, 15])
    parser.add_argument("--samples-per-snr", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    metas = sorted(glob.glob(os.path.join(args.sigmf_dir, "*.sigmf-meta")))
    if not metas:
        raise SystemExit(f"No SigMF recordings found in {args.sigmf_dir}")
    print(f"Found {len(metas)} SigMF recordings.")

    all_clean, all_noise = [], []
    src_rate = None
    for i, meta_path in enumerate(metas):
        print(f"  [{i+1}/{len(metas)}] {os.path.basename(meta_path)}")
        iq, meta = load_sigmf(meta_path)
        src_rate = float(meta["global"]["core:sample_rate"])
        iq_ds = resample_to_target(iq, src_rate)
        iq_real = np.real(iq_ds).astype(np.float32)

        sig, noise = split_signal_noise_segments(
            iq_real if iq_real.ndim == 1 else iq_real,
            meta,
        )
        all_clean.append(segment_into_blocks(sig))
        all_noise.append(segment_into_blocks(noise))

    clean_blocks = np.concatenate(all_clean, axis=0) if all_clean else np.zeros((0, 1024))
    noise_blocks = np.concatenate(all_noise, axis=0) if all_noise else np.zeros((0, 1024))
    print(f"Total: {len(clean_blocks):,} clean blocks, {len(noise_blocks):,} noise blocks.")

    clean, noisy, snr = compose_pairs(
        clean_blocks, noise_blocks,
        target_snr_range=tuple(args.target_snr_range),
        rng=rng,
    )
    print(f"Composed {len(clean):,} paired frames.")

    out_dir = os.path.join(os.path.dirname(__file__), "datasets", args.output)
    os.makedirs(out_dir, exist_ok=True)
    save_dataset(
        out_dir, clean, noisy, snr,
        test_frac=args.test_frac,
        test_snr_points=args.test_snr_points,
        samples_per_snr=args.samples_per_snr,
        rng=rng,
        src_rate=src_rate or 0.0,
        target_snr_range=tuple(args.target_snr_range),
    )
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
