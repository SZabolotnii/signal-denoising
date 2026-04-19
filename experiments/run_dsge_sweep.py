#!/usr/bin/env python3
"""DSGE Sweep: compare Variant A vs B, different bases, S=2,3.

Runs 24 configs: 2 variants × 3 bases × 2 orders × 2 noise types.
Each config trains a HybridDSGE_UNet with corrected DSGE implementation.
"""
import gc
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Disable multiprocessing workers entirely to avoid fork errors on macOS
os.environ["OMP_NUM_THREADS"] = "1"

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.training_hybrid import HybridUnetTrainer

DATASET = ROOT / "data_generation/datasets/deep_space_polygauss_qpsk_bs1024_n400000_0310b7e7"

with open(DATASET / "dataset_config.json") as f:
    cfg = json.load(f)

COMMON = dict(
    dataset_path=DATASET,
    batch_size=4096,
    epochs=50,
    learning_rate=3e-4,
    signal_len=cfg["block_size"],
    fs=cfg["sample_rate"],
    nperseg=128,
    noverlap=96,
    random_state=42,
    data_fraction=0.05,
    device="cpu",
)

VARIANTS = ["A", "B"]
BASES = ["fractional", "polynomial", "robust"]
ORDERS = [2, 3]
NOISE_TYPES = ["gaussian", "non_gaussian"]

# Fractional powers: sign(x)|x|^p — no linear term
FRACTIONAL_POWERS = {2: [0.5, 1.5], 3: [0.5, 1.5, 2.0]}
# Polynomial powers: x^p with p >= 2 only (no linear x^1)
POLYNOMIAL_POWERS = {2: [2, 3], 3: [2, 3, 4]}
# Robust: uses pool functions, powers just determine count
ROBUST_POWERS = {2: [0, 1], 3: [0, 1, 2]}

POWERS_MAP = {
    "fractional": FRACTIONAL_POWERS,
    "polynomial": POLYNOMIAL_POWERS,
    "robust": ROBUST_POWERS,
}

results = []
total = len(VARIANTS) * len(BASES) * len(ORDERS) * len(NOISE_TYPES)
i = 0

print(f"{'=' * 70}")
print(f"DSGE Sweep: {total} configurations")
print(f"Dataset: {DATASET.name}")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"{'=' * 70}")

for variant in VARIANTS:
    for basis in BASES:
        for order in ORDERS:
            for noise_type in NOISE_TYPES:
                i += 1
                name = f"v{variant}_{basis}_S{order}_{noise_type}"
                print(f"\n{'#' * 70}")
                print(f"# [{i}/{total}] {name}")
                print(f"{'#' * 70}")

                powers = POWERS_MAP[basis][order]
                try:
                    result = HybridUnetTrainer(
                        **COMMON,
                        noise_type=noise_type,
                        dsge_order=order,
                        dsge_basis=basis,
                        dsge_variant=variant,
                        dsge_powers=powers,
                    ).train()
                    results.append(result)
                    snr = result.get("val_snr", float("nan"))
                    print(f"  -> val_SNR = {snr:.2f} dB")
                except Exception as exc:
                    print(f"  ERROR: {exc}")
                    results.append({
                        "model": name,
                        "noise_type": noise_type,
                        "error": str(exc),
                    })
                finally:
                    gc.collect()

# Summary
print(f"\n{'=' * 70}")
print(f"DSGE Sweep Summary")
print(f"{'=' * 70}")
print(f"  {'Model':<55} {'val_SNR':>9} {'test_SNR':>9}")
print(f"  {'-' * 73}")
for r in results:
    name = r.get("model", "?")
    if r.get("error"):
        print(f"  {name:<55}  ERROR: {r['error'][:25]}")
    else:
        val_snr = r.get("val_snr") or float("nan")
        test_snr = (r.get("test_metrics") or {}).get("SNR", float("nan"))
        print(f"  {name:<55} {val_snr:>8.2f} dB {test_snr:>8.2f} dB")

# Save results
out_path = ROOT / "experiments" / f"dsge_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to: {out_path}")
