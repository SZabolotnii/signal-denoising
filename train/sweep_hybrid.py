#!/usr/bin/env python3
"""
Architecture sweep for HybridDSGE_UNet.

Tests all DSGE basis types and polynomial orders. Each configuration
gets its own W&B run for easy comparison in the W&B UI.

Usage:
    python train/sweep_hybrid.py \
        --dataset data_generation/datasets/deep_space_..._39075e4f \
        --noise-type non_gaussian \
        --epochs 30 \
        --wandb-project sd-science

Results are printed as a comparison table and saved to
<dataset>/weights/sweep_hybrid_report_<timestamp>.md
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# ── sweep grid ────────────────────────────────────────────────────────────────
# Each entry: (dsge_basis, dsge_powers, dsge_order)
# 'robust' has a fixed set of 3 functions (tanh, sigmoid, atan), order is always 3.
SWEEP_CONFIGS = [
    # fractional: sign(x)|x|^p — preserves sign, good for QPSK/FSK (S2–S5)
    ("fractional", [0.5, 1.5],                  2),
    ("fractional", [0.5, 1.0, 2.0],             3),
    ("fractional", [0.3, 0.7, 1.5, 2.5],        4),
    ("fractional", [0.3, 0.7, 1.0, 2.0, 3.0],  5),
    # polynomial: x^p — classical nonlinear projection (S2–S5)
    ("polynomial", [2.0, 3.0],                  2),
    ("polynomial", [2.0, 3.0, 4.0],             3),
    ("polynomial", [2.0, 3.0, 4.0, 5.0],        4),
    ("polynomial", [2.0, 3.0, 4.0, 5.0, 6.0],  5),
    # trigonometric: sin(f*x) — periodic features, f=1..S (S2–S5)
    ("trigonometric", [1.0, 2.0],               2),
    ("trigonometric", [1.0, 2.0, 3.0],          3),
    ("trigonometric", [1.0, 2.0, 3.0, 4.0],     4),
    ("trigonometric", [1.0, 2.0, 3.0, 4.0, 5.0], 5),
    # robust: tanh / sigmoid / atan — suppresses outliers (fixed 3-function basis)
    ("robust", None, 3),
]


def parse_args():
    p = argparse.ArgumentParser(description="DSGE architecture sweep for HybridDSGE_UNet")
    p.add_argument("--dataset",       required=True,
                   help="Path to dataset folder")
    p.add_argument("--noise-type",    default="non_gaussian",
                   choices=["gaussian", "non_gaussian"])
    p.add_argument("--epochs",        type=int,   default=30)
    p.add_argument("--batch-size",    type=int,   default=32)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--nperseg",       type=int,   default=32)
    p.add_argument("--lambda",        type=float, default=0.01, dest="tikhonov_lambda")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--wandb-project", default="")
    p.add_argument("--configs",       type=str,   default="all",
                   help="Comma-separated config indices (0-based) or 'all'")
    return p.parse_args()


def main():
    args = parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    if not dataset_path.exists():
        print(f"ERROR: dataset not found: {dataset_path}")
        sys.exit(1)

    with open(dataset_path / "dataset_config.json") as f:
        cfg = json.load(f)

    signal_len = cfg["block_size"]
    fs         = cfg["sample_rate"]
    noverlap   = args.nperseg // 2

    if args.configs == "all":
        configs_to_run = list(range(len(SWEEP_CONFIGS)))
    else:
        configs_to_run = [int(i) for i in args.configs.split(",")]

    print(f"Dataset  : {dataset_path.name}")
    print(f"Config   : block_size={signal_len}, sample_rate={fs}")
    print(f"Sweep    : {len(configs_to_run)} configurations")
    print(f"Epochs   : {args.epochs}")
    print()

    from train.training_hybrid import HybridUnetTrainer

    all_results = []

    for idx in configs_to_run:
        basis, powers, order = SWEEP_CONFIGS[idx]
        tag = f"{basis}_S{order}" + ("" if powers is None else f"_p{'_'.join(str(p) for p in powers)}")
        print(f"\n{'='*60}")
        print(f"Config [{idx+1}/{len(configs_to_run)}]: basis={basis}, order={order}, powers={powers}")
        print(f"{'='*60}")

        try:
            trainer = HybridUnetTrainer(
                dataset_path=dataset_path,
                noise_type=args.noise_type,
                dsge_order=order,
                dsge_basis=basis,
                dsge_powers=powers,
                tikhonov_lambda=args.tikhonov_lambda,
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.lr,
                signal_len=signal_len,
                fs=fs,
                nperseg=args.nperseg,
                noverlap=noverlap,
                random_state=args.seed,
                wandb_project=args.wandb_project,
            )
            result = trainer.train()
            result['config_tag'] = tag
            all_results.append(result)

        except Exception as exc:
            print(f"ERROR in config {tag}: {exc}")
            all_results.append({
                'model': f"HybridDSGE_{tag}",
                'config_tag': tag,
                'error': str(exc),
            })

    # ── comparison table ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SWEEP RESULTS")
    print("=" * 70)
    print(f"{'Config':<35} {'Val SNR':>9} {'Test SNR':>9} {'Test MSE':>10}")
    print("-" * 70)
    for r in all_results:
        if 'error' in r:
            print(f"{r['config_tag']:<35} {'ERROR':>9}")
            continue
        val_snr  = r.get('val_snr', float('nan'))
        test_snr = r.get('test_metrics', {}).get('SNR', float('nan'))
        test_mse = r.get('test_metrics', {}).get('MSE', float('nan'))
        print(f"{r['config_tag']:<35} {val_snr:>8.2f}dB {test_snr:>8.2f}dB {test_mse:>10.6f}")

    # ── save report ───────────────────────────────────────────────────────────
    weights_dir = dataset_path / "weights"
    weights_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    md_path = weights_dir / f"sweep_hybrid_report_{timestamp}.md"
    with open(md_path, "w") as f:
        f.write("# HybridDSGE_UNet Architecture Sweep\n\n")
        f.write(f"**Dataset:** `{dataset_path.name}`  \n")
        f.write(f"**Noise type:** {args.noise_type}  \n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  \n")
        f.write(f"**Epochs:** {args.epochs} | **Batch:** {args.batch_size} | **LR:** {args.lr}\n\n")
        f.write("## Results\n\n")
        f.write("| Config | Val SNR | Test SNR | Test MSE | Weights |\n")
        f.write("|--------|--------:|--------:|---------:|---------|\n")
        for r in all_results:
            if 'error' in r:
                f.write(f"| {r['config_tag']} | ERROR | — | — | — |\n")
                continue
            val_snr  = r.get('val_snr', float('nan'))
            test_snr = r.get('test_metrics', {}).get('SNR', float('nan'))
            test_mse = r.get('test_metrics', {}).get('MSE', float('nan'))
            wname = Path(r.get('weights_path', '')).name
            f.write(f"| {r['config_tag']} | {val_snr:.2f} dB | {test_snr:.2f} dB | "
                    f"{test_mse:.6f} | `{wname}` |\n")

    json_path = weights_dir / f"sweep_hybrid_report_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n📋 Sweep report → {md_path}")

    # highlight winner
    valid = [r for r in all_results if 'error' not in r and r.get('test_metrics', {}).get('SNR') is not None]
    if valid:
        best = max(valid, key=lambda r: r['test_metrics']['SNR'])
        print(f"🏆 Best config: {best['config_tag']} "
              f"(test SNR = {best['test_metrics']['SNR']:.2f} dB)")


if __name__ == "__main__":
    main()
