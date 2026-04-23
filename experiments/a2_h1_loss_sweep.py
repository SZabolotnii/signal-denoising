#!/usr/bin/env python3
"""A2-H1: Loss sweep for DSGE + non-Gaussian training failure.

Tests whether SmoothL1(beta=0.02) — the current default — is compatible with
DSGE-Hybrid-UNet training under heavy-tailed noise. Alternative losses:
MSE, SmoothL1 β∈{0.1, 1.0}, Huber δ=1.0, Charbonnier ε=1e-3.

Matrix: 6 losses × 2 scenarios × 3 seeds = 36 runs.

Scenarios (partial, per plan §A2-H1):
  - FPV telemetry: 25% data subset (~25k samples), 15 epochs
  - deep_space:    10% data subset (~40k samples), 15 epochs

Output:
  - experiments/results/a2_h1_<ts>.json — per-run val_SNR / test_SNR
  - experiments/results/a2_h1_<ts>.md   — summary table

Runtime estimate: ~5 h on M3 Max CPU (matches plan budget).

Usage:
  python experiments/a2_h1_loss_sweep.py
  python experiments/a2_h1_loss_sweep.py --smoke  # 1 loss × 1 scenario × 1 seed
"""
from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.training_hybrid import HybridUnetTrainer  # noqa: E402


DATASETS = {
    "fpv":        ROOT / "data_generation/datasets/fpv_telemetry_polygauss_qpsk_bs1024_n100000_953c56e8",
    "deep_space": ROOT / "data_generation/datasets/deep_space_polygauss_qpsk_bs1024_n400000_0310b7e7",
}

SCENARIO_FRACTIONS = {"fpv": 0.25, "deep_space": 0.10}

# (loss_name, kwargs) — kwargs merged into Trainer ctor
LOSS_CONFIGS = [
    ("mse",                {"loss_name": "mse"}),
    ("smoothl1_b0.02",     {"loss_name": "smoothl1", "robust_beta": 0.02}),
    ("smoothl1_b0.1",      {"loss_name": "smoothl1", "robust_beta": 0.1}),
    ("smoothl1_b1.0",      {"loss_name": "smoothl1", "robust_beta": 1.0}),
    ("huber_d1.0",         {"loss_name": "huber",   "huber_delta": 1.0}),
    ("charbonnier_e1e-3",  {"loss_name": "charbonnier", "charbonnier_eps": 1e-3}),
]

SEEDS = [42, 43, 44]


def run_one(scenario: str, loss_tag: str, loss_kwargs: dict, seed: int,
            epochs: int, batch_size: int) -> dict:
    """Train one config. Returns metrics dict."""
    ds_path = DATASETS[scenario]
    with open(ds_path / "dataset_config.json") as f:
        cfg = json.load(f)

    trainer = HybridUnetTrainer(
        dataset_path=ds_path,
        noise_type="non_gaussian",
        dsge_order=3,
        dsge_basis="robust",
        dsge_variant="A",
        unet_width=16,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=3e-4,
        signal_len=cfg["block_size"],
        fs=cfg["sample_rate"],
        nperseg=128,
        noverlap=96,
        random_state=seed,
        data_fraction=SCENARIO_FRACTIONS[scenario],
        device="cpu",
        **loss_kwargs,
    )
    result = trainer.train()
    return {
        "scenario": scenario,
        "loss": loss_tag,
        "seed": seed,
        "epochs": epochs,
        "best_val_snr_db": float(result.get("val_snr", float("nan"))),
        "final_test_snr_db": float(result.get("test_metrics", {}).get("SNR", float("nan"))),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--smoke", action="store_true", help="1 loss × 1 scenario × 1 seed")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--scenarios", nargs="+", default=list(DATASETS.keys()))
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--losses", nargs="+", default=None,
                   help=f"Loss tags to run (default: all). Options: {[t for t,_ in LOSS_CONFIGS]}")
    args = p.parse_args()

    loss_configs = LOSS_CONFIGS
    if args.losses:
        wanted = set(args.losses)
        loss_configs = [(t, k) for t, k in LOSS_CONFIGS if t in wanted]

    if args.smoke:
        loss_configs = loss_configs[:1]
        args.scenarios = args.scenarios[:1]
        args.seeds = args.seeds[:1]
        args.epochs = 2

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"a2_h1_{ts}.json"
    md_path = out_dir / f"a2_h1_{ts}.md"

    results = []
    total = len(args.scenarios) * len(loss_configs) * len(args.seeds)
    i = 0
    for scenario in args.scenarios:
        for loss_tag, loss_kwargs in loss_configs:
            for seed in args.seeds:
                i += 1
                print(f"\n[{i}/{total}] scenario={scenario} loss={loss_tag} seed={seed}")
                try:
                    rec = run_one(scenario, loss_tag, loss_kwargs, seed,
                                  epochs=args.epochs, batch_size=args.batch_size)
                except Exception as e:
                    rec = {"scenario": scenario, "loss": loss_tag, "seed": seed,
                           "error": str(e)}
                    print(f"  ERROR: {e}")
                results.append(rec)
                with open(json_path, "w") as f:
                    json.dump(results, f, indent=2)
                gc.collect()

    # Summary table
    import statistics
    rows = []
    for scenario in args.scenarios:
        for loss_tag, _ in loss_configs:
            snrs = [r.get("best_val_snr_db") for r in results
                    if r.get("scenario") == scenario and r.get("loss") == loss_tag
                    and r.get("best_val_snr_db") is not None]
            snrs = [s for s in snrs if s == s]  # drop NaN
            if snrs:
                mu = statistics.mean(snrs)
                sd = statistics.stdev(snrs) if len(snrs) > 1 else 0.0
            else:
                mu = sd = float("nan")
            rows.append((scenario, loss_tag, len(snrs), mu, sd))

    with open(md_path, "w") as f:
        f.write(f"# A2-H1 Loss Sweep — {ts}\n\n")
        f.write(f"Model: HybridDSGE-UNet, Variant A, robust basis S=3, width=16. ")
        f.write(f"Noise: non_gaussian. Epochs: {args.epochs}. Batch: {args.batch_size}.\n\n")
        f.write("| scenario | loss | n | val_SNR μ (dB) | σ (dB) |\n")
        f.write("|---|---|---|---|---|\n")
        for scenario, loss_tag, n, mu, sd in rows:
            f.write(f"| {scenario} | {loss_tag} | {n} | {mu:.3f} | {sd:.3f} |\n")

    print(f"\n✅ Done. Results → {json_path}\n   Summary → {md_path}")


if __name__ == "__main__":
    main()
