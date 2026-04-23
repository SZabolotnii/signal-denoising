#!/usr/bin/env python3
"""A2-H3: Robust per-channel normalization.

Motivation: H1, H2, H4, H5, H6 have collectively ruled out every DSGE-side
lever as the mechanism behind NG-training failures. H3 is the last A2 probe —
the NN-input normalization. Current pipeline uses `_p99` (99th percentile)
as the per-channel scale reference. Under heavy-tailed NG noise the top-1%
tail still dominates p99, so the input scales seen by the NN can vary strongly
across signals.

H3 tests whether MAD-based normalization (median(|x|) × 1.4826, std-equivalent
under Gaussian, insensitive to the extreme tail) stabilizes training.

Design: 2 scenarios × {p99, mad} × 3 seeds = 12 runs.
Baseline conditions match H1/H4: Variant A, robust S=3, width=16,
SmoothL1(β=0.02), ratio mask, non_gaussian, 8 epochs.
FPV 25% / deep_space 10%.

Predictions (from priority-decision analysis):
  • deep_space ratio-arm dead zone: unlikely to move (dead zone is a
    parameterization property, not an absolute-scale property).
  • deep_space additive-arm divergence (H2): was μ=−0.99 σ=0.75 — plausibly
    bounded by MAD; but additive isn't tested here (this is ratio mask).
  • FPV bimodality: unlikely to change (H6 ruled out DSGE-side causes).
  → Most likely outcome: all 4 (method, scenario) cells reproduce the
    baseline. This would be the final piece of evidence that NG failures are
    NN/loss/mask-side, not normalization-side.
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
SEEDS = [42, 43, 44]
METHODS = ["p99", "mad"]


def run_one(scenario: str, method: str, seed: int, epochs: int, batch_size: int) -> dict:
    ds_path = DATASETS[scenario]
    with open(ds_path / "dataset_config.json") as f:
        cfg = json.load(f)
    trainer = HybridUnetTrainer(
        dataset_path=ds_path,
        noise_type="non_gaussian",
        dsge_order=3, dsge_basis="robust", dsge_variant="A",
        unet_width=16,
        batch_size=batch_size, epochs=epochs, learning_rate=3e-4,
        signal_len=cfg["block_size"], fs=cfg["sample_rate"],
        nperseg=128, noverlap=96,
        random_state=seed, data_fraction=SCENARIO_FRACTIONS[scenario], device="cpu",
        dsge_norm_method=method,
    )
    result = trainer.train()
    return {
        "scenario": scenario,
        "method": method,
        "seed": seed,
        "epochs": epochs,
        "best_val_snr_db": float(result.get("val_snr", float("nan"))),
        "final_test_snr_db": float(result.get("test_metrics", {}).get("SNR", float("nan"))),
        "gen_element_norm": float(trainer.dsge.gen_element_norm),
        "dsge_K": list(map(float, trainer.dsge.K.tolist())),
        "dsge_k0": float(trainer.dsge.k0),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--methods", nargs="+", default=METHODS)
    p.add_argument("--scenarios", nargs="+", default=list(DATASETS.keys()))
    args = p.parse_args()

    if args.smoke:
        args.seeds = args.seeds[:1]
        args.scenarios = args.scenarios[:1]
        args.methods = args.methods[-1:]  # only mad under smoke
        args.epochs = 2

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"a2_h3_{ts}.json"
    md_path = out_dir / f"a2_h3_{ts}.md"

    results = []
    total = len(args.scenarios) * len(args.methods) * len(args.seeds)
    i = 0
    for scenario in args.scenarios:
        for method in args.methods:
            for seed in args.seeds:
                i += 1
                print(f"\n[{i}/{total}] scenario={scenario} method={method} seed={seed}")
                try:
                    rec = run_one(scenario, method, seed,
                                  epochs=args.epochs, batch_size=args.batch_size)
                except Exception as e:
                    rec = {"scenario": scenario, "method": method, "seed": seed, "error": str(e)}
                    import traceback; traceback.print_exc()
                    print(f"  ERROR: {e}")
                results.append(rec)
                with open(json_path, "w") as f:
                    json.dump(results, f, indent=2, default=str)
                gc.collect()

    import statistics
    with open(md_path, "w") as f:
        f.write(f"# A2-H3 robust per-channel normalization — {ts}\n\n")
        f.write("HybridDSGE-UNet, Variant A, robust S=3, width=16, SmoothL1(β=0.02), ratio mask, ")
        f.write(f"non_gaussian. epochs={args.epochs}.\n\n")
        f.write("method=p99: 99th percentile scale. method=mad: median(|x|)×1.4826.\n\n")
        f.write("| scenario | method | n_runs | val_SNR μ (dB) | σ (dB) |\n|---|---|---|---|---|\n")
        for scenario in args.scenarios:
            for method in args.methods:
                snrs = [r["best_val_snr_db"] for r in results
                        if r.get("scenario") == scenario and r.get("method") == method
                        and r.get("best_val_snr_db") == r.get("best_val_snr_db")]
                if snrs:
                    mu = statistics.mean(snrs)
                    sd = statistics.stdev(snrs) if len(snrs) > 1 else 0.0
                else:
                    mu = sd = float("nan")
                f.write(f"| {scenario} | {method} | {len(snrs)} | {mu:.3f} | {sd:.3f} |\n")

    print(f"\n✅ Done. Results → {json_path}\n   Summary → {md_path}")


if __name__ == "__main__":
    main()
