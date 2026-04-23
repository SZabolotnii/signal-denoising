#!/usr/bin/env python3
"""A2-H2: Ratio-mask collapse hypothesis test.

Does swapping the sigmoid ratio-mask for an additive residual correction
recover non-Gaussian training performance?

Configs: mask ∈ {ratio, additive} × seed ∈ {42, 43, 44}, non_gaussian noise,
FPV 25%, 15 epochs. Uses default SmoothL1(β=0.02) loss (H1 baseline condition).

Epoch logs include mask statistics (min/max/μ/σ) for distribution analysis.

Runtime estimate: 6 runs × ~9 min = ~1 h on M3 Max CPU.
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
# Match H1 data fractions so H1↔H2 results are directly comparable.
SCENARIO_FRACTIONS = {"fpv": 0.25, "deep_space": 0.10}
SEEDS = [42, 43, 44]
MASK_TYPES = ["ratio", "additive"]


def run_one(scenario: str, mask_type: str, seed: int, epochs: int, batch_size: int) -> dict:
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
        mask_type=mask_type,
    )
    result = trainer.train()
    return {
        "scenario": scenario,
        "mask_type": mask_type,
        "seed": seed,
        "epochs": epochs,
        "best_val_snr_db": float(result.get("val_snr", float("nan"))),
        "final_test_snr_db": float(result.get("test_metrics", {}).get("SNR", float("nan"))),
        "final_mask_stats": getattr(trainer, "_last_mask_stats", None),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--mask-types", nargs="+", default=MASK_TYPES)
    p.add_argument("--scenarios", nargs="+", default=list(DATASETS.keys()))
    args = p.parse_args()

    if args.smoke:
        args.seeds = args.seeds[:1]
        args.scenarios = args.scenarios[:1]
        args.epochs = 2

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"a2_h2_{ts}.json"
    md_path = out_dir / f"a2_h2_{ts}.md"

    results = []
    total = len(args.scenarios) * len(args.mask_types) * len(args.seeds)
    i = 0
    for scenario in args.scenarios:
        for mask_type in args.mask_types:
            for seed in args.seeds:
                i += 1
                print(f"\n[{i}/{total}] scenario={scenario} mask_type={mask_type} seed={seed}")
                try:
                    rec = run_one(scenario, mask_type, seed,
                                  epochs=args.epochs, batch_size=args.batch_size)
                except Exception as e:
                    rec = {"scenario": scenario, "mask_type": mask_type, "seed": seed, "error": str(e)}
                    print(f"  ERROR: {e}")
                results.append(rec)
                with open(json_path, "w") as f:
                    json.dump(results, f, indent=2)
                gc.collect()

    import statistics
    with open(md_path, "w") as f:
        f.write(f"# A2-H2 Mask-type Test — {ts}\n\n")
        f.write("HybridDSGE-UNet, Variant A, robust basis S=3, width=16, ")
        f.write(f"non_gaussian, SmoothL1(β=0.02), epochs={args.epochs}.\n\n")
        f.write("| scenario | mask_type | n | val_SNR μ (dB) | σ (dB) |\n|---|---|---|---|---|\n")
        for scenario in args.scenarios:
            for mask_type in args.mask_types:
                snrs = [r["best_val_snr_db"] for r in results
                        if r.get("scenario") == scenario and r.get("mask_type") == mask_type
                        and r.get("best_val_snr_db") == r.get("best_val_snr_db")]
                if snrs:
                    mu = statistics.mean(snrs)
                    sd = statistics.stdev(snrs) if len(snrs) > 1 else 0.0
                else:
                    mu = sd = float("nan")
                f.write(f"| {scenario} | {mask_type} | {len(snrs)} | {mu:.3f} | {sd:.3f} |\n")

    print(f"\n✅ Done. Results → {json_path}\n   Summary → {md_path}")


if __name__ == "__main__":
    main()
