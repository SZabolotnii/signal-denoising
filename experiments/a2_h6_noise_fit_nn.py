#!/usr/bin/env python3
"""A2-H6: Noise-fitted DSGE as feature extractor for the NN.

Motivation: H5 (Phase 1) showed that noise-subspace DSGE as a *standalone
denoiser* collapses to trivial near-identity solutions and destroys signal.
H6 tests whether feeding noise-fitted DSGE features into the neural network
still helps, because the NN sees all channels (original + reconstruction +
residual) and can learn which combinations extract signal.

Three conditions × 2 scenarios × 3 seeds = 18 runs:
  - fit_target=signal: current baseline (DSGE fit on clean/noisy pairs)
  - fit_target=noise : DSGE fit with target=noise, input=noise
  - fit_target=n2n   : DSGE fit with target=noise_B (shuffled), input=noise_A.
                       For iid noise, theoretically K→0 (sanity check).

Other settings: Variant A, robust S=3, width=16, SmoothL1(β=0.02),
ratio mask, non_gaussian noise. FPV 25%, deep_space 10%. 8 epochs.

Runtime estimate: 18 runs × ~6 min = ~1.8 h on M3 Max CPU (in parallel with H2).
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
FIT_TARGETS = ["signal", "noise", "n2n"]


def run_one(scenario: str, fit_target: str, seed: int, epochs: int, batch_size: int) -> dict:
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
        dsge_fit_target=fit_target,
    )
    result = trainer.train()
    return {
        "scenario": scenario,
        "fit_target": fit_target,
        "seed": seed,
        "epochs": epochs,
        "best_val_snr_db": float(result.get("val_snr", float("nan"))),
        "final_test_snr_db": float(result.get("test_metrics", {}).get("SNR", float("nan"))),
        "dsge_K": list(map(float, trainer.dsge.K.tolist())),
        "dsge_k0": float(trainer.dsge.k0),
        "gen_element_norm": float(trainer.dsge.gen_element_norm),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--fit-targets", nargs="+", default=FIT_TARGETS)
    p.add_argument("--scenarios", nargs="+", default=list(DATASETS.keys()))
    args = p.parse_args()

    if args.smoke:
        args.seeds = args.seeds[:1]
        args.scenarios = args.scenarios[:1]
        args.fit_targets = args.fit_targets[:1]
        args.epochs = 2

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"a2_h6_{ts}.json"
    md_path = out_dir / f"a2_h6_{ts}.md"

    results = []
    total = len(args.scenarios) * len(args.fit_targets) * len(args.seeds)
    i = 0
    for scenario in args.scenarios:
        for fit_target in args.fit_targets:
            for seed in args.seeds:
                i += 1
                print(f"\n[{i}/{total}] scenario={scenario} fit_target={fit_target} seed={seed}")
                try:
                    rec = run_one(scenario, fit_target, seed,
                                  epochs=args.epochs, batch_size=args.batch_size)
                except Exception as e:
                    rec = {"scenario": scenario, "fit_target": fit_target, "seed": seed, "error": str(e)}
                    import traceback; traceback.print_exc()
                    print(f"  ERROR: {e}")
                results.append(rec)
                with open(json_path, "w") as f:
                    json.dump(results, f, indent=2, default=str)
                gc.collect()

    import statistics
    with open(md_path, "w") as f:
        f.write(f"# A2-H6 Noise-fitted DSGE as NN features — {ts}\n\n")
        f.write("HybridDSGE-UNet, Variant A, robust S=3, width=16, SmoothL1(β=0.02), ratio mask, ")
        f.write(f"non_gaussian. epochs={args.epochs}.\n\n")
        f.write("| scenario | fit_target | n | val_SNR μ (dB) | σ (dB) |\n|---|---|---|---|---|\n")
        for scenario in args.scenarios:
            for ft in args.fit_targets:
                snrs = [r["best_val_snr_db"] for r in results
                        if r.get("scenario") == scenario and r.get("fit_target") == ft
                        and r.get("best_val_snr_db") == r.get("best_val_snr_db")]
                if snrs:
                    mu = statistics.mean(snrs)
                    sd = statistics.stdev(snrs) if len(snrs) > 1 else 0.0
                else:
                    mu = sd = float("nan")
                f.write(f"| {scenario} | {ft} | {len(snrs)} | {mu:.3f} | {sd:.3f} |\n")

    print(f"\n✅ Done. Results → {json_path}\n   Summary → {md_path}")


if __name__ == "__main__":
    main()
