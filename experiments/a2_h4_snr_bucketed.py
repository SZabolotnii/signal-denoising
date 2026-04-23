#!/usr/bin/env python3
"""A2-H4: SNR-bucketed Generating Element (class-specific DSGE fit).

Motivation: H1, H2, H6 all confirm that on deep_space every NN training
configuration collapses to ≈0 dB regardless of loss, mask type, or DSGE
input channel. H4 is the first probe that varies the *DSGE itself* — it
fits separate K, k0 per SNR bucket (quantile-binned per-sample SNR) and
routes signals via oracle ground-truth SNR (available through
`snr_values.npy`).

Design: 2 scenarios × 2 conditions (bins=0 baseline, bins=3 class-specific)
× 3 seeds = 12 runs. Same baseline conditions as H6 for comparability:
Variant A, robust S=3, width=16, SmoothL1(β=0.02), ratio mask, non_gaussian,
FPV 25% / deep_space 10%, 8 epochs.

Predictions:
  • FPV: bucketing may stabilize the bimodal seed behavior if the +8 dB
    basin is really an SNR-adequate GE regime rather than a training-noise
    artifact. Less likely though given H6's DSGE-independence finding.
  • deep_space: if bucketing helps, per-bucket K should specialize for
    the low-SNR regime where a single averaged GE fit is most compromised.
    If it still collapses, fundamental limitation is confirmed.
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
BIN_VALUES = [0, 3]  # 0 = global baseline, 3 = class-specific


def run_one(scenario: str, n_bins: int, seed: int, epochs: int, batch_size: int) -> dict:
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
        dsge_fit_target="signal",
        dsge_snr_bins=n_bins,
    )
    result = trainer.train()
    rec = {
        "scenario": scenario,
        "n_bins": n_bins,
        "seed": seed,
        "epochs": epochs,
        "best_val_snr_db": float(result.get("val_snr", float("nan"))),
        "final_test_snr_db": float(result.get("test_metrics", {}).get("SNR", float("nan"))),
        "gen_element_norm": float(trainer.dsge.gen_element_norm),
    }
    if n_bins == 0:
        rec["dsge_K"] = list(map(float, trainer.dsge.K.tolist()))
        rec["dsge_k0"] = float(trainer.dsge.k0)
    else:
        rec["dsge_K_bins"] = [list(map(float, Kb.tolist())) for Kb in trainer.dsge.K_bins]
        rec["dsge_k0_bins"] = list(map(float, trainer.dsge.k0_bins))
        rec["snr_edges"] = [float(e) for e in trainer.dsge.snr_edges.tolist()]
        rec["bin_means"] = list(map(float, trainer.dsge.bin_means))
    return rec


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--n-bins", nargs="+", type=int, default=BIN_VALUES)
    p.add_argument("--scenarios", nargs="+", default=list(DATASETS.keys()))
    args = p.parse_args()

    if args.smoke:
        args.seeds = args.seeds[:1]
        args.scenarios = args.scenarios[:1]
        args.n_bins = args.n_bins[-1:]  # only test bucketed path under smoke
        args.epochs = 2

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"a2_h4_{ts}.json"
    md_path = out_dir / f"a2_h4_{ts}.md"

    results = []
    total = len(args.scenarios) * len(args.n_bins) * len(args.seeds)
    i = 0
    for scenario in args.scenarios:
        for n_bins in args.n_bins:
            for seed in args.seeds:
                i += 1
                print(f"\n[{i}/{total}] scenario={scenario} n_bins={n_bins} seed={seed}")
                try:
                    rec = run_one(scenario, n_bins, seed,
                                  epochs=args.epochs, batch_size=args.batch_size)
                except Exception as e:
                    rec = {"scenario": scenario, "n_bins": n_bins, "seed": seed, "error": str(e)}
                    import traceback; traceback.print_exc()
                    print(f"  ERROR: {e}")
                results.append(rec)
                with open(json_path, "w") as f:
                    json.dump(results, f, indent=2, default=str)
                gc.collect()

    import statistics
    with open(md_path, "w") as f:
        f.write(f"# A2-H4 SNR-bucketed DSGE — {ts}\n\n")
        f.write("HybridDSGE-UNet, Variant A, robust S=3, width=16, SmoothL1(β=0.02), ratio mask, ")
        f.write(f"non_gaussian. epochs={args.epochs}.\n\n")
        f.write("n_bins=0: global single-GE (baseline). n_bins=3: class-specific K per SNR bucket.\n\n")
        f.write("| scenario | n_bins | n_runs | val_SNR μ (dB) | σ (dB) |\n|---|---|---|---|---|\n")
        for scenario in args.scenarios:
            for nb in args.n_bins:
                snrs = [r["best_val_snr_db"] for r in results
                        if r.get("scenario") == scenario and r.get("n_bins") == nb
                        and r.get("best_val_snr_db") == r.get("best_val_snr_db")]
                if snrs:
                    mu = statistics.mean(snrs)
                    sd = statistics.stdev(snrs) if len(snrs) > 1 else 0.0
                else:
                    mu = sd = float("nan")
                f.write(f"| {scenario} | {nb} | {len(snrs)} | {mu:.3f} | {sd:.3f} |\n")

    print(f"\n✅ Done. Results → {json_path}\n   Summary → {md_path}")


if __name__ == "__main__":
    main()
