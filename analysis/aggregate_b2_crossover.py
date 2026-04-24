#!/usr/bin/env python3
"""B2 cross-seed crossover aggregator — builds the 4-cell (G→G, G→NG, NG→G, NG→NG)
publication table from compare_report.py CSV outputs across B2 run_dirs.

Input: 4 comparison_data_*.csv (one per B2 run_dir, 2 seeds × 2 train_noise).
Output: b2_crossover.md/.json with μ±σ per (model, train_noise, test_noise).
"""
from __future__ import annotations
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "data_generation/datasets/deep_space_polygauss_qpsk_bs1024_n400000_0310b7e7"

B2_RUNS = {
    42: {"gaussian":     "run_20260423_ed46f843", "non_gaussian": "run_20260423_3d8957d2"},
    43: {"gaussian":     "run_20260423_2b4b99cc", "non_gaussian": "run_20260424_20fc5edf"},
}
SEEDS = sorted(B2_RUNS.keys())
TRAIN_NOISES = ["gaussian", "non_gaussian"]
TEST_NOISES  = ["gaussian", "non_gaussian"]
MODELS = ["UnetAutoencoder", "ResNetAutoencoder", "HybridDSGE_UNet_robust_S3_vA", "Wavelet"]


def _latest_csv(run_dir: Path) -> Path:
    csvs = sorted(run_dir.glob("comparison_data_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No comparison_data_*.csv in {run_dir}")
    return csvs[-1]


def collect(ds_root: Path) -> dict:
    """Returns data[model][train_noise][test_noise] = list of per-seed aggregate SNR."""
    data: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for seed in SEEDS:
        for train_nt in TRAIN_NOISES:
            run_dir = ds_root / "runs" / B2_RUNS[seed][train_nt]
            csv_path = _latest_csv(run_dir)
            with open(csv_path) as f:
                rdr = csv.DictReader(f)
                for row in rdr:
                    if row["snr_in_db"] != "all":
                        continue
                    model = row["model"]
                    test_nt = row["noise_tested"]
                    snr = float(row["snr_out_db"])
                    data[model][train_nt][test_nt].append(snr)
    return data


def _agg(vals: list[float]) -> tuple[float, float, int]:
    vals = [v for v in vals if v == v]
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan"), 0
    mu = statistics.mean(vals)
    sd = statistics.stdev(vals) if n > 1 else 0.0
    return mu, sd, n


def build_md(data: dict, out_path: Path):
    lines = []
    lines.append("# B2 deep_space — crossover matrix (n=2 seeds)\n\n")
    lines.append("Cross-evaluation: each model trained on `noise_trained` is "
                 "evaluated on test sets for both noise types. Source: 4 "
                 "`compare_report.py` CSV outputs (see §11.8 for authoritative "
                 "inference fixes).\n\n")

    lines.append("## 1. Crossover μ±σ [dB] (n=2 seeds; 'all' = aggregate SNR_out)\n\n")
    lines.append("| Model | Train | G→G | G→NG | NG→G | NG→NG |\n")
    lines.append("|---|---|---:|---:|---:|---:|\n")
    for model in MODELS:
        if model not in data:
            continue
        # One row per model: four cells corresponding to the (train, test) combos
        cells = []
        cells.append(model)
        cells.append("μ ± σ")
        for train_nt, test_nt in [("gaussian", "gaussian"),
                                   ("gaussian", "non_gaussian"),
                                   ("non_gaussian", "gaussian"),
                                   ("non_gaussian", "non_gaussian")]:
            vals = data[model].get(train_nt, {}).get(test_nt, [])
            mu, sd, n = _agg(vals)
            cells.append(f"{mu:+.2f} ± {sd:.2f}" if n else "—")
        lines.append("| " + " | ".join(cells) + " |\n")
    lines.append("\n")

    # Per-seed raw table
    lines.append("## 2. Per-seed crossover [dB]\n\n")
    lines.append("| Seed | Model | G→G | G→NG | NG→G | NG→NG |\n")
    lines.append("|---|---|---:|---:|---:|---:|\n")
    for i, seed in enumerate(SEEDS):
        for model in MODELS:
            if model not in data:
                continue
            row = [str(seed), model]
            for train_nt, test_nt in [("gaussian", "gaussian"),
                                       ("gaussian", "non_gaussian"),
                                       ("non_gaussian", "gaussian"),
                                       ("non_gaussian", "non_gaussian")]:
                vals = data[model].get(train_nt, {}).get(test_nt, [])
                row.append(f"{vals[i]:+.2f}" if i < len(vals) else "—")
            lines.append("| " + " | ".join(row) + " |\n")
    lines.append("\n")

    # Key deltas
    lines.append("## 3. Key deltas for central hypothesis\n\n")
    lines.append("**Hypothesis:** NG-training generalizes better on NG test "
                 "(NG→NG > G→NG), and maintains or loses minimally on G test "
                 "(NG→G ≥ G→G − ε).\n\n")
    lines.append("| Model | Δ_NG_test = NG→NG − G→NG | Δ_G_test = NG→G − G→G |\n")
    lines.append("|---|---:|---:|\n")
    for model in MODELS:
        if model not in data:
            continue
        def μ(tn, tt):
            vals = data[model].get(tn, {}).get(tt, [])
            mu, _, _ = _agg(vals)
            return mu
        d_ng = μ("non_gaussian", "non_gaussian") - μ("gaussian", "non_gaussian")
        d_g = μ("non_gaussian", "gaussian") - μ("gaussian", "gaussian")
        lines.append(f"| {model} | {d_ng:+.2f} | {d_g:+.2f} |\n")
    lines.append("\n")

    out_path.write_text("".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ds", type=Path, default=DS)
    p.add_argument("--out", type=Path,
                   default=ROOT / "experiments/results/b2_crossover.md")
    p.add_argument("--out-json", type=Path,
                   default=ROOT / "experiments/results/b2_crossover.json")
    args = p.parse_args()

    data = collect(args.ds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    build_md(data, args.out)

    json_safe = {
        model: {
            tn: {tt: vals for tt, vals in by_tt.items()}
            for tn, by_tt in by_tn.items()
        }
        for model, by_tn in data.items()
    }
    with open(args.out_json, "w") as f:
        json.dump(json_safe, f, indent=2)

    print(f"✅ Markdown → {args.out}")
    print(f"✅ JSON     → {args.out_json}")


if __name__ == "__main__":
    main()
