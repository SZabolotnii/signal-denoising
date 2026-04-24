#!/usr/bin/env python3
"""B2 main cross-seed aggregator — deep_space, 2 seeds, 2 noise types.

Analog of aggregate_b1 adapted for deep_space SNR grid (−20..+3 dB) and
the 4 run_dirs produced by experiments/b2_main.sh (killed after [4/6]).

Emits both:
  * `overall_testset` — test_metrics.SNR (pooled SNR over all test samples;
    consistent with aggregate_b1 "overall" column).
  * `overall_per_snr_mean` — unweighted mean of per-SNR-bin SNR_out;
    more informative on deep_space where low-SNR bins dominate pooled SNR.
"""
from __future__ import annotations
import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "data_generation/datasets/deep_space_polygauss_qpsk_bs1024_n400000_0310b7e7"

B2_RUNS = {
    42: {"gaussian":     "run_20260423_ed46f843", "non_gaussian": "run_20260423_3d8957d2"},
    43: {"gaussian":     "run_20260423_2b4b99cc", "non_gaussian": "run_20260424_20fc5edf"},
}
SNR_LEVELS = ["m20dB", "m17dB", "m15dB", "m12dB", "m10dB",
              "m7dB", "m5dB", "m3dB", "p0dB", "p3dB"]

MODEL_ORDER = ["UnetAutoencoder", "ResNetAutoencoder", "HybridDSGE_UNet", "Wavelet"]


def _load_report(run_dir: Path) -> dict:
    reports = sorted(run_dir.glob("training_report_*.json"))
    if not reports:
        raise FileNotFoundError(f"No training_report in {run_dir}")
    with open(reports[-1]) as f:
        return json.load(f)


def _agg(vals: list[float]) -> tuple[float, float, int]:
    vals = [v for v in vals if v == v]
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan"), 0
    mu = statistics.mean(vals)
    sd = statistics.stdev(vals) if n > 1 else 0.0
    return mu, sd, n


def collect(ds_root: Path, runs: dict) -> dict:
    out = {}
    seeds = sorted(runs.keys())
    for seed in seeds:
        for nt, run_id in runs[seed].items():
            rd = ds_root / "runs" / run_id
            rep = _load_report(rd)
            for m in rep["models"]:
                raw_name = m["model"]
                if raw_name.startswith("HybridDSGE_UNet"):
                    model = "HybridDSGE_UNet"
                else:
                    model = raw_name
                slot = out.setdefault(model, {}).setdefault(nt, {
                    "overall_testset": [],
                    "per_snr": {l: [] for l in SNR_LEVELS},
                    "wavelet_test_mse": [],
                    "val_snr": [],
                    "raw_name": raw_name,
                    "seeds": [],
                })
                slot["seeds"].append(seed)
                if model == "Wavelet":
                    test_mse = float(m.get("test_metrics", {}).get("MSE", "nan"))
                    slot["wavelet_test_mse"].append(test_mse)
                else:
                    snr_overall = float(m.get("test_metrics", {}).get("SNR", "nan"))
                    slot["overall_testset"].append(snr_overall)
                    for lbl in SNR_LEVELS:
                        v = m.get("per_snr_results", {}).get(lbl, {}).get("SNR")
                        slot["per_snr"][lbl].append(
                            float(v) if v is not None else float("nan")
                        )
                if "val_snr" in m and m["val_snr"] is not None:
                    slot["val_snr"].append(float(m["val_snr"]))
    return out


def _per_snr_mean_per_seed(slot: dict) -> list[float]:
    """Returns per-seed list of unweighted mean-of-bins SNR_out.
    Each seed contributes a scalar = mean over 10 SNR bins."""
    per_bin = slot.get("per_snr", {})
    if not per_bin or not any(per_bin.values()):
        return []
    n_seeds = len(next(iter(per_bin.values())))
    out = []
    for i in range(n_seeds):
        vals = [per_bin[lbl][i] for lbl in SNR_LEVELS
                if i < len(per_bin[lbl]) and per_bin[lbl][i] == per_bin[lbl][i]]
        if vals:
            out.append(sum(vals) / len(vals))
    return out


def build_md(data: dict, out_path: Path):
    lines = []
    lines.append("# B2 deep_space main — cross-seed aggregate (2 seeds: 42, 43)\n\n")
    lines.append("Source: `experiments/b2_main.sh`, killed after [4/6] at 13:45 "
                 "EEST 2026-04-24. 4 run_dirs in "
                 "`data_generation/datasets/deep_space_polygauss_qpsk_bs1024_n400000_0310b7e7/runs/`.\n\n")
    lines.append("Metrics authority: `training_report_*.json` per trainer's "
                 "`denoise_numpy`. Two aggregates reported:\n")
    lines.append("- **overall (testset)** — `test_metrics.SNR` (pooled SNR, "
                 "low-SNR dominated).\n")
    lines.append("- **overall (per-SNR mean)** — unweighted mean of 10 per-SNR-bin "
                 "SNR_out values (equal weighting).\n\n")

    # Table 1: Per-noise overall aggregates
    lines.append("## 1. Overall SNR μ ± σ [dB] (n=2 seeds)\n\n")
    lines.append("| Model | Metric | Gaussian | Non-Gaussian | Δ(NG−G) |\n")
    lines.append("|---|---|---|---|---:|\n")
    for model in MODEL_ORDER:
        if model not in data:
            continue
        if model == "Wavelet":
            vals_g = data[model].get("gaussian", {}).get("wavelet_test_mse", [])
            vals_ng = data[model].get("non_gaussian", {}).get("wavelet_test_mse", [])
            mu_g, sd_g, n_g = _agg(vals_g)
            mu_ng, sd_ng, n_ng = _agg(vals_ng)
            lines.append(f"| {model} | test_MSE | {mu_g:.4f} ± {sd_g:.4f} (n={n_g}) "
                         f"| {mu_ng:.4f} ± {sd_ng:.4f} (n={n_ng}) | — |\n")
            continue
        for metric_key, metric_label in [
            ("overall_testset", "testset SNR"),
            ("per_snr_mean",    "per-SNR mean"),
        ]:
            if metric_key == "per_snr_mean":
                vals_g = _per_snr_mean_per_seed(data[model].get("gaussian", {}))
                vals_ng = _per_snr_mean_per_seed(data[model].get("non_gaussian", {}))
            else:
                vals_g = data[model].get("gaussian", {}).get(metric_key, [])
                vals_ng = data[model].get("non_gaussian", {}).get(metric_key, [])
            mu_g, sd_g, n_g = _agg(vals_g)
            mu_ng, sd_ng, n_ng = _agg(vals_ng)
            delta = (mu_ng - mu_g) if (n_g and n_ng) else float("nan")
            lines.append(
                f"| {model} | {metric_label} "
                f"| {mu_g:.2f} ± {sd_g:.2f} (n={n_g}) "
                f"| {mu_ng:.2f} ± {sd_ng:.2f} (n={n_ng}) "
                f"| {delta:+.2f} |\n"
            )
    lines.append("\n")

    # Table 2: Per-seed raw
    lines.append("## 2. Per-seed test SNR [dB] (raw)\n\n")
    lines.append("| Seed | Noise | UNet (testset) | UNet (per-SNR mean) "
                 "| ResNet (testset) | ResNet (per-SNR mean) "
                 "| Hybrid (testset) | Hybrid (per-SNR mean) "
                 "| Wavelet (test_MSE) |\n")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for seed in sorted(B2_RUNS.keys()):
        for nt in ["gaussian", "non_gaussian"]:
            nt_lbl = "G" if nt == "gaussian" else "NG"
            row = [str(seed), nt_lbl]
            for model in ["UnetAutoencoder", "ResNetAutoencoder", "HybridDSGE_UNet"]:
                slot = data.get(model, {}).get(nt, {})
                idx = sorted(B2_RUNS.keys()).index(seed)
                vals_ts = slot.get("overall_testset", [])
                per_seed_psm = _per_snr_mean_per_seed(slot)
                row.append(f"{vals_ts[idx]:.2f}" if idx < len(vals_ts) else "—")
                row.append(f"{per_seed_psm[idx]:.2f}" if idx < len(per_seed_psm) else "—")
            wvals = data.get("Wavelet", {}).get(nt, {}).get("wavelet_test_mse", [])
            idx = sorted(B2_RUNS.keys()).index(seed)
            row.append(f"{wvals[idx]:.4f}" if idx < len(wvals) else "—")
            lines.append("| " + " | ".join(row) + " |\n")
    lines.append("\n")

    # Table 3: Per-SNR curves (μ ± σ)
    lines.append("## 3. Per-SNR breakdown (μ ± σ across 2 seeds) [SNR_out dB]\n\n")
    for nt in ["gaussian", "non_gaussian"]:
        lines.append(f"### {nt}\n\n")
        header = "| Model |" + "".join(f" {l} |" for l in SNR_LEVELS) + "\n"
        sep = "|---|" + ("---:|" * len(SNR_LEVELS)) + "\n"
        lines.append(header)
        lines.append(sep)
        for model in ["UnetAutoencoder", "ResNetAutoencoder", "HybridDSGE_UNet"]:
            slot = data.get(model, {}).get(nt, {})
            cells = [model]
            for lbl in SNR_LEVELS:
                vals = slot.get("per_snr", {}).get(lbl, [])
                mu, sd, n = _agg(vals)
                cells.append(f"{mu:+.2f}±{sd:.2f}" if n else "—")
            lines.append("| " + " | ".join(cells) + " |\n")
        lines.append("\n")

    # Table 4: Validation SNR
    lines.append("## 4. Validation SNR (μ ± σ) [dB]\n\n")
    lines.append("| Model | Gaussian | Non-Gaussian |\n|---|---:|---:|\n")
    for model in MODEL_ORDER:
        if model not in data or model == "Wavelet":
            continue
        row = [model]
        for nt in ["gaussian", "non_gaussian"]:
            vals = data[model].get(nt, {}).get("val_snr", [])
            mu, sd, n = _agg(vals)
            row.append(f"{mu:.2f} ± {sd:.2f} (n={n})" if n else "—")
        lines.append("| " + " | ".join(row) + " |\n")
    lines.append("\n")

    out_path.write_text("".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ds", type=Path, default=DS)
    p.add_argument("--out", type=Path,
                   default=ROOT / "experiments/results/b2_aggregate.md")
    p.add_argument("--out-json", type=Path,
                   default=ROOT / "experiments/results/b2_aggregate.json")
    args = p.parse_args()

    data = collect(args.ds, B2_RUNS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    build_md(data, args.out)

    json_safe = {}
    for model, by_nt in data.items():
        json_safe[model] = {}
        for nt, slot in by_nt.items():
            json_safe[model][nt] = {
                "overall_testset": slot["overall_testset"],
                "per_snr_mean_per_seed": _per_snr_mean_per_seed(slot),
                "per_snr": slot["per_snr"],
                "wavelet_test_mse": slot["wavelet_test_mse"],
                "val_snr": slot["val_snr"],
                "seeds": slot["seeds"],
            }
    with open(args.out_json, "w") as f:
        json.dump(json_safe, f, indent=2)

    print(f"✅ Markdown → {args.out}")
    print(f"✅ JSON     → {args.out_json}")


if __name__ == "__main__":
    main()
