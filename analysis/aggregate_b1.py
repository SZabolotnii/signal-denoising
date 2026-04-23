#!/usr/bin/env python3
"""B1 cross-seed aggregator.

Pulls training_report.json from each B1 run_dir and produces:
  • overall test SNR μ±σ per (model × train_noise)
  • per-SNR curve μ±σ per (model × train_noise × snr_in)

Source of truth: each trainer's own denoise_numpy + per_snr eval (saved to
training_report.json). compare_report.py was found to have inference bugs:
ResNet mask not applied (~13 dB error), HybridDSGE regex skipped _vA suffix.
"""
from __future__ import annotations
import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FPV_DS = ROOT / "data_generation/datasets/fpv_telemetry_polygauss_qpsk_bs1024_n100000_953c56e8"

B1_RUNS = {
    42: {"gaussian":     "run_20260421_267176a3", "non_gaussian": "run_20260421_92a2e0c4"},
    43: {"gaussian":     "run_20260421_fe4d376a", "non_gaussian": "run_20260422_d98d21b3"},
    44: {"gaussian":     "run_20260422_3d92fb74", "non_gaussian": "run_20260422_1023e287"},
}
SNR_LEVELS = ["m5dB", "m2dB", "p0dB", "p3dB", "p5dB", "p8dB", "p10dB", "p12dB", "p15dB", "p18dB"]


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
    """Returns {model: {train_noise: {'overall': [snr_per_seed], 'per_snr': {label: [snr_per_seed]}, 'wavelet_test_mse': [...]}}}."""
    out = {}
    for seed, by_noise in runs.items():
        for nt, run_id in by_noise.items():
            rd = ds_root / "runs" / run_id
            rep = _load_report(rd)
            for m in rep["models"]:
                raw_name = m["model"]
                # Collapse HybridDSGE variant identifiers to a single bucket so
                # we aggregate over basis/order/variant. B1 only used one combo
                # (robust_S3_vA), but this keeps downstream sweeps compatible.
                if raw_name.startswith("HybridDSGE_UNet"):
                    model = "HybridDSGE_UNet"
                else:
                    model = raw_name
                slot = out.setdefault(model, {}).setdefault(nt, {"overall": [], "per_snr": {l: [] for l in SNR_LEVELS}, "wavelet_test_mse": [], "n_params": None, "val_snr": [], "raw_name": raw_name})
                if model == "Wavelet":
                    test_mse = float(m.get("test_metrics", {}).get("MSE", "nan"))
                    slot["wavelet_test_mse"].append(test_mse)
                else:
                    snr_overall = float(m.get("test_metrics", {}).get("SNR", "nan"))
                    slot["overall"].append(snr_overall)
                    for lbl in SNR_LEVELS:
                        v = m.get("per_snr_results", {}).get(lbl, {}).get("SNR")
                        slot["per_snr"][lbl].append(float(v) if v is not None else float("nan"))
                if "val_snr" in m and m["val_snr"] is not None:
                    slot["val_snr"].append(float(m["val_snr"]))
    return out


def build_md(data: dict, out_path: Path):
    lines = []
    lines.append("# B1 FPV main — cross-seed aggregate (3 seeds: 42, 43, 44)\n")
    lines.append("**Source:** `training_report.json` from each run (uses each Trainer's own `denoise_numpy`).\n")
    lines.append("**Note:** `compare_report.py` is broken for ResNet (mask not applied) and HybridDSGE (regex skips `_vA` suffix); aggregator bypasses it.\n\n")

    # Overall table
    lines.append("## 1. Overall test SNR (μ ± σ) [dB]\n\n")
    lines.append("Computed on the held-out 25%-test split via each trainer's `denoise_numpy`.\n\n")
    lines.append("| Model | n_params | Gaussian train | Non-Gaussian train |\n|---|---:|---:|---:|\n")
    model_order = ["UnetAutoencoder", "ResNetAutoencoder", "HybridDSGE_UNet", "Wavelet"]
    for model in model_order:
        if model not in data:
            continue
        row = [model, "—"]
        for nt in ["gaussian", "non_gaussian"]:
            slot = data[model].get(nt, {})
            if model == "Wavelet":
                vals = slot.get("wavelet_test_mse", [])
                if vals:
                    mu, sd, n = _agg(vals)
                    row.append(f"MSE {mu:.4f} ± {sd:.4f} (n={n})")
                else:
                    row.append("—")
            else:
                vals = slot.get("overall", [])
                mu, sd, n = _agg(vals)
                row.append(f"{mu:.2f} ± {sd:.2f} (n={n})" if n else "—")
        lines.append("| " + " | ".join(row) + " |\n")
    lines.append("\n")

    # Per-seed raw table for transparency
    lines.append("## 2. Per-seed test SNR [dB] (raw)\n\n")
    lines.append("| Seed | Noise train | UNet | ResNet | HybridDSGE | Wavelet (test_MSE) |\n|---|---|---:|---:|---:|---:|\n")
    for seed in sorted(B1_RUNS.keys()):
        for nt in ["gaussian", "non_gaussian"]:
            row = [str(seed), nt[:1].upper() + ("G" if nt == "gaussian" else "G")]
            row[1] = "G" if nt == "gaussian" else "NG"
            for model in ["UnetAutoencoder", "ResNetAutoencoder", "HybridDSGE_UNet"]:
                idx = list(sorted(B1_RUNS.keys())).index(seed)
                vals = data.get(model, {}).get(nt, {}).get("overall", [])
                row.append(f"{vals[idx]:.2f}" if idx < len(vals) else "—")
            wvals = data.get("Wavelet", {}).get(nt, {}).get("wavelet_test_mse", [])
            idx = list(sorted(B1_RUNS.keys())).index(seed)
            row.append(f"{wvals[idx]:.4f}" if idx < len(wvals) else "—")
            lines.append("| " + " | ".join(row) + " |\n")
    lines.append("\n")

    # Per-SNR curve table (μ ± σ)
    lines.append("## 3. Per-SNR breakdown (μ ± σ across 3 seeds) [SNR_out dB]\n\n")
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
                cells.append(f"{mu:.2f}±{sd:.2f}" if n else "—")
            lines.append("| " + " | ".join(cells) + " |\n")
        lines.append("\n")

    # Validation SNR (training-set sanity)
    lines.append("## 4. Validation SNR (μ ± σ) [dB] — training-set sanity\n\n")
    lines.append("| Model | Gaussian | Non-Gaussian |\n|---|---:|---:|\n")
    for model in model_order:
        if model not in data:
            continue
        if model == "Wavelet":
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
    p.add_argument("--ds", type=Path, default=FPV_DS, help="Dataset root")
    p.add_argument("--out", type=Path, default=ROOT / "experiments/results/b1_aggregate.md")
    p.add_argument("--out-json", type=Path, default=ROOT / "experiments/results/b1_aggregate.json")
    args = p.parse_args()

    data = collect(args.ds, B1_RUNS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    build_md(data, args.out)

    # JSON dump for downstream stats
    json_safe = {}
    for model, by_nt in data.items():
        json_safe[model] = {}
        for nt, slot in by_nt.items():
            json_safe[model][nt] = {
                "overall": slot["overall"],
                "per_snr": slot["per_snr"],
                "wavelet_test_mse": slot["wavelet_test_mse"],
                "val_snr": slot["val_snr"],
            }
    with open(args.out_json, "w") as f:
        json.dump(json_safe, f, indent=2)

    print(f"✅ Markdown → {args.out}")
    print(f"✅ JSON     → {args.out_json}")


if __name__ == "__main__":
    main()
