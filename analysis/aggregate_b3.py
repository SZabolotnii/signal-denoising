#!/usr/bin/env python3
"""B3 zero-shot cross-seed aggregator.

Reads per-run JSON outputs of experiments/b3_real_sdr_zeroshot.py and emits
cross-seed μ±σ tables by (scenario, train_noise, model).
"""
from __future__ import annotations
import glob
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
B3_GLOB = str(ROOT / "experiments/results/b3_zeroshot_run_*.json")

# Map run_id → (scenario, seed, train_noise)
RUN_MAP = {
    # FPV B1 (per §11.1)
    "run_20260421_267176a3": ("FPV", 42, "gaussian"),
    "run_20260421_92a2e0c4": ("FPV", 42, "non_gaussian"),
    "run_20260421_fe4d376a": ("FPV", 43, "gaussian"),
    "run_20260422_d98d21b3": ("FPV", 43, "non_gaussian"),
    "run_20260422_3d92fb74": ("FPV", 44, "gaussian"),
    "run_20260422_1023e287": ("FPV", 44, "non_gaussian"),
    # deep_space B2 (per §14)
    "run_20260423_ed46f843": ("deep_space", 42, "gaussian"),
    "run_20260423_3d8957d2": ("deep_space", 42, "non_gaussian"),
    "run_20260423_2b4b99cc": ("deep_space", 43, "gaussian"),
    "run_20260424_20fc5edf": ("deep_space", 43, "non_gaussian"),
}


def _model_base(name: str) -> str:
    """Collapse run-specific suffixes so we aggregate over seeds."""
    if name.startswith("HybridDSGE_UNet"):
        return "HybridDSGE_UNet"
    for trunc in ("_non_gaussian", "_gaussian"):
        if name.endswith(trunc):
            return name[: -len(trunc)]
    return name


def _agg(vals: list[float]) -> tuple[float, float, int]:
    vals = [v for v in vals if v == v]
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan"), 0
    mu = statistics.mean(vals)
    sd = statistics.stdev(vals) if n > 1 else 0.0
    return mu, sd, n


def collect() -> dict:
    """data[scenario][train_noise][model] = list of per-seed mean-SNR scalars."""
    data: dict = {}
    for path in sorted(glob.glob(B3_GLOB)):
        with open(path) as f:
            payload = json.load(f)
        b1 = Path(payload["b1_run"]).name
        if b1 not in RUN_MAP:
            continue
        scenario, seed, train_nt = RUN_MAP[b1]
        results = payload["results"]
        for model_name, per_snr in results.items():
            if not per_snr:
                continue
            snrs = [v["SNR"] for v in per_snr.values() if v.get("SNR") is not None]
            if not snrs:
                continue
            mean_snr = sum(snrs) / len(snrs)
            mb = _model_base(model_name)
            (data.setdefault(scenario, {})
                 .setdefault(train_nt, {})
                 .setdefault(mb, [])).append(mean_snr)
    return data


def build_md(data: dict, out: Path):
    lines = ["# B3 zero-shot on RadioML 2018.01A — cross-seed aggregate\n\n"]
    lines.append("Models trained on synthetic polygauss datasets (FPV B1 3 seeds, "
                 "deep_space B2 2 seeds) evaluated zero-shot on real RadioML 2018 "
                 "BPSK+QPSK frames at SNR levels matched to the training scenario. "
                 "Metric: mean SNR_out across per-SNR bins.\n\n")

    lines.append("## 1. Per-scenario × train_noise aggregate (mean SNR ± σ, dB)\n\n")
    for scenario in ["FPV", "deep_space"]:
        if scenario not in data:
            continue
        lines.append(f"### {scenario}\n\n")
        lines.append("| Model | G-trained μ±σ | NG-trained μ±σ | Δ(NG−G) |\n")
        lines.append("|---|---|---|---:|\n")
        # Collect union of models
        models = sorted({m for nt in data[scenario].values() for m in nt.keys()})
        for model in models:
            g_vals = data[scenario].get("gaussian", {}).get(model, [])
            ng_vals = data[scenario].get("non_gaussian", {}).get(model, [])
            mu_g, sd_g, n_g = _agg(g_vals)
            mu_ng, sd_ng, n_ng = _agg(ng_vals)
            delta = (mu_ng - mu_g) if (n_g and n_ng) else float("nan")
            lines.append(
                f"| {model} "
                f"| {mu_g:+.2f} ± {sd_g:.2f} (n={n_g}) "
                f"| {mu_ng:+.2f} ± {sd_ng:.2f} (n={n_ng}) "
                f"| {delta:+.2f} |\n"
            )
        lines.append("\n")

    # Per-seed raw
    lines.append("## 2. Per-seed raw mean SNR (dB)\n\n")
    for scenario in ["FPV", "deep_space"]:
        if scenario not in data:
            continue
        lines.append(f"### {scenario}\n\n")
        # Rebuild seed-indexed table
        by_seed: dict = {}
        for run_id, (sc, seed, train_nt) in RUN_MAP.items():
            if sc != scenario:
                continue
            by_seed.setdefault(seed, {}).setdefault(train_nt, {})
        seeds = sorted(by_seed.keys())
        lines.append("| Seed | Train | " + " | ".join(sorted({m for nt in data[scenario].values() for m in nt})) + " |\n")
        n_models = len({m for nt in data[scenario].values() for m in nt})
        lines.append("|---|---|" + "---:|" * n_models + "\n")
        # Re-iterate per-seed per-train using data order (same order we appended)
        for seed in seeds:
            for train_nt in ["gaussian", "non_gaussian"]:
                row = [str(seed), "G" if train_nt == "gaussian" else "NG"]
                for model in sorted({m for nt in data[scenario].values() for m in nt}):
                    vals = data[scenario].get(train_nt, {}).get(model, [])
                    idx = seeds.index(seed)
                    row.append(f"{vals[idx]:+.2f}" if idx < len(vals) else "—")
                lines.append("| " + " | ".join(row) + " |\n")
        lines.append("\n")

    out.write_text("".join(lines))


def main():
    data = collect()
    out_md = ROOT / "experiments/results/b3_aggregate.md"
    out_json = ROOT / "experiments/results/b3_aggregate.json"
    build_md(data, out_md)
    with open(out_json, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ {out_md}")
    print(f"✅ {out_json}")


if __name__ == "__main__":
    main()
