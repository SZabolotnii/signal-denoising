#!/usr/bin/env python3
"""B5 cross-seed aggregator — combines real-test eval results across 8 fine-tune
runs (2 scenarios × 2 noise × 2 seeds) and compares to B3 zero-shot baseline.
"""
from __future__ import annotations
import glob
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiments/results"


def _agg(vals):
    vals = [v for v in vals if v == v]
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan"), 0
    mu = statistics.mean(vals)
    sd = statistics.stdev(vals) if n > 1 else 0.0
    return mu, sd, n


def _per_run_mean(per_snr: dict) -> float:
    snrs = [v.get("SNR") for v in per_snr.values() if v.get("SNR") is not None]
    return sum(snrs) / len(snrs) if snrs else float("nan")


def collect_b5() -> dict:
    """Returns {scenario: {train_noise: [per-seed mean SNR on real test]}}."""
    data = {"FPV": {"gaussian": [], "non_gaussian": []},
            "deep_space": {"gaussian": [], "non_gaussian": []}}
    # Pick only the latest JSON per dataset (avoid double-counting reruns).
    by_ds: dict[str, str] = {}
    for p in sorted(glob.glob(str(RESULTS / "b5_eval_*_realnoise_*.json"))):
        with open(p) as f:
            ds_name = json.load(f)["b5_dataset"]
        by_ds[ds_name] = p
    for json_path in by_ds.values():
        with open(json_path) as f:
            payload = json.load(f)
        scen = "FPV" if "fpv" in Path(payload["b5_dataset"]).name else "deep_space"
        for run_label, r in payload["runs"].items():
            meta = r.get("meta", {})
            seed = meta.get("seed")
            train_nt = "non_gaussian" if "non_gaussian" in run_label else "gaussian"
            mean_snr = _per_run_mean(r["per_snr"])
            data[scen][train_nt].append((seed, mean_snr))
    return data


def collect_b3() -> dict:
    """Same shape as B5 but from b3_aggregate.json."""
    j = json.loads((RESULTS / "b3_aggregate.json").read_text())
    out = {}
    for scen in ["FPV", "deep_space"]:
        out[scen] = {}
        for nt in ["gaussian", "non_gaussian"]:
            unet_vals = j.get(scen, {}).get(nt, {}).get("UnetAutoencoder", [])
            out[scen][nt] = unet_vals
    return out


def build_md(b5: dict, b3: dict, out: Path):
    lines = ["# B5 fine-tune vs B3 zero-shot — UNet on real RadioML 2018\n\n"]
    lines.append("Same UNet architecture, same B1/B2 pretrained checkpoints. "
                 "B3: zero-shot eval. B5: pretrained → fine-tuned 10 ep on "
                 "(synthetic_clean + real_noise_injection) at lr=1e-4, partial=0.25.\n\n")
    lines.append("## 1. Mean SNR_out (dB) on actual RadioML test, μ±σ\n\n")
    lines.append("| Scenario | Train noise | B3 zero-shot | B5 fine-tune | Δ improvement |\n")
    lines.append("|---|---|---:|---:|---:|\n")
    for scen in ["FPV", "deep_space"]:
        for nt in ["gaussian", "non_gaussian"]:
            b3_vals = b3.get(scen, {}).get(nt, [])
            b5_vals = [v for _, v in b5.get(scen, {}).get(nt, [])]
            mu3, sd3, n3 = _agg(b3_vals)
            mu5, sd5, n5 = _agg(b5_vals)
            delta = (mu5 - mu3) if (n3 and n5) else float("nan")
            lines.append(
                f"| {scen} | {nt} "
                f"| {mu3:+.2f} ± {sd3:.2f} (n={n3}) "
                f"| {mu5:+.2f} ± {sd5:.2f} (n={n5}) "
                f"| **{delta:+.2f}** |\n"
            )
    lines.append("\n")

    lines.append("## 2. Per-seed B5 raw\n\n")
    lines.append("| Scenario | Train | Seed | Mean SNR (dB) |\n|---|---|---|---:|\n")
    for scen in ["FPV", "deep_space"]:
        for nt in ["gaussian", "non_gaussian"]:
            for seed, val in sorted(b5.get(scen, {}).get(nt, [])):
                lines.append(f"| {scen} | {nt} | {seed} | {val:+.3f} |\n")
    lines.append("\n")

    out.write_text("".join(lines))


def main():
    b5 = collect_b5()
    b3 = collect_b3()
    out_md = RESULTS / "b5_aggregate.md"
    out_json = RESULTS / "b5_aggregate.json"
    build_md(b5, b3, out_md)
    payload = {
        "b5_per_scen_per_noise": {
            scen: {nt: [{"seed": s, "mean_snr": v} for s, v in lst]
                   for nt, lst in by_nt.items()}
            for scen, by_nt in b5.items()
        },
        "b3_baseline": b3,
    }
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"✅ {out_md}")
    print(f"✅ {out_json}")


if __name__ == "__main__":
    main()
