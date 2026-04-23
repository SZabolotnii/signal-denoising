#!/usr/bin/env python3
"""A2-H5: Noise-subspace DSGE diagnostic (Phase 1 — no NN).

Hypothesis: fitting DSGE on noise-only data produces a nonlinear noise model.
At inference on x̃ = s + n, K̂(x̃) approximates the noise component, and residual
r = x̃ − K̂(x̃) ≈ s.

Fit:  target = noise, input = noise   (noise-subspace DSGE).
Eval per SNR bin:
  (a) Var↓ noise = Var(n − K̂(n)) / Var(n)          — should be SMALL.
  (b) Var↓ mix   = Var((s+n) − K̂(s+n)) / Var(s+n)  — ≈ SNR_lin/(SNR_lin+1) if ideal.
  (c) Δcorr      = corr(residual, clean) − corr(mix, clean).
  (d) SNR gain   = 10 log10(Var(clean)/Var(residual − clean)) − input SNR (dB).

No NN training — all numpy. ~2–5 min runtime on CPU.

Usage:
  python experiments/a2_h5_noise_dsge_diag.py
  python experiments/a2_h5_noise_dsge_diag.py --bases robust polynomial --scenarios fpv
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.dsge_layer import DSGEFeatureExtractor  # noqa: E402

DATASETS = {
    "fpv":        ROOT / "data_generation/datasets/fpv_telemetry_polygauss_qpsk_bs1024_n100000_953c56e8",
    "deep_space": ROOT / "data_generation/datasets/deep_space_polygauss_qpsk_bs1024_n400000_0310b7e7",
}

DEFAULT_POWERS = {
    "fractional":    [0.5, 1.5, 2.0],
    "polynomial":    [2, 3, 4],
    "robust":        [1, 1, 1],  # robust_basis ignores values, uses len only
    "trigonometric": [1.0, 2.0, 3.0],
}

SNR_PATTERN = re.compile(r"test_(m|p)(\d+)dB_non_gaussian\.npy$")


def list_snr_bins(test_dir: Path) -> list[tuple[int, Path]]:
    """Return [(snr_db_int, non_gaussian_path), ...] sorted ascending."""
    bins = []
    for p in sorted(test_dir.glob("test_*dB_non_gaussian.npy")):
        m = SNR_PATTERN.search(p.name)
        if not m:
            continue
        sign, val = m.group(1), int(m.group(2))
        snr = -val if sign == "m" else val
        bins.append((snr, p))
    return sorted(bins, key=lambda t: t[0])


def reconstruct_batch(ext: DSGEFeatureExtractor, X: np.ndarray) -> np.ndarray:
    """Vectorized K̂ = k₀ + Σ kᵢ·φᵢ(X) for batch [N, T]."""
    phi = ext._basis_fn(X, ext.powers)  # [S, N, T]
    return ext.k0 + np.einsum("s,snt->nt", ext.K, phi)


def batch_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_c = a - a.mean(axis=1, keepdims=True)
    b_c = b - b.mean(axis=1, keepdims=True)
    num = (a_c * b_c).mean(axis=1)
    den = a_c.std(axis=1) * b_c.std(axis=1) + 1e-12
    return num / den


def eval_snr_bin(ext: DSGEFeatureExtractor, clean: np.ndarray, mix: np.ndarray,
                 noise: np.ndarray) -> dict:
    """Evaluate diagnostics on a single SNR bin."""
    n_hat = reconstruct_batch(ext, noise)
    noise_res = noise - n_hat
    vr_noise = np.var(noise_res, axis=1) / (np.var(noise, axis=1) + 1e-12)

    m_hat = reconstruct_batch(ext, mix)
    mix_res = mix - m_hat
    vr_mix = np.var(mix_res, axis=1) / (np.var(mix, axis=1) + 1e-12)

    corr_residual = batch_corr(mix_res, clean)
    corr_baseline = batch_corr(mix, clean)

    var_clean = np.var(clean, axis=1)
    var_noise_in = np.var(noise, axis=1)
    snr_in = 10 * np.log10(var_clean / (var_noise_in + 1e-12))

    residual_noise = mix_res - clean  # what noise survived after denoising
    var_res_noise = np.var(residual_noise, axis=1)
    snr_out = 10 * np.log10(var_clean / (var_res_noise + 1e-12))
    snr_gain = snr_out - snr_in

    return {
        "var_reduction_noise_mean": float(vr_noise.mean()),
        "var_reduction_mix_mean":   float(vr_mix.mean()),
        "corr_residual_clean":      float(corr_residual.mean()),
        "corr_baseline":            float(corr_baseline.mean()),
        "corr_improvement":         float((corr_residual - corr_baseline).mean()),
        "snr_in_db":                float(snr_in.mean()),
        "snr_out_db":               float(snr_out.mean()),
        "snr_gain_db_mean":         float(snr_gain.mean()),
        "snr_gain_db_std":          float(snr_gain.std()),
        "n_signals":                int(len(clean)),
    }


def run_one(scenario: str, basis: str, order: int, n_fit: int, n_eval_per_snr: int) -> dict:
    ds_path = DATASETS[scenario]
    train_dir = ds_path / "train"
    test_dir = ds_path / "test"

    noise_train_full = np.load(train_dir / "non_gaussian_noise_only.npy", mmap_mode="r")
    n_fit = min(n_fit, len(noise_train_full))
    noise_fit = np.asarray(noise_train_full[:n_fit], dtype=np.float32)

    powers = DEFAULT_POWERS[basis][:order]
    ext = DSGEFeatureExtractor(
        basis_type=basis,
        powers=powers,
        tikhonov_lambda=0.01,
        stft_params={"nperseg": 128, "noverlap": 96, "fs": 8192},
    )
    # Fit target=noise, input=noise (noise-subspace DSGE)
    ext.fit(noise_fit, noise_fit)

    # Per-SNR evaluation
    bins = list_snr_bins(test_dir)
    per_snr = {}
    for snr_db, mix_path in bins:
        mix_all = np.load(mix_path, mmap_mode="r")
        suffix = mix_path.name.replace("_non_gaussian.npy", "")
        clean_path = test_dir / f"{suffix}_clean.npy"
        noise_path = test_dir / f"{suffix}_non_gaussian_noise_only.npy"
        if not (clean_path.exists() and noise_path.exists()):
            continue

        n = min(n_eval_per_snr, len(mix_all))
        clean = np.asarray(np.load(clean_path, mmap_mode="r")[:n], dtype=np.float32)
        mix = np.asarray(mix_all[:n], dtype=np.float32)
        noise = np.asarray(np.load(noise_path, mmap_mode="r")[:n], dtype=np.float32)

        per_snr[str(snr_db)] = eval_snr_bin(ext, clean, mix, noise)

    # Aggregate stats
    gains = [v["snr_gain_db_mean"] for v in per_snr.values()]
    corrs = [v["corr_improvement"] for v in per_snr.values()]

    return {
        "scenario": scenario,
        "basis": basis,
        "order": order,
        "powers": powers,
        "n_fit": n_fit,
        "n_eval_per_snr": n_eval_per_snr,
        "K": ext.K.tolist(),
        "k0": float(ext.k0),
        "gen_element_norm": float(ext.gen_element_norm),
        "snr_gain_across_bins_mean": float(np.mean(gains)) if gains else float("nan"),
        "snr_gain_across_bins_min":  float(np.min(gains)) if gains else float("nan"),
        "snr_gain_across_bins_max":  float(np.max(gains)) if gains else float("nan"),
        "corr_improvement_mean":     float(np.mean(corrs)) if corrs else float("nan"),
        "per_snr": per_snr,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--scenarios", nargs="+", default=list(DATASETS.keys()))
    p.add_argument("--bases", nargs="+", default=list(DEFAULT_POWERS.keys()))
    p.add_argument("--order", type=int, default=3)
    p.add_argument("--n-fit", type=int, default=5000)
    p.add_argument("--n-eval-per-snr", type=int, default=1000)
    args = p.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"a2_h5_noise_dsge_{ts}.json"
    md_path = out_dir / f"a2_h5_noise_dsge_{ts}.md"

    results = []
    total = len(args.scenarios) * len(args.bases)
    i = 0
    for scenario in args.scenarios:
        for basis in args.bases:
            i += 1
            print(f"\n[{i}/{total}] scenario={scenario} basis={basis} order={args.order}")
            try:
                rec = run_one(scenario, basis, args.order, args.n_fit, args.n_eval_per_snr)
            except Exception as e:
                rec = {"scenario": scenario, "basis": basis, "error": str(e)}
                print(f"  ERROR: {e}")
                import traceback; traceback.print_exc()
                results.append(rec)
                with open(json_path, "w") as f:
                    json.dump(results, f, indent=2, default=str)
                continue
            print(f"  K={[round(k, 4) for k in rec['K']]} k0={rec['k0']:+.3e} "
                  f"gen_norm={rec['gen_element_norm']:.4f}")
            print(f"  SNR gain μ across bins: {rec['snr_gain_across_bins_mean']:+.3f} dB "
                  f"(min {rec['snr_gain_across_bins_min']:+.3f}, max {rec['snr_gain_across_bins_max']:+.3f})")
            print(f"  Δcorr mean: {rec['corr_improvement_mean']:+.4f}")
            # Print per-SNR table for this config
            print(f"  per-SNR gains:")
            for snr, v in sorted(rec['per_snr'].items(), key=lambda kv: int(kv[0])):
                print(f"    {int(snr):+3d} dB: var↓noise={v['var_reduction_noise_mean']:.3f} "
                      f"Δcorr={v['corr_improvement']:+.4f} "
                      f"SNR gain={v['snr_gain_db_mean']:+.3f} dB")
            results.append(rec)
            with open(json_path, "w") as f:
                json.dump(results, f, indent=2, default=str)

    # Markdown
    with open(md_path, "w") as f:
        f.write(f"# A2-H5 Noise-subspace DSGE Diagnostic — {ts}\n\n")
        f.write("**Setup:** DSGE fit with target=noise, input=noise (noise-subspace). ")
        f.write(f"n_fit={args.n_fit} samples, n_eval={args.n_eval_per_snr} per SNR bin, order=S={args.order}. ")
        f.write("No NN training.\n\n")
        f.write("## Aggregate table\n\n")
        f.write("| scenario | basis | μ SNR gain (dB) | min | max | Δcorr mean |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in results:
            if "error" in r:
                f.write(f"| {r['scenario']} | {r['basis']} | ERROR | | | |\n")
                continue
            f.write(f"| {r['scenario']} | {r['basis']} | "
                    f"{r['snr_gain_across_bins_mean']:+.3f} | "
                    f"{r['snr_gain_across_bins_min']:+.3f} | "
                    f"{r['snr_gain_across_bins_max']:+.3f} | "
                    f"{r['corr_improvement_mean']:+.4f} |\n")
        f.write("\n## Per-SNR breakdown\n\n")
        for r in results:
            if "error" in r:
                continue
            f.write(f"### {r['scenario']} / {r['basis']}  (K={[round(k,4) for k in r['K']]}, k0={r['k0']:+.3e})\n\n")
            f.write("| SNR in (dB) | Var↓ noise | Var↓ mix | corr(res,c) | corr(mix,c) | Δcorr | SNR out (dB) | SNR gain (dB) |\n")
            f.write("|---|---|---|---|---|---|---|---|\n")
            for snr, v in sorted(r['per_snr'].items(), key=lambda kv: int(kv[0])):
                f.write(f"| {int(snr):+3d} | "
                        f"{v['var_reduction_noise_mean']:.3f} | "
                        f"{v['var_reduction_mix_mean']:.3f} | "
                        f"{v['corr_residual_clean']:+.3f} | "
                        f"{v['corr_baseline']:+.3f} | "
                        f"{v['corr_improvement']:+.4f} | "
                        f"{v['snr_out_db']:+.2f} | "
                        f"{v['snr_gain_db_mean']:+.3f} |\n")
            f.write("\n")

    print(f"\n✅ Done. Results → {json_path}\n   Summary → {md_path}")


if __name__ == "__main__":
    main()
