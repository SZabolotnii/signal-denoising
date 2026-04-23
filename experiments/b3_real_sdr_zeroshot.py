#!/usr/bin/env python3
"""Phase B3 — Real SDR zero-shot evaluation.

Loads models pretrained on synthetic FPV (B1 run_dir) and evaluates them on a
real-SDR dataset adapted to the pipeline format (`data_generation/load_radioml2018.py`
or `load_dronedetect.py` output directory).

Pipeline reuse: compare_report._load_* loaders give us denoise callables for
UNet, ResNet, Hybrid, Wavelet, VAE, Transformer bound to the B1 checkpoints.
snr_curve.evaluate_per_snr walks test_<snr>_<noise>.npy / test_<snr>_clean.npy
files — the adapter outputs match this schema, so zero-shot eval is a direct
call.

Output:
    <out-dir>/b3_zeroshot_<src-run-id>_<real-ds-id>_<ts>.{md,json}

Typical use:
    python experiments/b3_real_sdr_zeroshot.py \
        --b1-run data_generation/datasets/fpv_telemetry_polygauss_qpsk_bs1024_n100000_953c56e8/runs/run_20260421_267176a3 \
        --real-dataset data_generation/datasets/radioml2018_bpsk_qpsk_<uuid> \
        --test-noise non_gaussian

Notes:
    * B1 runs are per-seed × per-noise. For cross-seed aggregation, call this
      script once per run_dir and stitch in a post-processing pass (like
      `analysis/aggregate_b1.py`).
    * Real-dataset test dir is expected at `<real-dataset>/test/`.
    * Only noise_type that exists in the real dataset will be evaluated; if the
      adapter produced only `non_gaussian` files (typical for real SDR), pass
      `--test-noise non_gaussian`.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the compare_report model loaders (fixed in §11.8 — authoritative inference).
from train.compare_report import discover_runs  # noqa: E402
from train.snr_curve import evaluate_per_snr  # noqa: E402


def _load_dataset_config(real_ds: Path) -> dict:
    cfg_path = real_ds / "dataset_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing dataset_config.json in {real_ds}")
    with open(cfg_path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description="B3 zero-shot real-SDR evaluation.")
    ap.add_argument("--b1-run", required=True, type=Path,
                    help="B1 run_dir (synthetic-trained models).")
    ap.add_argument("--real-dataset", required=True, type=Path,
                    help="Adapter output dir (RadioML/DroneDetect).")
    ap.add_argument("--test-noise", default="non_gaussian",
                    choices=["gaussian", "non_gaussian"],
                    help="Which noise subset of the real test dir to evaluate.")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--nperseg", type=int, default=128)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("experiments/results"))
    args = ap.parse_args()

    b1_run: Path = args.b1_run
    real_ds: Path = args.real_dataset
    test_dir = real_ds / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Missing test/ under {real_ds}")

    # Config from the *real* dataset — ensures nperseg etc. match the adapter.
    real_cfg = _load_dataset_config(real_ds)
    # Sanity check block size
    bs = real_cfg.get("block_size") or real_cfg.get("signal_len")
    if bs is not None and bs != 1024:
        print(f"[warn] real dataset block_size={bs} ≠ 1024 — models trained on "
              f"1024-length blocks; results may be miscalibrated.", file=sys.stderr)

    # B1 run cfg drives the model arch (DSGE order/basis/variant in the run name).
    b1_cfg_path = b1_run.parent.parent / "dataset_config.json"
    b1_cfg = json.load(open(b1_cfg_path))

    # Discover models present in the B1 run — same mechanism compare_report uses.
    entries = discover_runs(b1_run, b1_cfg, nperseg=args.nperseg)
    if not entries:
        raise RuntimeError(f"No model sub-dirs discovered under {b1_run}")

    print(f"[b3] loaded {len(entries)} models from {b1_run.name}: "
          f"{list(entries.keys())}")
    print(f"[b3] evaluating on {real_ds.name}/test/{args.test_noise}")

    results = {}
    for name, info in entries.items():
        print(f"  • {name} …", flush=True)
        per_snr = evaluate_per_snr(info["denoise_fn"], test_dir, args.test_noise,
                                   batch_size=args.batch_size)
        if not per_snr:
            print(f"    [skip] no per-SNR test files matched in {test_dir} "
                  f"for noise={args.test_noise}")
        results[name] = per_snr

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / f"b3_zeroshot_{b1_run.name}_{real_ds.name[:24]}_{ts}.json"
    md_path = json_path.with_suffix(".md")

    payload = {
        "b1_run": str(b1_run),
        "real_dataset": str(real_ds),
        "test_noise": args.test_noise,
        "nperseg": args.nperseg,
        "timestamp": ts,
        "results": results,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    lines = [
        f"# B3 zero-shot real-SDR eval — {ts}",
        "",
        f"- Source models: `{b1_run}`",
        f"- Real dataset:  `{real_ds}` (noise={args.test_noise})",
        "",
        "## Per-model per-SNR SNR (dB)",
        "",
    ]
    # Collect all SNR labels across models (they should match — real test dir
    # produces a fixed grid per adapter).
    all_labels = sorted({lbl for m in results.values() for lbl in m.keys()},
                        key=lambda s: float(s.replace("m", "-").replace("p", "").replace("dB", "")))
    header = "| Model | " + " | ".join(all_labels) + " | mean |"
    sep = "|---|" + "|".join(["---:"] * (len(all_labels) + 1)) + "|"
    lines += [header, sep]
    for name, per_snr in results.items():
        row = [name]
        snrs = []
        for lbl in all_labels:
            if lbl in per_snr and per_snr[lbl].get("SNR") is not None:
                v = per_snr[lbl]["SNR"]
                row.append(f"{v:.2f}")
                snrs.append(v)
            else:
                row.append("—")
        if snrs:
            row.append(f"{sum(snrs) / len(snrs):.2f}")
        else:
            row.append("—")
        lines.append("| " + " | ".join(row) + " |")

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[b3] wrote {json_path}")
    print(f"[b3] wrote {md_path}")


if __name__ == "__main__":
    main()
