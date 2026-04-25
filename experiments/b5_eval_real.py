#!/usr/bin/env python3
"""Phase B5 evaluation — fine-tuned models vs actual RadioML test.

Loads model_best.pth from each B5 fine-tune run_dir (flat layout, marked by
b4_finetune_meta.json) and evaluates on the real RadioML test set
(non_gaussian split). Compares to B3 zero-shot baseline.
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

from train.compare_report import _load_unet  # noqa: E402
from train.snr_curve import evaluate_per_snr  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--b5-dataset", required=True, type=Path,
                    help="B5 dataset dir (contains runs/ with fine-tuned models).")
    ap.add_argument("--real-test-dir", required=True, type=Path,
                    help="Real-data dataset dir (contains test/ subdir).")
    ap.add_argument("--test-noise", default="non_gaussian")
    ap.add_argument("--nperseg", type=int, default=128)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "experiments/results")
    args = ap.parse_args()

    # Load real-test dataset config (gives block_size + sample_rate for STFT)
    real_cfg = json.loads((args.real_test_dir / "dataset_config.json").read_text())
    test_dir = args.real_test_dir / "test"

    runs_root = args.b5_dataset / "runs"
    run_dirs = [d for d in sorted(runs_root.iterdir())
                if d.is_dir() and (d / "model_best.pth").exists()]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.out.mkdir(parents=True, exist_ok=True)
    json_path = args.out / f"b5_eval_{args.b5_dataset.name}_{ts}.json"
    md_path = json_path.with_suffix(".md")

    results = {}
    for rd in run_dirs:
        # Run name encodes seed/noise via path? We need to read b4_finetune_meta
        # if present.
        meta_path = rd.parent / "b4_finetune_meta.json"
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                meta = {}
        rd_label = rd.name
        print(f"  • {rd_label}", flush=True)
        denoise = _load_unet(rd, real_cfg, nperseg=args.nperseg)
        per_snr = evaluate_per_snr(denoise, test_dir, args.test_noise,
                                    batch_size=256)
        import numpy as np
        def _jsonify(o):
            if isinstance(o, dict):
                return {k: _jsonify(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [_jsonify(x) for x in o]
            if isinstance(o, (np.floating, np.integer)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return o

        results[rd_label] = {
            "meta": meta,
            "per_snr": _jsonify(per_snr),
        }

    # Write JSON
    with open(json_path, "w") as f:
        json.dump({
            "b5_dataset": str(args.b5_dataset),
            "real_test_dir": str(args.real_test_dir),
            "test_noise": args.test_noise,
            "timestamp": ts,
            "runs": results,
        }, f, indent=2)

    # Build MD
    snr_labels = sorted({lbl for r in results.values() for lbl in r["per_snr"]},
                        key=lambda s: float(s.replace("m", "-").replace("p", "").replace("dB", "")))
    lines = [
        f"# B5 fine-tuned eval on real {args.real_test_dir.name} — {ts}",
        "",
        f"- Source B5 dataset: `{args.b5_dataset}`",
        f"- Real test dir: `{args.real_test_dir}`",
        f"- Test noise: {args.test_noise}",
        "",
        "## Per-run SNR_out (dB) on actual RadioML test",
        "",
        "| Run | " + " | ".join(snr_labels) + " | mean |",
        "|---|" + "|".join(["---:"] * (len(snr_labels) + 1)) + "|",
    ]
    for rd_label, r in results.items():
        cells = [rd_label]
        snrs = []
        for lbl in snr_labels:
            v = r["per_snr"].get(lbl, {}).get("SNR")
            if v is None:
                cells.append("—")
            else:
                cells.append(f"{v:.2f}")
                snrs.append(v)
        cells.append(f"{sum(snrs)/len(snrs):.2f}" if snrs else "—")
        lines.append("| " + " | ".join(cells) + " |")
    md_path.write_text("\n".join(lines) + "\n")

    print(f"[b5-eval] wrote {md_path}")
    print(f"[b5-eval] wrote {json_path}")


if __name__ == "__main__":
    main()
