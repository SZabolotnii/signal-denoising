#!/usr/bin/env python3
"""Phase B4 — fine-tune pretrained synthetic-trained models on RadioML train split.

Loads a B1 or B2 pretrained checkpoint, continues training on the adapted
RadioML subset with a reduced learning rate (1e-4) for a small number of
epochs, evaluates on the same per-SNR test bins as B3.

Purpose: verify that sim-to-real gap (B3 showed −0.02..−1.59 dB zero-shot
across all models) can be closed via partial adaptation on real data.

Usage:
    python experiments/b4_finetune.py \\
        --b1-run <path to pretrained run_dir> \\
        --real-dataset <path to radioml adapted subset> \\
        --model unet \\
        --noise-type non_gaussian \\
        --seed 42 \\
        --epochs 10 \\
        --lr 1e-4 \\
        --partial 0.25
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.training_uae import UnetAutoencoderTrainer  # noqa: E402
from train.training_resnet import ResNetAutoencoderTrainer  # noqa: E402

TRAINERS = {
    "unet":   UnetAutoencoderTrainer,
    "resnet": ResNetAutoencoderTrainer,
}


def _find_model_checkpoint(b1_run: Path, model_name: str, noise_type: str) -> Path:
    """Locate model_best.pth in the given run_dir matching model + noise_type."""
    subdirs = [d for d in b1_run.iterdir() if d.is_dir() and d.name.endswith(f"_{noise_type}")]
    # Pick the one whose prefix matches
    class_prefix = {
        "unet":   "UnetAutoencoder",
        "resnet": "ResNetAutoencoder",
    }[model_name]
    matches = [d for d in subdirs if d.name.startswith(class_prefix)]
    if not matches:
        raise FileNotFoundError(f"No {class_prefix}_{noise_type} subdir in {b1_run}")
    ckpt = matches[0] / "model_best.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"No model_best.pth in {matches[0]}")
    return ckpt


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--b1-run", required=True, type=Path,
                    help="Pretrained source run_dir (B1 FPV or B2 deep_space).")
    ap.add_argument("--real-dataset", required=True, type=Path,
                    help="Adapted RadioML subset dir.")
    ap.add_argument("--model", required=True, choices=list(TRAINERS))
    ap.add_argument("--noise-type", required=True, choices=["gaussian", "non_gaussian"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--partial", type=float, default=0.25)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--nperseg", type=int, default=128)
    args = ap.parse_args()

    ckpt = _find_model_checkpoint(args.b1_run, args.model, args.noise_type)
    print(f"[b4] pretrained checkpoint: {ckpt}")
    print(f"[b4] real dataset: {args.real_dataset}")

    # Infer block_size from dataset config (RadioML uses 1024)
    cfg_path = args.real_dataset / "dataset_config.json"
    block_size = 1024
    if cfg_path.exists():
        block_size = int(json.loads(cfg_path.read_text()).get("block_size", 1024))

    TrainerCls = TRAINERS[args.model]
    trainer = TrainerCls(
        dataset_path=args.real_dataset,
        noise_type=args.noise_type,
        epochs=args.epochs,
        learning_rate=args.lr,
        random_state=args.seed,
        data_fraction=args.partial,
        device=args.device,
        nperseg=args.nperseg,
        signal_len=block_size,
    )
    # Load pretrained weights before training
    state = torch.load(ckpt, map_location=trainer.device)
    trainer.model.load_state_dict(state)
    print(f"[b4] loaded pretrained weights into {args.model}")

    result = trainer.train()

    # Sidecar marker so post-hoc analysis can distinguish fine-tune runs.
    # Trainer flat layout: weights_path = <ds>/runs/run_xxx_Model_noise/model_best.pth
    # so weights_path.parent IS the run_dir (no extra subdir).
    run_dir = Path(result["weights_path"]).parent \
        if "weights_path" in result else None
    if run_dir and run_dir.exists():
        meta = {
            "phase": "B4_finetune",
            "source_checkpoint": str(ckpt),
            "source_run": str(args.b1_run),
            "real_dataset": str(args.real_dataset),
            "seed": args.seed,
            "epochs": args.epochs,
            "lr": args.lr,
            "partial": args.partial,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        marker = run_dir / "b4_finetune_meta.json"
        marker.write_text(json.dumps(meta, indent=2))
        print(f"[b4] wrote marker: {marker}")


if __name__ == "__main__":
    main()
