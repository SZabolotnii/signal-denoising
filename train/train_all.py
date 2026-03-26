#!/usr/bin/env python3
"""Unified training script — trains all (or selected) denoising models on a dataset.

Each model gets its own W&B run (when --wandb-project is set).
A Markdown + JSON report is generated in <dataset>/weights/ at the end.

Usage:
    python train/train_all.py \
        --dataset data_generation/datasets/deep_space_..._39075e4f \
        --noise-type non_gaussian \
        --models all \
        --epochs 50 \
        --wandb-project sd-science
"""

import argparse
import gc
import json
import sys
import uuid as _uuid_mod
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

try:
    import wandb
    WANDB_OK = True
except Exception:
    WANDB_OK = False

# Transformer first — largest VRAM consumer (O(T²) attention), trains safely
# before GPU memory gets fragmented by smaller models.
ALL_MODELS = ["transformer", "unet", "vae", "resnet", "hybrid", "wavelet"]

# Per-model batch sizes measured on ~8 GiB GPU with signal_len=1024, nperseg=128.
# Values calibrated from actual vram= logs; target ≤ 6.5 GiB peak (fwd+bwd).
# --batch-size overrides all of these when explicitly provided.
MODEL_BATCH_SIZES = {
    "transformer": 128,   # O(T²) attention at T=1024; measured 4.99 GB
    "unet":        1024,  # torch.stft on GPU; measured 6.47 GB — ~1 GB margin on 8 GB card
    "vae":         8192,  # torch.stft on GPU; measured 3.91 GB
    "resnet":      2048,  # torch.stft on GPU; measured 4.82 GB
    "hybrid":      4096,  # torch.stft 4-ch on GPU; measured 5.16 GB
    "wavelet":     512,   # CPU-only
}

# Per-model default learning rates.
# Transformer needs a 10× higher LR — at 1e-4 it stalls at epoch 1 (loss ≈ E[x²],
# meaning it outputs near-zero and never escapes that plateau).
# All other models converge well but reach the fine-tuning plateau only after many
# epochs at 1e-4; 3e-4 shortens the initial descent without causing instability.
# --lr overrides all of these when explicitly provided.
MODEL_LEARNING_RATES = {
    "transformer": 1e-3,  # stuck at 1e-4; 10× increase needed to escape init plateau
    "unet":        1e-3,
    "vae":         1e-3,
    "resnet":      1e-3,
    "hybrid":      3e-4,
    "wavelet":     None,  # not applicable (grid search)
}


# ── model runners ─────────────────────────────────────────────────────────────

def run_unet(dataset_dir: Path, cfg: dict, args) -> dict:
    from train.training_uae import UnetAutoencoderTrainer
    print("\n" + "=" * 60)
    print("=== UNet (Mask + STFT, MSELoss) ===")
    print("=" * 60)
    bs = args.batch_size if args.batch_size is not None else MODEL_BATCH_SIZES["unet"]
    lr = args.lr if args.lr is not None else MODEL_LEARNING_RATES["unet"]
    return UnetAutoencoderTrainer(
        dataset_path=dataset_dir,
        noise_type=args.noise_type,
        batch_size=bs,
        epochs=args.epochs,
        learning_rate=lr,
        signal_len=cfg["block_size"],
        fs=cfg["sample_rate"],
        nperseg=args.nperseg,
        noverlap=args.nperseg * 3 // 4,
        random_state=args.seed,
        wandb_project=args.wandb_project,
        data_fraction=args.partial_train,
        output_dir=args.shared_run_dir,
    ).train()


def run_resnet(dataset_dir: Path, cfg: dict, args) -> dict:
    from train.training_resnet import ResNetAutoencoderTrainer
    print("\n" + "=" * 60)
    print("=== ResNet (STFT autoencoder, MSELoss) ===")
    print("=" * 60)
    bs = args.batch_size if args.batch_size is not None else MODEL_BATCH_SIZES["resnet"]
    lr = args.lr if args.lr is not None else MODEL_LEARNING_RATES["resnet"]
    return ResNetAutoencoderTrainer(
        dataset_path=dataset_dir,
        noise_type=args.noise_type,
        batch_size=bs,
        epochs=args.epochs,
        learning_rate=lr,
        signal_len=cfg["block_size"],
        fs=cfg["sample_rate"],
        nperseg=args.nperseg,
        random_state=args.seed,
        wandb_project=args.wandb_project,
        data_fraction=args.partial_train,
        output_dir=args.shared_run_dir,
    ).train()


def run_vae(dataset_dir: Path, cfg: dict, args) -> dict:
    from train.training_vae import VAETrainer
    print("\n" + "=" * 60)
    print("=== VAE (SpectrogramVAE, MSELoss + KL) ===")
    print("=" * 60)
    bs = args.batch_size if args.batch_size is not None else MODEL_BATCH_SIZES["vae"]
    lr = args.lr if args.lr is not None else MODEL_LEARNING_RATES["vae"]
    return VAETrainer(
        dataset_path=dataset_dir,
        noise_type=args.noise_type,
        batch_size=bs,
        epochs=args.epochs,
        learning_rate=lr,
        signal_len=cfg["block_size"],
        fs=cfg["sample_rate"],
        nperseg=args.nperseg,
        random_state=args.seed,
        wandb_project=args.wandb_project,
        data_fraction=args.partial_train,
        output_dir=args.shared_run_dir,
    ).train()


def run_transformer(dataset_dir: Path, cfg: dict, args) -> dict:
    from train.training_transformer import TransformerTrainer
    print("\n" + "=" * 60)
    print("=== Transformer (time-domain, MSELoss) ===")
    print("=" * 60)
    bs = args.batch_size if args.batch_size is not None else MODEL_BATCH_SIZES["transformer"]
    lr = args.lr if args.lr is not None else MODEL_LEARNING_RATES["transformer"]
    return TransformerTrainer(
        dataset_path=dataset_dir,
        noise_type=args.noise_type,
        batch_size=bs,
        epochs=args.epochs,
        learning_rate=lr,
        random_state=args.seed,
        wandb_project=args.wandb_project,
        data_fraction=args.partial_train,
        output_dir=args.shared_run_dir,
    ).train()


def run_wavelet(dataset_dir: Path, cfg: dict, args) -> dict | None:
    from train.wavelet_grid_search import grid_search_wavelet
    import numpy as np
    print("\n" + "=" * 60)
    print("=== Wavelet (grid search) ===")
    print("=" * 60)
    noisy = np.load(dataset_dir / "train" / f"{args.noise_type}_signals.npy")
    clean = np.load(dataset_dir / "train" / "clean_signals.npy")
    if args.partial_train < 1.0:
        n = max(1, int(len(noisy) * args.partial_train))
        noisy, clean = noisy[:n], clean[:n]
    best_params, val_mse, test_mse = grid_search_wavelet(noisy, clean, random_state=args.seed)
    print(f"  Best params: {best_params}")
    print(f"  Val MSE: {val_mse:.6f}  Test MSE: {test_mse:.6f}")
    run_dir = args.shared_run_dir / f"Wavelet_{args.noise_type}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_path = run_dir / "best_params.json"
    with open(save_path, "w") as f:
        json.dump({"best_params": best_params, "val_mse": val_mse, "test_mse": test_mse}, f, indent=2)
    print(f"  Saved: {save_path}")
    return {
        'model': 'Wavelet', 'noise_type': args.noise_type,
        'val_snr': None, 'test_metrics': {'MSE': test_mse},
        'weights_path': str(save_path), 'per_snr_results': {},
    }


def run_hybrid(dataset_dir: Path, cfg: dict, args) -> dict:
    from train.training_hybrid import HybridUnetTrainer
    print("\n" + "=" * 60)
    print("=== HybridDSGE_UNet (robust basis S=3, U-Net mask, MSELoss) ===")
    print("=" * 60)
    bs = args.batch_size if args.batch_size is not None else MODEL_BATCH_SIZES["hybrid"]
    lr = args.lr if args.lr is not None else MODEL_LEARNING_RATES["hybrid"]
    return HybridUnetTrainer(
        dataset_path=dataset_dir,
        noise_type=args.noise_type,
        dsge_order=3,
        dsge_basis='robust',
        batch_size=bs,
        epochs=args.epochs,
        learning_rate=lr,
        signal_len=cfg["block_size"],
        fs=cfg["sample_rate"],
        nperseg=args.nperseg,
        noverlap=args.nperseg * 3 // 4,
        random_state=args.seed,
        wandb_project=args.wandb_project,
        data_fraction=args.partial_train,
        output_dir=args.shared_run_dir,
    ).train()


_RUNNERS = {
    "unet":        run_unet,
    "resnet":      run_resnet,
    "vae":         run_vae,
    "transformer": run_transformer,
    "wavelet":     run_wavelet,
    "hybrid":      run_hybrid,
}


# ── report ────────────────────────────────────────────────────────────────────

def generate_report(results: list, dataset_dir: Path, args, weights_dir: Path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset_dir.name,
        "noise_type": args.noise_type,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "models": [r for r in results if r is not None],
    }

    json_path = weights_dir / f"training_report_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    md_path = weights_dir / f"training_report_{timestamp}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Training Report\n\n")
        f.write(f"**Dataset:** `{dataset_dir.name}`  \n")
        f.write(f"**Noise type:** {args.noise_type}  \n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  \n")
        f.write(f"**Epochs:** {args.epochs} | **Batch:** {args.batch_size} | **LR:** {args.lr}  \n\n")
        f.write("## Results\n\n")
        f.write("| Model | Val SNR | Test SNR | Test MSE | Weights |\n")
        f.write("|-------|--------:|--------:|---------:|---------|\n")
        for r in results:
            if r is None:
                continue
            val_snr  = r.get('val_snr')
            test_snr = r.get('test_metrics', {}).get('SNR')
            test_mse = r.get('test_metrics', {}).get('MSE')
            val_str  = f"{val_snr:.2f} dB"  if val_snr  is not None else "—"
            snr_str  = f"{test_snr:.2f} dB" if test_snr is not None else "—"
            mse_str  = f"{test_mse:.6f}"    if test_mse is not None else "—"
            wpath = Path(r.get('weights_path', ''))
            try:
                wname = str(wpath.relative_to(weights_dir))
            except ValueError:
                wname = wpath.name
            f.write(f"| {r['model']} ({r.get('noise_type','')}) | {val_str} | {snr_str} | {mse_str} | `{wname}` |\n")

        # per-SNR table (first model that has it)
        for r in results:
            if r and r.get('per_snr_results'):
                f.write("\n## Per-SNR Performance\n\n")
                f.write("| Model | SNR_in | SNR_out | MSE |\n")
                f.write("|-------|-------:|--------:|----:|\n")
                for lbl, m in sorted(r['per_snr_results'].items(),
                                     key=lambda kv: kv[1]['snr_in_db']):
                    f.write(f"| {r['model']} | {m['snr_in_db']:.0f} dB | "
                            f"{m['SNR']:.2f} dB | {m['MSE']:.6f} |\n")

    print(f"\n📋 Report → {md_path}")
    return md_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train all denoising models on a dataset")
    p.add_argument("--dataset",       required=True,
                   help="Path to dataset folder (absolute or relative to project root)")
    p.add_argument("--noise-types",   default="all",
                   help="Comma-separated or 'all'. Options: gaussian, non_gaussian")
    p.add_argument("--models",        default="all",
                   help=f"Comma-separated or 'all'. Options: {', '.join(ALL_MODELS)}")
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--batch-size",    type=int,   default=None,
                   help="Override batch size for all models (default: per-model from MODEL_BATCH_SIZES)")
    p.add_argument("--lr",            type=float, default=None,
                   help="Learning rate override for all models (default: per-model from MODEL_LEARNING_RATES)")
    p.add_argument("--nperseg",       type=int,   default=128,
                   help="STFT window size for spectral models (default 128 for 1024-sample signals)")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--wandb-project", default="",
                   help="W&B project name (empty = disable)")
    p.add_argument("--partial-train", type=float, default=1.0, metavar="FRACTION",
                   help="Fraction of dataset to use (0 < f <= 1). Useful for quick debug runs.")
    return p.parse_args()


def main():
    args = parse_args()

    dataset_dir = Path(args.dataset)
    if not dataset_dir.is_absolute():
        dataset_dir = ROOT / dataset_dir
    if not dataset_dir.exists():
        print(f"ERROR: dataset not found: {dataset_dir}")
        sys.exit(1)

    with open(dataset_dir / "dataset_config.json") as f:
        cfg = json.load(f)

    print(f"Dataset : {dataset_dir.name}")
    print(f"Config  : block_size={cfg['block_size']}, sample_rate={cfg['sample_rate']}, "
          f"scenario={cfg.get('scenario', '?')}")
    noise_types = (
        ["gaussian", "non_gaussian"] if args.noise_types == "all"
        else [n.strip() for n in args.noise_types.split(",")]
    )
    if not (0.0 < args.partial_train <= 1.0):
        print(f"ERROR: --partial-train must be in (0, 1], got {args.partial_train}")
        sys.exit(1)
    lr_display = f"{args.lr}" if args.lr is not None else "per-model"
    print(f"Training: noise_types={noise_types}, epochs={args.epochs}, "
          f"batch={args.batch_size}, lr={lr_display}"
          + (f", partial={args.partial_train:.0%}" if args.partial_train < 1.0 else ""))

    run_date = datetime.now().strftime("%Y%m%d")
    run_uid  = _uuid_mod.uuid4().hex[:8]
    shared_run_dir = dataset_dir / "runs" / f"run_{run_date}_{run_uid}"
    shared_run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir : {shared_run_dir.relative_to(dataset_dir)}")
    args.shared_run_dir = shared_run_dir

    if not args.wandb_project:
        reason = "wandb not installed" if not WANDB_OK else "no --wandb-project given"
        print(f"[W&B] Logging disabled ({reason})")
    else:
        print(f"[W&B] Logging enabled → project='{args.wandb_project}' (one run per model)")

    models_to_train = (
        ALL_MODELS if args.models == "all"
        else [m.strip() for m in args.models.split(",")]
    )

    results = []
    for noise_type in noise_types:
        args.noise_type = noise_type
        print(f"\n{'#' * 60}")
        print(f"# Noise type: {noise_type}")
        print(f"{'#' * 60}")
        for m in models_to_train:
            runner = _RUNNERS.get(m)
            if runner is None:
                print(f"Unknown model '{m}', skipping")
                continue
            try:
                result = runner(dataset_dir, cfg, args)
                results.append(result)
            except Exception as exc:
                print(f"ERROR training {m} ({noise_type}): {exc}")
                results.append({'model': m, 'noise_type': noise_type, 'error': str(exc)})
                exc.__traceback__ = None  # release GPU tensor refs held in traceback frames
            finally:
                gc.collect()
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    generate_report(results, dataset_dir, args, shared_run_dir)
    print(f"\n✅ Done. Weights and report saved to: {shared_run_dir}")


if __name__ == "__main__":
    main()
