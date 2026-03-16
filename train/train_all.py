#!/usr/bin/env python3
"""Unified training script for all signal denoising models.

Usage:
    python train/train_all.py \
        --dataset data_generation/datasets/deep_space_polygauss_nonstationary_bpsk_bs256_n50000_39075e4f \
        --noise-type non_gaussian \
        --models all \
        --epochs 50

Models: unet, resnet, vae, transformer, wavelet  (or "all")
Noise types: gaussian, non_gaussian
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import stft, istft
from torch.utils.data import DataLoader, TensorDataset, random_split

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

try:
    import wandb
    WANDB_OK = True
except Exception:
    WANDB_OK = False

from models.autoencoder_unet import UnetAutoencoder
from models.autoencoder_resnet import ResNetAutoencoder
from models.autoencoder_vae import SpectrogramVAE
from models.time_series_trasformer import TimeSeriesTransformer
from metrics import MeanSquaredError, MeanAbsoluteError, RootMeanSquaredError, SignalToNoiseRatio
from train.wavelet_grid_search import grid_search_wavelet

ALL_MODELS = ["unet", "resnet", "vae", "transformer", "wavelet", "hybrid"]


# ─── metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MSE":  MeanSquaredError.calculate(y_true, y_pred),
        "MAE":  MeanAbsoluteError.calculate(y_true, y_pred),
        "RMSE": RootMeanSquaredError.calculate(y_true, y_pred),
        "SNR":  SignalToNoiseRatio.calculate(y_true, y_pred),
    }


def print_metrics(metrics: dict):
    for k, v in metrics.items():
        print(f"    {k}: {v:.2f} dB" if k == "SNR" else f"    {k}: {v:.6f}")


# ─── data ─────────────────────────────────────────────────────────────────────

def load_dataset(dataset_dir: Path, noise_type: str):
    clean = np.load(dataset_dir / "train" / "clean_signals.npy")
    noisy = np.load(dataset_dir / "train" / f"{noise_type}_signals.npy")
    return clean, noisy


def make_loaders(clean: np.ndarray, noisy: np.ndarray,
                 batch_size: int, val_frac: float = 0.15, seed: int = 42,
                 unsqueeze_last: bool = False):
    if unsqueeze_last:
        X = torch.tensor(noisy, dtype=torch.float32).unsqueeze(-1)  # (N, T, 1)
        y = torch.tensor(clean, dtype=torch.float32).unsqueeze(-1)
    else:
        X = torch.tensor(noisy, dtype=torch.float32)   # (N, T)
        y = torch.tensor(clean, dtype=torch.float32)

    ds = TensorDataset(X, y)
    val_len = int(val_frac * len(ds))
    train_len = len(ds) - val_len
    train_set, val_set = random_split(
        ds, [train_len, val_len],
        generator=torch.Generator().manual_seed(seed)
    )
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(val_set,   batch_size=batch_size),
    )


# ─── STFT utilities ───────────────────────────────────────────────────────────

def stft_mag_phase(x: np.ndarray, fs: int, nperseg: int, noverlap: int, pad: int):
    x_pad = np.pad(x, pad, mode="reflect")
    _, _, Zxx = stft(x_pad, fs=fs, nperseg=nperseg, noverlap=noverlap,
                     window="hann", boundary=None, padded=False)
    return np.abs(Zxx).astype(np.float32), np.angle(Zxx).astype(np.float32)


def istft_mag_phase(mag, phase, fs: int, nperseg: int, noverlap: int, pad: int, target_len: int):
    _, rec = istft(mag * np.exp(1j * phase), fs=fs, nperseg=nperseg, noverlap=noverlap,
                   window="hann", input_onesided=True, boundary=None)
    rec = rec[pad: pad + target_len]
    if len(rec) < target_len:
        rec = np.pad(rec, (0, target_len - len(rec)))
    return rec.astype(np.float32)


def batch_to_specs(batch: np.ndarray, fs: int, nperseg: int, noverlap: int, pad: int):
    mags, phases = [], []
    for x in batch:
        m, p = stft_mag_phase(x, fs, nperseg, noverlap, pad)
        mags.append(m)
        phases.append(p)
    return np.stack(mags), np.stack(phases)


def specs_to_signals(mags: np.ndarray, phases: np.ndarray,
                     fs: int, nperseg: int, noverlap: int, pad: int, target_len: int):
    return np.stack([
        istft_mag_phase(m, p, fs, nperseg, noverlap, pad, target_len)
        for m, p in zip(mags, phases)
    ])


# ─── spectral loss (used by UNet) ─────────────────────────────────────────────

def _stft_mag_torch(x: torch.Tensor, nperseg: int, noverlap: int) -> torch.Tensor:
    hop = nperseg - noverlap
    win = torch.hann_window(nperseg, periodic=True, device=x.device, dtype=x.dtype)
    return torch.abs(torch.stft(x, n_fft=nperseg, hop_length=hop, win_length=nperseg,
                                window=win, center=True, return_complex=True))


def multi_res_stft_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    total = 0.0
    for n, ov in [(32, 16), (16, 8), (64, 32)]:
        S_hat = _stft_mag_torch(x_hat, n, ov)
        S     = _stft_mag_torch(x,     n, ov)
        l1 = torch.mean(torch.abs(torch.log1p(S_hat) - torch.log1p(S)))
        sc = (torch.linalg.norm(S_hat - S, ord='fro', dim=(1, 2)) /
              (torch.linalg.norm(S, ord='fro', dim=(1, 2)) + 1e-12)).mean()
        total = total + l1 + 0.5 * sc
    return total / 3


# ─── W&B helpers ──────────────────────────────────────────────────────────────

def _wandb_init(name: str, args):
    if not (WANDB_OK and args.wandb_project):
        return
    import uuid
    wandb.init(project=args.wandb_project,
               name=f"{name}_{args.noise_type}_{uuid.uuid4().hex[:6]}",
               config=vars(args), reinit=True)


def _wandb_log(metrics: dict, step: int, args):
    if WANDB_OK and args.wandb_project:
        wandb.log(metrics, step=step)


def _wandb_finish(args):
    if WANDB_OK and args.wandb_project:
        wandb.finish()


# ─── UNet ─────────────────────────────────────────────────────────────────────

def train_unet(clean, noisy, cfg, args, weights_dir, device):
    print("\n=== UNet ===")
    signal_len = cfg["block_size"]
    fs         = cfg["sample_rate"]
    nperseg    = args.nperseg
    noverlap   = nperseg // 2          # 50% Hann COLA
    pad        = nperseg // 2

    sample_mag, _ = stft_mag_phase(clean[0], fs, nperseg, noverlap, pad)
    input_shape = sample_mag.shape
    print(f"  spectrogram shape: {input_shape}")

    train_loader, val_loader = make_loaders(clean, noisy, args.batch_size, seed=args.seed)
    model     = UnetAutoencoder(input_shape).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    _wandb_init("UNet", args)
    best_val, best_sd = float("inf"), None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for noisy_b, clean_b in train_loader:
            nm, phases = batch_to_specs(noisy_b.numpy(), fs, nperseg, noverlap, pad)
            cm, _      = batch_to_specs(clean_b.numpy(), fs, nperseg, noverlap, pad)

            nm_t = torch.tensor(nm, device=device).unsqueeze(1)   # (B,1,F,T')
            cm_t = torch.tensor(cm, device=device).unsqueeze(1)

            mask    = model(nm_t)
            out_mag = mask * nm_t
            base_loss = torch.mean(torch.abs(torch.log1p(out_mag) - torch.log1p(cm_t)))

            rec = specs_to_signals(out_mag.squeeze(1).detach().cpu().numpy(),
                                   phases, fs, nperseg, noverlap, pad, signal_len)
            mr_loss = multi_res_stft_loss(torch.tensor(rec, device=device), clean_b.to(device))
            loss = base_loss + 0.5 * mr_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        val_loss = _val_unet(model, val_loader, fs, nperseg, noverlap, pad, signal_len, device)
        _log_epoch("UNet", epoch, args.epochs, epoch_loss / len(train_loader), val_loss)
        _wandb_log({"train_loss": epoch_loss / len(train_loader), "val_loss": val_loss}, epoch, args)

        if val_loss < best_val:
            best_val = val_loss
            best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    _save(best_sd, weights_dir / f"UNet_{args.noise_type}_best.pth")
    _wandb_finish(args)


def _val_unet(model, loader, fs, nperseg, noverlap, pad, signal_len, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for noisy_b, clean_b in loader:
            nm, phases = batch_to_specs(noisy_b.numpy(), fs, nperseg, noverlap, pad)
            cm, _      = batch_to_specs(clean_b.numpy(), fs, nperseg, noverlap, pad)
            nm_t = torch.tensor(nm, device=device).unsqueeze(1)
            cm_t = torch.tensor(cm, device=device).unsqueeze(1)
            mask    = model(nm_t)
            out_mag = mask * nm_t
            base_loss = torch.mean(torch.abs(torch.log1p(out_mag) - torch.log1p(cm_t)))
            rec = specs_to_signals(out_mag.squeeze(1).cpu().numpy(),
                                   phases, fs, nperseg, noverlap, pad, signal_len)
            mr_loss = multi_res_stft_loss(torch.tensor(rec, device=device), clean_b.to(device))
            total += (base_loss + 0.5 * mr_loss).item()
    return total / len(loader)


# ─── ResNet ───────────────────────────────────────────────────────────────────

def train_resnet(clean, noisy, cfg, args, weights_dir, device):
    print("\n=== ResNet ===")
    fs       = cfg["sample_rate"]
    nperseg  = args.nperseg
    noverlap = int(nperseg * 0.75)
    pad      = nperseg // 2

    sample_mag, _ = stft_mag_phase(clean[0], fs, nperseg, noverlap, pad)
    input_shape = sample_mag.shape
    print(f"  spectrogram shape: {input_shape}")

    train_loader, val_loader = make_loaders(clean, noisy, args.batch_size, seed=args.seed)
    model     = ResNetAutoencoder(input_shape).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn   = nn.MSELoss()

    _wandb_init("ResNet", args)
    best_val, best_sd = float("inf"), None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for noisy_b, clean_b in train_loader:
            nm, _ = batch_to_specs(noisy_b.numpy(), fs, nperseg, noverlap, pad)
            cm, _ = batch_to_specs(clean_b.numpy(), fs, nperseg, noverlap, pad)
            nm_t = torch.tensor(nm, device=device).unsqueeze(1)
            cm_t = torch.tensor(cm, device=device).unsqueeze(1)
            loss = loss_fn(model(nm_t), cm_t)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_loss += loss.item()

        val_loss = _val_spec(model, val_loader, fs, nperseg, noverlap, pad, loss_fn, device)
        _log_epoch("ResNet", epoch, args.epochs, epoch_loss / len(train_loader), val_loss)
        _wandb_log({"train_loss": epoch_loss / len(train_loader), "val_loss": val_loss}, epoch, args)

        if val_loss < best_val:
            best_val = val_loss
            best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    _save(best_sd, weights_dir / f"ResNet_{args.noise_type}_best.pth")
    _wandb_finish(args)


def _val_spec(model, loader, fs, nperseg, noverlap, pad, loss_fn, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for noisy_b, clean_b in loader:
            nm, _ = batch_to_specs(noisy_b.numpy(), fs, nperseg, noverlap, pad)
            cm, _ = batch_to_specs(clean_b.numpy(), fs, nperseg, noverlap, pad)
            nm_t = torch.tensor(nm, device=device).unsqueeze(1)
            cm_t = torch.tensor(cm, device=device).unsqueeze(1)
            total += loss_fn(model(nm_t), cm_t).item()
    return total / len(loader)


# ─── VAE ──────────────────────────────────────────────────────────────────────

def train_vae(clean, noisy, cfg, args, weights_dir, device):
    print("\n=== VAE ===")
    fs       = cfg["sample_rate"]
    nperseg  = args.nperseg
    noverlap = nperseg // 2
    pad      = nperseg // 2

    sample_mag, _ = stft_mag_phase(clean[0], fs, nperseg, noverlap, pad)
    freq_bins, time_frames = sample_mag.shape
    print(f"  spectrogram shape: ({freq_bins}, {time_frames})")

    train_loader, val_loader = make_loaders(clean, noisy, args.batch_size, seed=args.seed)
    model     = SpectrogramVAE(freq_bins=freq_bins, time_frames=time_frames).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn   = nn.MSELoss(reduction="sum")

    _wandb_init("VAE", args)
    best_val, best_sd = float("inf"), None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for noisy_b, clean_b in train_loader:
            nm, _ = batch_to_specs(noisy_b.numpy(), fs, nperseg, noverlap, pad)
            cm, _ = batch_to_specs(clean_b.numpy(), fs, nperseg, noverlap, pad)
            nm_t = torch.tensor(nm, device=device).unsqueeze(1)
            cm_t = torch.tensor(cm, device=device).unsqueeze(1)
            recon, mu, logvar = model(nm_t)
            kl   = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = (loss_fn(recon, cm_t) + kl) / nm_t.size(0)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_loss += loss.item()

        val_loss = _val_vae(model, val_loader, fs, nperseg, noverlap, pad, loss_fn, device)
        _log_epoch("VAE", epoch, args.epochs, epoch_loss / len(train_loader), val_loss)
        _wandb_log({"train_loss": epoch_loss / len(train_loader), "val_loss": val_loss}, epoch, args)

        if val_loss < best_val:
            best_val = val_loss
            best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    _save(best_sd, weights_dir / f"VAE_{args.noise_type}_best.pth")
    _wandb_finish(args)


def _val_vae(model, loader, fs, nperseg, noverlap, pad, loss_fn, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for noisy_b, clean_b in loader:
            nm, _ = batch_to_specs(noisy_b.numpy(), fs, nperseg, noverlap, pad)
            cm, _ = batch_to_specs(clean_b.numpy(), fs, nperseg, noverlap, pad)
            nm_t = torch.tensor(nm, device=device).unsqueeze(1)
            cm_t = torch.tensor(cm, device=device).unsqueeze(1)
            recon, mu, logvar = model(nm_t)
            kl   = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            total += ((loss_fn(recon, cm_t) + kl) / nm_t.size(0)).item()
    return total / len(loader)


# ─── Transformer ──────────────────────────────────────────────────────────────

def train_transformer(clean, noisy, cfg, args, weights_dir, device):
    print("\n=== Transformer ===")
    train_loader, val_loader = make_loaders(
        clean, noisy, args.batch_size, seed=args.seed, unsqueeze_last=True
    )
    model     = TimeSeriesTransformer(input_dim=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn   = nn.MSELoss()

    _wandb_init("Transformer", args)
    best_val, best_sd = float("inf"), None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for noisy_b, clean_b in train_loader:
            noisy_b, clean_b = noisy_b.to(device), clean_b.to(device)
            loss = loss_fn(model(noisy_b), clean_b)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_loss += loss.item()

        val_loss = _val_transformer(model, val_loader, loss_fn, device)
        _log_epoch("Transformer", epoch, args.epochs, epoch_loss / len(train_loader), val_loss)
        _wandb_log({"train_loss": epoch_loss / len(train_loader), "val_loss": val_loss}, epoch, args)

        if val_loss < best_val:
            best_val = val_loss
            best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    _save(best_sd, weights_dir / f"Transformer_{args.noise_type}_best.pth")
    _wandb_finish(args)


def _val_transformer(model, loader, loss_fn, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for noisy_b, clean_b in loader:
            noisy_b, clean_b = noisy_b.to(device), clean_b.to(device)
            total += loss_fn(model(noisy_b), clean_b).item()
    return total / len(loader)


# ─── Wavelet ──────────────────────────────────────────────────────────────────

def train_wavelet(clean, noisy, cfg, args, weights_dir):
    print("\n=== Wavelet (grid search) ===")
    best_params, val_mse, test_mse = grid_search_wavelet(noisy, clean, random_state=args.seed)
    print(f"  Best params: {best_params}")
    print(f"  Val MSE: {val_mse:.6f}, Test MSE: {test_mse:.6f}")
    save_path = weights_dir / f"Wavelet_{args.noise_type}_best_params.json"
    with open(save_path, "w") as f:
        json.dump({"best_params": best_params, "val_mse": val_mse, "test_mse": test_mse}, f, indent=2)
    print(f"  Saved: {save_path}")


# ─── Hybrid DSGE+UNet ─────────────────────────────────────────────────────────

def train_hybrid(dataset_dir: Path, cfg: dict, args):
    print("\n=== HybridDSGE_UNet ===")
    from train.training_hybrid import HybridUnetTrainer
    signal_len = cfg["block_size"]
    fs         = cfg["sample_rate"]
    noverlap   = args.nperseg // 2

    trainer = HybridUnetTrainer(
        dataset_path=dataset_dir,
        noise_type=args.noise_type,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        signal_len=signal_len,
        fs=fs,
        nperseg=args.nperseg,
        noverlap=noverlap,
        random_state=args.seed,
        wandb_project=args.wandb_project,
    )
    trainer.train()


# ─── shared helpers ───────────────────────────────────────────────────────────

def _log_epoch(name: str, epoch: int, total: int, train_loss: float, val_loss: float):
    if epoch == 1 or epoch % 10 == 0 or epoch == total:
        print(f"  [{epoch:03d}/{total}] train={train_loss:.5f}  val={val_loss:.5f}")


def _save(state_dict, path: Path):
    torch.save(state_dict, path)
    print(f"  Saved: {path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train all denoising models on a dataset")
    p.add_argument("--dataset", required=True,
                   help="Path to dataset folder (absolute or relative to project root)")
    p.add_argument("--noise-type", default="non_gaussian",
                   choices=["gaussian", "non_gaussian"])
    p.add_argument("--models", default="all",
                   help=f"Comma-separated or 'all'. Options: {', '.join(ALL_MODELS)}")
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch-size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--nperseg",    type=int,   default=32,
                   help="STFT window size for spectral models (default 32 for 256-sample signals)")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--wandb-project", default="",
                   help="W&B project name (empty = disable)")
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
          f"scenario={cfg['scenario']}")
    print(f"Training: noise_type={args.noise_type}, epochs={args.epochs}, "
          f"batch={args.batch_size}, lr={args.lr}")

    weights_dir = dataset_dir / "weights"
    weights_dir.mkdir(exist_ok=True)

    clean, noisy = load_dataset(dataset_dir, args.noise_type)
    print(f"Data    : clean {clean.shape}, noisy {noisy.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}\n")

    if not args.wandb_project:
        reason = "wandb not installed" if not WANDB_OK else "no --wandb-project given"
        global WANDB_OK
        WANDB_OK = False
        print(f"[W&B] Logging disabled ({reason})")
    else:
        print(f"[W&B] Logging enabled → project='{args.wandb_project}'")

    models_to_train = (
        ALL_MODELS if args.models == "all"
        else [m.strip() for m in args.models.split(",")]
    )

    for m in models_to_train:
        if   m == "unet":        train_unet(clean, noisy, cfg, args, weights_dir, device)
        elif m == "resnet":      train_resnet(clean, noisy, cfg, args, weights_dir, device)
        elif m == "vae":         train_vae(clean, noisy, cfg, args, weights_dir, device)
        elif m == "transformer": train_transformer(clean, noisy, cfg, args, weights_dir, device)
        elif m == "wavelet":     train_wavelet(clean, noisy, cfg, args, weights_dir)
        elif m == "hybrid":      train_hybrid(dataset_dir, cfg, args)
        else:                    print(f"Unknown model: {m}, skipping")

    print("\n✅ Done. Weights saved to:", weights_dir)


if __name__ == "__main__":
    main()
