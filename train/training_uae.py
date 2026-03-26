import argparse
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

try:
    import wandb
    WANDB_OK = True
except Exception:
    WANDB_OK = False

from tqdm import tqdm

from models.autoencoder_unet import UnetAutoencoder
from metrics import MeanSquaredError, MeanAbsoluteError, RootMeanSquaredError, SignalToNoiseRatio
from train.losses import select_loss
from train.snr_curve import evaluate_per_snr, print_snr_table, plot_snr_curve, log_snr_curve_wandb, save_training_curves

MODEL_NAME = 'UnetAutoencoder'


# ── Compatibility helpers for compare_report.py ─────────────────────────────
# compare_report.py historically imported these from train.training_uae.
# Keep them here to avoid breaking reporting scripts.
def stft_mag_phase(x: np.ndarray, fs: int, nperseg: int, noverlap: int, pad: int = 0):
    """Return (magnitude, phase) for a single 1D signal using scipy STFT."""
    from scipy.signal import stft as _stft

    if pad > 0:
        x = np.pad(x, (pad, pad), mode="reflect")
    _, _, Zxx = _stft(x, fs=fs, nperseg=nperseg, noverlap=noverlap, boundary=None)
    return np.abs(Zxx).astype(np.float32), np.angle(Zxx).astype(np.float32)


def istft_from_mag_phase(
    mag: np.ndarray,
    phase: np.ndarray,
    fs: int,
    nperseg: int,
    noverlap: int,
    pad: int,
    signal_len: int,
):
    """Inverse STFT for one sample; crops reflect padding and ensures signal_len."""
    from scipy.signal import istft as _istft

    Z = mag * np.exp(1j * phase)
    _, x_rec = _istft(Z, fs=fs, nperseg=nperseg, noverlap=noverlap, input_onesided=True, boundary=None)
    if pad > 0 and len(x_rec) >= 2 * pad:
        x_rec = x_rec[pad: -pad]
    x_rec = x_rec[:signal_len] if len(x_rec) >= signal_len else np.pad(x_rec, (0, signal_len - len(x_rec)))
    return x_rec.astype(np.float32)


def _stft_mag_torch(x: torch.Tensor, nperseg: int, noverlap: int) -> torch.Tensor:
    hop = nperseg - noverlap
    win = torch.hann_window(nperseg, periodic=True, device=x.device, dtype=x.dtype)
    return torch.abs(torch.stft(x, n_fft=nperseg, hop_length=hop, win_length=nperseg,
                                window=win, center=True, return_complex=True))


def multi_res_stft_loss(x_hat: torch.Tensor, x: torch.Tensor,
                        configs=((32, 16), (64, 32), (16, 8))) -> torch.Tensor:
    total = 0.0
    for n, ov in configs:
        S_hat = _stft_mag_torch(x_hat, n, ov)
        S     = _stft_mag_torch(x,     n, ov)
        l1 = torch.mean(torch.abs(torch.log1p(S_hat) - torch.log1p(S)))
        sc = (torch.linalg.norm(S_hat - S, ord='fro', dim=(1, 2)) /
              (torch.linalg.norm(S, ord='fro', dim=(1, 2)) + 1e-12)).mean()
        total = total + l1 + 0.5 * sc
    return total / len(configs)


class UnetAutoencoderTrainer:
    def __init__(self, dataset_path: Path, noise_type="non_gaussian",
                 batch_size=512, epochs=30, learning_rate=3e-4,
                 signal_len=256, fs=8192, nperseg=128, noverlap=96, random_state=42,
                 wandb_project="", device=None, data_fraction=1.0, output_dir=None):
        self.dataset_path = Path(dataset_path)
        self.noise_type = noise_type
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = learning_rate
        self.signal_len = signal_len
        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = nperseg * 3 // 4   # 75% overlap (Hann COLA satisfied)
        self.pad = nperseg // 2
        self.random_state = random_state
        self.data_fraction = data_fraction
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.run_id = uuid.uuid4().hex[:8]
        self.run_date = datetime.now().strftime("%Y%m%d")
        self.dataset_uid = self.dataset_path.name.split('_')[-1]

        if WANDB_OK and wandb_project:
            run_name = f"{MODEL_NAME}_{noise_type}_{self.dataset_uid}_{self.run_id}"
            wandb.init(project=wandb_project, name=run_name, reinit=True, config={
                "model": MODEL_NAME, "noise_type": noise_type,
                "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
                "random_state": random_state, "fs": fs, "nperseg": nperseg,
                "dataset": self.dataset_path.name,
            })
            print(f"[W&B] Logging enabled → project='{wandb_project}', run='{run_name}'")
        else:
            reason = "wandb not installed" if not WANDB_OK else "no --wandb-project given"
            print(f"[W&B] Logging disabled ({reason})")

        self.train_loader, self.val_loader, self.test_loader, self.input_shape = self._load_data()
        self.model = UnetAutoencoder(self.input_shape).to(self.device)

    # ── STFT helpers (GPU-batched, no scipy) ──────────────────────────────────

    def _stft_batch(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T] → complex [B, F, T'] — Hann window, center=True."""
        win = torch.hann_window(self.nperseg, device=x.device)
        return torch.stft(x, n_fft=self.nperseg,
                          hop_length=self.nperseg - self.noverlap,
                          win_length=self.nperseg, window=win,
                          center=True, pad_mode='reflect',
                          onesided=True, return_complex=True)

    def _istft_batch(self, spec: torch.Tensor) -> torch.Tensor:
        """complex [B, F, T'] → [B, signal_len]"""
        win = torch.hann_window(self.nperseg, device=spec.device)
        return torch.istft(spec, n_fft=self.nperseg,
                           hop_length=self.nperseg - self.noverlap,
                           win_length=self.nperseg, window=win,
                           center=True, onesided=True, length=self.signal_len)

    # ── data ──────────────────────────────────────────────────────────────────

    def _load_data(self):
        noisy = np.load(self.dataset_path / "train" / f"{self.noise_type}_signals.npy")
        clean = np.load(self.dataset_path / "train" / "clean_signals.npy")
        if self.data_fraction < 1.0:
            n = max(1, int(len(noisy) * self.data_fraction))
            noisy, clean = noisy[:n], clean[:n]
        assert noisy.shape[1] == self.signal_len, \
            f"Signal length mismatch: expected {self.signal_len}, got {noisy.shape[1]}"

        dummy = torch.zeros(1, self.signal_len)
        spec0 = self._stft_batch(dummy)
        input_shape = (int(spec0.shape[1]), int(spec0.shape[2]))

        dataset = TensorDataset(
            torch.tensor(noisy, dtype=torch.float32),
            torch.tensor(clean, dtype=torch.float32),
        )
        total = len(dataset)
        val_len  = int(0.25 * total)
        test_len = int(0.25 * total)
        train_len = total - val_len - test_len
        train_set, val_set, test_set = random_split(
            dataset, [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(self.random_state),
        )
        pin = torch.cuda.is_available()
        return (
            DataLoader(train_set, batch_size=self.batch_size, shuffle=True,
                       num_workers=4, pin_memory=pin, persistent_workers=True),
            DataLoader(val_set,   batch_size=self.batch_size,
                       num_workers=4, pin_memory=pin, persistent_workers=True),
            DataLoader(test_set,  batch_size=self.batch_size,
                       num_workers=4, pin_memory=pin, persistent_workers=True),
            input_shape,
        )

    # ── inference ─────────────────────────────────────────────────────────────

    def denoise_numpy(self, noisy: np.ndarray) -> np.ndarray:
        """[N, T] → [N, T], batched STFT → mask → ISTFT."""
        self.model.eval()
        x = torch.tensor(noisy, dtype=torch.float32, device=self.device)
        spec = self._stft_batch(x)
        mag = spec.abs().unsqueeze(1)
        with torch.no_grad():
            out_mag = self.model(mag) * mag
        out_spec = out_mag.squeeze(1) * torch.exp(1j * torch.angle(spec))
        return self._istft_batch(out_spec).cpu().numpy()

    # ── validation ────────────────────────────────────────────────────────────

    def _compute_val_snr(self) -> float:
        all_true, all_pred = [], []
        for noisy, clean in tqdm(self.val_loader, desc="  val SNR", leave=False, unit="batch"):
            all_pred.append(self.denoise_numpy(noisy.numpy()))
            all_true.append(clean.numpy())
        return float(SignalToNoiseRatio.calculate(
            np.concatenate(all_true), np.concatenate(all_pred)
        ))

    def _compute_val_loss(self, loss_fn) -> float:
        self.model.eval()
        total = 0.0
        with torch.no_grad():
            for noisy, clean in tqdm(self.val_loader, desc="  val loss", leave=False, unit="batch"):
                nm_t = self._stft_batch(noisy.to(self.device)).abs().unsqueeze(1)
                cm_t = self._stft_batch(clean.to(self.device)).abs().unsqueeze(1)
                total += loss_fn(self.model(nm_t) * nm_t, cm_t).item()
        return total / len(self.val_loader)

    # ── training loop ─────────────────────────────────────────────────────────

    def train(self) -> dict:
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = select_loss(self.noise_type)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=3, factor=0.5, threshold=0.01
        )
        best_val_loss = float("inf")
        best_val_snr  = float("-inf")
        best_sd = None
        train_history, val_snr_history = [], []
        no_improve = 0
        early_stop_patience = 7

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:02d}/{self.epochs}", leave=False, unit="batch")
            for noisy, clean in pbar:
                nm_t = self._stft_batch(noisy.to(self.device)).abs().unsqueeze(1)
                cm_t = self._stft_batch(clean.to(self.device)).abs().unsqueeze(1)
                loss = loss_fn(self.model(nm_t) * nm_t, cm_t)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.5f}")

            vram_str = (f" | vram={torch.cuda.max_memory_allocated() / 1024**3:.2f}GB"
                        if torch.cuda.is_available() else "")

            val_loss = self._compute_val_loss(loss_fn)
            val_snr  = self._compute_val_snr()

            scheduler.step(val_loss)
            lr_now = optimizer.param_groups[0]['lr']

            if WANDB_OK and hasattr(wandb, 'run') and wandb.run:
                wandb.log({
                    "train/mse_loss": epoch_loss / len(self.train_loader),
                    "val/mse_loss":   val_loss,
                    "val/snr_db":     val_snr,
                    "train/lr":       lr_now,
                }, step=epoch)

            print(f"Epoch {epoch:02d}/{self.epochs} | "
                  f"train={epoch_loss / len(self.train_loader):.5f} | "
                  f"val_loss={val_loss:.5f} | val_SNR={val_snr:.2f} dB | "
                  f"lr={lr_now:.2e}{vram_str}")

            train_history.append(epoch_loss / len(self.train_loader))
            val_snr_history.append(val_snr)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_snr  = val_snr
                best_sd = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= early_stop_patience:
                    print(f"  Early stopping: no improvement for {early_stop_patience} epochs")
                    break

        # ── save ──────────────────────────────────────────────────────────────
        if self.output_dir is not None:
            run_dir = self.output_dir / f"{MODEL_NAME}_{self.noise_type}"
        else:
            run_dir = self.dataset_path / "runs" / f"run_{self.run_date}_{self.run_id}_{MODEL_NAME}_{self.noise_type}"
        run_dir.mkdir(parents=True, exist_ok=True)
        save_path = run_dir / "model_best.pth"
        save_training_curves(
            train_history, val_snr_history,
            run_dir / "figures" / "training_curves.png",
            MODEL_NAME, self.noise_type,
        )
        torch.save(best_sd, save_path)
        print(f"✅ Best model saved → {save_path}")
        self.model.load_state_dict(best_sd)

        # ── test metrics ──────────────────────────────────────────────────────
        test_metrics = self._evaluate_test()

        # ── per-SNR curves ────────────────────────────────────────────────────
        per_snr = {}
        test_dir = self.dataset_path / "test"
        if test_dir.exists():
            per_snr = evaluate_per_snr(self.denoise_numpy, test_dir, self.noise_type)
            print_snr_table(per_snr, MODEL_NAME)
            plot_snr_curve(
                per_snr, MODEL_NAME,
                save_path=run_dir / "figures" / "snr_curve.png",
            )
            log_snr_curve_wandb(per_snr, MODEL_NAME)

        if WANDB_OK and hasattr(wandb, 'run') and wandb.run:
            wandb.finish()

        return {
            'model': MODEL_NAME, 'noise_type': self.noise_type,
            'dataset_uid': self.dataset_uid, 'run_id': self.run_id,
            'val_snr': best_val_snr, 'test_metrics': test_metrics,
            'per_snr_results': per_snr, 'weights_path': str(save_path),
        }

    def _evaluate_test(self) -> dict:
        all_true, all_pred = [], []
        for noisy, clean in self.test_loader:
            all_pred.append(self.denoise_numpy(noisy.numpy()))
            all_true.append(clean.numpy())
        y_true = np.concatenate(all_true)
        y_pred = np.concatenate(all_pred)
        metrics = {
            "MSE":  MeanSquaredError.calculate(y_true, y_pred),
            "MAE":  MeanAbsoluteError.calculate(y_true, y_pred),
            "RMSE": RootMeanSquaredError.calculate(y_true, y_pred),
            "SNR":  SignalToNoiseRatio.calculate(y_true, y_pred),
        }
        if WANDB_OK and hasattr(wandb, 'run') and wandb.run:
            wandb.log({f"test/{k.lower()}": v for k, v in metrics.items()})
        print("\n📊 Final Test Metrics (time domain):")
        for k, v in metrics.items():
            print(f"  {k}: {v:.2f} dB" if k == "SNR" else f"  {k}: {v:.6f}")
        return metrics


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train UNet autoencoder for signal denoising")
    p.add_argument("--dataset",       required=True)
    p.add_argument("--noise-type",    default="non_gaussian", choices=["gaussian", "non_gaussian"])
    p.add_argument("--epochs",        type=int,   default=30)
    p.add_argument("--batch-size",    type=int,   default=512)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--nperseg",       type=int,   default=128)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--wandb-project", default="")
    args = p.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path

    with open(dataset_path / "dataset_config.json") as f:
        cfg = json.load(f)

    print(f"Dataset: {dataset_path.name}")
    print(f"Config:  block_size={cfg['block_size']}, sample_rate={cfg['sample_rate']}, "
          f"noise_type={args.noise_type}")

    UnetAutoencoderTrainer(
        dataset_path=dataset_path,
        noise_type=args.noise_type,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        signal_len=cfg["block_size"],
        fs=cfg["sample_rate"],
        nperseg=args.nperseg,
        noverlap=args.nperseg // 2,
        random_state=args.seed,
        wandb_project=args.wandb_project,
    ).train()
