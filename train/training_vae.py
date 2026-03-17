import argparse
import json
import sys
import uuid
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
from scipy.signal import stft, istft
from torch.utils.data import DataLoader, TensorDataset, random_split

try:
    import wandb
    WANDB_OK = True
except Exception:
    WANDB_OK = False

from models.autoencoder_vae import SpectrogramVAE
from metrics import MeanSquaredError, MeanAbsoluteError, RootMeanSquaredError, SignalToNoiseRatio
from train.snr_curve import evaluate_per_snr, print_snr_table, plot_snr_curve, log_snr_curve_wandb, save_training_curves

MODEL_NAME = 'SpectrogramVAE'


class VAETrainer:
    def __init__(self, dataset_path: Path, noise_type="non_gaussian",
                 batch_size=32, epochs=50, learning_rate=1e-4,
                 signal_len=256, fs=8192, nperseg=32, random_state=42,
                 wandb_project=""):
        self.dataset_path = Path(dataset_path)
        self.noise_type = noise_type
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = learning_rate
        self.signal_len = signal_len
        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = nperseg // 2
        self.pad = nperseg // 2
        self.random_state = random_state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.run_id = uuid.uuid4().hex[:8]
        self.dataset_uid = self.dataset_path.name.split('_')[-1]

        if WANDB_OK and wandb_project:
            run_name = f"{MODEL_NAME}_{noise_type}_{self.dataset_uid}_{self.run_id}"
            wandb.init(project=wandb_project, name=run_name, reinit=True, config={
                "model": MODEL_NAME, "noise_type": noise_type,
                "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
                "random_state": random_state, "dataset": self.dataset_path.name,
            })
            print(f"[W&B] Logging enabled → project='{wandb_project}', run='{run_name}'")
        else:
            reason = "wandb not installed" if not WANDB_OK else "no --wandb-project given"
            print(f"[W&B] Logging disabled ({reason})")

        self.train_loader, self.val_loader, self.test_loader, \
            self.freq_bins, self.time_frames = self.load_data()
        self.model = SpectrogramVAE(
            freq_bins=self.freq_bins, time_frames=self.time_frames
        ).to(self.device)

    # ── data ──────────────────────────────────────────────────────────────────

    def _signal_to_mag_tensor(self, signal_batch: torch.Tensor) -> torch.Tensor:
        padded = nn.functional.pad(signal_batch, (self.pad, self.pad), mode="reflect")
        mags = []
        for s in padded.squeeze(1).cpu().numpy():
            _, _, Zxx = stft(s, fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap)
            mags.append(np.abs(Zxx))
        return torch.tensor(np.stack(mags), dtype=torch.float32).unsqueeze(1).to(self.device)

    def load_data(self):
        noisy = np.load(self.dataset_path / "train" / f"{self.noise_type}_signals.npy")
        clean = np.load(self.dataset_path / "train" / "clean_signals.npy")
        assert noisy.shape[1] == self.signal_len

        X = torch.tensor(noisy[:, :self.signal_len], dtype=torch.float32).unsqueeze(1)
        y = torch.tensor(clean[:, :self.signal_len], dtype=torch.float32).unsqueeze(1)

        dataset = TensorDataset(X, y)
        total = len(dataset)
        val_len  = int(0.15 * total)
        test_len = int(0.15 * total)
        train_len = total - val_len - test_len
        train_set, val_set, test_set = random_split(
            dataset, [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(self.random_state),
        )

        example = self._signal_to_mag_tensor(X[:1])
        _, _, freq_bins, time_frames = example.shape

        return (
            DataLoader(train_set, batch_size=self.batch_size, shuffle=True),
            DataLoader(val_set,   batch_size=self.batch_size),
            DataLoader(test_set,  batch_size=self.batch_size),
            freq_bins, time_frames,
        )

    # ── inference ─────────────────────────────────────────────────────────────

    def denoise_numpy(self, noisy: np.ndarray) -> np.ndarray:
        """[N, T] → [N, T]"""
        t = torch.tensor(noisy, dtype=torch.float32).unsqueeze(1)
        return self._denoise_batch(t.to(self.device)).squeeze(1).cpu().numpy()

    def _denoise_batch(self, signal_batch: torch.Tensor) -> torch.Tensor:
        """[N, 1, T] → [N, 1, T]"""
        self.model.eval()
        padded = nn.functional.pad(signal_batch, (self.pad, self.pad), mode='reflect')
        mags, phases = [], []
        for s in padded.squeeze(1).cpu().numpy():
            _, _, Zxx = stft(s, fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap)
            mags.append(np.abs(Zxx))
            phases.append(np.angle(Zxx))
        spec_t = torch.tensor(np.stack(mags), dtype=torch.float32).unsqueeze(1).to(self.device)
        with torch.no_grad():
            out_mag, _, _ = self.model(spec_t)
            out_mag = out_mag.squeeze(1).cpu().numpy()
        rec_signals = []
        for mag, phase in zip(out_mag, phases):
            _, rec = istft(mag * np.exp(1j * phase), fs=self.fs,
                           nperseg=self.nperseg, noverlap=self.noverlap)
            rec = rec[self.pad: self.pad + self.signal_len]
            if len(rec) < self.signal_len:
                rec = np.pad(rec, (0, self.signal_len - len(rec)))
            rec_signals.append(rec.astype(np.float32))
        return torch.tensor(np.stack(rec_signals), dtype=torch.float32).unsqueeze(1)

    # ── validation ────────────────────────────────────────────────────────────

    def _compute_val_snr(self) -> float:
        all_true, all_pred = [], []
        for X_batch, y_batch in self.val_loader:
            pred = self.denoise_numpy(X_batch.squeeze(1).numpy())
            all_pred.append(pred)
            all_true.append(y_batch.squeeze(1).numpy())
        return float(SignalToNoiseRatio.calculate(
            np.concatenate(all_true), np.concatenate(all_pred)
        ))

    def _compute_val_loss(self, loss_fn) -> float:
        self.model.eval()
        total = 0.0
        with torch.no_grad():
            for X_batch, _ in self.val_loader:
                spec = self._signal_to_mag_tensor(X_batch.to(self.device))
                recon, mu, logvar = self.model(spec)
                kl   = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = (loss_fn(recon, spec) + kl) / spec.size(0)
                total += loss.item()
        return total / len(self.val_loader)

    # ── training loop ─────────────────────────────────────────────────────────

    def train(self) -> dict:
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # HuberLoss with reduction="sum" to keep the same scale as original MSELoss(sum)
        loss_fn = nn.HuberLoss(delta=1.0, reduction="sum")
        best_val_snr = float("-inf")
        best_sd = None
        train_history, val_snr_history = [], []

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0.0

            for X_batch, _ in self.train_loader:
                spec = self._signal_to_mag_tensor(X_batch.to(self.device))
                recon, mu, logvar = self.model(spec)
                kl   = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = (loss_fn(recon, spec) + kl) / spec.size(0)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                total_loss += loss.item()

            val_loss = self._compute_val_loss(loss_fn)
            val_snr  = self._compute_val_snr()

            if WANDB_OK and hasattr(wandb, 'run') and wandb.run:
                wandb.log({
                    "train/huber_kl_loss": total_loss / len(self.train_loader),
                    "val/huber_kl_loss":   val_loss,
                    "val/snr_db":          val_snr,
                }, step=epoch)

            print(f"Epoch {epoch:02d}/{self.epochs} | "
                  f"train={total_loss / len(self.train_loader):.5f} | "
                  f"val_loss={val_loss:.5f} | val_SNR={val_snr:.2f} dB")

            train_history.append(total_loss / len(self.train_loader))
            val_snr_history.append(val_snr)

            if val_snr > best_val_snr:
                best_val_snr = val_snr
                best_sd = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        weights_dir = self.dataset_path / "weights"
        weights_dir.mkdir(exist_ok=True)
        save_name = f"{MODEL_NAME}_{self.noise_type}_{self.dataset_uid}_best.pth"
        save_training_curves(
            train_history, val_snr_history,
            weights_dir / "figures" / "training" / f"training_curves_{MODEL_NAME}_{self.noise_type}.png",
            MODEL_NAME, self.noise_type,
        )
        save_path = weights_dir / save_name
        torch.save(best_sd, save_path)
        print(f"✅ Best model saved → {save_path}")
        self.model.load_state_dict(best_sd)

        test_metrics = self._evaluate_test()

        per_snr = {}
        test_dir = self.dataset_path / "test"
        if test_dir.exists():
            per_snr = evaluate_per_snr(self.denoise_numpy, test_dir, self.noise_type)
            print_snr_table(per_snr, MODEL_NAME)
            plot_snr_curve(
                per_snr, MODEL_NAME,
                save_path=weights_dir / f"snr_curve_{MODEL_NAME}_{self.noise_type}.png",
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
        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                pred = self.denoise_numpy(X_batch.squeeze(1).numpy())
                all_pred.append(pred)
                all_true.append(y_batch.squeeze(1).numpy())
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
        print("\n📊 Final Test Metrics:")
        for name, val in metrics.items():
            print(f"  {name}: {val:.2f} dB" if name == "SNR" else f"  {name}: {val:.6f}")
        return metrics


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train VAE for signal denoising")
    p.add_argument("--dataset",       required=True)
    p.add_argument("--noise-type",    default="non_gaussian", choices=["gaussian", "non_gaussian"])
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--batch-size",    type=int,   default=32)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--nperseg",       type=int,   default=32)
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

    VAETrainer(
        dataset_path=dataset_path,
        noise_type=args.noise_type,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        signal_len=cfg["block_size"],
        fs=cfg["sample_rate"],
        nperseg=args.nperseg,
        random_state=args.seed,
        wandb_project=args.wandb_project,
    ).train()
