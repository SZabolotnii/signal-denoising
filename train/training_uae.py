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
from scipy.signal import stft, istft

try:
    import wandb
    WANDB_OK = True
except Exception:
    WANDB_OK = False

from tqdm import tqdm

from models.autoencoder_unet import UnetAutoencoder
from metrics import MeanSquaredError, MeanAbsoluteError, RootMeanSquaredError, SignalToNoiseRatio
from train.snr_curve import evaluate_per_snr, print_snr_table, plot_snr_curve, log_snr_curve_wandb, save_training_curves

MODEL_NAME = 'UnetAutoencoder'
WINDOW = 'hann'


def stft_mag_phase(x, fs, nperseg, noverlap, pad):
    x_pad = np.pad(x, pad, mode="reflect")
    _, _, Zxx = stft(x_pad, fs=fs, nperseg=nperseg, noverlap=noverlap,
                     window=WINDOW, boundary=None, padded=False)
    return np.abs(Zxx).astype(np.float32), np.angle(Zxx).astype(np.float32)


def istft_from_mag_phase(mag, phase, fs, nperseg, noverlap, pad, target_len):
    _, rec = istft(mag * np.exp(1j * phase), fs=fs, nperseg=nperseg, noverlap=noverlap,
                   window=WINDOW, input_onesided=True, boundary=None)
    rec = rec[pad: pad + target_len]
    if len(rec) < target_len:
        rec = np.pad(rec, (0, target_len - len(rec)))
    return rec.astype(np.float32)


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
                 batch_size=512, epochs=30, learning_rate=1e-4,
                 signal_len=256, fs=8192, nperseg=128, noverlap=96, random_state=42,
                 wandb_project="", device=None):
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

    # ── data ──────────────────────────────────────────────────────────────────

    def _load_data(self):
        noisy = np.load(self.dataset_path / "train" / f"{self.noise_type}_signals.npy")
        clean = np.load(self.dataset_path / "train" / "clean_signals.npy")
        assert noisy.shape[1] == self.signal_len, \
            f"Signal length mismatch: expected {self.signal_len}, got {noisy.shape[1]}"

        mag0, _ = stft_mag_phase(clean[0], self.fs, self.nperseg, self.noverlap, self.pad)
        input_shape = mag0.shape

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
        return (
            DataLoader(train_set, batch_size=self.batch_size, shuffle=True),
            DataLoader(val_set,   batch_size=self.batch_size),
            DataLoader(test_set,  batch_size=self.batch_size),
            input_shape,
        )

    # ── inference ─────────────────────────────────────────────────────────────

    def denoise_numpy(self, noisy: np.ndarray) -> np.ndarray:
        """[N, T] → [N, T], batched STFT → mask → ISTFT."""
        self.model.eval()
        mags, phases = [], []
        for x in noisy:
            m, p = stft_mag_phase(x, self.fs, self.nperseg, self.noverlap, self.pad)
            mags.append(m)
            phases.append(p)
        mags_np = np.stack(mags)
        mags_t = torch.tensor(mags_np, dtype=torch.float32, device=self.device).unsqueeze(1)
        with torch.no_grad():
            masks = self.model(mags_t).squeeze(1).cpu().numpy()
        out_mags = masks * mags_np
        return np.stack([
            istft_from_mag_phase(m, p, self.fs, self.nperseg, self.noverlap, self.pad, self.signal_len)
            for m, p in zip(out_mags, phases)
        ])

    # ── validation ────────────────────────────────────────────────────────────

    def _compute_val_snr(self) -> float:
        all_true, all_pred = [], []
        for noisy, clean in self.val_loader:
            all_pred.append(self.denoise_numpy(noisy.numpy()))
            all_true.append(clean.numpy())
        return float(SignalToNoiseRatio.calculate(
            np.concatenate(all_true), np.concatenate(all_pred)
        ))

    def _compute_val_loss(self, loss_fn) -> float:
        self.model.eval()
        total = 0.0
        with torch.no_grad():
            for noisy, clean in self.val_loader:
                noisy_np = noisy.numpy()
                clean_np = clean.numpy()
                mags, clean_mags = [], []
                for xn, xc in zip(noisy_np, clean_np):
                    nm, _ = stft_mag_phase(xn, self.fs, self.nperseg, self.noverlap, self.pad)
                    cm, _ = stft_mag_phase(xc, self.fs, self.nperseg, self.noverlap, self.pad)
                    mags.append(nm); clean_mags.append(cm)
                nm_t = torch.tensor(np.stack(mags),       dtype=torch.float32, device=self.device).unsqueeze(1)
                cm_t = torch.tensor(np.stack(clean_mags), dtype=torch.float32, device=self.device).unsqueeze(1)
                mask    = self.model(nm_t)
                out_mag = mask * nm_t
                total += loss_fn(out_mag, cm_t).item()
        return total / len(self.val_loader)

    # ── training loop ─────────────────────────────────────────────────────────

    def train(self) -> dict:
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        best_val_snr = float("-inf")
        best_sd = None
        train_history, val_snr_history = [], []

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0.0

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:02d}/{self.epochs}", leave=False, unit="batch")
            for noisy, clean in pbar:
                noisy_np = noisy.numpy()
                clean_np = clean.numpy()
                mags, clean_mags = [], []
                for xn, xc in zip(noisy_np, clean_np):
                    nm, _ = stft_mag_phase(xn, self.fs, self.nperseg, self.noverlap, self.pad)
                    cm, _ = stft_mag_phase(xc, self.fs, self.nperseg, self.noverlap, self.pad)
                    mags.append(nm); clean_mags.append(cm)

                nm_t = torch.tensor(np.stack(mags),       dtype=torch.float32, device=self.device).unsqueeze(1)
                cm_t = torch.tensor(np.stack(clean_mags), dtype=torch.float32, device=self.device).unsqueeze(1)

                mask    = self.model(nm_t)
                out_mag = mask * nm_t
                loss = loss_fn(out_mag, cm_t)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.5f}")

            val_loss = self._compute_val_loss(loss_fn)
            val_snr  = self._compute_val_snr()

            if WANDB_OK and hasattr(wandb, 'run') and wandb.run:
                wandb.log({
                    "train/mse_loss": epoch_loss / len(self.train_loader),
                    "val/mse_loss":   val_loss,
                    "val/snr_db":     val_snr,
                }, step=epoch)

            print(f"Epoch {epoch:02d}/{self.epochs} | "
                  f"train={epoch_loss / len(self.train_loader):.5f} | "
                  f"val_loss={val_loss:.5f} | val_SNR={val_snr:.2f} dB")

            train_history.append(epoch_loss / len(self.train_loader))
            val_snr_history.append(val_snr)

            if val_snr > best_val_snr:
                best_val_snr = val_snr
                best_sd = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        # ── save ──────────────────────────────────────────────────────────────
        run_dir = self.dataset_path / "weights" / "runs" / f"run_{self.run_date}_{self.run_id}_{MODEL_NAME}_{self.noise_type}"
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
