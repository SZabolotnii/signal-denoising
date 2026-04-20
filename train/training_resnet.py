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
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

try:
    import wandb
    WANDB_OK = True
except Exception:
    WANDB_OK = False

from tqdm import tqdm

from models.autoencoder_resnet import ResNetAutoencoder
from metrics import MeanSquaredError, MeanAbsoluteError, RootMeanSquaredError, SignalToNoiseRatio
from train.losses import select_loss
from train.snr_curve import evaluate_per_snr, print_snr_table, plot_snr_curve, log_snr_curve_wandb, save_training_curves

MODEL_NAME = 'ResNetAutoencoder'


class ResNetAutoencoderTrainer:
    def __init__(self, dataset_path: Path, noise_type="non_gaussian",
                 batch_size=1024, epochs=50, learning_rate=6e-4,
                 signal_len=256, fs=8192, nperseg=128, random_state=42,
                 wandb_project="", data_fraction=1.0, output_dir=None, device=None):
        self.dataset_path = Path(dataset_path)
        self.noise_type = noise_type
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = learning_rate
        self.signal_len = signal_len
        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = int(nperseg * 0.75)
        self.pad = nperseg // 2
        self.random_state = random_state
        from train.repro_utils import set_global_seed
        set_global_seed(random_state)
        self.data_fraction = data_fraction
        self.output_dir = Path(output_dir) if output_dir is not None else None
        from train.device_utils import get_device
        self.device = get_device(device)

        self.run_id = uuid.uuid4().hex[:8]
        self.run_date = datetime.now().strftime("%Y%m%d")
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

        self.train_loader, self.val_loader, self.test_loader, self.input_shape = self.load_data()
        self.model = ResNetAutoencoder(self.input_shape).to(self.device)

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

    def _signal_to_mag_tensor(self, signal_batch: torch.Tensor) -> torch.Tensor:
        """[B, 1, T] → [B, 1, F, T'] spectral magnitude."""
        spec = self._stft_batch(signal_batch.squeeze(1).to(self.device))
        return spec.abs().unsqueeze(1)

    def _precompute_stft_mag(self, signals: np.ndarray, chunk_size: int = 50000) -> torch.Tensor:
        """Precompute STFT magnitudes on CPU in chunks. [N, T] → [N, 1, F, T']."""
        chunks = []
        for i in range(0, len(signals), chunk_size):
            x = torch.tensor(signals[i:i + chunk_size], dtype=torch.float32)
            chunks.append(self._stft_batch(x).abs().unsqueeze(1))
        return torch.cat(chunks)

    def load_data(self):
        noisy = np.load(self.dataset_path / "train" / f"{self.noise_type}_signals.npy")
        clean = np.load(self.dataset_path / "train" / "clean_signals.npy")
        if self.data_fraction < 1.0:
            n = max(1, int(len(noisy) * self.data_fraction))
            noisy, clean = noisy[:n], clean[:n]
        assert noisy.shape[1] == self.signal_len

        # Precompute STFT magnitudes on CPU (one-time cost)
        print("  Precomputing STFT magnitudes on CPU …")
        noisy_mag = self._precompute_stft_mag(noisy)
        clean_mag = self._precompute_stft_mag(clean)
        input_shape = (int(noisy_mag.shape[2]), int(noisy_mag.shape[3]))
        print(f"  Done: {noisy_mag.shape} per set, "
              f"{2 * noisy_mag.nelement() * 4 / 1e9:.1f} GB total")

        noisy_raw = torch.tensor(noisy, dtype=torch.float32).unsqueeze(1)  # [N,1,T]
        clean_raw = torch.tensor(clean, dtype=torch.float32).unsqueeze(1)

        dataset = TensorDataset(noisy_mag, clean_mag, noisy_raw, clean_raw)
        total = len(dataset)
        val_len  = int(0.25 * total)
        test_len = int(0.25 * total)
        train_len = total - val_len - test_len
        train_set, val_set, test_set = random_split(
            dataset, [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(self.random_state),
        )

        from train.device_utils import get_dataloader_kwargs
        dl_kw = get_dataloader_kwargs(self.device)
        return (
            DataLoader(train_set, batch_size=self.batch_size, shuffle=True, **dl_kw),
            DataLoader(val_set,   batch_size=self.batch_size, **dl_kw),
            DataLoader(test_set,  batch_size=self.batch_size, **dl_kw),
            input_shape,
        )

    # ── inference ─────────────────────────────────────────────────────────────

    def denoise_numpy(self, noisy: np.ndarray) -> np.ndarray:
        """[N, T] → [N, T]"""
        t = torch.tensor(noisy, dtype=torch.float32).unsqueeze(1)   # [N, 1, T]
        result = self._denoise_batch(t.to(self.device))              # [N, 1, T]
        return result.squeeze(1).cpu().numpy()

    def _denoise_batch(self, signal_batch: torch.Tensor) -> torch.Tensor:
        """[N, 1, T] → [N, 1, T]"""
        self.model.eval()
        x = signal_batch.squeeze(1).to(self.device)
        spec = self._stft_batch(x)
        mag = spec.abs().unsqueeze(1)
        with torch.no_grad():
            out_mag = self.model(mag) * mag
        phase = spec / (spec.abs() + 1e-8)
        out_spec = out_mag.squeeze(1) * phase
        return self._istft_batch(out_spec).unsqueeze(1)

    # ── validation ────────────────────────────────────────────────────────────

    def _compute_val_snr(self) -> float:
        all_true, all_pred = [], []
        for _nm, _cm, noisy_raw, clean_raw in tqdm(self.val_loader, desc="  val SNR", leave=False, unit="batch"):
            pred = self.denoise_numpy(noisy_raw.squeeze(1).numpy())
            all_pred.append(pred)
            all_true.append(clean_raw.squeeze(1).numpy())
        return float(SignalToNoiseRatio.calculate(
            np.concatenate(all_true), np.concatenate(all_pred)
        ))

    def _compute_val_loss(self, loss_fn) -> float:
        self.model.eval()
        total = 0.0
        with torch.no_grad():
            for noisy_mag, clean_mag, _, _ in tqdm(self.val_loader, desc="  val loss", leave=False, unit="batch"):
                nm = noisy_mag.to(self.device)
                cm = clean_mag.to(self.device)
                total += loss_fn(self.model(nm) * nm, cm).item()
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
        early_stop_patience = 5

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0.0
            from train.device_utils import reset_peak_memory
            reset_peak_memory(self.device)

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:02d}/{self.epochs}", leave=False, unit="batch")
            for noisy_mag, clean_mag, _, _ in pbar:
                nm = noisy_mag.to(self.device)
                cm = clean_mag.to(self.device)
                mask = self.model(nm)
                loss = loss_fn(mask * nm, cm)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.5f}")

            from train.device_utils import format_vram_str
            vram_str = format_vram_str(self.device)

            val_loss = self._compute_val_loss(loss_fn)
            val_snr  = self._compute_val_snr()

            scheduler.step(val_loss)
            lr_now = optimizer.param_groups[0]['lr']

            if WANDB_OK and hasattr(wandb, 'run') and wandb.run:
                wandb.log({
                    "train/mse_loss": total_loss / len(self.train_loader),
                    "val/mse_loss":   val_loss,
                    "val/snr_db":     val_snr,
                    "train/lr":       lr_now,
                }, step=epoch)

            print(f"Epoch {epoch:02d}/{self.epochs} | "
                  f"train={total_loss / len(self.train_loader):.5f} | "
                  f"val_loss={val_loss:.5f} | val_SNR={val_snr:.2f} dB | "
                  f"lr={lr_now:.2e}{vram_str}")

            train_history.append(total_loss / len(self.train_loader))
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

        test_metrics = self._evaluate_test()

        per_snr = {}
        test_dir = self.dataset_path / "test"
        if test_dir.exists():
            per_snr = evaluate_per_snr(self.denoise_numpy, test_dir, self.noise_type, batch_size=self.batch_size)
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
        with torch.no_grad():
            for _, _, noisy_raw, clean_raw in self.test_loader:
                pred = self.denoise_numpy(noisy_raw.squeeze(1).numpy())
                all_pred.append(pred)
                all_true.append(clean_raw.squeeze(1).numpy())
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
    p = argparse.ArgumentParser(description="Train ResNet autoencoder for signal denoising")
    p.add_argument("--dataset",       required=True)
    p.add_argument("--noise-type",    default="non_gaussian", choices=["gaussian", "non_gaussian"])
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--batch-size",    type=int,   default=1024)
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

    ResNetAutoencoderTrainer(
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
