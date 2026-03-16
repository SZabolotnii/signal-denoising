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
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from scipy.signal import stft, istft

try:
    import wandb
    WANDB_OK = True
except Exception:
    WANDB_OK = False

from models.autoencoder_unet import UnetAutoencoder
from metrics import MeanSquaredError, MeanAbsoluteError, RootMeanSquaredError, SignalToNoiseRatio

WINDOW = 'hann'
EPS = 1e-12


def stft_mag_phase(x, fs, nperseg, noverlap, pad):
    x_pad = np.pad(x, pad, mode="reflect")
    _, _, Zxx = stft(x_pad, fs=fs, nperseg=nperseg, noverlap=noverlap, window=WINDOW, boundary=None, padded=False)
    return np.abs(Zxx).astype(np.float32), np.angle(Zxx).astype(np.float32)


def istft_from_mag_phase(mag, phase, fs, nperseg, noverlap, pad, target_len):
    _, rec = istft(mag * np.exp(1j * phase),
                   fs=fs, nperseg=nperseg, noverlap=noverlap,
                   window='hann', input_onesided=True, boundary=None)
    rec = rec[pad: pad + target_len]
    if len(rec) < target_len:
        rec = np.pad(rec, (0, target_len - len(rec)))
    elif len(rec) > target_len:
        rec = rec[:target_len]
    return rec.astype(np.float32)


def stft_mag_torch(x: torch.Tensor, nperseg: int, noverlap: int) -> torch.Tensor:
    hop = nperseg - noverlap
    window = torch.hann_window(nperseg, periodic=True, device=x.device, dtype=x.dtype)
    X = torch.stft(x, n_fft=nperseg, hop_length=hop, win_length=nperseg,
                   window=window, center=True, return_complex=True)
    return torch.abs(X)


def spectral_convergence(S_hat: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    diff = S_hat - S
    num = torch.linalg.norm(diff, ord='fro', dim=(1, 2))
    den = torch.linalg.norm(S, ord='fro', dim=(1, 2)) + 1e-12
    return (num / den).mean()


def multi_res_stft_loss(x_hat: torch.Tensor, x: torch.Tensor,
                        configs=((128, 64), (256, 128), (64, 32)),
                        alpha=1.0, beta=0.5) -> torch.Tensor:
    total = 0.0
    for nperseg, noverlap in configs:
        S_hat = stft_mag_torch(x_hat, nperseg, noverlap)
        S = stft_mag_torch(x, nperseg, noverlap)
        l1 = torch.mean(torch.abs(torch.log1p(S_hat) - torch.log1p(S)))
        sc = spectral_convergence(S_hat, S)
        total = total + alpha * l1 + beta * sc
    return total / len(configs)


class UnetAutoencoderTrainer:
    def __init__(self, dataset_path: Path, noise_type="non_gaussian",
                 batch_size=32, epochs=30, learning_rate=1e-4,
                 signal_len=256, fs=8192, nperseg=32, noverlap=16, random_state=42,
                 wandb_project="", device=None):
        self.dataset_path = Path(dataset_path)
        self.noise_type = noise_type
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = learning_rate
        self.signal_len = signal_len
        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.pad = self.nperseg // 2
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if self.noverlap != self.nperseg // 2:
            print(f"[INFO] Adjusting noverlap from {self.noverlap} to {self.nperseg // 2} "
                  f"for Hann COLA consistency.")
            self.noverlap = self.nperseg // 2

        if WANDB_OK and wandb_project:
            run_name = f"MaskUNet_{noise_type}_{uuid.uuid4().hex[:8]}"
            wandb.init(project=wandb_project, name=run_name, config={
                "model": "MaskUNet",
                "noise_type": noise_type,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "random_state": random_state,
                "fs": fs, "nperseg": nperseg, "noverlap": noverlap
            })
            print(f"[W&B] Logging enabled → project='{wandb_project}', run='{run_name}'")
        else:
            reason = "wandb not installed" if not WANDB_OK else "no --wandb-project given"
            print(f"[W&B] Logging disabled ({reason})")

        self.train_loader, self.val_loader, self.test_loader, self.input_shape = self._load_data()
        self.model = UnetAutoencoder(self.input_shape).to(self.device)

    def _load_data(self):
        noisy = np.load(self.dataset_path / "train" / f"{self.noise_type}_signals.npy")
        clean = np.load(self.dataset_path / "train" / "clean_signals.npy")
        assert noisy.shape[1] == self.signal_len and clean.shape[1] == self.signal_len, \
            f"Signal length mismatch: expected {self.signal_len}, got noisy={noisy.shape[1]}, clean={clean.shape[1]}"

        mag0, _ = stft_mag_phase(clean[0], fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap, pad=self.pad)
        input_shape = mag0.shape

        dataset = TensorDataset(
            torch.tensor(noisy, dtype=torch.float32),
            torch.tensor(clean, dtype=torch.float32)
        )
        total_len = len(dataset)
        val_len = int(0.15 * total_len)
        test_len = int(0.15 * total_len)
        train_len = total_len - val_len - test_len

        train_set, val_set, test_set = random_split(
            dataset, [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(self.random_state)
        )

        return (
            DataLoader(train_set, batch_size=self.batch_size, shuffle=True),
            DataLoader(val_set, batch_size=self.batch_size),
            DataLoader(test_set, batch_size=self.batch_size),
            input_shape
        )

    def train(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        best_val = float("inf")
        best_sd = None

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0.0

            for noisy, clean in self.train_loader:
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)

                noisy_np = noisy.cpu().numpy()
                clean_np = clean.cpu().numpy()

                noisy_mag_list, clean_mag_list, phase_list = [], [], []
                for xn, xc in zip(noisy_np, clean_np):
                    nm, np_phase = stft_mag_phase(xn, self.fs, self.nperseg, self.noverlap, self.pad)
                    cm, _ = stft_mag_phase(xc, self.fs, self.nperseg, self.noverlap, self.pad)
                    noisy_mag_list.append(nm)
                    clean_mag_list.append(cm)
                    phase_list.append(np_phase)

                noisy_mag = torch.tensor(np.stack(noisy_mag_list), dtype=torch.float32, device=self.device).unsqueeze(1)
                clean_mag = torch.tensor(np.stack(clean_mag_list), dtype=torch.float32, device=self.device).unsqueeze(1)

                mask = self.model(noisy_mag)
                out_mag = mask * noisy_mag
                base_loss = torch.mean(torch.abs(torch.log1p(out_mag) - torch.log1p(clean_mag)))

                out_mag_np = out_mag.squeeze(1).detach().cpu().numpy()
                rec_list = []
                for om, ph in zip(out_mag_np, phase_list):
                    rec = istft_from_mag_phase(om, ph, self.fs, self.nperseg, self.noverlap, self.pad, self.signal_len)
                    rec_list.append(rec)
                rec_batch = torch.tensor(np.stack(rec_list), device=self.device)

                mr_loss = multi_res_stft_loss(
                    x_hat=rec_batch, x=clean,
                    configs=((32, 16), (64, 32), (16, 8)), alpha=1.0, beta=0.5
                )

                loss = base_loss + 0.5 * mr_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())

            val_loss = self._validate()

            if WANDB_OK and hasattr(wandb, 'run') and wandb.run:
                wandb.log({"train_loss": epoch_loss / len(self.train_loader),
                           "val_loss": val_loss}, step=epoch)

            print(f"Epoch {epoch:02d} | train_loss={epoch_loss / len(self.train_loader):.5f} | val_loss={val_loss:.5f}")

            if val_loss < best_val:
                best_val = val_loss
                best_sd = self.model.state_dict()

        weights_dir = self.dataset_path / "weights"
        weights_dir.mkdir(exist_ok=True)
        save_path = weights_dir / f"UnetAutoencoder_{self.noise_type}_best.pth"
        torch.save(best_sd, save_path)
        print(f"✅ Best model saved to: {save_path}")
        self.model.load_state_dict(best_sd)

        self.evaluate_metrics(self.test_loader)

    def _validate(self):
        self.model.eval()
        total = 0.0
        with torch.no_grad():
            for noisy, clean in self.val_loader:
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)

                noisy_np = noisy.cpu().numpy()
                clean_np = clean.cpu().numpy()
                noisy_mag_list, clean_mag_list, phase_list = [], [], []
                for xn, xc in zip(noisy_np, clean_np):
                    nm, np_phase = stft_mag_phase(xn, self.fs, self.nperseg, self.noverlap, self.pad)
                    cm, _ = stft_mag_phase(xc, self.fs, self.nperseg, self.noverlap, self.pad)
                    noisy_mag_list.append(nm)
                    clean_mag_list.append(cm)
                    phase_list.append(np_phase)

                noisy_mag = torch.tensor(np.stack(noisy_mag_list), dtype=torch.float32, device=self.device).unsqueeze(1)
                clean_mag = torch.tensor(np.stack(clean_mag_list), dtype=torch.float32, device=self.device).unsqueeze(1)

                mask = self.model(noisy_mag)
                out_mag = mask * noisy_mag
                base_loss = torch.mean(torch.abs(torch.log1p(out_mag) - torch.log1p(clean_mag)))

                out_mag_np = out_mag.squeeze(1).cpu().numpy()
                rec_list = []
                for om, ph in zip(out_mag_np, phase_list):
                    rec = istft_from_mag_phase(om, ph, self.fs, self.nperseg, self.noverlap, self.pad, self.signal_len)
                    rec_list.append(rec)
                rec_batch = torch.tensor(np.stack(rec_list), device=self.device)

                mr_loss = multi_res_stft_loss(
                    x_hat=rec_batch, x=clean,
                    configs=((32, 16), (64, 32), (16, 8)), alpha=1.0, beta=0.5
                )

                loss = base_loss + 0.5 * mr_loss
                total += float(loss.item())

        return total / len(self.val_loader)

    def denoise_batch(self, noisy_batch):
        self.model.eval()
        noisy_np = noisy_batch.cpu().numpy()
        out_rec = []
        with torch.no_grad():
            for xn in noisy_np:
                nm, ph = stft_mag_phase(xn, self.fs, self.nperseg, self.noverlap, self.pad)
                nm_t = torch.tensor(nm, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
                mask = self.model(nm_t)
                om = (mask.squeeze().cpu().numpy() * nm)
                rec = istft_from_mag_phase(om, ph, self.fs, self.nperseg, self.noverlap, self.pad, self.signal_len)
                out_rec.append(rec)
        return np.stack(out_rec)

    def evaluate_metrics(self, loader):
        all_true, all_pred = [], []
        with torch.no_grad():
            for noisy, clean in loader:
                noisy = noisy.to(self.device)
                pred = self.denoise_batch(noisy)
                all_pred.append(pred)
                all_true.append(clean.cpu().numpy())
        y_true = np.concatenate(all_true, axis=0)
        y_pred = np.concatenate(all_pred, axis=0)

        metrics = {
            "MSE":  MeanSquaredError.calculate(y_true, y_pred),
            "MAE":  MeanAbsoluteError.calculate(y_true, y_pred),
            "RMSE": RootMeanSquaredError.calculate(y_true, y_pred),
            "SNR":  SignalToNoiseRatio.calculate(y_true, y_pred),
        }
        if WANDB_OK and hasattr(wandb, 'run') and wandb.run:
            wandb.log({f"test_{k.lower()}": v for k, v in metrics.items()})
        print("\n📊 Final Test Metrics (time domain):")
        for k, v in metrics.items():
            print(f"  {k}: {v:.2f} dB" if k == "SNR" else f"  {k}: {v:.6f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train UNet autoencoder for signal denoising")
    p.add_argument("--dataset", required=True,
                   help="Path to dataset folder (e.g. data_generation/datasets/<name>)")
    p.add_argument("--noise-type", default="non_gaussian", choices=["gaussian", "non_gaussian"])
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--batch-size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--nperseg",    type=int,   default=32)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--wandb-project", default="")
    args = p.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path

    with open(dataset_path / "dataset_config.json") as f:
        cfg = json.load(f)

    signal_len = cfg["block_size"]
    fs         = cfg["sample_rate"]
    noverlap   = args.nperseg // 2

    print(f"Dataset: {dataset_path.name}")
    print(f"Config:  block_size={signal_len}, sample_rate={fs}, noise_type={args.noise_type}")

    trainer = UnetAutoencoderTrainer(
        dataset_path=dataset_path,
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
