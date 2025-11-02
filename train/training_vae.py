import uuid

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from scipy.signal import stft
from torch.utils.data import DataLoader, TensorDataset, random_split

wandb.login(key="")

from models.autoencoder_vae import SpectrogramVAE
from metrics import MeanSquaredError, MeanAbsoluteError, RootMeanSquaredError, SignalToNoiseRatio


class VAETrainer:
    def __init__(self, dataset_type="gaussian", batch_size=32, epochs=50, learning_rate=1e-3,
                 signal_len=1024, fs=1024, nperseg=256, random_state=42,
                 wandb_project="spectrogram-vae"):
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = learning_rate
        self.signal_len = signal_len
        self.fs = fs
        self.nperseg = 128
        self.noverlap = 96
        self.pad = self.nperseg // 2
        self.random_state = random_state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        run_name = f"VAE_{dataset_type}_{uuid.uuid4().hex[:8]}"
        wandb.init(project=wandb_project, name=run_name, config={
            "model": "VAE",
            "dataset": dataset_type,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "random_state": random_state
        })

        self.train_loader, self.val_loader, self.test_loader, self.freq_bins, self.time_frames = self.load_data()

        self.model = SpectrogramVAE(freq_bins=self.freq_bins,
                                    time_frames=self.time_frames).to(self.device)

    def signal_to_mag(self, signal_batch):
        # signal_batch: [B, 1, T]
        padded = nn.functional.pad(signal_batch, (self.pad, self.pad), mode="reflect")
        signals_np = padded.squeeze(1).cpu().numpy()

        mags = []
        for s in signals_np:
            _, _, Zxx = stft(s, fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap)
            mags.append(np.abs(Zxx))

        return torch.tensor(np.stack(mags), dtype=torch.float32).unsqueeze(1).to(self.device)

    def load_data(self):
        noisy = np.load(f"../dataset/{self.dataset_type}_signals.npy")
        clean = np.load("../dataset/clean_signals.npy")

        assert noisy.shape[
                   1] == self.signal_len, f"Noisy signal length mismatch: expected {self.signal_len}, got {noisy.shape[1]}"
        assert clean.shape[
                   1] == self.signal_len, f"Clean signal length mismatch: expected {self.signal_len}, got {clean.shape[1]}"

        X = torch.tensor(noisy[:, :self.signal_len], dtype=torch.float32).unsqueeze(1)
        y = torch.tensor(clean[:, :self.signal_len], dtype=torch.float32).unsqueeze(1)

        dataset = TensorDataset(X, y)
        total_len = len(dataset)
        val_len = int(0.15 * total_len)
        test_len = int(0.15 * total_len)
        train_len = total_len - val_len - test_len

        train_set, val_set, test_set = random_split(
            dataset, [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(self.random_state)
        )

        # Get shape of one spectrogram
        example = self.signal_to_mag(X[:1])
        _, _, freq_bins, time_frames = example.shape

        return (
            DataLoader(train_set, batch_size=self.batch_size, shuffle=True),
            DataLoader(val_set, batch_size=self.batch_size),
            DataLoader(test_set, batch_size=self.batch_size),
            freq_bins,
            time_frames
        )

    def train(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss(reduction="sum")

        best_val_loss = float("inf")
        best_weights = None

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0

            train_y_true, train_y_pred = [], []

            for X_batch, _ in self.train_loader:
                X_batch = X_batch.to(self.device)
                spec = self.signal_to_mag(X_batch)

                recon, mu, logvar = self.model(spec)
                rec_loss = loss_fn(recon, spec)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = (rec_loss + kl_loss) / spec.size(0)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                # Save for metrics
                train_y_pred.append(recon.detach().cpu().numpy())
                train_y_true.append(spec.cpu().numpy())

            # Compute train metrics on spectrogram domain
            train_metrics = self.compute_epoch_metrics_from_numpy(train_y_true, train_y_pred)

            # Validation
            val_loss, val_metrics = self.evaluate_loss_and_metrics(self.val_loader, loss_fn)

            # Log all metrics
            wandb.log({
                "train_loss": total_loss / len(self.train_loader),
                "train_mse": train_metrics["MSE"],
                "train_mae": train_metrics["MAE"],
                "train_rmse": train_metrics["RMSE"],
                "train_snr": train_metrics["SNR"],
                "val_loss": val_loss,
                "val_mse": val_metrics["MSE"],
                "val_mae": val_metrics["MAE"],
                "val_rmse": val_metrics["RMSE"],
                "val_snr": val_metrics["SNR"],
            }, step=epoch)

            print(f"Epoch {epoch:02d} | "
                  f"Train Loss: {total_loss / len(self.train_loader):.2f} | "
                  f"Val Loss: {val_loss:.2f} | "
                  f"Val MSE: {val_metrics['MSE']:.2f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = self.model.state_dict()

        torch.save(best_weights, f"../weights/SpectrogramVAE_{self.dataset_type}_best.pth")
        print("✅ Best model saved.")
        self.model.load_state_dict(best_weights)

        self.evaluate_metrics(self.test_loader)

    def compute_epoch_metrics_from_numpy(self, true_list, pred_list):
        y_true = np.concatenate(true_list)
        y_pred = np.concatenate(pred_list)
        return self.compute_metrics(y_true, y_pred)

    def compute_metrics(self, y_true, y_pred):
        return {
            "MSE": MeanSquaredError.calculate(y_true, y_pred),
            "MAE": MeanAbsoluteError.calculate(y_true, y_pred),
            "RMSE": RootMeanSquaredError.calculate(y_true, y_pred),
            "SNR": SignalToNoiseRatio.calculate(y_true, y_pred),
        }

    def evaluate_loss_and_metrics(self, loader, criterion):
        self.model.eval()
        total_loss = 0
        val_y_true, val_y_pred = [], []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device)
                spec = self.signal_to_mag(X_batch)
                recon, mu, logvar = self.model(spec)
                rec_loss = criterion(recon, spec)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = (rec_loss + kl_loss) / spec.size(0)
                total_loss += loss.item()
                val_y_pred.append(recon.cpu().numpy())
                val_y_true.append(spec.cpu().numpy())
        val_metrics = self.compute_epoch_metrics_from_numpy(val_y_true, val_y_pred)
        return total_loss / len(loader), val_metrics

    def evaluate_metrics(self, loader):
        self.model.eval()
        all_true, all_pred = [], []

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                denoised = self.denoise_batch(X_batch).cpu().numpy()
                all_pred.append(denoised)
                all_true.append(y_batch.numpy())

        y_true = np.concatenate(all_true)
        y_pred = np.concatenate(all_pred)

        metrics = {
            "MSE": MeanSquaredError.calculate(y_true, y_pred),
            "MAE": MeanAbsoluteError.calculate(y_true, y_pred),
            "RMSE": RootMeanSquaredError.calculate(y_true, y_pred),
            "SNR": SignalToNoiseRatio.calculate(y_true, y_pred),
        }

        wandb.log({f"test_{k.lower()}": v for k, v in metrics.items()})
        print("\n📊 Final Test Metrics:")
        for name, val in metrics.items():
            print(f"{name}: {val:.4f}" if name != "SNR" else f"{name}: {val:.2f} dB")

    def denoise_batch(self, signal_batch):
        """
        Runs STFT -> VAE -> ISTFT to produce denoised time-domain signals
        """
        padded = nn.functional.pad(signal_batch, (self.pad, self.pad), mode='reflect')
        signals_np = padded.squeeze(1).cpu().numpy()

        mags, phases = [], []
        for s in signals_np:
            _, _, Zxx = stft(s, fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap)
            mags.append(np.abs(Zxx))
            phases.append(np.angle(Zxx))

        spec_tensor = torch.tensor(np.stack(mags), dtype=torch.float32).unsqueeze(1).to(self.device)
        out_mag, _, _ = self.model(spec_tensor)
        out_mag = out_mag.squeeze(1).detach().cpu().numpy()

        # Reconstruct time-domain
        from scipy.signal import istft
        rec_signals = []
        for mag, phase in zip(out_mag, phases):
            Zxx_denoised = mag * np.exp(1j * phase)
            _, rec = istft(Zxx_denoised, fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap)
            rec = rec[self.pad: self.pad + self.signal_len]
            if len(rec) < self.signal_len:
                rec = np.pad(rec, (0, self.signal_len - len(rec)))
            elif len(rec) > self.signal_len:
                rec = rec[:self.signal_len]
            rec_signals.append(rec)

        return torch.tensor(np.stack(rec_signals), dtype=torch.float32).unsqueeze(1)


if __name__ == "__main__":
    input_dim = 1
    dataset_type = "non_gaussian"  # or "non_gaussian"
    random_state = 42
    batch_size = 32
    epochs = 50
    learning_rate = 1e-4
    wandb_project = "signal-denoising"
    signal_len = 2144

    trainer = VAETrainer(
        dataset_type=dataset_type,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        random_state=random_state,
        wandb_project=wandb_project,
        signal_len=signal_len
    )
    trainer.train()
