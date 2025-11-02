import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from scipy.signal import stft, istft
import wandb
import uuid

from models.autoencoder_unet import UnetAutoencoder  # Adjust path
from metrics import MeanSquaredError, MeanAbsoluteError, RootMeanSquaredError, SignalToNoiseRatio

wandb.login(key="")


class UnetAutoencoderTrainer:
    def __init__(self, dataset_type="gaussian", batch_size=32, epochs=30, learning_rate=1e-4,
                 signal_len=1024, fs=1024, nperseg=256, random_state=42,
                 wandb_project="autoencoder", device=None):
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
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        run_name = f"UAE_{dataset_type}_{uuid.uuid4().hex[:8]}"
        wandb.init(project=wandb_project, name=run_name, config={
            "model": "UAE",
            "dataset": dataset_type,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "random_state": random_state
        })

        self.train_loader, self.val_loader, self.test_loader, self.input_shape = self.load_data()

        self.model = UnetAutoencoder(self.input_shape).to(self.device)

    def signal_to_mag(self, signal_batch):
        mags = []
        for s in signal_batch:
            _, _, Zxx = stft(s, fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap)
            mag = np.abs(Zxx)
            mags.append(mag)
        return torch.tensor(np.stack(mags), dtype=torch.float32).unsqueeze(1).to(self.device)

    def load_data(self):
        noisy = np.load(f"../dataset/{self.dataset_type}_signals.npy")
        clean = np.load("../dataset/clean_signals.npy")

        assert noisy.shape[1] == self.signal_len, "Signal length mismatch"
        assert clean.shape[1] == self.signal_len, "Signal length mismatch"

        # Save shapes for reference
        # Get input_shape for model
        _, _, Zxx = stft(clean[0], fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap)
        mag = np.abs(Zxx)
        input_shape = mag.shape

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
        loss_fn = nn.MSELoss()
        best_val_loss = float("inf")
        best_weights = None

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0
            train_true, train_pred = [], []

            for noisy, clean in self.train_loader:
                noisy, clean = noisy.to(self.device), clean.to(self.device)
                noisy_mag = self.signal_to_mag(noisy.cpu().numpy())
                clean_mag = self.signal_to_mag(clean.cpu().numpy())

                output = self.model(noisy_mag)
                loss = loss_fn(output, clean_mag)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                train_true.append(clean_mag.cpu().numpy())
                train_pred.append(output.detach().cpu().numpy())

            train_metrics = self.compute_epoch_metrics_from_numpy(train_true, train_pred)
            val_loss, val_metrics = self.evaluate_loss_and_metrics(self.val_loader, loss_fn)

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

            print(f"Epoch {epoch:02d} | Train Loss: {total_loss / len(self.train_loader):.6f} | "
                  f"Val Loss: {val_loss:.6f} | Val MSE: {val_metrics['MSE']:.2f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = self.model.state_dict()

        model_path = f"../weights/UnetAutoencoder_{self.dataset_type}_best.pth"
        torch.save(best_weights, model_path)
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
        val_true, val_pred = [], []
        with torch.no_grad():
            for noisy, clean in loader:
                noisy, clean = noisy.to(self.device), clean.to(self.device)
                noisy_mag = self.signal_to_mag(noisy.cpu().numpy())
                clean_mag = self.signal_to_mag(clean.cpu().numpy())

                output = self.model(noisy_mag)
                loss = criterion(output, clean_mag)
                total_loss += loss.item()
                val_pred.append(output.cpu().numpy())
                val_true.append(clean_mag.cpu().numpy())
        val_metrics = self.compute_epoch_metrics_from_numpy(val_true, val_pred)
        return total_loss / len(loader), val_metrics

    def evaluate_metrics(self, loader):
        self.model.eval()
        all_true, all_pred = [], []

        with torch.no_grad():
            for noisy, clean in loader:
                noisy, clean = noisy.to(self.device), clean.to(self.device)
                denoised = self.denoise_batch(noisy)
                all_pred.append(denoised)
                all_true.append(clean.cpu().numpy())

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
        # Denoising in spectrogram domain and reconstruct to time domain
        signal_batch_np = signal_batch.cpu().numpy()
        mags = []
        phases = []
        for s in signal_batch_np:
            _, _, Zxx = stft(s, fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap)
            mag = np.abs(Zxx)
            mags.append(mag)
            phases.append(np.angle(Zxx))

        mags_tensor = torch.tensor(np.stack(mags), dtype=torch.float32).unsqueeze(1).to(self.device)
        out_mag = self.model(mags_tensor).squeeze(1).detach().cpu().numpy()
        rec_signals = []
        for mag, phase in zip(out_mag, phases):
            Zxx_denoised = mag * np.exp(1j * phase)
            _, rec = istft(Zxx_denoised, fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap)
            if len(rec) < self.signal_len:
                rec = np.pad(rec, (0, self.signal_len - len(rec)))
            elif len(rec) > self.signal_len:
                rec = rec[:self.signal_len]
            rec_signals.append(rec)
        return np.stack(rec_signals)

if __name__ == "__main__":
    input_dim = 1
    dataset_type = "gaussian"  # or "non_gaussian"
    random_state = 42
    batch_size = 32
    epochs = 50
    learning_rate = 1e-4
    wandb_project = "signal-denoising"
    signal_len = 2144

    trainer = UnetAutoencoderTrainer(
        dataset_type="non_gaussian",
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        random_state=random_state,
        wandb_project=wandb_project,
        signal_len=signal_len
    )
    trainer.train()
