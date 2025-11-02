import uuid

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader, TensorDataset, random_split

from metrics import MeanSquaredError, MeanAbsoluteError, RootMeanSquaredError, SignalToNoiseRatio
from models.time_series_trasformer import TimeSeriesTransformer

wandb.login(key="")


class Trainer:
    def __init__(self, model, model_name, dataset_type="gaussian", batch_size=32, epochs=50,
                 learning_rate=1e-3, random_state=42, wandb_project="signal-denoising", device=None):
        self.model = model
        self.model_name = model_name
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = learning_rate
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

        # Unique run name
        run_name = f"{model_name}_{dataset_type}_{uuid.uuid4().hex[:8]}"
        wandb.init(project=wandb_project, name=run_name, config={
            "model": model_name,
            "dataset": dataset_type,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "random_state": random_state
        })

    def load_data(self):
        noisy_signals = np.load(f"../dataset/{self.dataset_type}_signals.npy")
        clean_signals = np.load("../dataset/clean_signals.npy")

        X = torch.tensor(noisy_signals, dtype=torch.float32).unsqueeze(-1)  # [B, T, 1]
        y = torch.tensor(clean_signals, dtype=torch.float32).unsqueeze(-1)  # [B, T, 1]

        dataset = TensorDataset(X, y)
        total_len = len(dataset)
        val_len = int(0.15 * total_len)
        test_len = int(0.15 * total_len)
        train_len = total_len - val_len - test_len

        return random_split(dataset, [train_len, val_len, test_len],
                            generator=torch.Generator().manual_seed(self.random_state))

    def train(self):
        train_set, val_set, test_set = self.load_data()
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.batch_size)
        test_loader = DataLoader(test_set, batch_size=self.batch_size)

        self.model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        best_val_loss = float('inf')
        best_weights = None

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_train_outputs = []
            epoch_train_targets = []
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                epoch_train_outputs.append(y_pred.detach().cpu().numpy())
                epoch_train_targets.append(y_batch.cpu().numpy())

            val_metrics = self.compute_epoch_metrics(val_loader)
            train_metrics = self.compute_epoch_metrics_from_numpy(epoch_train_outputs, epoch_train_targets)

            wandb.log({
                "train_mse": train_metrics["MSE"],
                "train_mae": train_metrics["MAE"],
                "train_rmse": train_metrics["RMSE"],
                "train_snr": train_metrics["SNR"],
                "val_mse": val_metrics["MSE"],
                "val_mae": val_metrics["MAE"],
                "val_rmse": val_metrics["RMSE"],
                "val_snr": val_metrics["SNR"],
                "train_loss": train_loss / len(train_loader),
            }, step=epoch)

            if val_metrics["MSE"] < best_val_loss:
                best_val_loss = val_metrics["MSE"]
                best_weights = self.model.state_dict()

            print(f"Epoch {epoch:02d} | "
                  f"Train Loss: {train_metrics['train_loss']:.4f} | "
                  f"Train MSE: {train_metrics['MSE']:.4f} | "
                  f"Val MSE: {val_metrics['MSE']:.4f}")

        # Save best model
        model_path = f"../weights/{self.model_name}_{self.dataset_type}_best.pth"
        torch.save(best_weights, model_path)
        print(f"✅ Best model saved to {model_path}")
        self.model.load_state_dict(best_weights)

        self.final_eval_metrics = self.evaluate_metrics(test_loader)

    def compute_epoch_metrics(self, loader):
        self.model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                preds = self.model(X_batch).cpu().squeeze().numpy()
                truths = y_batch.squeeze().numpy()
                y_pred.append(preds)
                y_true.append(truths)
        return self.compute_metrics(np.concatenate(y_true), np.concatenate(y_pred))

    def compute_epoch_metrics_from_numpy(self, pred_list, true_list):
        y_pred = np.concatenate(pred_list)
        y_true = np.concatenate(true_list)
        return self.compute_metrics(y_true, y_pred)

    def compute_metrics(self, y_true, y_pred):
        return {
            "MSE": MeanSquaredError.calculate(y_true, y_pred),
            "MAE": MeanAbsoluteError.calculate(y_true, y_pred),
            "RMSE": RootMeanSquaredError.calculate(y_true, y_pred),
            "SNR": SignalToNoiseRatio.calculate(y_true, y_pred),
        }

    def evaluate_metrics(self, loader):
        metrics = self.compute_epoch_metrics(loader)
        wandb.log({
            "test_mse": metrics["MSE"],
            "test_mae": metrics["MAE"],
            "test_rmse": metrics["RMSE"],
            "test_snr": metrics["SNR"],
        })

        print("\n📊 Final Test Metrics:")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}" if name != "SNR" else f"{name}: {value:.2f} dB")

        return metrics

    def evaluate(self, loader, criterion):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                y_pred = self.model(X_batch)
                loss = criterion(y_pred, y_batch)
                total_loss += loss.item()
        return total_loss / len(loader)


if __name__ == "__main__":
    input_dim = 1
    dataset_type = "non_gaussian"  # or "non_gaussian"
    random_state = 42
    batch_size = 32
    epochs = 50
    learning_rate = 1e-4
    wandb_project = "signal-denoising"

    model = TimeSeriesTransformer(input_dim=input_dim)

    trainer = Trainer(
        model=model,
        model_name="TimeSeriesTransformer",
        dataset_type=dataset_type,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        random_state=random_state,
        wandb_project=wandb_project,
    )

    trainer.train()
