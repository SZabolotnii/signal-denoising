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
from torch.utils.data import DataLoader, TensorDataset, random_split

try:
    import wandb
    WANDB_OK = True
except Exception:
    WANDB_OK = False

from metrics import MeanSquaredError, MeanAbsoluteError, RootMeanSquaredError, SignalToNoiseRatio
from models.time_series_trasformer import TimeSeriesTransformer


class TransformerTrainer:
    def __init__(self, model, model_name, dataset_path: Path, noise_type="non_gaussian",
                 batch_size=32, epochs=50, learning_rate=1e-3, random_state=42,
                 wandb_project="", device=None):
        self.model = model
        self.model_name = model_name
        self.dataset_path = Path(dataset_path)
        self.noise_type = noise_type
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = learning_rate
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

        if WANDB_OK and wandb_project:
            run_name = f"{model_name}_{noise_type}_{uuid.uuid4().hex[:8]}"
            wandb.init(project=wandb_project, name=run_name, config={
                "model": model_name,
                "noise_type": noise_type,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "random_state": random_state
            })
            print(f"[W&B] Logging enabled → project='{wandb_project}', run='{run_name}'")
        else:
            reason = "wandb not installed" if not WANDB_OK else "no --wandb-project given"
            print(f"[W&B] Logging disabled ({reason})")

    def load_data(self):
        noisy_signals = np.load(self.dataset_path / "train" / f"{self.noise_type}_signals.npy")
        clean_signals = np.load(self.dataset_path / "train" / "clean_signals.npy")

        X = torch.tensor(noisy_signals, dtype=torch.float32).unsqueeze(-1)  # (N, T, 1)
        y = torch.tensor(clean_signals, dtype=torch.float32).unsqueeze(-1)

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
        val_loader   = DataLoader(val_set,   batch_size=self.batch_size)
        test_loader  = DataLoader(test_set,  batch_size=self.batch_size)

        self.model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        best_val_loss = float('inf')
        best_weights = None

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = 0.0
            epoch_train_outputs, epoch_train_targets = [], []

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

            if WANDB_OK and hasattr(wandb, 'run') and wandb.run:
                wandb.log({
                    "train_loss": train_loss / len(train_loader),
                    "train_mse": train_metrics["MSE"],
                    "val_mse": val_metrics["MSE"],
                }, step=epoch)

            print(f"Epoch {epoch:02d} | "
                  f"Train Loss: {train_loss / len(train_loader):.4f} | "
                  f"Train MSE: {train_metrics['MSE']:.4f} | "
                  f"Val MSE: {val_metrics['MSE']:.4f}")

            if val_metrics["MSE"] < best_val_loss:
                best_val_loss = val_metrics["MSE"]
                best_weights = self.model.state_dict()

        weights_dir = self.dataset_path / "weights"
        weights_dir.mkdir(exist_ok=True)
        save_path = weights_dir / f"{self.model_name}_{self.noise_type}_best.pth"
        torch.save(best_weights, save_path)
        print(f"✅ Best model saved to {save_path}")
        self.model.load_state_dict(best_weights)
        self.evaluate_metrics(test_loader)

    def compute_epoch_metrics(self, loader):
        self.model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                preds  = self.model(X_batch).cpu().squeeze().numpy()
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
            "MSE":  MeanSquaredError.calculate(y_true, y_pred),
            "MAE":  MeanAbsoluteError.calculate(y_true, y_pred),
            "RMSE": RootMeanSquaredError.calculate(y_true, y_pred),
            "SNR":  SignalToNoiseRatio.calculate(y_true, y_pred),
        }

    def evaluate_metrics(self, loader):
        metrics = self.compute_epoch_metrics(loader)
        if WANDB_OK and hasattr(wandb, 'run') and wandb.run:
            wandb.log({f"test_{k.lower()}": v for k, v in metrics.items()})
        print("\n📊 Final Test Metrics:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.2f} dB" if name == "SNR" else f"  {name}: {value:.6f}")
        return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train Transformer for signal denoising")
    p.add_argument("--dataset", required=True,
                   help="Path to dataset folder (e.g. data_generation/datasets/<name>)")
    p.add_argument("--noise-type", default="non_gaussian", choices=["gaussian", "non_gaussian"])
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch-size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--seed",       type=int,   default=42)
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

    model = TimeSeriesTransformer(input_dim=1)

    trainer = TransformerTrainer(
        model=model,
        model_name="TimeSeriesTransformer",
        dataset_path=dataset_path,
        noise_type=args.noise_type,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        random_state=args.seed,
        wandb_project=args.wandb_project,
    )
    trainer.train()
