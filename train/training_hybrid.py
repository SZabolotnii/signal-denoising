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

from models.hybrid_unet import HybridDSGE_UNet
from models.dsge_layer import DSGEFeatureExtractor
from metrics import MeanSquaredError, MeanAbsoluteError, RootMeanSquaredError, SignalToNoiseRatio


class HybridUnetTrainer:
    """
    Тренер гібридної DSGE + U-Net моделі.

    Пайплайн:
      1. Завантаження clean/noisy з dataset_path/train/
      2. DSGEFeatureExtractor.fit() ТІЛЬКИ на тренувальних даних
      3. 4-канальний препроцесинг: [STFT(x̃), STFT(φ₁), STFT(φ₂), STFT(φ₃)]
      4. Тренування HybridDSGE_UNet (MSE, Adam)
      5. Збереження ваг і стану DSGE в dataset_path/weights/
    """

    def __init__(
        self,
        dataset_path: Path,
        noise_type: str = 'non_gaussian',
        dsge_order: int = 3,
        dsge_basis: str = 'fractional',
        dsge_powers: list | None = None,
        tikhonov_lambda: float = 0.01,
        batch_size: int = 32,
        epochs: int = 30,
        learning_rate: float = 1e-4,
        signal_len: int = 256,
        fs: int = 8192,
        nperseg: int = 32,
        noverlap: int = 16,
        random_state: int = 42,
        wandb_project: str = '',
        device: str | None = None,
    ):
        self.dataset_path = Path(dataset_path)
        self.noise_type = noise_type
        self.dsge_order = dsge_order
        self.dsge_basis = dsge_basis
        self.dsge_powers = dsge_powers or [0.5, 1.5, 2.0]
        self.tikhonov_lambda = tikhonov_lambda
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = learning_rate
        self.signal_len = signal_len
        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.random_state = random_state
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        if WANDB_OK and wandb_project:
            run_name = f"HybridDSGE_{noise_type}_S{dsge_order}_{uuid.uuid4().hex[:6]}"
            wandb.init(
                project=wandb_project,
                name=run_name,
                config={
                    'model': 'HybridDSGE_UNet',
                    'noise_type': noise_type,
                    'dsge_order': dsge_order,
                    'dsge_basis': dsge_basis,
                    'dsge_powers': self.dsge_powers,
                    'tikhonov_lambda': tikhonov_lambda,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'random_state': random_state,
                },
            )
            print(f"[W&B] Logging enabled → project='{wandb_project}', run='{run_name}'")
        else:
            reason = "wandb not installed" if not WANDB_OK else "no --wandb-project given"
            print(f"[W&B] Logging disabled ({reason})")

        self.train_loader, self.val_loader, self.test_loader, self.input_shape, \
            self.train_clean, self.train_noisy = self._load_data()

        print(f"[Info] Fitting DSGEFeatureExtractor on {len(self.train_clean)} train samples…")
        self.dsge = DSGEFeatureExtractor(
            basis_type=dsge_basis,
            powers=self.dsge_powers,
            tikhonov_lambda=tikhonov_lambda,
            stft_params={'nperseg': nperseg, 'noverlap': noverlap, 'fs': fs},
        )
        self.dsge.fit(self.train_clean, self.train_noisy)
        self.dsge.check_generating_element_norm()
        print(f"[Info] DSGE ready: {self.dsge}")

        self.model = HybridDSGE_UNet(
            input_shape=self.input_shape,
            dsge_order=dsge_order,
        ).to(self.device)
        print(f"[Info] Model params: {self.model.param_count():,}")

    # ──────────────────────────────────────────────────
    #  Data loading
    # ──────────────────────────────────────────────────

    def _load_data(self):
        noisy = np.load(self.dataset_path / "train" / f"{self.noise_type}_signals.npy")
        clean = np.load(self.dataset_path / "train" / "clean_signals.npy")

        assert noisy.shape[1] == self.signal_len, \
            f"Signal length mismatch: expected {self.signal_len}, got {noisy.shape[1]}"

        _, _, Zxx = stft(clean[0], fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap)
        input_shape = np.abs(Zxx).shape

        dataset = TensorDataset(
            torch.tensor(noisy, dtype=torch.float32),
            torch.tensor(clean, dtype=torch.float32),
        )
        total = len(dataset)
        val_len = int(0.15 * total)
        test_len = int(0.15 * total)
        train_len = total - val_len - test_len

        g = torch.Generator().manual_seed(self.random_state)
        train_set, val_set, test_set = random_split(
            dataset, [train_len, val_len, test_len], generator=g
        )

        train_idx = train_set.indices
        train_clean = clean[train_idx]
        train_noisy = noisy[train_idx]

        return (
            DataLoader(train_set, batch_size=self.batch_size, shuffle=True),
            DataLoader(val_set,   batch_size=self.batch_size),
            DataLoader(test_set,  batch_size=self.batch_size),
            input_shape,
            train_clean,
            train_noisy,
        )

    # ──────────────────────────────────────────────────
    #  Preprocessing: 4-channel input
    # ──────────────────────────────────────────────────

    def _batch_to_4ch(self, signal_batch: np.ndarray) -> torch.Tensor:
        """
        [N, T] → (B, 1+S, F, T') з нормалізованими DSGE-каналами.

        Нормалізація DSGE-каналів критична: без неї великі φ₃(x)
        призводять до колапсу в тривіальний нуль-маску.
        """
        stft_mags = []
        dsge_mags_list = [[] for _ in range(self.dsge_order)]

        for s in signal_batch:
            _, _, Zxx = stft(s, fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap)
            stft_mags.append(np.abs(Zxx))
            dsge_specs = self.dsge.compute_dsge_spectrograms(s)  # [S, F, T']
            for i in range(self.dsge_order):
                dsge_mags_list[i].append(dsge_specs[i])

        stft_stack = np.stack(stft_mags)  # [B, F, T']
        stft_ref_max = stft_stack.max() + 1e-8
        channels = [stft_stack]

        for i in range(self.dsge_order):
            dsge_ch = np.stack(dsge_mags_list[i])
            dsge_max = dsge_ch.max() + 1e-8
            channels.append(dsge_ch * (stft_ref_max / dsge_max))

        return torch.tensor(
            np.stack(channels, axis=1), dtype=torch.float32
        ).to(self.device)  # [B, 1+S, F, T']

    def _signal_to_mag(self, signal_batch: np.ndarray) -> torch.Tensor:
        """1-канальна STFT для target (чистий сигнал)."""
        mags = []
        for s in signal_batch:
            _, _, Zxx = stft(s, fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap)
            mags.append(np.abs(Zxx))
        return torch.tensor(np.stack(mags), dtype=torch.float32).unsqueeze(1).to(self.device)

    # ──────────────────────────────────────────────────
    #  Metrics
    # ──────────────────────────────────────────────────

    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        return {
            'MSE':  MeanSquaredError.calculate(y_true, y_pred),
            'MAE':  MeanAbsoluteError.calculate(y_true, y_pred),
            'RMSE': RootMeanSquaredError.calculate(y_true, y_pred),
            'SNR':  SignalToNoiseRatio.calculate(y_true, y_pred),
        }

    def _evaluate(self, loader: DataLoader, loss_fn: nn.Module):
        self.model.eval()
        total_loss = 0
        all_true, all_pred = [], []
        with torch.no_grad():
            for noisy, clean in loader:
                x4 = self._batch_to_4ch(noisy.cpu().numpy())
                clean_mag = self._signal_to_mag(clean.cpu().numpy())
                out_mask = self.model(x4)
                out = out_mask * x4[:, 0:1, :, :]
                loss = loss_fn(out, clean_mag)
                total_loss += loss.item()
                all_true.append(clean_mag.cpu().numpy())
                all_pred.append(out.cpu().numpy())

        metrics = self._compute_metrics(np.concatenate(all_true), np.concatenate(all_pred))
        return total_loss / len(loader), metrics

    def _denoise_batch(self, signal_batch: np.ndarray) -> np.ndarray:
        """STFT → mask → ISTFT → часова область."""
        phases = [
            np.angle(stft(s, fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap)[2])
            for s in signal_batch
        ]
        x4 = self._batch_to_4ch(signal_batch)
        noisy_mag_ch = x4[:, 0, :, :].cpu().numpy()
        with torch.no_grad():
            out_mag = (self.model(x4).squeeze(1).cpu().numpy() * noisy_mag_ch)

        rec = []
        for mag, phase in zip(out_mag, phases):
            _, r = istft(mag * np.exp(1j * phase), fs=self.fs,
                         nperseg=self.nperseg, noverlap=self.noverlap)
            r = r[:self.signal_len] if len(r) >= self.signal_len \
                else np.pad(r, (0, self.signal_len - len(r)))
            rec.append(r)
        return np.stack(rec)

    # ──────────────────────────────────────────────────
    #  Training loop
    # ──────────────────────────────────────────────────

    def train(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        best_val_loss = float('inf')
        best_weights = None

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0
            tr_true, tr_pred = [], []

            for noisy, clean in self.train_loader:
                x4 = self._batch_to_4ch(noisy.cpu().numpy())
                clean_mag = self._signal_to_mag(clean.cpu().numpy())
                out_mask = self.model(x4)
                out = out_mask * x4[:, 0:1, :, :]
                loss = loss_fn(out, clean_mag)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                tr_true.append(clean_mag.cpu().numpy())
                tr_pred.append(out.detach().cpu().numpy())

            tr_metrics = self._compute_metrics(np.concatenate(tr_true), np.concatenate(tr_pred))
            val_loss, val_metrics = self._evaluate(self.val_loader, loss_fn)

            if WANDB_OK and hasattr(wandb, 'run') and wandb.run:
                wandb.log({
                    'train_loss': total_loss / len(self.train_loader),
                    'val_loss': val_loss,
                    **{f'train_{k.lower()}': v for k, v in tr_metrics.items()},
                    **{f'val_{k.lower()}': v   for k, v in val_metrics.items()},
                }, step=epoch)

            print(f"Epoch {epoch:02d}/{self.epochs} | "
                  f"train={total_loss / len(self.train_loader):.5f} | "
                  f"val={val_loss:.5f} | "
                  f"val_MSE={val_metrics['MSE']:.4f} | "
                  f"val_SNR={val_metrics['SNR']:.2f} dB")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = {k: v.clone() for k, v in self.model.state_dict().items()}

        weights_dir = self.dataset_path / "weights"
        weights_dir.mkdir(exist_ok=True)

        model_path = weights_dir / f"HybridDSGE_UNet_{self.noise_type}_S{self.dsge_order}_best.pth"
        torch.save(best_weights, model_path)
        print(f"✅ Best model saved → {model_path}")

        dsge_path = weights_dir / f"dsge_state_{self.noise_type}_S{self.dsge_order}.npz"
        self.dsge.save_state(str(dsge_path))

        self.model.load_state_dict(best_weights)
        self._final_test_eval()

        if WANDB_OK and hasattr(wandb, 'run') and wandb.run:
            wandb.finish()

    def _final_test_eval(self):
        self.model.eval()
        all_true, all_pred = [], []
        for noisy, clean in self.test_loader:
            all_pred.append(self._denoise_batch(noisy.cpu().numpy()))
            all_true.append(clean.cpu().numpy())

        metrics = self._compute_metrics(np.concatenate(all_true), np.concatenate(all_pred))
        if WANDB_OK and hasattr(wandb, 'run') and wandb.run:
            wandb.log({f'test_{k.lower()}': v for k, v in metrics.items()})

        print('\n📊 Final Test Metrics (time domain):')
        for name, val in metrics.items():
            print(f"  {name}: {val:.2f} dB" if name == 'SNR' else f"  {name}: {val:.6f}")


# ──────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Train HybridDSGE_UNet')
    p.add_argument('--dataset', required=True,
                   help='Path to dataset folder (e.g. data_generation/datasets/<name>)')
    p.add_argument('--noise-type',    default='non_gaussian', choices=['gaussian', 'non_gaussian'])
    p.add_argument('--epochs',        type=int,   default=30)
    p.add_argument('--batch-size',    type=int,   default=32)
    p.add_argument('--lr',            type=float, default=1e-4)
    p.add_argument('--dsge-order',    type=int,   default=3)
    p.add_argument('--dsge-basis',    type=str,   default='fractional',
                   choices=['fractional', 'polynomial', 'trigonometric', 'robust'])
    p.add_argument('--dsge-powers',   type=float, nargs='+', default=None)
    p.add_argument('--lambda',        type=float, default=0.01, dest='tikhonov_lambda')
    p.add_argument('--nperseg',       type=int,   default=32)
    p.add_argument('--seed',          type=int,   default=42)
    p.add_argument('--wandb-project', default='')
    args = p.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path

    with open(dataset_path / 'dataset_config.json') as f:
        cfg = json.load(f)

    signal_len = cfg['block_size']
    fs         = cfg['sample_rate']
    noverlap   = args.nperseg // 2

    print(f"Dataset: {dataset_path.name}")
    print(f"Config:  block_size={signal_len}, sample_rate={fs}, noise_type={args.noise_type}")

    trainer = HybridUnetTrainer(
        dataset_path=dataset_path,
        noise_type=args.noise_type,
        dsge_order=args.dsge_order,
        dsge_basis=args.dsge_basis,
        dsge_powers=args.dsge_powers,
        tikhonov_lambda=args.tikhonov_lambda,
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
