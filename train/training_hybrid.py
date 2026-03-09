"""
Тренування гібридної моделі HybridDSGE_UNet.

Пайплайн (порівняно з training_uae.py):
  1. Завантаження clean/noisy датасетів.
  2. DSGEFeatureExtractor.fit() ТІЛЬКИ на тренувальних даних.
  3. 4-канальний препроцесинг: [STFT(x̃), STFT(φ₁), STFT(φ₂), STFT(φ₃)].
  4. Тренування HybridDSGE_UNet (MSE, Adam, lr=1e-4, 30 епох).
  5. Wandb-логування + збереження ваг і стану DSGE.

Запуск:
    cd train/
    python training_hybrid.py               # non_gaussian, 30 epochs
    python training_hybrid.py --dataset gaussian --epochs 50 --no-wandb
"""

import argparse
import os
import sys
import uuid

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import stft, istft
from torch.utils.data import DataLoader, TensorDataset, random_split

# Додаємо корінь проєкту в sys.path (для запуску з будь-якої директорії)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.hybrid_unet import HybridDSGE_UNet
from models.dsge_layer import DSGEFeatureExtractor
from metrics import MeanSquaredError, MeanAbsoluteError, RootMeanSquaredError, SignalToNoiseRatio


# ──────────────────────────────────────────────────────
#  Trainer
# ──────────────────────────────────────────────────────

class HybridUnetTrainer:
    """
    Тренер гібридної DSGE + U-Net моделі.

    Parameters
    ----------
    dataset_type : str          'gaussian' або 'non_gaussian'
    dsge_order : int            порядок апроксимації S (кількість DSGE-каналів)
    dsge_basis : str            тип базису ('fractional', 'polynomial', 'trigonometric', 'robust')
    dsge_powers : list[float]   степені для базису (для fractional: [0.5, 1.5, 2.0])
    tikhonov_lambda : float     регуляризація Тихонова
    batch_size, epochs, lr      гіперпараметри тренування
    use_wandb : bool            вмикає/вимикає wandb
    """

    def __init__(
        self,
        dataset_type: str = 'non_gaussian',
        dsge_order: int = 3,
        dsge_basis: str = 'fractional',
        dsge_powers: list | None = None,
        tikhonov_lambda: float = 0.01,
        batch_size: int = 32,
        epochs: int = 30,
        learning_rate: float = 1e-4,
        signal_len: int = 2144,
        fs: int = 1024,
        nperseg: int = 128,
        noverlap: int = 96,
        random_state: int = 42,
        wandb_project: str = 'signal-denoising',
        use_wandb: bool = True,
        device: str | None = None,
    ):
        self.dataset_type = dataset_type
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
        self.use_wandb = use_wandb
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Шляхи
        self._root = os.path.join(os.path.dirname(__file__), '..')
        self._weights_dir = os.path.join(self._root, 'weights')
        os.makedirs(self._weights_dir, exist_ok=True)

        # Wandb
        if self.use_wandb:
            import wandb
            run_name = f"HybridDSGE_{dataset_type}_S{dsge_order}_{uuid.uuid4().hex[:6]}"
            wandb.init(
                project=wandb_project,
                name=run_name,
                config={
                    'model': 'HybridDSGE_UNet',
                    'dataset': dataset_type,
                    'dsge_order': dsge_order,
                    'dsge_basis': dsge_basis,
                    'dsge_powers': dsge_powers,
                    'tikhonov_lambda': tikhonov_lambda,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'random_state': random_state,
                },
            )
            self.wandb = wandb
        else:
            self.wandb = None

        # Завантаження даних і DSGE-fit
        self.train_loader, self.val_loader, self.test_loader, self.input_shape, \
            self.train_clean, self.train_noisy = self._load_data()

        print(f"[Info] Fitting DSGEFeatureExtractor on {len(self.train_clean)} train samples…")
        self.dsge = DSGEFeatureExtractor(
            basis_type=dsge_basis,
            powers=dsge_powers or [0.5, 1.5, 2.0],
            tikhonov_lambda=tikhonov_lambda,
            stft_params={'nperseg': nperseg, 'noverlap': noverlap, 'fs': fs},
        )
        self.dsge.fit(self.train_clean, self.train_noisy)
        self.dsge.check_generating_element_norm()
        print(f"[Info] DSGE ready: {self.dsge}")

        # Модель
        self.model = HybridDSGE_UNet(
            input_shape=self.input_shape,
            dsge_order=dsge_order,
        ).to(self.device)
        print(f"[Info] Model params: {self.model.param_count():,}")

    # ──────────────────────────────────────────────────
    #  Завантаження і спліт даних
    # ──────────────────────────────────────────────────

    def _load_data(self):
        noisy = np.load(os.path.join(self._root, 'dataset', f'{self.dataset_type}_signals.npy'))
        clean = np.load(os.path.join(self._root, 'dataset', 'clean_signals.npy'))

        assert noisy.shape[1] == self.signal_len, "Signal length mismatch"
        assert clean.shape[1] == self.signal_len, "Signal length mismatch"

        # Форма спектрограми
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
        train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len], generator=g)

        # Зберігаємо тренувальні clean/noisy для fit()
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
    #  Препроцесинг: 4-канальний вхід
    # ──────────────────────────────────────────────────

    def _batch_to_4ch(self, signal_batch: np.ndarray) -> torch.Tensor:
        """
        Перетворює batch сигналів у 4-канальний тензор:
          канал 0:   |STFT(x̃)|              (не змінюється)
          канали 1-S: |STFT(φᵢ(x̃))| / scale  (нормалізовані до масштабу STFT)

        Нормалізація DSGE-каналів критична: без неї великі значення φ₃(x)=sign(x)|x|²
        (до 1.37) проти STFT (до 0.69) призводять до колапсу в тривіальний нуль-маску.

        Returns
        -------
        tensor : torch.Tensor, shape (B, 1+S, F, T')
        """
        stft_mags = []
        dsge_mags_list = [[] for _ in range(self.dsge_order)]

        for s in signal_batch:
            # Основна STFT-спектрограма
            _, _, Zxx = stft(s, fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap)
            stft_mags.append(np.abs(Zxx))

            # DSGE-спектрограми
            dsge_specs = self.dsge.compute_dsge_spectrograms(s)  # [S, F, T']
            for i in range(self.dsge_order):
                dsge_mags_list[i].append(dsge_specs[i])

        # Stack по batch: [B, F, T']
        stft_stack = np.stack(stft_mags)  # [B, F, T']
        all_channels = [stft_stack]

        # Нормалізуємо кожен DSGE-канал до масштабу STFT-каналу:
        # ділимо на max усього батчу (стійко до нуля)
        stft_ref_max = stft_stack.max() + 1e-8
        for i in range(self.dsge_order):
            dsge_ch = np.stack(dsge_mags_list[i])  # [B, F, T']
            dsge_max = dsge_ch.max() + 1e-8
            # Приводимо DSGE до того ж абсолютного масштабу, що й STFT
            dsge_normalized = dsge_ch * (stft_ref_max / dsge_max)
            all_channels.append(dsge_normalized)

        tensor = torch.tensor(
            np.stack(all_channels, axis=1),  # [B, 1+S, F, T']
            dtype=torch.float32,
        ).to(self.device)
        return tensor

    def _signal_to_mag(self, signal_batch: np.ndarray) -> torch.Tensor:
        """1-канальна версія для обчислення таргет (чистий сигнал)."""
        mags = []
        for s in signal_batch:
            _, _, Zxx = stft(s, fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap)
            mags.append(np.abs(Zxx))
        return torch.tensor(np.stack(mags), dtype=torch.float32).unsqueeze(1).to(self.device)

    # ──────────────────────────────────────────────────
    #  Метрики
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
                noisy_np = noisy.cpu().numpy()
                clean_np = clean.cpu().numpy()

                x4 = self._batch_to_4ch(noisy_np)          # (B, 4, F, T')
                clean_mag = self._signal_to_mag(clean_np)  # (B, 1, F, T')

                out_mask = self.model(x4)
                # Apply mask to the original noisy STFT magnitude (channel 0)
                # x4[:, 0:1, :, :] contains the |STFT(x̃)|
                noisy_mag_ch = x4[:, 0:1, :, :]
                out = out_mask * noisy_mag_ch
                
                loss = loss_fn(out, clean_mag)
                total_loss += loss.item()
                all_true.append(clean_mag.cpu().numpy())
                all_pred.append(out.cpu().numpy())

        metrics = self._compute_metrics(
            np.concatenate(all_true), np.concatenate(all_pred)
        )
        return total_loss / len(loader), metrics

    # ──────────────────────────────────────────────────
    #  Повний денойзинг у часовій області (для фінального тесту)
    # ──────────────────────────────────────────────────

    def _denoise_batch(self, signal_batch: np.ndarray) -> np.ndarray:
        """STFT → mask → ISTFT → часова область."""
        phases = []
        mags_np = []
        for s in signal_batch:
            _, _, Zxx = stft(s, fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap)
            mags_np.append(np.abs(Zxx))
            phases.append(np.angle(Zxx))

        x4 = self._batch_to_4ch(signal_batch)  # (B, 4, F, T')
        noisy_mag_ch = x4[:, 0, :, :].cpu().numpy()  # (B, F, T')
        
        with torch.no_grad():
            out_mask = self.model(x4).squeeze(1).cpu().numpy()  # (B, F, T')
            out_mag = out_mask * noisy_mag_ch  # Apply mask!

        rec = []
        for mag, phase in zip(out_mag, phases):
            Zxx_d = mag * np.exp(1j * phase)
            _, r = istft(Zxx_d, fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap)
            r = r[:self.signal_len] if len(r) >= self.signal_len else np.pad(r, (0, self.signal_len - len(r)))
            rec.append(r)
        return np.stack(rec)

    # ──────────────────────────────────────────────────
    #  Тренування
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
                noisy_np = noisy.cpu().numpy()
                clean_np = clean.cpu().numpy()

                x4 = self._batch_to_4ch(noisy_np)
                noisy_mag_ch = x4[:, 0:1, :, :] # Channel 0 is STFT
                clean_mag = self._signal_to_mag(clean_np)

                out_mask = self.model(x4)
                # Ratio mask logic: Model predicts 0..1 mask, we multiply by input magnitude
                out = out_mask * noisy_mag_ch
                
                loss = loss_fn(out, clean_mag)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                tr_true.append(clean_mag.cpu().numpy())
                tr_pred.append(out.detach().cpu().numpy())

            tr_metrics = self._compute_metrics(np.concatenate(tr_true), np.concatenate(tr_pred))
            val_loss, val_metrics = self._evaluate(self.val_loader, loss_fn)

            log = {
                'epoch': epoch,
                'train_loss': total_loss / len(self.train_loader),
                'val_loss': val_loss,
                **{f'train_{k.lower()}': v for k, v in tr_metrics.items()},
                **{f'val_{k.lower()}': v   for k, v in val_metrics.items()},
            }
            if self.wandb:
                self.wandb.log(log, step=epoch)

            print(
                f"Epoch {epoch:02d}/{self.epochs} | "
                f"Train: {log['train_loss']:.6f} | "
                f"Val: {val_loss:.6f} | "
                f"Val MSE: {val_metrics['MSE']:.4f} | "
                f"Val SNR: {val_metrics['SNR']:.2f} dB"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = {k: v.clone() for k, v in self.model.state_dict().items()}

        # Збереження
        model_path = os.path.join(
            self._weights_dir,
            f'HybridDSGE_UNet_{self.dataset_type}_S{self.dsge_order}_best.pth',
        )
        torch.save(best_weights, model_path)
        print(f'✅ Best model saved → {model_path}')

        dsge_path = os.path.join(
            self._weights_dir,
            f'dsge_state_{self.dataset_type}_S{self.dsge_order}.npz',
        )
        self.dsge.save_state(dsge_path)

        # Фінальна оцінка на тесті
        self.model.load_state_dict(best_weights)
        self._final_test_eval()

    # ──────────────────────────────────────────────────
    #  Фінальна оцінка (часова область)
    # ──────────────────────────────────────────────────

    def _final_test_eval(self):
        self.model.eval()
        all_true, all_pred = [], []
        for noisy, clean in self.test_loader:
            denoised = self._denoise_batch(noisy.cpu().numpy())
            all_pred.append(denoised)
            all_true.append(clean.cpu().numpy())

        y_true = np.concatenate(all_true)
        y_pred = np.concatenate(all_pred)
        metrics = self._compute_metrics(y_true, y_pred)

        if self.wandb:
            self.wandb.log({f'test_{k.lower()}': v for k, v in metrics.items()})
            self.wandb.finish()

        print('\n📊 Final Test Metrics (time domain):')
        for name, val in metrics.items():
            unit = ' dB' if name == 'SNR' else ''
            print(f'  {name}: {val:.4f}{unit}')


# ──────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Train HybridDSGE_UNet')
    p.add_argument('--dataset',    type=str,   default='non_gaussian', choices=['gaussian', 'non_gaussian'])
    p.add_argument('--epochs',     type=int,   default=30)
    p.add_argument('--batch-size', type=int,   default=32)
    p.add_argument('--lr',         type=float, default=1e-4)
    p.add_argument('--dsge-order', type=int,   default=3)
    p.add_argument('--dsge-basis', type=str,   default='fractional',
                   choices=['fractional', 'polynomial', 'trigonometric', 'robust'])
    p.add_argument('--dsge-powers',type=float, nargs='+', default=None,
                   help='List of powers/frequencies for the basis. Space-separated.')
    p.add_argument('--lambda',     type=float, default=0.01, dest='tikhonov_lambda')
    p.add_argument('--no-wandb',   action='store_true')
    p.add_argument('--device',     type=str,   default=None)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    trainer = HybridUnetTrainer(
        dataset_type=args.dataset,
        dsge_order=args.dsge_order,
        dsge_basis=args.dsge_basis,
        dsge_powers=args.dsge_powers,
        tikhonov_lambda=args.tikhonov_lambda,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        signal_len=2144,
        fs=1024,
        nperseg=128,
        noverlap=96,
        use_wandb=not args.no_wandb,
        device=args.device,
    )
    trainer.train()
