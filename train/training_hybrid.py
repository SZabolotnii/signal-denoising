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
from train.snr_curve import evaluate_per_snr, print_snr_table, plot_snr_curve, log_snr_curve_wandb, save_training_curves


def _model_name(dsge_basis: str, dsge_order: int) -> str:
    return f"HybridDSGE_UNet_{dsge_basis}_S{dsge_order}"


class HybridUnetTrainer:
    """
    Trainer for HybridDSGE_UNet.

    Pipeline:
      1. Load clean/noisy from dataset_path/train/
      2. DSGEFeatureExtractor.fit() on training data only (no data leakage)
      3. 4-channel preprocessing: [STFT(x̃), STFT(φ₁), STFT(φ₂), STFT(φ₃)]
         with per-channel DSGE normalisation
      4. Train with HuberLoss (robust to impulsive noise)
      5. Best model selected by max(val_SNR)
      6. Save weights + DSGE state + per-SNR curves
    """

    def __init__(
        self,
        dataset_path: Path,
        noise_type: str = 'non_gaussian',
        dsge_order: int = 3,
        dsge_basis: str = 'fractional',
        dsge_powers: list | None = None,
        tikhonov_lambda: float = 0.01,
        batch_size: int = 256,
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

        self.run_id = uuid.uuid4().hex[:8]
        self.dataset_uid = self.dataset_path.name.split('_')[-1]
        self.model_name = _model_name(dsge_basis, dsge_order)

        if WANDB_OK and wandb_project:
            run_name = f"{self.model_name}_{noise_type}_{self.dataset_uid}_{self.run_id}"
            wandb.init(project=wandb_project, name=run_name, reinit=True, config={
                'model': self.model_name, 'noise_type': noise_type,
                'dsge_order': dsge_order, 'dsge_basis': dsge_basis,
                'dsge_powers': self.dsge_powers, 'tikhonov_lambda': tikhonov_lambda,
                'epochs': epochs, 'batch_size': batch_size, 'learning_rate': learning_rate,
                'random_state': random_state, 'dataset': self.dataset_path.name,
            })
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

    # ── data ──────────────────────────────────────────────────────────────────

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
        val_len  = int(0.25 * total)
        test_len = int(0.25 * total)
        train_len = total - val_len - test_len
        g = torch.Generator().manual_seed(self.random_state)
        train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len], generator=g)

        train_clean = clean[train_set.indices]
        train_noisy = noisy[train_set.indices]

        return (
            DataLoader(train_set, batch_size=self.batch_size, shuffle=True),
            DataLoader(val_set,   batch_size=self.batch_size),
            DataLoader(test_set,  batch_size=self.batch_size),
            input_shape,
            train_clean,
            train_noisy,
        )

    # ── preprocessing ─────────────────────────────────────────────────────────

    def _batch_to_4ch(self, signal_batch: np.ndarray) -> torch.Tensor:
        """[N, T] → (B, 1+S, F, T') with normalised DSGE channels."""
        stft_mags = []
        dsge_mags_list = [[] for _ in range(self.dsge_order)]
        for s in signal_batch:
            _, _, Zxx = stft(s, fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap)
            stft_mags.append(np.abs(Zxx))
            dsge_specs = self.dsge.compute_dsge_spectrograms(s)  # [S, F, T']
            for i in range(self.dsge_order):
                dsge_mags_list[i].append(dsge_specs[i])

        stft_stack = np.stack(stft_mags)          # [B, F, T']
        stft_ref_max = stft_stack.max() + 1e-8
        channels = [stft_stack]
        for i in range(self.dsge_order):
            ch = np.stack(dsge_mags_list[i])
            channels.append(ch * (stft_ref_max / (ch.max() + 1e-8)))

        return torch.tensor(
            np.stack(channels, axis=1), dtype=torch.float32
        ).to(self.device)  # [B, 1+S, F, T']

    def _signal_to_clean_mag(self, signal_batch: np.ndarray) -> torch.Tensor:
        mags = []
        for s in signal_batch:
            _, _, Zxx = stft(s, fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap)
            mags.append(np.abs(Zxx))
        return torch.tensor(np.stack(mags), dtype=torch.float32).unsqueeze(1).to(self.device)

    # ── inference ─────────────────────────────────────────────────────────────

    def denoise_numpy(self, noisy: np.ndarray) -> np.ndarray:
        """[N, T] → [N, T]"""
        return self._denoise_batch(noisy)

    def _denoise_batch(self, signal_batch: np.ndarray) -> np.ndarray:
        phases = [
            np.angle(stft(s, fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap)[2])
            for s in signal_batch
        ]
        x4 = self._batch_to_4ch(signal_batch)
        noisy_mag = x4[:, 0, :, :].cpu().numpy()
        self.model.eval()
        with torch.no_grad():
            out_mag = self.model(x4).squeeze(1).cpu().numpy() * noisy_mag

        rec = []
        for mag, phase in zip(out_mag, phases):
            _, r = istft(mag * np.exp(1j * phase), fs=self.fs,
                         nperseg=self.nperseg, noverlap=self.noverlap)
            r = r[:self.signal_len] if len(r) >= self.signal_len \
                else np.pad(r, (0, self.signal_len - len(r)))
            rec.append(r.astype(np.float32))
        return np.stack(rec)

    # ── validation ────────────────────────────────────────────────────────────

    def _compute_val_snr(self) -> float:
        all_true, all_pred = [], []
        for noisy, clean in self.val_loader:
            all_pred.append(self.denoise_numpy(noisy.numpy()))
            all_true.append(clean.numpy())
        return float(SignalToNoiseRatio.calculate(
            np.concatenate(all_true), np.concatenate(all_pred)
        ))

    def _evaluate_loader(self, loader: DataLoader, loss_fn: nn.Module):
        self.model.eval()
        total_loss = 0.0
        all_true, all_pred = [], []
        with torch.no_grad():
            for noisy, clean in loader:
                x4 = self._batch_to_4ch(noisy.numpy())
                clean_mag = self._signal_to_clean_mag(clean.numpy())
                out = self.model(x4) * x4[:, 0:1, :, :]
                total_loss += loss_fn(out, clean_mag).item()
                all_true.append(clean_mag.cpu().numpy())
                all_pred.append(out.cpu().numpy())
        metrics = {
            'MSE':  MeanSquaredError.calculate(np.concatenate(all_true), np.concatenate(all_pred)),
            'MAE':  MeanAbsoluteError.calculate(np.concatenate(all_true), np.concatenate(all_pred)),
            'RMSE': RootMeanSquaredError.calculate(np.concatenate(all_true), np.concatenate(all_pred)),
            'SNR':  SignalToNoiseRatio.calculate(np.concatenate(all_true), np.concatenate(all_pred)),
        }
        return total_loss / len(loader), metrics

    # ── training loop ─────────────────────────────────────────────────────────

    def train(self) -> dict:
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.HuberLoss(delta=1.0)
        best_val_snr = float('-inf')
        best_sd = None
        train_history, val_snr_history = [], []

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0.0

            for noisy, clean in self.train_loader:
                x4 = self._batch_to_4ch(noisy.numpy())
                clean_mag = self._signal_to_clean_mag(clean.numpy())
                out = self.model(x4) * x4[:, 0:1, :, :]
                loss = loss_fn(out, clean_mag)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                total_loss += loss.item()

            val_loss, val_metrics = self._evaluate_loader(self.val_loader, loss_fn)
            val_snr = self._compute_val_snr()

            if WANDB_OK and hasattr(wandb, 'run') and wandb.run:
                wandb.log({
                    'train/huber_loss': total_loss / len(self.train_loader),
                    'val/huber_loss':   val_loss,
                    'val/snr_db':       val_snr,
                    **{f'val/{k.lower()}': v for k, v in val_metrics.items()},
                }, step=epoch)

            print(f"Epoch {epoch:02d}/{self.epochs} | "
                  f"train={total_loss / len(self.train_loader):.5f} | "
                  f"val_loss={val_loss:.5f} | val_SNR={val_snr:.2f} dB")

            train_history.append(total_loss / len(self.train_loader))
            val_snr_history.append(val_snr)

            if val_snr > best_val_snr:
                best_val_snr = val_snr
                best_sd = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        # ── save ──────────────────────────────────────────────────────────────
        run_dir = self.dataset_path / "weights" / "runs" / f"{self.model_name}_{self.noise_type}"
        run_dir.mkdir(parents=True, exist_ok=True)

        model_path = run_dir / "model_best.pth"
        save_training_curves(
            train_history, val_snr_history,
            run_dir / "figures" / "training_curves.png",
            self.model_name, self.noise_type,
        )
        torch.save(best_sd, model_path)
        print(f"✅ Best model saved → {model_path}")

        dsge_path = run_dir / "dsge_state.npz"
        self.dsge.save_state(str(dsge_path))

        self.model.load_state_dict(best_sd)

        # ── test metrics ──────────────────────────────────────────────────────
        test_metrics = self._evaluate_test()

        # ── per-SNR curves ────────────────────────────────────────────────────
        per_snr = {}
        test_dir = self.dataset_path / "test"
        if test_dir.exists():
            per_snr = evaluate_per_snr(self.denoise_numpy, test_dir, self.noise_type)
            print_snr_table(per_snr, self.model_name)
            plot_snr_curve(
                per_snr, self.model_name,
                save_path=run_dir / "figures" / "snr_curve.png",
            )
            log_snr_curve_wandb(per_snr, self.model_name)

        if WANDB_OK and hasattr(wandb, 'run') and wandb.run:
            wandb.finish()

        return {
            'model': self.model_name, 'noise_type': self.noise_type,
            'dataset_uid': self.dataset_uid, 'run_id': self.run_id,
            'val_snr': best_val_snr, 'test_metrics': test_metrics,
            'per_snr_results': per_snr, 'weights_path': str(model_path),
            'dsge_path': str(dsge_path),
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
        print('\n📊 Final Test Metrics (time domain):')
        for name, val in metrics.items():
            print(f"  {name}: {val:.2f} dB" if name == 'SNR' else f"  {name}: {val:.6f}")
        return metrics


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Train HybridDSGE_UNet')
    p.add_argument('--dataset',       required=True)
    p.add_argument('--noise-type',    default='non_gaussian', choices=['gaussian', 'non_gaussian'])
    p.add_argument('--epochs',        type=int,   default=30)
    p.add_argument('--batch-size',    type=int,   default=256)
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

    print(f"Dataset: {dataset_path.name}")
    print(f"Config:  block_size={cfg['block_size']}, sample_rate={cfg['sample_rate']}, "
          f"noise_type={args.noise_type}")

    HybridUnetTrainer(
        dataset_path=dataset_path,
        noise_type=args.noise_type,
        dsge_order=args.dsge_order,
        dsge_basis=args.dsge_basis,
        dsge_powers=args.dsge_powers,
        tikhonov_lambda=args.tikhonov_lambda,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        signal_len=cfg['block_size'],
        fs=cfg['sample_rate'],
        nperseg=args.nperseg,
        noverlap=args.nperseg // 2,
        random_state=args.seed,
        wandb_project=args.wandb_project,
    ).train()
