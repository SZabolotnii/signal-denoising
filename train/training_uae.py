import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from scipy.signal import stft, istft
import uuid

try:
    import wandb

    WANDB_OK = True
except Exception:
    WANDB_OK = False

from models.autoencoder_unet import UnetAutoencoder
from metrics import MeanSquaredError, MeanAbsoluteError, RootMeanSquaredError, SignalToNoiseRatio

# ---------- глобальні параметри STFT (узгоджені з інференсом/візуалізацією) ----------
WINDOW = 'hann'
EPS = 1e-12


def stft_mag_phase(x, fs, nperseg, noverlap, pad):
    """STFT із reflect-pad; повертає (mag, phase)."""
    x_pad = np.pad(x, pad, mode="reflect")
    _, _, Zxx = stft(x_pad, fs=fs, nperseg=nperseg, noverlap=noverlap, window=WINDOW, boundary=None, padded=False)
    return np.abs(Zxx).astype(np.float32), np.angle(Zxx).astype(np.float32)


def istft_from_mag_phase(mag, phase, fs, nperseg, noverlap, pad, target_len):
    # mag, phase: (F, T')
    _, rec = istft(mag * np.exp(1j * phase),
                   fs=fs, nperseg=nperseg, noverlap=noverlap,
                   window='hann', input_onesided=True, boundary=None)
    rec = rec[pad: pad + target_len]
    if len(rec) < target_len:
        rec = np.pad(rec, (0, target_len - len(rec)))
    elif len(rec) > target_len:
        rec = rec[:target_len]
    return rec.astype(np.float32)


# ==== FIX 2: Torch STFT-модуль для лоссу ====
def stft_mag_torch(x: torch.Tensor, nperseg: int, noverlap: int) -> torch.Tensor:
    """
    x: (B, T) -> |STFT|(B, F, T')
    Використовує Hann і hop = nperseg - noverlap.
    """
    hop = nperseg - noverlap
    window = torch.hann_window(nperseg, periodic=True, device=x.device, dtype=x.dtype)
    X = torch.stft(x, n_fft=nperseg, hop_length=hop, win_length=nperseg,
                   window=window, center=True, return_complex=True)
    return torch.abs(X)  # (B, F, T')


# ---------- multi-resolution STFT loss (на часовому сигналі) ----------
def spectral_convergence(S_hat: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """
    S_hat, S: (B, F, T) — лінійні амплітуди спектрів.
    Повертає середній по батчу SC.
    """
    diff = S_hat - S  # (B, F, T)
    # Frobenius-норма по (F, T) для кожного елемента батча:
    num = torch.linalg.norm(diff, ord='fro', dim=(1, 2))  # (B,)
    den = torch.linalg.norm(S, ord='fro', dim=(1, 2)) + 1e-12  # (B,)
    return (num / den).mean()


def stft_mag_db_torch(x, fs, nperseg, noverlap):
    """STFT у Torch для батча (B,T) → (B, F, T'), повертає лінійну амплітуду."""
    # Для простоти: використовуємо torch.stft (комплексний вихід, потрібно PyTorch>=1.8)
    # Щоб мати той самий hop, беремо hop_length = nperseg - noverlap
    hop = nperseg - noverlap
    window = torch.hann_window(nperseg, periodic=True, device=x.device, dtype=x.dtype)
    # torch.stft повертає (B, F, T', 2) якщо return_complex=False
    X = torch.stft(x, n_fft=nperseg, hop_length=hop, win_length=nperseg,
                   window=window, center=True, return_complex=True)
    mag = torch.abs(X)  # (B,F,T')
    return mag


def multi_res_stft_loss(x_hat: torch.Tensor, x: torch.Tensor,
                        configs=((128, 64), (256, 128), (64, 32)),
                        alpha=1.0, beta=0.5) -> torch.Tensor:
    """
    ==== FIX 3: конфіги з COLA для Hann ====
    Використовує три масштаби з noverlap = nperseg//2, щоб уникнути NOLA-попереджень.
    """
    total = 0.0
    for nperseg, noverlap in configs:
        S_hat = stft_mag_torch(x_hat, nperseg, noverlap)  # (B, F, T')
        S = stft_mag_torch(x, nperseg, noverlap)
        l1 = torch.mean(torch.abs(torch.log1p(S_hat) - torch.log1p(S)))
        sc = spectral_convergence(S_hat, S)
        total = total + alpha * l1 + beta * sc
    return total / len(configs)


# ---------- тренер ----------
class UnetAutoencoderTrainer:
    def __init__(self, dataset_type="gaussian", batch_size=32, epochs=30, learning_rate=1e-4,
                 signal_len=2144, fs=1024, nperseg=128, noverlap=96, random_state=42,
                 wandb_project="signal-denoising", device=None):
        self.dataset_type = dataset_type
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
        # ==== FIX 4: якщо Hann — примусово COLA (noverlap = nperseg // 2) ====
        if self.noverlap != self.nperseg // 2:
            print(f"[INFO] Adjusting noverlap from {self.noverlap} to {self.nperseg // 2} "
                  f"for Hann COLA consistency.")
            self.noverlap = self.nperseg // 2

        # WANDB (опційно)
        if WANDB_OK:
            run_name = f"MaskUNet_{dataset_type}_{uuid.uuid4().hex[:8]}"
            wandb.init(project=wandb_project, name=run_name, config={
                "model": "MaskUNet",
                "dataset": dataset_type,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "random_state": random_state,
                "fs": fs, "nperseg": nperseg, "noverlap": noverlap
            })

        self.train_loader, self.val_loader, self.test_loader, self.input_shape = self._load_data()
        self.model = UnetAutoencoder(self.input_shape).to(self.device)

    def _load_data(self):
        noisy = np.load(f"../dataset/{self.dataset_type}_signals.npy")
        clean = np.load("../dataset/clean_signals.npy")
        assert noisy.shape[1] == self.signal_len and clean.shape[1] == self.signal_len, "Signal length mismatch"

        # Отримаємо форму спектрограми для моделі (F,T) на базових параметрах STFT
        mag0, _ = stft_mag_phase(clean[0], fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap, pad=self.pad)
        input_shape = mag0.shape  # (F,T)

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

    # --------- основний train-ітератор ---------
    def train(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        best_val = float("inf")
        best_sd = None

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0.0

            for noisy, clean in self.train_loader:
                noisy = noisy.to(self.device)  # (B,T)
                clean = clean.to(self.device)  # (B,T)

                # STFT (на CPU, але можна оптимізувати під Torch dsp)
                noisy_np = noisy.cpu().numpy()
                clean_np = clean.cpu().numpy()

                # Отримуємо магнітуди та фази з reflect-pad — БАЗОВА роздільність
                noisy_mag_list, clean_mag_list, phase_list = [], [], []
                for xn, xc in zip(noisy_np, clean_np):
                    nm, np_phase = stft_mag_phase(xn, self.fs, self.nperseg, self.noverlap, self.pad)
                    cm, _ = stft_mag_phase(xc, self.fs, self.nperseg, self.noverlap, self.pad)
                    noisy_mag_list.append(nm)
                    clean_mag_list.append(cm)
                    phase_list.append(np_phase)

                noisy_mag = torch.tensor(np.stack(noisy_mag_list), dtype=torch.float32, device=self.device).unsqueeze(
                    1)  # (B,1,F,T)
                clean_mag = torch.tensor(np.stack(clean_mag_list), dtype=torch.float32, device=self.device).unsqueeze(
                    1)  # (B,1,F,T)

                # ---- MASK PREDICTION ----  # CHANGED
                mask = self.model(noisy_mag)  # (B,1,F,T) ∈ [0,1]
                out_mag = mask * noisy_mag  # застосовуємо маску до вхідної амплітуди

                # ---- Base loss у log-амплітуді ----  # CHANGED
                base_loss = torch.mean(torch.abs(torch.log1p(out_mag) - torch.log1p(clean_mag)))

                # ---- Реконструюємо часовий сигнал і даємо multi-res STFT loss ----  # NEW
                out_mag_np = out_mag.squeeze(1).detach().cpu().numpy()
                rec_list = []
                for om, ph in zip(out_mag_np, phase_list):
                    rec = istft_from_mag_phase(om, ph, self.fs, self.nperseg, self.noverlap, self.pad, self.signal_len)
                    rec_list.append(rec)
                rec_batch = torch.tensor(np.stack(rec_list), device=self.device)  # (B,T)

                # Multi-resolution STFT loss на часовому сигналі
                mr_loss = multi_res_stft_loss(
                    x_hat=rec_batch, x=clean,
                    configs=((128, 64), (256, 128), (64, 32)), alpha=1.0, beta=0.5
                )

                loss = base_loss + 0.5 * mr_loss  # ваги можна тюнити  # CHANGED

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.item())

            # Валідація
            val_loss = self._validate()

            if WANDB_OK:
                wandb.log({"train_loss": epoch_loss / len(self.train_loader),
                           "val_loss": val_loss}, step=epoch)

            print(f"Epoch {epoch:02d} | train_loss={epoch_loss / len(self.train_loader):.5f} | val_loss={val_loss:.5f}")

            if val_loss < best_val:
                best_val = val_loss
                best_sd = self.model.state_dict()

        # Збереження кращої моделі
        save_path = f"../weights/UnetAutoencoder_{self.dataset_type}_best.pth"
        torch.save(best_sd, save_path)
        print(f"✅ Best model saved to: {save_path}")
        self.model.load_state_dict(best_sd)

        # Фінальні метрики на тесті (часова область)
        self.evaluate_metrics(self.test_loader)

    def _validate(self):
        self.model.eval()
        total = 0.0
        with torch.no_grad():
            for noisy, clean in self.val_loader:
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)

                # базова STFT
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

                # реконструкція та multi-res
                out_mag_np = out_mag.squeeze(1).cpu().numpy()
                rec_list = []
                for om, ph in zip(out_mag_np, phase_list):
                    rec = istft_from_mag_phase(om, ph, self.fs, self.nperseg, self.noverlap, self.pad, self.signal_len)
                    rec_list.append(rec)
                rec_batch = torch.tensor(np.stack(rec_list), device=self.device)

                mr_loss = multi_res_stft_loss(
                    x_hat=rec_batch, x=clean,
                    configs=((128, 64), (256, 128), (64, 32)), alpha=1.0, beta=0.5
                )

                loss = base_loss + 0.5 * mr_loss
                total += float(loss.item())

        return total / len(self.val_loader)

    # --------- оцінка в часовій області ---------
    def denoise_batch(self, noisy_batch):
        self.model.eval()
        noisy_np = noisy_batch.cpu().numpy()
        out_rec = []
        with torch.no_grad():
            for xn in noisy_np:
                nm, ph = stft_mag_phase(xn, self.fs, self.nperseg, self.noverlap, self.pad)
                nm_t = torch.tensor(nm, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
                mask = self.model(nm_t)  # (1,1,F,T)
                om = (mask.squeeze().cpu().numpy() * nm)  # застосовуємо маску
                rec = istft_from_mag_phase(om, ph, self.fs, self.nperseg, self.noverlap, self.pad, self.signal_len)
                out_rec.append(rec)
        return np.stack(out_rec)

    def evaluate_metrics(self, loader):
        all_true, all_pred = [], []
        with torch.no_grad():
            for noisy, clean in loader:
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)
                pred = self.denoise_batch(noisy)  # (B,T) numpy
                all_pred.append(pred)
                all_true.append(clean.cpu().numpy())
        y_true = np.concatenate(all_true, axis=0)
        y_pred = np.concatenate(all_pred, axis=0)

        metrics = {
            "MSE": MeanSquaredError.calculate(y_true, y_pred),
            "MAE": MeanAbsoluteError.calculate(y_true, y_pred),
            "RMSE": RootMeanSquaredError.calculate(y_true, y_pred),
            "SNR": SignalToNoiseRatio.calculate(y_true, y_pred),
        }
        if WANDB_OK:
            wandb.log({f"test_{k.lower()}": v for k, v in metrics.items()})
        print("\n📊 Final Test Metrics (time domain):")
        for k, v in metrics.items():
            print(f"{k}: {v:.6f}" if k != "SNR" else f"{k}: {v:.2f} dB")


# ---------- запуск ----------
if __name__ == "__main__":
    trainer = UnetAutoencoderTrainer(
        dataset_type="gaussian",  # або "gaussian"
        batch_size=16,
        epochs=80,
        learning_rate=1e-4,
        signal_len=2144,
        fs=1024,
        nperseg=128,
        noverlap=96,
        random_state=42,
        wandb_project="signal-denoising"
    )
    trainer.train()
