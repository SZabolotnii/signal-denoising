import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from models.autoencoder_unet_v3 import UnetAutoencoder

# ---- метрики (мають працювати на numpy) ----
from metrics import MeanSquaredError, MeanAbsoluteError, RootMeanSquaredError, SignalToNoiseRatio

# ----------------- STFT/ISTFT налаштування -----------------
WINDOW = 'hann'
EPS = 1e-8


def torch_stft(x, nperseg, noverlap):
    hop = nperseg - noverlap
    win = torch.hann_window(
        nperseg,
        periodic=True,
        device=x.device,
        dtype=x.dtype,  # x тут real, тож все OK
    )
    X = torch.stft(
        x,
        n_fft=nperseg,
        hop_length=hop,
        win_length=nperseg,
        window=win,
        center=True,
        return_complex=True,
    )
    return X  # (B, F, T') complex


def torch_istft(X, nperseg, noverlap, length=None):
    """
    X: (B, F, T') комплексний тензор STFT.
    ВАЖЛИВО: вікно має бути real dtype (НЕ complex), і на тому ж девайсі.
    """
    hop = nperseg - noverlap
    win = torch.hann_window(
        nperseg,
        periodic=True,
        device=X.device,
        dtype=(X.real.dtype if X.is_complex() else X.dtype),  # <-- ключова різниця
    )
    x = torch.istft(
        X,
        n_fft=nperseg,
        hop_length=hop,
        win_length=nperseg,
        window=win,
        center=True,
        length=length,
        return_complex=False,
    )
    return x


def multi_res_stft_loss_time(x_hat, x, configs=((128, 64), (256, 128), (64, 32)),
                             alpha=1.0, beta=0.5):
    """MRSTFT лосс у часовій області (диференційований, вся математика в torch)."""
    total = 0.0
    for nperseg, noverlap in configs:
        Xh = torch_stft(x_hat, nperseg, noverlap)
        X  = torch_stft(x,    nperseg, noverlap)
        Sh = torch.abs(Xh)
        S  = torch.abs(X)

        l1 = torch.mean(torch.abs(torch.log1p(Sh) - torch.log1p(S)))

        diff = Sh - S
        num = torch.linalg.norm(diff, ord='fro', dim=(1, 2))
        den = torch.linalg.norm(S,   ord='fro', dim=(1, 2)) + 1e-8
        sc = (num / den).mean()

        total = total + alpha * l1 + beta * sc
    return total / len(configs)


class RunningStats:
    """Обчислення μ, σ для log1p(|STFT|) на всьому train set (стабільна стандартизація)."""
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0  # sum of squares of diffs

    def update(self, x):
        # x: (B,1,F,T) torch/tensor -> використовуємо .mean()
        x = x.detach()
        b = x.numel()
        val = x.mean().item()
        # для дисперсії приблизний агрегаційний метод
        var_local = x.var(unbiased=False).item()

        # оновлення через агрегування «батч-статистик»
        # перетворимо в еквівалентні суми
        old_n = self.n
        self.n += b
        if old_n == 0:
            self.mean = val
            self.M2 = var_local * b
        else:
            delta = val - self.mean
            self.mean += delta * (b / self.n)
            # M2_agg = M2_a + M2_b + delta^2 * n_a * n_b / n
            self.M2 = self.M2 + var_local * b + delta * delta * old_n * b / self.n

    def get(self):
        if self.n == 0:
            return 0.0, 1.0
        var = self.M2 / self.n
        std = math.sqrt(max(var, 1e-8))
        return self.mean, std


class UnetAutoencoderTrainer:
    def __init__(self, dataset_type="gaussian", batch_size=16, epochs=80, lr=3e-4,
                 signal_len=2144, fs=1024, nperseg=128, noverlap=64, random_state=42, device=None):
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.signal_len = signal_len
        self.fs = fs
        self.nperseg = nperseg
        # COLA для Hann: hop = nperseg//2
        self.noverlap = nperseg // 2 if noverlap != nperseg // 2 else noverlap
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Дані
        self.train_loader, self.val_loader, self.test_loader = self._load_data()

        # Форма спектра для моделі
        with torch.no_grad():
            x0 = next(iter(self.train_loader))[0][:1].to(self.device)  # (1,T)
            S0 = torch_stft(x0, self.nperseg, self.noverlap)           # (1,F,T')
            F, T_ = S0.shape[-2:]
            input_shape = (F, T_)

        self.model = UnetAutoencoder(input_shape).to(self.device)

        # Статистика для стандартизації log1p(|STFT|)
        self.mu, self.sigma = self._compute_logmag_stats(self.train_loader)

        # Оптимізатор, шедулер, кліпінг
        self.opt = optim.AdamW(self.model.parameters(), lr=self.lr, betas=(0.9, 0.99), weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=self.epochs)
        self.grad_clip = 1.0

    def _load_data(self):
        noisy = np.load(f"../dataset/{self.dataset_type}_signals.npy")
        clean = np.load(f"../dataset/clean_signals.npy")
        assert noisy.shape == clean.shape
        assert noisy.shape[1] == self.signal_len

        X = torch.tensor(noisy, dtype=torch.float32)
        Y = torch.tensor(clean, dtype=torch.float32)

        ds = TensorDataset(X, Y)
        N = len(ds)
        val_len = int(0.15 * N)
        test_len = int(0.15 * N)
        train_len = N - val_len - test_len

        g = torch.Generator().manual_seed(self.random_state)
        tr, va, te = random_split(ds, [train_len, val_len, test_len], generator=g)

        return (DataLoader(tr, batch_size=self.batch_size, shuffle=True),
                DataLoader(va, batch_size=self.batch_size),
                DataLoader(te, batch_size=self.batch_size))

    @torch.no_grad()
    def _compute_logmag_stats(self, loader, max_batches=50):
        """Фіксовані μ,σ по train для стабільності навчання."""
        stats = RunningStats()
        seen = 0
        for b, (x_noisy, _) in enumerate(loader):
            x_noisy = x_noisy.to(self.device)  # (B,T)
            S = torch_stft(x_noisy, self.nperseg, self.noverlap)  # (B,F,T')
            Mag = torch.abs(S) + EPS
            Log = torch.log1p(Mag)  # (B,F,T')
            Log = Log.unsqueeze(1)   # (B,1,F,T')
            stats.update(Log)
            seen += 1
            if seen >= max_batches:
                break
        mu, sigma = stats.get()
        print(f"[Stats] log-mag mean={mu:.4f}, std={sigma:.4f}")
        return mu, sigma

    def _forward_batch(self, x_noisy, x_clean):
        """
        Повний граф:
        - STFT (torch)
        - Вхід у модель: нормований log1p(|S_noisy|)
        - Ціль: IRM та log1p(|S_clean|)
        - Прогноз: mask
        - Відновлення часу: torch.istft(mask * |S_noisy| з фазою S_noisy)
        """
        # (B,T) → (B,F,T')
        S_noisy = torch_stft(x_noisy, self.nperseg, self.noverlap)
        S_clean = torch_stft(x_clean, self.nperseg, self.noverlap)

        Mag_noisy = torch.abs(S_noisy) + EPS
        Mag_clean = torch.abs(S_clean) + EPS

        # Вхід у модель: нормований лог-амп
        Log_noisy = torch.log1p(Mag_noisy)  # (B,F,T')
        Log_noisy = (Log_noisy - self.mu) / self.sigma
        model_in = Log_noisy.unsqueeze(1)   # (B,1,F,T')

        # IRM як таргет маски
        irm = torch.clamp(Mag_clean / (Mag_noisy + EPS), 0.0, 1.0).unsqueeze(1)  # (B,1,F,T')

        # Прогноз маски
        mask = self.model(model_in)         # (B,1,F,T'), sigmoid

        # Лінійна амплітуда після маски
        Mag_hat = mask.squeeze(1) * Mag_noisy  # (B,F,T')

        # Лог-амп магнітуд: порівняння з clean
        Log_hat = torch.log1p(Mag_hat)
        Log_clean = torch.log1p(Mag_clean)

        # Збір комплексного спектру для ISTFT: беремо фазу S_noisy
        phase = torch.angle(S_noisy)  # (B,F,T')
        S_hat = Mag_hat * torch.exp(1j * phase)
        x_hat = torch_istft(S_hat, self.nperseg, self.noverlap, length=x_noisy.shape[-1])  # (B,T)

        return {
            "mask": mask,
            "irm": irm,
            "log_hat": Log_hat,
            "log_clean": Log_clean,
            "x_hat": x_hat
        }

    def train(self):
        best_val = float("inf")
        best_state = None

        for ep in range(1, self.epochs + 1):
            self.model.train()
            run_loss = 0.0

            for x_noisy, x_clean in self.train_loader:
                x_noisy = x_noisy.to(self.device)
                x_clean = x_clean.to(self.device)

                out = self._forward_batch(x_noisy, x_clean)

                # Лоси
                l_mask = torch.mean(torch.abs(out["mask"] - out["irm"]))  # L1 по масці
                l_mag  = torch.mean(torch.abs(out["log_hat"] - out["log_clean"]))  # L1 у log-ампл.
                l_mrstft = multi_res_stft_loss_time(out["x_hat"], x_clean)

                loss = l_mask + l_mag + 0.5 * l_mrstft

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.opt.step()

                run_loss += loss.item()

            self.scheduler.step()
            val_loss = self._validate()
            print(f"Epoch {ep:03d} | train={run_loss/len(self.train_loader):.5f} | val={val_loss:.5f}")

            if val_loss < best_val:
                best_val = val_loss
                best_state = self.model.state_dict()

        path = f"../weights/UnetAutoencoder_{self.dataset_type}_best.pth"
        torch.save(best_state, path)
        print(f"✅ Best model saved to {path}")
        self.model.load_state_dict(best_state)

        # фінальні метрики на тесті
        self.evaluate_metrics(self.test_loader)

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        total = 0.0
        for x_noisy, x_clean in self.val_loader:
            x_noisy = x_noisy.to(self.device)
            x_clean = x_clean.to(self.device)
            out = self._forward_batch(x_noisy, x_clean)
            l_mask = torch.mean(torch.abs(out["mask"] - out["irm"]))
            l_mag  = torch.mean(torch.abs(out["log_hat"] - out["log_clean"]))
            l_mrstft = multi_res_stft_loss_time(out["x_hat"], x_clean)
            loss = l_mask + l_mag + 0.5 * l_mrstft
            total += loss.item()
        return total / len(self.val_loader)

    @torch.no_grad()
    def denoise_batch(self, x_noisy):
        """Інференс: (B,T) → (B,T) часова область."""
        self.model.eval()
        x_noisy = x_noisy.to(self.device)
        S_noisy = torch_stft(x_noisy, self.nperseg, self.noverlap)
        Mag_noisy = torch.abs(S_noisy) + EPS
        Log_noisy = torch.log1p(Mag_noisy)
        Log_noisy = (Log_noisy - self.mu) / self.sigma
        inp = Log_noisy.unsqueeze(1)
        mask = self.model(inp).squeeze(1)
        Mag_hat = mask * Mag_noisy
        phase = torch.angle(S_noisy)
        S_hat = Mag_hat * torch.exp(1j * phase)
        x_hat = torch_istft(S_hat, self.nperseg, self.noverlap, length=x_noisy.shape[-1])
        return x_hat.cpu().numpy()

    @torch.no_grad()
    def evaluate_metrics(self, loader):
        all_true, all_pred = [], []
        for x_noisy, x_clean in loader:
            pred = self.denoise_batch(x_noisy)  # (B,T) numpy
            all_pred.append(pred)
            all_true.append(x_clean.numpy())

        y_true = np.concatenate(all_true, axis=0)
        y_pred = np.concatenate(all_pred, axis=0)

        print("\n=== Test-set metrics (mean) ===")
        print(f"MSE : {MeanSquaredError.calculate(y_true, y_pred):.6f}")
        print(f"MAE : {MeanAbsoluteError.calculate(y_true, y_pred):.6f}")
        print(f"RMSE: {RootMeanSquaredError.calculate(y_true, y_pred):.6f}")
        print(f"SNR : {SignalToNoiseRatio.calculate(y_true, y_pred):.2f} dB")


if __name__ == "__main__":
    trainer = UnetAutoencoderTrainer(
        dataset_type="non_gaussian",  # або "gaussian"
        batch_size=16,
        epochs=50,
        lr=3e-4,
        signal_len=2144,
        fs=1024,
        nperseg=128,
        noverlap=64,   # буде примусово nperseg//2
        random_state=42
    )
    trainer.train()
