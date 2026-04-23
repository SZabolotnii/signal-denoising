"""
DSGE Feature Extractor — реалізація розкладу в просторі з порідним елементом
(Decomposition in Space with Generating Element / Простір Кунченка).

Теорія:
    Порідний вектор X (чистий сигнал) наближається узгодженим вектором Y:
        X ≈ Y = k₀ + Σᵢ kᵢ · φᵢ(x̃)
    де φᵢ — нелінійні базисні функції від зашумлених спостережень x̃,
    а оптимальні коефіцієнти K знаходяться з KF = B (з регуляризацією Тіхонова).

Інтеграція з U-Net:
    φᵢ(x̃) → STFT → |Zxx_i| — DSGE-спектрограма для i-го каналу.
    4-канальний вхід U-Net = [STFT(x̃), STFT(φ₁(x̃)), STFT(φ₂(x̃)), STFT(φ₃(x̃))].
"""

import numpy as np
from scipy.signal import stft
from scipy.special import erf as _erf


# ──────────────────────────────────────────────────────
#  Доступні типи базисів
# ──────────────────────────────────────────────────────

def fractional_basis(x: np.ndarray, powers: list[float]) -> np.ndarray:
    """
    Дробово-степеневий базис: φᵢ(x) = sign(x) · |x|^pᵢ.
    Зберігає знак — критично для QPSK/FSK (фаза несе інформацію).
    """
    return np.array([np.sign(x) * np.abs(x) ** p for p in powers])


def polynomial_basis(x: np.ndarray, powers: list[float]) -> np.ndarray:
    """Поліноміальний базис: φᵢ(x) = x^pᵢ  (цілі степені ≥ 2, без лінійного x¹).

    DSGE decomposition requires strictly nonlinear basis functions —
    linear term φ(x) = x is excluded because it makes Z = 0 trivially.
    Powers < 2 are silently skipped with a warning.
    """
    valid = [p for p in powers if int(p) >= 2]
    if len(valid) < len(powers):
        skipped = [p for p in powers if int(p) < 2]
        import warnings
        warnings.warn(
            f"polynomial_basis: skipping linear/constant powers {skipped} — "
            f"DSGE requires strictly nonlinear basis (φ(x) ≠ x). Using {valid}."
        )
    if not valid:
        raise ValueError("polynomial_basis: no valid powers ≥ 2 provided")
    return np.array([x ** int(p) for p in valid])


def trigonometric_basis(x: np.ndarray, freqs: list[float]) -> np.ndarray:
    """Тригонометричний базис: φᵢ(x) = sin(fᵢ · x)."""
    return np.array([np.sin(f * x) for f in freqs])


# Pool of robust saturation functions (ordered by diversity).
# All suppress outliers; used in order when S > 3.
_ROBUST_POOL = [
    lambda x: np.tanh(x),                    # tanh         ∈ (-1, 1)
    lambda x: 1.0 / (1.0 + np.exp(-x)),      # sigmoid      ∈ (0, 1)
    lambda x: np.arctan(x),                   # arctan       ∈ (-π/2, π/2)
    lambda x: x / (1.0 + np.abs(x)),          # softsign     ∈ (-1, 1)
    lambda x: _erf(x),                        # error func   ∈ (-1, 1)
]


def robust_basis(x: np.ndarray, powers: list[float]) -> np.ndarray:
    """Робастний базис насичення порядку S = len(powers).

    Функції вибираються з пулу по порядку:
      S=3: [tanh, sigmoid, arctan]
      S=4: + softsign  x/(1+|x|)
      S=5: + erf(x)

    Parameters
    ----------
    x : np.ndarray
    powers : list[float]
        Використовується лише len(powers) — визначає кількість функцій.
        Самі значення ігноруються (базис фіксований).
    """
    s = len(powers)
    if s > len(_ROBUST_POOL):
        raise ValueError(
            f"robust_basis підтримує порядок до {len(_ROBUST_POOL)}, отримано {s}"
        )
    return np.array([_ROBUST_POOL[i](x) for i in range(s)])


# Реєстр доступних базисів
_BASIS_REGISTRY = {
    'fractional': fractional_basis,
    'polynomial': polynomial_basis,
    'trigonometric': trigonometric_basis,
    'robust': robust_basis,
}


# ──────────────────────────────────────────────────────
#  Головний клас
# ──────────────────────────────────────────────────────

class DSGEFeatureExtractor:
    """
    Обчислення DSGE-ознак для знешумлення сигналів.

    Parameters
    ----------
    basis_type : str
        Тип базисних функцій: 'fractional' | 'polynomial' | 'trigonometric' | 'robust'.
    powers : list[float]
        Степені або частоти для відповідного basis_type.
        За замовчуванням: [0.5, 1.5, 2.0] для fractional (оптимальний з HAR-статті).
    tikhonov_lambda : float
        Коефіцієнт λ для регуляризації Тихонова (запобігає виродженості F).
    stft_params : dict
        Параметри STFT: {'nperseg': 32, 'noverlap': 16, 'fs': 8192}.
    """

    def __init__(
        self,
        basis_type: str = 'fractional',
        powers: list[float] | None = None,
        tikhonov_lambda: float = 0.01,
        stft_params: dict | None = None,
    ):
        self.basis_type = basis_type
        self.powers = powers if powers is not None else [0.5, 1.5, 2.0]
        self.tikhonov_lambda = tikhonov_lambda
        self.stft_params = stft_params or {'nperseg': 32, 'noverlap': 16, 'fs': 8192}

        if basis_type not in _BASIS_REGISTRY:
            raise ValueError(
                f"Unknown basis_type '{basis_type}'. "
                f"Available: {list(_BASIS_REGISTRY.keys())}"
            )

        self._basis_fn = _BASIS_REGISTRY[basis_type]

        # Стан після fit()
        self._fitted = False
        self.S: int = len(self.powers)
        self.psi_0: float | None = None
        self.psi: np.ndarray | None = None
        self.K: np.ndarray | None = None
        self.k0: float | None = None
        self.gen_element_norm: float | None = None

        # Class-specific fit state (H4): populated only by fit_class_specific()
        self.K_bins: list | None = None
        self.k0_bins: list | None = None
        self.psi_bins: list | None = None
        self.psi_0_bins: list | None = None
        self.snr_edges: np.ndarray | None = None
        self.n_bins: int = 0
        self.bin_means: list | None = None

    # ──────────────────────────────────────────────────
    #  Обчислення базису
    # ──────────────────────────────────────────────────

    def _apply_basis(self, x: np.ndarray) -> np.ndarray:
        """Застосовує базисні функції. Повертає [S, len(x)]."""
        return self._basis_fn(x, self.powers)

    # ──────────────────────────────────────────────────
    #  Fit: навчання на тренувальних даних
    # ──────────────────────────────────────────────────

    def _solve_bucket(self, clean: np.ndarray, noisy: np.ndarray):
        """Solve (psi_0, psi, K, k0, S) for a single bucket of paired signals."""
        psi_0 = float(clean.mean())
        phi_all = self._basis_fn(noisy, self.powers).transpose(1, 0, 2)  # [N, S, T]
        S = phi_all.shape[1]
        psi = phi_all.mean(axis=(0, 2))  # [S]

        phi_centered = phi_all - psi[np.newaxis, :, np.newaxis]
        x_centered = clean - psi_0

        phi_flat = phi_centered.transpose(1, 0, 2).reshape(S, -1)  # [S, N*T]
        F = (phi_flat @ phi_flat.T) / phi_flat.shape[1]            # [S, S]
        F_reg = F + self.tikhonov_lambda * np.eye(S)

        x_flat = x_centered.reshape(1, -1)
        B = (phi_flat @ x_flat.T).ravel() / phi_flat.shape[1]      # [S]

        K = np.linalg.solve(F_reg, B)
        k0 = psi_0 - float(K @ psi)
        return psi_0, psi, K, k0, S

    def fit(self, clean_signals: np.ndarray, noisy_signals: np.ndarray) -> 'DSGEFeatureExtractor':
        """
        Обчислює статистики та коефіцієнти K на тренувальних даних.

        Parameters
        ----------
        clean_signals : np.ndarray, shape [N, T]
        noisy_signals : np.ndarray, shape [N, T]
        """
        clean_norm = (clean_signals - clean_signals.mean(axis=1, keepdims=True)) / (
            clean_signals.std(axis=1, keepdims=True) + 1e-8
        )
        gen_element = clean_norm.mean(axis=0)
        self.gen_element_norm = float(np.linalg.norm(gen_element))

        self.psi_0, self.psi, self.K, self.k0, self.S = \
            self._solve_bucket(clean_signals, noisy_signals)

        # Reset class-specific state (in case fit() called after fit_class_specific())
        self.K_bins = None
        self.k0_bins = None
        self.psi_bins = None
        self.psi_0_bins = None
        self.snr_edges = None
        self.n_bins = 0
        self.bin_means = None

        self._fitted = True
        return self

    def fit_class_specific(
        self,
        clean_signals: np.ndarray,
        noisy_signals: np.ndarray,
        snr_db: np.ndarray,
        n_bins: int = 3,
    ) -> 'DSGEFeatureExtractor':
        """
        Clase-specific fit: окремі K, k0, psi на кожен SNR bucket (H4).

        Bucket edges — квантильні (рівна кількість семплів у кожному).
        Глобальні K, k0, psi також зберігаються як fallback.

        Parameters
        ----------
        clean_signals, noisy_signals : np.ndarray, shape [N, T]
        snr_db : np.ndarray, shape [N]
            Per-sample SNR (dB), використовується для бакетизації.
        n_bins : int
            Кількість bucket-ів (типово 3).
        """
        assert len(snr_db) == len(clean_signals), (
            f"snr_db length {len(snr_db)} != N signals {len(clean_signals)}"
        )
        if n_bins < 2:
            raise ValueError(f"n_bins must be ≥2, got {n_bins}")

        # Generating element norm (global)
        clean_norm = (clean_signals - clean_signals.mean(axis=1, keepdims=True)) / (
            clean_signals.std(axis=1, keepdims=True) + 1e-8
        )
        gen_element = clean_norm.mean(axis=0)
        self.gen_element_norm = float(np.linalg.norm(gen_element))

        # Quantile-based bin edges for balanced buckets
        quantiles = np.linspace(0, 1, n_bins + 1)
        edges = np.quantile(snr_db, quantiles)
        edges = edges.astype(float)
        edges[0] = -np.inf
        edges[-1] = np.inf

        self.K_bins = []
        self.k0_bins = []
        self.psi_bins = []
        self.psi_0_bins = []
        self.snr_edges = edges
        self.n_bins = n_bins
        self.bin_means = []

        print(f"[DSGE] Class-specific fit: {n_bins} buckets, quantile edges "
              f"[{', '.join(f'{e:+.2f}' for e in edges)}] dB")
        for b in range(n_bins):
            in_bin = (snr_db >= edges[b]) & (snr_db < edges[b + 1])
            n_in = int(in_bin.sum())
            if n_in < 10:
                raise ValueError(
                    f"bucket {b} has only {n_in} samples (< 10); "
                    f"try reducing n_bins or increasing data_fraction"
                )
            snr_mean = float(snr_db[in_bin].mean())
            psi_0_b, psi_b, K_b, k0_b, S_b = self._solve_bucket(
                clean_signals[in_bin], noisy_signals[in_bin]
            )
            self.K_bins.append(K_b)
            self.k0_bins.append(k0_b)
            self.psi_bins.append(psi_b)
            self.psi_0_bins.append(psi_0_b)
            self.bin_means.append(snr_mean)
            K_str = ", ".join(f"{k:+.3f}" for k in K_b)
            print(f"  bucket {b}  n={n_in:6d}  μSNR={snr_mean:+6.2f} dB  "
                  f"K=[{K_str}]  k0={k0_b:+.4f}")

        # Global fallback fit (used when snr_db not provided at inference)
        self.psi_0, self.psi, self.K, self.k0, self.S = \
            self._solve_bucket(clean_signals, noisy_signals)

        self._fitted = True
        return self

    def bucket_for_snr(self, snr_db: float) -> int:
        """Return bucket index for a given SNR. Requires fit_class_specific()."""
        if self.K_bins is None:
            raise RuntimeError("bucket_for_snr requires fit_class_specific()")
        for b in range(self.n_bins):
            if self.snr_edges[b] <= snr_db < self.snr_edges[b + 1]:
                return b
        return self.n_bins - 1

    # ──────────────────────────────────────────────────
    #  compute_basis
    # ──────────────────────────────────────────────────

    def compute_basis(self, noisy_signal: np.ndarray) -> np.ndarray:
        """Обчислює базисні функції від одного сигналу. Повертає [S, T]."""
        return self._apply_basis(noisy_signal)

    # ──────────────────────────────────────────────────
    #  compute_dsge_spectrograms
    # ──────────────────────────────────────────────────

    def compute_dsge_spectrograms(self, noisy_signal: np.ndarray) -> np.ndarray:
        """
        STFT від кожного φᵢ(x̃). Повертає [S, F, T'].
        Параметри STFT ідентичні до основної спектрограми.
        """
        phi = self.compute_basis(noisy_signal)  # [S, T]
        nperseg = self.stft_params.get('nperseg', 32)
        noverlap = self.stft_params.get('noverlap', 16)
        fs = self.stft_params.get('fs', 8192)

        mags = []
        for i in range(self.S):
            _, _, Zxx = stft(phi[i], fs=fs, nperseg=nperseg, noverlap=noverlap)
            mags.append(np.abs(Zxx))

        return np.array(mags)  # [S, F, T']

    # ──────────────────────────────────────────────────
    #  reconstruct: пряме DSGE-наближення
    # ──────────────────────────────────────────────────

    def reconstruct(self, noisy_signal: np.ndarray, snr_db: float | None = None) -> np.ndarray:
        """Y = k₀ + Σ kᵢ·φᵢ(x̃). Вимагає fit().

        If class-specific fit is active and snr_db is provided, routes to the
        matching bucket's (K_b, k0_b). Otherwise uses global (K, k0).
        """
        self._check_fitted()
        phi = self.compute_basis(noisy_signal)
        if self.K_bins is not None and snr_db is not None:
            b = self.bucket_for_snr(float(snr_db))
            return self.k0_bins[b] + (self.K_bins[b][:, np.newaxis] * phi).sum(axis=0)
        return self.k0 + (self.K[:, np.newaxis] * phi).sum(axis=0)

    # ──────────────────────────────────────────────────
    #  DSGE channel formation for Hybrid UNet
    # ──────────────────────────────────────────────────

    def compute_dsge_channels_A(
        self, noisy_signal: np.ndarray, snr_db: float | None = None
    ) -> np.ndarray:
        """Variant A: reconstruction + residual.

        Returns [2, F, T'] — two STFT magnitude spectrograms:
          channel 0: |STFT(x̂_dsge)| — optimal DSGE reconstruction
          channel 1: |STFT(Z_dsge)| — residual (what DSGE couldn't explain)

        where x̂ = k₀ + Σ kᵢ·φᵢ(x̃),  Z = x̃ - x̂.

        If class-specific fit active and snr_db provided → routes per bucket.
        """
        self._check_fitted()
        x_hat = self.reconstruct(noisy_signal, snr_db=snr_db)  # [T]
        z_residual = noisy_signal - x_hat                       # [T]

        nperseg = self.stft_params.get('nperseg', 32)
        noverlap = self.stft_params.get('noverlap', 16)
        fs = self.stft_params.get('fs', 8192)

        _, _, Zxx_hat = stft(x_hat, fs=fs, nperseg=nperseg, noverlap=noverlap)
        _, _, Zxx_res = stft(z_residual, fs=fs, nperseg=nperseg, noverlap=noverlap)

        return np.array([np.abs(Zxx_hat), np.abs(Zxx_res)])  # [2, F, T']

    def compute_dsge_channels_B(
        self, noisy_signal: np.ndarray, snr_db: float | None = None
    ) -> np.ndarray:
        """Variant B: reconstruction + weighted basis channels.

        Returns [1+S, F, T'] — STFT magnitude spectrograms:
          channel 0: |STFT(x̂_dsge)|         — optimal DSGE reconstruction
          channels 1..S: |STFT(kᵢ·φᵢ(x̃))| — individually weighted basis components

        Each basis contribution kᵢ·φᵢ(x̃) shows how much that nonlinear
        component contributes to the reconstruction.

        If class-specific fit active and snr_db provided → routes per bucket
        for both x̂ and the per-basis weights kᵢ.
        """
        self._check_fitted()
        x_hat = self.reconstruct(noisy_signal, snr_db=snr_db)  # [T]
        phi = self.compute_basis(noisy_signal)                  # [S, T]

        # Select K per bucket if class-specific and snr_db provided
        if self.K_bins is not None and snr_db is not None:
            b = self.bucket_for_snr(float(snr_db))
            K_use = self.K_bins[b]
        else:
            K_use = self.K

        nperseg = self.stft_params.get('nperseg', 32)
        noverlap = self.stft_params.get('noverlap', 16)
        fs = self.stft_params.get('fs', 8192)

        # Channel 0: DSGE reconstruction
        _, _, Zxx_hat = stft(x_hat, fs=fs, nperseg=nperseg, noverlap=noverlap)
        mags = [np.abs(Zxx_hat)]

        # Channels 1..S: weighted basis components
        for i in range(self.S):
            weighted = K_use[i] * phi[i]  # kᵢ·φᵢ(x̃)
            _, _, Zxx_i = stft(weighted, fs=fs, nperseg=nperseg, noverlap=noverlap)
            mags.append(np.abs(Zxx_i))

        return np.array(mags)  # [1+S, F, T']

    # ──────────────────────────────────────────────────
    #  check_generating_element_norm
    # ──────────────────────────────────────────────────

    def check_generating_element_norm(self, threshold: float = 0.1) -> bool:
        self._check_fitted()
        norm = self.gen_element_norm
        print(f"[DSGE] Generating element ‖X‖ = {norm:.4f} "
              f"({'OK' if norm >= threshold else 'DEGENERATE — consider clustering'})")
        return norm >= threshold

    # ──────────────────────────────────────────────────
    #  Серіалізація
    # ──────────────────────────────────────────────────

    def save_state(self, path: str) -> None:
        self._check_fitted()
        np.savez(
            path,
            psi_0=np.array([self.psi_0]),
            psi=self.psi,
            K=self.K,
            k0=np.array([self.k0]),
            gen_element_norm=np.array([self.gen_element_norm]),
            S=np.array([self.S]),
            powers=np.array(self.powers),
            tikhonov_lambda=np.array([self.tikhonov_lambda]),
        )
        print(f"[DSGE] State saved → {path}")

    @classmethod
    def load_state(cls, path: str, basis_type: str = 'fractional',
                   stft_params: dict | None = None) -> 'DSGEFeatureExtractor':
        data = np.load(path)
        extractor = cls(
            basis_type=basis_type,
            powers=list(data['powers']),
            tikhonov_lambda=data['tikhonov_lambda'].item(),
            stft_params=stft_params,
        )
        extractor.psi_0 = data['psi_0'].item()
        extractor.psi = data['psi']
        extractor.K = data['K']
        extractor.k0 = data['k0'].item()
        extractor.gen_element_norm = data['gen_element_norm'].item()
        extractor.S = data['S'].item()
        extractor._fitted = True
        return extractor

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("DSGEFeatureExtractor is not fitted yet. Call fit() first.")

    def __repr__(self) -> str:
        state = "fitted" if self._fitted else "not fitted"
        return (f"DSGEFeatureExtractor(basis_type='{self.basis_type}', "
                f"powers={self.powers}, S={self.S}, lambda={self.tikhonov_lambda}, state={state})")
