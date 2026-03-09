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
    """Поліноміальний базис: φᵢ(x) = x^pᵢ  (цілі степені)."""
    return np.array([x ** int(p) for p in powers])


def trigonometric_basis(x: np.ndarray, freqs: list[float]) -> np.ndarray:
    """Тригонометричний базис: φᵢ(x) = sin(fᵢ · x)."""
    return np.array([np.sin(f * x) for f in freqs])


def robust_basis(x: np.ndarray, **kwargs) -> np.ndarray:
    """Робастний базис: tanh, sigmoid, atan — пригнічують викиди."""
    return np.array([
        np.tanh(x),
        1.0 / (1.0 + np.exp(-x)),   # sigmoid
        np.arctan(x),
    ])


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

    Реалізує:
    1. Формування породжуючого елемента з тренувальних даних.
    2. Обчислення базисних функцій φᵢ(x̃) від зашумленого сигналу.
    3. Розв'язання системи KF = B з регуляризацією Тихонова → коефіцієнти K.
    4. Генерацію DSGE-спектрограм |STFT{φᵢ(x̃)}| для U-Net.
    5. Пряме DSGE-наближення Y = k₀ + Σ kᵢ·φᵢ(x̃) (опційно).

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
        Параметри STFT: {'nperseg': 128, 'noverlap': 96, 'fs': 1024}.
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
        self.stft_params = stft_params or {'nperseg': 128, 'noverlap': 96, 'fs': 1024}

        if basis_type not in _BASIS_REGISTRY:
            raise ValueError(
                f"Unknown basis_type '{basis_type}'. "
                f"Available: {list(_BASIS_REGISTRY.keys())}"
            )

        self._basis_fn = _BASIS_REGISTRY[basis_type]

        # Стан після fit()
        self._fitted = False
        self.S: int = len(self.powers)           # порядок апроксимації
        self.psi_0: float | None = None          # E{x}
        self.psi: np.ndarray | None = None       # E{φᵢ(x̃)}, shape [S]
        self.K: np.ndarray | None = None         # коефіцієнти [S]
        self.k0: float | None = None             # вільний член
        self.gen_element_norm: float | None = None  # ‖X⁽ᵐ⁾‖ для діагностики

    # ──────────────────────────────────────────────────
    #  Обчислення базису
    # ──────────────────────────────────────────────────

    def _apply_basis(self, x: np.ndarray) -> np.ndarray:
        """Застосовує базисні функції. Повертає [S, len(x)]."""
        if self.basis_type == 'robust':
            return self._basis_fn(x)
        else:
            return self._basis_fn(x, self.powers)

    # ──────────────────────────────────────────────────
    #  Fit: навчання на тренувальних даних
    # ──────────────────────────────────────────────────

    def fit(
        self,
        clean_signals: np.ndarray,
        noisy_signals: np.ndarray,
    ) -> 'DSGEFeatureExtractor':
        """
        Обчислює статистики та коефіцієнти K на тренувальних даних.

        Parameters
        ----------
        clean_signals : np.ndarray, shape [N, T]
            N чистих сигналів довжиною T.
        noisy_signals : np.ndarray, shape [N, T]
            N зашумлених сигналів.

        Returns
        -------
        self
        """
        N, T = clean_signals.shape
        self.S = len(self.powers)  # оновлюємо на випадок robust

        # ── 1. Формування породжуючого елемента ──────────
        # Нормалізуємо чисті сигнали та усереднюємо (спрощений варіант без кластеризації)
        clean_norm = (clean_signals - clean_signals.mean(axis=1, keepdims=True)) / (
            clean_signals.std(axis=1, keepdims=True) + 1e-8
        )
        gen_element = clean_norm.mean(axis=0)  # [T]
        self.gen_element_norm = float(np.linalg.norm(gen_element))

        # ── 2. Глобальні середні ──────────────────────────
        # Ψ₀ = E{x} по всіх вибірках і всіх точках
        self.psi_0 = float(clean_signals.mean())

        # ── 3. Обчислення φᵢ для всіх зашумлених сигналів ──
        # phi_all: [N, S, T]
        phi_all = np.array([self._apply_basis(noisy_signals[i]) for i in range(N)])  # [N, S, T]

        # Ψᵢ = E{φᵢ(x̃)} по N*T елементах
        self.psi = phi_all.mean(axis=(0, 2))  # [S]

        # ── 4. Центровані базисні функції та чисті сигнали ──
        # Центровані φᵢ: [N, S, T]
        phi_centered = phi_all - self.psi[np.newaxis, :, np.newaxis]

        # Центровані чисті сигнали: [N, T]
        x_centered = clean_signals - self.psi_0

        # ── 5. Матриця F (корелянти) — розмір [S, S] ──────
        # Fᵢⱼ = E{[φᵢ(x̃)−Ψᵢ][φⱼ(x̃)−Ψⱼ]}
        # Швидко через reshape: phi_centered → [S, N*T]
        phi_flat = phi_centered.transpose(1, 0, 2).reshape(self.S, -1)  # [S, N*T]
        F = (phi_flat @ phi_flat.T) / phi_flat.shape[1]                 # [S, S]

        # Регуляризація Тихонова
        F_reg = F + self.tikhonov_lambda * np.eye(self.S)

        # ── 6. Вектор B (взаємні кореляції) — розмір [S] ──
        # Bᵢ = E{[x−Ψ₀][φᵢ(x̃)−Ψᵢ]}
        x_flat = x_centered.reshape(1, -1)  # [1, N*T]
        B = (phi_flat @ x_flat.T).ravel() / phi_flat.shape[1]  # [S]

        # ── 7. Розв'язання KF = B → K ──────────────────────
        self.K = np.linalg.solve(F_reg, B)  # [S]

        # Вільний член: k₀ = Ψ₀ − Σ kᵢ Ψᵢ
        self.k0 = self.psi_0 - float(self.K @ self.psi)

        self._fitted = True
        return self

    # ──────────────────────────────────────────────────
    #  compute_basis: базисні функції від одного сигналу
    # ──────────────────────────────────────────────────

    def compute_basis(self, noisy_signal: np.ndarray) -> np.ndarray:
        """
        Обчислює базисні функції від одного зашумленого сигналу.

        Parameters
        ----------
        noisy_signal : np.ndarray, shape [T]

        Returns
        -------
        phi : np.ndarray, shape [S, T]
            φᵢ(x̃) для i = 0..S-1.
        """
        return self._apply_basis(noisy_signal)  # [S, T]

    # ──────────────────────────────────────────────────
    #  compute_dsge_spectrograms
    # ──────────────────────────────────────────────────

    def compute_dsge_spectrograms(self, noisy_signal: np.ndarray) -> np.ndarray:
        """
        Обчислює DSGE-спектрограми: STFT від кожного φᵢ(x̃).

        Параметри STFT ідентичні до основної спектрограми сигналу —
        забезпечує сумісність розмірів для конкатенації каналів.

        Parameters
        ----------
        noisy_signal : np.ndarray, shape [T]

        Returns
        -------
        dsge_mags : np.ndarray, shape [S, F, T']
            Амплітудні спектрограми від S базисних функцій.
            F = nperseg//2 + 1, T' залежить від довжини сигналу та noverlap.
        """
        phi = self.compute_basis(noisy_signal)  # [S, T]
        nperseg = self.stft_params.get('nperseg', 128)
        noverlap = self.stft_params.get('noverlap', 96)
        fs = self.stft_params.get('fs', 1024)

        mags = []
        for i in range(self.S):
            _, _, Zxx = stft(phi[i], fs=fs, nperseg=nperseg, noverlap=noverlap)
            mags.append(np.abs(Zxx))  # [F, T']

        return np.array(mags)  # [S, F, T']

    # ──────────────────────────────────────────────────
    #  reconstruct: пряме DSGE-наближення
    # ──────────────────────────────────────────────────

    def reconstruct(self, noisy_signal: np.ndarray) -> np.ndarray:
        """
        Пряме DSGE-наближення чистого сигналу: Y = k₀ + Σ kᵢ·φᵢ(x̃).

        Вимагає попереднього виклику fit().

        Parameters
        ----------
        noisy_signal : np.ndarray, shape [T]

        Returns
        -------
        Y : np.ndarray, shape [T]
            Узгоджений вектор (DSGE-оцінка чистого сигналу).
        """
        self._check_fitted()
        phi = self.compute_basis(noisy_signal)  # [S, T]
        Y = self.k0 + (self.K[:, np.newaxis] * phi).sum(axis=0)  # [T]
        return Y

    # ──────────────────────────────────────────────────
    #  check_generating_element_norm
    # ──────────────────────────────────────────────────

    def check_generating_element_norm(self, threshold: float = 0.1) -> bool:
        """
        Перевіряє, чи є породжуючий елемент інформативним.

        Returns
        -------
        bool
            True, якщо норма достатня (елемент НЕ вироджений).
        """
        self._check_fitted()
        norm = self.gen_element_norm
        print(f"[DSGE] Generating element ‖X‖ = {norm:.4f} "
              f"({'OK' if norm >= threshold else 'DEGENERATE — consider clustering'})")
        return norm >= threshold

    # ──────────────────────────────────────────────────
    #  Серіалізація стану (для збереження разом з моделлю)
    # ──────────────────────────────────────────────────

    def save_state(self, path: str) -> None:
        """Зберігає стан DSGEFeatureExtractor у .npz файл."""
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
    def load_state(cls, path: str, basis_type: str = 'fractional', stft_params: dict | None = None) -> 'DSGEFeatureExtractor':
        """Відновлює DSGEFeatureExtractor зі збереженого .npz файлу."""
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

    # ──────────────────────────────────────────────────
    #  Внутрішнє
    # ──────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "DSGEFeatureExtractor is not fitted yet. Call fit() first."
            )

    def __repr__(self) -> str:
        state = "fitted" if self._fitted else "not fitted"
        return (
            f"DSGEFeatureExtractor("
            f"basis_type='{self.basis_type}', "
            f"powers={self.powers}, "
            f"S={self.S}, "
            f"lambda={self.tikhonov_lambda}, "
            f"state={state})"
        )
