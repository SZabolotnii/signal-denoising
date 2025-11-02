import numpy as np
import pywt
import matplotlib.pyplot as plt
from metrics import MeanSquaredError


class WaveletDenoising:
    """
    Контрольований вейвлет-денойзинг 1D сигналів з per-level порогуванням (опційно),
    порогуванням лише детальних коефіцієнтів (cD_j), керованими режимами країв.
    """

    def __init__(self,
                 wavelet: str = 'db4',
                 level: int | None = None,
                 thresh_mode: str = 'soft',  # 'soft' або 'hard'
                 per_level: bool = True,  # окремий поріг на кожному рівні
                 ext_mode: str = 'symmetric'):  # 'symmetric', 'periodization', ...
        self.wavelet = wavelet
        self.level = level
        self.thresh_mode = thresh_mode
        self.per_level = per_level
        self.ext_mode = ext_mode

    # ---- утиліти ----
    @staticmethod
    def _mad(x: np.ndarray) -> float:
        """Median Absolute Deviation."""
        return np.median(np.abs(x - np.median(x)))

    def _estimate_sigma_from_detail(self, d: np.ndarray) -> float:
        """Оцінка σ з детальних коефіцієнтів (MAD/0.6745)."""
        if d.size == 0:
            return 0.0
        return self._mad(d) / 0.6745

    @staticmethod
    def _universal_threshold(sigma: float, n: int) -> float:
        """Універсальний поріг Donoho–Johnstone."""
        if sigma <= 0 or n <= 1:
            return 0.0
        return sigma * np.sqrt(2.0 * np.log(n))

    def _threshold_coeffs(self, coeffs: list[np.ndarray]) -> list[np.ndarray]:
        """
        Порогування ТІЛЬКИ детальних коефіцієнтів (coeffs[1:]).
        coeffs[0] (cA) не чіпаємо.
        """
        cA, *details = coeffs
        if not details:
            return coeffs

        new_details = []
        if self.per_level:
            # окремий σ і поріг для кожного рівня
            for d in details:
                sigma_j = self._estimate_sigma_from_detail(d)
                thr_j = self._universal_threshold(sigma_j, len(d))
                new_details.append(pywt.threshold(d, value=thr_j, mode=self.thresh_mode))
        else:
            # один σ (з найдрібнішого рівня) і один поріг для всіх рівнів
            sigma = self._estimate_sigma_from_detail(details[0])
            n = max(len(d) for d in details)
            thr = self._universal_threshold(sigma, n)
            for d in details:
                new_details.append(pywt.threshold(d, value=thr, mode=self.thresh_mode))

        return [cA] + new_details

    # ---- основний метод ----
    def denoise(self, signal: np.ndarray) -> np.ndarray:
        """
        Вейвлет-знешумлення 1D сигналу.
        Повертає сигнал тієї ж довжини (паддінг/обрізання за потреби).
        """
        # безпечний рівень декомпозиції
        w = pywt.Wavelet(self.wavelet)
        max_level = pywt.dwt_max_level(len(signal), w.dec_len)
        level = max_level if (self.level is None or self.level > max_level) else self.level

        # розклад
        coeffs = pywt.wavedec(signal, wavelet=self.wavelet, level=level, mode=self.ext_mode)

        # порогування деталей
        coeffs_thr = self._threshold_coeffs(coeffs)

        # реконструкція
        rec = pywt.waverec(coeffs_thr, wavelet=self.wavelet, mode=self.ext_mode)

        # вирівнювання довжини
        if len(rec) < len(signal):
            rec = np.pad(rec, (0, len(signal) - len(rec)))
        elif len(rec) > len(signal):
            rec = rec[:len(signal)]
        return rec

    # опційно: зручний сеттер для GridSearch
    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        return self

    def visualize(self,
                  original_signal: np.ndarray,
                  noisy_signal: np.ndarray,
                  denoised_signal: np.ndarray,
                  xlim: tuple[float, float] | None = None,
                  ylim: tuple[float, float] | None = None,
                  title: str = "Wavelet Denoising Result"):
        """
        Малює 3 ряди: оригінальний, зашумлений, знешумлений.
        Параметри:
          - xlim: (xmin, xmax) для зуму по осі X
          - ylim: (ymin, ymax) для масштабу по осі Y
          - title: заголовок графіка
        """
        n = len(original_signal)
        t = np.arange(n)

        plt.figure(figsize=(14, 6))
        plt.plot(t, original_signal, label='а) Clean', linewidth=2, color='black')
        plt.plot(t, noisy_signal, label='б) Noisy', alpha=0.6, color='gray')
        plt.plot(t, denoised_signal, label='в) Wavelet Denoised', linestyle='--', linewidth=2)

        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.title(title)
        if xlim is not None:
            plt.xlim(*xlim)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


# Приклад використання
if __name__ == "__main__":
    # Генерація тестового сигналу
    np.random.seed(0)
    t = np.linspace(0, 1, 1000, endpoint=False)
    clean_signal = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 10 * t)
    noise = np.random.normal(0, 0.5, len(t))
    noisy_signal = clean_signal + noise

    # Ініціалізація та використання вейвлет-знешумлення
    wavelet_denoiser = WaveletDenoising(wavelet='db6', level=4)
    denoised_signal = wavelet_denoiser.denoise(noisy_signal)

    # Обчислення MSE
    mse = MeanSquaredError.calculate(clean_signal, denoised_signal)
    print(f"Mean Squared Error (MSE): {mse:.6f}")

    # Візуалізація результатів
    wavelet_denoiser.visualize(clean_signal, noisy_signal, denoised_signal)
