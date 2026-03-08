import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from tqdm.auto import tqdm


class SignalDatasetGenerator:
    """
    Генератор датасету для знешумлення сигналів.

    Підтримує два профілі сценаріїв:
      - "fpv_telemetry": вузькосмуговий цифровий зв'язок (FPV-дрони, ELRS-подібні системи)
      - "deep_space":    дальній космічний зв'язок з екстремально низьким SNR

    Два профілі шуму:
      - Гауссовий (AWGN) — baseline
      - Негауссовий (суміш імпульсного, кольорового, полігауссового) — для порівняння
    """

    SCENARIOS = {
        "fpv_telemetry": {
            # Baseband після downconversion; реальна несуча 433/868/915 МГц або 2.4 ГГц
            "modulations": ["qpsk", "cpfsk", "gfsk"],
            "carrier_range": (300, 1800),   # Гц у базовій смузі
            "snr_range":     (-5.0, 15.0),  # дБ
        },
        "deep_space": {
            # Baseband після downconversion; реальна несуча X/Ka-band (8–32 ГГц)
            "modulations": ["bpsk", "qpsk"],
            "carrier_range": (100, 800),    # Гц у базовій смузі
            "snr_range":     (-20.0, 0.0),  # дБ
        },
    }

    # Допустимі типи негауссового шуму
    NON_GAUSSIAN_NOISE_TYPES = ("impulse", "pink", "red", "polygauss", "polygauss_nonstationary")

    def __init__(
        self,
        num_samples: int,
        sample_rate: int = 8192,
        duration: float = 0.25,
        scenario: str = "fpv_telemetry",
        snr_range: tuple[float, float] | None = None,
        non_gaussian_noise_types: list[str] | None = None,
        non_gaussian_mix_mode: str = "fixed",
    ):
        """
        Parameters
        ----------
        num_samples              : кількість прикладів у датасеті
        sample_rate              : частота дискретизації [Гц]; Найквіст = sample_rate/2
        duration                 : тривалість одного сигналу [с]
        scenario                 : "fpv_telemetry" або "deep_space"
        snr_range                : (snr_min_dB, snr_max_dB) — перекриває дефолт сценарію
        non_gaussian_noise_types : список типів негауссового шуму для використання.
                                   Доступні: "impulse", "pink", "red", "polygauss",
                                             "polygauss_nonstationary".
                                   За замовчуванням: ["polygauss"].
        non_gaussian_mix_mode    : "fixed"  — використовувати всі вказані типи одночасно;
                                   "random" — випадково вибрати підмножину (хоча б один).
        """
        assert scenario in self.SCENARIOS, f"Unknown scenario: {scenario}. Choose from {list(self.SCENARIOS)}"
        assert non_gaussian_mix_mode in ("fixed", "random"), \
            "non_gaussian_mix_mode must be 'fixed' or 'random'"

        if non_gaussian_noise_types is None:
            non_gaussian_noise_types = ["polygauss"]
        unknown = set(non_gaussian_noise_types) - set(self.NON_GAUSSIAN_NOISE_TYPES)
        assert not unknown, f"Unknown noise types: {unknown}. Available: {self.NON_GAUSSIAN_NOISE_TYPES}"

        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.duration = duration
        self.scenario = scenario
        self.config = self.SCENARIOS[scenario]
        self.carrier_range = self.config["carrier_range"]
        self.snr_range = snr_range if snr_range is not None else self.config["snr_range"]
        self.non_gaussian_noise_types = list(non_gaussian_noise_types)
        self.non_gaussian_mix_mode = non_gaussian_mix_mode
        self._signal_len = int(sample_rate * duration)

    # ──────────────────────────────────────────────────────────────────────────
    # Утиліти
    # ──────────────────────────────────────────────────────────────────────────

    def _time_vector(self) -> np.ndarray:
        return np.linspace(0, self.duration, self._signal_len, endpoint=False)

    def _random_carrier(self) -> float:
        return random.uniform(*self.carrier_range)

    def _random_symbol_rate(self, carrier_freq: float) -> float:
        """Symbol rate = carrier / k, де k ∈ [6, 12] — 6–12 циклів несучої на символ."""
        return carrier_freq / random.uniform(6.0, 12.0)

    # ──────────────────────────────────────────────────────────────────────────
    # Модуляції
    # ──────────────────────────────────────────────────────────────────────────

    def generate_bpsk_signal(
        self,
        carrier_freq: float | None = None,
        symbol_rate: float | None = None,
        amplitude: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """BPSK: phase ∈ {0, π} per symbol."""
        if carrier_freq is None:
            carrier_freq = self._random_carrier()
        if symbol_rate is None:
            symbol_rate = self._random_symbol_rate(carrier_freq)

        t = self._time_vector()
        num_symbols = max(1, int(self.duration * symbol_rate))
        bits = np.random.randint(0, 2, num_symbols)
        phases = np.where(bits == 0, 0.0, np.pi)

        samples_per_symbol = self._signal_len / num_symbols
        signal = np.zeros(self._signal_len)
        for i, phase in enumerate(phases):
            start = int(i * samples_per_symbol)
            end = min(int((i + 1) * samples_per_symbol), self._signal_len)
            signal[start:end] = amplitude * np.cos(2 * np.pi * carrier_freq * t[start:end] + phase)

        return t, signal

    def generate_qpsk_signal(
        self,
        carrier_freq: float | None = None,
        symbol_rate: float | None = None,
        amplitude: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """QPSK: phase ∈ {0, π/2, π, 3π/2}."""
        phase_map = {(0, 0): 0.0, (0, 1): np.pi / 2, (1, 0): np.pi, (1, 1): 3 * np.pi / 2}

        if carrier_freq is None:
            carrier_freq = self._random_carrier()
        if symbol_rate is None:
            symbol_rate = self._random_symbol_rate(carrier_freq)

        t = self._time_vector()
        num_symbols = max(1, int(self.duration * symbol_rate))
        bits = np.random.randint(0, 2, num_symbols * 2)
        symbols = [(bits[i], bits[i + 1]) for i in range(0, len(bits), 2)]

        samples_per_symbol = self._signal_len / num_symbols
        signal = np.zeros(self._signal_len)
        for i, sym in enumerate(symbols):
            phase = phase_map[sym]
            start = int(i * samples_per_symbol)
            end = min(int((i + 1) * samples_per_symbol), self._signal_len)
            signal[start:end] = amplitude * np.cos(2 * np.pi * carrier_freq * t[start:end] + phase)

        return t, signal

    def generate_cpfsk_signal(
        self,
        carrier_freq: float | None = None,
        bit_rate: float | None = None,
        h: float | None = None,
        amplitude: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        CPFSK (Continuous Phase FSK) з векторизованим розрахунком фази.

        freq₀ = carrier - h·bit_rate/2
        freq₁ = carrier + h·bit_rate/2

        Фаза інтегрується поcлідовно → неперервність гарантована.

        Parameters
        ----------
        h : modulation index. h=0.5 → MSK (мінімальне зміщення).
        """
        if carrier_freq is None:
            carrier_freq = self._random_carrier()
        if bit_rate is None:
            bit_rate = self._random_symbol_rate(carrier_freq)
        if h is None:
            h = random.uniform(0.5, 1.5)

        freq_dev = h * bit_rate / 2.0
        freq0 = carrier_freq - freq_dev
        freq1 = carrier_freq + freq_dev

        num_bits = max(1, int(self.duration * bit_rate))
        bits = np.random.randint(0, 2, num_bits)

        # Розгортаємо біти до частоти на кожному відліку (векторизовано)
        samples_per_bit = self._signal_len / num_bits
        freq_sequence = np.repeat(
            np.where(bits == 1, freq1, freq0),
            int(round(samples_per_bit)),
        )[: self._signal_len]
        if len(freq_sequence) < self._signal_len:
            freq_sequence = np.pad(freq_sequence, (0, self._signal_len - len(freq_sequence)), mode="edge")

        # Інтегруємо частоту → неперервна фаза
        dt = 1.0 / self.sample_rate
        phase = 2.0 * np.pi * np.cumsum(freq_sequence) * dt
        signal = amplitude * np.cos(phase)

        t = self._time_vector()
        return t, signal

    def generate_gfsk_signal(
        self,
        carrier_freq: float | None = None,
        bit_rate: float | None = None,
        h: float | None = None,
        BT: float = 0.4,
        amplitude: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        GFSK: CPFSK з гаусівським фільтром на бітовому потоці.

        BT — bandwidth-time product:
          0.3 → Bluetooth classic
          0.5 → BLE, типові FPV-системи

        Гаусів фільтр згладжує фазові переходи між бітами,
        зменшуючи вихідну смугу сигналу (ближче до MSK).
        """
        if carrier_freq is None:
            carrier_freq = self._random_carrier()
        if bit_rate is None:
            bit_rate = self._random_symbol_rate(carrier_freq)
        if h is None:
            h = random.uniform(0.3, 0.5)

        num_bits = max(1, int(self.duration * bit_rate))
        bits = np.random.randint(0, 2, num_bits)

        # ±1 на рівні відліків
        samples_per_bit = self._signal_len / num_bits
        bit_signal = np.repeat(2 * bits - 1, int(round(samples_per_bit)))[: self._signal_len]
        if len(bit_signal) < self._signal_len:
            bit_signal = np.pad(bit_signal, (0, self._signal_len - len(bit_signal)), mode="edge")

        # σ гаусівського фільтра (у відліках)
        sigma_samples = (np.sqrt(np.log(2)) * self.sample_rate) / (2.0 * np.pi * BT * bit_rate)
        filtered = gaussian_filter1d(bit_signal.astype(float), sigma=sigma_samples)

        # FM-модуляція: інтегруємо миттєву частоту
        freq_dev = h * bit_rate / 2.0
        inst_freq = carrier_freq + freq_dev * filtered
        phase = 2.0 * np.pi * np.cumsum(inst_freq) / self.sample_rate
        signal = amplitude * np.cos(phase)

        t = self._time_vector()
        return t, signal

    # ──────────────────────────────────────────────────────────────────────────
    # Шум
    # ──────────────────────────────────────────────────────────────────────────

    def _ou_trajectories(self, n: int, K: int, theta: float, sigma: float) -> np.ndarray:
        """
        K незалежних траєкторій процесу Орнштейна-Уленбека довжиною n відліків.

        Дискретна рекурсія: x[t] = α·x[t-1] + β·ε[t]
          α = 1 − θ·dt  (коефіцієнт загасання за крок)
          β = σ·√dt     (амплітуда шуму)

        Реалізовано через scipy.signal.lfilter — O(n·K), без Python-циклу.
        Час кореляції OU: τ = 1/θ секунд.

        Returns
        -------
        ndarray (K, n)
        """
        from scipy.signal import lfilter
        dt    = 1.0 / self.sample_rate
        alpha = max(0.0, 1.0 - theta * dt)
        beta  = sigma * np.sqrt(dt)
        eps   = np.random.randn(K, n)
        # IIR: b=[beta], a=[1, -alpha]  →  y[t] = alpha·y[t-1] + beta·eps[t]
        return lfilter([beta], [1.0, -alpha], eps, axis=1)

    def _polygauss_nonstationary_component(self, n: int, K: int = 4) -> np.ndarray:
        """
        Нестаціонарний полігауссовий шум (polygauss_nonstationary).

        Суміш K гаусіан, параметри якої плавно дрейфують у часі через
        процеси Орнштейна-Уленбека (OU):

        Крок 1 — Траєкторії параметрів
          Ваги w_k(t):  K траєкторій OU → softmax → w_k(t) ∈ (0,1), Σw_k=1
          Дисперсії:    K траєкторій OU у log-просторі → exp → σ_k(t) > 0;
                        різні log-зміщення на компоненту (фоновий шум ↔ викиди)
          Середні:      K-1 вільних траєкторій OU + обмеження нульового середнього:
                        μ_K(t) = −Σ_{k<K} w_k(t)·μ_k(t) / w_K(t)

        Крок 2 — Поточкова генерація (векторизована)
          Для кожного t вибрати активну компоненту k(t) методом рулетки
          (накопичених ваг) та згенерувати відлік N(μ_{k(t)}, σ²_{k(t)}).

        Parameters
        ----------
        K : кількість компонент (≥ 3)
        """
        assert K >= 3, "K must be >= 3 for polygauss_nonstationary"

        # ── Крок 1. Траєкторії параметрів ────────────────────────────────────
        # Ваги: softmax(OU) — τ ≈ 0.125 с (повільний дрейф відносно 0.25 с вікна)
        logits  = self._ou_trajectories(n, K, theta=8.0, sigma=3.0)   # (K, n)
        logits -= logits.max(axis=0)                                    # стабільний softmax
        exp_l   = np.exp(logits)
        weights = exp_l / exp_l.sum(axis=0)                            # (K, n)

        # Дисперсії: exp(OU) > 0; log-зміщення розподіляють компоненти від
        # слабкого теплового шуму (−0.5) до сильних викидів (+1.0)
        log_var = self._ou_trajectories(n, K, theta=5.0, sigma=2.0)   # (K, n)
        offsets = np.linspace(-0.5, 1.0, K)
        stds    = np.exp(log_var + offsets[:, np.newaxis])             # (K, n), > 0

        # Середні: K-1 вільних + μ_K забезпечує нульове загальне середнє
        means         = np.zeros((K, n))
        means[:K - 1] = self._ou_trajectories(n, K - 1, theta=10.0, sigma=2.0)
        means[K - 1]  = (
            -np.sum(weights[:K - 1] * means[:K - 1], axis=0)
            / (weights[K - 1] + 1e-9)
        )

        # ── Крок 2. Векторизована генерація відліків ──────────────────────────
        # Метод рулетки: k(t) = перший індекс де cumsum ≥ u(t)
        cum_w = np.cumsum(weights, axis=0)                             # (K, n)
        u     = np.random.uniform(0.0, 1.0, n)
        k_idx = (cum_w < u[np.newaxis, :]).sum(axis=0).clip(0, K - 1) # (n,)

        # Fancy indexing: selected_means[t] = means[k_idx[t], t]
        t_idx            = np.arange(n)
        selected_means   = means[k_idx, t_idx]
        selected_stds    = stds[k_idx, t_idx]
        noise            = selected_means + selected_stds * np.random.randn(n)

        return noise

    def _noise_std_for_snr(self, signal: np.ndarray, snr_db: float) -> float:
        """Обчислює std гауссового шуму для заданого SNR (дБ)."""
        signal_power = np.mean(signal ** 2)
        if signal_power < 1e-12:
            signal_power = 1.0
        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        return float(np.sqrt(noise_power))

    def _add_gaussian_noise(self, signal: np.ndarray, snr_db: float) -> np.ndarray:
        """AWGN — гауссовий білий шум заданого SNR."""
        std = self._noise_std_for_snr(signal, snr_db)
        return signal + np.random.normal(0.0, std, self._signal_len)

    def _add_non_gaussian_noise(self, signal: np.ndarray, snr_db: float) -> np.ndarray:
        """
        Негауссові завади, масштабовані до заданого SNR.

        Доступні типи (задаються через non_gaussian_noise_types):
          - "impulse"                  : імпульсний шум (ESC-перешкоди, EM-спалахи)
          - "pink"                     : рожевий шум (1/f), нахил спектру ≈ -1
          - "red"                      : червоний шум (1/f²), нахил ≈ -2
          - "polygauss"                : стаціонарна суміш гауссіан, рівномірний спектр
          - "polygauss_nonstationary"  : суміш гауссіан з параметрами що дрейфують
                                         через процеси Орнштейна-Уленбека (K ≥ 3)

        Режими (non_gaussian_mix_mode):
          - "fixed"  : використовувати всі вказані типи одночасно
          - "random" : випадково вибрати підмножину (хоча б один)
        """
        n = self._signal_len
        noise = np.zeros(n)

        if self.non_gaussian_mix_mode == "fixed":
            chosen = list(self.non_gaussian_noise_types)
        else:  # "random"
            chosen = [t for t in self.non_gaussian_noise_types if random.random() < 0.5]
            if not chosen:
                chosen = [random.choice(self.non_gaussian_noise_types)]

        for noise_type in chosen:
            component = np.zeros(n)
            if noise_type == "impulse":
                prob = random.uniform(0.005, 0.02)
                amp = random.uniform(3.0, 8.0)
                component = np.random.choice([0.0, amp], size=n, p=[1.0 - prob, prob])

            elif noise_type == "pink":
                white = np.random.randn(n)
                component = np.cumsum(white)
                component /= np.max(np.abs(component)) + 1e-9

            elif noise_type == "red":
                white = np.random.randn(n)
                component = np.cumsum(np.cumsum(white))
                component /= np.max(np.abs(component)) + 1e-9

            elif noise_type == "polygauss":
                k = random.randint(2, 5)
                weights = np.random.dirichlet(np.ones(k))
                means = np.random.uniform(-2.0, 2.0, k)
                stds = np.random.uniform(0.3, 1.5, k)
                choices = np.random.choice(k, size=n, p=weights)
                component = np.fromiter(
                    (np.random.normal(means[c], stds[c]) for c in choices),
                    dtype=float, count=n,
                )

            elif noise_type == "polygauss_nonstationary":
                K = random.randint(3, 5)
                component = self._polygauss_nonstationary_component(n, K=K)

            noise += component

        # Масштабуємо сумарний шум до цільового SNR.
        # Використовуємо RMS (а не std), щоб коректно врахувати
        # ненульове середнє у pink/red (random walk) та polygauss.
        target_rms = self._noise_std_for_snr(signal, snr_db)  # = sqrt(noise_power)
        current_rms = np.sqrt(np.mean(noise ** 2))
        if current_rms > 1e-9:
            noise = noise * (target_rms / current_rms)

        return signal + noise

    # ──────────────────────────────────────────────────────────────────────────
    # Вибір модуляції за профілем
    # ──────────────────────────────────────────────────────────────────────────

    def _generate_signal(self) -> np.ndarray:
        mod = random.choice(self.config["modulations"])
        if mod == "bpsk":
            _, signal = self.generate_bpsk_signal()
        elif mod == "qpsk":
            _, signal = self.generate_qpsk_signal()
        elif mod == "cpfsk":
            _, signal = self.generate_cpfsk_signal()
        elif mod == "gfsk":
            _, signal = self.generate_gfsk_signal()
        else:
            raise ValueError(f"Unknown modulation: {mod}")
        return signal

    # ──────────────────────────────────────────────────────────────────────────
    # Генерація датасетів
    # ──────────────────────────────────────────────────────────────────────────

    def generate_dataset(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Тренувальний датасет з ВАРІАТИВНИМ SNR (рівномірно з snr_range).

        Returns
        -------
        clean            : (N, L) — чисті сигнали
        gaussian_noisy   : (N, L) — з AWGN
        non_gaussian_noisy: (N, L) — з негауссовим шумом
        snr_values       : (N,)   — SNR кожного прикладу в дБ
        """
        clean_signals, gaussian_signals, non_gaussian_signals, snr_values = [], [], [], []

        for _ in tqdm(range(self.num_samples), desc="Generating train dataset", unit="sig"):
            signal = self._generate_signal()
            snr_db = random.uniform(*self.snr_range)

            clean_signals.append(signal)
            gaussian_signals.append(self._add_gaussian_noise(signal, snr_db))
            non_gaussian_signals.append(self._add_non_gaussian_noise(signal, snr_db))
            snr_values.append(snr_db)

        return (
            np.array(clean_signals),
            np.array(gaussian_signals),
            np.array(non_gaussian_signals),
            np.array(snr_values),
        )

    def generate_test_dataset(
        self,
        snr_values: tuple | list = (-15, -10, -5, 0, 5, 10),
        samples_per_snr: int | None = None,
    ) -> dict[float, dict[str, np.ndarray]]:
        """
        Тестовий датасет з ФІКСОВАНИМ SNR — для побудови ROC-кривих.

        Parameters
        ----------
        snr_values      : перелік SNR-точок [дБ]
        samples_per_snr : кількість прикладів на кожну SNR-точку.
                          За замовчуванням = self.num_samples.

        Returns
        -------
        dict { snr_db -> {"clean": ndarray, "gaussian": ndarray, "non_gaussian": ndarray} }
        """
        if samples_per_snr is None:
            samples_per_snr = self.num_samples

        result: dict[float, dict[str, np.ndarray]] = {}

        for snr_db in tqdm(snr_values, desc="Generating test dataset (SNR points)", unit="SNR"):
            clean_list, gauss_list, non_gauss_list = [], [], []
            for _ in tqdm(range(samples_per_snr), desc=f"  SNR={snr_db:+.0f} dB",
                          unit="sig", leave=False):
                signal = self._generate_signal()
                clean_list.append(signal)
                gauss_list.append(self._add_gaussian_noise(signal, snr_db))
                non_gauss_list.append(self._add_non_gaussian_noise(signal, snr_db))

            result[float(snr_db)] = {
                "clean":       np.array(clean_list),
                "gaussian":    np.array(gauss_list),
                "non_gaussian": np.array(non_gauss_list),
            }

        return result


# ──────────────────────────────────────────────────────────────────────────────
# DatasetExplorer
# ──────────────────────────────────────────────────────────────────────────────

class DatasetExplorer:
    def __init__(
        self,
        clean_signals: np.ndarray,
        gaussian_dataset: np.ndarray,
        non_gaussian_dataset: np.ndarray,
        snr_values: np.ndarray | None = None,
        sample_rate: int = 8192,
    ):
        self.clean_signals = clean_signals
        self.gaussian_dataset = gaussian_dataset
        self.non_gaussian_dataset = non_gaussian_dataset
        self.snr_values = snr_values
        self.sample_rate = sample_rate

    def visualize_sample(self, idx: int, dataset_type: str = "clean"):
        """Візуалізує сигнал з датасету у часовій та частотній областях."""
        data_map = {
            "clean":       self.clean_signals,
            "gaussian":    self.gaussian_dataset,
            "non_gaussian": self.non_gaussian_dataset,
        }
        if dataset_type not in data_map:
            raise ValueError(f"dataset_type must be one of {list(data_map)}")

        signal = data_map[dataset_type][idx]
        n = len(signal)
        t = np.arange(n) / self.sample_rate

        snr_label = ""
        if self.snr_values is not None:
            snr_label = f"  |  SNR = {self.snr_values[idx]:.1f} dB"

        fig, axes = plt.subplots(2, 1, figsize=(12, 7))

        axes[0].plot(t, signal, linewidth=0.8)
        axes[0].set_title(f"[{dataset_type}] sample #{idx}{snr_label} — time domain")
        axes[0].set_xlabel("Time [s]")
        axes[0].set_ylabel("Amplitude")
        axes[0].grid(True)

        freqs = np.fft.rfftfreq(n, d=1.0 / self.sample_rate)
        spectrum = np.abs(np.fft.rfft(signal))
        axes[1].plot(freqs, 20 * np.log10(spectrum + 1e-9), linewidth=0.8)
        axes[1].set_title("Spectrum (magnitude, dB)")
        axes[1].set_xlabel("Frequency [Hz]")
        axes[1].set_ylabel("Magnitude [dB]")
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()

    def save_dataset(self, path: str, dataset_type: str = "clean"):
        """Зберігає масив датасету у .npy файл."""
        data_map = {
            "clean":       self.clean_signals,
            "gaussian":    self.gaussian_dataset,
            "non_gaussian": self.non_gaussian_dataset,
        }
        if dataset_type not in data_map:
            raise ValueError(f"dataset_type must be one of {list(data_map)}")
        np.save(path, data_map[dataset_type])

    @staticmethod
    def save_test_dataset(result: dict, base_path: str):
        """
        Зберігає тестовий датасет з фіксованим SNR.
        Формат файлів: {base_path}/test_{snr:+.0f}dB_{noise_type}.npy
        """
        import os
        os.makedirs(base_path, exist_ok=True)
        for snr_db, arrays in result.items():
            tag = f"{snr_db:+.0f}dB".replace("+", "p").replace("-", "m")
            for noise_type, arr in arrays.items():
                fname = os.path.join(base_path, f"test_{tag}_{noise_type}.npy")
                np.save(fname, arr)
        print(f"Test dataset saved to '{base_path}' "
              f"({len(result)} SNR levels × 3 noise types).")


# ──────────────────────────────────────────────────────────────────────────────
# Запуск генерації:
#   python dataset/generation.py                              # defaults: deep_space + polygauss
#   python dataset/generation.py --deep_space --polygauss
#   python dataset/generation.py --fpv --polygauss_impulse
#   python dataset/generation.py --help                         # defaults: deep_space + polygauss
#   python dataset/generation.py --deep_space --polygauss
#   python dataset/generation.py --fpv --polygauss_impulse
#   python dataset/generation.py --deep_space --polygauss --num_train 5000
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, os, time

    # ── CLI ───────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Generate signal denoising dataset.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    scenario_group = parser.add_mutually_exclusive_group()
    scenario_group.add_argument(
        "--deep_space", action="store_true",
        help="Deep space scenario: BPSK/QPSK, carrier 100–800 Hz, SNR −20..0 dB  [default]",
    )
    scenario_group.add_argument(
        "--fpv", action="store_true",
        help="FPV telemetry scenario: QPSK/CPFSK/GFSK, carrier 300–1800 Hz, SNR −5..+15 dB",
    )

    noise_group = parser.add_mutually_exclusive_group()
    noise_group.add_argument(
        "--polygauss", action="store_true",
        help="Non-Gaussian noise: polygaussian only  [default]",
    )
    noise_group.add_argument(
        "--polygauss_impulse", action="store_true",
        help="Non-Gaussian noise: polygaussian + impulse (fixed combination)",
    )
    noise_group.add_argument(
        "--polygauss_nonstationary", action="store_true",
        help="Non-Gaussian noise: non-stationary poly-Gaussian via Ornstein-Uhlenbeck (K∈[3,5])",
    )
    noise_group.add_argument(
        "--all_noise", action="store_true",
        help="Non-Gaussian noise: impulse + pink + red + polygauss (random subset per sample)",
    )

    parser.add_argument("--num_train",      type=int, default=15_000,
                        help="Number of training samples (default: 15000)")
    parser.add_argument("--samples_per_snr", type=int, default=500,
                        help="Test samples per SNR point (default: 500)")

    args = parser.parse_args()

    # ── Розв'язання параметрів ────────────────────────────────────────────────
    SCENARIO = "fpv_telemetry" if args.fpv else "deep_space"

    if args.polygauss_impulse:
        NOISE_TYPES = ["polygauss", "impulse"]
        MIX_MODE    = "fixed"
        noise_tag   = "polygauss_impulse"
    elif args.polygauss_nonstationary:
        NOISE_TYPES = ["polygauss_nonstationary"]
        MIX_MODE    = "fixed"
        noise_tag   = "polygauss_nonstationary"
    elif args.all_noise:
        NOISE_TYPES = ["impulse", "pink", "red", "polygauss"]
        MIX_MODE    = "random"
        noise_tag   = "all_noise"
    else:                           # --polygauss або за замовчуванням
        NOISE_TYPES = ["polygauss"]
        MIX_MODE    = "fixed"
        noise_tag   = "polygauss"

    NUM_TRAIN       = args.num_train
    SAMPLES_PER_SNR = args.samples_per_snr
    SAMPLE_RATE     = 8192
    DURATION        = 0.25          # → 2048 відліків на сигнал

    # SNR-точки для тестового датасету (для ROC-кривих):
    #   deep_space: -20 dB "підлога" → 0 dB "стеля", густіше у перехідній зоні −12..−5 дБ
    #   fpv:        -5 dB "підлога"  → +15 dB "стеля"
    TEST_SNR_POINTS = {
        "deep_space":    (-20, -15, -12, -10, -7, -5, 0),
        "fpv_telemetry": (-5, 0, 3, 5, 8, 10, 15),
    }[SCENARIO]

    out_dir   = f"{SCENARIO}_{noise_tag}"
    TRAIN_DIR = os.path.join(os.path.dirname(__file__), out_dir, "train")
    TEST_DIR  = os.path.join(os.path.dirname(__file__), out_dir, "test")
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR,  exist_ok=True)

    signal_len = int(SAMPLE_RATE * DURATION)
    est_mb = (NUM_TRAIN * 3 + len(TEST_SNR_POINTS) * SAMPLES_PER_SNR * 3) * signal_len * 4 / 1024 ** 2
    print(f"Сценарій    : {SCENARIO}")
    print(f"sample_rate : {SAMPLE_RATE} Hz  |  duration: {DURATION} s  →  {signal_len} відліків")
    print(f"Шум         : {NOISE_TYPES}  ({MIX_MODE})")
    print(f"Тренування  : {NUM_TRAIN:,} прикладів")
    print(f"Тест        : {len(TEST_SNR_POINTS)} SNR-точок × {SAMPLES_PER_SNR} прикладів  {TEST_SNR_POINTS} dB")
    print(f"Очікуваний розмір: ~{est_mb:.0f} MB\n")

    gen = SignalDatasetGenerator(
        num_samples=NUM_TRAIN,
        sample_rate=SAMPLE_RATE,
        duration=DURATION,
        scenario=SCENARIO,
        non_gaussian_noise_types=NOISE_TYPES,
        non_gaussian_mix_mode=MIX_MODE,
    )

    # ── Тренувальний датасет ──────────────────────────────────────────────────
    print("[ 1 / 2 ]  Тренувальний датасет...")
    t0 = time.time()
    clean, gaussian, non_gaussian, snr_arr = gen.generate_dataset()
    print(f"           Готово за {time.time() - t0:.1f} с  "
          f"| SNR: [{snr_arr.min():.1f}, {snr_arr.max():.1f}] дБ\n")

    np.save(os.path.join(TRAIN_DIR, "clean_signals.npy"),        clean.astype(np.float32))
    np.save(os.path.join(TRAIN_DIR, "gaussian_signals.npy"),     gaussian.astype(np.float32))
    np.save(os.path.join(TRAIN_DIR, "non_gaussian_signals.npy"), non_gaussian.astype(np.float32))
    np.save(os.path.join(TRAIN_DIR, "snr_values.npy"),           snr_arr.astype(np.float32))
    for fname in ["clean_signals.npy", "gaussian_signals.npy", "non_gaussian_signals.npy"]:
        mb = os.path.getsize(os.path.join(TRAIN_DIR, fname)) / 1024 ** 2
        print(f"  {fname:<30s}  {mb:6.1f} MB")

    # ── Тестовий датасет ─────────────────────────────────────────────────────
    print(f"\n[ 2 / 2 ]  Тестовий датасет ({len(TEST_SNR_POINTS)} SNR-точок × {SAMPLES_PER_SNR})...")
    t0 = time.time()
    gen_test = SignalDatasetGenerator(
        num_samples=SAMPLES_PER_SNR,
        sample_rate=SAMPLE_RATE,
        duration=DURATION,
        scenario=SCENARIO,
        non_gaussian_noise_types=NOISE_TYPES,
        non_gaussian_mix_mode=MIX_MODE,
    )
    test_data = gen_test.generate_test_dataset(
        snr_values=TEST_SNR_POINTS,
        samples_per_snr=SAMPLES_PER_SNR,
    )
    print(f"           Готово за {time.time() - t0:.1f} с\n")
    DatasetExplorer.save_test_dataset(test_data, base_path=TEST_DIR)

    # ── Верифікація SNR ───────────────────────────────────────────────────────
    print("\nВерифікація SNR (тестовий датасет):")
    for snr_target, arrays in test_data.items():
        c, g, ng = arrays["clean"], arrays["gaussian"], arrays["non_gaussian"]
        sp = np.mean(c ** 2, axis=1)
        g_snr  = np.mean(10 * np.log10(sp / np.mean((g  - c) ** 2, axis=1)))
        ng_snr = np.mean(10 * np.log10(sp / np.mean((ng - c) ** 2, axis=1)))
        print(f"  target={snr_target:+3.0f} dB  →  gaussian={g_snr:+.2f} dB  non_gaussian={ng_snr:+.2f} dB")

    total_mb = sum(
        os.path.getsize(os.path.join(root, f)) / 1024 ** 2
        for root, _, files in os.walk(os.path.join(os.path.dirname(__file__), out_dir))
        for f in files if f.endswith(".npy")
    )
    print(f"\nЗагальний розмір датасету: {total_mb:.1f} MB")
    print("Done.")

