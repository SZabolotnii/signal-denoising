import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from tqdm.auto import tqdm


class SignalDatasetGenerator:
    """
    Dataset generator for signal denoising experiments.

    Scenarios:
      - "fpv_telemetry": narrowband digital comms (FPV drones, ELRS-like)
      - "deep_space":    deep-space link, extreme low SNR

    Noise profiles:
      - Gaussian (AWGN) — baseline
      - Non-Gaussian (polygauss, impulse, pink, red)

    Signal is a real-valued simulated baseband/IF waveform.
    block_size defines samples per training example;
    latency at deployment = block_size / sample_rate.
    """

    SCENARIOS = {
        "fpv_telemetry": {
            "modulations": ["cpfsk", "gfsk", "qpsk"],
            "snr_range":   (-5.0, 15.0),
        },
        "deep_space": {
            "modulations": ["bpsk", "qpsk"],
            "snr_range":   (-20.0, 0.0),
        },
    }

    # Per-modulation realistic baseband/IF defaults
    MODULATION_DEFAULTS: dict[str, dict] = {
        "bpsk":  {"carrier_range": (100, 400),  "symbol_rate_ratio": (8, 16)},
        "qpsk":  {"carrier_range": (200, 600),  "symbol_rate_ratio": (6, 12)},
        "cpfsk": {"carrier_range": (300, 1000), "symbol_rate_ratio": (6, 10),
                  "h_range": (0.5, 1.5)},
        "gfsk":  {"carrier_range": (300, 1000), "symbol_rate_ratio": (6, 10),
                  "h_range": (0.3, 0.5), "BT": 0.4},
    }

    NON_GAUSSIAN_NOISE_TYPES = (
        "impulse", "pink", "red", "polygauss",
    )

    VALID_MODULATIONS = tuple(MODULATION_DEFAULTS.keys())
    # "random" is a valid modulation_type value meaning "pick randomly per signal"
    VALID_MODULATION_TYPES = ("random",) + VALID_MODULATIONS

    def __init__(
        self,
        num_samples: int,
        sample_rate: int = 8192,
        block_size: int = 1024,
        scenario: str = "fpv_telemetry",
        modulation_type: str = "qpsk",
        bits_per_symbol: int = 2,
        snr_range: tuple[float, float] | None = None,
        non_gaussian_noise_types: list[str] | None = None,
        non_gaussian_mix_mode: str = "fixed",
        polygauss_components: int = 3,
        polygauss_random_k: tuple[int, int] | None = None,
    ):
        """
        Parameters
        ----------
        num_samples              : number of examples in the datasets
        sample_rate              : ADC sample rate [Hz]
        block_size               : samples per training example
                                   (deployment latency = block_size / sample_rate)
        scenario                 : "deep_space" or "fpv_telemetry"
        modulation_type          : modulation used for every signal in the datasets.
                                   "random" — pick randomly from scenario's list per signal.
                                   Fixed choices: "bpsk", "qpsk", "cpfsk", "gfsk".
        bits_per_symbol          : log₂(M); determines constellation/alphabet size:
                                     1 → BPSK / binary FSK
                                     2 → QPSK / 4-FSK
                                     4 → 16-PSK / 16-FSK
                                     8 → 256-PSK / 256-FSK
        snr_range                : (min_dB, max_dB), overrides scenario default
        non_gaussian_noise_types : noise types to use; default ["polygauss"]
        non_gaussian_mix_mode    : "fixed" = all types summed; "random" = random subset per sample
        polygauss_components     : fixed number of GMM components K (default 3)
        polygauss_random_k       : (k_min, k_max) — randomise K per sample; overrides
                                   polygauss_components when set
        """
        assert scenario in self.SCENARIOS, \
            f"Unknown scenario: {scenario!r}. Choose from {list(self.SCENARIOS)}"
        assert non_gaussian_mix_mode in ("fixed", "random"), \
            "non_gaussian_mix_mode must be 'fixed' or 'random'"
        assert bits_per_symbol >= 1, "bits_per_symbol must be >= 1"
        assert modulation_type in self.VALID_MODULATION_TYPES, \
            f"Unknown modulation_type: {modulation_type!r}. Choose from {self.VALID_MODULATION_TYPES}"

        if non_gaussian_noise_types is None:
            non_gaussian_noise_types = ["polygauss"]
        unknown = set(non_gaussian_noise_types) - set(self.NON_GAUSSIAN_NOISE_TYPES)
        assert not unknown, f"Unknown noise types: {unknown}. Available: {self.NON_GAUSSIAN_NOISE_TYPES}"

        self.num_samples              = num_samples
        self.sample_rate              = sample_rate
        self.block_size               = block_size
        self.scenario                 = scenario
        self.config                   = self.SCENARIOS[scenario]
        self.modulation_type          = modulation_type
        self.bits_per_symbol          = bits_per_symbol
        self.snr_range                = snr_range if snr_range is not None else self.config["snr_range"]
        self.non_gaussian_noise_types = list(non_gaussian_noise_types)
        self.non_gaussian_mix_mode    = non_gaussian_mix_mode
        self.polygauss_components     = polygauss_components
        self.polygauss_random_k       = polygauss_random_k
        self._signal_len              = block_size
        self._t                       = np.linspace(
            0, block_size / sample_rate, block_size, endpoint=False
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────────

    def _time_vector(self) -> np.ndarray:
        return self._t

    def _random_carrier(self, modulation: str) -> float:
        return random.uniform(*self.MODULATION_DEFAULTS[modulation]["carrier_range"])

    def _random_symbol_rate(self, carrier_freq: float, modulation: str) -> float:
        ratio = random.uniform(*self.MODULATION_DEFAULTS[modulation]["symbol_rate_ratio"])
        return carrier_freq / ratio

    def _n_polygauss_components(self) -> int:
        """Return K: fixed polygauss_components, or random from polygauss_random_k."""
        if self.polygauss_random_k is not None:
            return random.randint(*self.polygauss_random_k)
        return self.polygauss_components

    # ──────────────────────────────────────────────────────────────────────────
    # Modulations
    # ──────────────────────────────────────────────────────────────────────────

    def generate_psk_signal(
        self,
        carrier_freq: float | None = None,
        symbol_rate: float | None = None,
        amplitude: float = 1.0,
        modulation: str = "bpsk",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        M-PSK with M = 2^bits_per_symbol equally spaced phases.

        bits_per_symbol=1 → BPSK (M=2)
        bits_per_symbol=2 → QPSK (M=4)
        bits_per_symbol=3 → 8-PSK
        bits_per_symbol=4 → 16-PSK
        etc.

        carrier_range and symbol_rate_ratio are taken from MODULATION_DEFAULTS[modulation].
        """
        M = 2 ** self.bits_per_symbol
        if carrier_freq is None:
            carrier_freq = self._random_carrier(modulation)
        if symbol_rate is None:
            symbol_rate = self._random_symbol_rate(carrier_freq, modulation)

        t = self._time_vector()
        duration = self.block_size / self.sample_rate
        num_symbols = max(1, int(duration * symbol_rate))
        samples_per_symbol = self._signal_len / num_symbols

        symbol_indices = np.random.randint(0, M, num_symbols)
        symbol_phases  = 2 * np.pi * symbol_indices / M

        phase_sequence = np.repeat(
            symbol_phases, int(round(samples_per_symbol))
        )[: self._signal_len]
        if len(phase_sequence) < self._signal_len:
            phase_sequence = np.pad(
                phase_sequence, (0, self._signal_len - len(phase_sequence)), mode="edge"
            )
        signal = amplitude * np.cos(2 * np.pi * carrier_freq * t + phase_sequence)
        return t, signal

    def generate_bpsk_signal(
        self,
        carrier_freq: float | None = None,
        symbol_rate: float | None = None,
        amplitude: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """BPSK — M-PSK with 'bpsk' carrier/symbol-rate defaults."""
        return self.generate_psk_signal(carrier_freq, symbol_rate, amplitude, modulation="bpsk")

    def generate_qpsk_signal(
        self,
        carrier_freq: float | None = None,
        symbol_rate: float | None = None,
        amplitude: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """QPSK — M-PSK with 'qpsk' carrier/symbol-rate defaults."""
        return self.generate_psk_signal(carrier_freq, symbol_rate, amplitude, modulation="qpsk")

    def generate_cpfsk_signal(
        self,
        carrier_freq: float | None = None,
        bit_rate: float | None = None,
        h: float | None = None,
        amplitude: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        M-ary CPFSK with M = 2^bits_per_symbol frequencies.

        Frequencies: f_k = carrier + (2k − M + 1) · h · symbol_rate / 2,  k = 0…M−1
        Phase is integrated sample-by-sample → phase continuity guaranteed.

        bits_per_symbol=1 → binary FSK (2 frequencies)
        bits_per_symbol=2 → 4-FSK
        etc.
        """
        M = 2 ** self.bits_per_symbol
        if carrier_freq is None:
            carrier_freq = self._random_carrier("cpfsk")
        if bit_rate is None:
            bit_rate = self._random_symbol_rate(carrier_freq, "cpfsk")
        if h is None:
            h = random.uniform(*self.MODULATION_DEFAULTS["cpfsk"]["h_range"])

        freq_dev     = h * bit_rate / 2.0
        freq_offsets = np.array([(2 * k - M + 1) * freq_dev for k in range(M)])
        freqs        = carrier_freq + freq_offsets  # (M,)

        duration     = self.block_size / self.sample_rate
        num_symbols  = max(1, int(duration * bit_rate))
        symbol_indices = np.random.randint(0, M, num_symbols)

        samples_per_symbol = self._signal_len / num_symbols
        freq_sequence = np.repeat(
            freqs[symbol_indices], int(round(samples_per_symbol))
        )[: self._signal_len]
        if len(freq_sequence) < self._signal_len:
            freq_sequence = np.pad(
                freq_sequence, (0, self._signal_len - len(freq_sequence)), mode="edge"
            )

        dt    = 1.0 / self.sample_rate
        phase  = 2.0 * np.pi * np.cumsum(freq_sequence) * dt
        signal = amplitude * np.cos(phase)

        t = self._time_vector()
        return t, signal

    def generate_gfsk_signal(
        self,
        carrier_freq: float | None = None,
        bit_rate: float | None = None,
        h: float | None = None,
        BT: float | None = None,
        amplitude: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        M-ary GFSK: M-ary CPFSK with Gaussian pre-filtering of the symbol stream.

        BT — bandwidth-time product:
          0.3 → Bluetooth classic
          0.4 → BLE / typical FPV

        Symbol values are normalised to [−1, +1] so max frequency deviation
        = h·bit_rate/2 regardless of M.
        """
        M = 2 ** self.bits_per_symbol
        if carrier_freq is None:
            carrier_freq = self._random_carrier("gfsk")
        if bit_rate is None:
            bit_rate = self._random_symbol_rate(carrier_freq, "gfsk")
        if h is None:
            h = random.uniform(*self.MODULATION_DEFAULTS["gfsk"]["h_range"])
        if BT is None:
            BT = self.MODULATION_DEFAULTS["gfsk"]["BT"]

        duration    = self.block_size / self.sample_rate
        num_symbols = max(1, int(duration * bit_rate))
        symbol_indices = np.random.randint(0, M, num_symbols)

        # Normalise to [-1, +1]: (2k − M + 1) / (M − 1) for M > 1
        if M > 1:
            symbol_values = (2 * symbol_indices - (M - 1)).astype(float) / (M - 1)
        else:
            symbol_values = np.ones(num_symbols, dtype=float)

        samples_per_symbol = self._signal_len / num_symbols
        symbol_signal = np.repeat(
            symbol_values, int(round(samples_per_symbol))
        )[: self._signal_len]
        if len(symbol_signal) < self._signal_len:
            symbol_signal = np.pad(
                symbol_signal, (0, self._signal_len - len(symbol_signal)), mode="edge"
            )

        sigma_samples = (
            (np.sqrt(np.log(2)) * self.sample_rate) / (2.0 * np.pi * BT * bit_rate)
        )
        filtered = gaussian_filter1d(symbol_signal, sigma=sigma_samples)

        freq_dev  = h * bit_rate / 2.0
        inst_freq = carrier_freq + freq_dev * filtered
        phase     = 2.0 * np.pi * np.cumsum(inst_freq) / self.sample_rate
        signal    = amplitude * np.cos(phase)

        t = self._time_vector()
        return t, signal

    # ──────────────────────────────────────────────────────────────────────────
    # Noise
    # ──────────────────────────────────────────────────────────────────────────

    def _noise_std_for_snr(self, signal: np.ndarray, snr_db: float) -> float:
        """RMS noise amplitude for target SNR [dB]."""
        signal_power = np.mean(signal ** 2)
        if signal_power < 1e-12:
            signal_power = 1.0
        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        return float(np.sqrt(noise_power))

    def _add_gaussian_noise(self, signal: np.ndarray, snr_db: float) -> np.ndarray:
        """AWGN — white Gaussian noise at specified SNR."""
        std = self._noise_std_for_snr(signal, snr_db)
        return signal + np.random.normal(0.0, std, self._signal_len)

    def _generate_non_gaussian_noise(self, signal: np.ndarray, snr_db: float) -> np.ndarray:
        """
        Generate non-Gaussian noise scaled to target SNR. Returns pure noise array.

        Available types (set via non_gaussian_noise_types):
          - "impulse"               : sparse impulse noise (EM spikes)
          - "pink"                  : 1/f noise, spectral slope ≈ −10 dB/dec
          - "red"                   : 1/f² noise, slope ≈ −20 dB/dec
          - "polygauss"             : stationary GMM, K = _n_polygauss_components()


        Modes (non_gaussian_mix_mode):
          - "fixed"  : all listed types combined
          - "random" : random subset per sample (at least one)
        """
        n     = self._signal_len
        noise = np.zeros(n)

        if self.non_gaussian_mix_mode == "fixed":
            chosen = list(self.non_gaussian_noise_types)
        else:
            chosen = [t for t in self.non_gaussian_noise_types if random.random() < 0.5]
            if not chosen:
                chosen = [random.choice(self.non_gaussian_noise_types)]

        for noise_type in chosen:
            component = np.zeros(n)

            if noise_type == "impulse":
                prob      = random.uniform(0.005, 0.02)
                amp       = random.uniform(3.0, 8.0)
                component = np.random.choice(
                    [0.0, amp], size=n, p=[1.0 - prob, prob]
                )

            elif noise_type == "pink":
                white     = np.random.randn(n)
                component = np.cumsum(white)
                component /= np.max(np.abs(component)) + 1e-9

            elif noise_type == "red":
                white     = np.random.randn(n)
                np.cumsum(white, out=white)
                component = np.cumsum(white)
                component /= np.max(np.abs(component)) + 1e-9

            elif noise_type == "polygauss":
                k       = self._n_polygauss_components()
                weights = np.random.dirichlet(np.ones(k))
                means   = np.random.uniform(-2.0, 2.0, k)
                stds    = np.random.uniform(0.3, 1.5, k)
                choices = np.random.choice(k, size=n, p=weights)
                component = np.random.normal(means[choices], stds[choices])

            noise += component

        # Scale combined noise to target SNR using RMS (handles non-zero mean noise)
        target_rms  = self._noise_std_for_snr(signal, snr_db)
        current_rms = np.sqrt(np.mean(noise ** 2))
        if current_rms > 1e-9:
            noise = noise * (target_rms / current_rms)

        return noise

    def _add_non_gaussian_noise(self, signal: np.ndarray, snr_db: float) -> np.ndarray:
        """Signal + non-Gaussian noise at target SNR."""
        return signal + self._generate_non_gaussian_noise(signal, snr_db)

    # ──────────────────────────────────────────────────────────────────────────
    # Modulation dispatch
    # ──────────────────────────────────────────────────────────────────────────

    def _generate_signal(self) -> np.ndarray:
        mod = random.choice(self.config["modulations"]) \
            if self.modulation_type == "random" else self.modulation_type
        if mod in ("bpsk", "qpsk"):
            _, signal = self.generate_psk_signal(modulation=mod)
        elif mod == "cpfsk":
            _, signal = self.generate_cpfsk_signal()
        elif mod == "gfsk":
            _, signal = self.generate_gfsk_signal()
        else:
            raise ValueError(f"Unknown modulation: {mod!r}")
        return signal

    # ──────────────────────────────────────────────────────────────────────────
    # Dataset generation
    # ──────────────────────────────────────────────────────────────────────────

    def generate_dataset(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Training datasets with variable SNR (uniform from snr_range).

        Returns
        -------
        clean                  : (N, block_size)
        gaussian_noisy         : (N, block_size)
        non_gaussian_noisy     : (N, block_size)  signal + noise
        non_gaussian_noise_only: (N, block_size)  pure noise component
        snr_values             : (N,)
        """
        N, L = self.num_samples, self.block_size
        clean          = np.empty((N, L), dtype=np.float32)
        gaussian_noisy = np.empty((N, L), dtype=np.float32)
        non_gauss      = np.empty((N, L), dtype=np.float32)
        noise_only     = np.empty((N, L), dtype=np.float32)
        snr_values     = np.empty(N,      dtype=np.float32)

        for i in tqdm(range(N), desc="Generating train datasets", unit="sig"):
            signal = self._generate_signal()
            snr_db = random.uniform(*self.snr_range)
            noise  = self._generate_non_gaussian_noise(signal, snr_db)

            clean[i]          = signal
            gaussian_noisy[i] = self._add_gaussian_noise(signal, snr_db)
            non_gauss[i]      = signal + noise
            noise_only[i]     = noise
            snr_values[i]     = snr_db

        return clean, gaussian_noisy, non_gauss, noise_only, snr_values

    def generate_test_dataset(
        self,
        snr_values: tuple | list = (-15, -10, -5, 0, 5, 10),
        samples_per_snr: int | None = None,
    ) -> dict[float, dict[str, np.ndarray]]:
        """
        Test datasets with fixed SNR points — for SNR-vs-metric curves.

        Returns
        -------
        dict { snr_db -> {"clean", "gaussian", "non_gaussian", "non_gaussian_noise_only"} }
        """
        if samples_per_snr is None:
            samples_per_snr = self.num_samples

        result: dict[float, dict[str, np.ndarray]] = {}

        N, L = samples_per_snr, self.block_size
        for snr_db in tqdm(snr_values, desc="Generating test datasets", unit="SNR"):
            clean_arr     = np.empty((N, L), dtype=np.float32)
            gauss_arr     = np.empty((N, L), dtype=np.float32)
            non_gauss_arr = np.empty((N, L), dtype=np.float32)
            noise_arr     = np.empty((N, L), dtype=np.float32)

            for i in tqdm(range(N), desc=f"  SNR={snr_db:+.0f} dB",
                          unit="sig", leave=False):
                signal = self._generate_signal()
                noise  = self._generate_non_gaussian_noise(signal, snr_db)
                clean_arr[i]     = signal
                gauss_arr[i]     = self._add_gaussian_noise(signal, snr_db)
                non_gauss_arr[i] = signal + noise
                noise_arr[i]     = noise

            result[float(snr_db)] = {
                "clean":                    clean_arr,
                "gaussian":                 gauss_arr,
                "non_gaussian":             non_gauss_arr,
                "non_gaussian_noise_only":  noise_arr,
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
        self.clean_signals        = clean_signals
        self.gaussian_dataset     = gaussian_dataset
        self.non_gaussian_dataset = non_gaussian_dataset
        self.snr_values           = snr_values
        self.sample_rate          = sample_rate

    def visualize_sample(self, idx: int, dataset_type: str = "clean"):
        """Visualise one signal in time and frequency domains."""
        data_map = {
            "clean":        self.clean_signals,
            "gaussian":     self.gaussian_dataset,
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

        freqs    = np.fft.rfftfreq(n, d=1.0 / self.sample_rate)
        spectrum = np.abs(np.fft.rfft(signal))
        axes[1].plot(freqs, 20 * np.log10(spectrum + 1e-9), linewidth=0.8)
        axes[1].set_title("Spectrum (magnitude, dB)")
        axes[1].set_xlabel("Frequency [Hz]")
        axes[1].set_ylabel("Magnitude [dB]")
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()

    def save_dataset(self, path: str, dataset_type: str = "clean"):
        """Save datasets array to .npy file."""
        data_map = {
            "clean":        self.clean_signals,
            "gaussian":     self.gaussian_dataset,
            "non_gaussian": self.non_gaussian_dataset,
        }
        if dataset_type not in data_map:
            raise ValueError(f"dataset_type must be one of {list(data_map)}")
        np.save(path, data_map[dataset_type])

    @staticmethod
    def save_test_dataset(result: dict, base_path: str):
        """
        Save fixed-SNR test datasets.
        File format: {base_path}/test_{snr}dB_{noise_type}.npy
        """
        import os
        os.makedirs(base_path, exist_ok=True)
        for snr_db, arrays in result.items():
            tag = f"{snr_db:+.0f}dB".replace("+", "p").replace("-", "m")
            for noise_type, arr in arrays.items():
                fname = os.path.join(base_path, f"test_{tag}_{noise_type}.npy")
                np.save(fname, arr)
        print(f"Test datasets saved to '{base_path}' "
              f"({len(result)} SNR levels × 3 noise types).")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
#
#   python datasets/generation.py                                 # defaults
#   python datasets/generation.py --deep_space --polygauss
#   python datasets/generation.py --fpv --polygauss --modulation qpsk
#   python datasets/generation.py --deep_space --polygauss --bits_per_symbol 2
#   python datasets/generation.py --help
#
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, os, time

    parser = argparse.ArgumentParser(
        description="Generate signal denoising datasets.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Scenario
    scenario_group = parser.add_mutually_exclusive_group()
    scenario_group.add_argument(
        "--deep_space", action="store_true",
        help="Deep space: BPSK/QPSK, SNR −20..0 dB",
    )
    scenario_group.add_argument(
        "--fpv", action="store_true",
        help="FPV telemetry: QPSK/CPFSK/GFSK, SNR −5..+15 dB  [default]",
    )

    # Noise type
    noise_group = parser.add_mutually_exclusive_group()
    noise_group.add_argument(
        "--polygauss", action="store_true",
        help="Non-Gaussian noise: stationary polygauss",
    )
    noise_group.add_argument(
        "--polygauss_impulse", action="store_true",
        help="Non-Gaussian noise: polygauss + impulse (fixed combination)",
    )
    noise_group.add_argument(
        "--all_noise", action="store_true",
        help="Non-Gaussian noise: impulse + pink + red + polygauss (random subset)",
    )

    # Signal parameters
    parser.add_argument(
        "--modulation_type",
        choices=SignalDatasetGenerator.VALID_MODULATION_TYPES,
        default="qpsk",
        help="Modulation used for all signals (default: qpsk).\n"
             "'random' — pick randomly from scenario's list per signal.",
    )
    parser.add_argument(
        "--bits_per_symbol", type=int, default=2,
        help="log₂(M): constellation/alphabet size M = 2^bps (default: 2)",
    )
    parser.add_argument(
        "--block_size", type=int, default=1024,
        help="Samples per training example (default: 1024)\n"
             "Deployment latency = block_size / sample_rate",
    )

    # Polygauss components
    parser.add_argument(
        "--polygauss_components", type=int, default=3,
        help="Fixed number of GMM components K (default: 3)",
    )
    parser.add_argument(
        "--polygauss_random_k", type=int, nargs=2, metavar=("K_MIN", "K_MAX"),
        default=None,
        help="Randomise K per sample in [K_MIN, K_MAX]; overrides --polygauss_components",
    )

    # Dataset size
    parser.add_argument("--num_train",       type=int, default=400_000,
                        help="Training samples (default: 400000)")
    parser.add_argument("--samples_per_snr", type=int, default=10_000,
                        help="Test samples per SNR point (default: 10000)")

    args = parser.parse_args()

    # Resolve parameters
    SCENARIO = "fpv_telemetry" if args.fpv else "deep_space"

    if args.polygauss_impulse:
        NOISE_TYPES = ["polygauss", "impulse"]
        MIX_MODE    = "fixed"
        noise_tag   = "polygauss_impulse"
    elif args.polygauss:
        NOISE_TYPES = ["polygauss"]
        MIX_MODE    = "fixed"
        noise_tag   = "polygauss"
    elif args.all_noise:
        NOISE_TYPES = ["impulse", "pink", "red", "polygauss"]
        MIX_MODE    = "random"
        noise_tag   = "all_noise"
    else:  # default: polygauss
        NOISE_TYPES = ["polygauss"]
        MIX_MODE    = "fixed"
        noise_tag   = "polygauss"

    BLOCK_SIZE      = args.block_size
    SAMPLE_RATE     = 8192
    NUM_TRAIN       = args.num_train
    SAMPLES_PER_SNR = args.samples_per_snr

    POLY_RANDOM_K = tuple(args.polygauss_random_k) if args.polygauss_random_k else None

    TEST_SNR_POINTS = {
        "deep_space":    (-20, -17, -15, -12, -10, -7, -5, -3, 0, 3),
        "fpv_telemetry": (-5, -2, 0, 3, 5, 8, 10, 12, 15, 18),
    }[SCENARIO]

    import json, uuid
    from datetime import datetime

    # Build unique run folder
    uid            = uuid.uuid4().hex[:8]
    noise_tag_full = "_".join(NOISE_TYPES)
    folder_name = (
        f"{SCENARIO}_{noise_tag_full}_{args.modulation_type}"
        f"_bs{BLOCK_SIZE}_n{NUM_TRAIN}_{uid}"
    )
    DATA_GEN_DIR = os.path.join(os.path.dirname(__file__), "datasets")
    RUN_DIR   = os.path.join(DATA_GEN_DIR, folder_name)
    TRAIN_DIR = os.path.join(RUN_DIR, "train")
    TEST_DIR  = os.path.join(RUN_DIR, "test")
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR,  exist_ok=True)

    # Save dataset_config.json
    config = {
        "uid":                  uid,
        "created_at":           datetime.now().isoformat(timespec="seconds"),
        "folder":               folder_name,
        "scenario":             SCENARIO,
        "modulation_type":      args.modulation_type,
        "bits_per_symbol":      args.bits_per_symbol,
        "block_size":           BLOCK_SIZE,
        "sample_rate":          SAMPLE_RATE,
        "snr_range":            list(SignalDatasetGenerator.SCENARIOS[SCENARIO]["snr_range"]),
        "noise_types":          NOISE_TYPES,
        "mix_mode":             MIX_MODE,
        "polygauss_components": args.polygauss_components,
        "polygauss_random_k":   list(POLY_RANDOM_K) if POLY_RANDOM_K else None,
        "num_train":            NUM_TRAIN,
        "test_snr_points":      list(TEST_SNR_POINTS),
        "samples_per_snr":      SAMPLES_PER_SNR,
    }
    with open(os.path.join(RUN_DIR, "dataset_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    latency_ms = 1000 * BLOCK_SIZE / SAMPLE_RATE
    est_mb = (
        (NUM_TRAIN * 3 + len(TEST_SNR_POINTS) * SAMPLES_PER_SNR * 3)
        * BLOCK_SIZE * 4 / 1024 ** 2
    )

    print(f"Run folder        : {folder_name}")
    print(f"Scenario          : {SCENARIO}")
    print(f"Sample rate       : {SAMPLE_RATE} Hz")
    print(f"Block size        : {BLOCK_SIZE} samples  →  latency {latency_ms:.1f} ms")
    print(f"Modulation        : {args.modulation_type}  |  bits/symbol: {args.bits_per_symbol}")
    print(f"Noise             : {NOISE_TYPES}  ({MIX_MODE})")
    if POLY_RANDOM_K:
        print(f"Polygauss K       : random {POLY_RANDOM_K}")
    else:
        print(f"Polygauss K       : fixed {args.polygauss_components}")
    print(f"Train samples     : {NUM_TRAIN:,}")
    print(f"Test SNR points   : {TEST_SNR_POINTS}  ×  {SAMPLES_PER_SNR} samples")
    print(f"Estimated size    : ~{est_mb:.0f} MB\n")

    gen = SignalDatasetGenerator(
        num_samples=NUM_TRAIN,
        sample_rate=SAMPLE_RATE,
        block_size=BLOCK_SIZE,
        scenario=SCENARIO,
        modulation_type=args.modulation_type,
        bits_per_symbol=args.bits_per_symbol,
        non_gaussian_noise_types=NOISE_TYPES,
        non_gaussian_mix_mode=MIX_MODE,
        polygauss_components=args.polygauss_components,
        polygauss_random_k=POLY_RANDOM_K,
    )

    # Training datasets
    print("[ 1 / 2 ]  Training datasets...")
    t0 = time.time()
    clean, gaussian, non_gaussian, noise_only, snr_arr = gen.generate_dataset()
    print(f"           Done in {time.time() - t0:.1f} s  "
          f"| SNR: [{snr_arr.min():.1f}, {snr_arr.max():.1f}] dB\n")

    np.save(os.path.join(TRAIN_DIR, "clean_signals.npy"),               clean.astype(np.float32))
    np.save(os.path.join(TRAIN_DIR, "gaussian_signals.npy"),            gaussian.astype(np.float32))
    np.save(os.path.join(TRAIN_DIR, "non_gaussian_signals.npy"),        non_gaussian.astype(np.float32))
    np.save(os.path.join(TRAIN_DIR, "non_gaussian_noise_only.npy"),     noise_only.astype(np.float32))
    np.save(os.path.join(TRAIN_DIR, "snr_values.npy"),                  snr_arr.astype(np.float32))
    for fname in ["clean_signals.npy", "gaussian_signals.npy", "non_gaussian_signals.npy",
                  "non_gaussian_noise_only.npy"]:
        mb = os.path.getsize(os.path.join(TRAIN_DIR, fname)) / 1024 ** 2
        print(f"  {fname:<30s}  {mb:6.1f} MB")

    # Test datasets
    print(f"\n[ 2 / 2 ]  Test datasets ({len(TEST_SNR_POINTS)} SNR points × {SAMPLES_PER_SNR})...")
    t0 = time.time()
    gen_test = SignalDatasetGenerator(
        num_samples=SAMPLES_PER_SNR,
        sample_rate=SAMPLE_RATE,
        block_size=BLOCK_SIZE,
        scenario=SCENARIO,
        modulation_type=args.modulation_type,
        bits_per_symbol=args.bits_per_symbol,
        non_gaussian_noise_types=NOISE_TYPES,
        non_gaussian_mix_mode=MIX_MODE,
        polygauss_components=args.polygauss_components,
        polygauss_random_k=POLY_RANDOM_K,
    )
    test_data = gen_test.generate_test_dataset(
        snr_values=TEST_SNR_POINTS,
        samples_per_snr=SAMPLES_PER_SNR,
    )
    print(f"           Done in {time.time() - t0:.1f} s\n")
    DatasetExplorer.save_test_dataset(test_data, base_path=TEST_DIR)

    # SNR verification
    print("\nSNR verification (test datasets):")
    for snr_target, arrays in test_data.items():
        c, g, ng = arrays["clean"], arrays["gaussian"], arrays["non_gaussian"]
        sp     = np.mean(c ** 2, axis=1)
        g_snr  = np.mean(10 * np.log10(sp / np.mean((g  - c) ** 2, axis=1)))
        ng_snr = np.mean(10 * np.log10(sp / np.mean((ng - c) ** 2, axis=1)))
        print(f"  target={snr_target:+3.0f} dB  →  "
              f"gaussian={g_snr:+.2f} dB  non_gaussian={ng_snr:+.2f} dB")

    total_mb = sum(
        os.path.getsize(os.path.join(root, f)) / 1024 ** 2
        for root, _, files in os.walk(RUN_DIR)
        for f in files if f.endswith(".npy")
    )
    print(f"\nDataset saved to  : {RUN_DIR}")
    print(f"Total size        : {total_mb:.1f} MB")
    print("Done.")
