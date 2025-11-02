import numpy as np
import matplotlib.pyplot as plt
import random


class SignalDatasetGenerator:
    def __init__(self, num_samples, sample_rate, duration, freq_range=(10, 100)):
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.duration = duration
        self.freq_range = freq_range  # Діапазон частот для генерації сигналів

    def generate_qpsk_signal(self, symbol_rate, amplitude=1):
        """Генерує QPSK-сигнал."""
        phase_map = {
            (0, 0): 0,
            (0, 1): np.pi / 2,
            (1, 0): np.pi,
            (1, 1): 3 * np.pi / 2
        }
        num_symbols = int(self.sample_rate * self.duration // symbol_rate)
        bits = np.random.randint(0, 2, num_symbols * 2)
        symbols = [(bits[i], bits[i + 1]) for i in range(0, len(bits), 2)]

        # Випадкова несуча частота
        carrier_freq = random.uniform(*self.freq_range)

        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), endpoint=False)
        signal = np.zeros_like(t)

        symbol_duration = 1 / symbol_rate
        for i, symbol in enumerate(symbols):
            phase = phase_map[symbol]
            start_idx = int(i * symbol_duration * self.sample_rate)
            end_idx = int((i + 1) * symbol_duration * self.sample_rate)
            signal[start_idx:end_idx] = amplitude * np.cos(2 * np.pi * carrier_freq * t[start_idx:end_idx] + phase)

        return t, signal

    def generate_fsk_signal(self, bit_rate, amplitude=1):
        """Генерує FSK-сигнал."""
        num_bits = int(self.sample_rate * self.duration // bit_rate)
        bits = np.random.randint(0, 2, num_bits)

        # Випадкові частоти для бітів "0" і "1"
        freq0 = random.uniform(*self.freq_range)
        freq1 = random.uniform(*self.freq_range)

        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), endpoint=False)
        signal = np.zeros_like(t)

        bit_duration = 1 / bit_rate
        for i, bit in enumerate(bits):
            freq = freq0 if bit == 0 else freq1
            start_idx = int(i * bit_duration * self.sample_rate)
            end_idx = int((i + 1) * bit_duration * self.sample_rate)
            signal[start_idx:end_idx] = amplitude * np.cos(2 * np.pi * freq * t[start_idx:end_idx])

        return t, signal

    def generate_gaussian_noise(self, mean=0, std=1):
        """Генерує білий шум (гауссові завади)."""
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), endpoint=False)
        noise = np.random.normal(mean, std, len(t))
        return t, noise

    def generate_impulse_noise(self, probability=0.01, amplitude=5):
        """Генерує імпульсний шум (негауссовий)."""
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), endpoint=False)
        noise = np.random.choice([0, amplitude], size=len(t), p=[1 - probability, probability])
        return t, noise

    def generate_colored_noise(self, color="pink", amplitude=1):
        """Генерує кольоровий шум (рожевий або червоний)."""
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), endpoint=False)
        white_noise = np.random.randn(len(t))

        if color == "pink":
            pink_filter = np.cumsum(white_noise)
            noise = amplitude * pink_filter / np.max(np.abs(pink_filter))
        elif color == "red":
            red_filter = np.cumsum(np.cumsum(white_noise))
            noise = amplitude * red_filter / np.max(np.abs(red_filter))
        else:
            raise ValueError("Unsupported noise color. Use 'pink' or 'red'.")

        return t, noise

    def generate_wifi_like_noise(self, num_carriers=64, bandwidth=20e6, amplitude=0.5):
        """Генерує шум, подібний до Wi-Fi сигналу."""
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), endpoint=False)
        noise = np.zeros_like(t)
        carrier_spacing = bandwidth / num_carriers

        for i in range(num_carriers):
            freq = (i - num_carriers // 2) * carrier_spacing
            noise += amplitude * np.cos(2 * np.pi * freq * t + np.random.uniform(0, 2 * np.pi))

        return t, noise

    def generate_polygauss_noise(self, components=3, weights=None, means=None, stds=None):
        """
        Generates Poly-Gaussian noise: a mixture of multiple Gaussian distributions.

        Parameters:
            components (int): Number of Gaussian components.
            weights (list or None): Weights for each Gaussian component (should sum to 1).
            means (list or None): Means of each Gaussian component.
            stds (list or None): Standard deviations of each Gaussian component.

        Returns:
            t (np.ndarray): Time vector.
            noise (np.ndarray): Generated poly-Gaussian noise.
        """
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), endpoint=False)
        n = len(t)

        # Default random parameters if not provided
        if weights is None:
            weights = np.random.dirichlet(np.ones(components))
        if means is None:
            means = np.random.uniform(-2, 2, components)
        if stds is None:
            stds = np.random.uniform(0.5, 1.5, components)

        # Choose component for each point
        component_choices = np.random.choice(components, size=n, p=weights)
        noise = np.array([
            np.random.normal(loc=means[c], scale=stds[c]) for c in component_choices
        ])

        return t, noise

    def generate_dataset(self):
        """Генерує датасет із сигналами та шумами."""
        clean_signals = []
        gaussian_signals = []
        non_gaussian_signals = []

        for _ in range(self.num_samples):
            # Випадковий вибір типу сигналу
            if random.choice([True, False]):
                _, signal = self.generate_qpsk_signal(symbol_rate=10)
            else:
                _, signal = self.generate_fsk_signal(bit_rate=10)

            clean_signals.append(signal)

            # Додавання гауссового шуму
            _, gaussian_noise = self.generate_gaussian_noise()
            gaussian_signals.append(signal + gaussian_noise)

            # Випадковий вибір і комбінація негауссових шумів
            non_gaussian_noise = np.zeros_like(signal)
            if random.choice([True, False]):
                _, impulse_noise = self.generate_impulse_noise()
                non_gaussian_noise += impulse_noise
            if random.choice([True, False]):
                _, pink_noise = self.generate_colored_noise(color="pink")
                non_gaussian_noise += pink_noise
            if random.choice([True, False]):
                _, red_noise = self.generate_colored_noise(color="red")
                non_gaussian_noise += red_noise
            if random.choice([True, False]):
                _, wifi_noise = self.generate_wifi_like_noise()
                non_gaussian_noise += wifi_noise
            if random.choice([True, False]):
                components = random.randint(2, 5)  # Random number of components between 2 and 5
                weights = np.random.dirichlet(np.ones(components))
                means = np.random.uniform(-2, 2, components)
                stds = np.random.uniform(0.3, 1.5, components)

                _, polygauss_noise = self.generate_polygauss_noise(
                    components=components,
                    weights=weights,
                    means=means,
                    stds=stds
                )
                non_gaussian_noise += polygauss_noise

            non_gaussian_signals.append(signal + non_gaussian_noise)

        return np.array(clean_signals), np.array(gaussian_signals), np.array(non_gaussian_signals)


class DatasetExplorer:
    def __init__(self, clean_signals, gaussian_dataset, non_gaussian_dataset):
        self.clean_signals = clean_signals
        self.gaussian_dataset = gaussian_dataset
        self.non_gaussian_dataset = non_gaussian_dataset

    def visualize_sample(self, idx, dataset_type="clean"):
        """Візуалізує сигнал із датасету."""
        if dataset_type == "clean":
            signal = self.clean_signals[idx]
        elif dataset_type == "gaussian":
            signal = self.gaussian_dataset[idx]
        elif dataset_type == "non_gaussian":
            signal = self.non_gaussian_dataset[idx]
        else:
            raise ValueError("Invalid dataset type. Use 'clean', 'gaussian', or 'non_gaussian'.")

        plt.figure(figsize=(12, 6))
        plt.plot(signal)
        plt.title(f"Signal {idx} from {dataset_type} dataset")
        plt.xlabel("Time (samples)")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.show()

    def save_dataset(self, filename, dataset_type="clean"):
        """Зберігає датасет у файл."""
        if dataset_type == "clean":
            np.save(filename, self.clean_signals)
        elif dataset_type == "gaussian":
            np.save(filename, self.gaussian_dataset)
        elif dataset_type == "non_gaussian":
            np.save(filename, self.non_gaussian_dataset)
        else:
            raise ValueError("Invalid dataset type. Use 'clean', 'gaussian', or 'non_gaussian'.")


# Генерація датасету
generator = SignalDatasetGenerator(num_samples=5000, sample_rate=2144, duration=1, freq_range=(10, 100))
clean_signals, gaussian_dataset, non_gaussian_dataset = generator.generate_dataset()

# Робота з датасетом
explorer = DatasetExplorer(clean_signals, gaussian_dataset, non_gaussian_dataset)
explorer.visualize_sample(0, dataset_type="clean")
explorer.visualize_sample(0, dataset_type="gaussian")
explorer.visualize_sample(0, dataset_type="non_gaussian")

# Збереження датасету
explorer.save_dataset("clean_signals.npy", dataset_type="clean")
explorer.save_dataset("gaussian_signals.npy", dataset_type="gaussian")
explorer.save_dataset("non_gaussian_signals.npy", dataset_type="non_gaussian")
