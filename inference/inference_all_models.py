import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import stft, istft
from models.autoencoder_unet import UnetAutoencoder
from models.time_series_trasformer import TimeSeriesTransformer
from models.wavelet import WaveletDenoising

# ----- Конфігурація -----
dataset_type = "non_gaussian"  # "gaussian" або "non_gaussian"
random_state = 42
sample_index = 0
signal_len = 2144
fs = 1024
nperseg = 128
noverlap = 96
pad = nperseg // 2

# шляхи до збережених ваг моделей
unet_weights_path = "../weights/UnetAutoencoder_non_gaussian_best.pth"
transformer_weights_path = "../weights/TimeSeriesTransformer_non_gaussian_best.pth"
wavelet_type = 'db8'
wavelet_level = None

# ----- Завантаження даних -----
noisy = np.load(f"../data_generation/{dataset_type}_signals.npy")
clean = np.load("../data_generation/clean_signals.npy")
assert noisy.shape == clean.shape

# Спліт даних
N = len(noisy)
indices = np.arange(N)
np.random.seed(random_state)
np.random.shuffle(indices)
test_indices = indices[int(0.7 * N):]

X_test = noisy[test_indices]
y_test = clean[test_indices]

test_noisy = X_test[sample_index]
test_clean = y_test[sample_index]

# ----- Wavelet Denoising -----
best_params = {
    "gaussian": {'wavelet': 'db6', 'level': 3, 'thresh_mode': 'soft', 'per_level': False, 'ext_mode': 'symmetric'},
    "non_gaussian": {'wavelet': 'db4', 'level': 4, 'thresh_mode': 'soft', 'per_level': False,
                     'ext_mode': 'periodization'}}

wavelet_denoiser = WaveletDenoising(**best_params[dataset_type])
wavelet_denoised = wavelet_denoiser.denoise(test_noisy)
wavelet_denoised = wavelet_denoised[:signal_len]

# ----- Transformer Denoising -----
transformer = TimeSeriesTransformer(input_dim=1)
transformer.load_state_dict(torch.load(transformer_weights_path, map_location='cpu'))
transformer.eval()

with torch.no_grad():
    transformer_input = torch.tensor(test_noisy, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    transformer_output = transformer(transformer_input).squeeze().numpy()

transformer_output = transformer_output[:signal_len]

# ----- UNet Autoencoder Denoising -----
# Підготовка вхідних даних для UNet
padded = np.pad(test_noisy, pad_width=pad, mode='reflect')
_, _, Zxx = stft(padded, fs=fs, nperseg=nperseg, noverlap=noverlap, boundary=None)
mag = np.abs(Zxx)
phase = np.angle(Zxx)
input_shape = mag.shape

unet = UnetAutoencoder(input_shape=input_shape)
unet.load_state_dict(torch.load(unet_weights_path, map_location='cpu'))
unet.eval()

with torch.no_grad():
    input_mag_tensor = torch.tensor(mag, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    output_mag = unet(input_mag_tensor).squeeze().numpy()

Zxx_denoised = output_mag * np.exp(1j * phase)
_, unet_output = istft(Zxx_denoised, fs=fs, nperseg=nperseg, noverlap=noverlap, input_onesided=True)
# Явно обрізаємо або доповнюємо результат до потрібної довжини
if len(unet_output) < signal_len:
    unet_output = np.pad(unet_output, (0, signal_len - len(unet_output)))
else:
    unet_output = unet_output[:signal_len]

# unet_output = unet_output[pad:pad + signal_len]
# unet_output = unet_output[:signal_len]

# ----- Візуалізація усіх моделей -----
time_axis = np.arange(signal_len) / fs

plt.figure(figsize=(15, 6))
plt.plot(time_axis, test_clean, label='а) Оригінальний сигнал', linewidth=2, color='black')
plt.plot(time_axis, test_noisy, label='b) Зашумлений сигнал', alpha=0.5, color='gray')
plt.plot(time_axis, wavelet_denoised, label='c) Відновлення Wavelet', linestyle='-.', linewidth=2)
plt.plot(time_axis, transformer_output, label='d) Відновлення Transformer', linestyle='--', linewidth=2)
plt.plot(time_axis, unet_output, label='e) Відновлення UAE', linestyle='-', linewidth=2)

plt.xlabel("Час (с)")
plt.ylabel("Амплітуда")
plt.title(f"Порівняння знешумлення сигналу (негаусові завади)")
plt.xlim([0.15, 0.25])
plt.ylim([-2, 2])
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
