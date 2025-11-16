import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import stft, istft
from models.autoencoder_unet_v2 import UnetAutoencoder
from models.time_series_trasformer import TimeSeriesTransformer
from models.wavelet import WaveletDenoising

# ----- Конфігурація -----
dataset_type = "gaussian"  # "gaussian" або "non_gaussian"
random_state = 42
sample_index = 0
signal_len = 2144
fs = 1024
nperseg = 128
noverlap = 96
pad = nperseg // 2

# шляхи до збережених ваг моделей
unet_weights_path = f"../weights/UnetAutoencoder_{dataset_type}_best.pth"
transformer_weights_path = f"../weights/TimeSeriesTransformer_{dataset_type}_best.pth"
wavelet_type = 'db8'
wavelet_level = None

# ----- Завантаження даних -----
noisy = np.load(f"../dataset/{dataset_type}_signals.npy")
clean = np.load("../dataset/clean_signals.npy")
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
plt.title(f"Порівняння знешумлення сигналу ({dataset_type} завади)")
plt.xlim([0.15, 0.25])
plt.ylim([-2, 2])
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


from metrics import MeanSquaredError, MeanAbsoluteError, RootMeanSquaredError, SignalToNoiseRatio

# ======= УТИЛІТИ ДЛЯ ОЦІНКИ =======
def crop_or_pad(x, target_len):
    """Приводить сигнал x до довжини target_len (обрізання/доповнення нулями)."""
    if len(x) < target_len:
        return np.pad(x, (0, target_len - len(x)))
    elif len(x) > target_len:
        return x[:target_len]
    return x

def compute_all_metrics(clean_arr, pred_arr):
    """Обчислює всі метрики з вашого класу metrics.* для 1D масивів однакової довжини."""
    return {
        "MSE": MeanSquaredError.calculate(clean_arr, pred_arr),
        "MAE": MeanAbsoluteError.calculate(clean_arr, pred_arr),
        "RMSE": RootMeanSquaredError.calculate(clean_arr, pred_arr),
        "SNR": SignalToNoiseRatio.calculate(clean_arr, pred_arr),
    }

# Допоміжні обчислювачі для кожної моделі (щоб перевикористати в циклі)
def wavelet_denoise_signal(x_noisy):
    x_den = wavelet_denoiser.denoise(x_noisy)
    return crop_or_pad(x_den, signal_len)

@torch.no_grad()
def transformer_denoise_signal(x_noisy):
    x_in = torch.tensor(x_noisy, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
    x_out = transformer(x_in).squeeze().cpu().numpy()                             # [T]
    return crop_or_pad(x_out, signal_len)

@torch.no_grad()
def unet_denoise_signal(x_noisy):
    # STFT з віддзеркальним паддінгом (такими ж параметрами як вище)
    padded = np.pad(x_noisy, pad_width=pad, mode='reflect')
    _, _, Zxx = stft(padded, fs=fs, nperseg=nperseg, noverlap=noverlap, boundary=None)
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)

    # Прогін через U-Net по амплітуді
    inp = torch.tensor(mag, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,F,T]
    out_mag = unet(inp).squeeze().cpu().numpy()                             # [F,T]

    # ISTFT + вирівнювання довжини
    Zxx_denoised = out_mag * np.exp(1j * phase)
    _, rec = istft(Zxx_denoised, fs=fs, nperseg=nperseg, noverlap=noverlap, input_onesided=True)
    # зняти паддінг (повернутись до оригінальної довжини) та підрівняти
    rec = rec[pad:pad + signal_len]
    return crop_or_pad(rec, signal_len)

# ======= ОБЧИСЛЕННЯ МЕТРИК НА ВСЬОМУ ТЕСТ-НАБОРІ =======
wavelet_metrics = {"MSE": [], "MAE": [], "RMSE": [], "SNR": []}
transformer_metrics = {"MSE": [], "MAE": [], "RMSE": [], "SNR": []}
unet_metrics = {"MSE": [], "MAE": [], "RMSE": [], "SNR": []}

for i in range(len(X_test)):
    x_noisy = X_test[i]
    x_clean = y_test[i]

    # Wavelet
    pred_w = wavelet_denoise_signal(x_noisy)
    m_w = compute_all_metrics(x_clean, pred_w)
    for k, v in m_w.items():
        wavelet_metrics[k].append(v)

    # Transformer
    pred_t = transformer_denoise_signal(x_noisy)
    m_t = compute_all_metrics(x_clean, pred_t)
    for k, v in m_t.items():
        transformer_metrics[k].append(v)

    # U-Net
    pred_u = unet_denoise_signal(x_noisy)
    m_u = compute_all_metrics(x_clean, pred_u)
    for k, v in m_u.items():
        unet_metrics[k].append(v)

# Середні значення по тестовому набору
def mean_dict(d):
    return {k: float(np.mean(v)) for k, v in d.items()}

wavelet_mean = mean_dict(wavelet_metrics)
transformer_mean = mean_dict(transformer_metrics)
unet_mean = mean_dict(unet_metrics)

print("\n=== Test-set metrics (mean over test set) ===")
print(f"Wavelet     | MSE={wavelet_mean['MSE']:.6f}  MAE={wavelet_mean['MAE']:.6f}  "
      f"RMSE={wavelet_mean['RMSE']:.6f}  SNR={wavelet_mean['SNR']:.2f} dB")
print(f"Transformer | MSE={transformer_mean['MSE']:.6f}  MAE={transformer_mean['MAE']:.6f}  "
      f"RMSE={transformer_mean['RMSE']:.6f}  SNR={transformer_mean['SNR']:.2f} dB")
print(f"U-Net (UAE) | MSE={unet_mean['MSE']:.6f}  MAE={unet_mean['MAE']:.6f}  "
      f"RMSE={unet_mean['RMSE']:.6f}  SNR={unet_mean['SNR']:.2f} dB")

# (необовʼязково) невелика табличка в консолі:
try:
    import pandas as pd
    df = pd.DataFrame([
        {"Model": "Wavelet", **wavelet_mean},
        {"Model": "Transformer", **transformer_mean},
        {"Model": "U-Net (UAE)", **unet_mean},
    ])
    print("\nSummary table:\n", df.to_string(index=False))
except Exception:
    pass

