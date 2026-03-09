# compare_models_with_new_unet.py
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import stft, istft

# ======= МОДЕЛІ =======
from models.autoencoder_unet_v2 import UnetAutoencoder   # НОВА U-Net
from models.time_series_trasformer import TimeSeriesTransformer
from models.wavelet import WaveletDenoising

# ======= МЕТРИКИ =======
from metrics import MeanSquaredError, MeanAbsoluteError, RootMeanSquaredError, SignalToNoiseRatio

# ----- Конфігурація -----
dataset_type = "non_gaussian"  # "gaussian" або "non_gaussian"
random_state = 42
sample_index = 0
signal_len = 2144

# шляхи
RAW_DIR       = "../dataset"
SPECTRO_ROOT  = f"../spectro_dataset/{dataset_type}"
unet_weights_path        = f"../weights/UnetAutoencoder_{dataset_type}_best.pth"
transformer_weights_path = f"../weights/TimeSeriesTransformer_{dataset_type}_best.pth"

# wavelet best params
best_params = {
    "gaussian":      {'wavelet': 'db6', 'level': 3, 'thresh_mode': 'soft', 'per_level': False, 'ext_mode': 'symmetric'},
    "non_gaussian":  {'wavelet': 'db4', 'level': 4, 'thresh_mode': 'soft', 'per_level': False, 'ext_mode': 'periodization'},
}

# =========================
# Допоміжні утиліти
# =========================
def crop_or_pad(x, target_len):
    if len(x) < target_len:
        return np.pad(x, (0, target_len - len(x)))
    elif len(x) > target_len:
        return x[:target_len]
    return x

def compute_all_metrics(clean_arr, pred_arr):
    return {
        "MSE":  MeanSquaredError.calculate(clean_arr, pred_arr),
        "MAE":  MeanAbsoluteError.calculate(clean_arr, pred_arr),
        "RMSE": RootMeanSquaredError.calculate(clean_arr, pred_arr),
        "SNR":  SignalToNoiseRatio.calculate(clean_arr, pred_arr),
    }

def istft_unpad(mag, phase, fs, nperseg, noverlap, window, pad):
    Z = mag * np.exp(1j * phase)
    _, x_pad = istft(Z, fs=fs, nperseg=nperseg, noverlap=noverlap,
                     window=window, input_onesided=True, boundary=None)
    return x_pad[pad:-pad] if pad > 0 and x_pad.size >= 2 * pad else x_pad

# =========================
# Завантаження даних та мета
# =========================
# сирі дані (часова область)
noisy_all = np.load(os.path.join(RAW_DIR, f"{dataset_type}_signals.npy"))
clean_all = np.load(os.path.join(RAW_DIR, "clean_signals.npy"))
assert noisy_all.shape == clean_all.shape

# мета для STFT/спліту/нормалізації
import json
with open(os.path.join(SPECTRO_ROOT, "meta.json"), "r") as f:
    meta = json.load(f)
fs       = int(meta["fs"])
nperseg  = int(meta["nperseg"])
noverlap = int(meta["noverlap"])
window   = meta.get("window", "hann")
pad      = int(meta.get("pad", nperseg // 2))
test_idx = np.array(meta["split_indices"]["test_idx"], dtype=np.int64)

# статистики нормалізації для U-Net (лог-амплітуда train_X)
train_X = np.load(os.path.join(SPECTRO_ROOT, "train_X.npy"))  # log1p|N|
train_mean = float(train_X.mean())
train_std  = float(train_X.std() + 1e-8)

# формуємо тестові набори з сирих даних за тим самим сплітом
X_test = noisy_all[test_idx]
y_test = clean_all[test_idx]

# =========================
# Ініціалізація моделей
# =========================
# Wavelet
wavelet_denoiser = WaveletDenoising(**best_params[dataset_type])

# Transformer
transformer = TimeSeriesTransformer(input_dim=1)
transformer.load_state_dict(torch.load(transformer_weights_path, map_location='cpu'))
transformer.eval()

# U-Net (нова)
# Щоб побудувати правильний розмір входу (F, T'), робимо STFT одного прикладу
padded_probe = np.pad(X_test[0], pad_width=pad, mode='reflect')
_, _, Z_probe = stft(padded_probe, fs=fs, nperseg=nperseg, noverlap=noverlap, boundary=None)
F, Tprime = Z_probe.shape
unet = UnetAutoencoder(input_shape=(F, Tprime))
unet.load_state_dict(torch.load(unet_weights_path, map_state_dict := {'map_location':'cpu'} if isinstance(unet_weights_path, str) else {}))
unet.eval()

# =========================
# Обгортки-денойзери
# =========================
def wavelet_denoise_signal(x_noisy):
    return crop_or_pad(wavelet_denoiser.denoise(x_noisy), signal_len)

@torch.no_grad()
def transformer_denoise_signal(x_noisy):
    x_in = torch.tensor(x_noisy, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # [1,T,1]
    x_out = transformer(x_in).squeeze().cpu().numpy()
    return crop_or_pad(x_out, signal_len)

@torch.no_grad()
def unet_denoise_signal(x_noisy):
    # STFT з reflect-паддінгом
    x_pad = np.pad(x_noisy, pad, mode='reflect')
    _, _, Z = stft(x_pad, fs=fs, nperseg=nperseg, noverlap=noverlap, boundary=None, window=window)
    mag = np.abs(Z)              # |N| (лінійно)
    phase = np.angle(Z)

    # інпут для моделі: X_in = zscore(log1p(|N|))
    X_log = np.log1p(mag).astype(np.float32)
    X_in  = (X_log - train_mean) / train_std
    X_in_t = torch.from_numpy(X_in).unsqueeze(0).unsqueeze(0)  # [1,1,F,T']

    pred_mask = unet(X_in_t).squeeze().cpu().numpy()           # [F,T'], σ∈[0,1]

    den_mag = pred_mask * mag                                  # |Ĉ| = M·|N|
    x_den = istft_unpad(den_mag, phase, fs, nperseg, noverlap, window, pad).astype(np.float32)

    return crop_or_pad(x_den, signal_len)

# =========================
# Візуалізація одного прикладу
# =========================
sample_noisy = X_test[sample_index]
sample_clean = y_test[sample_index]

pred_w = wavelet_denoise_signal(sample_noisy)
pred_t = transformer_denoise_signal(sample_noisy)
pred_u = unet_denoise_signal(sample_noisy)

time_axis = np.arange(signal_len) / fs
plt.figure(figsize=(15, 6))
plt.plot(time_axis, sample_clean, label='a) Оригінальний', linewidth=2, color='black')
plt.plot(time_axis, sample_noisy, label='b) Зашумлений', alpha=0.5, color='gray')
plt.plot(time_axis, pred_w, label='c) Wavelet', linestyle='-.', linewidth=2)
plt.plot(time_axis, pred_t, label='d) Transformer', linestyle='--', linewidth=2)
plt.plot(time_axis, pred_u, label='e) U-Net (new)', linestyle='-', linewidth=2)
plt.xlabel("Час (с)"); plt.ylabel("Амплітуда")
plt.title(f"Порівняння знешумлення сигналу ({dataset_type})")
plt.xlim([0.15, 0.25]); plt.ylim([-2, 2])
plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

# =========================
# Метрики на всьому test
# =========================
wavelet_metrics    = {"MSE": [], "MAE": [], "RMSE": [], "SNR": []}
transformer_metrics= {"MSE": [], "MAE": [], "RMSE": [], "SNR": []}
unet_metrics       = {"MSE": [], "MAE": [], "RMSE": [], "SNR": []}

for i in range(len(X_test)):
    x_noisy = X_test[i]
    x_clean = y_test[i]

    # Wavelet
    m = compute_all_metrics(x_clean, wavelet_denoise_signal(x_noisy))
    for k,v in m.items(): wavelet_metrics[k].append(v)

    # Transformer
    m = compute_all_metrics(x_clean, transformer_denoise_signal(x_noisy))
    for k,v in m.items(): transformer_metrics[k].append(v)

    # U-Net (NEW PATH)
    m = compute_all_metrics(x_clean, unet_denoise_signal(x_noisy))
    for k,v in m.items(): unet_metrics[k].append(v)

def mean_dict(d): return {k: float(np.mean(v)) for k,v in d.items()}

wavelet_mean     = mean_dict(wavelet_metrics)
transformer_mean = mean_dict(transformer_metrics)
unet_mean        = mean_dict(unet_metrics)

print("\n=== Test-set metrics (mean over TEST split) ===")
print(f"Wavelet     | MSE={wavelet_mean['MSE']:.6f}  MAE={wavelet_mean['MAE']:.6f}  RMSE={wavelet_mean['RMSE']:.6f}  SNR={wavelet_mean['SNR']:.2f} dB")
print(f"Transformer | MSE={transformer_mean['MSE']:.6f}  MAE={transformer_mean['MAE']:.6f}  RMSE={transformer_mean['RMSE']:.6f}  SNR={transformer_mean['SNR']:.2f} dB")
print(f"U-Net (new) | MSE={unet_mean['MSE']:.6f}  MAE={unet_mean['MAE']:.6f}  RMSE={unet_mean['RMSE']:.6f}  SNR={unet_mean['SNR']:.2f} dB")
