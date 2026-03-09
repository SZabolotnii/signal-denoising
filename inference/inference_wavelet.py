import numpy as np
from models.wavelet import WaveletDenoising
from metrics import MeanSquaredError, MeanAbsoluteError, RootMeanSquaredError, SignalToNoiseRatio
import matplotlib.pyplot as plt

# ---- Configuration ----
dataset_type = "non_gaussian"  # or "non_gaussian"
random_state = 42
wavelet = 'db4'
level = 4
sample_index = 0  # Change to visualize another test sample

# ---- Load datasets and split identically to ML models ----
noisy = np.load(f"../data_generation/{dataset_type}_signals.npy")
clean = np.load("../data_generation/clean_signals.npy")

best_params = {
    "gaussian": {'wavelet': 'db6', 'level': 3, 'thresh_mode': 'soft', 'per_level': False, 'ext_mode': 'symmetric'},
    "non_gaussian": {'wavelet': 'db4', 'level': 4, 'thresh_mode': 'soft', 'per_level': False,
                     'ext_mode': 'periodization'}}

signal_len = noisy.shape[1]
assert noisy.shape == clean.shape

# Make same split
N = len(noisy)
val_len = int(0.15 * N)
test_len = int(0.15 * N)
train_len = N - val_len - test_len

# Reproducible split
indices = np.arange(N)
rng = np.random.default_rng(random_state)
rng.shuffle(indices)
train_indices = indices[:train_len]
val_indices = indices[train_len:train_len + val_len]
test_indices = indices[train_len + val_len:]

X_test = noisy[test_indices]
y_test = clean[test_indices]

# ---- Denoising and metrics ----
wavelet_denoiser = WaveletDenoising(**best_params[dataset_type])

all_metrics = {
    "MSE": [],
    "MAE": [],
    "RMSE": [],
    "SNR": [],
}

denoised_signals = []

for x_noisy, x_clean in zip(X_test, y_test):
    x_denoised = wavelet_denoiser.denoise(x_noisy)[:signal_len]  # trim/pad as needed
    # Padding or cropping if necessary
    if len(x_denoised) < signal_len:
        x_denoised = np.pad(x_denoised, (0, signal_len - len(x_denoised)))
    elif len(x_denoised) > signal_len:
        x_denoised = x_denoised[:signal_len]
    denoised_signals.append(x_denoised)
    # Metrics
    all_metrics["MSE"].append(MeanSquaredError.calculate(x_clean, x_denoised))
    all_metrics["MAE"].append(MeanAbsoluteError.calculate(x_clean, x_denoised))
    all_metrics["RMSE"].append(RootMeanSquaredError.calculate(x_clean, x_denoised))
    all_metrics["SNR"].append(SignalToNoiseRatio.calculate(x_clean, x_denoised))

# Aggregate metrics over test set
print("Wavelet Denoising Test Set Metrics:")
print(f"Mean MSE:  {np.mean(all_metrics['MSE']):.6f}")
print(f"Mean MAE:  {np.mean(all_metrics['MAE']):.6f}")
print(f"Mean RMSE: {np.mean(all_metrics['RMSE']):.6f}")
print(f"Mean SNR:  {np.mean(all_metrics['SNR']):.2f} dB")

# ---- Visualization for a sample ----
idx = sample_index
plt.figure(figsize=(14, 6))
plt.plot(y_test[idx], label="Clean Signal", linewidth=2)
plt.plot(X_test[idx], label="Noisy Signal", alpha=0.5)
plt.plot(denoised_signals[idx], label="Wavelet Denoised", linestyle='--', linewidth=2)
plt.title(f"Wavelet Denoising - Test sample #{idx}")
plt.legend()
plt.ylim([-2,2])
plt.xlim([0,400])
plt.grid(True)
plt.tight_layout()
plt.show()
