import numpy as np
import torch
from scipy.signal import stft, istft
import matplotlib.pyplot as plt

from models.autoencoder_resnet import ResNetAutoencoder  # <- your ResNet AE model

# --- Configurable Parameters ---
dataset_type = "gaussian"         # or "non_gaussian"
signal_len = 2144
fs = 1024
nperseg = 128
noverlap = 96
pad = nperseg // 2
random_state = 42
model_weights_path = "../weights/ResNetAutoencoder_gaussian_best.pth"
sample_index = 0  # CHANGE THIS TO CHOOSE TEST ITEM

# --- Load Data (should be identical to trainer) ---
noisy = np.load(f"../data_generation/{dataset_type}_signals.npy")
clean = np.load("../data_generation/clean_signals.npy")

assert noisy.shape[1] == signal_len and clean.shape[1] == signal_len, "Signal length mismatch!"

X = torch.tensor(noisy[:, :signal_len], dtype=torch.float32).unsqueeze(1)
y = torch.tensor(clean[:, :signal_len], dtype=torch.float32).unsqueeze(1)

dataset = torch.utils.data.TensorDataset(X, y)
total_len = len(dataset)
val_len = int(0.15 * total_len)
test_len = int(0.15 * total_len)
train_len = total_len - val_len - test_len

# Consistent split
train_set, val_set, test_set = torch.utils.data.random_split(
    dataset, [train_len, val_len, test_len],
    generator=torch.Generator().manual_seed(random_state)
)

# --- Load Model (must match trained dimensions) ---
# Determine freq_bins and time_frames using the same logic as training
with torch.no_grad():
    test_example = X[0:1]
    padded = torch.nn.functional.pad(test_example, (pad, pad), mode="reflect")
    s = padded.squeeze(1).cpu().numpy()[0]
    _, _, Zxx = stft(s, fs=fs, nperseg=nperseg, noverlap=noverlap)
    freq_bins, time_frames = Zxx.shape
    input_shape = (freq_bins, time_frames)

model = ResNetAutoencoder(input_shape=input_shape)
model.load_state_dict(torch.load(model_weights_path, map_location='cpu'))
model.eval()

# --- Choose Test Item ---
test_X, test_y = test_set[sample_index]
test_X = test_X.unsqueeze(0)  # shape [1, 1, T]
test_y = test_y.squeeze().numpy()  # shape [T,]

# --- Denoising (STFT -> ResNet AE -> ISTFT) ---
with torch.no_grad():
    padded = torch.nn.functional.pad(test_X, (pad, pad), mode="reflect")
    s = padded.squeeze(0).squeeze(0).numpy()
    _, _, Zxx = stft(s, fs=fs, nperseg=nperseg, noverlap=noverlap)
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)
    mag_tensor = torch.tensor(mag, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    out_mag = model(mag_tensor)
    out_mag = out_mag.squeeze(0).squeeze(0).numpy()
    Zxx_denoised = out_mag * np.exp(1j * phase)
    _, rec = istft(Zxx_denoised, fs=fs, nperseg=nperseg, noverlap=noverlap)
    rec = rec[pad:pad+signal_len]
    if len(rec) < signal_len:
        rec = np.pad(rec, (0, signal_len - len(rec)))
    elif len(rec) > signal_len:
        rec = rec[:signal_len]

# --- Visualization ---
t = np.arange(signal_len) / fs
plt.figure(figsize=(12, 5))
plt.plot(t, test_y, label='Clean', linewidth=2)
plt.plot(t, test_X.squeeze().numpy(), label='Noisy', alpha=0.6)
plt.plot(t, rec, label='Denoised', linestyle='--', linewidth=2)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.ylim([-2, 2])
plt.xlim([0, 0.8])
plt.title(f"Sample #{sample_index} from test set ({dataset_type})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
