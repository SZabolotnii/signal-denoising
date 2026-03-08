import numpy as np
import torch
import matplotlib.pyplot as plt

from models.time_series_trasformer import TimeSeriesTransformer  # Adjust import as needed

# --- Configurable Parameters ---
dataset_type = "gaussian"          # or "non_gaussian"
signal_len = 1000
random_state = 42
batch_size = 32
input_dim = 1
model_weights_path = "../weights/TimeSeriesTransformer_gaussian_best.pth"
sample_index = 0  # CHANGE THIS TO CHOOSE TEST ITEM

# --- Load Data (identical to trainer) ---
noisy = np.load(f"../data_generation/{dataset_type}_signals.npy")
clean = np.load("../data_generation/clean_signals.npy")

assert noisy.shape[1] == signal_len and clean.shape[1] == signal_len, "Signal length mismatch!"

X = torch.tensor(noisy[:, :signal_len], dtype=torch.float32).unsqueeze(-1)  # shape [N, T, 1]
y = torch.tensor(clean[:, :signal_len], dtype=torch.float32).unsqueeze(-1)  # shape [N, T, 1]

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

# --- Load Model (dimensions must match training) ---
model = TimeSeriesTransformer(input_dim=input_dim)
# Load the state dict
state_dict = torch.load(model_weights_path, map_location='cpu')

# Remove "model." prefix from all keys if present
new_state_dict = {}
for k, v in state_dict.items():
    new_k = k.replace("model.", "") if k.startswith("model.") else k
    new_state_dict[new_k] = v

model.load_state_dict(new_state_dict)
model.eval()

# --- Choose Test Item ---
test_X, test_y = test_set[sample_index]
test_X_in = test_X.unsqueeze(0)  # shape [1, T, 1]
test_y = test_y.squeeze().numpy()  # shape [T,]

# --- Denoising (model inference) ---
with torch.no_grad():
    # Model expects input of shape [B, T, 1]
    denoised = model(test_X_in).squeeze(0).squeeze(-1).cpu().numpy()  # shape [T,]

# --- Visualization ---
t = np.arange(signal_len)
plt.figure(figsize=(12, 5))
plt.plot(t, test_y, label='Clean', linewidth=2)
plt.plot(t, test_X.squeeze().numpy(), label='Noisy', alpha=0.6)
plt.plot(t, denoised, label='Denoised', linestyle='--', linewidth=2)
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.title(f"Sample #{sample_index} from test set ({dataset_type})")
plt.legend()
plt.ylim([-2,2])
plt.xlim([0,400])
plt.grid(True)
plt.tight_layout()
plt.show()
