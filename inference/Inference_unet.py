import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import stft, istft

from models.autoencoder_unet import UnetAutoencoder

# -----------------------------
# Конфіг (узгоджено з тренуванням)
# -----------------------------
fs = 1024
nperseg = 128
noverlap = 96
window = 'hann'
pad = nperseg // 2
signal_len = 2144
random_state = 42
sample_index = 0

UNET_WEIGHTS = {
    "gaussian":      "../weights/UnetAutoencoder_gaussian_best.pth",
    "non_gaussian":  "../weights/UnetAutoencoder_non_gaussian_best.pth",
}

# -----------------------------
# Утиліти
# -----------------------------
_EPS = 1e-12  # стабілізатор для логарифма

def stft_mag_phase_1d(x, fs, nperseg, noverlap, window='hann'):
    """STFT з віддзеркальним паддінгом; повертає (f, t, mag, phase)."""
    x_pad = np.pad(x, pad, mode="reflect")
    f, t, Zxx = stft(
        x_pad, fs=fs, nperseg=nperseg, noverlap=noverlap,
        window=window, boundary=None, padded=False, return_onesided=True
    )
    mag = np.abs(Zxx).astype(np.float32)
    phase = np.angle(Zxx).astype(np.float32)
    return f, t, mag, phase

def istft_from_mag_phase(mag, phase, fs, nperseg, noverlap, window='hann', target_len=None):
    """ISTFT + зняття паддінга та вирівнювання довжини до target_len."""
    Zxx = mag * np.exp(1j * phase)
    _, rec = istft(
        Zxx, fs=fs, nperseg=nperseg, noverlap=noverlap,
        window=window, input_onesided=True, boundary=None
    )
    # зняти паддінг
    if target_len is None:
        target_len = signal_len
    rec = rec[pad: pad + target_len]
    # підрівняти
    if len(rec) < target_len:
        rec = np.pad(rec, (0, target_len - len(rec)))
    elif len(rec) > target_len:
        rec = rec[:target_len]
    return rec

def spectrogram_db(x, fs, nperseg, noverlap, window='hann'):
    """Амплітудна спектрограма в dB з тим самим STFT-процесом, що й для моделі."""
    f, t, mag, _ = stft_mag_phase_1d(x, fs, nperseg, noverlap, window=window)
    S_db = 20.0 * np.log10(np.maximum(mag, _EPS))
    return f, t, S_db

def compute_common_cmap_range(S_list_db, mode='percentile', pmin=5, pmax=95, fixed_range=None):
    """
    Повертає (vmin, vmax) для спільної шкали.
    mode='percentile' -> за перцентилями (pmin, pmax)
    mode='fixed'      -> fixed_range = (vmin, vmax) у dB
    """
    if mode == 'fixed' and fixed_range is not None:
        return fixed_range
    flat = np.concatenate([S.ravel() for S in S_list_db])
    vmin = np.percentile(flat, pmin)
    vmax = np.percentile(flat, pmax)
    # страховка: не допустити vmin >= vmax
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin, vmax = np.min(flat), np.max(flat)
    return float(vmin), float(vmax)

def plot_triplet(clean, noisy, denoised, fs, nperseg, noverlap, window='hann',
                 title="", cmap='viridis', cmap_mode='percentile', fixed_db_range=None):
    """
    Малює одну фігуру з 3 спектрограмами (clean/noisy/denoised) зі спільною шкалою dB.
    - cmap_mode: 'percentile' (за замовчуванням) або 'fixed'
    - fixed_db_range: (vmin, vmax) якщо використовуєте 'fixed'
    """
    f_c, t_c, S_clean = spectrogram_db(clean,   fs, nperseg, noverlap, window)
    f_n, t_n, S_noisy = spectrogram_db(noisy,   fs, nperseg, noverlap, window)
    f_d, t_d, S_deno  = spectrogram_db(denoised,fs, nperseg, noverlap, window)

    # Єдина шкала кольорів
    vmin, vmax = compute_common_cmap_range(
        [S_clean, S_noisy, S_deno],
        mode=('fixed' if fixed_db_range else 'percentile'),
        fixed_range=fixed_db_range
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True, constrained_layout=True)
    panels = [
        ("а) Чистий",      f_c, t_c, S_clean),
        ("б) Зашумлений",  f_n, t_n, S_noisy),
        ("в) Знешумлений (U-Net)", f_d, t_d, S_deno),
    ]
    im = None
    for ax, (ttl, f, t, S) in zip(axes, panels):
        im = ax.pcolormesh(t, f, S, shading="gouraud", vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title(ttl)
        ax.set_xlabel("Час, с")
        ax.set_xlim(t[0], t[-1])
    axes[0].set_ylabel("Частота, Гц")
    fig.suptitle(title)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label("Амплітуда, dB")
    plt.show()

def load_unet_for_dataset(dataset_type, example_signal):
    """Створює U-Net із правильною input_shape для поточних STFT-параметрів."""
    _, _, mag, _ = stft_mag_phase_1d(example_signal, fs, nperseg, noverlap, window=window)
    input_shape = mag.shape  # (freq_bins, time_frames)
    model = UnetAutoencoder(input_shape=input_shape)
    sd = torch.load(UNET_WEIGHTS[dataset_type], map_location='cpu')
    model.load_state_dict(sd)
    model.eval()
    return model

def unet_denoise_signal(x_noisy, model):
    """STFT → U-Net (на амплітуді) → ISTFT; повертає часовий ряд довжини signal_len."""
    _, _, mag, phase = stft_mag_phase_1d(x_noisy, fs, nperseg, noverlap, window=window)
    x_in = torch.from_numpy(mag).unsqueeze(0).unsqueeze(0).float()
    with torch.no_grad():
        out_mag = model(x_in).squeeze(0).squeeze(0).cpu().numpy()
    rec = istft_from_mag_phase(out_mag, phase, fs, nperseg, noverlap, window=window, target_len=signal_len)
    return rec

# -----------------------------
# Побудова 3 спектрограм для одного датасету
# -----------------------------
def plot_triplet_for_sample(dataset_type="gaussian", sample_index=0, title_suffix=""):
    noisy = np.load(f"../dataset/{dataset_type}_signals.npy")
    clean = np.load(f"../dataset/clean_signals.npy")
    assert noisy.shape[1] == signal_len and clean.shape[1] == signal_len

    # як у тренуванні (70/15/15)
    N = len(noisy)
    rng = np.random.default_rng(random_state)
    idx = np.arange(N); rng.shuffle(idx)
    test_idx = idx[int(0.7 * N):]
    x_noisy = noisy[test_idx][sample_index]
    x_clean = clean[test_idx][sample_index]

    model = load_unet_for_dataset(dataset_type, example_signal=x_noisy)
    x_deno = unet_denoise_signal(x_noisy, model)

    plot_triplet(
        clean=x_clean, noisy=x_noisy, denoised=x_deno,
        fs=fs, nperseg=nperseg, noverlap=noverlap, window=window,
        title=f"Спектрограми ({dataset_type}) — чистий / зашумлений / U-Net",
        # ВАРІАНТ 1: фіксований діапазон dB (ретельно під статтю)
        # fixed_db_range=(-100, -20),
        # ВАРІАНТ 2 (за замовчуванням): перцентилі 5–95 у всіх трьох панелях
        cmap='viridis'
    )

# -----------------------------
# Приклад виклику
# -----------------------------
if __name__ == "__main__":
    # non_gaussian
    plot_triplet_for_sample(dataset_type="non_gaussian", sample_index=0)

    # # gaussian
    # plot_triplet_for_sample(dataset_type="gaussian", sample_index=0)
