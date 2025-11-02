import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import stft, istft

from models.autoencoder_unet import UnetAutoencoder

# -----------------------------
# Конфіг
# -----------------------------
fs = 1024
nperseg = 128
noverlap = 96
pad = nperseg // 2
signal_len = 2144
random_state = 42
sample_index = 0

# Вкажіть шляхи до ваг U-Net для кожного типу датасету
UNET_WEIGHTS = {
    "gaussian":      "../weights/UnetAutoencoder_gaussian_best.pth",
    "non_gaussian":  "../weights/UnetAutoencoder_non_gaussian_best.pth",
}

# -----------------------------
# Утиліти
# -----------------------------
def stft_mag_phase_1d(x, fs, nperseg, noverlap):
    """STFT з віддзеркальним паддінгом; повертає (mag, phase, Zxx)."""
    x_pad = np.pad(x, pad, mode="reflect")
    f, t, Zxx = stft(x_pad, fs=fs, nperseg=nperseg, noverlap=noverlap, boundary=None)
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)
    return f, t, mag, phase, Zxx

def istft_from_mag_phase(mag, phase, fs, nperseg, noverlap, target_len):
    """ISTFT з відтинанням паддінга та вирівнюванням довжини."""
    Zxx = mag * np.exp(1j * phase)
    _, rec = istft(Zxx, fs=fs, nperseg=nperseg, noverlap=noverlap, input_onesided=True)
    # зняти паддінг
    rec = rec[pad: pad + target_len]
    # підрівняти
    if len(rec) < target_len:
        rec = np.pad(rec, (0, target_len - len(rec)))
    elif len(rec) > target_len:
        rec = rec[:target_len]
    return rec

def spectrogram_db(x, fs, nperseg, noverlap):
    """Амплітудний спектр у dB з тим самим STFT-процесом, що для моделі."""
    _, t, mag, _, _ = stft_mag_phase_1d(x, fs, nperseg, noverlap)
    # лог-шкала у dB (додаємо eps, щоб уникнути -inf)
    S_db = 20.0 * np.log10(np.maximum(mag, 1e-12))
    return t, S_db

def plot_triplet(clean, noisy, denoised, fs, nperseg, noverlap, title=""):
    """Одна фігура з 3 спектрограмами (clean / noisy / denoised) із загальною шкалою."""
    t_c, S_clean = spectrogram_db(clean,   fs, nperseg, noverlap)
    t_n, S_noisy = spectrogram_db(noisy,   fs, nperseg, noverlap)
    t_d, S_deno  = spectrogram_db(denoised,fs, nperseg, noverlap)

    # спільна шкала кольорів
    vmin = np.percentile(np.concatenate([S_clean.ravel(), S_noisy.ravel(), S_deno.ravel()]), 5)
    vmax = np.percentile(np.concatenate([S_clean.ravel(), S_noisy.ravel(), S_deno.ravel()]), 95)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True, constrained_layout=True)
    for ax, S, t, ttl in zip(
        axes,
        [S_clean, S_noisy, S_deno],
        [t_c, t_n, t_d],
        ["а) Чистий", "б) Зашумлений", "в) Знешумлений (U-Net)"],
    ):
        # частоти з STFT: nperseg точок -> вісь частот однакова для всіх викликів
        freqs = np.linspace(0, fs/2, S.shape[0])  # onesided
        im = ax.pcolormesh(t, freqs, S, shading="gouraud", vmin=vmin, vmax=vmax)
        ax.set_title(ttl)
        ax.set_xlabel("Час, с")
        ax.set_xlim(t[0], t[-1])
    axes[0].set_ylabel("Частота, Гц")
    fig.suptitle(title)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)
    cbar.set_label("Амплітуда, dB")
    plt.show()

def load_unet_for_dataset(dataset_type, example_signal):
    """Створює U-Net з правильною input_shape під конкретний датасет/параметри STFT."""
    # отримати форму спектрограми на основі прикладу
    _, _, mag, _, _ = stft_mag_phase_1d(example_signal, fs, nperseg, noverlap)
    input_shape = mag.shape  # (freq_bins, time_frames)
    model = UnetAutoencoder(input_shape=input_shape)
    sd = torch.load(UNET_WEIGHTS[dataset_type], map_location='cpu')
    model.load_state_dict(sd)
    model.eval()
    return model

def unet_denoise_signal(x_noisy, model):
    """STFT -> U-Net (на амплітуді) -> ISTFT; повертає часовий ряд."""
    # STFT
    _, _, mag, phase, _ = stft_mag_phase_1d(x_noisy, fs, nperseg, noverlap)
    # у модель: (B,C,H,W) = (1,1,freq,time)
    x_in = torch.from_numpy(mag).unsqueeze(0).unsqueeze(0).float()
    with torch.no_grad():
        out_mag = model(x_in).squeeze(0).squeeze(0).cpu().numpy()
    # ISTFT
    rec = istft_from_mag_phase(out_mag, phase, fs, nperseg, noverlap, target_len=signal_len)
    return rec

# -----------------------------
# Побудова 3 спектрограм для одного датасету
# -----------------------------
def plot_triplet_for_sample(dataset_type="gaussian", sample_index=0, title_suffix=""):
    # завантаження даних
    noisy = np.load(f"../dataset/{dataset_type}_signals.npy")
    clean = np.load(f"../dataset/clean_signals.npy")
    assert noisy.shape[1] == signal_len and clean.shape[1] == signal_len

    # відокремимо спліт як при тренуванні (70/15/15)
    N = len(noisy)
    rng = np.random.default_rng(random_state)
    idx = np.arange(N); rng.shuffle(idx)
    test_idx = idx[int(0.7*N):]  # останні 30%
    x_noisy = noisy[test_idx][sample_index]
    x_clean = clean[test_idx][sample_index]

    # створити модель під форму спектрограми й розмірності
    model = load_unet_for_dataset(dataset_type, example_signal=x_noisy)

    # деноїз
    x_deno = unet_denoise_signal(x_noisy, model)

    # побудова 3 спектрограм
    plot_triplet(
        clean=x_clean,
        noisy=x_noisy,
        denoised=x_deno,
        fs=fs, nperseg=nperseg, noverlap=noverlap,
        title=f"Спектрограми ({dataset_type}) {title_suffix}".strip()
    )

# -----------------------------
# Приклад виклику
# -----------------------------
if __name__ == "__main__":
    # Варіант 1: для негаусових завад
    plot_triplet_for_sample(dataset_type="non_gaussian", sample_index=0,
                            title_suffix="— чистий / зашумлений / U-Net")

    # Варіант 2: для гаусових завад (розкоментуйте за потреби)
    # plot_triplet_for_sample(dataset_type="gaussian", sample_index=0,
    #                         title_suffix="— чистий / зашумлений / U-Net")
