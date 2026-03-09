# inference_no_cli_linear.py
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.signal import istft, stft

# =========================
# CONFIG
# =========================
DATA_ROOT   = "../spectro_dataset"
DATASET     = "gaussian"
WEIGHTS     = "../weights/UnetAutoencoder_gaussian_best.pth"
RAW_DIR     = "../../dataset"
PLOTS_OUT   = "./inference_plots"
BATCH_SIZE  = 32
USE_CPU     = False

# =========================
# IMPORTS
# =========================
from models.autoencoder_unet import UnetAutoencoder
from metrics import (
    MeanSquaredError,
    MeanAbsoluteError,
    RootMeanSquaredError,
    SignalToNoiseRatio,
)

# =========================
# HELPERS
# =========================
def istft_unpad(mag, phase, fs, nperseg, noverlap, window, pad):
    """
    ISTFT із поверненням reflect-паддінгу, що додавався перед прямим STFT.
    mag, phase: (F, T') — лінійна амплітуда та фаза.
    """
    Z = mag * np.exp(1j * phase)
    _, x_pad = istft(
        Z, fs=fs, nperseg=nperseg, noverlap=noverlap,
        window=window, input_onesided=True, boundary=None
    )
    if pad > 0 and x_pad.size >= 2 * pad:
        x = x_pad[pad:-pad]
    else:
        x = x_pad
    return x.astype(np.float32)

def stft_mag_db_timefreq(x, fs, nperseg, noverlap, window="hann", pad=None, eps=1e-8):
    """
    Гарна спектрограма для відображення:
    - STFT прямо з часової хвилі (опційно reflect-pad),
    - Амплітуда → dB: 20*log10(|X|+eps),
    - Повертає (f, t, S_db).
    """
    if pad is None:
        pad = nperseg // 2
    x_pad = np.pad(x, pad, mode="reflect") if pad > 0 else x
    f, t, Z = stft(x_pad, fs=fs, nperseg=nperseg, noverlap=noverlap,
                   window=window, boundary=None, padded=False)
    S = np.abs(Z)
    S_db = 20.0 * np.log10(S + eps)
    return f, t, S_db

def plot_triptych_clean_noisy_denoised(y_clean, y_noisy, y_denoised,
                                       fs, nperseg, noverlap, window, pad,
                                       title="Спектрограми (test) — чистий / зашумлений / знешумлений (U-Net)",
                                       save_path=None):
    # Спектрограми в dB напряму з часу:
    f_c, t_c, S_c = stft_mag_db_timefreq(y_clean,    fs, nperseg, noverlap, window, pad)
    f_n, t_n, S_n = stft_mag_db_timefreq(y_noisy,    fs, nperseg, noverlap, window, pad)
    f_d, t_d, S_d = stft_mag_db_timefreq(y_denoised, fs, nperseg, noverlap, window, pad)

    vmin = min(S_c.min(), S_n.min(), S_d.min())
    vmax = max(S_c.max(), S_n.max(), S_d.max())

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    ims = []

    im = axes[0].imshow(S_c, origin="lower", aspect="auto",
                        extent=[t_c[0], t_c[-1], f_c[0], f_c[-1]],
                        vmin=vmin, vmax=vmax)
    axes[0].set_title("a) Чистий")
    axes[0].set_xlabel("час, с"); axes[0].set_ylabel("част., Гц")
    ims.append(im)

    im = axes[1].imshow(S_n, origin="lower", aspect="auto",
                        extent=[t_n[0], t_n[-1], f_n[0], f_n[-1]],
                        vmin=vmin, vmax=vmax)
    axes[1].set_title("б) Зашумлений")
    axes[1].set_xlabel("час, с"); axes[1].set_ylabel("част., Гц")
    ims.append(im)

    im = axes[2].imshow(S_d, origin="lower", aspect="auto",
                        extent=[t_d[0], t_d[-1], f_d[0], f_d[-1]],
                        vmin=vmin, vmax=vmax)
    axes[2].set_title("в) Знешумлений (U-Net)")
    axes[2].set_xlabel("час, с"); axes[2].set_ylabel("част., Гц")
    ims.append(im)

    cbar = fig.colorbar(ims[-1], ax=axes.ravel().tolist(), shrink=0.9, pad=0.02)
    cbar.set_label("Амплітуда, dB")
    fig.suptitle(title, y=1.02, fontsize=11)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        print(f"[INFO] Saved triptych to: {save_path}")
    else:
        plt.show()

# =========================
# MAIN
# =========================
def run_inference():
    device = torch.device("cpu" if (USE_CPU or not torch.cuda.is_available()) else "cuda")
    root = os.path.join(DATA_ROOT, DATASET)

    # meta
    with open(os.path.join(root, "meta.json"), "r") as f:
        meta = json.load(f)
    fs        = int(meta["fs"])
    nperseg   = int(meta["nperseg"])
    noverlap  = int(meta["noverlap"])
    window    = meta.get("window", "hann")
    pad       = int(meta.get("pad", nperseg // 2))
    test_indices = np.array(meta["split_indices"]["test_idx"], dtype=np.int64)

    # data
    train_X = np.load(os.path.join(root, "train_X.npy"))
    test_X  = np.load(os.path.join(root, "test_X.npy"))       # log1p|N|
    test_Y  = np.load(os.path.join(root, "test_Y.npy"))       # log1p|C|
    test_PH = np.load(os.path.join(root, "test_phase.npy"))   # фаза noisy
    Nte, F, Tprime = test_X.shape
    print(f"[INFO] Loaded test set: N={Nte}, F={F}, T'={Tprime}")

    # normalization (як у тренуванні)
    mean = float(train_X.mean())
    std  = float(train_X.std() + 1e-8)
    X_in = (test_X - mean) / std
    X_in_t = torch.from_numpy(X_in).float().unsqueeze(1)

    # model
    model = UnetAutoencoder(input_shape=(F, Tprime)).to(device)
    state = torch.load(WEIGHTS, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"[INFO] Loaded weights: {WEIGHTS}")

    # inference
    dl = DataLoader(TensorDataset(X_in_t), batch_size=BATCH_SIZE, shuffle=False)
    pred_masks = []
    with torch.no_grad():
        for (xb,) in dl:
            pm = model(xb.to(device))
            pred_masks.append(pm.cpu().numpy())
    pred_masks = np.concatenate(pred_masks, axis=0)[:, 0]
    print("[INFO] Inference done.")

    # spectra (linear amplitude)
    noisy_lin = np.expm1(test_X).clip(min=0.0)   # |N|
    den_lin   = pred_masks * noisy_lin           # |Ĉ|
    clean_lin = np.expm1(test_Y).clip(min=0.0)   # |C|

    # Load raw signals for time-domain metrics + align to test_idx
    noisy_path = os.path.join(RAW_DIR, f"{DATASET}_signals.npy")
    clean_path = os.path.join(RAW_DIR, "clean_signals.npy")
    have_raw = os.path.isfile(noisy_path) and os.path.isfile(clean_path)

    mse_list, mae_list, rmse_list, snr_list = [], [], [], []
    mse_noisy, mae_noisy, rmse_noisy, snr_noisy = [], [], [], []

    if have_raw:
        raw_noisy_full = np.load(noisy_path)   # [N_total, T]
        raw_clean_full = np.load(clean_path)   # [N_total, T]
        raw_noisy = raw_noisy_full[test_indices]
        raw_clean = raw_clean_full[test_indices]
        assert raw_noisy.shape[0] == Nte == raw_clean.shape[0], \
            f"Raw/test size mismatch after indexing: raw_noisy={raw_noisy.shape[0]}, raw_clean={raw_clean.shape[0]}, test={Nte}"

        for i in range(Nte):
            # реконструкція з лінійних спектрів і фази noisy
            y_noisy = istft_unpad(noisy_lin[i], test_PH[i], fs, nperseg, noverlap, window, pad)
            y_den   = istft_unpad(den_lin[i],   test_PH[i], fs, nperseg, noverlap, window, pad)
            y_clean = raw_clean[i].astype(np.float32)

            T = min(len(y_clean), len(y_den), len(y_noisy))
            y_clean, y_den, y_noisy = y_clean[:T], y_den[:T], y_noisy[:T]

            # baseline (noisy vs clean)
            mse_noisy.append(MeanSquaredError.calculate(y_clean, y_noisy))
            mae_noisy.append(MeanAbsoluteError.calculate(y_clean, y_noisy))
            rmse_noisy.append(RootMeanSquaredError.calculate(y_clean, y_noisy))
            snr_noisy.append(SignalToNoiseRatio.calculate(y_clean, y_noisy))

            # denoised vs clean
            mse_list.append(MeanSquaredError.calculate(y_clean, y_den))
            mae_list.append(MeanAbsoluteError.calculate(y_clean, y_den))
            rmse_list.append(RootMeanSquaredError.calculate(y_clean, y_den))
            snr_list.append(SignalToNoiseRatio.calculate(y_clean, y_den))

        # Візуалізація з часових сигналів (idx=0)
        idx = 0
        y_clean0 = raw_clean[idx].astype(np.float32)
        y_noisy0 = istft_unpad(noisy_lin[idx], test_PH[idx], fs, nperseg, noverlap, window, pad)
        y_den0   = istft_unpad(den_lin[idx],   test_PH[idx], fs, nperseg, noverlap, window, pad)
        T0 = min(len(y_clean0), len(y_noisy0), len(y_den0))
        y_clean0, y_noisy0, y_den0 = y_clean0[:T0], y_noisy0[:T0], y_den0[:T0]

        os.makedirs(PLOTS_OUT, exist_ok=True)
        plot_triptych_clean_noisy_denoised(
            y_clean0, y_noisy0, y_den0,
            fs=fs, nperseg=nperseg, noverlap=noverlap, window=window, pad=pad,
            title=f"Спектрограми (test,{DATASET}) — чистий / зашумлений / знешумлений (U-Net)",
            save_path=os.path.join(PLOTS_OUT, "triptych_clean_noisy_denoised_idx0.png"),
        )

    else:
        print("[WARN] Raw data not found -> metrics in spectrogram (linear) domain; SNR is NaN.")
        for i in range(Nte):
            # baseline у спектральному просторі (лінійна амплітуда)
            mse_noisy.append(MeanSquaredError.calculate(clean_lin[i], noisy_lin[i]))
            mae_noisy.append(MeanAbsoluteError.calculate(clean_lin[i], noisy_lin[i]))
            rmse_noisy.append(RootMeanSquaredError.calculate(clean_lin[i], noisy_lin[i]))
            snr_noisy.append(np.nan)
            # denoised
            mse_list.append(MeanSquaredError.calculate(clean_lin[i], den_lin[i]))
            mae_list.append(MeanAbsoluteError.calculate(clean_lin[i], den_lin[i]))
            rmse_list.append(RootMeanSquaredError.calculate(clean_lin[i], den_lin[i]))
            snr_list.append(np.nan)

        # Для графіка, якщо немає RAW, відновимо "clean" із clean_lin + фазою noisy
        idx = 0
        y_noisy0 = istft_unpad(noisy_lin[idx], test_PH[idx], fs, nperseg, noverlap, window, pad)
        y_den0   = istft_unpad(den_lin[idx],   test_PH[idx], fs, nperseg, noverlap, window, pad)
        # "псевдо-clean" (лише для візуалізації, не для метрик):
        y_clean0 = istft_unpad(clean_lin[idx], test_PH[idx], fs, nperseg, noverlap, window, pad)
        os.makedirs(PLOTS_OUT, exist_ok=True)
        plot_triptych_clean_noisy_denoised(
            y_clean0, y_noisy0, y_den0,
            fs=fs, nperseg=nperseg, noverlap=noverlap, window=window, pad=pad,
            title=f"Спектрограми (test,{DATASET}) — *псевдо*-чистий / зашумлений / знешумлений (U-Net)",
            save_path=os.path.join(PLOTS_OUT, f"triptych_clean_noisy_denoised_{DATASET}_idx0.png"),
        )

    # ---- Aggregate & print metrics ----
    def avg(x): return float(np.nanmean(x)) if len(x) else np.nan

    print("\n=== TEST METRICS ===")
    print(f"Baseline (Noisy vs Clean):  MSE={avg(mse_noisy):.6f}  MAE={avg(mae_noisy):.6f}  RMSE={avg(rmse_noisy):.6f}  SNR={avg(snr_noisy):.2f} dB")
    print(f"Denoised (Model vs Clean): MSE={avg(mse_list):.6f}  MAE={avg(mae_list):.6f}  RMSE={avg(rmse_list):.6f}  SNR={avg(snr_list):.2f} dB")

if __name__ == "__main__":
    run_inference()
