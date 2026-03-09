#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Training script for Mask U-Net on spectrogram dataset.

Expects data created by build_spectrogram_dataset.py:
  ../spectro_dataset/{dataset_type}/
      train_X.npy [N,F,T'], train_Y.npy [N,F,T'], train_mask.npy [N,F,T']
      val_X.npy   [..],      val_Y.npy   [..],    val_mask.npy   [..]
      test_X.npy  [..],      test_Y.npy  [..],    test_mask.npy  [..]
      test_phase.npy [N,F,T']   (фази тільки для тесту, для інференсу)
      meta.json    (fs, nperseg, noverlap, pad, shapes ...)

Loss (new):
  1) Маска (регресія): Smooth L1 між pred_mask і M
  2) L1 в лог-спектрі: |out_log - Y_log|
  3) L1 в лінійному масштабі: |expm1(out_log) - expm1(Y_log)|
  4) Spectral convergence: ||out_mag - clean_mag|| / ||clean_mag||

Сумарний лосс:
  L = λ_mask * L_mask +
      λ_log  * L_log  +
      λ_mag  * L_mag  +
      λ_sc   * L_sc

ВАЖЛИВО: Мережа отримує нормалізований вхід X_in = zscore(X_log),
але для реконструкції потрібен X_log (оригінальний лог-модуль), тому
датасет повертає і X_in, і X_log.
"""

import os
import json
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# опційно: WANDB (не обов'язково)
try:
    import wandb
    WANDB_OK = True
except Exception:
    WANDB_OK = False

# ----- модель (твоя версія з SE, даунсемплом по F і T) -----
from models.autoencoder_unet import UnetAutoencoder


# ------------------------
# Аугментації для спектрограм
# ------------------------

def rng_choice(rng, p: float) -> bool:
    return rng.random() < p

def aug_time_shift(spec, max_shift_frac=0.1, rng=None):
    """Циклічний шіфт по осі часу (T'). Працюємо в лог-амплітуді."""
    if rng is None:
        rng = np.random.default_rng()
    F, T = spec.shape
    max_shift = int(T * max_shift_frac)
    if max_shift <= 0:
        return spec
    k = rng.integers(-max_shift, max_shift + 1)
    if k == 0:
        return spec
    return np.roll(spec, k, axis=1)

def aug_freq_mask(spec, max_width_frac=0.15, p=0.5, rng=None):
    """Rectangular frequency mask ala SpecAugment (лог-амплітуда)."""
    if rng is None:
        rng = np.random.default_rng()
    if not rng_choice(rng, p):
        return spec
    F, T = spec.shape
    w = max(1, int(F * max_width_frac * rng.random()))
    f0 = rng.integers(0, max(1, F - w))
    out = spec.copy()
    out[f0:f0+w, :] = 0.0  # у лог-просторі 0≈log1p(0), тобто прибираємо енергію
    return out

def aug_time_mask(spec, max_width_frac=0.15, p=0.5, rng=None):
    """Rectangular time mask ala SpecAugment (лог-амплітуда)."""
    if rng is None:
        rng = np.random.default_rng()
    if not rng_choice(rng, p):
        return spec
    F, T = spec.shape
    w = max(1, int(T * max_width_frac * rng.random()))
    t0 = rng.integers(0, max(1, T - w))
    out = spec.copy()
    out[:, t0:t0+w] = 0.0
    return out

def aug_zoom_time(spec, zoom_min=0.9, zoom_max=1.1, rng=None):
    """Легкий zoom по часу з поверненням до початкової T' (без фліпу!)."""
    if rng is None:
        rng = np.random.default_rng()
    F, T = spec.shape
    z = rng.uniform(zoom_min, zoom_max)
    t_new = int(round(T * z))
    if t_new == T:
        return spec
    x = np.linspace(0, 1, T, endpoint=False)
    x_new = np.linspace(0, 1, t_new, endpoint=False)
    out = np.empty((F, t_new), dtype=spec.dtype)
    for i in range(F):
        out[i] = np.interp(x_new, x, spec[i])
    if t_new > T:
        start = (t_new - T) // 2
        out = out[:, start:start+T]
    else:
        pad_left = (T - t_new) // 2
        pad_right = T - t_new - pad_left
        out = np.pad(out, ((0, 0), (pad_left, pad_right)), mode="edge")
    return out

def apply_augmentations(X_log, Y_log, M, p_shift=0.8, p_masks=0.8, rng=None):
    """
    Узгоджено застосовуємо аугментації до X_log, Y_log, M (усі у F×T').
    Не робимо горизонтального фліпу.
    """
    if rng is None:
        rng = np.random.default_rng()
    Xo, Yo, Mo = X_log.copy(), Y_log.copy(), M.copy()

    if rng_choice(rng, p_shift):
        Xo = aug_time_shift(Xo, rng=rng)
        Yo = aug_time_shift(Yo, rng=rng)
        Mo = aug_time_shift(Mo, rng=rng)

    if rng_choice(rng, 0.6):
        Xo = aug_zoom_time(Xo, rng=rng)
        Yo = aug_zoom_time(Yo, rng=rng)
        Mo = aug_zoom_time(Mo, rng=rng)

    if rng_choice(rng, p_masks):
        # маскінг застосовуємо тільки до Xo (імітуємо випадкові пропуски енергії в noisy)
        Xo = aug_freq_mask(Xo, rng=rng)
        Xo = aug_time_mask(Xo, rng=rng)

    return Xo, Yo, Mo


# ------------------------
# Dataset / DataLoader
# ------------------------

class SpectroMaskDataset(Dataset):
    """
    Повертає кортеж з 4 елементів:
      X_in  : (1,F,T') — нормалізований інпут для моделі (z-score по train X)
      Y_log : (1,F,T') — таргет-лог-амплітуда clean (log1p|C|)
      M     : (1,F,T') — таргет-маска ∈[0,1]
      X_log : (1,F,T') — ОРИГІНАЛЬНИЙ лог noisy (log1p|N|), потрібен для реконструкції в лоссі
    """
    def __init__(self, root_dir, split, augment=False, seed=42, train_stats=None):
        self.X = np.load(os.path.join(root_dir, f"{split}_X.npy"))    # [N,F,T']  (це X_log)
        self.Y = np.load(os.path.join(root_dir, f"{split}_Y.npy"))    # [N,F,T']  (Y_log)
        self.M = np.load(os.path.join(root_dir, f"{split}_mask.npy")) # [N,F,T']  (mask)
        self.augment = augment
        self.rng = np.random.default_rng(seed)

        if train_stats is None and split != "train":
            raise ValueError("For non-train split pass train_stats={'mean':..., 'std':...}")
        if train_stats is None:
            self.mean = float(np.mean(self.X))
            self.std  = float(np.std(self.X) + 1e-8)
        else:
            self.mean = float(train_stats["mean"])
            self.std  = float(train_stats["std"]  + 1e-8)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        X_log = self.X[idx]  # (F,T')
        Y_log = self.Y[idx]
        M     = self.M[idx]

        if self.augment:
            X_log, Y_log, M = apply_augmentations(X_log, Y_log, M, rng=self.rng)

        # Нормалізуємо ТІЛЬКИ вхід для моделі
        X_in = (X_log - self.mean) / self.std

        # до PyTorch формату (1,F,T')
        X_in  = torch.tensor(X_in,  dtype=torch.float32).unsqueeze(0)
        Y_log = torch.tensor(Y_log, dtype=torch.float32).unsqueeze(0)
        M     = torch.tensor(M,     dtype=torch.float32).unsqueeze(0)
        X_log = torch.tensor(X_log, dtype=torch.float32).unsqueeze(0)

        return X_in, Y_log, M, X_log


# ------------------------
# Лоси
# ------------------------

def compute_loss_components(pred_mask, out_log, M, Y_log):
    """
    Повертає словник з компонентами лоссу:
      L_mask, L_log, L_mag, L_sc
    """
    # Маска (регресія)
    L_mask = torch.nn.functional.smooth_l1_loss(pred_mask, M)

    # Лог-спектр
    L_log = torch.mean(torch.abs(out_log - Y_log))

    # Лінійний спектр
    clean_mag = torch.expm1(Y_log)
    out_mag   = torch.expm1(out_log).clamp_min(0.0)

    L_mag = torch.mean(torch.abs(out_mag - clean_mag))

    # Spectral convergence
    num = torch.norm(out_mag - clean_mag, p='fro')
    den = torch.norm(clean_mag, p='fro') + 1e-8
    L_sc = num / den

    return {
        "mask": L_mask,
        "log":  L_log,
        "mag":  L_mag,
        "sc":   L_sc,
    }


def loss_fn(pred_mask, out_log, M, Y_log,
            w_mask=1.0, w_log=1.0, w_mag=1.0, w_sc=1.0):
    comps = compute_loss_components(pred_mask, out_log, M, Y_log)
    loss = (
        w_mask * comps["mask"] +
        w_log  * comps["log"]  +
        w_mag  * comps["mag"]  +
        w_sc   * comps["sc"]
    )
    return loss, comps


# ------------------------
# Тренувальні/валідаційні проходи
# ------------------------

def train_one_epoch(model, loader, opt, device, loss_weights):
    model.train()
    running = {"loss": 0.0, "mask": 0.0, "log": 0.0, "mag": 0.0, "sc": 0.0}
    n = 0

    for X_in, Y_log, M, X_log in loader:
        X_in  = X_in.to(device)   # (B,1,F,T)
        Y_log = Y_log.to(device)  # (B,1,F,T)
        M     = M.to(device)      # (B,1,F,T)
        X_log = X_log.to(device)  # (B,1,F,T)

        opt.zero_grad()
        pred_mask = model(X_in)   # (B,1,F,T), sigmoid всередині моделі

        # --- Реконструкція у лог-просторі ---
        noisy_lin = torch.expm1(X_log).clamp_min_(0.0)   # повертаємося в лінійну амплітуду
        out_lin   = pred_mask * noisy_lin                # застосовуємо маску до |N|
        out_log   = torch.log1p(out_lin.clamp_min(0.0))  # назад у лог

        # --- Складені втрати ---
        loss, comps = loss_fn(
            pred_mask, out_log, M, Y_log,
            w_mask=loss_weights["mask"],
            w_log=loss_weights["log"],
            w_mag=loss_weights["mag"],
            w_sc=loss_weights["sc"],
        )

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()

        bs = X_in.size(0)
        running["loss"] += float(loss.item())        * bs
        running["mask"] += float(comps["mask"].item()) * bs
        running["log"]  += float(comps["log"].item())  * bs
        running["mag"]  += float(comps["mag"].item())  * bs
        running["sc"]   += float(comps["sc"].item())   * bs
        n += bs

    for k in running:
        running[k] /= max(1, n)
    return running


@torch.no_grad()
def evaluate(model, loader, device, loss_weights):
    model.eval()
    running = {"loss": 0.0, "mask": 0.0, "log": 0.0, "mag": 0.0, "sc": 0.0}
    n = 0

    for X_in, Y_log, M, X_log in loader:
        X_in  = X_in.to(device)
        Y_log = Y_log.to(device)
        M     = M.to(device)
        X_log = X_log.to(device)

        pred_mask = model(X_in)

        noisy_lin = torch.expm1(X_log).clamp_min_(0.0)
        out_lin   = pred_mask * noisy_lin
        out_log   = torch.log1p(out_lin.clamp_min(0.0))

        loss, comps = loss_fn(
            pred_mask, out_log, M, Y_log,
            w_mask=loss_weights["mask"],
            w_log=loss_weights["log"],
            w_mag=loss_weights["mag"],
            w_sc=loss_weights["sc"],
        )

        bs = X_in.size(0)
        running["loss"] += float(loss.item())          * bs
        running["mask"] += float(comps["mask"].item()) * bs
        running["log"]  += float(comps["log"].item())  * bs
        running["mag"]  += float(comps["mag"].item())  * bs
        running["sc"]   += float(comps["sc"].item())   * bs
        n += bs

    for k in running:
        running[k] /= max(1, n)
    return running


# ------------------------
# Головна функція
# ------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    root = os.path.join(args.data_root, args.dataset_type)

    # Meta (для контролю параметрів/форм)
    with open(os.path.join(root, "meta.json"), "r") as f:
        meta = json.load(f)
    F, Tprime = meta["shapes"]["train"]["X"][1:]

    # --- Train stats for normalization (обчислюємо по train_X) ---
    train_X = np.load(os.path.join(root, "train_X.npy"))  # лог-амплітуда
    train_mean = float(train_X.mean())
    train_std  = float(train_X.std() + 1e-8)
    train_stats = {"mean": train_mean, "std": train_std}

    # Datasets / Loaders
    ds_train = SpectroMaskDataset(root, "train", augment=True,  seed=args.random_state,     train_stats=train_stats)
    ds_val   = SpectroMaskDataset(root, "val",   augment=False, seed=args.random_state + 1, train_stats=train_stats)

    g = torch.Generator()
    g.manual_seed(args.random_state)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,  num_workers=2, generator=g, drop_last=True)
    dl_val   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model
    model = UnetAutoencoder(input_shape=(F, Tprime)).to(device)

    # Optim / sched
    opt = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-6)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5, verbose=True)

    # Loss weights
    loss_weights = {
        "mask": args.lambda_mask,
        "log":  args.lambda_log,
        "mag":  args.lambda_mag,
        "sc":   args.lambda_sc,
    }

    # WANDB
    if WANDB_OK and args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"MaskUNet_{args.dataset_type}",
            config=dict(vars(args), F=F, Tprime=Tprime, mean=train_mean, std=train_std),
        )

    best_val = float("inf")
    best_state = None
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, dl_train, opt, device, loss_weights)
        va = evaluate(model, dl_val, device, loss_weights)

        sched.step(va["loss"])

        if WANDB_OK and args.wandb:
            wandb.log({
                "epoch": epoch,
                **{f"train_{k}": v for k, v in tr.items()},
                **{f"val_{k}":   v for k, v in va.items()},
                "lr": opt.param_groups[0]["lr"],
            })

        print(
            f"Epoch {epoch:03d} | "
            f"train: loss={tr['loss']:.4f} "
            f"(mask={tr['mask']:.4f} log={tr['log']:.4f} mag={tr['mag']:.4f} sc={tr['sc']:.4f}) | "
            f"val:   loss={va['loss']:.4f} "
            f"(mask={va['mask']:.4f} log={va['log']:.4f} mag={va['mag']:.4f} sc={va['sc']:.4f})"
        )

        # early stopping
        if va["loss"] < best_val - 1e-5:
            best_val = va["loss"]
            best_state = model.state_dict()
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    # Save best
    os.makedirs(args.weights_dir, exist_ok=True)
    save_path = os.path.join(args.weights_dir, f"UnetAutoencoder_{args.dataset_type}_best.pth")
    torch.save(best_state if best_state is not None else model.state_dict(), save_path)
    print(f"✅ Best model saved to: {save_path}")

    if WANDB_OK and args.wandb:
        wandb.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train Mask U-Net on spectrogram dataset")
    p.add_argument("--dataset_type", type=str, default="non_gaussian", choices=["gaussian", "non_gaussian"])
    p.add_argument("--data_root",    type=str, default="../../spectro_dataset")
    p.add_argument("--weights_dir",  type=str, default="weights")

    # reproducibility / loader
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--batch_size",   type=int, default=32)
    p.add_argument("--epochs",       type=int, default=60)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--patience",     type=int, default=12)
    p.add_argument("--cpu",          action="store_true")

    # loss weights (нові)
    p.add_argument("--lambda_mask", type=float, default=1.0)
    p.add_argument("--lambda_log",  type=float, default=1.0)
    p.add_argument("--lambda_mag",  type=float, default=1.0)
    p.add_argument("--lambda_sc",   type=float, default=1.0)

    # wandb
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="signal-denoising")

    args = p.parse_args()
    main(args)
