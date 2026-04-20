"""Reproducibility helpers for deterministic training across seeds.

Call `set_global_seed(seed)` once per process entry point (CLI main, Trainer
`__init__`). This seeds Python `random`, NumPy, and PyTorch CPU/CUDA/MPS
generators, plus best-effort deterministic flags.

Operator-level determinism is not guaranteed for every CUDA/MPS kernel; the
Phase A1 acceptance bar is σ(val_SNR) < 0.02 dB across two runs with the same
seed, which tolerates residual nondeterminism in cuDNN heuristics but catches
coarse seed leakage.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Seed all global RNGs used by the training pipeline.

    Parameters
    ----------
    seed : int
        Seed applied to random, NumPy, and all torch generators.
    deterministic : bool
        If True, sets cuDNN deterministic mode and
        `torch.use_deterministic_algorithms(True, warn_only=True)`.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # CUBLAS workspace config required for deterministic cuBLAS on CUDA ≥10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
