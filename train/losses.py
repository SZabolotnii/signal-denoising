"""Loss function utilities.

Goal
----
Use a robust loss only for non-Gaussian noise by default.

Policy
------
* gaussian     -> MSE (L2)
* non_gaussian -> SmoothL1 (Huber-like, robust to heavy tails / impulses)
"""

from __future__ import annotations

import torch
import torch.nn as nn


def select_loss(noise_type: str, *, robust_beta: float = 0.02) -> nn.Module:
    """Return a loss module for a given training noise regime.

    Parameters
    ----------
    noise_type:
        "gaussian" or "non_gaussian"
    robust_beta:
        SmoothL1 beta (transition point between L1 and L2). Smaller -> closer to L1.
    """
    if noise_type == "non_gaussian":
        # Robust to outliers (impulses, heavy tails)
        return nn.SmoothL1Loss(beta=robust_beta)
    return nn.MSELoss()


def select_recon_loss(noise_type: str, *, robust_beta: float = 0.02, reduction: str = "mean") -> nn.Module:
    """Same policy as select_loss but with configurable reduction.

    Useful for VAE where we may want to control reduction explicitly.
    """
    if noise_type == "non_gaussian":
        return nn.SmoothL1Loss(beta=robust_beta, reduction=reduction)
    return nn.MSELoss(reduction=reduction)
