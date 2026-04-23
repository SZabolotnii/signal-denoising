"""Loss function utilities.

Default policy (unchanged for backwards compat):
  gaussian     -> MSE
  non_gaussian -> SmoothL1(beta=0.02)

Phase A2-H1 (loss sweep) adds explicit selection: mse, smoothl1, huber,
charbonnier, l1. Pass `loss_name` to override the noise-type default.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CharbonnierLoss(nn.Module):
    """Differentiable approximation of L1: sqrt((x - y)^2 + eps^2).

    Stable near 0 (unlike pure L1) and less peaky than SmoothL1 with small beta.
    """

    def __init__(self, eps: float = 1e-3, reduction: str = "mean"):
        super().__init__()
        self.eps2 = eps * eps
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        v = torch.sqrt((pred - target) ** 2 + self.eps2)
        if self.reduction == "mean":
            return v.mean()
        if self.reduction == "sum":
            return v.sum()
        return v


def _build(loss_name: str, *, beta: float, huber_delta: float,
           charbonnier_eps: float, reduction: str) -> nn.Module:
    name = loss_name.lower()
    if name == "mse":
        return nn.MSELoss(reduction=reduction)
    if name == "l1":
        return nn.L1Loss(reduction=reduction)
    if name == "smoothl1":
        return nn.SmoothL1Loss(beta=beta, reduction=reduction)
    if name == "huber":
        return nn.HuberLoss(delta=huber_delta, reduction=reduction)
    if name == "charbonnier":
        return CharbonnierLoss(eps=charbonnier_eps, reduction=reduction)
    raise ValueError(f"Unknown loss_name: {loss_name!r}")


def select_loss(
    noise_type: str,
    *,
    loss_name: str | None = None,
    robust_beta: float = 0.02,
    huber_delta: float = 1.0,
    charbonnier_eps: float = 1e-3,
) -> nn.Module:
    """Return a loss module.

    If `loss_name` is given, it overrides the noise-type default.
    Otherwise: gaussian -> MSE, non_gaussian -> SmoothL1(beta=robust_beta).
    """
    if loss_name is not None:
        return _build(loss_name, beta=robust_beta, huber_delta=huber_delta,
                      charbonnier_eps=charbonnier_eps, reduction="mean")
    if noise_type == "non_gaussian":
        return nn.SmoothL1Loss(beta=robust_beta)
    return nn.MSELoss()


def select_recon_loss(
    noise_type: str,
    *,
    loss_name: str | None = None,
    robust_beta: float = 0.02,
    huber_delta: float = 1.0,
    charbonnier_eps: float = 1e-3,
    reduction: str = "mean",
) -> nn.Module:
    """Same as select_loss but with configurable reduction (for VAE etc.)."""
    if loss_name is not None:
        return _build(loss_name, beta=robust_beta, huber_delta=huber_delta,
                      charbonnier_eps=charbonnier_eps, reduction=reduction)
    if noise_type == "non_gaussian":
        return nn.SmoothL1Loss(beta=robust_beta, reduction=reduction)
    return nn.MSELoss(reduction=reduction)
