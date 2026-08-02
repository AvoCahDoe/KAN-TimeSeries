"""Reversible Instance Normalization (RevIN)."""

from __future__ import annotations

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """Per-channel instance normalization with reversible denorm.

    Args:
        num_features: Number of channels C.
        eps: Numerical stability.
        affine: Learnable affine scale/bias after norm.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("_mean", torch.zeros(1), persistent=False)
        self.register_buffer("_stdev", torch.ones(1), persistent=False)

    def forward(self, x: torch.Tensor, mode: str = "norm") -> torch.Tensor:
        """x: (B, L, C). mode in {'norm', 'denorm'}."""
        if mode == "norm":
            return self._normalize(x)
        if mode == "denorm":
            return self._denormalize(x)
        raise ValueError(f"Unknown RevIN mode: {mode}")

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        self._mean = x.mean(dim=1, keepdim=True).detach()
        self._stdev = (
            torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        )
        x = (x - self._mean) / self._stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps * 10)
        return x * self._stdev + self._mean
