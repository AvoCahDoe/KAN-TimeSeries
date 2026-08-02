"""Vectorized Chebyshev KAN layers (efficient-kan style)."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def chebyshev_basis(x: torch.Tensor, degree: int) -> torch.Tensor:
    """Evaluate T_0..T_degree at x in [-1,1].

    Args:
        x: (... , in_features)
        degree: max polynomial degree d
    Returns:
        (... , in_features, degree+1)
    """
    # T0 = 1, T1 = x, T_{n} = 2x T_{n-1} - T_{n-2}
    outs = [torch.ones_like(x), x]
    for _ in range(2, degree + 1):
        outs.append(2 * x * outs[-1] - outs[-2])
    # stack as (..., in, degree+1) — only keep up to degree
    stacked = torch.stack(outs[: degree + 1], dim=-1)
    return stacked


class ChebyKANLinear(nn.Module):
    """Single Chebyshev KAN edge layer: y_j = sum_i phi_{ij}(x_i).

    phi(x) = sum_{k=0}^d c_k T_k(tanh(x))
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        degree: int = 4,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        # coeffs: (out, in, degree+1)
        self.coeffs = nn.Parameter(
            torch.empty(out_features, in_features, degree + 1)
        )
        nn.init.kaiming_uniform_(self.coeffs, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in)
        x_t = torch.tanh(x)
        basis = chebyshev_basis(x_t, self.degree)  # (..., in, d+1)
        # einsum: out_j = sum_i sum_k coeffs[j,i,k] * basis[..., i, k]
        y = torch.einsum("...ik,oik->...o", basis, self.coeffs)
        if self.bias is not None:
            y = y + self.bias
        return y


class MLPLinear(nn.Module):
    """Plain MLP edge for ablation (basis=MLP)."""

    def __init__(self, in_features: int, out_features: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BSplineKANLinear(nn.Module):
    """Lightweight B-spline-like KAN via radial basis for ablation (not full pykan)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_knots: int = 8,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_knots = num_knots
        grid = torch.linspace(-1, 1, num_knots)
        self.register_buffer("grid", grid)
        self.coeffs = nn.Parameter(torch.empty(out_features, in_features, num_knots))
        nn.init.kaiming_uniform_(self.coeffs, a=math.sqrt(5))
        self.sigma = 2.0 / max(num_knots - 1, 1)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = torch.tanh(x)  # (..., in)
        # RBF around knots: (..., in, K)
        dist = x_t.unsqueeze(-1) - self.grid  # (..., in, K)
        basis = torch.exp(-0.5 * (dist / self.sigma) ** 2)
        y = torch.einsum("...ik,oik->...o", basis, self.coeffs)
        if self.bias is not None:
            y = y + self.bias
        return y


class ChebyKANBlock(nn.Module):
    """Stack of Chebyshev (or alt basis) layers with residual + LayerNorm."""

    def __init__(
        self,
        d_model: int,
        degree: int = 4,
        n_layers: int = 2,
        basis: str = "chebyshev",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.basis = basis
        layers: List[nn.Module] = []
        norms: List[nn.Module] = []
        drops: List[nn.Module] = []
        for _ in range(n_layers):
            if basis == "chebyshev":
                layers.append(ChebyKANLinear(d_model, d_model, degree=degree))
            elif basis == "bspline":
                layers.append(BSplineKANLinear(d_model, d_model, num_knots=degree + 4))
            elif basis == "mlp":
                layers.append(MLPLinear(d_model, d_model))
            else:
                raise ValueError(f"Unknown basis: {basis}")
            norms.append(nn.LayerNorm(d_model))
            drops.append(nn.Dropout(dropout))
        self.layers = nn.ModuleList(layers)
        self.norms = nn.ModuleList(norms)
        self.drops = nn.ModuleList(drops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer, norm, drop in zip(self.layers, self.norms, self.drops):
            x = norm(x + drop(layer(x)))
        return x


class TemporalChebyKAN(nn.Module):
    """Map lookback L×C → horizon representation H×d via channel mix + time proj."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        degree: int = 4,
        n_layers: int = 2,
        basis: str = "chebyshev",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.d_model = d_model
        self.in_proj = nn.Linear(enc_in, d_model)
        self.kan = ChebyKANBlock(
            d_model, degree=degree, n_layers=n_layers, basis=basis, dropout=dropout
        )
        self.time_proj = nn.Linear(seq_len, pred_len)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, C) -> (B, H, d_model)"""
        h = self.in_proj(x)  # (B, L, d)
        h = self.kan(h)
        h = h.transpose(1, 2)  # (B, d, L)
        h = self.time_proj(h)  # (B, d, H)
        h = h.transpose(1, 2)  # (B, H, d)
        return self.out_norm(h)
