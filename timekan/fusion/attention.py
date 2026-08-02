"""Cross-band attention fusion."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossBandAttention(nn.Module):
    """Fuse K band representations Z^(k) in R^{H x d}.

    alpha_k = softmax( Wq Z^(k) · (Wk Z_bar)^T / sqrt(d) )  [per time step via mean over d]
    Actually we use a practical multi-head style over the band axis:

    For each forecast step t:
      scores_k = (Wq z_t^{(k)}) · (Wk mean_k z_t) / sqrt(d)
      y_t = sum_k alpha_{t,k} * Wv z_t^{(k)}
    """

    def __init__(self, d_model: int, n_bands: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_bands = n_bands
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = d_model ** -0.5
        self.out = nn.Linear(d_model, d_model)

    def forward(
        self, band_reps: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            band_reps: list of K tensors (B, H, d)
        Returns:
            fused: (B, H, d)
            attn: (B, H, K) attention weights over bands
        """
        # stack -> (B, K, H, d)
        z = torch.stack(band_reps, dim=1)
        b, k, h, d = z.shape
        z_bar = z.mean(dim=1)  # (B, H, d)

        q = self.Wq(z)  # (B, K, H, d)
        key = self.Wk(z_bar).unsqueeze(1)  # (B, 1, H, d)
        # score per band & time: sum over d
        scores = (q * key).sum(dim=-1) * self.scale  # (B, K, H)
        scores = scores.permute(0, 2, 1)  # (B, H, K)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        v = self.Wv(z)  # (B, K, H, d)
        v = v.permute(0, 2, 1, 3)  # (B, H, K, d)
        fused = (attn.unsqueeze(-1) * v).sum(dim=2)  # (B, H, d)
        fused = self.out(fused)
        return fused, attn


class ConcatFusion(nn.Module):
    """Ablation: concatenate bands then project."""

    def __init__(self, d_model: int, n_bands: int):
        super().__init__()
        self.proj = nn.Linear(d_model * n_bands, d_model)

    def forward(
        self, band_reps: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        cat = torch.cat(band_reps, dim=-1)  # (B, H, K*d)
        return self.proj(cat), None


class MeanFusion(nn.Module):
    def forward(
        self, band_reps: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        stacked = torch.stack(band_reps, dim=0)
        return stacked.mean(dim=0), None


class LastBandFusion(nn.Module):
    def forward(
        self, band_reps: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return band_reps[-1], None


def build_fusion(mode: str, d_model: int, n_bands: int, dropout: float = 0.1) -> nn.Module:
    if mode == "attention":
        return CrossBandAttention(d_model, n_bands, dropout=dropout)
    if mode == "concat":
        return ConcatFusion(d_model, n_bands)
    if mode == "mean":
        return MeanFusion()
    if mode == "last":
        return LastBandFusion()
    raise ValueError(f"Unknown fusion mode: {mode}")
