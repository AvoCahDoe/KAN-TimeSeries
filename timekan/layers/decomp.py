"""Fixed FFT and adaptive learnable frequency-band decomposition."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _band_masks_fixed(freq_len: int, k_bands: int, device: torch.device) -> torch.Tensor:
    """Hard rectangular masks over DFT bins [0, freq_len).

    Returns (K, freq_len) float masks.
    """
    masks = torch.zeros(k_bands, freq_len, device=device)
    edges = torch.linspace(0, freq_len, k_bands + 1)
    for i in range(k_bands):
        lo = int(edges[i].item())
        hi = int(edges[i + 1].item())
        masks[i, lo:hi] = 1.0
    # ensure DC fully in lowest band
    masks[0, 0] = 1.0
    return masks


class FixedBandDecomp(nn.Module):
    """DFT → K hard frequency bands → iDFT → K time-domain sub-signals."""

    def __init__(self, k_bands: int = 3):
        super().__init__()
        self.k_bands = k_bands

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Args:
            x: (B, L, C)
        Returns:
            bands: list of K tensors (B, L, C)
            masks: (K, F) used masks (for viz)
        """
        b, l, c = x.shape
        # rFFT along time
        Xf = torch.fft.rfft(x, dim=1)  # (B, F, C)
        f = Xf.size(1)
        masks = _band_masks_fixed(f, self.k_bands, x.device)  # (K, F)
        bands: List[torch.Tensor] = []
        for k in range(self.k_bands):
            mk = masks[k].view(1, f, 1)
            Xk = Xf * mk
            xk = torch.fft.irfft(Xk, n=l, dim=1)
            bands.append(xk)
        return bands, masks


class AdaptiveBandDecomp(nn.Module):
    """Learnable soft frequency masks conditioned on the input spectrum."""

    def __init__(
        self,
        seq_len: int,
        enc_in: int,
        k_bands: int = 3,
        hidden: int = 64,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.k_bands = k_bands
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.freq_len = seq_len // 2 + 1
        self.temperature = temperature
        # condition on pooled log-magnitude spectrum
        self.gate = nn.Sequential(
            nn.Linear(self.freq_len, hidden),
            nn.GELU(),
            nn.Linear(hidden, k_bands * self.freq_len),
        )
        # soft prior toward increasing frequency bands
        prior = _band_masks_fixed(self.freq_len, k_bands, torch.device("cpu"))
        self.register_buffer("prior_masks", prior)

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Args:
            x: (B, L, C)
        Returns:
            bands: list of K (B, L, C)
            soft_masks: (B, K, F)
        """
        b, l, c = x.shape
        Xf = torch.fft.rfft(x, dim=1)  # (B, F, C)
        f = Xf.size(1)
        mag = torch.log1p(Xf.abs().mean(dim=-1))  # (B, F)
        logits = self.gate(mag).view(b, self.k_bands, f)  # (B, K, F)
        # mix with fixed prior so early training is stable
        prior = self.prior_masks.unsqueeze(0).expand(b, -1, -1)
        soft = torch.softmax(logits / self.temperature, dim=1)  # across bands
        soft = 0.7 * soft + 0.3 * prior
        # renormalize per frequency bin
        soft = soft / soft.sum(dim=1, keepdim=True).clamp_min(1e-6)

        bands: List[torch.Tensor] = []
        for k in range(self.k_bands):
            mk = soft[:, k, :].unsqueeze(-1)  # (B, F, 1)
            Xk = Xf * mk
            xk = torch.fft.irfft(Xk, n=l, dim=1)
            bands.append(xk)
        return bands, soft


def build_decomp(
    mode: str,
    seq_len: int,
    enc_in: int,
    k_bands: int = 3,
) -> nn.Module:
    if mode == "fixed":
        return FixedBandDecomp(k_bands=k_bands)
    if mode == "adaptive":
        return AdaptiveBandDecomp(seq_len=seq_len, enc_in=enc_in, k_bands=k_bands)
    raise ValueError(f"Unknown decomp mode: {mode}")
