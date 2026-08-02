"""TimeKAN full model and PlainKAN ablation."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from timekan.fusion.attention import build_fusion
from timekan.layers.cheby_kan import TemporalChebyKAN
from timekan.layers.decomp import build_decomp
from timekan.layers.revin import RevIN


DEFAULT_DEGREE_SCHEDULE = {
    1: [4],
    2: [2, 6],
    3: [2, 4, 6],
    4: [2, 3, 5, 7],
    5: [2, 3, 4, 5, 7],
}


class TimeKAN(nn.Module):
    """Frequency-decomposed Chebyshev KAN for LTSF.

    Stages: RevIN → band decomp → per-band M-KAN → fusion → linear → denorm.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        k_bands: int = 3,
        decomp_mode: str = "adaptive",
        fusion_mode: str = "attention",
        n_kan_layers: int = 2,
        degree_schedule: Optional[List[int]] = None,
        uniform_degree: Optional[int] = None,
        basis: str = "chebyshev",
        use_revin: bool = True,
        dropout: float = 0.1,
        c_out: Optional[int] = None,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.c_out = c_out or enc_in
        self.k_bands = k_bands
        self.use_revin = use_revin
        self.decomp_mode = decomp_mode
        self.fusion_mode = fusion_mode
        self.basis = basis

        if uniform_degree is not None:
            degrees = [uniform_degree] * k_bands
        elif degree_schedule is not None:
            degrees = list(degree_schedule)
            assert len(degrees) == k_bands
        else:
            degrees = DEFAULT_DEGREE_SCHEDULE.get(k_bands, [4] * k_bands)

        self.degrees = degrees
        self.revin = RevIN(enc_in, affine=True) if use_revin else None
        self.decomp = build_decomp(decomp_mode, seq_len, enc_in, k_bands=k_bands)
        self.band_kans = nn.ModuleList(
            [
                TemporalChebyKAN(
                    seq_len=seq_len,
                    pred_len=pred_len,
                    enc_in=enc_in,
                    d_model=d_model,
                    degree=deg,
                    n_layers=n_kan_layers,
                    basis=basis,
                    dropout=dropout,
                )
                for deg in degrees
            ]
        )
        self.fusion = build_fusion(fusion_mode, d_model, k_bands, dropout=dropout)
        self.head = nn.Linear(d_model, self.c_out)

        # caches for visualization
        self.last_bands: Optional[List[torch.Tensor]] = None
        self.last_masks: Optional[torch.Tensor] = None
        self.last_attn: Optional[torch.Tensor] = None

    def forward(
        self, x: torch.Tensor, return_aux: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: (B, L, C)
        Returns:
            yhat: (B, H, C)
        """
        if self.revin is not None:
            x = self.revin(x, "norm")

        bands, masks = self.decomp(x)
        self.last_bands = [b.detach() for b in bands]
        self.last_masks = masks.detach() if isinstance(masks, torch.Tensor) else masks

        band_reps = [kan(bk) for kan, bk in zip(self.band_kans, bands)]
        fused, attn = self.fusion(band_reps)
        self.last_attn = attn.detach() if attn is not None else None

        y = self.head(fused)  # (B, H, C)

        if self.revin is not None:
            y = self.revin(y, "denorm")

        if return_aux:
            return y, {
                "bands": self.last_bands,
                "masks": self.last_masks,
                "attn": self.last_attn,
            }
        return y


class PlainKAN(nn.Module):
    """Undecomposed Chebyshev KAN baseline (critical ablation)."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        degree: int = 4,
        n_kan_layers: int = 2,
        basis: str = "chebyshev",
        use_revin: bool = True,
        dropout: float = 0.1,
        c_out: Optional[int] = None,
    ):
        super().__init__()
        self.use_revin = use_revin
        self.c_out = c_out or enc_in
        self.revin = RevIN(enc_in, affine=True) if use_revin else None
        self.backbone = TemporalChebyKAN(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=enc_in,
            d_model=d_model,
            degree=degree,
            n_layers=n_kan_layers,
            basis=basis,
            dropout=dropout,
        )
        self.head = nn.Linear(d_model, self.c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.revin is not None:
            x = self.revin(x, "norm")
        h = self.backbone(x)
        y = self.head(h)
        if self.revin is not None:
            y = self.revin(y, "denorm")
        return y
