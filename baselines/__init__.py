"""Baseline forecasting models."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from timekan.layers.revin import RevIN
from timekan.models.timekan import PlainKAN, TimeKAN


class NaiveLast(nn.Module):
    """Repeat last observed value for H steps."""

    def __init__(self, seq_len: int, pred_len: int, enc_in: int, **kwargs):
        super().__init__()
        self.pred_len = pred_len
        self.enc_in = enc_in

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last = x[:, -1:, :]
        return last.repeat(1, self.pred_len, 1)


class DLinear(nn.Module):
    """DLinear: seasonal-trend decomposition + linear (Zeng et al.)."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        individual: bool = False,
        moving_avg: int = 25,
        **kwargs,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.individual = individual
        kernel = moving_avg if moving_avg % 2 == 1 else moving_avg + 1
        self.avg = nn.AvgPool1d(kernel_size=kernel, stride=1, padding=kernel // 2)
        if individual:
            self.Linear_Seasonal = nn.ModuleList(
                [nn.Linear(seq_len, pred_len) for _ in range(enc_in)]
            )
            self.Linear_Trend = nn.ModuleList(
                [nn.Linear(seq_len, pred_len) for _ in range(enc_in)]
            )
        else:
            self.Linear_Seasonal = nn.Linear(seq_len, pred_len)
            self.Linear_Trend = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        trend = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        # fix length if avgpool padding changes size slightly
        if trend.size(1) != x.size(1):
            trend = trend[:, : x.size(1), :]
        seasonal = x - trend
        if self.individual:
            seas_out, trend_out = [], []
            for i in range(self.enc_in):
                seas_out.append(self.Linear_Seasonal[i](seasonal[:, :, i]))
                trend_out.append(self.Linear_Trend[i](trend[:, :, i]))
            seas_out = torch.stack(seas_out, dim=-1)
            trend_out = torch.stack(trend_out, dim=-1)
        else:
            seas_out = self.Linear_Seasonal(seasonal.permute(0, 2, 1)).permute(0, 2, 1)
            trend_out = self.Linear_Trend(trend.permute(0, 2, 1)).permute(0, 2, 1)
        return seas_out + trend_out


class NLinear(nn.Module):
    """NLinear: subtract last value, linear, add back."""

    def __init__(self, seq_len: int, pred_len: int, enc_in: int, **kwargs):
        super().__init__()
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last = x[:, -1:, :]
        x = x - last
        y = self.linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        return y + last


class LSTMModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
        use_revin: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.use_revin = use_revin
        self.revin = RevIN(enc_in) if use_revin else None
        self.lstm = nn.LSTM(
            enc_in, d_model, num_layers=n_layers, batch_first=True, dropout=dropout
        )
        self.proj = nn.Linear(d_model, enc_in * pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.revin is not None:
            x = self.revin(x, "norm")
        out, _ = self.lstm(x)
        y = self.proj(out[:, -1, :])
        y = y.view(x.size(0), self.pred_len, -1)
        if self.revin is not None:
            y = self.revin(y, "denorm")
        return y


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous() if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.drop1(self.relu1(self.chomp1(self.conv1(x))))
        out = self.drop2(self.relu2(self.chomp2(self.conv2(out))))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        levels: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        use_revin: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.use_revin = use_revin
        self.revin = RevIN(enc_in) if use_revin else None
        layers = []
        for i in range(levels):
            dil = 2 ** i
            in_ch = enc_in if i == 0 else d_model
            layers.append(TemporalBlock(in_ch, d_model, kernel_size, dil, dropout))
        self.network = nn.Sequential(*layers)
        self.proj = nn.Linear(d_model * seq_len, enc_in * pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.revin is not None:
            x = self.revin(x, "norm")
        h = self.network(x.transpose(1, 2)).transpose(1, 2)
        y = self.proj(h.reshape(h.size(0), -1))
        y = y.view(x.size(0), self.pred_len, -1)
        if self.revin is not None:
            y = self.revin(y, "denorm")
        return y


class PatchTST(nn.Module):
    """Compact PatchTST-style channel-independent Transformer."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        n_heads: int = 4,
        e_layers: int = 2,
        patch_len: int = 16,
        stride: int = 8,
        dropout: float = 0.1,
        use_revin: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.patch_len = patch_len
        self.stride = stride
        self.use_revin = use_revin
        self.revin = RevIN(enc_in) if use_revin else None
        self.n_patches = max(1, (seq_len - patch_len) // stride + 1)
        self.patch_proj = nn.Linear(patch_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        self.head = nn.Linear(self.n_patches * d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C) — channel independent
        if self.revin is not None:
            x = self.revin(x, "norm")
        b, l, c = x.shape
        # unfold patches per channel
        xc = x.permute(0, 2, 1).reshape(b * c, 1, l)
        patches = xc.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # (B*C, 1, n_patches, patch_len) -> (B*C, n_patches, patch_len)
        patches = patches.squeeze(1)
        if patches.size(1) != self.n_patches:
            # pad/truncate
            if patches.size(1) > self.n_patches:
                patches = patches[:, : self.n_patches]
            else:
                pad = self.n_patches - patches.size(1)
                patches = F.pad(patches, (0, 0, 0, pad))
        h = self.patch_proj(patches)
        h = self.encoder(h)
        h = h.reshape(b * c, -1)
        y = self.head(h).view(b, c, self.pred_len).permute(0, 2, 1)
        if self.revin is not None:
            y = self.revin(y, "denorm")
        return y


class iTransformer(nn.Module):
    """Compact iTransformer: attention over variates."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        n_heads: int = 4,
        e_layers: int = 2,
        dropout: float = 0.1,
        use_revin: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.use_revin = use_revin
        self.revin = RevIN(enc_in) if use_revin else None
        self.embed = nn.Linear(seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=e_layers)
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.revin is not None:
            x = self.revin(x, "norm")
        # (B, L, C) -> (B, C, L)
        h = x.permute(0, 2, 1)
        h = self.embed(h)
        h = self.encoder(h)
        y = self.head(h).permute(0, 2, 1)  # (B, H, C)
        if self.revin is not None:
            y = self.revin(y, "denorm")
        return y


class InformerLite(nn.Module):
    """Lightweight Informer-style encoder (full ProbSparse omitted; standard attn)."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        n_heads: int = 4,
        e_layers: int = 2,
        dropout: float = 0.1,
        use_revin: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.use_revin = use_revin
        self.revin = RevIN(enc_in) if use_revin else None
        self.in_proj = nn.Linear(enc_in, d_model)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=e_layers)
        self.time_proj = nn.Linear(seq_len, pred_len)
        self.out = nn.Linear(d_model, enc_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.revin is not None:
            x = self.revin(x, "norm")
        h = self.in_proj(x) + self.pos[:, : x.size(1)]
        h = self.encoder(h)
        h = self.time_proj(h.transpose(1, 2)).transpose(1, 2)
        y = self.out(h)
        if self.revin is not None:
            y = self.revin(y, "denorm")
        return y


class AutoformerLite(nn.Module):
    """Series-decomp + moving average + linear (Autoformer spirit, compact)."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        moving_avg: int = 25,
        d_model: int = 64,
        use_revin: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.use_revin = use_revin
        self.revin = RevIN(enc_in) if use_revin else None
        kernel = moving_avg if moving_avg % 2 == 1 else moving_avg + 1
        self.avg = nn.AvgPool1d(kernel, stride=1, padding=kernel // 2)
        self.seasonal = nn.Linear(seq_len, pred_len)
        self.trend = nn.Linear(seq_len, pred_len)
        self.mix = nn.Linear(enc_in, enc_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.revin is not None:
            x = self.revin(x, "norm")
        trend = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)[:, : x.size(1)]
        seasonal = x - trend
        s = self.seasonal(seasonal.permute(0, 2, 1)).permute(0, 2, 1)
        t = self.trend(trend.permute(0, 2, 1)).permute(0, 2, 1)
        y = self.mix(s + t)
        if self.revin is not None:
            y = self.revin(y, "denorm")
        return y


class FEDformerLite(nn.Module):
    """Frequency-enhanced linear block (FEDformer spirit)."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        modes: int = 32,
        use_revin: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.modes = modes
        self.use_revin = use_revin
        self.revin = RevIN(enc_in) if use_revin else None
        self.freq_len = seq_len // 2 + 1
        self.modes = min(modes, self.freq_len)
        self.comp = nn.Parameter(
            torch.randn(2, self.modes, enc_in, enc_in) * 0.02
        )  # real/imag
        self.out_proj = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.revin is not None:
            x = self.revin(x, "norm")
        Xf = torch.fft.rfft(x, dim=1)  # (B, F, C)
        m = self.modes
        weight = torch.complex(self.comp[0], self.comp[1])  # (m, C, C)
        out_f = torch.zeros_like(Xf)
        # low-frequency modes
        Xm = Xf[:, :m, :]  # (B, m, C)
        # (B, m, C) x (m, C, C) -> (B, m, C)
        out_f[:, :m, :] = torch.einsum("bmc,mcd->bmd", Xm, weight)
        x_t = torch.fft.irfft(out_f, n=x.size(1), dim=1)
        y = self.out_proj(x_t.permute(0, 2, 1)).permute(0, 2, 1)
        if self.revin is not None:
            y = self.revin(y, "denorm")
        return y


class ARIMABaseline(nn.Module):
    """Non-trainable ARIMA wrapper used in eval (per-batch fit on CPU). Slow — for small tests."""

    def __init__(self, seq_len: int, pred_len: int, enc_in: int, order=(1, 0, 1), **kwargs):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.order = order
        # dummy param so optimizer doesn't break
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError:
            return x[:, -1:, :].repeat(1, self.pred_len, 1)

        b, l, c = x.shape
        device = x.device
        outs = []
        x_np = x.detach().cpu().numpy()
        for bi in range(b):
            chans = []
            for ci in range(c):
                series = x_np[bi, :, ci]
                try:
                    model = ARIMA(series, order=self.order)
                    fit = model.fit()
                    fc = fit.forecast(self.pred_len)
                except Exception:
                    fc = [series[-1]] * self.pred_len
                chans.append(fc)
            outs.append(chans)
        import numpy as np

        arr = np.transpose(np.array(outs), (0, 2, 1))  # (B, H, C)
        return torch.from_numpy(arr).float().to(device)


MODEL_REGISTRY = {
    "TimeKAN": TimeKAN,
    "PlainKAN": PlainKAN,
    "Naive": NaiveLast,
    "ARIMA": ARIMABaseline,
    "DLinear": DLinear,
    "NLinear": NLinear,
    "LSTM": LSTMModel,
    "TCN": TCNModel,
    "PatchTST": PatchTST,
    "iTransformer": iTransformer,
    "Informer": InformerLite,
    "Autoformer": AutoformerLite,
    "FEDformer": FEDformerLite,
}


def build_model(name: str, **kwargs) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name}. Choose from {list(MODEL_REGISTRY)}")
    import inspect

    cls = MODEL_REGISTRY[name]
    sig = inspect.signature(cls.__init__)
    params = sig.parameters
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_var_kw:
        return cls(**kwargs)
    allowed = {k: v for k, v in kwargs.items() if k in params}
    return cls(**allowed)
