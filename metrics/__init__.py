"""Forecasting metrics + finance extras."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def mse(pred, true) -> float:
    p, t = _to_numpy(pred), _to_numpy(true)
    return float(np.mean((p - t) ** 2))


def mae(pred, true) -> float:
    p, t = _to_numpy(pred), _to_numpy(true)
    return float(np.mean(np.abs(p - t)))


def rmse(pred, true) -> float:
    return float(np.sqrt(mse(pred, true)))


def mape(pred, true, eps: float = 1e-5) -> float:
    p, t = _to_numpy(pred), _to_numpy(true)
    denom = np.maximum(np.abs(t), eps)
    return float(np.mean(np.abs((p - t) / denom)) * 100.0)


def directional_accuracy(pred, true) -> float:
    """Sign agreement of returns (or first-diff of levels)."""
    p, t = _to_numpy(pred), _to_numpy(true)
    # if multi-step, use last dim flatten
    ps = np.sign(p.reshape(-1))
    ts = np.sign(t.reshape(-1))
    mask = ts != 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(ps[mask] == ts[mask]))


def illustrative_sharpe(
    pred_returns,
    true_returns,
    periods_per_year: int = 252,
) -> float:
    """Long/flat strategy: hold when predicted return > 0.

    Clearly illustrative — not a trading claim.
    Uses channel 0 / step 0 when inputs are multi-dimensional.
    """
    p = _to_numpy(pred_returns)
    t = _to_numpy(true_returns)
    while p.ndim > 1:
        p = p[..., 0]
    while t.ndim > 1:
        t = t[..., 0]
    p = p.reshape(-1)
    t = t.reshape(-1)
    n = min(len(p), len(t))
    p, t = p[:n], t[:n]
    positions = (p > 0).astype(np.float64)
    pnl = positions * t
    if pnl.std() < 1e-12:
        return 0.0
    return float(np.sqrt(periods_per_year) * pnl.mean() / pnl.std())

def compute_metrics(pred, true, finance: bool = False) -> Dict[str, float]:
    out = {
        "mse": mse(pred, true),
        "mae": mae(pred, true),
        "rmse": rmse(pred, true),
        "mape": mape(pred, true),
    }
    if finance:
        out["directional_accuracy"] = directional_accuracy(pred, true)
        out["illustrative_sharpe"] = illustrative_sharpe(pred, true)
    return out


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
