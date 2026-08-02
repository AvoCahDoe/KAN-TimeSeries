"""Quick sanity checks for core modules."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines import build_model
from timekan import TimeKAN, PlainKAN, FixedBandDecomp, AdaptiveBandDecomp


def test_shapes():
    b, l, c, h = 4, 96, 7, 96
    x = torch.randn(b, l, c)
    for name in [
        "TimeKAN",
        "PlainKAN",
        "DLinear",
        "NLinear",
        "LSTM",
        "TCN",
        "PatchTST",
        "iTransformer",
        "Informer",
        "Autoformer",
        "FEDformer",
        "Naive",
    ]:
        m = build_model(name, seq_len=l, pred_len=h, enc_in=c, d_model=32, k_bands=3)
        y = m(x)
        assert y.shape == (b, h, c), f"{name}: {y.shape}"
        print(f"OK {name} -> {tuple(y.shape)}")

    # aux
    m = TimeKAN(seq_len=l, pred_len=h, enc_in=c, d_model=32, k_bands=3)
    y, aux = m(x, return_aux=True)
    assert aux["attn"] is not None
    assert len(aux["bands"]) == 3
    print("OK TimeKAN aux")

    bands, masks = FixedBandDecomp(3)(x)
    assert len(bands) == 3
    bands2, soft = AdaptiveBandDecomp(l, c, 3)(x)
    assert soft.shape[1] == 3
    print("OK decomp")


if __name__ == "__main__":
    test_shapes()
    print("All sanity checks passed.")
