#!/usr/bin/env python
"""Sweep main table: models × datasets × horizons (budget-aware for 8GB GPU)."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exp.train import train_one, aggregate_summary

DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather"]
HORIZONS = [96, 192, 336, 720]
MODELS = [
    "Naive",
    "DLinear",
    "NLinear",
    "LSTM",
    "TCN",
    "PlainKAN",
    "TimeKAN",
    "PatchTST",
    "iTransformer",
    "Informer",
    "Autoformer",
    "FEDformer",
]


def base_cfg(dataset, model, horizon, seed=2021, epochs=8):
    max_ch = 50 if dataset == "Electricity" else None
    cfg = {
        "seed": seed,
        "device": "cuda",
        "data": {
            "dataset": dataset,
            "seq_len": 96 if not dataset.startswith("ETTm") else 96,
            "pred_len": horizon,
            "max_channels": max_ch,
        },
        "model": {
            "name": model,
            "d_model": 48 if dataset == "Electricity" else 64,
            "k_bands": 3,
            "decomp_mode": "adaptive",
            "fusion_mode": "attention",
            "n_kan_layers": 2,
            "basis": "chebyshev",
            "use_revin": True,
            "dropout": 0.1,
        },
        "train": {
            "batch_size": 16 if dataset == "Electricity" else 32,
            "epochs": epochs,
            "lr": 0.001,
            "patience": 3,
        },
    }
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["ETTh1"])
    p.add_argument("--horizons", nargs="+", type=int, default=[96])
    p.add_argument("--models", nargs="+", default=["TimeKAN", "DLinear", "PlainKAN", "Naive"])
    p.add_argument("--seeds", nargs="+", type=int, default=[2021])
    p.add_argument("--epochs", type=int, default=8)
    args = p.parse_args()

    for ds in args.datasets:
        for h in args.horizons:
            for model in args.models:
                for seed in args.seeds:
                    cfg = base_cfg(ds, model, h, seed=seed, epochs=args.epochs)
                    print(f"\n==== {ds} {model} H={h} seed={seed} ====")
                    try:
                        train_one(cfg)
                    except Exception as e:
                        print(f"FAILED: {e}")
    aggregate_summary()


if __name__ == "__main__":
    main()
