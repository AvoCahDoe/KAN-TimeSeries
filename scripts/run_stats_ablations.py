#!/usr/bin/env python
"""Multi-seed + K-band ablation helper + Wilcoxon aggregation."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exp.train import train_one, aggregate_summary, load_config


def run_k_bands(epochs: int = 6):
    for k in (1, 2, 3, 4, 5):
        cfg = {
            "seed": 2021,
            "device": "cuda",
            "tag": f"k_bands_{k}",
            "data": {"dataset": "ETTh1", "seq_len": 96, "pred_len": 96},
            "model": {
                "name": "TimeKAN",
                "d_model": 64,
                "k_bands": k,
                "decomp_mode": "fixed",
                "fusion_mode": "attention",
                "basis": "chebyshev",
                "use_revin": True,
            },
            "train": {"batch_size": 32, "epochs": epochs, "lr": 0.001, "patience": 3},
        }
        print(f"=== K={k} ===")
        train_one(cfg)


def run_seeds(models=("TimeKAN", "DLinear", "PlainKAN"), seeds=(2021, 2022, 2023), epochs=6):
    for model in models:
        for seed in seeds:
            cfg = {
                "seed": seed,
                "device": "cuda",
                "data": {"dataset": "ETTh1", "seq_len": 96, "pred_len": 96},
                "model": {
                    "name": model,
                    "d_model": 64,
                    "k_bands": 3,
                    "decomp_mode": "adaptive",
                    "fusion_mode": "attention",
                    "basis": "chebyshev",
                    "use_revin": True,
                },
                "train": {"batch_size": 32, "epochs": epochs, "lr": 0.001, "patience": 3},
            }
            print(f"=== {model} seed={seed} ===")
            train_one(cfg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["k_bands", "seeds", "both"], default="both")
    p.add_argument("--epochs", type=int, default=6)
    args = p.parse_args()
    if args.mode in ("k_bands", "both"):
        run_k_bands(args.epochs)
    if args.mode in ("seeds", "both"):
        run_seeds(epochs=args.epochs)
    aggregate_summary()


if __name__ == "__main__":
    main()
