#!/usr/bin/env python
"""Download or synthesize datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import download_ett, download_finance, ensure_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity", "Finance"],
    )
    args = p.parse_args()
    for name in args.datasets:
        if name.startswith("ETT"):
            download_ett([name] if name in ("ETTh1", "ETTh2", "ETTm1", "ETTm2") else None)
        path = ensure_dataset(name)
        print(f"OK {name}: {path}")


if __name__ == "__main__":
    main()
