#!/usr/bin/env python
"""Regenerate all figures from results/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exp.train import aggregate_summary
from viz import make_all_figures


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=str, default=str(ROOT / "results"))
    p.add_argument("--out", type=str, default=str(ROOT / "figures"))
    args = p.parse_args()
    results = Path(args.results)
    aggregate_summary(results)
    make_all_figures(results, Path(args.out))


if __name__ == "__main__":
    main()
