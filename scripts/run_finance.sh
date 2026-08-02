#!/usr/bin/env bash
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python scripts/download_data.py --datasets Finance
python exp/train.py --config configs/timekan_finance.yaml
python -c "from exp.train import aggregate_summary; aggregate_summary()"
python scripts/make_figures.py
