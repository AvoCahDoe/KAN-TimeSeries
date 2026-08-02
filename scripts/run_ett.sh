#!/usr/bin/env bash
# Quick ETT smoke: TimeKAN + key baselines on ETTh1 H=96
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python scripts/download_data.py --datasets ETTh1
python exp/train.py --config configs/timekan_etth1_h96.yaml
python exp/train.py --config configs/dlinear_etth1_h96.yaml
python exp/train.py --config configs/plainkan_etth1_h96.yaml
python exp/train.py --config configs/lstm_etth1_h96.yaml
python -c "from exp.train import aggregate_summary; aggregate_summary()"
python scripts/make_figures.py
echo "Done. See results/ and figures/"
