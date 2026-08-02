#!/usr/bin/env bash
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
for cfg in configs/ablations/*.yaml; do
  echo "=== $cfg ==="
  python exp/train.py --config "$cfg"
done
python -c "from exp.train import aggregate_summary; aggregate_summary()"
python scripts/make_figures.py
