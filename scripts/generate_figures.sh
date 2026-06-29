#!/usr/bin/env bash
# generate_figures.sh — Regenerate all publication figures
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
echo "Generating publication figures..."
python3 benchmark/pub_figures_final.py
echo "Figures saved to paper/figures/"
