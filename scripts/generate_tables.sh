#!/usr/bin/env bash
# generate_tables.sh — Regenerate all LaTeX tables
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
echo "Generating LaTeX tables..."
python3 paper/tables/generate_tables.py
echo "Tables saved to paper/tables/"
