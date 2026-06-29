#!/usr/bin/env bash
# run_tests.sh — Run full test suite
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
echo "Running 199 unit tests..."
python3 -m pytest tests/ -v --tb=short
echo "All tests passed."
