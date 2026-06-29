#!/usr/bin/env bash
# run_synthetic.sh — Run all synthetic MC experiments
set -euo pipefail
N_MC=${N_MC:-50}
WORKERS=${WORKERS:-8}
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
echo "Running synthetic benchmark (N=$N_MC, workers=$WORKERS)..."
python3 benchmark/run_full_benchmark.py --phase standard --n_mc "$N_MC" --workers "$WORKERS"
python3 benchmark/run_full_benchmark.py --phase dropout  --n_mc "$N_MC" --workers "$WORKERS"
python3 benchmark/run_full_benchmark.py --phase hyperparam --n_mc 15   --workers "$WORKERS"
python3 benchmark/run_ablations.py
python3 -c "
import sys,os; sys.path.insert(0,'.')
from benchmark.pub_figures_final import main as run_thresh_sweep
# threshold sweep runs automatically in ablations
"
echo "Synthetic experiments complete."
