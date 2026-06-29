#!/usr/bin/env bash
# reproduce_all.sh — Single command to reproduce all results from scratch.
# Usage: bash reproduce_all.sh [--n_mc N] [--workers W]
# Default: N=50 (fast estimators), N=25 (UKF variants), 8 workers
set -euo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"
N_MC=${N_MC:-50}
WORKERS=${WORKERS:-8}

echo "================================================================"
echo "  EKF Multi-Sensor Fusion — Full Reproduction"
echo "  Estimators: EKF, UKF, Adaptive-EKF, Adaptive-UKF,"
echo "              ES-EKF, MACE-EKF, MACE-UKF"
echo "  N_MC=$N_MC (fast); N=25 (UKF variants)"
echo "================================================================"

echo ""
echo "[1/7] Install dependencies..."
pip install -r requirements.txt -q

echo ""
echo "[2/7] Run test suite (199 tests)..."
python3 -m pytest tests/ -q --tb=short
echo "  PASSED: 199/199"

echo ""
echo "[3/7] Standard benchmark (7 est × 3 traj × 3 noise × N=$N_MC)..."
export N_MC WORKERS
python3 -c "
import sys,os,csv; sys.path.insert(0,'.')
from concurrent.futures import ProcessPoolExecutor,as_completed
from benchmark.run_full_benchmark import run_single, aggregate_stats, save_csv, RESULTS_DIR
FAST=['EKF','ES-EKF','Adaptive-EKF','MACE-EKF']
SLOW=['UKF','Adaptive-UKF','MACE-UKF']
TRAJ=['figure8','circle','straight']
NOISE=['low','medium','high']
N_F=int(os.environ.get('N_MC',50))
N_S=min(N_F,25)
tasks=[(n,t,r,s,0.0) for n in FAST for t in TRAJ for r in NOISE for s in range(N_F)]
tasks+=[(n,t,r,s,0.0) for n in SLOW for t in TRAJ for r in NOISE for s in range(N_S)]
print(f'  Tasks: {len(tasks)}')
raw=[]
with ProcessPoolExecutor(max_workers=int(os.environ.get('WORKERS',8))) as ex:
    futs=[ex.submit(run_single,*t) for t in tasks]
    for i,f in enumerate(as_completed(futs)):
        raw.append(f.result())
        if (i+1)%200==0: print(f'  {i+1}/{len(tasks)}')
save_csv(raw,os.path.join(RESULTS_DIR,'n50_benchmark_standard.csv'))
save_csv(aggregate_stats(raw,['estimator','trajectory','noise_regime']),
         os.path.join(RESULTS_DIR,'n50_stats_standard.csv'))
print(f'  Done: {len(raw)} runs')
"

echo ""
echo "[4/7] Dropout robustness benchmark..."
python3 -c "
import sys,os; sys.path.insert(0,'.')
from concurrent.futures import ProcessPoolExecutor,as_completed
from benchmark.run_full_benchmark import run_single, aggregate_stats, save_csv, RESULTS_DIR
FAST=['EKF','ES-EKF','Adaptive-EKF','MACE-EKF']
SLOW=['UKF','Adaptive-UKF','MACE-UKF']
DROPOUTS=[0.0,0.10,0.30,0.50,0.70]
N_F=int(os.environ.get('N_MC',50)); N_S=min(N_F,25)
tasks=[(n,'figure8','medium',s,dr) for n in FAST for dr in DROPOUTS for s in range(N_F)]
tasks+=[(n,'figure8','medium',s,dr) for n in SLOW for dr in DROPOUTS for s in range(N_S)]
raw=[]
with ProcessPoolExecutor(max_workers=int(os.environ.get('WORKERS',8))) as ex:
    futs=[ex.submit(run_single,*t) for t in tasks]
    for f in as_completed(futs): raw.append(f.result())
save_csv(raw,os.path.join(RESULTS_DIR,'n50_benchmark_dropout.csv'))
save_csv(aggregate_stats(raw,['estimator','dropout']),
         os.path.join(RESULTS_DIR,'n50_stats_dropout.csv'))
print(f'  Done: {len(raw)} dropout runs')
"

echo ""
echo "[5/7] Ablation study + threshold sweep + hyperparam grid..."
python3 benchmark/run_ablations.py
python3 -c "
import sys,os; sys.path.insert(0,'.')
from benchmark.run_full_benchmark import run_hyperparam_sensitivity, save_csv, RESULTS_DIR
rows=run_hyperparam_sensitivity(n_mc=15)
save_csv(rows,os.path.join(RESULTS_DIR,'hyperparam_sensitivity.csv'))
print(f'  Hyperparam: {len(rows)} cells')
"

echo ""
echo "[6/7] Generate all 12 publication figures..."
python3 benchmark/pub_figures_final.py

echo ""
echo "[7/7] Generate LaTeX tables..."
python3 paper/tables/generate_tables.py

echo ""
echo "================================================================"
echo "  Reproduction complete."
echo "  Results:    results/"
echo "  Figures:    paper/figures/ (12 figures, PDF+PNG)"
echo "  Tables:     paper/tables/"
echo "  Tests:      199/199 passed"
echo "================================================================"
