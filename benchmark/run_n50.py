"""
N=50 Monte Carlo Upgrade
Runs all 7 estimators × 3 trajectories × 3 noise × N=50 MC.
Seeds 0..49 for reproducibility.
Author: Medisetti Renukeswar
"""
from __future__ import annotations
import csv, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concurrent.futures import ProcessPoolExecutor, as_completed
from benchmark.run_full_benchmark import run_single, aggregate_stats, save_csv, RESULTS_DIR

N_MC = 50
ESTIMATORS = ['EKF','UKF','Adaptive-EKF','Adaptive-UKF','ES-EKF','MACE-EKF','MACE-UKF']
TRAJECTORIES = ['figure8','circle','straight']
NOISE_REGIMES = ['low','medium','high']
DROPOUT_RATES = [0.0, 0.10, 0.30, 0.50, 0.70]

def run_standard():
    tasks = [(n,t,r,s,0.0) for n in ESTIMATORS for t in TRAJECTORIES
             for r in NOISE_REGIMES for s in range(N_MC)]
    raw = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(run_single, *t) for t in tasks]
        for i, f in enumerate(as_completed(futs)):
            raw.append(f.result())
            if (i+1) % 200 == 0:
                print(f'  standard {i+1}/{len(tasks)}')
    save_csv(raw, os.path.join(RESULTS_DIR, 'n50_benchmark_standard.csv'))
    stats = aggregate_stats(raw, ['estimator','trajectory','noise_regime'])
    save_csv(stats, os.path.join(RESULTS_DIR, 'n50_stats_standard.csv'))
    print(f'Standard: {len(raw)} runs, {len(stats)} conditions')
    return raw, stats

def run_dropout():
    tasks = [(n,'figure8','medium',s,dr) for n in ESTIMATORS
             for dr in DROPOUT_RATES for s in range(N_MC)]
    raw = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(run_single, *t) for t in tasks]
        for i, f in enumerate(as_completed(futs)):
            raw.append(f.result())
            if (i+1) % 100 == 0:
                print(f'  dropout {i+1}/{len(tasks)}')
    save_csv(raw, os.path.join(RESULTS_DIR, 'n50_benchmark_dropout.csv'))
    stats = aggregate_stats(raw, ['estimator','dropout'])
    save_csv(stats, os.path.join(RESULTS_DIR, 'n50_stats_dropout.csv'))
    print(f'Dropout: {len(raw)} runs, {len(stats)} conditions')
    return raw, stats

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['standard','dropout','all'], default='all')
    args = p.parse_args()
    if args.phase in ('standard','all'):
        print('Running N=50 standard benchmark...')
        run_standard()
    if args.phase in ('dropout','all'):
        print('Running N=50 dropout benchmark...')
        run_dropout()
    print('Done.')
