"""
Full Benchmark Runner — Phase 8 (All Estimators)
=================================================
Runs MC experiments for all 6 estimators:
  EKF, UKF, Adaptive-EKF, Adaptive-UKF, ES-EKF, MACE-EKF, MACE-UKF

Conditions:
  3 trajectories × 3 noise regimes × N_MC=30 = 630 runs (standard)
  + 4 VO-dropout levels × 7 estimators × N_MC=30 = 840 runs (degradation)

Outputs:
  results/full_benchmark_standard.csv
  results/full_benchmark_dropout.csv
  results/full_stats_standard.csv
  results/full_stats_dropout.csv

Author: Medisetti Renukeswar
"""

from __future__ import annotations

import csv
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict

import numpy as np
from scipy.stats import wilcoxon, chi2 as chi2_dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.trajectories import TrajectoryGenerator, TrajectoryType
from simulation.research_sensor_sim import ResearchSensorSimulator, NoiseRegime
from ekf_core.ekf_estimator import EKFEstimator
from ekf_core.ukf_estimator import UKFEstimator
from ekf_core.adaptive_estimator import AdaptiveEKFEstimator, AdaptiveUKFEstimator
from ekf_core.esekf_estimator import ESEKFEstimator
from ekf_core.mace_estimator import MACEEKFEstimator, MACEUKFEstimator
from ekf_core.metrics import (
    compute_ate, compute_rpe, compute_rmse_position, compute_rmse_heading,
    compute_nees, average_nees_bounds,
)

# ── Config ───────────────────────────────────────────────────────────────────
N_MC         = 30
SIM_DURATION = 40.0
DT_IMU       = 0.01
DT_CAM       = 1 / 30
LOG_EVERY    = 10

TRAJECTORIES: list[str] = ["figure8", "circle", "straight"]
NOISE_REGIMES: list[str] = ["low", "medium", "high"]
DROPOUT_RATES: list[float] = [0.0, 0.10, 0.30, 0.50, 0.70]

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Chi-squared consistency bounds for N_MC=30, dof=3
_LB, _UB = average_nees_bounds(dof=3, n_runs=N_MC)

ESTIMATOR_NAMES = ["EKF", "UKF", "Adaptive-EKF", "Adaptive-UKF", "ES-EKF", "MACE-EKF", "MACE-UKF"]


def make_estimator(name: str) -> object:
    """Factory: create a fresh estimator instance by name."""
    if name == "EKF":
        return EKFEstimator()
    elif name == "UKF":
        return UKFEstimator()
    elif name == "Adaptive-EKF":
        return AdaptiveEKFEstimator()
    elif name == "Adaptive-UKF":
        return AdaptiveUKFEstimator()
    elif name == "ES-EKF":
        return ESEKFEstimator()
    elif name == "MACE-EKF":
        return MACEEKFEstimator()
    elif name == "MACE-UKF":
        return MACEUKFEstimator()
    raise ValueError(f"Unknown estimator: {name}")


# ── Single run ───────────────────────────────────────────────────────────────

def run_single(
    estimator_name: str,
    trajectory: str,
    noise_regime: str,
    seed: int,
    dropout_rate: float = 0.0,
) -> dict:
    """
    Execute one full simulation run and return a metrics dict.

    Parameters
    ----------
    dropout_rate : Fraction of camera updates to suppress [0, 1].
    """
    rng = np.random.default_rng(seed)

    traj = TrajectoryGenerator(trajectory_type=trajectory, scale=3.0)
    sim  = ResearchSensorSimulator(
        trajectory=traj, noise_regime=noise_regime,
        dt_imu=DT_IMU, dt_cam=DT_CAM, seed=seed,
    )
    est = make_estimator(estimator_name)

    # Initialise at t=0 ground truth
    px0, py0, th0, vx0, vy0, om0 = traj.get_state(0.0)
    x0 = np.array([px0, py0, th0, vx0, vy0, om0])
    P0 = np.diag([0.5, 0.5, 0.3, 0.5, 0.5, 0.1])
    est.reset(x0, P0)

    gt_x, gt_y, gt_th = [], [], []
    est_x, est_y, est_th = [], [], []
    nis_list, nees_list = [], []

    t = 0.0
    cam_timer = 0.0
    step = 0
    t_start = time.perf_counter()

    while t <= SIM_DURATION:
        px_gt, py_gt, th_gt, vx_gt, vy_gt, om_gt = traj.get_state(t)

        vx_m, vy_m, om_m = sim.get_imu(t)
        est.predict(vx_m, vy_m, om_m)

        cam_timer += DT_IMU
        if cam_timer >= DT_CAM:
            cam_timer = 0.0
            # Apply dropout
            if rng.random() >= dropout_rate:
                px_c, py_c, th_c = sim.get_camera(t)
                result = est.update_camera(px_c, py_c, th_c)
                nis_list.append(result.get("nis", float("nan")))

        if step % LOG_EVERY == 0:
            x_est, P_est = est.get_state()
            gt_x.append(px_gt); gt_y.append(py_gt); gt_th.append(th_gt)
            est_x.append(x_est[0]); est_y.append(x_est[1]); est_th.append(x_est[2])

            # NEES on pos+heading sub-state
            x_true = np.array([px_gt, py_gt, th_gt])
            x_e    = np.array([x_est[0], x_est[1], x_est[2]])
            P_sub  = P_est[np.ix_([0,1,2],[0,1,2])]
            nees_val = compute_nees(
                np.concatenate([x_true, np.zeros(3)]),
                np.concatenate([x_e, np.zeros(3)]),
                np.block([[P_sub, np.zeros((3,3))],[np.zeros((3,3)), np.eye(3)]]),
                state_indices=[0,1,2],
            )
            nees_list.append(nees_val)

        t += DT_IMU
        step += 1

    runtime_ms = (time.perf_counter() - t_start) * 1000.0

    gt_x = np.array(gt_x); gt_y = np.array(gt_y)
    est_x = np.array(est_x); est_y = np.array(est_y)

    ate       = compute_ate(gt_x, gt_y, est_x, est_y)
    rpe       = compute_rpe(gt_x, gt_y, est_x, est_y)
    rmse_pos  = compute_rmse_position(gt_x, gt_y, est_x, est_y)
    rmse_hdg  = compute_rmse_heading(np.array(gt_th), np.array(est_th))

    valid_nis = [v for v in nis_list if not math.isnan(v)]
    valid_nees = [v for v in nees_list if not math.isnan(v)]
    mean_nis   = float(np.mean(valid_nis))  / 3.0 if valid_nis else float("nan")  # ANIS
    mean_nees  = float(np.mean(valid_nees)) / 3.0 if valid_nees else float("nan")  # ANEES

    return dict(
        estimator=estimator_name, trajectory=trajectory,
        noise_regime=noise_regime, dropout=dropout_rate, seed=seed,
        ate=ate, rpe=rpe, rmse_pos=rmse_pos, rmse_hdg=rmse_hdg,
        anis=mean_nis, anees=mean_nees,
        runtime_ms=runtime_ms,
    )


# ── Aggregate stats ──────────────────────────────────────────────────────────

def aggregate_stats(rows: list[dict], group_keys: list[str]) -> list[dict]:
    """Compute mean±std, 95%CI, consistency verdicts per group."""
    from itertools import groupby
    import operator

    sorted_rows = sorted(rows, key=lambda r: tuple(r[k] for k in group_keys))
    results = []
    for key_vals, group in groupby(sorted_rows, key=lambda r: tuple(r[k] for k in group_keys)):
        g = list(group)
        ate_vals  = np.array([r["ate"]  for r in g])
        rpe_vals  = np.array([r["rpe"]  for r in g])
        anis_vals = np.array([r["anis"] for r in g if not math.isnan(r["anis"])])
        anees_vals= np.array([r["anees"]for r in g if not math.isnan(r["anees"])])
        rt_vals   = np.array([r["runtime_ms"] for r in g])
        n = len(ate_vals)

        def ci95(arr):
            if len(arr) < 2:
                return 0.0
            return 1.96 * np.std(arr, ddof=1) / math.sqrt(len(arr))

        anis_mean  = float(np.mean(anis_vals))  if len(anis_vals)  else float("nan")
        anees_mean = float(np.mean(anees_vals)) if len(anees_vals) else float("nan")
        nis_ok  = _LB <= anis_mean  <= _UB if not math.isnan(anis_mean)  else False
        nees_ok = _LB <= anees_mean <= _UB if not math.isnan(anees_mean) else False

        row = {k: v for k, v in zip(group_keys, key_vals)}
        row.update(dict(
            n=n,
            ate_mean=float(np.mean(ate_vals)), ate_std=float(np.std(ate_vals, ddof=1)),
            ate_ci=ci95(ate_vals),
            rpe_mean=float(np.mean(rpe_vals)), rpe_std=float(np.std(rpe_vals, ddof=1)),
            anis_mean=anis_mean,
            anis_std=float(np.std(anis_vals, ddof=1)) if len(anis_vals)>1 else 0.0,
            anees_mean=anees_mean,
            anees_std=float(np.std(anees_vals, ddof=1)) if len(anees_vals)>1 else 0.0,
            nis_ok=nis_ok, nees_ok=nees_ok,
            rt_mean=float(np.mean(rt_vals)), rt_std=float(np.std(rt_vals, ddof=1)),
            lb=_LB, ub=_UB,
        ))
        results.append(row)
    return results


def save_csv(rows: list[dict], path: str) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {path}")


# ── Hyperparameter sensitivity ────────────────────────────────────────────────

def run_hyperparam_sensitivity(n_mc: int = 15) -> list[dict]:
    """Grid search over W ∈ {5,10,20,50} and α ∈ {0.05,0.1,0.2}."""
    W_vals = [5, 10, 20, 50]
    A_vals = [0.05, 0.1, 0.2]
    rows = []
    total = len(W_vals) * len(A_vals) * n_mc
    done = 0
    for W in W_vals:
        for alpha in A_vals:
            run_rows = []
            for seed in range(n_mc):
                rng = np.random.default_rng(seed + 9000)
                traj = TrajectoryGenerator(trajectory_type="figure8", scale=3.0)
                sim  = ResearchSensorSimulator(
                    trajectory=traj, noise_regime="medium",
                    dt_imu=DT_IMU, dt_cam=DT_CAM, seed=seed + 9000,
                )
                est = AdaptiveEKFEstimator(window=W, alpha_smooth=alpha)
                px0, py0, th0, vx0, vy0, om0 = traj.get_state(0.0)
                est.reset(np.array([px0, py0, th0, vx0, vy0, om0]),
                          np.diag([0.5,0.5,0.3,0.5,0.5,0.1]))
                gt_x, gt_y, gt_th, est_x, est_y, est_th, nis_list = [], [], [], [], [], [], []
                t = 0.0; cam_timer = 0.0; step = 0
                while t <= SIM_DURATION:
                    px_gt, py_gt, th_gt, vx_gt, vy_gt, om_gt = traj.get_state(t)
                    vx_m, vy_m, om_m = sim.get_imu(t)
                    est.predict(vx_m, vy_m, om_m)
                    cam_timer += DT_IMU
                    if cam_timer >= DT_CAM:
                        cam_timer = 0.0
                        px_c, py_c, th_c = sim.get_camera(t)
                        result = est.update_camera(px_c, py_c, th_c)
                        nis_list.append(result.get("nis", float("nan")))
                    if step % LOG_EVERY == 0:
                        x_est, _ = est.get_state()
                        gt_x.append(px_gt); gt_y.append(py_gt)
                        est_x.append(x_est[0]); est_y.append(x_est[1])
                    t += DT_IMU; step += 1
                valid_nis = [v for v in nis_list if not math.isnan(v)]
                anis = float(np.mean(valid_nis)) / 3.0 if valid_nis else float("nan")
                ate  = compute_ate(np.array(gt_x), np.array(gt_y), np.array(est_x), np.array(est_y))
                run_rows.append(dict(W=W, alpha=alpha, seed=seed, ate=ate, anis=anis))
                done += 1
            # Aggregate
            ates   = [r["ate"]  for r in run_rows]
            aniss  = [r["anis"] for r in run_rows if not math.isnan(r["anis"])]
            anis_m = float(np.mean(aniss)) if aniss else float("nan")
            rows.append(dict(
                W=W, alpha=alpha,
                ate_mean=float(np.mean(ates)), ate_std=float(np.std(ates, ddof=1)),
                anis_mean=anis_m,
                nis_ok=(_LB <= anis_m <= _UB) if not math.isnan(anis_m) else False,
            ))
            print(f"  Hyperparam W={W} α={alpha}: ATE={np.mean(ates):.4f} ANIS={anis_m:.3f}")
    return rows


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["standard","dropout","hyperparam","all"], default="all")
    parser.add_argument("--n_mc", type=int, default=N_MC)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    print("=" * 60)
    print("  Full Benchmark — All Estimators")
    print(f"  N_MC={args.n_mc}, workers={args.workers}")
    print("=" * 60)

    if args.phase in ("standard", "all"):
        print("\n[1/3] Standard benchmark (all estimators × trajectories × noise)...")
        tasks = []
        for name in ESTIMATOR_NAMES:
            for traj in TRAJECTORIES:
                for noise in NOISE_REGIMES:
                    for seed in range(args.n_mc):
                        tasks.append((name, traj, noise, seed, 0.0))

        raw_rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_single, *t): t for t in tasks}
            done = 0
            for fut in as_completed(futs):
                raw_rows.append(fut.result())
                done += 1
                if done % 100 == 0:
                    print(f"    {done}/{len(tasks)} runs complete")

        save_csv(raw_rows, os.path.join(RESULTS_DIR, "full_benchmark_standard.csv"))
        stats = aggregate_stats(raw_rows, ["estimator", "trajectory", "noise_regime"])
        save_csv(stats, os.path.join(RESULTS_DIR, "full_stats_standard.csv"))
        print(f"  Standard benchmark complete: {len(raw_rows)} runs")

    if args.phase in ("dropout", "all"):
        print("\n[2/3] Dropout robustness benchmark...")
        tasks = []
        for name in ESTIMATOR_NAMES:
            for dropout in DROPOUT_RATES:
                for seed in range(args.n_mc):
                    tasks.append((name, "figure8", "medium", seed, dropout))

        raw_rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_single, *t): t for t in tasks}
            done = 0
            for fut in as_completed(futs):
                raw_rows.append(fut.result())
                done += 1
                if done % 50 == 0:
                    print(f"    {done}/{len(tasks)} dropout runs complete")

        save_csv(raw_rows, os.path.join(RESULTS_DIR, "full_benchmark_dropout.csv"))
        stats = aggregate_stats(raw_rows, ["estimator", "dropout"])
        save_csv(stats, os.path.join(RESULTS_DIR, "full_stats_dropout.csv"))
        print(f"  Dropout benchmark complete: {len(raw_rows)} runs")

    if args.phase in ("hyperparam", "all"):
        print("\n[3/3] Hyperparameter sensitivity (W × α grid)...")
        hp_rows = run_hyperparam_sensitivity(n_mc=15)
        save_csv(hp_rows, os.path.join(RESULTS_DIR, "hyperparam_sensitivity.csv"))
        print(f"  Hyperparam grid complete: {len(hp_rows)} cells")

    print("\nAll benchmarks complete.")


if __name__ == "__main__":
    main()
