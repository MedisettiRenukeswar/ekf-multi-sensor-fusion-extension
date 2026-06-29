"""
Benchmark Runner
================
Runs Monte Carlo experiments across:
  - 3 trajectories: figure8, circle, straight
  - 3 noise regimes: low, medium, high
  - 4 estimators:   EKF, UKF, Adaptive-EKF, Adaptive-UKF

For each condition:
  - N_MC = 30 Monte Carlo runs
  - Computes: ATE, RPE, RMSE-position, RMSE-heading, mean-NIS, mean-NEES, runtime
  - Reports mean ± std and 95% CI for each metric

Outputs:
  benchmark/benchmark_results.csv
  benchmark/summary_table.txt

Author: Medisetti Renukeswar (research extension)
"""

from __future__ import annotations

import csv
import math
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Callable

import numpy as np

from simulation.trajectories import TrajectoryGenerator, TrajectoryType
from simulation.research_sensor_sim import ResearchSensorSimulator, NoiseRegime
from ekf_core.ekf_estimator import EKFEstimator
from ekf_core.ukf_estimator import UKFEstimator
from ekf_core.adaptive_estimator import AdaptiveEKFEstimator, AdaptiveUKFEstimator
from ekf_core.metrics import (
    compute_ate,
    compute_rpe,
    compute_rmse_position,
    compute_rmse_heading,
    compute_nees,
    average_nees_bounds,
    monte_carlo_statistics,
)

# ─────────────────────────── configuration ────────────────────────────────────

N_MC         = 30      # Monte Carlo runs per condition
SIM_DURATION = 40.0    # seconds
DT_IMU       = 0.01    # 100 Hz
DT_CAM       = 1 / 30  # ~30 Hz
TRAJ_SCALE   = 3.0
LOG_EVERY    = 10      # log every N IMU steps

TRAJECTORIES: list[TrajectoryType] = ["figure8", "circle", "straight"]
NOISE_REGIMES: list[NoiseRegime]   = ["low", "medium", "high"]

ESTIMATOR_NAMES = ["EKF", "UKF", "Adaptive-EKF", "Adaptive-UKF"]

RESULTS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────── result dataclass ─────────────────────────────────

@dataclass
class RunResult:
    """Stores all metrics for a single MC run."""
    ate:          float = 0.0
    rpe:          float = 0.0
    rmse_pos:     float = 0.0
    rmse_heading: float = 0.0
    mean_nis:     float = 0.0
    mean_nees:    float = 0.0
    runtime_ms:   float = 0.0


@dataclass
class ConditionResult:
    """Aggregated statistics over N_MC runs for one experimental condition."""
    trajectory:   str = ""
    noise_regime: str = ""
    estimator:    str = ""
    n_runs:       int = 0

    # Per-metric statistics (mean, std, ci_lower, ci_upper)
    ate_mean:          float = 0.0
    ate_std:           float = 0.0
    ate_ci_lower:      float = 0.0
    ate_ci_upper:      float = 0.0
    rpe_mean:          float = 0.0
    rpe_std:           float = 0.0
    rmse_pos_mean:     float = 0.0
    rmse_pos_std:      float = 0.0
    rmse_heading_mean: float = 0.0
    rmse_heading_std:  float = 0.0
    nis_mean:          float = 0.0
    nis_std:           float = 0.0
    nis_lb:            float = 0.0   # 95% chi2 lower bound
    nis_ub:            float = 0.0   # 95% chi2 upper bound
    nees_mean:         float = 0.0
    nees_std:          float = 0.0
    nees_lb:           float = 0.0
    nees_ub:           float = 0.0
    runtime_ms_mean:   float = 0.0
    runtime_ms_std:    float = 0.0
    nis_consistent:    bool  = False
    nees_consistent:   bool  = False


# ─────────────────────────── estimator factory ────────────────────────────────

def make_estimator(name: str) -> EKFEstimator | UKFEstimator:
    """Instantiate a fresh estimator by name."""
    if name == "EKF":
        return EKFEstimator(dt=DT_IMU)
    elif name == "UKF":
        return UKFEstimator(dt=DT_IMU)
    elif name == "Adaptive-EKF":
        return AdaptiveEKFEstimator(dt=DT_IMU, window=20, adapt_R=True, adapt_Q=False,
                                    alpha_smooth=0.1)
    elif name == "Adaptive-UKF":
        return AdaptiveUKFEstimator(dt=DT_IMU, window=20, adapt_R=True, adapt_Q=False,
                                    alpha_smooth=0.1)
    else:
        raise ValueError(f"Unknown estimator: {name}")


# ─────────────────────────── single MC run ────────────────────────────────────

def run_single(
    estimator_name: str,
    traj_type: TrajectoryType,
    noise_regime: NoiseRegime,
    seed: int,
) -> RunResult:
    """
    Execute one full simulation run and return all performance metrics.
    """
    traj = TrajectoryGenerator(
        trajectory_type=traj_type,
        duration=SIM_DURATION,
        scale=TRAJ_SCALE,
    )
    sim = ResearchSensorSimulator(
        trajectory=traj,
        noise_regime=noise_regime,
        dt_imu=DT_IMU,
        dt_cam=DT_CAM,
        seed=seed,
    )
    est = make_estimator(estimator_name)

    # Initialise at ground truth
    px0, py0, th0, vx0, vy0, om0 = traj.get_state(0.0)
    x0 = np.array([px0, py0, th0, vx0, vy0, om0])
    P0 = np.diag([0.5, 0.5, 0.3, 0.5, 0.5, 0.1])
    est.reset(x0, P0)

    # Storage
    gt_x_log: list[float] = []
    gt_y_log: list[float] = []
    gt_th_log: list[float] = []
    est_x_log: list[float] = []
    est_y_log: list[float] = []
    est_th_log: list[float] = []
    nis_log: list[float] = []
    nees_log: list[float] = []

    t = 0.0
    cam_timer = 0.0
    step = 0

    t_start = time.perf_counter()

    while t <= SIM_DURATION:
        px_gt, py_gt, th_gt, vx_gt, vy_gt, om_gt = traj.get_state(t)

        vx_imu, vy_imu, om_imu = sim.get_imu(t)
        est.predict(vx_imu, vy_imu, om_imu)

        cam_timer += DT_IMU
        if cam_timer >= DT_CAM:
            cam_timer = 0.0
            px_c, py_c, th_c = sim.get_camera(t)
            result = est.update_camera(px_c, py_c, th_c)
            nis_log.append(result["nis"])

        if step % LOG_EVERY == 0:
            x_gt = np.array([px_gt, py_gt, th_gt, vx_gt, vy_gt, om_gt])
            x_est, P_est = est.get_state()

            gt_x_log.append(px_gt)
            gt_y_log.append(py_gt)
            gt_th_log.append(th_gt)
            est_x_log.append(x_est[0])
            est_y_log.append(x_est[1])
            est_th_log.append(x_est[2])

            nees = compute_nees(x_gt, x_est, P_est, state_indices=[0, 1, 2])
            if math.isfinite(nees):
                nees_log.append(nees)

        t += DT_IMU
        step += 1

    runtime_ms = (time.perf_counter() - t_start) * 1000.0

    gt_x_arr   = np.array(gt_x_log)
    gt_y_arr   = np.array(gt_y_log)
    gt_th_arr  = np.array(gt_th_log)
    est_x_arr  = np.array(est_x_log)
    est_y_arr  = np.array(est_y_log)
    est_th_arr = np.array(est_th_log)

    ate      = compute_ate(gt_x_arr, gt_y_arr, est_x_arr, est_y_arr)
    rpe      = compute_rpe(gt_x_arr, gt_y_arr, est_x_arr, est_y_arr)
    rmse_pos = compute_rmse_position(gt_x_arr, gt_y_arr, est_x_arr, est_y_arr)
    rmse_hdg = compute_rmse_heading(gt_th_arr, est_th_arr)

    mean_nis  = float(np.mean(nis_log))  if nis_log  else float("nan")
    mean_nees = float(np.mean(nees_log)) if nees_log else float("nan")

    return RunResult(
        ate=ate,
        rpe=rpe,
        rmse_pos=rmse_pos,
        rmse_heading=rmse_hdg,
        mean_nis=mean_nis,
        mean_nees=mean_nees,
        runtime_ms=runtime_ms,
    )


# ─────────────────────────── Monte Carlo loop ─────────────────────────────────

def run_monte_carlo(
    estimator_name: str,
    traj_type: TrajectoryType,
    noise_regime: NoiseRegime,
    n_runs: int = N_MC,
    base_seed: int = 0,
) -> ConditionResult:
    """
    Run N_MC independent simulations and aggregate statistics.
    """
    results: list[RunResult] = []
    for i in range(n_runs):
        seed = base_seed + i * 137    # deterministic, non-overlapping seeds
        r = run_single(estimator_name, traj_type, noise_regime, seed)
        results.append(r)

    ate_samples   = np.array([r.ate          for r in results])
    rpe_samples   = np.array([r.rpe          for r in results])
    rpos_samples  = np.array([r.rmse_pos     for r in results])
    rhdg_samples  = np.array([r.rmse_heading for r in results])
    nis_samples   = np.array([r.mean_nis     for r in results])
    nees_samples  = np.array([r.mean_nees    for r in results])
    rt_samples    = np.array([r.runtime_ms   for r in results])

    ate_stats  = monte_carlo_statistics(ate_samples)
    nis_stats  = monte_carlo_statistics(nis_samples)
    nees_stats = monte_carlo_statistics(nees_samples)

    # Chi-squared consistency bounds (dof=3 for position+heading state)
    nis_lb,  nis_ub  = average_nees_bounds(dof=3, n_runs=n_runs)
    nees_lb, nees_ub = average_nees_bounds(dof=3, n_runs=n_runs)

    # Normalise ANEES / ANIS by dof
    anees = nees_stats["mean"] / 3.0
    anis  = nis_stats["mean"]  / 3.0

    return ConditionResult(
        trajectory=traj_type,
        noise_regime=noise_regime,
        estimator=estimator_name,
        n_runs=n_runs,
        ate_mean=ate_stats["mean"],
        ate_std=ate_stats["std"],
        ate_ci_lower=ate_stats["ci_lower"],
        ate_ci_upper=ate_stats["ci_upper"],
        rpe_mean=float(np.mean(rpe_samples)),
        rpe_std=float(np.std(rpe_samples, ddof=1)),
        rmse_pos_mean=float(np.mean(rpos_samples)),
        rmse_pos_std=float(np.std(rpos_samples, ddof=1)),
        rmse_heading_mean=float(np.mean(rhdg_samples)),
        rmse_heading_std=float(np.std(rhdg_samples, ddof=1)),
        nis_mean=anis,
        nis_std=nis_stats["std"] / 3.0,
        nis_lb=nis_lb,
        nis_ub=nis_ub,
        nees_mean=anees,
        nees_std=nees_stats["std"] / 3.0,
        nees_lb=nees_lb,
        nees_ub=nees_ub,
        runtime_ms_mean=float(np.mean(rt_samples)),
        runtime_ms_std=float(np.std(rt_samples, ddof=1)),
        nis_consistent= nis_lb  <= anis  <= nis_ub,
        nees_consistent=nees_lb <= anees <= nees_ub,
    )


# ─────────────────────────── full benchmark ───────────────────────────────────

def run_full_benchmark(n_runs: int = N_MC) -> list[ConditionResult]:
    """
    Execute the complete benchmark: 3 traj × 3 noise × 4 estimators.
    """
    all_results: list[ConditionResult] = []
    total = len(TRAJECTORIES) * len(NOISE_REGIMES) * len(ESTIMATOR_NAMES)
    done  = 0

    for traj in TRAJECTORIES:
        for noise in NOISE_REGIMES:
            for est_name in ESTIMATOR_NAMES:
                done += 1
                print(f"  [{done:2d}/{total}]  {est_name:14s}  {traj:8s}  {noise:6s}  "
                      f"({n_runs} runs) ...", end=" ", flush=True)
                t0 = time.perf_counter()
                cr = run_monte_carlo(est_name, traj, noise, n_runs=n_runs)
                elapsed = time.perf_counter() - t0
                print(f"ATE={cr.ate_mean:.4f}±{cr.ate_std:.4f}  "
                      f"NIS={'✓' if cr.nis_consistent else '✗'}{cr.nis_mean:.2f}  "
                      f"NEES={'✓' if cr.nees_consistent else '✗'}{cr.nees_mean:.2f}  "
                      f"[{elapsed:.1f}s]")
                all_results.append(cr)

    return all_results


# ─────────────────────────── output ───────────────────────────────────────────

def save_csv(results: list[ConditionResult], path: str) -> None:
    """Save all results to CSV."""
    if not results:
        return
    fieldnames = list(asdict(results[0]).keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    print(f"\nSaved: {path}")


def save_summary_table(results: list[ConditionResult], path: str) -> None:
    """Save human-readable summary table."""
    lines: list[str] = []
    lines.append("=" * 120)
    lines.append("EKF / UKF Benchmark — Multi-Trajectory, Multi-Noise-Regime")
    lines.append("Medisetti Renukeswar  |  June 2026")
    lines.append("=" * 120)
    lines.append(f"{'Trajectory':<10} {'Noise':<8} {'Estimator':<15} "
                 f"{'ATE(m)':<18} {'RPE(m)':<16} {'RMSE-pos':<14} "
                 f"{'RMSE-hdg(rad)':<16} {'ANIS':<14} {'ANEES':<14} "
                 f"{'RT(ms)':<12} {'NIS-ok':<8} {'NEES-ok'}")
    lines.append("-" * 120)

    for r in results:
        ate_str  = f"{r.ate_mean:.4f}±{r.ate_std:.4f}"
        rpe_str  = f"{r.rpe_mean:.4f}±{r.rpe_std:.4f}"
        rpos_str = f"{r.rmse_pos_mean:.4f}±{r.rmse_pos_std:.4f}"
        rhdg_str = f"{r.rmse_heading_mean:.4f}±{r.rmse_heading_std:.4f}"
        nis_str  = f"{r.nis_mean:.3f}±{r.nis_std:.3f}"
        nees_str = f"{r.nees_mean:.3f}±{r.nees_std:.3f}"
        rt_str   = f"{r.runtime_ms_mean:.1f}±{r.runtime_ms_std:.1f}"
        lines.append(
            f"{r.trajectory:<10} {r.noise_regime:<8} {r.estimator:<15} "
            f"{ate_str:<18} {rpe_str:<16} {rpos_str:<14} "
            f"{rhdg_str:<16} {nis_str:<14} {nees_str:<14} "
            f"{rt_str:<12} {'YES' if r.nis_consistent else 'NO':<8} "
            f"{'YES' if r.nees_consistent else 'NO'}"
        )

    lines.append("=" * 120)
    lines.append(f"\nConsistency bounds at 95% CI (chi2, dof=3, N={results[0].n_runs} runs)")
    if results:
        lb, ub = results[0].nis_lb, results[0].nis_ub
        lines.append(f"  ANIS / ANEES consistent if in [{lb:.3f}, {ub:.3f}]")

    text = "\n".join(lines)
    print("\n" + text)
    with open(path, "w") as f:
        f.write(text)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("  EKF/UKF Benchmark — Monte Carlo Evaluation")
    print("  Medisetti Renukeswar")
    print("=" * 60)

    results = run_full_benchmark(n_runs=N_MC)

    csv_path = os.path.join(RESULTS_DIR, "benchmark_results.csv")
    txt_path = os.path.join(RESULTS_DIR, "summary_table.txt")

    save_csv(results, csv_path)
    save_summary_table(results, txt_path)

    print("\nBenchmark complete.")
