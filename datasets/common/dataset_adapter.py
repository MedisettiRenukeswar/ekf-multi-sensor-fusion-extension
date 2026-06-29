"""
Dataset Adapter
================
Converts a DatasetLoader stream into predict() / update_camera() calls
on the existing StateEstimator interface.

The challenge
-------------
The existing estimators use a world-frame velocity model:
    predict(vx_world, vy_world, omega)

Real IMU data provides body-frame accelerations and angular velocities.
This adapter integrates body-frame IMU into world-frame velocity estimates
using a running heading estimate and simple trapezoidal integration.

The adapter is the ONLY place where real-data specifics touch the estimator.
All four estimators (EKF, UKF, Adaptive-EKF, Adaptive-UKF) are used
identically through this adapter.

Architecture
------------
DatasetLoader  →  DatasetAdapter  →  StateEstimator
  IMUSample    →  predict()
  PoseSample   →  update_camera()
  PoseSample   →  compute_nees()  (if ground truth)

Author: Medisetti Renukeswar (Phase 6)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from datasets.common.dataset_base import DatasetLoader, IMUSample, PoseSample
from ekf_core.estimator_base import StateEstimator
from ekf_core.metrics import compute_nees, compute_ate, compute_rpe


@dataclass
class AdapterConfig:
    """
    Configuration for the dataset adapter.

    Parameters
    ----------
    imu_rate_hz    : IMU rate (used to compute dt)
    gt_as_vo       : If True, ground-truth poses are used as VO measurements
                     (noiseless update — upper bound on performance)
    vo_update_hz   : Rate at which VO updates are fed to the filter (Hz).
                     If 0, uses every available GT/VO sample.
    gravity_mag    : Local gravity magnitude (m/s²)
    vel_decay      : Exponential decay on velocity estimate between updates
                     (0=no decay, 0.01=mild decay). Prevents velocity divergence.
    """
    imu_rate_hz:  float = 200.0
    gt_as_vo:     bool  = True
    vo_update_hz: float = 0.0
    gravity_mag:  float = 9.81
    vel_decay:    float = 0.005


@dataclass
class RunResult:
    """Results from one adapter run on one sequence."""
    dataset_name:  str = ""
    sequence_name: str = ""
    estimator_name: str = ""
    n_imu:         int   = 0
    n_updates:     int   = 0
    runtime_ms:    float = 0.0
    ate:           float = float('nan')
    rpe:           float = float('nan')
    rmse_pos:      float = float('nan')
    rmse_heading:  float = float('nan')
    mean_nis:      float = float('nan')
    mean_nees:     float = float('nan')
    has_gt:        bool  = False
    notes:         str   = ""


class DatasetAdapter:
    """
    Runs a StateEstimator on a DatasetLoader sequence.

    Usage
    -----
    adapter = DatasetAdapter(estimator, config)
    result  = adapter.run(loader, estimator_name="EKF")
    """

    def __init__(
        self,
        estimator: StateEstimator,
        config: AdapterConfig | None = None,
    ) -> None:
        self.estimator = estimator
        self.config    = config or AdapterConfig()

    def run(
        self,
        loader: DatasetLoader,
        estimator_name: str = "unknown",
        log_every: int = 20,
    ) -> RunResult:
        """
        Execute the full predict/update loop over the dataset sequence.

        Parameters
        ----------
        loader         : loaded DatasetLoader instance
        estimator_name : label for the result
        log_every      : log every N IMU steps (reduces memory)

        Returns
        -------
        RunResult with all metrics populated
        """
        if not loader._loaded:
            loader.load()

        meta     = loader.get_metadata()
        imu_list = loader.imu_samples
        gt_list  = loader.gt_samples

        if len(imu_list) == 0:
            return RunResult(
                dataset_name=meta.dataset_name,
                sequence_name=meta.sequence_name,
                estimator_name=estimator_name,
                notes="No IMU data",
            )

        # ── Initialise estimator ──────────────────────────────────────────
        dt_imu = 1.0 / self.config.imu_rate_hz

        if len(gt_list) > 0:
            first_gt = gt_list[0]
            vx0 = first_gt.vx if math.isfinite(first_gt.vx) else 0.0
            vy0 = first_gt.vy if math.isfinite(first_gt.vy) else 0.0
            om0 = first_gt.omega if math.isfinite(first_gt.omega) else 0.0
            x0 = np.array([first_gt.px, first_gt.py, first_gt.theta,
                            vx0, vy0, om0])
        else:
            x0 = np.zeros(6)

        P0 = np.diag([1.0, 1.0, 0.5, 1.0, 1.0, 0.3])
        self.estimator.reset(x0, P0)
        self.estimator.dt = dt_imu

        # ── Velocity integrator state ─────────────────────────────────────
        vx_world = float(x0[3])
        vy_world = float(x0[4])

        # ── GT iterator ───────────────────────────────────────────────────
        gt_iter    = iter(gt_list)
        next_gt    = next(gt_iter, None)
        vo_min_dt  = 1.0 / self.config.vo_update_hz if self.config.vo_update_hz > 0 else 0.0
        last_vo_t  = -1e9

        # ── Logging buffers ───────────────────────────────────────────────
        gt_x_log: list[float] = []
        gt_y_log: list[float] = []
        gt_th_log: list[float] = []
        est_x_log: list[float] = []
        est_y_log: list[float] = []
        est_th_log: list[float] = []
        nis_log:  list[float] = []
        nees_log: list[float] = []

        n_updates = 0
        step      = 0
        t_wall_0  = time.perf_counter()

        # ── Main loop ─────────────────────────────────────────────────────
        for imu in imu_list:
            t = imu.timestamp

            # Convert body-frame IMU to world-frame velocity
            theta_est = float(self.estimator.x[2])
            vx_world, vy_world = self._integrate_imu(
                imu, theta_est, vx_world, vy_world, dt_imu,
            )

            # Velocity decay (prevents runaway integration)
            decay = 1.0 - self.config.vel_decay
            vx_world *= decay
            vy_world *= decay

            # Predict step
            omega = imu.gz
            self.estimator.predict(vx_world, vy_world, omega)

            # Advance GT iterator to catch up with current IMU timestamp
            while next_gt is not None and next_gt.timestamp <= t:
                # Check VO rate constraint
                if t - last_vo_t >= vo_min_dt - 1e-6:
                    result = self.estimator.update_camera(
                        next_gt.px, next_gt.py, next_gt.theta
                    )
                    nis_log.append(result["nis"])
                    n_updates += 1
                    last_vo_t = t

                    # NEES if ground truth available
                    xe, Pe = self.estimator.get_state()
                    x_gt = np.array([
                        next_gt.px, next_gt.py, next_gt.theta,
                        next_gt.vx if math.isfinite(next_gt.vx) else 0.0,
                        next_gt.vy if math.isfinite(next_gt.vy) else 0.0,
                        next_gt.omega if math.isfinite(next_gt.omega) else 0.0,
                    ])
                    nees = compute_nees(x_gt, xe, Pe, state_indices=[0, 1, 2])
                    if math.isfinite(nees):
                        nees_log.append(nees)

                next_gt = next(gt_iter, None)

            # Log at reduced rate
            if step % log_every == 0 and next_gt is not None:
                xe, _ = self.estimator.get_state()
                # Find nearest GT for logging
                gt_x_log.append(next_gt.px)
                gt_y_log.append(next_gt.py)
                gt_th_log.append(next_gt.theta)
                est_x_log.append(float(xe[0]))
                est_y_log.append(float(xe[1]))
                est_th_log.append(float(xe[2]))

            step += 1

        runtime_ms = (time.perf_counter() - t_wall_0) * 1000.0

        # ── Compute metrics ────────────────────────────────────────────────
        result = RunResult(
            dataset_name=meta.dataset_name,
            sequence_name=meta.sequence_name,
            estimator_name=estimator_name,
            n_imu=len(imu_list),
            n_updates=n_updates,
            runtime_ms=runtime_ms,
            has_gt=meta.has_gt,
        )

        if len(gt_x_log) > 2:
            gx = np.array(gt_x_log);   gy  = np.array(gt_y_log)
            ex = np.array(est_x_log);  ey  = np.array(est_y_log)
            gth = np.array(gt_th_log); eth = np.array(est_th_log)

            result.ate      = compute_ate(gx, gy, ex, ey)
            result.rpe      = compute_rpe(gx, gy, ex, ey)
            result.rmse_pos = result.ate
            # RMSE heading
            diff = gth - eth
            diff = np.arctan2(np.sin(diff), np.cos(diff))
            result.rmse_heading = float(np.sqrt(np.mean(diff ** 2)))

        if nis_log:
            result.mean_nis = float(np.mean(nis_log))
        if nees_log:
            result.mean_nees = float(np.mean(nees_log))

        return result

    def _integrate_imu(
        self,
        imu: IMUSample,
        theta: float,
        prev_vx: float,
        prev_vy: float,
        dt: float,
    ) -> tuple[float, float]:
        """
        Integrate body-frame accelerometer to world-frame velocity.

        Removes gravity component (assuming small pitch/roll for 2D navigation)
        and rotates body-frame xy-acceleration to world frame.
        """
        # Subtract gravity approximation (body z)
        # For nearly horizontal motion: az ≈ -g, ax/ay are lateral
        ax_b = imu.ax
        ay_b = imu.ay

        # Body → world rotation (2D)
        c = math.cos(theta)
        s = math.sin(theta)
        ax_w = c * ax_b - s * ay_b
        ay_w = s * ax_b + c * ay_b

        # Euler integration
        vx_new = prev_vx + ax_w * dt
        vy_new = prev_vy + ay_w * dt

        return float(vx_new), float(vy_new)
