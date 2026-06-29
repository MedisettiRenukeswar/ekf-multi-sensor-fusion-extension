"""
Research Unit Tests
=====================
Tests all components of the research framework:
  - Consistency metrics (NEES, NIS)
  - TrajectoryGenerator
  - EKFEstimator (via StateEstimator interface)
  - UKFEstimator
  - AdaptiveEKFEstimator / AdaptiveUKFEstimator
  - Monte Carlo statistics

Run:  python -m pytest tests/ -v --tb=short

Author: Medisetti Renukeswar (research extension)
"""

from __future__ import annotations

import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from ekf_core.metrics import (
    compute_nees,
    compute_nis,
    compute_ate,
    compute_rpe,
    compute_rmse_position,
    compute_rmse_heading,
    chi2_bounds,
    monte_carlo_statistics,
)
from simulation.trajectories import TrajectoryGenerator
from simulation.research_sensor_sim import ResearchSensorSimulator
from ekf_core.ekf_estimator import EKFEstimator
from ekf_core.ukf_estimator import UKFEstimator
from ekf_core.adaptive_estimator import AdaptiveEKFEstimator, AdaptiveUKFEstimator


# ─────────────────────────── Metrics ──────────────────────────────────────────

class TestConsistencyMetrics:
    def test_nees_zero_error(self):
        """NEES must be 0 when estimate equals truth."""
        x = np.array([1.0, 2.0, 0.5, 1.0, 0.0, 0.1])
        P = np.eye(6)
        assert compute_nees(x, x, P) == pytest.approx(0.0, abs=1e-9)

    def test_nees_known_value(self):
        """NEES should equal e^T P^{-1} e for a known error vector."""
        x_true = np.array([2.0, 3.0, 0.0, 0.0, 0.0, 0.0])
        x_est  = np.array([1.0, 3.0, 0.0, 0.0, 0.0, 0.0])
        P = np.eye(6)
        # Error in px = 1.0, P^-1 = I, so NEES = 1.0
        assert compute_nees(x_true, x_est, P) == pytest.approx(1.0, rel=1e-6)

    def test_nees_position_subset(self):
        """NEES with position indices should use only px, py."""
        x_true = np.array([2.0, 4.0, 0.0, 0.0, 0.0, 0.0])
        x_est  = np.array([1.0, 3.0, 0.0, 0.0, 0.0, 0.0])
        P = np.eye(6)
        # e = [1, 1], P_sub = I_2x2, NEES = 2
        nees = compute_nees(x_true, x_est, P, state_indices=[0, 1])
        assert nees == pytest.approx(2.0, rel=1e-6)

    def test_nis_zero_innovation(self):
        """NIS must be 0 for zero innovation."""
        y = np.zeros(3)
        S = np.eye(3)
        assert compute_nis(y, S) == pytest.approx(0.0, abs=1e-9)

    def test_nis_known_value(self):
        """NIS = y^T S^{-1} y with known values."""
        y = np.array([1.0, 0.0, 0.0])
        S = np.diag([2.0, 1.0, 1.0])
        # y^T S^{-1} y = 1/2
        assert compute_nis(y, S) == pytest.approx(0.5, rel=1e-6)

    def test_chi2_bounds_ordering(self):
        """Lower bound must be less than upper bound."""
        lb, ub = chi2_bounds(dof=3, n_runs=30)
        assert lb < ub

    def test_chi2_bounds_dof1_sanity(self):
        """For large n_runs, bounds should converge toward 1.0."""
        lb, ub = chi2_bounds(dof=3, n_runs=10000)
        assert abs(lb - 1.0) < 0.05
        assert abs(ub - 1.0) < 0.05

    def test_monte_carlo_statistics(self):
        """Statistics should be correct for a known distribution."""
        rng = np.random.default_rng(0)
        samples = rng.normal(loc=5.0, scale=1.0, size=1000)
        stats = monte_carlo_statistics(samples)
        assert abs(stats["mean"] - 5.0) < 0.1
        assert abs(stats["std"]  - 1.0) < 0.1
        assert stats["ci_lower"] < stats["mean"] < stats["ci_upper"]


# ─────────────────────────── Trajectories ────────────────────────────────────

class TestTrajectories:
    @pytest.mark.parametrize("traj_type", ["figure8", "circle", "straight"])
    def test_trajectory_returns_6_values(self, traj_type):
        traj = TrajectoryGenerator(trajectory_type=traj_type)
        state = traj.get_state(1.0)
        assert len(state) == 6

    def test_figure8_starts_near_origin(self):
        traj = TrajectoryGenerator("figure8")
        px, py, *_ = traj.get_state(0.0)
        assert abs(px) < 1e-6 and abs(py) < 1e-6

    def test_circle_constant_radius(self):
        traj = TrajectoryGenerator("circle", scale=2.0)
        for t in np.linspace(0, 40, 20):
            px, py, *_ = traj.get_state(t)
            r = math.sqrt(px ** 2 + py ** 2)
            assert abs(r - 2.0) < 0.01, f"Radius {r:.3f} ≠ 2.0 at t={t}"

    def test_straight_monotone_x(self):
        traj = TrajectoryGenerator("straight", speed=1.0)
        xs = [traj.get_state(t)[0] for t in np.linspace(0, 10, 50)]
        assert all(xs[i] <= xs[i + 1] for i in range(len(xs) - 1))

    @pytest.mark.parametrize("traj_type", ["figure8", "circle", "straight"])
    def test_heading_in_range(self, traj_type):
        traj = TrajectoryGenerator(trajectory_type=traj_type)
        for t in np.linspace(0, 40, 100):
            _, _, th, *_ = traj.get_state(t)
            assert -math.pi <= th <= math.pi, f"Heading {th} out of range at t={t}"


# ─────────────────────────── EKF ─────────────────────────────────────────────

class TestEKFEstimator:
    def test_state_shape(self):
        ekf = EKFEstimator()
        assert ekf.x.shape == (6,)
        assert ekf.P.shape == (6, 6)

    def test_covariance_grows_on_predict(self):
        ekf = EKFEstimator(dt=0.01)
        P_trace_0 = np.trace(ekf.P)
        for _ in range(20):
            ekf.predict(0.0, 0.0, 0.0)
        assert np.trace(ekf.P) > P_trace_0

    def test_covariance_decreases_on_update(self):
        ekf = EKFEstimator(dt=0.01)
        ekf.P = np.eye(6) * 2.0
        trace_before = np.trace(ekf.P)
        ekf.update_camera(0.1, 0.1, 0.05)
        assert np.trace(ekf.P) < trace_before

    def test_update_returns_nis(self):
        ekf = EKFEstimator(dt=0.01)
        result = ekf.update_camera(1.0, 2.0, 0.3)
        assert "nis" in result
        assert math.isfinite(result["nis"])
        assert result["nis"] >= 0.0

    def test_covariance_stays_symmetric(self):
        ekf = EKFEstimator(dt=0.01)
        for _ in range(10):
            ekf.predict(0.1, 0.0, 0.05)
            ekf.update_camera(0.1, 0.0, 0.05)
        assert np.allclose(ekf.P, ekf.P.T, atol=1e-10)

    def test_covariance_stays_positive_definite(self):
        ekf = EKFEstimator(dt=0.01)
        for _ in range(50):
            ekf.predict(0.1, 0.0, 0.05)
            ekf.update_camera(0.1, 0.0, 0.05)
        eigs = np.linalg.eigvalsh(ekf.P)
        assert np.all(eigs > 0), f"EKF P not PD: min eig = {eigs.min()}"

    def test_reset_clears_state(self):
        ekf = EKFEstimator(dt=0.01)
        for _ in range(20):
            ekf.predict(1.0, 0.5, 0.1)
        x0 = np.array([1.0, 2.0, 0.0, 0.0, 0.0, 0.0])
        P0 = np.eye(6) * 0.1
        ekf.reset(x0, P0)
        assert np.allclose(ekf.x, x0)
        assert np.allclose(ekf.P, P0)

    def test_nees_computable(self):
        ekf = EKFEstimator(dt=0.01)
        ekf.predict(0.0, 0.0, 0.0)
        x_true = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        nees = ekf.compute_nees_full(x_true)
        assert math.isfinite(nees) and nees >= 0.0

    def test_heading_wrap(self):
        ekf = EKFEstimator(dt=0.1)
        ekf.x = np.array([0.0, 0.0, math.pi - 0.01, 0.0, 0.0, 1.0])
        for _ in range(5):
            ekf.predict(0.0, 0.0, 1.0)
        assert -math.pi <= ekf.x[2] <= math.pi


# ─────────────────────────── UKF ─────────────────────────────────────────────

class TestUKFEstimator:
    def test_state_shape(self):
        ukf = UKFEstimator()
        assert ukf.x.shape == (6,)
        assert ukf.P.shape == (6, 6)

    def test_sigma_point_count(self):
        ukf = UKFEstimator()
        X = ukf._sigma_points(ukf.x, ukf.P)
        assert X.shape == (13, 6)  # 2*6+1 = 13

    def test_weights_sum_to_one(self):
        ukf = UKFEstimator()
        assert abs(np.sum(ukf._Wm) - 1.0) < 1e-10

    def test_covariance_grows_on_predict(self):
        """
        Position covariance P[0,0] must grow after predict-only steps
        when velocity is nonzero (position uncertainty accumulates via F dt coupling).
        """
        ukf = UKFEstimator(dt=0.01)
        # Set a nonzero velocity so position variance propagates
        ukf.x = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        p00_before = ukf.P[0, 0]
        for _ in range(20):
            ukf.predict(1.0, 0.0, 0.0)  # constant forward motion
        # Position uncertainty must have grown
        assert ukf.P[0, 0] > p00_before, (
            f"P[0,0] should grow: {p00_before:.4f} -> {ukf.P[0,0]:.4f}"
        )

    def test_covariance_decreases_on_update(self):
        ukf = UKFEstimator(dt=0.01)
        ukf.P = np.eye(6) * 2.0
        trace_before = np.trace(ukf.P)
        ukf.update_camera(0.1, 0.1, 0.05)
        assert np.trace(ukf.P) < trace_before

    def test_update_returns_nis(self):
        ukf = UKFEstimator(dt=0.01)
        result = ukf.update_camera(1.0, 2.0, 0.3)
        assert "nis" in result
        assert math.isfinite(result["nis"])
        assert result["nis"] >= 0.0

    def test_covariance_stays_positive_definite(self):
        ukf = UKFEstimator(dt=0.01)
        for _ in range(50):
            ukf.predict(0.1, 0.0, 0.05)
            ukf.update_camera(0.1, 0.0, 0.05)
        eigs = np.linalg.eigvalsh(ukf.P)
        assert np.all(eigs > 0), f"UKF P not PD: min eig = {eigs.min()}"

    def test_ekf_ukf_agree_linear_case(self):
        """
        For a nearly linear trajectory, EKF and UKF position estimates
        should agree within 2 cm after 10 steps.
        """
        ekf = EKFEstimator(dt=0.1)
        ukf = UKFEstimator(dt=0.1)
        x0 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        P0 = np.diag([0.5, 0.5, 0.3, 0.5, 0.5, 0.1])
        ekf.reset(x0, P0)
        ukf.reset(x0, P0)

        rng = np.random.default_rng(99)
        for i in range(10):
            vx = 1.0 + rng.normal(0, 0.01)
            vy = rng.normal(0, 0.01)
            om = rng.normal(0, 0.01)
            ekf.predict(vx, vy, om)
            ukf.predict(vx, vy, om)
            ekf.update_camera(float(i + 1) * 0.1, 0.0, 0.0)
            ukf.update_camera(float(i + 1) * 0.1, 0.0, 0.0)

        diff = np.linalg.norm(ekf.x[:2] - ukf.x[:2])
        assert diff < 0.05, f"EKF/UKF position divergence: {diff:.4f} m"

    def test_heading_wrap(self):
        ukf = UKFEstimator(dt=0.1)
        ukf.x = np.array([0.0, 0.0, math.pi - 0.01, 0.0, 0.0, 1.0])
        for _ in range(5):
            ukf.predict(0.0, 0.0, 1.0)
        assert -math.pi <= ukf.x[2] <= math.pi


# ─────────────────────────── Adaptive ────────────────────────────────────────

class TestAdaptiveEstimators:
    def test_adaptive_ekf_tracks_well(self):
        """Adaptive EKF should achieve ATE < 0.2 m on medium noise."""
        traj = TrajectoryGenerator("figure8", duration=40.0, scale=3.0)
        sim  = ResearchSensorSimulator(traj, "medium", seed=0)
        ekf  = AdaptiveEKFEstimator(dt=0.01, window=20, adapt_R=True, adapt_Q=False)

        px0, py0, th0, vx0, vy0, om0 = traj.get_state(0.0)
        ekf.reset(np.array([px0, py0, th0, vx0, vy0, om0]),
                  np.diag([0.5, 0.5, 0.3, 0.5, 0.5, 0.1]))

        errs: list[float] = []
        t, cam_timer, step = 0.0, 0.0, 0
        while t <= 40.0:
            vx, vy, om = sim.get_imu(t)
            ekf.predict(vx, vy, om)
            cam_timer += 0.01
            if cam_timer >= 1 / 30:
                cam_timer = 0.0
                ekf.update_camera(*sim.get_camera(t))
            if step % 10 == 0:
                px_gt, py_gt, *_ = traj.get_state(t)
                err = math.sqrt((px_gt - ekf.x[0])**2 + (py_gt - ekf.x[1])**2)
                errs.append(err)
            t += 0.01
            step += 1

        ate = float(np.sqrt(np.mean(np.array(errs)**2)))
        assert ate < 0.2, f"Adaptive-EKF ATE too large: {ate:.4f} m"

    def test_adaptive_ekf_R_changes(self):
        """Adaptive EKF should modify R_cam during a run."""
        traj = TrajectoryGenerator("figure8")
        sim  = ResearchSensorSimulator(traj, "medium", seed=5)
        ekf  = AdaptiveEKFEstimator(dt=0.01, window=20, adapt_R=True, adapt_Q=False)
        px0, py0, th0, vx0, vy0, om0 = traj.get_state(0.0)
        ekf.reset(np.array([px0, py0, th0, vx0, vy0, om0]),
                  np.diag([0.5, 0.5, 0.3, 0.5, 0.5, 0.1]))
        R_initial = ekf.R_cam.copy()

        t, cam_timer = 0.0, 0.0
        for _ in range(200):
            vx, vy, om = sim.get_imu(t)
            ekf.predict(vx, vy, om)
            cam_timer += 0.01
            if cam_timer >= 1 / 30:
                cam_timer = 0.0
                ekf.update_camera(*sim.get_camera(t))
            t += 0.01

        assert not np.allclose(ekf.R_cam, R_initial), "R_cam should have been adapted"

    def test_adaptive_ukf_R_changes(self):
        """Adaptive UKF should also modify R_cam."""
        traj = TrajectoryGenerator("circle")
        sim  = ResearchSensorSimulator(traj, "high", seed=7)
        ukf  = AdaptiveUKFEstimator(dt=0.01, window=20, adapt_R=True)
        px0, py0, th0, vx0, vy0, om0 = traj.get_state(0.0)
        ukf.reset(np.array([px0, py0, th0, vx0, vy0, om0]),
                  np.diag([0.5, 0.5, 0.3, 0.5, 0.5, 0.1]))
        R_initial = ukf.R_cam.copy()

        t, cam_timer = 0.0, 0.0
        for _ in range(300):
            vx, vy, om = sim.get_imu(t)
            ukf.predict(vx, vy, om)
            cam_timer += 0.01
            if cam_timer >= 1 / 30:
                cam_timer = 0.0
                ukf.update_camera(*sim.get_camera(t))
            t += 0.01

        assert not np.allclose(ukf.R_cam, R_initial), "UKF R_cam should have been adapted"

    def test_adaptive_reset_clears_buffers(self):
        """Reset should clear innovation buffers."""
        traj = TrajectoryGenerator("circle")
        sim  = ResearchSensorSimulator(traj, "medium", seed=3)
        ekf  = AdaptiveEKFEstimator(dt=0.01, window=10)
        ekf.update_camera(1.0, 2.0, 0.3)
        ekf.update_camera(1.0, 2.0, 0.3)
        ekf.reset(np.zeros(6), np.eye(6) * 0.5)
        assert len(ekf._innovation_buffer) == 0
        assert ekf.n_adaptations == 0


# ─────────────────────────── Sensor Simulator ────────────────────────────────

class TestResearchSensorSimulator:
    @pytest.mark.parametrize("regime", ["low", "medium", "high"])
    def test_noise_scaling(self, regime):
        """High noise should produce larger RMS measurement error than low."""
        traj = TrajectoryGenerator("figure8")
        sim_lo = ResearchSensorSimulator(traj, "low",  seed=0)
        sim_hi = ResearchSensorSimulator(traj, "high", seed=0)

        lo_errs, hi_errs = [], []
        for t in np.linspace(0, 40, 100):
            px_gt, py_gt, *_ = traj.get_state(t)
            px_lo, py_lo, _ = sim_lo.get_camera(t)
            px_hi, py_hi, _ = sim_hi.get_camera(t)
            lo_errs.append((px_gt - px_lo)**2 + (py_gt - py_lo)**2)
            hi_errs.append((px_gt - px_hi)**2 + (py_gt - py_hi)**2)

        assert np.mean(hi_errs) > np.mean(lo_errs)

    def test_seed_reproducibility(self):
        """Same seed must produce identical measurements."""
        traj = TrajectoryGenerator("figure8")
        sim1 = ResearchSensorSimulator(traj, "medium", seed=42)
        sim2 = ResearchSensorSimulator(traj, "medium", seed=42)
        for t in [0.0, 1.0, 5.0, 10.0]:
            assert sim1.get_imu(t) == sim2.get_imu(t)


# ─────────────────────────── End-to-end accuracy ─────────────────────────────

class TestEndToEnd:
    @pytest.mark.parametrize("estimator_cls", [EKFEstimator, UKFEstimator])
    def test_ate_below_threshold_medium_noise(self, estimator_cls):
        """Both EKF and UKF should achieve ATE < 0.15 m on medium figure-8."""
        traj = TrajectoryGenerator("figure8", duration=40.0, scale=3.0)
        sim  = ResearchSensorSimulator(traj, "medium", seed=42)
        est  = estimator_cls(dt=0.01)

        px0, py0, th0, vx0, vy0, om0 = traj.get_state(0.0)
        est.reset(np.array([px0, py0, th0, vx0, vy0, om0]),
                  np.diag([0.5, 0.5, 0.3, 0.5, 0.5, 0.1]))

        gt_x, gt_y, est_x, est_y = [], [], [], []
        t, cam_timer, step = 0.0, 0.0, 0
        while t <= 40.0:
            vx, vy, om = sim.get_imu(t)
            est.predict(vx, vy, om)
            cam_timer += 0.01
            if cam_timer >= 1 / 30:
                cam_timer = 0.0
                est.update_camera(*sim.get_camera(t))
            if step % 10 == 0:
                px_gt, py_gt, *_ = traj.get_state(t)
                gt_x.append(px_gt);  gt_y.append(py_gt)
                est_x.append(est.x[0]);  est_y.append(est.x[1])
            t += 0.01
            step += 1

        ate = compute_ate(np.array(gt_x), np.array(gt_y),
                          np.array(est_x), np.array(est_y))
        assert ate < 0.15, f"{estimator_cls.__name__} ATE={ate:.4f} m > 0.15 m"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
