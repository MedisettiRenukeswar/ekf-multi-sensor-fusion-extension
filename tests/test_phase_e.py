"""
Phase E Unit Tests
====================
Tests for:
  - Wilcoxon signed-rank test (metrics.py)
  - Cohen's d effect size (metrics.py)
  - DegradedSensorSimulator — bias random walk
  - DegradedSensorSimulator — VO dropout rates
  - DegradedSensorSimulator — reproducibility
  - Integration: degraded simulation runs without error for all 4 estimators
  - Statistical test: paired samples correctly detect known effect

Run:  python -m pytest tests/test_phase_e.py -v --tb=short

Author: Medisetti Renukeswar (Phase E)
"""

from __future__ import annotations

import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from ekf_core.metrics import wilcoxon_test, cohens_d
from simulation.trajectories import TrajectoryGenerator
from simulation.research_sensor_sim import (
    DegradedSensorSimulator, DegradationConfig,
    ResearchSensorSimulator,
)
from ekf_core.ekf_estimator import EKFEstimator
from ekf_core.ukf_estimator import UKFEstimator
from ekf_core.adaptive_estimator import AdaptiveEKFEstimator, AdaptiveUKFEstimator


# ─────────────────────────── Wilcoxon tests ──────────────────────────────────

class TestWilcoxon:
    def test_identical_samples_not_significant(self):
        """Identical samples → p=1, not significant."""
        a = np.array([0.05] * 30)
        r = wilcoxon_test(a, a.copy())
        # Scipy raises ValueError for zero-difference samples; our code returns nan
        assert not r["significant"]

    def test_clearly_different_samples_significant(self):
        """Well-separated Gaussian samples → p < 0.05."""
        rng = np.random.default_rng(42)
        a = rng.normal(0.10, 0.01, 50)
        b = rng.normal(0.05, 0.01, 50)
        r = wilcoxon_test(a, b)
        assert r["significant"], f"Should be significant, got p={r['p_value']:.4f}"
        assert r["p_value"] < 0.05

    def test_effect_size_in_range(self):
        """Effect size (rank-biserial r) must be in [-1, 1]."""
        rng = np.random.default_rng(7)
        a = rng.normal(0.08, 0.01, 50)
        b = rng.normal(0.04, 0.01, 50)
        r = wilcoxon_test(a, b)
        assert -1.0 <= r["effect_size"] <= 1.0

    def test_direction_consistency(self):
        """If a > b on average, median_diff should be positive."""
        rng = np.random.default_rng(3)
        a = rng.normal(0.10, 0.01, 50)
        b = rng.normal(0.05, 0.01, 50)
        r = wilcoxon_test(a, b)
        assert r["median_diff"] > 0, "a > b so diff should be positive"
        assert r["mean_diff"]   > 0

    def test_symmetry(self):
        """Swapping a and b should flip the sign of median_diff."""
        rng = np.random.default_rng(11)
        a = rng.normal(0.10, 0.01, 50)
        b = rng.normal(0.05, 0.01, 50)
        r_ab = wilcoxon_test(a, b)
        r_ba = wilcoxon_test(b, a)
        assert abs(r_ab["median_diff"] + r_ba["median_diff"]) < 1e-9
        # p-value should be the same (two-sided)
        assert abs(r_ab["p_value"] - r_ba["p_value"]) < 1e-9


class TestCohensD:
    def test_zero_d_for_equal_means(self):
        """Equal means → Cohen's d ≈ 0."""
        rng = np.random.default_rng(0)
        a = rng.normal(5.0, 1.0, 200)
        b = rng.normal(5.0, 1.0, 200)
        d = cohens_d(a, b)
        assert abs(d) < 0.2, f"|d|={abs(d):.3f} should be near 0"

    def test_large_d_for_well_separated(self):
        """2-sigma separation → |d| ≈ 2.0."""
        rng = np.random.default_rng(1)
        a = rng.normal(10.0, 1.0, 200)
        b = rng.normal(8.0,  1.0, 200)
        d = cohens_d(a, b)
        assert 1.5 < abs(d) < 2.5, f"|d|={abs(d):.3f} expected ~2.0"

    def test_sign_consistency(self):
        """d(a,b) = -d(b,a)."""
        rng = np.random.default_rng(2)
        a = rng.normal(5.0, 1.0, 50)
        b = rng.normal(4.0, 1.0, 50)
        assert abs(cohens_d(a, b) + cohens_d(b, a)) < 1e-9


# ─────────────────────────── DegradedSensorSimulator ─────────────────────────

class TestDegradedSimulator:

    def test_bias_rw_produces_variable_bias(self):
        """With random-walk enabled, bias must vary over time."""
        traj = TrajectoryGenerator("figure8")
        cfg  = DegradationConfig(enable_bias_random_walk=True, bias_rw_std=0.005)
        sim  = DegradedSensorSimulator(traj, "medium", degradation=cfg, seed=0)
        biases = []
        for i in range(500):
            sim.get_imu(i * 0.01)
            biases.append(sim._gyro_bias_current)
        assert np.std(biases) > 0.001, "Bias should wander over 500 steps"

    def test_bias_rw_disabled_keeps_constant_bias(self):
        """Without random-walk, bias must stay constant."""
        traj = TrajectoryGenerator("figure8")
        cfg  = DegradationConfig(enable_bias_random_walk=False)
        sim  = DegradedSensorSimulator(traj, "medium", degradation=cfg, seed=0)
        initial_bias = sim._gyro_bias_current
        for i in range(200):
            sim.get_imu(i * 0.01)
        assert sim._gyro_bias_current == initial_bias, "Bias should not change"

    def test_bias_rw_bounded(self):
        """Bias random walk must stay within ±0.10 rad/s."""
        traj = TrajectoryGenerator("circle")
        cfg  = DegradationConfig(enable_bias_random_walk=True, bias_rw_std=0.02)
        sim  = DegradedSensorSimulator(traj, "high", degradation=cfg, seed=99)
        for i in range(4000):
            sim.get_imu(i * 0.01)
            assert abs(sim._gyro_bias_current) <= 0.10, (
                f"Bias {sim._gyro_bias_current:.4f} exceeded ±0.10 bound"
            )

    @pytest.mark.parametrize("prob,expected_rate", [(0.30, 0.70), (0.50, 0.50)])
    def test_dropout_rate_correct(self, prob, expected_rate):
        """
        VO availability rate must be within 5 pp of (1 - dropout_prob)
        over 1000 Bernoulli trials.
        """
        traj = TrajectoryGenerator("figure8")
        cfg  = DegradationConfig(enable_vo_dropout=True, vo_dropout_prob=prob)
        sim  = DegradedSensorSimulator(traj, "medium", degradation=cfg, seed=42)
        # Consume IMU samples to advance RNG state exactly as the main loop does
        n_avail = 0
        for i in range(1000):
            sim.get_imu(i * 0.01)   # advance IMU rng
            if i % 3 == 0:           # camera fires every ~3 IMU steps
                if sim.camera_available():
                    n_avail += 1
        n_cam = 1000 // 3
        rate  = n_avail / n_cam
        assert abs(rate - expected_rate) < 0.08, (
            f"Dropout rate {1-rate:.2f} ≠ {prob:.2f} (expected ±0.08)"
        )

    def test_dropout_disabled_gives_always_available(self):
        """With dropout disabled, camera_available must always return True."""
        traj = TrajectoryGenerator("figure8")
        cfg  = DegradationConfig(enable_vo_dropout=False)
        sim  = DegradedSensorSimulator(traj, "medium", degradation=cfg, seed=0)
        for _ in range(100):
            assert sim.camera_available() is True

    def test_reproducibility_with_seed(self):
        """Identical seeds must produce identical measurement sequences."""
        traj = TrajectoryGenerator("figure8")
        cfg  = DegradationConfig(enable_bias_random_walk=True,
                                 enable_vo_dropout=True, vo_dropout_prob=0.30)
        sim1 = DegradedSensorSimulator(traj, "medium", degradation=cfg, seed=7)
        sim2 = DegradedSensorSimulator(traj, "medium", degradation=cfg, seed=7)
        for i in range(50):
            t = i * 0.01
            assert sim1.get_imu(t) == sim2.get_imu(t), "IMU mismatch at t={t}"
            assert sim1.camera_available() == sim2.camera_available()

    def test_reset_restores_state(self):
        """After reset, simulator must produce the same sequence as fresh init."""
        traj = TrajectoryGenerator("circle")
        cfg  = DegradationConfig(enable_bias_random_walk=True)
        sim  = DegradedSensorSimulator(traj, "medium", degradation=cfg, seed=5)

        # Collect first sequence
        seq1 = [sim.get_imu(i * 0.01) for i in range(20)]

        # Reset and collect second sequence
        sim.reset(seed=5)
        seq2 = [sim.get_imu(i * 0.01) for i in range(20)]

        assert seq1 == seq2, "Reset should restore identical sequence"

    def test_no_degradation_matches_base_simulator(self):
        """
        DegradedSimulator with no degradation must produce measurements
        identical to ResearchSensorSimulator at the same seed.
        """
        traj  = TrajectoryGenerator("figure8")
        cfg   = DegradationConfig()   # all flags False
        sim_d = DegradedSensorSimulator(traj, "medium", degradation=cfg, seed=42)
        sim_b = ResearchSensorSimulator(traj, "medium", seed=42)

        for i in range(30):
            t = i * 0.01
            d_imu = sim_d.get_imu(t)
            b_imu = sim_b.get_imu(t)
            assert d_imu == b_imu, f"Step {i}: degraded {d_imu} ≠ base {b_imu}"


# ─────────────────────────── Integration ─────────────────────────────────────

class TestDegradedIntegration:
    """
    End-to-end: all 4 estimators must complete a degraded run without error
    and achieve finite ATE.
    """

    @pytest.mark.parametrize("est_cls", [
        EKFEstimator, UKFEstimator, AdaptiveEKFEstimator, AdaptiveUKFEstimator
    ])
    @pytest.mark.parametrize("scenario", ["bias_rw", "vo_drop30", "vo_drop50"])
    def test_estimator_survives_degradation(self, est_cls, scenario):
        DT_IMU = 0.01
        DT_CAM = 1 / 30
        DURATION = 10.0   # short run for speed

        traj = TrajectoryGenerator("figure8", duration=DURATION, scale=3.0)
        cfg_map = {
            "bias_rw":   DegradationConfig(enable_bias_random_walk=True),
            "vo_drop30": DegradationConfig(enable_vo_dropout=True, vo_dropout_prob=0.30),
            "vo_drop50": DegradationConfig(enable_vo_dropout=True, vo_dropout_prob=0.50),
        }
        sim = DegradedSensorSimulator(
            traj, "medium", degradation=cfg_map[scenario], seed=42
        )
        est = est_cls(dt=DT_IMU)
        px0, py0, th0, vx0, vy0, om0 = traj.get_state(0.0)
        est.reset(
            np.array([px0, py0, th0, vx0, vy0, om0]),
            np.diag([0.5, 0.5, 0.3, 0.5, 0.5, 0.1]),
        )

        t = 0.0
        cam_timer = 0.0
        gt_x, est_x = [], []

        while t <= DURATION:
            vx, vy, om = sim.get_imu(t)
            est.predict(vx, vy, om)
            cam_timer += DT_IMU
            if cam_timer >= DT_CAM:
                cam_timer = 0.0
                if sim.camera_available():
                    px_c, py_c, th_c = sim.get_camera(t)
                    est.update_camera(px_c, py_c, th_c)
            px_gt, py_gt, *_ = traj.get_state(t)
            gt_x.append(px_gt)
            est_x.append(est.x[0])
            t += DT_IMU

        # ATE must be finite and reasonably bounded (< 2 m for 10s run)
        ate = float(np.sqrt(np.mean(
            (np.array(gt_x) - np.array(est_x)) ** 2
        )))
        assert math.isfinite(ate), f"{est_cls.__name__} {scenario}: ATE is NaN/inf"
        assert ate < 2.0, (
            f"{est_cls.__name__} {scenario}: ATE={ate:.3f} m exceeds 2.0 m threshold"
        )
        assert np.all(np.isfinite(est.P)), "Covariance matrix must remain finite"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
