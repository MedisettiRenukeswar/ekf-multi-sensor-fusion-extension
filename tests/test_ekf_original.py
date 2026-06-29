"""
Unit Tests — EKF Core
Tests fundamental filter properties.

Run: python -m pytest tests/ -v

Author: Medisetti Renukeswar
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import math
import pytest
from ekf_core.ekf import EKF2DRobot


class TestEKFInitialisation:
    def test_state_shape(self):
        ekf = EKF2DRobot()
        assert ekf.x.shape == (6,), "State must be 6-dimensional"

    def test_covariance_shape(self):
        ekf = EKF2DRobot()
        assert ekf.P.shape == (6, 6), "Covariance must be 6x6"

    def test_covariance_symmetric(self):
        ekf = EKF2DRobot()
        assert np.allclose(ekf.P, ekf.P.T), "Covariance must be symmetric"

    def test_covariance_positive_definite(self):
        ekf = EKF2DRobot()
        eigenvalues = np.linalg.eigvalsh(ekf.P)
        assert np.all(eigenvalues > 0), "Covariance must be positive definite"


class TestEKFPrediction:
    def test_identity_motion_zero_innovation(self):
        """Zero velocity prediction should not move the state."""
        ekf = EKF2DRobot(dt=0.01)
        ekf.x = np.array([1.0, 2.0, 0.5, 0.0, 0.0, 0.0])
        ekf.predict(0.0, 0.0, 0.0)
        assert abs(ekf.x[0] - 1.0) < 1e-9, "px should not change with zero velocity"
        assert abs(ekf.x[1] - 2.0) < 1e-9, "py should not change with zero velocity"

    def test_forward_motion(self):
        """Constant velocity should update position correctly."""
        ekf = EKF2DRobot(dt=0.1)
        ekf.x = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        ekf.predict(1.0, 0.0, 0.0)
        assert abs(ekf.x[0] - 0.1) < 1e-6, f"px should be 0.1, got {ekf.x[0]}"

    def test_covariance_grows_on_predict(self):
        """Covariance must grow (or stay same) after prediction (no measurement)."""
        ekf = EKF2DRobot(dt=0.01)
        P_before = ekf.P.copy()
        for _ in range(10):
            ekf.predict(0.0, 0.0, 0.0)
        # Trace of P should increase
        assert np.trace(ekf.P) >= np.trace(P_before), \
            "Covariance trace should not decrease without measurements"

    def test_heading_normalised(self):
        """Heading should stay in (-pi, pi) after large rotation."""
        ekf = EKF2DRobot(dt=0.1)
        ekf.x = np.array([0.0, 0.0, math.pi - 0.01, 0.0, 0.0, 1.0])
        for _ in range(10):
            ekf.predict(0.0, 0.0, 1.0)
        assert -math.pi <= ekf.x[2] <= math.pi, \
            f"Heading {ekf.x[2]} out of range"


class TestEKFUpdate:
    def test_high_noise_small_correction(self):
        """High measurement noise R → small Kalman gain → small state change."""
        ekf = EKF2DRobot(dt=0.01)
        ekf.x = np.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.1])
        ekf.R_cam = np.diag([100.0, 100.0, 100.0])  # very noisy camera
        x_before = ekf.x.copy()
        ekf.update_camera(5.0, 5.0, 1.0)  # large measurement
        delta = np.linalg.norm(ekf.x[:3] - x_before[:3])
        assert delta < 0.5, f"High R should cause small update, got delta={delta:.4f}"

    def test_low_noise_large_correction(self):
        """Low measurement noise R → large Kalman gain → state moves toward measurement."""
        ekf = EKF2DRobot(dt=0.01)
        ekf.x = np.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.1])
        ekf.P = np.eye(6) * 10.0  # very uncertain prior
        ekf.R_cam = np.diag([0.001, 0.001, 0.001])  # very accurate camera
        target = np.array([3.0, 4.0, 0.5])
        ekf.update_camera(*target)
        # State should move close to measurement
        assert abs(ekf.x[0] - 3.0) < 0.1, f"px={ekf.x[0]:.3f} should be near 3.0"
        assert abs(ekf.x[1] - 4.0) < 0.1, f"py={ekf.x[1]:.3f} should be near 4.0"

    def test_covariance_decreases_after_update(self):
        """Covariance should decrease after a measurement update."""
        ekf = EKF2DRobot(dt=0.01)
        ekf.P = np.eye(6) * 2.0
        P_trace_before = np.trace(ekf.P)
        ekf.update_camera(0.1, 0.1, 0.05)
        assert np.trace(ekf.P) < P_trace_before, \
            "Covariance trace should decrease after measurement"

    def test_covariance_remains_symmetric(self):
        """Covariance must remain symmetric after update."""
        ekf = EKF2DRobot(dt=0.01)
        ekf.update_camera(1.0, 2.0, 0.3)
        assert np.allclose(ekf.P, ekf.P.T, atol=1e-10), \
            "Covariance must stay symmetric after update"


class TestEKFConvergence:
    def test_filter_converges_to_true_position(self):
        """
        Filter should converge to ground truth when given accurate measurements.
        ATE after 100 updates with accurate camera should be < 0.1 m.
        """
        ekf = EKF2DRobot(dt=0.01)
        ekf.x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        ekf.R_cam = np.diag([0.01, 0.01, 0.01])

        true_px, true_py, true_th = 2.0, 3.0, 0.7

        for _ in range(100):
            ekf.predict(0.0, 0.0, 0.0)
            # Small noise measurement near true position
            ekf.update_camera(
                true_px + np.random.normal(0, 0.01),
                true_py + np.random.normal(0, 0.01),
                true_th + np.random.normal(0, 0.01)
            )

        pos_err = math.sqrt((ekf.x[0] - true_px)**2 + (ekf.x[1] - true_py)**2)
        assert pos_err < 0.1, f"Filter should converge, ATE={pos_err:.4f} m"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
