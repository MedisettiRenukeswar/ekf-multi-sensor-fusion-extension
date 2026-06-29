"""
EKF State Estimator
====================
Extended Kalman Filter implementation conforming to StateEstimator interface.

This is a direct refactor of the original ekf.py onto the abstract base class.
All mathematical content (motion model, Jacobian, Joseph-form update) is
preserved unchanged.  Additional methods expose innovation statistics (NIS)
for consistency analysis.

State vector: x = [px, py, theta, vx_world, vy_world, omega]

Prediction model (world-frame, linear in velocities):
    px(t+dt)    = px + vx * dt
    py(t+dt)    = py + vy * dt
    theta(t+dt) = theta + omega * dt
    vx, vy, omega replaced by IMU measurement

Jacobian F:
    F = I,  except  F[0,3]=dt, F[1,4]=dt, F[2,5]=dt

Update model (VO observation):
    z = [px, py, theta]
    H = [[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0]]

Author: Medisetti Renukeswar
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ekf_core.estimator_base import StateEstimator


class EKFEstimator(StateEstimator):
    """
    Extended Kalman Filter for 2-D robot localisation.

    Parameters
    ----------
    dt    : IMU integration timestep (s)
    Q     : Process noise covariance (6×6).  If None, uses tuned defaults.
    R_cam : Camera measurement noise covariance (3×3).  If None, uses defaults.
    """

    def __init__(
        self,
        dt: float = 0.01,
        Q: np.ndarray | None = None,
        R_cam: np.ndarray | None = None,
    ) -> None:
        super().__init__(dt)

        self.x = np.zeros(6)
        self.P = np.diag([0.5, 0.5, 0.3, 0.5, 0.5, 0.1])

        # Default Q — tuned for slow ground robot
        self.Q: np.ndarray = Q if Q is not None else np.diag([
            1e-4,   # px
            1e-4,   # py
            1e-3,   # theta
            0.05,   # vx_world
            0.05,   # vy_world
            0.02,   # omega
        ])

        # Default R_cam — calibrated from VO noise model
        self.R_cam: np.ndarray = R_cam if R_cam is not None else np.diag([
            0.08,   # px  (m)
            0.08,   # py  (m)
            0.03,   # theta (rad)
        ])

    # ------------------------------------------------------------------
    # Prediction step
    # ------------------------------------------------------------------

    def predict(
        self,
        vx_meas: float,
        vy_meas: float,
        om_meas: float,
    ) -> None:
        """
        EKF prediction step.

        Propagates state through the world-frame linear motion model and
        updates covariance via the analytic Jacobian F.
        """
        dt = self.dt
        vx = float(self.x[3])
        vy = float(self.x[4])

        # Nonlinear state propagation
        self.x[0] += vx * dt
        self.x[1] += vy * dt
        self.x[2] += om_meas * dt
        self.x[3]  = vx_meas
        self.x[4]  = vy_meas
        self.x[5]  = om_meas

        self.x[2] = self._wrap_angle(self.x[2])

        # Analytic Jacobian
        F = np.eye(6)
        F[0, 3] = dt   # d(px) / d(vx)
        F[1, 4] = dt   # d(py) / d(vy)
        F[2, 5] = dt   # d(theta) / d(omega)  — kept for completeness;
                       # omega replaced by measurement so this row isn't used
                       # in state update, but is correct for covariance.

        self.P = F @ self.P @ F.T + self.Q

    # ------------------------------------------------------------------
    # Update step
    # ------------------------------------------------------------------

    def update_camera(
        self,
        px_meas: float,
        py_meas: float,
        th_meas: float,
    ) -> dict[str, Any]:
        """
        EKF update step using Visual Odometry pose measurement.

        Uses the Joseph-form covariance update for numerical stability.

        Returns
        -------
        dict with keys: innovation, S, nis, K
        """
        z = np.array([px_meas, py_meas, th_meas])

        H = np.zeros((3, 6))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0

        # Innovation with heading wrap
        y = z - H @ self.x
        y[2] = self._wrap_angle(y[2])

        # Innovation covariance
        S = H @ self.P @ H.T + self.R_cam

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y
        self.x[2] = self._wrap_angle(self.x[2])

        # Joseph-form covariance update (numerically stable)
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R_cam @ K.T

        # Store for NIS computation
        self._last_innovation = y.copy()
        self._last_S = S.copy()

        nis = float(y @ np.linalg.inv(S) @ y)

        return {
            "innovation": y,
            "S": S,
            "nis": nis,
            "K": K,
        }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, x0: np.ndarray, P0: np.ndarray) -> None:
        """Reset filter to specified initial conditions."""
        self.x = x0.copy()
        self.P = P0.copy()
        self._last_innovation = None
        self._last_S = None
