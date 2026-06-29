"""
Error-State Extended Kalman Filter (ES-EKF)
============================================
Implements a 2D Error-State EKF for IMU-visual odometry fusion.

Formulation
-----------
The state is decomposed into:
  - Nominal state x_nom:  propagated deterministically via the motion model
  - Error state δx:       small perturbation estimated by the EKF

State vector (both nominal and error):
  x = [px, py, θ, vx, vy, ω]  ∈ ℝ⁶

For 2D with small-angle heading errors, composition is additive:
  x_true = x_nom ⊕ δx  ≈  x_nom + δx  (with heading wrap)

The error-state Jacobian F_δ differs from the standard EKF Jacobian in
that it propagates the error around the nominal trajectory, reducing
linearisation errors on curved paths.

Error-state propagation:
  δx_{k+1} = F_δ δx_k + w_k,   w_k ~ N(0, Q)

Error-state Jacobian (world-frame model):
  F_δ = I,  F_δ[0,3]=dt,  F_δ[1,4]=dt,  F_δ[2,5]=dt
  (identical structure to EKF Jacobian for this linear model,
   but applied to the error state rather than the full state)

Observation model (VO provides position + heading):
  z = x_nom[0:3] + H δx[0:3] + v_k,  v_k ~ N(0, R)
  Innovation:  y = z - x_nom[0:3]

After each update, the error-state estimate is injected into the
nominal state and the error state is reset to zero:
  x_nom ← x_nom + δx̂
  δx̂   ← 0
  P     ← (I - KH) P (I - KH)^T + KRK^T   (reset covariance)

References
----------
Sola, J. (2017). "Quaternion kinematics for the error-state Kalman filter."
arXiv:1711.02508.  Section 3 (2D simplification used here).

Mourikis, A. I. & Roumeliotis, S. I. (2007). "A multi-state constraint
Kalman filter for vision-aided inertial navigation." ICRA 2007.

Author: Medisetti Renukeswar
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ekf_core.estimator_base import StateEstimator


class ESEKFEstimator(StateEstimator):
    """
    Error-State EKF for 2D robot localisation.

    Parameters
    ----------
    dt    : IMU integration timestep (s)
    Q     : Process noise covariance (6×6)
    R_cam : Camera/VO measurement noise covariance (3×3)
    """

    def __init__(
        self,
        dt: float = 0.01,
        Q: np.ndarray | None = None,
        R_cam: np.ndarray | None = None,
    ) -> None:
        super().__init__(dt)

        # Nominal state:  [px, py, θ, vx, vy, ω]
        self.x_nom: np.ndarray = np.zeros(6)

        # Error-state mean (reset to 0 after each injection)
        self.dx: np.ndarray = np.zeros(6)

        # Error-state covariance
        self.P: np.ndarray = np.diag([0.5, 0.5, 0.3, 0.5, 0.5, 0.1])

        # x is kept as alias pointing to x_nom for base-class metric calls
        self.x: np.ndarray = self.x_nom

        self.Q: np.ndarray = Q if Q is not None else np.diag([
            1e-4, 1e-4, 1e-3, 0.05, 0.05, 0.02,
        ])
        self.R_cam: np.ndarray = R_cam if R_cam is not None else np.diag([
            0.08, 0.08, 0.03,
        ])

        # Observation matrix (selects px, py, θ from error state)
        self._H: np.ndarray = np.zeros((3, 6))
        self._H[0, 0] = self._H[1, 1] = self._H[2, 2] = 1.0

    # ------------------------------------------------------------------
    # Prediction step (nominal propagation + error-state covariance)
    # ------------------------------------------------------------------

    def predict(
        self,
        vx_meas: float,
        vy_meas: float,
        om_meas: float,
    ) -> None:
        """
        ES-EKF prediction step.

        1. Propagate the nominal state via the motion model.
        2. Propagate the error-state covariance via the linearised
           error-state dynamics F_δ (same structure as standard EKF
           Jacobian for this world-frame model).
        """
        dt = self.dt

        # 1. Nominal state propagation  (deterministic, no noise)
        vx = float(self.x_nom[3])
        vy = float(self.x_nom[4])

        self.x_nom[0] += vx * dt
        self.x_nom[1] += vy * dt
        self.x_nom[2] += om_meas * dt
        self.x_nom[3]  = vx_meas
        self.x_nom[4]  = vy_meas
        self.x_nom[5]  = om_meas
        self.x_nom[2]  = self._wrap_angle(self.x_nom[2])

        # 2. Error-state covariance propagation
        #    F_δ is the error-state Jacobian — for this 2D world-frame
        #    model it has the same sparsity pattern as the standard EKF F.
        F_delta = np.eye(6)
        F_delta[0, 3] = dt   # d(δpx) / d(δvx)
        F_delta[1, 4] = dt   # d(δpy) / d(δvy)
        F_delta[2, 5] = dt   # d(δθ)  / d(δω)

        self.P = F_delta @ self.P @ F_delta.T + self.Q

    # ------------------------------------------------------------------
    # Update step (VO measurement)
    # ------------------------------------------------------------------

    def update_camera(
        self,
        px_meas: float,
        py_meas: float,
        th_meas: float,
    ) -> dict[str, Any]:
        """
        ES-EKF update step.

        The VO measurement z = [px_meas, py_meas, θ_meas] is interpreted
        as a noisy observation of the true pose.  The innovation is formed
        against the nominal state:

          y = z - H x_nom

        The Kalman gain and Joseph-form update operate on the error-state
        covariance P.  After the update, the error-state estimate δx̂ is
        injected into the nominal state and reset to zero.

        Returns
        -------
        dict with keys: innovation, S, K, nis
        """
        z = np.array([px_meas, py_meas, th_meas])
        H = self._H

        # Innovation against nominal state
        y = z - H @ self.x_nom
        y[2] = self._wrap_angle(y[2])

        # Innovation covariance
        S = H @ self.P @ H.T + self.R_cam
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # Error-state update
        self.dx = K @ y

        # Joseph-form covariance update (numerically stable)
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R_cam @ K.T

        # Inject error state into nominal and reset
        self.x_nom += self.dx
        self.x_nom[2] = self._wrap_angle(self.x_nom[2])
        self.dx = np.zeros(6)

        # NIS for consistency monitoring
        try:
            S_inv = np.linalg.inv(S)
            nis = float(y @ S_inv @ y)
        except np.linalg.LinAlgError:
            nis = float("nan")

        return {"innovation": y, "S": S, "K": K, "nis": nis}

    # ------------------------------------------------------------------
    # Interface conformance
    # ------------------------------------------------------------------

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        return self.x_nom.copy(), self.P.copy()

    def get_position(self) -> tuple[float, float, float]:
        return float(self.x_nom[0]), float(self.x_nom[1]), float(self.x_nom[2])

    def reset(self, x0: np.ndarray, P0: np.ndarray) -> None:
        self.x_nom = x0.copy()
        self.x = self.x_nom
        self.dx = np.zeros(6)
        self.P = P0.copy()
