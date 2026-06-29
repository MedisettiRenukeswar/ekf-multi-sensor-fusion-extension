"""
UKF State Estimator
=====================
Unscented Kalman Filter implementation conforming to StateEstimator interface.

The UKF propagates a set of deterministically chosen sigma points through the
(possibly nonlinear) motion model instead of linearising via a Jacobian.
This avoids first-order linearisation errors and typically gives better
covariance estimates when the motion model is highly nonlinear.

Sigma Point Generation (Merwe Scaled Unscented Transform)
----------------------------------------------------------
Given state dimension n, parameters alpha, beta, kappa:
    lambda = alpha^2 * (n + kappa) - n
    c      = n + lambda

    2n+1 sigma points:
        X_0     = x_hat
        X_i     = x_hat + sqrt((n+lambda) * P)_i   for i = 1..n
        X_{n+i} = x_hat - sqrt((n+lambda) * P)_i   for i = 1..n

Weights:
    W_m_0 = lambda / c
    W_c_0 = lambda / c + (1 - alpha^2 + beta)
    W_m_i = W_c_i = 1 / (2*c)   for i = 1..2n

Motion model is identical to EKF: world-frame integration.
Observation model is the same linear H (position + heading).

Reference
---------
Wan & Merwe (2000), "The Unscented Kalman Filter for Nonlinear Estimation".

Author: Medisetti Renukeswar (research extension)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ekf_core.estimator_base import StateEstimator


class UKFEstimator(StateEstimator):
    """
    Unscented Kalman Filter for 2-D robot localisation.

    Parameters
    ----------
    dt    : IMU integration timestep (s)
    Q     : Process noise covariance (6×6).
    R_cam : Camera measurement noise covariance (3×3).
    alpha : Spread of sigma points around mean (default 1e-3, "tight").
    beta  : Prior knowledge of distribution (default 2 for Gaussian).
    kappa : Secondary scaling parameter (default 0).
    """

    def __init__(
        self,
        dt: float = 0.01,
        Q: np.ndarray | None = None,
        R_cam: np.ndarray | None = None,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ) -> None:
        super().__init__(dt)

        self.x = np.zeros(6)
        self.P = np.diag([0.5, 0.5, 0.3, 0.5, 0.5, 0.1])

        self.Q: np.ndarray = Q if Q is not None else np.diag([
            1e-4, 1e-4, 1e-3, 0.05, 0.05, 0.02,
        ])
        self.R_cam: np.ndarray = R_cam if R_cam is not None else np.diag([
            0.08, 0.08, 0.03,
        ])

        # UKF tuning parameters
        self.alpha = alpha
        self.beta  = beta
        self.kappa = kappa

        n = self.n
        lam = alpha ** 2 * (n + kappa) - n
        self._lam  = lam
        self._c    = n + lam

        # Mean and covariance weights
        self._Wm = np.full(2 * n + 1, 1.0 / (2.0 * self._c))
        self._Wc = np.full(2 * n + 1, 1.0 / (2.0 * self._c))
        self._Wm[0] = lam / self._c
        self._Wc[0] = lam / self._c + (1.0 - alpha ** 2 + beta)

    # ------------------------------------------------------------------
    # Sigma point generation
    # ------------------------------------------------------------------

    def _sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """
        Generate 2n+1 sigma points from mean x and covariance P.

        Uses Cholesky decomposition of (n+lambda)*P for numerical stability.

        Returns
        -------
        X : ndarray of shape (2n+1, n)
        """
        n = self.n
        try:
            A = np.linalg.cholesky(self._c * P)
        except np.linalg.LinAlgError:
            # Regularise if P is not positive definite
            P_reg = P + 1e-9 * np.eye(n)
            A = np.linalg.cholesky(self._c * P_reg)

        X = np.zeros((2 * n + 1, n))
        X[0] = x
        for i in range(n):
            X[i + 1]     = x + A[:, i]
            X[i + 1 + n] = x - A[:, i]

        # Wrap heading for all sigma points
        X[:, 2] = np.arctan2(np.sin(X[:, 2]), np.cos(X[:, 2]))
        return X

    # ------------------------------------------------------------------
    # Process model applied to a single sigma point
    # ------------------------------------------------------------------

    def _process_sigma(
        self,
        xi: np.ndarray,
        vx_meas: float,
        vy_meas: float,
        om_meas: float,
    ) -> np.ndarray:
        """
        Apply the motion model to a single sigma point xi.

        World-frame integration model (same as EKF):
            px += vx * dt
            py += vy * dt
            theta += omega_meas * dt
            vx, vy, omega = measurement
        """
        dt = self.dt
        xo = xi.copy()
        xo[0] += xi[3] * dt
        xo[1] += xi[4] * dt
        xo[2] += om_meas * dt
        xo[3]  = vx_meas
        xo[4]  = vy_meas
        xo[5]  = om_meas
        xo[2]  = self._wrap_angle(xo[2])
        return xo

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
        UKF prediction step.

        Propagates sigma points through the motion model, then recovers
        predicted mean and covariance.
        """
        n = self.n
        X = self._sigma_points(self.x, self.P)

        # Propagate sigma points
        X_pred = np.array([
            self._process_sigma(X[i], vx_meas, vy_meas, om_meas)
            for i in range(2 * n + 1)
        ])

        # Predicted mean
        x_pred = np.einsum("i,ij->j", self._Wm, X_pred)
        x_pred[2] = self._wrap_angle(x_pred[2])

        # Predicted covariance
        P_pred = self.Q.copy()
        for i in range(2 * n + 1):
            diff = X_pred[i] - x_pred
            diff[2] = self._wrap_angle(diff[2])
            P_pred += self._Wc[i] * np.outer(diff, diff)

        self.x = x_pred
        self.P = P_pred

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
        UKF update step using Visual Odometry pose measurement.

        Observation model is linear (H matrix), so sigma point propagation
        through the observation model reduces to the standard Kalman update
        but is computed via the UKF unscented transform for consistency.

        Returns
        -------
        dict with keys: innovation, S, nis, K
        """
        n = self.n
        z = np.array([px_meas, py_meas, th_meas])

        X = self._sigma_points(self.x, self.P)

        # Propagate sigma points through observation model (linear H)
        H = np.zeros((3, 6))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0

        Z_sigma = np.array([H @ X[i] for i in range(2 * n + 1)])

        # Predicted measurement mean
        z_pred = np.einsum("i,ij->j", self._Wm, Z_sigma)
        z_pred[2] = self._wrap_angle(z_pred[2])

        # Innovation covariance S and cross-covariance Pxz
        S = self.R_cam.copy()
        Pxz = np.zeros((n, 3))
        for i in range(2 * n + 1):
            dz = Z_sigma[i] - z_pred
            dz[2] = self._wrap_angle(dz[2])
            dx = X[i] - self.x
            dx[2] = self._wrap_angle(dx[2])
            S   += self._Wc[i] * np.outer(dz, dz)
            Pxz += self._Wc[i] * np.outer(dx, dz)

        # Kalman gain
        K = Pxz @ np.linalg.inv(S)

        # Innovation
        y = z - z_pred
        y[2] = self._wrap_angle(y[2])

        # State and covariance update
        self.x = self.x + K @ y
        self.x[2] = self._wrap_angle(self.x[2])

        # Standard covariance update (Joseph form not used for UKF;
        # Joseph form applies to the linear case — UKF uses Pxz-based update)
        self.P = self.P - K @ S @ K.T

        # Ensure symmetry and positive definiteness
        self.P = 0.5 * (self.P + self.P.T)
        min_eig = np.min(np.linalg.eigvalsh(self.P))
        if min_eig < 1e-9:
            self.P += (1e-9 - min_eig) * np.eye(n)

        # Store for NIS
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
