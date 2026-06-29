"""
Adaptive Covariance Estimator
================================
Implements innovation-based adaptive covariance tuning for both EKF and UKF.

Algorithm — Innovation-Based Adaptive Estimation (IAE / Mohamed-Schwarz)
--------------------------------------------------------------------------
The core idea: under a well-tuned filter the sample covariance of the
innovation sequence y_k should match the theoretical innovation covariance S_k.

  C_yy_hat = (1/W) * sum_{j=k-W+1}^{k}  y_j * y_j^T   (windowed sample covariance)

If C_yy_hat ≠ S_k the filter is mis-calibrated.  We adapt R and Q so that
C_yy_hat ≈ S_k.

Measurement noise adaptation (R adaptation):
    R_new = C_yy_hat - H * P_minus * H^T
    R_new = clip(R_new, R_min, R_max)

Process noise adaptation (Q adaptation):
    Q_new = K * C_yy_hat * K^T
    Q_new = clip(Q_new, Q_min, Q_max)

Only diagonal elements of Q and R are adapted; off-diagonal coupling is
preserved from the initial matrix.  This is a deliberate simplification to
avoid ill-conditioning.

References
----------
Mohamed & Schwarz (1999). "Adaptive Kalman filtering for INS/GPS".
Akhlaghi et al. (2017). "Adaptive adjustment of noise covariance in Kalman filter
for dynamic state estimation."

Author: Medisetti Renukeswar (research extension)
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np

from ekf_core.ekf_estimator import EKFEstimator
from ekf_core.ukf_estimator import UKFEstimator
from ekf_core.estimator_base import StateEstimator


class AdaptiveEKFEstimator(EKFEstimator):
    """
    EKF with online innovation-based adaptive covariance estimation.

    Adapts R_cam (measurement noise) and optionally Q (process noise)
    using a sliding window of innovations.

    Parameters
    ----------
    dt            : IMU timestep (s)
    Q, R_cam      : Initial covariance matrices
    window        : Innovation window length W (default 20)
    adapt_R       : Whether to adapt measurement noise R
    adapt_Q       : Whether to adapt process noise Q
    alpha_smooth  : Exponential smoothing factor for adaptation (0 < alpha <= 1)
                    alpha=1.0 means full immediate update (no smoothing).
    R_min_diag    : Minimum allowed diagonal values for R (positivity constraint)
    R_max_diag    : Maximum allowed diagonal values for R (stability constraint)
    Q_min_diag    : Minimum diagonal values for Q
    Q_max_diag    : Maximum diagonal values for Q
    """

    def __init__(
        self,
        dt: float = 0.01,
        Q: np.ndarray | None = None,
        R_cam: np.ndarray | None = None,
        window: int = 20,
        adapt_R: bool = True,
        adapt_Q: bool = False,
        alpha_smooth: float = 0.1,
        R_min_diag: np.ndarray | None = None,
        R_max_diag: np.ndarray | None = None,
        Q_min_diag: np.ndarray | None = None,
        Q_max_diag: np.ndarray | None = None,
    ) -> None:
        super().__init__(dt=dt, Q=Q, R_cam=R_cam)

        self.window       = window
        self.adapt_R      = adapt_R
        self.adapt_Q      = adapt_Q
        self.alpha_smooth = alpha_smooth

        # Constraints
        self.R_min = R_min_diag if R_min_diag is not None else np.array([1e-4, 1e-4, 1e-4])
        self.R_max = R_max_diag if R_max_diag is not None else np.array([1.0,  1.0,  0.5])
        self.Q_min = Q_min_diag if Q_min_diag is not None else np.array([1e-6, 1e-6, 1e-5, 1e-4, 1e-4, 1e-4])
        self.Q_max = Q_max_diag if Q_max_diag is not None else np.array([0.01, 0.01, 0.05, 0.5,  0.5,  0.2])

        # Innovation buffer
        self._innovation_buffer: deque[np.ndarray] = deque(maxlen=window)
        # Store P_minus for Q adaptation
        self._P_minus_buffer: deque[np.ndarray] = deque(maxlen=window)
        self._K_buffer: deque[np.ndarray] = deque(maxlen=window)

        # Adaptation history for analysis
        self.R_history: list[np.ndarray] = []
        self.Q_history: list[np.ndarray] = []
        self.n_adaptations: int = 0

    def update_camera(
        self,
        px_meas: float,
        py_meas: float,
        th_meas: float,
    ) -> dict[str, Any]:
        """
        Adaptive EKF update step.

        Runs the standard EKF update, then adapts Q and/or R based on
        the windowed innovation covariance.
        """
        # Store P_minus before update
        P_minus = self.P.copy()

        # Standard EKF update
        result = super().update_camera(px_meas, py_meas, th_meas)

        y = result["innovation"]
        K = result["K"]

        # Buffer innovation
        self._innovation_buffer.append(y.copy())
        self._P_minus_buffer.append(P_minus)
        self._K_buffer.append(K.copy())

        # Adapt once the buffer is full
        if len(self._innovation_buffer) >= self.window:
            self._adapt_covariances()
            self.n_adaptations += 1

        # Log history
        self.R_history.append(np.diag(self.R_cam).copy())
        self.Q_history.append(np.diag(self.Q).copy())

        return result

    def _adapt_covariances(self) -> None:
        """
        Perform one adaptation step using the current innovation window.
        """
        Y = np.array(list(self._innovation_buffer))    # (W, 3)
        W = len(Y)

        # Sample innovation covariance
        C_yy = (Y.T @ Y) / W                           # (3, 3)

        H = np.zeros((3, 6))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0

        P_minus_mean = np.mean(self._P_minus_buffer, axis=0)

        # --- Adapt R ---
        if self.adapt_R:
            R_new_full = C_yy - H @ P_minus_mean @ H.T
            # Take diagonal only for stability
            R_new_diag = np.diag(R_new_full)
            R_new_diag = np.clip(R_new_diag, self.R_min, self.R_max)
            # Exponential smoothing
            R_current_diag = np.diag(self.R_cam)
            R_updated_diag = (
                (1.0 - self.alpha_smooth) * R_current_diag
                + self.alpha_smooth * R_new_diag
            )
            self.R_cam = np.diag(R_updated_diag)

        # --- Adapt Q ---
        if self.adapt_Q:
            K_mean = np.mean(self._K_buffer, axis=0)    # (6, 3)
            Q_new_full = K_mean @ C_yy @ K_mean.T       # (6, 6)
            Q_new_diag = np.clip(np.diag(Q_new_full), self.Q_min, self.Q_max)
            Q_current_diag = np.diag(self.Q)
            Q_updated_diag = (
                (1.0 - self.alpha_smooth) * Q_current_diag
                + self.alpha_smooth * Q_new_diag
            )
            self.Q = np.diag(Q_updated_diag)

    def reset(self, x0: np.ndarray, P0: np.ndarray) -> None:
        super().reset(x0, P0)
        self._innovation_buffer.clear()
        self._P_minus_buffer.clear()
        self._K_buffer.clear()
        self.R_history.clear()
        self.Q_history.clear()
        self.n_adaptations = 0


class AdaptiveUKFEstimator(UKFEstimator):
    """
    UKF with online innovation-based adaptive covariance estimation.

    Same adaptation algorithm as AdaptiveEKFEstimator, applied to UKF.

    Parameters
    ----------
    Same as AdaptiveEKFEstimator, plus UKF-specific alpha, beta, kappa.
    """

    def __init__(
        self,
        dt: float = 0.01,
        Q: np.ndarray | None = None,
        R_cam: np.ndarray | None = None,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
        window: int = 20,
        adapt_R: bool = True,
        adapt_Q: bool = False,
        alpha_smooth: float = 0.1,
        R_min_diag: np.ndarray | None = None,
        R_max_diag: np.ndarray | None = None,
        Q_min_diag: np.ndarray | None = None,
        Q_max_diag: np.ndarray | None = None,
    ) -> None:
        super().__init__(dt=dt, Q=Q, R_cam=R_cam, alpha=alpha, beta=beta, kappa=kappa)

        self.window       = window
        self.adapt_R      = adapt_R
        self.adapt_Q      = adapt_Q
        self.alpha_smooth = alpha_smooth

        self.R_min = R_min_diag if R_min_diag is not None else np.array([1e-4, 1e-4, 1e-4])
        self.R_max = R_max_diag if R_max_diag is not None else np.array([1.0,  1.0,  0.5])
        self.Q_min = Q_min_diag if Q_min_diag is not None else np.array([1e-6, 1e-6, 1e-5, 1e-4, 1e-4, 1e-4])
        self.Q_max = Q_max_diag if Q_max_diag is not None else np.array([0.01, 0.01, 0.05, 0.5,  0.5,  0.2])

        self._innovation_buffer: deque[np.ndarray] = deque(maxlen=window)
        self._S_buffer: deque[np.ndarray] = deque(maxlen=window)
        self._K_buffer: deque[np.ndarray] = deque(maxlen=window)
        self._P_before_buffer: deque[np.ndarray] = deque(maxlen=window)

        self.R_history: list[np.ndarray] = []
        self.Q_history: list[np.ndarray] = []
        self.n_adaptations: int = 0

    def update_camera(
        self,
        px_meas: float,
        py_meas: float,
        th_meas: float,
    ) -> dict[str, Any]:
        P_before = self.P.copy()
        result = super().update_camera(px_meas, py_meas, th_meas)

        self._innovation_buffer.append(result["innovation"].copy())
        self._S_buffer.append(result["S"].copy())
        self._K_buffer.append(result["K"].copy())
        self._P_before_buffer.append(P_before)

        if len(self._innovation_buffer) >= self.window:
            self._adapt_covariances()
            self.n_adaptations += 1

        self.R_history.append(np.diag(self.R_cam).copy())
        self.Q_history.append(np.diag(self.Q).copy())

        return result

    def _adapt_covariances(self) -> None:
        Y = np.array(list(self._innovation_buffer))    # (W, 3)
        W = len(Y)
        C_yy = (Y.T @ Y) / W

        H = np.zeros((3, 6))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        P_mean = np.mean(self._P_before_buffer, axis=0)

        if self.adapt_R:
            R_new_full = C_yy - H @ P_mean @ H.T
            R_new_diag = np.clip(np.diag(R_new_full), self.R_min, self.R_max)
            R_current  = np.diag(self.R_cam)
            R_updated  = (1 - self.alpha_smooth) * R_current + self.alpha_smooth * R_new_diag
            self.R_cam = np.diag(R_updated)

        if self.adapt_Q:
            K_mean = np.mean(self._K_buffer, axis=0)
            Q_new_diag = np.clip(np.diag(K_mean @ C_yy @ K_mean.T), self.Q_min, self.Q_max)
            Q_current  = np.diag(self.Q)
            Q_updated  = (1 - self.alpha_smooth) * Q_current + self.alpha_smooth * Q_new_diag
            self.Q = np.diag(Q_updated)

    def reset(self, x0: np.ndarray, P0: np.ndarray) -> None:
        super().reset(x0, P0)
        self._innovation_buffer.clear()
        self._S_buffer.clear()
        self._K_buffer.clear()
        self._P_before_buffer.clear()
        self.R_history.clear()
        self.Q_history.clear()
        self.n_adaptations = 0


class AdaptiveEKFQR(AdaptiveEKFEstimator):
    """
    EKF with simultaneous online adaptation of both Q and R.

    Adaptive mode: ``adaptive_mode = "QR"``

    The R adaptation follows Mohamed & Schwarz (1999).
    The Q adaptation uses the Kalman-gain-weighted innovation covariance:

        Q_new = K * C_yy * K^T

    where C_yy is the windowed sample innovation covariance and K is the
    mean Kalman gain over the window.  Diagonal clipping enforces
    physical constraints on both matrices.

    This class is a strict extension of AdaptiveEKFEstimator (R-only);
    it adds Q adaptation while preserving the identical interface.

    Parameters
    ----------
    All parameters identical to AdaptiveEKFEstimator, plus:
    adaptive_mode : str  — "QR" (both) or "R" (R-only, same as parent)
    """

    adaptive_mode: str = "QR"

    def __init__(
        self,
        dt: float = 0.01,
        Q: np.ndarray | None = None,
        R_cam: np.ndarray | None = None,
        window: int = 20,
        alpha_smooth: float = 0.1,
        adaptive_mode: str = "QR",
        R_min_diag: np.ndarray | None = None,
        R_max_diag: np.ndarray | None = None,
        Q_min_diag: np.ndarray | None = None,
        Q_max_diag: np.ndarray | None = None,
    ) -> None:
        adapt_q = (adaptive_mode.upper() == "QR")
        super().__init__(
            dt=dt, Q=Q, R_cam=R_cam,
            window=window, adapt_R=True, adapt_Q=adapt_q,
            alpha_smooth=alpha_smooth,
            R_min_diag=R_min_diag, R_max_diag=R_max_diag,
            Q_min_diag=Q_min_diag, Q_max_diag=Q_max_diag,
        )
        self.adaptive_mode = adaptive_mode.upper()
