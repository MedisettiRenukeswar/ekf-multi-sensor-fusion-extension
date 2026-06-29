"""
MACE-χ²: Mahalanobis-Gated Adaptive Covariance Estimation
===========================================================
Extends the Mohamed-Schwarz (1999) innovation-based adaptive estimator
with a chi-squared outlier gate.

Motivation
----------
The standard adaptive estimator (AdaptiveEKFEstimator) includes ALL
innovations in the window covariance C_yy, including outliers from:
  - VO dropout (missing frames)
  - Lighting discontinuities
  - Fast motion blur

A single large outlier innovation can inflate C_yy substantially,
causing R to over-adapt and the filter to reduce trust in ALL future
measurements — exactly the opposite of the intended behaviour.

MACE-χ² Fix
-----------
Before including innovation y_k in the window, compute the
Mahalanobis distance squared:

  χ²_k = y_k^T S_k^{-1} y_k

Under the filter's assumed Gaussian model:
  χ²_k ~ χ²(m)   where m = observation dimension (3 here)

A 99% gate threshold:
  τ = χ²_inv(0.99, df=m) = 11.345  (for m=3)

Innovations with χ²_k > τ are rejected from the window.
The gate retains 99% of valid innovations while rejecting outliers.

Effective window size W_eff may be < W when many innovations are
gated; if W_eff == 0, R adaptation is skipped for that step.

Mathematical Justification
--------------------------
The 99% chi-squared gate is the standard Mahalanobis outlier test
used in data association (Bar-Shalom et al., 2001, Chapter 7).
Applied to the innovation window, it makes the covariance estimator
robust to non-Gaussian innovation tails caused by sensor dropout.

References
----------
Bar-Shalom, Y., Li, X.R., Kirubarajan, T. (2001). Estimation with
  Applications to Tracking and Navigation. Wiley. (Ch. 7, data assoc.)
Mohamed, A.H. & Schwarz, K.P. (1999). Adaptive Kalman filtering for
  INS/GPS. Journal of Geodesy 73(4):193–203.
Akhlaghi, S., Zhou, N., Huang, Z. (2017). Adaptive adjustment of noise
  covariance in Kalman filter for dynamic state estimation. IEEE PESGM.

Author: Medisetti Renukeswar
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np
from scipy.stats import chi2 as chi2_dist

from ekf_core.adaptive_estimator import AdaptiveEKFEstimator, AdaptiveUKFEstimator


# Pre-compute the 99% gate threshold for m=3 observation dimensions
_CHI2_GATE_DEFAULT: float = float(chi2_dist.ppf(0.99, df=3))  # ≈ 11.345


class MACEEKFEstimator(AdaptiveEKFEstimator):
    """
    EKF with Mahalanobis-Gated Adaptive Covariance Estimation (MACE-χ²).

    Identical to AdaptiveEKFEstimator (R-only mode) but gates outlier
    innovations before computing the window covariance C_yy.

    Parameters
    ----------
    chi2_gate     : Mahalanobis threshold τ (default χ²_inv(0.99, 3) ≈ 11.345).
                    Set to None to disable gating (reduces to vanilla Adaptive-EKF).
    All other parameters identical to AdaptiveEKFEstimator.

    Additional Attributes
    ---------------------
    n_gated       : Total count of gated (rejected) innovations.
    gate_fraction : Running fraction of gated innovations.
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
        chi2_gate: float | None = _CHI2_GATE_DEFAULT,
        R_min_diag: np.ndarray | None = None,
        R_max_diag: np.ndarray | None = None,
        Q_min_diag: np.ndarray | None = None,
        Q_max_diag: np.ndarray | None = None,
    ) -> None:
        super().__init__(
            dt=dt, Q=Q, R_cam=R_cam,
            window=window, adapt_R=adapt_R, adapt_Q=adapt_Q,
            alpha_smooth=alpha_smooth,
            R_min_diag=R_min_diag, R_max_diag=R_max_diag,
            Q_min_diag=Q_min_diag, Q_max_diag=Q_max_diag,
        )
        self.chi2_gate: float | None = chi2_gate

        # Gating statistics
        self.n_total_innovations: int = 0
        self.n_gated: int = 0

        # Gated innovation buffer (separate from parent's full buffer)
        self._gated_buffer: deque[np.ndarray] = deque(maxlen=window)
        self._gated_P_minus: deque[np.ndarray] = deque(maxlen=window)
        self._gated_K: deque[np.ndarray] = deque(maxlen=window)

    @property
    def gate_fraction(self) -> float:
        """Fraction of innovations rejected by the chi-squared gate."""
        if self.n_total_innovations == 0:
            return 0.0
        return self.n_gated / self.n_total_innovations

    def update_camera(
        self,
        px_meas: float,
        py_meas: float,
        th_meas: float,
    ) -> dict[str, Any]:
        """
        MACE-χ² update step.

        Computes the Mahalanobis distance of the innovation before
        adding it to the adaptation window.  Outlier innovations
        (χ² > gate threshold) are excluded from window covariance
        computation but are still applied to the state estimate.
        """
        # Store P_minus before standard update
        P_minus = self.P.copy()

        # Run the parent EKF update (state + covariance update, NIS computed)
        result = super(AdaptiveEKFEstimator, self).update_camera(
            px_meas, py_meas, th_meas
        )

        y = result["innovation"]
        K = result["K"]
        S = result["S"]

        self.n_total_innovations += 1

        # Chi-squared gate
        is_outlier = False
        if self.chi2_gate is not None:
            try:
                S_inv = np.linalg.inv(S)
                chi2_val = float(y @ S_inv @ y)
            except np.linalg.LinAlgError:
                chi2_val = 0.0
            is_outlier = chi2_val > self.chi2_gate

        if is_outlier:
            self.n_gated += 1
            # Do NOT add to gated buffer — outlier excluded from C_yy
        else:
            self._gated_buffer.append(y.copy())
            self._gated_P_minus.append(P_minus)
            self._gated_K.append(K.copy())

        # Also buffer in parent (for history logging only — we override _adapt)
        self._innovation_buffer.append(y.copy())
        self._P_minus_buffer.append(P_minus)
        self._K_buffer.append(K.copy())

        # Trigger adaptation when gated buffer has enough valid innovations
        if len(self._gated_buffer) >= self.window:
            self._adapt_mace()
            self.n_adaptations += 1

        self.R_history.append(np.diag(self.R_cam).copy())
        self.Q_history.append(np.diag(self.Q).copy())

        return result

    def _adapt_mace(self) -> None:
        """
        MACE adaptation using only gated (non-outlier) innovations.
        """
        Y = np.array(list(self._gated_buffer))   # (W_eff, 3)
        W_eff = len(Y)

        if W_eff == 0:
            return  # All innovations gated — skip adaptation

        C_yy = (Y.T @ Y) / W_eff   # (3, 3)

        H = np.zeros((3, 6))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        P_minus_mean = np.mean(self._gated_P_minus, axis=0)

        if self.adapt_R:
            R_new_full = C_yy - H @ P_minus_mean @ H.T
            R_new_diag = np.clip(np.diag(R_new_full), self.R_min, self.R_max)
            R_current  = np.diag(self.R_cam)
            R_updated  = (1 - self.alpha_smooth) * R_current + self.alpha_smooth * R_new_diag
            self.R_cam = np.diag(R_updated)

        if self.adapt_Q:
            K_mean = np.mean(self._gated_K, axis=0)
            Q_new_diag = np.clip(np.diag(K_mean @ C_yy @ K_mean.T), self.Q_min, self.Q_max)
            Q_current  = np.diag(self.Q)
            Q_updated  = (1 - self.alpha_smooth) * Q_current + self.alpha_smooth * Q_new_diag
            self.Q = np.diag(Q_updated)

    def reset(self, x0: np.ndarray, P0: np.ndarray) -> None:
        super().reset(x0, P0)
        self._gated_buffer.clear()
        self._gated_P_minus.clear()
        self._gated_K.clear()
        self.n_total_innovations = 0
        self.n_gated = 0


class MACEUKFEstimator(AdaptiveUKFEstimator):
    """
    UKF with Mahalanobis-Gated Adaptive Covariance Estimation (MACE-χ²).

    Same gating algorithm as MACEEKFEstimator, applied to UKF.

    Parameters
    ----------
    chi2_gate     : Mahalanobis threshold τ (default ≈ 11.345 for m=3).
    All other parameters identical to AdaptiveUKFEstimator.
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
        chi2_gate: float | None = _CHI2_GATE_DEFAULT,
        R_min_diag: np.ndarray | None = None,
        R_max_diag: np.ndarray | None = None,
        Q_min_diag: np.ndarray | None = None,
        Q_max_diag: np.ndarray | None = None,
    ) -> None:
        super().__init__(
            dt=dt, Q=Q, R_cam=R_cam,
            alpha=alpha, beta=beta, kappa=kappa,
            window=window, adapt_R=adapt_R, adapt_Q=adapt_Q,
            alpha_smooth=alpha_smooth,
            R_min_diag=R_min_diag, R_max_diag=R_max_diag,
            Q_min_diag=Q_min_diag, Q_max_diag=Q_max_diag,
        )
        self.chi2_gate: float | None = chi2_gate
        self.n_total_innovations: int = 0
        self.n_gated: int = 0

        self._gated_buffer: deque[np.ndarray] = deque(maxlen=window)
        self._gated_P_before: deque[np.ndarray] = deque(maxlen=window)
        self._gated_K: deque[np.ndarray] = deque(maxlen=window)

    @property
    def gate_fraction(self) -> float:
        if self.n_total_innovations == 0:
            return 0.0
        return self.n_gated / self.n_total_innovations

    def update_camera(
        self,
        px_meas: float,
        py_meas: float,
        th_meas: float,
    ) -> dict[str, Any]:
        P_before = self.P.copy()
        result = super(AdaptiveUKFEstimator, self).update_camera(
            px_meas, py_meas, th_meas
        )
        y = result["innovation"]
        S = result["S"]
        K = result["K"]

        self.n_total_innovations += 1

        is_outlier = False
        if self.chi2_gate is not None:
            try:
                chi2_val = float(y @ np.linalg.inv(S) @ y)
            except np.linalg.LinAlgError:
                chi2_val = 0.0
            is_outlier = chi2_val > self.chi2_gate

        if is_outlier:
            self.n_gated += 1
        else:
            self._gated_buffer.append(y.copy())
            self._gated_P_before.append(P_before)
            self._gated_K.append(K.copy())

        self._innovation_buffer.append(y.copy())
        self._P_before_buffer.append(P_before)
        self._K_buffer.append(K.copy())
        self._S_buffer.append(S.copy())

        if len(self._gated_buffer) >= self.window:
            self._adapt_mace()
            self.n_adaptations += 1

        self.R_history.append(np.diag(self.R_cam).copy())
        self.Q_history.append(np.diag(self.Q).copy())

        return result

    def _adapt_mace(self) -> None:
        Y = np.array(list(self._gated_buffer))
        W_eff = len(Y)
        if W_eff == 0:
            return
        C_yy = (Y.T @ Y) / W_eff
        H = np.zeros((3, 6))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        P_mean = np.mean(self._gated_P_before, axis=0)
        if self.adapt_R:
            R_new_diag = np.clip(np.diag(C_yy - H @ P_mean @ H.T), self.R_min, self.R_max)
            R_updated  = (1 - self.alpha_smooth) * np.diag(self.R_cam) + self.alpha_smooth * R_new_diag
            self.R_cam = np.diag(R_updated)
        if self.adapt_Q:
            K_mean = np.mean(self._gated_K, axis=0)
            Q_new_diag = np.clip(np.diag(K_mean @ C_yy @ K_mean.T), self.Q_min, self.Q_max)
            Q_updated  = (1 - self.alpha_smooth) * np.diag(self.Q) + self.alpha_smooth * Q_new_diag
            self.Q = np.diag(Q_updated)

    def reset(self, x0: np.ndarray, P0: np.ndarray) -> None:
        super().reset(x0, P0)
        self._gated_buffer.clear()
        self._gated_P_before.clear()
        self._gated_K.clear()
        self.n_total_innovations = 0
        self.n_gated = 0
