"""
Abstract State Estimator Base Class
=====================================
Defines the common interface for all estimators (EKF, UKF, Adaptive variants).

The interface enforces:
  - predict(vx, vy, omega) -> None
  - update_camera(px, py, theta) -> dict   (returns innovation diagnostics)
  - get_state() -> (x, P)
  - reset(x0, P0) -> None

Consistency diagnostics (NEES, NIS) are computed here when ground truth is
available, rather than duplicating the logic in each estimator subclass.

Author: Medisetti Renukeswar (research extension)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ekf_core.metrics import compute_nees, compute_nis


class StateEstimator(ABC):
    """
    Abstract base class for all state estimators.

    State vector convention (shared across all subclasses):
        x = [px, py, theta, vx, vy, omega]   (indices 0–5)

    Parameters
    ----------
    dt : IMU integration timestep (s)
    """

    #: Indices of the position sub-state used in NEES position-only evaluation
    POS_IDX: tuple[int, ...] = (0, 1)
    #: Indices of the full observable state
    OBS_IDX: tuple[int, ...] = (0, 1, 2)

    def __init__(self, dt: float = 0.01) -> None:
        self.dt  = dt
        self.n   = 6    # state dimension
        self.m   = 3    # observation dimension

        # State and covariance — subclass __init__ must initialise these
        self.x: np.ndarray = np.zeros(6)
        self.P: np.ndarray = np.eye(6)

        # Buffers for the most-recent update diagnostics
        self._last_innovation: np.ndarray | None = None
        self._last_S: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Abstract interface — every subclass must implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def predict(
        self,
        vx_meas: float,
        vy_meas: float,
        om_meas: float,
    ) -> None:
        """
        Prediction step using IMU measurement.

        Parameters
        ----------
        vx_meas  : World-frame x velocity from IMU (m/s)
        vy_meas  : World-frame y velocity from IMU (m/s)
        om_meas  : Angular velocity from gyroscope (rad/s)
        """

    @abstractmethod
    def update_camera(
        self,
        px_meas: float,
        py_meas: float,
        th_meas: float,
    ) -> dict[str, Any]:
        """
        Update step using Visual Odometry measurement.

        Parameters
        ----------
        px_meas : Measured x position (m)
        py_meas : Measured y position (m)
        th_meas : Measured heading (rad)

        Returns
        -------
        dict with keys:
            innovation : np.ndarray (3,)
            S          : np.ndarray (3, 3)  innovation covariance
            nis        : float              NIS value
            K          : np.ndarray (6, 3) Kalman gain
        """

    @abstractmethod
    def reset(
        self,
        x0: np.ndarray,
        P0: np.ndarray,
    ) -> None:
        """Reset the filter to a given initial state and covariance."""

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Return copies of current state estimate and covariance."""
        return self.x.copy(), self.P.copy()

    def get_position(self) -> tuple[float, float, float]:
        """Return (px, py, theta) as plain floats."""
        return float(self.x[0]), float(self.x[1]), float(self.x[2])

    def compute_nees_position(self, x_true: np.ndarray) -> float:
        """
        Compute position-only NEES using the current estimate and covariance.

        Parameters
        ----------
        x_true : Full ground-truth state vector (6,)
        """
        return compute_nees(x_true, self.x, self.P, state_indices=list(self.POS_IDX))

    def compute_nees_full(self, x_true: np.ndarray) -> float:
        """
        Compute NEES over the full observable state [px, py, theta].
        """
        return compute_nees(x_true, self.x, self.P, state_indices=list(self.OBS_IDX))

    def compute_nis_last(self) -> float:
        """
        Return the NIS from the most-recent update_camera call.
        Returns NaN if no update has been performed yet.
        """
        if self._last_innovation is None or self._last_S is None:
            return float("nan")
        return compute_nis(self._last_innovation, self._last_S)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return math.atan2(math.sin(angle), math.cos(angle))
