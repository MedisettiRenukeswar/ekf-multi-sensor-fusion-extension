"""
Common Dataset Interface
=========================
Defines the shared data structures and abstract base class that all
dataset loaders (EuRoC, TUM-VI, KITTI) must implement.

The interface is deliberately minimal so that the Phase 6 benchmark
runner is fully dataset-agnostic.

Design rationale
----------------
The existing estimators use a world-frame velocity model:
    predict(vx_world, vy_world, omega)
    update_camera(px, py, theta)

Real IMU data provides body-frame accelerations and angular velocities.
The adapter layer in each dataset loader is responsible for converting
body-frame IMU to world-frame velocity increments using the current
heading estimate (dead-reckoning pre-integration).

Ground-truth poses are 6-DOF (3D position + quaternion orientation).
The loader projects these to 2D (x, y, yaw) for compatibility with
the existing 2D estimator state vector [px, py, theta, vx, vy, omega].

All timestamps are in seconds (float64) relative to sequence start.

Author: Medisetti Renukeswar (Phase 6)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator


import numpy as np


# ---------------------------------------------------------------------------
# Data sample types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IMUSample:
    """
    One IMU measurement.

    Parameters
    ----------
    timestamp : seconds since sequence start
    ax, ay, az : accelerometer measurements in body frame (m/s²)
    gx, gy, gz : gyroscope measurements in body frame (rad/s)
    """
    timestamp: float
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float


@dataclass(frozen=True)
class PoseSample:
    """
    One ground-truth or visual-odometry pose sample.

    Parameters
    ----------
    timestamp : seconds since sequence start
    px, py    : 2D position projected from 3D (m)
    theta     : yaw angle (rad), wrapped to (-pi, pi]
    vx, vy    : 2D velocity (m/s), may be NaN if not available
    omega     : angular velocity (rad/s), may be NaN if not available
    """
    timestamp: float
    px: float
    py: float
    theta: float
    vx: float = float('nan')
    vy: float = float('nan')
    omega: float = float('nan')


@dataclass
class SequenceMetadata:
    """
    Metadata for one dataset sequence.

    Parameters
    ----------
    dataset_name  : "EuRoC" | "TUM-VI" | "KITTI"
    sequence_name : e.g. "MH_01_easy", "corridor1", "00"
    duration_s    : total sequence duration (seconds)
    imu_rate_hz   : nominal IMU rate
    gt_rate_hz    : ground-truth pose rate
    n_imu         : number of IMU samples
    n_gt          : number of ground-truth samples
    has_gt        : True if ground-truth is available
    motion_type   : "handheld" | "drone" | "car" | "synthetic"
    difficulty    : "easy" | "medium" | "hard" | "n/a"
    notes         : free-form notes
    """
    dataset_name:  str
    sequence_name: str
    duration_s:    float = 0.0
    imu_rate_hz:   float = 200.0
    gt_rate_hz:    float = 0.0
    n_imu:         int   = 0
    n_gt:          int   = 0
    has_gt:        bool  = True
    motion_type:   str   = "unknown"
    difficulty:    str   = "n/a"
    notes:         str   = ""


# ---------------------------------------------------------------------------
# Abstract loader base class
# ---------------------------------------------------------------------------

class DatasetLoader(ABC):
    """
    Abstract base class for all dataset loaders.

    A loader reads one sequence from disk and exposes its IMU stream
    and ground-truth pose stream through iterators.

    Subclasses implement ``_load()`` which populates
    ``self._imu_samples`` and ``self._gt_samples``.

    Usage
    -----
    loader = EuRoCLoader("/data/EuRoC/MH_01_easy")
    if loader.is_available():
        loader.load()
        for imu in loader.imu_iter():
            ...
        for gt in loader.gt_iter():
            ...
    else:
        # data not present on disk; use synthetic emulation
        pass
    """

    def __init__(self) -> None:
        self._imu_samples: list[IMUSample] = []
        self._gt_samples:  list[PoseSample] = []
        self._loaded:      bool = False
        self.metadata:     SequenceMetadata | None = None

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the dataset files exist on disk."""

    @abstractmethod
    def _load(self) -> None:
        """
        Read data from disk and populate ``_imu_samples`` and ``_gt_samples``.
        Called by ``load()``; must not be called directly.
        """

    @abstractmethod
    def get_metadata(self) -> SequenceMetadata:
        """Return sequence metadata (can be called before load)."""

    def load(self) -> None:
        """Load data from disk. Idempotent."""
        if not self._loaded:
            self._load()
            self._loaded = True
            self.metadata = self.get_metadata()

    def imu_iter(self) -> Iterator[IMUSample]:
        """Iterate over IMU samples in chronological order."""
        if not self._loaded:
            raise RuntimeError("Call load() before iterating.")
        return iter(self._imu_samples)

    def gt_iter(self) -> Iterator[PoseSample]:
        """Iterate over ground-truth pose samples in chronological order."""
        if not self._loaded:
            raise RuntimeError("Call load() before iterating.")
        return iter(self._gt_samples)

    @property
    def imu_samples(self) -> list[IMUSample]:
        return self._imu_samples

    @property
    def gt_samples(self) -> list[PoseSample]:
        return self._gt_samples

    @property
    def n_imu(self) -> int:
        return len(self._imu_samples)

    @property
    def n_gt(self) -> int:
        return len(self._gt_samples)


# ---------------------------------------------------------------------------
# 3D → 2D projection utilities
# ---------------------------------------------------------------------------

def quat_to_yaw(qw: float, qx: float, qy: float, qz: float) -> float:
    """
    Extract yaw angle from a unit quaternion (w, x, y, z).

    Yaw = rotation about the world z-axis, wrapped to (-pi, pi].
    For ground robots and MAVs in near-horizontal flight, yaw is the
    dominant heading angle relevant to 2D localisation.

    Formula: yaw = atan2(2(wz + xy), 1 - 2(y² + z²))
    """
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def rotation_matrix_z(theta: float) -> np.ndarray:
    """2D rotation matrix for angle theta (rad)."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def body_imu_to_world_velocity(
    ax_b: float, ay_b: float,
    gx_b: float, gy_b: float, gz_b: float,
    theta_world: float,
    dt: float,
    prev_vx: float,
    prev_vy: float,
) -> tuple[float, float, float]:
    """
    Convert body-frame IMU to world-frame velocity using Euler integration.

    This is the adapter between real 3D IMU data and the existing
    estimator's predict(vx_world, vy_world, omega) interface.

    Parameters
    ----------
    ax_b, ay_b   : Body-frame x/y accelerations (m/s²)
    gx_b,gy_b,gz_b : Body-frame angular velocities (rad/s)
    theta_world  : Current heading estimate (rad)
    dt           : Time step (s)
    prev_vx,vy   : Previous world-frame velocity estimate (m/s)

    Returns
    -------
    (vx_world, vy_world, omega_world)

    Notes
    -----
    Gravity is assumed to be along world -z; body-frame xy acceleration
    is rotated to world frame and integrated into velocity.
    """
    # Body → world rotation (2D projection of 3D rotation about z)
    R = rotation_matrix_z(theta_world)
    a_body = np.array([ax_b, ay_b])
    a_world = R @ a_body

    # Simple Euler integration
    vx_world = prev_vx + a_world[0] * dt
    vy_world = prev_vy + a_world[1] * dt

    # Use gz as the yaw rate (standard for 2D ground/MAV navigation)
    omega = gz_b

    return float(vx_world), float(vy_world), float(omega)
