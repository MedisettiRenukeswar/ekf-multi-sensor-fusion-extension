"""
Research Sensor Simulator
==========================
Generates realistic IMU and Visual Odometry measurements from any
TrajectoryGenerator instance.

Noise Regimes
-------------
low    : ~50 % of nominal MPU-6050 specs — ideal conditions
medium : nominal MPU-6050 specs — baseline (matches original project)
high   : ~3× nominal specs — degraded / outdoor conditions

The sensor model is identical in structure to the original sensor_sim.py
but parameterised through a noise regime for reproducible multi-condition
experiments.

Author: Medisetti Renukeswar (research extension)
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from simulation.trajectories import TrajectoryGenerator

NoiseRegime = Literal["low", "medium", "high"]

# Noise scaling factors relative to nominal (medium)
_REGIME_SCALE: dict[str, float] = {
    "low":    0.4,
    "medium": 1.0,
    "high":   3.0,
}


class ResearchSensorSimulator:
    """
    Sensor simulator for benchmark experiments.

    Parameters
    ----------
    trajectory  : TrajectoryGenerator instance supplying ground-truth state
    noise_regime: "low" | "medium" | "high"
    dt_imu      : IMU timestep (s)
    dt_cam      : Camera / VO timestep (s)
    seed        : Random seed for reproducibility
    """

    # Nominal (medium) noise parameters — MPU-6050 class
    _GYRO_NOISE_STD_NOM   = 0.008   # rad/s
    _GYRO_BIAS_NOM        = 0.003   # rad/s constant
    _ACCEL_NOISE_STD_NOM  = 0.04    # m/s²
    _CAM_POS_STD_NOM      = 0.06    # m
    _CAM_THETA_STD_NOM    = 0.025   # rad
    _VO_DRIFT_RATE_NOM    = 0.0008  # m / m

    def __init__(
        self,
        trajectory: TrajectoryGenerator,
        noise_regime: NoiseRegime = "medium",
        dt_imu: float = 0.01,
        dt_cam: float = 1 / 30,
        seed: int = 42,
    ) -> None:
        self.trajectory   = trajectory
        self.noise_regime = noise_regime
        self.dt_imu       = dt_imu
        self.dt_cam       = dt_cam

        scale = _REGIME_SCALE[noise_regime]

        self.gyro_noise_std  = self._GYRO_NOISE_STD_NOM  * scale
        self.gyro_bias       = self._GYRO_BIAS_NOM        * scale
        self.accel_noise_std = self._ACCEL_NOISE_STD_NOM  * scale
        self.cam_pos_std     = self._CAM_POS_STD_NOM      * scale
        self.cam_theta_std   = self._CAM_THETA_STD_NOM    * scale
        self.vo_drift_rate   = self._VO_DRIFT_RATE_NOM    * scale

        self._rng = np.random.default_rng(seed)
        self._vo_drift: np.ndarray = np.zeros(2)
        self._last_gt_pos: np.ndarray = np.zeros(2)

    def reset(self, seed: int | None = None) -> None:
        """Reset internal drift accumulators and optionally re-seed the RNG."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._vo_drift = np.zeros(2)
        self._last_gt_pos = np.zeros(2)

    def get_imu(self, t: float) -> tuple[float, float, float]:
        """
        Return noisy IMU measurement (vx_world, vy_world, omega).

        Notes
        -----
        Returns world-frame velocity estimates (integrated accelerometer)
        and gyroscope-derived angular velocity, consistent with the
        original EKF motion model.
        """
        _, _, _, vx_gt, vy_gt, omega_gt = self.trajectory.get_state(t)

        vx_meas = vx_gt + self._rng.normal(0.0, self.accel_noise_std * 0.1)
        vy_meas = vy_gt + self._rng.normal(0.0, self.accel_noise_std * 0.1)
        om_meas = omega_gt + self.gyro_bias + self._rng.normal(0.0, self.gyro_noise_std)

        return float(vx_meas), float(vy_meas), float(om_meas)

    def get_camera(self, t: float) -> tuple[float, float, float]:
        """
        Return noisy Visual Odometry pose measurement (px, py, theta).
        """
        px_gt, py_gt, th_gt, vx, vy, _ = self.trajectory.get_state(t)

        dist = math.sqrt(
            (px_gt - self._last_gt_pos[0]) ** 2
            + (py_gt - self._last_gt_pos[1]) ** 2
        )
        self._vo_drift += self._rng.normal(
            0.0, self.vo_drift_rate * dist + 1e-6, 2
        )
        self._last_gt_pos = np.array([px_gt, py_gt])

        px_meas = px_gt + self._vo_drift[0] + self._rng.normal(0.0, self.cam_pos_std)
        py_meas = py_gt + self._vo_drift[1] + self._rng.normal(0.0, self.cam_pos_std)
        th_meas = th_gt + self._rng.normal(0.0, self.cam_theta_std)

        return float(px_meas), float(py_meas), float(th_meas)

    def noise_summary(self) -> dict[str, float]:
        """Return a dict of effective noise parameters for this regime."""
        return {
            "regime": self.noise_regime,          # type: ignore[dict-item]
            "gyro_noise_std":  self.gyro_noise_std,
            "gyro_bias":       self.gyro_bias,
            "accel_noise_std": self.accel_noise_std,
            "cam_pos_std":     self.cam_pos_std,
            "cam_theta_std":   self.cam_theta_std,
            "vo_drift_rate":   self.vo_drift_rate,
        }


# ---------------------------------------------------------------------------
# Degradation scenarios  (Phase E)
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DegradationConfig:
    """
    Configuration for sensor degradation scenarios.

    Scenario A — Time-varying IMU bias (random walk):
        The gyroscope bias is no longer constant; it performs a Gaussian
        random walk with diffusion coefficient `bias_rw_std` (rad/s/sqrt(s)).
        This models temperature drift, vibration, and ageing effects that
        cause bias to wander over time.
        Physical basis: Allan variance gyro bias instability, typically
        0.001–0.01 rad/s for MEMS gyros.

    Scenario B — Visual Odometry dropout:
        Camera measurements are randomly unavailable with probability
        `vo_dropout_prob` at each camera trigger.  When dropout occurs the
        filter runs predict-only until the next successful measurement.
        This models feature-poor scenes, motion blur, and lighting failures.
        dropout_prob = 0.30 → 30% of frames lost
        dropout_prob = 0.50 → 50% of frames lost

    Parameters
    ----------
    enable_bias_random_walk : bool
        Activate time-varying IMU bias (Scenario A).
    bias_rw_std : float
        Gyro bias random-walk diffusion (rad/s / sqrt(Hz)).
        Applied as  bias += N(0, bias_rw_std * sqrt(dt))  each IMU step.
    enable_vo_dropout : bool
        Activate VO dropout (Scenario B).
    vo_dropout_prob : float
        Probability [0,1) that any given camera frame is dropped.
    """
    enable_bias_random_walk: bool  = False
    bias_rw_std:             float = 0.002   # rad/s/sqrt(Hz) — moderate wander

    enable_vo_dropout:  bool  = False
    vo_dropout_prob:    float = 0.30          # 30% default


class DegradedSensorSimulator(ResearchSensorSimulator):
    """
    ResearchSensorSimulator extended with physically-motivated degradation
    scenarios for robustness evaluation.

    Inherits all noise-regime parameters from ResearchSensorSimulator.
    Degradation is additive on top of the chosen noise regime.

    Parameters
    ----------
    trajectory      : TrajectoryGenerator
    noise_regime    : "low" | "medium" | "high"
    degradation     : DegradationConfig
    dt_imu, dt_cam  : timesteps
    seed            : reproducibility seed
    """

    def __init__(
        self,
        trajectory: "TrajectoryGenerator",
        noise_regime: "NoiseRegime" = "medium",
        degradation: DegradationConfig | None = None,
        dt_imu: float = 0.01,
        dt_cam: float = 1 / 30,
        seed: int = 42,
    ) -> None:
        super().__init__(
            trajectory=trajectory,
            noise_regime=noise_regime,
            dt_imu=dt_imu,
            dt_cam=dt_cam,
            seed=seed,
        )
        self.degradation = degradation or DegradationConfig()

        # Mutable bias state for random-walk scenario
        self._gyro_bias_current: float = self.gyro_bias

    def reset(self, seed: int | None = None) -> None:
        """Reset base simulator and bias state."""
        super().reset(seed)
        self._gyro_bias_current = self.gyro_bias

    def get_imu(self, t: float) -> tuple[float, float, float]:
        """
        Return IMU measurement with optional time-varying bias random walk.

        When bias_random_walk is enabled the gyro bias diffuses by
            delta_bias ~ N(0, bias_rw_std^2 * dt_imu)
        at every call, producing a realistic random-walk trajectory in bias
        space.  The bias is bounded at ±0.1 rad/s to prevent catastrophic
        divergence in long simulations.
        """
        _, _, _, vx_gt, vy_gt, omega_gt = self.trajectory.get_state(t)

        # Update bias random walk
        if self.degradation.enable_bias_random_walk:
            drift = self._rng.normal(
                0.0,
                self.degradation.bias_rw_std * math.sqrt(self.dt_imu),
            )
            self._gyro_bias_current = float(
                np.clip(self._gyro_bias_current + drift, -0.10, 0.10)
            )
        else:
            self._gyro_bias_current = self.gyro_bias

        vx_meas = vx_gt + self._rng.normal(0.0, self.accel_noise_std * 0.1)
        vy_meas = vy_gt + self._rng.normal(0.0, self.accel_noise_std * 0.1)
        om_meas = (omega_gt
                   + self._gyro_bias_current
                   + self._rng.normal(0.0, self.gyro_noise_std))

        return float(vx_meas), float(vy_meas), float(om_meas)

    def camera_available(self) -> bool:
        """
        Return True if the camera measurement is available this frame.

        When VO dropout is enabled each call draws a Bernoulli sample.
        The RNG state is advanced identically whether or not we actually
        call get_camera(), keeping sequences reproducible.
        """
        if self.degradation.enable_vo_dropout:
            return bool(self._rng.random() >= self.degradation.vo_dropout_prob)
        return True

    def degradation_summary(self) -> dict:
        """Return effective degradation parameters for logging."""
        return {
            "bias_rw":        self.degradation.enable_bias_random_walk,
            "bias_rw_std":    self.degradation.bias_rw_std,
            "vo_dropout":     self.degradation.enable_vo_dropout,
            "vo_dropout_prob": self.degradation.vo_dropout_prob,
        }
