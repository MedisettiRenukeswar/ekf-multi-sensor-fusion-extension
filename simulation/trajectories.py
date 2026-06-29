"""
Trajectory Generator
=====================
Ground-truth trajectory primitives used in all benchmark experiments.

Trajectories
------------
figure8  : Lemniscate of Bernoulli — exercises left + right turns
circle   : Constant-radius circular path — sustained angular velocity
straight : Straight line with constant velocity — no rotation stress

Each trajectory returns the full 6-DOF state at time t:
  (px, py, theta, vx, vy, omega)

Design principle
----------------
Trajectories are parameterised so that the robot travels at a comparable
average speed (~1 m/s) across all three types, ensuring fair comparison.

Author: Medisetti Renukeswar (research extension)
"""

from __future__ import annotations

import math
from typing import Literal

TrajectoryType = Literal["figure8", "circle", "straight"]


class TrajectoryGenerator:
    """
    Generates analytical ground-truth trajectories.

    Parameters
    ----------
    trajectory_type : one of "figure8", "circle", "straight"
    duration        : total simulation time (s)
    scale           : spatial scale factor (m) — adjusts radius / amplitude
    speed           : nominal forward speed (m/s) for straight / circle
    """

    def __init__(
        self,
        trajectory_type: TrajectoryType = "figure8",
        duration: float = 40.0,
        scale: float = 3.0,
        speed: float = 1.0,
    ) -> None:
        self.trajectory_type = trajectory_type
        self.duration = duration
        self.scale = scale
        self.speed = speed

        # Derived frequency so that trajectory fits naturally in `duration`
        if trajectory_type == "figure8":
            # One full figure-8 in `duration` seconds
            self._omega = 2 * math.pi / duration
        elif trajectory_type == "circle":
            # One full circle in `duration` seconds
            self._omega = 2 * math.pi / duration
        else:  # straight
            self._omega = 0.0

    def get_state(self, t: float) -> tuple[float, float, float, float, float, float]:
        """
        Return ground-truth state (px, py, theta, vx, vy, omega) at time t.
        """
        if self.trajectory_type == "figure8":
            return self._figure8(t)
        elif self.trajectory_type == "circle":
            return self._circle(t)
        else:
            return self._straight(t)

    # ------------------------------------------------------------------
    # Figure-8 (Lemniscate of Bernoulli)
    # ------------------------------------------------------------------
    def _figure8(self, t: float) -> tuple[float, float, float, float, float, float]:
        """
        Lemniscate parameterisation.
        x(t) = A * sin(w*t)
        y(t) = A * sin(w*t) * cos(w*t)  = A/2 * sin(2*w*t)
        """
        w = self._omega * 2   # two lobes in one period
        A = self.scale

        px = A * math.sin(w * t)
        py = A * math.sin(w * t) * math.cos(w * t)

        vx = A * w * math.cos(w * t)
        vy = A * w * math.cos(2 * w * t)

        theta = math.atan2(vy, vx)

        # Numerical angular velocity
        dt_small = 1e-4
        vx2 = A * w * math.cos(w * (t + dt_small))
        vy2 = A * w * math.cos(2 * w * (t + dt_small))
        th2 = math.atan2(vy2, vx2)
        omega = math.atan2(math.sin(th2 - theta), math.cos(th2 - theta)) / dt_small

        return px, py, theta, vx, vy, omega

    # ------------------------------------------------------------------
    # Circular trajectory
    # ------------------------------------------------------------------
    def _circle(self, t: float) -> tuple[float, float, float, float, float, float]:
        """
        Counter-clockwise circle of radius `scale`.
        Constant angular velocity omega = 2*pi / duration.
        """
        w = self._omega
        R = self.scale

        px = R * math.cos(w * t - math.pi / 2)  # start at (0, -R)
        py = R * math.sin(w * t - math.pi / 2)

        vx = -R * w * math.sin(w * t - math.pi / 2)
        vy =  R * w * math.cos(w * t - math.pi / 2)

        theta = math.atan2(vy, vx)
        omega = float(w)  # constant

        return px, py, theta, vx, vy, omega

    # ------------------------------------------------------------------
    # Straight line
    # ------------------------------------------------------------------
    def _straight(self, t: float) -> tuple[float, float, float, float, float, float]:
        """
        Straight-line motion along x-axis at constant speed.
        Gentle sinusoidal y-perturbation to avoid a degenerate 1-D case.
        """
        vx = self.speed
        vy = 0.05 * self.speed * math.cos(0.5 * t)   # small transverse wobble

        px = self.speed * t
        py = 0.1 * self.scale * math.sin(0.5 * t)

        theta = math.atan2(vy, vx)

        dt_small = 1e-4
        vy2 = 0.05 * self.speed * math.cos(0.5 * (t + dt_small))
        vx2 = self.speed
        th2 = math.atan2(vy2, vx2)
        omega = math.atan2(math.sin(th2 - theta), math.cos(th2 - theta)) / dt_small

        return px, py, theta, vx, vy, omega
