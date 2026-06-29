"""
Extended Kalman Filter — 2D Ground Robot
IMU + Camera (Visual Odometry) Fusion

State vector: x = [px, py, theta, vx, vy, omega]  (6-DOF)
  px, py   : position in world frame (m)
  theta    : heading angle (rad)
  vx, vy   : velocity in world frame (m/s)
  omega    : angular velocity (rad/s)

Sensors:
  IMU   : provides angular velocity (omega) at high rate (~100 Hz)
  Camera: provides pose estimate (px, py, theta) from Visual Odometry at ~30 Hz

Author : Medisetti Renukeswar
Project: Multi-Sensor Fusion EKF State Estimation
"""

import numpy as np
import math


class EKF2DRobot:
    """
    Extended Kalman Filter for 2D differential-drive robot.
    Uses world-frame velocity model for stability.
    """

    def __init__(self, dt: float = 0.01):
        self.dt = dt
        self.n  = 6

        # State: [px, py, theta, vx_w, vy_w, omega]  (world-frame velocities)
        self.x = np.zeros(6)

        # Initial covariance
        self.P = np.diag([0.5, 0.5, 0.3, 0.5, 0.5, 0.1])

        # Process noise Q — tuned for slow ground robot
        self.Q = np.diag([
            1e-4,   # px
            1e-4,   # py
            1e-3,   # theta
            0.05,   # vx_world
            0.05,   # vy_world
            0.02,   # omega
        ])

        # Camera measurement noise R
        self.R_cam = np.diag([
            0.08,   # px  (m)
            0.08,   # py  (m)
            0.03,   # theta (rad)
        ])

    def predict(self, vx_meas: float, vy_meas: float, om_meas: float):
        """
        Prediction step.
        vx_meas, vy_meas: world-frame velocity from IMU integration
        om_meas: angular velocity from gyroscope
        """
        dt  = self.dt
        th  = float(self.x[2])
        vx  = float(self.x[3])
        vy  = float(self.x[4])

        # Motion model (world frame — linear)
        self.x[0] += vx * dt
        self.x[1] += vy * dt
        self.x[2] += om_meas * dt
        self.x[3]  = vx_meas
        self.x[4]  = vy_meas
        self.x[5]  = om_meas

        self.x[2] = math.atan2(math.sin(self.x[2]), math.cos(self.x[2]))

        # Jacobian (world-frame model is nearly linear)
        F = np.eye(6)
        F[0, 3] = dt   # dpx/dvx
        F[1, 4] = dt   # dpy/dvy

        self.P = F @ self.P @ F.T + self.Q

    def update_camera(self, px_meas: float, py_meas: float, th_meas: float):
        """Update step using Visual Odometry measurement."""
        z = np.array([px_meas, py_meas, th_meas])

        H = np.zeros((3, 6))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0

        y = z - H @ self.x
        y[2] = math.atan2(math.sin(y[2]), math.cos(y[2]))

        S = H @ self.P @ H.T + self.R_cam
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y

        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R_cam @ K.T

        self.x[2] = math.atan2(math.sin(self.x[2]), math.cos(self.x[2]))

    def get_state(self):
        return self.x.copy(), self.P.copy()

    def get_position(self):
        return float(self.x[0]), float(self.x[1]), float(self.x[2])
