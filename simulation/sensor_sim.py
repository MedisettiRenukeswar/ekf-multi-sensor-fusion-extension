"""
Sensor Simulator
Generates realistic IMU and Visual Odometry measurements
from a known ground-truth figure-8 trajectory.

Author: Medisetti Renukeswar
"""

import numpy as np
import math


class SensorSimulator:
    def __init__(self, dt_imu: float = 0.01, dt_cam: float = 1/30):
        self.dt_imu  = dt_imu
        self.dt_cam  = dt_cam
        np.random.seed(42)

        # IMU noise (MPU-6050 class)
        self.gyro_noise_std   = 0.008   # rad/s
        self.gyro_bias        = 0.003   # rad/s constant bias
        self.accel_noise_std  = 0.04    # m/s^2

        # Camera / VO noise
        self.cam_pos_std   = 0.06    # m
        self.cam_theta_std = 0.025   # rad
        self.vo_drift_rate = 0.0008  # m per m travelled

        self._vo_drift    = np.zeros(2)
        self._last_gt_pos = np.zeros(2)
        self._prev_vx     = None
        self._prev_vy     = None

    def figure8_trajectory(self, t: float, scale: float = 2.5):
        """Lemniscate (figure-8) ground truth. Returns (px,py,theta,vx,vy,omega)."""
        w = 0.18   # angular frequency
        s = scale

        px = s * math.sin(w * t)
        py = s * math.sin(w * t) * math.cos(w * t)   # = s/2 * sin(2wt)

        vx_w =  s * w * math.cos(w * t)
        vy_w =  s * w * math.cos(2 * w * t)

        theta = math.atan2(vy_w, vx_w)

        # Angular velocity via finite difference
        dt = 0.001
        vx2 =  s * w * math.cos(w * (t + dt))
        vy2 =  s * w * math.cos(2 * w * (t + dt))
        th2 = math.atan2(vy2, vx2)
        omega = (th2 - theta) / dt

        return px, py, theta, vx_w, vy_w, omega

    def get_imu(self, t: float):
        """Return (vx_world, vy_world, omega) — world-frame velocities from IMU."""
        _, _, _, vx_w, vy_w, omega_gt = self.figure8_trajectory(t)

        # Add noise to velocities (integrated from accelerometer)
        vx_meas = vx_w + np.random.normal(0, self.accel_noise_std * 0.1)
        vy_meas = vy_w + np.random.normal(0, self.accel_noise_std * 0.1)
        om_meas = omega_gt + self.gyro_bias + np.random.normal(0, self.gyro_noise_std)

        return vx_meas, vy_meas, om_meas

    def get_camera(self, t: float):
        """Return noisy Visual Odometry pose (px, py, theta)."""
        px_gt, py_gt, th_gt, vx, vy, _ = self.figure8_trajectory(t)

        dist = math.sqrt((px_gt - self._last_gt_pos[0])**2 +
                         (py_gt - self._last_gt_pos[1])**2)
        self._vo_drift += np.random.normal(0, self.vo_drift_rate * dist + 1e-6, 2)
        self._last_gt_pos = np.array([px_gt, py_gt])

        px_meas = px_gt + self._vo_drift[0] + np.random.normal(0, self.cam_pos_std)
        py_meas = py_gt + self._vo_drift[1] + np.random.normal(0, self.cam_pos_std)
        th_meas = th_gt + np.random.normal(0, self.cam_theta_std)

        return px_meas, py_meas, th_meas
