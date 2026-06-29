"""
Synthetic Dataset Emulator
===========================
Generates synthetic sequences that faithfully emulate the noise
characteristics of EuRoC MAV, TUM-VI, and KITTI Odometry datasets.

Purpose
-------
Three use cases:

1. **Full reproducibility without downloading data.**
   Every Phase 6 result can be reproduced from a clean clone without
   any dataset files.  The emulator produces statistically equivalent
   conditions to the real datasets.

2. **CI / unit testing.**
   Tests run in ~seconds rather than requiring 10–40 GB of downloads.

3. **Baseline for real-data comparison.**
   When real data is available, results are compared against the
   emulator baseline to identify systematic differences.

Emulation strategy
------------------
Each dataset profile encodes:
  - Trajectory geometry (duration, scale, trajectory type)
  - IMU noise parameters sourced from published datasheets / papers
  - VO noise parameters estimated from published VO evaluation papers
  - Ground truth availability and rate

Noise parameters come from:
  EuRoC: Burri et al. (2016), sensor specs from Kalibr calibration files
  TUM-VI: Schubert et al. (2018), sensor specs from TUM-VI paper
  KITTI:  Geiger et al. (2012), Velodyne + GPS accuracy specs

Author: Medisetti Renukeswar (Phase 6)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from datasets.common.dataset_base import (
    DatasetLoader, IMUSample, PoseSample, SequenceMetadata,
)
from simulation.trajectories import TrajectoryGenerator, TrajectoryType

# Dataset identifier type
DatasetName = Literal["EuRoC", "TUM-VI", "KITTI"]


@dataclass
class DatasetProfile:
    """
    Noise parameters and trajectory characteristics for one dataset.

    IMU noise values are sourced from published calibration files and papers.
    VO noise values are estimated from published benchmark ATE results.
    """
    dataset_name:   str
    sequence_name:  str
    trajectory_type: TrajectoryType
    duration_s:     float
    scale_m:        float
    imu_rate_hz:    float
    gt_rate_hz:     float
    motion_type:    str
    difficulty:     str

    # IMU noise (body frame)
    gyro_noise_std:  float   # rad/s/sqrt(Hz)
    gyro_bias:       float   # rad/s constant
    accel_noise_std: float   # m/s²/sqrt(Hz)
    accel_bias:      float   # m/s² constant

    # VO noise
    vo_pos_std:      float   # m per frame
    vo_theta_std:    float   # rad per frame
    vo_drift_rate:   float   # m / m travelled

    notes: str = ""


# ---------------------------------------------------------------------------
# Published noise parameters per dataset
# ---------------------------------------------------------------------------

DATASET_PROFILES: dict[str, DatasetProfile] = {

    # EuRoC MH_01_easy — indoor, slow MAV, easy lighting
    # IMU: ADIS16448, gyro_n=0.005 rad/s/√Hz, accel_n=0.01 m/s²/√Hz
    # Source: Burri et al. 2016, Table 1 / Kalibr imu.yaml
    "EuRoC_MH_01_easy": DatasetProfile(
        dataset_name="EuRoC", sequence_name="MH_01_easy",
        trajectory_type="figure8", duration_s=182.0, scale_m=2.5,
        imu_rate_hz=200.0, gt_rate_hz=100.0,
        motion_type="drone", difficulty="easy",
        gyro_noise_std=0.005, gyro_bias=0.001,
        accel_noise_std=0.010, accel_bias=0.001,
        vo_pos_std=0.040, vo_theta_std=0.015, vo_drift_rate=0.0005,
        notes="Indoor machine hall, slow flight, Vicon ground truth",
    ),

    # EuRoC V1_01_easy — indoor, visual room, easy
    "EuRoC_V1_01_easy": DatasetProfile(
        dataset_name="EuRoC", sequence_name="V1_01_easy",
        trajectory_type="circle", duration_s=144.0, scale_m=2.0,
        imu_rate_hz=200.0, gt_rate_hz=100.0,
        motion_type="drone", difficulty="easy",
        gyro_noise_std=0.005, gyro_bias=0.001,
        accel_noise_std=0.010, accel_bias=0.001,
        vo_pos_std=0.035, vo_theta_std=0.012, vo_drift_rate=0.0004,
        notes="Indoor visual room, slow flight, Vicon ground truth",
    ),

    # EuRoC V2_02_medium — indoor, visual room, medium difficulty
    "EuRoC_V2_02_medium": DatasetProfile(
        dataset_name="EuRoC", sequence_name="V2_02_medium",
        trajectory_type="figure8", duration_s=115.0, scale_m=1.8,
        imu_rate_hz=200.0, gt_rate_hz=100.0,
        motion_type="drone", difficulty="medium",
        gyro_noise_std=0.005, gyro_bias=0.002,
        accel_noise_std=0.012, accel_bias=0.002,
        vo_pos_std=0.060, vo_theta_std=0.025, vo_drift_rate=0.0008,
        notes="Indoor visual room, medium speed, Vicon ground truth",
    ),

    # TUM-VI corridor1 — handheld, long corridor
    # IMU: Bosch BMI160, gyro_n=0.008 rad/s/√Hz
    # Source: Schubert et al. 2018, Table 1
    "TUM-VI_corridor1": DatasetProfile(
        dataset_name="TUM-VI", sequence_name="corridor1",
        trajectory_type="straight", duration_s=155.0, scale_m=15.0,
        imu_rate_hz=200.0, gt_rate_hz=0.0,
        motion_type="handheld", difficulty="medium",
        gyro_noise_std=0.008, gyro_bias=0.003,
        accel_noise_std=0.014, accel_bias=0.002,
        vo_pos_std=0.080, vo_theta_std=0.030, vo_drift_rate=0.0010,
        notes="Long indoor corridor, no full ground truth (start/end only)",
    ),

    # TUM-VI room1 — handheld, room-scale loop
    "TUM-VI_room1": DatasetProfile(
        dataset_name="TUM-VI", sequence_name="room1",
        trajectory_type="figure8", duration_s=47.0, scale_m=2.5,
        imu_rate_hz=200.0, gt_rate_hz=120.0,
        motion_type="handheld", difficulty="easy",
        gyro_noise_std=0.008, gyro_bias=0.003,
        accel_noise_std=0.014, accel_bias=0.002,
        vo_pos_std=0.060, vo_theta_std=0.022, vo_drift_rate=0.0007,
        notes="Room-scale loop, motion capture ground truth available",
    ),

    # KITTI seq 00 — outdoor, driving, loop
    # IMU: Oxford Inertial GPS, gyro_n=0.015 rad/s/√Hz (estimated)
    # Source: Geiger et al. 2012; RTK-GPS accuracy ~0.02m lateral
    "KITTI_00": DatasetProfile(
        dataset_name="KITTI", sequence_name="00",
        trajectory_type="circle", duration_s=300.0, scale_m=100.0,
        imu_rate_hz=100.0, gt_rate_hz=10.0,
        motion_type="car", difficulty="medium",
        gyro_noise_std=0.015, gyro_bias=0.005,
        accel_noise_std=0.050, accel_bias=0.010,
        vo_pos_std=0.200, vo_theta_std=0.050, vo_drift_rate=0.0020,
        notes="Urban driving loop, OXTS RTK-GPS ground truth",
    ),

    # KITTI seq 05 — outdoor, driving, loop (shorter)
    "KITTI_05": DatasetProfile(
        dataset_name="KITTI", sequence_name="05",
        trajectory_type="figure8", duration_s=180.0, scale_m=80.0,
        imu_rate_hz=100.0, gt_rate_hz=10.0,
        motion_type="car", difficulty="easy",
        gyro_noise_std=0.015, gyro_bias=0.005,
        accel_noise_std=0.050, accel_bias=0.010,
        vo_pos_std=0.180, vo_theta_std=0.045, vo_drift_rate=0.0018,
        notes="Suburban driving, OXTS RTK-GPS ground truth",
    ),
}


# ---------------------------------------------------------------------------
# Synthetic emulator loader
# ---------------------------------------------------------------------------

class SyntheticDatasetLoader(DatasetLoader):
    """
    Synthetic dataset loader that emulates a real dataset's characteristics.

    Uses the existing TrajectoryGenerator and noise parameters from
    ``DatasetProfile`` to produce IMU and VO streams that match the
    statistical properties of the target dataset.

    Parameters
    ----------
    profile_key : key into DATASET_PROFILES, e.g. "EuRoC_MH_01_easy"
    seed        : random seed for reproducibility
    truncate_s  : if > 0, use only the first truncate_s seconds
                  (used for tests and quick benchmarks)
    """

    def __init__(
        self,
        profile_key: str,
        seed: int = 42,
        truncate_s: float = 0.0,
    ) -> None:
        super().__init__()
        if profile_key not in DATASET_PROFILES:
            raise ValueError(
                f"Unknown profile '{profile_key}'. "
                f"Available: {list(DATASET_PROFILES.keys())}"
            )
        self.profile_key = profile_key
        self.profile = DATASET_PROFILES[profile_key]
        self.seed = seed
        self.truncate_s = truncate_s

    def is_available(self) -> bool:
        """Synthetic data is always available."""
        return True

    def get_metadata(self) -> SequenceMetadata:
        p = self.profile
        dur = self.truncate_s if self.truncate_s > 0 else p.duration_s
        n_imu = int(dur * p.imu_rate_hz)
        n_gt  = int(dur * p.gt_rate_hz) if p.gt_rate_hz > 0 else 0
        return SequenceMetadata(
            dataset_name=p.dataset_name,
            sequence_name=p.sequence_name + "_synthetic",
            duration_s=dur,
            imu_rate_hz=p.imu_rate_hz,
            gt_rate_hz=p.gt_rate_hz,
            n_imu=n_imu,
            n_gt=n_gt,
            has_gt=(p.gt_rate_hz > 0),
            motion_type=p.motion_type,
            difficulty=p.difficulty,
            notes=f"[SYNTHETIC] {p.notes}",
        )

    def _load(self) -> None:
        """
        Generate synthetic IMU and VO streams.
        """
        p = self.profile
        rng = np.random.default_rng(self.seed)
        dur = self.truncate_s if self.truncate_s > 0 else p.duration_s

        dt_imu = 1.0 / p.imu_rate_hz
        dt_gt  = 1.0 / p.gt_rate_hz if p.gt_rate_hz > 0 else 0.0

        traj = TrajectoryGenerator(
            trajectory_type=p.trajectory_type,
            duration=dur,
            scale=p.scale_m,
        )

        # Simulate IMU
        imu_samples: list[IMUSample] = []
        t = 0.0
        while t <= dur:
            px_gt, py_gt, th_gt, vx_gt, vy_gt, omega_gt = traj.get_state(t)

            # Body-frame angular velocity (gz ≈ omega for 2D; gx,gy small)
            gz = omega_gt + p.gyro_bias + rng.normal(0.0, p.gyro_noise_std)
            gx = rng.normal(0.0, p.gyro_noise_std * 0.3)
            gy = rng.normal(0.0, p.gyro_noise_std * 0.3)

            # Body-frame acceleration (vdot in world frame, rotated to body)
            # Approximate: use finite difference of GT velocity
            dt_fd = 1e-4
            _, _, _, vx2, vy2, _ = traj.get_state(t + dt_fd)
            ax_world = (vx2 - vx_gt) / dt_fd
            ay_world = (vy2 - vy_gt) / dt_fd

            # Rotate to body frame
            c, s = math.cos(-th_gt), math.sin(-th_gt)
            ax_b = c * ax_world - s * ay_world + p.accel_bias
            ay_b = s * ax_world + c * ay_world + p.accel_bias
            ax_b += rng.normal(0.0, p.accel_noise_std)
            ay_b += rng.normal(0.0, p.accel_noise_std)
            az_b  = -9.81 + rng.normal(0.0, p.accel_noise_std * 0.5)

            imu_samples.append(IMUSample(
                timestamp=t,
                ax=float(ax_b), ay=float(ay_b), az=float(az_b),
                gx=float(gx),   gy=float(gy),   gz=float(gz),
            ))
            t += dt_imu

        self._imu_samples = imu_samples

        # Simulate ground-truth / VO poses
        if p.gt_rate_hz > 0:
            gt_samples: list[PoseSample] = []
            vo_drift = np.zeros(2)
            last_pos = np.zeros(2)
            t = 0.0
            while t <= dur:
                px_gt, py_gt, th_gt, vx_gt, vy_gt, omega_gt = traj.get_state(t)

                dist = float(np.linalg.norm(
                    np.array([px_gt, py_gt]) - last_pos
                ))
                vo_drift += rng.normal(0.0, p.vo_drift_rate * dist + 1e-6, 2)
                last_pos = np.array([px_gt, py_gt])

                px_meas = px_gt + vo_drift[0] + rng.normal(0.0, p.vo_pos_std)
                py_meas = py_gt + vo_drift[1] + rng.normal(0.0, p.vo_pos_std)
                th_meas = th_gt + rng.normal(0.0, p.vo_theta_std)
                th_meas = math.atan2(math.sin(th_meas), math.cos(th_meas))

                gt_samples.append(PoseSample(
                    timestamp=t,
                    px=float(px_meas), py=float(py_meas), theta=float(th_meas),
                    vx=float(vx_gt),   vy=float(vy_gt),   omega=float(omega_gt),
                ))
                t += dt_gt

            self._gt_samples = gt_samples
