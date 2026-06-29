"""
TUM-VI Dataset Loader
======================
Reads the TUM Visual-Inertial Dataset in its published format.

Dataset structure expected on disk
------------------------------------
<root>/
  dso/
    imu.txt               # timestamp[ns] ax ay az gx gy gz
    cam0/
      images/             # (not used)
  mocap/
    imu_mocap.txt         # timestamp[ns] px py pz qw qx qy qz
  dataset-<name>_512_16/
    dso/
      imu0/
        data.csv          # same as dso/imu.txt in some versions

  Note: TUM-VI provides ground truth only at sequence start and end
  (loop closure points) for most sequences. Room sequences have full
  motion-capture ground truth.

Download
--------
  bash datasets/scripts/download_tumvi.sh
  # or manually from: https://vision.in.tum.de/data/datasets/visual-inertial-dataset
  # room1: https://vision.in.tum.de/tumvi/exported/euroc/512_16/dataset-room1_512_16.tar.gz

Reference
---------
Schubert et al., "The TUM VI Benchmark for Evaluating Visual-Inertial Odometry,"
IROS 2018. https://doi.org/10.1109/IROS.2018.8593419

Author: Medisetti Renukeswar (Phase 6)
"""

from __future__ import annotations

import csv
import math
import os

import numpy as np

from datasets.common.dataset_base import (
    DatasetLoader, IMUSample, PoseSample, SequenceMetadata, quat_to_yaw,
)


class TUMVILoader(DatasetLoader):
    """
    Loader for TUM-VI dataset sequences.

    Supports two layout variants:
      - EuRoC-style ASL CSV (exported format)
      - Raw TUM-VI txt format (dso/imu.txt + mocap/imu_mocap.txt)

    Parameters
    ----------
    root_dir      : path to sequence root directory
    sequence_name : e.g. "room1", "corridor1"
    has_full_gt   : True for room sequences (mocap), False for corridor
    """

    # EuRoC-style (preferred if present)
    IMU_CSV_EUROC = os.path.join("mav0", "imu0", "data.csv")
    GT_CSV_EUROC  = os.path.join("mav0", "mocap0", "data.csv")

    # Raw TUM-VI format
    IMU_TXT = os.path.join("dso", "imu.txt")
    GT_TXT  = os.path.join("mocap", "imu_mocap.txt")

    def __init__(
        self,
        root_dir: str,
        sequence_name: str = "unknown",
        has_full_gt: bool = True,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.sequence_name = sequence_name
        self.has_full_gt = has_full_gt
        self._use_euroc_format: bool = False

    def is_available(self) -> bool:
        root = self.root_dir
        euroc_ok = (
            os.path.isfile(os.path.join(root, self.IMU_CSV_EUROC))
        )
        raw_ok = (
            os.path.isfile(os.path.join(root, self.IMU_TXT))
        )
        return euroc_ok or raw_ok

    def get_metadata(self) -> SequenceMetadata:
        n_imu = len(self._imu_samples)
        n_gt  = len(self._gt_samples)
        dur   = self._imu_samples[-1].timestamp if n_imu > 0 else 0.0
        return SequenceMetadata(
            dataset_name="TUM-VI",
            sequence_name=self.sequence_name,
            duration_s=dur,
            imu_rate_hz=200.0,
            gt_rate_hz=120.0 if self.has_full_gt else 0.0,
            n_imu=n_imu,
            n_gt=n_gt,
            has_gt=self.has_full_gt and n_gt > 0,
            motion_type="handheld",
            difficulty="medium",
            notes="TUM-VI — Schubert et al. 2018",
        )

    def _load(self) -> None:
        root = self.root_dir
        if os.path.isfile(os.path.join(root, self.IMU_CSV_EUROC)):
            self._use_euroc_format = True
            self._imu_samples = self._read_imu_euroc()
            self._gt_samples  = self._read_gt_euroc()
        else:
            self._use_euroc_format = False
            self._imu_samples = self._read_imu_raw()
            self._gt_samples  = self._read_gt_raw()

    # ── EuRoC-style (exported TUM-VI) ─────────────────────────────────────

    def _read_imu_euroc(self) -> list[IMUSample]:
        path = os.path.join(self.root_dir, self.IMU_CSV_EUROC)
        samples: list[IMUSample] = []
        t0: float | None = None
        with open(path, newline="") as f:
            for row in csv.reader(f):
                if not row or row[0].strip().startswith("#"):
                    continue
                ts_ns = int(row[0].strip())
                if t0 is None:
                    t0 = ts_ns
                t_s = (ts_ns - t0) * 1e-9
                # Same column order as EuRoC: ts, wx, wy, wz, ax, ay, az
                samples.append(IMUSample(
                    timestamp=t_s,
                    ax=float(row[4]), ay=float(row[5]), az=float(row[6]),
                    gx=float(row[1]), gy=float(row[2]), gz=float(row[3]),
                ))
        return samples

    def _read_gt_euroc(self) -> list[PoseSample]:
        path = os.path.join(self.root_dir, self.GT_CSV_EUROC)
        if not os.path.isfile(path):
            return []
        samples: list[PoseSample] = []
        t0: float | None = None
        with open(path, newline="") as f:
            for row in csv.reader(f):
                if not row or row[0].strip().startswith("#"):
                    continue
                ts_ns = int(row[0].strip())
                if t0 is None:
                    t0 = ts_ns
                t_s = (ts_ns - t0) * 1e-9
                px, py = float(row[1]), float(row[2])
                yaw = quat_to_yaw(float(row[4]), float(row[5]),
                                   float(row[6]), float(row[7]))
                samples.append(PoseSample(timestamp=t_s, px=px, py=py, theta=yaw))
        return samples

    # ── Raw TUM-VI format ──────────────────────────────────────────────────

    def _read_imu_raw(self) -> list[IMUSample]:
        """
        TUM-VI raw IMU format:
          timestamp[ns] ax ay az gx gy gz   (space-separated)
        """
        path = os.path.join(self.root_dir, self.IMU_TXT)
        samples: list[IMUSample] = []
        t0: float | None = None
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                ts_ns = int(parts[0])
                if t0 is None:
                    t0 = ts_ns
                t_s = (ts_ns - t0) * 1e-9
                ax, ay, az = float(parts[1]), float(parts[2]), float(parts[3])
                gx, gy, gz = float(parts[4]), float(parts[5]), float(parts[6])
                samples.append(IMUSample(
                    timestamp=t_s,
                    ax=ax, ay=ay, az=az,
                    gx=gx, gy=gy, gz=gz,
                ))
        return samples

    def _read_gt_raw(self) -> list[PoseSample]:
        """
        TUM-VI mocap format:
          timestamp[ns] px py pz qw qx qy qz   (space-separated)
        """
        path = os.path.join(self.root_dir, self.GT_TXT)
        if not os.path.isfile(path):
            return []
        samples: list[PoseSample] = []
        t0: float | None = None
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                ts_ns = int(parts[0])
                if t0 is None:
                    t0 = ts_ns
                t_s = (ts_ns - t0) * 1e-9
                px, py = float(parts[1]), float(parts[2])
                yaw = quat_to_yaw(float(parts[4]), float(parts[5]),
                                   float(parts[6]), float(parts[7]))
                samples.append(PoseSample(timestamp=t_s, px=px, py=py, theta=yaw))
        return samples
