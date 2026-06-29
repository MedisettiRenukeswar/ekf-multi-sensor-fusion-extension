"""
Phase 6 Unit Tests
===================
Tests for all Phase 6 components:
  - DatasetBase (IMUSample, PoseSample, quat_to_yaw)
  - SyntheticDatasetLoader for all 6 profiles
  - EuRoCLoader (parse logic with mock data)
  - TUMVILoader (parse logic with mock data)
  - KITTILoader (parse logic with mock data)
  - DatasetAdapter (predict/update loop)
  - Integration: all 4 estimators run on synthetic sequences

Run:
  python -m pytest tests/test_phase6.py -v --tb=short

Author: Medisetti Renukeswar (Phase 6)
"""

from __future__ import annotations

import csv
import math
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.common.dataset_base import (
    IMUSample, PoseSample, SequenceMetadata, DatasetLoader,
    quat_to_yaw, body_imu_to_world_velocity,
)
from datasets.common.synthetic_loader import (
    SyntheticDatasetLoader, DATASET_PROFILES,
)
from datasets.common.dataset_adapter import DatasetAdapter, AdapterConfig
from datasets.euroc.euroc_loader import EuRoCLoader
from datasets.tumvi.tumvi_loader import TUMVILoader
from datasets.kitti.kitti_loader import KITTILoader
from ekf_core.ekf_estimator import EKFEstimator
from ekf_core.ukf_estimator import UKFEstimator
from ekf_core.adaptive_estimator import AdaptiveEKFEstimator, AdaptiveUKFEstimator


# ─────────────────────────── DatasetBase ─────────────────────────────────────

class TestDatasetBase:
    def test_imu_sample_frozen(self):
        s = IMUSample(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        with pytest.raises((AttributeError, TypeError)):
            s.timestamp = 1.0  # type: ignore

    def test_pose_sample_defaults_nan(self):
        s = PoseSample(0.0, 1.0, 2.0, 0.5)
        assert math.isnan(s.vx)
        assert math.isnan(s.vy)
        assert math.isnan(s.omega)

    def test_quat_to_yaw_identity(self):
        """Identity quaternion → yaw = 0."""
        yaw = quat_to_yaw(1.0, 0.0, 0.0, 0.0)
        assert abs(yaw) < 1e-9

    def test_quat_to_yaw_90_degrees(self):
        """90-degree rotation about z → yaw = pi/2."""
        # q = cos(45°), 0, 0, sin(45°)
        c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
        yaw = quat_to_yaw(c, 0.0, 0.0, s)
        assert abs(yaw - math.pi / 2) < 1e-6

    def test_quat_to_yaw_minus_90(self):
        c, s = math.cos(-math.pi / 4), math.sin(-math.pi / 4)
        yaw = quat_to_yaw(c, 0.0, 0.0, s)
        assert abs(yaw - (-math.pi / 2)) < 1e-6

    def test_quat_to_yaw_range(self):
        """Yaw must always be in (-pi, pi]."""
        rng = np.random.default_rng(0)
        for _ in range(100):
            q = rng.normal(size=4)
            q /= np.linalg.norm(q)
            yaw = quat_to_yaw(*q)
            assert -math.pi <= yaw <= math.pi

    def test_body_imu_world_velocity_zero_accel(self):
        """Zero acceleration → velocity unchanged."""
        vx, vy, omega = body_imu_to_world_velocity(
            0.0, 0.0, 0.0, 0.0, 0.5,
            theta_world=0.0, dt=0.01,
            prev_vx=1.0, prev_vy=0.5,
        )
        assert abs(vx - 1.0) < 1e-9
        assert abs(vy - 0.5) < 1e-9
        assert abs(omega - 0.5) < 1e-9


# ─────────────────────────── SyntheticDatasetLoader ──────────────────────────

class TestSyntheticLoader:
    @pytest.mark.parametrize("profile_key", list(DATASET_PROFILES.keys()))
    def test_all_profiles_available(self, profile_key: str):
        loader = SyntheticDatasetLoader(profile_key, seed=0, truncate_s=5.0)
        assert loader.is_available()

    @pytest.mark.parametrize("profile_key", list(DATASET_PROFILES.keys()))
    def test_all_profiles_load(self, profile_key: str):
        loader = SyntheticDatasetLoader(profile_key, seed=0, truncate_s=5.0)
        loader.load()
        assert loader.n_imu > 0, f"{profile_key}: no IMU samples"

    def test_imu_timestamps_monotone(self):
        loader = SyntheticDatasetLoader("EuRoC_MH_01_easy", seed=0, truncate_s=5.0)
        loader.load()
        ts = [s.timestamp for s in loader.imu_samples]
        assert all(ts[i] < ts[i + 1] for i in range(len(ts) - 1))

    def test_gt_timestamps_monotone(self):
        loader = SyntheticDatasetLoader("EuRoC_MH_01_easy", seed=0, truncate_s=5.0)
        loader.load()
        if loader.n_gt > 1:
            ts = [s.timestamp for s in loader.gt_samples]
            assert all(ts[i] < ts[i + 1] for i in range(len(ts) - 1))

    def test_reproducibility(self):
        """Same seed → identical samples."""
        a = SyntheticDatasetLoader("TUM-VI_room1", seed=42, truncate_s=2.0)
        b = SyntheticDatasetLoader("TUM-VI_room1", seed=42, truncate_s=2.0)
        a.load(); b.load()
        assert a.n_imu == b.n_imu
        for sa, sb in zip(a.imu_samples[:10], b.imu_samples[:10]):
            assert sa == sb

    def test_different_seeds_differ(self):
        a = SyntheticDatasetLoader("EuRoC_MH_01_easy", seed=0,  truncate_s=2.0)
        b = SyntheticDatasetLoader("EuRoC_MH_01_easy", seed=99, truncate_s=2.0)
        a.load(); b.load()
        # At least some IMU samples must differ
        diffs = [sa.gz != sb.gz for sa, sb in zip(a.imu_samples, b.imu_samples)]
        assert any(diffs)

    def test_metadata_populated(self):
        loader = SyntheticDatasetLoader("KITTI_00", seed=0, truncate_s=10.0)
        loader.load()
        meta = loader.get_metadata()
        assert meta.dataset_name == "KITTI"
        assert meta.n_imu > 0
        assert meta.duration_s > 0

    def test_truncation_respected(self):
        full  = SyntheticDatasetLoader("EuRoC_MH_01_easy", seed=0, truncate_s=0.0)
        short = SyntheticDatasetLoader("EuRoC_MH_01_easy", seed=0, truncate_s=10.0)
        full.load(); short.load()
        # Short should have far fewer samples
        assert short.n_imu < full.n_imu

    def test_unknown_profile_raises(self):
        with pytest.raises(ValueError, match="Unknown profile"):
            SyntheticDatasetLoader("nonexistent_profile")


# ─────────────────────────── EuRoCLoader (mock files) ─────────────────────────

def write_euroc_mock(tmpdir: str) -> str:
    """Create minimal EuRoC directory structure with 5 IMU + 3 GT samples."""
    imu_dir = os.path.join(tmpdir, "mav0", "imu0")
    gt_dir  = os.path.join(tmpdir, "mav0", "state_groundtruth_estimate0")
    os.makedirs(imu_dir, exist_ok=True)
    os.makedirs(gt_dir,  exist_ok=True)

    with open(os.path.join(imu_dir, "data.csv"), "w") as f:
        f.write("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y,w_RS_S_z,a_RS_S_x,a_RS_S_y,a_RS_S_z\n")
        for i in range(5):
            ts = 1000000000 + i * 5000000  # 200 Hz
            f.write(f"{ts},0.01,0.00,0.15,0.05,0.02,-9.80\n")

    with open(os.path.join(gt_dir, "data.csv"), "w") as f:
        f.write("# timestamp [ns], p_RS_R_x, p_RS_R_y, p_RS_R_z, "
                "q_RS_w, q_RS_x, q_RS_y, q_RS_z, "
                "v_RS_R_x, v_RS_R_y, v_RS_R_z, ...\n")
        for i in range(3):
            ts  = 1000000000 + i * 10000000  # 100 Hz
            px  = float(i) * 0.1
            qw  = math.cos(math.pi / 8)
            qz  = math.sin(math.pi / 8)
            f.write(f"{ts},{px},0.05,0.0,{qw},0.0,0.0,{qz},0.5,0.0,0.0\n")

    return tmpdir


class TestEuRoCLoader:
    def test_not_available_for_missing_dir(self):
        loader = EuRoCLoader("/nonexistent/path", "MH_01")
        assert not loader.is_available()

    def test_loads_mock_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_euroc_mock(tmp)
            loader = EuRoCLoader(tmp, "mock")
            assert loader.is_available()
            loader.load()
            assert loader.n_imu == 5
            assert loader.n_gt  == 3

    def test_timestamps_start_at_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_euroc_mock(tmp)
            loader = EuRoCLoader(tmp, "mock")
            loader.load()
            assert abs(loader.imu_samples[0].timestamp) < 1e-9

    def test_gt_theta_in_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_euroc_mock(tmp)
            loader = EuRoCLoader(tmp, "mock")
            loader.load()
            for s in loader.gt_samples:
                assert -math.pi <= s.theta <= math.pi

    def test_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_euroc_mock(tmp)
            loader = EuRoCLoader(tmp, "mock")
            loader.load()
            meta = loader.get_metadata()
            assert meta.dataset_name == "EuRoC"
            assert meta.n_imu == 5
            assert meta.n_gt  == 3


# ─────────────────────────── TUMVILoader (mock files) ────────────────────────

def write_tumvi_mock(tmpdir: str) -> str:
    imu_dir = os.path.join(tmpdir, "dso")
    gt_dir  = os.path.join(tmpdir, "mocap")
    os.makedirs(imu_dir, exist_ok=True)
    os.makedirs(gt_dir,  exist_ok=True)

    with open(os.path.join(imu_dir, "imu.txt"), "w") as f:
        f.write("# timestamp[ns] ax ay az gx gy gz\n")
        for i in range(6):
            ts = 1000000000 + i * 5000000
            f.write(f"{ts} 0.05 0.02 -9.80 0.01 0.00 0.15\n")

    with open(os.path.join(gt_dir, "imu_mocap.txt"), "w") as f:
        f.write("# timestamp[ns] px py pz qw qx qy qz\n")
        for i in range(4):
            ts = 1000000000 + i * 8333333  # ~120 Hz
            px = float(i) * 0.05
            f.write(f"{ts} {px} 0.1 0.0 1.0 0.0 0.0 0.0\n")

    return tmpdir


class TestTUMVILoader:
    def test_not_available_for_missing_dir(self):
        loader = TUMVILoader("/nonexistent/path", "room1")
        assert not loader.is_available()

    def test_loads_raw_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_tumvi_mock(tmp)
            loader = TUMVILoader(tmp, "mock", has_full_gt=True)
            assert loader.is_available()
            loader.load()
            assert loader.n_imu == 6
            assert loader.n_gt  == 4

    def test_timestamps_start_at_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_tumvi_mock(tmp)
            loader = TUMVILoader(tmp, "mock")
            loader.load()
            assert abs(loader.imu_samples[0].timestamp) < 1e-9


# ─────────────────────────── KITTILoader (mock files) ─────────────────────────

def write_kitti_mock(tmpdir: str) -> str:
    seq_dir   = os.path.join(tmpdir, "sequences", "00")
    poses_dir = os.path.join(tmpdir, "poses")
    os.makedirs(seq_dir,   exist_ok=True)
    os.makedirs(poses_dir, exist_ok=True)

    # times.txt — 5 frames at 10 Hz
    with open(os.path.join(seq_dir, "times.txt"), "w") as f:
        for i in range(5):
            f.write(f"{i * 0.1:.6f}\n")

    # poses/00.txt — forward motion along x
    with open(os.path.join(poses_dir, "00.txt"), "w") as f:
        for i in range(5):
            px = float(i) * 1.0
            f.write(f"1 0 0 {px}  0 1 0 0  0 0 1 0\n")

    return tmpdir


class TestKITTILoader:
    def test_not_available_for_missing_dir(self):
        loader = KITTILoader("/nonexistent/path", "00")
        assert not loader.is_available()

    def test_loads_mock_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_kitti_mock(tmp)
            loader = KITTILoader(tmp, "00")
            assert loader.is_available()
            loader.load()
            assert loader.n_gt  == 5
            assert loader.n_imu > 0   # synthesised from poses

    def test_gt_monotone_forward(self):
        """Forward motion → px should increase."""
        with tempfile.TemporaryDirectory() as tmp:
            write_kitti_mock(tmp)
            loader = KITTILoader(tmp, "00")
            loader.load()
            pxs = [s.px for s in loader.gt_samples]
            assert all(pxs[i] <= pxs[i + 1] for i in range(len(pxs) - 1))


# ─────────────────────────── DatasetAdapter ──────────────────────────────────

class TestDatasetAdapter:
    @pytest.mark.parametrize("est_cls", [
        EKFEstimator, UKFEstimator, AdaptiveEKFEstimator, AdaptiveUKFEstimator
    ])
    def test_adapter_runs_all_estimators(self, est_cls):
        """All four estimators must complete a synthetic run without error."""
        loader = SyntheticDatasetLoader("EuRoC_MH_01_easy", seed=0, truncate_s=5.0)
        loader.load()
        dt = 1.0 / loader.get_metadata().imu_rate_hz
        est = est_cls(dt=dt)
        adapter = DatasetAdapter(est, AdapterConfig(imu_rate_hz=200.0))
        result = adapter.run(loader, estimator_name=est_cls.__name__)
        assert result.n_imu > 0
        assert math.isfinite(result.runtime_ms)

    def test_ate_finite_with_gt(self):
        """ATE must be finite when ground truth is available."""
        loader = SyntheticDatasetLoader("EuRoC_MH_01_easy", seed=0, truncate_s=5.0)
        loader.load()
        assert loader.n_gt > 0
        dt = 1.0 / loader.get_metadata().imu_rate_hz
        est = EKFEstimator(dt=dt)
        adapter = DatasetAdapter(est, AdapterConfig(imu_rate_hz=200.0))
        result = adapter.run(loader)
        assert math.isfinite(result.ate), f"ATE is not finite: {result.ate}"

    def test_nis_is_positive(self):
        """Mean NIS must be positive when updates occur."""
        loader = SyntheticDatasetLoader("TUM-VI_room1", seed=5, truncate_s=5.0)
        loader.load()
        dt = 1.0 / loader.get_metadata().imu_rate_hz
        est = EKFEstimator(dt=dt)
        adapter = DatasetAdapter(est, AdapterConfig(imu_rate_hz=200.0))
        result = adapter.run(loader)
        if result.n_updates > 0:
            assert result.mean_nis > 0

    @pytest.mark.parametrize("profile", ["EuRoC_MH_01_easy", "TUM-VI_room1",
                                          "KITTI_00"])
    def test_all_dataset_profiles(self, profile):
        """Adapter must complete for all dataset profiles."""
        loader = SyntheticDatasetLoader(profile, seed=0, truncate_s=5.0)
        loader.load()
        dt = 1.0 / loader.get_metadata().imu_rate_hz
        est = EKFEstimator(dt=dt)
        adapter = DatasetAdapter(est, AdapterConfig(imu_rate_hz=dt ** -1))
        result = adapter.run(loader)
        assert result.n_imu > 0

    def test_covariance_stays_finite(self):
        """Estimator covariance must remain finite after full run."""
        loader = SyntheticDatasetLoader("EuRoC_V1_01_easy", seed=3, truncate_s=5.0)
        loader.load()
        dt = 1.0 / loader.get_metadata().imu_rate_hz
        est = UKFEstimator(dt=dt)
        adapter = DatasetAdapter(est, AdapterConfig(imu_rate_hz=200.0))
        adapter.run(loader)
        _, P = est.get_state()
        assert np.all(np.isfinite(P)), "UKF covariance contains NaN/Inf"

    def test_adaptive_ekf_adapts_R(self):
        """Adaptive-EKF R must change during a synthetic dataset run."""
        loader = SyntheticDatasetLoader("EuRoC_MH_01_easy", seed=0, truncate_s=10.0)
        loader.load()
        dt = 1.0 / loader.get_metadata().imu_rate_hz
        est = AdaptiveEKFEstimator(dt=dt, window=20, adapt_R=True,
                                   adapt_Q=False, alpha_smooth=0.1)
        R_init = est.R_cam.copy()
        adapter = DatasetAdapter(est, AdapterConfig(imu_rate_hz=200.0))
        adapter.run(loader)
        assert not np.allclose(est.R_cam, R_init), "R should have been adapted"


# ─────────────────────────── Integration ─────────────────────────────────────

class TestPhase6Integration:
    def test_all_six_profiles_full_run(self):
        """All 6 synthetic profiles complete with finite ATE."""
        failures = []
        for profile_key in DATASET_PROFILES:
            loader = SyntheticDatasetLoader(profile_key, seed=0, truncate_s=5.0)
            loader.load()
            dt  = 1.0 / loader.get_metadata().imu_rate_hz
            est = EKFEstimator(dt=dt)
            adapter = DatasetAdapter(est, AdapterConfig(imu_rate_hz=dt ** -1))
            result = adapter.run(loader, estimator_name="EKF")
            if result.n_updates > 0 and not math.isfinite(result.ate):
                failures.append(f"{profile_key}: ATE not finite")
        assert not failures, "\n".join(failures)

    def test_existing_tests_still_pass(self):
        """Verify existing filter tests are not broken by Phase 6 imports."""
        from ekf_core.ekf_estimator import EKFEstimator as E
        from ekf_core.ukf_estimator import UKFEstimator as U
        ekf = E(dt=0.01)
        ukf = U(dt=0.01)
        x0  = np.zeros(6)
        P0  = np.eye(6)
        ekf.reset(x0, P0)
        ukf.reset(x0, P0)
        ekf.predict(1.0, 0.0, 0.1)
        ukf.predict(1.0, 0.0, 0.1)
        assert np.all(np.isfinite(ekf.P))
        assert np.all(np.isfinite(ukf.P))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
