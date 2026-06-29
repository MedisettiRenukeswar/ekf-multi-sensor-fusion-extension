"""
Phase 7 Unit Tests
====================
Tests for Phase 7 real-dataset evaluation infrastructure.

Key concerns tested:
  1. Provenance enforcement — RealResult.data_source is always "real_dataset"
  2. No synthetic fallback — check_availability() returns False correctly
  3. Runner exits gracefully with zero results when no data present
  4. All sequence specs resolve to correct loader classes
  5. EuRoC/TUM-VI/KITTI loaders correctly report is_available()=False
     for non-existent paths
  6. RealResult.assert_real() raises on wrong data_source
  7. Table formatters produce non-empty output for real results
  8. Empty-result path produces correct placeholder output
  9. Consistency bounds for N=1 are wider than N=30 (sanity)

Run:
  python -m pytest tests/test_phase7.py -v --tb=short

Author: Medisetti Renukeswar (Phase 7)
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

from benchmark.run_phase7_real import (
    RealResult, SEQUENCE_CATALOGUE, check_availability,
    make_table1, make_table2, make_table3,
    write_real_dataset_report, write_summary,
)
from ekf_core.metrics import average_nees_bounds
from datasets.euroc.euroc_loader import EuRoCLoader
from datasets.tumvi.tumvi_loader import TUMVILoader
from datasets.kitti.kitti_loader import KITTILoader


# ─────────────────────────── Provenance enforcement ──────────────────────────

class TestProvenance:
    def test_real_result_default_source(self):
        """RealResult must default to 'real_dataset'."""
        r = RealResult()
        assert r.data_source == "real_dataset"

    def test_assert_real_passes_on_correct_source(self):
        """assert_real() must not raise for valid result."""
        r = RealResult(data_source="real_dataset")
        r.assert_real()   # no exception

    def test_assert_real_raises_on_synthetic(self):
        """assert_real() must raise if data_source is not 'real_dataset'."""
        r = RealResult(data_source="synthetic")
        with pytest.raises(AssertionError, match="real_dataset"):
            r.assert_real()

    def test_assert_real_raises_on_blank(self):
        r = RealResult(data_source="")
        with pytest.raises(AssertionError):
            r.assert_real()

    def test_result_fields_present(self):
        """All required metric fields must exist."""
        r = RealResult()
        for field in ["ate", "rpe", "rmse_pos", "rmse_heading",
                      "anis", "anees", "nis_consistent", "nees_consistent"]:
            assert hasattr(r, field), f"Missing field: {field}"


# ─────────────────────────── Sequence catalogue ───────────────────────────────

class TestSequenceCatalogue:
    def test_catalogue_non_empty(self):
        assert len(SEQUENCE_CATALOGUE) > 0

    def test_all_datasets_represented(self):
        datasets = {s.dataset for s in SEQUENCE_CATALOGUE}
        assert "EuRoC" in datasets
        assert "TUM-VI" in datasets
        assert "KITTI" in datasets

    def test_euroc_sequences_present(self):
        euroc = [s for s in SEQUENCE_CATALOGUE if s.dataset == "EuRoC"]
        names = [s.sequence_name for s in euroc]
        assert "MH_01_easy"   in names
        assert "V1_01_easy"   in names
        assert "V2_02_medium" in names

    def test_imu_rates_positive(self):
        for spec in SEQUENCE_CATALOGUE:
            assert spec.imu_rate_hz > 0, f"{spec.sequence_name}: imu_rate_hz <= 0"

    def test_loader_classes_correct(self):
        for spec in SEQUENCE_CATALOGUE:
            if spec.dataset == "EuRoC":
                assert spec.loader_cls is EuRoCLoader
            elif spec.dataset == "TUM-VI":
                assert spec.loader_cls is TUMVILoader
            elif spec.dataset == "KITTI":
                assert spec.loader_cls is KITTILoader


# ─────────────────────────── Availability check ──────────────────────────────

class TestAvailabilityCheck:
    def test_no_env_vars_all_missing(self, monkeypatch):
        """Without env vars, all sequences must be MISSING."""
        import benchmark.run_phase7_real as p7
        monkeypatch.setattr(p7, "EUROC_ROOT", "")
        monkeypatch.setattr(p7, "TUMVI_ROOT", "")
        monkeypatch.setattr(p7, "KITTI_ROOT", "")
        status = check_availability()
        n_avail = sum(1 for v in status.values() if v["available"])
        assert n_avail == 0

    def test_nonexistent_root_all_missing(self, monkeypatch, tmp_path):
        """Pointing root to empty dir → all sequences missing."""
        import benchmark.run_phase7_real as p7
        monkeypatch.setattr(p7, "EUROC_ROOT", str(tmp_path))
        monkeypatch.setattr(p7, "TUMVI_ROOT", str(tmp_path))
        monkeypatch.setattr(p7, "KITTI_ROOT", str(tmp_path))
        status = check_availability()
        n_avail = sum(1 for v in status.values() if v["available"])
        assert n_avail == 0

    def test_status_has_all_sequences(self):
        status = check_availability()
        assert len(status) == len(SEQUENCE_CATALOGUE)

    def test_status_reason_populated(self):
        """Every unavailable entry must have a non-empty reason."""
        status = check_availability()
        for key, info in status.items():
            if not info["available"]:
                assert info["reason"], f"{key}: missing reason for unavailability"


# ─────────────────────────── Loaders — unavailability ─────────────────────────

class TestLoaderUnavailability:
    def test_euroc_loader_missing_path(self):
        loader = EuRoCLoader("/nonexistent/path", "MH_01")
        assert not loader.is_available()

    def test_tumvi_loader_missing_path(self):
        loader = TUMVILoader("/nonexistent/path", "room1")
        assert not loader.is_available()

    def test_kitti_loader_missing_path(self):
        loader = KITTILoader("/nonexistent/path", "00")
        assert not loader.is_available()

    def test_euroc_loader_empty_dir(self, tmp_path):
        loader = EuRoCLoader(str(tmp_path), "MH_01")
        assert not loader.is_available()

    def test_kitti_loader_empty_dir(self, tmp_path):
        loader = KITTILoader(str(tmp_path), "00")
        assert not loader.is_available()


# ─────────────────────────── Table formatters ─────────────────────────────────

class TestTableFormatters:
    def setup_method(self):
        lb, ub = average_nees_bounds(dof=3, n_runs=1)
        self.lb, self.ub = lb, ub

    def test_table1_empty_contains_placeholder(self):
        text = make_table1([], self.lb, self.ub)
        assert "no real dataset" in text.lower()
        assert "TABLE 1" in text

    def test_table2_empty_contains_placeholder(self):
        text = make_table2([], self.lb, self.ub)
        assert "no real dataset" in text.lower()
        assert "TABLE 2" in text

    def test_table3_empty_contains_placeholder(self):
        text = make_table3([])
        assert "no real dataset" in text.lower()

    def test_table1_with_real_result(self):
        r = RealResult(
            data_source="real_dataset",
            dataset_name="EuRoC", sequence_name="MH_01_easy",
            estimator_name="EKF",
            ate=0.0456, rpe=0.0321, rmse_pos=0.0456,
            rmse_heading=0.012, n_updates=540, runtime_ms=312.0,
        )
        text = make_table1([r], self.lb, self.ub)
        assert "EuRoC" in text
        assert "MH_01_easy" in text
        assert "EKF" in text
        assert "0.0456" in text

    def test_table2_with_real_result(self):
        r = RealResult(
            data_source="real_dataset",
            dataset_name="EuRoC", sequence_name="MH_01_easy",
            estimator_name="Adaptive-EKF",
            anis=0.85, anees=0.79,
            nis_consistent=True, nees_consistent=True,
        )
        text = make_table2([r], self.lb, self.ub)
        assert "MH_01_easy" in text
        assert "Adaptive-EKF" in text
        assert "YES" in text

    def test_table_rejects_synthetic_source(self):
        """Tables must raise if given a synthetic result."""
        r = RealResult(data_source="synthetic", dataset_name="EuRoC",
                       sequence_name="X", estimator_name="EKF")
        with pytest.raises(AssertionError):
            make_table1([r], self.lb, self.ub)


# ─────────────────────────── Report writers ───────────────────────────────────

class TestReportWriters:
    def test_write_report_no_data(self, tmp_path):
        path = str(tmp_path / "REAL_DATASET_REPORT.md")
        write_real_dataset_report([], path)
        assert os.path.isfile(path)
        content = open(path).read()
        assert "No Real Data Available" in content or "No results" in content

    def test_write_summary_no_data(self, tmp_path):
        path = str(tmp_path / "REAL_DATASET_SUMMARY.md")
        write_summary([], path)
        assert os.path.isfile(path)
        content = open(path).read()
        assert "No results" in content or "not present" in content

    def test_write_report_with_real_result(self, tmp_path):
        r = RealResult(
            data_source="real_dataset",
            dataset_name="EuRoC", sequence_name="MH_01_easy",
            estimator_name="EKF", ate=0.045, anis=0.03,
        )
        path = str(tmp_path / "report.md")
        write_real_dataset_report([r], path)
        content = open(path).read()
        assert "EuRoC" in content
        assert "MH_01_easy" in content

    def test_write_report_rejects_synthetic(self, tmp_path):
        r = RealResult(data_source="synthetic")
        path = str(tmp_path / "report.md")
        with pytest.raises(AssertionError):
            write_real_dataset_report([r], path)


# ─────────────────────────── Consistency bounds sanity ────────────────────────

class TestConsistencyBounds:
    def test_n1_bounds_wider_than_n30(self):
        """Single-run bounds must be wider than N=30 MC bounds."""
        lb1, ub1 = average_nees_bounds(dof=3, n_runs=1)
        lb30, ub30 = average_nees_bounds(dof=3, n_runs=30)
        assert lb1 < lb30, "N=1 lower bound should be lower than N=30"
        assert ub1 > ub30, "N=1 upper bound should be higher than N=30"

    def test_bounds_positive(self):
        lb, ub = average_nees_bounds(dof=3, n_runs=1)
        assert lb > 0 and ub > lb

    def test_bounds_contain_unity(self):
        """Both N=1 and N=30 bounds must bracket 1.0."""
        for n in [1, 30]:
            lb, ub = average_nees_bounds(dof=3, n_runs=n)
            assert lb < 1.0 < ub, f"N={n}: 1.0 not in [{lb},{ub}]"


# ─────────────────────────── Full runner — no-data path ──────────────────────

class TestRunnerNoData:
    def test_runner_exits_cleanly_no_data(self, tmp_path, monkeypatch):
        """Runner must not raise when no datasets are present."""
        import benchmark.run_phase7_real as p7
        monkeypatch.setattr(p7, "EUROC_ROOT", "")
        monkeypatch.setattr(p7, "TUMVI_ROOT", "")
        monkeypatch.setattr(p7, "KITTI_ROOT", "")
        monkeypatch.setattr(p7, "RAW_DIR",   str(tmp_path))
        monkeypatch.setattr(p7, "TABLE_DIR", str(tmp_path))
        monkeypatch.setattr(p7, "PLOT_DIR",  str(tmp_path))
        monkeypatch.setattr(p7, "ROOT",      str(tmp_path))

        # Should complete without exception
        status = p7.check_availability()
        n_avail = sum(1 for v in status.values() if v["available"])
        assert n_avail == 0

        # Calling main table/report functions with empty results must not raise
        lb, ub = average_nees_bounds(dof=3, n_runs=1)
        p7.make_table1([], lb, ub)
        p7.make_table2([], lb, ub)
        p7.make_table3([])
        p7.write_real_dataset_report([], str(tmp_path / "report.md"))
        p7.write_summary([],            str(tmp_path / "summary.md"))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
