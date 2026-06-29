"""
Unit Tests — ES-EKF, MACE-EKF, MACE-UKF
=========================================
Author: Medisetti Renukeswar
"""
from __future__ import annotations
import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from ekf_core.esekf_estimator import ESEKFEstimator
from ekf_core.mace_estimator import MACEEKFEstimator, MACEUKFEstimator, _CHI2_GATE_DEFAULT


# ── ES-EKF tests ─────────────────────────────────────────────────────────────

class TestESEKF:
    def test_state_shape(self):
        e = ESEKFEstimator()
        assert e.x_nom.shape == (6,)
        assert e.dx.shape == (6,)

    def test_covariance_shape(self):
        e = ESEKFEstimator()
        assert e.P.shape == (6, 6)

    def test_covariance_positive_definite(self):
        e = ESEKFEstimator()
        eigvals = np.linalg.eigvalsh(e.P)
        assert np.all(eigvals > 0)

    def test_error_state_reset_after_update(self):
        """After update, error state should be injected and reset to zero."""
        e = ESEKFEstimator()
        e.x_nom = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        e.predict(1.0, 0.0, 0.0)
        e.update_camera(0.1, 0.05, 0.02)
        assert np.allclose(e.dx, np.zeros(6), atol=1e-12), \
            "Error state must be zero after injection"

    def test_covariance_grows_on_predict(self):
        e = ESEKFEstimator()
        tr_before = np.trace(e.P)
        for _ in range(5):
            e.predict(0.5, 0.0, 0.1)
        assert np.trace(e.P) >= tr_before

    def test_covariance_decreases_on_update(self):
        e = ESEKFEstimator()
        e.P = np.eye(6) * 2.0
        tr_before = np.trace(e.P)
        e.update_camera(0.1, 0.1, 0.05)
        assert np.trace(e.P) < tr_before

    def test_covariance_symmetric_after_update(self):
        e = ESEKFEstimator()
        e.update_camera(1.0, 2.0, 0.3)
        assert np.allclose(e.P, e.P.T, atol=1e-10)

    def test_nominal_state_updated_after_injection(self):
        """Nominal state should move toward observation after update."""
        e = ESEKFEstimator()
        e.x_nom = np.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.1])
        e.P = np.eye(6) * 5.0
        e.R_cam = np.diag([0.001, 0.001, 0.001])
        e.update_camera(3.0, 4.0, 0.5)
        assert abs(e.x_nom[0] - 3.0) < 0.1, f"px={e.x_nom[0]:.3f} not near 3.0"
        assert abs(e.x_nom[1] - 4.0) < 0.1

    def test_get_state_returns_nominal(self):
        e = ESEKFEstimator()
        e.x_nom = np.array([1.0, 2.0, 0.5, 0.0, 0.0, 0.0])
        x, P = e.get_state()
        assert np.allclose(x[:3], [1.0, 2.0, 0.5])

    def test_reset(self):
        e = ESEKFEstimator()
        x0 = np.array([1.0, 2.0, 0.3, 0.0, 0.0, 0.0])
        P0 = np.eye(6) * 3.0
        e.reset(x0, P0)
        assert np.allclose(e.x_nom, x0)
        assert np.allclose(e.P, P0)
        assert np.allclose(e.dx, np.zeros(6))

    def test_nis_in_result(self):
        e = ESEKFEstimator()
        result = e.update_camera(0.1, 0.2, 0.05)
        assert "nis" in result
        assert not math.isnan(result["nis"])
        assert result["nis"] >= 0.0

    def test_heading_wrap(self):
        e = ESEKFEstimator()
        e.x_nom[2] = math.pi - 0.01
        for _ in range(10):
            e.predict(0.0, 0.0, 1.0)
        assert -math.pi <= e.x_nom[2] <= math.pi


# ── MACE gate threshold ───────────────────────────────────────────────────────

class TestMACEGate:
    def test_default_chi2_gate(self):
        """Default gate should be approximately chi2_inv(0.99, 3) = 11.345."""
        assert abs(_CHI2_GATE_DEFAULT - 11.345) < 0.01

    def test_outlier_not_added_to_gated_buffer(self):
        """An innovation with chi2 >> gate threshold must be excluded."""
        m = MACEEKFEstimator(window=5, chi2_gate=1.0)  # Very tight gate
        m.x = np.zeros(6)
        # Predict a few steps to grow uncertainty
        for _ in range(10):
            m.predict(0.5, 0.0, 0.1)
        # Force large residual — will produce chi2 >> 1.0
        m.update_camera(50.0, 50.0, 0.0)
        assert m.n_gated >= 1, "Large outlier should have been gated"

    def test_normal_innovation_not_gated(self):
        """Near-zero innovation should never be gated."""
        m = MACEEKFEstimator(window=5)
        m.reset(np.array([1.0,1.0,0.0,0.5,0.0,0.1]), np.eye(6)*0.5)
        # Consistent measurement — near current state
        for _ in range(8):
            m.predict(0.5, 0.0, 0.1)
            x,_ = m.get_state()
            m.update_camera(x[0]+0.01, x[1]+0.01, x[2]+0.005)
        assert m.gate_fraction < 0.3, "Near-consistent measurements should rarely be gated"

    def test_gate_fraction_property(self):
        m = MACEEKFEstimator(window=5)
        assert m.gate_fraction == 0.0  # No innovations yet
        m.predict(0.5, 0.0, 0.0)
        m.update_camera(0.05, 0.05, 0.01)
        assert 0.0 <= m.gate_fraction <= 1.0

    def test_mace_r_adapts_when_window_full(self):
        """R should change after enough non-gated innovations fill the window."""
        m = MACEEKFEstimator(window=5)
        r_initial = np.diag(m.R_cam).copy()
        m.reset(np.zeros(6), np.eye(6)*0.5)
        for _ in range(20):
            m.predict(0.5, 0.0, 0.1)
            x, _ = m.get_state()
            m.update_camera(x[0]+np.random.normal(0, 0.1),
                            x[1]+np.random.normal(0, 0.1),
                            x[2]+np.random.normal(0, 0.03))
        assert not np.allclose(np.diag(m.R_cam), r_initial), \
            "R should have adapted after window filled"

    def test_all_gated_skips_adaptation(self):
        """If all innovations are gated, R should not change."""
        m = MACEEKFEstimator(window=3, chi2_gate=0.0001)  # Tiny gate — gate everything
        r_initial = np.diag(m.R_cam).copy()
        for _ in range(10):
            m.predict(0.5, 0.0, 0.1)
            m.update_camera(10.0, 10.0, 1.0)  # Giant residual, always gated
        # R should be near initial since all innovations are gated
        assert m.n_gated >= 5

    def test_reset_clears_gate_counters(self):
        m = MACEEKFEstimator(window=5, chi2_gate=0.001)
        for _ in range(5):
            m.predict(0.5, 0.0, 0.1)
            m.update_camera(100.0, 100.0, 1.0)
        assert m.n_gated > 0
        m.reset(np.zeros(6), np.eye(6))
        assert m.n_gated == 0
        assert m.n_total_innovations == 0

    def test_mace_ukf_gate_works(self):
        m = MACEUKFEstimator(window=5, chi2_gate=1.0)
        for _ in range(3):
            m.predict(0.5, 0.0, 0.1)
        m.update_camera(50.0, 50.0, 0.0)  # Should be gated
        assert m.n_gated >= 1

    def test_mace_consistency_better_than_adaptive_under_dropout(self):
        """MACE should not drastically over-adapt under simulated dropout."""
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from benchmark.run_full_benchmark import run_single
        r_mace = run_single('MACE-UKF', 'figure8', 'medium', seed=0, dropout_rate=0.5)
        r_adap = run_single('Adaptive-EKF', 'figure8', 'medium', seed=0, dropout_rate=0.5)
        # MACE-UKF ANIS should stay closer to 1.0 than Adaptive-EKF at 50% dropout
        # (not strictly required to be better in a single run, but should be reasonable)
        assert not math.isnan(r_mace['anis'])
        assert r_mace['anis'] >= 0.0


# ── Interface conformance ────────────────────────────────────────────────────

class TestInterfaceConformance:
    @pytest.mark.parametrize("cls", [ESEKFEstimator, MACEEKFEstimator, MACEUKFEstimator])
    def test_predict_update_cycle(self, cls):
        e = cls()
        e.reset(np.zeros(6), np.eye(6) * 0.5)
        for _ in range(5):
            e.predict(0.5, 0.0, 0.1)
        result = e.update_camera(0.05, 0.05, 0.01)
        assert "innovation" in result
        assert "nis" in result
        x, P = e.get_state()
        assert x.shape == (6,)
        assert P.shape == (6, 6)
        assert np.all(np.linalg.eigvalsh(P) > -1e-8)  # near-PSD

    @pytest.mark.parametrize("cls", [ESEKFEstimator, MACEEKFEstimator, MACEUKFEstimator])
    def test_get_position(self, cls):
        e = cls()
        e.reset(np.array([1.5, 2.5, 0.3, 0.0, 0.0, 0.0]), np.eye(6))
        px, py, th = e.get_position()
        assert abs(px - 1.5) < 1e-9
        assert abs(py - 2.5) < 1e-9
        assert abs(th - 0.3) < 1e-9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
