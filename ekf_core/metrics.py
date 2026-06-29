"""
Filter Consistency and Accuracy Metrics
========================================
Implements NEES, NIS, ATE, RPE, RMSE and chi-squared consistency bounds
for evaluating EKF / UKF estimators.

Definitions
-----------
NEES  — Normalized Estimation Error Squared
          nees_k = (x_true - x_est)^T P^{-1} (x_true - x_est)
          Under correct filter: nees_k ~ chi²(n_state)

NIS   — Normalized Innovation Squared
          nis_k  = y^T S^{-1} y
          Under correct filter: nis_k ~ chi²(n_obs)

Both are computed per time step and averaged over Monte Carlo runs.
Consistency bounds are derived from the chi-squared distribution at 95 % CI.

Author : Medisetti Renukeswar (research extension)
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.stats import chi2


# ---------------------------------------------------------------------------
# Per-step metrics
# ---------------------------------------------------------------------------

def compute_nees(
    x_true: np.ndarray,
    x_est: np.ndarray,
    P: np.ndarray,
    state_indices: Sequence[int] | None = None,
) -> float:
    """
    Compute Normalized Estimation Error Squared (NEES) for one time step.

    Parameters
    ----------
    x_true        : Ground-truth state vector  (n,)
    x_est         : Filter state estimate       (n,)
    P             : Filter covariance matrix    (n, n)
    state_indices : Subset of state indices to use (e.g. [0,1] for position).
                    If None, uses all states.

    Returns
    -------
    nees : float — scalar NEES value

    Notes
    -----
    NEES is undefined when P is singular; returns NaN in that case.
    """
    if state_indices is not None:
        idx = list(state_indices)
        e = x_true[idx] - x_est[idx]
        P_sub = P[np.ix_(idx, idx)]
    else:
        e = x_true - x_est
        P_sub = P

    # Heading wrap-around for index 2 if present
    if state_indices is not None and 2 in state_indices:
        i2 = list(state_indices).index(2)
        e[i2] = math.atan2(math.sin(e[i2]), math.cos(e[i2]))
    elif state_indices is None and len(e) > 2:
        e[2] = math.atan2(math.sin(e[2]), math.cos(e[2]))

    try:
        P_inv = np.linalg.inv(P_sub)
        return float(e @ P_inv @ e)
    except np.linalg.LinAlgError:
        return float("nan")


def compute_nis(
    innovation: np.ndarray,
    S: np.ndarray,
) -> float:
    """
    Compute Normalized Innovation Squared (NIS) for one update step.

    Parameters
    ----------
    innovation : Innovation vector y = z - H x̂⁻   (m,)
    S          : Innovation covariance S = HPH^T+R  (m, m)

    Returns
    -------
    nis : float
    """
    try:
        S_inv = np.linalg.inv(S)
        return float(innovation @ S_inv @ innovation)
    except np.linalg.LinAlgError:
        return float("nan")


# ---------------------------------------------------------------------------
# Trajectory-level accuracy metrics
# ---------------------------------------------------------------------------

def compute_ate(
    gt_x: np.ndarray,
    gt_y: np.ndarray,
    est_x: np.ndarray,
    est_y: np.ndarray,
) -> float:
    """
    Absolute Trajectory Error — RMSE of per-step Euclidean position errors.
    """
    err = np.sqrt((gt_x - est_x) ** 2 + (gt_y - est_y) ** 2)
    return float(np.sqrt(np.mean(err ** 2)))


def compute_rpe(
    gt_x: np.ndarray,
    gt_y: np.ndarray,
    est_x: np.ndarray,
    est_y: np.ndarray,
    step: int = 10,
) -> float:
    """
    Relative Pose Error — RMSE of relative translational errors over *step* frames.
    """
    n = len(gt_x)
    errors: list[float] = []
    for i in range(0, n - step, step):
        dgt = math.sqrt(
            (gt_x[i + step] - gt_x[i]) ** 2 + (gt_y[i + step] - gt_y[i]) ** 2
        )
        dest = math.sqrt(
            (est_x[i + step] - est_x[i]) ** 2 + (est_y[i + step] - est_y[i]) ** 2
        )
        errors.append((dgt - dest) ** 2)
    return float(np.sqrt(np.mean(errors))) if errors else 0.0


def compute_rmse_position(
    gt_x: np.ndarray,
    gt_y: np.ndarray,
    est_x: np.ndarray,
    est_y: np.ndarray,
) -> float:
    """RMSE of position magnitude error."""
    err = np.sqrt((gt_x - est_x) ** 2 + (gt_y - est_y) ** 2)
    return float(np.sqrt(np.mean(err ** 2)))


def compute_rmse_heading(
    gt_th: np.ndarray,
    est_th: np.ndarray,
) -> float:
    """RMSE of heading error with angle wrapping."""
    diff = gt_th - est_th
    diff = np.arctan2(np.sin(diff), np.cos(diff))
    return float(np.sqrt(np.mean(diff ** 2)))


# ---------------------------------------------------------------------------
# Consistency bounds (chi-squared)
# ---------------------------------------------------------------------------

def chi2_bounds(
    dof: int,
    n_runs: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """
    95 % confidence bounds on the *average* NEES / NIS over n_runs Monte Carlo runs.

    The average of n_runs independent chi²(dof) values divided by dof
    is tested against chi²(n_runs * dof) / (n_runs * dof).

    Returns
    -------
    (lower, upper) : bounds on the normalised average NEES/NIS value
                     (i.e., divide raw sum by dof before comparing)
    """
    total_dof = n_runs * dof
    lower = chi2.ppf(alpha / 2, df=total_dof) / total_dof
    upper = chi2.ppf(1 - alpha / 2, df=total_dof) / total_dof
    return float(lower), float(upper)


def average_nees_bounds(
    dof: int,
    n_runs: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """
    Bounds on the *time-averaged* NEES at 95% CI, averaged over n_runs.
    Returns (lower, upper) for the normalised statistic ANEES = mean(nees)/dof.
    """
    return chi2_bounds(dof, n_runs, alpha)


# ---------------------------------------------------------------------------
# Monte Carlo aggregation
# ---------------------------------------------------------------------------

def monte_carlo_statistics(
    samples: np.ndarray,
    confidence: float = 0.95,
) -> dict[str, float]:
    """
    Compute mean, std, and confidence interval for a 1-D array of Monte Carlo samples.

    Parameters
    ----------
    samples    : 1-D array of scalar metric values across MC runs
    confidence : CI level (default 0.95)

    Returns
    -------
    dict with keys: mean, std, ci_lower, ci_upper, median
    """
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if len(samples) == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "ci_lower": float("nan"), "ci_upper": float("nan"),
                "median": float("nan")}

    mean = float(np.mean(samples))
    std = float(np.std(samples, ddof=1))
    n = len(samples)
    # Normal approximation CI (valid for n >= 30)
    from scipy.stats import t as t_dist
    t_crit = float(t_dist.ppf((1 + confidence) / 2, df=n - 1))
    margin = t_crit * std / math.sqrt(n)
    return {
        "mean": mean,
        "std": std,
        "ci_lower": mean - margin,
        "ci_upper": mean + margin,
        "median": float(np.median(samples)),
    }


# ---------------------------------------------------------------------------
# Statistical significance tests  (Phase E)
# ---------------------------------------------------------------------------

from scipy.stats import wilcoxon as _wilcoxon, rankdata as _rankdata


def wilcoxon_test(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    alternative: str = "two-sided",
) -> dict[str, float]:
    """
    Wilcoxon signed-rank test for paired samples.

    Tests H0: median difference between paired samples A and B is zero.
    Appropriate for non-Gaussian metric distributions common in robotics.

    Parameters
    ----------
    samples_a   : 1-D array from estimator A (N,)
    samples_b   : 1-D array from estimator B (N,)  — paired with A
    alternative : "two-sided" | "greater" | "less"

    Returns
    -------
    dict with keys:
        statistic : W statistic
        p_value   : two-tailed p-value
        significant : bool  (p < 0.05)
        effect_size : rank-biserial correlation r = W / (N*(N+1)/2)
        median_diff : median(a - b)
        mean_diff   : mean(a - b)
    """
    a = np.asarray(samples_a, dtype=float)
    b = np.asarray(samples_b, dtype=float)
    diff = a - b

    # Remove ties (zero differences) for effect size calculation
    n_nonzero = int(np.sum(diff != 0))

    if n_nonzero < 2:
        return {
            "statistic":   float("nan"),
            "p_value":     float("nan"),
            "significant": False,
            "effect_size": float("nan"),
            "median_diff": float(np.median(diff)),
            "mean_diff":   float(np.mean(diff)),
        }

    stat, pval = _wilcoxon(a, b, alternative=alternative, zero_method="wilcox")

    # Rank-biserial correlation as effect size  r = 1 - 2W / (N*(N+1)/2)
    # For signed-rank test: r = W_plus / (W_plus + W_minus) * 2 - 1  (simplified)
    max_w = n_nonzero * (n_nonzero + 1) / 2.0
    effect_size = float(1.0 - 2.0 * stat / max_w) if max_w > 0 else float("nan")

    return {
        "statistic":   float(stat),
        "p_value":     float(pval),
        "significant": bool(pval < 0.05),
        "effect_size": effect_size,
        "median_diff": float(np.median(diff)),
        "mean_diff":   float(np.mean(diff)),
    }


def cohens_d(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
) -> float:
    """
    Cohen's d effect size for two independent samples.

    d = (mean_a - mean_b) / pooled_std

    Conventions: |d| < 0.2 negligible, 0.2–0.5 small,
                 0.5–0.8 medium, > 0.8 large.
    """
    a = np.asarray(samples_a, dtype=float)
    b = np.asarray(samples_b, dtype=float)
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return float("nan")
    pooled = math.sqrt(
        ((n_a - 1) * np.var(a, ddof=1) + (n_b - 1) * np.var(b, ddof=1))
        / (n_a + n_b - 2)
    )
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else float("nan")
