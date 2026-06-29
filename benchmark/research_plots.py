"""
Research Visualization
=======================
Generates publication-quality plots from benchmark results:
  1. ATE comparison bar chart (3 noise × 4 estimators × 3 trajectories)
  2. NIS / NEES consistency plot with chi-squared bounds
  3. Adaptive covariance trace (R adaptation over time)
  4. Estimator accuracy vs noise-regime scatter

Author: Medisetti Renukeswar (research extension)
"""

from __future__ import annotations

import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─────────────────────────── load CSV ─────────────────────────────────────────

def load_results(csv_path: str) -> list[dict]:
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def flt(d: dict, key: str) -> float:
    return float(d[key])


# ─────────────────────────── style ────────────────────────────────────────────

BG     = "#0a0f1a"
ACCENT = "#00e5ff"
COLORS = {
    "EKF":          "#00e5ff",
    "UKF":          "#00ff88",
    "Adaptive-EKF": "#b96dff",
    "Adaptive-UKF": "#ffd600",
}
NOISE_ORDER = ["low", "medium", "high"]
EST_ORDER   = ["EKF", "UKF", "Adaptive-EKF", "Adaptive-UKF"]
TRAJ_ORDER  = ["figure8", "circle", "straight"]


def style_ax(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors="#aaccdd", labelsize=8)
    ax.xaxis.label.set_color("#aaccdd")
    ax.yaxis.label.set_color("#aaccdd")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1a3a5a")
    ax.grid(True, alpha=0.15, color="#00e5ff", linewidth=0.5)


# ─────────────────────────── Plot 1: ATE comparison ──────────────────────────

def plot_ate_comparison(results: list[dict], out_dir: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    fig.patch.set_facecolor(BG)
    fig.suptitle("ATE Comparison — EKF vs UKF vs Adaptive Variants\n(mean ± std, N=10 Monte Carlo runs)",
                 color="white", fontsize=12, y=1.01)

    x_positions = np.arange(len(NOISE_ORDER))
    width = 0.18

    for ax_idx, traj in enumerate(TRAJ_ORDER):
        ax = axes[ax_idx]
        style_ax(ax)

        for i, est in enumerate(EST_ORDER):
            ate_means, ate_stds = [], []
            for noise in NOISE_ORDER:
                row = next((r for r in results
                            if r["trajectory"] == traj
                            and r["noise_regime"] == noise
                            and r["estimator"] == est), None)
                if row:
                    ate_means.append(flt(row, "ate_mean"))
                    ate_stds.append(flt(row, "ate_std"))
                else:
                    ate_means.append(0.0)
                    ate_stds.append(0.0)

            offset = (i - 1.5) * width
            bars = ax.bar(x_positions + offset, ate_means, width,
                          color=COLORS[est], alpha=0.85,
                          yerr=ate_stds, capsize=3,
                          error_kw={"ecolor": "white", "alpha": 0.6},
                          label=est)

        ax.set_title(f"Trajectory: {traj}", fontsize=10)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(["Low", "Medium", "High"])
        ax.set_xlabel("Noise Regime")
        ax.set_ylabel("ATE (m) RMSE") if ax_idx == 0 else None

    # Legend
    handles = [mpatches.Patch(color=COLORS[e], label=e) for e in EST_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               facecolor="#0e1a2a", edgecolor="#1a3a5a", labelcolor="white",
               fontsize=9, bbox_to_anchor=(0.5, -0.06))

    plt.tight_layout()
    path = os.path.join(out_dir, "ate_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────── Plot 2: NIS / NEES consistency ──────────────────

def plot_consistency(results: list[dict], out_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Filter Consistency — ANIS and ANEES\n(dashed lines = 95% chi² bounds)",
                 color="white", fontsize=12)

    # Get bounds (same for all rows)
    lb = flt(results[0], "nis_lb")
    ub = flt(results[0], "nis_ub")

    for metric_idx, (metric_key, title) in enumerate([
        ("nis_mean",  "Average NIS (ANIS)"),
        ("nees_mean", "Average NEES (ANEES)"),
    ]):
        ax = axes[metric_idx]
        style_ax(ax)

        # Horizontal consistency bounds
        ax.axhline(lb, color="#ff6b35", linestyle="--", linewidth=1.2,
                   alpha=0.8, label=f"95% lower ({lb:.3f})")
        ax.axhline(ub, color="#ff6b35", linestyle="--", linewidth=1.2,
                   alpha=0.8, label=f"95% upper ({ub:.3f})")
        ax.axhline(1.0, color="white", linestyle=":", linewidth=0.8, alpha=0.5,
                   label="Ideal (1.0)")

        # Group by estimator and noise
        x_labels = []
        x_pos = []
        counter = 0
        for noise in NOISE_ORDER:
            for est in EST_ORDER:
                # Average over trajectories
                vals = [flt(r, metric_key) for r in results
                        if r["noise_regime"] == noise and r["estimator"] == est]
                if vals:
                    val = np.mean(vals)
                    color = COLORS[est]
                    ax.bar(counter, val, color=color, alpha=0.8, width=0.7)
                    x_labels.append(f"{est[:3]}")
                    x_pos.append(counter)
                    counter += 1
            counter += 0.5  # gap between noise groups

        # Noise group labels
        group_centers = [1.5, 5.0, 8.5]
        for gc, noise in zip(group_centers, NOISE_ORDER):
            ax.text(gc, -0.05, noise, ha="center", va="top",
                    color="#aaccdd", fontsize=8,
                    transform=ax.get_xaxis_transform())

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Normalised Statistic")
        ax.legend(fontsize=7, facecolor="#0e1a2a", edgecolor="#1a3a5a",
                  labelcolor="white", loc="upper left")

    plt.tight_layout()
    path = os.path.join(out_dir, "consistency_plot.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────── Plot 3: Adaptive R trace ────────────────────────

def plot_adaptive_trace(out_dir: str) -> None:
    """Run one Adaptive-EKF and one Adaptive-UKF on medium figure-8 and plot R trace."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from simulation.trajectories import TrajectoryGenerator
    from simulation.research_sensor_sim import ResearchSensorSimulator
    from ekf_core.adaptive_estimator import AdaptiveEKFEstimator, AdaptiveUKFEstimator

    traj = TrajectoryGenerator("figure8", duration=40.0, scale=3.0)
    sim  = ResearchSensorSimulator(traj, "high", seed=42)

    DT_IMU = 0.01
    DT_CAM = 1 / 30

    # --- Adaptive EKF ---
    aekf = AdaptiveEKFEstimator(dt=DT_IMU, window=20, adapt_R=True, adapt_Q=False,
                                 alpha_smooth=0.1)
    px0, py0, th0, vx0, vy0, om0 = traj.get_state(0.0)
    aekf.reset(np.array([px0, py0, th0, vx0, vy0, om0]),
               np.diag([0.5, 0.5, 0.3, 0.5, 0.5, 0.1]))

    t, cam_timer = 0.0, 0.0
    while t <= 40.0:
        vx, vy, om = sim.get_imu(t)
        aekf.predict(vx, vy, om)
        cam_timer += DT_IMU
        if cam_timer >= DT_CAM:
            cam_timer = 0.0
            aekf.update_camera(*sim.get_camera(t))
        t += DT_IMU

    R_hist_ekf = np.array(aekf.R_history)      # (N_updates, 3)

    # --- Adaptive UKF ---
    sim.reset(seed=42)
    aukf = AdaptiveUKFEstimator(dt=DT_IMU, window=20, adapt_R=True, adapt_Q=False,
                                 alpha_smooth=0.1)
    aukf.reset(np.array([px0, py0, th0, vx0, vy0, om0]),
               np.diag([0.5, 0.5, 0.3, 0.5, 0.5, 0.1]))

    t, cam_timer = 0.0, 0.0
    while t <= 40.0:
        vx, vy, om = sim.get_imu(t)
        aukf.predict(vx, vy, om)
        cam_timer += DT_IMU
        if cam_timer >= DT_CAM:
            cam_timer = 0.0
            aukf.update_camera(*sim.get_camera(t))
        t += DT_IMU

    R_hist_ukf = np.array(aukf.R_history)

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Adaptive R Covariance Trace — High Noise, Figure-8\n"
                 "(dashed = true noise variance = 0.0036 m²)",
                 color="white", fontsize=11)

    labels = ["R_px (m²)", "R_py (m²)", "R_theta (rad²)"]
    colors_r = [ACCENT, "#00ff88", "#ffd600"]
    true_R = [0.06 ** 2 * 9, 0.06 ** 2 * 9, 0.025 ** 2 * 9]   # high regime: 3x noise

    for ax_idx, (R_hist, title) in enumerate([
        (R_hist_ekf, "Adaptive-EKF"),
        (R_hist_ukf, "Adaptive-UKF"),
    ]):
        ax = axes[ax_idx]
        style_ax(ax)
        t_updates = np.linspace(0, 40, len(R_hist))

        for j, (lbl, col) in enumerate(zip(labels, colors_r)):
            ax.plot(t_updates, R_hist[:, j], color=col, linewidth=1.2,
                    alpha=0.85, label=lbl)
            ax.axhline(true_R[j], color=col, linestyle="--", linewidth=0.8, alpha=0.4)

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Diagonal R value")
        ax.legend(fontsize=8, facecolor="#0e1a2a", edgecolor="#1a3a5a",
                  labelcolor="white")

    plt.tight_layout()
    path = os.path.join(out_dir, "adaptive_R_trace.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────── Plot 4: accuracy vs noise ───────────────────────

def plot_accuracy_vs_noise(results: list[dict], out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(BG)
    style_ax(ax)
    ax.set_title("ATE vs Noise Regime — Averaged over All Trajectories",
                 color="white", fontsize=11)

    x_noise = [0, 1, 2]
    for est in EST_ORDER:
        ates = []
        for noise in NOISE_ORDER:
            vals = [flt(r, "ate_mean") for r in results
                    if r["noise_regime"] == noise and r["estimator"] == est]
            ates.append(np.mean(vals) if vals else float("nan"))
        ax.plot(x_noise, ates, color=COLORS[est], marker="o",
                linewidth=2.0, markersize=7, label=est)

    ax.set_xticks(x_noise)
    ax.set_xticklabels(["Low", "Medium", "High"])
    ax.set_xlabel("Noise Regime")
    ax.set_ylabel("Mean ATE (m)")
    ax.legend(facecolor="#0e1a2a", edgecolor="#1a3a5a", labelcolor="white", fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, "ate_vs_noise.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────── main ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    RESULTS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmark", "benchmark")
    CSV_PATH     = os.path.join(RESULTS_DIR, "benchmark_results.csv")
    PLOT_DIR     = RESULTS_DIR

    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found — run run_benchmark.py first")
        sys.exit(1)

    results = load_results(CSV_PATH)
    print(f"Loaded {len(results)} conditions from {CSV_PATH}")

    plot_ate_comparison(results, PLOT_DIR)
    plot_consistency(results, PLOT_DIR)
    plot_accuracy_vs_noise(results, PLOT_DIR)
    plot_adaptive_trace(PLOT_DIR)

    print("\nAll research plots generated.")
