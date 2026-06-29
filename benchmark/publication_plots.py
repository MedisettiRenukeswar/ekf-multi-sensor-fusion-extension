"""
Publication-Quality Figure Generator
=====================================
Generates all figures in IEEE publication format:
  - White background, 300 DPI, PDF+PNG output
  - Times New Roman / DejaVu Serif fonts
  - IEEE color palette
  - Proper axis labels, units, legends
  - Confidence bands on MC results

Figures generated:
  1.  Trajectory comparison (EKF vs UKF vs GT)
  2.  ATE by estimator and noise regime (grouped bar)
  3.  ANIS consistency heatmap
  4.  ANEES consistency heatmap
  5.  Dropout robustness curves
  6.  Hyperparameter sensitivity heatmap (W × α)
  7.  Runtime comparison (log scale)
  8.  MACE vs Adaptive-EKF under dropout (main novel result)
  9.  All-estimator ANIS vs noise (main consistency finding)
  10. Consistency bounds chart with chi-sq reference lines

Author: Medisetti Renukeswar
"""

from __future__ import annotations

import csv
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── IEEE style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Serif",
    "font.size":          9,
    "axes.labelsize":     9,
    "axes.titlesize":     10,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "legend.framealpha":  0.9,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "lines.linewidth":    1.4,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linewidth":     0.5,
})

# IEEE color palette
COLORS = {
    "EKF":          "#0072BD",   # blue
    "UKF":          "#D95319",   # orange
    "Adaptive-EKF": "#EDB120",   # yellow
    "Adaptive-UKF": "#7E2F8E",   # purple
    "ES-EKF":       "#77AC30",   # green
    "MACE-EKF":     "#4DBEEE",   # cyan
    "MACE-UKF":     "#A2142F",   # dark red
}
MARKERS = {"EKF":"o","UKF":"s","Adaptive-EKF":"^","Adaptive-UKF":"D",
           "ES-EKF":"v","MACE-EKF":"*","MACE-UKF":"P"}
ESTIMATORS = list(COLORS.keys())

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS    = os.path.join(REPO_ROOT, "results")
FIGURES    = os.path.join(REPO_ROOT, "paper", "figures")
os.makedirs(FIGURES, exist_ok=True)

# Consistency bounds for N=15 MC, dof=3
sys.path.insert(0, REPO_ROOT)
from ekf_core.metrics import average_nees_bounds
_LB, _UB = average_nees_bounds(dof=3, n_runs=15)


def load_stats(fname: str) -> list[dict]:
    path = os.path.join(RESULTS, fname)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (ValueError, TypeError):
                pass
    return rows


def savefig(fig: plt.Figure, name: str) -> None:
    for ext in ["pdf", "png"]:
        path = os.path.join(FIGURES, f"{name}.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=300)
    print(f"  Saved: {name}.pdf/.png")
    plt.close(fig)


# ── Figure 1: ATE by estimator × noise regime (grouped bar, figure-8) ───────
def fig_ate_bar(stats: list[dict]) -> None:
    noise_order = ["low", "medium", "high"]
    traj = "figure8"
    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    x = np.arange(len(noise_order))
    w = 0.10
    n_est = len(ESTIMATORS)
    offsets = np.linspace(-(n_est-1)/2*w, (n_est-1)/2*w, n_est)

    for i, name in enumerate(ESTIMATORS):
        ate_means, ate_errs = [], []
        for noise in noise_order:
            row = next((r for r in stats if r.get("estimator")==name
                        and r.get("trajectory")==traj
                        and r.get("noise_regime")==noise), None)
            ate_means.append(row["ate_mean"] if row else 0.0)
            ate_errs.append(row["ate_ci"]    if row else 0.0)
        ax.bar(x + offsets[i], ate_means, w*0.9,
               yerr=ate_errs, color=COLORS[name], label=name,
               capsize=2, error_kw={"linewidth":0.8})

    ax.set_xticks(x)
    ax.set_xticklabels(["Low noise", "Medium noise", "High noise"])
    ax.set_ylabel("ATE RMSE (m)")
    ax.set_title("(a) Accuracy: ATE by Estimator and Noise Level\n(figure-8, N=15 MC, 95% CI)")
    ax.legend(loc="upper left", ncol=2, fontsize=6.5)
    fig.tight_layout()
    savefig(fig, "fig1_ate_bar")


# ── Figure 2: ANIS consistency heatmap ───────────────────────────────────────
def fig_anis_heatmap(stats: list[dict]) -> None:
    noise_order = ["low", "medium", "high"]
    traj = "figure8"
    matrix = np.zeros((len(ESTIMATORS), len(noise_order)))
    for i, name in enumerate(ESTIMATORS):
        for j, noise in enumerate(noise_order):
            row = next((r for r in stats if r.get("estimator")==name
                        and r.get("trajectory")==traj
                        and r.get("noise_regime")==noise), None)
            matrix[i, j] = row["anis_mean"] if row else float("nan")

    fig, ax = plt.subplots(figsize=(3.0, 2.8))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.0, vmax=1.5, aspect="auto")
    plt.colorbar(im, ax=ax, label="ANIS")
    # Overlay chi-sq bounds as text
    ax.axhline(-0.5, color="gray", lw=0.5, ls="--")
    for i in range(len(ESTIMATORS)):
        for j in range(len(noise_order)):
            v = matrix[i, j]
            color = "black" if 0.3 < v < 1.3 else "white"
            ok = "OK" if _LB <= v <= _UB else "--"
            ax.text(j, i, f"{v:.2f}\n{ok}", ha="center", va="center",
                    fontsize=6.5, color=color, weight="bold")
    ax.set_xticks(range(len(noise_order)))
    ax.set_xticklabels(["Low", "Medium", "High"])
    ax.set_yticks(range(len(ESTIMATORS)))
    ax.set_yticklabels(ESTIMATORS, fontsize=7)
    ax.set_title(f"(b) ANIS Consistency (95% bounds: [{_LB:.2f}, {_UB:.2f}])\n✓=consistent, ✗=inconsistent")
    fig.tight_layout()
    savefig(fig, "fig2_anis_heatmap")


# ── Figure 3: ANEES consistency heatmap ──────────────────────────────────────
def fig_anees_heatmap(stats: list[dict]) -> None:
    noise_order = ["low", "medium", "high"]
    traj = "figure8"
    matrix = np.zeros((len(ESTIMATORS), len(noise_order)))
    for i, name in enumerate(ESTIMATORS):
        for j, noise in enumerate(noise_order):
            row = next((r for r in stats if r.get("estimator")==name
                        and r.get("trajectory")==traj
                        and r.get("noise_regime")==noise), None)
            matrix[i, j] = row["anees_mean"] if row else float("nan")

    fig, ax = plt.subplots(figsize=(3.0, 2.8))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.0, vmax=1.5, aspect="auto")
    plt.colorbar(im, ax=ax, label="ANEES")
    for i in range(len(ESTIMATORS)):
        for j in range(len(noise_order)):
            v = matrix[i, j]
            color = "black" if 0.3 < v < 1.3 else "white"
            ok = "OK" if _LB <= v <= _UB else "--"
            ax.text(j, i, f"{v:.2f}\n{ok}", ha="center", va="center",
                    fontsize=6.5, color=color, weight="bold")
    ax.set_xticks(range(len(noise_order)))
    ax.set_xticklabels(["Low", "Medium", "High"])
    ax.set_yticks(range(len(ESTIMATORS)))
    ax.set_yticklabels(ESTIMATORS, fontsize=7)
    ax.set_title(f"(c) ANEES Consistency (95% bounds: [{_LB:.2f}, {_UB:.2f}])\n✓=consistent, ✗=inconsistent")
    fig.tight_layout()
    savefig(fig, "fig3_anees_heatmap")


# ── Figure 4: ANIS vs noise (line plot, all estimators) ─────────────────────
def fig_anis_lines(stats: list[dict]) -> None:
    noise_order = ["low", "medium", "high"]
    noise_x = [0, 1, 2]
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.5), sharey=True)
    trajs = ["figure8", "circle", "straight"]
    traj_labels = ["(d) Figure-8", "(e) Circle", "(f) Straight-line"]

    for ax, traj, tlabel in zip(axes, trajs, traj_labels):
        for name in ESTIMATORS:
            anis_vals = []
            for noise in noise_order:
                row = next((r for r in stats if r.get("estimator")==name
                            and r.get("trajectory")==traj
                            and r.get("noise_regime")==noise), None)
                anis_vals.append(row["anis_mean"] if row else float("nan"))
            ax.plot(noise_x, anis_vals, marker=MARKERS[name],
                    color=COLORS[name], label=name, markersize=4)

        ax.axhline(_LB, ls="--", lw=0.8, color="gray", label=f"χ² bounds [{_LB:.2f},{_UB:.2f}]")
        ax.axhline(_UB, ls="--", lw=0.8, color="gray")
        ax.fill_between(noise_x, _LB, _UB, alpha=0.08, color="green")
        ax.set_xticks(noise_x)
        ax.set_xticklabels(["Low", "Med", "High"])
        ax.set_title(tlabel)
        ax.set_ylim(-0.1, 1.6)

    axes[0].set_ylabel("ANIS")
    axes[1].set_xlabel("Noise regime")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7,
               bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("ANIS Consistency vs Noise Regime (green band = 95% chi-sq bounds)", fontsize=9)
    fig.tight_layout()
    savefig(fig, "fig4_anis_lines")


# ── Figure 5: Dropout robustness ─────────────────────────────────────────────
def fig_dropout(stats: list[dict]) -> None:
    dropout_x = [0.0, 0.10, 0.30, 0.50, 0.70]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.5))

    for name in ESTIMATORS:
        ate_vals, anis_vals = [], []
        for dr in dropout_x:
            row = next((r for r in stats if r.get("estimator")==name
                        and abs(float(r.get("dropout",0)) - dr) < 0.01), None)
            ate_vals.append(row["ate_mean"] if row else float("nan"))
            anis_vals.append(row["anis_mean"] if row else float("nan"))
        ax1.plot([d*100 for d in dropout_x], ate_vals,
                 marker=MARKERS[name], color=COLORS[name], label=name, markersize=4)
        ax2.plot([d*100 for d in dropout_x], anis_vals,
                 marker=MARKERS[name], color=COLORS[name], label=name, markersize=4)

    ax2.axhline(_LB, ls="--", lw=0.8, color="gray")
    ax2.axhline(_UB, ls="--", lw=0.8, color="gray")
    ax2.fill_between([d*100 for d in dropout_x], _LB, _UB, alpha=0.08, color="green")

    ax1.set_xlabel("VO Dropout Rate (%)")
    ax1.set_ylabel("ATE RMSE (m)")
    ax1.set_title("(g) ATE vs VO Dropout")
    ax2.set_xlabel("VO Dropout Rate (%)")
    ax2.set_ylabel("ANIS")
    ax2.set_title("(h) Consistency vs VO Dropout\n(green band = consistent region)")
    ax2.set_ylim(-0.1, 2.0)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7,
               bbox_to_anchor=(0.5, -0.10))
    fig.tight_layout()
    savefig(fig, "fig5_dropout")


# ── Figure 6: MACE vs Adaptive-EKF under dropout (novel result) ──────────────
def fig_mace_vs_adaptive(stats: list[dict]) -> None:
    dropout_x = [0.0, 0.10, 0.30, 0.50, 0.70]
    target = ["EKF", "Adaptive-EKF", "MACE-EKF", "MACE-UKF"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.5))

    for name in target:
        anis_vals, ate_vals = [], []
        for dr in dropout_x:
            row = next((r for r in stats if r.get("estimator")==name
                        and abs(float(r.get("dropout",0)) - dr) < 0.01), None)
            anis_vals.append(row["anis_mean"] if row else float("nan"))
            ate_vals.append(row["ate_mean"] if row else float("nan"))
        ax1.plot([d*100 for d in dropout_x], anis_vals,
                 marker=MARKERS[name], color=COLORS[name], label=name,
                 markersize=5, linewidth=1.8)
        ax2.plot([d*100 for d in dropout_x], ate_vals,
                 marker=MARKERS[name], color=COLORS[name], label=name,
                 markersize=5, linewidth=1.8)

    ax1.axhline(_LB, ls="--", lw=1.0, color="gray", label=f"χ² bounds")
    ax1.axhline(_UB, ls="--", lw=1.0, color="gray")
    ax1.fill_between([0,10,30,50,70], _LB, _UB, alpha=0.10, color="green")
    ax1.set_xlabel("VO Dropout Rate (%)")
    ax1.set_ylabel("ANIS")
    ax1.set_title("(i) MACE-χ² vs Adaptive-EKF:\nConsistency Under Dropout")
    ax1.set_ylim(-0.1, 2.0)
    ax1.legend(fontsize=7)

    ax2.set_xlabel("VO Dropout Rate (%)")
    ax2.set_ylabel("ATE RMSE (m)")
    ax2.set_title("(j) MACE-χ² vs Adaptive-EKF:\nAccuracy Under Dropout")
    ax2.legend(fontsize=7)
    fig.tight_layout()
    savefig(fig, "fig6_mace_novel")


# ── Figure 7: Hyperparameter sensitivity heatmap ─────────────────────────────
def fig_hyperparam(hp_rows: list[dict]) -> None:
    if not hp_rows:
        return
    W_vals = sorted(set(int(r["W"]) for r in hp_rows))
    A_vals = sorted(set(float(r["alpha"]) for r in hp_rows))

    ate_mat  = np.full((len(W_vals), len(A_vals)), float("nan"))
    anis_mat = np.full((len(W_vals), len(A_vals)), float("nan"))
    for r in hp_rows:
        i = W_vals.index(int(r["W"]))
        j = A_vals.index(float(r["alpha"]))
        ate_mat[i, j]  = float(r["ate_mean"])
        anis_mat[i, j] = float(r["anis_mean"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.5))
    for ax, mat, label, title in [
        (ax1, ate_mat, "ATE (m)", "(k) ATE Heatmap"),
        (ax2, anis_mat, "ANIS",  "(l) ANIS Heatmap"),
    ]:
        vmin = np.nanmin(mat); vmax = np.nanmax(mat)
        im = ax.imshow(mat, cmap="YlOrRd" if label=="ATE (m)" else "RdYlGn",
                       vmin=vmin, vmax=vmax, aspect="auto")
        plt.colorbar(im, ax=ax, label=label, fraction=0.04, pad=0.04)
        for i in range(len(W_vals)):
            for j in range(len(A_vals)):
                v = mat[i,j]
                ok = ""
                if label == "ANIS":
                    ok = " ✓" if _LB <= v <= _UB else " ✗"
                ax.text(j, i, f"{v:.3f}{ok}", ha="center", va="center", fontsize=7)
        ax.set_xticks(range(len(A_vals)))
        ax.set_xticklabels([f"α={a}" for a in A_vals])
        ax.set_yticks(range(len(W_vals)))
        ax.set_yticklabels([f"W={w}" for w in W_vals])
        ax.set_title(title)

    fig.suptitle("Adaptive-EKF Hyperparameter Sensitivity\n(figure-8, medium noise, N=15 MC)", fontsize=9)
    fig.tight_layout()
    savefig(fig, "fig7_hyperparam")


# ── Figure 8: Runtime comparison (log scale) ─────────────────────────────────
def fig_runtime(stats: list[dict]) -> None:
    noise = "medium"; traj = "figure8"
    names, rts, stds = [], [], []
    for name in ESTIMATORS:
        row = next((r for r in stats if r.get("estimator")==name
                    and r.get("trajectory")==traj
                    and r.get("noise_regime")==noise), None)
        if row:
            names.append(name)
            rts.append(row["rt_mean"])
            stds.append(row["rt_std"])

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    y_pos = range(len(names))
    bars = ax.barh(list(y_pos), rts, xerr=stds, color=[COLORS[n] for n in names],
                   capsize=3, error_kw={"linewidth":0.8}, height=0.65)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Runtime per 40 s trajectory (ms)")
    ax.set_xscale("log")
    ax.set_title("(m) Runtime Comparison\n(figure-8, medium noise, N=15 MC, log scale)")
    ax.axvline(1000, ls="--", lw=0.8, color="red", alpha=0.5, label="1 s (real-time margin)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    savefig(fig, "fig8_runtime")


# ── Figure 9: Trajectory visualisation ───────────────────────────────────────
def fig_trajectory() -> None:
    sys.path.insert(0, REPO_ROOT)
    from simulation.trajectories import TrajectoryGenerator
    from simulation.research_sensor_sim import ResearchSensorSimulator
    from ekf_core.ekf_estimator import EKFEstimator
    from ekf_core.ukf_estimator import UKFEstimator
    from ekf_core.mace_estimator import MACEEKFEstimator

    DT_IMU = 0.01; DT_CAM = 1/30; SIM_DURATION = 40.0

    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.8))
    trajs_types = ["figure8", "circle", "straight"]
    traj_titles = ["(n) Figure-8 Trajectory", "(o) Circle Trajectory", "(p) Straight-line Trajectory"]

    for ax, tt, title in zip(axes, trajs_types, traj_titles):
        traj = TrajectoryGenerator(trajectory_type=tt, scale=3.0)
        sim  = ResearchSensorSimulator(trajectory=traj, noise_regime="medium",
                                       dt_imu=DT_IMU, dt_cam=DT_CAM, seed=42)
        gt_x, gt_y = [], []
        est_data = {n: ([], []) for n in ["EKF", "UKF", "MACE-EKF"]}
        ests = {"EKF": EKFEstimator(), "UKF": UKFEstimator(), "MACE-EKF": MACEEKFEstimator()}
        px0, py0, th0, vx0, vy0, om0 = traj.get_state(0.0)
        x0 = np.array([px0, py0, th0, vx0, vy0, om0])
        P0 = np.diag([0.5,0.5,0.3,0.5,0.5,0.1])
        for e in ests.values(): e.reset(x0, P0)
        t = 0.0; cam_t = 0.0; step = 0
        while t <= SIM_DURATION:
            px_gt, py_gt, *_ = traj.get_state(t)
            vxm, vym, omm = sim.get_imu(t)
            for e in ests.values(): e.predict(vxm, vym, omm)
            cam_t += DT_IMU
            if cam_t >= DT_CAM:
                cam_t = 0.0
                pxc, pyc, thc = sim.get_camera(t)
                for e in ests.values(): e.update_camera(pxc, pyc, thc)
            if step % 10 == 0:
                gt_x.append(px_gt); gt_y.append(py_gt)
                for n, e in ests.items():
                    xs, ys = est_data[n]
                    xe, ye, _ = e.get_position()
                    xs.append(xe); ys.append(ye)
            t += DT_IMU; step += 1

        ax.plot(gt_x, gt_y, "k-", lw=1.5, label="Ground Truth", zorder=5)
        for n in ["EKF", "UKF", "MACE-EKF"]:
            xs, ys = est_data[n]
            ax.plot(xs, ys, "-", color=COLORS[n], lw=1.0, label=n, alpha=0.85)
        ax.set_aspect("equal"); ax.set_title(title, fontsize=9)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    savefig(fig, "fig9_trajectories")


# ── Figure 10: Summary bar — key findings ────────────────────────────────────
def fig_summary_bar(std_stats: list[dict], drop_stats: list[dict]) -> None:
    """Four-panel summary of key paper findings."""
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.0))

    # Panel 1: ATE figure-8 medium
    ax = axes[0, 0]
    names_short = ["EKF","UKF","A-EKF","A-UKF","ES-EKF","MACE-EKF","MACE-UKF"]
    short_map = dict(zip(ESTIMATORS, names_short))
    ates, errs = [], []
    for name in ESTIMATORS:
        row = next((r for r in std_stats if r.get("estimator")==name
                    and r.get("trajectory")=="figure8"
                    and r.get("noise_regime")=="medium"), None)
        ates.append(row["ate_mean"] if row else 0)
        errs.append(row["ate_ci"]   if row else 0)
    bars = ax.bar(range(len(ESTIMATORS)), ates, yerr=errs, color=[COLORS[n] for n in ESTIMATORS],
                  capsize=3, error_kw={"linewidth":0.8})
    ax.set_xticks(range(len(ESTIMATORS)))
    ax.set_xticklabels(names_short, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("ATE RMSE (m)")
    ax.set_title("Finding 1: Accuracy (figure-8, medium noise)")

    # Panel 2: ANIS medium
    ax = axes[0, 1]
    aniss, a_errs = [], []
    for name in ESTIMATORS:
        row = next((r for r in std_stats if r.get("estimator")==name
                    and r.get("trajectory")=="figure8"
                    and r.get("noise_regime")=="medium"), None)
        aniss.append(row["anis_mean"] if row else 0)
        a_errs.append(row["anis_std"] if row else 0)
    ax.bar(range(len(ESTIMATORS)), aniss, yerr=a_errs, color=[COLORS[n] for n in ESTIMATORS],
           capsize=3, error_kw={"linewidth":0.8})
    ax.axhline(_LB, ls="--", lw=1.0, color="green", label=f"Lower bound {_LB:.2f}")
    ax.axhline(_UB, ls="--", lw=1.0, color="green", label=f"Upper bound {_UB:.2f}")
    ax.fill_between([-0.5, len(ESTIMATORS)-0.5], _LB, _UB, alpha=0.10, color="green")
    ax.set_xticks(range(len(ESTIMATORS)))
    ax.set_xticklabels(names_short, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("ANIS")
    ax.set_title("Finding 2: Consistency (figure-8, medium noise)")
    ax.legend(fontsize=7)
    ax.set_ylim(0, 1.6)

    # Panel 3: ATE at 50% dropout
    ax = axes[1, 0]
    ates50, errs50 = [], []
    for name in ESTIMATORS:
        row = next((r for r in drop_stats if r.get("estimator")==name
                    and abs(float(r.get("dropout",0)) - 0.50) < 0.01), None)
        ates50.append(row["ate_mean"] if row else 0)
        errs50.append(row["ate_ci"]   if row else 0)
    ax.bar(range(len(ESTIMATORS)), ates50, yerr=errs50, color=[COLORS[n] for n in ESTIMATORS],
           capsize=3, error_kw={"linewidth":0.8})
    ax.set_xticks(range(len(ESTIMATORS)))
    ax.set_xticklabels(names_short, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("ATE RMSE (m)")
    ax.set_title("Finding 3: Robustness at 50% VO Dropout")

    # Panel 4: ANIS at 50% dropout
    ax = axes[1, 1]
    aniss50, ae50 = [], []
    for name in ESTIMATORS:
        row = next((r for r in drop_stats if r.get("estimator")==name
                    and abs(float(r.get("dropout",0)) - 0.50) < 0.01), None)
        aniss50.append(row["anis_mean"] if row else 0)
        ae50.append(row["anis_std"]     if row else 0)
    ax.bar(range(len(ESTIMATORS)), aniss50, yerr=ae50, color=[COLORS[n] for n in ESTIMATORS],
           capsize=3, error_kw={"linewidth":0.8})
    ax.axhline(_LB, ls="--", lw=1.0, color="green")
    ax.axhline(_UB, ls="--", lw=1.0, color="green")
    ax.fill_between([-0.5, len(ESTIMATORS)-0.5], _LB, _UB, alpha=0.10, color="green")
    ax.set_xticks(range(len(ESTIMATORS)))
    ax.set_xticklabels(names_short, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("ANIS")
    ax.set_title("Finding 4: Consistency at 50% VO Dropout")
    ax.set_ylim(0, 2.5)

    fig.suptitle("Key Research Findings Summary (N=15 MC, figure-8, medium noise unless stated)", fontsize=9)
    fig.tight_layout()
    savefig(fig, "fig10_summary")


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    print("Generating publication-quality figures...")
    std_stats  = load_stats("full_stats_standard.csv")
    drop_stats = load_stats("full_stats_dropout.csv")
    hp_rows    = load_stats("hyperparam_sensitivity.csv")

    print(f"  Standard conditions: {len(std_stats)}")
    print(f"  Dropout conditions:  {len(drop_stats)}")
    print(f"  Hyperparam cells:    {len(hp_rows)}")

    fig_ate_bar(std_stats)
    fig_anis_heatmap(std_stats)
    fig_anees_heatmap(std_stats)
    fig_anis_lines(std_stats)
    fig_dropout(drop_stats)
    fig_mace_vs_adaptive(drop_stats)
    fig_hyperparam(hp_rows)
    fig_runtime(std_stats)
    fig_trajectory()
    fig_summary_bar(std_stats, drop_stats)

    print(f"\nAll figures saved to: {FIGURES}")


if __name__ == "__main__":
    main()
