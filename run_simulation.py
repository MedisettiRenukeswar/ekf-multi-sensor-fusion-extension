"""
Multi-Sensor Fusion EKF — Main Simulation
==========================================
Runs the full EKF fusion loop:
  - IMU prediction at 100 Hz
  - Camera (VO) update at 30 Hz
  - Pure IMU dead reckoning for comparison
  - Computes ATE, RPE metrics
  - Saves result plots

Usage:
    python run_simulation.py

Author : Medisetti Renukeswar
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
from ekf_core.ekf import EKF2DRobot
from simulation.sensor_sim import SensorSimulator


# ── Simulation Parameters ────────────────────────────────────────────────────
SIM_DURATION   = 40.0    # seconds
DT_IMU         = 0.01    # 100 Hz
DT_CAM         = 1/30    # ~30 Hz
TRAJ_SCALE     = 3.0     # figure-8 scale (meters)

# ── Output ───────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_simulation():
    print("=" * 60)
    print("  EKF Multi-Sensor Fusion — IMU + Camera")
    print("  Medisetti Renukeswar")
    print("=" * 60)

    sim  = SensorSimulator(dt_imu=DT_IMU, dt_cam=DT_CAM)
    ekf  = EKF2DRobot(dt=DT_IMU)

    # Initialise EKF at ground truth starting position
    px0, py0, th0, vx0, _, om0 = sim.figure8_trajectory(0.0)
    ekf.x = np.array([px0, py0, th0, vx0, 0.0, om0])

    # ── Storage ──────────────────────────────────────────────────────────────
    t_log     = []
    gt_x, gt_y, gt_th = [], [], []
    ekf_x, ekf_y, ekf_th = [], [], []
    imu_x, imu_y = [], []        # pure IMU dead reckoning (no fusion)
    sigma_x, sigma_y = [], []    # 1-sigma uncertainty bounds

    # Pure IMU dead reckoning state
    dr_px, dr_py, dr_th = px0, py0, th0
    dr_vx, dr_vy = vx0, 0.0

    t = 0.0
    cam_timer = 0.0
    step = 0

    print(f"\nRunning {SIM_DURATION}s simulation...")

    while t <= SIM_DURATION:
        # ── Ground Truth ─────────────────────────────────────────────────────
        px_gt, py_gt, th_gt, vx_gt, vy_gt, om_gt = sim.figure8_trajectory(t)

        # ── IMU Measurement ───────────────────────────────────────────────────
        ax, ay, om = sim.get_imu(t)

        # ── EKF Prediction ────────────────────────────────────────────────────
        ekf.predict(ax, ay, om)

        # ── Camera Update (at ~30 Hz) ─────────────────────────────────────────
        cam_timer += DT_IMU
        if cam_timer >= DT_CAM:
            cam_timer = 0.0
            px_c, py_c, th_c = sim.get_camera(t)
            ekf.update_camera(px_c, py_c, th_c)

        # ── Pure IMU Dead Reckoning ───────────────────────────────────────────
        cos_dr = math.cos(dr_th)
        sin_dr = math.sin(dr_th)
        dr_px += (dr_vx * cos_dr - dr_vy * sin_dr) * DT_IMU
        dr_py += (dr_vx * sin_dr + dr_vy * cos_dr) * DT_IMU
        dr_th += om * DT_IMU
        dr_vx += ax * DT_IMU
        dr_vy += ay * DT_IMU

        # ── Log ──────────────────────────────────────────────────────────────
        if step % 10 == 0:   # log every 10 IMU steps = 10 Hz
            ekf_state, P = ekf.get_state()
            t_log.append(t)
            gt_x.append(px_gt);  gt_y.append(py_gt);  gt_th.append(th_gt)
            ekf_x.append(np.asarray(ekf_state[0]).flat[0])
            ekf_y.append(np.asarray(ekf_state[1]).flat[0])
            ekf_th.append(np.asarray(ekf_state[2]).flat[0])
            imu_x.append(dr_px);  imu_y.append(dr_py)
            sigma_x.append(float(np.sqrt(np.asarray(P[0,0]).flat[0])))
            sigma_y.append(float(np.sqrt(np.asarray(P[1,1]).flat[0])))

        t += DT_IMU
        step += 1

    print(f"  Steps: {step}  |  Log points: {len(t_log)}")

    # ── Convert to numpy ─────────────────────────────────────────────────────
    gt_x   = np.array(gt_x);   gt_y   = np.array(gt_y)
    ekf_x  = np.array(ekf_x);  ekf_y  = np.array(ekf_y)
    imu_x  = np.array(imu_x);  imu_y  = np.array(imu_y)
    sigma_x = np.array(sigma_x); sigma_y = np.array(sigma_y)
    t_arr  = np.array(t_log)

    # ── Metrics ──────────────────────────────────────────────────────────────
    ate_ekf = compute_ate(gt_x, gt_y, ekf_x, ekf_y)
    ate_imu = compute_ate(gt_x, gt_y, imu_x, imu_y)
    rpe_ekf = compute_rpe(gt_x, gt_y, ekf_x, ekf_y)

    print(f"\n{'─'*50}")
    print(f"  RESULTS")
    print(f"{'─'*50}")
    print(f"  ATE  (EKF fusion)       : {ate_ekf:.4f} m  RMSE")
    print(f"  ATE  (IMU only)         : {ate_imu:.4f} m  RMSE")
    print(f"  RPE  (EKF, per step)    : {rpe_ekf:.4f} m  RMSE")
    print(f"  Improvement vs IMU-only : {(1 - ate_ekf/ate_imu)*100:.1f}%")
    print(f"{'─'*50}\n")

    # ── Plot 1: Trajectory ────────────────────────────────────────────────────
    plot_trajectory(gt_x, gt_y, ekf_x, ekf_y, imu_x, imu_y,
                    ate_ekf, ate_imu, RESULTS_DIR)

    # ── Plot 2: Position Error Over Time ─────────────────────────────────────
    plot_error(t_arr, gt_x, gt_y, ekf_x, ekf_y, imu_x, imu_y,
               sigma_x, sigma_y, RESULTS_DIR)

    # ── Plot 3: Kalman Gain & Uncertainty ─────────────────────────────────────
    plot_uncertainty(t_arr, sigma_x, sigma_y, RESULTS_DIR)

    # ── Save metrics ──────────────────────────────────────────────────────────
    with open(os.path.join(RESULTS_DIR, 'metrics.txt'), 'w') as f:
        f.write("EKF Multi-Sensor Fusion — Results\n")
        f.write("Medisetti Renukeswar\n")
        f.write("=" * 40 + "\n")
        f.write(f"Simulation duration  : {SIM_DURATION} s\n")
        f.write(f"IMU rate             : {1/DT_IMU:.0f} Hz\n")
        f.write(f"Camera rate          : {1/DT_CAM:.0f} Hz\n")
        f.write(f"Trajectory           : Figure-8, scale {TRAJ_SCALE} m\n")
        f.write("=" * 40 + "\n")
        f.write(f"ATE EKF fusion       : {ate_ekf:.4f} m\n")
        f.write(f"ATE IMU-only         : {ate_imu:.4f} m\n")
        f.write(f"RPE EKF              : {rpe_ekf:.4f} m\n")
        f.write(f"Improvement          : {(1 - ate_ekf/ate_imu)*100:.1f}%\n")

    print("  Plots saved to results/")
    print("  Done.\n")
    return ate_ekf, ate_imu, rpe_ekf


# ── Metrics ──────────────────────────────────────────────────────────────────
def compute_ate(gt_x, gt_y, est_x, est_y):
    """Absolute Trajectory Error — RMSE of position errors."""
    err = np.sqrt((gt_x - est_x)**2 + (gt_y - est_y)**2)
    return float(np.sqrt(np.mean(err**2)))


def compute_rpe(gt_x, gt_y, est_x, est_y, step=10):
    """Relative Pose Error — RMSE of relative motion errors."""
    n = len(gt_x)
    errors = []
    for i in range(0, n - step, step):
        dgt  = math.sqrt((gt_x[i+step]-gt_x[i])**2  + (gt_y[i+step]-gt_y[i])**2)
        dest = math.sqrt((est_x[i+step]-est_x[i])**2 + (est_y[i+step]-est_y[i])**2)
        errors.append((dgt - dest)**2)
    return float(np.sqrt(np.mean(errors))) if errors else 0.0


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_trajectory(gt_x, gt_y, ekf_x, ekf_y, imu_x, imu_y,
                    ate_ekf, ate_imu, out_dir):
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#0a0f1a')
    ax.set_facecolor('#0a0f1a')

    ax.plot(gt_x,  gt_y,  color='#00ff88', linewidth=2.5,
            label='Ground Truth', zorder=3)
    ax.plot(ekf_x, ekf_y, color='#00e5ff', linewidth=2.0, linestyle='-',
            label=f'EKF Fusion  (ATE={ate_ekf:.3f} m)', zorder=4, alpha=0.9)
    ax.plot(imu_x, imu_y, color='#ff6b35', linewidth=1.2, linestyle='--',
            label=f'IMU Only    (ATE={ate_imu:.3f} m)', zorder=2, alpha=0.7)

    # Start/end markers
    ax.scatter([gt_x[0]],  [gt_y[0]],  s=120, color='#ffd600', zorder=5,
               marker='o', label='Start')
    ax.scatter([gt_x[-1]], [gt_y[-1]], s=120, color='#ff3b5c', zorder=5,
               marker='x', linewidths=3)

    ax.set_title('EKF Multi-Sensor Fusion — Trajectory\n'
                 'IMU + Camera (Visual Odometry)',
                 color='white', fontsize=14, pad=15)
    ax.set_xlabel('X Position (m)', color='#aaccdd')
    ax.set_ylabel('Y Position (m)', color='#aaccdd')
    ax.legend(facecolor='#0e1a2a', edgecolor='#1a3a5a', labelcolor='white',
              fontsize=10)
    ax.tick_params(colors='#aaccdd')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1a3a5a')
    ax.grid(True, alpha=0.15, color='#00e5ff')
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'trajectory.png'), dpi=150,
                bbox_inches='tight', facecolor='#0a0f1a')
    plt.close()
    print("  Saved: results/trajectory.png")


def plot_error(t_arr, gt_x, gt_y, ekf_x, ekf_y, imu_x, imu_y,
               sigma_x, sigma_y, out_dir):
    ekf_err = np.sqrt((gt_x - ekf_x)**2 + (gt_y - ekf_y)**2)
    imu_err = np.sqrt((gt_x - imu_x)**2 + (gt_y - imu_y)**2)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.patch.set_facecolor('#0a0f1a')

    # Position error
    ax = axes[0]
    ax.set_facecolor('#0a0f1a')
    ax.plot(t_arr, ekf_err, color='#00e5ff', linewidth=1.5,
            label='EKF Position Error')
    ax.plot(t_arr, imu_err, color='#ff6b35', linewidth=1.2, linestyle='--',
            alpha=0.7, label='IMU-Only Error')
    ax.fill_between(t_arr, 0, ekf_err, alpha=0.15, color='#00e5ff')
    ax.set_ylabel('Position Error (m)', color='#aaccdd')
    ax.set_title('Position Error Over Time', color='white', fontsize=12)
    ax.legend(facecolor='#0e1a2a', edgecolor='#1a3a5a', labelcolor='white')
    ax.tick_params(colors='#aaccdd')
    ax.grid(True, alpha=0.15, color='#00e5ff')
    for spine in ax.spines.values(): spine.set_edgecolor('#1a3a5a')

    # 1-sigma uncertainty
    ax2 = axes[1]
    ax2.set_facecolor('#0a0f1a')
    ax2.plot(t_arr, sigma_x, color='#b96dff', linewidth=1.5,
             label='σ_x (position uncertainty)')
    ax2.plot(t_arr, sigma_y, color='#ffd600', linewidth=1.5, linestyle='--',
             label='σ_y (position uncertainty)')
    ax2.set_xlabel('Time (s)', color='#aaccdd')
    ax2.set_ylabel('1-sigma (m)', color='#aaccdd')
    ax2.set_title('EKF Uncertainty (1σ) — Converges as measurements arrive',
                  color='white', fontsize=12)
    ax2.legend(facecolor='#0e1a2a', edgecolor='#1a3a5a', labelcolor='white')
    ax2.tick_params(colors='#aaccdd')
    ax2.grid(True, alpha=0.15, color='#00e5ff')
    for spine in ax2.spines.values(): spine.set_edgecolor('#1a3a5a')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'position_error.png'), dpi=150,
                bbox_inches='tight', facecolor='#0a0f1a')
    plt.close()
    print("  Saved: results/position_error.png")


def plot_uncertainty(t_arr, sigma_x, sigma_y, out_dir):
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#0a0f1a')
    ax.set_facecolor('#0a0f1a')

    ax.fill_between(t_arr, 0, np.array(sigma_x)*2,
                    alpha=0.25, color='#00e5ff', label='2σ_x bound')
    ax.fill_between(t_arr, 0, np.array(sigma_y)*2,
                    alpha=0.25, color='#b96dff', label='2σ_y bound')
    ax.plot(t_arr, sigma_x, color='#00e5ff', linewidth=1.5)
    ax.plot(t_arr, sigma_y, color='#b96dff', linewidth=1.5)

    ax.set_title('EKF Covariance Convergence — Filter Confidence Over Time',
                 color='white', fontsize=12)
    ax.set_xlabel('Time (s)', color='#aaccdd')
    ax.set_ylabel('1σ Position Uncertainty (m)', color='#aaccdd')
    ax.legend(facecolor='#0e1a2a', edgecolor='#1a3a5a', labelcolor='white')
    ax.tick_params(colors='#aaccdd')
    ax.grid(True, alpha=0.15, color='#00e5ff')
    for spine in ax.spines.values(): spine.set_edgecolor('#1a3a5a')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'uncertainty.png'), dpi=150,
                bbox_inches='tight', facecolor='#0a0f1a')
    plt.close()
    print("  Saved: results/uncertainty.png")


if __name__ == '__main__':
    run_simulation()
