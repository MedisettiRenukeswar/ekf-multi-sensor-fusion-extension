"""
Phase E — Research Validation Benchmark
=========================================
N=30 Monte Carlo runs, all estimators batched per condition for maximal
parallel throughput.  Incremental saves prevent data loss on timeout.

Conditions:
  Standard:     3 traj × 3 noise × 4 estimators
  Degradation:  3 scenarios × 2 traj × 1 noise × 4 estimators
    A — Time-varying IMU bias random walk
    B — VO dropout 30%
    C — VO dropout 50%

Statistical tests:
  Wilcoxon signed-rank (paired by seed) for all 4 pairwise comparisons
  Cohen's d effect size

Author: Medisetti Renukeswar (Phase E)
"""

from __future__ import annotations

import concurrent.futures
import csv
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Literal

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from simulation.trajectories import TrajectoryGenerator, TrajectoryType
from simulation.research_sensor_sim import (
    ResearchSensorSimulator, NoiseRegime,
    DegradedSensorSimulator, DegradationConfig,
)
from ekf_core.ekf_estimator import EKFEstimator
from ekf_core.ukf_estimator import UKFEstimator
from ekf_core.adaptive_estimator import AdaptiveEKFEstimator, AdaptiveUKFEstimator
from ekf_core.metrics import (
    compute_ate, compute_rpe, compute_rmse_position, compute_rmse_heading,
    compute_nees, average_nees_bounds, monte_carlo_statistics,
    wilcoxon_test, cohens_d,
)

# ── config ───────────────────────────────────────────────────────────────────

N_MC         = 30
SIM_DURATION = 40.0
DT_IMU       = 0.01
DT_CAM       = 1 / 30
TRAJ_SCALE   = 3.0
LOG_EVERY    = 10
N_WORKERS    = 4
BASE_SEED    = 2000      # independent from Phase D seeds

TRAJECTORIES  = ["figure8", "circle", "straight"]
NOISE_REGIMES = ["low", "medium", "high"]
ESTIMATORS    = ["EKF", "UKF", "Adaptive-EKF", "Adaptive-UKF"]

COMPARISONS = [
    ("EKF",          "UKF",           "EKF_vs_UKF"),
    ("UKF",          "Adaptive-UKF",  "UKF_vs_AdaptiveUKF"),
    ("EKF",          "Adaptive-EKF",  "EKF_vs_AdaptiveEKF"),
    ("Adaptive-EKF", "Adaptive-UKF",  "AdaptiveEKF_vs_AdaptiveUKF"),
]

DEG_TRAJS = ["figure8", "circle"]
DEG_NOISE = "medium"
DEG_CONFIGS = {
    "bias_rw":   DegradationConfig(enable_bias_random_walk=True,  bias_rw_std=0.002),
    "vo_drop30": DegradationConfig(enable_vo_dropout=True, vo_dropout_prob=0.30),
    "vo_drop50": DegradationConfig(enable_vo_dropout=True, vo_dropout_prob=0.50),
}

OUT_DIR = os.path.join(_HERE, "phase_e")
os.makedirs(OUT_DIR, exist_ok=True)

seeds = [BASE_SEED + i * 137 for i in range(N_MC)]


# ── estimator factory ────────────────────────────────────────────────────────

def _make(name: str):
    if name == "EKF":          return EKFEstimator(dt=DT_IMU)
    if name == "UKF":          return UKFEstimator(dt=DT_IMU)
    if name == "Adaptive-EKF": return AdaptiveEKFEstimator(dt=DT_IMU, window=20,
                                         adapt_R=True, adapt_Q=False, alpha_smooth=0.1)
    if name == "Adaptive-UKF": return AdaptiveUKFEstimator(dt=DT_IMU, window=20,
                                         adapt_R=True, adapt_Q=False, alpha_smooth=0.1)
    raise ValueError(name)


# ── single simulation ─────────────────────────────────────────────────────────

def _simulate(
    est_name: str,
    traj_type: str,
    noise_regime: str,
    seed: int,
    degradation=None,
) -> dict:
    traj = TrajectoryGenerator(traj_type, duration=SIM_DURATION, scale=TRAJ_SCALE)
    if degradation is not None:
        sim = DegradedSensorSimulator(traj, noise_regime,
                                      degradation=degradation,
                                      dt_imu=DT_IMU, dt_cam=DT_CAM, seed=seed)
    else:
        sim = ResearchSensorSimulator(traj, noise_regime,
                                      dt_imu=DT_IMU, dt_cam=DT_CAM, seed=seed)
    est = _make(est_name)
    px0, py0, th0, vx0, vy0, om0 = traj.get_state(0.0)
    est.reset(np.array([px0, py0, th0, vx0, vy0, om0]),
              np.diag([0.5, 0.5, 0.3, 0.5, 0.5, 0.1]))

    gx, gy, gth, ex, ey, eth = [], [], [], [], [], []
    nis_buf, nees_buf = [], []
    t = 0.0; cam_t = 0.0; step = 0
    t0w = time.perf_counter()

    while t <= SIM_DURATION:
        px_gt, py_gt, th_gt, vx_gt, vy_gt, om_gt = traj.get_state(t)
        vxi, vyi, omi = sim.get_imu(t)
        est.predict(vxi, vyi, omi)

        cam_t += DT_IMU
        if cam_t >= DT_CAM:
            cam_t = 0.0
            ok = (sim.camera_available()
                  if isinstance(sim, DegradedSensorSimulator) else True)
            if ok:
                px_c, py_c, th_c = sim.get_camera(t)
                res = est.update_camera(px_c, py_c, th_c)
                nis_buf.append(res["nis"])

        if step % LOG_EVERY == 0:
            xe, Pe = est.get_state()
            x_gt = np.array([px_gt, py_gt, th_gt, vx_gt, vy_gt, om_gt])
            gx.append(px_gt); gy.append(py_gt); gth.append(th_gt)
            ex.append(xe[0]); ey.append(xe[1]); eth.append(xe[2])
            nees = compute_nees(x_gt, xe, Pe, state_indices=[0,1,2])
            if math.isfinite(nees): nees_buf.append(nees)

        t += DT_IMU; step += 1

    rt = (time.perf_counter() - t0w) * 1000.0
    gxa = np.array(gx); gya = np.array(gy); gtha = np.array(gth)
    exa = np.array(ex); eya = np.array(ey); etha = np.array(eth)

    return dict(
        ate=compute_ate(gxa,gya,exa,eya),
        rpe=compute_rpe(gxa,gya,exa,eya),
        rmse_pos=compute_rmse_position(gxa,gya,exa,eya),
        rmse_hdg=compute_rmse_heading(gtha, etha),
        mean_nis=float(np.mean(nis_buf))  if nis_buf  else float("nan"),
        mean_nees=float(np.mean(nees_buf)) if nees_buf else float("nan"),
        runtime_ms=rt,
    )


# ── worker wrappers (picklable) ───────────────────────────────────────────────

def _wstd(args):
    est, traj, noise, seed = args
    return _simulate(est, traj, noise, seed, None)

def _wdeg(args):
    est, traj, noise, seed, cfg = args
    return _simulate(est, traj, noise, seed, cfg)


# ── parallel batch per condition ──────────────────────────────────────────────

def run_condition(
    scenario: str,
    traj: str,
    noise: str,
    degradation=None,
    tag: str = "",
) -> dict[str, list[dict]]:
    """
    Submit all estimator×seed jobs for one condition into a single pool.
    Returns {estimator_name: [list of run dicts]}.
    """
    jobs = []
    for est in ESTIMATORS:
        for s in seeds:
            if degradation is not None:
                jobs.append((est, traj, noise, s, degradation))
            else:
                jobs.append((est, traj, noise, s))

    worker = _wdeg if degradation is not None else _wstd

    with concurrent.futures.ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        all_runs = list(ex.map(worker, jobs))

    # Un-flatten: jobs are grouped [est0*N_MC, est1*N_MC, ...]
    result: dict[str, list[dict]] = {e: [] for e in ESTIMATORS}
    for i, est in enumerate(ESTIMATORS):
        result[est] = all_runs[i*N_MC : (i+1)*N_MC]

    mean_ates = {e: np.mean([r["ate"] for r in result[e]]) for e in ESTIMATORS}
    print(f"  {tag:<38} " +
          "  ".join(f"{e[:4]}={mean_ates[e]:.4f}" for e in ESTIMATORS))
    return result


# ── aggregation ───────────────────────────────────────────────────────────────

def aggregate(
    runs: list[dict],
    scenario: str, estimator: str, traj: str, noise: str,
) -> dict:
    n   = len(runs)
    ate = np.array([r["ate"]       for r in runs])
    rpe = np.array([r["rpe"]       for r in runs])
    rp  = np.array([r["rmse_pos"]  for r in runs])
    rh  = np.array([r["rmse_hdg"]  for r in runs])
    nis = np.array([r["mean_nis"]  for r in runs])
    ne  = np.array([r["mean_nees"] for r in runs])
    rt  = np.array([r["runtime_ms"]for r in runs])

    lb, ub = average_nees_bounds(dof=3, n_runs=n)
    s      = monte_carlo_statistics(ate)
    anees  = float(np.nanmean(ne))  / 3.0
    anis   = float(np.nanmean(nis)) / 3.0

    return dict(
        scenario=scenario, estimator=estimator,
        trajectory=traj,   noise_regime=noise,  n_runs=n,
        ate_mean=s["mean"],  ate_std=s["std"],
        ate_ci_lo=s["ci_lower"], ate_ci_hi=s["ci_upper"],
        ate_median=s["median"],
        rpe_mean=float(np.mean(rpe)),  rpe_std=float(np.std(rpe,ddof=1)),
        rmse_pos_mean=float(np.mean(rp)), rmse_pos_std=float(np.std(rp,ddof=1)),
        rmse_hdg_mean=float(np.mean(rh)), rmse_hdg_std=float(np.std(rh,ddof=1)),
        anis_mean=anis,  anis_std=float(np.nanstd(nis,ddof=1))/3.0,
        anis_lb=lb,      anis_ub=ub,
        anees_mean=anees, anees_std=float(np.nanstd(ne,ddof=1))/3.0,
        anees_lb=lb,      anees_ub=ub,
        rt_mean=float(np.mean(rt)), rt_std=float(np.std(rt,ddof=1)),
        nis_ok=  lb <= anis  <= ub,
        nees_ok= lb <= anees <= ub,
    )


# ── significance tests ────────────────────────────────────────────────────────

def sig_tests_for_condition(
    ate_map: dict[str, np.ndarray],
    scenario: str, traj: str, noise: str,
) -> list[dict]:
    rows = []
    for est_a, est_b, cname in COMPARISONS:
        a = ate_map.get(est_a)
        b = ate_map.get(est_b)
        if a is None or b is None: continue
        wt = wilcoxon_test(a, b)
        cd = cohens_d(a, b)
        winner = "none"
        if wt["significant"]:
            winner = est_a if wt["mean_diff"] < 0 else est_b
        rows.append(dict(
            scenario=scenario, trajectory=traj, noise_regime=noise,
            comparison=cname, estimator_a=est_a, estimator_b=est_b,
            n_runs=len(a),
            mean_a=float(np.mean(a)), mean_b=float(np.mean(b)),
            median_diff=wt["median_diff"],  mean_diff=wt["mean_diff"],
            w_stat=wt["statistic"], p_value=wt["p_value"],
            significant=wt["significant"],
            effect_r=wt["effect_size"], cohens_d=cd,
            winner=winner,
        ))
    return rows


# ── CSV savers ────────────────────────────────────────────────────────────────

def _save(rows: list[dict], path: str) -> None:
    if not rows: return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"  Saved: {path}")


def _save_raw(
    raw: dict[str, dict[str, list[float]]],
    path: str,
) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario","trajectory","noise_regime","estimator","run_idx","ate"])
        for ckey, ed in raw.items():
            scen, traj, noise = ckey.split("|")
            for est, vals in ed.items():
                for i, v in enumerate(vals):
                    w.writerow([scen, traj, noise, est, i, f"{v:.6f}"])
    print(f"  Saved: {path}")


# ── summary text ──────────────────────────────────────────────────────────────

def _summary(
    std_agg: list[dict], deg_agg: list[dict],
    sig: list[dict], n: int,
) -> str:
    lb, ub = average_nees_bounds(dof=3, n_runs=n)
    L = []

    def h(t): L.append("\n" + "="*125 + f"\n  {t}\n" + "="*125)

    h(f"PHASE E — RESEARCH VALIDATION  |  N={n} Monte Carlo  |  Medisetti Renukeswar  |  June 2026")
    L.append(f"  Consistency bounds (95% CI, dof=3, N={n}): [{lb:.4f}, {ub:.4f}]")

    # header row
    def rhdr():
        L.append(
            f"  {'Scenario':<16} {'Traj':<9} {'Noise':<7} {'Estimator':<15} "
            f"{'ATE mean±std':<20} {'95% CI':<24} "
            f"{'ANIS':<14} {'ANEES':<14} {'NIS':<5} {'NEES'}"
        )
        L.append("  " + "-"*120)

    def frow(r: dict) -> str:
        ci = f"[{r['ate_ci_lo']:.4f},{r['ate_ci_hi']:.4f}]"
        nis  = f"{r['anis_mean']:.3f}±{r['anis_std']:.3f}"
        nees = f"{r['anees_mean']:.3f}±{r['anees_std']:.3f}"
        return (
            f"  {r['scenario']:<16} {r['trajectory']:<9} {r['noise_regime']:<7} "
            f"{r['estimator']:<15} "
            f"{r['ate_mean']:.4f}±{r['ate_std']:.4f}        "
            f"{ci:<24} {nis:<14} {nees:<14} "
            f"{'Y' if r['nis_ok'] else 'N':<5} {'Y' if r['nees_ok'] else 'N'}"
        )

    h("TABLE 1 — Standard Conditions"); rhdr()
    for r in std_agg: L.append(frow(r))

    h("TABLE 2 — Degradation Scenarios"); rhdr()
    for r in deg_agg: L.append(frow(r))

    h("TABLE 3 — Wilcoxon Signed-Rank Tests (ATE, paired N=30, p<0.05)")
    L.append(
        f"  {'Scenario':<14} {'Traj':<9} {'Noise':<7} {'Comparison':<35} "
        f"{'Mean-A':<8} {'Mean-B':<8} {'MedDiff':<9} "
        f"{'W':<8} {'p-val':<9} {'Sig':<5} {'r':<7} {'|d|':<7} {'Winner'}"
    )
    L.append("  " + "-"*125)
    for r in sig:
        L.append(
            f"  {r['scenario']:<14} {r['trajectory']:<9} {r['noise_regime']:<7} "
            f"{r['comparison']:<35} "
            f"{r['mean_a']:<8.4f} {r['mean_b']:<8.4f} {r['median_diff']:<9.4f} "
            f"{r['w_stat']:<8.1f} {r['p_value']:<9.4f} "
            f"{'YES' if r['significant'] else 'NO':<5} "
            f"{r['effect_r']:<7.3f} {abs(r['cohens_d']):<7.3f} {r['winner']}"
        )

    h("STATISTICAL CONCLUSIONS")

    # consistency
    def nis_ok(rows): return sum(1 for r in rows if r["nis_ok"])
    def nees_ok(rows): return sum(1 for r in rows if r["nees_ok"])
    adapt_std = [r for r in std_agg if "Adaptive" in r["estimator"]]
    fixed_std  = [r for r in std_agg if "Adaptive" not in r["estimator"]]
    adapt_deg = [r for r in deg_agg if "Adaptive" in r["estimator"]]
    fixed_deg  = [r for r in deg_agg if "Adaptive" not in r["estimator"]]

    L.append(f"\n  A) ACCURACY — Significant pairwise comparisons:")
    for cname in [c[2] for c in COMPARISONS]:
        sub = [r for r in sig if r["comparison"] == cname]
        ns  = sum(1 for r in sub if r["significant"])
        ds  = [abs(r["cohens_d"]) for r in sub if math.isfinite(r["cohens_d"])]
        pct = 100.0*ns/len(sub) if sub else 0.0
        L.append(f"     {cname:<35}  {ns:2d}/{len(sub):2d} sig ({pct:.0f}%)  "
                 f"median |d|={np.median(ds):.3f}" if ds else
                 f"     {cname:<35}  {ns:2d}/{len(sub):2d} sig ({pct:.0f}%)")

    L.append(f"\n  B) CONSISTENCY — Standard conditions "
             f"(N={len(std_agg)} total, bounds [{lb:.3f},{ub:.3f}]):")
    L.append(f"     Fixed  EKF/UKF  — NIS ok: {nis_ok(fixed_std)}/{len(fixed_std)}  "
             f"NEES ok: {nees_ok(fixed_std)}/{len(fixed_std)}")
    L.append(f"     Adaptive EKF/UKF — NIS ok: {nis_ok(adapt_std)}/{len(adapt_std)}  "
             f"NEES ok: {nees_ok(adapt_std)}/{len(adapt_std)}")

    L.append(f"\n  C) ROBUSTNESS — Degradation scenarios "
             f"(N={len(deg_agg)} total):")
    L.append(f"     Fixed   — NIS ok: {nis_ok(fixed_deg)}/{len(fixed_deg)}  "
             f"NEES ok: {nees_ok(fixed_deg)}/{len(fixed_deg)}  "
             f"mean ATE: {np.mean([r['ate_mean'] for r in fixed_deg]):.4f} m")
    L.append(f"     Adaptive — NIS ok: {nis_ok(adapt_deg)}/{len(adapt_deg)}  "
             f"NEES ok: {nees_ok(adapt_deg)}/{len(adapt_deg)}  "
             f"mean ATE: {np.mean([r['ate_mean'] for r in adapt_deg]):.4f} m")

    # per-scenario degradation ATE
    for scen in ["bias_rw", "vo_drop30", "vo_drop50"]:
        sub_f = [r for r in fixed_deg  if r["scenario"] == scen]
        sub_a = [r for r in adapt_deg  if r["scenario"] == scen]
        if sub_f and sub_a:
            L.append(f"     {scen:<12} fixed ATE={np.mean([r['ate_mean'] for r in sub_f]):.4f}  "
                     f"adaptive ATE={np.mean([r['ate_mean'] for r in sub_a]):.4f}")

    L.append("\n")
    return "\n".join(L)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print(f"  Phase E — Research Validation  |  N={N_MC} MC  |  {N_WORKERS} workers")
    print("  Medisetti Renukeswar")
    print("=" * 70)

    all_std_agg:  list[dict] = []
    all_deg_agg:  list[dict] = []
    all_sig:      list[dict] = []
    std_raw:      dict[str, dict[str, list[float]]] = {}
    deg_raw:      dict[str, dict[str, list[float]]] = {}

    # ── STANDARD ────────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  [1/2] Standard conditions  (3 traj × 3 noise × 4 est × {N_MC} runs)")
    print(f"{'─'*70}")
    t_std = time.perf_counter()

    for traj in TRAJECTORIES:
        for noise in NOISE_REGIMES:
            tag = f"std | {traj} | {noise}"
            runs_map = run_condition("standard", traj, noise,
                                     degradation=None, tag=tag)
            ckey = f"standard|{traj}|{noise}"
            std_raw[ckey] = {e: [r["ate"] for r in runs_map[e]] for e in ESTIMATORS}

            ate_map = {e: np.array(std_raw[ckey][e]) for e in ESTIMATORS}
            sig_rows = sig_tests_for_condition(ate_map, "standard", traj, noise)
            all_sig.extend(sig_rows)

            for est in ESTIMATORS:
                all_std_agg.append(
                    aggregate(runs_map[est], "standard", est, traj, noise)
                )

    print(f"  Standard total: {time.perf_counter()-t_std:.1f}s")

    # incremental save
    _save(all_std_agg, os.path.join(OUT_DIR, "standard_stats.csv"))
    _save(all_sig,     os.path.join(OUT_DIR, "standard_sig.csv"))
    _save_raw(std_raw, os.path.join(OUT_DIR, "standard_N30_raw.csv"))

    # ── DEGRADATION ──────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  [2/2] Degradation scenarios  "
          f"(3 scenarios × 2 traj × 4 est × {N_MC} runs)")
    print(f"{'─'*70}")
    t_deg = time.perf_counter()

    for scen_name, deg_cfg in DEG_CONFIGS.items():
        for traj in DEG_TRAJS:
            tag = f"deg | {scen_name} | {traj}"
            runs_map = run_condition(scen_name, traj, DEG_NOISE,
                                     degradation=deg_cfg, tag=tag)
            ckey = f"{scen_name}|{traj}|{DEG_NOISE}"
            deg_raw[ckey] = {e: [r["ate"] for r in runs_map[e]] for e in ESTIMATORS}

            ate_map = {e: np.array(deg_raw[ckey][e]) for e in ESTIMATORS}
            sig_rows = sig_tests_for_condition(ate_map, scen_name, traj, DEG_NOISE)
            all_sig.extend(sig_rows)

            for est in ESTIMATORS:
                all_deg_agg.append(
                    aggregate(runs_map[est], scen_name, est, traj, DEG_NOISE)
                )

    print(f"  Degradation total: {time.perf_counter()-t_deg:.1f}s")

    _save(all_deg_agg, os.path.join(OUT_DIR, "degradation_stats.csv"))
    _save_raw(deg_raw, os.path.join(OUT_DIR, "degradation_N30_raw.csv"))

    # all sig together
    _save(all_sig, os.path.join(OUT_DIR, "significance_tests.csv"))

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    summary = _summary(all_std_agg, all_deg_agg, all_sig, N_MC)
    path = os.path.join(OUT_DIR, "phase_e_summary.txt")
    with open(path, "w") as f:
        f.write(summary)
    print(summary)
    print(f"\n  Saved: {path}")
    print("  Phase E complete.")


if __name__ == "__main__":
    main()
