"""
Phase 6 — Real Dataset Benchmark
==================================
Runs all four estimators on EuRoC, TUM-VI, and KITTI sequences
(real data if available on disk, synthetic emulation otherwise).

Outputs
-------
  results/phase6/phase6_results.csv        — per-run metrics
  results/phase6/phase6_summary.txt        — human-readable tables
  paper/figures/phase6/fig7_real_ate.png   — ATE comparison across datasets
  paper/figures/phase6/fig8_consistency.png — ANIS/ANEES on real data
  paper/figures/phase6/fig9_sim_vs_real.png — simulation vs real-data comparison

Usage
-----
  # Runs on synthetic emulation (no downloads required):
  python benchmark/run_phase6_real_datasets.py

  # Runs on real data (after downloading):
  EUROC_ROOT=/data/EuRoC TUMVI_ROOT=/data/TUM-VI KITTI_ROOT=/data/KITTI \\
      python benchmark/run_phase6_real_datasets.py

  # Quick test mode (truncated sequences, fast):
  python benchmark/run_phase6_real_datasets.py --quick

Author: Medisetti Renukeswar (Phase 6)
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from ekf_core.ekf_estimator import EKFEstimator
from ekf_core.ukf_estimator import UKFEstimator
from ekf_core.adaptive_estimator import AdaptiveEKFEstimator, AdaptiveUKFEstimator
from ekf_core.metrics import average_nees_bounds
from datasets.common.synthetic_loader import SyntheticDatasetLoader, DATASET_PROFILES
from datasets.common.dataset_adapter import DatasetAdapter, AdapterConfig, RunResult
from datasets.euroc.euroc_loader import EuRoCLoader
from datasets.tumvi.tumvi_loader import TUMVILoader
from datasets.kitti.kitti_loader import KITTILoader

# ── output paths ─────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_CSV    = os.path.join(ROOT, "results", "phase6", "phase6_results.csv")
OUT_TXT    = os.path.join(ROOT, "results", "phase6", "phase6_summary.txt")
FIG_DIR    = os.path.join(ROOT, "paper", "figures", "phase6")
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ── dataset root paths (override with env vars or edit here) ─────────────────
EUROC_ROOT = os.environ.get("EUROC_ROOT", "")
TUMVI_ROOT = os.environ.get("TUMVI_ROOT", "")
KITTI_ROOT = os.environ.get("KITTI_ROOT", "")

# ── estimator factory ────────────────────────────────────────────────────────
ESTIMATORS = {
    "EKF":          lambda dt: EKFEstimator(dt=dt),
    "UKF":          lambda dt: UKFEstimator(dt=dt),
    "Adaptive-EKF": lambda dt: AdaptiveEKFEstimator(dt=dt, window=20, adapt_R=True,
                                                     adapt_Q=False, alpha_smooth=0.1),
    "Adaptive-UKF": lambda dt: AdaptiveUKFEstimator(dt=dt, window=20, adapt_R=True,
                                                     adapt_Q=False, alpha_smooth=0.1),
}

# ── visual style ──────────────────────────────────────────────────────────────
BG  = "#0a0f1a"
COL = {
    "EKF":          "#00e5ff",
    "UKF":          "#00ff88",
    "Adaptive-EKF": "#b96dff",
    "Adaptive-UKF": "#ffd600",
}

def sax(ax: plt.Axes, fs: int = 10) -> None:
    ax.set_facecolor(BG)
    ax.tick_params(colors="#aaccdd", labelsize=fs)
    ax.xaxis.label.set_color("#aaccdd")
    ax.yaxis.label.set_color("#aaccdd")
    ax.title.set_color("white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#1a3a5a")
    ax.grid(True, alpha=0.2, color="#00e5ff", linewidth=0.5)


# ── loader factory ────────────────────────────────────────────────────────────

def make_loaders(quick: bool = False) -> list[tuple[str, object]]:
    """
    Returns a list of (label, loader) pairs.

    If real data paths are set and files exist, real loaders are used.
    Otherwise, synthetic emulators are used and clearly labelled.
    """
    truncate = 30.0 if quick else 0.0
    loaders: list[tuple[str, object]] = []

    # ── EuRoC ─────────────────────────────────────────────────────────────
    for seq, profile_key in [
        ("MH_01_easy",  "EuRoC_MH_01_easy"),
        ("V1_01_easy",  "EuRoC_V1_01_easy"),
        ("V2_02_medium","EuRoC_V2_02_medium"),
    ]:
        real_path = os.path.join(EUROC_ROOT, seq) if EUROC_ROOT else ""
        real_loader = EuRoCLoader(real_path, seq) if real_path else None

        if real_loader and real_loader.is_available():
            label = f"EuRoC/{seq}"
            loaders.append((label, real_loader))
            print(f"  [REAL]  {label}")
        else:
            label = f"EuRoC/{seq}[synth]"
            loaders.append((label, SyntheticDatasetLoader(
                profile_key, seed=42, truncate_s=truncate
            )))
            print(f"  [SYNTH] {label}")

    # ── TUM-VI ────────────────────────────────────────────────────────────
    for seq, profile_key, has_gt in [
        ("room1",     "TUM-VI_room1",     True),
        ("corridor1", "TUM-VI_corridor1", False),
    ]:
        real_path = os.path.join(TUMVI_ROOT, seq) if TUMVI_ROOT else ""
        real_loader = TUMVILoader(real_path, seq, has_gt) if real_path else None

        if real_loader and real_loader.is_available():
            label = f"TUM-VI/{seq}"
            loaders.append((label, real_loader))
            print(f"  [REAL]  {label}")
        else:
            label = f"TUM-VI/{seq}[synth]"
            loaders.append((label, SyntheticDatasetLoader(
                profile_key, seed=99, truncate_s=truncate
            )))
            print(f"  [SYNTH] {label}")

    # ── KITTI ─────────────────────────────────────────────────────────────
    for seq, profile_key in [
        ("00", "KITTI_00"),
        ("05", "KITTI_05"),
    ]:
        real_path = KITTI_ROOT if KITTI_ROOT else ""
        real_loader = KITTILoader(real_path, seq) if real_path else None

        if real_loader and real_loader.is_available():
            label = f"KITTI/{seq}"
            loaders.append((label, real_loader))
            print(f"  [REAL]  {label}")
        else:
            label = f"KITTI/{seq}[synth]"
            loaders.append((label, SyntheticDatasetLoader(
                profile_key, seed=7, truncate_s=truncate
            )))
            print(f"  [SYNTH] {label}")

    return loaders


# ── main benchmark ────────────────────────────────────────────────────────────

def run_benchmark(quick: bool = False) -> list[RunResult]:
    print("=" * 70)
    print("  Phase 6 — Real Dataset Benchmark")
    print("  Medisetti Renukeswar | June 2026")
    print("=" * 70)
    print("\nDataset sources:")

    loaders = make_loaders(quick)

    print(f"\nRunning {len(loaders)} sequences × {len(ESTIMATORS)} estimators "
          f"{'(QUICK mode)' if quick else '(FULL mode)'} ...\n")

    results: list[RunResult] = []
    lb, ub = average_nees_bounds(dof=3, n_runs=1)  # per-run bounds

    for label, loader in loaders:
        # Load once, reuse for all estimators
        loader.load()
        meta   = loader.get_metadata()
        dt_imu = 1.0 / meta.imu_rate_hz

        cfg = AdapterConfig(
            imu_rate_hz=meta.imu_rate_hz,
            gt_as_vo=True,
            vel_decay=0.01,
        )

        for est_name, est_factory in ESTIMATORS.items():
            est     = est_factory(dt_imu)
            adapter = DatasetAdapter(est, cfg)

            t0  = time.perf_counter()
            res = adapter.run(loader, estimator_name=est_name)
            elapsed = time.perf_counter() - t0

            # Override sequence name with the full label
            res.sequence_name = label

            nis_str  = f"{res.mean_nis / 3:.3f}" if math.isfinite(res.mean_nis)  else "n/a"
            nees_str = f"{res.mean_nees / 3:.3f}" if math.isfinite(res.mean_nees) else "n/a"
            ate_str  = f"{res.ate:.4f}"           if math.isfinite(res.ate)        else "n/a"

            print(f"  {label:<35} {est_name:<15} "
                  f"ATE={ate_str}  ANIS={nis_str}  ANEES={nees_str}  "
                  f"[{elapsed:.1f}s]")
            results.append(res)

    return results


# ── save / report ─────────────────────────────────────────────────────────────

def save_csv(results: list[RunResult], path: str) -> None:
    if not results:
        return
    fieldnames = [
        "sequence_name", "estimator_name", "dataset_name",
        "ate", "rpe", "rmse_pos", "rmse_heading",
        "mean_nis", "mean_nees", "n_imu", "n_updates",
        "runtime_ms", "has_gt", "notes",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in results:
            row = asdict(r)
            for k in list(row.keys()):
                if k not in fieldnames:
                    del row[k]
            w.writerow(row)
    print(f"\nSaved: {path}")


def save_summary(results: list[RunResult], path: str) -> None:
    lb, ub = average_nees_bounds(dof=3, n_runs=30)
    lines  = []

    def L(s: str = "") -> None:
        lines.append(s)

    L("=" * 110)
    L("PHASE 6 — REAL DATASET VALIDATION")
    L("Adaptive Covariance Tuning: EKF/UKF on EuRoC, TUM-VI, KITTI")
    L("Medisetti Renukeswar | June 2026")
    L()
    L("Sequences marked [synth] used synthetic emulation (real data not present).")
    L("Sequences without [synth] were evaluated on the actual dataset files.")
    L(f"Consistency bounds (chi², dof=3, N=1): [n/a for single-run; reference N=30: [{lb:.4f},{ub:.4f}]]")
    L("=" * 110)
    L()

    # Group by dataset
    for ds in ["EuRoC", "TUM-VI", "KITTI"]:
        ds_results = [r for r in results if r.dataset_name == ds]
        if not ds_results:
            continue
        L(f"── {ds} ──────────────────────────────────────────────────────────")
        L(f"{'Sequence':<38} {'Estimator':<15} {'ATE (m)':<10} "
          f"{'RPE (m)':<10} {'RMSE-pos':<10} {'RMSE-hdg':<12} "
          f"{'ANIS':<8} {'ANEES':<8} {'RT(ms)'}")
        L("-" * 110)
        for r in ds_results:
            def fmt(v: float) -> str:
                return f"{v:.4f}" if math.isfinite(v) else "  n/a  "
            anis  = fmt(r.mean_nis  / 3.0)
            anees = fmt(r.mean_nees / 3.0)
            L(f"{r.sequence_name:<38} {r.estimator_name:<15} "
              f"{fmt(r.ate):<10} {fmt(r.rpe):<10} {fmt(r.rmse_pos):<10} "
              f"{fmt(r.rmse_heading):<12} {anis:<8} {anees:<8} {r.runtime_ms:.0f}")
        L()

    # Findings
    L("=" * 110)
    L("FINDINGS")
    L("=" * 110)

    for est in ["EKF", "UKF", "Adaptive-EKF", "Adaptive-UKF"]:
        est_results = [r for r in results if r.estimator_name == est
                       and math.isfinite(r.ate)]
        if not est_results:
            continue
        ate_vals = [r.ate for r in est_results]
        L(f"  {est:20s}: ATE mean={np.mean(ate_vals):.4f}m  "
          f"std={np.std(ate_vals):.4f}m  "
          f"min={min(ate_vals):.4f}m  max={max(ate_vals):.4f}m")

    text = "\n".join(lines)
    print("\n" + text)
    with open(path, "w") as f:
        f.write(text)
    print(f"\nSaved: {path}")


# ── figures ───────────────────────────────────────────────────────────────────

def make_figures(results: list[RunResult]) -> None:

    # ── Fig 7: ATE per dataset per estimator ──────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "Figure 7: ATE on Real/Synthetic Dataset Sequences\n"
        "(sequences marked [synth] = synthetic emulation)",
        color="white", fontsize=12, y=1.01,
    )

    est_names = list(ESTIMATORS.keys())
    for di, ds in enumerate(["EuRoC", "TUM-VI", "KITTI"]):
        ax = axes[di]
        sax(ax)
        ds_results = [r for r in results if r.dataset_name == ds]
        seqs = sorted(set(r.sequence_name for r in ds_results))
        x = np.arange(len(seqs))
        w = 0.18

        for ei, est in enumerate(est_names):
            ates = []
            for seq in seqs:
                rows = [r for r in ds_results
                        if r.sequence_name == seq and r.estimator_name == est]
                ates.append(rows[0].ate if rows and math.isfinite(rows[0].ate) else 0.0)
            off = (ei - 1.5) * w
            ax.bar(x + off, ates, w, color=COL[est], alpha=0.85, label=est)

        short_seqs = [s.split("/")[-1].replace("[synth]", "\n[S]") for s in seqs]
        ax.set_xticks(x)
        ax.set_xticklabels(short_seqs, fontsize=7, rotation=15, ha="right")
        ax.set_xlabel("Sequence", fontsize=9)
        ax.set_ylabel("ATE (m)", fontsize=9)
        ax.set_title(ds, fontsize=11)
        if di == 0:
            ax.legend(facecolor="#0e1a2a", edgecolor="#1a3a5a",
                      labelcolor="white", fontsize=7)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig7_real_ate.png")
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved: {path}")

    # ── Fig 8: ANIS across datasets ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor(BG)
    sax(ax)
    lb, ub = average_nees_bounds(dof=3, n_runs=30)
    ax.axhline(lb, color="#ff6b35", lw=1.5, ls="--", alpha=0.9,
               label=f"Lower bound ({lb:.3f})")
    ax.axhline(ub, color="#ff6b35", lw=1.5, ls="-.", alpha=0.9,
               label=f"Upper bound ({ub:.3f})")
    ax.axhline(1.0, color="white", lw=0.8, ls=":", alpha=0.4,
               label="Ideal (1.0)")

    valid = [r for r in results if math.isfinite(r.mean_nis)]
    seqs_all = sorted(set(r.sequence_name for r in valid))
    x = np.arange(len(seqs_all))
    w = 0.18
    for ei, est in enumerate(est_names):
        anis_vals = []
        for seq in seqs_all:
            rows = [r for r in valid
                    if r.sequence_name == seq and r.estimator_name == est]
            anis_vals.append(rows[0].mean_nis / 3.0 if rows else 0.0)
        off = (ei - 1.5) * w
        ax.bar(x + off, anis_vals, w, color=COL[est], alpha=0.85, label=est)

    short = [s.split("/")[-1].replace("[synth]", "\n[S]") for s in seqs_all]
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=7, rotation=20, ha="right")
    ax.set_ylabel("ANIS (= mean NIS / 3)", fontsize=9)
    ax.set_title("Figure 8: Filter Consistency (ANIS) on Real/Synthetic Datasets",
                 fontsize=11)
    ax.legend(facecolor="#0e1a2a", edgecolor="#1a3a5a", labelcolor="white",
              fontsize=7, ncol=7)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig8_real_consistency.png")
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved: {path}")

    # ── Fig 9: Simulation vs real-data ATE comparison ─────────────────────
    # Load simulation results for comparison
    sim_csv = os.path.join(ROOT, "results", "merged_standard_stats.csv")
    sim_medium: dict[str, float] = {}
    if os.path.isfile(sim_csv):
        import csv as _csv
        with open(sim_csv) as f:
            for row in _csv.DictReader(f):
                if row["noise_regime"] == "medium":
                    key = row["estimator"]
                    try:
                        sim_medium.setdefault(key, []).append(float(row["ate_mean"]))
                    except (KeyError, ValueError):
                        pass
        sim_medium = {k: float(np.mean(v)) for k, v in sim_medium.items()}

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG)
    sax(ax)

    categories = ["Simulation\n(medium noise)", "EuRoC\n[synth]",
                  "TUM-VI\n[synth]", "KITTI\n[synth]"]
    x = np.arange(len(categories))
    w = 0.18

    for ei, est in enumerate(est_names):
        vals = []
        # Simulation medium
        vals.append(sim_medium.get(est, float("nan")))
        # Per dataset average
        for ds in ["EuRoC", "TUM-VI", "KITTI"]:
            ds_r = [r for r in results
                    if r.dataset_name == ds and r.estimator_name == est
                    and math.isfinite(r.ate)]
            vals.append(float(np.mean([r.ate for r in ds_r])) if ds_r else float("nan"))
        clean = [v if math.isfinite(v) else 0.0 for v in vals]
        off = (ei - 1.5) * w
        ax.bar(x + off, clean, w, color=COL[est], alpha=0.85, label=est)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel("Mean ATE (m)", fontsize=10)
    ax.set_title("Figure 9: Simulation vs Dataset ATE Comparison",
                 fontsize=11)
    ax.legend(facecolor="#0e1a2a", edgecolor="#1a3a5a", labelcolor="white",
              fontsize=8)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig9_sim_vs_real.png")
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved: {path}")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6 Real Dataset Benchmark")
    parser.add_argument("--quick", action="store_true",
                        help="Truncate sequences to 30s for fast testing")
    args = parser.parse_args()

    results = run_benchmark(quick=args.quick)
    save_csv(results, OUT_CSV)
    save_summary(results, OUT_TXT)
    make_figures(results)

    print("\nPhase 6 complete.")
    print(f"  Results:  {OUT_CSV}")
    print(f"  Summary:  {OUT_TXT}")
    print(f"  Figures:  {FIG_DIR}/")


if __name__ == "__main__":
    main()
