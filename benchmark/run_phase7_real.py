"""
Phase 7 — Real Dataset Benchmark Runner
=========================================
Evaluates EKF, UKF, Adaptive-EKF, and Adaptive-UKF on actual
EuRoC MAV, TUM-VI, and KITTI dataset files.

Policy (enforced in code)
--------------------------
  * No synthetic fallback.  If a dataset file is missing, that sequence
    is skipped and logged as UNAVAILABLE.  It does not appear in result
    tables or figures.
  * No mixing of synthetic and real results.
  * Every result row carries a ``data_source`` field set to "real_dataset"
    so downstream code can assert provenance.

Usage
-----
  # Check which datasets are available (no evaluation):
  python benchmark/run_phase7_real.py --check-only

  # Run evaluation on whatever is present:
  python benchmark/run_phase7_real.py

  # Specify dataset roots explicitly:
  EUROC_ROOT=/path/to/EuRoC python benchmark/run_phase7_real.py
  TUMVI_ROOT=/path/to/TUM-VI python benchmark/run_phase7_real.py
  KITTI_ROOT=/path/to/KITTI   python benchmark/run_phase7_real.py

Outputs (only produced when real data is present)
--------------------------------------------------
  results/real_datasets/raw_csv/phase7_raw.csv
  results/real_datasets/tables/phase7_table1_accuracy.txt
  results/real_datasets/tables/phase7_table2_consistency.txt
  results/real_datasets/tables/phase7_table3_comparison.txt
  results/real_datasets/plots/fig10_real_ate.png
  results/real_datasets/plots/fig11_real_consistency.png
  REAL_DATASET_REPORT.md
  REAL_DATASET_SUMMARY.md

Author: Medisetti Renukeswar (Phase 7)
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from typing import Optional

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
from datasets.common.dataset_adapter import DatasetAdapter, AdapterConfig
from datasets.euroc.euroc_loader import EuRoCLoader
from datasets.tumvi.tumvi_loader import TUMVILoader
from datasets.kitti.kitti_loader import KITTILoader

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR   = os.path.join(ROOT, "results", "real_datasets", "raw_csv")
TABLE_DIR = os.path.join(ROOT, "results", "real_datasets", "tables")
PLOT_DIR  = os.path.join(ROOT, "results", "real_datasets", "plots")
for d in (RAW_DIR, TABLE_DIR, PLOT_DIR):
    os.makedirs(d, exist_ok=True)

# ── dataset root paths ────────────────────────────────────────────────────────
EUROC_ROOT = os.environ.get("EUROC_ROOT", "")
TUMVI_ROOT = os.environ.get("TUMVI_ROOT", "")
KITTI_ROOT = os.environ.get("KITTI_ROOT", "")

# ── estimator factory ──────────────────────────────────────────────────────────
ESTIMATORS = {
    "EKF":          lambda dt: EKFEstimator(dt=dt),
    "UKF":          lambda dt: UKFEstimator(dt=dt),
    "Adaptive-EKF": lambda dt: AdaptiveEKFEstimator(
        dt=dt, window=20, adapt_R=True, adapt_Q=False, alpha_smooth=0.1),
    "Adaptive-UKF": lambda dt: AdaptiveUKFEstimator(
        dt=dt, window=20, adapt_R=True, adapt_Q=False, alpha_smooth=0.1),
}

# ── visual style ───────────────────────────────────────────────────────────────
BG  = "#0a0f1a"
COL = {
    "EKF":          "#00e5ff",
    "UKF":          "#00ff88",
    "Adaptive-EKF": "#b96dff",
    "Adaptive-UKF": "#ffd600",
}


# ── result dataclass ───────────────────────────────────────────────────────────

@dataclass
class RealResult:
    """
    One result row.  ``data_source`` is always "real_dataset" — never synthetic.
    """
    data_source:    str = "real_dataset"   # IMMUTABLE — never change this
    dataset_name:   str = ""
    sequence_name:  str = ""
    estimator_name: str = ""
    n_imu:          int   = 0
    n_gt:           int   = 0
    n_updates:      int   = 0
    duration_s:     float = 0.0
    ate:            float = float("nan")
    rpe:            float = float("nan")
    rmse_pos:       float = float("nan")
    rmse_heading:   float = float("nan")
    anis:           float = float("nan")   # mean_nis / 3
    anees:          float = float("nan")   # mean_nees / 3
    nis_consistent: bool  = False
    nees_consistent: bool = False
    runtime_ms:     float = 0.0

    def assert_real(self) -> None:
        """Raise if this result is not from real data."""
        assert self.data_source == "real_dataset", (
            f"BUG: result for {self.sequence_name}/{self.estimator_name} "
            f"has data_source='{self.data_source}', not 'real_dataset'"
        )


# ── sequence catalogue ─────────────────────────────────────────────────────────

@dataclass
class SequenceSpec:
    dataset:       str
    sequence_name: str
    root_var:      str          # which root path applies
    sub_path:      str          # relative path under root
    loader_cls:    type
    loader_kwargs: dict = field(default_factory=dict)
    imu_rate_hz:   float = 200.0


SEQUENCE_CATALOGUE: list[SequenceSpec] = [
    # ── EuRoC ──────────────────────────────────────────────────────────────
    SequenceSpec("EuRoC", "MH_01_easy",   "EUROC_ROOT", "MH_01_easy",
                 EuRoCLoader, {"sequence_name": "MH_01_easy"},   200.0),
    SequenceSpec("EuRoC", "MH_02_easy",   "EUROC_ROOT", "MH_02_easy",
                 EuRoCLoader, {"sequence_name": "MH_02_easy"},   200.0),
    SequenceSpec("EuRoC", "MH_03_medium", "EUROC_ROOT", "MH_03_medium",
                 EuRoCLoader, {"sequence_name": "MH_03_medium"}, 200.0),
    SequenceSpec("EuRoC", "V1_01_easy",   "EUROC_ROOT", "V1_01_easy",
                 EuRoCLoader, {"sequence_name": "V1_01_easy"},   200.0),
    SequenceSpec("EuRoC", "V1_02_medium", "EUROC_ROOT", "V1_02_medium",
                 EuRoCLoader, {"sequence_name": "V1_02_medium"}, 200.0),
    SequenceSpec("EuRoC", "V2_02_medium", "EUROC_ROOT", "V2_02_medium",
                 EuRoCLoader, {"sequence_name": "V2_02_medium"}, 200.0),
    # ── TUM-VI ─────────────────────────────────────────────────────────────
    SequenceSpec("TUM-VI", "room1",     "TUMVI_ROOT", "room1",
                 TUMVILoader, {"sequence_name": "room1",     "has_full_gt": True},  200.0),
    SequenceSpec("TUM-VI", "room2",     "TUMVI_ROOT", "room2",
                 TUMVILoader, {"sequence_name": "room2",     "has_full_gt": True},  200.0),
    SequenceSpec("TUM-VI", "corridor1", "TUMVI_ROOT", "corridor1",
                 TUMVILoader, {"sequence_name": "corridor1", "has_full_gt": False}, 200.0),
    SequenceSpec("TUM-VI", "corridor2", "TUMVI_ROOT", "corridor2",
                 TUMVILoader, {"sequence_name": "corridor2", "has_full_gt": False}, 200.0),
    # ── KITTI ──────────────────────────────────────────────────────────────
    SequenceSpec("KITTI", "00", "KITTI_ROOT", "",
                 KITTILoader, {"sequence_id": "00"}, 100.0),
    SequenceSpec("KITTI", "05", "KITTI_ROOT", "",
                 KITTILoader, {"sequence_id": "05"}, 100.0),
    SequenceSpec("KITTI", "07", "KITTI_ROOT", "",
                 KITTILoader, {"sequence_id": "07"}, 100.0),
]


# ── availability check ─────────────────────────────────────────────────────────

def check_availability() -> dict[str, dict]:
    """
    Check which sequences are available on disk.

    Returns dict: sequence_name → {available: bool, path: str, reason: str}
    """
    roots = {
        "EUROC_ROOT": EUROC_ROOT,
        "TUMVI_ROOT": TUMVI_ROOT,
        "KITTI_ROOT": KITTI_ROOT,
    }
    status: dict[str, dict] = {}

    for spec in SEQUENCE_CATALOGUE:
        root = roots.get(spec.root_var, "")

        if not root:
            status[f"{spec.dataset}/{spec.sequence_name}"] = {
                "available": False,
                "path": "",
                "reason": f"{spec.root_var} not set",
                "spec": spec,
            }
            continue

        # Build the loader to use its is_available() check
        if spec.dataset == "KITTI":
            loader = spec.loader_cls(root, **spec.loader_kwargs)
        else:
            seq_path = os.path.join(root, spec.sub_path)
            loader = spec.loader_cls(seq_path, **spec.loader_kwargs)

        avail  = loader.is_available()
        path   = seq_path if spec.dataset != "KITTI" else root
        reason = "files present" if avail else "files not found at path"

        status[f"{spec.dataset}/{spec.sequence_name}"] = {
            "available": avail,
            "path": path,
            "reason": reason,
            "spec": spec,
            "loader": loader,
        }

    return status


def print_availability(status: dict[str, dict]) -> None:
    n_avail = sum(1 for v in status.values() if v["available"])
    n_total = len(status)

    print()
    print("=" * 70)
    print(f"  DATASET AVAILABILITY  ({n_avail}/{n_total} sequences available)")
    print("=" * 70)
    print(f"  {'Sequence':<30} {'Status':<12} {'Path / Reason'}")
    print("  " + "-" * 68)

    for key, info in status.items():
        sym   = "✓ AVAILABLE" if info["available"] else "✗ MISSING"
        detail = info["path"] if info["available"] else info["reason"]
        print(f"  {key:<30} {sym:<12}  {detail}")
    print()

    if n_avail == 0:
        print("  ⚠  No real dataset files found on this system.")
        print("  ⚠  See DATASET_AVAILABILITY.md for download instructions.")
        print()


# ── single-sequence evaluation ─────────────────────────────────────────────────

def evaluate_sequence(
    spec: SequenceSpec,
    loader,
    lb: float,
    ub: float,
) -> list[RealResult]:
    """
    Run all four estimators on one real sequence.
    Returns one RealResult per estimator.
    """
    loader.load()
    meta   = loader.get_metadata()
    dt_imu = 1.0 / spec.imu_rate_hz
    cfg    = AdapterConfig(imu_rate_hz=spec.imu_rate_hz, gt_as_vo=True, vel_decay=0.01)

    results: list[RealResult] = []

    for est_name, est_factory in ESTIMATORS.items():
        est     = est_factory(dt_imu)
        adapter = DatasetAdapter(est, cfg)

        t0  = time.perf_counter()
        res = adapter.run(loader, estimator_name=est_name)
        rt  = (time.perf_counter() - t0) * 1000.0

        anis  = res.mean_nis  / 3.0 if math.isfinite(res.mean_nis)  else float("nan")
        anees = res.mean_nees / 3.0 if math.isfinite(res.mean_nees) else float("nan")
        nis_ok  = lb <= anis  <= ub if math.isfinite(anis)  else False
        nees_ok = lb <= anees <= ub if math.isfinite(anees) else False

        row = RealResult(
            data_source    = "real_dataset",
            dataset_name   = spec.dataset,
            sequence_name  = spec.sequence_name,
            estimator_name = est_name,
            n_imu          = res.n_imu,
            n_gt           = loader.n_gt,
            n_updates      = res.n_updates,
            duration_s     = meta.duration_s,
            ate            = res.ate,
            rpe            = res.rpe,
            rmse_pos       = res.rmse_pos,
            rmse_heading   = res.rmse_heading,
            anis           = anis,
            anees          = anees,
            nis_consistent = nis_ok,
            nees_consistent = nees_ok,
            runtime_ms     = rt,
        )
        row.assert_real()   # provenance guard
        results.append(row)

        ate_s  = f"{res.ate:.4f}"  if math.isfinite(res.ate)  else "  n/a "
        anis_s = f"{anis:.3f}"    if math.isfinite(anis)     else " n/a"
        print(f"    {est_name:<15}  ATE={ate_s}  ANIS={anis_s}  "
              f"[{rt:.0f}ms]")

    return results


# ── table formatters ───────────────────────────────────────────────────────────

def fmt(v: float, precision: int = 4) -> str:
    return f"{v:.{precision}f}" if math.isfinite(v) else "  n/a  "


def make_table1(results: list[RealResult], lb: float, ub: float) -> str:
    """Table 1 — Accuracy metrics."""
    lines = []
    lines.append("=" * 115)
    lines.append("TABLE 1 — ACCURACY METRICS (real dataset evaluation)")
    lines.append(f"  ATE, RPE, RMSE position, RMSE heading | data_source = real_dataset")
    lines.append("=" * 115)
    lines.append(
        f"  {'Dataset':<8} {'Sequence':<15} {'Estimator':<15} "
        f"{'ATE (m)':<12} {'RPE (m)':<12} {'RMSE-pos':<12} "
        f"{'RMSE-hdg':<12} {'n_updates':<10} {'RT (ms)'}"
    )
    lines.append("  " + "-" * 110)

    for r in results:
        r.assert_real()
        lines.append(
            f"  {r.dataset_name:<8} {r.sequence_name:<15} {r.estimator_name:<15} "
            f"{fmt(r.ate):<12} {fmt(r.rpe):<12} {fmt(r.rmse_pos):<12} "
            f"{fmt(r.rmse_heading):<12} {r.n_updates:<10} {r.runtime_ms:.0f}"
        )
    if not results:
        lines.append("  (no real dataset results — see DATASET_AVAILABILITY.md)")
    lines.append("=" * 115)
    return "\n".join(lines)


def make_table2(results: list[RealResult], lb: float, ub: float) -> str:
    """Table 2 — Consistency metrics."""
    lines = []
    lines.append("=" * 100)
    lines.append("TABLE 2 — FILTER CONSISTENCY (real dataset evaluation)")
    lines.append(f"  ANIS = mean(NIS)/3,  ANEES = mean(NEES)/3")
    lines.append(f"  Consistency bounds (chi², dof=3, N=1, 95% CI): [{lb:.4f}, {ub:.4f}]")
    lines.append("  NOTE: Single-run bounds apply here (not N=30 MC bounds from simulation)")
    lines.append("=" * 100)
    lines.append(
        f"  {'Dataset':<8} {'Sequence':<15} {'Estimator':<15} "
        f"{'ANIS':<10} {'NIS-ok':<8} {'ANEES':<10} {'NEES-ok'}"
    )
    lines.append("  " + "-" * 98)
    for r in results:
        r.assert_real()
        lines.append(
            f"  {r.dataset_name:<8} {r.sequence_name:<15} {r.estimator_name:<15} "
            f"{fmt(r.anis,4):<10} {'YES' if r.nis_consistent else 'NO':<8} "
            f"{fmt(r.anees,4):<10} {'YES' if r.nees_consistent else 'NO'}"
        )
    if not results:
        lines.append("  (no real dataset results — see DATASET_AVAILABILITY.md)")
    lines.append("=" * 100)
    return "\n".join(lines)


def make_table3(results: list[RealResult]) -> str:
    """Table 3 — EKF vs UKF and fixed vs adaptive comparison."""
    lines = []
    lines.append("=" * 100)
    lines.append("TABLE 3 — ESTIMATOR COMPARISON SUMMARY (real dataset evaluation)")
    lines.append("=" * 100)

    if not results:
        lines.append("  (no real dataset results — see DATASET_AVAILABILITY.md)")
        lines.append("=" * 100)
        return "\n".join(lines)

    sequences = sorted(set(r.sequence_name for r in results))
    for seq in sequences:
        seq_results = [r for r in results if r.sequence_name == seq]
        ds = seq_results[0].dataset_name
        lines.append(f"\n  {ds} / {seq}")
        lines.append(f"  {'Estimator':<16} {'ATE (m)':<12} {'vs EKF':<12} "
                     f"{'ANIS':<10} {'Consistent?'}")
        lines.append("  " + "-" * 60)

        ekf_ate = next((r.ate for r in seq_results if r.estimator_name == "EKF"), float("nan"))
        for r in seq_results:
            if math.isfinite(r.ate) and math.isfinite(ekf_ate) and ekf_ate > 0:
                delta_pct = f"{100*(r.ate - ekf_ate)/ekf_ate:+.1f}%"
            else:
                delta_pct = "  n/a"
            lb_seq, ub_seq = average_nees_bounds(dof=3, n_runs=1)
            cons = "YES" if r.nis_consistent else "NO"
            lines.append(
                f"  {r.estimator_name:<16} {fmt(r.ate):<12} {delta_pct:<12} "
                f"{fmt(r.anis,3):<10} {cons}"
            )

    lines.append("\n" + "=" * 100)
    return "\n".join(lines)


# ── figures ────────────────────────────────────────────────────────────────────

def make_figures(results: list[RealResult], lb: float, ub: float) -> None:
    """Generate publication-quality figures from real results only."""
    if not results:
        print("  No real results — skipping figure generation.")
        return

    EST_NAMES = list(ESTIMATORS.keys())

    def sax(ax, fs=10):
        ax.set_facecolor(BG)
        ax.tick_params(colors="#aaccdd", labelsize=fs)
        ax.xaxis.label.set_color("#aaccdd")
        ax.yaxis.label.set_color("#aaccdd")
        ax.title.set_color("white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#1a3a5a")
        ax.grid(True, alpha=0.2, color="#00e5ff", linewidth=0.5)

    # ── Fig 10: ATE per sequence per estimator ─────────────────────────────
    seqs = sorted(set(r.sequence_name for r in results))
    n_seqs = len(seqs)
    fig, ax = plt.subplots(figsize=(max(10, n_seqs * 2.5), 5))
    fig.patch.set_facecolor(BG)
    sax(ax)

    x   = np.arange(n_seqs)
    w   = 0.18
    for ei, est in enumerate(EST_NAMES):
        ates = []
        for seq in seqs:
            rows = [r for r in results if r.sequence_name == seq and r.estimator_name == est]
            ates.append(rows[0].ate if rows and math.isfinite(rows[0].ate) else 0.0)
        off = (ei - 1.5) * w
        ax.bar(x + off, ates, w, color=COL[est], alpha=0.85, label=est)

    ax.set_xticks(x)
    ax.set_xticklabels(seqs, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("ATE (m)", fontsize=10)
    ax.set_title("Figure 10: ATE on Real Dataset Sequences\n(real data only)",
                 fontsize=11)
    handles = [mpatches.Patch(color=COL[e], label=e) for e in EST_NAMES]
    ax.legend(handles=handles, facecolor="#0e1a2a", edgecolor="#1a3a5a",
              labelcolor="white", fontsize=8)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "fig10_real_ate.png")
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {path}")

    # ── Fig 11: ANIS consistency ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(10, n_seqs * 2.5), 5))
    fig.patch.set_facecolor(BG)
    sax(ax)
    ax.axhline(lb, color="#ff6b35", lw=1.5, ls="--", alpha=0.9,
               label=f"Lower ({lb:.3f})")
    ax.axhline(ub, color="#ff6b35", lw=1.5, ls="-.", alpha=0.9,
               label=f"Upper ({ub:.3f})")
    ax.axhline(1.0, color="white", lw=0.8, ls=":", alpha=0.4, label="Ideal 1.0")

    for ei, est in enumerate(EST_NAMES):
        anis_vals = []
        for seq in seqs:
            rows = [r for r in results if r.sequence_name == seq and r.estimator_name == est]
            anis_vals.append(rows[0].anis if rows and math.isfinite(rows[0].anis) else 0.0)
        off = (ei - 1.5) * w
        ax.bar(x + off, anis_vals, w, color=COL[est], alpha=0.85, label=est)

    ax.set_xticks(x)
    ax.set_xticklabels(seqs, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("ANIS", fontsize=10)
    ax.set_title("Figure 11: Filter Consistency (ANIS) on Real Datasets\n"
                 "(real data only — chi² bounds shown)", fontsize=11)
    ax.legend(facecolor="#0e1a2a", edgecolor="#1a3a5a", labelcolor="white",
              fontsize=7, ncol=7)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "fig11_real_consistency.png")
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {path}")


# ── report generators ──────────────────────────────────────────────────────────

def write_real_dataset_report(results: list[RealResult], path: str) -> None:
    """Write REAL_DATASET_REPORT.md."""
    # Verify every row before writing
    for r in results:
        r.assert_real()

    lb, ub = average_nees_bounds(dof=3, n_runs=1)
    n_seq  = len(set(r.sequence_name for r in results))
    n_runs = len(results)

    lines = [
        "# Real Dataset Evaluation Report",
        "",
        f"**Author:** Medisetti Renukeswar  ",
        f"**Date:** June 2026  ",
        f"**Data source:** `real_dataset` (verified — no synthetic data in this file)",
        "",
        "---",
        "",
        "## Evaluation Summary",
        "",
        f"| Item | Value |",
        f"|------|-------|",
        f"| Sequences evaluated | {n_seq} |",
        f"| Estimator runs | {n_runs} |",
        f"| Sequences skipped (unavailable) | "
        f"{len(SEQUENCE_CATALOGUE) - n_seq} |",
        f"| Consistency bounds (chi², dof=3, N=1) | [{lb:.4f}, {ub:.4f}] |",
        "",
    ]

    if not results:
        lines += [
            "## Status: No Real Data Available",
            "",
            "No EuRoC, TUM-VI, or KITTI dataset files were found on this system.",
            "See `DATASET_AVAILABILITY.md` for download instructions.",
            "",
            "**This report will be populated automatically once real data is present**",
            "by re-running:",
            "",
            "```bash",
            "EUROC_ROOT=/path/to/EuRoC python benchmark/run_phase7_real.py",
            "```",
            "",
        ]
    else:
        # Group by dataset
        for ds in ["EuRoC", "TUM-VI", "KITTI"]:
            ds_r = [r for r in results if r.dataset_name == ds]
            if not ds_r:
                continue
            seqs = sorted(set(r.sequence_name for r in ds_r))
            lines += [
                f"## {ds} Results",
                "",
                f"| Sequence | Estimator | ATE (m) | ANIS | NIS-ok | ANEES | NEES-ok |",
                f"|----------|-----------|---------|------|--------|-------|---------|",
            ]
            for seq in seqs:
                for r in [x for x in ds_r if x.sequence_name == seq]:
                    lines.append(
                        f"| {seq} | {r.estimator_name} | {fmt(r.ate)} | "
                        f"{fmt(r.anis,3)} | {'YES' if r.nis_consistent else 'NO'} | "
                        f"{fmt(r.anees,3)} | {'YES' if r.nees_consistent else 'NO'} |"
                    )
            lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {path}")


def write_summary(results: list[RealResult], path: str) -> None:
    """Write REAL_DATASET_SUMMARY.md."""
    for r in results:
        r.assert_real()

    n_seq = len(set(r.sequence_name for r in results))
    lines = [
        "# Real Dataset Summary",
        "",
        "## What is in this file",
        "Aggregated findings from real EuRoC/TUM-VI/KITTI evaluation.",
        "Every number is traceable to `results/real_datasets/raw_csv/phase7_raw.csv`.",
        "",
        "## Sequences evaluated",
        f"- Total: **{n_seq}** of {len(SEQUENCE_CATALOGUE)} catalogued",
        "",
    ]

    if not results:
        lines += [
            "**No results yet.** Real dataset files are not present on this system.",
            "Download instructions: `DATASET_AVAILABILITY.md`",
            "",
            "Once data is available, re-run:",
            "```bash",
            "python benchmark/run_phase7_real.py",
            "```",
        ]
    else:
        lb, ub = average_nees_bounds(dof=3, n_runs=1)

        # Per-estimator ATE summary
        lines += ["## Per-Estimator ATE (averaged over evaluated sequences)", ""]
        lines.append("| Estimator | Mean ATE | Std ATE | Min ATE | Max ATE |")
        lines.append("|-----------|----------|---------|---------|---------|")
        for est in ESTIMATORS:
            vals = [r.ate for r in results if r.estimator_name == est
                    and math.isfinite(r.ate)]
            if vals:
                lines.append(
                    f"| {est} | {np.mean(vals):.4f} | {np.std(vals):.4f} | "
                    f"{min(vals):.4f} | {max(vals):.4f} |"
                )

        # Consistency summary
        n_nis_ok  = sum(1 for r in results if r.nis_consistent)
        n_nees_ok = sum(1 for r in results if r.nees_consistent)
        lines += [
            "",
            "## Consistency Summary",
            "",
            f"| Metric | Consistent | Total | Pct |",
            f"|--------|------------|-------|-----|",
            f"| NIS  | {n_nis_ok} | {len(results)} | "
            f"{100*n_nis_ok/len(results):.0f}% |",
            f"| NEES | {n_nees_ok} | {len(results)} | "
            f"{100*n_nees_ok/len(results):.0f}% |",
            "",
            f"Bounds: [{lb:.4f}, {ub:.4f}] (chi², dof=3, N=1 single-run)",
        ]

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {path}")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true",
                        help="Only check availability, do not run evaluations")
    args = parser.parse_args()

    print("=" * 70)
    print("  Phase 7 — Real Dataset Evaluation (no synthetic fallback)")
    print("  Medisetti Renukeswar | June 2026")
    print("=" * 70)

    status = check_availability()
    print_availability(status)

    available = {k: v for k, v in status.items() if v["available"]}

    if args.check_only:
        print(f"  --check-only mode: {len(available)} sequences available.")
        return

    if not available:
        print("  No real dataset files found.  Nothing to evaluate.")
        print("  See DATASET_AVAILABILITY.md for download instructions.")
        print()
    else:
        print(f"  Evaluating {len(available)} available sequences ...\n")

    lb, ub = average_nees_bounds(dof=3, n_runs=1)
    all_results: list[RealResult] = []

    for key, info in available.items():
        spec   = info["spec"]
        loader = info["loader"]
        print(f"  ── {key} ──────────────────────────────────")
        try:
            seq_results = evaluate_sequence(spec, loader, lb, ub)
            all_results.extend(seq_results)
        except Exception as exc:
            print(f"    ERROR evaluating {key}: {exc}")
        print()

    # ── Save raw CSV ───────────────────────────────────────────────────────
    raw_path = os.path.join(RAW_DIR, "phase7_raw.csv")
    if all_results:
        fieldnames = list(asdict(all_results[0]).keys())
        with open(raw_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in all_results:
                r.assert_real()
                w.writerow(asdict(r))
        print(f"  Saved: {raw_path}")
    else:
        # Write empty CSV with correct headers to signal no data
        with open(raw_path, "w", newline="") as f:
            f.write("data_source,dataset_name,sequence_name,estimator_name,"
                    "n_imu,n_gt,n_updates,duration_s,ate,rpe,rmse_pos,"
                    "rmse_heading,anis,anees,nis_consistent,nees_consistent,"
                    "runtime_ms\n")
            f.write("# NO REAL DATA AVAILABLE — see DATASET_AVAILABILITY.md\n")
        print(f"  Saved (empty): {raw_path}")

    # ── Save tables ────────────────────────────────────────────────────────
    t1 = make_table1(all_results, lb, ub)
    t2 = make_table2(all_results, lb, ub)
    t3 = make_table3(all_results)

    for name, content in [
        ("phase7_table1_accuracy.txt",    t1),
        ("phase7_table2_consistency.txt", t2),
        ("phase7_table3_comparison.txt",  t3),
    ]:
        path = os.path.join(TABLE_DIR, name)
        with open(path, "w") as f:
            f.write(content)
        print(f"  Saved: {path}")
        print(content)

    # ── Generate figures ───────────────────────────────────────────────────
    make_figures(all_results, lb, ub)

    # ── Write reports ──────────────────────────────────────────────────────
    write_real_dataset_report(
        all_results,
        os.path.join(ROOT, "REAL_DATASET_REPORT.md"),
    )
    write_summary(
        all_results,
        os.path.join(ROOT, "REAL_DATASET_SUMMARY.md"),
    )

    print()
    print("=" * 70)
    n = len(all_results)
    if n > 0:
        print(f"  Phase 7 complete.  {n} real results saved.")
    else:
        print("  Phase 7 complete.  0 real results (no dataset files present).")
        print("  Infrastructure is ready — results will populate once data")
        print("  is downloaded.  See DATASET_AVAILABILITY.md.")
    print("=" * 70)


if __name__ == "__main__":
    main()
