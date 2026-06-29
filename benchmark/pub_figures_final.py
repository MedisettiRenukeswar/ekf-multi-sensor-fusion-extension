"""
Final Publication Figure Generator — N=50 data, honest ablation results.
All figures IEEE-format: white bg, 300 DPI, DejaVu Serif, PDF+PNG.
Author: Medisetti Renukeswar
"""
from __future__ import annotations
import csv, math, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    "font.family": "DejaVu Serif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7.5, "legend.framealpha": 0.9,
    "figure.dpi": 300, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.4, "axes.grid": True,
    "grid.alpha": 0.3, "grid.linewidth": 0.5,
    "axes.facecolor": "white", "figure.facecolor": "white",
})

COLORS = {
    "EKF": "#0072BD", "UKF": "#D95319",
    "Adaptive-EKF": "#EDB120", "Adaptive-UKF": "#7E2F8E",
    "ES-EKF": "#77AC30", "MACE-EKF": "#4DBEEE", "MACE-UKF": "#A2142F",
    "EKF-Fixed": "#0072BD", "ChiGate-EKF": "#77AC30",
    "Huber-EKF": "#FF6600", "Tukey-EKF": "#AA00AA",
}
MARKERS = {
    "EKF": "o", "UKF": "s", "Adaptive-EKF": "^", "Adaptive-UKF": "D",
    "ES-EKF": "v", "MACE-EKF": "*", "MACE-UKF": "P",
    "EKF-Fixed": "o", "ChiGate-EKF": "v",
    "Huber-EKF": "^", "Tukey-EKF": "s",
}
ESTIMATORS = ["EKF", "UKF", "Adaptive-EKF", "Adaptive-UKF",
              "ES-EKF", "MACE-EKF", "MACE-UKF"]
ABLATION_NAMES = ["EKF-Fixed", "ChiGate-EKF", "Adaptive-EKF",
                  "Huber-EKF", "Tukey-EKF", "MACE-EKF"]

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES  = os.path.join(REPO, "results")
FIG  = os.path.join(REPO, "paper", "figures")
os.makedirs(FIG, exist_ok=True)

sys.path.insert(0, REPO)
from ekf_core.metrics import average_nees_bounds

# Bounds per N
LB50, UB50 = average_nees_bounds(dof=3, n_runs=50)  # 0.787, 1.239
LB25, UB25 = average_nees_bounds(dof=3, n_runs=25)  # 0.706, 1.345
LB30, UB30 = average_nees_bounds(dof=3, n_runs=30)  # 0.729, 1.313

def N_bounds(est):
    """Return appropriate bounds based on N for this estimator."""
    if est in ["UKF", "Adaptive-UKF", "MACE-UKF"]:
        return LB25, UB25
    return LB50, UB50

def load(f):
    p = os.path.join(RES, f)
    if not os.path.exists(p):
        return []
    rows = list(csv.DictReader(open(p)))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except Exception:
                pass
    return rows

def savefig(fig, name):
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"),
                    bbox_inches="tight", dpi=300,
                    facecolor="white", edgecolor="none")
    print(f"  {name}.pdf/.png")
    plt.close(fig)

def get_row(rows, **kw):
    for r in rows:
        if all(abs(r.get(k, -999) - v) < 0.01
               if isinstance(v, float) else str(r.get(k, "")) == str(v)
               for k, v in kw.items()):
            return r
    return None

# ── Figure 1: ATE grouped bar — all 7 estimators, figure-8 ───────────────
def fig1_ate_bar(std):
    noise_labels = ["Low", "Medium", "High"]
    noise_keys   = ["low", "medium", "high"]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    x = np.arange(3)
    n = len(ESTIMATORS)
    w = 0.10
    offsets = np.linspace(-(n-1)/2*w, (n-1)/2*w, n)
    for i, est in enumerate(ESTIMATORS):
        means, errs = [], []
        for nk in noise_keys:
            r = get_row(std, estimator=est, trajectory="figure8", noise_regime=nk)
            means.append(r["ate_mean"] if r else 0)
            errs.append(r["ate_ci"]    if r else 0)
        ax.bar(x + offsets[i], means, w*0.88, yerr=errs,
               color=COLORS[est], label=est, capsize=2,
               error_kw={"linewidth": 0.7})
    ax.set_xticks(x); ax.set_xticklabels(noise_labels)
    ax.set_ylabel("ATE RMSE (m)")
    ax.set_title("(a) Accuracy: ATE by Estimator × Noise\n(figure-8, N≥25 MC, 95% CI)")
    ax.legend(ncol=2, fontsize=6.5, loc="upper left")
    fig.tight_layout()
    savefig(fig, "fig1_ate_bar")

# ── Figure 2: ANIS consistency heatmap ───────────────────────────────────
def fig2_anis_heatmap(std):
    noise_keys = ["low", "medium", "high"]
    mat = np.zeros((len(ESTIMATORS), 3))
    for i, est in enumerate(ESTIMATORS):
        lb, ub = N_bounds(est)
        for j, nk in enumerate(noise_keys):
            r = get_row(std, estimator=est, trajectory="figure8", noise_regime=nk)
            mat[i, j] = r["anis_mean"] if r else float("nan")
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0.0, vmax=1.5, aspect="auto")
    plt.colorbar(im, ax=ax, label="ANIS", fraction=0.04)
    for i, est in enumerate(ESTIMATORS):
        lb, ub = N_bounds(est)
        for j in range(3):
            v = mat[i, j]
            ok = "in" if lb <= v <= ub else "out"
            tc = "black" if 0.25 < v < 1.3 else "white"
            ax.text(j, i, f"{v:.2f}\n({ok})", ha="center", va="center",
                    fontsize=6, color=tc, weight="bold")
    ax.set_xticks([0,1,2]); ax.set_xticklabels(["Low", "Med", "High"])
    ax.set_yticks(range(len(ESTIMATORS))); ax.set_yticklabels(ESTIMATORS, fontsize=7)
    ax.set_title("(b) ANIS Consistency\n(in/out = within 95% χ² bounds)")
    fig.tight_layout()
    savefig(fig, "fig2_anis_heatmap")

# ── Figure 3: ANEES heatmap ───────────────────────────────────────────────
def fig3_anees_heatmap(std):
    noise_keys = ["low", "medium", "high"]
    mat = np.zeros((len(ESTIMATORS), 3))
    for i, est in enumerate(ESTIMATORS):
        for j, nk in enumerate(noise_keys):
            r = get_row(std, estimator=est, trajectory="figure8", noise_regime=nk)
            mat[i, j] = r["anees_mean"] if r else float("nan")
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0.0, vmax=1.5, aspect="auto")
    plt.colorbar(im, ax=ax, label="ANEES", fraction=0.04)
    for i, est in enumerate(ESTIMATORS):
        lb, ub = N_bounds(est)
        for j in range(3):
            v = mat[i, j]
            ok = "in" if lb <= v <= ub else "out"
            tc = "black" if 0.25 < v < 1.3 else "white"
            ax.text(j, i, f"{v:.2f}\n({ok})", ha="center", va="center",
                    fontsize=6, color=tc, weight="bold")
    ax.set_xticks([0,1,2]); ax.set_xticklabels(["Low", "Med", "High"])
    ax.set_yticks(range(len(ESTIMATORS))); ax.set_yticklabels(ESTIMATORS, fontsize=7)
    ax.set_title("(c) ANEES Consistency\n(in/out = within 95% χ² bounds)")
    fig.tight_layout()
    savefig(fig, "fig3_anees_heatmap")

# ── Figure 4: ANIS vs noise — 3 trajectories ─────────────────────────────
def fig4_anis_lines(std):
    noise_x = [0, 1, 2]
    trajs = ["figure8", "circle", "straight"]
    titles = ["(d) Figure-8", "(e) Circle", "(f) Straight-line"]
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.4), sharey=True)
    for ax, traj, title in zip(axes, trajs, titles):
        for est in ESTIMATORS:
            lb, ub = N_bounds(est)
            vals = []
            for nk in ["low", "medium", "high"]:
                r = get_row(std, estimator=est, trajectory=traj, noise_regime=nk)
                vals.append(r["anis_mean"] if r else float("nan"))
            ax.plot(noise_x, vals, marker=MARKERS[est],
                    color=COLORS[est], label=est, markersize=4)
        # Draw bounds — use N=50 bounds as reference
        ax.axhline(LB50, ls="--", lw=0.9, color="gray")
        ax.axhline(UB50, ls="--", lw=0.9, color="gray")
        ax.fill_between(noise_x, LB50, UB50, alpha=0.07, color="green")
        ax.set_xticks(noise_x); ax.set_xticklabels(["Low", "Med", "High"])
        ax.set_title(title); ax.set_ylim(-0.05, 1.55)
    axes[0].set_ylabel("ANIS")
    axes[1].set_xlabel("Noise regime")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("ANIS vs Noise Regime (shaded = 95% χ² consistent band, N=50)",
                 fontsize=9)
    fig.tight_layout()
    savefig(fig, "fig4_anis_lines")

# ── Figure 5: Dropout — ATE and ANIS lines ───────────────────────────────
def fig5_dropout(drop):
    drs = [0.0, 0.10, 0.30, 0.50, 0.70]
    drx = [d*100 for d in drs]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.5))
    for est in ESTIMATORS:
        lb, ub = N_bounds(est)
        ate_v, anis_v = [], []
        for dr in drs:
            r = get_row(drop, estimator=est, dropout=dr)
            ate_v.append(r["ate_mean"]   if r else float("nan"))
            anis_v.append(r["anis_mean"] if r else float("nan"))
        ax1.plot(drx, ate_v, marker=MARKERS[est], color=COLORS[est],
                 label=est, markersize=4)
        ax2.plot(drx, anis_v, marker=MARKERS[est], color=COLORS[est],
                 label=est, markersize=4)
    ax2.axhline(LB25, ls="--", lw=0.9, color="gray", alpha=0.7)
    ax2.axhline(UB25, ls="--", lw=0.9, color="gray", alpha=0.7)
    ax2.fill_between(drx, LB25, UB25, alpha=0.06, color="green")
    ax1.set_xlabel("VO Dropout (%)"); ax1.set_ylabel("ATE RMSE (m)")
    ax1.set_title("(g) ATE vs VO Dropout")
    ax2.set_xlabel("VO Dropout (%)"); ax2.set_ylabel("ANIS")
    ax2.set_title("(h) Consistency vs VO Dropout\n(shaded = consistent band)")
    ax2.set_ylim(-0.05, 2.2)
    h, l = ax1.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7,
               bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    savefig(fig, "fig5_dropout")

# ── Figure 6: Ablation — MACE vs alternatives ────────────────────────────
def fig6_ablation(abl):
    drs = [0.0, 0.10, 0.30, 0.50, 0.70]
    drx = [d*100 for d in drs]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.6))
    for est in ABLATION_NAMES:
        ate_v, anis_v = [], []
        for dr in drs:
            r = get_row(abl, estimator=est, dropout=dr)
            ate_v.append(r["ate_mean"]   if r else float("nan"))
            anis_v.append(r["anis_mean"] if r else float("nan"))
        lw = 2.0 if est in ("MACE-EKF", "Adaptive-EKF") else 1.2
        ls = "-" if est in ("MACE-EKF", "Adaptive-EKF") else "--"
        ax1.plot(drx, ate_v, marker=MARKERS.get(est,"o"),
                 color=COLORS.get(est,"gray"), label=est,
                 markersize=4, linewidth=lw, linestyle=ls)
        ax2.plot(drx, anis_v, marker=MARKERS.get(est,"o"),
                 color=COLORS.get(est,"gray"), label=est,
                 markersize=4, linewidth=lw, linestyle=ls)
    ax2.axhline(LB30, ls=":", lw=1.0, color="green", label=f"Bounds [{LB30:.2f},{UB30:.2f}]")
    ax2.axhline(UB30, ls=":", lw=1.0, color="green")
    ax2.fill_between(drx, LB30, UB30, alpha=0.07, color="green")
    ax1.set_xlabel("VO Dropout (%)"); ax1.set_ylabel("ATE RMSE (m)")
    ax1.set_title("(i) Ablation: Accuracy\n(bold = MACE-EKF vs Adaptive-EKF)")
    ax2.set_xlabel("VO Dropout (%)"); ax2.set_ylabel("ANIS")
    ax2.set_title("(j) Ablation: Consistency\n(N=30, shaded = χ² bounds)")
    ax2.set_ylim(-0.05, 2.3)
    h, l = ax1.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7,
               bbox_to_anchor=(0.5, -0.10))
    fig.tight_layout()
    savefig(fig, "fig6_ablation")

# ── Figure 7: MACE threshold sweep ───────────────────────────────────────
def fig7_threshold(thresh):
    if not thresh:
        return
    taus = sorted(set(r["threshold"] for r in thresh if r["threshold"] < 1e9))
    taus_plot = taus + [25.0]  # stand-in for inf on plot
    labels = [f"{t:.0f}" for t in taus] + ["∞\n(Adaptive)"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.0, 2.5))
    for dr, color, ls, label in [(0.0,"#0072BD","-","0% dropout"),
                                   (0.30,"#EDB120","--","30% dropout"),
                                   (0.50,"#D95319","-.","50% dropout")]:
        ate_v, anis_v = [], []
        for tau in taus:
            r = get_row(thresh, threshold=tau, dropout=dr)
            ate_v.append(r["ate_mean"]   if r else float("nan"))
            anis_v.append(r["anis_mean"] if r else float("nan"))
        # inf = vanilla adaptive
        r_inf = get_row(thresh, threshold=float("inf"), dropout=dr)
        ate_v.append(r_inf["ate_mean"]   if r_inf else float("nan"))
        anis_v.append(r_inf["anis_mean"] if r_inf else float("nan"))
        ax1.plot(range(len(taus_plot)), ate_v, color=color,
                 linestyle=ls, marker="o", markersize=3.5, label=label)
        ax2.plot(range(len(taus_plot)), anis_v, color=color,
                 linestyle=ls, marker="o", markersize=3.5, label=label)
    ax2.axhline(LB30, ls=":", lw=0.9, color="green")
    ax2.axhline(UB30, ls=":", lw=0.9, color="green")
    ax2.fill_between(range(len(taus_plot)), LB30, UB30, alpha=0.07, color="green")
    for ax, ylabel, title in [
        (ax1,"ATE RMSE (m)","(k) ATE vs Chi-sq Gate Threshold"),
        (ax2,"ANIS","(l) ANIS vs Chi-sq Gate Threshold"),
    ]:
        ax.set_xticks(range(len(taus_plot)))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_xlabel("Gate threshold τ")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7)
    ax2.set_ylim(-0.05, 2.5)
    fig.suptitle("MACE-EKF Chi-sq Threshold Sensitivity\n(figure-8, medium noise, N=20 MC)",
                 fontsize=9)
    fig.tight_layout()
    savefig(fig, "fig7_threshold")

# ── Figure 8: Hyperparameter heatmap ─────────────────────────────────────
def fig8_hyperparam(hp):
    if not hp:
        return
    W_vals = sorted(set(int(r["W"]) for r in hp))
    A_vals = sorted(set(float(r["alpha"]) for r in hp))
    ate_m  = np.full((len(W_vals), len(A_vals)), float("nan"))
    anis_m = np.full((len(W_vals), len(A_vals)), float("nan"))
    for r in hp:
        i = W_vals.index(int(r["W"]))
        j = A_vals.index(float(r["alpha"]))
        ate_m[i,j]  = float(r["ate_mean"])
        anis_m[i,j] = float(r["anis_mean"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.4))
    for ax, mat, label, title, cmap in [
        (ax1, ate_m,  "ATE (m)",  "(m) ATE", "YlOrRd"),
        (ax2, anis_m, "ANIS",     "(n) ANIS", "RdYlGn"),
    ]:
        im = ax.imshow(mat, cmap=cmap, vmin=np.nanmin(mat),
                       vmax=np.nanmax(mat), aspect="auto")
        plt.colorbar(im, ax=ax, label=label, fraction=0.04)
        for i in range(len(W_vals)):
            for j in range(len(A_vals)):
                v = mat[i, j]
                mark = ""
                if label == "ANIS":
                    mark = "*" if LB30 <= v <= UB30 else ""
                ax.text(j, i, f"{v:.3f}{mark}", ha="center", va="center",
                        fontsize=7)
        ax.set_xticks(range(len(A_vals)))
        ax.set_xticklabels([f"α={a}" for a in A_vals])
        ax.set_yticks(range(len(W_vals)))
        ax.set_yticklabels([f"W={w}" for w in W_vals])
        ax.set_title(title)
    fig.suptitle("Adaptive-EKF Hyperparameter Sensitivity (* = consistent)",
                 fontsize=9)
    fig.tight_layout()
    savefig(fig, "fig8_hyperparam")

# ── Figure 9: Runtime ─────────────────────────────────────────────────────
def fig9_runtime(std):
    names, rts, stds = [], [], []
    for est in ESTIMATORS:
        r = get_row(std, estimator=est, trajectory="figure8", noise_regime="medium")
        if r:
            names.append(est)
            rts.append(r["rt_mean"])
            stds.append(r["rt_std"])
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.barh(range(len(names)), rts, xerr=stds,
            color=[COLORS[n] for n in names],
            capsize=3, error_kw={"linewidth": 0.7}, height=0.6)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("Runtime per 40 s trajectory (ms)")
    ax.set_title("(o) Runtime Comparison\n(log scale, N=50 MC, figure-8, medium noise)")
    ax.axvline(1000, ls="--", lw=0.8, color="red", alpha=0.6, label="1 s boundary")
    ax.legend(fontsize=7)
    fig.tight_layout()
    savefig(fig, "fig9_runtime")

# ── Figure 10: MACE-UKF vs Adaptive-UKF (key UKF finding) ───────────────
def fig10_ukf_mace(drop):
    drs = [0.0, 0.10, 0.30, 0.50, 0.70]
    drx = [d*100 for d in drs]
    targets = ["UKF", "Adaptive-UKF", "MACE-UKF"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.6))
    for est in targets:
        ate_v, anis_v = [], []
        for dr in drs:
            r = get_row(drop, estimator=est, dropout=dr)
            ate_v.append(r["ate_mean"]   if r else float("nan"))
            anis_v.append(r["anis_mean"] if r else float("nan"))
        lw = 2.2 if "MACE" in est else 1.4
        ax1.plot(drx, ate_v, marker=MARKERS[est], color=COLORS[est],
                 label=est, linewidth=lw, markersize=5)
        ax2.plot(drx, anis_v, marker=MARKERS[est], color=COLORS[est],
                 label=est, linewidth=lw, markersize=5)
    ax2.axhline(LB25, ls="--", lw=1.0, color="green", label=f"Bounds N=25")
    ax2.axhline(UB25, ls="--", lw=1.0, color="green")
    ax2.fill_between(drx, LB25, UB25, alpha=0.10, color="green")
    # Annotate 70% dropout
    for est in targets:
        r = get_row(drop, estimator=est, dropout=0.70)
        if r:
            ax2.annotate(f"{r['anis_mean']:.3f}",
                         xy=(70, r["anis_mean"]),
                         xytext=(72, r["anis_mean"]+0.02),
                         fontsize=6.5, color=COLORS[est])
    ax1.set_xlabel("VO Dropout (%)"); ax1.set_ylabel("ATE RMSE (m)")
    ax1.set_title("(p) UKF Family: ATE vs Dropout")
    ax2.set_xlabel("VO Dropout (%)"); ax2.set_ylabel("ANIS")
    ax2.set_title("(q) UKF Family: Consistency vs Dropout\n"
                  "(MACE-UKF consistent at 70% dropout)")
    ax2.set_ylim(0.0, 1.8)
    ax1.legend(fontsize=8); ax2.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "fig10_ukf_mace")

# ── Figure 11: 4-panel summary ───────────────────────────────────────────
def fig11_summary(std, drop, abl):
    names_s = ["EKF","UKF","A-EKF","A-UKF","ES-EKF","MACE-EKF","MACE-UKF"]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0))
    cols = [COLORS[e] for e in ESTIMATORS]
    # Panel 1: ATE medium figure-8
    ax = axes[0,0]
    ates = []; errs = []
    for est in ESTIMATORS:
        r = get_row(std, estimator=est, trajectory="figure8", noise_regime="medium")
        ates.append(r["ate_mean"] if r else 0)
        errs.append(r["ate_ci"]   if r else 0)
    ax.bar(range(7), ates, yerr=errs, color=cols, capsize=2,
           error_kw={"linewidth":0.7})
    ax.set_xticks(range(7)); ax.set_xticklabels(names_s, rotation=35,
                                                  ha="right", fontsize=7)
    ax.set_ylabel("ATE RMSE (m)")
    ax.set_title("1. Accuracy (fig-8, medium, N≥25)")
    # Panel 2: ANIS medium figure-8
    ax = axes[0,1]
    aniss = []
    for est in ESTIMATORS:
        r = get_row(std, estimator=est, trajectory="figure8", noise_regime="medium")
        aniss.append(r["anis_mean"] if r else 0)
    ax.bar(range(7), aniss, color=cols)
    ax.axhline(LB50, ls="--", lw=0.9, color="green")
    ax.axhline(UB50, ls="--", lw=0.9, color="green")
    ax.fill_between([-0.5,6.5], LB50, UB50, alpha=0.08, color="green")
    ax.set_xticks(range(7)); ax.set_xticklabels(names_s, rotation=35,
                                                  ha="right", fontsize=7)
    ax.set_ylabel("ANIS"); ax.set_ylim(0, 1.5)
    ax.set_title("2. Consistency (fig-8, medium, N≥25)")
    # Panel 3: Ablation ATE at 50% dropout
    ax = axes[1,0]
    abl_colors = [COLORS.get(e,"gray") for e in ABLATION_NAMES]
    ates3 = []
    for est in ABLATION_NAMES:
        r = get_row(abl, estimator=est, dropout=0.50)
        ates3.append(r["ate_mean"] if r else 0)
    bars = ax.bar(range(6), ates3, color=abl_colors)
    short = ["EKF-Fix","ChiGate","Adaptive","Huber","Tukey","MACE"]
    ax.set_xticks(range(6)); ax.set_xticklabels(short, rotation=30,
                                                  ha="right", fontsize=7)
    ax.set_ylabel("ATE RMSE (m)")
    ax.set_title("3. Ablation: ATE at 50% Dropout (N=30)")
    # Panel 4: MACE-UKF vs Adap-UKF ANIS vs dropout
    ax = axes[1,1]
    drs=[0.0,0.10,0.30,0.50,0.70]
    drx=[d*100 for d in drs]
    for est in ["Adaptive-UKF","MACE-UKF"]:
        v=[]; 
        for dr in drs:
            r=get_row(drop,estimator=est,dropout=dr)
            v.append(r["anis_mean"] if r else float("nan"))
        ax.plot(drx,v,marker=MARKERS[est],color=COLORS[est],
                label=est,linewidth=2.0,markersize=5)
    ax.axhline(LB25,ls="--",lw=0.9,color="green")
    ax.axhline(UB25,ls="--",lw=0.9,color="green")
    ax.fill_between(drx,LB25,UB25,alpha=0.10,color="green")
    ax.set_xlabel("VO Dropout (%)"); ax.set_ylabel("ANIS")
    ax.set_title("4. MACE-UKF Consistent at 70% Dropout")
    ax.set_ylim(0.4, 1.5); ax.legend(fontsize=8)
    fig.suptitle("Key Research Findings — N≥25 MC, Simulation Only",
                 fontsize=10, weight="bold")
    fig.tight_layout()
    savefig(fig, "fig11_summary")

# ── Figure 12: Trajectory visualisation ──────────────────────────────────
def fig12_trajectories():
    from simulation.trajectories import TrajectoryGenerator
    from simulation.research_sensor_sim import ResearchSensorSimulator
    from ekf_core.ekf_estimator import EKFEstimator
    from ekf_core.ukf_estimator import UKFEstimator
    from ekf_core.mace_estimator import MACEUKFEstimator
    DT_IMU=0.01; DT_CAM=1/30; SIM=40.0
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8))
    for ax, tt, title in zip(axes,
            ["figure8","circle","straight"],
            ["(r) Figure-8","(s) Circle","(t) Straight"]):
        traj = TrajectoryGenerator(trajectory_type=tt, scale=3.0)
        sim  = ResearchSensorSimulator(trajectory=traj, noise_regime="medium",
                                       dt_imu=DT_IMU, dt_cam=DT_CAM, seed=42)
        gt_x, gt_y = [], []
        est_data = {n: ([], []) for n in ["EKF","UKF","MACE-UKF"]}
        ests = {"EKF": EKFEstimator(), "UKF": UKFEstimator(),
                "MACE-UKF": MACEUKFEstimator()}
        px0,py0,th0,vx0,vy0,om0 = traj.get_state(0.0)
        x0 = np.array([px0,py0,th0,vx0,vy0,om0])
        P0 = np.diag([0.5,0.5,0.3,0.5,0.5,0.1])
        for e in ests.values(): e.reset(x0, P0)
        t=0.0; cam_t=0.0; step=0
        while t <= SIM:
            px_gt,py_gt,*_ = traj.get_state(t)
            vxm,vym,omm = sim.get_imu(t)
            for e in ests.values(): e.predict(vxm,vym,omm)
            cam_t += DT_IMU
            if cam_t >= DT_CAM:
                cam_t=0.0
                pxc,pyc,thc = sim.get_camera(t)
                for e in ests.values(): e.update_camera(pxc,pyc,thc)
            if step % 10 == 0:
                gt_x.append(px_gt); gt_y.append(py_gt)
                for n, e in ests.items():
                    xe,ye,_ = e.get_position()
                    est_data[n][0].append(xe); est_data[n][1].append(ye)
            t += DT_IMU; step += 1
        ax.plot(gt_x, gt_y, "k-", lw=2.0, label="Ground Truth", zorder=5)
        for n in ["EKF","UKF","MACE-UKF"]:
            xs,ys = est_data[n]
            ax.plot(xs, ys, "-", color=COLORS[n], lw=1.1,
                    label=n, alpha=0.85)
        ax.set_aspect("equal"); ax.set_title(title, fontsize=9)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7.5,
               bbox_to_anchor=(0.5,-0.04))
    fig.tight_layout()
    savefig(fig, "fig12_trajectories")

def main():
    print("Loading data...")
    std    = load("n50_stats_standard.csv")
    drop   = load("n50_stats_dropout.csv")
    abl    = load("ablation_stats.csv")
    hp     = load("hyperparam_sensitivity.csv")
    thresh = load("mace_threshold_sweep.csv")
    print(f"  standard={len(std)} dropout={len(drop)} "
          f"ablation={len(abl)} hp={len(hp)} threshold={len(thresh)}")
    print("Generating figures...")
    fig1_ate_bar(std)
    fig2_anis_heatmap(std)
    fig3_anees_heatmap(std)
    fig4_anis_lines(std)
    fig5_dropout(drop)
    fig6_ablation(abl)
    fig7_threshold(thresh)
    fig8_hyperparam(hp)
    fig9_runtime(std)
    fig10_ukf_mace(drop)
    fig11_summary(std, drop, abl)
    fig12_trajectories()
    print(f"\nAll figures saved to: {FIG}")

if __name__ == "__main__":
    main()
