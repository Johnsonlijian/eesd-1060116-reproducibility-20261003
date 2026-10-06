"""Figure 6: transferred full-scale semi-active MR bank versus active references.

The figure uses the validated 3-story Ohtori plant and the FEMA moderate plus deep-yield
records. The Yang--Spencer--Carlson--Sain (2002) 200-kN device is transferred as three
devices per floor; the comparison preserves the datasheet force/current limits and does
not claim exact force matching or frame-specific calibration.

All values are read from ``results/eesd_semiactive_mr_fullscale.json``. The primary
closed-loop comparison gives mean J values of 0.954 (powered MR), 0.848 (unpowered MR),
0.625 (clipped-LQR reference), and 0.593 (in-QP reference). The powered MR bank reduces
peak interstory drift by 20% on average (range 2--36%); the unpowered bank reduces it by
22%. The message is therefore device-transfer performance: MR reduces drift, while the
field-controlled increment is passive-dominated and the active reference cells have lower J.

Run: ``python paper_figures/src/fig_eesd_semiactive.py``
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results"
OUT = {ext: PROJECT_ROOT / "paper_figures" / "output" / ext for ext in ("svg", "pdf", "png")}

C_INK = "#111827"
C_GREY = "#6b7280"
C_UNC = "#9ca3af"     # uncontrolled
C_MRC = "#c2410c"     # MR clipped-optimal (powered)
C_MRO = "#f59e0b"     # MR unpowered passive
C_ACL = "#1d4ed8"     # active clipped-LQR
C_AQP = "#15803d"     # active in-QP
C_REF = "#b91c1c"     # uncontrolled = 1.0 reference


def _load():
    with open(RESULTS / "eesd_semiactive_mr_fullscale.json", encoding="utf-8") as f:
        return json.load(f)


def fig_semiactive():
    d = _load()
    bar1 = d["BAR1_reduces_vs_uncontrolled_3story"]
    bar2 = d["BAR2_vs_active_and_avoids_failures_3story"]
    field = bar1["field_control_value_vs_unpowered"]
    rows = d["per_row_3story"]

    # ---- mean J per strategy, from the summary blocks (manuscript-matched rounding) ----
    J_mrclip = bar1["primary_J"]["mean"]            # 0.955
    J_mroff = field["mr_off_J_mean"]                # 0.848
    J_actclip = bar2["mean_J_act_clipLQR"]          # 0.625
    J_actinqp = bar2["mean_J_act_inqp"]             # 0.593

    # ---- per-case J for the overlaid points ----
    per = {
        "unc": [1.0] * len(rows),
        "mrclip": [r["mr_clip_J"] for r in rows],
        "mroff": [r["mr_off_J"] for r in rows],
        "actclip": [r["act_clipLQR_J"] for r in rows],
        "actinqp": [r["act_inqp_J"] for r in rows],
    }

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.4, 4.8),
                                   gridspec_kw={"width_ratios": [1.55, 1.0]})

    # ========================= Panel A: cost J =========================
    labels = ["Uncontrolled", "MR\nclipped-opt.\n(powered)", "MR\nzero-current",
              "Active\nclipped-LQR", "Active\nin-QP"]
    means = [1.0, J_mrclip, J_mroff, J_actclip, J_actinqp]
    colors = [C_UNC, C_MRC, C_MRO, C_ACL, C_AQP]
    perkeys = ["unc", "mrclip", "mroff", "actclip", "actinqp"]
    x = np.arange(len(labels))

    axA.axhline(1.0, color=C_REF, lw=1.3, ls="--", zorder=2)
    axA.text(len(labels) - 0.5, 1.01, "uncontrolled = 1.0", ha="right", va="bottom",
             fontsize=7.6, color=C_REF, weight="bold")
    axA.bar(x, means, color=colors, alpha=0.85, width=0.66, zorder=3)
    # per-case markers (jittered) to show the spread behind each mean
    rng = np.random.default_rng(7)
    for xi, k in zip(x, perkeys):
        if k == "unc":
            continue
        pts = per[k]
        jit = (rng.random(len(pts)) - 0.5) * 0.30
        axA.scatter(xi + jit, pts, s=13, color="#1f2937", alpha=0.48, zorder=5,
                    edgecolors="white", linewidths=0.4)
    for xi, v in zip(x, means):
        axA.text(xi, v + 0.018, f"{v:.3f}", ha="center", va="bottom",
                 fontsize=8.8, weight="bold", color=C_INK,
                 bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.78))

    # annotate the two organizing facts
    axA.text(1.55, 1.245,
             f"powered $J$ is +{field['primary_minus_off_dJ_mean']:.3f} higher\nthan zero-current MR",
             ha="center", va="center", fontsize=6.8, color=C_MRC,
             bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=C_MRC, lw=0.8), zorder=7)
    axA.text(3.55, 1.155, "device transfer:\nactive reference cells lower $J$",
             ha="center", va="center", fontsize=6.8, color=C_AQP,
             bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=C_AQP, lw=0.8), zorder=7)

    axA.set_xticks(x); axA.set_xticklabels(labels, fontsize=7.8)
    axA.set_ylabel("normalized control cost  $J$  (uncontrolled = 1.0)", fontsize=8.8)
    axA.set_ylim(0, 1.32)
    axA.grid(axis="y", color="#e5e7eb", lw=0.5, zorder=0)
    axA.set_title("A  Control cost: powered versus zero-current operation",
                  fontsize=8.8, weight="bold", loc="left")

    # ========================= Panel B: drift reduction =========================
    red_clip = bar1["drift_reduction_pct"]
    red_off_mean = field["mr_off_drift_reduction_pct_mean"]
    red_clip_per = [r["mr_clip_drift_red_pct"] for r in rows]
    red_off_per = [r["mr_off_drift_red_pct"] for r in rows]

    xb = np.array([0, 1])
    means_b = [red_clip["mean"], red_off_mean]
    colors_b = [C_MRC, C_MRO]
    bars = axB.bar(xb, means_b, color=colors_b, alpha=0.85, width=0.55, zorder=3)
    # per-case markers
    for xi, pts in zip(xb, (red_clip_per, red_off_per)):
        jit = (np.random.default_rng(3).random(len(pts)) - 0.5) * 0.26
        axB.scatter(xi + jit, pts, s=13, color="#1f2937", alpha=0.48, zorder=5,
                    edgecolors="white", linewidths=0.4)
    for xi, v in zip(xb, means_b):
        axB.text(xi, v + 0.5, f"{v:.1f}%", ha="center", va="bottom",
                 fontsize=9.0, weight="bold", color=C_INK,
                 bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.78))
    # range note on the powered bar
    axB.text(0, red_clip["mean"] * 0.50,
             f"range {red_clip['min']:.0f}-{red_clip['max']:.0f}%\n(all reduce)",
             ha="center", va="center", fontsize=6.4, color="white", weight="bold")

    axB.set_xticks(xb)
    axB.set_xticklabels(["MR clipped-opt.\n(powered)", "MR zero-current"], fontsize=8.0)
    axB.set_ylabel("peak interstory-drift reduction\nvs uncontrolled (%)", fontsize=8.6)
    all_reductions = np.asarray(red_clip_per + red_off_per, dtype=float)
    axB.set_ylim(min(0.0, float(all_reductions.min())) - 2.0, float(all_reductions.max()) + 3.0)
    axB.axhline(0.0, color="#6b7280", lw=0.8)
    axB.grid(axis="y", color="#e5e7eb", lw=0.5, zorder=0)
    axB.set_title("B  Mean peak-drift reduction: powered versus\nzero-current operation",
                  fontsize=8.8, weight="bold", loc="left")

    fig.suptitle("Transferred full-scale MR bank (Yang et al. 2002, 200 kN x3/floor; not exactly force-matched): "
                 "powered and zero-current operation, with active reference controllers",
                 fontsize=9.8, weight="bold", y=0.995)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.24, wspace=0.26)
    _save(fig, "fig_eesd_semiactive")


def _save(fig, name):
    for ext, d in OUT.items():
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{name}.{ext}", dpi=300 if ext == "png" else None,
                    bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)
    print("wrote", name)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig_semiactive()


if __name__ == "__main__":
    main()




