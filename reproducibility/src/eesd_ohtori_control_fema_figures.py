r"""Figures for the E4 FEMA P695 robustness sweep (reads results/eesd_ohtori_control_results_fema.json).

Four clean panels, Agg backend, publication style (no draft/HONEST overlays):
  (A) 2x2 descriptive variance shares: pooled + per-tier, with the leave-one-record spread on the
      saturation share as an error bar -- the "saturation dominates" claim across the record family.
  (B) in-QP deep-yield failure: dJ = J_inQP - J_clip vs uncontrolled peak drift, colored win/lose,
      with the 3.5% deep-yield line -- the in-QP failure boundary.
  (C) softening-aware NULL: per-record margin (J_elastic - J_softened) histogram with the mean and
      the leave-one-record band -- the airtight null.
  (D) best dJ predictor scatter (the index that best orders the in-QP failures across FEMA).

Run:  python src/eesd_ohtori_control_fema_figures.py  
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).absolute().parents[1]
RESULTS = PROJECT_ROOT / "results"
if not (RESULTS / "eesd_ohtori_control_results_fema.json").exists():
    RESULTS = PROJECT_ROOT.parent / "analysis"
FIGDIR = Path(os.environ.get("EESD_FIGURE_OUTPUT_DIR", str(PROJECT_ROOT.parent / "figures")))
RES = json.loads((RESULTS / "eesd_ohtori_control_results_fema.json").read_text())

# Base-record reference (for the across-evidence contrast bar in panel A)
BASE = json.loads((RESULTS / "eesd_ohtori_control_results.json").read_text())


def main():
    (FIGDIR / "png").mkdir(parents=True, exist_ok=True)
    (FIGDIR / "pdf").mkdir(parents=True, exist_ok=True)
    (FIGDIR / "svg").mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 9,
        "axes.linewidth": 0.8,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2))   # 2x2 so each panel can breathe
    axA, axB, axC, axD = axes.flatten()

    rows = RES["per_row"]
    c1 = RES["claim1_saturation_dominates_2x2"]
    c2 = RES["claim2_inqp_deepyield_failure"]
    c3 = RES["claim3_softening_aware_null"]
    tiers = list(RES["subset"]["tiers"].keys())

    # -------- Panel A: descriptive variance shares (FEMA pooled + per tier + base-record reference) --------
    ov = c1["overall_all_rows"]
    fema_pooled = [ov["ss_A_frac"] * 100, ov["ss_B_frac"] * 100, ov["ss_AB_frac"] * 100]
    e3 = BASE["analysis3_2x2_factorial"]["overall_all_cases"]
    e3_shares = [e3["ss_A_frac"] * 100, e3["ss_B_frac"] * 100, e3["ss_AB_frac"] * 100]
    comp = ["frequency\n(tuning)", "saturation\n(handling)", "interaction"]
    xc = np.arange(3); w = 0.38
    # leave-one-record error on the saturation share (1-sigma)
    sat_err = c1["leave_one_record_spread"]["ss_B_sat_std"] * 100
    errs = [0, sat_err, 0]
    axA.bar(xc - w / 2, fema_pooled, w, yerr=errs, capsize=3,
            color=["#dd8452", "#55a868", "#8172b3"], edgecolor="black", linewidth=0.6,
            label="FEMA P695 (44 evals)", error_kw={"elinewidth": 1.0})
    axA.bar(xc + w / 2, e3_shares, w, color=["#dd8452", "#55a868", "#8172b3"], edgecolor="black",
            linewidth=0.6, hatch="///", alpha=0.55, label="Base-record set")
    for xi, v in zip(xc - w / 2, fema_pooled):
        axA.text(xi, v + 2.5, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    for xi, v in zip(xc + w / 2, e3_shares):
        axA.text(xi, v + 2.5, f"{v:.0f}", ha="center", va="bottom", fontsize=7, color="0.35")
    axA.set_xticks(xc); axA.set_xticklabels(comp, fontsize=7.5)
    axA.set_ylabel("share of across-cell cost variance (%)")
    axA.set_title("A  2$\\times$2 descriptive variance shares\nprimary score, pooled record sets",
                  fontsize=8.4)
    axA.legend(fontsize=6.8, frameon=False, loc="upper left")
    axA.set_ylim(0, max(max(fema_pooled), max(e3_shares)) * 1.3)
    axA.grid(axis="y", alpha=0.3, linewidth=0.6)

    # -------- Panel B: in-QP failure boundary (dJ vs uncontrolled drift) --------
    dJ = np.array([r["dJ_inqp_minus_clip"] for r in rows])
    udrift = np.array([r["unc_dmax_pct"] for r in rows])
    wins = dJ < 0
    axB.axhline(0, color="0.5", lw=0.8)
    axB.axvline(3.5, color="#c44e52", lw=1.0, ls="--", label="deep-yield (3.5% drift)")
    axB.scatter(udrift[wins], dJ[wins], s=34, c="#4c72b0", edgecolor="black", linewidth=0.4,
                label=f"in-QP wins ({wins.sum()})", zorder=3)
    axB.scatter(udrift[~wins], dJ[~wins], s=46, c="#c44e52", marker="^", edgecolor="black",
                linewidth=0.4, label=f"in-QP loses ({(~wins).sum()})", zorder=3)
    axB.set_xlabel("uncontrolled peak drift (%)")
    axB.set_ylabel(r"$\Delta J = J_{\mathrm{in\text{-}QP}} - J_{\mathrm{clip}}$")
    fr_d = c2["fail_rate_deepyield_unc_gt_3p5pct"]; fr_s = c2["fail_rate_shallow_unc_le_3p5pct"]
    axB.set_title(f"B  in-QP deep-yield failure\nfail rate: deep {fr_d:.0%} vs shallow {fr_s:.0%}",
                  fontsize=8.4)
    axB.legend(fontsize=6.6, frameon=False, loc="upper left")
    axB.grid(alpha=0.3, linewidth=0.6)

    # -------- Panel C: softening-aware NULL margin histogram --------
    margin = np.array([r["J_LQR_elastic"] - r["J_LQR_softened"] for r in rows])
    axC.axvline(0, color="0.5", lw=0.9)
    axC.hist(margin, bins=14, color="#55a868", edgecolor="black", linewidth=0.5, alpha=0.85)
    mu = c3["mean_margin_elastic_minus_softened"]
    band = c3["leave_one_record_margin"]["std"]
    axC.axvline(mu, color="#c44e52", lw=1.6, label=f"mean {mu:+.4f}")
    axC.axvspan(mu - band, mu + band, color="#c44e52", alpha=0.12)
    axC.set_xlabel(r"softening-aware margin  $J_{\mathrm{elastic}}-J_{\mathrm{softened}}$")
    axC.set_ylabel("number of (record, tier) cases")
    helps = c3["n_softening_helps"]; ntot = c3["n_total"]
    axC.set_title(f"C  No consistent benefit from retuning\nhelps {helps}/{ntot}, "
                  f"$\\sigma${c3['margin_std']:.4f}", fontsize=8.4)
    axC.legend(fontsize=6.8, frameon=False, loc="upper right")
    axC.grid(axis="y", alpha=0.3, linewidth=0.6)

    # -------- Panel D: best dJ predictor scatter --------
    best = c2["best_predictor_by_abs_rho"]
    bp = c2["boundary_predictors"][best]
    xvals = np.array([r[best] for r in rows])
    predictor_label = "ductility proxy" if best == "ductility_proxy" else best.replace("_", " ")
    axD.axhline(0, color="0.5", lw=0.8)
    axD.scatter(xvals[wins], dJ[wins], s=34, c="#4c72b0", edgecolor="black", linewidth=0.4,
                label="in-QP wins", zorder=3)
    axD.scatter(xvals[~wins], dJ[~wins], s=46, c="#c44e52", marker="^", edgecolor="black",
                linewidth=0.4, label="in-QP loses", zorder=3)
    axD.set_xlabel(predictor_label)
    axD.set_ylabel(r"$\Delta J$")
    axD.set_title(f"D  in-QP failure predictor\n{predictor_label}: $\\rho$={bp['spearman_rho_vs_dJ']:+.2f}, "
                  f"AUC={bp['auc_abs']:.2f}", fontsize=8.4)
    axD.legend(fontsize=6.8, frameon=False, loc="upper left")
    axD.grid(alpha=0.3, linewidth=0.6)

    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        dest = FIGDIR / f"Figure04_fig_eesd_ohtori_control_fema.{ext}"
        fig.savefig(dest, bbox_inches="tight")
        print(f"wrote {dest}")
    plt.close(fig)


if __name__ == "__main__":
    main()





