"""EESD authority-allocation synthesis figure (publication-grade) + honest falsification.

Panel A (the organizing PRINCIPLE): net benefit of each candidate enhancement over its
  saturation-aware baseline (J units, positive = the enhancement helps). Only the one that
  changes the ALLOCATION of the bounded force (in-QP vs post-hoc clip) helps; the three that
  refine the internal MODEL (design frequency, online stiffness, semi-active field force)
  are null or negative -> authority allocation dominates model fidelity.
Panel B (the BOUNDARY / falsification): the in-QP benefit against softening depth at both
  heights. The 3-story loses to clip below alpha_eff ~ 0.18; the 9-story, forced to a DEEPER
  alpha_eff = 0.135 (~10% drift), still wins. No universal softening-depth threshold governs
  the failure: it is a bounded, short-frame-specific corner.

NO re-run: reads results/eesd_ohtori_control_fema_rows.csv (3-story, demand-logged) and
results/eesd_ohtori_control9_deepsweep.json (9-story forced-deep). The three null margins are
the paper's reported values (frequency-tuning, yield-adaptive, semi-active MR).
"""
from __future__ import annotations
from pathlib import Path
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("EESD_PROJECT_ROOT", PACKAGE_ROOT))
RES = Path(os.environ.get("EESD_OUTPUT_DIR", PACKAGE_ROOT / "analysis"))
OUT = {e: ROOT / "paper_figures" / "output" / e for e in ("svg", "pdf", "png")}

C3, C9 = "#2166ac", "#c0392b"
C_HELP, C_NULL = "#1f9e6e", "#9aa6b2"
TRIVIAL = 1e-3


def main():
    d3 = pd.read_csv(RES / "eesd_ohtori_control_fema_rows.csv")
    d3 = d3[~d3["dJ_inqp_minus_clip"].isna()].copy()
    d3["benefit"] = -d3["dJ_inqp_minus_clip"]
    n3 = len(d3)
    inqp_wins = int((d3["dJ_inqp_minus_clip"] < 0).sum())
    inqp_winrate = inqp_wins / n3                        # in-QP beats clip (allocation)

    js = json.loads((RES / "eesd_ohtori_control9_deepsweep.json").read_text(encoding="utf-8"))
    pc = js["claim2_inqp_deepyield_failure"]["per_case"]
    r9 = pd.DataFrame({"alpha": [float(c["alpha_eff_min"]) for c in pc],
                       "benefit": [-float(c["dJ_inqp_minus_clip"]) for c in pc],
                       "drift": [float(c["unc_dmax_pct"]) for c in pc]})
    ml3 = d3[d3["dJ_inqp_minus_clip"] > TRIVIAL]
    onset3 = float(ml3["alpha_eff_min"].max())          # shallowest 3-story loss = failure onset
    a9_min = float(r9["alpha"].min()); d9_max = float(r9["drift"].max())

    # ---- Panel A: win rate = fraction of cases each enhancement beats its saturation-aware
    #      baseline. 50% = half the cases. Counts from the paper's per-case results:
    #      in-QP 32/44 (data); frequency 12/44; yield-adaptive 3/18; MR 6/44.
    bars = [
        ("Authority allocation\n(in-QP vs clip)", inqp_winrate, C_HELP),
        ("Design-frequency\nretuning", 12.0 / 44.0, C_NULL),
        ("Online stiffness update\n(yield-adaptive)", 3.0 / 18.0, C_NULL),
        ("Semi-active field\nmodulation (MR)", 6.0 / 44.0, C_NULL),
    ]

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "mathtext.fontset": "dejavusans"})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.0, 5.2), gridspec_kw={"width_ratios": [1.12, 1.0]})

    # ---------- Panel A: the principle (win rate vs the 50% baseline line) ----------
    ypos = np.arange(len(bars))[::-1].astype(float)
    for y, (label, val, col) in zip(ypos, bars):
        axA.barh(y, val, height=0.56, color=col, edgecolor="white", zorder=4)
        axA.text(-0.03, y, label, ha="right", va="center", fontsize=8.7, color="#222", weight="bold")
        helps = val > 0.5
        axA.text(val + 0.02, y, f"{val*100:.0f}%  wins",
                 ha="left", va="center", fontsize=8.6, color=C_HELP if helps else "#5b6770",
                 weight="bold")
    axA.axvline(0.5, color="#333", lw=1.7, ls="--", zorder=5)
    axA.text(0.5, ypos[0] + 0.55, "= half the cases", ha="center", va="bottom",
             fontsize=7.8, color="#333")
    axA.annotate("", xy=(1.05, ypos[0] + 0.42), xytext=(1.05, ypos[0] - 0.42),
                 arrowprops=dict(arrowstyle="-", color=C_HELP, lw=2.8))
    axA.text(1.09, ypos[0], "allocation:\nlarger tested change", ha="left", va="center", fontsize=8.2,
             color=C_HELP, weight="bold")
    axA.annotate("", xy=(1.05, ypos[3] - 0.42), xytext=(1.05, ypos[1] + 0.42),
                 arrowprops=dict(arrowstyle="-", color=C_NULL, lw=2.8))
    axA.text(1.09, ypos[2] + 0.5, "other tested changes:\n<30% wins", ha="left", va="center",
             fontsize=8.2, color="#5b6770", weight="bold")
    axA.set_xlim(0, 1.42); axA.set_ylim(-0.7, len(bars) - 0.05)
    axA.set_yticks([]); axA.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axA.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    axA.set_xlabel("fraction of cases with lower J than the respective baseline", fontsize=9.2)
    axA.set_title("A   Win fractions in separate comparisons",
                  loc="left", fontsize=10.6, weight="bold", color="#222")
    for s in ("top", "right", "left"):
        axA.spines[s].set_visible(False)

    # ---------- Panel B: the bounded, non-universal failure corner ----------
    axB.axhspan(-0.3, 0, color="#f6dcd8", alpha=0.6, zorder=1)
    axB.text(0.62, -0.20, "in-QP loses\nto clip", fontsize=8.4, color=C3, weight="bold",
             ha="center", va="center", zorder=2)
    axB.scatter(d3["alpha_eff_min"], d3["benefit"], s=46, marker="o", facecolor=C3,
                edgecolor="white", lw=0.6, alpha=0.9, label="3-story (stiff)", zorder=4)
    axB.scatter(r9["alpha"], r9["benefit"], s=52, marker="s", facecolor=C9, edgecolor="white",
                lw=0.6, alpha=0.9, label="9-story (flexible, forced deep)", zorder=4)
    axB.axhline(0, color="#333", lw=1.2, zorder=3)
    ytop = axB.get_ylim()[1]
    axB.axvline(onset3, color=C3, ls="--", lw=1.4, zorder=3)
    axB.axvline(a9_min, color=C9, ls=":", lw=1.6, zorder=3)
    axB.set_xlabel(r"softening depth  $\alpha_{\mathrm{eff,min}}=(\omega_1^{\mathrm{tan}}/\omega_1)^2$", fontsize=9.5)
    axB.set_ylabel(r"optimal-allocation benefit  $J_{\mathrm{clip}}-J_{\mathrm{in\text{-}QP}}$", fontsize=9.5)
    axB.set_title("B   The in-QP failure is a bounded short-frame corner,\n     not a universal softening threshold",
                  loc="left", fontsize=11.0, weight="bold", color="#222")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C3, markeredgecolor="white",
               markersize=7, label="3-story: losses observed below $\\alpha\\approx0.18$"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=C9, markeredgecolor="white",
               markersize=7, label=f"9-story: $\\alpha\\approx{a9_min:.2f}$, ~{d9_max:.0f}% drift, still wins"),
        Line2D([0], [0], color=C3, lw=1.4, ls="--", label="3-story loss onset"),
        Line2D([0], [0], color=C9, lw=1.6, ls=":", label="deep forced 9-story point"),
    ]
    axB.legend(handles=handles, fontsize=7.6, frameon=True, loc="upper right",
               facecolor="white", edgecolor="#dddddd", framealpha=0.95)
    axB.grid(True, color="#eee", lw=0.5)
    for s in ("top", "right"):
        axB.spines[s].set_visible(False)

    fig.suptitle("Separate response-level comparisons of allocation and model updates",
                 fontsize=13.0, weight="bold", y=0.995, color="#222")
    fig.subplots_adjust(left=0.205, right=0.985, top=0.88, bottom=0.115, wspace=0.42)
    for ext, dout in OUT.items():
        dout.mkdir(parents=True, exist_ok=True)
        fig.savefig(dout / f"fig_eesd_h_authority.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print("=== AUTHORITY-ALLOCATION FIGURE (publication-grade) ===")
    print(f"in-QP win rate = {inqp_winrate*100:.0f}% ({inqp_wins}/{n3})")
    print(f"3-story failure onset alpha = {onset3:.3f}; 9-story deepest alpha = {a9_min:.3f} at {d9_max:.1f}% drift")
    print("wrote fig_eesd_h_authority")


if __name__ == "__main__":
    main()



