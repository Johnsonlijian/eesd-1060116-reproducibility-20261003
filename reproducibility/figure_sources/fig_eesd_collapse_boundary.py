r"""Collapse-limit boundary at both heights.

The figure uses the full 22-record IDA results for the response-validated nonlinear frame models. It keeps
elastic and softened controller designs visible because the corrected nine-story design uses its
height-specific frozen softening value (alpha_design = 0.36249). The collapse-limit comparison is
reported as a bounded result: elastic in-QP is essentially neutral relative to clipping, while the
softened allocation changes sign between heights. The finite-record fragility summaries are
comparative descriptors, not inferential population estimates.

Run with the project environment: python paper_figures/src/fig_eesd_collapse_boundary.py
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = {e: ROOT / "paper_figures" / "output" / e for e in ("pdf", "png", "svg")}
J3 = RES / "eesd_ida_fragility.json"
J9 = RES / "eesd_ida_fragility9.json"

C_CLIP = "#C0504D"
C_INQP = "#2E86AB"
C_UNC = "#9aa6b2"
C_SOFT = "#7B61A8"


def frag_panel(ax, d, title):
    cells = d["cells"]; scales = d["scales"]
    curves = [
        ("none_none", "uncontrolled", C_UNC, (0, (1, 1)), 1.1),
        ("elastic_clip", "clip (elastic)", C_CLIP, "-", 1.8),
        ("elastic_inqp", "in-QP (elastic)", C_INQP, "-", 1.8),
        ("softened_inqp", "in-QP (softened)", C_SOFT, "--", 1.5),
    ]
    for ck, lab, col, ls, lw in curves:
        fr = cells[ck]["fragility"]; pc = fr["P_collapse_by_scale"]
        ys = [pc.get(str(s), pc.get(s, np.nan)) for s in scales]
        ax.plot(scales, ys, color=col, ls=ls, lw=lw, marker="o" if ck != "none_none" else None,
                ms=3, label=lab)
        med = fr.get("median_capacity", math.inf)
        if math.isfinite(med):
            ax.axvline(med, color=col, lw=0.65, alpha=0.35)
    ax.set_xlabel("intensity scale  (x ATC-63 normalized)", fontsize=9.4)
    ax.set_ylabel("P(collapse)", fontsize=9.4)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title, loc="left", fontsize=10.6, weight="bold")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="#64748b", lw=0.9, alpha=0.55))
    labels.append("median collapse capacity")
    ax.legend(handles, labels, fontsize=7.2, frameon=False, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    d3 = json.loads(J3.read_text()); d9 = json.loads(J9.read_text())

    def med(d, ck):
        return d["cells"][ck]["fragility"]["median_capacity"]
    gains = {
        "3-story": (med(d3, "elastic_inqp") - med(d3, "elastic_clip"),
                    med(d3, "softened_inqp") - med(d3, "softened_clip")),
        "9-story": (med(d9, "elastic_inqp") - med(d9, "elastic_clip"),
                    med(d9, "softened_inqp") - med(d9, "softened_clip")),
    }

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "mathtext.fontset": "dejavusans", "axes.linewidth": 0.9})
    fig = plt.figure(figsize=(13.8, 4.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.08, 1.08, 0.96], wspace=0.30)
    axA, axB, axC = (fig.add_subplot(gs[i]) for i in range(3))

    frag_panel(axA, d3, "A  3-story collapse fragility")
    frag_panel(axB, d9, "B  9-story collapse fragility")
    axB.text(0.97, 0.06, "elastic allocation ~neutral;\nsoftened in-QP +0.16",
             transform=axB.transAxes, ha="right", fontsize=8.1, color=C_SOFT, style="italic")

    labels = list(gains.keys()); x = np.arange(len(labels)); width = 0.30
    ge = [gains[h][0] for h in labels]; gsft = [gains[h][1] for h in labels]
    b1 = axC.bar(x - width/2, ge, width, color=C_INQP, label="elastic design", zorder=3)
    b2 = axC.bar(x + width/2, gsft, width, color=C_SOFT, label="softened design", zorder=3)
    axC.axhline(0, color="#111827", lw=0.9)
    for bars, vals, col in [(b1, ge, C_INQP), (b2, gsft, C_SOFT)]:
        for bar, g in zip(bars, vals):
            axC.text(bar.get_x()+bar.get_width()/2, g + (0.025 if g >= 0 else -0.025), f"{g:+.2f}",
                     ha="center", va="bottom" if g >= 0 else "top", fontsize=8.6, weight="bold", color=col)
    axC.set_xticks(x); axC.set_xticklabels(labels, fontsize=9.3)
    axC.set_ylabel("in-QP minus clip\nmedian collapse capacity", fontsize=9.1)
    axC.set_title("C  Allocation gain at\n    the collapse limit", loc="left", fontsize=10.2, weight="bold")
    axC.legend(fontsize=7.7, frameon=False, loc="upper left")
    axC.set_ylim(-0.16, 0.28)
    for s in ("top", "right"):
        axC.spines[s].set_visible(False)

    fig.suptitle("Collapse-limit boundary for the height-specific frozen designs (IDA, 22 FEMA P695 records, response-validated Ohtori frame models)",
                 fontsize=10.9, weight="bold", y=1.005)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.84, bottom=0.16)
    for ext, dout in OUT.items():
        dout.mkdir(parents=True, exist_ok=True)
        fig.savefig(dout / f"fig_eesd_collapse_boundary.{ext}", dpi=300 if ext == "png" else None,
                    bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"wrote fig_eesd_collapse_boundary | 3-story elastic {ge[0]:+.3f}, softened {gsft[0]:+.3f} | 9-story elastic {ge[1]:+.3f}, softened {gsft[1]:+.3f}")


if __name__ == "__main__":
    main()


