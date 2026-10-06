"""EESD-honest concept / architecture figure (opening figure).

Narrative, left to right:
  seismic input -> a YIELDING frame driven through a SATURATING actuator (|u|<=Umax)
  -> the controller has two independent design axes: (1) the design FREQUENCY
     (elastic w1 vs softened w1*sqrt(alpha)) and (2) the actuator-CONSTRAINT handling
     (post-hoc clip vs in-QP) -> the systematic finding: constraint-handling DOMINATES
     the cost variance (82-99% across heights), frequency retuning is a null.

Schematic (no data file); the variance shares are the paper's reported descriptive two-factor decomposition.
Run:  python paper_figures/src/fig_eesd_h_concept.py
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = {e: ROOT / "paper_figures" / "output" / e for e in ("pdf", "png", "svg")}

C_INK = "#1f2937"
C_GREY = "#6b7280"
C_BOX = "#f1f5f9"
C_EDGE = "#94a3b8"
C_SAT = "#C0504D"     # saturation / constraint accent
C_FREQ = "#3B6EA8"    # frequency accent
C_WIN = "#2E8B57"     # dominant finding


def box(ax, x, y, w, h, title, body, ec=C_EDGE, fc=C_BOX, tcol=C_INK,
        title_fs=9.4, body_fs=8.0, title_y=0.16, body_y=0.46):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=ec, lw=1.6, zorder=3))
    ax.text(x + w / 2, y + h - title_y, title, ha="center", va="top", fontsize=title_fs,
            weight="bold", color=tcol, zorder=4)
    ax.text(x + w / 2, y + h - body_y, body, ha="center", va="top", fontsize=body_fs,
            color=C_INK, zorder=4)


def arrow(ax, x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=16,
                 color=C_INK, lw=2.0, zorder=2))


def mini_frame(ax, x0, y0, h=1.5, w=0.9):
    """A 3-story stick frame with a controllable actuator brace and a saturation note."""
    for j in range(4):
        ax.plot([x0, x0 + w], [y0 + j * h / 3, y0 + j * h / 3], color=C_INK, lw=1.6, zorder=4)
    for xi in (x0, x0 + w):
        ax.plot([xi, xi], [y0, y0 + h], color=C_INK, lw=1.6, zorder=4)
    # actuator (diagonal brace) on the first story
    ax.plot([x0, x0 + w], [y0, y0 + h / 3], color=C_SAT, lw=3.0, zorder=5)
    ax.add_patch(Circle((x0 + w / 2, y0 + h / 6), 0.07, fc="white", ec=C_SAT, lw=1.6, zorder=6))
    # ground hatch
    for k in range(5):
        ax.plot([x0 - 0.1 + k * 0.27, x0 - 0.25 + k * 0.27], [y0, y0 - 0.16], color=C_INK, lw=0.9)
    ax.plot([x0 - 0.15, x0 + w + 0.05], [y0, y0], color=C_INK, lw=1.4)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "svg.fonttype": "none",
                         "mathtext.fontset": "dejavusans"})
    fig, ax = plt.subplots(figsize=(13.4, 4.6))
    ax.set_xlim(0, 13.4)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    # 1. seismic input + yielding/saturating plant
    box(ax, 0.2, 1.4, 2.7, 2.5, "Yielding frame",
        "saturating actuator\n"
        r"$|u|\leq U_{\max}$",
        ec=C_SAT, title_fs=9.2, body_fs=7.4, title_y=0.18, body_y=0.56)
    mini_frame(ax, 1.15, 1.55, h=1.05, w=0.8)
    ax.text(0.34, 1.22, "Open Ohtori benchmark", fontsize=7.2, color=C_GREY,
            ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.86))
    ax.text(0.12, 4.56, "seismic input (FEMA P695 far-field)", fontsize=8.0,
            color=C_INK, ha="left", va="top")
    ax.add_patch(FancyArrowPatch((0.7, 4.28), (0.7, 3.96), arrowstyle="-|>",
                 mutation_scale=11, color=C_INK, lw=1.3))

    arrow(ax, 2.95, 3.65, 2.65)
    ax.text(3.30, 2.88, "evaluation\nflow", ha="center", va="bottom",
            fontsize=7.0, color=C_GREY, style="italic")

    # 2. the controller with two design axes (a 2x2)
    cx, cy, cw, ch = 3.7, 1.0, 4.5, 3.3
    ax.add_patch(FancyBboxPatch((cx, cy), cw, ch, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc="#fbfdff", ec=C_EDGE, lw=1.6, zorder=3))
    ax.text(cx + cw / 2, cy + ch - 0.12, "Controller: two independent design axes",
            ha="center", va="top", fontsize=9.4, weight="bold", color=C_INK)
    # 2x2 cells
    gx, gy = cx + 1.35, cy + 0.55
    gw, gh = 1.35, 1.05
    cells = [["elastic\n+ clip", "elastic\n+ in-QP"], ["softened\n+ clip", "softened\n+ in-QP"]]
    for r in range(2):
        for c in range(2):
            ax.add_patch(Rectangle((gx + c * gw, gy + (1 - r) * gh), gw, gh, fc="#eef2f7",
                                   ec=C_EDGE, lw=1.1, zorder=4))
            ax.text(gx + c * gw + gw / 2, gy + (1 - r) * gh + gh / 2, cells[r][c],
                    ha="center", va="center", fontsize=7.8, color=C_INK, zorder=5)
    # axis labels
    ax.text(gx - 0.32, gy + gh, "design\nfrequency", ha="right", va="center",
            fontsize=8.2, color=C_FREQ, weight="bold")
    ax.annotate("", xy=(gx - 0.30, gy), xytext=(gx - 0.30, gy + 2 * gh),
                arrowprops=dict(arrowstyle="-", color=C_FREQ, lw=1.4))
    ax.text(gx - 0.55, gy + 2 * gh - 0.1, r"$\omega_1$", fontsize=8.5, color=C_FREQ, ha="right")
    ax.text(gx - 0.55, gy + 0.1, r"$\omega_1\sqrt{\alpha}$", fontsize=8.5, color=C_FREQ, ha="right")
    ax.text(gx + gw / 2, gy - 0.24, "post-hoc clip", ha="center", va="top", fontsize=7.6,
            color=C_SAT)
    ax.text(gx + gw + gw / 2, gy - 0.24, "in-QP", ha="center", va="top", fontsize=7.6,
            color=C_SAT)
    ax.text(gx + gw, gy - 0.70, "actuator-constraint handling", ha="center", va="top",
            fontsize=8.2, color=C_SAT, weight="bold")
    arrow(ax, 8.25, 8.95, 2.65)

    # 3. the finding: descriptive two-factor variance shares
    fx, fy, fw, fh = 9.0, 1.0, 4.2, 3.3
    ax.add_patch(FancyBboxPatch((fx, fy), fw, fh, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc="#f0fbf4", ec=C_WIN, lw=1.8, zorder=3))
    ax.text(fx + fw / 2, fy + fh - 0.22, "Under the primary score",
            ha="center", va="top", fontsize=9.1, weight="bold", color=C_INK, zorder=4)
    ax.text(fx + fw / 2, fy + fh - 0.55, "descriptive variance shares, pooled FEMA P695 set",
            ha="center", va="top", fontsize=7.2, color=C_GREY, zorder=4)

    bx, bw = fx + 0.48, 2.72
    shares = [("Actuator-constraint handling", 82, C_SAT),
              ("Frequency tuning", 14, C_FREQ),
              ("Interaction", 4, "#9aa6b2")]
    row_y = [3.02, 2.31, 1.60]
    for (lbl, pct, col), yy in zip(shares, row_y):
        ax.text(bx, yy + 0.28, lbl, fontsize=7.7, color=C_INK, va="bottom", ha="left", zorder=4)
        ax.text(bx + bw + 0.20, yy + 0.28, f"{pct}%", fontsize=7.7, color=C_INK,
                va="bottom", ha="right", zorder=4)
        ax.add_patch(Rectangle((bx, yy), bw, 0.24, fc="white", ec="#dbe5df", lw=0.8, zorder=4))
        ax.add_patch(Rectangle((bx, yy), bw * pct / 100.0, 0.24, fc=col, ec="none", zorder=5))

    ax.text(fx + fw / 2, fy + 0.34, "No consistent benefit from retuning",
            fontsize=7.6, color=C_WIN, ha="center", va="center", style="italic", zorder=4,
            bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="#cfe8d7", alpha=0.92))

    fig.suptitle("Comparing actuator-bound handling and frequency retuning on yielding frames",
                 fontsize=11.6, weight="bold", y=0.99)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.08)
    for ext, dout in OUT.items():
        dout.mkdir(parents=True, exist_ok=True)
        fig.savefig(dout / f"fig_eesd_h_concept.{ext}", dpi=300 if ext == "png" else None,
                    bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print("wrote fig_eesd_h_concept")


if __name__ == "__main__":
    main()



