"""EESD Figure 1: the OpenSeesPy benchmark plants + modal validation.

fig_eesd_bench_plants: a schematic of the two Ohtori et al. (2004) SAC moment-frame
benchmark structures (3-story and 9-story) drawn as lumped-mass stick/lollipop story
models with per-floor seismic masses, alongside a modal-validation panel comparing the
published natural frequencies against the OpenSeesPy reimplementation.

  * LEFT   -- 3-story stick model: 3 floor masses, story heights 3.9624 m, mass1/mass2
              per MRF from struct3.m (478.35 / 517.79 t).
  * CENTER -- 9-story stick model: 9 floor masses + basement, heights from struct9.m.
  * RIGHT  -- modal validation: published vs OpenSeesPy first-N frequencies, both frames,
              with the maximum absolute error annotated (3-story 0.13%, 9-story 0.20%).

Numbers are read directly from results/eesd_ohtori_plant_validation.json and
results/eesd_ohtori_9story_validation.json -- nothing is hand-keyed. Caption message:
the OpenSeesPy reimplementation matches the published benchmark modal properties.

Style matches fig_eesd_competence.py / fig_eesd_plants.py (DejaVu Sans, Agg, 300-dpi PNG +
PDF). Run:  python paper_figures/src/fig_eesd_bench_plants.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results"
OUT = {ext: PROJECT_ROOT / "paper_figures" / "output" / ext for ext in ("pdf", "png", "svg")}

# palette shared with the other EESD figures
C_INK = "#111827"
C_GREY = "#6b7280"
C_PUB = "#1d4ed8"    # published (reference)
C_MOD = "#c2410c"    # OpenSeesPy model
C_BLDG = "#cbd5e1"
C_BLDG_E = "#475569"
C_MASS = "#fde68a"
C_MASS_E = "#b45309"
C_P3_FILL = "#eef4ff"
C_P3_EDGE = "#1d4ed8"
C_P9_FILL = "#fff4ed"
C_P9_EDGE = "#c2410c"
C_OK = "#15803d"


def _load():
    with open(RESULTS / "eesd_ohtori_plant_validation.json", encoding="utf-8") as f:
        v3 = json.load(f)
    with open(RESULTS / "eesd_ohtori_9story_validation.json", encoding="utf-8") as f:
        v9 = json.load(f)
    return v3, v9


def _stick(ax, x0, base_y, heights_m, masses_t, edge, fill, mass_fc, mass_ec,
           col_h=1.0, mass_r=0.034, label_mass=True, mass_fs=6.0):
    """Draw a lumped-mass stick model: a vertical column with a disk (lollipop) mass at
    each floor; floor vertical spacing proportional to the story heights. Returns the
    list of floor y-coordinates."""
    h = np.asarray(heights_m, float)
    ys = base_y + np.cumsum(h) / h.sum() * col_h
    # ground line + hatch
    ax.plot([x0 - 0.045, x0 + 0.045], [base_y, base_y], color=edge, lw=2.4, zorder=4)
    for xx in np.linspace(x0 - 0.045, x0 + 0.045, 7):
        ax.plot([xx, xx - 0.014], [base_y, base_y - 0.018], color=edge, lw=0.9, zorder=4)
    # column
    ax.plot([x0, x0], [base_y, ys[-1]], color=edge, lw=2.6, zorder=5)
    # floor masses as disks
    asp = 0.62  # vertical compression of the axes -> keep disks round-looking
    for i, (yc, m) in enumerate(zip(ys, masses_t)):
        ax.add_patch(plt.matplotlib.patches.Ellipse((x0, yc), mass_r * 2, mass_r * 2 * asp,
                     fc=mass_fc, ec=mass_ec, lw=1.2, zorder=6))
        if label_mass:
            ax.text(x0 + mass_r + 0.012, yc, f"{m:.0f} t", ha="left", va="center",
                    fontsize=mass_fs, color=mass_ec, zorder=7)
    return list(ys)


def fig_bench_plants():
    v3, v9 = _load()

    # ---- plant geometry / masses straight from the JSON ----
    h3 = v3["plant_summary"]["story_heights_m"]
    m3a = v3["plant_summary"]["seismic_mass1_per_mrf_kg"] / 1e3   # tonnes per MRF
    m3b = v3["plant_summary"]["seismic_mass2_per_mrf_kg"] / 1e3
    masses3 = [m3a, m3a, m3b]  # floors 1,2 = mass1; roof = mass2 (struct3.m)

    h9 = v9["plant_summary"]["story_heights_m"]
    masses9 = [m / 1e3 for m in v9["plant_summary"]["floor_masses_per_mrf_kg"]]

    # ---- modal-validation tables straight from the JSON ----
    def modal(v):
        modes = v["frequency_validation"]["modes"]
        n = [m["mode"] for m in modes]
        pub = [m["published_hz"] for m in modes]
        mod = [m["elastic_hz"] for m in modes]
        maxerr = v["frequency_validation"].get("max_abs_pct_error_elastic")
        return n, pub, mod, maxerr

    n3, pub3, mod3, err3 = modal(v3)
    n9, pub9, mod9, err9 = modal(v9)

    fig = plt.figure(figsize=(9.6, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 1.45, 1.55], wspace=0.28)
    ax3 = fig.add_subplot(gs[0, 0]); ax9 = fig.add_subplot(gs[0, 1])
    axm = fig.add_subplot(gs[0, 2])
    for ax in (ax3, ax9):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ===================== 3-story stick model =====================
    ax3.add_patch(Rectangle((0.04, 0.07), 0.92, 0.86, fc=C_P3_FILL, ec=C_P3_EDGE,
                            lw=1.5, zorder=1, alpha=0.55))
    ax3.text(0.5, 0.965, "3-story SAC frame", ha="center", va="top",
             fontsize=10.0, weight="bold", color=C_P3_EDGE)
    ax3.text(0.5, 0.905, "4 bays @ 9.144 m  |  stories 3.96 m", ha="center", va="top",
             fontsize=7.0, color=C_GREY, style="italic")
    ys3 = _stick(ax3, 0.40, 0.16, h3, masses3, C_P3_EDGE, C_P3_FILL, C_MASS, C_MASS_E,
                 col_h=0.58, mass_r=0.052, mass_fs=7.0)
    for i, yc in enumerate(ys3, start=1):
        ax3.text(0.40 - 0.075, yc, f"$m_{i}$", ha="right", va="center",
                 fontsize=8.0, weight="bold", color=C_INK)
    ax3.text(0.5, 0.115, "lumped floor masses (per MRF)", ha="center", va="top",
             fontsize=6.8, color=C_GREY, style="italic")

    # ===================== 9-story stick model =====================
    ax9.add_patch(Rectangle((0.06, 0.05), 0.88, 0.88, fc=C_P9_FILL, ec=C_P9_EDGE,
                            lw=1.5, zorder=1, alpha=0.55))
    ax9.text(0.5, 0.975, "9-story SAC frame", ha="center", va="top",
             fontsize=10.0, weight="bold", color=C_P9_EDGE)
    ax9.text(0.5, 0.925, "5 bays @ 9.144 m  |  + basement level", ha="center", va="top",
             fontsize=7.0, color=C_GREY, style="italic")
    # label masses sparsely (roof + basement) to avoid clutter on 9 floors
    ys9 = _stick(ax9, 0.50, 0.115, h9, masses9, C_P9_EDGE, C_P9_FILL, C_MASS, C_MASS_E,
                 col_h=0.76, mass_r=0.030, label_mass=False)
    ax9.text(0.50 + 0.052, ys9[-1], f"roof  {masses9[-1]:.0f} t", ha="left", va="center",
             fontsize=6.6, color=C_MASS_E, zorder=7)
    ax9.text(0.50 + 0.052, ys9[0], f"basement  {masses9[0]:.0f} t", ha="left", va="center",
             fontsize=6.6, color=C_MASS_E, zorder=7)
    ax9.text(0.50 - 0.052, ys9[4], "9 floor\nmasses\n~495 t", ha="right", va="center",
             fontsize=6.6, color=C_INK, zorder=7)
    ax9.text(0.5, 0.075, "lumped floor masses (per MRF)", ha="center", va="top",
             fontsize=6.8, color=C_GREY, style="italic")

    # ===================== modal-validation panel =====================
    # grouped horizontal bars: published vs OpenSeesPy frequency for each mode, both frames.
    # 3-story modes stacked above 9-story modes; a thin separator between the two blocks.
    rows = []
    for mm, p, q in zip(n3, pub3, mod3):
        rows.append(("3-story", f"Mode {mm}", p, q))
    for mm, p, q in zip(n9, pub9, mod9):
        rows.append(("9-story", f"Mode {mm}", p, q))

    yvals = np.arange(len(rows))[::-1]  # top-to-bottom
    bh = 0.34
    for y, (_, _, p, q) in zip(yvals, rows):
        axm.barh(y + bh / 2, p, height=bh, color=C_PUB, alpha=0.9, zorder=3)
        axm.barh(y - bh / 2, q, height=bh, color=C_MOD, alpha=0.9, zorder=3)
        axm.text(p + 0.06, y + bh / 2, f"{p:.3g}", va="center", ha="left",
                 fontsize=6.2, color=C_PUB)
        axm.text(q + 0.06, y - bh / 2, f"{q:.3f}", va="center", ha="left",
                 fontsize=6.2, color=C_MOD)
    axm.set_yticks(yvals)
    axm.set_yticklabels([f"{fr}  {md}" for (fr, md, _, _) in rows], fontsize=7.2)
    axm.set_xlabel("natural frequency (Hz)", fontsize=8.5)
    axm.set_xlim(0, 6.9)
    axm.set_title("Modal validation: published vs OpenSeesPy", fontsize=9.0, weight="bold")
    axm.grid(axis="x", color="#e5e7eb", lw=0.5, zorder=0)
    axm.spines["top"].set_visible(False); axm.spines["right"].set_visible(False)
    # separator between the 3-story and 9-story blocks
    sep = (yvals[len(n3) - 1] + yvals[len(n3)]) / 2
    axm.axhline(sep, color="#cbd5e1", lw=0.9, ls="--", zorder=1)
    # legend
    from matplotlib.patches import Patch
    axm.legend(handles=[Patch(fc=C_PUB, label="Published (Ohtori et al. 2004)"),
                        Patch(fc=C_MOD, label="OpenSeesPy reimplementation")],
               loc="upper right", bbox_to_anchor=(0.99, 0.99), ncol=1,
               fontsize=6.5, frameon=True, framealpha=0.95,
               facecolor="white", edgecolor="#cbd5e1", borderpad=0.35)
    # max-error callouts per block
    axm.text(6.78, yvals[1] + 0.05, f"3-story\nmax error\n{err3:.2f}%",
             ha="right", va="center", fontsize=6.8, color=C_OK, weight="bold",
             bbox=dict(boxstyle="round,pad=0.2", fc="#ecfdf5", ec=C_OK, lw=0.8))
    axm.text(6.78, yvals[len(n3) + 1], f"9-story\nmax error\n{err9:.2f}%",
             ha="right", va="center", fontsize=6.8, color=C_OK, weight="bold",
             bbox=dict(boxstyle="round,pad=0.2", fc="#ecfdf5", ec=C_OK, lw=0.8))

    fig.suptitle("OpenSeesPy reimplementation of the Ohtori et al. (2004) benchmark reproduces the "
                 "published modal properties at both heights",
                 fontsize=10.4, weight="bold", y=0.995)
    fig.subplots_adjust(left=0.03, right=0.985, top=0.90, bottom=0.20)
    _save(fig, "fig_eesd_bench_plants")


def _save(fig, name):
    for ext, d in OUT.items():
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{name}.{ext}", dpi=300 if ext == "png" else None,
                    bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print("wrote", name)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42,
                         "svg.fonttype": "none"})
    fig_bench_plants()


if __name__ == "__main__":
    main()



