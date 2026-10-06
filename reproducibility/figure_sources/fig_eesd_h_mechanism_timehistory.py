r"""EESD mechanism figure (real-plant time-history evidence for the authority-allocation principle).

ONE message, measured on the VALIDATED nonlinear Ohtori plant in a saturated deep-yield case (actuator
saturated ~90% of the time):
  (A) PERTURB the design frequency (elastic->softened, holding clip) and the bounded force
      barely moves; PERTURB the allocation rule (clip->in-QP, holding tuning) and the SAME
      bounded authority is reshaped. The two difference signals make this undeniable:
      RMS(delta-u_internal) << RMS(delta-u_allocation).
  (B) That force difference propagates to the drift response: the four controlled runs
      cluster by ALLOCATION (clip vs in-QP), not by tuning -- the two tunings overlie.
  (C) Across four strong records the allocation axis decorrelates the control force far
      more than the internal-model axis. This is why the two-factor decomposition assigns the variance to
      allocation: it is visible in the force itself, not just in the summary statistic.

Data: results/eesd_mechanism_th_northrdg_x1.5.npz + _verdict.json, produced by
eesd_mechanism_timehistory.py (openseespy 3.8.0). Every curve is a logged closed-loop run.

Run: python paper_figures/src/fig_eesd_h_mechanism_timehistory.py
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = {e: ROOT / "paper_figures" / "output" / e for e in ("svg", "pdf", "png")}
NPZ = RES / "eesd_mechanism_th_northrdg_x1.5.npz"
VERDICT = RES / "eesd_mechanism_timehistory_verdict.json"

C_FID = "#2E86AB"     # internal-model perturbation: blue
C_ALLOC = "#C0504D"   # allocation perturbation: red
C_UNC = "#9aa6b2"     # uncontrolled


def govdrift(arr):
    """governing inter-story drift envelope: max over floors of |drift| at each step."""
    return np.max(np.abs(arr), axis=1)


def main():
    d = np.load(NPZ)
    t = d["t"]; alpha = float(d["alpha_eff"])
    uec = d["u_elastic_clip"]; usc = d["u_softened_clip"]; uei = d["u_elastic_inqp"]
    # difference signals at the floor where the allocation effect is largest (kN)
    fl = int(np.argmax(np.mean(np.abs(uec - uei), axis=0)))
    du_fid = (usc[:, fl] - uec[:, fl]) / 1e3        # internal-model perturbation: softened - elastic (clip held)
    du_all = (uei[:, fl] - uec[:, fl]) / 1e3        # allocation perturbation: in-QP - clip (elastic held)
    rms_fid = float(np.sqrt(np.mean(du_fid ** 2)))
    rms_all = float(np.sqrt(np.mean(du_all ** 2)))

    sw = json.loads(VERDICT.read_text())["sweep"]
    cases, fid, alloc = [], [], []
    label = {"northrdg_x1.0": "North x1.0", "northrdg_x1.5": "North x1.5",
             "northrdg_x2.0": "North x2.0", "kobe_x1.0": "Kobe x1.0"}
    for v in sw:
        if "error" in v:
            continue
        cases.append(label.get(v["case"], v["case"]))
        fid.append(v["H1_model_fidelity_null_under_saturation"]["corr_u_elastic_vs_softened_CLIP"])
        alloc.append(v["H2_allocation_dominates"]["corr_u_clip_vs_inqp_ELASTIC"])

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none",
                         "mathtext.fontset": "dejavusans", "axes.linewidth": 0.9})
    fig = plt.figure(figsize=(13.8, 5.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.18, 1.12, 0.92], wspace=0.38)
    axA = fig.add_subplot(gs[0]); axB = fig.add_subplot(gs[1]); axC = fig.add_subplot(gs[2])

    # ---- Panel A: the two perturbation difference signals (the core contrast) ----
    t0, t1 = 3.0, 12.0
    m = (t >= t0) & (t <= t1)
    axA.axhline(0, color="#cbd5e1", lw=0.9, zorder=1)
    axA.plot(t[m], du_all[m], color=C_ALLOC, lw=1.25, zorder=4,
             label=f"allocation RMS {rms_all:.0f} kN")
    axA.plot(t[m], du_fid[m], color=C_FID, lw=1.4, zorder=5,
             label=f"internal-model RMS {rms_fid:.0f} kN")
    axA.set_title("A  Force perturbation: allocation dominates", loc="left",
                  fontsize=10.6, weight="bold", pad=24)
    axA.set_ylabel(r"$\Delta$ story actuator force  (kN)", fontsize=9.8)
    axA.set_xlabel("time  (s)", fontsize=9.8)
    axA.set_xlim(t0, t1)
    axA.legend(fontsize=7.0, frameon=True, loc="upper right", ncol=1,
               facecolor="white", edgecolor="#cbd5e1", framealpha=0.92)

    # ---- Panel B: the drift consequence -- runs cluster by allocation, not tuning ----
    axB.plot(t[m], govdrift(d["drift_unc"])[m] * 100, color=C_UNC, lw=1.0, zorder=2,
             label="uncontrolled")
    axB.plot(t[m], govdrift(d["drift_elastic_clip"])[m] * 100, color=C_ALLOC, lw=1.5, zorder=4,
             label="clip  (elastic)")
    axB.plot(t[m], govdrift(d["drift_softened_clip"])[m] * 100, color=C_ALLOC, lw=1.1,
             ls=(0, (1, 1.4)), zorder=4, label="clip  (softened)")
    axB.plot(t[m], govdrift(d["drift_elastic_inqp"])[m] * 100, color=C_FID, lw=1.5, zorder=5,
             label="in-QP  (elastic)")
    axB.plot(t[m], govdrift(d["drift_softened_inqp"])[m] * 100, color=C_FID, lw=1.1,
             ls=(0, (1, 1.4)), zorder=5, label="in-QP  (softened)")
    axB.set_title("B  Drift response clusters by allocation, not tuning", loc="left",
                  fontsize=10.6, weight="bold")
    axB.set_ylabel("governing inter-story drift  (%)", fontsize=9.8)
    axB.set_xlabel("time  (s)", fontsize=9.8)
    axB.set_xlim(t0, t1)
    axB.legend(fontsize=7.8, frameon=False, loc="upper right", ncol=1, handlelength=2.0)
    axB.text(0.08, 0.04, "the two tunings overlie within each allocation",
             transform=axB.transAxes, ha="left", fontsize=8.6, style="italic", color="#475569")

    # ---- Panel C: the decorrelation contrast across 4 records ----
    x = np.arange(len(cases)); w = 0.38
    axC.bar(x - w / 2, fid, w, color=C_FID, label="internal-model\n(elastic vs softened)", zorder=3)
    axC.bar(x + w / 2, alloc, w, color=C_ALLOC, label="allocation\n(clip vs in-QP)", zorder=3)
    axC.axhline(1.0, color="#e5e7eb", lw=0.8)
    axC.set_ylim(0, 1.10)
    axC.set_xticks(x); axC.set_xticklabels(cases, fontsize=8.0, rotation=18, ha="right")
    axC.set_ylabel("force-history correlation\n(perturb axis off $\\to$ on)", fontsize=9.0)
    axC.set_title("C  Holds across records", loc="left", fontsize=10.4, weight="bold")
    axC.legend(fontsize=7.1, frameon=True, loc="upper left", bbox_to_anchor=(1.02, 1.0),
               facecolor="white", edgecolor="#cbd5e1", framealpha=0.92)
    for xi, (fv, av) in enumerate(zip(fid, alloc)):
        axC.text(xi - w / 2, fv + 0.015, f"{fv:.2f}", ha="center", fontsize=7.0, color=C_FID)
        axC.text(xi + w / 2, av + 0.015, f"{av:.2f}", ha="center", fontsize=7.0, color=C_ALLOC)
    for ax in (axA, axB, axC):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.suptitle("In the saturated deep-yield test, the bounded control force is governed by authority allocation, not design-frequency tuning\n"
                 f"(Ohtori 3-story, $\\alpha_{{\\rm eff}}={alpha:.2f}$; actuator at its bound ~90% of the time)",
                 fontsize=10.9, weight="bold", y=0.985)
    fig.subplots_adjust(left=0.062, right=0.99, top=0.80, bottom=0.18)
    for ext, dout in OUT.items():
        dout.mkdir(parents=True, exist_ok=True)
        fig.savefig(dout / f"fig_eesd_h_mechanism_timehistory.{ext}",
                    dpi=300 if ext == "png" else None, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"wrote fig_eesd_h_mechanism_timehistory  (floor {fl}, RMS_fid={rms_fid:.1f} kN "
          f"vs RMS_alloc={rms_all:.1f} kN, ratio {rms_all / max(rms_fid, 1e-9):.1f}x)")


if __name__ == "__main__":
    main()





