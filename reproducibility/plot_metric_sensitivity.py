from pathlib import Path
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

package_root = Path(__file__).resolve().parents[1]
round_dir = package_root
src = round_dir / "analysis" / "metric_sensitivity_summary_2026-09-27.json"
figdir = Path(os.environ.get("EESD_FIGURE_OUTPUT_DIR", str(round_dir / "figures")))
figdir.mkdir(parents=True, exist_ok=True)
data = json.loads(src.read_text(encoding="utf-8"))
order = ["primary", "balanced", "response_dominant", "rms_drift_emphasis", "effort_aware"]
cells = ["elastic_clip", "softened_clip", "elastic_inqp", "softened_inqp"]
labels = ["Elastic\nclip", "Softened\nclip", "Elastic\nin-QP", "Softened\nin-QP"]
means = {(r["weight_set"], r["cell"]): r["mean_J"] for r in data["summary"]}
x = np.arange(len(order))
width = 0.18
colors = ["#6b7280", "#9ca3af", "#1f77b4", "#4c78a8"]
fig, ax = plt.subplots(figsize=(7.2, 4.2))
for i, cell in enumerate(cells):
    vals = [means[(w, cell)] for w in order]
    ax.bar(x + (i - 1.5)*width, vals, width, color=colors[i], edgecolor="black", linewidth=0.35, label=labels[i].replace("\n", " "))
ax.set_xticks(x, ["Primary", "Balanced", "Response-\ndominant", "RMS-drift\nemphasis", "Effort-aware"])
ax.set_ylabel("Mean normalized composite cost $J$")
ax.set_ylim(0.45, 0.72)
ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
ax.grid(axis="y", color="#d1d5db", linewidth=0.6, alpha=0.8)
ax.set_axisbelow(True)
ax.legend(ncol=2, frameon=False, fontsize=8, loc="upper right")
ax.text(0.01, 1.02, "All 44 FEMA record/tier evaluations; lower is better", transform=ax.transAxes, fontsize=8)
fig.tight_layout()
for name in ["Figure09_fig_eesd_metric_sensitivity.pdf", "Figure09_fig_eesd_metric_sensitivity.png"]:
    fig.savefig(figdir / name, dpi=300, bbox_inches="tight")
print(figdir / "Figure09_fig_eesd_metric_sensitivity.pdf")






