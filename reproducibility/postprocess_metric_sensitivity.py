from pathlib import Path
import json
import pandas as pd
import numpy as np

package_root = Path(__file__).resolve().parents[1]
round_dir = package_root
src = round_dir / "analysis" / "component_metrics_full_2026-09-27.json"
out_json = round_dir / "analysis" / "metric_sensitivity_summary_2026-09-27.json"
out_csv = round_dir / "analysis" / "metric_sensitivity_summary_2026-09-27.csv"
out_md = round_dir / "analysis" / "metric_sensitivity_summary_2026-09-27.md"

rows = json.loads(src.read_text(encoding="utf-8"))["rows"]
weights = {
    "primary": (0.70, 0.30, 0.02),
    "balanced": (0.50, 0.50, 0.00),
    "response_dominant": (0.80, 0.20, 0.00),
    "rms_drift_emphasis": (0.90, 0.10, 0.00),
    "effort_aware": (0.70, 0.20, 0.10),
}
cells = {
    "elastic_clip": ("elastic_clip_d_rms", "elastic_clip_d_max", "elastic_clip_u_peak"),
    "softened_clip": ("softened_clip_d_rms", "softened_clip_d_max", "softened_clip_u_peak"),
    "elastic_inqp": ("elastic_inqp_d_rms", "elastic_inqp_d_max", "elastic_inqp_u_peak"),
    "softened_inqp": ("softened_inqp_d_rms", "softened_inqp_d_max", "softened_inqp_u_peak"),
}
u_max = 5e5
records = []
for row in rows:
    for wname, (wrms, wpeak, weff) in weights.items():
        for cell, (krms, kpeak, ku) in cells.items():
            j = wrms * row[krms] / row["unc_d_rms"] + wpeak * row[kpeak] / row["unc_d_max"] + weff * row[ku] / u_max
            records.append({
                "record_id": row["record_id"], "earthquake": row["earthquake"], "tier": row["tier"],
                "weight_set": wname, "cell": cell, "J": float(j)
            })
df = pd.DataFrame(records)
summary = []
for (wname, cell), g in df.groupby(["weight_set", "cell"], sort=False):
    summary.append({
        "weight_set": wname, "cell": cell, "n": int(len(g)), "mean_J": float(g.J.mean()),
        "median_J": float(g.J.median()), "std_J": float(g.J.std(ddof=1)),
        "min_J": float(g.J.min()), "max_J": float(g.J.max()),
    })
summary_df = pd.DataFrame(summary)
rankings = {}
for wname in weights:
    means = summary_df[summary_df.weight_set == wname].set_index("cell").mean_J.sort_values()
    rankings[wname] = means.index.tolist()
# Main comparison: in-QP vs clipping at each tuning level.
pair = []
for wname in weights:
    sub = df[df.weight_set == wname]
    for tuning in ["elastic", "softened"]:
        a = sub[sub.cell == f"{tuning}_clip"].set_index(["record_id", "tier"]).J
        b = sub[sub.cell == f"{tuning}_inqp"].set_index(["record_id", "tier"]).J
        d = b - a
        pair.append({
            "weight_set": wname, "tuning": tuning, "n": int(len(d)),
            "inqp_wins": int((d < 0).sum()), "clip_wins": int((d > 0).sum()),
            "ties": int((d == 0).sum()), "mean_delta_inqp_minus_clip": float(d.mean()),
            "median_delta_inqp_minus_clip": float(d.median()),
        })
pair_df = pd.DataFrame(pair)
result = {
    "source": "analysis/component_metrics_full_2026-09-27.json",
    "n_rows": len(rows),
    "weights": {k: list(v) for k, v in weights.items()},
    "summary": summary,
    "rankings_low_to_high": rankings,
    "pairwise_inqp_minus_clip": pair,
}
out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
summary_df.to_csv(out_csv, index=False)
lines = [
    "# Metric-weight sensitivity (44 FEMA record/tier rows)",
    "",
    "The component rerun retains RMS drift, peak drift, peak force, and separate peak acceleration for all four factorial cells. The five weight sets are applied to normalized RMS drift, peak drift, and active-force effort.",
    "",
    "## Mean composite score (lower is better)",
    "",
    "| weight set | elastic/clip | softened/clip | elastic/in-QP | softened/in-QP | lowest mean |",
    "|---|---:|---:|---:|---:|---|",
]
for wname in weights:
    vals = summary_df[summary_df.weight_set == wname].set_index("cell").mean_J
    lowest = vals.idxmin()
    lines.append(f"| {wname} | {vals['elastic_clip']:.4f} | {vals['softened_clip']:.4f} | {vals['elastic_inqp']:.4f} | {vals['softened_inqp']:.4f} | {lowest} |")
lines += ["", "## In-QP minus clipping", "", "| weight set | tuning | n | in-QP wins | clipping wins | ties | mean delta |", "|---|---|---:|---:|---:|---:|---:|"]
for _, r in pair_df.iterrows():
    lines.append(f"| {r.weight_set} | {r.tuning} | {int(r.n)} | {int(r.inqp_wins)} | {int(r.clip_wins)} | {int(r.ties)} | {r.mean_delta_inqp_minus_clip:+.4f} |")
lines += ["", "Peak acceleration is retained in the raw component output but is not inserted into the primary score. The ranking statement is therefore explicitly conditional on this response/effort family."]
out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(summary_df.to_string(index=False))
print(pair_df.to_string(index=False))




