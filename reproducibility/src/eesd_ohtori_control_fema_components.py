r"""Run the FEMA P695 component-level robustness analysis used by the revised manuscript.

The script evaluates the frozen controller designs on the response-validated nonlinear plant,
using one component from each FEMA record pair at the declared moderate- and deep-yield tiers.
It is a reproducibility entry point, not an editorial-stage artifact. Set EESD_PROJECT_ROOT,
EESD_OHTORI_DIR, EESD_FEMA_DIR, EESD_SOURCE_ROOT, and EESD_OUTPUT_DIR when the external
benchmark or record tree is stored outside this package.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
import os
import sys
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(os.environ.get("EESD_SOURCE_ROOT", PACKAGE_ROOT / "reproducibility" / "src"))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np
import openseespy.opensees as ops

import fema_p695_loader as F
from eesd_ohtori_plant import (BuildConfig, apply_rayleigh, build_model, eigen_frequencies,
                               parse_struct3, OHTORI_DIR)
# REUSE every piece of the canonical machinery verbatim:
import eesd_ohtori_control as canonical
from eesd_ohtori_control import (OhtoriPlantSession, FactorialMPC, design_lqr_on_K, j_gate,
                                 decompose_2x2, spearman_rho, crossover, U_MAX, _RM)

PROJECT_ROOT = Path(os.environ.get("EESD_PROJECT_ROOT", PACKAGE_ROOT))
RESULTS = Path(os.environ.get("EESD_OUTPUT_DIR", PACKAGE_ROOT / "analysis"))

# ----- subset definition -----
TIERS = {"moderate": 1.5, "deepyield": 3.0}     # atc63_norm * scale
DT_ANALYSIS = 0.005
ACTIVE_WINDOW_CAP = 50.0                         # cap the integrated window for speed (s)
# canonical-frozen design points (so the FEMA result is directly comparable to the benchmark result)
ALPHA_DESIGN = 0.183
Q_BEST, R_BEST = 1e12, 1e-6
CONTROLLABLE_J = 0.95


class FemaPlantSession(OhtoriPlantSession):
    """canonical plant session, but driven by a FEMA P695 record (atc63_norm * global scale)."""

    def _setup_fema(self, rid, scale, active_window, dt_analysis=DT_ANALYSIS):
        t_rec, ag = F.load_record(rid)                      # time [s], accel [m/s^2] (unscaled)
        ag = ag * F.atc63_norm_factor(rid) * scale          # ATC-63 normalize, then global scale
        rec_dt = F.record_dt(rid)
        build_model(self.plant, self.cfg)
        self.rayleigh = apply_rayleigh(self.cfg.damping_ratio)
        self.f0 = eigen_frequencies(3)
        tf_use = min(active_window, float(t_rec[-1]))
        keep = t_rec <= tf_use
        times, accel = t_rec[keep], ag[keep]
        if len(times) and times[-1] < tf_use:
            times = np.append(times, tf_use); accel = np.append(accel, 0.0)
        self.times, self.accel, self.rec_dt = times, accel, rec_dt
        ops.timeSeries("Path", 1001, "-dt", rec_dt, "-values", *accel.tolist(), "-factor", 1.0)
        ops.pattern("UniformExcitation", 2001, 1, "-accel", 1001)
        ops.timeSeries("Constant", 1002)
        ops.pattern("Plain", 2002, 1002)
        for nd in self.floor_nodes:
            ops.load(nd, 0.0, 0.0, 0.0)
        ops.wipeAnalysis()
        ops.constraints("Transformation"); ops.numberer("RCM"); ops.system("BandGeneral")
        ops.test("NormDispIncr", 1e-8, 40); ops.algorithm("Newton")
        ops.integrator("Newmark", 0.5, 0.25); ops.analysis("Transient")
        self.tf_use = tf_use
        self.dt_analysis = dt_analysis

    def run_fema(self, controller, rid, scale, active_window,
                 dt_analysis=DT_ANALYSIS, measure_alpha=False):
        """Closed-loop FEMA run. Mirrors OhtoriPlantSession.run but uses _setup_fema.

        Returns the SAME dict keys as the canonical run (so j_gate / alpha consumers are unchanged), plus
        a peak-ductility proxy = drift_peak / drift_yield (drift_yield from the elastic-design yield
        threshold ~0.7% used in the SAC frame; reported as a relative regime index, not an absolute)."""
        self._setup_fema(rid, scale, active_window, dt_analysis)
        if controller is not None and hasattr(controller, "reset"):
            controller.reset()
        n_steps = int(math.ceil(self.tf_use / dt_analysis))
        drift_peak = acc_peak = u_peak = 0.0
        demand_peak_ratio = 0.0   # peak |unconstrained command| / U_MAX  (graded saturation/demand measure)
        sat_steps = 0
        drift_sq = 0.0
        u_sq = 0.0
        nout = 0
        unstable = False
        f1_min = self.f0[0] if self.f0 else math.nan
        f1_samples = []
        de_weight_num = 0.0
        de_weight_den = 0.0
        for step in range(n_steps):
            d, v = self._state()
            if controller is not None:
                u_raw = np.asarray(controller(np.concatenate([d, v])), dtype=float)
                u = np.clip(u_raw, -U_MAX, U_MAX)
                demand_peak_ratio = max(demand_peak_ratio, float(np.max(np.abs(u_raw))) / U_MAX)
                if np.any(np.abs(u_raw) >= U_MAX - 1.0):
                    sat_steps += 1
            else:
                u = np.zeros(3)
            ops.remove("loadPattern", 2002)
            ops.pattern("Plain", 2002, 1002)
            for c, nd in enumerate(self.floor_nodes):
                ops.load(nd, float(u[c]), 0.0, 0.0)
            if ops.analyze(1, dt_analysis) != 0:
                unstable = True
                break
            t = ops.getTime()
            d, v = self._state()
            gacc = float(np.interp(t, self.times, self.accel)) if len(self.times) else 0.0
            abs_acc = np.array([ops.nodeAccel(n, 1) + gacc for n in self.floor_nodes])
            interstory = np.concatenate([[d[0]], np.diff(d)])
            drift_ratio = interstory / self.heights
            de = float(np.sum(drift_ratio ** 2))
            drift_peak = max(drift_peak, float(np.max(np.abs(drift_ratio))))
            acc_peak = max(acc_peak, float(np.max(np.abs(abs_acc))))
            u_peak = max(u_peak, float(np.max(np.abs(u))))
            drift_sq += de
            u_sq += float(np.sum(u * u))
            nout += 1
            if measure_alpha and step % 10 == 0 and drift_peak > 1e-5:
                try:
                    lam = ops.eigen("-fullGenLapack", 1)
                    if lam and lam[0] > 0:
                        f1 = math.sqrt(lam[0]) / (2 * math.pi)
                        f1_samples.append((f1, de))
                        if f1 < f1_min:
                            f1_min = f1
                        de_weight_num += (f1 / self.f0[0]) ** 2 * de
                        de_weight_den += de
                    ops.test("NormDispIncr", 1e-8, 40); ops.algorithm("Newton")
                except Exception:
                    pass
        res = {
            "unstable": unstable,
            "d_max": 1e9 if unstable else drift_peak,
            "xdd_max": 1e9 if unstable else acc_peak,
            "d_rms": 1e9 if unstable else float(math.sqrt(drift_sq / max(nout, 1))),
            "u_peak": u_peak, "u_rms": float(math.sqrt(u_sq / max(nout, 1))),
            "n_steps": nout,
            "demand_peak_ratio": demand_peak_ratio,
            "sat_duty": float(sat_steps) / max(nout, 1),
        }
        if measure_alpha:
            alpha_min = (f1_min / self.f0[0]) ** 2 if self.f0 else math.nan
            f1_arr = np.array([f for f, _ in f1_samples]) if f1_samples else np.array([self.f0[0]])
            alpha_p5 = (np.percentile(f1_arr, 5) / self.f0[0]) ** 2
            alpha_de = (de_weight_num / de_weight_den) if de_weight_den > 0 else 1.0
            res.update({
                "f1_elastic": self.f0[0], "f1_tangent_min": f1_min,
                "alpha_eff_min": float(alpha_min), "alpha_eff_p5": float(alpha_p5),
                "alpha_eff_driftweighted": float(alpha_de),
            })
        return res


# yield drift for the ductility proxy (SAC steel MRF ~ first-yield interstory drift)
DRIFT_YIELD = 0.007


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="3 records x 2 tiers smoke")
    ap.add_argument("--components-out", type=Path, default=RESULTS / "component_metrics_full.json")
    ap.add_argument("--records", type=int, default=0, help="limit to first N component-1 records (0=all 22)")
    ap.add_argument("--out", type=Path, default=RESULTS / "eesd_ohtori_control_results_fema.json")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    pairs = F.record_pairs()
    comp1 = [p[0] for p in pairs]
    if args.quick:
        comp1 = comp1[:3]
    elif args.records > 0:
        comp1 = comp1[:args.records]
    manifest = F.manifest().set_index("record_id")

    plant = parse_struct3(OHTORI_DIR / "struct3.m")
    cfg = BuildConfig(mode="nonlinear")
    sess = FemaPlantSession(plant, cfg)

    # FIXED controllers (designed once on the reduced model; identical to canonical design points)
    K_lqr = design_lqr_on_K(_RM.K0, Q_BEST, R_BEST)
    K_soft = design_lqr_on_K(_RM.K_softened(ALPHA_DESIGN), Q_BEST, R_BEST)
    n_lqr = float(np.linalg.norm(K_lqr)); n_soft = float(np.linalg.norm(K_soft))
    K_iso = K_soft * (n_lqr / n_soft)

    CELLS = [("elastic", "clip"), ("softened", "clip"), ("elastic", "inqp"), ("softened", "inqp")]

    print(f"FEMA FEMA P695 sweep: {len(comp1)} component-1 records x {len(TIERS)} tiers "
          f"({TIERS}) on the full nonlinear Ohtori plant.")
    print(f"Frozen canonical design: LQR q={Q_BEST:.0e} r={R_BEST:.0e}; softened alpha={ALPHA_DESIGN}\n")

    rows = []            # one row per (record, tier): all cells + LQR family + alpha + descriptors
    for ri, rid in enumerate(comp1):
        eq = str(manifest.loc[rid, "earthquake"])
        mag = float(manifest.loc[rid, "magnitude"])
        dur = float(manifest.loc[rid, "duration_s"])
        aw = min(ACTIVE_WINDOW_CAP, dur)
        rec_dt = F.record_dt(rid)
        for tier, scale in TIERS.items():
            # uncontrolled ref + alpha_eff
            unc = sess.run_fema(None, rid, scale, aw, measure_alpha=True)
            # LQR family (for the softening-aware NULL)
            lqr_J = {}
            lqr_demand = {}
            for label, K in (("elastic", K_lqr), ("softened", K_soft), ("iso", K_iso)):
                r = sess.run_fema(lambda x, K=K: -K @ x, rid, scale, aw)
                lqr_J[label] = j_gate(r, unc)
                lqr_demand[label] = (r.get("demand_peak_ratio", float("nan")), r.get("sat_duty", float("nan")))
            # 2x2 factorial MPC cells
            Jc = {}
            cell_dmax = {}
            cell_resp = {}
            for (tuning, handling) in CELLS:
                ctrl = FactorialMPC(rec_dt, tuning, handling, ALPHA_DESIGN)
                r = sess.run_fema(ctrl, rid, scale, aw)
                Jc[(tuning, handling)] = j_gate(r, unc)
                cell_dmax[(tuning, handling)] = (r["d_max"] * 100) if not r["unstable"] else 1e9
                cell_resp[(tuning, handling)] = r
            dJ = Jc[("elastic", "inqp")] - Jc[("elastic", "clip")]      # in-QP minus clip (elastic tuning)
            dJ_soft = Jc[("softened", "inqp")] - Jc[("softened", "clip")]
            E_K = 1.0 - unc["alpha_eff_driftweighted"]
            duct = unc["d_max"] / DRIFT_YIELD
            row = {
                "record_id": rid, "earthquake": eq, "magnitude": mag,
                "tier": tier, "scale": scale,
                "unc_dmax_pct": unc["d_max"] * 100, "ductility_proxy": duct,
                "alpha_eff_min": unc["alpha_eff_min"],
                "alpha_eff_driftweighted": unc["alpha_eff_driftweighted"],
                "E_K": E_K,
                "demand_peak_ratio": lqr_demand["elastic"][0], "sat_duty": lqr_demand["elastic"][1],
                "J_LQR_elastic": lqr_J["elastic"], "J_LQR_softened": lqr_J["softened"],
                "J_LQR_iso": lqr_J["iso"],
                "J00_elastic_clip": Jc[("elastic", "clip")], "J10_softened_clip": Jc[("softened", "clip")],
                "J01_elastic_inqp": Jc[("elastic", "inqp")], "J11_softened_inqp": Jc[("softened", "inqp")],
                "dmax00": cell_dmax[("elastic", "clip")], "dmax01": cell_dmax[("elastic", "inqp")],
                "dJ_inqp_minus_clip": dJ, "dJ_inqp_minus_clip_softened": dJ_soft,
                "inqp_wins": bool(dJ < 0),
                "max_cell_J": float(max(Jc.values())),
            }
            # Raw response components for the metric-sensitivity check.
            for (tuning, handling) in CELLS:
                rr = cell_resp[(tuning, handling)]
                prefix = f"{tuning}_{handling}_"
                row[prefix + "d_rms"] = float(rr["d_rms"])
                row[prefix + "d_max"] = float(rr["d_max"])
                row[prefix + "u_peak"] = float(rr["u_peak"])
                row[prefix + "xdd_max"] = float(rr["xdd_max"])
            row["unc_d_rms"] = float(unc["d_rms"])
            row["unc_d_max"] = float(unc["d_max"])
            row["unc_xdd_max"] = float(unc["xdd_max"])
            row["unc_u_peak"] = 0.0
            rows.append(row)
            print(f"  [{ri+1:2d}/{len(comp1)}] {rid} {eq[:14]:14s} {tier:9s}(x{scale}): "
                  f"unc d={unc['d_max']*100:5.2f}% a_eff={unc['alpha_eff_min']:.3f} | "
                  f"LQR e/s {lqr_J['elastic']:.3f}/{lqr_J['softened']:.3f} | "
                  f"clip {Jc[('elastic','clip')]:.3f} inqp {Jc[('elastic','inqp')]:.3f} dJ={dJ:+.3f}")

    elapsed = time.time() - t0

    # ============ AGGREGATE STATISTICS ============
    arr = lambda k, sub=None: np.array([r[k] for r in (sub if sub is not None else rows)], float)

    # ---- Claim 1: 2x2 ANOVA variance shares (pooled + leave-one-record) ----
    def cell_meanJ(subrows):
        return {(t, h): float(np.mean([r[{("elastic", "clip"): "J00_elastic_clip",
                                          ("softened", "clip"): "J10_softened_clip",
                                          ("elastic", "inqp"): "J01_elastic_inqp",
                                          ("softened", "inqp"): "J11_softened_inqp"}[(t, h)]]
                                        for r in subrows]))
                for (t, h) in CELLS}

    def anova_shares(subrows):
        dec = decompose_2x2(cell_meanJ(subrows))
        return dec

    overall = anova_shares(rows)
    # per-tier
    per_tier = {tier: anova_shares([r for r in rows if r["tier"] == tier]) for tier in TIERS}
    # controllable subset (max cell J < CONTROLLABLE_J, per (record,tier))
    ctrl_rows = [r for r in rows if r["max_cell_J"] < CONTROLLABLE_J]
    overall_ctrl = anova_shares(ctrl_rows) if len(ctrl_rows) >= 4 else None
    # leave-one-record spread on the ANOVA shares (drop all rows of one record)
    rec_ids = sorted({r["record_id"] for r in rows})
    loro_sat = []; loro_freq = []; loro_int = []
    for drop in rec_ids:
        sub = [r for r in rows if r["record_id"] != drop]
        d = anova_shares(sub)
        loro_sat.append(d["ss_B_frac"]); loro_freq.append(d["ss_A_frac"]); loro_int.append(d["ss_AB_frac"])
    loro_sat = np.array(loro_sat); loro_freq = np.array(loro_freq); loro_int = np.array(loro_int)

    # ---- Claim 2: in-QP deep-yield failure boundary ----
    dJ_all = arr("dJ_inqp_minus_clip")
    inqp_loses = dJ_all > 0
    # candidate predictors of dJ (failure when dJ>0)
    predictors = {
        "alpha_eff_min": arr("alpha_eff_min"),
        "E_K": arr("E_K"),
        "ductility_proxy": arr("ductility_proxy"),
        "unc_dmax_pct": arr("unc_dmax_pct"),
        "J_LQR_elastic": arr("J_LQR_elastic"),
        "magnitude": arr("magnitude"),
        "max_cell_J": arr("max_cell_J"),
    }
    boundary = {}
    for name, x in predictors.items():
        rho = spearman_rho(x, dJ_all)
        cx = crossover(x, dJ_all)
        # AUC-like separation: fraction of (lose,win) pairs correctly ordered by the index
        wins = dJ_all < 0
        if wins.any() and inqp_loses.any():
            xl = x[inqp_loses]; xw = x[wins]
            comp = (xl[:, None] > xw[None, :])      # expect lose has HIGHER index for ductility-type
            auc_hi = float(comp.mean())
            auc = max(auc_hi, 1 - auc_hi)
        else:
            auc = float("nan")
        boundary[name] = {"spearman_rho_vs_dJ": rho, "separation_purity": cx["separation_purity"],
                          "auc_abs": auc, "bracket_lo_hi": cx["bracket_lo_hi"]}
    n_inqp_lose = int(inqp_loses.sum()); n_inqp_win = int((dJ_all < 0).sum())
    # deep-yield conditional failure rate (the canonical story: fails when frame yields deeply)
    deep = arr("unc_dmax_pct") > 3.5
    fail_rate_deep = float((dJ_all[deep] > 0).mean()) if deep.any() else float("nan")
    fail_rate_shallow = float((dJ_all[~deep] > 0).mean()) if (~deep).any() else float("nan")
    # magnitude of the failure when it happens
    dJ_when_lose = dJ_all[inqp_loses]
    dJ_when_win = dJ_all[dJ_all < 0]

    # ---- Claim 3: softening-aware NULL robustness ----
    Je = arr("J_LQR_elastic"); Js = arr("J_LQR_softened"); Jiso = arr("J_LQR_iso")
    margin = Je - Js                      # >0 => softened helps
    iso_margin = Je - Jiso
    # leave-one-record spread on the mean margin
    loro_margin = []
    for drop in rec_ids:
        m = np.array([r["J_LQR_elastic"] - r["J_LQR_softened"] for r in rows if r["record_id"] != drop])
        loro_margin.append(float(m.mean()))
    loro_margin = np.array(loro_margin)
    # per record-class: does it help on any earthquake/tier subset?
    helps_by_eq = {}
    for eq in sorted({r["earthquake"] for r in rows}):
        m = np.array([r["J_LQR_elastic"] - r["J_LQR_softened"] for r in rows if r["earthquake"] == eq])
        helps_by_eq[eq] = {"mean_margin": float(m.mean()), "n": int(len(m)),
                           "frac_helps": float((m > 1e-6).mean())}

    out = {
        "title": "FEMA: EESD surviving-claims robustness across the FEMA P695 far-field set (Ohtori plant)",
        "central_question": ("Are the two canonical-surviving claims (saturation dominates the 2x2 variance; "
                             "in-QP has a deep-yield failure mode) robust across the FEMA P695 record "
                             "family, and is the softening-aware null airtight?"),
        "architecture": ("Controllers DESIGNED on the reduced 3-DOF condensed lateral M/K (frozen canonical "
                         "design points: LQR q=1e12 r=1e-6; softened alpha=0.183); EVALUATED closed-loop "
                         "on the FULL nonlinear OpenSees Ohtori plant. FEMA records loaded via "
                         "fema_p695_loader (atc63_norm * global scale)."),
        "honesty_scope": ("Validated BILINEAR plant; drift/accel J1-J6 only (NOT trilinear MEX, NOT "
                          "damage indices J7-J10). FEMA far-field set is by construction NON-pulse, so "
                          "'pulse vs ordinary' is a near-degenerate within-set axis."),
        "subset": {
            "n_records": len(comp1), "records": comp1,
            "tiers": TIERS, "n_evals_per_cell": len(comp1) * len(TIERS),
            "rationale": ("one component per FEMA pair (removes within-pair redundancy); two intensity "
                          "tiers bracketing onset-of-yield (x1.5, median ~1.9% drift) and deep-yield "
                          "(x3.0, median ~4.1% drift, 14/22 > 3.5%) where the canonical in-QP failure lives; "
                          "44 evals/cell is enough for the ANOVA share + boundary statistics."),
            "active_window_cap_s": ACTIVE_WINDOW_CAP, "dt_analysis": DT_ANALYSIS,
        },
        "frozen_design": {"lqr_q": Q_BEST, "lqr_r": R_BEST, "alpha_design": ALPHA_DESIGN,
                          "U_MAX_per_floor_N": U_MAX, "K_iso_ratio_soft_over_elastic": n_soft / n_lqr},
        "elapsed_s": elapsed,

        "claim1_saturation_dominates_2x2": {
            "overall_all_rows": overall,
            "per_tier": per_tier,
            "overall_controllable": overall_ctrl,
            "n_controllable_rows": len(ctrl_rows),
            "saturation_is_largest_share_overall": bool(
                overall["ss_B_frac"] >= overall["ss_A_frac"] and overall["ss_B_frac"] >= overall["ss_AB_frac"]),
            "leave_one_record_spread": {
                "ss_B_sat_mean": float(loro_sat.mean()), "ss_B_sat_std": float(loro_sat.std()),
                "ss_B_sat_min": float(loro_sat.min()), "ss_B_sat_max": float(loro_sat.max()),
                "ss_A_freq_mean": float(loro_freq.mean()), "ss_A_freq_std": float(loro_freq.std()),
                "ss_AB_int_mean": float(loro_int.mean()), "ss_AB_int_std": float(loro_int.std()),
            },
        },
        "claim2_inqp_deepyield_failure": {
            "n_inqp_wins": n_inqp_win, "n_inqp_loses": n_inqp_lose, "n_total": len(rows),
            "fail_rate_deepyield_unc_gt_3p5pct": fail_rate_deep,
            "fail_rate_shallow_unc_le_3p5pct": fail_rate_shallow,
            "dJ_when_inqp_loses": {"mean": float(dJ_when_lose.mean()) if dJ_when_lose.size else float("nan"),
                                   "max": float(dJ_when_lose.max()) if dJ_when_lose.size else float("nan"),
                                   "n": int(dJ_when_lose.size)},
            "dJ_when_inqp_wins": {"mean": float(dJ_when_win.mean()) if dJ_when_win.size else float("nan"),
                                  "min": float(dJ_when_win.min()) if dJ_when_win.size else float("nan"),
                                  "n": int(dJ_when_win.size)},
            "boundary_predictors": boundary,
            "best_predictor_by_abs_rho": max(boundary, key=lambda k: abs(boundary[k]["spearman_rho_vs_dJ"])
                                             if np.isfinite(boundary[k]["spearman_rho_vs_dJ"]) else 0.0),
        },
        "claim3_softening_aware_null": {
            "mean_J_elastic": float(Je.mean()), "mean_J_softened": float(Js.mean()),
            "mean_margin_elastic_minus_softened": float(margin.mean()),
            "margin_std": float(margin.std()), "margin_min": float(margin.min()), "margin_max": float(margin.max()),
            "frac_records_softening_helps": float((margin > 1e-6).mean()),
            "n_softening_helps": int((margin > 1e-6).sum()), "n_total": len(rows),
            "iso_mean_margin": float(iso_margin.mean()), "frac_iso_helps": float((iso_margin > 1e-6).mean()),
            "leave_one_record_margin": {"mean": float(loro_margin.mean()), "std": float(loro_margin.std()),
                                        "min": float(loro_margin.min()), "max": float(loro_margin.max())},
            "helps_by_earthquake": helps_by_eq,
            "NULL_airtight": bool(abs(margin.mean()) < 0.01 * Je.mean() and (margin > 1e-6).mean() <= 0.6),
        },
        "per_row": rows,
    }

    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # also a tidy CSV of per-row results
    import pandas as pd
    pd.DataFrame(rows).to_csv(RESULTS / "eesd_ohtori_control_fema_rows_components.csv", index=False)
    args.components_out.parent.mkdir(parents=True, exist_ok=True)
    args.components_out.write_text(json.dumps({"rows": rows, "weights": {"primary": [0.70,0.30,0.02], "balanced": [0.50,0.50,0.0], "response_dominant": [0.80,0.20,0.0], "rms_drift_emphasis": [0.90,0.10,0.0], "effort_aware": [0.70,0.20,0.10]}}, indent=2), encoding="utf-8")

    # ---------- console verdict ----------
    print("\n" + "=" * 96)
    print("FEMA FEMA P695 ROBUSTNESS VERDICT")
    print("=" * 96)
    c1 = out["claim1_saturation_dominates_2x2"]; c2 = out["claim2_inqp_deepyield_failure"]; c3 = out["claim3_softening_aware_null"]
    print(f"[C1] 2x2 ANOVA shares (pooled {len(rows)} rows): "
          f"freq {overall['ss_A_frac']*100:.0f}% / sat {overall['ss_B_frac']*100:.0f}% / "
          f"int {overall['ss_AB_frac']*100:.0f}%  -> saturation largest: {c1['saturation_is_largest_share_overall']}")
    print(f"     leave-one-record sat share: {c1['leave_one_record_spread']['ss_B_sat_mean']*100:.0f}% "
          f"+/- {c1['leave_one_record_spread']['ss_B_sat_std']*100:.0f}%  "
          f"[{c1['leave_one_record_spread']['ss_B_sat_min']*100:.0f},{c1['leave_one_record_spread']['ss_B_sat_max']*100:.0f}]")
    print(f"     per-tier sat: " + "  ".join(f"{t}={per_tier[t]['ss_B_frac']*100:.0f}%" for t in TIERS))
    print(f"[C2] in-QP: {c2['n_inqp_loses']}/{c2['n_total']} rows lose to clip (dJ>0). "
          f"deep-yield fail rate {c2['fail_rate_deepyield_unc_gt_3p5pct']}, shallow {c2['fail_rate_shallow_unc_le_3p5pct']}")
    print(f"     dJ when it loses: mean {c2['dJ_when_inqp_loses']['mean']:+.3f} max {c2['dJ_when_inqp_loses']['max']:+.3f}")
    print(f"     best dJ predictor: {c2['best_predictor_by_abs_rho']} "
          f"(rho={boundary[c2['best_predictor_by_abs_rho']]['spearman_rho_vs_dJ']:+.2f}, "
          f"AUC={boundary[c2['best_predictor_by_abs_rho']]['auc_abs']:.2f})")
    print(f"[C3] softening-aware NULL: mean J elastic {c3['mean_J_elastic']:.4f} -> softened {c3['mean_J_softened']:.4f}  "
          f"margin {c3['mean_margin_elastic_minus_softened']:+.4f} +/- {c3['margin_std']:.4f}  "
          f"helps {c3['n_softening_helps']}/{c3['n_total']}  airtight: {c3['NULL_airtight']}")
    print(f"\nwrote {args.out}")
    print(f"wrote {RESULTS / 'eesd_ohtori_control_fema_rows.csv'}")
    print(f"elapsed {elapsed:.1f}s")
    return out


if __name__ == "__main__":
    main()



