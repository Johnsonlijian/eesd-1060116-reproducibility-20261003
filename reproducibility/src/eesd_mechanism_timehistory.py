r"""DEEP MECHANISM extraction for the EESD authority-allocation paper.

Question this script answers with REAL OpenSees time-histories (not assertion):
WHY does the saturation-handling AXIS (clip vs in-QP) dominate the model-fidelity
AXIS (elastic vs softened design frequency)?

It re-runs the VALIDATED full nonlinear Ohtori 3-story plant closed-loop under the
2x2 factorial {elastic,softened} x {clip,inqp} at a DEEP-YIELD case, logging the FULL
per-step control force u(t), inter-story drift(t), and the actuator-saturation flag.
For the in-QP cells it also reconstructs the controller's INTERNAL predicted state
trajectory at the deepest-yield step (the full horizon solution lives in controller.Uprev)
and records what the plant ACTUALLY does over that same horizon -- so the model-plant
divergence that defeats the multi-step in-QP planner can be MEASURED, not asserted.

Three falsifiable mechanism hypotheses, tested on the data:
  H1 (model-fidelity null): under saturation the elastic and softened CLIP force
      histories are nearly identical -> retuning the design frequency cannot move a
      bounded, saturated force. Test: corr(u_elastic_clip, u_softened_clip) ~ 1,
      |drift_peak_elastic - drift_peak_softened| small.
  H2 (allocation dominates): clip vs in-QP force histories DIFFER materially even at
      the same tuning -> how the bounded authority is spent is the live lever.
      Test: corr(u_clip, u_inqp) well below 1; drift_peak differs.
  H3 (in-QP deep-yield failure): the in-QP plan, built on the fixed (elastic) linear
      model, predicts a horizon trajectory that diverges from the yielded plant; the
      solver commits multi-step to a stale model while clip stays myopically robust.
      Test: predicted-vs-actual horizon drift RMS divergence grows with yield depth.

Honesty: whatever the correlations / divergences say is reported as found. The cells,
records and U_MAX are the paper's; nothing is retuned to favor a hypothesis.

ENV: .venv312\Scripts\python.exe (openseespy 3.8.0).
Run:  .venv312\Scripts\python.exe src\eesd_mechanism_timehistory.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import openseespy.opensees as ops

import eesd_ohtori_control as EC
from eesd_ohtori_control import (FactorialMPC, OhtoriPlantSession, U_MAX, TD_DRIFT, N_HORIZON,
                                 R_SCALE, _RM, _continuous_ss, _discretize_zoh, _build_condensed)
from eesd_ohtori_plant import parse_struct3, BuildConfig, OHTORI_DIR

RESULTS = EC.RESULTS
OUT_NPZ = RESULTS / "eesd_mechanism_timehistory.npz"
OUT_JSON = RESULTS / "eesd_mechanism_timehistory_verdict.json"

# deep-yield demonstration case: strong near-fault record, scaled to drive the controlled
# structure well into yield (the precondition for the softening story to be testable at all).
DEMO = ("northrdg", "n", 1.5, 100.0, 30.0)
DT = 0.005
YIELD_TRIGGER = 0.015   # capture the in-QP horizon plan at the first step drift exceeds 1.5%


def design_condensed(tuning: str, alpha_eff: float):
    """Recompute the design-stiffness condensed prediction matrices Phi, Gamma (for trajectory
    reconstruction) -- exactly the matrices the FactorialMPC's QP is built on."""
    Kdes = _RM.K0 if tuning == "elastic" else _RM.K_softened(alpha_eff)
    A, B, n = _continuous_ss(Kdes)
    Ad, Bd = _discretize_zoh(A, B, DT)
    Phi, Gamma, nx, nu = _build_condensed(Ad, Bd, N_HORIZON)
    return Phi, Gamma, nx, nu


class MechSession(OhtoriPlantSession):
    """Plant session that logs FULL per-step histories and (for in-QP) the predicted horizon plan."""

    def run_logged(self, controller, record, var, intensity, tf, active_window,
                   dt_analysis=DT, capture_plan=False, plan_matrices=None):
        self._setup(record, var, intensity, tf, active_window, dt_analysis)
        if controller is not None and hasattr(controller, "reset"):
            controller.reset()
        n_steps = int(math.ceil(self.tf_use / dt_analysis))
        T, DR, UU, ACC = [], [], [], []
        unstable = False
        # plan capture state
        plan = None
        plan_armed = capture_plan
        plan_record_left = 0
        plan_actual_d = []
        for step in range(n_steps):
            d, v = self._state()
            x = np.concatenate([d, v])
            if controller is not None:
                u = np.clip(controller(x), -U_MAX, U_MAX)
            else:
                u = np.zeros(3)
            # ---- arm/capture the in-QP horizon plan at first deep-yield step ----
            interstory_now = np.concatenate([[d[0]], np.diff(d)]) / self.heights
            if plan_armed and controller is not None and float(np.max(np.abs(interstory_now))) > YIELD_TRIGGER:
                Phi, Gamma, nx, nu = plan_matrices
                U_full = np.asarray(controller.Uprev, float)          # full horizon solution
                pred_states = Phi @ x + Gamma @ U_full                # N blocks of [d(3),v(3)]
                pred_d = pred_states.reshape(N_HORIZON, nx)[:, :3]    # predicted floor disp over horizon
                pred_drift = (np.column_stack([pred_d[:, 0], np.diff(pred_d, axis=1)[:, 0],
                                               np.diff(pred_d, axis=1)[:, 1]]) / self.heights)
                plan = {"step": step, "t": float(ops.getTime()),
                        "x": x.copy(), "U_full": U_full.copy(),
                        "pred_drift": pred_drift.copy()}
                plan_armed = False
                plan_record_left = N_HORIZON
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
            interstory = np.concatenate([[d[0]], np.diff(d)]) / self.heights
            T.append(t); DR.append(interstory.copy()); UU.append(u.copy()); ACC.append(abs_acc.copy())
            if plan_record_left > 0:
                plan_actual_d.append(interstory.copy())
                plan_record_left -= 1
        out = {"t": np.array(T), "drift": np.array(DR), "u": np.array(UU), "acc": np.array(ACC),
               "unstable": unstable}
        if plan is not None:
            plan["actual_drift"] = np.array(plan_actual_d)
            out["plan"] = plan
        return out


def corr(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def sat_fraction(u):
    """fraction of steps with at least one floor actuator at the saturation bound."""
    return float(np.mean(np.any(np.abs(u) >= 0.999 * U_MAX, axis=1)))


def analyze_case(record, var, I, tf, aw, save_npz=False):
    print(f"=== DEEP MECHANISM time-histories on VALIDATED Ohtori plant: {record} x{I} ===")
    plant = parse_struct3(OHTORI_DIR / "struct3.m")
    cfg = BuildConfig(mode="nonlinear")

    # 1) uncontrolled + measure alpha_eff (design the softened controller at the MEASURED softening)
    unc = MechSession(plant, cfg).run_logged(None, record, var, I, tf, aw)
    # measure alpha_eff from the uncontrolled tangent-frequency drop (same protocol as the paper)
    res_alpha = OhtoriPlantSession(plant, cfg).run(None, record, var, I, tf, aw, measure_alpha=True)
    alpha_eff = float(res_alpha.get("alpha_eff_min", 0.2))
    alpha_eff = min(max(alpha_eff, 1e-3), 0.999)
    print(f"  measured alpha_eff (uncontrolled tangent) = {alpha_eff:.4f}  "
          f"-> omega_s/omega_1 = {math.sqrt(alpha_eff):.3f}")
    print(f"  uncontrolled drift_peak = {np.max(np.abs(unc['drift'])):.4%}")

    # 2) the 2x2 factorial, full histories; capture the in-QP horizon plan at deep yield
    cells = {}
    for tuning in ("elastic", "softened"):
        plan_mats = design_condensed(tuning, alpha_eff)
        for handling in ("clip", "inqp"):
            ctrl = FactorialMPC(DT, tuning, handling, alpha_eff)
            log = MechSession(plant, cfg).run_logged(ctrl, record, var, I, tf, aw,
                                           capture_plan=(handling == "inqp"),
                                           plan_matrices=plan_mats)
            cells[(tuning, handling)] = log
            dpk = np.max(np.abs(log["drift"])) if len(log["drift"]) else float("nan")
            print(f"  {tuning:8s} {handling:4s}: drift_peak={dpk:.4%}  "
                  f"sat_frac={sat_fraction(log['u']):.3f}  unstable={log['unstable']}")

    # ---- H1: model-fidelity null under saturation (elastic-clip vs softened-clip force) ----
    ec, sc = cells[("elastic", "clip")], cells[("softened", "clip")]
    ei, si = cells[("elastic", "inqp")], cells[("softened", "inqp")]
    n = min(len(ec["u"]), len(sc["u"]), len(ei["u"]), len(si["u"]))

    def dpk(c):
        return float(np.max(np.abs(c["drift"]))) if len(c["drift"]) else float("nan")

    H1 = {
        "corr_u_elastic_vs_softened_CLIP": corr(ec["u"][:n], sc["u"][:n]),
        "drift_peak_elastic_clip": dpk(ec), "drift_peak_softened_clip": dpk(sc),
        "abs_drift_peak_diff_clip": abs(dpk(ec) - dpk(sc)),
        "sat_frac_elastic_clip": sat_fraction(ec["u"]), "sat_frac_softened_clip": sat_fraction(sc["u"]),
    }
    # ---- H2: allocation dominates (clip vs in-QP force at the SAME tuning) ----
    H2 = {
        "corr_u_clip_vs_inqp_ELASTIC": corr(ec["u"][:n], ei["u"][:n]),
        "corr_u_clip_vs_inqp_SOFTENED": corr(sc["u"][:n], si["u"][:n]),
        "drift_peak_elastic_clip": dpk(ec), "drift_peak_elastic_inqp": dpk(ei),
        "drift_peak_softened_clip": dpk(sc), "drift_peak_softened_inqp": dpk(si),
    }
    # ---- H3: in-QP deep-yield failure (predicted vs actual horizon drift divergence) ----
    H3 = {}
    for tag, c in (("elastic_inqp", ei), ("softened_inqp", si)):
        pl = c.get("plan")
        if pl is not None and len(pl.get("actual_drift", [])) >= 3:
            m = min(len(pl["pred_drift"]), len(pl["actual_drift"]))
            pred = pl["pred_drift"][:m]
            act = pl["actual_drift"][:m]
            # compare the controlled-DOF drift magnitude over the horizon
            pred_mag = np.max(np.abs(pred), axis=1)
            act_mag = np.max(np.abs(act), axis=1)
            rms_div = float(np.sqrt(np.mean((pred_mag - act_mag) ** 2)))
            H3[tag] = {
                "trigger_t": pl["t"], "horizon_steps": int(m),
                "pred_drift_horizon_end": float(pred_mag[-1]),
                "actual_drift_horizon_end": float(act_mag[-1]),
                "rms_pred_minus_actual": rms_div,
                "pred_over_actual_end": float(pred_mag[-1] / max(act_mag[-1], 1e-9)),
            }

    verdict = {
        "title": "EESD authority-allocation mechanism: real-plant time-history evidence",
        "case": f"{record}_x{I}", "alpha_eff_measured": alpha_eff,
        "uncontrolled_drift_peak": float(np.max(np.abs(unc["drift"]))),
        "H1_model_fidelity_null_under_saturation": H1,
        "H2_allocation_dominates": H2,
        "H3_inqp_deepyield_planmismatch": H3,
    }
    if save_npz:
        def ds(a, k=2):
            return np.asarray(a)[::k]
        pl = ei.get("plan", {})
        np.savez_compressed(
            RESULTS / f"eesd_mechanism_th_{record}_x{I}.npz",
            t=ds(ec["t"]), drift_unc=ds(unc["drift"]),
            u_elastic_clip=ds(ec["u"]), u_softened_clip=ds(sc["u"]),
            u_elastic_inqp=ds(ei["u"]), u_softened_inqp=ds(si["u"]),
            drift_elastic_clip=ds(ec["drift"]), drift_softened_clip=ds(sc["drift"]),
            drift_elastic_inqp=ds(ei["drift"]), drift_softened_inqp=ds(si["drift"]),
            plan_pred_drift=np.asarray(pl.get("pred_drift", np.zeros((0, 3)))),
            plan_actual_drift=np.asarray(pl.get("actual_drift", np.zeros((0, 3)))),
            plan_t=float(pl.get("t", 0.0)),
            alpha_eff=alpha_eff, U_MAX=U_MAX,
        )
        print(f"  wrote {OUT_NPZ.name} (deepest-case histories for the figure)")
    print(json.dumps(verdict, indent=2))
    return verdict


# the deep-yield sweep: span shallow -> deep yield on strong records so the in-QP failure
# REGIME and the plan-divergence growth with yield depth are both captured on one plant.
SWEEP = [
    ("northrdg", "n", 1.0, 100.0, 30.0),
    ("northrdg", "n", 1.5, 100.0, 30.0),
    ("northrdg", "n", 2.0, 100.0, 30.0),
    ("kobe", "k", 1.0, 180.0, 60.0),
]


def main():
    verdicts = []
    for (record, var, I, tf, aw) in SWEEP:
        save = (record == "northrdg" and I >= 1.5)   # deep-yield cases -> npz for figure
        try:
            verdicts.append(analyze_case(record, var, I, tf, aw, save_npz=save))
        except Exception as e:
            verdicts.append({"case": f"{record}_x{I}", "error": repr(e)})
            print(f"  CASE FAILED {record} x{I}: {e!r}")
    OUT_JSON.write_text(json.dumps({"sweep": verdicts}, indent=2))
    print(f"\n=== SWEEP COMPLETE: {len(verdicts)} cases -> {OUT_JSON.name} ===")
    print("\ncase            alpha_eff  unc_drift  H1corr  H2corr  inqp-clip(E)  predDiv(E)")
    for v in verdicts:
        if "error" in v:
            print(f"  {v['case']:14s}  FAILED {v['error'][:40]}"); continue
        h1, h2, h3 = (v["H1_model_fidelity_null_under_saturation"],
                      v["H2_allocation_dominates"], v["H3_inqp_deepyield_planmismatch"])
        margin = h2["drift_peak_elastic_inqp"] - h2["drift_peak_elastic_clip"]  # >0 => in-QP worse
        pdiv = h3.get("elastic_inqp", {}).get("rms_pred_minus_actual", float("nan"))
        print(f"  {v['case']:14s}  {v['alpha_eff_measured']:.4f}    {v['uncontrolled_drift_peak']:.4f}   "
              f"{h1['corr_u_elastic_vs_softened_CLIP']:.4f}  {h2['corr_u_clip_vs_inqp_ELASTIC']:.4f}  "
              f"{margin:+.5f}     {pdiv:.5f}")


if __name__ == "__main__":
    main()



