r"""SUPER-ENHANCEMENT 3 (EESD): near-fault PULSE demand, closing the paper's explicit
"non-pulse far-field only" limitation. The FEMA P695 far-field set is by construction non-pulse;
this probes whether the authority-allocation principle survives a coherent near-fault velocity
pulse, which loads a saturating actuator far harder than ordinary broadband motion.

We use synthetic Mavroeidis-Papageorgiou (2003) velocity pulses (a standard analytical near-fault
pulse), sweeping pulse period Tp around the structure's fundamental period and three amplitudes, and
run the same 2x2 factorial {elastic,softened}x{clip,in-QP} on the VALIDATED 3-story plant. These
are a controlled probe of pulse-type demand, NOT a substitute for recorded near-fault motions; the
honest claim is robustness of the principle to demand TYPE, reported as found.

ENV: .venv312. Run: ... src\eesd_nearfault_pulse.py [--smoke]
"""
from __future__ import annotations
import sys
import json
import math
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

SRC = Path(__file__).absolute().parent          # .absolute(): keep mapped W:, not raw UNC
RESULTS = SRC.parent / "results"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DT = 0.005
TPS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]     # pulse periods bracketing T1~1.0s (3-story)
VPS = [0.6, 1.0, 1.6]                            # peak pulse velocities (m/s), near-fault range
GAMMA = 2.0                                      # oscillation parameter (number of half cycles)
NU = 0.0                                         # phase
CELLS = [("elastic", "clip"), ("softened", "clip"), ("elastic", "inqp"), ("softened", "inqp"), (None, None)]
ALPHA_DESIGN = 0.183
MAX_WORKERS = 32
WORKER_SENTINEL = "RESULT_JSON:"


def mp_pulse_accel(Tp, Vp, gamma=GAMMA, nu=NU, dt=DT, tail=6.0):
    """Mavroeidis-Papageorgiou velocity pulse -> ground acceleration (m/s^2)."""
    fp = 1.0 / Tp
    dur = gamma / fp
    t0 = dur / 2.0 + 1.0
    t_end = dur + 1.0 + tail
    t = np.arange(0.0, t_end, dt)
    v = np.zeros_like(t)
    m = np.abs(t - t0) <= dur / 2.0
    tt = t[m] - t0
    v[m] = 0.5 * Vp * (1.0 + np.cos(2 * np.pi * fp * tt / gamma)) * np.cos(2 * np.pi * fp * tt + nu)
    a = np.gradient(v, dt)                        # ground acceleration
    return t, a


def run_one(task):
    Tp, Vp, tuning, handling = task
    try:
        import numpy as np
        import openseespy.opensees as ops
        from eesd_ohtori_control import FactorialMPC, U_MAX, decompose_2x2, j_gate  # noqa
        from eesd_ohtori_control_fema import FemaPlantSession, DT_ANALYSIS
        from eesd_ohtori_plant import parse_struct3, BuildConfig, OHTORI_DIR, build_model, apply_rayleigh, eigen_frequencies

        class PulseSession(FemaPlantSession):
            def _setup(self, record, var, intensity, tf, active_window, dt_analysis):
                build_model(self.plant, self.cfg)
                self.rayleigh = apply_rayleigh(self.cfg.damping_ratio)
                self.f0 = eigen_frequencies(3)
                times, accel = self._pulse
                self.times, self.accel, self.rec_dt = times, accel, dt_analysis
                ops.timeSeries("Path", 1001, "-dt", dt_analysis, "-values", *accel.tolist(), "-factor", 1.0)
                ops.pattern("UniformExcitation", 2001, 1, "-accel", 1001)
                ops.timeSeries("Constant", 1002)
                ops.pattern("Plain", 2002, 1002)
                for nd in self.floor_nodes:
                    ops.load(nd, 0.0, 0.0, 0.0)
                ops.wipeAnalysis()
                ops.constraints("Transformation"); ops.numberer("RCM"); ops.system("BandGeneral")
                ops.test("NormDispIncr", 1e-8, 40); ops.algorithm("Newton")
                ops.integrator("Newmark", 0.5, 0.25); ops.analysis("Transient")
                self.tf_use = min(tf, float(times[-1])); self.dt_analysis = dt_analysis

        t, a = mp_pulse_accel(Tp, Vp)
        plant = parse_struct3(OHTORI_DIR / "struct3.m")
        cfg = BuildConfig(mode="nonlinear")
        sess = PulseSession(plant, cfg)
        sess._pulse = (t, a)
        ctrl = None if handling is None else FactorialMPC(DT_ANALYSIS, tuning, handling, ALPHA_DESIGN)
        res = sess.run(ctrl, "pulse", "p", 1.0, float(t[-1]), float(t[-1]), dt_analysis=DT_ANALYSIS)
        return {"Tp": Tp, "Vp": Vp, "tuning": tuning or "none", "handling": handling or "none",
                "d_max": float(res["d_max"]), "d_rms": float(res.get("d_rms", float("nan"))),
                "u_peak": float(res.get("u_peak", float("nan"))), "unstable": bool(res["unstable"])}
    except Exception as e:
        import traceback
        return {"Tp": Tp, "Vp": Vp, "tuning": tuning or "none", "handling": handling or "none",
                "error": traceback.format_exc()[-300:]}


def dispatch(task):
    Tp, Vp, tuning, handling = task
    args = [sys.executable, str(SRC / "eesd_nearfault_pulse.py"), "--worker",
            str(Tp), str(Vp), tuning or "none", handling or "none"]
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=600, cwd=str(SRC))
        for line in p.stdout.splitlines():
            if line.startswith(WORKER_SENTINEL):
                return json.loads(line[len(WORKER_SENTINEL):])
        return {"task": list(task), "error": "no-result rc=%d %s" % (p.returncode, (p.stderr or "")[-160:])}
    except Exception as e:
        return {"task": list(task), "error": repr(e)[:160]}


def main():
    smoke = "--smoke" in sys.argv
    from eesd_ohtori_control import decompose_2x2, j_gate
    tps = TPS[:2] if smoke else TPS
    vps = VPS[:1] if smoke else VPS
    tasks = [(tp, vp, t, h) for tp in tps for vp in vps for (t, h) in CELLS]
    print(f"=== near-fault pulse: {len(tps)} Tp x {len(vps)} Vp x {len(CELLS)} cell = {len(tasks)} runs ===", flush=True)
    results, done = [], 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for fut in as_completed([ex.submit(dispatch, t) for t in tasks]):
            results.append(fut.result()); done += 1
            if done % 20 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} done", flush=True)
    errs = [r for r in results if "error" in r]
    if errs:
        print(f"  WARNING {len(errs)} errored | {errs[0]['error']}", flush=True)

    # per pulse (Tp,Vp): the 2x2 J-cost decomposition (J = drift-based gate vs uncontrolled)
    by_pulse = {}
    for r in results:
        if "error" in r:
            continue
        by_pulse.setdefault((r["Tp"], r["Vp"]), {})[(r["tuning"], r["handling"])] = r
    rows, sat_shares = [], []
    for (tp, vp), cells in by_pulse.items():
        unc = cells.get(("none", "none"))
        if unc is None or not all((t, h) in cells for (t, h) in [("elastic", "clip"), ("softened", "clip"), ("elastic", "inqp"), ("softened", "inqp")]):
            continue
        J = {}
        for (t, h) in [("elastic", "clip"), ("softened", "clip"), ("elastic", "inqp"), ("softened", "inqp")]:
            J[(t, "clip" if h == "clip" else "inqp")] = j_gate(cells[(t, h)], unc)
        J00 = J[("elastic", "clip")]; J10 = J[("softened", "clip")]
        J01 = J[("elastic", "inqp")]; J11 = J[("softened", "inqp")]
        e_om = 0.5 * ((J00 - J10) + (J01 - J11))         # frequency-tuning main effect
        e_sat = 0.5 * ((J00 - J01) + (J10 - J11))        # saturation-handling main effect
        e_int = 0.5 * ((J00 - J10) - (J01 - J11))        # interaction
        ss = e_om ** 2 + e_sat ** 2 + e_int ** 2
        if ss > 1e-12:
            sat_shares.append(e_sat ** 2 / ss)           # orthogonal two-factor variance share (sum-of-squares)
        rows.append({"Tp": tp, "Vp": vp, "unc_drift": unc["d_max"],
                     "J_elastic_clip": J00, "J_elastic_inqp": J01,
                     "sat_share": (e_sat ** 2 / ss) if ss > 1e-12 else float("nan"),
                     "e_sat": e_sat, "e_omega": e_om})
    sat_mean = float(np.mean(sat_shares)) if sat_shares else float("nan")
    inqp_better = [r for r in rows if r["J_elastic_inqp"] < r["J_elastic_clip"]]
    out = {"title": "EESD near-fault pulse robustness (synthetic Mavroeidis-Papageorgiou)",
           "n_pulses": len(rows), "saturation_variance_share_mean": sat_mean,
           "inqp_beats_clip_fraction": (len(inqp_better) / len(rows)) if rows else float("nan"),
           "rows": rows}
    (RESULTS / ("eesd_nearfault_pulse_smoke.json" if smoke else "eesd_nearfault_pulse.json")).write_text(json.dumps(out, indent=2))
    print(f"\n=== NEAR-FAULT PULSE: {len(rows)} pulses ===", flush=True)
    print(f"  saturation-handling variance share (mean over pulses): {sat_mean:.3f}", flush=True)
    print(f"  in-QP beats clip (elastic): {len(inqp_better)}/{len(rows)} pulses", flush=True)
    print(f"  uncontrolled drift range: {min(r['unc_drift'] for r in rows):.3f}-{max(r['unc_drift'] for r in rows):.3f}" if rows else "  no rows", flush=True)
    print(f"wrote eesd_nearfault_pulse{'_smoke' if smoke else ''}.json", flush=True)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        i = sys.argv.index("--worker")
        _t = None if sys.argv[i + 3] == "none" else sys.argv[i + 3]
        _h = None if sys.argv[i + 4] == "none" else sys.argv[i + 4]
        print(WORKER_SENTINEL + json.dumps(run_one((float(sys.argv[i + 1]), float(sys.argv[i + 2]), _t, _h))), flush=True)
    else:
        main()


