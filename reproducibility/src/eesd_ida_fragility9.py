r"""SUPER-ENHANCEMENT 1b (EESD): corrected nine-story incremental dynamic analysis and collapse fragility.

This script evaluates the same five closed-loop conditions as the three-story IDA on the
response-validated nine-story nonlinear plant. The retained analysis freezes the canonical
height-specific \(\alpha_{\mathrm{design}}=0.3624908788\) and uses 22 FEMA P695 far-field
records over 13 intensity levels. The corrected run completed all 1,430 analyses with zero
errors. The resulting elastic in-QP allocation is nearly neutral to clipping at the collapse
boundary (median gain \(-0.012\)); the softened height-specific in-QP design is higher by
about \(+0.158\). These are bounded, record-set comparative results rather than a universal
collapse-limit rule.

Run: python src/eesd_ida_fragility9.py [--smoke]
"""
from __future__ import annotations
import os
import sys
import json
import math
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

SRC = Path(__file__).resolve().parent
PACKAGE_ROOT = SRC.parents[1]
PROJECT_ROOT = Path(os.environ.get("EESD_PROJECT_ROOT", PACKAGE_ROOT))
RESULTS = Path(os.environ.get("EESD_OUTPUT_DIR", PACKAGE_ROOT / "analysis"))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SCALES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0]  # 9-story is milder; extend
CELLS = [("elastic", "clip"), ("softened", "clip"), ("elastic", "inqp"), ("softened", "inqp"), (None, None)]
COLLAPSE_DRIFT = 0.10
ALPHA_DESIGN = 0.3624908787712646
MAX_WORKERS = 8
WORKER_SENTINEL = "RESULT_JSON:"


def run_one(task):
    rid, scale, tuning, handling = task
    try:
        import numpy as np
        import openseespy.opensees as ops
        import eesd_ohtori_control9 as E9
        from eesd_ohtori_control9 import Ohtori9PlantSession, FactorialMPC, U_MAX
        from eesd_ohtori_plant9 import (parse_struct9, BuildConfig, OHTORI_DIR, build_model,
                                        apply_rayleigh, eigen_frequencies)
        import fema_p695_loader as F
        # Optional explicit path remap for worker subprocesses when the record location differs.
        record_unc = os.environ.get("EESD_RECORD_UNC_PREFIX")
        record_mapped = os.environ.get("EESD_RECORD_MAPPED_PREFIX", "")
        for _a in ("FF_DIR", "MANIFEST"):
            _v = getattr(F, _a, None)
            if _v is not None and record_unc and record_unc in str(_v):
                setattr(F, _a, Path(str(_v).replace(record_unc, record_mapped)))

        class FemaSession9(Ohtori9PlantSession):
            def _setup(self, record, var, intensity, tf, active_window, dt_analysis):
                build_model(self.plant, self.cfg)
                self.rayleigh = apply_rayleigh(self.cfg.damping_ratio)
                self.f0 = eigen_frequencies(3)
                t_rec, ag = F.load_record(record)
                ag = ag * F.atc63_norm_factor(record) * intensity
                rec_dt = F.record_dt(record)
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

        plant = parse_struct9(OHTORI_DIR / "struct9.m")
        cfg = BuildConfig(mode="nonlinear")
        sess = FemaSession9(plant, cfg)
        ctrl = None if handling is None else FactorialMPC(0.005, tuning, handling, ALPHA_DESIGN)
        res = sess.run(ctrl, rid, "f", scale, 100.0, 50.0, dt_analysis=0.005)
        dmax = float(res["d_max"])
        unstable = bool(res["unstable"])
        collapse = unstable or (dmax > COLLAPSE_DRIFT) or (dmax >= 1e8)
        return {"rid": rid, "scale": scale, "tuning": tuning or "none", "handling": handling or "none",
                "d_max": float(min(dmax, 1.0)), "unstable": unstable, "collapse": bool(collapse)}
    except Exception as e:
        import traceback
        return {"rid": rid, "scale": scale, "tuning": tuning or "none", "handling": handling or "none",
                "d_max": float("nan"), "unstable": True, "collapse": True, "error": traceback.format_exc()[-600:]}


def dispatch(task):
    rid, scale, tuning, handling = task
    args = [sys.executable, str(SRC / "eesd_ida_fragility9.py"), "--worker",
            rid, str(scale), tuning or "none", handling or "none"]
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=900, cwd=str(SRC))
        for line in p.stdout.splitlines():
            if line.startswith(WORKER_SENTINEL):
                return json.loads(line[len(WORKER_SENTINEL):])
        return {"rid": rid, "scale": scale, "tuning": tuning or "none", "handling": handling or "none",
                "d_max": float("nan"), "unstable": True, "collapse": True,
                "error": "no-result rc=%d %s" % (p.returncode, (p.stderr or "")[-160:])}
    except Exception as e:
        return {"rid": rid, "scale": scale, "tuning": tuning or "none", "handling": handling or "none",
                "d_max": float("nan"), "unstable": True, "collapse": True, "error": repr(e)[:160]}


def collapse_capacity(points):
    cs = sorted((p["scale"] for p in points if p["collapse"]))
    return cs[0] if cs else math.inf


def lognormal_fragility(caps, scales):
    finite = [c for c in caps if math.isfinite(c)]
    pcurve = {s: float(np.mean([c <= s for c in caps])) for s in scales}
    if len(finite) >= 3:
        ln = np.log(finite); med = float(np.exp(np.mean(ln))); beta = float(np.std(ln, ddof=1))
    else:
        med = float("inf") if not finite else float(np.median(finite)); beta = float("nan")
    return {"median_capacity": med, "beta": beta, "n_collapsed": len(finite),
            "n_records": len(caps), "P_collapse_by_scale": pcurve}


def main():
    smoke = "--smoke" in sys.argv
    import fema_p695_loader as F
    rids = [p[0] for p in F.record_pairs()]
    scales = SCALES
    if smoke:
        rids, scales = rids[:2], [1.0, 3.0, 6.0]
    all_tasks = [(r, s, t, h) for r in rids for s in scales for (t, h) in CELLS]
    prog = RESULTS / "eesd_ida_fragility9_progress.jsonl"
    results, done_keys = [], set()
    if prog.exists() and not smoke:                       # resume: skip already-computed points
        for line in prog.read_text().splitlines():
            try:
                r = json.loads(line); results.append(r)
                done_keys.add((r["rid"], float(r["scale"]), r["tuning"], r["handling"]))
            except Exception:
                pass
    tasks = [t for t in all_tasks if (t[0], float(t[1]), t[2] or "none", t[3] or "none") not in done_keys]
    print(f"=== 9-story IDA/fragility: {len(all_tasks)} runs | {len(results)} resumed | {len(tasks)} to run ===", flush=True)
    done = 0
    with prog.open("a") as pf:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            for fut in as_completed([ex.submit(dispatch, t) for t in tasks]):
                r = fut.result(); results.append(r)
                pf.write(json.dumps(r) + "\n"); pf.flush()
                done += 1
                if done % 50 == 0 or done == len(tasks):
                    print(f"  {done}/{len(tasks)} done", flush=True)
    errs = [r for r in results if "error" in r]
    if errs:
        print(f"  WARNING {len(errs)} errored | {errs[0]['error']}", flush=True)
    cells = {}
    for (t, h) in CELLS:
        ck = f"{t or 'none'}_{h or 'none'}"
        by_rid = {}
        for r in results:
            if r["tuning"] == (t or "none") and r["handling"] == (h or "none"):
                by_rid.setdefault(r["rid"], []).append(r)
        caps = [collapse_capacity(p) for p in by_rid.values()]
        ida = {}
        for s in scales:
            ds = [min(ps[0]["d_max"], COLLAPSE_DRIFT) if not ps[0]["collapse"] else COLLAPSE_DRIFT
                  for ps in ([pp for pp in pts if abs(pp["scale"] - s) < 1e-9] for pts in by_rid.values()) if ps]
            ida[s] = float(np.median(ds)) if ds else float("nan")
        cells[ck] = {"fragility": lognormal_fragility(caps, scales), "median_ida_drift": ida}
    med = lambda c: cells[c]["fragility"]["median_capacity"]
    contrast = {"median_collapse_capacity": {c: cells[c]["fragility"]["median_capacity"] for c in cells},
                "allocation_gain_elastic_inqp_over_clip": med("elastic_inqp") - med("elastic_clip"),
                "allocation_gain_softened_inqp_over_clip": med("softened_inqp") - med("softened_clip"),
                "fidelity_gain_softened_over_elastic_clip": med("softened_clip") - med("elastic_clip")}
    out = {"title": "EESD 9-story IDA + collapse fragility", "height": 9, "n_records": len(rids),
           "scales": scales, "collapse_drift": COLLAPSE_DRIFT, "cells": cells, "contrast": contrast}
    (RESULTS / ("eesd_ida_fragility9_smoke.json" if smoke else "eesd_ida_fragility9.json")).write_text(json.dumps(out, indent=2))
    print("\n=== 9-STORY COLLAPSE FRAGILITY (median capacity) ===", flush=True)
    for ck in ["none_none", "elastic_clip", "softened_clip", "elastic_inqp", "softened_inqp"]:
        if ck in cells:
            f = cells[ck]["fragility"]
            mc = f["median_capacity"]
            print(f"  {ck:16s}  median={mc if math.isinf(mc) else round(mc,2)}  beta={f['beta']:.3f}  collapsed {f['n_collapsed']}/{f['n_records']}", flush=True)
    print("  ALLOCATION gain (in-QP - clip): elastic %+.3f | softened %+.3f" % (
        contrast["allocation_gain_elastic_inqp_over_clip"], contrast["allocation_gain_softened_inqp_over_clip"]), flush=True)
    print(f"wrote eesd_ida_fragility9{'_smoke' if smoke else ''}.json", flush=True)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        i = sys.argv.index("--worker")
        _t = None if sys.argv[i + 3] == "none" else sys.argv[i + 3]
        _h = None if sys.argv[i + 4] == "none" else sys.argv[i + 4]
        print(WORKER_SENTINEL + json.dumps(run_one((sys.argv[i + 1], float(sys.argv[i + 2]), _t, _h))), flush=True)
    else:
        main()



