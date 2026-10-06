r"""SUPER-ENHANCEMENT 1 (EESD): Incremental Dynamic Analysis + collapse fragility.

The published EESD result quantifies the authority-allocation principle on MEAN drift
(2x2 ANOVA, J-cost). This adds the earthquake-engineering gold-standard metric: does the
allocation axis (clip vs in-QP) move the COLLAPSE CAPACITY and the collapse FRAGILITY of
the controlled frame, not just the mean response?

Method: full IDA on the VALIDATED nonlinear Ohtori 3-story plant. For each of the 22 FEMA
P695 far-field records (one component per pair), scale from 0.5 to 7.0 x ATC-63-normalized
intensity and run the closed loop for the four 2x2 cells {elastic,softened}x{clip,in-QP}
plus uncontrolled. Collapse = dynamic instability or governing drift > 10%. Per record/cell
the collapse capacity is the lowest scale at incipient collapse; pooled over records this
gives a lognormal collapse fragility per cell, and the allocation-vs-fidelity contrast is
read on the fragility medians, exactly as in the mean-drift ANOVA but at the collapse limit.

Honest: whatever the fragilities say is reported. The scale axis is the IM (proportional to
Sa since records are ATC-63 normalized); the plant is the validated bilinear surrogate, so
"collapse" here is a drift-based sidesway proxy, not a trilinear damage-index collapse.

ENV: python (openseespy). Heavy parallel (96-core box).
Run:  python src\eesd_ida_fragility.py --smoke   (2 records, validate)
      python src\eesd_ida_fragility.py           (full 22-record IDA)
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

# Keep worker paths package-relative or explicitly supplied through environment variables.
SRC = Path(__file__).resolve().parent
PACKAGE_ROOT = SRC.parents[1]
PROJECT_ROOT = Path(os.environ.get("EESD_PROJECT_ROOT", PACKAGE_ROOT))
RESULTS = Path(os.environ.get("EESD_OUTPUT_DIR", PACKAGE_ROOT / "analysis"))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SCALES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0]
CELLS = [("elastic", "clip"), ("softened", "clip"),
         ("elastic", "inqp"), ("softened", "inqp"), (None, None)]   # last = uncontrolled
COLLAPSE_DRIFT = 0.10
MAX_WORKERS = 46


def run_one(task):
    """Top-level worker (picklable). One closed-loop IDA point. Robust to instability."""
    rid, scale, tuning, handling = task
    try:
        from eesd_ohtori_control import FactorialMPC
        from eesd_ohtori_control_fema import FemaPlantSession, ALPHA_DESIGN, DT_ANALYSIS, ACTIVE_WINDOW_CAP
        from eesd_ohtori_plant import parse_struct3, BuildConfig, OHTORI_DIR
        plant = parse_struct3(OHTORI_DIR / "struct3.m")
        cfg = BuildConfig(mode="nonlinear")
        sess = FemaPlantSession(plant, cfg)
        ctrl = None if handling is None else FactorialMPC(DT_ANALYSIS, tuning, handling, ALPHA_DESIGN)
        res = sess.run_fema(ctrl, rid, scale, ACTIVE_WINDOW_CAP, DT_ANALYSIS)
        dmax = float(res["d_max"])
        unstable = bool(res["unstable"])
        collapse = unstable or (dmax > COLLAPSE_DRIFT) or (dmax >= 1e8)
        return {"rid": rid, "scale": scale, "tuning": tuning or "none", "handling": handling or "none",
                "d_max": float(min(dmax, 1.0)), "unstable": unstable, "collapse": bool(collapse)}
    except Exception as e:  # never let one point kill the sweep
        return {"rid": rid, "scale": scale, "tuning": tuning or "none", "handling": handling or "none",
                "d_max": float("nan"), "unstable": True, "collapse": True, "error": repr(e)[:120]}


WORKER_SENTINEL = "RESULT_JSON:"


def run_task_subprocess(task):
    """Run one IDA point in an ISOLATED subprocess: clean per-task openseespy import (the DLL does
    not survive a Windows process-pool spawn) and full crash isolation (a segfault fails only this
    point). This is the proven Start-Process-style parallelism for this box."""
    rid, scale, tuning, handling = task
    args = [sys.executable, str(Path(__file__).absolute()), "--worker",
            rid, str(scale), tuning or "none", handling or "none"]
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=600, cwd=str(SRC))
        for line in p.stdout.splitlines():
            if line.startswith(WORKER_SENTINEL):
                return json.loads(line[len(WORKER_SENTINEL):])
        return {"rid": rid, "scale": scale, "tuning": tuning or "none", "handling": handling or "none",
                "d_max": float("nan"), "unstable": True, "collapse": True,
                "error": "no-result rc=%d %s" % (p.returncode, (p.stderr or p.stdout)[-160:])}
    except Exception as e:
        return {"rid": rid, "scale": scale, "tuning": tuning or "none", "handling": handling or "none",
                "d_max": float("nan"), "unstable": True, "collapse": True, "error": repr(e)[:160]}


def cell_key(tuning, handling):
    return f"{tuning}_{handling}"


def collapse_capacity(points):
    """lowest scale at incipient collapse for one (rid, cell); inf if never collapses in range."""
    cs = sorted((p["scale"] for p in points if p["collapse"]))
    return cs[0] if cs else math.inf


def lognormal_fragility(capacities, scales):
    """counting fragility P(collapse<=scale) and a lognormal (median, beta) MLE on finite capacities."""
    finite = [c for c in capacities if math.isfinite(c)]
    n = len(capacities)
    pcurve = {s: float(np.mean([c <= s for c in capacities])) for s in scales}
    if len(finite) >= 3:
        ln = np.log(finite)
        med = float(np.exp(np.mean(ln)))
        beta = float(np.std(ln, ddof=1))
    else:
        med = float("inf") if not finite else float(np.median(finite))
        beta = float("nan")
    return {"median_capacity": med, "beta": beta, "n_collapsed": len(finite),
            "n_records": n, "P_collapse_by_scale": pcurve}


def main():
    smoke = "--smoke" in sys.argv
    import fema_p695_loader as F
    pairs = F.record_pairs() if hasattr(F, "record_pairs") and callable(F.record_pairs) else None
    if pairs:
        rids = [p[0] for p in pairs]           # one component per pair (22)
    else:
        rids = list(F.all_record_ids())[::2]
    scales = SCALES
    if smoke:
        rids = rids[:2]
        scales = [1.0, 3.0, 5.0]
    tasks = [(rid, s, t, h) for rid in rids for s in scales for (t, h) in CELLS]
    print(f"=== EESD IDA/fragility: {len(rids)} records x {len(scales)} scales x {len(CELLS)} cells "
          f"= {len(tasks)} runs, {MAX_WORKERS} workers ===", flush=True)

    results = []
    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(run_task_subprocess, t) for t in tasks]
        for fut in as_completed(futs):
            results.append(fut.result())
            done += 1
            if done % 50 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} done", flush=True)
    errs = [r for r in results if "error" in r]
    if errs:
        print(f"  WARNING {len(errs)}/{len(results)} errored | sample: {errs[0].get('error')}", flush=True)
    nz = [r for r in results if not r.get("collapse") ]
    print(f"  non-collapse runs: {len(nz)}/{len(results)} | sample d_max: "
          f"{[round(r['d_max'],3) for r in results[:6]]}", flush=True)

    # ---- aggregate per cell: IDA curves + collapse fragility ----
    cells = {}
    for (t, h) in CELLS:
        ck = cell_key(t or "none", h or "none")
        by_rid = {}
        for r in results:
            if r["tuning"] == (t or "none") and r["handling"] == (h or "none"):
                by_rid.setdefault(r["rid"], []).append(r)
        caps = [collapse_capacity(pts) for pts in by_rid.values()]
        frag = lognormal_fragility(caps, scales)
        # median IDA drift at each scale (over records, capping collapsed at the drift limit)
        ida = {}
        for s in scales:
            ds = []
            for pts in by_rid.values():
                ps = [p for p in pts if abs(p["scale"] - s) < 1e-9]
                if ps:
                    ds.append(min(ps[0]["d_max"], COLLAPSE_DRIFT) if not ps[0]["collapse"] else COLLAPSE_DRIFT)
            ida[s] = float(np.median(ds)) if ds else float("nan")
        cells[ck] = {"fragility": frag, "median_ida_drift": ida}

    # ---- the allocation-vs-fidelity contrast at the collapse limit ----
    def med(ck):
        return cells[ck]["fragility"]["median_capacity"]
    contrast = {
        "median_collapse_capacity": {ck: cells[ck]["fragility"]["median_capacity"] for ck in cells},
        "allocation_gain_elastic_inqp_over_clip": (med("elastic_inqp") - med("elastic_clip")),
        "allocation_gain_softened_inqp_over_clip": (med("softened_inqp") - med("softened_clip")),
        "fidelity_gain_softened_over_elastic_clip": (med("softened_clip") - med("elastic_clip")),
        "fidelity_gain_softened_over_elastic_inqp": (med("softened_inqp") - med("elastic_inqp")),
    }
    out = {"title": "EESD IDA + collapse fragility (validated Ohtori 3-story)",
           "n_records": len(rids), "scales": scales, "collapse_drift": COLLAPSE_DRIFT,
           "cells": cells, "contrast": contrast, "raw_n": len(results)}
    tag = "_smoke" if smoke else ""
    (RESULTS / f"eesd_ida_fragility{tag}.json").write_text(json.dumps(out, indent=2))
    print("\n=== COLLAPSE FRAGILITY (median collapse-capacity scale, higher = safer) ===", flush=True)
    for ck in ["none_none", "elastic_clip", "softened_clip", "elastic_inqp", "softened_inqp"]:
        if ck in cells:
            f = cells[ck]["fragility"]
            print(f"  {ck:16s}  median={f['median_capacity']:.2f}  beta={f['beta']:.3f}  "
                  f"collapsed {f['n_collapsed']}/{f['n_records']}")
    print("\n  ALLOCATION gain (in-QP - clip):  elastic %+.2f | softened %+.2f" % (
        contrast["allocation_gain_elastic_inqp_over_clip"],
        contrast["allocation_gain_softened_inqp_over_clip"]))
    print("  FIDELITY  gain (softened - elastic): clip %+.2f | in-QP %+.2f" % (
        contrast["fidelity_gain_softened_over_elastic_clip"],
        contrast["fidelity_gain_softened_over_elastic_inqp"]))
    print(f"\nwrote eesd_ida_fragility{tag}.json")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        _i = sys.argv.index("--worker")
        _rid = sys.argv[_i + 1]
        _scale = float(sys.argv[_i + 2])
        _tuning = None if sys.argv[_i + 3] == "none" else sys.argv[_i + 3]
        _handling = None if sys.argv[_i + 4] == "none" else sys.argv[_i + 4]
        print(WORKER_SENTINEL + json.dumps(run_one((_rid, _scale, _tuning, _handling))), flush=True)
    else:
        main()



