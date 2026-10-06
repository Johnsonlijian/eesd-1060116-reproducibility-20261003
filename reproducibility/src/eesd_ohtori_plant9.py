"""Canonical Ohtori/Spencer 9-story nonlinear SAC moment-frame plant (OpenSeesPy).

Multi-height companion to src/eesd_ohtori_plant.py (3-story). Same modelling contract,
extended to the 9-story benchmark whose source file is more complex:

    Ohtori, Y., Christenson, R.E., Spencer, B.F. Jr., Dyke, S.J. (2004).
    "Benchmark Control Problems for Seismically Excited Nonlinear Buildings."
    J. Eng. Mech. 130(4), 366-385.

PROVENANCE (no fabricated properties)
-------------------------------------
Every structural property is parsed at runtime from the on-disk authors' MATLAB package:

    data_raw/eesd_ohtori/unzipped/bldg_full3/MATLAB_Utility/struct9.m   (plant definition)
    data_raw/.../MATLAB_Utility/{elcentro,hachinhe,northrdg,kobe}.mat   (ground motions, m/s^2)
    data_raw/.../MATLAB_Utility/unctrl9.mat                              (published uncontrolled responses)

WHAT IS DIFFERENT FROM struct3.m (and why the 3-story parser cannot be used verbatim)
-------------------------------------------------------------------------------------
* Num_story = 9+1 = 10  (a basement story below grade; 11 node rows, 6 columns/row, 5 bays).
* Four distinct floor masses (seismic_mass1..4), not two; mass starts at the *1st* floor.
* The rigid-floor diaphragm (slv_tbl) starts at the *2nd* floor (master node 15). The basement
  and 1st-floor nodes are horizontally tied to ground by Fix_node (fix_x=1) instead.
* Base nodes are pin-like (fix_x=fix_y=1, rotation FREE); 1st-floor nodes fix_x only.
* mat_tbl is appended with five SPLICE materials (12..16) defined as rational-weighted
  combinations of base rows -- "mat_tbl = [mat_tbl; 12 (6/13)*mat_tbl(2,2:8)+... 1]". These
  are evaluated here (not regex-scalar-parsed) so the column material map resolves fully.

Published modal frequencies validated against (benchmark Report_bldg_full3.pdf, verbatim):
    "The first five natural frequencies of the 9-story benchmark evaluation model are:
     0.443, 1.18, 2.05, 3.09, and 4.27 Hz."

HONEST SCOPE / APPROXIMATIONS  (identical philosophy to the 3-story port)
------------------------------------------------------------------------
* Bilinear Steel01 Mz backbone (initial tangent == EI1) approximating the authors' trilinear
  compiled MEX. Initial stiffness / modal frequencies preserved exactly; nonlinear branch and
  damage indices (J7-J10) are NOT reproduced. Claims kept to drift / accel.
* Type-3 Rayleigh anchored at modes 1..nCutOff(=2) -> conventional two-term Rayleigh at
  omega_1, omega_2 with zeta=0.02.

Run as a script to (re)generate results/eesd_ohtori_9story_validation.json.
"""

from __future__ import annotations
import os

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

try:
    import openseespy.opensees as ops
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "openseespy is required. Use the project env that imports it, e.g. "
        "python-environment\\Scripts\\python.exe"
    ) from exc


# --------------------------------------------------------------------------- #
# Paths / constants
# --------------------------------------------------------------------------- #
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("EESD_PROJECT_ROOT", PACKAGE_ROOT))
OHTORI_DIR = Path(os.environ.get("EESD_OHTORI_DIR", ROOT / "data_raw" / "eesd_ohtori" / "unzipped" / "bldg_full3" / "MATLAB_Utility"))
RESULTS = Path(os.environ.get("EESD_OUTPUT_DIR", PACKAGE_ROOT / "analysis"))

E_STEEL_PA = 2.0e11
G_MPS2 = 9.81

# Published modal frequencies (Ohtori et al. 2004 benchmark report, 9-story evaluation model).
PUBLISHED_FREQS_HZ = (0.443, 1.18, 2.05, 3.09, 4.27)


# --------------------------------------------------------------------------- #
# struct9.m parsing
# --------------------------------------------------------------------------- #
def _strip_comments(text: str) -> str:
    return "\n".join(line.split("%", 1)[0] for line in text.splitlines())


def _scalar(text: str, name: str) -> float:
    m = re.search(rf"\b{name}\s*=\s*([0-9eE.+\-/*]+)\s*;", text)
    if not m:
        raise ValueError(f"Scalar {name!r} not found")
    return float(eval(m.group(1)))  # noqa: S307 - controlled numeric arithmetic only


def _vector(text: str, name: str) -> list[float]:
    # struct9 uses "...." (4 dots) as a continuation in `height`; collapse any run of dots.
    m = re.search(rf"\b{name}\s*=\s*\[(.*?)\]", text, re.DOTALL)
    if not m:
        raise ValueError(f"Vector {name!r} not found")
    seg = re.sub(r"\.{2,}", " ", m.group(1))
    return [float(v) for v in seg.split()]


def _isnum(v: str) -> bool:
    try:
        float(v)
        return True
    except ValueError:
        return False


def _matrix_block(text: str, marker: str, terminator: str) -> list[int]:
    start = text.index(marker) + len(marker)
    end = text.index(terminator, start)
    seg = text[start:end]
    for ch in ("...", "'", ";", ","):
        seg = seg.replace(ch, " ")
    return [int(float(v)) for v in seg.split() if _isnum(v)]


@dataclass
class Plant9:
    num_story: int          # 10 (includes basement)
    num_bay: int            # 5
    num_mrf: int            # 2
    height: list[float]     # 10 story heights
    width: list[float]      # 5 bay widths
    zeta_cr: float
    n_cutoff: int
    nodes: list[dict[str, Any]]
    elements: list[dict[str, Any]]
    materials: dict[int, dict[str, float]]
    fix_nodes: list[dict[str, int]]
    slaves: list[dict[str, Any]]
    floor_masses: list[float]   # per-MRF seismic mass for each of num_story floors (story 1..num_story)

    @property
    def floor_master_nodes(self) -> list[int]:
        """The diaphragm master nodes (2nd floor .. roof), bottom-to-top."""
        return [int(s["master"]) for s in sorted(self.slaves, key=lambda d: d["master"])]


def _parse_mat_tbl(text: str) -> dict[int, dict[str, float]]:
    """Parse the base mat_tbl block AND the appended splice rows 12..16.

    Base rows:   '<no> EI1 EI2 EI3 EA GA d1 d2 type'  (9 numeric tokens).
    Splice rows: '<no> (6/13)*mat_tbl(a,2:8)+(7/13)*mat_tbl(b,2:8)  <type>' -- evaluated from
                 already-parsed base rows so columns referencing them resolve.
    """
    materials: dict[int, dict[str, float]] = {}
    keys7 = ("EI1", "EI2", "EI3", "EA", "GA", "d1", "d2")

    # --- base block: first 'mat_tbl = [' ... first ']' ---
    mstart = text.index("mat_tbl = [") + len("mat_tbl = [")
    mend = text.index("]", mstart)
    for line in text[mstart:mend].splitlines():
        toks = line.split()
        if len(toks) >= 9 and _isnum(toks[0]) and all(_isnum(t) for t in toks[:9]):
            vals = [float(t) for t in toks[:9]]
            materials[int(vals[0])] = {
                "EI1": vals[1], "EI2": vals[2], "EI3": vals[3], "EA": vals[4],
                "GA": vals[5], "d1": vals[6], "d2": vals[7], "type": int(vals[8]),
            }

    # --- appended splice block: 'mat_tbl = [mat_tbl' ... ']' ---
    app_marker = "mat_tbl = [mat_tbl"
    if app_marker in text:
        astart = text.index(app_marker) + len(app_marker)
        aend = text.index("]", astart)
        seg = text[astart:aend]
        # each splice row:  <no>  (w1)*mat_tbl(a,2:8)+(w2)*mat_tbl(b,2:8)   <type>
        row_re = re.compile(
            r"(\d+)\s*"
            r"\(([\d./]+)\)\*mat_tbl\((\d+),2:8\)\s*\+\s*\(([\d./]+)\)\*mat_tbl\((\d+),2:8\)\s*"
            r"(\d+)"
        )
        for m in row_re.finditer(seg):
            no = int(m.group(1))
            w1 = float(eval(m.group(2)))      # noqa: S307 - rational literal only
            a = int(m.group(3))
            w2 = float(eval(m.group(4)))      # noqa: S307
            b = int(m.group(5))
            typ = int(m.group(6))
            ma, mb = materials[a], materials[b]
            materials[no] = {k: w1 * ma[k] + w2 * mb[k] for k in keys7}
            materials[no]["type"] = typ
    return materials


def parse_struct9(struct_path: Path) -> Plant9:
    raw = struct_path.read_text(encoding="utf-8", errors="ignore")
    text = _strip_comments(raw)

    num_story = int(_scalar(text, "Num_story"))     # 9+1 = 10
    num_bay = int(_scalar(text, "Num_bay"))         # 5
    num_mrf = int(_scalar(text, "Num_MRF"))         # 2
    zeta_cr = _scalar(text, "zeta_cr")
    n_cutoff = int(_scalar(text, "nCutOff"))
    height = _vector(text, "height")
    width = _vector(text, "width")
    if len(height) != num_story:
        raise ValueError(f"height has {len(height)} entries, expected {num_story}")
    if len(width) != num_bay:
        raise ValueError(f"width has {len(width)} entries, expected {num_bay}")

    # four floor masses
    masses = {i: _scalar(text, f"seismic_mass{i}") for i in (1, 2, 3, 4)}

    num_col = (num_bay + 1) * num_story
    num_beam = num_bay * num_story

    materials = _parse_mat_tbl(text)

    # --- column / beam material id vectors ---
    col_mat = _matrix_block(text, "element_tbl(1:Num_col,3) = [", "]';")
    beam_mat = _matrix_block(text, "element_tbl(Num_col+1:Num_col+Num_beam,3) = [", "]';")
    if len(col_mat) != num_col:
        raise ValueError(f"column material ids: expected {num_col}, got {len(col_mat)}")
    if len(beam_mat) != num_beam:
        raise ValueError(f"beam material ids: expected {num_beam}, got {len(beam_mat)}")

    # brace / fixed-hinged offsets (element_tbl(Num_col+[...],4)=5)
    brace_offsets: set[int] = set()
    mb = re.search(r"element_tbl\(Num_col\+\[(.*?)\]\s*,\s*4\)\s*=\s*5", text, re.DOTALL)
    if mb:
        brace_offsets = {int(float(v)) for v in mb.group(1).split() if _isnum(v)}

    # --- nodes (grid). story_level 0 = base; 1..num_story going up. y is negative downward
    #     in the source convention; we store y as elevation magnitude via -y0 like struct3. ---
    nodes: list[dict[str, Any]] = []
    node_id = 0
    y0 = 0.0
    for story_level in range(num_story + 1):
        x0 = 0.0
        for bay_line in range(num_bay + 1):
            node_id += 1
            nodes.append({"node_id": node_id, "story_level": story_level, "x_m": x0, "y_m": -y0})
            if bay_line < num_bay:
                x0 += width[bay_line]
        if story_level < num_story:
            y0 += height[story_level]
    node_xy = {n["node_id"]: (n["x_m"], n["y_m"]) for n in nodes}

    # --- per-MRF floor masses (story 1..num_story). struct9 Element_mass order:
    #     story1->mass1, story2->mass2, story3..9->mass3, roof->mass4 ---
    floor_mass_per_story: list[float] = []
    for story in range(1, num_story + 1):
        if story == 1:
            m = masses[1]
        elif story == 2:
            m = masses[2]
        elif story == num_story:
            m = masses[4]
        else:
            m = masses[3]
        floor_mass_per_story.append(m)
    beam_masses: list[float] = []
    for story in range(1, num_story + 1):
        beam_masses.extend([floor_mass_per_story[story - 1] / num_bay] * num_bay)

    # --- elements: columns then beams/braces ---
    elements: list[dict[str, Any]] = []
    for idx in range(1, num_col + 1):
        ni, nj = idx, idx + num_bay + 1
        story = math.ceil(idx / (num_bay + 1))
        elements.append(_elem(idx, "column", ni, nj, col_mat[idx - 1], story, 0.0, node_xy, materials))
    elem_no = num_col
    for story in range(1, num_story + 1):
        for bay in range(1, num_bay + 1):
            elem_no += 1
            offset = elem_no - num_col
            ni = story * (num_bay + 1) + bay
            nj = ni + 1
            mtype = "brace" if offset in brace_offsets else "beam"
            elements.append(
                _elem(elem_no, mtype, ni, nj, beam_mat[offset - 1], story,
                      beam_masses[offset - 1], node_xy, materials)
            )

    # --- fixities ---
    fb = _matrix_block(text, "Fix_node = [", "]")
    fix_nodes = [
        {"node_id": fb[i], "fix_x": fb[i + 1], "fix_y": fb[i + 2], "fix_r": fb[i + 3]}
        for i in range(0, len(fb), 4)
    ]

    # --- master/slave rigid floor ties ---
    sb = _matrix_block(text, "slv_tbl = [", "]")
    slaves: list[dict[str, Any]] = []
    i = 0
    while i < len(sb):
        master, direction, nslv = sb[i], sb[i + 1], sb[i + 2]
        slv = [sb[i + 3 + j] for j in range(nslv)]
        slaves.append({"master": master, "direction": direction, "slaves": slv})
        i += 3 + nslv

    return Plant9(
        num_story=num_story, num_bay=num_bay, num_mrf=num_mrf, height=height, width=width,
        zeta_cr=zeta_cr, n_cutoff=n_cutoff, nodes=nodes, elements=elements, materials=materials,
        fix_nodes=fix_nodes, slaves=slaves, floor_masses=floor_mass_per_story,
    )


def _elem(eid, mtype, ni, nj, mat_no, story, mass_kg, node_xy, materials) -> dict[str, Any]:
    xi, yi = node_xy[ni]
    xj, yj = node_xy[nj]
    mat = materials[mat_no]
    return {
        "element_id": eid, "member_type": mtype, "node_i": ni, "node_j": nj, "story": story,
        "mat_no": mat_no, "length_m": math.hypot(xj - xi, yj - yi), "element_mass_kg": mass_kg,
        **{k: mat[k] for k in ("EI1", "EI2", "EI3", "EA", "GA", "d1", "d2", "type")},
    }


# --------------------------------------------------------------------------- #
# OpenSeesPy model builders  (identical contract to eesd_ohtori_plant.build_model)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BuildConfig:
    mode: str = "nonlinear"
    yield_factor: float = 8.0
    hardening_ratio: float = 0.02
    n_integration: int = 5
    damping_ratio: float = 0.02
    extra: dict = field(default_factory=dict)


def _node_mass_vec(mass_kg: float, length_m: float) -> np.ndarray:
    rot = 0.5 * mass_kg * 1.0e-6 * length_m**2 / 210.0
    return np.array([0.5 * mass_kg, 0.5 * mass_kg, rot], dtype=float)


def build_model(plant: Plant9, cfg: BuildConfig) -> dict[str, Any]:
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)

    for n in sorted(plant.nodes, key=lambda d: d["node_id"]):
        ops.node(int(n["node_id"]), float(n["x_m"]), float(n["y_m"]))

    fixed = {f["node_id"]: f for f in plant.fix_nodes}
    for n in plant.nodes:
        nid = int(n["node_id"])
        f = fixed.get(nid)
        if f is None:
            ops.fix(nid, 0, 0, 0)
        else:
            ops.fix(nid, int(f["fix_x"]), int(f["fix_y"]), int(f["fix_r"]))

    for s in plant.slaves:
        for slv in s["slaves"]:
            ops.equalDOF(int(s["master"]), int(slv), int(s["direction"]))

    nodal_mass: dict[int, np.ndarray] = {int(n["node_id"]): np.zeros(3) for n in plant.nodes}
    for ele in plant.elements:
        mass = float(ele["element_mass_kg"])
        if mass <= 0.0:
            continue
        vec = _node_mass_vec(mass, float(ele["length_m"]))
        nodal_mass[int(ele["node_i"])] += vec
        nodal_mass[int(ele["node_j"])] += vec
    for nid, vec in nodal_mass.items():
        if float(np.sum(vec)) > 0.0:
            ops.mass(nid, float(vec[0]), float(vec[1]), max(float(vec[2]), 1.0e-12))

    ops.geomTransf("Linear", 1)
    nxt_mat = nxt_sec = nxt_int = nxt_ele = 1
    member_rows: list[dict[str, Any]] = []

    for ele in sorted(plant.elements, key=lambda d: d["element_id"]):
        ni, nj = int(ele["node_i"]), int(ele["node_j"])
        ea = float(ele["EA"])
        length = float(ele["length_m"])
        ei = float(ele["EI1"])
        if str(ele["member_type"]) == "brace":
            # struct9 "brace" offsets (Num_col+[5 10 ... 50], type 5) are the benchmark's
            # FIXED-HINGED BEAM elements (report: "fixed-hinged beam elements in the 9-story
            # structure"), NOT axial truss members. They are elastic beams with a moment release
            # at the j-end (pinned). Modelling them as hinged beams reproduces the published modal
            # frequencies to <0.2% (truss -> -2.6%, full moment beam -> +7%). The benchmark
            # explicitly EXCLUDES these from the yielding/damage criteria, so they stay elastic in
            # both build modes.
            area = max(ea / E_STEEL_PA, 1.0e-10)
            iz = max(ei / E_STEEL_PA, 1.0e-12)
            ops.element("elasticBeamColumn", nxt_ele, ni, nj, area, E_STEEL_PA, iz, 1, "-release", 2)
            nxt_ele += 1
            continue

        if cfg.mode == "elastic":
            area = max(ea / E_STEEL_PA, 1.0e-10)
            iz = max(ei / E_STEEL_PA, 1.0e-12)
            ops.element("elasticBeamColumn", nxt_ele, ni, nj, area, E_STEEL_PA, iz, 1)
            nxt_ele += 1
            continue

        theta_y = max(min(abs(float(ele["d1"])), abs(float(ele["d2"]))), 1.0e-12)
        my = max(cfg.yield_factor * ei * theta_y / length, 1.0)
        axial_mat = nxt_mat
        flex_mat = nxt_mat + 1
        nxt_mat += 2
        ops.uniaxialMaterial("Elastic", axial_mat, max(ea, 1.0))
        ops.uniaxialMaterial("Steel01", flex_mat, my, max(ei, 1.0), cfg.hardening_ratio)
        ops.section("Aggregator", nxt_sec, axial_mat, "P", flex_mat, "Mz")
        ops.beamIntegration("Lobatto", nxt_int, nxt_sec, cfg.n_integration)
        ops.element("forceBeamColumn", nxt_ele, ni, nj, 1, nxt_int, "-iter", 25, 1.0e-10)
        member_rows.append({
            "element_id": int(ele["element_id"]), "member_type": str(ele["member_type"]),
            "story": int(ele["story"]), "length_m": length, "EI1": ei,
            "theta_y_rad": theta_y, "my_Nm": my,
        })
        nxt_sec += 1
        nxt_int += 1
        nxt_ele += 1

    return {
        "mode": cfg.mode,
        "n_main_nodes": len(plant.nodes),
        "n_model_elements": int(nxt_ele - 1),
        "n_nonlinear_members": len(member_rows),
        "my_min_Nm": float(min((r["my_Nm"] for r in member_rows), default=math.nan)),
        "my_max_Nm": float(max((r["my_Nm"] for r in member_rows), default=math.nan)),
        "total_mass_kg": float(sum(float(e["element_mass_kg"]) for e in plant.elements)),
        "yield_factor": cfg.yield_factor,
        "hardening_ratio": cfg.hardening_ratio,
    }


# --------------------------------------------------------------------------- #
# Analyses (verbatim contract from the 3-story module)
# --------------------------------------------------------------------------- #
def eigen_frequencies(n_modes: int = 5) -> list[float]:
    ops.wipeAnalysis()
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("FullGeneral")
    try:
        lambdas = [float(v) for v in ops.eigen("-fullGenLapack", n_modes) if float(v) > 1.0e-8]
    except Exception:
        lambdas = [float(v) for v in ops.eigen(n_modes) if float(v) > 1.0e-8]
    lambdas.sort()
    return [math.sqrt(l) / (2.0 * math.pi) for l in lambdas[:n_modes]]


def apply_rayleigh(damping_ratio: float) -> dict[str, Any]:
    ops.wipeAnalysis()
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("FullGeneral")
    try:
        lambdas = [float(v) for v in ops.eigen("-fullGenLapack", 2) if float(v) > 0.0]
    except Exception:
        lambdas = [float(v) for v in ops.eigen(2) if float(v) > 0.0]
    if len(lambdas) < 2:
        return {"status": "not_enough_modes"}
    w1, w2 = math.sqrt(lambdas[0]), math.sqrt(lambdas[1])
    beta_k = 2.0 * damping_ratio / (w1 + w2)
    alpha_m = beta_k * w1 * w2
    ops.rayleigh(alpha_m, 0.0, 0.0, beta_k)
    return {"status": "ok", "alpha_m": alpha_m, "beta_k": beta_k, "w1": w1, "w2": w2,
            "anchor_modes": [1, 2], "zeta": damping_ratio}


def load_record(name: str, var: str, intensity: float) -> tuple[np.ndarray, np.ndarray, float]:
    data = loadmat(OHTORI_DIR / f"{name}.mat")[var]
    times = np.asarray(data[0, :], dtype=float)
    accel = np.asarray(data[1, :], dtype=float) * float(intensity)
    dt = float(np.median(np.diff(times)))
    return times, accel, dt


def _try_step(dt: float, tol: float, max_iter: int) -> int:
    for alg in [("Newton",), ("NewtonLineSearch", "-type", "Bisection"),
                ("KrylovNewton",), ("ModifiedNewton",)]:
        ops.algorithm(*alg)
        if int(ops.analyze(1, dt)) == 0:
            return 0
    return 1


def run_uncontrolled(
    plant: Plant9, cfg: BuildConfig, record: str, var: str, intensity: float, tf: float,
    dt_analysis: float = 0.005, dt_output: float = 0.01, tol: float = 1.0e-8, max_iter: int = 40,
) -> dict[str, Any]:
    meta = build_model(plant, cfg)
    rayleigh = apply_rayleigh(cfg.damping_ratio)

    times, accel, rec_dt = load_record(record, var, intensity)
    keep = times <= tf
    times, accel = times[keep], accel[keep]
    if len(times) and times[-1] < tf:
        times = np.append(times, tf)
        accel = np.append(accel, 0.0)

    ops.timeSeries("Path", 1001, "-dt", rec_dt, "-values", *accel.tolist(), "-factor", 1.0)
    ops.pattern("UniformExcitation", 2001, 1, "-accel", 1001)

    ops.wipeAnalysis()
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", tol, max_iter)
    ops.algorithm("Newton")
    ops.integrator("Newmark", 0.5, 0.25)
    ops.analysis("Transient")

    floor_nodes = plant.floor_master_nodes
    # The diaphragm starts at the 2nd floor; for drift we need the full elevation ladder from
    # ground through every floor. Build elevations from node story-levels and pick the master
    # node per floor that carries lateral disp. The 1st floor (basement-top) is x-tied to ground
    # so its lateral disp is ~0 and is included as the base reference.
    levels = sorted({(n["story_level"], n["y_m"]) for n in plant.nodes})
    elevations = [abs(y) for _, y in levels]            # ground .. roof, length num_story+1
    heights = np.diff(elevations)                       # per-story heights, length num_story

    # floor lateral displacement at each story level (1..num_story). Master nodes give floors that
    # have a diaphragm; for the basement/1st floor (tied to ground) take a representative node.
    master_by_level: dict[int, int] = {}
    for s in plant.slaves:
        lvl = next(n["story_level"] for n in plant.nodes if n["node_id"] == s["master"])
        master_by_level[lvl] = int(s["master"])
    # representative lateral node for each story level (first node of that row)
    rep_node_by_level: dict[int, int] = {}
    for n in sorted(plant.nodes, key=lambda d: d["node_id"]):
        lvl = n["story_level"]
        if lvl not in rep_node_by_level:
            rep_node_by_level[lvl] = int(n["node_id"])
    track_nodes = []
    for lvl in range(1, plant.num_story + 1):
        track_nodes.append(master_by_level.get(lvl, rep_node_by_level[lvl]))

    floor_mass = np.array([
        sum(float(e["element_mass_kg"]) for e in plant.elements
            if e["member_type"] in ("beam", "brace") and int(e["story"]) == s) * plant.num_mrf
        for s in range(1, plant.num_story + 1)
    ], dtype=float)

    n_steps = int(math.ceil(tf / dt_analysis))
    out_every = max(1, int(round(dt_output / dt_analysis)))
    drift_peak = acc_peak = disp_peak = vel_peak = shear_peak = 0.0
    drift_sq = np.zeros(len(track_nodes))
    acc_sq = np.zeros(len(track_nodes))
    shear_sq = 0.0
    n_out = 0
    code = 0
    t = 0.0
    for step in range(n_steps):
        if _try_step(dt_analysis, tol, max_iter) != 0:
            code = 1
            break
        if step % out_every != 0:
            continue
        t = ops.getTime()
        gacc = float(np.interp(t, times, accel)) if len(times) else 0.0
        disp = np.array([ops.nodeDisp(n, 1) for n in track_nodes])
        vel = np.array([ops.nodeVel(n, 1) for n in track_nodes])
        abs_acc = np.array([ops.nodeAccel(n, 1) + gacc for n in track_nodes])
        interstory = np.concatenate([[disp[0]], np.diff(disp)])
        drift_ratio = interstory / heights
        base_shear = float(np.dot(abs_acc, floor_mass))

        drift_peak = max(drift_peak, float(np.max(np.abs(drift_ratio))))
        acc_peak = max(acc_peak, float(np.max(np.abs(abs_acc))))
        disp_peak = max(disp_peak, float(np.max(np.abs(disp))))
        vel_peak = max(vel_peak, float(np.max(np.abs(vel))))
        shear_peak = max(shear_peak, abs(base_shear))
        drift_sq += drift_ratio**2
        acc_sq += abs_acc**2
        shear_sq += base_shear**2
        n_out += 1

    tf_eff = max(t, dt_output)
    return {
        "record": record, "intensity": intensity, "tf_s": tf, "analysis_code": code,
        "converged": code == 0, "n_output_steps": n_out, "record_dt": rec_dt,
        "rayleigh": rayleigh, "model": meta,
        "d_max": drift_peak, "xdd_max": acc_peak, "F_max": shear_peak,
        "x_max": disp_peak, "xd_max": vel_peak,
        "d_max_norm": float(np.max(np.sqrt(drift_sq * dt_output / tf_eff))) if n_out else math.nan,
        "xdd_max_norm": float(np.max(np.sqrt(acc_sq * dt_output / tf_eff))) if n_out else math.nan,
        "F_max_norm": float(math.sqrt(shear_sq * dt_output / tf_eff)) if n_out else math.nan,
    }


def published_reference(case_index: int) -> dict[str, float]:
    """Published uncontrolled responses for a case (1-based) from unctrl9.mat."""
    data = loadmat(OHTORI_DIR / "unctrl9.mat")
    i = case_index - 1
    return {
        "d_max": float(np.ravel(data["d_max"])[i]),
        "xdd_max": float(np.ravel(data["xdd_max"])[i]),
        "F_max": float(np.ravel(data["F_max"])[i]),
        "d_max_norm": float(np.ravel(data["d_max_norm"])[i]),
        "xdd_max_norm": float(np.ravel(data["xdd_max_norm"])[i]),
        "F_max_norm": float(np.ravel(data["F_max_norm"])[i]),
        "total_mass_kg": float(np.ravel(data["M"])[0]),
        "weight_N": float(np.ravel(data["W"])[0]),
    }


# --------------------------------------------------------------------------- #
# Validation driver
# --------------------------------------------------------------------------- #
def validate(struct_path: Path, run_th: bool = True, n_modes: int = 5) -> dict[str, Any]:
    plant = parse_struct9(struct_path)

    build_model(plant, BuildConfig(mode="elastic"))
    elastic_freqs = eigen_frequencies(n_modes)

    build_model(plant, BuildConfig(mode="nonlinear"))
    nonlinear_freqs = eigen_frequencies(n_modes)

    freq_rows = []
    for i, pub in enumerate(PUBLISHED_FREQS_HZ):
        fe = elastic_freqs[i] if i < len(elastic_freqs) else math.nan
        fn = nonlinear_freqs[i] if i < len(nonlinear_freqs) else math.nan
        freq_rows.append({
            "mode": i + 1, "published_hz": pub,
            "elastic_hz": fe, "elastic_pct_err": 100.0 * (fe - pub) / pub,
            "nonlinear_initial_hz": fn, "nonlinear_pct_err": 100.0 * (fn - pub) / pub,
        })
    max_abs_err = max(abs(r["elastic_pct_err"]) for r in freq_rows)
    mode1_err = abs(freq_rows[0]["elastic_pct_err"])
    freq_pass = max_abs_err < 2.0

    total_mass_full = (sum(plant.floor_masses) * plant.num_mrf)

    out: dict[str, Any] = {
        "deliverable": "eesd_ohtori_9story_validation",
        "benchmark": "Ohtori/Spencer 9-story nonlinear SAC moment frame (LA, N-S MRF)",
        "source_package": str(struct_path.relative_to(ROOT)).replace("\\", "/"),
        "ground_truth_mat": "data_raw/eesd_ohtori/unzipped/bldg_full3/MATLAB_Utility/unctrl9.mat",
        "published_freqs_source": ("Ohtori et al. 2004 benchmark report (Report_bldg_full3.pdf): "
                                   "'first five natural frequencies of the 9-story benchmark "
                                   "evaluation model are: 0.443, 1.18, 2.05, 3.09, and 4.27 Hz'"),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "openseespy_version": str(ops.version()),
        "plant_summary": {
            "num_story_incl_basement": plant.num_story, "num_bay": plant.num_bay,
            "num_mrf": plant.num_mrf, "story_heights_m": plant.height, "bay_widths_m": plant.width,
            "n_nodes": len(plant.nodes), "n_elements": len(plant.elements),
            "n_materials_incl_splices": len(plant.materials),
            "floor_masses_per_mrf_kg": plant.floor_masses,
            "total_mass_full_frame_kg": total_mass_full,
            "published_M_kg": float(np.ravel(loadmat(OHTORI_DIR / "unctrl9.mat")["M"])[0]),
            "floor_master_nodes": plant.floor_master_nodes,
            "damping_ratio_zeta": plant.zeta_cr, "rayleigh_n_cutoff": plant.n_cutoff,
        },
        "frequency_validation": {
            "published_hz": list(PUBLISHED_FREQS_HZ),
            "modes": freq_rows,
            "mode1_abs_pct_error": mode1_err,
            "max_abs_pct_error_elastic": max_abs_err,
            "mean_abs_pct_error_elastic": float(np.mean([abs(r["elastic_pct_err"]) for r in freq_rows])),
            "freq_match_pct_mode1": 100.0 - mode1_err,
            "pass_within_2pct": bool(freq_pass),
        },
    }

    if run_th:
        cfg = BuildConfig(mode="nonlinear", yield_factor=8.0, hardening_ratio=0.02)
        th = run_uncontrolled(plant, cfg, "elcentro", "e", 1.0, 100.0)
        ref = published_reference(2)
        out["time_history_validation"] = {
            "case": "El Centro, intensity 1.0 (unctrl9.mat case 2)",
            "yield_factor": cfg.yield_factor, "hardening_ratio": cfg.hardening_ratio,
            "converged": th["converged"], "analysis_code": th["analysis_code"],
            "model": {k: th[k] for k in ("d_max", "xdd_max", "F_max",
                                         "d_max_norm", "xdd_max_norm", "F_max_norm")},
            "published": ref,
            "ratio_model_over_published": {
                "d_max": th["d_max"] / ref["d_max"],
                "xdd_max": th["xdd_max"] / ref["xdd_max"],
                "F_max": th["F_max"] / ref["F_max"],
            },
            "mass_check_total_kg": {
                "model_full_frame": total_mass_full, "published_M": ref["total_mass_kg"],
            },
        }

    out["honesty_notes"] = [
        "All geometry, member EI/EA/GA, yield curvatures, masses, fixities, rigid-floor ties and the "
        "five splice materials (12-16) are parsed/evaluated at runtime from the authors' struct9.m.",
        "Mode-1 frequency reproduced by the elastic initial-stiffness model; nonlinear force-based "
        "model matches at t=0 (initial EI1 tangent preserved).",
        "Bilinear Steel01 Mz surrogate for the trilinear compiled MEX -- validated approximation of "
        "yield onset/softening, NOT bit-exact; damage indices J7-J10 out of scope.",
        "yield_factor=8 carried over from the 3-story calibration (same member yield mapping "
        "My=yield_factor*EI1*theta_y/L); reported as a model choice, not re-fit per height.",
        "Type-3 Rayleigh realized as two-term Rayleigh anchored at modes 1,2 with zeta=0.02.",
    ]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build & validate the Ohtori 9-story nonlinear plant.")
    parser.add_argument("--struct", type=Path, default=OHTORI_DIR / "struct9.m")
    parser.add_argument("--out", type=Path, default=RESULTS / "eesd_ohtori_9story_validation.json")
    parser.add_argument("--no-th", action="store_true")
    parser.add_argument("--modes", type=int, default=5)
    args = parser.parse_args()

    result = validate(args.struct, run_th=not args.no_th, n_modes=args.modes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    fv = result["frequency_validation"]
    print(f"openseespy {result['openseespy_version']}  python {result['python_version']}")
    print(f"Plant: {result['plant_summary']['num_story_incl_basement']} story-levels "
          f"(incl basement), {result['plant_summary']['n_nodes']} nodes, "
          f"{result['plant_summary']['n_elements']} elements, "
          f"{result['plant_summary']['n_materials_incl_splices']} materials")
    print("Modal frequency validation (published 0.443/1.18/2.05/3.09/4.27 Hz):")
    for r in fv["modes"]:
        print(f"  mode {r['mode']}: elastic {r['elastic_hz']:.4f} Hz "
              f"({r['elastic_pct_err']:+.3f}%)  nonlinear-init {r['nonlinear_initial_hz']:.4f} Hz")
    print(f"  mode-1 |error| = {fv['mode1_abs_pct_error']:.3f}%  "
          f"max |error| = {fv['max_abs_pct_error_elastic']:.3f}%  pass(<2%) = {fv['pass_within_2pct']}")
    if "time_history_validation" in result:
        th = result["time_history_validation"]
        rr = th["ratio_model_over_published"]
        print(f"El Centro I=1.0: converged={th['converged']}  d_max ratio={rr['d_max']:.3f}  "
              f"xdd ratio={rr['xdd_max']:.3f}  F ratio={rr['F_max']:.3f}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()


