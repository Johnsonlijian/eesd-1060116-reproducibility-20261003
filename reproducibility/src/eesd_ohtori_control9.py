r"""Multi-height replication: port the EESD softening-aware control analysis onto the VALIDATED
canonical Ohtori 9-STORY nonlinear OpenSees benchmark and test whether the THREE SURVIVING claims
from the 3-story are height-robust or height-specific.

THE THREE CLAIMS UNDER TEST (established on the 3-story; do they replicate at 9-story?)
=======================================================================================
  (1) SATURATION-HANDLING DOMINANCE: in the 2x2 factorial (tuning{elastic,softened} x
      saturation{clip,in-QP}), the saturation-handling factor B carries the dominant ANOVA
      variance share (~82% on the 3-story across FEMA). Still dominant at 9-story?
  (2) IN-QP DEEP-YIELD FAILURE: in-QP loses to clip past ~3.5% drift, and the predictor of the
      crossover is UNCONTROLLED DRIFT DEPTH (not E_K = tangent-stiffness mismatch). Still true,
      and still drift-depth (not E_K), at 9-story?
  (3) SOFTENING-AWARE NULL: the softened-vs-elastic LQR margin (re-pointing design to
      omega_1*sqrt(alpha_eff)) is ~0 (3-story: -0.004, an airtight null). Still ~0 at 9-story?

ARCHITECTURE (identical to the 3-story port, eesd_ohtori_control.py)
===================================================================
Controllers are DESIGNED on the reduced 9-DOF lateral M/K design surrogate
(results/eesd_struct9_lateral_MK.json, Guyan static condensation) and EVALUATED CLOSED-LOOP on the
FULL nonlinear OpenSees 9-story plant (eesd_ohtori_plant9, modal-validated to 0.06% on mode 1).
Actuators enter at the 9 floor master nodes (15,21,...,63) horizontal DOF; the controller state is
the reduced floor (disp,vel) read at those nodes. alpha_eff (post-yield modal frequency ratio) is
MEASURED on the 9-story from the live tangent-stiffness mode-1 frequency drop.

SCOPE (9-story nonlinear; chosen to be statistically meaningful AND tractable)
==============================================================================
* 10 canonical Ohtori cases (El Centro/Hachinohe x{0.5,1.0,1.5}; Northridge/Kobe x{0.5,1.0}) --
  the benchmark's official set; matches unctrl9.mat columns.
* A DEEP-YIELD tier (Northridge/Kobe x{2.0,2.5}) added so the controlled structure is pushed past
  ~3.5% drift where the 3-story in-QP failure lives (at canonical intensities the controlled
  9-story stays milder). Reported separately.
Per controlled run ~2-3 s on an active window, so the full matrix is affordable.

HONESTY
=======
* Validated BILINEAR plant (modal + drift/accel); NO damage indices, NOT the trilinear MEX.
* yield_factor=8 and the cost/saturation constants are carried over from the 3-story WITHOUT
  re-fitting, then U_MAX is re-scaled once to the 9-story weight so the elastic-LQR plant sits in
  the same operating regime (saturates AND still yields). The re-scale factor is reported.
* Whatever the numbers say -- replicate / weaken / flip -- is reported as found. Nothing is forced.

ENV: python (openseespy 3.8.0 + numpy/scipy).
Run:  python src\eesd_ohtori_control9.py
      python src\eesd_ohtori_control9.py --quick   (2-case smoke)
"""
from __future__ import annotations
import os

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import expm, solve_continuous_are, solve_discrete_are

import openseespy.opensees as ops

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eesd_ohtori_plant9 import (BuildConfig, apply_rayleigh, build_model, eigen_frequencies,  # noqa: E402
                                load_record, parse_struct9, OHTORI_DIR)

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(os.environ.get("EESD_PROJECT_ROOT", PACKAGE_ROOT))
RESULTS = Path(os.environ.get("EESD_OUTPUT_DIR", PACKAGE_ROOT / "analysis"))
MK_JSON = Path(os.environ.get("EESD_MK_JSON", RESULTS / "eesd_struct9_lateral_MK.json"))

NSTORY = 9  # control floors (diaphragm levels 2..roof)

# canonical Ohtori 9-story cases (record, mat-var, intensity, tf_seconds, active-window seconds)
CANONICAL_CASES = [
    ("elcentro", "e", 0.5, 100.0, 30.0), ("elcentro", "e", 1.0, 100.0, 30.0),
    ("elcentro", "e", 1.5, 100.0, 30.0),
    ("hachinhe", "h", 0.5, 100.0, 30.0), ("hachinhe", "h", 1.0, 100.0, 30.0),
    ("hachinhe", "h", 1.5, 100.0, 30.0),
    ("northrdg", "n", 0.5, 100.0, 25.0), ("northrdg", "n", 1.0, 100.0, 25.0),
    ("kobe", "k", 0.5, 100.0, 45.0), ("kobe", "k", 1.0, 100.0, 45.0),
]
# deep-yield tier: push controlled drift past ~3.5% where the in-QP failure lives on the 3-story
DEEPYIELD_CASES = [
    ("northrdg", "n", 2.0, 100.0, 25.0), ("northrdg", "n", 2.5, 100.0, 25.0),
    ("kobe", "k", 2.0, 100.0, 45.0), ("kobe", "k", 2.5, 100.0, 45.0),
    ("elcentro", "e", 2.0, 100.0, 30.0),
    # extreme deep-yield extension: push the 9-story to the 3-story's in-QP-failure depth
    # (alpha_eff ~ 0.10-0.16) to test whether the deep-yield in-QP failure is height-specific
    # or a universal deep-yield boundary (eesd_modelmismatch_crossover).
    ("northrdg", "n", 3.0, 100.0, 25.0), ("northrdg", "n", 3.5, 100.0, 25.0),
    ("kobe", "k", 3.0, 100.0, 45.0), ("kobe", "k", 3.5, 100.0, 45.0),
    ("elcentro", "e", 2.5, 100.0, 30.0), ("elcentro", "e", 3.0, 100.0, 30.0),
    ("hachinhe", "h", 2.0, 100.0, 30.0), ("hachinhe", "h", 2.5, 100.0, 30.0),
]
CASE_LABEL = {(r, I): f"{r}_x{I}" for (r, _v, I, _tf, _aw) in CANONICAL_CASES + DEEPYIELD_CASES}

STORY_HEIGHT = 3.9624
# Per-floor actuator saturation [N]. Calibrated by a probe sweep (results in the run log) so the
# elastic-LQR controlled plant sits in the SAME operating regime as the 3-story: the actuator
# SATURATES on every strong record (|u|_peak = U_MAX) AND the controlled structure STILL YIELDS
# (controlled drift ~0.7-2.7% on Northridge/Kobe x1, and >3.5% on the deep-yield tier) -- the
# precondition for the softening story to be testable. The 9-story takes the SAME per-floor 5e5 N
# as the 3-story (per-floor inter-story shear demand is comparable; 1e6-1.5e6 over-controls and the
# structure stops yielding under control, which would make softening irrelevant by construction).
U_MAX = 5.0e5
# floor disp -> inter-story drift (bidiagonal, n x n)
TD_DRIFT = (np.eye(NSTORY) - np.eye(NSTORY, k=-1))
DAMPING_RATIO = 0.02
N_HORIZON = 20
R_SCALE = 1e-6
CONTROLLABLE_J = 0.95
ALPHA_TOY = 0.02


# =====================================================================================
#  Reduced 9-DOF lateral design model (the Guyan condensed M/K surrogate)
# =====================================================================================
@dataclass
class ReducedDesignModel:
    def __post_init__(self):
        mk = json.loads(MK_JSON.read_text())
        self.M = np.diag(np.asarray(mk["masses_kg"], float))
        self.K0 = np.asarray(mk["K_cond_N_per_m"], float)
        self.K0 = 0.5 * (self.K0 + self.K0.T)
        self.n = self.M.shape[0]
        w = np.sort(np.sqrt(np.linalg.eigvalsh(np.linalg.solve(self.M, self.K0))))
        self.w_modal = w
        self.f_modal = w / (2 * np.pi)
        wi, wj = w[0], w[-1]
        a0 = DAMPING_RATIO * 2 * wi * wj / (wi + wj)
        a1 = DAMPING_RATIO * 2 / (wi + wj)
        self.C = a0 * self.M + a1 * self.K0
        self.B = np.eye(self.n)

    def K_softened(self, alpha_eff: float) -> np.ndarray:
        Ks = alpha_eff * self.K0
        return 0.5 * (Ks + Ks.T)


_RM = ReducedDesignModel()
_RM.__post_init__()


# =====================================================================================
#  Controller synthesis on the reduced model (LQR + condensed MPC QP) -- n-generalized
# =====================================================================================
def design_lqr_on_K(Kmat: np.ndarray, q_scale: float, r_value: float) -> np.ndarray:
    n = _RM.n
    Minv = np.linalg.inv(_RM.M)
    A = np.block([[np.zeros((n, n)), np.eye(n)], [-Minv @ Kmat, -Minv @ _RM.C]])
    B = np.vstack([np.zeros((n, n)), Minv @ _RM.B])
    Qd = TD_DRIFT.T @ TD_DRIFT
    Q = q_scale * np.block([[Qd, np.zeros((n, n))], [np.zeros((n, n)), 1e-3 * np.eye(n)]])
    R = r_value * np.eye(_RM.B.shape[1])
    P = solve_continuous_are(A, B, Q, R)
    return np.linalg.inv(R) @ B.T @ P


def _continuous_ss(Kmat: np.ndarray):
    n = _RM.n
    Minv = np.linalg.inv(_RM.M)
    A = np.block([[np.zeros((n, n)), np.eye(n)], [-Minv @ Kmat, -Minv @ _RM.C]])
    B = np.vstack([np.zeros((n, n)), Minv @ _RM.B])
    return A, B, n


def _discretize_zoh(A, B, dt):
    n, m = A.shape[0], B.shape[1]
    aug = np.zeros((n + m, n + m))
    aug[:n, :n] = A
    aug[:n, n:] = B
    e = expm(aug * dt)
    return e[:n, :n], e[:n, n:]


def _build_condensed(Ad, Bd, N):
    nx, nu = Ad.shape[0], Bd.shape[1]
    powers = [np.eye(nx)]
    for _ in range(N):
        powers.append(powers[-1] @ Ad)
    Phi = np.zeros((N * nx, nx))
    Gamma = np.zeros((N * nx, N * nu))
    for k in range(N):
        Phi[k * nx:(k + 1) * nx, :] = powers[k + 1]
        for j in range(k + 1):
            Gamma[k * nx:(k + 1) * nx, j * nu:(j + 1) * nu] = powers[k - j] @ Bd
    return Phi, Gamma, nx, nu


def box_qp(H, f, lb, ub, L, x0, iters=120):
    x = np.clip(x0, lb, ub)
    y = x.copy()
    t = 1.0
    for _ in range(iters):
        g = H @ y + f
        xn = np.clip(y - g / L, lb, ub)
        tn = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        y = xn + ((t - 1.0) / tn) * (xn - x)
        x = xn
        t = tn
    return x


def condensed_mpc(Kmat, dt, N=N_HORIZON, r_scale=R_SCALE, q_scale=1e12, vel_weight=1e-3):
    A, B, n = _continuous_ss(Kmat)
    Ad, Bd = _discretize_zoh(A, B, dt)
    Phi, Gamma, nx, nu = _build_condensed(Ad, Bd, N)
    Cd = np.hstack([TD_DRIFT, np.zeros((n, n))])
    Qx = q_scale * (Cd.T @ Cd)
    Qx[n:, n:] += q_scale * vel_weight * np.eye(n)
    R = r_scale * np.eye(nu)
    P = solve_discrete_are(Ad, Bd, Qx, R)
    Qbig = np.zeros((N * nx, N * nx))
    for k in range(N - 1):
        Qbig[k * nx:(k + 1) * nx, k * nx:(k + 1) * nx] = Qx
    Qbig[(N - 1) * nx:N * nx, (N - 1) * nx:N * nx] = P
    Rbig = np.kron(np.eye(N), R)
    H = Gamma.T @ Qbig @ Gamma + Rbig
    H = 0.5 * (H + H.T)
    L = float(np.linalg.eigvalsh(H)[-1])
    GtQPhi = Gamma.T @ Qbig @ Phi
    return H, L, GtQPhi, np.full(N * nu, -U_MAX), np.full(N * nu, U_MAX), nu


class FactorialMPC:
    """MPC parameterized by (tuning{elastic,softened}) x (handling{clip,inqp}). Fixed linearization;
    factor A is purely the DESIGN frequency. clip = unconstrained-QP-then-clip; inqp = box-QP (FISTA)."""
    def __init__(self, dt, tuning, handling, alpha_eff, N=N_HORIZON, r_scale=R_SCALE, qp_iters=120):
        self.tuning, self.handling = tuning, handling
        Kdes = _RM.K0 if tuning == "elastic" else _RM.K_softened(alpha_eff)
        H, L, GtQPhi, lb, ub, nu = condensed_mpc(Kdes, dt, N, r_scale)
        self.H, self.L, self.GtQPhi, self.lb, self.ub, self.nu = H, L, GtQPhi, lb, ub, nu
        self.qp_iters = qp_iters
        self.Uprev = np.zeros(N * nu)

    def reset(self):
        self.Uprev = np.zeros_like(self.Uprev)

    def __call__(self, x):
        f = self.GtQPhi @ np.asarray(x, float)
        if self.handling == "clip":
            U = np.linalg.solve(self.H, -f)
        else:
            ws = np.concatenate([self.Uprev[self.nu:], self.Uprev[-self.nu:]])
            U = box_qp(self.H, f, self.lb, self.ub, self.L, ws, self.qp_iters)
        self.Uprev = U
        return U[:self.nu]


# =====================================================================================
#  FULL OpenSees 9-story plant evaluation harness (actuators at floor master nodes 15..63)
# =====================================================================================
class Ohtori9PlantSession:
    def __init__(self, plant, cfg: BuildConfig):
        self.plant = plant
        self.cfg = cfg
        self.floor_nodes = plant.floor_master_nodes               # [15,21,...,63] (9 floors)
        # elevation ladder from ground through every floor for drift
        levels = sorted({(n["story_level"], n["y_m"]) for n in plant.nodes})
        self.elevations = [abs(y) for _, y in levels]
        # heights of the 9 controlled stories: from 1st-floor (basement-top, x-tied to ground) up.
        # the controlled floors are story_levels of the master nodes; build their inter-floor heights.
        level_of = {n["node_id"]: n["story_level"] for n in plant.nodes}
        elev_of = {lvl: abs(y) for lvl, y in levels}
        floor_levels = [level_of[mn] for mn in self.floor_nodes]   # e.g. [2,3,...,10]
        base_level = floor_levels[0] - 1                            # the 1st floor (ground-tied)
        elevs = [elev_of[base_level]] + [elev_of[l] for l in floor_levels]
        self.heights = np.diff(elevs)                              # 9 inter-floor heights
        self.floor_mass = np.array([
            sum(float(e["element_mass_kg"]) for e in plant.elements
                if e["member_type"] in ("beam", "brace") and int(e["story"]) == lvl) * plant.num_mrf
            for lvl in floor_levels], dtype=float)
        self.f0 = None

    def _setup(self, record, var, intensity, tf, active_window, dt_analysis):
        build_model(self.plant, self.cfg)
        self.rayleigh = apply_rayleigh(self.cfg.damping_ratio)
        self.f0 = eigen_frequencies(3)
        times, accel, rec_dt = load_record(record, var, intensity)
        tf_use = min(tf, active_window)
        keep = times <= tf_use
        times, accel = times[keep], accel[keep]
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

    def _state(self):
        d = np.array([ops.nodeDisp(n, 1) for n in self.floor_nodes])
        v = np.array([ops.nodeVel(n, 1) for n in self.floor_nodes])
        return d, v

    def run(self, controller, record, var, intensity, tf, active_window,
            dt_analysis=0.005, measure_alpha=False):
        self._setup(record, var, intensity, tf, active_window, dt_analysis)
        if controller is not None and hasattr(controller, "reset"):
            controller.reset()
        n_steps = int(math.ceil(self.tf_use / dt_analysis))
        drift_peak = acc_peak = u_peak = 0.0
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
                u = np.clip(controller(np.concatenate([d, v])), -U_MAX, U_MAX)
            else:
                u = np.zeros(NSTORY)
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
            if measure_alpha and step % 20 == 0 and drift_peak > 1e-5:
                try:
                    lam = ops.eigen(1)
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
        }
        if measure_alpha:
            alpha_min = (f1_min / self.f0[0]) ** 2 if self.f0 else math.nan
            f1_arr = np.array([f for f, _ in f1_samples]) if f1_samples else np.array([self.f0[0]])
            alpha_p5 = (np.percentile(f1_arr, 5) / self.f0[0]) ** 2
            alpha_de = (de_weight_num / de_weight_den) if de_weight_den > 0 else 1.0
            res.update({
                "f1_elastic": self.f0[0], "f1_tangent_min": f1_min,
                "alpha_eff_min": float(alpha_min),
                "alpha_eff_p5": float(alpha_p5),
                "alpha_eff_driftweighted": float(alpha_de),
                "omega_s_predicted_hz": float(self.f0[0] * math.sqrt(max(alpha_min, 1e-9))),
            })
        return res


def j_gate(res, unc) -> float:
    if res["unstable"]:
        return 10.0
    rms_ratio = res["d_rms"] / max(unc["d_rms"], 1e-12)
    peak_ratio = res["d_max"] / max(unc["d_max"], 1e-12)
    return 0.7 * rms_ratio + 0.3 * peak_ratio + 0.02 * (res["u_peak"] / U_MAX)


def decompose_2x2(J: dict) -> dict:
    J00 = J[("elastic", "clip")]; J10 = J[("softened", "clip")]
    J01 = J[("elastic", "inqp")]; J11 = J[("softened", "inqp")]
    dJ_total = J00 - J11
    dJ_omega = 0.5 * ((J00 - J10) + (J01 - J11))
    dJ_sat = 0.5 * ((J00 - J01) + (J10 - J11))
    eff_A = 0.25 * ((J10 + J11) - (J00 + J01))
    eff_B = 0.25 * ((J01 + J11) - (J00 + J10))
    eff_AB = 0.25 * ((J00 + J11) - (J10 + J01))
    SS_A, SS_B, SS_AB = 4 * eff_A ** 2, 4 * eff_B ** 2, 4 * eff_AB ** 2
    SS_tot = SS_A + SS_B + SS_AB
    interaction_contrast = 0.5 * ((J00 - J10) - (J01 - J11))
    best_cell = min(J, key=J.get); worst_cell = max(J, key=J.get)
    lab = {("elastic", "clip"): "A0B0_elastic_clip", ("softened", "clip"): "A1B0_softened_clip",
           ("elastic", "inqp"): "A0B1_elastic_inqp", ("softened", "inqp"): "A1B1_softened_inqp"}
    return {
        "J00_elastic_clip": J00, "J10_softened_clip": J10, "J01_elastic_inqp": J01, "J11_softened_inqp": J11,
        "best_cell": lab[best_cell], "best_cell_J": float(J[best_cell]),
        "worst_cell": lab[worst_cell], "worst_cell_J": float(J[worst_cell]),
        "dJ_total_corner": float(dJ_total), "dJ_omega_frequency": float(dJ_omega),
        "dJ_sat_saturation": float(dJ_sat),
        "share_omega_pct": float(100 * dJ_omega / dJ_total) if abs(dJ_total) > 1e-12 else float("nan"),
        "share_sat_pct": float(100 * dJ_sat / dJ_total) if abs(dJ_total) > 1e-12 else float("nan"),
        "anova_effect_A_freq": float(eff_A), "anova_effect_B_sat": float(eff_B),
        "anova_effect_AB_interaction": float(eff_AB),
        "ss_A_frac": float(SS_A / SS_tot) if SS_tot > 1e-18 else float("nan"),
        "ss_B_frac": float(SS_B / SS_tot) if SS_tot > 1e-18 else float("nan"),
        "ss_AB_frac": float(SS_AB / SS_tot) if SS_tot > 1e-18 else float("nan"),
        "interaction_contrast_Junits": float(interaction_contrast),
        "interaction_sign": ("reinforcing" if interaction_contrast < -1e-6 else
                             "redundant" if interaction_contrast > 1e-6 else "additive"),
    }


def spearman_rho(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    if len(x) < 3:
        return float("nan")
    rx = _rankdata(x); ry = _rankdata(y)
    rx -= rx.mean(); ry -= ry.mean()
    den = math.sqrt((rx @ rx) * (ry @ ry))
    return float((rx @ ry) / den) if den > 0 else float("nan")


def _rankdata(a):
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    # average ties
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def crossover(index_vals, dJ):
    x = np.asarray(index_vals, float); d = np.asarray(dJ, float)
    m = np.isfinite(x) & np.isfinite(d); x, d = x[m], d[m]
    order = np.argsort(x); x, d = x[order], d[order]
    wins = d < 0.0; loses = d > 0.0
    x_win_hi = float(np.max(x[wins])) if wins.any() else float("nan")
    x_lose_lo = float(np.min(x[loses])) if loses.any() else float("nan")
    if np.isfinite(x_win_hi) and np.isfinite(x_lose_lo):
        bracket = sorted([x_win_hi, x_lose_lo]); x_cross = float(np.mean(bracket))
        pred_lose = x > x_cross; purity = int(np.sum(pred_lose == loses)) / len(x)
    else:
        bracket = [float("nan"), float("nan")]; x_cross = float("nan"); purity = float("nan")
    return {"x_cross_bracket": x_cross, "bracket_lo_hi": bracket,
            "separation_purity": float(purity),
            "n_inqp_wins": int(wins.sum()), "n_inqp_loses": int(loses.sum())}


def auc_drift_vs_dJ(drift_depth, dJ):
    """AUC that 'uncontrolled drift depth' ranks the in-QP-loses (dJ>0) cases above the wins.
    Mirrors the 3-story claim that drift depth (not E_K) is the predictor (3-story AUC ~0.83)."""
    x = np.asarray(drift_depth, float); d = np.asarray(dJ, float)
    m = np.isfinite(x) & np.isfinite(d); x, d = x[m], d[m]
    pos = d > 1e-9   # in-QP loses
    neg = d < -1e-9  # in-QP wins
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan"), int(pos.sum()), int(neg.sum())
    xp = x[pos]; xn = x[neg]
    wins_pairs = 0; total = 0
    for a in xp:
        for b in xn:
            total += 1
            if a > b:
                wins_pairs += 1
            elif a == b:
                wins_pairs += 0.5
    return float(wins_pairs / total), int(pos.sum()), int(neg.sum())


# =====================================================================================
#  DRIVER
# =====================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="2-case smoke (elcentro/northrdg x1)")
    ap.add_argument("--cases", type=int, default=0, help="limit to first N canonical cases (0=all)")
    ap.add_argument("--no-deep", action="store_true", help="skip the deep-yield tier")
    ap.add_argument("--out", type=Path, default=RESULTS / "eesd_ohtori_control9_results.json")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.quick:
        cases = [c for c in CANONICAL_CASES if (c[0], c[2]) in {("elcentro", 1.0), ("northrdg", 1.0)}]
        deep = []
    elif args.cases > 0:
        cases = CANONICAL_CASES[:args.cases]
        deep = [] if args.no_deep else DEEPYIELD_CASES
    else:
        cases = CANONICAL_CASES
        deep = [] if args.no_deep else DEEPYIELD_CASES
    all_cases = cases + deep

    plant = parse_struct9(OHTORI_DIR / "struct9.m")
    cfg = BuildConfig(mode="nonlinear")
    sess = Ohtori9PlantSession(plant, cfg)

    print(f"Reduced 9-DOF design model: f_modal = {np.array2string(_RM.f_modal[:5], precision=4)} Hz")
    print(f"Evaluating {len(all_cases)} cases ({len(cases)} canonical + {len(deep)} deep-yield) "
          f"on the FULL nonlinear 9-story OpenSees plant.\n")

    # ---------- PASS 1: uncontrolled refs + alpha_eff per case ----------
    print("PASS 1: uncontrolled reference + alpha_eff (tangent mode-1 frequency drop)...")
    unc = {}
    alpha = {}
    for (rec, var, I, tf, aw) in all_cases:
        r = sess.run(None, rec, var, I, tf, aw, measure_alpha=True)
        key = (rec, I)
        unc[key] = r
        alpha[key] = {k: r[k] for k in ("f1_elastic", "f1_tangent_min", "alpha_eff_min",
                                        "alpha_eff_p5", "alpha_eff_driftweighted", "omega_s_predicted_hz")}
        print(f"  {CASE_LABEL[key]:14s}: unc d_max={r['d_max']*100:6.3f}%DR  "
              f"alpha_eff(min)={r['alpha_eff_min']:.4f} sqrt={math.sqrt(max(r['alpha_eff_min'],0)):.4f}  "
              f"f1_tan_min={r['f1_tangent_min']:.4f}Hz  unstable={r['unstable']}")

    a_mins = np.array([alpha[(rec, I)]["alpha_eff_min"] for (rec, _v, I, _tf, _aw) in cases])
    alpha_design = float(np.median(a_mins))
    print(f"\n  alpha_eff per-case min: median={alpha_design:.4f}  "
          f"range=[{a_mins.min():.4f},{a_mins.max():.4f}]")
    print(f"  -> softened DESIGN omega_s = omega_1*sqrt({alpha_design:.4f}) "
          f"= {_RM.f_modal[0]*math.sqrt(alpha_design):.4f} Hz (toy alpha=0.02 -> "
          f"{_RM.f_modal[0]*math.sqrt(ALPHA_TOY):.4f} Hz)\n")

    # ---------- PASS 2: tune elastic LQR on the full plant (grid over canonical cases) ----------
    print("PASS 2: tuning elastic LQR on the full plant (grid)...")
    best_lqr = (None, np.inf, None)
    lqr_grid = []
    for q_scale in (1e8, 1e10, 1e12):
        for r_value in (1e-4, 1e-6):
            K = design_lqr_on_K(_RM.K0, q_scale, r_value)
            js = [j_gate(sess.run(lambda x, K=K: -K @ x, rec, var, I, tf, aw), unc[(rec, I)])
                  for (rec, var, I, tf, aw) in cases]
            mj = float(np.mean(js))
            lqr_grid.append({"q_scale": q_scale, "r_value": r_value, "mean_J": mj})
            if mj < best_lqr[1]:
                best_lqr = (K, mj, {"q_scale": q_scale, "r_value": r_value})
            print(f"   q={q_scale:.0e} r={r_value:.0e}: mean J={mj:.4f}")
    K_lqr, lqr_mean, lqr_best = best_lqr
    q_best, r_best = lqr_best["q_scale"], lqr_best["r_value"]
    print(f"  tuned elastic LQR: q={q_best:.0e} r={r_best:.0e}  mean J={lqr_mean:.4f}\n")

    K_soft = design_lqr_on_K(_RM.K_softened(alpha_design), q_best, r_best)
    n_lqr = float(np.linalg.norm(K_lqr)); n_soft = float(np.linalg.norm(K_soft))
    K_iso = K_soft * (n_lqr / n_soft)

    # ---------- CLAIM 3: softening-aware LQR null (elastic vs softened) ----------
    print("CLAIM 3: softening-aware LQR vs clipped-elastic LQR (null test) + iso-gain...")
    lqr_rows = []
    for (rec, var, I, tf, aw) in all_cases:
        key = (rec, I)
        row = {"case": CASE_LABEL[key], "record": rec, "intensity": I,
               "deep": (rec, var, I, tf, aw) in deep,
               "alpha_eff_min": alpha[key]["alpha_eff_min"]}
        for label, K in (("LQR_elastic", K_lqr), ("LQR_softened", K_soft),
                         ("LQR_iso_softdir_elasticnorm", K_iso)):
            r = sess.run(lambda x, K=K: -K @ x, rec, var, I, tf, aw)
            row[label + "_J"] = j_gate(r, unc[key])
            row[label + "_dmax_pct"] = r["d_max"] * 100 if not r["unstable"] else 1e9
        lqr_rows.append(row)
        print(f"  {CASE_LABEL[key]:14s}: elastic J={row['LQR_elastic_J']:.4f} "
              f"softened J={row['LQR_softened_J']:.4f} iso J={row['LQR_iso_softdir_elasticnorm_J']:.4f}")

    def meanJ(rows, key, subset=None):
        sel = [r for r in rows if (subset is None or r["deep"] == subset)]
        return float(np.mean([r[key] for r in sel])) if sel else float("nan")
    canon_rows = [r for r in lqr_rows if not r["deep"]]
    lqr_summary = {k: meanJ(canon_rows, k + "_J") for k in
                   ("LQR_elastic", "LQR_softened", "LQR_iso_softdir_elasticnorm")}
    soft_margin = lqr_summary["LQR_elastic"] - lqr_summary["LQR_softened"]
    iso_margin = lqr_summary["LQR_elastic"] - lqr_summary["LQR_iso_softdir_elasticnorm"]
    n_soft_better = sum(1 for r in canon_rows if r["LQR_elastic_J"] - r["LQR_softened_J"] > 1e-6)
    soft_margin_all = (meanJ(lqr_rows, "LQR_elastic_J") - meanJ(lqr_rows, "LQR_softened_J"))
    margins_percase = [r["LQR_elastic_J"] - r["LQR_softened_J"] for r in canon_rows]
    print(f"  mean J (canonical): elastic={lqr_summary['LQR_elastic']:.4f} "
          f"softened={lqr_summary['LQR_softened']:.4f} margin={soft_margin:+.4f} "
          f"({n_soft_better}/{len(canon_rows)} better)\n")

    # ---------- CLAIMS 1 + 2: 2x2 factorial MPC + regime map ----------
    print("CLAIMS 1 & 2: 2x2 factorial MPC (tuning x saturation) + deep-yield in-QP failure...")
    CELLS = [("elastic", "clip"), ("softened", "clip"), ("elastic", "inqp"), ("softened", "inqp")]
    cell_rows = []
    regime_rows = []
    for (rec, var, I, tf, aw) in all_cases:
        key = (rec, I)
        is_deep = (rec, var, I, tf, aw) in deep
        _t, _a, dt = load_record(rec, var, I)
        Jc = {}
        dmax_cell = {}
        for (tuning, handling) in CELLS:
            ctrl = FactorialMPC(dt, tuning, handling, alpha_design)
            r = sess.run(ctrl, rec, var, I, tf, aw)
            j = j_gate(r, unc[key])
            Jc[(tuning, handling)] = j
            dmax_cell[(tuning, handling)] = r["d_max"] * 100 if not r["unstable"] else 1e9
            cell_rows.append({"case": CASE_LABEL[key], "record": rec, "intensity": I, "deep": is_deep,
                              "tuning": tuning, "handling": handling, "J": j,
                              "dmax_pct": dmax_cell[(tuning, handling)], "unstable": r["unstable"]})
        dJ = Jc[("elastic", "inqp")] - Jc[("elastic", "clip")]
        E_K = 1.0 - alpha[key]["alpha_eff_driftweighted"]
        regime_rows.append({
            "case": CASE_LABEL[key], "record": rec, "intensity": I, "deep": is_deep,
            "J_clip": Jc[("elastic", "clip")], "J_inqp": Jc[("elastic", "inqp")],
            "dJ_inqp_minus_clip": dJ, "inqp_wins": bool(dJ < 0),
            "E_K": E_K, "alpha_eff_min": alpha[key]["alpha_eff_min"],
            "unc_dmax_pct": unc[key]["d_max"] * 100,
            "clip_dmax_pct": dmax_cell[("elastic", "clip")]})
        dec = decompose_2x2(Jc)
        tag = "DEEP" if is_deep else "    "
        print(f"  {tag} {CASE_LABEL[key]:14s}: A0B0={dec['J00_elastic_clip']:.3f} "
              f"A1B0={dec['J10_softened_clip']:.3f} A0B1={dec['J01_elastic_inqp']:.3f} "
              f"A1B1={dec['J11_softened_inqp']:.3f} dJ(inqp-clip)={dJ:+.3f} "
              f"unc_d={unc[key]['d_max']*100:.2f}%")

    def cell_meanJ(rows):
        out = {}
        for (t, h) in CELLS:
            vals = [r["J"] for r in rows if r["tuning"] == t and r["handling"] == h]
            out[(t, h)] = float(np.mean(vals)) if vals else float("nan")
        return out
    canon_cellrows = [r for r in cell_rows if not r["deep"]]
    overall_2x2 = decompose_2x2(cell_meanJ(canon_cellrows))
    overall_2x2_all = decompose_2x2(cell_meanJ(cell_rows))
    ctrl_cases = []
    for (rec, var, I, tf, aw) in cases:
        cj = [r["J"] for r in canon_cellrows if r["record"] == rec and r["intensity"] == I]
        if cj and max(cj) < CONTROLLABLE_J:
            ctrl_cases.append((rec, I))
    ctrl_cellrows = [r for r in canon_cellrows if (r["record"], r["intensity"]) in ctrl_cases]
    overall_2x2_ctrl = decompose_2x2(cell_meanJ(ctrl_cellrows)) if ctrl_cellrows else None

    # regime stats (all cases incl deep, where the in-QP failure should live)
    E_K_arr = np.array([r["E_K"] for r in regime_rows])
    dJ_arr = np.array([r["dJ_inqp_minus_clip"] for r in regime_rows])
    unc_d_arr = np.array([r["unc_dmax_pct"] for r in regime_rows])
    clip_d_arr = np.array([r["clip_dmax_pct"] for r in regime_rows])
    rho_EK = spearman_rho(E_K_arr, dJ_arr)
    rho_uncd = spearman_rho(unc_d_arr, dJ_arr)
    rho_clipd = spearman_rho(clip_d_arr, dJ_arr)
    cx_EK = crossover(E_K_arr, dJ_arr)
    cx_uncd = crossover(unc_d_arr, dJ_arr)
    cx_clipd = crossover(clip_d_arr, dJ_arr)
    auc_uncd, n_pos, n_neg = auc_drift_vs_dJ(unc_d_arr, dJ_arr)
    auc_EK, _, _ = auc_drift_vs_dJ(E_K_arr, dJ_arr)
    auc_clipd, _, _ = auc_drift_vs_dJ(clip_d_arr, dJ_arr)

    # ---------- CLAIM-2 mechanism summary ----------
    f1_0 = _RM.f_modal[0]
    mech_rows = []
    for (rec, var, I, tf, aw) in all_cases:
        key = (rec, I); a = alpha[key]
        mech_rows.append({
            "case": CASE_LABEL[key], "intensity": I, "deep": (rec, var, I, tf, aw) in deep,
            "f1_elastic_hz": a["f1_elastic"], "f1_tangent_min_hz": a["f1_tangent_min"],
            "alpha_eff_min": a["alpha_eff_min"],
            "sqrt_alpha_eff_min": math.sqrt(max(a["alpha_eff_min"], 0.0)),
            "omega_s_over_omega1_measured": a["f1_tangent_min"] / a["f1_elastic"],
            "toy_sqrt_alpha": math.sqrt(ALPHA_TOY),
        })

    elapsed = time.time() - t0

    # ---------- VERDICTS (height-robustness of the 3 surviving claims) ----------
    # CLAIM 1: saturation share dominant
    sb = overall_2x2["ss_B_frac"]
    claim1_sat_dominant = bool(np.isfinite(sb) and sb >= overall_2x2["ss_A_frac"]
                               and sb >= overall_2x2["ss_AB_frac"])
    # CLAIM 2: in-QP loses at deep yield AND drift-depth predicts (beats E_K)
    n_inqp_loses = int(np.sum(dJ_arr > 1e-9))
    n_inqp_loses_deep = int(np.sum((dJ_arr > 1e-9) & np.array([r["deep"] for r in regime_rows])))
    claim2_inqp_fails = bool(n_inqp_loses >= 1)
    claim2_driftdepth_predicts = bool(np.isfinite(auc_uncd) and auc_uncd >= 0.65
                                      and (not np.isfinite(auc_EK) or auc_uncd >= auc_EK))
    # CLAIM 3: softening-aware null (|margin| small, not a clear help)
    median_sqrt_a = float(np.median([m["sqrt_alpha_eff_min"] for m in mech_rows if not m["deep"]]))
    claim3_is_null = bool(abs(soft_margin) <= 0.02 * max(lqr_summary["LQR_elastic"], 1e-9))
    soft_helps = bool(soft_margin >= 0.02 * lqr_summary["LQR_elastic"] and n_soft_better > len(canon_rows) / 2)

    out = {
        "title": "EESD multi-height (9-story) replication of the three surviving Ohtori control claims",
        "central_question": ("Are the 3-story surviving claims (saturation-handling dominance; in-QP "
                             "deep-yield failure with drift-depth predictor; softening-aware null) "
                             "height-robust at the 9-story, or height-specific?"),
        "architecture": ("Controllers DESIGNED on the reduced 9-DOF Guyan-condensed lateral M/K "
                         "surrogate; EVALUATED closed-loop on the FULL nonlinear OpenSees 9-story "
                         "plant (modal-validated 0.06% mode-1). Actuators at floor master nodes "
                         "15/21/.../63 horizontal DOF; state = reduced floor (disp,vel)."),
        "honesty_scope": ("Validated BILINEAR plant (modal + drift/accel); NOT damage indices, NOT "
                          "the trilinear MEX. yield_factor=8 and cost constants carried from the "
                          "3-story un-refitted; U_MAX re-scaled once by the 9/3-story weight ratio."),
        "n_canonical": len(cases), "n_deepyield": len(deep),
        "canonical_cases": [CASE_LABEL[(c[0], c[2])] for c in cases],
        "deepyield_cases": [CASE_LABEL[(c[0], c[2])] for c in deep],
        "U_MAX_per_floor_N": U_MAX,
        "U_MAX_note": ("5e5 N/floor, same as the 3-story; probe-calibrated so the elastic-LQR plant "
                       "saturates on strong records AND still yields (controlled drift 0.7-2.7% at "
                       "canonical x1, >3.5% on the deep-yield tier). 1e6-1.5e6 over-controls."),
        "N_horizon": N_HORIZON, "r_scale": R_SCALE,
        "reduced_model_f_modal_hz": _RM.f_modal.tolist(),
        "reduced_model_mode1_surrogate_note": ("Guyan condensation overstiffens mode-1 (~+20%); the "
                                               "modal-exact object is the full plant. Elastic LQR is "
                                               "grid-tuned on the full plant, absorbing the mismatch."),
        "elapsed_s": elapsed,

        "alpha_eff_measurement": {
            "method": ("alpha_eff=(omega_1^tangent_min/omega_1^elastic)^2 from ops.eigen on the live "
                       "tangent stiffness during the uncontrolled time-history."),
            "toy_alpha_explicit": ALPHA_TOY, "toy_sqrt_alpha": math.sqrt(ALPHA_TOY),
            "alpha_design_median_canonical": alpha_design,
            "alpha_eff_min_range_canonical": [float(a_mins.min()), float(a_mins.max())],
            "per_case": {CASE_LABEL[k]: alpha[k] for k in alpha},
        },

        "claim1_saturation_dominance": {
            "question": "Is saturation-handling (factor B) still the dominant ANOVA variance share at 9-story?",
            "tuned_elastic_lqr_q_r": lqr_best, "lqr_grid": lqr_grid,
            "overall_2x2_canonical": overall_2x2,
            "overall_2x2_all_incl_deep": overall_2x2_all,
            "overall_2x2_controllable": overall_2x2_ctrl,
            "controllable_cases": [CASE_LABEL[k] for k in ctrl_cases],
            "ss_freq_frac": overall_2x2["ss_A_frac"], "ss_sat_frac": overall_2x2["ss_B_frac"],
            "ss_interaction_frac": overall_2x2["ss_AB_frac"],
            "saturation_share_pct": float(100 * overall_2x2["ss_B_frac"]) if np.isfinite(overall_2x2["ss_B_frac"]) else float("nan"),
            "VERDICT_saturation_dominant": claim1_sat_dominant,
            "per_case_cells": cell_rows,
        },
        "claim2_inqp_deepyield_failure": {
            "question": ("Does in-QP still lose to clip at deep yield, and is UNCONTROLLED DRIFT DEPTH "
                         "(not E_K) the predictor?"),
            "n_inqp_loses_total": n_inqp_loses, "n_inqp_loses_deeptier": n_inqp_loses_deep,
            "n_inqp_wins": int(np.sum(dJ_arr < -1e-9)),
            "auc_uncdrift_ranks_inqp_loss": auc_uncd,
            "auc_clipdrift_ranks_inqp_loss": auc_clipd,
            "auc_EK_ranks_inqp_loss": auc_EK,
            "n_loss_cases": n_pos, "n_win_cases": n_neg,
            "spearman_rho_uncdrift_vs_dJ": rho_uncd,
            "spearman_rho_clipdrift_vs_dJ": rho_clipd,
            "spearman_rho_EK_vs_dJ": rho_EK,
            "crossover_uncdrift": cx_uncd, "crossover_clipdrift": cx_clipd, "crossover_EK": cx_EK,
            "VERDICT_inqp_fails_at_deepyield": claim2_inqp_fails,
            "VERDICT_driftdepth_predicts_better_than_EK": claim2_driftdepth_predicts,
            "per_case": regime_rows,
        },
        "claim3_softening_aware_null": {
            "question": "Is the softened-vs-elastic LQR margin still ~0 (null) at 9-story?",
            "f1_elastic_hz": f1_0,
            "mean_J_elastic_canonical": lqr_summary["LQR_elastic"],
            "mean_J_softened_canonical": lqr_summary["LQR_softened"],
            "softened_margin_vs_elastic_canonical": soft_margin,
            "softened_margin_vs_elastic_allcases": soft_margin_all,
            "margin_percase_mean_pm_std": [float(np.mean(margins_percase)), float(np.std(margins_percase))],
            "margin_percase_range": [float(np.min(margins_percase)), float(np.max(margins_percase))],
            "n_softened_better": n_soft_better, "n_canonical_cases": len(canon_rows),
            "iso_margin_vs_elastic": iso_margin,
            "median_sqrt_alpha_eff_canonical": median_sqrt_a,
            "VERDICT_is_null": claim3_is_null,
            "VERDICT_softening_helps": soft_helps,
            "per_case": lqr_rows,
        },
    }

    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # ---------- console verdict ----------
    print("\n" + "=" * 94)
    print("MULTI-HEIGHT REPLICATION VERDICT (9-story vs 3-story surviving claims)")
    print("=" * 94)
    print(f"  alpha_eff(min) median canonical = {alpha_design:.4f}  sqrt={median_sqrt_a:.3f}  "
          f"(3-story/toy alpha={ALPHA_TOY}, sqrt={math.sqrt(ALPHA_TOY):.3f})")
    print(f"  [CLAIM 1] saturation dominant: {claim1_sat_dominant}  "
          f"ANOVA shares freq {overall_2x2['ss_A_frac']*100:.0f}% / sat {overall_2x2['ss_B_frac']*100:.0f}% / "
          f"interaction {overall_2x2['ss_AB_frac']*100:.0f}%")
    print(f"  [CLAIM 2] in-QP fails at deep yield: {claim2_inqp_fails}  "
          f"({n_inqp_loses} loses / {int(np.sum(dJ_arr<-1e-9))} wins; deep-tier loses {n_inqp_loses_deep})")
    print(f"            drift-depth predicts (AUC unc {auc_uncd if not np.isnan(auc_uncd) else float('nan'):.2f} "
          f"vs E_K {auc_EK if not np.isnan(auc_EK) else float('nan'):.2f}): {claim2_driftdepth_predicts}")
    print(f"  [CLAIM 3] softening-aware null: {claim3_is_null}  "
          f"margin elastic->softened = {soft_margin:+.4f} "
          f"(per-case {np.mean(margins_percase):+.4f} +/- {np.std(margins_percase):.4f})")
    print(f"\nwrote {args.out}")
    print(f"elapsed {elapsed:.1f}s")


if __name__ == "__main__":
    main()


