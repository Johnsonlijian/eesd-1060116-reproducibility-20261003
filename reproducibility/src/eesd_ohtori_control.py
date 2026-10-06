r"""Port the EESD softening-aware control analysis onto the VALIDATED canonical Ohtori 3-story
nonlinear OpenSees benchmark, and answer THE key question: does the toy-3-DOF EESD story replicate
on the real benchmark?

ARCHITECTURE (the methodological upgrade over the toy paper)
===========================================================
TOY PAPER  : controllers were BOTH designed and evaluated on the fast reduced 3-DOF bilinear plant
             (eesd_struct3_plant.Struct3ReducedPlant), whose softening is a single EPP story spring
             with an EXPLICIT post-yield ratio alpha=0.02 (so omega_s = omega_1*sqrt(0.02) = 0.141*omega_1).
THIS PORT  : controllers are DESIGNED on the reduced 3-DOF lateral M/K (the REAL condensed matrices,
             results/eesd_struct3_lateral_MK.json) but EVALUATED CLOSED-LOOP on the FULL nonlinear
             OpenSees Ohtori plant (eesd_ohtori_plant, openseespy 3.8.0, modal-validated to 0.13%,
             El Centro drift to 4%). Actuator forces enter at the floor master nodes (6,11,16) on the
             horizontal DOF; the controller state is the reduced floor (disp,vel) read from those nodes
             -- the same state the reduced-model controller was designed on.

THE alpha RE-DERIVATION (the scientific core)
=============================================
The toy alpha=0.02 is a MODEL CHOICE, not a property of the SAC frame. On the real plant softening comes
from many bilinear members yielding transiently. We MEASURE the effective post-yield modal stiffness
ratio alpha_eff per record from the TANGENT-stiffness mode-1 frequency drop at peak yielding:
    alpha_eff = (omega_1^tangent_min / omega_1^elastic)^2      (eigen on the live tangent K).
We then test whether omega_s = omega_1*sqrt(alpha_eff) is the controller-relevant frequency on the REAL
plant, and we build the softening-aware controllers by re-pointing the design to that MEASURED omega_s
(K_soft = alpha_eff * K0, mass unchanged -> mode-1 at omega_1*sqrt(alpha_eff) exactly), the faithful
generalization of the toy's alpha*k_story.

THE FIVE ANALYSES (ported verbatim in structure; numbers recomputed on the real plant)
======================================================================================
 1. Softening-aware LQR vs clipped-elastic LQR: does retuning to omega_s still help? mean J + per-case.
 2. The 1/sqrt(alpha) frequency-ratio mechanism: does the real plant's post-yield modal frequency follow
    omega_1*sqrt(alpha_eff)? Quantified per record.
 3. 2x2 factorial: tuning{elastic,softened} x saturation{clip,in-QP}; ANOVA variance shares on real plant.
 4. Regime map: E_K (tangent-stiffness mismatch) -> in-QP vs clip sign.
 5. Iso-gain ablation: is the benefit DIRECTION (frequency) not MAGNITUDE (gain norm) on the real plant?

HONESTY
=======
* The plant is a VALIDATED BILINEAR approximation (modal + drift/J1-J6; NOT damage indices J7-J10, NOT the
  exact trilinear MEX). Claims kept to drift/accel response.
* Replication is the whole point. Whatever the numbers say -- full / partial / no replication -- is
  reported as found. Nothing is forced.

ENV: run with python (openseespy 3.8.0 + numpy/scipy/pandas).
Run:  python src\eesd_ohtori_control.py            (4 SAC records, 10 canonical cases)
      python src\eesd_ohtori_control.py --quick    (El Centro/Northridge x1 smoke)
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

from eesd_ohtori_plant import (BuildConfig, apply_rayleigh, build_model, eigen_frequencies,
                               load_record, parse_struct3, OHTORI_DIR)

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(os.environ.get("EESD_PROJECT_ROOT", PACKAGE_ROOT))
RESULTS = Path(os.environ.get("EESD_OUTPUT_DIR", PACKAGE_ROOT / "analysis"))
MK_JSON = Path(os.environ.get("EESD_MK_JSON", RESULTS / "eesd_struct3_lateral_MK.json"))

# ---- canonical Ohtori cases (4 SAC records; the FEMA P695 set is appended in the record-robustness run) ----
# (record, mat-var, intensity, tf_seconds, active-window seconds for speed)
CANONICAL_CASES = [
    ("elcentro", "e", 0.5, 100.0, 45.0), ("elcentro", "e", 1.0, 100.0, 45.0),
    ("elcentro", "e", 1.5, 100.0, 45.0),
    ("hachinhe", "h", 0.5, 100.0, 40.0), ("hachinhe", "h", 1.0, 100.0, 40.0),
    ("hachinhe", "h", 1.5, 100.0, 40.0),
    ("northrdg", "n", 0.5, 100.0, 30.0), ("northrdg", "n", 1.0, 100.0, 30.0),
    ("northrdg", "n", 1.5, 100.0, 30.0),
    ("kobe", "k", 0.5, 180.0, 60.0), ("kobe", "k", 1.0, 180.0, 60.0),
]
CASE_LABEL = {(r, I): f"{r}_x{I}" for (r, _v, I, _tf, _aw) in CANONICAL_CASES}

# control / cost constants (mirror the toy paper exactly so the comparison is apples-to-apples)
STORY_HEIGHT = 3.9624
# Per-floor actuator saturation [N] on the FULL frame. Calibrated (probe sweep) so the elastic-LQR
# controlled plant sits in the SAME operating regime as the toy paper: the actuator SATURATES on every
# strong record (|u|_peak = U_MAX) AND the controlled structure STILL YIELDS (peak drift ~3.5-3.7% on
# Northridge/Kobe x1), giving elastic-LQR mean J ~ 0.6. This is the precondition for the softening story
# to even be testable -- with unlimited authority the structure never yields under control and softening
# is irrelevant by construction. (Toy used 2e5 N on a reduced model directly driving 3 DOFs; on the full
# 2-MRF frame, mass ~2.95e6 kg, the toy-equivalent authority is ~5e5 N/floor.)
U_MAX = 5.0e5
TD_DRIFT = np.array([[1, 0, 0], [-1, 1, 0], [0, -1, 1]], float)  # floor disp -> inter-story drift
DAMPING_RATIO = 0.02
N_HORIZON = 20
R_SCALE = 1e-6
CONTROLLABLE_J = 0.95
ALPHA_TOY = 0.02                      # the toy plant's explicit post-yield ratio (for contrast only)


# =====================================================================================
#  Reduced 3-DOF lateral design model (the REAL condensed M/K; controllers designed here)
# =====================================================================================
@dataclass
class ReducedDesignModel:
    """Reduced 3-DOF lateral design model from the validated condensed M/K (real stiff-base distribution).

    Used ONLY to SYNTHESIZE controllers (LQR gains, MPC condensed QP). Evaluation is on the full plant.
    """
    def __post_init__(self):
        mk = json.loads(MK_JSON.read_text())
        self.M = np.diag(np.asarray(mk["masses_kg"], float))
        self.K0 = np.asarray(mk["K_cond_N_per_m"], float)
        self.K0 = 0.5 * (self.K0 + self.K0.T)
        self.n = 3
        w = np.sort(np.sqrt(np.linalg.eigvalsh(np.linalg.solve(self.M, self.K0))))
        self.w_modal = w
        self.f_modal = w / (2 * np.pi)
        # Rayleigh 2% at modes 1 and 3 (same convention as the toy reduced plant)
        wi, wj = w[0], w[2]
        a0 = DAMPING_RATIO * 2 * wi * wj / (wi + wj)
        a1 = DAMPING_RATIO * 2 / (wi + wj)
        self.C = a0 * self.M + a1 * self.K0
        # one actuator per floor (B = identity selection on the 3 floor DOFs)
        self.B = np.eye(self.n)

    def K_softened(self, alpha_eff: float) -> np.ndarray:
        """Softened design stiffness: K_soft = alpha_eff * K0 -> mode 1 at omega_1*sqrt(alpha_eff).

        Direct generalization of the toy's alpha*k_story, but with the MEASURED alpha_eff (not 0.02).
        Scaling the whole condensed matrix preserves the mode shapes and shifts every modal frequency by
        sqrt(alpha_eff); for a mode-1-dominated drift objective this re-points the controller's design
        frequency to the measured softened fundamental, which is exactly the 1/sqrt(alpha) repair."""
        Ks = alpha_eff * self.K0
        return 0.5 * (Ks + Ks.T)


_RM = ReducedDesignModel()
_RM.__post_init__()


# =====================================================================================
#  Controller synthesis on the reduced model (LQR + condensed MPC QP), reused from the toy formulas
# =====================================================================================
def design_lqr_on_K(Kmat: np.ndarray, q_scale: float, r_value: float) -> np.ndarray:
    """LQR gain on an arbitrary design stiffness Kmat (elastic K0 or softened Ks). Same Q/R as the toy."""
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
    """FISTA box-constrained QP (same self-contained solver as the toy paper)."""
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
    """Condensed QP (H, L, GtQPhi, lb, ub, nu) for an MPC whose prediction model uses stiffness Kmat."""
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
    """MPC parameterized by (tuning in {elastic,softened}) x (handling in {clip,inqp}).

    tuning fixes the internal prediction stiffness: elastic K0 vs softened alpha_eff*K0.
    handling: clip = unconstrained-QP-then-clip; inqp = box-constrained FISTA QP.
    Fixed linearization (no online yield scheduling) -- factor A is purely the DESIGN frequency.
    """
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
#  FULL OpenSees plant evaluation harness (actuator forces at floor master nodes 6,11,16)
# =====================================================================================
class OhtoriPlantSession:
    """Wraps the validated full nonlinear plant for repeated controlled time-histories.

    Build once per (record,intensity): assembles the model, Rayleigh damping, the ground-motion
    UniformExcitation, and a Plain control pattern whose nodal loads are updated each step. Exposes
    reduced floor (disp,vel) as the controller state and computes drift/accel/effort + J_gate.
    """
    def __init__(self, plant, cfg: BuildConfig):
        self.plant = plant
        self.cfg = cfg
        self.floor_nodes = plant.floor_master_nodes               # [6, 11, 16]
        levels = sorted({(n["story_level"], n["y_m"]) for n in plant.nodes})
        self.elevations = [abs(y) for _, y in levels]
        self.heights = np.diff(self.elevations)
        self.floor_mass = np.array([
            sum(float(e["element_mass_kg"]) for e in plant.elements
                if e["member_type"] in ("beam", "brace") and int(e["story"]) == s) * plant.num_mrf
            for s in range(1, plant.num_story + 1)], dtype=float)
        self.f0 = None

    def _setup(self, record, var, intensity, tf, active_window, dt_analysis):
        build_model(self.plant, self.cfg)
        self.rayleigh = apply_rayleigh(self.cfg.damping_ratio)
        self.f0 = eigen_frequencies(3)
        times, accel, rec_dt = load_record(record, var, intensity)
        # trim to the active window (+small tail) for speed; the published unctrl peaks occur here
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
        """Controlled closed-loop run. controller(state[6])->force[3] (or None for uncontrolled).
        Returns drift/accel/effort peaks+rms and (optionally) the tangent-stiffness alpha_eff history."""
        self._setup(record, var, intensity, tf, active_window, dt_analysis)
        if controller is not None and hasattr(controller, "reset"):
            controller.reset()
        n_steps = int(math.ceil(self.tf_use / dt_analysis))
        drift_peak = acc_peak = u_peak = 0.0
        drift_sq = 0.0
        u_sq = 0.0
        nout = 0
        unstable = False
        yielded_steps = 0
        # alpha_eff (tangent mode-1 freq) sampling
        f1_min = self.f0[0] if self.f0 else math.nan
        f1_samples = []
        de_weight_num = 0.0
        de_weight_den = 0.0
        for step in range(n_steps):
            d, v = self._state()
            if controller is not None:
                u = np.clip(controller(np.concatenate([d, v])), -U_MAX, U_MAX)
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
                    # restore transient analysis objects clobbered by eigen
                    ops.test("NormDispIncr", 1e-8, 40); ops.algorithm("Newton")
                except Exception:
                    pass
        dur = max(nout * dt_analysis, dt_analysis)
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


# =====================================================================================
#  J_gate (identical formula to the toy paper, computed on the REAL plant drift)
# =====================================================================================
def j_gate(res, unc) -> float:
    if res["unstable"]:
        return 10.0
    rms_ratio = res["d_rms"] / max(unc["d_rms"], 1e-12)
    peak_ratio = res["d_max"] / max(unc["d_max"], 1e-12)
    return 0.7 * rms_ratio + 0.3 * peak_ratio + 0.02 * (res["u_peak"] / U_MAX)


# =====================================================================================
#  2x2 factorial decomposition algebra (verbatim from eesd_loss_decomposition)
# =====================================================================================
def decompose_2x2(J: dict) -> dict:
    J00 = J[("elastic", "clip")]; J10 = J[("softened", "clip")]
    J01 = J[("elastic", "inqp")]; J11 = J[("softened", "inqp")]
    dJ_total = J00 - J11
    dJ_omega = 0.5 * ((J00 - J10) + (J01 - J11))
    dJ_sat = 0.5 * ((J00 - J01) + (J10 - J11))
    Jbar = 0.25 * (J00 + J10 + J01 + J11)
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
    import pandas as pd
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    if len(x) < 3:
        return float("nan")
    rx = np.array(pd.Series(x).rank().to_numpy(), dtype=float, copy=True)
    ry = np.array(pd.Series(y).rank().to_numpy(), dtype=float, copy=True)
    rx -= rx.mean(); ry -= ry.mean()
    den = math.sqrt((rx @ rx) * (ry @ ry))
    return float((rx @ ry) / den) if den > 0 else float("nan")


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


# =====================================================================================
#  DRIVER
# =====================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="2-case smoke (elcentro/northrdg x1)")
    ap.add_argument("--cases", type=int, default=0, help="limit to first N canonical cases (0=all)")
    ap.add_argument("--out", type=Path, default=RESULTS / "eesd_ohtori_control_results.json")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.quick:
        cases = [c for c in CANONICAL_CASES if (c[0], c[2]) in {("elcentro", 1.0), ("northrdg", 1.0)}]
    elif args.cases > 0:
        cases = CANONICAL_CASES[:args.cases]
    else:
        cases = CANONICAL_CASES

    plant = parse_struct3(OHTORI_DIR / "struct3.m")
    cfg = BuildConfig(mode="nonlinear")
    sess = OhtoriPlantSession(plant, cfg)

    print(f"Reduced design model: f_modal = {np.array2string(_RM.f_modal, precision=4)} Hz")
    print(f"Evaluating {len(cases)} canonical cases on the FULL nonlinear OpenSees plant.\n")

    # ---------- PASS 1: uncontrolled refs + alpha_eff measurement per case ----------
    print("PASS 1: uncontrolled reference + alpha_eff (tangent mode-1 frequency drop)...")
    unc = {}
    alpha = {}
    for (rec, var, I, tf, aw) in cases:
        r = sess.run(None, rec, var, I, tf, aw, measure_alpha=True)
        key = (rec, I)
        unc[key] = r
        alpha[key] = {k: r[k] for k in ("f1_elastic", "f1_tangent_min", "alpha_eff_min",
                                        "alpha_eff_p5", "alpha_eff_driftweighted", "omega_s_predicted_hz")}
        print(f"  {CASE_LABEL[key]:16s}: unc d_max={r['d_max']*100:6.3f}%DR  "
              f"alpha_eff(min)={r['alpha_eff_min']:.4f} sqrt={math.sqrt(r['alpha_eff_min']):.4f}  "
              f"f1_tan_min={r['f1_tangent_min']:.4f}Hz  unstable={r['unstable']}")

    # representative alpha_eff for the softened DESIGN: use the median over yielding cases (honest single
    # design point, like the toy used a single alpha=0.02). Also report the per-case-design sensitivity.
    a_mins = np.array([alpha[(rec, I)]["alpha_eff_min"] for (rec, _v, I, _tf, _aw) in cases])
    alpha_design = float(np.median(a_mins))
    print(f"\n  alpha_eff per-case min: median={alpha_design:.4f}  range=[{a_mins.min():.4f},{a_mins.max():.4f}]")
    print(f"  -> softened DESIGN frequency omega_s = omega_1*sqrt({alpha_design:.4f}) "
          f"= {_RM.f_modal[0]*math.sqrt(alpha_design):.4f} Hz (toy used alpha=0.02 -> "
          f"{_RM.f_modal[0]*math.sqrt(ALPHA_TOY):.4f} Hz)\n")

    # ---------- tuned elastic LQR baseline (grid on the reduced model, scored on full plant) ----------
    print("PASS 2: tuning elastic LQR on the full plant (grid)...")
    best_lqr = (None, np.inf, None)
    lqr_grid = []
    for q_scale in (1e8, 1e10, 1e12):
        for r_value in (1e-4, 1e-6):
            K = design_lqr_on_K(_RM.K0, q_scale, r_value)
            js = []
            for (rec, var, I, tf, aw) in cases:
                r = sess.run(lambda x, K=K: -K @ x, rec, var, I, tf, aw)
                js.append(j_gate(r, unc[(rec, I)]))
            mj = float(np.mean(js))
            lqr_grid.append({"q_scale": q_scale, "r_value": r_value, "mean_J": mj})
            if mj < best_lqr[1]:
                best_lqr = (K, mj, {"q_scale": q_scale, "r_value": r_value})
            print(f"   q={q_scale:.0e} r={r_value:.0e}: mean J={mj:.4f}")
    K_lqr, lqr_mean, lqr_best = best_lqr
    q_best, r_best = lqr_best["q_scale"], lqr_best["r_value"]
    print(f"  tuned elastic LQR: q={q_best:.0e} r={r_best:.0e}  mean J={lqr_mean:.4f}\n")

    # softened-design LQR (single design point at alpha_design) + iso-norm variant
    K_soft = design_lqr_on_K(_RM.K_softened(alpha_design), q_best, r_best)
    n_lqr = float(np.linalg.norm(K_lqr)); n_soft = float(np.linalg.norm(K_soft))
    K_iso = K_soft * (n_lqr / n_soft)              # softened direction, elastic norm (iso-gain control)

    # best plain-DETUNED elastic LQR (honesty control: gain magnitude, not frequency)
    detuned_best = (None, np.inf, None)
    for qd in (q_best, q_best * 1e-2, q_best * 1e-4):
        for rd in (r_best, r_best * 1e2, r_best * 1e4):
            Kd = design_lqr_on_K(_RM.K0, qd, rd)
            js = [j_gate(sess.run(lambda x, Kd=Kd: -Kd @ x, rec, var, I, tf, aw), unc[(rec, I)])
                  for (rec, var, I, tf, aw) in cases]
            mj = float(np.mean(js))
            if mj < detuned_best[1]:
                detuned_best = (Kd, mj, (qd, rd))
    Kd_best, detuned_mean, detuned_qr = detuned_best

    # ---------- ANALYSIS 1 + 5: LQR family (elastic / softened / iso / detuned) per case ----------
    print("ANALYSIS 1 & 5: softening-aware LQR vs clipped-elastic LQR + iso-gain ablation...")
    lqr_rows = []
    for (rec, var, I, tf, aw) in cases:
        key = (rec, I)
        row = {"case": CASE_LABEL[key], "record": rec, "intensity": I,
               "alpha_eff_min": alpha[key]["alpha_eff_min"]}
        for label, K in (("LQR_elastic", K_lqr), ("LQR_softened", K_soft),
                         ("LQR_iso_softdir_elasticnorm", K_iso), ("LQR_detuned_elastic", Kd_best)):
            r = sess.run(lambda x, K=K: -K @ x, rec, var, I, tf, aw)
            row[label + "_J"] = j_gate(r, unc[key])
            row[label + "_dmax_pct"] = r["d_max"] * 100 if not r["unstable"] else 1e9
        lqr_rows.append(row)
        print(f"  {CASE_LABEL[key]:16s}: elastic J={row['LQR_elastic_J']:.4f} "
              f"softened J={row['LQR_softened_J']:.4f} iso J={row['LQR_iso_softdir_elasticnorm_J']:.4f} "
              f"detuned J={row['LQR_detuned_elastic_J']:.4f}")

    def meanJ(rows, key):
        return float(np.mean([r[key] for r in rows]))
    lqr_summary = {k: meanJ(lqr_rows, k + "_J") for k in
                   ("LQR_elastic", "LQR_softened", "LQR_iso_softdir_elasticnorm", "LQR_detuned_elastic")}
    soft_margin = lqr_summary["LQR_elastic"] - lqr_summary["LQR_softened"]
    iso_margin = lqr_summary["LQR_elastic"] - lqr_summary["LQR_iso_softdir_elasticnorm"]
    n_soft_better = sum(1 for r in lqr_rows if r["LQR_elastic_J"] - r["LQR_softened_J"] > 1e-6)
    n_iso_better = sum(1 for r in lqr_rows if r["LQR_elastic_J"] - r["LQR_iso_softdir_elasticnorm_J"] > 1e-6)
    print(f"  mean J: elastic={lqr_summary['LQR_elastic']:.4f} softened={lqr_summary['LQR_softened']:.4f} "
          f"(margin {soft_margin:+.4f}, {n_soft_better}/{len(lqr_rows)} better)  "
          f"iso={lqr_summary['LQR_iso_softdir_elasticnorm']:.4f} (margin {iso_margin:+.4f})  "
          f"detuned={lqr_summary['LQR_detuned_elastic']:.4f}\n")

    # ---------- ANALYSIS 3 + 4: 2x2 factorial MPC + regime map (per case) ----------
    print("ANALYSIS 3 & 4: 2x2 factorial MPC (tuning x saturation) + regime map...")
    CELLS = [("elastic", "clip"), ("softened", "clip"), ("elastic", "inqp"), ("softened", "inqp")]
    cell_rows = []
    regime_rows = []
    for (rec, var, I, tf, aw) in cases:
        key = (rec, I)
        a_eff = alpha_design  # softened MPC uses the same single design alpha as the LQR
        dt = sess.rec_dt if hasattr(sess, "rec_dt") else 0.02
        # need dt: use the record dt (load once)
        _t, _a, dt = load_record(rec, var, I)
        Jc = {}
        for (tuning, handling) in CELLS:
            ctrl = FactorialMPC(dt, tuning, handling, a_eff)
            r = sess.run(ctrl, rec, var, I, tf, aw)
            j = j_gate(r, unc[key])
            Jc[(tuning, handling)] = j
            cell_rows.append({"case": CASE_LABEL[key], "record": rec, "intensity": I,
                              "tuning": tuning, "handling": handling, "J": j,
                              "dmax_pct": r["d_max"] * 100 if not r["unstable"] else 1e9,
                              "unstable": r["unstable"]})
        # regime: dJ = J_inqp(elastic) - J_clip(elastic); E_K from alpha_eff (drift-weighted mismatch)
        dJ = Jc[("elastic", "inqp")] - Jc[("elastic", "clip")]
        # E_K analog on the real plant: relative mode-1 tangent-stiffness mismatch = 1 - alpha_eff_driftweighted
        E_K = 1.0 - alpha[key]["alpha_eff_driftweighted"]
        regime_rows.append({"case": CASE_LABEL[key], "record": rec, "intensity": I,
                            "J_clip": Jc[("elastic", "clip")], "J_inqp": Jc[("elastic", "inqp")],
                            "dJ_inqp_minus_clip": dJ, "inqp_wins": bool(dJ < 0),
                            "E_K": E_K, "alpha_eff_min": alpha[key]["alpha_eff_min"]})
        dec = decompose_2x2(Jc)
        print(f"  {CASE_LABEL[key]:16s}: A0B0={dec['J00_elastic_clip']:.3f} A1B0={dec['J10_softened_clip']:.3f} "
              f"A0B1={dec['J01_elastic_inqp']:.3f} A1B1={dec['J11_softened_inqp']:.3f} "
              f"best={dec['best_cell']} dJ={dJ:+.3f}")

    # aggregate 2x2 over all cases (mean cell J)
    def cell_meanJ(rows):
        out = {}
        for (t, h) in CELLS:
            vals = [r["J"] for r in rows if r["tuning"] == t and r["handling"] == h]
            out[(t, h)] = float(np.mean(vals))
        return out
    overall_2x2 = decompose_2x2(cell_meanJ(cell_rows))
    # controllable subset (max cell J < CONTROLLABLE_J per case)
    ctrl_cases = []
    for (rec, var, I, tf, aw) in cases:
        cj = [r["J"] for r in cell_rows if r["record"] == rec and r["intensity"] == I]
        if cj and max(cj) < CONTROLLABLE_J:
            ctrl_cases.append((rec, I))
    ctrl_cellrows = [r for r in cell_rows if (r["record"], r["intensity"]) in ctrl_cases]
    overall_2x2_ctrl = decompose_2x2(cell_meanJ(ctrl_cellrows)) if ctrl_cellrows else None

    # regime stats
    E_K_arr = np.array([r["E_K"] for r in regime_rows])
    dJ_arr = np.array([r["dJ_inqp_minus_clip"] for r in regime_rows])
    rho_EK = spearman_rho(E_K_arr, dJ_arr)
    cx_EK = crossover(E_K_arr, dJ_arr)
    rho_alpha = spearman_rho([r["alpha_eff_min"] for r in regime_rows], dJ_arr)

    # ---------- ANALYSIS 2: 1/sqrt(alpha) mechanism check ----------
    # On the real plant we MEASURE omega_1^tangent_min directly; the "prediction" omega_1*sqrt(alpha_eff)
    # is exact by construction of alpha_eff (alpha_eff := (f_tan/f0)^2). The HONEST test is whether this
    # measured softened frequency is (a) far from elastic (mechanism active) and (b) near the toy's
    # 0.141*f1 (toy replication) or instead mild. We report the spread + the contrast with the toy.
    f1_0 = _RM.f_modal[0]
    mech_rows = []
    for (rec, var, I, tf, aw) in cases:
        key = (rec, I)
        a = alpha[key]
        mech_rows.append({
            "case": CASE_LABEL[key], "intensity": I,
            "f1_elastic_hz": a["f1_elastic"], "f1_tangent_min_hz": a["f1_tangent_min"],
            "alpha_eff_min": a["alpha_eff_min"],
            "sqrt_alpha_eff_min": math.sqrt(max(a["alpha_eff_min"], 0.0)),
            "omega_s_over_omega1_measured": a["f1_tangent_min"] / a["f1_elastic"],
            "toy_sqrt_alpha": math.sqrt(ALPHA_TOY),
        })

    elapsed = time.time() - t0

    # ---------- VERDICTS (honest replication assessment) ----------
    # 1: softening-aware LQR helps if margin>=2% of elastic mean AND majority of cases better
    soft_helps = bool(soft_margin >= 0.02 * lqr_summary["LQR_elastic"] and n_soft_better > len(lqr_rows) / 2)
    soft_beats_detuned = bool(lqr_summary["LQR_softened"] < detuned_mean - 1e-6)
    iso_survives = bool(iso_margin >= 1e-4 and n_iso_better > len(lqr_rows) / 2)
    # 2: 1/sqrt(alpha) "replicates the toy magnitude" only if median sqrt(alpha_eff) ~ toy 0.141 (within 2x)
    median_sqrt_a = float(np.median([m["sqrt_alpha_eff_min"] for m in mech_rows]))
    mech_matches_toy_magnitude = bool(0.5 * math.sqrt(ALPHA_TOY) <= median_sqrt_a <= 2.0 * math.sqrt(ALPHA_TOY))
    mech_active = bool(median_sqrt_a < 0.85)   # frequency genuinely drops (softening is real)
    # 3: saturation dominates if ss_B_frac is the largest share (toy: ~equal three-way)
    sb = overall_2x2["ss_B_frac"]
    # 4: regime boundary holds if purity high
    regime_holds = bool(np.isfinite(cx_EK["separation_purity"]) and cx_EK["separation_purity"] >= 0.8
                        and cx_EK["n_inqp_wins"] >= 1 and cx_EK["n_inqp_loses"] >= 1)

    out = {
        "title": "EESD softening-aware control ported to the validated Ohtori 3-story nonlinear benchmark",
        "central_question": "Does the toy-3-DOF EESD story replicate on the real OpenSees benchmark?",
        "architecture": ("Controllers DESIGNED on the reduced 3-DOF condensed lateral M/K (real "
                         "stiff-base distribution); EVALUATED closed-loop on the FULL nonlinear OpenSees "
                         "Ohtori plant (openseespy 3.8.0, modal-validated 0.13%). Actuators at floor "
                         "master nodes 6/11/16 horizontal DOF; state = reduced floor (disp,vel)."),
        "honesty_scope": ("Validated BILINEAR approximation (modal + drift/J1-J6); NOT damage indices "
                          "J7-J10, NOT the exact trilinear MEX. Claims kept to drift/accel response."),
        "n_cases": len(cases), "cases": [CASE_LABEL[(c[0], c[2])] for c in cases],
        "U_MAX_per_floor_N": U_MAX, "N_horizon": N_HORIZON, "r_scale": R_SCALE,
        "reduced_model_f_modal_hz": _RM.f_modal.tolist(),
        "elapsed_s": elapsed,

        "alpha_eff_measurement": {
            "method": ("alpha_eff = (omega_1^tangent_min / omega_1^elastic)^2 from ops.eigen on the LIVE "
                       "tangent stiffness during the uncontrolled time-history (mode-1 frequency drop at "
                       "peak yielding). Also drift-energy-weighted and 5th-percentile variants."),
            "toy_alpha_explicit": ALPHA_TOY,
            "toy_sqrt_alpha": math.sqrt(ALPHA_TOY),
            "alpha_design_median": alpha_design,
            "alpha_eff_min_range": [float(a_mins.min()), float(a_mins.max())],
            "per_case": {CASE_LABEL[k]: alpha[k] for k in alpha},
        },
        "analysis1_softening_aware_lqr": {
            "tuned_elastic_lqr_q_r": lqr_best, "lqr_grid": lqr_grid,
            "mean_J": lqr_summary,
            "softened_margin_vs_elastic": soft_margin,
            "n_softened_better": n_soft_better, "n_cases": len(lqr_rows),
            "softened_beats_detuned_elastic": soft_beats_detuned,
            "detuned_elastic_best_mean_J": detuned_mean, "detuned_qr": list(detuned_qr),
            "per_case": lqr_rows,
            "VERDICT_softening_helps": soft_helps,
        },
        "analysis2_inv_sqrt_alpha_mechanism": {
            "f1_elastic_hz": f1_0,
            "median_sqrt_alpha_eff": median_sqrt_a,
            "toy_sqrt_alpha": math.sqrt(ALPHA_TOY),
            "mechanism_frequency_drops": mech_active,
            "matches_toy_magnitude_within_2x": mech_matches_toy_magnitude,
            "per_case": mech_rows,
        },
        "analysis3_2x2_factorial": {
            "cells": ["A0B0_elastic_clip", "A1B0_softened_clip", "A0B1_elastic_inqp", "A1B1_softened_inqp"],
            "overall_all_cases": overall_2x2,
            "overall_controllable": overall_2x2_ctrl,
            "controllable_cases": [CASE_LABEL[k] for k in ctrl_cases],
            "saturation_is_largest_share": bool(sb >= overall_2x2["ss_A_frac"] and sb >= overall_2x2["ss_AB_frac"]),
            "per_case_cells": cell_rows,
        },
        "analysis4_regime_map": {
            "index": "E_K = 1 - alpha_eff_driftweighted (tangent-stiffness mismatch)",
            "spearman_rho_EK_vs_dJ": rho_EK,
            "spearman_rho_alpha_min_vs_dJ": rho_alpha,
            "crossover": cx_EK,
            "per_case": regime_rows,
            "VERDICT_regime_boundary_holds": regime_holds,
        },
        "analysis5_iso_gain": {
            "gain_norms": {"K_elastic_fro": n_lqr, "K_softened_fro": n_soft,
                           "ratio_soft_over_elastic": n_soft / n_lqr},
            "iso_margin_vs_elastic": iso_margin,
            "n_iso_better": n_iso_better,
            "VERDICT_frequency_benefit_survives_isonorm": iso_survives,
        },
    }

    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # ---------- console verdict ----------
    print("\n" + "=" * 92)
    print("REPLICATION VERDICT (real Ohtori plant vs toy 3-DOF)")
    print("=" * 92)
    print(f"  alpha_eff(min) median = {alpha_design:.4f}  (TOY alpha = {ALPHA_TOY})  "
          f"sqrt: real {median_sqrt_a:.3f} vs toy {math.sqrt(ALPHA_TOY):.3f}")
    print(f"  [A1] softening-aware LQR helps: {soft_helps}  "
          f"(elastic {lqr_summary['LQR_elastic']:.4f} -> softened {lqr_summary['LQR_softened']:.4f}, "
          f"margin {soft_margin:+.4f}, {n_soft_better}/{len(lqr_rows)}; beats detuned: {soft_beats_detuned})")
    print(f"  [A2] 1/sqrt(alpha) mechanism active (freq drops): {mech_active}; "
          f"matches toy magnitude (within 2x): {mech_matches_toy_magnitude}")
    print(f"  [A3] 2x2 ANOVA shares: freq {overall_2x2['ss_A_frac']*100:.0f}% / "
          f"sat {overall_2x2['ss_B_frac']*100:.0f}% / interaction {overall_2x2['ss_AB_frac']*100:.0f}%  "
          f"(saturation largest: {out['analysis3_2x2_factorial']['saturation_is_largest_share']})")
    print(f"  [A4] regime boundary holds (purity>=0.8): {regime_holds}  "
          f"(purity {cx_EK['separation_purity']}, {cx_EK['n_inqp_wins']} win / {cx_EK['n_inqp_loses']} lose, "
          f"rho_EK {rho_EK:+.2f})")
    print(f"  [A5] iso-gain (frequency benefit survives matched ||K||): {iso_survives}  "
          f"(margin {iso_margin:+.4f}, {n_iso_better}/{len(lqr_rows)})")
    print(f"\nwrote {args.out}")
    print(f"elapsed {elapsed:.1f}s")


if __name__ == "__main__":
    main()


