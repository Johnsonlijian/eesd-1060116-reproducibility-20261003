"""Loader for the FEMA P695 (ATC-63) far-field ground-motion set, staged for the EESD
record-robustness sweep.

Interface mirrors src/eesd_struct3_plant.load_record so the existing 1/sqrt(alpha) / margin-trend
harness can consume these records the same way it consumes the 4 Ohtori benchmark records:

    t, ag = load_record(record_id)          # t [s], ag [m/s^2]   (matches eesd_struct3_plant)

UNITS NOTE -- this is the load-bearing detail:
  * The Ohtori benchmark .mat records store acceleration in m/s^2; the EESD plant
    (eesd_struct3_plant.Struct3ReducedPlant.step) takes ground accel `ag` in m/s^2.
  * The FEMA P695 2c "Sorted" single-column files store acceleration in g.
  * Therefore load_record() multiplies the raw g series by g=9.80665 and returns m/s^2,
    so it is a DROP-IN substitute for eesd_struct3_plant.load_record. Use load_record_g()
    if you want the raw g series instead.

INTENSITY / SCALING (how the harness already scales):
  The harness multiplies the returned `ag` by a scalar `intensity`/`scale` (see
  eesd_struct3_plant.__main__ and eesd_margin_law.get_record). These are UNSCALED records.
  To put the suite at a common intensity per the ATC-63 Appendix A method, multiply each
  record by its `atc63_norm_factor` (from the manifest / summary Table A-4D); both horizontal
  components of a pair share the same factor. After normalization, apply the global ATC-63
  spectral scale factor (a single number for the whole set) for a target Sa -- that global
  factor is NOT in this summary file and is chosen by the analyst for the target intensity.

Records are identified by their 6-digit ATC-63 EQ ID (e.g. "120111" = Northridge Beverly Hills,
component 1). Convenience lookups by PEER RSN + component are also provided.
"""
from __future__ import annotations
import os

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(os.environ.get("EESD_PROJECT_ROOT", PACKAGE_ROOT))
FF_DIR = Path(os.environ.get("EESD_FEMA_DIR", PROJECT_ROOT / "data" / "fema_p695_farfield"))
MANIFEST = FF_DIR / "manifest.csv"

GRAVITY = 9.80665  # m/s^2 per g (PEER/ATC-63 records are in g)


@lru_cache(maxsize=1)
def manifest() -> pd.DataFrame:
    """Return the staged-record manifest (one row per component)."""
    df = pd.read_csv(MANIFEST, dtype={"record_id": str, "record_pair_id": str})
    return df


def _row(record_id: str) -> pd.Series:
    df = manifest()
    hit = df[df["record_id"] == str(record_id)]
    if hit.empty:
        raise KeyError(f"record_id {record_id!r} not in manifest "
                       f"(valid examples: {list(df['record_id'].head(3))} ...)")
    return hit.iloc[0]


@lru_cache(maxsize=64)
def load_record_g(record_id: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (time[s], accel[g], dt[s]) for a FEMA P695 record component.

    Reads the 2c single-column sorted text file (one acceleration value in g per line) and
    builds the time vector from the dt recorded in the PEER .AT2 header (manifest column dt_s).
    """
    r = _row(record_id)
    dt = float(r["dt_s"])
    # sorted_file is a forward-slash relative path in the manifest; join part-by-part so it
    # resolves correctly regardless of platform / UNC drive mapping.
    path = FF_DIR.joinpath(*str(r["sorted_file"]).split("/"))
    ag_g = np.asarray([float(x) for x in path.read_text().split()], float)
    t = np.arange(len(ag_g), dtype=float) * dt
    return t, ag_g, dt


def load_record(record_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Drop-in for eesd_struct3_plant.load_record: return (time[s], ground_accel[m/s^2]).

    Acceleration is converted from g to m/s^2 so the EESD plant (which expects m/s^2) can step
    these records identically to the Ohtori benchmark records.
    """
    t, ag_g, _dt = load_record_g(record_id)
    return t, ag_g * GRAVITY


def record_dt(record_id: str) -> float:
    """dt [s] for a record (matches float(np.median(np.diff(t))) the harness computes)."""
    return float(_row(record_id)["dt_s"])


def atc63_norm_factor(record_id: str) -> float:
    """ATC-63 (Appendix A) normalization factor for the record's pair (Table A-4D).

    Multiply a record by this to normalize its peak ground velocity to the set median before
    applying any global spectral scale. Both components of a pair share the same factor.
    """
    return float(_row(record_id)["atc63_norm_factor"])


def all_record_ids() -> list[str]:
    """All 44 component record ids, ordered as in the manifest."""
    return list(manifest()["record_id"])


def record_pairs() -> list[tuple[str, str]]:
    """The 22 (component-1, component-2) record-id pairs."""
    df = manifest()
    pairs = []
    for pid, grp in df.groupby("record_pair_id", sort=True):
        ids = sorted(grp["record_id"])
        pairs.append((ids[0], ids[1]))
    return pairs


def find_by_rsn(rsn: int, component: int = 1) -> str:
    """Return the record_id for a PEER RSN + horizontal component (1 or 2)."""
    df = manifest()
    hit = df[(df["peer_rsn"] == int(rsn)) & (df["component"] == int(component))]
    if hit.empty:
        raise KeyError(f"No record for RSN {rsn} component {component}")
    return str(hit.iloc[0]["record_id"])


if __name__ == "__main__":
    df = manifest()
    print(f"FEMA P695 far-field set: {len(df)} components, "
          f"{df['record_pair_id'].nunique()} pairs, {df['earthquake'].nunique()} events")
    print(f"dt values [s]: {sorted(df['dt_s'].unique())}")
    print(f"duration [s]:  {df['duration_s'].min():.2f} .. {df['duration_s'].max():.2f}")
    print(f"PGA [g]:       {df['pga_g'].min():.3f} .. {df['pga_g'].max():.3f}")
    # smoke: load the three named sanity records via RSN
    for rsn, label in [(953, "Northridge Beverly Hills"), (767, "Loma Prieta Gilroy#3"),
                       (1244, "Chi-Chi CHY101")]:
        rid = find_by_rsn(rsn, 1)
        t, ag_ms2 = load_record(rid)
        _, ag_g, dt = load_record_g(rid)
        print(f"  RSN {rsn:5d} {label:26s} id={rid} dt={dt} npts={len(t)} "
              f"dur={t[-1]:.2f}s PGA={np.max(np.abs(ag_g)):.3f}g "
              f"({np.max(np.abs(ag_ms2)):.3f} m/s2) norm={atc63_norm_factor(rid):.4f}")


