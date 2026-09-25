"""TrackingParticle truth helix at the POCA to the beamline, as CMSSW defines it.

The nano truth columns disagree about where they are evaluated:

  * tp_d0 / tp_z0 (L1TrackTruthTableProducer, on the L1TTrack tables) and d0 / z0
    (L1TrackingParticleTableProducer, the L1TTP table) are at the POCA to the beamline.
  * tp_phi / phi are the momentum azimuth at the PRODUCTION VERTEX.

A helix built from (phi, d0, z0) is therefore inconsistent: for a TP whose vertex
sits away from its POCA (secondaries, b/c decays) the production azimuth differs
from phi0 by the angle the track turns between the two points: negligible for
prompt TPs, but 0.6 mrad at the 95th percentile of the perfect-track TPs in the
ttbar PU200 calibration sample and tens of mrad for Geant4 secondaries. Nanos made from 2026-09-24 carry the POCA
azimuth as tp_phi0 / phi0; for older nanos :func:`poca_phi0` recomputes it with the
producers' own formula (agreement with the stored column: < 3e-7 rad).
"""

from __future__ import annotations

import numpy as np

# c [m/ns scaled] * B(0,0,0) / 100: pT [GeV] = KPT / |rInv [1/cm]|. The producers
# take B at the origin from the field map (3.8112 T for the Phase-2 3.8 T map);
# the L1 tracks' own pT * |rInv| gives the same value (0.011426).
KPT_CMSSW = 0.29979246 * 3.8112 / 100.0


def poca_phi0(phi, pt, charge, vx, vy, kpt=KPT_CMSSW):
    """Azimuth at the POCA to the beamline of the helix through (vx, vy) with
    production azimuth `phi` (L1TrackTruthTableProducer's formula). Entries with
    pt <= 0 or charge == 0 (unmatched sentinels, neutrals) return `phi` unchanged."""
    phi, pt, charge, vx, vy = (np.asarray(a, dtype=np.float64) for a in (phi, pt, charge, vx, vy))
    ok = (pt > 0) & (charge != 0)
    r2 = np.where(ok, charge * kpt / np.where(ok, pt, 1.0) / 2.0, 1.0)
    x0p = -vx - np.sin(phi) / (2.0 * r2)
    y0p = -vy + np.cos(phi) / (2.0 * r2)
    return np.where(ok, np.arctan2(-r2 * x0p, r2 * y0p), phi)


def tp_phi0(cols, prefix="tp_", kpt=KPT_CMSSW):
    """The truth phi0 for a table of TP columns (a mapping of arrays): the stored
    `<prefix>phi0` when the nano has it, else recomputed from `<prefix>phi, pt,
    charge, vx, vy`. prefix "tp_" for the L1TTrack truth columns, "" for L1TTP."""
    if f"{prefix}phi0" in cols:
        return np.asarray(cols[f"{prefix}phi0"], dtype=np.float64)
    return poca_phi0(*(cols[f"{prefix}{c}"] for c in ("phi", "pt", "charge", "vx", "vy")), kpt=kpt)
