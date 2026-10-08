"""Truth helix from POCA parameters (definition sheet v1, items `frame.d0`, `frame.phi0`).

Conventions (CMSSW L1 / TTTrack, the beam line = the z axis):
  * the POCA to the beam line is (x0, y0) = (d0 sin phi0, -d0 cos phi0), i.e.
    d0 = x0 sin phi0 - y0 cos phi0 = -dxy (offline convention);
  * phi0 = momentum azimuth AT THE POCA (tp_phi0 / L1TTP_phi0, or truth_helix.poca_phi0;
    tp_phi / L1TTP_phi are at the production vertex);
  * transverse radius R = pT / KPT [cm] with KPT = 0.29979246 * 3.8112 / 100 (B at the origin);
    in B along +z the azimuth of a particle of charge q turns as dphi/ds = -q / R.
"""
from __future__ import annotations

import numpy as np

KPT_CMSSW = 0.29979246 * 3.8112 / 100.0


def poca_xy(d0, phi0):
    d0, phi0 = np.asarray(d0, np.float64), np.asarray(phi0, np.float64)
    return d0 * np.sin(phi0), -d0 * np.cos(phi0)


def at_radius(r, pt, charge, phi0, d0, z0=None, tanl=None, kpt=KPT_CMSSW, n_newton=8):
    """(x, y, z, s) where the helix leaving its POCA outward first reaches transverse
    radius r [cm]. s = transverse path length from the POCA. NaN where it never does."""
    r, pt, q = (np.asarray(a, np.float64) for a in (r, pt, charge))
    phi0 = np.asarray(phi0, np.float64)
    x0, y0 = poca_xy(d0, phi0)
    kappa = -q * kpt / np.where(pt > 0, pt, np.nan)          # dphi/ds
    s = np.sqrt(np.clip(r ** 2 - x0 ** 2 - y0 ** 2, 0.0, None))  # straight-line start
    for _ in range(n_newton):
        ph = phi0 + kappa * s
        with np.errstate(invalid="ignore", divide="ignore"):
            small = np.abs(kappa * s) < 1e-9
            x = x0 + np.where(small, s * np.cos(phi0), (np.sin(ph) - np.sin(phi0)) / kappa)
            y = y0 + np.where(small, s * np.sin(phi0), -(np.cos(ph) - np.cos(phi0)) / kappa)
            f = x * x + y * y - r * r
            fp = 2.0 * (x * np.cos(ph) + y * np.sin(ph))
            s = s - f / fp
    ph = phi0 + kappa * s
    with np.errstate(invalid="ignore", divide="ignore"):
        small = np.abs(kappa * s) < 1e-9
        x = x0 + np.where(small, s * np.cos(phi0), (np.sin(ph) - np.sin(phi0)) / kappa)
        y = y0 + np.where(small, s * np.sin(phi0), -(np.cos(ph) - np.cos(phi0)) / kappa)
    bad = ~np.isfinite(s) | (s < 0) | (np.abs(np.hypot(x, y) - r) > 1e-6)
    x, y = np.where(bad, np.nan, x), np.where(bad, np.nan, y)
    z = None if z0 is None else np.asarray(z0, np.float64) + s * np.asarray(tanl, np.float64)
    return x, y, z, np.where(bad, np.nan, s)


def wrap(dphi):
    return (np.asarray(dphi) + np.pi) % (2 * np.pi) - np.pi
