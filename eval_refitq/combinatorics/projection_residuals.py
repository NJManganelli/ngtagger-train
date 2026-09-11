"""Does the layer-C projection window have to widen with d0?  Phi vs z.

THE QUESTION THIS SETTLES. The seeder projects a cluster pair to a third layer
and searches for a matching cluster. It can search on azimuth or on z, and the
choice decides whether displaced seeding is affordable:

  phi at layer C carries the pair's curvature, and a pair-derived curvature
  ASSUMES d0 = 0. A real d0 biases it by d0 * (1/rA - 1/rB) / (c * dr), which is
  10.1 per cm for IT L1L2. So the phi window must open by ~NSIG * d0/r_inner and
  the candidate-triplet count grows LINEARLY in d0.

  z at layer C is z0 + r * cot(theta). A TRANSVERSE impact parameter perturbs
  neither term to first order, so the z window should be d0-INDEPENDENT.

If that holds, the seeder should search on z and use phi only to confirm, which
is the opposite of what tracklet_topology_cost.py currently does. This script
measures it rather than assuming it, because the whole reordering rests on it.

Method: take the CORRECT cluster triple -- three clusters of one TrackingParticle
in layers A, B, C -- derive (kappa, phi0, cot, z0) from the A-B cluster pair
alone, project to C's own radius, and histogram the residual against the TP's
own d0. No combinatorics: this is a resolution measurement, not a cost model.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import awkward as ak
import numpy as np
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tracklet_topology_cost as M  # noqa: E402

IT_TABLE = "L1TSmartPixelsCluster"
C_BEND = 0.29979246 * 3.8 / 2.0 / 100.0
TP_KEY_SHIFT = 20
COLS = ["layer", "globalR", "globalZ", "globalPhi", "tpIdx", "tpPt",
        "tpVx", "tpVy", "tpPhi"]
D0_EDGES_CM = [0.0, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 1e9]


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def robust_sigma(x):
    if len(x) < 8:
        return float("nan")
    q16, q84 = np.percentile(x, [15.865, 84.135])
    return float(0.5 * (q84 - q16))


def load(path, nev):
    """One or many input files; see tracklet_topology_cost.expand_inputs."""
    return M.load_flat(path, IT_TABLE, COLS, nev)[:2]


def first_cluster_per_tp(D, layer, sel):
    """One cluster index per (event, TP) in `layer`; keys sorted."""
    m = sel & (D["layer"] == layer) & (D["tpIdx"] >= 0)
    idx = np.flatnonzero(m)
    k = D["event"][idx].astype(np.int64) * (1 << TP_KEY_SHIFT) + D["tpIdx"][idx]
    o = np.argsort(k, kind="stable")
    k, idx = k[o], idx[o]
    first = np.r_[True, k[1:] != k[:-1]]
    return k[first], idx[first]


def correct_triples(D, la, lb, lc, sel):
    """Cluster indices (A, B, C) of the ONE correct triple per TP."""
    ka, ia = first_cluster_per_tp(D, la, sel)
    kb, ib = first_cluster_per_tp(D, lb, sel)
    kc, ic = first_cluster_per_tp(D, lc, sel)
    common = np.intersect1d(np.intersect1d(ka, kb), kc)
    if not len(common):
        return (np.empty(0, np.int64),) * 3
    return (ia[np.searchsorted(ka, common)],
            ib[np.searchsorted(kb, common)],
            ic[np.searchsorted(kc, common)])


def residuals(D, gA, gB, gC):
    """Project the A-B cluster pair to C's own radius; return (dphi, dz)."""
    drp = D["globalR"][gB] - D["globalR"][gA]
    ok = np.abs(drp) > 0.5
    drp = np.where(ok, drp, 1e9)
    kap = wrap(D["globalPhi"][gA] - D["globalPhi"][gB]) / (C_BEND * drp)
    cot = (D["globalZ"][gB] - D["globalZ"][gA]) / drp
    z0p = D["globalZ"][gA] - D["globalR"][gA] * cot
    phi0 = wrap(D["globalPhi"][gA] + C_BEND * D["globalR"][gA] * kap)
    rC = D["globalR"][gC]
    dphi = wrap(D["globalPhi"][gC] - wrap(phi0 - C_BEND * rC * kap))
    dz = D["globalZ"][gC] - (z0p + rC * cot)
    return dphi, dz, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-n", "--nev", type=int, default=None)
    ap.add_argument("--ptmin", type=float, default=2.0,
                    help="isolate the d0 effect from multiple scattering")
    a = ap.parse_args()

    D, nev = load(a.input, a.nev)
    print(f"{len(D['layer'])/nev:.0f} clusters/event over {nev} events, "
          f"pT > {a.ptmin} GeV\n")
    sel = D["tpPt"] >= a.ptmin
    d0_all = np.abs(-D["tpVx"] * np.sin(D["tpPhi"]) + D["tpVy"] * np.cos(D["tpPhi"]))

    for (la, lb, lc) in ((1, 2, 3), (2, 3, 4)):
        gA, gB, gC = correct_triples(D, la, lb, lc, sel)
        if not len(gA):
            print(f"L{la}L{lb}->L{lc}: no correct triples"); continue
        dphi, dz, ok = residuals(D, gA, gB, gC)
        d0 = d0_all[gA]
        print(f"=== L{la}L{lb} -> L{lc}   ({ok.sum()} correct triples) ===")
        print(f"{'TP |d0|':>16}{'n':>8}{'sigma(dphi)':>14}{'sigma(dz)':>12}"
              f"{'phi vs core':>13}{'z vs core':>11}")
        print(f"{'':>16}{'':>8}{'[mrad]':>14}{'[um]':>12}{'':>13}{'':>11}")
        ref_p = ref_z = None
        for lo, hi in zip(D0_EDGES_CM[:-1], D0_EDGES_CM[1:]):
            m = ok & (d0 >= lo) & (d0 < hi)
            if m.sum() < 8:
                continue
            sp = robust_sigma(dphi[m]) * 1e3
            sz = robust_sigma(dz[m]) * 1e4
            if ref_p is None:
                ref_p, ref_z = sp, sz
            hi_s = "inf" if hi > 1e8 else f"{hi*1e4:.0f}"
            print(f"{lo*1e4:>9.0f}-{hi_s:<6}{m.sum():>8}{sp:>14.3f}{sz:>12.1f}"
                  f"{sp/ref_p:>12.1f}x{sz/ref_z:>10.1f}x")
        print()


if __name__ == "__main__":
    main()
