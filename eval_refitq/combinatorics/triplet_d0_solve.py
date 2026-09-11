"""Can three clusters determine (phi0, d0, kappa) exactly, and how well?

WHY THIS EXISTS. A cluster PAIR gives two azimuths at two radii. That is two
measurements for three unknowns, so the pair can only solve for curvature by
ASSUMING d0 = 0, and a real d0 then biases the curvature by
    kappa_bias = d0 * (1/rA - 1/rB) / (c * dr)
which is 10.34 per cm for IT L1L2. Covering d0 = 500 um therefore needs the
curvature gate opened by 0.517, against kappa_max = 0.500 at pT = 2 GeV -- so the
pT threshold is destroyed to buy heavy-flavour acceptance, and a seed labelled
2 GeV actually accepts 0.98 GeV.

A TRIPLET has three azimuths for three unknowns: zero remaining r-phi degrees of
freedom, d0 SOLVED rather than assumed, and no curvature widening needed at all.
This is the "exact three-point d0 solve". The model, to first order in d0/r and
kappa*r, is linear in its unknowns:

    phi(r) = phi0 + A/r + B*r ,      A = +d0 ,  B = -c*kappa

The sign of A is MEASURED, not asserted: a robust regression of the fitted A
on the TrackingParticle's own d0 gives slope +1.11, so d0 = +A. An earlier
revision wrote d0 = -A, which makes the residual -2*d0 and so grows linearly
with d0 -- it read as sigma(d0) degrading from 46 um to 3631 um across the d0
bins, i.e. as a resolution problem rather than a sign error. The core bin hid
it, because there d0 ~ 0 and the residual is just sigma(A) either way.

so it is one 3x3 solve per cluster triple, done in closed form here rather than
by np.linalg.solve so it vectorises over every triple at once.

THIS SCRIPT DOES NOT SEED ANYTHING. It takes the ONE correct cluster triple per
TrackingParticle and asks how well the solve recovers the TP's own d0 and
curvature -- i.e. whether the mechanism is worth building on. The signs of A and
B are verified against truth rather than asserted, because a sign convention
error here would look like a resolution problem.
"""
from __future__ import annotations
import argparse
import awkward as ak
import numpy as np
import uproot

IT_TABLE = "L1TSmartPixelsCluster"
C_BEND = 0.29979246 * 3.8 / 2.0 / 100.0
TP_KEY_SHIFT = 20
COLS = ["layer", "globalR", "globalZ", "globalPhi", "tpIdx", "tpPt",
        "tpVx", "tpVy", "tpPhi", "tpEta"]
D0_EDGES_CM = [0.0, 0.005, 0.01, 0.05, 0.1, 0.5, 1e9]


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def robust_sigma(x):
    if len(x) < 8:
        return float("nan")
    q16, q84 = np.percentile(x, [15.865, 84.135])
    return float(0.5 * (q84 - q16))


def load(path, nev):
    A = uproot.open(f"{path}:Events").arrays(
        [f"{IT_TABLE}_{c}" for c in COLS], entry_stop=nev)
    n = ak.to_numpy(ak.num(A[f"{IT_TABLE}_layer"]))
    D = {c: ak.to_numpy(ak.flatten(A[f"{IT_TABLE}_{c}"])) for c in COLS}
    D["event"] = np.repeat(np.arange(len(n)), n)
    return D, len(n)


def first_cluster_per_tp(D, layer, sel):
    m = sel & (D["layer"] == layer) & (D["tpIdx"] >= 0)
    idx = np.flatnonzero(m)
    k = D["event"][idx].astype(np.int64) * (1 << TP_KEY_SHIFT) + D["tpIdx"][idx]
    o = np.argsort(k, kind="stable")
    k, idx = k[o], idx[o]
    first = np.r_[True, k[1:] != k[:-1]]
    return k[first], idx[first]


def correct_triples(D, la, lb, lc, sel):
    ka, ia = first_cluster_per_tp(D, la, sel)
    kb, ib = first_cluster_per_tp(D, lb, sel)
    kc, ic = first_cluster_per_tp(D, lc, sel)
    common = np.intersect1d(np.intersect1d(ka, kb), kc)
    if not len(common):
        return (np.empty(0, np.int64),) * 3
    return (ia[np.searchsorted(ka, common)],
            ib[np.searchsorted(kb, common)],
            ic[np.searchsorted(kc, common)])


def solve_triplet(r1, p1, r2, p2, r3, p3):
    """Closed-form solve of phi_i = phi0 + A/r_i + B*r_i for every triple.

    Azimuths are differenced against the innermost cluster before solving, so the
    2x2 that remains is immune to the 2*pi wrap that a raw phi0 would carry.
    """
    u2, u3 = 1.0 / r2 - 1.0 / r1, 1.0 / r3 - 1.0 / r1
    v2, v3 = r2 - r1, r3 - r1
    d2, d3 = wrap(p2 - p1), wrap(p3 - p1)
    det = u2 * v3 - u3 * v2
    bad = np.abs(det) < 1e-12
    det = np.where(bad, 1.0, det)
    A = (d2 * v3 - d3 * v2) / det
    B = (u2 * d3 - u3 * d2) / det
    phi0 = wrap(p1 - A / r1 - B * r1)
    return phi0, A, B, ~bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-n", "--nev", type=int, default=None)
    ap.add_argument("--ptmin", type=float, default=2.0)
    a = ap.parse_args()

    D, nev = load(a.input, a.nev)
    print(f"{len(D['layer'])/nev:.0f} clusters/event over {nev} events, "
          f"pT > {a.ptmin} GeV\n")
    sel = D["tpPt"] >= a.ptmin
    d0_true_all = (-D["tpVx"] * np.sin(D["tpPhi"])
                   + D["tpVy"] * np.cos(D["tpPhi"]))

    for (la, lb, lc) in ((1, 2, 3), (2, 3, 4)):
        gA, gB, gC = correct_triples(D, la, lb, lc, sel)
        if not len(gA):
            print(f"L{la}L{lb}L{lc}: no correct triples"); continue
        r = [D["globalR"][g] for g in (gA, gB, gC)]
        p = [D["globalPhi"][g] for g in (gA, gB, gC)]
        phi0, A, B, ok = solve_triplet(r[0], p[0], r[1], p[1], r[2], p[2])
        d0_fit, kap_fit = A, -B / C_BEND
        d0_tp = d0_true_all[gA]
        # pair-derived kappa for comparison: assumes d0 = 0
        drp = r[1] - r[0]
        kap_pair = wrap(p[0] - p[1]) / (C_BEND * drp)
        kap_tp = 1.0 / D["tpPt"][gA]

        print(f"=== L{la}L{lb}L{lc}   ({ok.sum()} correct triples) ===")
        # SIGN AND SCALE CHECK FIRST, because a flipped sign reads as bad
        # resolution. By SLOPE, not correlation: the solve has a degenerate tail
        # (near-zero determinant) whose outliers drive Pearson r to ~0 even when
        # the bulk is recovered perfectly, which is how the sign error above
        # survived its own check. Both slopes should be ~1.
        bulk = ok & (np.abs(A) < 1.0) & (np.abs(B) < 0.1)
        sd = np.polyfit(d0_tp[bulk], d0_fit[bulk], 1)[0]
        sk = np.polyfit(kap_tp[bulk], np.abs(kap_fit[bulk]), 1)[0]
        print(f"  sign/scale (bulk {bulk.sum()} of {ok.sum()}): "
              f"slope d0_fit/d0_true = {sd:+.3f}, "
              f"slope |kappa_fit|/(1/pT) = {sk:+.3f}")
        print(f"  {'TP |d0|':>14}{'n':>7}{'sigma(d0) fit':>15}"
              f"{'sig(kappa) 3pt':>16}{'sig(kappa) pair':>17}")
        for lo, hi in zip(D0_EDGES_CM[:-1], D0_EDGES_CM[1:]):
            m = ok & (np.abs(d0_tp) >= lo) & (np.abs(d0_tp) < hi)
            if m.sum() < 8:
                continue
            s_d0 = robust_sigma(d0_fit[m] - d0_tp[m]) * 1e4
            s_k3 = robust_sigma(np.abs(kap_fit[m]) - kap_tp[m])
            s_kp = robust_sigma(np.abs(kap_pair[m]) - kap_tp[m])
            hs = "inf" if hi > 1e8 else f"{hi*1e4:.0f}"
            print(f"  {lo*1e4:>7.0f}-{hs:<6}{m.sum():>7}{s_d0:>13.0f}um"
                  f"{s_k3:>16.4f}{s_kp:>17.4f}")
        print()


if __name__ == "__main__":
    main()
