#!/usr/bin/env python
"""Where do findable prompt tracks die? A per-stage TRUTH accounting.

MOTIVATION. The prompt-core efficiency band (|d0| < 100 um, n = 4002 findable)
sits at 0.674 -- a third of core prompt tracks with clusters on all three layers
are not found, and d0 is irrelevant there by construction. That loss is ~10x
larger than the entire d0/heavy-flavour effect. Every attempt so far to name its
cause has been an inference and several have been wrong, so this script does not
infer: it follows the ONE correct pair for each findable track and records the
first stage that rejects it.

METHOD. For a findable TP (above pT floor, cluster on all three layers of the
configuration), take its OWN clusters on the seed pair's two layers and its own
cluster on the projection layer. Push that single correct combination through the
identical gate sequence the seeder uses and record where it first fails. A track
can also be lost by SURVIVING every gate and then losing the best-residual
comparison to a wrong cluster; that is counted separately as "lost_at_arbitration"
because it is a different failure -- the gates were fine, the choice was not.

Nothing here is a rate measurement. It is a census of survival for the correct
combination, which is what an efficiency loss actually is.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tracklet_topology_cost as M   # noqa: E402

T = "L1TSmartPixelsCluster"
C_BEND = 0.29979246 * 3.8 / 2.0 / 100.0
Z_LUMI, NSIG = 15.0, 3.0
SIG_PULL_INFLATE = 1.21
THETA_MS_MRAD = 1.36

STAGES = ["no_cluster_on_layer", "alpha_veto", "phi_window", "kappa_cut",
          "z0_luminous", "alpha_consistency", "z0_consistency",
          "projection_phi", "projection_z", "lost_at_arbitration", "FOUND"]


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def load(path, nev):
    """Clusters with their TP's L1TTP d0 at the POCA [cm] joined as tp_d0."""
    cols = ["layer", "globalR", "globalZ", "globalPhi", "globalClusterPhi",
            "globalClusterCotTheta", "sigGlobalClusterPhi", "sigGlobalClusterCotTheta",
            "sigY", "tpIdx", "tpPt"]
    return M.load_flat(path, T, cols, nev, tp=("tp_d0",))[:2]


def trace_one(D, ia, ib, ic, ptmin, d0_cm, rA, rB, dr):
    """Push ONE correct triple through the gates; return the first failing stage."""
    kmax = 1.0 / ptmin
    ms = (THETA_MS_MRAD * 1e-3 / ptmin) * np.sqrt(2.0)
    # single-cluster alpha veto
    for i in (ia, ib, ic):
        h = wrap(D["globalPhi"][i] - D["globalClusterPhi"][i])
        ka = np.sin(h) / (C_BEND * D["globalR"][i])
        sk = D["sigGlobalClusterPhi"][i] * abs(np.cos(h)) / (C_BEND * D["globalR"][i])
        if abs(ka) > kmax + NSIG * sk:
            return "alpha_veto"
    d0_allow = d0_cm / min(rA, rB)
    kap_slack = abs(1.0 / rA - 1.0 / rB) / (C_BEND * max(dr, 0.1)) * d0_cm
    half_w = C_BEND * dr * kmax + ms + 5e-4 + d0_allow
    if abs(wrap(D["globalPhi"][ia] - D["globalPhi"][ib])) > half_w:
        return "phi_window"
    drp = D["globalR"][ib] - D["globalR"][ia]
    if abs(drp) < 0.5:
        return "phi_window"
    kap = wrap(D["globalPhi"][ia] - D["globalPhi"][ib]) / (C_BEND * drp)
    cot = (D["globalZ"][ib] - D["globalZ"][ia]) / drp
    z0 = D["globalZ"][ia] - D["globalR"][ia] * cot
    if abs(kap) > kmax + kap_slack:
        return "kappa_cut"
    if abs(z0) > Z_LUMI:
        return "z0_luminous"
    for i in (ia, ib):
        h = wrap(D["globalPhi"][i] - D["globalClusterPhi"][i])
        ka = np.sin(h) / (C_BEND * D["globalR"][i])
        sk = D["sigGlobalClusterPhi"][i] * abs(np.cos(h)) / (C_BEND * D["globalR"][i])
        if abs(kap - ka) > NSIG * max(sk, 1e-9):
            return "alpha_consistency"
    sza = D["globalR"][ia] * D["sigGlobalClusterCotTheta"][ia] * SIG_PULL_INFLATE
    szb = D["globalR"][ib] * D["sigGlobalClusterCotTheta"][ib] * SIG_PULL_INFLATE
    z0a = D["globalZ"][ia] - D["globalR"][ia] * D["globalClusterCotTheta"][ia]
    z0b = D["globalZ"][ib] - D["globalR"][ib] * D["globalClusterCotTheta"][ib]
    if abs(z0a - z0b) > NSIG * np.hypot(sza, szb):
        return "z0_consistency"
    # PROJECTION: to the TARGET CLUSTER'S OWN radius, and with the MEASURED
    # per-cluster sigY. Using the layer-median radius here while comparing to the
    # cluster's actual z was a dz = (r_actual - r_median)*cot error of ~1 cm
    # against an ~800 um window, and it produced a spurious 52.7% loss at this
    # stage. sigY is measured at 12.5 um, not the 30 um previously assumed.
    phi0 = wrap(D["globalPhi"][ia] + C_BEND * D["globalR"][ia] * kap)
    rC = D["globalR"][ic]
    skap = np.sqrt(2.0) * 1e-3 / (C_BEND * max(dr, 0.1))
    sph = np.hypot(5e-4, C_BEND * rC * skap) + ms + d0_allow
    sct = np.hypot(max(D["sigY"][ia], 1e-6), max(D["sigY"][ib], 1e-6)) / max(dr, 0.1)
    szp = np.hypot(max(D["sigY"][ic], 1e-6), rC * sct) + rC * ms
    phiC = wrap(phi0 - C_BEND * rC * kap)
    zC = z0 + rC * cot
    if abs(wrap(D["globalPhi"][ic] - phiC)) > NSIG * sph:
        return "projection_phi"
    if abs(D["globalZ"][ic] - zC) > NSIG * szp:
        return "projection_z"
    return "FOUND", phiC, NSIG * sph, zC, NSIG * szp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-n", "--nev", type=int, default=40)
    ap.add_argument("--ptmin", type=float, default=2.0)
    ap.add_argument("--d0", type=float, default=0.0, help="prompt d0 allowance [cm]")
    ap.add_argument("--pair", default="1,2", help="seed layers, e.g. 1,2")
    ap.add_argument("--proj", type=int, default=3)
    ap.add_argument("--d0max", type=float, default=100e-4,
                    help="restrict to TPs below this |d0| [cm]; 0 = no cut")
    ap.add_argument("-o", "--out", default="seed_loss_accounting.json")
    a = ap.parse_args()
    la, lb = (int(x) for x in a.pair.split(","))
    lc = a.proj
    D, nev = load(a.input, a.nev)
    rA = np.median(D["globalR"][D["layer"] == la])
    rB = np.median(D["globalR"][D["layer"] == lb])
    rC = np.median(D["globalR"][D["layer"] == lc])
    dr = rB - rA
    tally = {s: 0 for s in STAGES}
    arb = 0
    for e in range(nev):
        m = D["event"] == e
        idx = np.flatnonzero(m)
        tp, ly, pt = D["tpIdx"][idx], D["layer"][idx], D["tpPt"][idx]
        d0 = np.abs(D["tp_d0"][idx])
        ok = (tp >= 0) & (pt >= a.ptmin) & np.isfinite(d0)
        if a.d0max > 0:
            ok &= d0 < a.d0max
        if not ok.any():
            continue
        for t in np.unique(tp[ok]):
            sel = ok & (tp == t)
            got = {}
            for L in (la, lb, lc):
                w = np.flatnonzero(sel & (ly == L))
                if len(w):
                    got[L] = idx[w[0]]
            if len(got) < 3:
                continue                      # not findable for THIS configuration
            r = trace_one(D, got[la], got[lb], got[lc], a.ptmin, a.d0, rA, rB, dr)
            if isinstance(r, tuple):
                # survived every gate: does a WRONG cluster win arbitration?
                _, phiC, wphi, zC, wz = r
                cand = np.flatnonzero(m & (D["layer"] == lc))
                near = cand[(np.abs(wrap(D["globalPhi"][cand] - phiC)) <= wphi)
                            & (np.abs(D["globalZ"][cand] - zC) <= wz)]
                if len(near):
                    best = near[np.argmin(np.abs(wrap(D["globalPhi"][near] - phiC)))]
                    if best != got[lc]:
                        tally["lost_at_arbitration"] += 1
                        arb += 1
                        continue
                tally["FOUND"] += 1
            else:
                tally[r] += 1
    tot = sum(tally.values())
    R = {"n_events": nev, "ptmin": a.ptmin, "d0_allowance_cm": a.d0,
         "d0max_cm": a.d0max, "seed": f"L{la}L{lb}->L{lc}",
         "n_findable": tot, "tally": tally,
         "fractions": {k: v / max(tot, 1) for k, v in tally.items()}}
    print(f"seed L{la}L{lb}->L{lc}  pT>{a.ptmin}  d0_allow={a.d0*1e4:.0f}um  "
          f"|d0|<{a.d0max*1e4:.0f}um   findable={tot}")
    print(f"{'stage':24s}{'n':>7}{'frac':>9}")
    for s in STAGES:
        if tally[s]:
            print(f"{s:24s}{tally[s]:7d}{tally[s]/max(tot,1):9.3f}")
    json.dump(R, open(a.out, "w"), indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
