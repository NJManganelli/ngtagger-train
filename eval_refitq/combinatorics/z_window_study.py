"""How wide does an IT seed's z road into the OT actually have to be?

THE TABLE CANNOT ANSWER THIS. rphimatchcut_/zmatchcut_ are indexed by OT seed,
and an IT seed's lever arm into the OT matches no OT seed's -- it extrapolates
from r ~ 3-15 cm to r ~ 23-108 cm. So the road has to be DERIVED, and the
honest derivation is to measure the residual it has to contain.

Method: take every TrackingParticle with clusters on the two chosen IT layers
and a genuine stub in an OT barrel layer, build the seed's r-z line from the IT
pair exactly as run_seed does (cot from the two clusters, z0 = zA - rA*cot),
project to the stub's own radius, and histogram z_stub - z_pred. Binned in pT
because the scattering term scales as 1/pT, which is the whole reason the 1 GeV
floor needs a wider road than the 2 GeV one.
"""
from __future__ import annotations
import sys, argparse, json
from pathlib import Path
import numpy as np
import awkward as ak
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tracklet_topology_cost as M   # noqa: E402
import kf_emulation as KF            # noqa: E402

# per-layer OT stub sigma_z, the calibrated values the census installs
SIGZ_OT = {1: 0.081, 2: 0.113, 3: 0.190, 4: 1.773, 5: 1.771, 6: 1.867}
IT = ["layer", "globalR", "globalZ", "globalPhi", "sigY", "tpIdx", "tpPt"]
OT = ["layer", "isBarrel", "r", "z", "tpIdx", "tpPt", "tpGenuine"]


def load(spec, nev):
    srcs = [f"{p}:Events" for p in M.expand_inputs(spec)]
    A = uproot.concatenate(srcs, [f"L1TSmartPixelsCluster_{c}" for c in IT]
                           + [f"L1TOTStub_{c}" for c in OT])
    if nev is not None:
        A = A[:nev]
    def flat(pre, cols):
        n = ak.to_numpy(ak.num(A[f"{pre}_{cols[0]}"]))
        D = {c: ak.to_numpy(ak.flatten(A[f"{pre}_{c}"])) for c in cols}
        D["event"] = np.repeat(np.arange(len(n)), n)
        return D, len(n)
    I, nev_ = flat("L1TSmartPixelsCluster", IT)
    O, _ = flat("L1TOTStub", OT)
    return I, O, nev_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-n", "--nev", type=int, default=100)
    ap.add_argument("--pairs", default="1,2 1,4 3,4 2,3")
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()

    I, O, nev = load(a.input, a.nev)
    ik = I["event"].astype(np.int64) * 100000 + I["tpIdx"]
    ok_i = I["tpIdx"] >= 0
    ob = (O["isBarrel"] > 0) & (O["tpGenuine"] > 0) & (O["tpIdx"] >= 0)
    ok_key = O["event"].astype(np.int64) * 100000 + O["tpIdx"]
    print(f"{nev} events, {ok_i.sum():,} IT clusters with truth, "
          f"{ob.sum():,} genuine barrel OT stubs")

    PT_BINS = [(1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 1e9)]
    out = {}
    for spec in a.pairs.split():
        la, lb = (int(x) for x in spec.split(","))
        sa = ok_i & (I["layer"] == la)
        sb = ok_i & (I["layer"] == lb)
        ka, kb = ik[sa], ik[sb]
        # ONE CLUSTER PER LAYER PER PARTICLE. A particle with two clusters on a
        # layer (module overlap, or a second crossing) makes intersect1d pick an
        # arbitrary one of each, and a pair drawn from two different crossings
        # gives a nonsense r-z line. That alone put p99.9 of the residual at
        # 60-500 cm while the core sigma stayed at 1-6 mm -- a tail made
        # entirely of pairings the seeder would never form.
        uka, cnta = np.unique(ka, return_counts=True)
        ukb, cntb = np.unique(kb, return_counts=True)
        solo = np.intersect1d(uka[cnta == 1], ukb[cntb == 1])
        com, p1, p2 = np.intersect1d(ka, kb, return_indices=True)
        keep0 = np.isin(com, solo)
        com, p1, p2 = com[keep0], p1[keep0], p2[keep0]
        A_, B_ = np.flatnonzero(sa)[p1], np.flatnonzero(sb)[p2]
        rA, zA = I["globalR"][A_], I["globalZ"][A_]
        rB, zB = I["globalR"][B_], I["globalZ"][B_]
        dr = rB - rA
        good = np.abs(dr) > 0.5
        cot = np.where(good, (zB - zA) / np.where(good, dr, 1e9), 0.0)
        z0 = zA - rA * cot
        # AND ONLY SEEDS THAT WOULD BE FORMED. it_pairs gates on
        # |z0| <= Z_LUMI before any projection, so a residual measured without
        # that gate describes candidates the system never builds.
        good &= np.abs(z0) <= M.Z_LUMI
        com, A_, B_ = com[good], A_[good], B_[good]
        rA, zA, rB, zB = rA[good], zA[good], rB[good], zB[good]
        dr, cot, z0 = dr[good], cot[good], z0[good]
        pt = I["tpPt"][A_]
        # index the seed line by its TP key so OT stubs can be joined to it
        order = np.argsort(com)
        key_s = com[order]
        print(f"\n=== IT seed IL{la}+IL{lb}  (lever arm {np.median(np.abs(dr)):.1f} cm, "
              f"{len(com):,} particles) ===")
        print(f"{'OT layer':>9s}{'pT band':>12s}{'n':>8s}{'sigma':>9s}"
              f"{'p99':>9s}{'window':>11s}{'contains':>12s}{'w/sigma':>8s}")
        for ol in range(1, 7):
            so = ob & (O["layer"] == ol)
            if not so.any():
                continue
            oi = np.flatnonzero(so)
            pos = np.searchsorted(key_s, ok_key[oi])
            pos = np.clip(pos, 0, max(len(key_s) - 1, 0))
            hit = (len(key_s) > 0) & (key_s[pos] == ok_key[oi])
            oi, pos = oi[hit], order[pos[hit]]
            if len(oi) < 50:
                continue
            r_ot, z_ot = O["r"][oi], O["z"][oi]
            res = z_ot - (z0[pos] + r_ot * cot[pos])
            ptv = O["tpPt"][oi]
            for lo, hi in PT_BINS:
                m = (ptv >= lo) & (ptv < hi) & np.isfinite(res)
                if m.sum() < 40:
                    continue
                x = res[m]
                sig = 1.4826 * np.median(np.abs(x - np.median(x)))
                p99 = np.percentile(np.abs(x), 99)
                p999 = np.percentile(np.abs(x), 99.9)
                # THE WINDOW THE CODE ACTUALLY DERIVES. it_project builds
                #   szp = hypot(sigY_stub, r*sigma_cot) + r*theta_MS ; window = 3*szp
                # and sigY_stub for an OT stub is the CALIBRATED per-layer sigma
                # that _unify installs, not a nominal strip length. An earlier
                # version of this study hardcoded 0.2 cm there and made OL4-OL6
                # look as though the window were 0.4 sigma wide; the calibrated
                # values (0.08/0.11/0.19/1.77/1.77/1.87) are what is really used
                # and they agree with the sigma measured here to a few percent.
                sct = np.hypot(np.median(I["sigY"][A_]), np.median(I["sigY"][B_])) \
                    / max(abs(np.median(dr)), 0.1)
                ms = (M.THETA_MS_MRAD * 1e-3 / max(lo, 0.5)) * np.sqrt(2.0)
                rr = np.median(r_ot[m])
                sz = np.hypot(SIGZ_OT[ol], rr * sct) + rr * ms
                model = M.NSIG * sz
                cont = float(np.mean(np.abs(x) <= model))
                w99 = float(np.percentile(np.abs(x), 99))
                lab = f"{lo:g}-{hi:g}" if hi < 1e8 else f">{lo:g}"
                print(f"{'OL%d' % ol:>9s}{lab:>12s}{int(m.sum()):8d}"
                      f"{sig:9.3f}{w99:9.2f}{model:11.2f}{100 * cont:11.1f}%"
                      f"{model / max(sig, 1e-6):8.1f}")
                out[f"IL{la}IL{lb}_OL{ol}_{lab}"] = {
                    "n": int(m.sum()), "sigma_cm": float(sig),
                    "p99_cm": float(p99), "p999_cm": float(p999),
                    "model_3sigma_cm": float(model)}
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
