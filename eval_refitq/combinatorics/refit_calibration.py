"""Calibration data for the refit covariance: the target cluster against the SEED.

For every PERFECT track and every IT barrel layer, the majority-owner TP's
cluster (the target) is recorded against the UN-UPDATED OT-only projection onto
its own module (ot_projection.evaluate_layer, study 19's code), together with
everything needed to evaluate a trial covariance model offline without
re-projecting:

  residual     r = cluster - projection in (u, v, cotAlpha, cotBeta), module frame
  J            d(u, v, cotA, cotB)/d(rInv, phi, tanL, z0, d0) at that module
  C            the OT fit's helix covariance (L1TTrack_cov_*)
  sig          the cluster's own errors (CPE sigX, sigY; sensor sigAlpha, sigBeta)
  truth        the matched TP's helix projected onto the SAME module plane
               (u, v, cotA, cotB), and CMSSW's own helix-propagated true angles
               (tpLocalCotAlpha/Beta) as an independent check of that projection

Perfect tracks only: every stub belongs to the target's TP, so the target is
unambiguous and the OT fit is not pulled by another particle.

The truth helix has NO multiple scattering -- it is the TP's production helix --
so truth positions are trustworthy only where scattering between the
production vertex and the layer is negligible (high pT); the angle truth is
far less sensitive (a kink moves the direction by theta0, ~1e-4 rad at 5 GeV,
against sigAlpha ~0.02).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import awkward as ak
import numpy as np
import uproot
from ngtagger.truth_helix import tp_phi0

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ot_projection as OP          # noqa: E402
import ot_refit_projection as RP    # noqa: E402

TP_COLS = ["tp_pt", "tp_phi", "tp_tanL", "tp_z0", "tp_d0", "tp_charge", "tp_vx", "tp_vy"]
CL_TRUTH = ["tpLocalCotAlpha", "tpLocalCotBeta", "tpChargeFrac", "sizeX", "sizeY"]


def extract(files, geometry, nev=None, chunk=10, log=print):
    geo = OP.load_geometry(geometry)
    OP.check_inputs(files[0])
    extra = [f"L1TTrack_{c}" for c in TP_COLS] + [f"L1TSmartPixelsCluster_{c}" for c in CL_TRUTH]
    out = {k: [] for k in ("layer", "pt", "tanL", "d0", "r_mod", "res", "J", "C", "sig", "hasA",
                           "hasB", "meas", "truth", "cltruth", "chargeFrac", "sizeX", "sizeY",
                           "inside", "event", "track", "n_disk_stubs", "n_stubs")}
    done = 0
    ev_base = 0
    trk_base = 0
    for A in uproot.iterate([f"{f}:Events" for f in files], OP.branches(False) + extra,
                            step_size=chunk):
        if nev is not None and done >= nev:
            break
        if nev is not None and done + len(A) > nev:
            A = A[:nev - done]
        ctx = OP.prepare_chunk(A, geo)
        T, K, a, C = ctx["T"], ctx["K"], ctx["a"], ctx["C"]
        kpt = RP.pt_constant(T)
        tp = {c: ak.to_numpy(ak.flatten(A[f"L1TTrack_{c}"])).astype(np.float64) for c in TP_COLS}
        kt = {c: ak.to_numpy(ak.flatten(A[f"L1TSmartPixelsCluster_{c}"])).astype(np.float64)
              for c in CL_TRUTH}
        a_tp = np.stack([tp["tp_charge"] * kpt / np.where(tp["tp_pt"] > 0, tp["tp_pt"], 1.0),
                         tp_phi0(tp, kpt=kpt), tp["tp_tanL"], tp["tp_z0"], tp["tp_d0"]], 1)
        use = (ctx["cls"] == 0) & (tp["tp_pt"] > 0)
        # OT stubs in endcap disks per track: forward OT fits may need their own seed floor
        Sst = OP._flat(A, "L1TTrackStub", ["trackIdx", "isBarrel"])
        toff = np.concatenate([[0], np.cumsum(ctx["T"]["_n"])])
        gst = toff[Sst["event"]] + Sst["trackIdx"].astype(np.int64)
        n_disk = np.bincount(gst, weights=Sst["isBarrel"] == 0, minlength=ctx["ntrk"]).astype(np.int16)
        n_st = np.bincount(gst, minlength=ctx["ntrk"]).astype(np.int16)
        for L in OP.IT_LAYERS:
            ev = OP.evaluate_layer(ctx, a, C, L, geo, 4.0, use)
            if ev is None:
                continue
            c, pj, pm = ev["cand"], ev["pj"], ev["pm"]
            m = c["isA"]
            t, ci, p = c["t"][m], c["ci"][m], c["pair"][m]
            mi = pm[p]
            tr = OP.project_to_modules(a_tp[t], OP.cylinder_s(a_tp[t], geo["R"][L]), geo, mi)
            ok = tr["ok"]
            t, ci, p, mi = t[ok], ci[ok], p[ok], mi[ok]
            sel = {k: v[ok] for k, v in tr.items()}
            out["layer"].append(np.full(len(t), L, np.int8))
            out["pt"].append(T["pt"][t].astype(np.float32))
            out["tanL"].append(a[t, 2].astype(np.float32))
            out["d0"].append(tp["tp_d0"][t].astype(np.float32))
            out["r_mod"].append(np.hypot(geo["origin"][mi, 0], geo["origin"][mi, 1]).astype(np.float32))
            out["res"].append(np.stack([c["du"][m][ok], c["dv"][m][ok], c["dcA"][m][ok],
                                        c["dcB"][m][ok]], 1).astype(np.float64))
            out["J"].append(pj["J"][p].astype(np.float64))
            out["C"].append(C[t].astype(np.float64))
            out["sig"].append(np.stack([K["sigX"][ci], K["sigY"][ci], K["sigAlpha"][ci],
                                        K["sigBeta"][ci]], 1).astype(np.float64))
            out["hasA"].append(K["hasAlpha"][ci] > 0)
            out["hasB"].append(K["hasBeta"][ci] > 0)
            out["meas"].append(np.stack([K["localX"][ci], K["localY"][ci], K["localCotAlpha"][ci],
                                         K["localCotBeta"][ci]], 1).astype(np.float64))
            out["truth"].append(np.stack([sel["u"], sel["v"], sel["cotA"], sel["cotB"]], 1))
            out["cltruth"].append(np.stack([kt["tpLocalCotAlpha"][ci], kt["tpLocalCotBeta"][ci]], 1))
            out["chargeFrac"].append(kt["tpChargeFrac"][ci].astype(np.float32))
            out["sizeX"].append(kt["sizeX"][ci].astype(np.float32))
            out["sizeY"].append(kt["sizeY"][ci].astype(np.float32))
            out["inside"].append(c["inside"][m][ok])
            out["event"].append((ctx["tev"][t] + ev_base).astype(np.int32))
            out["track"].append((t + trk_base).astype(np.int64))
            out["n_disk_stubs"].append(n_disk[t])
            out["n_stubs"].append(n_st[t])
        done += len(A)
        ev_base += len(A)
        trk_base += ctx["ntrk"]
        log(f"    calibration extract: {done} events")
    return {k: np.concatenate(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--inputs", nargs="+", required=True)
    ap.add_argument("--geometry", required=True)
    ap.add_argument("-n", "--nev", type=int, default=None)
    ap.add_argument("-o", "--out", required=True, help=".npz")
    a = ap.parse_args()
    import glob
    files = [f for g in a.inputs for f in (sorted(glob.glob(g)) or [g])]
    D = extract(files, a.geometry, a.nev)
    np.savez_compressed(a.out, **D)
    print(f"wrote {a.out}: {len(D['layer']):,} target clusters")


if __name__ == "__main__":
    main()
