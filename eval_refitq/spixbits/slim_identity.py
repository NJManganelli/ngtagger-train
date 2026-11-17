"""Baseline identity (stats-review-02, finding 3): the tables are changes from canon_v5's "implemented now", while the
taggers read the slim tier, a separate grid production. Before the parametrization is used on the slim tier, today's
slim-tier tracks and canon_v5's must agree (different events, same process) within 3 sd per cell in: the 68 % widths
of the d0 and z0 residuals, their pull widths, and the quantiles of every deployed-head output.

  python slim_identity.py [--nfiles 6]      -> checks/slim_tier_identity.json

canon_v5 side: the extended scan's "implemented now" (replay; validated against the stored CMSSW result).
Slim tier side: the first --nfiles complete files per mask on the SD card; L1TSmartPixelsExtTrack with its otTp_* truth
and head_* columns (head_clean is a probability; compared as a margin).
"""
import argparse
import glob
import json

import awkward as ak
import numpy as np
import uproot

import param_closure as PC
import param_model as PM
import study as ST

BASE = "/Volumes/NJM256GBSD/smartpixels/slimtaggertier/RelValTTbar_14TeV/slimtaggertier_slimtaggertier_SP{mask}_RelValTTbar_14TeV/*/0000/taggertier_*.root"
T_ = "L1TSmartPixelsExtTrack_"
COLS = ["rInv", "phi", "tanL", "z0", "d0", "cov_d0_d0", "cov_z0_z0", "spixRefitPerformed", "otTpIdx", "otTp_pt", "otTp_eta", "otTp_d0",
        "otTp_z0"] + [f"head_{h}" for h in PM.HEADS]


def slim(mask, nfiles):
    files = sorted(glob.glob(BASE.format(mask=mask)))[:nfiles]
    parts, ev0, marker = [], 0, None
    for f in files:
        with uproot.open(f) as fh:
            t = fh["Events"]
            A = t.arrays([T_ + c for c in COLS])
        n = ak.to_numpy(ak.num(A[T_ + "d0"]))
        d = {c: ak.to_numpy(ak.flatten(A[T_ + c])).astype(np.float64) for c in COLS}
        d["ev"] = np.repeat(np.arange(len(n)) + ev0, n)
        ev0 += len(n)
        parts.append(d)
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}, files


def summary(d0, z0, s_d0, s_z0, tp_d0, tp_z0, tp_pt, tp_eta, matched, refit, margins, ev):
    """{quantity: (value, bootstrap sd)}."""
    rng = np.random.default_rng(1)
    u, inv = np.unique(ev, return_inverse=True)
    W = [rng.multinomial(len(u), np.full(len(u), 1 / len(u)))[inv].astype(float) for _ in range(60)]
    out = {}

    def put(key, x, m, q=0.6827):
        if m.sum() < 200:
            return
        xv = x[m]
        out[key] = (float(np.quantile(xv, q)), float(np.std([PC.wq(xv, w[m], q) for w in W])), int(m.sum()))
    aeta = np.abs(tp_eta)
    for ie, (elo, ehi) in enumerate(((0, 1.1), (1.1, 1.8), (1.8, 2.4))):
        for plo, phi in ((2, 5), (5, 1e9)):
            m = matched & refit & (aeta >= elo) & (aeta < ehi) & (tp_pt >= plo) & (tp_pt < phi)
            cell = f"eta{elo}-{ehi}|pt{plo}-{'inf' if phi > 1e8 else phi}"
            put(f"d0 w68 um|{cell}", np.abs(d0 - tp_d0) * 1e4, m)
            put(f"z0 w68 um|{cell}", np.abs(z0 - tp_z0) * 1e4, m)
            put(f"d0 pull w68|{cell}", np.abs(d0 - tp_d0) / s_d0, m)
            put(f"z0 pull w68|{cell}", np.abs(z0 - tp_z0) / s_z0, m)
    for w, wm in (("prompt", np.abs(d0) <= 0.02), ("displaced", np.abs(d0) > 0.02)):
        for i, h in enumerate(PM.HEADS):
            for q in (0.16, 0.5, 0.84):
                put(f"head {h} q{int(q * 100)}|{w}", margins[:, i], refit & wm & np.isfinite(margins[:, i]), q)
    out["refit fraction"] = (float(refit.mean()), float(np.std([np.sum(w * refit) / w.sum() for w in W])), int(len(refit)))
    return out


def main():
    import ext_scan as XS
    ap = argparse.ArgumentParser()
    ap.add_argument("--nfiles", type=int, default=6)
    a = ap.parse_args()
    res = {"what": __doc__.split("\n\n")[0], "scenarios": {}}
    for scen in ST.SCENARIOS:
        S, files = slim(scen, a.nfiles)
        m = S["head_clean"]
        marg = np.stack([np.log(np.clip(m, 1e-9, 1 - 1e-9) / (1 - np.clip(m, 1e-9, 1 - 1e-9)))] + [S[f"head_{h}"] for h in PM.HEADS[1:]], 1)
        slim_sum = summary(S["d0"], S["z0"], np.sqrt(S["cov_d0_d0"]), np.sqrt(S["cov_z0_z0"]), S["otTp_d0"], S["otTp_z0"], S["otTp_pt"],
                           S["otTp_eta"], S["otTpIdx"] >= 0, S["spixRefitPerformed"] > 0, marg, S["ev"])
        T, today, _, truth, _ = PC.load(XS.out_root(), scen, ST.SCAN_FILES, [])
        ev = np.unique(T["event_key"], return_inverse=True)[1]
        can = summary(today["a"][:, 4], today["a"][:, 3], np.sqrt(today["cov_diag"][:, 4]), np.sqrt(today["cov_diag"][:, 3]), T["tp_d0"],
                      T["tp_z0"], T["tp_pt"], T["tp_eta"], T["matched"], today["refit"], today["margins"], ev)
        rows, bad = {}, []
        for k in can:
            if k not in slim_sum:
                continue
            (v1, s1, n1), (v2, s2, n2) = can[k], slim_sum[k]
            sd = float(np.hypot(s1, s2))
            z = (v2 - v1) / sd if sd > 0 else 0.0
            rows[k] = {"canon_v5": v1, "slim_tier": v2, "sd": sd, "difference_in_sd": z, "n_canon": n1, "n_slim": n2}
            if abs(z) > 3:
                bad.append(f"{k}: canon_v5 {v1:.4g} vs slim tier {v2:.4g} ({z:+.1f} sd)")
        res["scenarios"][scen] = {"slim_files": len(files), "slim_events": int(S["ev"].max() + 1), "slim_tracks": int(len(S["d0"])),
                                  "canon_tracks": int(len(T["run"])), "quantities_compared": len(rows), "beyond_3_sd": bad, "rows": rows}
        print(scen, "compared", len(rows), "beyond 3 sd:", len(bad), bad[:8], flush=True)
    json.dump(res, open(ST.LOCAL / "checks" / "slim_tier_identity.json", "w"), indent=1)


if __name__ == "__main__":
    main()
