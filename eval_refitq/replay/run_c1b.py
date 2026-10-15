"""Driver of the C1b jobs (the replay's disc calibration; /Volumes/WDMac/smartpixels_scratch/wpc/c1b/PLAN.md rev 4).
Every job runs through ops/spheavy --frozen --on-busy wait, in the background.

  overlap  §1 item 2, files 1-2, AAAA only (one copy of each event): the same-surface overlap pairs on IL1
           and ID1-ID4; T0 (the toy closure in the real disc pair geometry, 1000 replicas, the planted R within
           3 % where the replica mean is known to 1 %, with the no-rotation known-bad), D0 (the IL1 closure against the same clusters' truth-residual
           R: the pooled ratio's one-sided 95 % bounds within [0.90, 1.10], per axis), D2 (per disc, the
           overlap R against v3's R at the pairs' own cells, within 1 +- 0.1 widened by t1(N) x the bootstrap
           sd; the same- vs opposite-facing agreement; B' < 0.02 cm GeV) and the facing-sign known-bad's
           size. Writes c1b/overlap/overlap_summary.json and the pairs.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import arms as AR
import common as CM
import disc_r as DR
import gates as G
from common import OP

C1B_OUT = CM.OUT / "c1b"
IL1, DISCS = 1, (11, 12, 13, 14)
N_TOY, N_TOY_KB = 1000, 40                    # rev 4 amendment A3: enough replicas that the mean's error is <= 1 %
TOY_TOL, TOY_SE_MAX = 0.03, 0.01
D0_BAND = (0.90, 1.10)
D2_TOL = 0.10
NEV_FILE = 100


def _concat(parts):
    parts = [p for p in parts if p]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def judged(n_clusters):
    return n_clusters >= 400                      # >= 200 pairs: two clusters per pair


def overlap(a):
    t0 = time.time()
    geo = OP.load_geometry(CM.GEOMETRY)
    tools = AR.Tools(None, calibration=AR.CALIBRATION, angle_table=None)
    tables = tools.cal["R_tables"]
    parts = []
    for fi, f in enumerate(CM.FIT_FILES):
        path = CM.input_path("AAAA", f)
        for start, A in CM.iterate(path, AR.branches(path), chunk=a.chunk, nev=a.nev):
            ctx = AR.prepare(A, geo)
            parts.append(DR.pairs_of_chunk(ctx, geo, (IL1,) + DISCS, start=fi * NEV_FILE + start))
        print(f"[{time.time() - t0:7.1f}s] AAAA f{f:02d}: pairs collected", flush=True)
    P = _concat(parts)
    nev = NEV_FILE * len(CM.FIT_FILES)
    reps = G.replicas(nev=nev)
    res = {"plan": "c1b/PLAN.md rev 4 §1 item 2, §3 T0 D0 D2", "files": [str(CM.input_path("AAAA", f)) for f in CM.FIT_FILES],
           "n_pairs": {str(s): int((P["surface"] == s).sum()) for s in (IL1,) + DISCS}, "material": [], "minor": []}
    rng = np.random.default_rng(20261015)
    planted = DR.planted_widths(tables)

    # T0: the toy closure on the real disc pairs (and its no-rotation known-bad). A bin is judged when it holds
    # >= 200 pairs and the replica mean's standard error is <= 1 % (amendment A3); others are reported.
    res["T0"] = {}
    for s in DISCS:
        sel = P["surface"] == s
        out = {}
        jb = None
        for lab, rot, nrep in (("model", True, N_TOY), ("known_bad_no_rotation", False, N_TOY_KB)):
            fits = [DR.toy(P, tables, sel, planted, rng, rotation=rot) for _ in range(nrep)]
            rel = {ax: np.array([f[f"R_{ax}"] for f in fits]) / planted[ax] - 1 for ax in ("u", "v")}
            dev = {ax: np.nanmean(rel[ax], 0) for ax in ("u", "v")}
            se = {ax: np.nanstd(rel[ax], 0) / np.sqrt(np.sum(np.isfinite(rel[ax]), 0).clip(1)) for ax in ("u", "v")}
            nn = {ax: fits[0][f"n_{ax}"] for ax in ("u", "v")}
            if jb is None:                            # the known-bad is judged on the model's judged bins
                jb = {ax: [b for b in range(len(dev[ax])) if judged(nn[ax][b]) and se[ax][b] <= TOY_SE_MAX]
                      for ax in ("u", "v")}
            bad = [f"{ax}{b}" for ax in ("u", "v") for b in jb[ax] if not abs(dev[ax][b]) <= TOY_TOL]
            out[lab] = {"deviation": dev, "se": se, "n_clusters": nn, "judged": jb, "failing_bins": bad,
                        "replicas": nrep, "B_prime": np.nanmean([f["B_prime"] for f in fits], 0)}
        res["T0"][str(s)] = out
        if out["model"]["failing_bins"]:
            res["material"].append(f"T0 surface {s}: planted R not recovered within 3 % in {out['model']['failing_bins']}")
        if not out["known_bad_no_rotation"]["failing_bins"]:
            res["material"].append(f"T0 surface {s}: the no-rotation known-bad passes (the toy cannot see the mixing)"
                                   + ("" if any(jb.values()) else "; no bin judged"))
        print(f"[{time.time() - t0:7.1f}s] T0 surface {s}", flush=True)

    # fits with the event bootstrap: IL1 and the discs; facing splits on the discs
    fits = {s: DR.bootstrap(P, tables, s, reps) for s in (IL1,) + DISCS}

    # D0: the IL1 closure, like-for-like (the same clusters' truth-residual R), pooled ratio per axis
    sel = P["surface"] == IL1
    ev = P["ev"].astype(np.int64)
    res["D0"] = {}
    for ax in ("u", "v"):
        ref, n = DR.truth_width(P, tables, sel, ax)
        ratios = []
        for r, w in zip(fits[IL1], [None] + list(reps)):
            if w is None:
                rr = ref
            else:
                k = np.repeat(np.flatnonzero(sel), w[ev[sel]])
                Q = {kk: v[k] for kk, v in P.items()}
                rr, _ = DR.truth_width(Q, tables, np.ones(len(k), bool), ax)
            x = r[f"R_{ax}"] / rr
            m = np.isfinite(x) & (n > 0)
            ratios.append(float(np.sum(n[m] * x[m]) / np.sum(n[m])) if m.any() else np.nan)
        lo, hi = np.nanpercentile(ratios[1:], [5, 95])
        v3c, _ = DR.v3_at_cells(P, tables, sel, ax)
        res["D0"][ax] = {"pooled_ratio": ratios[0], "bounds95": [lo, hi], "pass": D0_BAND[0] <= lo and hi <= D0_BAND[1],
                         "per_bin_ratio": fits[IL1][0][f"R_{ax}"] / ref, "n_clusters": n,
                         "edge_vs_module": ref / v3c}
        if not res["D0"][ax]["pass"]:
            res["material"].append(f"D0 {ax}: IL1 pooled ratio {ratios[0]:.3f} [{lo:.3f}, {hi:.3f}] outside [0.90, 1.10]")
    print(f"[{time.time() - t0:7.1f}s] D0", flush=True)

    # D2 per disc; the facing agreement; B'
    res["D2"] = {}
    disc_fail = False
    for s in DISCS:
        sel = P["surface"] == s
        e = {}
        for ax in ("u", "v"):
            v3c, n = DR.v3_at_cells(P, tables, sel, ax)
            R = np.array([f[f"R_{ax}"] for f in fits[s]])
            ratio = R / v3c
            sd = np.nanstd(ratio[1:], 0)
            nj = sum(judged(x) for x in n)
            W = G.t_one(max(nj, 1))
            fail = [b for b in range(len(n)) if judged(n[b]) and np.isfinite(ratio[0][b])
                    and abs(ratio[0][b] - 1) > D2_TOL + W * sd[b]]
            e[ax] = {"R": R[0], "v3_at_cells": v3c, "ratio": ratio[0], "sd": sd, "n_clusters": n, "W": W, "failing": fail}
            disc_fail |= bool(fail)
        bp = np.array([f["B_prime"] for f in fits[s]])
        e["B_prime"], e["B_prime_sd"] = bp[0], np.nanstd(bp[1:], 0)
        e["B_prime_ok"] = bool(np.all(np.nan_to_num(bp[0]) < DR.B_PRIME_MAX))
        if not e["B_prime_ok"]:
            res["material"].append(f"D2 surface {s}: B' {np.round(bp[0], 4).tolist()} >= {DR.B_PRIME_MAX} cm GeV")
        # same- vs opposite-facing, paired bootstrap
        same = P["facing1"] == P["facing2"]
        fs = {lab: DR.bootstrap({k: v[m_] for k, v in P.items()}, tables, s, reps)
              for lab, m_ in (("same", same), ("opposite", ~same))}
        fac = {}
        for ax in ("u", "v"):
            d = np.array([a_[f"R_{ax}"] - b_[f"R_{ax}"] for a_, b_ in zip(fs["same"], fs["opposite"])])
            sd = np.nanstd(d[1:], 0)
            z = d[0] / np.where(sd > 0, sd, np.nan)
            nmin = np.minimum(fs["same"][0][f"n_{ax}"], fs["opposite"][0][f"n_{ax}"])
            bad = [b for b in range(len(z)) if judged(nmin[b]) and np.isfinite(z[b]) and abs(z[b]) > 2]
            fac[ax] = {"same": fs["same"][0][f"R_{ax}"], "opposite": fs["opposite"][0][f"R_{ax}"], "z": z, "disagree": bad}
            if bad:
                res["material"].append(f"D2 surface {s} {ax}: same- vs opposite-facing R disagree in bins {bad}")
        e["facing"] = fac
        kb = DR.fit_surface(P, tables, sel, drop_facing=-1)
        e["known_bad_facing_sign_dropped"] = {"B_prime": kb["B_prime"], "R_u": kb["R_u"], "R_v": kb["R_v"]}
        res["D2"][str(s)] = e
        print(f"[{time.time() - t0:7.1f}s] D2 surface {s}", flush=True)
    res["D2_decision"] = "build disc R tables from the overlap R" if disc_fail else "keep v3's R"
    res["seconds"] = round(time.time() - t0, 1)
    out_dir = os.path.join(a.out, "overlap")
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "overlap_pairs.npz"), **P)
    with open(os.path.join(out_dir, "overlap_summary.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    print(json.dumps({k: res[k] for k in ("n_pairs", "material", "D2_decision")}, indent=1, default=str))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("job", choices=("overlap",))
    ap.add_argument("--nev", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--out", default=str(C1B_OUT))
    a = ap.parse_args(argv)
    return {"overlap": overlap}[a.job](a)


if __name__ == "__main__":
    raise SystemExit(main())
