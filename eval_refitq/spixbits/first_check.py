"""First check (PLAN.md section 8): the replay's "implemented now" (arms.v3 on the pinned payload, the producer's kpt,
float32 geometry, the nano's float32 OT helix) against the CMSSW v3 result stored in the canon_v5 nano.

closure_v3.py's rule for nano inputs: every track is judged at its first disagreement (accepted flag and cluster per
visited crossing). A near-tie is a first disagreement whose margin is <= max(eps, 1e-6 (1 + |score|)); eps is frozen
here, on file 4, as 10 x the largest difference between the replay's and the nano's two-layer chi2 of the applied
candidate (rankChi2) on agreeing accepted crossings. The nano does not store the committing scores, so the margin of
a disagreeing window is the replay's own (second-lowest minus lowest committing score; the rank margin where there
is none). MATERIAL: a first disagreement that is not a near-tie; near-ties > 0.1 % of crossings; on tracks with
identical choices, a state beyond 1e-3 sigma + 2 storage quanta or a covariance element beyond
1e-3 sqrt(C_ii C_jj) + 2 storage quanta (the quantum of each stored column is measured from its values).

  python first_check.py [--nev 10] [--eps <frozen value>] [--file 4]
"""
import argparse
import json
import time

import awkward as ak
import numpy as np

import study as ST
from study import AR, CM, OP
from common import ORP

HPAR = ("rInv", "phi", "tanL", "z0", "d0")
HIT = "L1TSmartPixelsRefitHit"
TRK = "L1TSmartPixelsTrack"
HCOLS = ("trackIdx", "layer", "hitAccepted", "selClusterIdx", "kchoice", "rankChi2", "skipPrice", "detId",
         "recoLocalX", "recoLocalY")
TCOLS = HPAR + tuple(f"cov_{HPAR[i]}_{HPAR[j]}" for i in range(5) for j in range(i, 5)) + \
    ("l1TrackIdx", "spixRefitPerformed", "spixNAcceptedHits")


def quantum(col):
    """Per value, the storage quantum of a float32 column: 2^(exponent - kept mantissa bits), the kept bits measured
    as 23 - the smallest number of trailing zero mantissa bits over the column's nonzero values."""
    x = np.asarray(col, np.float32)
    bits = x.view(np.uint32) & np.uint32(0x7FFFFF)
    nz = bits[(x != 0) & np.isfinite(x)]
    if not len(nz):
        return np.zeros(len(x)), 23
    tz = 23
    for b in range(23):
        if (nz & np.uint32((1 << (b + 1)) - 1)).any():
            tz = b
            break
    kept = 23 - tz
    e = np.floor(np.log2(np.where(x != 0, np.abs(x.astype(np.float64)), 1.0)))
    return np.where(x != 0, np.exp2(e - kept), 0.0), kept


def run(a):
    geo = ST.geometry()
    res = {}
    for scen in a.scen:
        path = ST.input_path(scen, a.file)
        cfg = CM.producer_config(path)
        assert CM.mask_of(cfg) == scen
        tools = ST.tools(cfg)
        arm = AR.v3(cfg)
        br = list(dict.fromkeys(AR.branches(path) + [f"{HIT}_{c}" for c in HCOLS] + [f"{TRK}_{c}" for c in TCOLS]))
        S = {"crossings": 0, "agree": 0, "tracks": 0, "identical": 0, "first": [], "dscore": [], "refit_mismatch": 0,
             "da": [], "dC": [], "da_raw": [], "worst": []}
        t0 = time.time()
        for start, A in CM.iterate(path, br, chunk=10, nev=a.nev):
            ctx = AR.prepare(A, geo)
            if a.positions == "hitfloat":
                # the nano's cluster table keeps 16 mantissa bits of localX / localY; the refit hit table keeps CMSSW's
                # float positions of the clusters it accepted: use those copies for them
                Hx = {c: ak.to_numpy(ak.flatten(A[f"{HIT}_{c}"])) for c in ("hitAccepted", "selClusterIdx",
                                                                          "recoLocalX", "recoLocalY")}
                hev_ = np.repeat(np.arange(len(A)), ak.to_numpy(ak.num(A[f"{HIT}_layer"])))
                am = (Hx["hitAccepted"] > 0) & (Hx["selClusterIdx"] >= 0)
                cg = ctx["coff"][hev_[am]] + Hx["selClusterIdx"][am]
                K2 = dict(ctx["K"])
                for kc, hc in (("localX", "recoLocalX"), ("localY", "recoLocalY")):
                    v = ctx["K"][kc].astype(np.float64).copy()
                    v[cg] = Hx[hc][am]
                    K2[kc] = v
                ctx["K"] = K2
            out = AR.run_arm(ctx, geo, cfg, scen, arm, tools)
            rec, seq = out["rec"], out["seq"]
            ntrk = ctx["ntrk"]
            toff = np.concatenate([[0], np.cumsum(ctx["T"]["_n"])])
            n_sp = ak.to_numpy(ak.num(A[f"{TRK}_d0"]))
            assert (n_sp == ctx["T"]["_n"]).all(), "the refit track table is not parallel to L1TTrack"
            l1 = ak.to_numpy(ak.flatten(A[f"{TRK}_l1TrackIdx"]))
            assert (l1 == np.arange(ntrk) - toff[ctx["tev"]]).all(), "l1TrackIdx is not the track's own index"
            H = {c: ak.to_numpy(ak.flatten(A[f"{HIT}_{c}"])) for c in HCOLS}
            hev = np.repeat(np.arange(len(A)), ak.to_numpy(ak.num(A[f"{HIT}_layer"])))
            gt = toff[hev] + H["trackIdx"].astype(np.int64)
            step = ORP._row_steps(seq, gt, H["layer"].astype(np.int64))
            ns = seq.shape[1]
            c_vis = np.zeros((ntrk, ns), bool)
            c_acc = np.zeros((ntrk, ns), bool)
            c_ci = np.full((ntrk, ns), -1, np.int64)
            c_rank = np.full((ntrk, ns), np.nan)
            ok = step >= 0
            S.setdefault("off_path_rows", 0)
            S["off_path_rows"] += int((~ok).sum())
            acc = (H["hitAccepted"] > 0) & (H["selClusterIdx"] >= 0)
            c_vis[gt[ok], step[ok]] = True
            c_acc[gt[ok], step[ok]] = acc[ok]
            c_ci[gt[ok], step[ok]] = np.where(acc[ok], H["selClusterIdx"][ok], -1)
            c_rank[gt[ok], step[ok]] = np.where(H["rankChi2"][ok] > -900, H["rankChi2"][ok], np.nan)
            r_acc = rec["acc"]
            r_ci = np.where(r_acc, rec["ci"] - ctx["coff"][ctx["tev"]][:, None], -1)
            vis = (rec["visited"] | c_vis) & (seq >= 0)
            same = (r_acc == c_acc) & (r_ci == c_ci)
            S["crossings"] += int(vis.sum())
            S["agree"] += int((vis & same).sum())
            both = vis & same & r_acc & np.isfinite(c_rank) & np.isfinite(rec["rankchi2"]) \
                & ~(vis & ~same).any(1)[:, None]                  # on tracks whose choices all agree
            S["dscore"].append(np.abs(c_rank[both] - rec["rankchi2"][both]))
            T = {c: ak.to_numpy(ak.flatten(A[f"{TRK}_{c}"])) for c in TCOLS}
            c_refit = T["spixRefitPerformed"] > 0
            has = np.trace(ctx["C"], axis1=1, axis2=2) > 0
            ev_ids = [(int(A["run"][e]), int(A["luminosityBlock"][e]), int(A["event"][e])) for e in range(len(A))] \
                if "run" in A.fields else None
            for t in np.flatnonzero(has | c_vis.any(1)):
                S["tracks"] += 1
                bad = np.flatnonzero(vis[t] & ~same[t])
                if len(bad):
                    j = int(bad[0])
                    ks = rec["kscores"][t, j]
                    fin = np.sort(ks[np.isfinite(ks)])
                    m = [float(fin[1] - fin[0])] if len(fin) >= 2 else []
                    if np.isfinite(rec["rmargin"][t, j]):
                        m.append(float(rec["rmargin"][t, j]))
                    S["first"].append({"event_entry": int(start + ctx["tev"][t]), "track": int(t - toff[ctx["tev"][t]]),
                                       "step": j, "surface": int(seq[t, j]), "replay": [bool(r_acc[t, j]), int(r_ci[t, j])],
                                       "cmssw": [bool(c_acc[t, j]), int(c_ci[t, j])],
                                       "margin": min(m) if m else None,
                                       "score": float(fin[0]) if len(fin) else None})
                    continue
                S["identical"] += 1
                if bool(out["refit"][t]) != bool(c_refit[t]):
                    S["refit_mismatch"] += 1
                    continue
                if not out["refit"][t]:
                    continue
            idt = np.flatnonzero((has | c_vis.any(1)) & ~(vis & ~same).any(1) & out["refit"] & c_refit)
            if len(idt):
                sg = np.sqrt(np.clip(np.einsum("nii->ni", out["C"][idt]), 0, None))
                for i, p in enumerate(HPAR):
                    q, _ = quantum(T[p])
                    d = np.abs(out["a"][idt, i] - T[p][idt].astype(np.float64))
                    S["da_raw"].append(d / sg[:, i])
                    S["da"].append((d - 2 * q[idt]) / sg[:, i])
                for i in range(5):
                    for j in range(i, 5):
                        col = T[f"cov_{HPAR[i]}_{HPAR[j]}"]
                        q, _ = quantum(col)
                        d = np.abs(out["C"][idt, i, j] - col[idt].astype(np.float64))
                        S["dC"].append((d - 2 * q[idt]) / (sg[:, i] * sg[:, j]))
            if start == 0:
                S["kept_mantissa_bits"] = {c: int(quantum(T[c])[1]) for c in TCOLS[:20]}
        ds = np.concatenate(S["dscore"]) if S["dscore"] else np.zeros(0)
        res[scen] = {"seconds": time.time() - t0, "tracks": S["tracks"], "crossings": S["crossings"],
                     "agreeing": S["agree"], "tracks_identical": S["identical"],
                     "refit_flag_mismatch": S["refit_mismatch"], "off_path_rows": S["off_path_rows"],
                     "max_dscore_agreeing": float(ds.max()) if len(ds) else 0.0, "n_dscore": int(len(ds)),
                     "dscore_q": [float(np.quantile(ds, q)) for q in (0.5, 0.99, 0.999)] if len(ds) else [],
                     "first_disagreements": S["first"],
                     "state_max_over_sigma_raw": float(np.max(np.concatenate(S["da_raw"]))) if S["da_raw"] else 0.0,
                     "state_max_excess": float(np.max(np.concatenate(S["da"]))) if S["da"] else 0.0,
                     "state_over_sigma_q": [float(np.quantile(np.concatenate(S["da_raw"]), q)) for q in (0.5, 0.99, 0.999)]
                     if S["da_raw"] else [],
                     "cov_max_excess": float(np.max(np.concatenate(S["dC"]))) if S["dC"] else 0.0,
                     "kept_mantissa_bits": S.get("kept_mantissa_bits")}
        print(scen, json.dumps({k: v for k, v in res[scen].items() if k != "first_disagreements"}), flush=True)
    eps = a.eps if a.eps is not None else 10 * max(r["max_dscore_agreeing"] for r in res.values())
    verdict = {"eps": eps, "eps_frozen_here": a.eps is None, "material": []}
    for scen, r in res.items():
        hard, near = [], []
        for f in r["first_disagreements"]:
            f["eps"] = max(eps, 1e-6 * (1 + abs(f["score"] or 0.0)))
            f["near_tie"] = f["margin"] is not None and f["margin"] <= f["eps"]
            (near if f["near_tie"] else hard).append(f)
        r.update(hard=len(hard), near_ties=len(near), cap=0.001 * r["crossings"])
        if hard:
            verdict["material"].append(f"{scen}: {len(hard)} first disagreements that are not near-ties")
        if len(near) > r["cap"]:
            verdict["material"].append(f"{scen}: {len(near)} near-ties > 0.1 % of {r['crossings']} crossings")
        if r["state_max_excess"] > 1e-3:
            verdict["material"].append(f"{scen}: state beyond 1e-3 sigma + 2 quanta ({r['state_max_excess']:.3g})")
        if r["cov_max_excess"] > 1e-3:
            verdict["material"].append(f"{scen}: covariance beyond 1e-3 + 2 quanta ({r['cov_max_excess']:.3g})")
        if r["refit_flag_mismatch"]:
            verdict["material"].append(f"{scen}: {r['refit_flag_mismatch']} tracks with identical choices but another refit flag")
    out = {"file": a.file, "nev": a.nev, "positions": a.positions, "verdict": verdict, "scenarios": res}
    (ST.LOCAL / "checks").mkdir(exist_ok=True)
    name = ST.LOCAL / "checks" / f"first_check_f{a.file:02d}_{a.positions}{'' if a.nev is None else f'_{a.nev}ev'}.json"
    json.dump(out, open(name, "w"), indent=1, default=float)
    print(json.dumps(verdict, indent=1))
    print({s: {k: r[k] for k in ("crossings", "agreeing", "hard", "near_ties", "tracks", "tracks_identical")} for s, r in res.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=int, default=4)
    ap.add_argument("--nev", type=int, default=None)
    ap.add_argument("--eps", type=float, default=None)
    ap.add_argument("--positions", choices=("clustertable", "hitfloat"), default="clustertable",
                    help="cluster positions: the nano's cluster table (16 mantissa bits; what the study runs on) or, "
                         "for the clusters CMSSW accepted, the refit hit table's float copies")
    ap.add_argument("--scen", nargs="+", default=list(ST.SCENARIOS))
    run(ap.parse_args())
