"""Entry-gate statistics (PLAN §4, §5, §10), computed offline from gatedata.py's outputs, plus the P0.2
transfer / closure check.

  p02_closure  P0.2's per-event refit-v2 references (canon_v2 files 3-10) recomputed on canon_v3 from CMSSW's
               own tracks with fom.py: bit-identity certifies the canon_v2 -> canon_v3 transfer and fom.py
  e1a          the table import (INN's own-cluster angle pulls), P0.3's test (i) form; its false-veto rate
  e1b          the coverage of the 99 % cuts (calibrated model), per bin and pooled
  e23          the ORP-cal identity (E2 / E3) and the reported angle-import effect on the pull cores
  e4           the forced replay's final parameters against L1TSmartPixelsTrack (file 4 = known-good)
  e5           PM against refit v2 (i)-(iv), PM-B, the known-bad mirrors
  e7           K's skip-loss and the wrong-hit rejection gained
Every error is the event-level bootstrap (400 replicas, seed fixed), paired across arms of one file.
"""
from __future__ import annotations

import json
import os

import awkward as ak
import numpy as np
from scipy import stats

import arms as AR
import common as CM
import cuts as CU
import engine as E
import fom as FM
from common import OP
from ngtagger.spdefs import angles as SA
from ngtagger.definitions.estimators import core

P02_REF = "/Volumes/WDMac/smartpixels_scratch/p02_refit_audit/p02_reference_per_event.npz"
SP_COLS = ("d0", "z0")
N_REP = 400
SEED = 20261014
NEV = 100


def t_two(n, alpha=0.01):
    return float(stats.norm.isf((1.0 - (1.0 - alpha) ** (1.0 / max(int(n), 1))) / 2.0))


def t_one(n, alpha=0.01):
    return float(stats.norm.isf(1.0 - (1.0 - alpha) ** (1.0 / max(int(n), 1))))


def replicas(nev=NEV, n=N_REP, seed=SEED):
    return np.random.default_rng(seed).multinomial(nev, np.full(nev, 1.0 / nev), size=n)


def load(scen, f, out=None, skip=()):
    """{run: {field: array}} of gatedata's output; the runs in skip are never read (a blinded run)."""
    z = np.load(os.path.join(out or str(CM.OUT / "entry"), f"{scen}_f{f:02d}.npz"))
    d = {}
    for k in z.files:
        run, field = k.split("__", 1)
        if run not in skip:
            d.setdefault(run, {})[field] = z[k]
    return d


# ---------------------------------------------------------------- P0.2 closure
def canonical_event_index(A):
    """Per event of the file: its rank of (run, lumi, event) (P0.2's index within a file)."""
    run, lumi, evn = (ak.to_numpy(A[k]).astype(np.int64) for k in ("run", "luminosityBlock", "event"))
    rank = np.empty(len(run), np.int64)
    rank[np.lexsort((evn, lumi, run))] = np.arange(len(run))
    return rank


def file_counts(scen, f, nev=None):
    """CMSSW refit v2's per-event H / P counts on canon_v3 file f (events in P0.2's canonical order)."""
    path = CM.input_path(scen, f)
    geo = OP.load_geometry(CM.GEOMETRY)
    br = AR.branches(path) + [f"{CM.SP_TRACK}_{c}" for c in SP_COLS] + ["run", "luminosityBlock", "event"]
    A = next(CM.iterate(path, list(dict.fromkeys(br)), chunk=100000, nev=nev))[1]
    ctx = AR.prepare(A, geo)
    H = E.hit_table(A)
    truth = FM.track_truth(ctx)
    sp = {c: ak.to_numpy(ak.flatten(A[f"{CM.SP_TRACK}_{c}"])).astype(np.float64) for c in SP_COLS}
    hm = FM.harm(sp["d0"], sp["z0"], ctx["T"]["d0"], ctx["T"]["z0"], truth)
    rank = canonical_event_index(A)
    ev = rank[ctx["tev"]]
    t, surf, acc, wrong = FM.hits_of_cmssw(H, ctx)
    return FM.per_event_counts(ev, len(rank), truth, hm, t, surf, acc, wrong)


def p02_closure(scen, f, nev=None):
    """{n_keys, identical, mismatches}: fom.py on CMSSW's file-f tracks vs P0.2's stored references."""
    got = file_counts(scen, f, nev)
    ref = np.load(P02_REF, allow_pickle=True)
    sl = slice(100 * (f - 3), 100 * (f - 2))
    mism = []
    for k, v in got.items():
        kind, key, nd = k.split("/")
        rk = (f"{scen}__H_either__{key}__num" if (kind == "H" and nd == "num") else
              f"{scen}__H__{key}__den" if kind == "H" else f"{scen}__P__{key}__{nd}")
        if rk not in ref.files:
            if np.any(v):                       # P0.2 skips a surface without crossings (an inactive one)
                mism.append((k, "missing in reference"))
            continue
        if not np.array_equal(ref[rk][sl].astype(np.float64), v.astype(np.float64)):
            mism.append((k, float(np.abs(ref[rk][sl] - v).sum())))
    return {"scenario": scen, "file": f, "n_keys": len(got), "identical": not mism, "mismatches": mism[:20]}


# ---------------------------------------------------------------- E1(a)
def _bin_stats(x, w=None):
    """(centre, width) of a pull sample: the table's estimators (25 %-trimmed mean, weighting width)."""
    if w is not None:
        x = np.repeat(x, w)
    if len(x) < SA.MIN_FIT:
        return np.nan, np.nan
    c = SA.centre_of(x)
    return c, SA.weighting_width(np.abs(x - c))


def e1a_bins(own, scen_table, clean=True, stat="e1"):
    """E1(a) bins: (angle, surface, signed key bin, orientation) over own class-T clusters (table criterion)
    with a clean history (PD8), barrel. Returns a list of dicts with the values and their event index."""
    out = []
    base = own["table_pop"] & (own["surface"] <= 4) & (own["clean"] if clean else True)
    for i, ang in enumerate(SA.ANGLES):
        x = own[stat][:, i]
        m0 = base & own["has"][:, i] & np.isfinite(x)
        keys = np.stack([own["surface"], own["keybin"][:, i], own["orient"]], 1)[m0]
        xs, evs = x[m0], own["ev"][m0]
        for k in np.unique(keys, axis=0):
            m = (keys == k).all(1)
            if m.sum() < SA.MIN_FIT:
                continue
            row = scen_table.rows_of("true", np.array([k[0]]), np.array([np.mean(SA.fine_bin_bounds(int(k[0]), int(k[1])))]),
                                     np.zeros(1, np.int8))[0]
            r = scen_table.doc["keys"]["true"]["rows"][row] if row >= 0 else None
            tsd = (r[ang]["sd_centre"], r[ang]["sd_width"]) if r and r.get(ang) else (0.0, 0.0)
            out.append({"angle": ang, "surface": int(k[0]), "keybin": int(k[1]), "orient": int(k[2]),
                        "x": xs[m], "ev": evs[m], "table_sd": tsd})
    return out


def e1a_judge(bins, W=None, nev=NEV, n_rep=N_REP, seed=SEED, sd_from=None):
    """P0.3 test (i) on E1's bins: centre within 0 +- 0.25 and width within 1 +- 0.1, each widened by
    W x max(bootstrap sd, table sd); W = t1(N), N = 2 x judged bin-angles. sd_from: reuse these bins'
    bootstrap sd (the false-veto resamples, PD10 (a))."""
    rep = replicas(nev, n_rep, seed)
    res = []
    for k, b in enumerate(bins):
        c, w = _bin_stats(b["x"])
        if sd_from is None:
            R = np.array([_bin_stats(b["x"], rep[r][b["ev"]]) for r in range(n_rep)])
            sdc, sdw = np.nanstd(R[:, 0]), np.nanstd(R[:, 1])
        else:
            sdc, sdw = sd_from[k]["sd_boot"]
        res.append({"angle": b["angle"], "surface": b["surface"], "keybin": b["keybin"], "orient": b["orient"],
                    "n": int(len(b["x"])), "centre": c, "width": w, "sd_boot": (sdc, sdw),
                    "sd": (max(sdc, b["table_sd"][0]), max(sdw, b["table_sd"][1]))})
    N = 2 * len(res)
    W = t_one(N) if W is None else W
    for r in res:
        r["fail_centre"] = bool(abs(r["centre"]) > 0.25 + W * r["sd"][0])
        r["fail_width"] = bool(abs(r["width"] - 1.0) > 0.1 + W * r["sd"][1])
        r["fail"] = r["fail_centre"] or r["fail_width"]
    return {"W": W, "N": N, "bins": res, "n_fail": int(sum(r["fail"] for r in res))}


def e1a_material(judged, own):
    """Failing bins holding > 2 % of their surface's class-T clusters (table criterion, clean, barrel)."""
    base = own["table_pop"] & (own["surface"] <= 4) & own["clean"]
    tot = {S: int((base & (own["surface"] == S)).sum()) for S in (1, 2, 3, 4)}
    return [r for r in judged["bins"] if r["fail"] and r["n"] > 0.02 * tot.get(r["surface"], 1)]


def e1a_false_veto(bins4, judged4, n_res=250, seed=SEED + 7):
    """PD10 (a): the rule on 250 event resamples of file 4, judged with file 4's bootstrap sd."""
    rng = np.random.default_rng(seed)
    W = judged4["W"]
    fails = 0
    for _ in range(n_res):
        wts = rng.multinomial(NEV, np.full(NEV, 1.0 / NEV))
        rs = [dict(b, x=np.repeat(b["x"], wts[b["ev"]]), ev=np.repeat(b["ev"], wts[b["ev"]])) for b in bins4]
        j = e1a_judge(rs, W=W, sd_from=judged4["bins"])
        fails += j["n_fail"] > 0
    return {"resamples": n_res, "vetoed": int(fails), "rate": fails / n_res, "criterion": 0.057}


# ---------------------------------------------------------------- E1(b)
def _lookup_n(cuts, tab_key, keys_chain):
    tab = cuts.doc["cuts"][tab_key]
    for k in keys_chain:
        e = tab.get(k)
        if e is not None and e["value"] is not None and e["n"] >= CU.MIN_N:
            return e["value"], e["n"], k
    e = tab["*"]
    return e["value"], e["n"], "*"


def e1b_consumer(cuts, consumer, key, stat, surface, first, ev, keybin=None, nev=NEV, cls=None):
    """Coverage of one consumer's cut on its population: per lookup bin (x class cls, e.g. the number
    of remaining inner hits, PD17) the exceedances, the exact order-statistic null with variance
    max(event bootstrap, exchangeable); pooled the one-sided 95 % shortfall bound against the 0.5 %
    margin (PD4 (h)), its sd the event bootstrap plus the cuts' fit variance per shared cut (PD17)."""
    kb = np.zeros(len(stat), np.int64) if keybin is None else np.asarray(keybin, np.int64)
    combos = np.stack([np.asarray(surface, np.int64), np.asarray(first, np.int64), kb], 1)
    uniq, inv = np.unique(combos, axis=0, return_inverse=True)
    looked = []
    for S, f, b in uniq.tolist():
        chain = (f"{S}|{f}|{b}", f"{S}|{f}", f"{S}") if keybin is not None else (f"{S}|{f}", f"{S}")
        looked.append(_lookup_n(cuts, key, chain))
    vals = np.array([x[0] for x in looked])[inv.ravel()]
    nfit = np.array([x[1] for x in looked])[inv.ravel()]
    bkey = np.array([x[2] for x in looked])[inv.ravel()]
    beyond = stat > vals
    lkey = bkey.copy()
    if cls is not None:
        bkey = np.char.add(np.char.add(bkey.astype(str), "|n"), np.asarray(cls).astype(str))
    rep = replicas(nev)
    out = {"bins": [], "consumer": consumer, "key": key}
    for k in np.unique(bkey):
        m = bkey == k
        n_t, mcount = int(m.sum()), int(beyond[m].sum())
        e0 = float(SA.expected_coverage(nfit[m][0], CU.LEVEL))
        v_ex = float(SA.exchangeable_coverage_variance(nfit[m][0], n_t, CU.LEVEL))
        cb = np.array([1 - np.sum(beyond[m] * rep[r][ev[m]]) / max(np.sum(rep[r][ev[m]]), 1) for r in range(N_REP)])
        var = max(v_ex, float(np.var(cb)))
        z = float(SA.coverage_z(mcount, n_t, e0, var))
        out["bins"].append({"bin": k, "n_test": n_t, "n_fit": int(nfit[m][0]), "beyond": mcount,
                            "coverage": 1 - mcount / n_t, "e0": e0, "var": var, "z": z})
    zs = np.array([b["z"] for b in out["bins"] if np.isfinite(b["z"])])
    out["t"] = t_two(len(zs))
    out["max_abs_z"] = float(np.max(np.abs(zs))) if len(zs) else None
    out["chi2_p"] = float(stats.chi2.sf(np.sum(zs ** 2), len(zs))) if len(zs) else None
    C = 1 - beyond.mean()
    Cb = np.array([1 - np.sum(beyond * rep[r][ev]) / max(np.sum(rep[r][ev]), 1) for r in range(N_REP)])
    fit_var = 0.0
    for k in np.unique(lkey):
        m = lkey == k
        e0 = float(SA.expected_coverage(nfit[m][0], CU.LEVEL))
        fit_var += (m.mean()) ** 2 * e0 * (1 - e0) / (nfit[m][0] + 2)
    sd = float(np.sqrt(np.var(Cb) + fit_var))
    out["pooled"] = {"n": int(len(stat)), "coverage": float(C), "sd": sd,
                     "shortfall_bound95": float(CU.LEVEL - C + 1.645 * sd), "margin": 0.005}
    out["pooled"]["pass"] = out["pooled"]["shortfall_bound95"] <= 0.005
    return out


def e1b_all(own, loo, cuts, scen, cov="calibrated"):
    """Every consumer's coverage on one file of scenario scen (populations PD1 + PD8; rr cuts per
    scenario, PD15)."""
    gate, window, rr = CU.populations(own, {"own_primary": loo["own_primary"], "clean_track": loo["clean_track"]})
    res = {}
    for dims in ("xy", "xya", "xyb", "xyab"):
        res[f"gate|{dims}"] = e1b_consumer(cuts, "gate", f"gate|{dims}|{cov}", own[f"g_{dims}"][gate],
                                           own["surface"][gate], own["first"][gate], own["ev"][gate])
        res[f"rr|{dims}"] = e1b_consumer(cuts, "rr", CU.table_key("rr", dims, cov, scen), loo[f"d_{dims}"][rr],
                                         loo["surface"][rr], loo["first"][rr], loo["ev"][rr])
    for i, ang in enumerate(SA.ANGLES):
        m = window & np.isfinite(own["z"][:, i])
        res[f"window|{ang}"] = e1b_consumer(cuts, "window", f"window|{ang}|{cov}", own["z"][m, i],
                                            own["surface"][m], own["first"][m], own["ev"][m], own["keybin"][m, i])
    return res


def e1b_verdict(scen, k, r, safe_direction_minor):
    """(MATERIAL lines, MINOR lines) of one consumer's E1(b) result. A failing pooled bound is MATERIAL;
    bins beyond t(N) are MATERIAL when they hold > 2 % of the pooled rows, else MINOR. With
    safe_direction_minor (PD15, PD16), over-coverage bins (z > +t) are MINOR whatever their size."""
    mat, minor = [], []
    if not r["pooled"]["pass"]:
        mat.append(f"E1(b) {scen} {k}: pooled shortfall bound {r['pooled']['shortfall_bound95']:.4f} > 0.005")
    bad = [b for b in r["bins"] if np.isfinite(b["z"]) and abs(b["z"]) > r["t"]]
    safe = [b for b in bad if b["z"] > 0] if safe_direction_minor else []
    unsafe = [b for b in bad if b not in safe]
    if safe:
        minor.append(f"E1(b) {scen} {k}: {len(safe)} bins over-covered beyond t(N) (safe direction): "
                     + ", ".join(f"{b['bin']} {b['coverage']:.4f} (z {b['z']:+.1f})" for b in safe))
    if unsafe:
        big = [b for b in unsafe if b["n_test"] > 0.02 * r["pooled"]["n"]]
        (mat if big else minor).append(f"E1(b) {scen} {k}: {len(unsafe)} bins beyond t(N)")
    return mat, minor


def e1b_class_verdict(scen, k, r, min_n=CU.MIN_N):
    """PD17's per-class rule on an e1b_consumer result with classes: the class-bins with n >= min_n are
    judged at t(N) over them; under-coverage beyond t is MATERIAL, over-coverage MINOR; a failing
    pooled bound is MATERIAL. Returns (MATERIAL lines, MINOR lines, t)."""
    judged = [b for b in r["bins"] if b["n_test"] >= min_n and np.isfinite(b["z"])]
    t = t_two(len(judged)) if judged else 0.0
    mat = [f"rr per class {scen} {k}: {b['bin']} {b['coverage']:.4f} (z {b['z']:+.1f}, n {b['n_test']})"
           for b in judged if b["z"] < -t]
    minor = [f"rr per class {scen} {k}: {b['bin']} over-covered {b['coverage']:.4f} (z {b['z']:+.1f}, n {b['n_test']})"
             for b in judged if b["z"] > t]
    if not r["pooled"]["pass"]:
        mat.append(f"rr per class {scen} {k}: pooled shortfall bound {r['pooled']['shortfall_bound95']:.4f} > 0.005")
    return mat, minor, t


# ---------------------------------------------------------------- E2 / E3
def e23(meta_identity, own_inn, own_pay):
    """The ORP-cal identity over every chunk, and the reported angle-import effect: the position and
    innovation pull cores of own primaries with a clean history, pT >= 5 GeV, per barrel surface, INN (the
    table) minus INN-payload (the old scales), paired."""
    ident = all(c["identical"] for c in meta_identity)
    rep = {}
    for S in (1, 2, 3, 4):
        e = {}
        for nm, R in (("table", own_inn), ("payload", own_pay)):
            m = R["primary"] & R["clean"] & (R["surface"] == S) & (R["tp_pt"] >= 5.0)
            for c in ("pu", "pv"):
                cr = core(R[c][m])
                e[f"{nm}_{c}"] = cr["rms0"] if cr else None
        rep[S] = e
    return {"identical_every_chunk": ident, "n_chunks": len(meta_identity),
            "max_abs_diff": max(c["max_abs_diff"] for c in meta_identity) if meta_identity else None,
            "reported_pull_cores": rep}


# ---------------------------------------------------------------- E4
def e4_stats(run, trk, sel):
    """|Δ| / σ_CMSSW of the final parameters (d0, z0, pT via rInv, phi, tanL) on the tracks sel."""
    out = {}
    pairs = (("d0", "d0", "cov_d0_d0"), ("z0", "z0", "cov_z0_z0"), ("pT", "rInv", "cov_rInv_rInv"),
             ("phi0", "phi", "cov_phi_phi"), ("tanL", "tanL", "cov_tanL_tanL"))
    for nm, p, c in pairs:
        d = run[p][sel] - trk[f"sp_{p}"][sel]
        if p == "phi":
            d = np.angle(np.exp(1j * d))
        with np.errstate(invalid="ignore", divide="ignore"):
            out[nm] = np.abs(d) / np.sqrt(trk[f"sp_{c}"][sel])
    return out


def e4(d3, d4, kb=None):
    """PD11: per parameter, file 3's p50 / p99 of |Δ| / σ and its fraction beyond file 4's p99.9, against
    file 4 (+ t1(45) combined sigma). kb: the known-bad forced run (file 3)."""
    W = t_one(45)
    rep = replicas()
    res = {}

    def sel_of(d, run):
        return run["has_rows"].astype(bool) & np.isfinite(d["cmssw_trk"]["sp_cov_d0_d0"]) if "has_rows" in run \
            else np.isfinite(d["cmssw_trk"]["sp_cov_d0_d0"])
    s3 = sel_of(d3, d3["forced"])
    s4 = sel_of(d4, d4["forced"])
    x3 = e4_stats(d3["forced"], d3["cmssw_trk"], s3)
    x4 = e4_stats(d4["forced"], d4["cmssw_trk"], s4)
    xk = e4_stats(kb, d3["cmssw_trk"], s3) if kb is not None else None
    ev3, ev4 = d3["cmssw_trk"]["ev"][s3], d4["cmssw_trk"]["ev"][s4]

    def q_boot(x, ev, qq):
        ok = np.isfinite(x)
        v = np.percentile(x[ok], qq)
        b = [np.percentile(y, qq) for y in (np.repeat(x[ok], rep[r][ev[ok]]) for r in range(100)) if len(y)]
        return float(v), float(np.std(b))

    for nm in x3:
        e = {}
        for qq in (50, 99):
            v3, s3_ = q_boot(x3[nm], ev3, qq)
            v4, s4_ = q_boot(x4[nm], ev4, qq)
            e[f"p{qq}"] = {"file3": v3, "file4": v4, "sd": float(np.hypot(s3_, s4_)), "pass": v3 <= v4 + W * np.hypot(s3_, s4_)}
            if xk is not None:
                vk, _ = q_boot(xk[nm], ev3, qq)
                e[f"p{qq}"]["known_bad"] = vk
                e[f"p{qq}"]["known_bad_fails"] = bool(vk > v4 + W * np.hypot(s3_, s4_))
        lim4 = np.nanpercentile(x4[nm], 99.9)
        ok3 = np.isfinite(x3[nm])
        f3 = float(np.mean(x3[nm][ok3] > lim4))
        sd = float(np.sqrt(0.001 * 0.999 / max(ok3.sum(), 1)))
        e["beyond_p999"] = {"file3": f3, "limit": 0.001 + W * sd, "pass": f3 <= 0.001 + W * sd}
        if xk is not None:
            okk = np.isfinite(xk[nm])
            fk = float(np.mean(xk[nm][okk] > lim4))
            e["beyond_p999"]["known_bad"] = fk
            e["beyond_p999"]["known_bad_fails"] = fk > 0.001 + W * sd
        e["pass"] = e["p50"]["pass"] and e["p99"]["pass"] and e["beyond_p999"]["pass"]
        res[nm] = e
    return {"W": W, "params": res}


# ---------------------------------------------------------------- E5
def _sum_by_event(counts, nev=NEV):
    """{key: per-event array} from concatenated per-chunk count blocks with an 'ev' column."""
    out = {}
    ev = counts["ev"].astype(np.int64)
    for k, v in counts.items():
        if k in ("ev", "draw"):
            continue
        out[k] = np.bincount(ev, weights=v, minlength=nev)
    return out


def e5_rates(risk, nev=NEV, draws=1):
    """Per surface: first disagreements / at-risk crossings, with the event-bootstrap sd."""
    rep = replicas(nev)
    out = {}
    for S in FM.SURF:
        m = risk["surface"] == S
        if not m.any():
            continue
        num = np.bincount(risk["ev"][m].astype(np.int64), weights=risk["first"][m], minlength=nev)
        den = np.bincount(risk["ev"][m].astype(np.int64), minlength=nev).astype(float)
        r = num.sum() / max(den.sum(), 1)
        rb = (rep @ num) / np.maximum(rep @ den, 1)
        out[FM.NAME[S]] = {"rate": float(r), "sd": float(np.std(rb)), "n": int(den.sum() / draws)}
    return out


def e5(d3, d4, scen):
    """PM against refit v2 on file 3, with the perturbation model of file 4 as the prediction."""
    W1 = t_one(24)
    res = {}
    obs = e5_rates(d3["PM/e5_risk"])
    pred = e5_rates(d4["e5_model_risk"], draws=10)
    res["i_rates"] = {}
    for S, o in obs.items():
        p = pred.get(S, {"rate": 0.0, "sd": 0.0})
        sd = float(np.hypot(o["sd"], p["sd"]))
        res["i_rates"][S] = {"observed": o["rate"], "predicted": p["rate"], "sd": sd, "n": o["n"],
                             "pass": o["rate"] <= p["rate"] + W1 * sd}
    first = d3["PM/e5_first"]
    n_dis = len(first["cause"])
    n_out = int((first["cause"] == 1).sum())
    n_nocr = int((first["cause"] == 2).sum())
    n_risk = int(len(d3["PM/e5_risk"]["surface"]))
    lim = float(stats.poisson.ppf(0.99, max(0.002 * n_risk, 1e-9)))          # PD12
    res["i_near_tie"] = {"disagreements": n_dis, "near_tie": int((first["cause"] == 0).sum()),
                         "outside_window": n_out, "no_cmssw_crossing": n_nocr, "at_risk_crossings": n_risk,
                         "poisson99_bound_of_0.2pct_at_risk": lim, "pass": n_out + n_nocr <= lim}
    # (ii)-(iii)a: pooled signed first-flip shift, observed vs predicted
    T = d3["cmssw_trk"]
    pr = (T["tp_pop"] > 0) & (T["tp_origin"] == 0)
    den_h = np.bincount(T["ev"][pr].astype(np.int64), minlength=NEV).astype(float)
    Hh = d3["cmssw_hit"]
    popm = (T["tp_pop"] > 0)[Hh["t"]] & Hh["acc"]
    den_p = np.bincount(Hh["ev"][popm].astype(np.int64), minlength=NEV).astype(float)
    ob = _sum_by_event(d3["e5_obs_counts"]) if "e5_obs_counts" in d3 else {}
    pr_c = _sum_by_event(d4["e5_model_counts"]) if "e5_model_counts" in d4 else {}
    T4, H4 = d4["cmssw_trk"], d4["cmssw_hit"]
    pr4 = (T4["tp_pop"] > 0) & (T4["tp_origin"] == 0)
    den_h4 = np.bincount(T4["ev"][pr4].astype(np.int64), minlength=NEV).astype(float)
    den_p4 = np.bincount(H4["ev"][((T4["tp_pop"] > 0)[H4["t"]] & H4["acc"])].astype(np.int64), minlength=NEV).astype(float)
    rep = replicas()
    W6 = t_two(6)

    def pooled(cnt, prefix, den, draws=1):
        num = sum((v for k, v in cnt.items() if k.startswith(prefix) and k.endswith("/num")), np.zeros(NEV)) / draws
        r = num.sum() / max(den.sum(), 1)
        rb = (rep @ num) / np.maximum(rep @ den, 1)
        return float(r), float(np.std(rb))

    a = {}
    for kind, d_obs, d_pred in (("H", den_h, den_h4), ("P", den_p, den_p4)):
        o, so = pooled(ob, f"{kind}/", d_obs)
        p, sp = pooled(pr_c, f"{kind}/", d_pred, draws=10)
        sd = float(np.hypot(so, sp))
        a[kind] = {"observed": o, "predicted": p, "sd": sd, "pass": abs(o - p) <= W6 * sd}
    res["ii_iii_a"] = a
    # (ii)-(iii)b: the full PM - v2 shift, pooled (equivalence) and per bin / surface (reported)
    PMt = d3["PM"]
    truth = {"d0": T["tp_d0"], "z0": T["tp_z0"]}
    h_pm = FM.harm(PMt["d0"], PMt["z0"], T["ot_d0"], T["ot_z0"], truth)["either"]
    h_v2 = FM.harm(T["sp_d0"], T["sp_z0"], T["ot_d0"], T["ot_z0"], truth)["either"]
    dh = np.bincount(T["ev"][pr].astype(np.int64), weights=(h_pm[pr].astype(float) - h_v2[pr]), minlength=NEV)
    rec_t, rec_j = np.nonzero(PMt["rec_acc"])
    ktp_wrong_pm = d3["PM/wrong"]["w"] if "PM/wrong" in d3 else None
    b = {}
    r = dh.sum() / max(den_h.sum(), 1)
    rb = (rep @ dh) / np.maximum(rep @ den_h, 1)
    sd = float(np.std(rb))
    b["H"] = {"shift": float(r), "sd": sd, "bound95": abs(float(r)) + 1.645 * sd, "margin": 0.003,
              "pass": abs(float(r)) + 1.645 * sd <= 0.003}
    if ktp_wrong_pm is not None:
        wpm = np.bincount(d3["PM/wrong"]["ev"].astype(np.int64), weights=ktp_wrong_pm, minlength=NEV)
        apm = np.bincount(d3["PM/wrong"]["ev"].astype(np.int64), minlength=NEV).astype(float)
        wv2 = np.bincount(Hh["ev"][popm].astype(np.int64), weights=(Hh["cls"][popm] != 0), minlength=NEV)
        p_pm, p_v2 = wpm.sum() / max(apm.sum(), 1), wv2.sum() / max(den_p.sum(), 1)
        pb_ = (rep @ wpm) / np.maximum(rep @ apm, 1) - (rep @ wv2) / np.maximum(rep @ den_p, 1)
        sd = float(np.std(pb_))
        b["P"] = {"shift": float(p_pm - p_v2), "sd": sd, "bound95": abs(p_pm - p_v2) + 1.645 * sd, "margin": 0.002,
                  "pass": abs(p_pm - p_v2) + 1.645 * sd <= 0.002}
    b["H_per_bin"] = {}
    aeta, pt = np.abs(T["tp_eta"]), T["tp_pt"]
    for (elo, ehi), (plo, phi) in FM.H_BINS:
        m = pr & (aeta >= elo) & (aeta < ehi) & (pt >= plo) & (pt < phi)
        if m.sum():
            b["H_per_bin"][FM.h_key((elo, ehi), (plo, phi))] = {
                "shift": float(np.mean(h_pm[m].astype(float) - h_v2[m])), "n": int(m.sum())}
    res["ii_iii_b"] = b
    return res


# ---------------------------------------------------------------- E7
def e7(d, pt_bins=(2.0, 3.0, 5.0, 10.0, np.inf)):
    """K's skip-loss: tracks whose K = 1 pick at the first-visited surface is the own primary; the fraction
    whose K = 2 + skip winner is another branch; per first-visited surface x pT bin and pooled. And the
    wrong-hit rejection gained: K = 1 picks that are wrong, replaced by the own primary or by skip."""
    k1, k2, own = d["INN"], d["INN-K2"], d["INN/own"]
    first_ci = k1["rec_ci"][:, 0]
    acc1 = k1["rec_acc"][:, 0].astype(bool)
    surf = k1["rec_surface"][:, 0]
    prim = own["primary"] & (own["step"] == 0)
    key = lambda o, c: o.astype(np.int64) * (1 << 31) + c
    is_prim = np.isin(key(np.arange(len(first_ci)), first_ci), key(own["o"][prim], own["ci"][prim])) & acc1
    T = d["cmssw_trk"]
    loss = is_prim & (k2["branch"] != 0)
    pt = T["ot_pt"]
    res = {"pooled": {"n": int(is_prim.sum()), "loss": float(loss.sum() / max(is_prim.sum(), 1))}, "bins": {}}
    rep = replicas()
    ev = T["ev"].astype(np.int64)
    num = np.bincount(ev, weights=loss, minlength=NEV)
    den = np.bincount(ev, weights=is_prim, minlength=NEV)
    res["pooled"]["sd"] = float(np.std((rep @ num) / np.maximum(rep @ den, 1)))
    res["pooled"]["lower95"] = res["pooled"]["loss"] - 1.645 * res["pooled"]["sd"]
    for S in np.unique(surf[is_prim]):
        for lo, hi in zip(pt_bins[:-1], pt_bins[1:]):
            m = is_prim & (surf == S) & (pt >= lo) & (pt < hi)
            if m.sum():
                n_ = np.bincount(ev[m], weights=loss[m], minlength=NEV)
                d_ = np.bincount(ev[m], minlength=NEV).astype(float)
                res["bins"][f"{int(S)}|{lo:g}-{hi:g}"] = {"n": int(m.sum()), "loss": float(loss[m].mean()),
                                                         "sd": float(np.std((rep @ n_) / np.maximum(rep @ d_, 1)))}
    # the wrong-hit rejection gained: K = 1 picks that are wrong, replaced by an own hit or by no hit
    w1 = k1["first_wrong"].astype(bool)
    gain = w1 & ~k2["first_wrong"].astype(bool)
    res["wrong_hit_rejection_gained"] = {"n_k1_wrong": int(w1.sum()), "replaced": float(gain.sum() / max(w1.sum(), 1)),
                                         "by_own": float((gain & k2["first_own"].astype(bool)).sum() / max(w1.sum(), 1))}
    return res


E7_LIMIT = 0.015      # PD2: the pooled skip-loss limit


def e7_compare(r3, r4):
    """PD2: file 4's pooled loss <= 1.5 % (else the design failed); per bin |file 3 - file 4| <= t(N)
    combined sigma; file 3's pooled one-sided 95 % lower bound <= 1.5 %."""
    common = [k for k in r3["bins"] if k in r4["bins"]]
    t = t_two(3 * len(common)) if common else 0.0
    fails = [k for k in common if abs(r3["bins"][k]["loss"] - r4["bins"][k]["loss"])
             > t * np.hypot(r3["bins"][k]["sd"], r4["bins"][k]["sd"])]
    return {"file4_pooled": r4["pooled"]["loss"], "file4_ok": r4["pooled"]["loss"] <= E7_LIMIT,
            "file3_pooled": r3["pooled"]["loss"], "file3_lower95": r3["pooled"]["lower95"],
            "file3_ok": r3["pooled"]["lower95"] <= E7_LIMIT, "bin_fails": fails, "t": t}


def e7_kb(d, kb):
    """The tail-probability comparison (the PD2 known-bad) on the same tracks: its pooled skip-loss."""
    dd = dict(d)
    dd["INN-K2"] = kb
    return e7(dd)["pooled"]


# ---------------------------------------------------------------- E5 (iv), PM-B, the mirrors' known-bads
def e5_iv(d3, d4):
    """PM's final parameters on its tracks whose selections all agree with CMSSW, against the forced
    replay with the cluster table's inputs on file 4 (PD11), with E4's rule; plus the d0 / z0 threshold
    fractions of PM and refit v2 (sheet fom.thresholds), paired, reported."""
    T = d3["cmssw_trk"]
    PMt = d3["PM"]
    dis_o = d3["PM/e5_first"]["o"] if "PM/e5_first" in d3 else np.zeros(0, np.int64)
    agree = PMt["refit"].astype(bool) & ~np.isin(np.arange(len(PMt["refit"])), dis_o) & np.isfinite(T["sp_cov_d0_d0"])
    x3 = e4_stats(PMt, T, agree)
    s4 = d4["forced-cluster"]["has_rows"].astype(bool) & np.isfinite(d4["cmssw_trk"]["sp_cov_d0_d0"])
    x4 = e4_stats(d4["forced-cluster"], d4["cmssw_trk"], s4)
    W = t_one(45)
    out = {"n_agree": int(agree.sum()), "params": {}}
    for nm in x3:
        a, b = x3[nm][np.isfinite(x3[nm])], x4[nm][np.isfinite(x4[nm])]
        e = {}
        for qq in (50, 99):
            va, vb = float(np.percentile(a, qq)), float(np.percentile(b, qq))
            sda = float(np.std([np.percentile(np.random.default_rng(i).choice(a, len(a)), qq) for i in range(50)]))
            sdb = float(np.std([np.percentile(np.random.default_rng(100 + i).choice(b, len(b)), qq) for i in range(50)]))
            e[f"p{qq}"] = {"pm": va, "known_good": vb, "pass": va <= vb + W * np.hypot(sda, sdb)}
        lim = float(np.percentile(b, 99.9))
        f = float(np.mean(a > lim))
        sd = float(np.sqrt(0.001 * 0.999 / max(len(a), 1)))
        e["beyond_p999"] = {"pm": f, "limit": 0.001 + W * sd, "pass": f <= 0.001 + W * sd}
        e["pass"] = e["p50"]["pass"] and e["p99"]["pass"] and e["beyond_p999"]["pass"]
        out["params"][nm] = e
    thr = {}
    for p, lims in (("d0", (0.01, 0.02, 0.05, 0.1, 0.5)), ("z0", (0.05, 0.1, 0.5))):
        rp = np.abs(PMt[p] - T[f"tp_{p}"])
        rv = np.abs(T[f"sp_{p}"] - T[f"tp_{p}"])
        pop = T["tp_pop"].astype(bool)
        for L in lims:
            thr[f"{p}>{L * 10:g}mm"] = {"pm": float(np.mean(rp[pop] > L)), "v2": float(np.mean(rv[pop] > L))}
    out["thresholds"] = thr
    return out


def pmb_identity(d3):
    """PM-B bit-identical to PM on the tracks whose PM visit list is barrel-only (PLAN §4 E5)."""
    PMt, PB = d3["PM"], d3["PM-B"]
    vis = PMt["rec_visited"].astype(bool)
    barrel_only = ~(vis & (PMt["rec_surface"] > 10)).any(1)
    ns = min(PMt["rec_ci"].shape[1], PB["rec_ci"].shape[1])
    same = barrel_only.copy()
    for p in ("rInv", "phi", "tanL", "z0", "d0"):
        same &= (PMt[p] == PB[p]) | ~barrel_only
    same &= (PMt["rec_ci"][:, :ns] == PB["rec_ci"][:, :ns]).all(1) | ~barrel_only
    return {"barrel_only_tracks": int(barrel_only.sum()), "identical": int(same.sum()),
            "pass": bool(same.sum() == barrel_only.sum())}


def e5_mirror_kb(d3, d4, name):
    """A known-bad mirror read with E5 (i)'s rules: it must fail a rate test or the non-near-tie bound."""
    W1 = t_one(24)
    obs = e5_rates(d3[f"{name}/e5_risk"])
    pred = e5_rates(d4["e5_model_risk"], draws=10)
    fail_rate = [S for S, o in obs.items()
                 if o["rate"] > pred.get(S, {"rate": 0.0})["rate"] + W1 * np.hypot(o["sd"], pred.get(S, {"sd": 0.0})["sd"])]
    first = d3[f"{name}/e5_first"]
    n_dis = len(first["cause"])
    n_out = int((first["cause"] >= 1).sum())
    lim = float(stats.poisson.ppf(0.99, max(0.002 * len(d3[f"{name}/e5_risk"]["surface"]), 1e-9)))      # PD12
    return {"rate_fail_surfaces": fail_rate, "disagreements": n_dis, "outside_window": n_out, "bound": lim,
            "fails": bool(fail_rate or n_out > lim)}


# ---------------------------------------------------------------- E1(b) known-bads
def e1b_known_bads(own, loo, cuts_doc_975, cov="calibrated"):
    """(1) a 3-sigma cut on the statistic in Gaussian terms (chi2 with the consumer's dims at 99.73 %;
    |z| <= 3 for the window); (2) the 97.5 % quantile used as the 99 % one (cuts refitted at 0.975 on the
    fit rows). Each must fail the per-bin or the pooled test."""
    out = {}
    gate, window, rr = CU.populations(own, {"own_primary": loo["own_primary"], "clean_track": loo["clean_track"]})
    for dims, k in (("xy", 2), ("xyab", 4)):
        cut = stats.chi2.ppf(0.9973, k)
        x = own[f"g_{dims}"][gate]
        out[f"gate|{dims}|3sigma"] = {"coverage": float(np.mean(x <= cut)), "fails": bool(np.mean(x <= cut) < 0.985)}
    for i, ang in enumerate(SA.ANGLES):
        x = own["z"][window, i]
        x = x[np.isfinite(x)]
        out[f"window|{ang}|3sigma"] = {"coverage": float(np.mean(x <= 3.0)), "fails": bool(np.mean(x <= 3.0) < 0.985)}
    if cuts_doc_975 is not None:
        c975 = CU.Cuts(cuts_doc_975)
        r = e1b_consumer(c975, "gate", f"gate|xy|{cov}", own["g_xy"][gate], own["surface"][gate], own["first"][gate],
                         own["ev"][gate])
        out["gate|xy|q97.5"] = {"pooled": r["pooled"], "max_abs_z": r["max_abs_z"], "t": r["t"],
                                "fails": bool((not r["pooled"]["pass"]) or (r["max_abs_z"] or 0) > r["t"])}
    return out
