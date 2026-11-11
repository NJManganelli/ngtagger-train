"""Fit the shipped parametrization on ALL scan events and write it as correctionlib JSON, one file per scenario
(PLAN.md section 7b): doc-archive/spix-bits-refit-study/parametrization/spixbits_<scenario>.json.gz, with the
reference helper spixbits_apply.py and a consistency check (helper through correctionlib == the fit code's numpy
evaluation, the same draws given to both).

  python param_export.py [--scen AAAA AAAI IAAA] [--root DIR] [--out DIR] [--allocs ...]

Corrections in a file (inputs in SmartPixels-study units: cm, GeV, 1/cm; `allocation` is one of the shipped names,
anything else raises):
  uniform, normal   the per-track draws: hashprng of (run, lumi, event, track, draw); draw 1 = state, 2 = rank,
                    3 = score (uniform); 11-14 = z0, tanL, 1/R, phi0 (normal)
  p_changed         share of tracks changing clusters (or refit flag) against today
  eps               inverse CDF of the same-cluster parameter change in units of today's sigma
  lambda, cov_scale, rho     its scale, the covariance-diagonal ratio, the copula correlations
  score_map, score_eta       same-cluster map of the head outputs and chi2-like inputs: coefficients and residual
  donor             the changed state: the donor's values, picked by the rank draw
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

import param_model as PM

OUT = Path.home() / "smartpixels/doc-archive/spix-bits-refit-study/parametrization"
INF = "inf"
V = {"allocation": ("string", "shipped allocation name"), "parameter": ("string", "rInv, phi0, tanL, z0 or d0"),
     "matched": ("int", "1 if the OT track has a tracking particle (otTpIdx >= 0), else 0"),
     "window": ("int", "today's window: 0 prompt (|d0| <= 0.02 cm), 1 displaced, 2 not refit today"),
     "window_arm": ("int", "window of the EMULATED d0: 0 prompt, 1 displaced"),
     "abs_eta": ("real", "|eta| of today's refit track (asinh of tanL)"), "pt": ("real", "today's refit pT [GeV]"),
     "mult": ("real", "today's spixMaxWindowMult"),
     "pull": ("real", "max(|d0 - truth d0| / sigma_d0, |z0 - truth z0| / sigma_z0) of today's track; 0 for a track without a particle"), "clean_margin": ("real", "today's clean head as a margin: logit of head_clean"),
     "u": ("real", "a uniform draw in [0, 1)"), "tail": ("real", "|2 u_rank - 1|"),
     "score": ("string", "head_clean (margin), head_d0, head_z0, head_tanl, head_rinv, head_phi (log sigma_hat) or a chi2-like column"),
     "coefficient": ("string", "a, b (or sigma for score_map)"), "quantity": ("string", "donor quantity"),
     "run": ("int", "run"), "lumi": ("int", "luminosity block"), "event": ("int", "event"),
     "track": ("int", "index of the track along axis 1 of the event's L1TSmartPixelsExtTrack collection"),
     "draw": ("int", "which draw (see the file description)")}


def var(n):
    return {"name": n, "type": V[n][0], "description": V[n][1]}


def binning(name, edges, content, flow="clamp"):
    return {"nodetype": "binning", "input": name, "edges": [float(e) if e != INF else INF for e in edges], "content": content, "flow": flow}


def category(name, items):
    return {"nodetype": "category", "input": name, "content": [{"key": k, "value": v} for k, v in items]}


def ints(name, n, fn):
    return category(name, [(i, fn(i)) for i in range(n)])


def eta_bins(fn):
    return binning("abs_eta", [0.0, *PM.ETA_EDGES, INF], [fn(i) for i in range(3)])


def pt_bins(fn):
    return binning("pt", [0.0, *PM.PT_EDGES, INF], [fn(i) for i in range(3)])


def inv_cdf_node(q):
    """Piecewise-linear inverse CDF in u with constant ends (the clamp)."""
    k = PM.KNOTS
    content = [float(q[0])]
    for i in range(len(k) - 1):
        slope = (q[i + 1] - q[i]) / (k[i + 1] - k[i])
        content.append({"nodetype": "formula", "expression": "[0]+[1]*x", "parser": "TFormula", "variables": ["u"],
                        "parameters": [float(q[i] - slope * k[i]), float(slope)]})
    content.append(float(q[-1]))
    return binning("u", [0.0, *k, 1.0], content)


def correction(name, desc, inputs, data, out="value"):
    return {"name": name, "description": desc, "version": 1, "inputs": [var(i) for i in inputs],
            "output": {"name": out, "type": "real"}, "data": data}


def build(tables, terc, meta):
    """The CorrectionSet dict from {allocation: tables of param_model.fit}."""
    al = list(tables)
    per = lambda fn: category("allocation", [(a, fn(tables[a])) for a in al])
    ids = ["run", "lumi", "event", "track", "draw"]
    cs = [correction("uniform", "uniform [0, 1) draw per track", ids, {"nodetype": "hashprng", "inputs": ids, "distribution": "stdflat"}),
          correction("normal", "standard normal draw per track", ids, {"nodetype": "hashprng", "inputs": ids, "distribution": "stdnormal"})]
    terc_edges = ["-inf", float(terc[0]), float(terc[1]), INF]
    cs.append(correction("p_changed", "share of tracks whose accepted clusters or refit flag differ from today's",
                         ["allocation", "matched", "window", "abs_eta", "pt", "mult", "clean_margin", "pull"],
                         per(lambda T: ints("matched", 2, lambda m: ints("window", 3, lambda w: eta_bins(lambda e: pt_bins(
                             lambda p: binning("mult", [0.0, *PM.MULT_EDGES, INF], [binning("clean_margin", terc_edges, [binning("pull", [0.0, *PM.PULL_EDGES, INF],
                                                                                                                         [float(T["p"][m, w, e, p, k, t, q]) for q in range(3)])
                                                                                                                 for t in range(3)])
                                                                                for k in range(4)]))))))))
    par = lambda fn: category("parameter", [(p, fn(p)) for p in PM.PARAMS])
    cs.append(correction("eps", "same-cluster change of a parameter in units of the sigma the allocation adds on the track, "
                                "sqrt(|C_arm - C_today| + (0.05 sigma_today)^2) with C_arm from cov_scale: inverse CDF at u",
                         ["allocation", "parameter", "abs_eta", "window", "u"],
                         per(lambda T: par(lambda p: eta_bins(lambda e: ints("window", 2, lambda w: inv_cdf_node(T["kernel"][p][e, w])))))))
    cs.append(correction("lambda", "scale of eps so that the 68 % residual width matches the forced refit's",
                         ["allocation", "parameter", "window", "abs_eta", "pt"],
                         per(lambda T: par(lambda p: ints("window", 2, lambda w: eta_bins(lambda e: pt_bins(
                             lambda i: float(T["lambda"][p][w, e, i]))))))))
    cs.append(correction("cov_scale", "same-cluster covariance-diagonal ratio arm / today = exp(a + b ln sigma_today), clipped to "
                                      "0.2-10000: the coefficient a or b", ["allocation", "parameter", "coefficient", "window", "abs_eta", "pt"],
                         per(lambda T: par(lambda p: category("coefficient", [(c, ints("window", 2, lambda w: eta_bins(lambda e: pt_bins(
                             lambda i: float(T["cov_scale"][p][w, e, i, k]))))) for k, c in enumerate(("a", "b"))])))))
    cs.append(correction("rho", "Gaussian-copula correlation of a parameter's draw with its partner's (z0 with d0, tanL with z0, "
                                "rInv and phi0 with d0); d0 has none", ["allocation", "parameter", "abs_eta", "window"],
                         per(lambda T: category("parameter", [(p, eta_bins(lambda e: ints("window", 2, lambda w: float(T["rho"][p][e, w]))))
                                                              for p in PM.DRAW_ORDER[1:]]))))
    sc = lambda fn: category("score", [(s, fn(s)) for s in PM.SCORES])
    cs.append(correction("score_map", "same-cluster map s' = a + b s + sigma eta of a score in its natural variable",
                         ["allocation", "score", "coefficient", "window_arm", "abs_eta"],
                         per(lambda T: sc(lambda s: category("coefficient", [(c, ints("window_arm", 2, lambda w: eta_bins(
                             lambda e: float(T["map"][s][w, e, i])))) for i, c in enumerate(("a", "b", "sigma"))])))))
    cs.append(correction("score_eta", "standardized residual of the same-cluster score map: inverse CDF at u, by |2 u_rank - 1|",
                         ["allocation", "score", "tail", "u"],
                         per(lambda T: sc(lambda s: binning("tail", [0.0, *PM.TAIL_EDGES, 1.0], [inv_cdf_node(T["eta"][s][k]) for k in range(3)])))))
    dq = []
    for grp, names in (("matched_ratio", PM.PARAMS), ("delta_ratio", PM.PARAMS), ("cov_ratio", PM.PARAMS), ("d_margin", PM.HEADS),
                       ("counts", PM.COUNTS), ("chi2", PM.CHI2)):
        dq += [(f"{grp}_{n}" if grp not in ("counts", "chi2") else n, grp, i) for i, n in enumerate(names)]
    edges = [float(x) for x in np.linspace(0, 1, PM.N_DONORS + 1)]
    r6 = lambda x: float(f"{x:.6g}")
    cs.append(correction("donor", "changed state: the donor's value, picked by the rank draw u. matched_ratio_<p>: (arm - truth) / "
                                  "sigma_today; delta_ratio_<p>: (arm - today) / sigma_today; cov_ratio_<p>: arm / today covariance diagonal; "
                                  "d_margin_<head>: change of the head output; the spix* columns: the arm's own values",
                         ["allocation", "quantity", "matched", "window", "abs_eta", "pull", "u"],
                         per(lambda T: category("quantity", [(q, ints("matched", 2, lambda m: ints("window", 3, lambda w: eta_bins(
                             lambda e: binning("pull", [0.0, *PM.PULL_EDGES, INF], [binning("u", edges, [r6(v) for v in T["donors"][g][m, w, e, j, :, i]])
                                                                                    for j in range(3)])))))
                                                             for q, g, i in dq]))))
    return {"schema_version": 2, "description": json.dumps(meta), "corrections": cs}


def main():
    import ext_scan as XS
    import param_closure as PC
    import study as ST
    ap = argparse.ArgumentParser()
    ap.add_argument("--scen", nargs="+", default=list(ST.SCENARIOS))
    ap.add_argument("--allocs", nargs="+", default=None)
    ap.add_argument("--files", type=int, nargs="+", default=list(ST.SCAN_FILES))
    ap.add_argument("--root", default="")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    import subprocess
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(a.root) if a.root else XS.out_root()
    allocs = a.allocs or [x for x in XS.SHIPPED if x != "implemented_now"]
    commit = subprocess.run(["git", "-C", str(Path(__file__).parent), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    payload = json.load(open(ST.CACHE / "checksums.json"))
    shutil.copy(Path(__file__).parent / "spixbits_apply.py", out / "spixbits_apply.py")
    summary = {}
    for scen in a.scen:
        T, today, arms, truth, m = PC.load(root, scen, a.files, allocs)
        D = PM.draws(T["run"], T["lumi"], T["event"], T["track"])
        terc = np.quantile(today["margins"][today["refit"], 0], [1 / 3, 2 / 3])
        allsel = np.ones(len(T["run"]), bool)
        tables = {al: PM.fit(today, arms[al], truth, D, terc, allsel) for al in allocs}
        meta = {"what": "SmartPixels output bits -> track refit: per-track parametrization of shipped bit allocations relative to "
                        "today's production (PLAN.md section 7b)", "scenario": scen, "allocations": allocs,
                "fit_sample": f"canon_v5 TTbar files {list(a.files)}, extended collection, every track ({len(T['run'])} tracks, "
                              f"{len(np.unique(T['event_key']))} events)", "scan_commit": commit,
                "payload_refit_calibration_sha256": payload["refit_calibration_v3.json"]["sha256"],
                "heads": "heads_cmssw reference models", "kpt_GeV_cm": m["kpt"], "rinv_sign_times_charge": m["rinv_sign"],
                "clean_margin_tercile_edges": [float(x) for x in terc], "knots": [float(k) for k in PM.KNOTS],
                "n_donors": PM.N_DONORS, "eps_unit": PM.EPS_UNIT, "donor_form": PM.DONOR_FORM}
        doc = build(tables, terc, meta)
        path = out / f"spixbits_{scen}.json.gz"
        with gzip.open(path, "wt") as fh:
            json.dump(doc, fh)
        # consistency: the helper through correctionlib against the numpy evaluation, the same draws given to both
        import correctionlib
        import spixbits_apply as SA
        cset = correctionlib.CorrectionSet.from_file(str(path))
        rng = np.random.default_rng(1)
        pick = np.sort(rng.choice(len(T["run"]), min(20000, len(T["run"])), replace=False))
        sub = lambda d: {k: (v[pick] if isinstance(v, np.ndarray) else v) for k, v in d.items()}
        worst = {}
        for al in (allocs[:3] + allocs[-1:]):
            ref = PM.apply(tables[al], sub(today), truth[pick], sub(D), terc)
            got = SA.apply(cset, al, a=today["a"][pick], cov_diag=today["cov_diag"][pick], refit=today["refit"][pick],
                           margins=today["margins"][pick], columns={c: today["inputs"][pick, PM.IN[c]] for c in PM.COUNTS + PM.CHI2},
                           truth=truth[pick], kpt=m["kpt"], draws=sub(D))
            sg_ = np.sqrt(today["cov_diag"][pick])                 # parameters: difference in units of today's sigma
            d = [float(np.nanmax(np.abs(PM.residual(got["a"].copy(), ref["a"])) / sg_)),
                 float(np.nanmax(np.abs(got["cov_diag"] / ref["cov_diag"] - 1))),
                 float(np.nanmax(np.abs(got["margins"] - ref["margins"]))), float(np.mean(got["changed"] != ref["changed"]))]
            d += [float(np.nanmax(np.abs(got[c] - ref[c]) / (np.abs(ref[c]) + 1e-6))) for c in PM.COUNTS + PM.CHI2]
            worst[al] = max(d)
        # the hashed draws of the file itself: usable, and uniform
        u = cset["uniform"].evaluate(T["run"][pick], T["lumi"][pick], T["event"][pick], T["track"][pick].astype(np.int64), 2)
        summary[scen] = {"file": str(path), "bytes": path.stat().st_size, "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
                         "helper_vs_numpy_max_difference (parameters in today's sigma, others relative)": worst, "hashprng_uniform_mean": float(np.mean(u)),
                         "tracks": int(len(T["run"])), "meta": meta}
        print(scen, json.dumps({k: v for k, v in summary[scen].items() if k != "meta"}, indent=1), flush=True)
    json.dump(summary, open(out / "export_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
