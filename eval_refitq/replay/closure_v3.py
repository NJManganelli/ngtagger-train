"""The v3 CMSSW closure, the replay side (v3cmssw/PLAN.md rev 1 §3 items 4-5).

  python closure_v3.py --nano <closure nano> --dump <CMSSW double dump .jsonl> --out <result .json>

The closure arm is arms.v3 (geometry-only window modules) with kpt = the producer's constant (WPC_KPT=cmssw, set
here). Tracks are paired by (run, lumi, event, track index along axis 1 of the event's OT track collection), never by
their index along axis 0 of a file.

Item 5, selection (free-running): per visited step, the accepted flag and the cluster (detId, index along axis 1 of the
event's cluster collection). Each track is judged at its first disagreement. Its margin is the committing two-layer
score margin (selMargin) of the window that disagrees. eps is frozen (--eps, 3.9e-5 from the dev events), with a
floor of 1e-6 (1 + |score|). A first disagreement with margin > eps fails; near-ties are capped at 0.1 % of crossings.
Score check: on tracks whose choices all agree, the crossings whose committing scores differ by more than eps are also
capped at 0.1 % of crossings. Also reported: the crossing agreement and, on tracks with identical choices, the final state and
covariance (|da_i| / sigma_i, |dC_ij| / sqrt(C_ii C_jj)).

Item 4, arithmetic (forced, double): the replay's forced mode on CMSSW's chosen clusters. At every accepted update, the
predicted (aPre / CPred) and filtered (aPost / CPost) state and covariance, and the final state with its inner material,
against the dump: |da_i| / sigma_i <= 1e-6 and |dC_ij| / sqrt(C_ii C_jj) <= 1e-6.

--inputs dump (the default, the plan's "both from the double closure dump"): the OT helix of every paired track is the
dump's double one. The nano stores rInv and phi0 in float32: one unit in the last place, about 6e-8 relative, which a
refit's shrunken sigma(phi0) turns into up to ~5e-4 sigma. --inputs nano: the nano's values, reported to show that.

--export DIR: also the closure arm's per-track output in the spec reference's format (DIR/<scen>_f<NN>.jsonl and
_forced.jsonl), for the port's cross-check.

--geometry f32 (the default): the module origins and frames rounded to float32, as CMSSW holds them (GeomDet surfaces).
The geometry file's 10-digit text, read as double, moves a projection by up to 1e-8 cm (2.4e-5 of the cluster sigma),
which shows up as up to 9e-6 sigma in the state. --geometry file: the file's values as read.
"""
from __future__ import annotations

import argparse
import json
import os

os.environ["WPC_KPT"] = "cmssw"                     # before common reads it

import numpy as np                                   # noqa: E402

import arms as AR                                    # noqa: E402
import common as CM                                  # noqa: E402
import cuts as CU                                    # noqa: E402
from common import OP                                # noqa: E402

TOL = 1e-6
NEAR_TIE_CAP = 0.001
PARAMS = ("rInv", "phi0", "tanL", "z0", "d0")


def _rel(a_rep, a_ref, C_ref):
    """max |da_i| / sigma_i (sigma from the reference covariance's diagonal)."""
    s = np.sqrt(np.clip(np.diag(C_ref), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.abs(np.asarray(a_rep) - np.asarray(a_ref)) / s
    return float(np.nanmax(np.where(s > 0, x, 0.0)))


def _relC(C_rep, C_ref):
    """max |dC_ij| / sqrt(C_ii C_jj) (the reference's diagonal)."""
    C_rep, C_ref = np.asarray(C_rep, float), np.asarray(C_ref, float)
    d = np.sqrt(np.clip(np.diag(C_ref), 0, None))
    den = np.outer(d, d)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.abs(C_rep - C_ref) / den
    return float(np.nanmax(np.where(den > 0, x, 0.0)))


def float32_geometry(geo):
    """The geometry with the module origins and frames rounded to float32 (CMSSW's GeomDet precision)."""
    g = dict(geo)
    for k in ("origin", "ex", "ey", "ez"):
        g[k] = np.asarray(geo[k]).astype(np.float32).astype(np.float64)
    return g


def run(a):
    geo = OP.load_geometry(CM.GEOMETRY)
    if a.geometry == "f32":
        geo = float32_geometry(geo)
    cfg = CM.producer_config(a.nano)
    scen = CM.mask_of(cfg)
    with open(CM.OUT / "c1b" / "scale" / "scalefit_summary.json") as fh:
        scale = float(json.load(fh)["scale"])
    tools = AR.Tools(cfg, cuts=CU.Cuts.load(str(CM.OUT / "c1b" / "cuts" / "wpc_cuts_c1b_v1.json")),
                     disc=AR.DiscTools(disc_scale=scale))
    arm = AR.v3(cfg)
    ref = {}
    with open(a.dump) as fh:
        for line in fh:
            r = json.loads(line)
            ref[(r["run"], r["lumi"], r["event"], r["track"])] = r
    br = list(dict.fromkeys(AR.branches(a.nano) + ["run", "luminosityBlock", "event"]))
    sel = {"crossings": 0, "agree": 0, "tracks": 0, "tracks_identical": 0, "first": [], "dscore": [], "score_over": 0,
           "final_da": [], "final_dC": []}
    frc = {"updates": 0, "max_da_pre": 0.0, "max_dC_pre": 0.0, "max_da_post": 0.0, "max_dC_post": 0.0,
           "max_da_final": 0.0, "max_dC_final": 0.0, "tracks": 0, "worst": []}
    unpaired = 0
    fh = ff = None
    if a.export:
        import export_v3 as X
        os.makedirs(a.export, exist_ok=True)
        stem = os.path.join(a.export, os.path.basename(a.nano).replace(".root", ""))
        fh, ff = open(stem + ".jsonl", "w"), open(stem + "_forced.jsonl", "w")
    for start, A in CM.iterate(a.nano, br, chunk=a.chunk):
        ctx = AR.prepare(A, geo)
        kpt = CM.pt_constant(ctx)
        tev = ctx["tev"]
        first = np.r_[0, np.cumsum(np.bincount(tev, minlength=len(A)))[:-1]]
        ids = [(int(A["run"][e]), int(A["luminosityBlock"][e]), int(A["event"][e])) for e in range(len(A))]
        if a.inputs == "dump":
            for t in range(ctx["ntrk"]):
                r = ref.get(ids[int(tev[t])] + (int(t - first[int(tev[t])]),))
                if r is not None:
                    ctx["a"][t] = [r["ot"][p] for p in PARAMS]
        out = AR.run_arm(ctx, geo, cfg, scen, arm, tools)
        rec, seq = out["rec"], out["seq"]
        if fh is not None:
            X.write_chunk(fh, ff, A, ctx, geo, cfg, scen, arm, tools, out)
        # item 5: the replay's free-running choices against CMSSW's
        for t in range(ctx["ntrk"]):
            e = int(tev[t])
            key = ids[e] + (int(t - first[e]),)
            r = ref.get(key)
            if r is None:
                unpaired += 1
                continue
            sel["tracks"] += 1
            cs = {s["step"]: s for s in r["steps"]}
            first_bad = None
            dsc = []
            for j in range(seq.shape[1]):
                if seq[t, j] < 0:
                    continue
                c = cs.get(j)
                if c is None or c["surface"] != int(seq[t, j]):
                    first_bad = first_bad or {"step": j, "why": "visit plan", "margin": None}
                    continue
                if not (rec["visited"][t, j] or c["visited"]):
                    continue
                sel["crossings"] += 1
                acc = bool(rec["acc"][t, j])
                ci = int(rec["ci"][t, j]) - int(ctx["coff"][e]) if acc else -1
                same = acc == c["accepted"] and (not acc or ci == c["cluster"])
                sel["agree"] += same
                ks = rec["kscores"][t, j]
                if same and c["scores"] is not None and np.isfinite(ks).any():
                    cmssw = np.array([np.nan if x is None else x for x in c["scores"]], float)
                    m = np.isfinite(cmssw) & np.isfinite(ks)
                    if m.any():
                        dsc.append(float(np.max(np.abs(cmssw[m] - ks[m]))))
                if not same and first_bad is None:
                    fin = np.sort(ks[np.isfinite(ks)])
                    rep_margin = float(fin[1] - fin[0]) if len(fin) >= 2 else None
                    margins = [x for x in (c["selMargin"], rep_margin) if x is not None]
                    sc = [x for x in (c["scores"] or []) if x is not None]
                    first_bad = {"step": j, "surface": int(seq[t, j]), "first_step": j == 0,
                                 "kchoice_cmssw": c["kchoice"], "kchoice_replay": int(rec["kchoice"][t, j]),
                                 "cluster_cmssw": c["cluster"], "cluster_replay": ci,
                                 "margin": min(margins) if margins else None,
                                 "score": float(min(sc)) if sc else None}
            if first_bad is not None:
                first_bad["track"] = list(key)
                sel["first"].append(first_bad)
            else:
                sel["tracks_identical"] += 1
                sel["dscore"] += dsc
                sel["score_over"] += sum(x > a.eps for x in dsc)
                if r["refit"]:
                    sel["final_da"].append(_rel([out["a"][t][i] for i in range(5)], [r["final"][p] for p in PARAMS],
                                                r["final_C"]))
                    sel["final_dC"].append(_relC(out["C"][t], r["final_C"]))
        # item 4: the forced mode on CMSSW's chosen clusters
        keys = [ids[int(tev[t])] + (int(t - first[int(tev[t])]),) for t in range(ctx["ntrk"])]
        o = np.array([t for t in range(ctx["ntrk"]) if keys[t] in ref and ref[keys[t]]["refit"]], np.int64)
        if not len(o):
            continue
        path, seq_ = AR.visit_plan(ctx, geo, cfg, scen, arm)
        sel_mi = np.full(seq_[o].shape, -1, np.int64)
        sel_ci = np.full(seq_[o].shape, -1, np.int64)
        for i, t in enumerate(o):
            e = int(tev[t])
            for s in ref[keys[t]]["steps"]:
                if s["accepted"]:
                    sel_mi[i, s["step"]] = OP.module_index(geo, np.array([s["detId"]], np.int64))[0]
                    sel_ci[i, s["step"]] = int(ctx["coff"][e]) + s["cluster"]
        F = AR.refit_forced(arm, ctx, geo, cfg, tools, o, seq_[o], sel_mi, sel_ci, path, kpt, history=True)
        AR.inner_material(arm, F, geo, tools, kpt)
        for i, t in enumerate(o):
            r = ref[keys[t]]
            frc["tracks"] += 1
            worst = 0.0
            for s in r["steps"]:
                if not s["accepted"]:
                    continue
                j = s["step"]
                frc["updates"] += 1
                d = {"max_da_pre": _rel(F.hist["pa"][i, j], s["aPre"], s["CPred"]),
                     "max_dC_pre": _relC(F.hist["pC"][i, j], s["CPred"]),
                     "max_da_post": _rel(F.hist["fa"][i, j], s["aPost"], s["CPost"]),
                     "max_dC_post": _relC(F.hist["fC"][i, j], s["CPost"])}
                for k, v in d.items():
                    frc[k] = max(frc[k], v)
                worst = max(worst, *d.values())
            da = _rel(F.a[i], [r["final"][p] for p in PARAMS], r["final_C"])
            dC = _relC(F.C[i], r["final_C"])
            frc["max_da_final"], frc["max_dC_final"] = max(frc["max_da_final"], da), max(frc["max_dC_final"], dC)
            worst = max(worst, da, dC)
            if worst > TOL:
                frc["worst"].append({"track": list(keys[t]), "worst": worst})
    if fh is not None:
        fh.close()
        ff.close()
    # verdicts
    eps = a.eps
    cap = NEAR_TIE_CAP * sel["crossings"]
    for f in sel["first"]:
        f["eps"] = max(eps, 1e-6 * (1 + abs(f["score"] or 0.0)))
        f["near_tie"] = f["margin"] is not None and f["margin"] <= f["eps"]
    hard = [f for f in sel["first"] if not f["near_tie"]]
    near = [f for f in sel["first"] if f["near_tie"]]
    res = {"plan": "v3cmssw/PLAN.md rev 1 §3 items 4-5", "nano": a.nano, "dump": a.dump, "scenario": scen,
           "inputs": a.inputs, "geometry": a.geometry,
           "kpt": "cmssw", "arm": "arms.v3 (truth_modules False, window_fallback all)",
           "unpaired_tracks": unpaired, "paired_tracks": sel["tracks"],
           "selection": {"crossings": sel["crossings"], "agreeing": sel["agree"],
                         "agreement": sel["agree"] / max(sel["crossings"], 1),
                         "tracks_identical": sel["tracks_identical"], "eps": eps,
                         "max_dscore_agreeing": max(sel["dscore"], default=0.0),
                         "score_checked_crossings": len(sel["dscore"]), "score_over_eps": sel["score_over"],
                         "first_disagreements": sel["first"], "hard": len(hard), "near_ties": len(near),
                         "cap": cap,
                         "final_max_da": max(sel["final_da"], default=0.0),
                         "final_max_dC": max(sel["final_dC"], default=0.0),
                         "passes": not hard and len(near) <= cap and sel["score_over"] <= cap
                                   and sel["agree"] / max(sel["crossings"], 1) >= 0.999},
           "forced": {**{k: v for k, v in frc.items() if k != "worst"}, "over_tolerance": frc["worst"][:50],
                      "n_over_tolerance": len(frc["worst"]), "tolerance": TOL,
                      "passes": not frc["worst"]}}
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(json.dumps({k: res[k] for k in ("scenario", "paired_tracks", "unpaired_tracks")}, indent=1))
    print(json.dumps({k: v for k, v in res["selection"].items() if k != "first_disagreements"}, indent=1, default=float))
    print(json.dumps({k: v for k, v in res["forced"].items() if k != "over_tolerance"}, indent=1, default=float))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--nano", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--export", default="", help="a directory for the per-track output in the reference format")
    ap.add_argument("--eps", type=float, default=3.9e-5, help="the frozen committing-score tolerance (dev events)")
    ap.add_argument("--inputs", choices=("dump", "nano"), default="dump", help="the OT helix: the dump's double or the "
                                                                              "nano's float32")
    ap.add_argument("--geometry", choices=("f32", "file"), default="f32", help="module origins and frames: float32 "
                                                                               "(CMSSW) or the file's text")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
