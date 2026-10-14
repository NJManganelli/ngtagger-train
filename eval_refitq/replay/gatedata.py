"""The entry gate's runs (PLAN §4, §6, §10): every arm once per scenario and file, outputs stored compactly
so that the gate statistics, their bootstraps, resamples and the known-bads' verdicts are computed offline
(gates.py).

  file 3 (gated), per scenario:  PM, PM-B, INN (own + leave-one-out), INN with the payload-era angle errors
          (own: E2 / E3's reported effect), INN K = 2 + skip (E7), the forced f32fma replay (E4), the
          ORP-cal identity (E2 / E3), the CMSSW record and refit-v2 tracks
  file 4 (known-good), per scenario: INN (own + leave-one-out), INN K = 2 + skip
  known-bads (AAAA, file 3): PM without the cap / truncation / empty-window recharge; the producer metric on
          ellipse candidates (ORP's `producer` variant's rule, as an engine arm: ORP does not expose its picks); INN with payload-era scales, the folded table, the global-sign centre, the
          angle-unit centre; the identity with the swapped R-table lookup and half-strength material; the
          forced replay with IL2 positions shifted 1 sigma; K with the tail-probability rule

Output: WPC_OUT/entry/<scen>_f<file>.npz (+ known_bad_AAAA_f03.npz), one key per array, "<run>__<field>".
"""
from __future__ import annotations

import argparse
import json
import os
import time

import awkward as ak
import numpy as np

import arms as AR
import common as CM
import cuts as CU
import diagnostics as DG
import engine as E
import fom as FM
import e5 as E5
import identity as ID
from common import OP, ORP

OWN_FIELDS = ("o", "ev", "step", "first", "surface", "ci", "mi", "clean", "e1", "t1", "pu", "pv", "keybin",
              "orient", "table_pop", "primary", "classT", "has", "g_xy", "g_xya", "g_xyb", "g_xyab", "z", "tp_pt")
N_DRAWS = 10                                     # E5's perturbation model (file 4)
CUTS = str(CM.OUT / "cuts" / "wpc_cuts_v1.json")
SP_COLS = ("d0", "z0", "pt", "phi", "tanL", "rInv", "cov_d0_d0", "cov_z0_z0", "cov_rInv_rInv", "cov_phi_phi",
           "cov_tanL_tanL")
PARAMS = ("rInv", "phi", "tanL", "z0", "d0")


def track_block(out, ctx, off):
    """Per track of an engine output: final parameters, refit flag, the step-wise record, and whether the
    hit accepted at the first visit step is the track's TP's (own) or another's / none (wrong)."""
    a = out["a"]
    d = {p: a[:, i] for i, p in enumerate(PARAMS)}
    d.update(refit=out["refit"], nacc=out["nacc"], branch=out["branch"], ev=ctx["tev"] + off)
    for k in ("acc", "ci", "mi", "surface", "gated", "visited"):
        d["rec_" + k] = out["rec"][k]
    ci0 = out["rec"]["ci"][:, 0]
    acc0 = out["rec"]["acc"][:, 0]
    own0 = acc0 & (ctx["K"]["tp_global"][np.maximum(ci0, 0)] == ctx["trk_tp"]) & (ctx["trk_tp"] >= 0)
    d["first_own"], d["first_wrong"] = own0, acc0 & ~own0
    d["cov_d0"] = out["C"][:, 4, 4]
    d["cov_z0"] = out["C"][:, 3, 3]
    return d


def loo_block(out, ctx, off, R):
    import run_entry as RE
    prim, clean_track = RE.loo_flags(out, R, ctx)
    L = out["loo"]
    d = {"o": L["o"], "ev": ctx["tev"][L["o"]] + off, "surface": L["surface"], "first": L["first"],
         "own_primary": prim, "clean_track": clean_track}
    d.update({f"d_{k}": v for k, v in L["dims"].items()})
    return d


def cmssw_block(A, ctx, off):
    """The CMSSW record (refit hit table) and refit-v2 tracks, plus the OT tracks and their truth."""
    H = E.hit_table(A)
    toff = np.concatenate([[0], np.cumsum(ctx["T"]["_n"])])
    t = toff[H["event"]] + H["trackIdx"].astype(np.int64)
    acc = (H["hitAccepted"] > 0) & (H["selClusterIdx"] >= 0)
    hit = {"t": t, "layer": H["layer"].astype(np.int64), "detId": H["detId"].astype(np.int64), "acc": acc,
           "ci": np.where(acc, ctx["coff"][H["event"]] + H["selClusterIdx"].astype(np.int64), -1),
           "cls": H["selHitClass"].astype(np.int64), "ev": H["event"] + off}
    for k in ("projLocalX", "projLocalY", "projCotAlpha", "projCotBeta", "projSigX", "projSigY", "pullX", "pullY"):
        x = H[k].astype(np.float64)
        hit[k] = np.where(x > -900, x, np.nan)
    sp = {c: ak.to_numpy(ak.flatten(A[f"{CM.SP_TRACK}_{c}"])).astype(np.float64) for c in SP_COLS}
    tr = FM.track_truth(ctx)
    trk = {"sp_" + c: v for c, v in sp.items()}
    trk.update({"ot_" + p: ctx["T"][p].astype(np.float64) for p in ("d0", "z0", "pt", "eta", "rInv", "phi", "tanL")})
    trk.update({"tp_" + k: np.asarray(v) for k, v in tr.items()})
    trk["ev"] = ctx["tev"] + off
    trk["trk_tp"] = ctx["trk_tp"]
    return hit, trk


MIRRORS = ("PM", "PM-nocap", "PM-notrunc", "PM-norecharge", "PM-ellipse")


def e5_block(add, ctx, geo, cfg, tools, scen, hit, trk, outs, start, tr_off, gated, rng):
    """E5 per chunk. File 3: each mirror's (PM and its known-bads) first disagreement with CMSSW per track
    and its near-tie class; PM's first-flip refits (observed shift). File 4: the perturbation model's first
    flips over N_DRAWS draws and their refits (predicted shift). Shift counts are per event of the file
    (event index = start + local)."""
    arm = AR.pm(cfg)
    path, seq = AR.visit_plan(ctx, geo, cfg, scen, arm)
    ntrk = ctx["ntrk"]
    steps = ORP._row_steps(seq, hit["t"], hit["layer"])
    row = np.full((ntrk, seq.shape[1]), -1, np.int64)
    ok = steps >= 0
    row[hit["t"][ok], steps[ok]] = np.flatnonzero(ok)
    have = row >= 0
    r = np.where(have, row, 0)
    base_ci = np.where(have & hit["acc"][r], hit["ci"][r], -1)
    base_mi = np.where(have, OP.module_index(geo, hit["detId"][r]), -1)
    truth = {k[3:]: v for k, v in trk.items() if k.startswith("tp_")}
    ev_local = ctx["tev"]
    nev = int(ev_local.max()) + 1 if ntrk else 0
    kpt = ORP.pt_constant(ctx["T"])
    model = E5.OffsetModel(str(CM.OUT / "tail" / f"tail_{scen}.npz"))
    emp = E5.earlier_max_pull_rows(hit)
    pb = E5.pull_bin(emp)
    tracks_all = np.trace(ctx["C"], axis1=1, axis2=2) > 0
    for name in [m for m in MIRRORS if m in outs] if gated else []:
        rec = outs[name]["rec"]
        cm = {"acc": base_ci >= 0, "ci": base_ci}
        _, first = E5.compare(rec, cm, tracks_all)
        t = np.flatnonzero(first >= 0)
        j = first[t]
        rr = row[t, j]
        pm_ci = np.where(rec["acc"][t, j], rec["ci"][t, j], -1)
        near = np.zeros(len(t), bool)
        has_rec = rr >= 0
        if has_rec.any():
            q = rr[has_rec]
            cor = model.corners(hit["layer"][q], pb[q])
            hits_near = np.zeros(has_rec.sum(), bool)
            for su, sv in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                pk = E5.pick(ctx, cfg, ctx["tev"][hit["t"][q]], hit["detId"][q], hit["layer"][q],
                             hit["projLocalX"][q] + su * cor[:, 0], hit["projLocalY"][q] + sv * cor[:, 1],
                             hit["projCotAlpha"][q], hit["projCotBeta"][q])
                hits_near |= pk == pm_ci[has_rec]
            near[has_rec] = hits_near
        cause = np.where(~has_rec, 2, np.where(near, 0, 1))      # 0 near-tie, 1 outside the window, 2 no CMSSW crossing
        add(f"{name}/e5_first", {"o": t + tr_off, "step": j, "surface": rec["surface"][t, j], "cause": cause,
                                 "pm_ci": pm_ci, "cm_ci": base_ci[t, j], "ev": ev_local[t] + start})
        at_risk = have & tracks_all[:, None] & ((first[:, None] < 0) | (np.arange(seq.shape[1])[None, :] <= first[:, None]))
        tt, jj = np.nonzero(at_risk)
        add(f"{name}/e5_risk", {"surface": hit["layer"][row[tt, jj]], "ev": ev_local[tt] + start,
                                "first": (first[tt] == jj).astype(np.int8)})
        if name == "PM" and len(t):
            new_mi = np.where(rec["acc"][t, j], rec["mi"][t, j], -1)
            b, fl = E5.flip_effect(arm, ctx, geo, cfg, tools, path, seq, base_mi, base_ci, t, j, pm_ci, new_mi, kpt)
            cnt = E5.counts_of_flip(ctx, truth, t, b, fl, trk["ot_d0"], trk["ot_z0"], ev_local, nev,
                                    seq[t, j], base_ci[t, j], pm_ci)
            add("e5_obs_counts", cnt | {"ev": np.arange(nev) + start})
    if gated:
        return
    # file 4: the perturbation model
    hr = np.flatnonzero(np.isfinite(hit["projLocalX"]))
    ref = E5.pick(ctx, cfg, ctx["tev"][hit["t"][hr]], hit["detId"][hr], hit["layer"][hr], hit["projLocalX"][hr],
                  hit["projLocalY"][hr], hit["projCotAlpha"][hr], hit["projCotBeta"][hr])
    for d in range(N_DRAWS):
        off = model.draw(hit["layer"][hr], pb[hr], rng)
        pk = E5.pick(ctx, cfg, ctx["tev"][hit["t"][hr]], hit["detId"][hr], hit["layer"][hr], hit["projLocalX"][hr] + off[:, 0],
                     hit["projLocalY"][hr] + off[:, 1], hit["projCotAlpha"][hr], hit["projCotBeta"][hr])
        flip = pk != ref
        tr = hit["t"][hr]
        st = steps[hr]
        fstep = np.full(ntrk, np.iinfo(np.int64).max)
        np.minimum.at(fstep, tr[flip & (st >= 0)], st[flip & (st >= 0)])
        firstf = np.where(fstep < np.iinfo(np.int64).max, fstep, -1)
        risk = (st >= 0) & ((firstf[tr] < 0) | (st <= firstf[tr]))
        add("e5_model_risk", {"surface": hit["layer"][hr][risk], "ev": ev_local[tr[risk]] + start,
                              "first": (firstf[tr[risk]] == st[risk]).astype(np.int8), "draw": np.full(risk.sum(), d)})
        t = np.flatnonzero(firstf >= 0)
        if not len(t):
            continue
        j = firstf[t]
        # the flipped crossing's perturbed pick, on CMSSW's recorded module
        sel = np.flatnonzero(flip & (st >= 0))
        key = tr[sel] * 64 + st[sel]
        kk = t * 64 + j
        pos = np.searchsorted(key, kk)
        new_ci = pk[sel][pos]
        new_mi = base_mi[t, j]
        b, fl = E5.flip_effect(arm, ctx, geo, cfg, tools, path, seq, base_mi, base_ci, t, j, new_ci, new_mi, kpt)
        cnt = E5.counts_of_flip(ctx, truth, t, b, fl, trk["ot_d0"], trk["ot_z0"], ev_local, nev,
                                seq[t, j], base_ci[t, j], new_ci)
        add("e5_model_counts", cnt | {"ev": np.arange(nev) + start, "draw": np.full(nev, d)})


def run_file(scen, f, nev, chunk, known_bad):
    path = CM.input_path(scen, f)
    cfg = CM.producer_config(path)
    assert CM.mask_of(cfg) == scen
    geo = OP.load_geometry(CM.GEOMETRY)
    cuts = CU.Cuts.load(CUTS)
    tools = AR.Tools(cfg, cuts=cuts)
    gated = f == CM.GATED_FILE
    runs = {"INN": AR.inn(cfg), "INN-K2": AR.inn(cfg, name="INN-K2", K=2)}
    if gated:
        runs.update({"PM": AR.pm(cfg), "PM-B": AR.pm_b(cfg), "INN-payload": AR.inn(cfg, name="INN-payload",
                                                                                    ang_err="payload")})
    kb = {}
    if known_bad:
        kb = {"PM-nocap": AR.pm(cfg, name="PM-nocap", cap=False),
              "PM-notrunc": AR.pm(cfg, name="PM-notrunc", truncate=False),
              "PM-norecharge": AR.pm(cfg, name="PM-norecharge", empty_recharge=False),
              "PM-ellipse": AR.pm(cfg, name="PM-ellipse", window="ellipse", jac="central", crossing="f64"),
              "INN-kb-payload": AR.inn(cfg, name="INN-kb-payload", ang_err="payload"),
              "INN-kb-folded": AR.inn(cfg, name="INN-kb-folded", table_variant="folded"),
              "INN-kb-globalsign": AR.inn(cfg, name="INN-kb-globalsign", table_variant="global_sign"),
              "INN-kb-angleunits": AR.inn(cfg, name="INN-kb-angleunits", table_variant="angle_units"),
              "INN-K2-tailprob": AR.inn(cfg, name="INN-K2-tailprob", K=2, k_score="tailprob")}
    parts = {}
    ident = []
    rng = np.random.default_rng(20261014 + 1000 * f + CM.SCENARIOS.index(scen))
    t0 = time.time()
    tr_off = 0
    nev_done = 0
    for start, A in CM.iterate(path, list(dict.fromkeys(AR.branches(path) + [f"{CM.SP_TRACK}_{c}" for c in SP_COLS]
                                                        + ["run", "luminosityBlock", "event"])), chunk=chunk, nev=nev):
        ctx = AR.prepare(A, geo)
        add = lambda k, d: parts.setdefault(k, []).append(d)
        outs = {}
        for name, arm in list(runs.items()) + list(kb.items()):
            own = name in ("INN", "INN-payload") or name.startswith("INN-kb-")
            loo = name == "INN"
            out = AR.run_arm(ctx, geo, cfg, scen, arm, tools, own=own, loo=loo)
            outs[name] = out
            tb = track_block(out, ctx, start)
            tb["o"] = np.arange(ctx["ntrk"]) + tr_off
            add(name, tb)
            if name in MIRRORS:
                ht, hs, _, hw = FM.hits_of_record(out["rec"], ctx)
                pop = FM.track_truth(ctx)["pop"][ht]
                add(f"{name}/wrong", {"w": hw[pop].astype(float), "ev": ctx["tev"][ht[pop]] + start,
                                      "surface": hs[pop]})
            if own:
                R = DG.mark_primary(out["own"])
                ob = {k: R[k] for k in OWN_FIELDS}
                ob["ev"] = ob["ev"] + start
                ob["o"] = ob["o"] + tr_off
                add(name + "/own", ob)
                if loo:
                    lb = loo_block(out, ctx, start, R)
                    lb["o"] = lb["o"] + tr_off
                    add(name + "/loo", lb)
        Hh = E.hit_table(A)
        for nm, fo in (("forced", E.ForcedOpts(crossing="f32fma")),
                       ("forced-cluster", E.ForcedOpts(crossing="f32fma", meas="cluster"))):
            if nm == "forced-cluster" and gated:
                continue
            Rf = E.forced(ctx, Hh, geo, cfg, scen, fo)
            add(nm, {**{p: Rf["a"][:, i] for i, p in enumerate(PARAMS)}, "has_rows": Rf["has_rows"],
                     "o": np.arange(ctx["ntrk"]) + tr_off})
        hit, trk = cmssw_block(A, ctx, start)
        e5_block(add, ctx, geo, cfg, tools, scen, hit, trk, outs, start, tr_off, gated, rng)
        hit["t"] = hit["t"] + tr_off
        add("cmssw_hit", hit)
        trk["o"] = np.arange(ctx["ntrk"]) + tr_off
        add("cmssw_trk", trk)
        if gated:
            ident.append(ID.orp_cal_identity(A, geo, cfg, scen, tools))
            if known_bad:
                Rk = E.forced(ctx, E.hit_table(A), geo, cfg, scen, E.ForcedOpts(crossing="f32fma", shift_il2_sigma=1.0))
                add("forced-kb-shift", {**{p: Rk["a"][:, i] for i, p in enumerate(PARAMS)},
                                        "o": np.arange(ctx["ntrk"]) + tr_off})
                for nm, ex in (("swap", {"rtable_swap": True}), ("halfmat", {"material_scale": 0.5})):
                    la, lb = [], []
                    with ID.recording(ORP, "kalman_update", la):
                        ORP.run_chunk(A, geo, scen, cfg["useAngles"], ("innovation",), 4.0, False, tools.cal,
                                      CM.REFIT_HIT, cfg)
                    with ID.recording(E, "kf_update", lb):
                        AR.run_arm(ctx, geo, cfg, scen, AR.orp_cal(cfg, **ex), tools, tracks=ctx["cls"] >= 0)
                    add(f"ident-kb-{nm}", {"identical": np.array([ID.compare_logs(la, lb, np.flatnonzero(ctx["cls"] >= 0))["identical"]])})
        tr_off += ctx["ntrk"]
        nev_done += len(A)
        print(f"[{time.time() - t0:7.1f}s] {scen} f{f:02d}: {nev_done} events", flush=True)
    data = {k: CM.concat(v) for k, v in parts.items()}
    meta = {"scen": scen, "file": f, "events": nev_done, "runs": list(runs) + list(kb), "cuts": CUTS,
            "identity_chunks": ident, "seconds": round(time.time() - t0, 1)}
    return data, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--file", type=int, choices=(CM.GATED_FILE, CM.PRED_FILE), required=True)
    ap.add_argument("--scen", nargs="+", default=list(CM.SCENARIOS))
    ap.add_argument("--known-bad", action="store_true", help="also the known-bad runs (AAAA, file 3)")
    ap.add_argument("--nev", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--out", default=str(CM.OUT / "entry"))
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    for scen in a.scen:
        kb = a.known_bad and scen == "AAAA" and a.file == CM.GATED_FILE
        data, meta = run_file(scen, a.file, a.nev, a.chunk, kb)
        np.savez_compressed(os.path.join(a.out, f"{scen}_f{a.file:02d}.npz"),
                            **{f"{run}__{k}": v for run, d in data.items() for k, v in d.items()})
        with open(os.path.join(a.out, f"{scen}_f{a.file:02d}.json"), "w") as fh:
            json.dump(meta, fh, indent=1, default=str)
        print(f"{scen} f{a.file:02d} done: {meta['seconds']} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
