"""WP-D0a before/after: runs A (format 26), B (27, OT module flags), C (28, stub credit) and
C2 (29, C with truth diagnostics) on the same events, one set of figures of merit, paired
event-bootstrap errors.

    # each census read by its OWN code root (a census refuses other formats); with
    # PYTHONPATH=ROOT/src so ngtagger comes from the same root
    python before_after.py extract LABEL CENSUS ROOT OUT.npz
    # the input's OT credits by route, for the denominators (code-independent)
    python before_after.py routes INPUT.root OUT.npz
    python before_after.py compare A.npz B.npz C.npz [C2.npz] --routes R.npz --universe C \
        [--pairs A:B,B:C,A:C] -o out.json

extract keeps, per TP of the census's own table: event, true pT, |eta|, |d0|, phi, z0,
production radius and the hit patterns; per build (the census's DR group): seed-truthful
(found_dr over the build's entries; definition sheet truth.seed_truthful of the census's
own format), delivered (a DR survivor of the build's full menu, DR recomputed from the
exported tracks as angle_stage_metrics does, owned by the TP: n_own >= 3 and more than half
the hits) and clean delivered (the same track without a wrong IT hit), all also per seed
class (IT, OT, mixed); per event: fitted tracks, DR survivors and fakes (>= 2 wrong hits;
truth.census_fake, raw) per seed class x arity. Two further columns:
  *_sens   a SENSITIVITY to the dominant-TP limit (the nano carries one TP per OT cluster,
           CMSSW credits every contributor; CMSSW truth spec sec. 6), not a bound: delivered
           and fake if every wrong OT stub that has a TP also credited the CURRENT owner
           (with full contributor lists another TP could become the owner instead);
  *_pos    (format >= 29) the same figures without OUTER-ONLY credits, the stubs credited to
           the owner only through their outer cluster and so holding another particle's
           position (truth.stub): seed-truthful from found_dr_pos, delivered with
           n_own - n_ot_right_outer_only, fake with n_wrong + n_ot_right_outer_only (same
           owner).

routes reads the input's OT barrel stubs as the census does (|stub eta| <= M.ETA_MATCHED) and
keeps, per TP, the layers credited by route: through the principal tpIdx, through a
position-carrying credit (tpIdx or tpIdxInner), and outer-only; plus every L1TTP row's
event-id and bunch crossing.

compare quotes every figure on each run's own census denominator (TPs with >= 4 layers,
IT and OT together, under that run's credit) and, for review 01 H1, on four denominators
built once from the routes and the universe run's table (the format >= 28 one), the same
TP set for every run (a TP missing from a run's table is not found by it):
  a  current      IT layers + OT layers by any credit (format 28's own denominator)
  b  position     IT layers + OT layers whose stub carries the TP's position
                  (principal or inner-cluster credit)
  c  fixed        IT layers + OT layers by the principal only: run B's population
  d  OT-only      >= 4 OT layers by any credit: the analogue of L1TrackNtupleMaker's
                  denominator (>= 4 OT stubs in >= 4 layers or disks; TP_minPt 1.9 there, the
                  bins below here; barrel only)
  dhs             d and hard interaction in time (L1TTP evt 0, bx 0), as its plot macro
at pT >= 2 GeV in |eta| < 1.1 (c), < 0.9 (h), 1.1-1.8 (f) and pT 1-2 GeV in |eta| < 1.1,
each |eta| range with its OT-truncated label (M.eta_range_label). Errors are the standard
deviation over event-bootstrap replicas (1000; the 100 events resampled with replacement,
every figure recomputed as a ratio of sums), PAIRED across runs (same replica weights: the
runs share their events), so a difference's error includes the correlation.
"""
from __future__ import annotations
import argparse, importlib, json, os, sys
from pathlib import Path
import numpy as np

N_REP = 1000
SEED = 20261009
SYS = {0: "it", 1: "ot", 2: "mixed"}
CATS = [(s, a) for s in (0, 1, 2) for a in (2, 3)]
ETA = {"c": (0.0, 1.1), "h": (0.0, 0.9), "f": (1.1, 1.8)}
EFFS = [("hi", "c"), ("hi", "h"), ("hi", "f"), ("lo", "c")]
PT = {"hi": (2.0, np.inf), "lo": (1.0, 2.0)}
DENS = {"a": "current: IT + OT layers by any credit",
        "b": "position-carrying: IT + OT layers by principal or inner-cluster credit",
        "c": "fixed: IT + OT layers by the principal only (run B's population)",
        "d": "OT-only: >= 4 OT layers by any credit",
        "dhs": "OT-only and hard interaction in time (L1TTP evt 0, bx 0)"}
HERE = Path(__file__).resolve().parents[1] / "combinatorics"


def popcount8(x):
    return np.unpackbits(np.asarray(x, np.uint8)[:, None], axis=1).sum(1)


# ------------------------------------------------------------------ extract
def _root_modules(root):
    d = os.path.join(os.path.realpath(root), "eval_refitq", "combinatorics")
    sys.path.insert(0, d)
    mods = [importlib.import_module(m) for m in ("tp_findability", "seed_arity", "kf_emulation")]
    import ngtagger
    for m in mods + [ngtagger]:
        if not os.path.realpath(m.__file__).startswith(os.path.realpath(root) + os.sep):
            raise SystemExit(f"{m.__name__} imported from {m.__file__}, not from {root}")
    return mods


def extract(label, census, root, out):
    TF, SA, KF = _root_modules(root)
    M = TF.M
    d = TF.census_dir(census)
    man = json.load(open(os.path.join(d, "manifest.json")))
    C = TF.load(d, with_tracks=True, verbose=False)
    fmt, nev = man["format"], int(C["n_events"])
    col = {c: j for j, c in enumerate(C["track_cols"])}
    pos_ok = "n_ot_right_outer_only" in col and "found_dr_pos" in C
    key = np.asarray(C["key"], np.int64)
    R = {"label": label, "format": fmt, "n_events": nev, "census": d, "code_root": root,
         "builds": list(C["builds"]), "inputs": [r["path"] for r in man["inputs"]], "has_pos": pos_ok}
    A = {"key": key, "event": np.asarray(C["event"], np.int32), "pt": C["pt"],
         "aeta": np.abs(C["eta"]), "ad0": np.abs(C["d0"]), "phi": C["phi"], "z0": C["z0"],
         "vr": C["vr"], "nlay": (popcount8(C["hit_it"]) + popcount8(C["hit_ot"])).astype(np.int8),
         "hit_it": C["hit_it"], "hit_ot": C["hit_ot"]}
    seed_cls = {i: KF.sysclass_of(C["seeds"][i].layers) for i in range(len(C["seeds"]))}
    for b, ent in C["builds"].items():
        seed = {k: np.zeros(len(key), bool) for k in ("all", *SYS.values())}
        seed_pos = {k: np.zeros(len(key), bool) for k in ("all", *SYS.values())}
        for i in ent:
            for S_, pos in ((seed, False), (seed_pos, True)):
                if pos and not pos_ok:
                    continue
                m = TF.seed_mask(C, i, after_dr=True, position=True) if pos else TF.seed_mask(C, i, after_dr=True)
                S_["all"] |= m
                S_[SYS[seed_cls[i]]] |= m
        es = [i for i in sorted(ent) if i in C["tracks"] and len(C["tracks"][i])]
        T = np.concatenate([C["tracks"][i] for i in es])
        H = np.concatenate([C["hits"][i] for i in es])
        surv = SA.duplicate_removal(H, T[:, col["rank_score"]].astype(np.float64))
        S = T[surv]
        n_own, nh, own = S[:, col["n_own"]], S[:, col["nhit"]], S[:, col["own_tpidx"]]
        nw = S[:, col["n_wrong"]]

        def held_with(n):
            return (own >= 0) & (n >= 3) & (n > 0.5 * nh)
        held = held_with(n_own)
        hid = S[:, col["n_ot_wrong"]] + S[:, col["n_ot_combinatoric"]]
        sel = {"deliv": held, "clean": held & (S[:, col["n_it_wrong"]] == 0),
               "deliv_sens": held_with(n_own + hid)}
        fakes = {"fake": nw >= 2, "fake_sens": nw - hid >= 2}
        if pos_ok:
            oo = S[:, col["n_ot_right_outer_only"]]
            sel["deliv_pos"] = held_with(n_own - oo)
            fakes["fake_pos"] = nw + oo >= 2
            A[f"{b}/n_outer_only_dr"] = np.bincount(S[:, col["event"]].astype(np.int64),
                                                    weights=oo, minlength=nev)
            A[f"{b}/n_ambiguous_dr"] = np.bincount(S[:, col["event"]].astype(np.int64),
                                                   weights=(S[:, col["n_tp_majority"]] >= 2), minlength=nev)
        okey = M.tp_key(S[:, col["event"]].astype(np.int64), np.maximum(own, 0).astype(np.int64))
        scl = S[:, col["sysclass"]].astype(int)
        ar = S[:, col["arity"]].astype(int)
        ev_t = T[:, col["event"]].astype(np.int64)
        ev_s = S[:, col["event"]].astype(np.int64)
        for nm, m in sel.items():
            A[f"{b}/{nm}"] = np.isin(key, okey[m])
            for k, v in SYS.items():
                A[f"{b}/{nm}_{v}"] = np.isin(key, okey[m & (scl == k)])
        A[f"{b}/n_fit"] = np.bincount(ev_t, minlength=nev)
        A[f"{b}/n_dr"] = np.bincount(ev_s, minlength=nev)
        for fk, f in fakes.items():
            A[f"{b}/n_{fk}"] = np.bincount(ev_s[f], minlength=nev)
            for s, a in CATS:
                m = (scl == s) & (ar == a)
                A[f"{b}/n_dr_{SYS[s]}{a}"] = np.bincount(ev_s[m], minlength=nev)
                A[f"{b}/n_{fk}_{SYS[s]}{a}"] = np.bincount(ev_s[m & f], minlength=nev)
        for k, v in seed.items():
            A[f"{b}/seed_{k}"] = v
        if pos_ok:
            for k, v in seed_pos.items():
                A[f"{b}/seed_pos_{k}"] = v
        print(f"{label} {b}: {len(T):,} fitted, {len(S):,} after DR", flush=True)
    np.savez_compressed(out, meta=json.dumps(R), **A)
    print(f"wrote {out}")


# ------------------------------------------------------------------ routes
def routes(path, out, nev=None):
    """Per TP (M.tp_key) of the input: its OT barrel layers (census selection) credited by
    the principal, by a position-carrying credit and outer-only; and every L1TTP row's
    event-id and bunch crossing."""
    import awkward as ak
    import uproot
    sys.path.insert(0, str(HERE))
    import tracklet_topology_cost as M
    br = [f"L1TOTStub_{c}" for c in ("layer", "isBarrel", "z", "r", *M.STUB_TP_COLS)] + ["L1TTP_evt", "L1TTP_bx"]
    A = uproot.open(f"{path}:Events").arrays(br, entry_stop=nev)
    n = ak.to_numpy(ak.num(A["L1TOTStub_layer"]))
    f = {c: ak.to_numpy(ak.flatten(A[f"L1TOTStub_{c}"])) for c in ("layer", "isBarrel", "z", "r", *M.STUB_TP_COLS)}
    ev = np.repeat(np.arange(len(n)), n)
    eta = np.abs(np.arcsinh(f["z"] / np.maximum(f["r"], 1e-6)))
    bar = (f["isBarrel"] > 0) & (eta <= M.ETA_MATCHED)
    cred = M.credited_tps(*(f[c][bar] for c in M.STUB_TP_COLS))
    lay, evb = f["layer"][bar].astype(np.int64), ev[bar]
    i, s_ = np.nonzero(M.SPN.tp_matched(cred))
    key = M.tp_key(evb[i], cred[i, s_])
    bit = (1 << (lay[i] - 1)).astype(np.uint8)
    uk, inv = np.unique(key, return_inverse=True)
    out_ = {"key": uk}
    for nm, m in (("principal", s_ == 0), ("position", s_ < M.POSITION_SLOTS),
                  ("outer_only", s_ >= M.POSITION_SLOTS), ("any", np.ones(len(s_), bool))):
        b = np.zeros(len(uk), np.uint8)
        np.bitwise_or.at(b, inv[m], bit[m])
        out_[nm] = b
    nt = ak.to_numpy(ak.num(A["L1TTP_evt"]))
    out_["tp_key"] = M.tp_key(np.repeat(np.arange(len(nt)), nt),
                              np.arange(int(nt.sum())) - np.repeat(np.cumsum(nt) - nt, nt))
    out_["tp_evt"] = ak.to_numpy(ak.flatten(A["L1TTP_evt"]))
    out_["tp_bx"] = ak.to_numpy(ak.flatten(A["L1TTP_bx"]))
    np.savez_compressed(out, meta=json.dumps({"input": path, "n_events": int(len(n))}), **out_)
    print(f"wrote {out}: {len(uk):,} TPs with an OT credit, {int(nt.sum()):,} L1TTP rows")


# ------------------------------------------------------------------ compare
def load(p):
    z = np.load(p, allow_pickle=False)
    return json.loads(str(z["meta"])), {k: z[k] for k in z.files if k != "meta"}


def kin_sel(Z, pt, eta):
    lo, hi = PT[pt]
    e0, e1 = ETA[eta]
    return (Z["pt"] >= lo) & (Z["pt"] < hi) & (Z["aeta"] >= e0) & (Z["aeta"] < e1)


def _ratio(num, den, W):
    return float(num.sum() / max(den.sum(), 1)), (W @ num) / np.maximum(W @ den, 1)


def figures(meta, Z, W):
    """{figure: (value, replicas)} for one run, on its own census denominator."""
    nev = meta["n_events"]
    out = {}
    effk = ["seed_all", "deliv", "clean", "deliv_sens"] + [f"{x}_{v}" for x in ("seed", "deliv")
                                                           for v in SYS.values()]
    if meta.get("has_pos"):
        effk += ["seed_pos_all", "deliv_pos"] + [f"seed_pos_{v}" for v in SYS.values()]
    for b in meta["builds"]:
        for pt, eta in EFFS:
            sel = (Z["nlay"] >= 4) & kin_sel(Z, pt, eta)
            den = np.bincount(Z["event"][sel], minlength=nev).astype(float)
            out[f"{b}/den_{pt}_{eta}"] = (float(den.sum()), W @ den)
            for k in effk:
                num = np.bincount(Z["event"][sel & Z[f"{b}/{k}"]], minlength=nev).astype(float)
                out[f"{b}/eff_{k}_{pt}_{eta}"] = _ratio(num, den, W)
        for k in ("n_fit", "n_dr"):
            n = Z[f"{b}/{k}"].astype(float)
            out[f"{b}/{k}_per_ev"] = (float(n.sum() / nev), (W @ n) / nev)
        for fk in ("fake", "fake_sens") + (("fake_pos",) if meta.get("has_pos") else ()):
            out[f"{b}/{fk}"] = _ratio(Z[f"{b}/n_{fk}"].astype(float), Z[f"{b}/n_dr"].astype(float), W)
            for s, a in CATS:
                c = f"{SYS[s]}{a}"
                out[f"{b}/{fk}_{c}"] = _ratio(Z[f"{b}/n_{fk}_{c}"].astype(float),
                                              Z[f"{b}/n_dr_{c}"].astype(float), W)
        if meta.get("has_pos"):
            for k in ("n_outer_only_dr", "n_ambiguous_dr"):
                n = Z[f"{b}/{k}"].astype(float)
                out[f"{b}/{k}_per_ev"] = (float(n.sum() / nev), (W @ n) / nev)
    return out


def denominators(U, Rt):
    """The review-01 denominators over the universe table U (a format >= 28 census
    table: every TP with any credit), from the input's routes Rt. Also checks that the
    routes reproduce the census's own hit patterns."""
    pos = np.clip(np.searchsorted(Rt["key"], U["key"]), 0, len(Rt["key"]) - 1)
    has = Rt["key"][pos] == U["key"]
    bits = {k: np.where(has, Rt[k][pos], 0).astype(np.uint8)
            for k in ("principal", "position", "outer_only", "any")}
    if not np.array_equal(bits["any"], U["hit_ot"]):
        raise SystemExit(f"routes do not reproduce the census's OT hit patterns on "
                         f"{int((bits['any'] != U['hit_ot']).sum())} TPs")
    tpos = np.clip(np.searchsorted(Rt["tp_key"], U["key"]), 0, len(Rt["tp_key"]) - 1)
    in_ttp = Rt["tp_key"][tpos] == U["key"]
    hard = in_ttp & (Rt["tp_evt"][tpos] == 0) & (Rt["tp_bx"][tpos] == 0)
    nit = popcount8(U["hit_it"])
    n_any, n_pos, n_pri = (popcount8(bits[k]) for k in ("any", "position", "principal"))
    return {"a": nit + n_any >= 4, "b": nit + n_pos >= 4, "c": nit + n_pri >= 4,
            "d": n_any >= 4, "dhs": (n_any >= 4) & hard}, {"nit": nit, "n_any": n_any, "n_pos": n_pos, "n_pri": n_pri,
                                       "hard": hard}


def found_on(Z, b, k, keys):
    """flag `k` of build b of run Z, on the TP keys `keys` (False where Z lacks the TP)."""
    p = np.clip(np.searchsorted(Z["key"], keys), 0, len(Z["key"]) - 1)
    return (Z["key"][p] == keys) & Z[f"{b}/{k}"][p]


def den_figures(runs, U, D, W, nev):
    """{run: {figure: (value, replicas)}} of seed-truthful, delivered and clean delivered on
    the four denominators D (masks over U), the same TPs for every run."""
    out = {}
    for lab, (meta, Z) in runs.items():
        r = out.setdefault(lab, {})
        for b in meta["builds"]:
            for dn, dm in D.items():
                for pt, eta in EFFS:
                    sel = dm & kin_sel(U, pt, eta)
                    den = np.bincount(U["event"][sel], minlength=nev).astype(float)
                    r[f"{b}/D{dn}/den_{pt}_{eta}"] = (float(den.sum()), W @ den)
                    for k in ("seed_all", "deliv", "clean"):
                        f = found_on(Z, b, k, U["key"][sel])
                        num = np.bincount(U["event"][sel][f], minlength=nev).astype(float)
                        r[f"{b}/D{dn}/eff_{k}_{pt}_{eta}"] = _ratio(num, den, W)
    return out


def decomposition(U, D, parts):
    """Review 01 H1: who joins the current denominator (a) from run B's (c)."""
    out = {}
    for pt, eta in (("hi", "c"), ("lo", "c")):
        k = kin_sel(U, pt, eta)
        new = k & D["a"] & ~D["c"]
        out[f"{pt}_{eta}"] = {
            "n_new_members": int(new.sum()),
            "reach_4_ot_layers (a CMSSW analogue exists)": int((new & (parts["n_any"] >= 4)).sum()),
            "only_through_outer_only_credits (not in the position denominator b)": int((new & ~D["b"]).sum()),
            "pattern_IT_layers_x_OT_layers_any": {f"{i}IT+{o}OT": int((new & (parts["nit"] == i)
                                                                        & (parts["n_any"] == o)).sum())
                                                 for i in range(5) for o in range(7)
                                                 if (new & (parts["nit"] == i) & (parts["n_any"] == o)).any()},
            "hard_interaction_in_time": int((new & parts["hard"]).sum())}
    return out


def outer_only(runs, U, D, W, nev, before="B", after="C", pos="C2"):
    """Review 01 M5: how much of the after - before change rests on OUTER-ONLY credits. On
    run B's fixed population (denominator c) per seed class: seed-truthful and delivered of
    `before`, `after` and of `pos` (a format >= 29 rerun of `after`) without outer-only
    credits; the raw fake fraction per seed class x arity likewise (DR survivors are the same
    tracks in all three). share = (after - pos) / (after - before): the part of the change
    the outer-only credits carry."""
    Zb, Za, Zp = runs[before][1], runs[after][1], runs[pos][1]
    res = {}
    for b in runs[pos][0]["builds"]:
        r = res.setdefault(b, {})
        for pt, eta in (("hi", "c"), ("lo", "c")):
            sel = D["c"] & kin_sel(U, pt, eta)
            den = np.bincount(U["event"][sel], minlength=nev).astype(float)
            for cls in ("all", *SYS.values()):
                for fig, kb, kp in (("seed", f"seed_{cls}", f"seed_pos_{cls}"),
                                    ("deliv", "deliv" if cls == "all" else f"deliv_{cls}",
                                     "deliv_pos" if cls == "all" else None)):
                    if kp is None:
                        continue
                    vals = {}
                    for nm, Z, k in ((before, Zb, kb), (after, Za, kb), (pos, Zp, kp)):
                        f = found_on(Z, b, k, U["key"][sel])
                        vals[nm] = _ratio(np.bincount(U["event"][sel][f], minlength=nev).astype(float), den, W)
                    r[f"{fig}_{cls}_{pt}_{eta}"] = _share(vals, before, after, pos)
        for c in ["", *[f"_{SYS[s_]}{a}" for s_, a in CATS]]:
            vals = {nm: _ratio(Z[f"{b}/n_{fk}{c}"].astype(float), Z[f"{b}/n_dr{c}"].astype(float), W)
                    for nm, Z, fk in ((before, Zb, "fake"), (after, Za, "fake"), (pos, Zp, "fake_pos"))}
            r[f"fake{c}"] = _share(vals, before, after, pos)
        for k in ("n_outer_only_dr", "n_ambiguous_dr"):
            n = Zp[f"{b}/{k}"].astype(float)
            r[f"{k}_per_ev"] = {"value": float(n.sum() / nev), "sd": float(np.std((W @ n) / nev, ddof=1))}
        n_dr = float(Zp[f"{b}/n_dr"].sum())
        r["frac_dr_survivors_ambiguous_owner"] = float(Zp[f"{b}/n_ambiguous_dr"].sum()) / max(n_dr, 1.0)
    return res


def _share(vals, before, after, pos):
    (vb, rb), (va, ra), (vp, rp) = vals[before], vals[after], vals[pos]
    d, do = va - vb, va - vp
    return {before: vb, after: va, f"{pos}_without_outer_only": vp,
            "change": d, "change_sd": float(np.std(ra - rb, ddof=1)),
            "outer_only_part": do, "outer_only_part_sd": float(np.std(ra - rp, ddof=1)),
            "share_outer_only": do / d if d else None}


def tp_changes(Zb, Zc):
    """What the stub credit did to the TP table: new TPs and OT layers gained."""
    kb, kc = Zb["key"], Zc["key"]
    new = ~np.isin(kc, kb)
    pos = np.clip(np.searchsorted(kc, kb), 0, len(kc) - 1)
    kept = bool(np.all(kc[pos] == kb))
    gain_ot = Zc["hit_ot"][pos] != Zb["hit_ot"]
    lost = ((Zc["hit_ot"][pos] & Zb["hit_ot"]) != Zb["hit_ot"]) | (Zc["hit_it"][pos] != Zb["hit_it"])
    kin = [c for c in ("pt", "aeta", "ad0", "phi", "z0", "vr", "event") if c in Zb and c in Zc]
    same = all(np.array_equal(Zc[c][pos], Zb[c], equal_nan=True) for c in kin)
    den = lambda Z: int(((Z["nlay"] >= 4) & (Z["pt"] >= 2.0) & (Z["aeta"] < 1.1)).sum())
    return {"n_tp_before": int(len(kb)), "n_tp_after": int(len(kc)), "every_tp_kept": kept,
            "frac_new_tps": float(new.sum() / len(kb)),
            "frac_new_tps_below_2gev": float((Zc["pt"][new] < 2.0).sum() / max(new.sum(), 1)),
            "frac_new_tps_pt_nan": float(np.isnan(Zc["pt"][new]).mean()) if new.any() else 0.0,
            "frac_tps_gaining_ot_layer": float(gain_ot.mean()),
            "n_tps_losing_a_layer": int(lost.sum()),
            "kinematics_of_common_tps_identical": bool(same), "kinematics_compared": kin,
            "den_hi_c_before": den(Zb), "den_hi_c_after": den(Zc)}


def _pair(F, x, y):
    d = {}
    for k in sorted(set(F[x]) & set(F[y])):
        vx, rx = F[x][k]
        vy, ry = F[y][k]
        sd = float(np.std(ry - rx, ddof=1))
        d[k] = {"before": vx, "after": vy, "diff": vy - vx, "sd": sd,
                "rel": (vy - vx) / vx if vx else None, "z": (vy - vx) / sd if sd > 0 else None}
    return d


def compare(paths, pairs, out, routes_path=None, universe=None):
    runs = {}
    for p in paths:
        meta, Z = load(p)
        runs[meta["label"]] = (meta, Z)
    nev = {m["n_events"] for m, _ in runs.values()}
    if len(nev) != 1:
        raise SystemExit(f"runs have different event counts {nev}")
    nev = nev.pop()
    W = np.random.default_rng(SEED).multinomial(nev, np.full(nev, 1.0 / nev), size=N_REP).astype(float)
    F = {lab: figures(m, Z, W) for lab, (m, Z) in runs.items()}
    sys.path.insert(0, str(HERE))
    import tracklet_topology_cost as M
    res = {"n_events": nev, "n_replicas": N_REP, "seed": SEED,
           "runs": {lab: {k: m.get(k) for k in ("format", "census", "code_root", "builds", "has_pos")}
                    for lab, (m, _) in runs.items()},
           "eta_labels": {k: M.eta_range_label(*v) for k, v in ETA.items()}, "figures": {},
           "pairs": {}, "tp_changes": {}}
    for lab, f in F.items():
        res["figures"][lab] = {k: {"value": v, "sd": float(np.std(r, ddof=1))} for k, (v, r) in f.items()}
    G = {}
    if routes_path:
        rmeta, Rt = load(routes_path)
        U = runs[universe][1]
        D, parts = denominators(U, Rt)
        G = den_figures(runs, U, D, W, nev)
        res["denominators"] = {"definitions": DENS, "universe": universe, "routes": rmeta,
                               "figures": {lab: {k: {"value": v, "sd": float(np.std(r, ddof=1))}
                                                 for k, (v, r) in g.items()} for lab, g in G.items()},
                               "decomposition_a_vs_c": decomposition(U, D, parts)}
        res["denominators"]["pairs"] = {}
        if "C2" in runs and runs["C2"][0].get("has_pos") and "B" in runs and "C" in runs:
            res["outer_only"] = outer_only(runs, U, D, W, nev)
    for pr in pairs:
        x, y = pr.split(":")
        res["pairs"][pr] = _pair(F, x, y)
        res["tp_changes"][pr] = tp_changes(runs[x][1], runs[y][1])
        if G:
            res["denominators"]["pairs"][pr] = _pair(G, x, y)
    json.dump(res, open(out, "w"), indent=1)
    print(f"wrote {out}")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("label")
    e.add_argument("census")
    e.add_argument("root")
    e.add_argument("out")
    r = sub.add_parser("routes")
    r.add_argument("input")
    r.add_argument("out")
    r.add_argument("-n", "--nev", type=int, default=None)
    c = sub.add_parser("compare")
    c.add_argument("npz", nargs="+")
    c.add_argument("--pairs", default="A:B,B:C,A:C")
    c.add_argument("--routes", default=None)
    c.add_argument("--universe", default="C")
    c.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    if a.cmd == "extract":
        extract(a.label, a.census, a.root, a.out)
    elif a.cmd == "routes":
        routes(a.input, a.out, a.nev)
    else:
        compare(a.npz, a.pairs.split(","), a.out, a.routes, a.universe)


if __name__ == "__main__":
    main()
