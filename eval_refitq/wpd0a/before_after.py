"""WP-D0a before/after: census runs on the same events (A format 26, B 27, C 28, C2 29,
D 30, E 31), one set of figures of merit, paired event-bootstrap errors.

    # each census read by its OWN code root (a census refuses other formats); with
    # PYTHONPATH=ROOT/src so ngtagger comes from the same root
    python before_after.py extract LABEL CENSUS ROOT OUT.npz
    python before_after.py compare A.npz B.npz ... --universe E [--pairs A:B,B:C,...] -o out.json

extract keeps, per TP of the census's own table: event, true pT, |eta|, |d0|, phi, z0,
production radius, the hit patterns, the charge and the findable flag (census format >= 30:
credited hits per TP in format 30, credited modules from format 31); per build (the census's DR group): seed-truthful (found_dr over the build's
entries; definition sheet truth.seed_truthful of the census's own format), delivered (a DR
survivor of the build's full menu, DR recomputed from the exported tracks as
angle_stage_metrics does, owned by the TP: n_own >= 3 and more than half the hits) and
clean delivered (the same track without a wrong IT hit); per event: fitted tracks, DR
survivors and fakes (>= 2 wrong hits; truth.census_fake, raw) per seed class x arity, and
ambiguous owners (n_tp_majority >= 2, format >= 29). The *_sens columns are a SENSITIVITY to
the dominant-TP limit (the nano carries one TP per OT cluster, CMSSW credits every
contributor; CMSSW truth spec sec. 6), not a bound: delivered and fake if every wrong OT
stub that has a TP also credited the CURRENT owner.

compare quotes every efficiency on the same TP sets for every run, built once from the
universe run's table (a census of format >= 31), a TP missing from a run's table counting as
not found by that run:
  findable   THE CENSUS DENOMINATOR (definition sheet census.findable; decisions of
             2026-10-13 and 2026-10-14): charged, credited hits on >= 4 distinct modules,
             IT clusters plus OT stubs in any mix, each OT stub one hit, a TP's hits on one
             module one hit (TF.findable of the universe run)
  ot_cmssw   comparison only: the barrel analogue of CMSSW's L1TrackNtupleMaker
             denominator, charged TPs with credited stubs in >= 4 OT layers (hence >= 4
             stubs), event-id 0 as its plot macro (L1TrackNtuplePlot.C:1315)
at pT >= 2 GeV in |eta| < 1.1 (c), < 0.9 (h), 1.1-1.8 (f) and pT 1-2 GeV in |eta| < 1.1,
each |eta| range with its OT-truncated label (M.eta_range_label). It also reports, per pT
bin, what counting modules instead of layers adds, what one hit per module removes from a
format-30 run's per-hit count (given one), and how many neutral TPs the charge removes. Errors are the standard deviation over event-bootstrap replicas
(1000; the 100 events resampled with replacement, every figure recomputed as a ratio of
sums), PAIRED across runs (same replica weights: the runs share their events), so a
difference's error includes the correlation.
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
PT_REPORT = [(0.0, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, np.inf)]
DENS = {"findable": "the census denominator: charged, credited hits on >= 4 distinct modules (IT "
                    "clusters + OT stubs, each stub one hit, one hit per module), decisions of "
                    "2026-10-13 and 2026-10-14",
        "ot_cmssw": "comparison only: charged, credited stubs in >= 4 OT layers, event-id 0 "
                    "(L1TrackNtupleMaker analogue, barrel)"}
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
    key = np.asarray(C["key"], np.int64)
    count = ("n_mod_it", "n_mod_ot") if "n_mod_it" in C else ("n_hit_it", "n_hit_ot")
    R = {"label": label, "format": fmt, "n_events": nev, "census": d, "code_root": root,
         "builds": list(C["builds"]), "inputs": [r["path"] for r in man["inputs"]],
         "findable_count": count[0][:5] if count[0] in C else None}
    A = {"key": key, "event": np.asarray(C["event"], np.int32), "pt": C["pt"],
         "aeta": np.abs(C["eta"]), "ad0": np.abs(C["d0"]), "phi": C["phi"], "z0": C["z0"],
         "vr": C["vr"], "hit_it": C["hit_it"], "hit_ot": C["hit_ot"]}
    if R["findable_count"]:
        A.update(n_it=C[count[0]], n_ot=C[count[1]], charge=C["charge"], findable=TF.findable(C))
    seed_cls = {i: KF.sysclass_of(C["seeds"][i].layers) for i in range(len(C["seeds"]))}
    for b, ent in C["builds"].items():
        seed = {k: np.zeros(len(key), bool) for k in ("all", *SYS.values())}
        for i in ent:
            m = TF.seed_mask(C, i, after_dr=True)
            seed["all"] |= m
            seed[SYS[seed_cls[i]]] |= m
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
        if "n_tp_majority" in col:
            A[f"{b}/n_ambiguous_dr"] = np.bincount(ev_s[S[:, col["n_tp_majority"]] >= 2], minlength=nev)
        for fk, f in fakes.items():
            A[f"{b}/n_{fk}"] = np.bincount(ev_s[f], minlength=nev)
            for s, a in CATS:
                m = (scl == s) & (ar == a)
                A[f"{b}/n_dr_{SYS[s]}{a}"] = np.bincount(ev_s[m], minlength=nev)
                A[f"{b}/n_{fk}_{SYS[s]}{a}"] = np.bincount(ev_s[m & f], minlength=nev)
        for k, v in seed.items():
            A[f"{b}/seed_{k}"] = v
        print(f"{label} {b}: {len(T):,} fitted, {len(S):,} after DR", flush=True)
    np.savez_compressed(out, meta=json.dumps(R), **A)
    print(f"wrote {out}")


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
    """{figure: (value, replicas)} for one run: track rates and raw fake fractions."""
    nev = meta["n_events"]
    out = {}
    for b in meta["builds"]:
        for k in ("n_fit", "n_dr"):
            n = Z[f"{b}/{k}"].astype(float)
            out[f"{b}/{k}_per_ev"] = (float(n.sum() / nev), (W @ n) / nev)
        if f"{b}/n_ambiguous_dr" in Z:
            out[f"{b}/frac_dr_ambiguous_owner"] = _ratio(Z[f"{b}/n_ambiguous_dr"].astype(float),
                                                         Z[f"{b}/n_dr"].astype(float), W)
        for fk in ("fake", "fake_sens"):
            out[f"{b}/{fk}"] = _ratio(Z[f"{b}/n_{fk}"].astype(float), Z[f"{b}/n_dr"].astype(float), W)
            for s, a in CATS:
                c = f"{SYS[s]}{a}"
                out[f"{b}/{fk}_{c}"] = _ratio(Z[f"{b}/n_{fk}_{c}"].astype(float),
                                              Z[f"{b}/n_dr_{c}"].astype(float), W)
    return out


def l1ttp_evt(path):
    """(sorted tp_key, event-id) of every L1TTP row of the input."""
    import awkward as ak
    import uproot
    sys.path.insert(0, str(HERE))
    import tracklet_topology_cost as M
    A = uproot.open(f"{path}:Events").arrays(["L1TTP_evt"])
    nt = ak.to_numpy(ak.num(A["L1TTP_evt"]))
    key = M.tp_key(np.repeat(np.arange(len(nt)), nt), np.arange(int(nt.sum())) - np.repeat(np.cumsum(nt) - nt, nt))
    return key, ak.to_numpy(ak.flatten(A["L1TTP_evt"]))


def denominators(U, umeta, hits=None):
    """The census denominator and the CMSSW-like comparison one over the universe table
    U (a census of format >= 31), plus, per pT bin, what counting modules instead of
    layers adds, what one hit per module removes from the per-hit count of a format-30
    run `hits` (same TP table), and how many neutral TPs the charge requirement removes."""
    tkey, tevt = l1ttp_evt(umeta["inputs"][0])
    p = np.clip(np.searchsorted(tkey, U["key"]), 0, len(tkey) - 1)
    hard = (tkey[p] == U["key"]) & (tevt[p] == 0)
    charged = np.isfinite(U["charge"]) & (U["charge"] != 0)
    n_mod = U["n_it"].astype(np.int64) + U["n_ot"]
    n_lay = popcount8(U["hit_it"]) + popcount8(U["hit_ot"])
    if not np.array_equal(U["findable"], charged & (n_mod >= 4)):
        raise SystemExit("the universe's findable flag disagrees with charge and >= 4 modules")
    D = {"findable": U["findable"], "ot_cmssw": charged & (popcount8(U["hit_ot"]) >= 4) & hard}
    per_hit = None
    if hits is not None:
        if not np.array_equal(hits["key"], U["key"]):
            raise SystemExit("the per-hit run's TP table differs from the universe's")
        per_hit = hits["findable"]
    report = {}
    for eta_nm, (e0, e1) in (("all_census", (0.0, np.inf)), ("c", ETA["c"])):
        for lo, hi in PT_REPORT:
            k = (U["pt"] >= lo) & (U["pt"] < hi) & (U["aeta"] >= e0) & (U["aeta"] < e1)
            layers = charged & (n_lay >= 4)
            r = {"findable": int((k & D["findable"]).sum()),
                 "charged_4_layers": int((k & layers).sum()),
                 "added_by_counting_modules_not_layers": int((k & D["findable"] & ~layers).sum()),
                 "layer_findable_not_findable": int((k & layers & ~D["findable"]).sum()),
                 "neutral_with_4_modules_excluded": int((k & ~charged & (n_mod >= 4)).sum())}
            if per_hit is not None:
                r["findable_per_hit_format30"] = int((k & per_hit).sum())
                r["removed_by_one_hit_per_module"] = int((k & per_hit & ~D["findable"]).sum())
                r["added_by_one_hit_per_module"] = int((k & ~per_hit & D["findable"]).sum())
            report[f"{eta_nm}/pt_{lo:g}_{hi:g}"] = r
    return D, report


def found_on(Z, b, k, keys):
    """flag `k` of build b of run Z, on the TP keys `keys` (False where Z lacks the TP)."""
    p = np.clip(np.searchsorted(Z["key"], keys), 0, len(Z["key"]) - 1)
    return (Z["key"][p] == keys) & Z[f"{b}/{k}"][p]


def den_figures(runs, U, D, W, nev):
    """{run: {figure: (value, replicas)}}: seed-truthful, delivered, clean delivered and the
    delivered sensitivity on the denominators D (masks over U), the same TPs for every run."""
    out = {}
    for lab, (meta, Z) in runs.items():
        r = out.setdefault(lab, {})
        for b in meta["builds"]:
            for dn, dm in D.items():
                for pt, eta in EFFS:
                    sel = dm & kin_sel(U, pt, eta)
                    den = np.bincount(U["event"][sel], minlength=nev).astype(float)
                    r[f"{b}/{dn}/den_{pt}_{eta}"] = (float(den.sum()), W @ den)
                    for k in ("seed_all", "deliv", "clean", "deliv_sens"):
                        f = found_on(Z, b, k, U["key"][sel])
                        num = np.bincount(U["event"][sel][f], minlength=nev).astype(float)
                        r[f"{b}/{dn}/eff_{k}_{pt}_{eta}"] = _ratio(num, den, W)
    return out


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
    return {"n_tp_before": int(len(kb)), "n_tp_after": int(len(kc)), "every_tp_kept": kept,
            "frac_new_tps": float(new.sum() / len(kb)),
            "frac_new_tps_below_2gev": float((Zc["pt"][new] < 2.0).sum() / max(new.sum(), 1)),
            "frac_tps_gaining_ot_layer": float(gain_ot.mean()),
            "n_tps_losing_a_layer": int(lost.sum()),
            "kinematics_of_common_tps_identical": bool(same), "kinematics_compared": kin}


def _pair(F, x, y):
    d = {}
    for k in sorted(set(F[x]) & set(F[y])):
        vx, rx = F[x][k]
        vy, ry = F[y][k]
        sd = float(np.std(ry - rx, ddof=1))
        d[k] = {"before": vx, "after": vy, "diff": vy - vx, "sd": sd,
                "rel": (vy - vx) / vx if vx else None, "z": (vy - vx) / sd if sd > 0 else None}
    return d


def compare(paths, pairs, out, universe, per_hit=None):
    runs = {}
    for p in paths:
        meta, Z = load(p)
        runs[meta["label"]] = (meta, Z)
    nev = {m["n_events"] for m, _ in runs.values()}
    if len(nev) != 1:
        raise SystemExit(f"runs have different event counts {nev}")
    nev = nev.pop()
    umeta, U = runs[universe]
    if umeta.get("findable_count") != "n_mod":
        raise SystemExit(f"the universe run {universe} must be a census of format >= 31")
    W = np.random.default_rng(SEED).multinomial(nev, np.full(nev, 1.0 / nev), size=N_REP).astype(float)
    sys.path.insert(0, str(HERE))
    import tracklet_topology_cost as M
    D, hits_report = denominators(U, umeta, runs[per_hit][1] if per_hit else None)
    F = {lab: figures(m, Z, W) for lab, (m, Z) in runs.items()}
    G = den_figures(runs, U, D, W, nev)
    sd = lambda d: {k: {"value": v, "sd": float(np.std(r, ddof=1))} for k, (v, r) in d.items()}
    res = {"n_events": nev, "n_replicas": N_REP, "seed": SEED,
           "runs": {lab: {k: m.get(k) for k in ("format", "census", "code_root", "builds")}
                    for lab, (m, _) in runs.items()},
           "eta_labels": {k: M.eta_range_label(*v) for k, v in ETA.items()},
           "denominators": {"definitions": DENS, "universe": universe},
           "modules_vs_layers_hits_and_charge": hits_report,
           "figures": {lab: sd(f) for lab, f in F.items()},
           "efficiencies": {lab: sd(g) for lab, g in G.items()},
           "pairs": {}, "tp_changes": {}}
    for pr in pairs:
        x, y = pr.split(":")
        res["pairs"][pr] = {**_pair(F, x, y), **_pair(G, x, y)}
        res["tp_changes"][pr] = tp_changes(runs[x][1], runs[y][1])
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
    c = sub.add_parser("compare")
    c.add_argument("npz", nargs="+")
    c.add_argument("--pairs", default="A:B,B:C,A:C")
    c.add_argument("--universe", required=True, help="the label of a census of format >= 31")
    c.add_argument("--per-hit", default=None,
                   help="the label of a format-30 census (findable per hit), to report what one hit per module removes")
    c.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    if a.cmd == "extract":
        extract(a.label, a.census, a.root, a.out)
    else:
        compare(a.npz, a.pairs.split(","), a.out, a.universe, a.per_hit)


if __name__ == "__main__":
    main()
