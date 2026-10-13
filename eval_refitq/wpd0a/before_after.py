"""WP-D0a before/after: runs A (format 26), B (27, OT module flags) and C (28, stub
credit) on the same events, one set of figures of merit, paired event-bootstrap errors.

    # each census read by its OWN code root (a census refuses other formats); with
    # PYTHONPATH=ROOT/src so ngtagger comes from the same root
    python before_after.py extract LABEL CENSUS ROOT OUT.npz
    python before_after.py compare A.npz B.npz C.npz [--pairs A:B,B:C,A:C] -o out.json

extract keeps, per TP (the census's own table): event, true pT, |eta|, |d0|, layers
hit and the hit patterns; per build (the census's DR group): seed-truthful (found_dr
over the build's entries; definition sheet truth.seed_truthful of the census's own
format), delivered (a DR survivor of the build's full menu, DR recomputed from the
exported tracks as angle_stage_metrics does, owned by the TP: n_own >= 3 and more
than half the hits) and clean delivered (the same track without a wrong IT hit), all
also per seed class (IT, OT, mixed); per event: fitted tracks, DR survivors and fakes
(>= 2 wrong hits; definition sheet truth.census_fake, raw) per seed class x arity. Also the
bound of the dominant-TP limit (the nano carries one TP per OT cluster, CMSSW credits every
contributor; cmssw_truth_spec.md section 6): "deliv_ub" and "fake_lb" count every wrong OT
stub that has a TP as crediting the owner as well.

compare: efficiencies over TPs with >= 4 layers at pT >= 2 GeV in |eta| < 1.1 (c),
< 0.9 (h), 1.1-1.8 (f), and pT 1-2 GeV in |eta| < 1.1; |eta| ranges carry the
OT-truncated label (M.eta_range_label). Errors are the standard deviation over
event-bootstrap replicas (1000; the 100 events resampled with replacement, every
figure recomputed as a ratio of sums), PAIRED across runs (same replica weights:
the runs share their events), so a difference's error includes the correlation.
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
    C = TF.load(d, with_tracks=True, verbose=False)
    fmt = json.load(open(os.path.join(d, "manifest.json")))["format"]
    nev = int(C["n_events"])
    col = {c: j for j, c in enumerate(C["track_cols"])}
    key = np.asarray(C["key"], np.int64)
    nl = (np.unpackbits(C["hit_it"][:, None], axis=1).sum(1)
          + np.unpackbits(C["hit_ot"][:, None], axis=1).sum(1))
    R = {"label": label, "format": fmt, "n_events": nev, "census": d, "code_root": root,
         "builds": list(C["builds"])}
    A = {"key": key, "event": np.asarray(C["event"], np.int32), "pt": C["pt"],
         "aeta": np.abs(C["eta"]), "ad0": np.abs(C["d0"]), "nlay": nl.astype(np.int8),
         "hit_it": C["hit_it"], "hit_ot": C["hit_ot"]}
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
        held = (own >= 0) & (n_own >= 3) & (n_own > 0.5 * nh)
        clean = held & (S[:, col["n_it_wrong"]] == 0)
        # BOUND on the dominant-TP limit (the nano carries one TP per OT cluster, CMSSW credits every
        # contributor): every wrong OT stub that has a TP (genuine of another TP, or combinatoric) also
        # credited to the owner. Unknown stubs have no TP; IT clusters stay dominant-only.
        hid = S[:, col["n_ot_wrong"]] + S[:, col["n_ot_combinatoric"]]
        held_ub = (own >= 0) & (n_own + hid >= 3) & (n_own + hid > 0.5 * nh)
        okey = M.tp_key(S[:, col["event"]].astype(np.int64), np.maximum(own, 0).astype(np.int64))
        scl = S[:, col["sysclass"]].astype(int)
        dl = {}
        for nm, sel in (("deliv", held), ("clean", clean), ("deliv_ub", held_ub)):
            dl[nm] = np.isin(key, okey[sel])
            for k, v in SYS.items():
                dl[f"{nm}_{v}"] = np.isin(key, okey[sel & (scl == k)])
        ev_t = T[:, col["event"]].astype(np.int64)
        ev_s = S[:, col["event"]].astype(np.int64)
        fake = S[:, col["n_wrong"]] >= 2
        A[f"{b}/n_fake_lb"] = np.bincount(ev_s[S[:, col["n_wrong"]] - hid >= 2], minlength=nev)
        A[f"{b}/n_fit"] = np.bincount(ev_t, minlength=nev)
        A[f"{b}/n_dr"] = np.bincount(ev_s, minlength=nev)
        A[f"{b}/n_fake"] = np.bincount(ev_s[fake], minlength=nev)
        ar = S[:, col["arity"]].astype(int)
        for s, a in CATS:
            m = (scl == s) & (ar == a)
            A[f"{b}/n_dr_{SYS[s]}{a}"] = np.bincount(ev_s[m], minlength=nev)
            A[f"{b}/n_fake_{SYS[s]}{a}"] = np.bincount(ev_s[m & fake], minlength=nev)
        for k, v in seed.items():
            A[f"{b}/seed_{k}"] = v
        for k, v in dl.items():
            A[f"{b}/{k}"] = v
        print(f"{label} {b}: {len(T):,} fitted, {len(S):,} after DR", flush=True)
    np.savez_compressed(out, meta=json.dumps(R), **A)
    print(f"wrote {out}")


# ------------------------------------------------------------------ compare
def load(p):
    z = np.load(p, allow_pickle=False)
    return json.loads(str(z["meta"])), {k: z[k] for k in z.files if k != "meta"}


def tp_sel(Z, pt, eta):
    lo, hi = PT[pt]
    e0, e1 = ETA[eta]
    return (Z["nlay"] >= 4) & (Z["pt"] >= lo) & (Z["pt"] < hi) & (Z["aeta"] >= e0) & (Z["aeta"] < e1)


def figures(meta, Z, W):
    """{figure: (value, replicas)} for one run."""
    nev = meta["n_events"]
    out = {}

    def ratio(num, den):
        return float(num.sum() / max(den.sum(), 1)), (W @ num) / np.maximum(W @ den, 1)

    for b in meta["builds"]:
        for pt, eta in EFFS:
            sel = tp_sel(Z, pt, eta)
            den = np.bincount(Z["event"][sel], minlength=nev).astype(float)
            out[f"{b}/den_{pt}_{eta}"] = (float(den.sum()), W @ den)
            for k in ["seed_all", "deliv", "clean", "deliv_ub"] + [f"{x}_{v}" for x in ("seed", "deliv")
                                                       for v in SYS.values()]:
                num = np.bincount(Z["event"][sel & Z[f"{b}/{k}"]], minlength=nev).astype(float)
                out[f"{b}/eff_{k}_{pt}_{eta}"] = ratio(num, den)
        for k in ("n_fit", "n_dr"):
            n = Z[f"{b}/{k}"].astype(float)
            out[f"{b}/{k}_per_ev"] = (float(n.sum() / nev), (W @ n) / nev)
        out[f"{b}/fake"] = ratio(Z[f"{b}/n_fake"].astype(float), Z[f"{b}/n_dr"].astype(float))
        out[f"{b}/fake_lb"] = ratio(Z[f"{b}/n_fake_lb"].astype(float), Z[f"{b}/n_dr"].astype(float))
        for s, a in CATS:
            c = f"{SYS[s]}{a}"
            out[f"{b}/fake_{c}"] = ratio(Z[f"{b}/n_fake_{c}"].astype(float), Z[f"{b}/n_dr_{c}"].astype(float))
    return out


def tp_changes(Zb, Zc):
    """What the stub credit did to the TP table: new TPs and OT layers gained."""
    kb, kc = Zb["key"], Zc["key"]
    new = ~np.isin(kc, kb)
    pos = np.clip(np.searchsorted(kc, kb), 0, len(kc) - 1)
    kept = bool(np.all(kc[pos] == kb))
    gain_ot = Zc["hit_ot"][pos] != Zb["hit_ot"]
    lost = ((Zc["hit_ot"][pos] & Zb["hit_ot"]) != Zb["hit_ot"]) | (Zc["hit_it"][pos] != Zb["hit_it"])
    kin = all(np.array_equal(Zc[c][pos], Zb[c], equal_nan=True) for c in ("pt", "aeta", "ad0", "event"))
    den = lambda Z: int(((Z["nlay"] >= 4) & (Z["pt"] >= 2.0) & (Z["aeta"] < 1.1)).sum())
    return {"n_tp_before": int(len(kb)), "n_tp_after": int(len(kc)), "every_tp_kept": kept,
            "frac_new_tps": float(new.sum() / len(kb)),
            "frac_new_tps_below_2gev": float((Zc["pt"][new] < 2.0).sum() / max(new.sum(), 1)),
            "frac_new_tps_pt_nan": float(np.isnan(Zc["pt"][new]).mean()) if new.any() else 0.0,
            "frac_tps_gaining_ot_layer": float(gain_ot.mean()),
            "n_tps_losing_a_layer": int(lost.sum()),
            "kinematics_of_common_tps_identical": bool(kin),
            "den_hi_c_before": den(Zb), "den_hi_c_after": den(Zc)}


def compare(paths, pairs, out):
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
    res = {"n_events": nev, "n_replicas": N_REP, "seed": SEED,
           "runs": {lab: {k: m[k] for k in ("format", "census", "code_root", "builds")}
                    for lab, (m, _) in runs.items()},
           "eta_labels": {}, "figures": {}, "pairs": {}, "tp_changes": {}}
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "combinatorics"))
    import tracklet_topology_cost as M
    res["eta_labels"] = {k: M.eta_range_label(*v) for k, v in ETA.items()}
    for lab, f in F.items():
        res["figures"][lab] = {k: {"value": v, "sd": float(np.std(r, ddof=1))} for k, (v, r) in f.items()}
    for pr in pairs:
        x, y = pr.split(":")
        common = sorted(set(F[x]) & set(F[y]))
        d = {}
        for k in common:
            vx, rx = F[x][k]
            vy, ry = F[y][k]
            diff = vy - vx
            sd = float(np.std(ry - rx, ddof=1))
            d[k] = {"before": vx, "after": vy, "diff": diff, "sd": sd,
                    "rel": diff / vx if vx else None, "z": diff / sd if sd > 0 else None}
        res["pairs"][pr] = d
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
    c.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    if a.cmd == "extract":
        extract(a.label, a.census, a.root, a.out)
    else:
        compare(a.npz, a.pairs.split(","), a.out)


if __name__ == "__main__":
    main()
