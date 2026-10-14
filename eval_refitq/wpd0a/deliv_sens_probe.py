"""WP-D0a: why does the delivered SENSITIVITY (before_after deliv_sens) fall from run B to run C?

  dump LABEL CENSUS ROOT OUT.npz   per build, the DR survivors' truth columns (read by the census's own root,
                                   DR recomputed exactly as before_after.extract does)
  compare B.npz C.npz UNIVERSE_ba.npz OUT.json
                                   B and C have identical tracks and DR (tracks_identical B-C), so survivor
                                   rows align; per build and bin, the findable TPs whose deliv_sens flips, and
                                   for each the tracks that carried it, classified.
  rules CENSUS ROOT B_CENSUS C_CENSUS OUT.json
                                   on the tracks of a census of format >= 32 (identical to B's and C's), each
                                   track's owner recomputed from its hits (the input re-read chunk by chunk as the
                                   census reads it) under credit {B: each hit's own tpIdx (formats <= 27), C:
                                   M.credited_tps} x ties {index: the larger index (formats <= 31), none: no owner
                                   (format >= 32, as CMSSW)}; B/index, C/index and C/none must reproduce the owners
                                   stored by B, C and CENSUS track for track. Per build and bin, on CENSUS's findable
                                   TPs: seed-independent delivered, clean delivered and deliv_sens in each cell, with
                                   event-bootstrap errors (before_after's replicas) and the C - B differences.
deliv_sens (before_after): a DR survivor with owner >= 0 and n = n_own + n_ot_wrong + n_ot_combinatoric
satisfying n >= 3 and n > nhit / 2 delivers its OWNER."""
import json, os, sys
import numpy as np

COLS = ("event", "own_tpidx", "n_own", "nhit", "n_ot_wrong", "n_ot_combinatoric", "n_ot_right", "n_ot_unknown",
        "n_it_right", "n_it_wrong")


def dump(label, census, root, out):
    sys.path.insert(0, os.path.join(os.environ["SP_FROZEN_ROOT"], "eval_refitq", "wpd0a"))
    import before_after as BA
    TF, SA, KF = BA._root_modules(root)
    d = TF.census_dir(census)
    C = TF.load(d, with_tracks=True, verbose=False)
    col = {c: j for j, c in enumerate(C["track_cols"])}
    A = {}
    for b, ent in C["builds"].items():
        es = [i for i in sorted(ent) if i in C["tracks"] and len(C["tracks"][i])]
        T = np.concatenate([C["tracks"][i] for i in es])
        H = np.concatenate([C["hits"][i] for i in es])
        surv = SA.duplicate_removal(H, T[:, col["rank_score"]].astype(np.float64))
        S = T[surv]
        for c in COLS:
            A[f"{b}/{c}"] = S[:, col[c]].astype(np.int64)
        A[f"{b}/hits"] = H[surv]
        print(label, b, len(S), "survivors", flush=True)
    np.savez_compressed(out, **A)


def compare(pb, pc, pu, out):
    import importlib
    sys.path.insert(0, os.path.join(os.environ["SP_FROZEN_ROOT"], "eval_refitq", "combinatorics"))
    M = importlib.import_module("tracklet_topology_cost")
    B, C, U = np.load(pb), np.load(pc), np.load(pu)
    res = {}
    for b in ("AAAA", "AAAI", "IAAA"):
        X = {r: {c: Z[f"{b}/{c}"] for c in COLS} for r, Z in (("B", B), ("C", C))}
        assert np.array_equal(X["B"]["event"], X["C"]["event"]) and np.array_equal(B[f"{b}/hits"], C[f"{b}/hits"])
        for r in "BC":
            x = X[r]
            n = x["n_own"] + x["n_ot_wrong"] + x["n_ot_combinatoric"]
            x["sens"] = (x["own_tpidx"] >= 0) & (n >= 3) & (n > 0.5 * x["nhit"])
            x["held"] = (x["own_tpidx"] >= 0) & (x["n_own"] >= 3) & (x["n_own"] > 0.5 * x["nhit"])
            x["n_sens"] = n
            x["okey"] = M.tp_key(x["event"], np.maximum(x["own_tpidx"], 0))
        xb, xc = X["B"], X["C"]
        same_owner = xb["own_tpidx"] == xc["own_tpidx"]
        for pt, (lo, hi) in (("lo", (1.0, 2.0)), ("hi", (2.0, np.inf))):
            k = U["findable"] & (U["pt"] >= lo) & (U["pt"] < hi) & (U["aeta"] < 1.1)
            keys = U["key"][k]
            fb = np.isin(keys, xb["okey"][xb["sens"]])
            fc = np.isin(keys, xc["okey"][xc["sens"]])
            lost, gained = keys[fb & ~fc], keys[~fb & fc]
            # the B tracks that delivered a lost TP (sens), and what C makes of the same rows
            rows = np.flatnonzero(xb["sens"] & np.isin(xb["okey"], lost))
            oc = ~same_owner[rows]
            new_owner_sens = xc["sens"][rows] & oc
            new_owner_findable = np.isin(xc["okey"][rows], U["key"][U["findable"]]) & oc
            # same owner, sens lost: n_own + hid moved
            so = rows[same_owner[rows]]
            r = {"n_findable": int(k.sum()), "sens_B": int(fb.sum()), "sens_C": int(fc.sum()),
                 "lost": int(len(lost)), "gained": int(len(gained)),
                 "deliv_B": int(np.isin(keys, xb["okey"][xb["held"]]).sum()),
                 "deliv_C": int(np.isin(keys, xc["okey"][xc["held"]]).sum()),
                 "tracks_carrying_lost": int(len(rows)),
                 "of_which_owner_changed_in_C": int(oc.sum()),
                 "owner_changed_new_owner_sens_in_C": int(new_owner_sens.sum()),
                 "owner_changed_new_owner_findable": int(new_owner_findable.sum()),
                 "owner_changed_to_none": int((xc["own_tpidx"][rows] < 0).sum()),
                 "of_which_same_owner": int(len(so)),
                 "same_owner_n_sens_B_minus_C": np.bincount(np.clip(xb["n_sens"][so] - xc["n_sens"][so], 0, 9)).tolist()
                 if len(so) else [],
                 # for the owner-changed rows: the lost TP's credits in C (n hits crediting it is not stored;
                 # bound it by B's n_own, which counted its own-tpIdx hits) vs the new owner's n_own in C
                 "owner_changed_B_n_own_hist": np.bincount(xb["n_own"][rows[oc]], minlength=7)[:10].tolist(),
                 "owner_changed_C_n_own_hist": np.bincount(xc["n_own"][rows[oc]], minlength=7)[:10].tolist(),
                 "owner_changed_nhit_hist": np.bincount(xb["nhit"][rows[oc]], minlength=7)[:10].tolist(),
                 "owner_changed_B_combinatoric_hist": np.bincount(xb["n_ot_combinatoric"][rows[oc]], minlength=5)[:8].tolist()}
            # gained TPs: why C delivers them
            grow = np.flatnonzero(xc["sens"] & np.isin(xc["okey"], gained))
            r["tracks_carrying_gained"] = int(len(grow))
            r["gained_owner_changed"] = int((~same_owner[grow]).sum())
            res[f"{b}/{pt}"] = r
            print(b, pt, json.dumps(r), flush=True)
    json.dump(res, open(out, "w"), indent=1)


def owner_index_ties(tp, valid):
    """The census owner of formats <= 31 (KF.majority_owner before format 32): the most
    credited TP, a tie to the larger index; unmatched never wins."""
    from ngtagger.data import spix_nano as SPN
    none = np.iinfo(np.int64).min
    v = np.where(valid & SPN.tp_matched(tp), np.asarray(tp, np.int64), np.int64(none))
    S = np.sort(v, axis=1)[:, ::-1]
    cur_v, cur_n = S[:, 0].copy(), (S[:, 0] != none).astype(np.int64)
    best_v, best_n = cur_v.copy(), cur_n.copy()
    for j in range(1, S.shape[1]):
        col = S[:, j]
        real = col != none
        same = real & (col == cur_v)
        cur_n = np.where(same, cur_n + 1, np.where(real, 1, 0))
        cur_v = np.where(same, cur_v, col)
        take = cur_n > best_n
        best_n = np.where(take, cur_n, best_n)
        best_v = np.where(take, cur_v, best_v)
    return np.where(best_n > 0, best_v, -1), best_n


def _stored(census, entry_tracks):
    """{entry: (own_tpidx, n_own)} of a census's stored track rows, by column name
    (tracks_identical.track_cols: the manifest's list, or this tree's prefix)."""
    import glob
    import tracks_identical as TI
    d = TI.census_dir(census)
    man = json.load(open(os.path.join(d, "manifest.json")))
    out = {}
    for sh in sorted(glob.glob(os.path.join(d, "chunk_*.npz"))):
        z = np.load(sh)
        for f in z.files:
            if f.startswith("trk_") and int(f[4:]) in entry_tracks:
                x = z[f]
                c = TI.track_cols(man, x.shape[1])
                out.setdefault(int(f[4:]), []).append(x[:, [c.index("own_tpidx"), c.index("n_own")]])
    return {e: np.concatenate(v) for e, v in out.items()}


def rules(census, root, census_b, census_c, out):
    sys.path.insert(0, os.path.join(os.environ["SP_FROZEN_ROOT"], "eval_refitq", "wpd0a"))
    import before_after as BA
    TF, SA, KF = BA._root_modules(root)
    M = TF.M
    d = TF.census_dir(census)
    meta = json.load(open(os.path.join(d, "manifest.json")))
    if meta["format"] < 32:
        raise SystemExit("rules needs a census of format >= 32 (n_tp_top, the no-tie owner)")
    C = TF.load(d, with_tracks=True, verbose=False)
    col = {c: j for j, c in enumerate(C["track_cols"])}
    nev = int(C["n_events"])
    sigz = {int(k): float(v) for k, v in meta["calibration"]["sigz_ot"].items()}
    if meta.get("ot_bend_cal"):
        M.set_ot_bend_cal({int(k): tuple(v) for k, v in meta["ot_bend_cal"].items()})
    cfg = meta["config"]
    if cfg.get("bench") or cfg["angle_res"].get("alpha_k", 1.0) < 1.0 or cfg["angle_res"].get("beta_k", 1.0) < 1.0:
        raise SystemExit("rules re-reads the input with full-float angles and no resolution knob only")
    spec = ",".join(r["path"] for r in meta["inputs"])
    import glob
    from ngtagger.data import spix_nano as SPN
    shards = sorted(glob.glob(os.path.join(d, "chunk_*.npz")))
    CELLS = [("B", "index"), ("C", "index"), ("C", "none")]
    per = {}                                   # entry -> list over chunks of {cell: arrays}
    for ci, ((U, Q), ev) in enumerate(TF.unified_chunks(spec, nev, cfg["chunk"], sigz, None,
                                                        {"alpha_k": 1.0, "beta_k": 1.0})):
        z = np.load(shards[ci])
        own_b = M.credited_tps(U["tpIdx"])     # the hit's own tpIdx alone (formats <= 27)
        is_ot_u = U["layer"] >= 11
        for f in z.files:
            if not f.startswith("hit_"):
                continue
            e = int(f[4:])
            G = z[f].astype(np.int64)
            on = G >= 0
            g = np.clip(G, 0, None)
            flg = np.where(on, U["ot_flag"][g], 0)
            is_ot = on & is_ot_u[g]
            is_it = on & ~is_ot_u[g]
            res = {"nh": on.sum(1)}
            for cr, tie in CELLS + [("B", "none")]:
                tps = np.where(on[..., None], (own_b if cr == "B" else U["tps"])[g], -1)
                n = tps.shape[-1]
                flat, von = tps.reshape(len(tps), -1), np.repeat(on, n, axis=1)
                if tie == "index":
                    own, n_own = owner_index_ties(flat, von)
                else:
                    own, n_own, _ = KF.majority_owner(flat, von)
                owned = SPN.tp_matched(own)
                right = on & owned[:, None] & (tps == own[:, None, None]).any(axis=-1)
                assert np.array_equal(right.sum(1), np.where(owned, n_own, 0))
                res[(cr, tie)] = {"own": own, "n_own": np.where(owned, n_own, 0),
                                  "hid": (is_ot & (flg == 1) & ~right).sum(1) + (is_ot & (flg == 2) & ~right).sum(1),
                                  "it_wrong": (is_it & ~right).sum(1)}
            per.setdefault(e, []).append(res)
        print("chunk", ci, flush=True)
    cat = {e: {k: (np.concatenate([r[k] for r in v]) if not isinstance(k, tuple) else
                   {q: np.concatenate([r[k][q] for r in v]) for q in v[0][k]})
               for k in v[0]} for e, v in per.items()}
    # VALIDATION: B/index, C/index, C/none reproduce B's, C's and this census's stored owners track for track
    valid = {}
    for lab, cen, cell in (("B", census_b, ("B", "index")), ("C", census_c, ("C", "index")),
                           ("this", census, ("C", "none"))):
        st = _stored(cen, set(cat))
        bad = sum(int(np.sum((st[e][:, 0] != cat[e][cell]["own"]) | (st[e][:, 1] != cat[e][cell]["n_own"])))
                  for e in cat)
        valid[f"{lab} stored owners vs {cell[0]}/{cell[1]}"] = {"tracks": int(sum(len(cat[e]["nh"]) for e in cat)),
                                                              "mismatches": bad}
        print("validation", lab, valid[f"{lab} stored owners vs {cell[0]}/{cell[1]}"], flush=True)
    W = np.random.default_rng(BA.SEED).multinomial(nev, np.full(nev, 1.0 / nev), size=BA.N_REP).astype(float)
    key = np.asarray(C["key"], np.int64)
    fnd = TF.findable(C)
    res = {"validation": valid, "cells": [f"{a}/{b}" for a, b in CELLS + [("B", "none")]], "eff": {}, "diff": {}}
    for b, ent in C["builds"].items():
        es = [i for i in sorted(ent) if i in C["tracks"] and len(C["tracks"][i])]
        T = np.concatenate([C["tracks"][i] for i in es])
        H = np.concatenate([C["hits"][i] for i in es])
        surv = SA.duplicate_removal(H, T[:, col["rank_score"]].astype(np.float64))
        ev = T[surv, col["event"]].astype(np.int64)
        nh = np.concatenate([cat[i]["nh"] for i in es])[surv]
        assert np.array_equal(nh, T[surv, col["nhit"]].astype(np.int64)), "hit tables do not align with the tracks"
        flags = {}
        for cell in CELLS + [("B", "none")]:
            x = {q: np.concatenate([cat[i][cell][q] for i in es])[surv] for q in ("own", "n_own", "hid", "it_wrong")}
            owned = x["own"] >= 0
            hw = lambda n: owned & (n >= 3) & (n > 0.5 * nh)
            okey = M.tp_key(ev, np.maximum(x["own"], 0))
            held = hw(x["n_own"])
            for nm, m in (("deliv", held), ("clean", held & (x["it_wrong"] == 0)), ("deliv_sens", hw(x["n_own"] + x["hid"]))):
                flags[(cell, nm)] = np.isin(key, okey[m])
        for pt, eta in BA.EFFS:
            sel = fnd & BA.kin_sel({"pt": C["pt"], "aeta": np.abs(C["eta"])}, pt, eta)
            den = np.bincount(np.asarray(C["event"])[sel], minlength=nev).astype(float)
            reps = {}
            for (cell, nm), f in flags.items():
                num = np.bincount(np.asarray(C["event"])[sel & f], minlength=nev).astype(float)
                v, r = BA._ratio(num, den, W)
                reps[(cell, nm)] = r
                res["eff"][f"{b}/{nm}_{pt}_{eta}/{cell[0]}/{cell[1]}"] = {"value": v, "sd": float(np.std(r, ddof=1)),
                                                                          "n": int(num.sum()), "den": int(den.sum())}
            for nm in ("deliv", "clean", "deliv_sens"):
                for tie in ("index", "none"):
                    a, c = res["eff"][f"{b}/{nm}_{pt}_{eta}/B/{tie}"], res["eff"][f"{b}/{nm}_{pt}_{eta}/C/{tie}"]
                    res["diff"][f"{b}/{nm}_{pt}_{eta}/C-B/{tie}"] = {
                        "value": c["value"] - a["value"],
                        "sd": float(np.std(reps[(("C", tie), nm)] - reps[(("B", tie), nm)], ddof=1))}
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    if sys.argv[1] == "dump":
        dump(*sys.argv[2:6])
    elif sys.argv[1] == "rules":
        rules(*sys.argv[2:7])
    else:
        compare(*sys.argv[2:6])
