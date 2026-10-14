"""WP-D0a: why does the delivered SENSITIVITY (before_after deliv_sens) fall from run B to run C?

  dump LABEL CENSUS ROOT OUT.npz   per build, the DR survivors' truth columns (read by the census's own root,
                                   DR recomputed exactly as before_after.extract does)
  compare B.npz C.npz UNIVERSE_ba.npz OUT.json
                                   B and C have identical tracks and DR (tracks_identical B-C), so survivor
                                   rows align; per build and bin, the findable TPs whose deliv_sens flips, and
                                   for each the tracks that carried it, classified.
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


if __name__ == "__main__":
    if sys.argv[1] == "dump":
        dump(*sys.argv[2:6])
    else:
        compare(*sys.argv[2:6])
