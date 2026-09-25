"""Compose a seed MENU by what each seed uniquely adds, not by what it costs.

Ranking seeds by cost answers the wrong question: the eight cheapest seeds may
all recover the same TrackingParticles, in which case seven of them are free and
worthless. What a system wants is the seed that adds the most tracks NOBODY ELSE
finds, then the next, and so on -- a greedy composition by marginal gain.

So this runs every candidate seed END TO END on the unified IT+OT hit table
(IL1-IL4 = SmartPixels layers, OL1-OL6 = OT barrel), records the SET of
TrackingParticles each recovers, and then:

  1. reports the pairwise overlap, the confusion-matrix view of which seeds are
     substitutes for each other and which are complements;
  2. composes greedily -- pick the best single seed, then repeatedly add whichever
     seed raises the union most -- giving a RANKED MENU with the marginal
     efficiency each entry buys and the cost it brings.

The denominator throughout is TPs findable by ANY candidate seed, so seeds with
different layer acceptances are compared on one population rather than each on
its own.

Layer naming follows the instrumented-layer convention: a 1110 SmartPixels design
instruments IL1, IL2, IL3 and not IL4, so seeds are written IL3+IL1, IL2+OL1 etc.
and any seed touching an uninstrumented layer can be dropped by --layers.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tracklet_topology_cost as M   # noqa: E402

NAME = {1: "IL1", 2: "IL2", 3: "IL3", 4: "IL4",
        11: "OL1", 12: "OL2", 13: "OL3", 14: "OL4", 15: "OL5", 16: "OL6"}
CODE = {v: k for k, v in NAME.items()}


def build(src, nev, ptmin):
    I, n, _ = M.load_flat(src, M.IT_TABLE, list(M.IT_COLS), nev)
    O, on, _ = M.load_ot(src, nev, tp=("tp_z0", "tp_tanL"))
    bar = (O["isBarrel"] > 0) & (O["eta"] <= M.ETA_MATCHED)
    nO = int(bar.sum())
    # OT stub z resolution, measured per layer against the stub's L1TTP helix
    # z0 + r*tanL, used as the z-search target sigma
    ob = {c: O[c][bar] for c in ("layer", "tpIdx", "tpPt", "z", "r", "tp_z0", "tp_tanL")}

    def rs(x):
        q = np.percentile(x, [15.865, 84.135])
        return float(0.5 * (q[1] - q[0]))
    sig = np.full(nO, 0.2)
    for L in range(1, 7):
        m = ob["layer"] == L
        g = m & (ob["tpIdx"] >= 0) & (ob["tpPt"] >= ptmin) & np.isfinite(ob["tp_z0"])
        if g.sum() < 200:
            continue
        sig[m] = rs(ob["z"][g] - (ob["tp_z0"][g] + ob["r"][g] * ob["tp_tanL"][g]))
    U = {"layer": np.r_[I["layer"], O["layer"][bar] + 10],
         "globalR": np.r_[I["globalR"], O["r"][bar]],
         "globalZ": np.r_[I["globalZ"], O["z"][bar]],
         "globalPhi": np.r_[I["globalPhi"], O["phi"][bar]],
         "sigY": np.r_[I["sigY"], sig],
         "tpIdx": np.r_[I["tpIdx"], O["tpIdx"][bar]],
         "tpPt": np.r_[I["tpPt"], O["tpPt"][bar]],
         "event": np.r_[I["event"], O["event"][bar]]}
    QI = M.it_prepare({c: I[c] for c in M.IT_COLS}, None)
    # an OT stub carries no alpha/beta: huge sigmas make every angle gate pass,
    # and the joint (z0,phi) pairing then treats it as a wildcard on phi alone.
    Q = {"kap_a": np.r_[QI["kap_a"], np.zeros(nO)],
         "s_kap": np.r_[QI["s_kap"], np.full(nO, 1e9)],
         "z0": np.r_[QI["z0"], np.zeros(nO)],
         "s_z0": np.r_[QI["s_z0"], np.full(nO, 1e9)],
         "ovf_a": np.r_[QI["ovf_a"], np.zeros(nO, bool)],
         "ovf_z": np.r_[QI["ovf_z"], np.zeros(nO, bool)]}
    return U, Q, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-n", "--nev", type=int, default=200)
    ap.add_argument("--ptmin", type=float, default=2.0)
    ap.add_argument("--layers", default="IL1,IL2,IL3,IL4,OL1,OL2,OL3,OL4,OL5,OL6",
                    help="instrumented layers; e.g. a 1110 design drops IL4")
    ap.add_argument("--min-acc", type=float, default=20.0,
                    help="skip seeds whose pair acceptance is below this TP/event")
    ap.add_argument("--max-seeds", type=int, default=12, help="menu length")
    ap.add_argument("-o", "--out", default="eval_refitq/combinatorics/results/"
                                           "seed_menu_composition.json")
    a = ap.parse_args()
    use = [CODE[x.strip()] for x in a.layers.split(",") if x.strip()]
    U, Q, nev = build(a.input, a.nev, a.ptmin)
    M.set_limits(M.triplets_for_budget(1.5), min(6.0, M.SAFE_RSS_FRAC * M.PHYS_RAM_GB))
    allidx = np.arange(len(U["layer"]))

    # candidate seeds: every pair of instrumented layers, projected to the
    # remaining instrumented layer with the largest acceptance
    FP = {}
    for L in use:
        m = (U["layer"] == L) & (U["tpIdx"] >= 0) & (U["tpPt"] >= a.ptmin)
        FP[L] = np.unique(M.tp_key(U["event"][m], U["tpIdx"][m]))
    cands = []
    for i, la in enumerate(use):
        for lb in use[i + 1:]:
            acc = len(np.intersect1d(FP[la], FP[lb])) / nev
            if acc < a.min_acc:
                continue
            best = max((L for L in use if L not in (la, lb)),
                       key=lambda L: len(np.intersect1d(
                           np.intersect1d(FP[la], FP[lb]), FP[L])))
            cands.append((la, lb, best, acc))
    print(f"{nev} events, pT > {a.ptmin}.  {len(cands)} candidate seeds "
          f"from layers {a.layers}")

    found, cost, info = {}, {}, {}
    for la, lb, lc, acc in cands:
        tag = f"{NAME[la]}+{NAME[lb]}>{NAME[lc]}"
        try:
            o = M.it_pair_seed(U, Q, allidx, la, lb, lc, a.ptmin, True, False, 0.0)
        except M.TooWide as e:
            print(f"  {tag:<18} INFEASIBLE ({e.n:.2g} candidate triplets)")
            continue
        if "_trip" not in o:
            continue
        found[tag] = M.recovered_keys(U, *o["_trip"])
        nc, nt = M.cand_purity(U, *o["_trip"])
        cost[tag] = {"pairs": o.get("tracklets", 0) / nev,
                     "cand": o.get("match_cand", 0) / nev,
                     "fit": o.get("tracks_to_fit", 0) / nev,
                     "fake": 1.0 - nt / max(nc, 1)}
        info[tag] = {"acc_tp_per_event": acc}
    if not found:
        raise SystemExit("no viable seeds")
    # common denominator: findable by ANY candidate seed
    denom = np.unique(np.concatenate([found[t] for t in found]))
    print(f"\ncommon denominator: {len(denom):,} TPs recovered by at least one seed\n")

    # greedy composition by marginal gain
    menu, have = [], np.empty(0, np.int64)
    pool = dict(found)
    while pool and len(menu) < a.max_seeds:
        best = max(pool, key=lambda t: len(np.setdiff1d(pool[t], have)))
        gain = len(np.setdiff1d(pool[best], have))
        have = np.union1d(have, pool.pop(best))
        menu.append({"seed": best, "marginal_tps": int(gain),
                     "cumulative_eff": float(len(have) / len(denom)),
                     "own_eff": float(len(found[best]) / len(denom)),
                     **cost[best], **info[best]})
    print(f"{'#':>3} {'seed':<18}{'own eff':>9}{'marginal':>10}{'cum eff':>9}"
          f"{'fake':>7}{'cand/ev':>10}{'fit/ev':>8}")
    for i, m in enumerate(menu, 1):
        print(f"{i:>3} {m['seed']:<18}{m['own_eff']:>9.3f}{m['marginal_tps']:>10,d}"
              f"{m['cumulative_eff']:>9.3f}{m['fake']:>7.3f}{m['cand']:>10,.0f}"
              f"{m['fit']:>8,.0f}")
    # pairwise overlap among the menu: substitutes vs complements
    tags = [m["seed"] for m in menu]
    ov = [[float(len(np.intersect1d(found[b], found[c])) / max(len(found[c]), 1))
           for c in tags] for b in tags]
    print(f"\noverlap M[i][j] = fraction of column j's TPs also found by row i")
    print(f"{'':<18}" + "".join(f"{t.split('>')[0]:>10}" for t in tags))
    for i, b in enumerate(tags):
        print(f"{b:<18}" + "".join(f"{ov[i][j]:>10.2f}" for j in range(len(tags))))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"n_events": nev, "pt_min": a.ptmin, "layers": a.layers,
               "denominator": int(len(denom)), "menu": menu,
               "overlap": {"tags": tags, "matrix": ov},
               "all_seeds": {t: {"n_found": int(len(found[t])), **cost[t], **info[t]}
                             for t in found}}, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
