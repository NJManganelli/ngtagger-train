"""Quick numbers for the pT-ranked cap of three clusters per sub-area (physicist's ruling, 2026-11-11), prompt
collection, AAAA, the scan files:
  - clusters of charged particles with true pT >= 2 GeV (class T, charge fraction >= 0.5) lost to the cap per layer,
    three kept at random against the three of highest pT from the bending angle;
  - how well the bending-angle pT tracks the true pT on those clusters;
  - network, float with the pT-ranked cap: d0 68 % width of b/c daughters with pT >= 5 GeV over "implemented now",
    beside the stored scan values for the random cap and for all clusters sent.

  python cap_quick.py     -> checks/cap_pt_ranked_quick.json
"""
import json

import numpy as np

import emulate as EM
import run_scan as RS
import study as ST
from study import AR, CM


def w68(x):
    return float(np.quantile(np.abs(x[np.isfinite(x)]), 0.6827))


def main():
    S = RS.Study()
    scen = "AAAA"
    reg = json.load(open(RS.HERE / "arms.json"))
    old = RS.load_registry(ST.LOCAL)                       # the first scan's frozen registry (unit composition)
    lost = {}
    ptq = []
    rows = {"ev": [], "now": [], "pt": [], "rnd": [], "all": [], "tp_d0": [], "sel": []}
    for fi, f in enumerate(ST.SCAN_FILES):
        truth, D = RS.run_unit(S, reg, f, scen, ["implemented_now", "network_float"])
        T = np.load(ST.LOCAL / "units" / "truth" / f"{scen}_f{f:02d}_truth.npz")
        assert np.array_equal(T["event"], truth["event"]) and np.array_equal(T["track"], truth["track"])
        stored = {}
        for name in ("implemented_now", "network_float", "network_float_allsent"):
            u = next(x for x in RS.units_of(old, ST.SCAN_FILES, ST.SCENARIOS) if name in x["arms"] and x["scen"] == scen and x["file"] == f)
            stored[name] = np.load(ST.LOCAL / "units" / u["name"])[f"{name}__a"]
        assert np.array_equal(stored["implemented_now"], D["implemented_now"]["a"])     # the same replay
        sel = (truth["origin"] == 1) & (truth["tp_pt"] >= 5)
        rows["ev"].append(truth["entry"].astype(np.int64) + 1000 * fi)
        rows["sel"].append(sel)
        rows["tp_d0"].append(truth["tp_d0"])
        rows["now"].append(D["implemented_now"]["a"][:, 4])
        rows["pt"].append(D["network_float"]["a"][:, 4])
        rows["rnd"].append(stored["network_float"][:, 4])
        rows["all"].append(stored["network_float_allsent"][:, 4])
        # cluster level
        path = ST.input_path(scen, f)
        keys = AR.OP._keys(str(path))
        br = list(dict.fromkeys(AR.branches(path) + ["run", "luminosityBlock", "event"]
                                + AR.SPN.cluster_branches(keys, list(EM.EXTRA_CLUSTER_COLS), disc_scope="all")))
        for start, A in CM.iterate(path, br, chunk=10):
            ctx = AR.prepare(A, S.geo)
            B = EM.base(ctx, S.geo, A, S.net, S.links, S.LT)
            surf = S.geo["surface"][np.maximum(ctx["K"]["mi"], 0)]
            tg = B["target"] & (ctx["K"]["mi"] >= 0)
            for s_ in np.unique(surf[tg]):
                m = tg & (surf == s_)
                e = lost.setdefault(int(s_), [0, 0, 0, 0])
                e[0] += int(m.sum())
                e[1] += int((~B["sent"][m]).sum())
                e[2] += int((~B["sent_pt"][m]).sum())
                e[3] += int((B["n_in_subarea"][m] > 3).sum())
            m1 = tg & (surf == 1)
            tp_pt = ctx["TP"]["pt"][np.maximum(ctx["K"]["tp_global"], 0)]
            ptq.append(np.stack([tp_pt[m1], B["pt_bend"][m1]], 1))
    R = {k: np.concatenate(v) for k, v in rows.items()}
    m = R["sel"]
    rng = np.random.default_rng(1)
    u, inv = np.unique(R["ev"][m], return_inverse=True)

    def ratio(a, b):
        xa, xb = np.abs(R[a][m] - R["tp_d0"][m]), np.abs(R[b][m] - R["tp_d0"][m])
        val = np.quantile(xa, 0.6827) / np.quantile(xb, 0.6827)
        reps = []
        for _ in range(400):
            o = np.repeat(np.arange(len(xa)), rng.multinomial(len(u), np.full(len(u), 1 / len(u)))[inv])
            reps.append(np.quantile(xa[o], 0.6827) / np.quantile(xb[o], 0.6827))
        return {"ratio": float(val), "band68": [float(np.quantile(reps, 0.16)), float(np.quantile(reps, 0.84))]}
    P = np.concatenate(ptq)
    name = lambda s_: f"ID{s_ - 10}" if s_ > 10 else f"IL{s_}"
    res = {"scenario": scen, "files": list(ST.SCAN_FILES),
           "clusters_of_charged_pt_ge_2_lost_to_the_cap": {name(k): {"clusters": v[0], "in_sub_areas_with_more_than_3": v[3] / v[0],
                                                                      "lost_random": v[1] / v[0], "lost_pt_ranked": v[2] / v[0]}
                                                           for k, v in sorted(lost.items())},
           "IL1_pt_from_bending_over_true_pt": {"n": int(len(P)), "q16_q50_q84": [float(x) for x in np.quantile(P[:, 1] / P[:, 0], [0.16, 0.5, 0.84])],
                                                "share_with_bending_pt_ge_2": float((P[:, 1] >= 2).mean())},
           "bc_daughters_pt_ge_5": {"tracks": int(m.sum()), "d0_w68_um_implemented_now": w68((R["now"] - R["tp_d0"])[m]) * 1e4,
                                    "network_float_pt_ranked_cap_over_implemented_now": ratio("pt", "now"),
                                    "network_float_random_cap_over_implemented_now (scan)": ratio("rnd", "now"),
                                    "network_float_all_sent_over_implemented_now (scan)": ratio("all", "now"),
                                    "pt_ranked_over_all_sent": ratio("pt", "all")}}
    json.dump(res, open(ST.LOCAL / "checks" / "cap_pt_ranked_quick.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
