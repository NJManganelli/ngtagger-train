"""Planted defects (PLAN.md section 8): separate short runs on file 4, AAAA, each with its expected size written
down here BEFORE it is run. Every defect is compared with the same arms run correctly in the same process.

  python defects.py [--nev 100]        -> checks/planted_defects.json

EXPECTED (pre-declared):
  x_y_swapped            the two position steps exchanged in the emulation (the errors keep the nominal assignment).
                         At step 150 um (4 sub-area bits), d0 68 % half-width / network-float's: correct run: the y
                         scan > 1.3 and the x scan < 1.1; with the defect the two exchange (x scan > 1.3, y scan < 1.1).
  errors_not_following   positions quantized at the y step 37.5 um (6 bits) and x step 150 um (4 bits) with the added
                         variance left out of the errors: RMS of the crossing-cluster pulls leaves 1: localX pull RMS
                         > 2 on IL1 (expected about sqrt(1 + 117 / 14) = 3) and > 1.5 on the discs; correct run:
                         within 1 +- 0.15. Barrel and discs read separately.
  one_draw_per_event     one network residual reused for every class-T cluster of an event: the share of distinct
                         (alpha residual) values among an event's class-T clusters falls from >= 0.99 to < 0.01; the
                         residual widths do not change (within 10 %).
  bending_with_code_beta the bending-angle-only arm built with the code's "beta" (the SmartPixels alpha) instead of the
                         code's "alpha": its IL1 wrong-hit share moves from the correct bending-only arm's value to
                         the no-angle arm's (closer to the no-angle arm than to the correct arm).
  facing_flipped         the facing bit flipped in the link key when decoding: the share of target clusters on an
                         underflow / overflow code per link leaves 0.5 % (median over links > 1 %; correct: 0.3-0.8 %).
  residual_negated       the network residual applied with the opposite sign: the mean of (reported - true) projected
                         on the sign of the angle table's mean, in units of the table's |mean| (clusters in the upper
                         half of |table mean|), flips: < -0.5 (correct: > +0.5).
"""
import argparse
import json

import numpy as np

import emulate as EM
import run_scan as RS
import study as ST
from study import AR, CM

F, SCEN = ST.DEV_FILE, "AAAA"


def w68(x):
    x = np.abs(x[np.isfinite(x)])
    return float(np.quantile(x, 0.6827)) if len(x) else float("nan")


def pull_rms(d, i, surfaces, slot_surface):
    m = d["hit_accepted"] & (d["hit_code"] == 1) & np.isin(slot_surface, surfaces)
    x = d["hit_abs_pull"][..., i][m].astype(float)
    x = x[(x > 0) & (x < 5)]
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else float("nan")


def wrong_share(d, surface, slot_surface):
    m = d["hit_accepted"] & (slot_surface == surface)
    return float((d["hit_code"][m] >= 3).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nev", type=int, default=100)
    a = ap.parse_args()
    S = RS.Study()
    reg = json.load(open(RS.HERE / "arms.json"))
    by = {r["name"]: r for r in reg["arms"]}

    def variant(name, base, **change):
        r = json.loads(json.dumps(by[base]))
        r["name"] = name
        r["spec"].update(change)
        reg["arms"].append(r)
        return name
    names = ["network_float", "down_x_4b", "down_y_4b", "alloc_6604", "positions_only_float",
             variant("defect_swap_down_x_4b", "down_x_4b", swap_xy=True),
             variant("defect_swap_down_y_4b", "down_y_4b", swap_xy=True),
             variant("ok_y6_x4", "network_float", step_y_um=37.5, step_x_um=150.0),
             variant("defect_errors_y6_x4", "network_float", step_y_um=37.5, step_x_um=150.0, errors_not_following=True),
             variant("defect_bending_code_beta", "alloc_6604", code_angles_override=["beta"])]
    truth, D = RS.run_unit(S, reg, F, SCEN, names, nev=a.nev)
    ss = truth["slot_surface"]
    prompt = truth["origin"] == 0
    res = {"nev": a.nev, "detected": {}}
    d0 = lambda n: w68((D[n]["a"][:, 4] - truth["tp_d0"])[prompt]) * 1e4
    r = {n: d0(n) / d0("network_float") for n in ("down_x_4b", "down_y_4b", "defect_swap_down_x_4b", "defect_swap_down_y_4b")}
    res["x_y_swapped"] = {"d0_w68_ratio_to_network_float": r}
    res["detected"]["x_y_swapped"] = bool(r["down_y_4b"] > 1.3 and r["down_x_4b"] < 1.1
                                          and r["defect_swap_down_x_4b"] > 1.3 and r["defect_swap_down_y_4b"] < 1.1)
    e = {n: {"IL1_localX": pull_rms(D[n], 0, [1], ss), "barrel_localX": pull_rms(D[n], 0, [1, 2, 3, 4], ss),
             "discs_localX": pull_rms(D[n], 0, [11, 12, 13, 14], ss), "barrel_localY": pull_rms(D[n], 1, [1, 2, 3, 4], ss),
             "discs_localY": pull_rms(D[n], 1, [11, 12, 13, 14], ss)} for n in ("ok_y6_x4", "defect_errors_y6_x4")}
    res["errors_not_following"] = e
    res["detected"]["errors_not_following"] = bool(e["defect_errors_y6_x4"]["IL1_localX"] > 2
                                                   and e["defect_errors_y6_x4"]["discs_localX"] > 1.5
                                                   and abs(e["ok_y6_x4"]["barrel_localX"] - 1) < 0.15
                                                   and abs(e["ok_y6_x4"]["discs_localX"] - 1) < 0.15)
    w = {n: wrong_share(D[n], 1, ss) for n in ("alloc_6604", "defect_bending_code_beta", "positions_only_float")}
    res["bending_with_code_beta"] = {"IL1_wrong_hit_share": w}
    res["detected"]["bending_with_code_beta"] = bool(abs(w["defect_bending_code_beta"] - w["positions_only_float"])
                                                     < abs(w["defect_bending_code_beta"] - w["alloc_6604"]))
    # cluster-level defects
    path = ST.input_path(SCEN, F)
    keys = AR.OP._keys(str(path))
    br = list(dict.fromkeys(AR.branches(path) + ["run", "luminosityBlock", "event"]
                            + AR.SPN.cluster_branches(keys, list(EM.EXTRA_CLUSTER_COLS), disc_scope="all")))
    LT = S.LT
    acc = {k: [] for k in ("distinct_ok", "distinct_bad", "w_ok", "w_bad")}
    flag = {"ok": {}, "flipped": {}}
    means = {"ok": [], "negated": [], "table": []}
    flipped = EM.Links(S.geo, CM.GEOMETRY)
    flipped.flip_facing = True
    for start, A in CM.iterate(path, br, chunk=10, nev=min(a.nev, 50)):
        ctx = AR.prepare(A, S.geo)
        B = EM.base(ctx, S.geo, A, S.net, S.links, LT)
        B1 = EM.base(ctx, S.geo, A, S.net, S.links, LT, defect="one_draw_per_event")
        B2 = EM.base(ctx, S.geo, A, S.net, flipped, LT)
        B3 = EM.base(ctx, S.geo, A, S.net, S.links, LT, defect="residual_negated")
        T = B["class_T"]
        ev = ctx["K"]["event"]
        for k, b in (("ok", B), ("bad", B1)):
            rA = (b["net_A"] - b["true_A"])[T]
            acc["distinct_" + k] += [len(np.unique(rA[ev[T] == e_])) / max((ev[T] == e_).sum(), 1) for e_ in np.unique(ev[T])]
            acc["w_" + k].append(rA[B["target"][T]])
        for k, b in (("ok", B), ("flipped", B2)):
            c = EM.emulate(ctx, b, {"angles": "network", "bits_alpha": 4, "bits_beta": 4}, LT)["K"]
            fl = (c["sb_flagAlpha"] != 0) | (c["sb_flagBeta"] != 0)
            for lk in np.unique(B["link"][B["target"]]):
                m = B["target"] & (B["link"] == lk)
                e_ = flag[k].setdefault(int(lk), [0, 0])
                e_[0] += int(fl[m].sum())
                e_[1] += int(m.sum())
        tgt = B["target"]
        edges, mtab, _ = LT.tables[("net", "B")]
        x = B["sB"][tgt] * B["true_B"][tgt]
        kk = (x[:, None] >= edges[B["pool"][tgt]]).sum(1)
        means["table"].append(mtab[B["pool"][tgt], kk])
        means["ok"].append(B["sB"][tgt] * (B["net_B"] - B["true_B"])[tgt])
        means["negated"].append(B["sB"][tgt] * (B3["net_B"] - B3["true_B"])[tgt])
    dist = {k: float(np.mean(acc["distinct_" + k])) for k in ("ok", "bad")}
    wid = {k: float(np.std(np.concatenate(acc["w_" + k]))) for k in ("ok", "bad")}
    res["one_draw_per_event"] = {"distinct_share": dist, "alpha_residual_rms_target": wid}
    res["detected"]["one_draw_per_event"] = bool(dist["ok"] >= 0.99 and dist["bad"] < 0.01)
    med = {k: float(np.median([v[0] / v[1] for v in flag[k].values() if v[1] >= 200])) for k in flag}
    res["facing_flipped"] = {"median_flagged_share_of_target_per_link": med}
    res["detected"]["facing_flipped"] = bool(med["flipped"] > 0.01 and 0.003 <= med["ok"] <= 0.008)
    tab = np.concatenate(means["table"])
    big = np.abs(tab) > np.quantile(np.abs(tab), 0.5)
    cor = {k: float(np.mean(np.concatenate(means[k])[big] * np.sign(tab[big])) / np.mean(np.abs(tab[big])))
           for k in ("ok", "negated")}
    res["residual_negated"] = {"mean_residual_along_table_mean_over_table_mean": cor,
                               "note": "the mean residual projected on the sign of the table's mean, in units of the "
                                       "table's mean |mean| (clusters in the upper half of |table mean|): +1 correct, -1 negated"}
    res["detected"]["residual_negated"] = bool(cor["ok"] > 0.5 and cor["negated"] < -0.5)
    json.dump(res, open(ST.LOCAL / "checks" / "planted_defects.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
