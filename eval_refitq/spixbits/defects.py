"""Planted defects (PLAN.md section 8): separate short runs on file 4, AAAA, each with its expected size written
down here BEFORE it is run. Every defect is compared with the same arms run correctly in the same process.

  python defects.py [--nev 100]        -> checks/planted_defects.json
(postscan.py runs the same parts as checkpointed units: two arm groups, the cluster-level part, the evaluation.)

EXPECTED (pre-declared; the statistics of three defects were re-specified after a 10-event smoke test of this script,
before any 100-event run: the stored hit pull is the innovation pull, which includes the prediction's variance, so the
"errors" defect is read at coarse steps and on the track pulls; the distinct-draw share is read on target clusters,
because an event's ~50,000 class-T clusters necessarily repeat network clusters; the x / y swap is planted in the
registry mapping, so that emulation and errors swap together):
  x_y_swapped            the two position outputs exchanged in the registry mapping (the "x" scan drives localX, the
                         "y" scan localY). At step 150 um (4 sub-area bits), prompt d0 68 % half-width / network-float's:
                         correct run: the y scan > 1.3 and the x scan < 1.1; with the defect the two exchange
                         (x scan > 1.3, y scan < 1.1): d0 responds to the x scan.
  errors_not_following   positions quantized at y step 150 um and x step 600 um with the added variance left out of the
                         errors (and of the window): the RMS (within +-5) of the crossing-cluster localX pulls is more
                         than 1.5 x the correct run's, on the barrel and on the discs separately, and the prompt d0 pull
                         RMS (within +-5) more than 1.3 x the correct run's; the correct run's hit pulls are within
                         1 +- 0.15.
  one_draw_per_event     one network residual reused for every class-T cluster of an event: the share of distinct
                         alpha residuals among an event's TARGET clusters falls from >= 0.95 to < 0.01.
  bending_with_code_beta the bending-angle-only arm built with the code's "beta" (the SmartPixels alpha) instead of the
                         code's "alpha". RE-DECLARED 2026-11-11 after the first run and the code review (the first
                         signature, "the IL1 wrong-hit share moves to the no-angle arm's", had no power: the correct
                         and the no-angle arm cannot be told apart on it): read the configuration from the stored
                         outputs. Correct arm: chi2_sum of the code beta is 0 on every track and chi2_sum of the code
                         alpha > 0 on more than 90 % of refit tracks; defect: the reverse. The IL1 wrong-hit shares of
                         the correct arm, the defect and a like-for-like no-angle arm (6/6/0/0) are reported only.
  facing_flipped         the facing bit flipped in the link key when decoding: the share of target clusters on an
                         underflow / overflow code (either angle) per link leaves its correct value (median over links
                         > 2 %; correct: 0.5-1.3 %, RE-DECLARED from 0.3-0.8 %, which was the per-angle value).
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


GROUPS = {"A": ["network_float", "down_x_4b", "down_y_4b", "defect_swap_down_x_4b", "defect_swap_down_y_4b"],
          "B": ["ok_y4_x2", "defect_errors_y4_x2", "alloc_6604", "defect_bending_code_beta", "ref_6600_no_angle"]}


def registry_with_defects():
    reg = json.load(open(RS.HERE / "arms.json"))
    by = {r["name"]: r for r in reg["arms"]}

    def variant(name, base, **change):
        r = json.loads(json.dumps(by[base]))
        r["name"] = name
        r["spec"].update(change)
        reg["arms"].append(r)
    variant("defect_swap_down_x_4b", "down_x_4b", step_x_um=None, step_y_um=150.0)
    variant("defect_swap_down_y_4b", "down_y_4b", step_y_um=None, step_x_um=150.0)
    variant("ok_y4_x2", "network_float", step_y_um=150.0, step_x_um=600.0)
    variant("defect_errors_y4_x2", "network_float", step_y_um=150.0, step_x_um=600.0, errors_not_following=True)
    variant("defect_bending_code_beta", "alloc_6604", code_angles_override=["beta"])
    variant("ref_6600_no_angle", "alloc_6604", send=[])
    return reg


def arms_part(S, group, out_dir, nev=None):
    """One group of defect arms on file 4 -> <out_dir>/defects_<group>_{truth,arms}.npz."""
    truth, D = RS.run_unit(S, registry_with_defects(), F, SCEN, GROUPS[group], nev=nev)
    RS.save_npz(out_dir / f"defects_{group}_truth.npz", truth)
    RS.save_npz(out_dir / f"defects_{group}_arms.npz", {f"{n}__{k}": v for n, d in D.items() for k, v in d.items()})


def clusters_part(S, out_dir, nev=50):
    """The cluster-level defects (one draw per event, facing flipped, residual negated) -> defects_clusters.json."""
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
    for start, A in CM.iterate(path, br, chunk=10, nev=nev):
        ctx = AR.prepare(A, S.geo)
        B = EM.base(ctx, S.geo, A, S.net, S.links, LT)
        B1 = EM.base(ctx, S.geo, A, S.net, S.links, LT, defect="one_draw_per_event")
        B2 = EM.base(ctx, S.geo, A, S.net, flipped, LT)
        B3 = EM.base(ctx, S.geo, A, S.net, S.links, LT, defect="residual_negated")
        T = B["target"]
        ev = ctx["K"]["event"]
        for k, b in (("ok", B), ("bad", B1)):
            rA = (b["net_A"] - b["true_A"])[T]
            acc["distinct_" + k] += [len(np.unique(rA[ev[T] == e_])) / max((ev[T] == e_).sum(), 1) for e_ in np.unique(ev[T])]
            acc["w_" + k].append(rA)
        for k, b in (("ok", B), ("flipped", B2)):
            c = EM.emulate(ctx, b, {"angles": "network", "bits_alpha": 4, "bits_beta": 4}, LT)["K"]
            fl = (c["sb_flagAlpha"] != 0) | (c["sb_flagBeta"] != 0)
            for lk in np.unique(B["link"][B["target"]]):
                m = B["target"] & (B["link"] == lk)
                e_ = flag[k].setdefault(int(lk), [0, 0])
                e_[0] += int(fl[m].sum())
                e_[1] += int(m.sum())
        tgt = B["target"]
        edges, mtab, _ = LT.tables[("net", "B")]                 # the tables are in the link's own frame
        x = B["true_B"][tgt]
        kk = (x[:, None] >= edges[B["pool"][tgt]]).sum(1)
        means["table"].append(mtab[B["pool"][tgt], kk])
        means["ok"].append((B["net_B"] - B["true_B"])[tgt])
        means["negated"].append((B3["net_B"] - B3["true_B"])[tgt])
    dist = {k: float(np.mean(acc["distinct_" + k])) for k in ("ok", "bad")}
    wid = {k: float(np.std(np.concatenate(acc["w_" + k]))) for k in ("ok", "bad")}
    med = {k: float(np.median([v[0] / v[1] for v in flag[k].values() if v[1] >= 200])) for k in flag}
    tab = np.concatenate(means["table"])
    big = np.abs(tab) > np.quantile(np.abs(tab), 0.5)
    cor = {k: float(np.mean(np.concatenate(means[k])[big] * np.sign(tab[big])) / np.mean(np.abs(tab[big])))
           for k in ("ok", "negated")}
    res = {"nev": nev, "one_draw_per_event": {"distinct_share": dist, "alpha_residual_rms_target": wid},
           "facing_flipped": {"median_flagged_share_of_target_per_link": med},
           "residual_negated": {"mean_residual_along_table_mean_over_table_mean": cor}}
    tmp = out_dir / "defects_clusters.json.tmp"
    json.dump(res, open(tmp, "w"), indent=1)
    tmp.rename(out_dir / "defects_clusters.json")


def evaluate(out_dir, checks_dir=None):
    """Every defect against its pre-declared reading, with the margins -> checks/planted_defects.json."""
    D, truth = {}, None
    for g in GROUPS:
        Z = np.load(out_dir / f"defects_{g}_arms.npz")
        T = np.load(out_dir / f"defects_{g}_truth.npz")
        truth = truth or {k: T[k] for k in T.files}
        for k in Z.files:
            n, fld = k.split("__", 1)
            D.setdefault(n, {})[fld] = Z[k]
    ss = truth["slot_surface"]
    prompt = truth["origin"] == 0
    res = {"file": F, "scenario": SCEN, "tracks": int(len(ss)), "code": "size-blind error from the MAD; flag cost "
           "without the quantization variance (branch spixbits, after the code review)", "detected": {}, "margins": {}}
    d0 = lambda n: w68((D[n]["a"][:, 4] - truth["tp_d0"])[prompt]) * 1e4
    r = {n: d0(n) / d0("network_float") for n in ("down_x_4b", "down_y_4b", "defect_swap_down_x_4b", "defect_swap_down_y_4b")}
    res["x_y_swapped"] = {"d0_w68_ratio_to_network_float": r}
    res["detected"]["x_y_swapped"] = bool(r["down_y_4b"] > 1.3 and r["down_x_4b"] < 1.1
                                          and r["defect_swap_down_x_4b"] > 1.3 and r["defect_swap_down_y_4b"] < 1.1)
    res["margins"]["x_y_swapped"] = "thresholds > 1.3 and < 1.1; see the ratios"

    def d0_pull_rms(n):
        x = ((D[n]["a"][:, 4] - truth["tp_d0"]) / np.sqrt(D[n]["cov"][:, 14]))[prompt & D[n]["refit"]]
        x = x[np.abs(x) < 5]
        return float(np.sqrt(np.mean(x ** 2)))
    e = {n: {"barrel_localX": pull_rms(D[n], 0, [1, 2, 3, 4], ss), "discs_localX": pull_rms(D[n], 0, [11, 12, 13, 14], ss),
             "barrel_localY": pull_rms(D[n], 1, [1, 2, 3, 4], ss), "discs_localY": pull_rms(D[n], 1, [11, 12, 13, 14], ss),
             "prompt_d0_pull": d0_pull_rms(n)} for n in ("ok_y4_x2", "defect_errors_y4_x2")}
    ok_, bad_ = e["ok_y4_x2"], e["defect_errors_y4_x2"]
    e["ratio_defect_over_correct"] = {k: bad_[k] / ok_[k] for k in ok_}
    res["errors_not_following"] = e
    res["detected"]["errors_not_following"] = bool(bad_["barrel_localX"] > 1.5 * ok_["barrel_localX"]
                                                   and bad_["discs_localX"] > 1.5 * ok_["discs_localX"]
                                                   and bad_["prompt_d0_pull"] > 1.3 * ok_["prompt_d0_pull"]
                                                   and abs(ok_["barrel_localX"] - 1) < 0.15
                                                   and abs(ok_["discs_localX"] - 1) < 0.15)
    res["margins"]["errors_not_following"] = "declared factors: localX hit pulls x 1.5 (barrel, discs), d0 pull x 1.3"
    w = {n: wrong_share(D[n], 1, ss) for n in ("alloc_6604", "defect_bending_code_beta", "ref_6600_no_angle")}
    cs = {n: {"code_alpha_chi2_positive_share_of_refit": float((D[n]["chi2_sum"][:, 2] > 0)[D[n]["refit"]].mean()),
              "code_beta_chi2_positive_share_of_refit": float((D[n]["chi2_sum"][:, 3] > 0)[D[n]["refit"]].mean())}
          for n in ("alloc_6604", "defect_bending_code_beta")}
    res["bending_with_code_beta"] = {"chi2_sum": cs, "IL1_wrong_hit_share (reported only)": w}
    a_, b_ = cs["alloc_6604"], cs["defect_bending_code_beta"]
    res["detected"]["bending_with_code_beta"] = bool(
        a_["code_beta_chi2_positive_share_of_refit"] == 0 and a_["code_alpha_chi2_positive_share_of_refit"] > 0.9
        and b_["code_alpha_chi2_positive_share_of_refit"] == 0 and b_["code_beta_chi2_positive_share_of_refit"] > 0.9)
    C = json.load(open(out_dir / "defects_clusters.json"))
    res.update({k: C[k] for k in ("one_draw_per_event", "facing_flipped", "residual_negated")})
    dist = C["one_draw_per_event"]["distinct_share"]
    med = C["facing_flipped"]["median_flagged_share_of_target_per_link"]
    cor = C["residual_negated"]["mean_residual_along_table_mean_over_table_mean"]
    res["detected"]["one_draw_per_event"] = bool(dist["ok"] >= 0.95 and dist["bad"] < 0.01)
    res["detected"]["facing_flipped"] = bool(med["flipped"] > 0.02 and 0.005 <= med["ok"] <= 0.013)
    res["detected"]["residual_negated"] = bool(cor["ok"] > 0.5 and cor["negated"] < -0.5)
    json.dump(res, open((checks_dir or ST.LOCAL / "checks") / "planted_defects.json", "w"), indent=1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nev", type=int, default=None)
    a = ap.parse_args()
    S = RS.Study()
    out = ST.LOCAL / "checks" / "postscan"
    out.mkdir(parents=True, exist_ok=True)
    for g in GROUPS:
        arms_part(S, g, out, a.nev)
    clusters_part(S, out, 50 if a.nev is None else min(a.nev, 50))
    print(json.dumps(evaluate(out), indent=1))


if __name__ == "__main__":
    main()
