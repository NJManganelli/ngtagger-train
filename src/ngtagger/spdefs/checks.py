"""Known-answer checks of the shared angle module (plan C.2 P8; C.4 rows "k knob" and "frame
sign"), in the fast gate's item form (ngtagger.spcheck.core.Item), each with the known-bad it
must fail. They run on 5 events of canon_v3 AAAA file 3 (the angle items' known-good after the
canon_v3 switch, C.4) and are not yet in the fast gate's item list (joining it is a one-line
edit of spcheck/gate.py, left to its owner).

    python -m ngtagger.spdefs.checks run        [--json OUT]   every item green on the known-good
    python -m ngtagger.spdefs.checks known-bad  [--json OUT]   every item red on its known-bad
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from ngtagger.spcheck.core import Ctx, Item, jsonable, now_utc
from ngtagger.spdefs import angles as SA

KNOWN_GOOD_INPUT = "/Volumes/WDMac/smartpixels_scratch/canon_v3/spstudy_AAAA_f03_100ev.root"
KNOWN_GOOD_EVENTS = 5
CL = "L1TSmartPixelsCluster"
COLS = ("surface", "detId", "tpEmuClass", "tpIdx", "hasAlpha", "hasBeta", "globalR", "globalZ", "globalPhi",
        "globalClusterPhi", "globalClusterCotTheta", "sigGlobalClusterPhi", "sigGlobalClusterCotTheta",
        "tpGlobalClusterPhi", "tpGlobalClusterCotTheta", "localCotAlpha", "localCotBeta", "tpLocalCotAlpha",
        "tpLocalCotBeta", "tpPt")
K_TESTED = (0.0, 0.05, 0.3, 0.67, 0.999)
FRAME_RATIO_BAND = (0.8, 1.2)   # frame-sign check: median p_loc / map(p_glob) per group (P0.3 amendment 1e)
FRAME_PULL_RANGE = (1.0, 10.0)  # ... over the clusters with 1 < |p_glob| < 10
FRAME_MIN_CLUSTERS = 50         # ... a group needs >= 50 of them to be judged


def _differ(x, y):
    """Element-wise 'not the same value' (NaN equals NaN: a bit-identity test)."""
    x, y = np.asarray(x), np.asarray(y)
    return ~((x == y) | (np.isnan(x) & np.isnan(y)))


def clusters(ctx: Ctx) -> dict:
    key = ("spdefs_clusters", ctx.input, ctx.events)
    if key not in ctx.cache:
        import awkward as ak

        A = ctx.arrays([f"{CL}_{c}" for c in COLS])
        n = ak.to_numpy(ak.num(A[f"{CL}_surface"]))
        D = {c: ak.to_numpy(ak.flatten(A[f"{CL}_{c}"])) for c in COLS}
        D["event"] = np.repeat(np.arange(len(n)), n)
        D["ok"] = SA.usable(D)
        D["T"] = SA.class_T(D["tpEmuClass"]) & D["ok"]
        D["C"] = (D["tpEmuClass"] == 0) & D["ok"]
        ids, sg = SA.orientation_from_clusters(D["detId"], D["localCotBeta"], D["globalClusterCotTheta"], D["ok"])
        D["orient"] = SA.orientation_lookup(D["detId"], ids, sg)
        ctx.cache[key] = D
    return ctx.cache[key]


def _fit_rows(D, key_cot, pulls, sel):
    """An in-sample table of the check's own input: (row id per cluster, centre, width) per angle,
    merged with the table's rule (rows = surface x facing x merged signed bins)."""
    fac = SA.facing(D["surface"], D["orient"])
    kv = SA.incidence_key(D["surface"], key_cot)
    fb = SA.fine_bin(D["surface"], kv)
    row = np.full(len(fb), -1, np.int64)
    est = {a: [] for a in pulls}
    nrow = 0
    for s in SA.SURFACES:
        for f in ((0,) if s in SA.BARREL else (-1, 1)):
            g = sel & (D["surface"] == s) & (fac == f) & (fb >= 0)
            cnt = np.bincount(fb[g], minlength=SA.N_FINE[s])
            bounds = [SA.fine_bin_bounds(s, b) for b in range(SA.N_FINE[s])]
            for grp in SA.merge_bins(bounds, cnt):
                m = g & np.isin(fb, grp)
                e = {a: SA.bin_estimates(pulls[a][m]) for a in pulls}
                if any(v is None for v in e.values()):
                    continue
                mm = (D["surface"] == s) & (fac == f) & np.isin(fb, grp)
                row[mm] = nrow
                for a in pulls:
                    est[a].append((e[a]["centre"], e[a]["width"]))
                nrow += 1
    return row, {a: np.array(v + [(np.nan, np.nan)]) for a, v in est.items()}


# ---------------------------------------------------------------- k knob
class KnobIdentity(Item):
    id = "spdefs.k_identity"
    title = "k = 1 is bit-identical to no knob (angles and sigmas, both modes)"
    c4_row = "k knob: k = 1 bit-identical to no knob"
    known_bad_desc = "the knob formula evaluated at k = 1 (ref + 1 x (angle - ref)) instead of the identity"
    rule_desc = "every angle and sigma at k = 1 equals the emitted value bit for bit, both angles and modes"

    def _stat(self, ctx, bad=False):
        D = clusters(ctx)
        out = {"input": ctx.input, "events": ctx.events, "n_class_T": int(D["T"].sum()), "differ": {}}
        for a in SA.ANGLES:
            e, s = SA.emitted(a, D), SA.sigma_of(a, D)
            for mode in SA.K_MODES:
                if bad:
                    t = SA.truth(a, D)
                    k2 = np.array(e, copy=True)
                    k2[D["T"]] = SA.WRAP[a](t[D["T"]] + 1.0 * SA.residual(a, e, t)[D["T"]])
                    r = {"a": k2, "sigma": SA.sigma_k(s, 1.0, mode)}
                else:
                    r = SA.knob(a, D, D["T"], 1.0, mode)
                out["differ"][f"{a}/{mode}"] = int(np.sum(_differ(r["a"], e)) + np.sum(_differ(r["sigma"], s)))
        return out

    def compute(self, ctx):
        return self._stat(ctx)

    def known_bad(self, ctx):
        return self._stat(ctx, bad=True)

    def judge(self, stat, ref):
        bad = {k: v for k, v in stat["differ"].items() if v}
        return not bad, ("bit-identical" if not bad else f"values differ at k = 1: {bad}")


class KnobContinuityFloor(Item):
    id = "spdefs.k_scaled_floor"
    title = "scaled mode: continuous as k -> 1 from below; finite at k = 0 (floor 0.05)"
    c4_row = "k knob: scaled k -> 1 from below continuous; scaled k = 0 finite"
    known_bad_desc = "the unfloored scaled sigma (k x sigma, the census's knob today): zero at k = 0"
    rule_desc = ("at k = 1 - 1e-9 every angle within 1e-8 rad and every sigma within 1e-8 relative of k = 1; at "
                 "k = 0 every class-T consumer pull finite and every sigma = 0.05 x published")

    def _stat(self, ctx, unfloored=False):
        D = clusters(ctx)
        out = {"input": ctx.input, "events": ctx.events}
        for a in SA.ANGLES:
            e, s, t = SA.emitted(a, D), SA.sigma_of(a, D), SA.truth(a, D)
            r1 = SA.knob(a, D, D["T"], 1 - 1e-9, "scaled")
            out[f"{a}/max_angle_jump"] = float(np.max(np.abs(SA.residual(a, r1["a"], e))[D["ok"]]))
            out[f"{a}/max_sigma_jump"] = float(np.max(np.abs(r1["sigma"] / s - 1)[D["ok"]]))
            r0 = SA.knob(a, D, D["T"], 0.0, "scaled")
            sig0 = s * 0.0 if unfloored else r0["sigma"]
            with np.errstate(divide="ignore", invalid="ignore"):
                p0 = SA.residual(a, r0["a"], t) / sig0
            out[f"{a}/k0_nonfinite_pulls"] = int(np.sum(~np.isfinite(p0[D["T"]])))
            out[f"{a}/k0_sigma_ratio_range"] = [float(np.min((sig0 / s)[D["ok"]])), float(np.max((sig0 / s)[D["ok"]]))]
        return out

    def compute(self, ctx):
        return self._stat(ctx)

    def known_bad(self, ctx):
        return self._stat(ctx, unfloored=True)

    def judge(self, stat, ref):
        why = []
        for a in SA.ANGLES:
            if stat[f"{a}/max_angle_jump"] > 1e-8 or stat[f"{a}/max_sigma_jump"] > 1e-8:
                why.append(f"{a}: discontinuous at k -> 1 ({stat[f'{a}/max_angle_jump']:.2g} rad, "
                           f"{stat[f'{a}/max_sigma_jump']:.2g} rel)")
            if stat[f"{a}/k0_nonfinite_pulls"]:
                why.append(f"{a}: {stat[f'{a}/k0_nonfinite_pulls']} non-finite class-T pulls at k = 0")
            lo, hi = stat[f"{a}/k0_sigma_ratio_range"]
            if abs(lo - SA.K_FLOOR) > 1e-12 or abs(hi - SA.K_FLOOR) > 1e-12:
                why.append(f"{a}: k = 0 sigma ratio {lo:.3g}-{hi:.3g}, not the floor {SA.K_FLOOR}")
        return not why, "continuous at 1, floored at 0" if not why else "; ".join(why)


class KnobClassC(Item):
    id = "spdefs.k_class_c"
    title = "class-C (and unlinked) angles bit-identical under any k"
    c4_row = "k knob: class-C angles bit-identical under any k"
    known_bad_desc = "the knob applied to every TP-linked cluster (class C moved toward its dominant TP's truth)"
    rule_desc = f"at k in {K_TESTED}, every non-class-T angle equals the emitted value bit for bit; class T moved"

    def _stat(self, ctx, bad=False):
        D = clusters(ctx)
        sel = (D["tpIdx"] >= 0) & D["ok"] if bad else D["T"]
        out = {"input": ctx.input, "events": ctx.events, "changed_non_T": {}, "moved_T": {}}
        for a in SA.ANGLES:
            e = SA.emitted(a, D)
            for k in K_TESTED:
                a2, moved = SA.improve(a, e, SA.truth(a, D), sel, k)
                out["changed_non_T"][f"{a}/{k}"] = int(np.sum(_differ(a2[~D["T"]], e[~D["T"]])))
                out["moved_T"][f"{a}/{k}"] = int(moved[D["T"]].sum())
        return out

    def compute(self, ctx):
        return self._stat(ctx)

    def known_bad(self, ctx):
        return self._stat(ctx, bad=True)

    def judge(self, stat, ref):
        bad = {k: v for k, v in stat["changed_non_T"].items() if v}
        idle = {k: v for k, v in stat["moved_T"].items() if v == 0}
        why = ([f"non-class-T angles changed: {bad}"] if bad else []) + ([f"class T not moved: {idle}"] if idle else [])
        return not why, "class C untouched, class T moved" if not why else "; ".join(why)


class KnobCentre(Item):
    id = "spdefs.k_centre"
    title = "subtracted centre = k x centre in both modes: class-T pulls after the table exactly 0 at k = 0"
    c4_row = "k knob: subtracted centre = k x centre in both modes (class-T centre 0 at k = 0)"
    known_bad_desc = "the k = 1 centre subtracted at k = 0"
    rule_desc = ("with an in-sample table (true key) at k = 0, every moved class-T cluster's pull after the table is 0 "
                 "(<= 1e-9: rounding of the wrapped angle) in both modes; at k = 0.5 in scaled mode the class-T pull is exactly 0.5 / 0.5 x (pull - c) / w")

    def _stat(self, ctx, bad=False):
        D = clusters(ctx)
        pulls = {a: SA.residual(a, SA.emitted(a, D), SA.truth(a, D)) / SA.sigma_of(a, D) for a in SA.ANGLES}
        row, est = _fit_rows(D, D["tpGlobalClusterCotTheta"], pulls, D["T"])
        out = {"input": ctx.input, "events": ctx.events, "rows": int(len(est["alpha"]) - 1)}
        for a in SA.ANGLES:
            c, w = est[a][row, 0], est[a][row, 1]
            t, s = SA.truth(a, D), SA.sigma_of(a, D)
            for mode in SA.K_MODES:
                r = SA.knob(a, D, D["T"], 0.0, mode)
                kc = 1.0 if bad else 0.0
                p = (SA.residual(a, r["a"], t) - kc * c * s) / (w * r["sigma"]) if bad else \
                    SA.table_pull(SA.residual(a, r["a"], t), s, r["sigma"], 0.0, c, w)
                m = D["T"] & r["moved"] & (row >= 0)
                out[f"{a}/{mode}/k0_max_abs_pull"] = float(np.max(np.abs(p[m]))) if m.any() else None
                out[f"{a}/{mode}/k0_n"] = int(m.sum())
            r = SA.knob(a, D, D["T"], 0.5, "scaled")
            p = SA.table_pull(SA.residual(a, r["a"], t), s, r["sigma"], 0.5, c, w)
            m = D["T"] & r["moved"] & (row >= 0)
            p1 = (pulls[a] - c) / w
            out[f"{a}/scaled_k0.5_max_dev"] = float(np.max(np.abs(p[m] - p1[m])))
        return out

    def compute(self, ctx):
        return self._stat(ctx)

    def known_bad(self, ctx):
        return self._stat(ctx, bad=True)

    def judge(self, stat, ref):
        why = []
        for a in SA.ANGLES:
            for mode in SA.K_MODES:
                v = stat[f"{a}/{mode}/k0_max_abs_pull"]
                if v is None or v > 1e-9:
                    why.append(f"{a}/{mode}: class-T pull after the table at k = 0 up to {v}")
            if stat[f"{a}/scaled_k0.5_max_dev"] > 1e-6:
                why.append(f"{a}: scaled k = 0.5 pulls differ from k = 1 by {stat[f'{a}/scaled_k0.5_max_dev']:.2g}")
        return not why, "centre scales with k in both modes" if not why else "; ".join(why)


# ---------------------------------------------------------------- frame sign and key conversion
class FrameSign(Item):
    id = "spdefs.frame_sign"
    title = "module-local consumer: the per-cluster frame map of the class-T beta pull, both module orientations"
    c4_row = "frame sign (the 2026-09-06 sense bug class)"
    known_bad_desc = "the global sign used on reversed-orientation modules (the orientation ignored in the map)"
    rule_desc = ("per surface x orientation x table row with >= 200 class-T clusters: the median, over its clusters "
                 f"with {FRAME_PULL_RANGE[0]:g} < |p_glob| < {FRAME_PULL_RANGE[1]:g} (>= {FRAME_MIN_CLUSTERS} of them; "
                 "fewer: reported, not judged), of p_loc / map(p_glob) within "
                 f"[{FRAME_RATIO_BAND[0]:g}, {FRAME_RATIO_BAND[1]:g}], where p_loc is the module-frame pull (residual of "
                 "atan(localCotBeta) / published sigma), p_glob the global one and map = to_local (local_beta_sign); "
                 "both orientations judged. Reported, not judged: the module-frame after-table centre (the table's "
                 "trimmed mean) per group. Redesigned 2026-10-14 (P0.3 amendment 1e, statistics review 02 R4-3): the "
                 "after-table centre depends on the location estimator (the module-frame and global disc-beta pulls "
                 "agree cluster by cluster but not exactly: 90 % spread about 0.4 pull units), so the check tests the "
                 "frame map itself; a sign error gives a ratio of about -1")

    def _stat(self, ctx, bad=False):
        D = clusters(ctx)
        pg = SA.residual("beta", SA.emitted("beta", D), SA.truth("beta", D)) / SA.sigma_of("beta", D)
        row, est = _fit_rows(D, D["tpGlobalClusterCotTheta"], {"beta": pg}, D["T"])
        c, w = est["beta"][row, 0], est["beta"][row, 1]
        o = D["orient"]
        o_map = np.ones_like(o) if bad else o
        r_loc = np.arctan(D["localCotBeta"].astype(np.float64)) - np.arctan(D["tpLocalCotBeta"].astype(np.float64))
        p_loc = r_loc / SA.sigma_of("beta", D)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = p_loc / SA.to_local("beta", pg, o_map, D["surface"])
        after = (p_loc - SA.to_local("beta", c, o_map, D["surface"])) / w
        in_range = (np.abs(pg) > FRAME_PULL_RANGE[0]) & (np.abs(pg) < FRAME_PULL_RANGE[1])
        groups = {}
        for s in SA.SURFACES:
            for so in (-1, 1):
                for rw in np.unique(row[(D["surface"] == s) & (row >= 0)]):
                    m = D["T"] & (D["surface"] == s) & (o == so) & (row == rw)
                    if m.sum() < SA.MIN_FIT:
                        continue
                    r = ratio[m & in_range]
                    r = r[np.isfinite(r)]
                    groups[f"{SA.SURFACE_NAMES[s]}/s{so:+d}/row{int(rw)}"] = {
                        "n": int(m.sum()), "n_ratio": int(len(r)),
                        "ratio_median": float(np.median(r)) if len(r) >= FRAME_MIN_CLUSTERS else None,
                        "after_centre_reported": SA.centre_of(after[m])}
        return {"input": ctx.input, "events": ctx.events, "bad": bad, "groups": groups}

    def compute(self, ctx):
        return self._stat(ctx)

    def known_bad(self, ctx):
        return self._stat(ctx, bad=True)

    def judge(self, stat, ref):
        g = {k: v for k, v in stat["groups"].items() if v["ratio_median"] is not None}
        lo, hi = FRAME_RATIO_BAND
        fails = {k: v for k, v in g.items() if not lo <= v["ratio_median"] <= hi}
        orients = {k.split("/")[1] for k in g}
        why = []
        if orients != {"s-1", "s+1"}:
            why.append(f"not both orientations judged ({sorted(orients)})")
        if fails:
            k = min(fails, key=lambda x: fails[x]["ratio_median"])
            why.append(f"{len(fails)} of {len(g)} groups outside [{lo:g}, {hi:g}]; lowest {k}: median ratio "
                       f"{fails[k]['ratio_median']:+.2f}")
        rr = [v["ratio_median"] for v in g.values()]
        n_skip = len(stat["groups"]) - len(g)
        return not why, (f"{len(g)} groups on both orientations ({n_skip} with < {FRAME_MIN_CLUSTERS} clusters in range "
                         f"reported), median ratio {min(rr):.2f}-{max(rr):.2f}" if not why else "; ".join(why))


class KeyFromLocal(Item):
    id = "spdefs.key_local"
    title = "the incidence key from module-local angles and the orientation sign equals the global key"
    c4_row = "frame sign: module-local conversion of the key"
    known_bad_desc = "the orientation sign ignored (every module taken as s = +1)"
    rule_desc = ("class T with pT >= 1 GeV (the table population), true angles: |key_local - key_global| <= "
                 "1e-3 (1 + |key|) for every cluster and no sign mismatch; below 1 GeV the clusters whose direction "
                 "is grazing in r-phi (radial and normal components of opposite sign) are counted, not judged")

    def _stat(self, ctx, bad=False):
        D = clusters(ctx)
        o = np.ones_like(D["orient"]) if bad else D["orient"]
        kg = SA.incidence_key(D["surface"], D["tpGlobalClusterCotTheta"])
        kl = SA.key_from_local(D["surface"], D["tpLocalCotAlpha"], D["tpLocalCotBeta"], o)
        m = D["T"] & (D["orient"] != 0) & (D["tpPt"] >= SA.PT_MIN)
        lo = D["T"] & (D["orient"] != 0) & (D["tpPt"] < SA.PT_MIN)
        d = np.abs(kl - kg)[m] / (1 + np.abs(kg[m]))
        return {"input": ctx.input, "events": ctx.events, "n": int(m.sum()), "max_rel": float(d.max()),
                "sign_mismatch": int(np.sum(np.sign(kl[m]) != np.sign(kg[m]))),
                "below_pt_min": {"n": int(lo.sum()), "sign_mismatch": int(np.sum(np.sign(kl[lo]) != np.sign(kg[lo])))},
                "unknown_orientation": int(np.sum(D["T"] & (D["orient"] == 0)))}

    def compute(self, ctx):
        return self._stat(ctx)

    def known_bad(self, ctx):
        return self._stat(ctx, bad=True)

    def judge(self, stat, ref):
        ok = stat["max_rel"] <= 1e-3 and stat["sign_mismatch"] == 0
        return ok, (f"{stat['n']} class-T clusters (pT >= {SA.PT_MIN:g}), max rel {stat['max_rel']:.1e}, sign "
                    f"mismatches {stat['sign_mismatch']}; below: {stat['below_pt_min']['sign_mismatch']} of "
                    f"{stat['below_pt_min']['n']}; unknown orientation {stat['unknown_orientation']}")


ITEMS = [KnobIdentity(), KnobContinuityFloor(), KnobClassC(), KnobCentre(), FrameSign(), KeyFromLocal()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="spdefs.checks")
    ap.add_argument("what", choices=("run", "known-bad"))
    ap.add_argument("--input", default=KNOWN_GOOD_INPUT)
    ap.add_argument("--events", type=int, default=KNOWN_GOOD_EVENTS)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    ctx = Ctx(input_path=a.input, events=a.events)
    res = {"mode": a.what, "time": now_utc(), "input": a.input, "events": a.events, "items": {}}
    ok = True
    for it in ITEMS:
        stat = it.compute(ctx) if a.what == "run" else it.known_bad(ctx)
        good, why = it.judge(stat, None)
        st = ("green" if good else "red") if a.what == "run" else ("red-as-required" if not good else "NOT-RED")
        ok &= (good if a.what == "run" else not good)
        res["items"][it.id] = {"status": st, "why": why, "rule": it.rule_desc, "known_bad": it.known_bad_desc,
                               "c4_row": it.c4_row, "stat": jsonable(stat)}
        print(f"{st:16s} {it.id:24s} {why[:200]}", flush=True)
    res["verdict"] = ("all green" if ok else "SOME RED") if a.what == "run" else \
        ("every known-bad red" if ok else "SOME KNOWN-BAD NOT RED")
    print(res["verdict"])
    if a.json:
        with open(a.json, "w") as f:
            json.dump(jsonable(res), f, indent=1, default=str)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
