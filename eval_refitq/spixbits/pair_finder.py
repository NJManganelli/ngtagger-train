"""Displaced track-pair finder, first step (design: doc-archive/l1-flow-bphys-study/DESIGN_displaced_pair_finder.md):
opposite-charge pairs of tracks, a common vertex from the OUTER-TRACKER helices, masses under three hypotheses, then the
per-track displacement heads (second version) as pair inputs. Cuts only: no network for the pairs.

  [TAGGER_SP_COLLECTION=prompt TAGGER_SP_OUT=/Volumes/WDMac/smartpixels_scratch/tagger_v5_prompt/heads_input] \\
      python pair_finder.py heads | pairs | analyze   [--masks AAAA AAAI IAAA]
  (without the two variables: the extended collection)

Events: the displacement heads' sample (canon_v5 TTbar files 1-3, 5, 7-10 and seven H->tautau files, 1,150 events,
"implemented now"; TTbar files 4 and 6 are not read). 200 pileup.

heads    OUT-OF-FOLD displacement-head outputs for every track: five folds by event identity; each fold's tracks are
         scored by a model trained on the other four (the stage-A settings, trial 0, no trial choice), so no track is
         scored by a model that saw its event. (The stage-A models themselves were trained on 80 % of these events.)
pairs    per event, every pair of outer-tracker tracks (the five-parameter fit of the collection) with opposite charge
         and Delta R(eta, phi0) < 1. The two circles in the transverse plane: where they cross, the crossing with the
         smaller |z1 - z2| among those ahead of both tracks' points of closest approach (s_i >= 0) is the vertex (z_i = z0_i + s_i tanL_i at the path length s_i from the track's point of closest
         approach to the beam line); where they do not, the midpoint of their closest approach, with the transverse
         gap kept. Per pair: vertex radius R and z, |dz|, gap, cos of the transverse angle between the vertex position
         and the pair momentum (pointing), pair pT, masses under pi pi, p pi (the proton is the track of higher
         momentum) and e e, with the momenta taken at the vertex. Kept: R < 60 cm, |dz| < 10 cm, gap < 5 cm.
         Truth: a pair is a TRUE K0S when its two tracks' particles are the pi+ and pi- of one K0S (displacement.k0s_pions:
         identical production vertex); OTHER TRUE VERTEX when the two particles share an identical production vertex
         beyond 0.5 cm otherwise (Lambda, conversions, interactions in material); RANDOM otherwise.
analyze  PRESELECTION (geometry only): windows on |dz|, gap, pointing and |m(pi pi) - m(K0S)|, each at the 98 % point
         of the true K0S pairs of the non-test events of the heads' split, then frozen: nothing is tuned on the test
         events and nothing else is tuned at all. Then one-variable scans, each a curve of the efficiency for true K0S
         (both pions tracked) against random pairs per event:
           without head quantities   min over the two tracks of |d0| / sigma(d0);  the vertex radius R
           with head quantities      min over the two tracks of P(class >= 1)  ("both displaced");
                                     min over the two tracks of P(the class that holds R on that track's trajectory)
                                     ("each track's range agrees with the vertex");
                                     the first, requiring also that the two tracks' low-high ranges overlap and R lies
                                     inside both (ranges widened by 20 %)
         split by the true decay radius. Reported on all events and on the test events of the heads' split alone.
Outputs under ~/smartpixels/scratch_local/spixbits/pair_finder/<collection>/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import displacement as D             # noqa: E402
import displacement_heads as DH      # noqa: E402
from displacement_heads import C     # noqa: E402  (tagger_sp/common.py)

from ngtagger.definitions.helix import KPT_CMSSW, wrap  # noqa: E402

OUT = Path.home() / "smartpixels/scratch_local/spixbits/pair_finder" / C.COLLECTION
N_FOLDS = 5
DR_MAX, R_MAX, DZ_KEEP, GAP_KEEP = 1.0, 60.0, 10.0, 5.0
M_PI, M_P, M_E, M_K0S = 0.13957, 0.93827, 0.000511, 0.497611
CONTAIN = 0.98
RANGE_TOL = 0.2
R_BINS = [0.0, 2.85, 5.95, 10.3, 14.5, 60.0]           # cm, true decay radius
RATES = (10.0, 1.0, 0.1)                                # random pairs per event at which the efficiency is quoted
OTHER_MIN_R = 0.5


def event_key(evid):
    return (evid[:, 0].astype(np.int64) * 1000003 + evid[:, 1].astype(np.int64)) * 1000003 + evid[:, 2].astype(np.int64)


def tracks(mask):
    """The heads' tracks (displacement_heads.load) with, per track, its event (a running number and the identity key), the
    position of its event in the file, and its particle's production vertex. Files 4 and 6 are dropped before use."""
    import awkward as ak
    import build as B
    import uproot
    T = DH.load(mask)
    man = json.load(open(C.manifest_path(mask)))
    ev, key, vtx, n_ev = [], [], [], 0
    for s, n, p in man["files"]:
        _, ntrk, evid = B.load_tracks(os.path.join(C.cache_dir("tracks"), C.tag(s, mask, n) + ".npz"))
        ev.append(np.repeat(n_ev + np.arange(len(ntrk)), ntrk))
        key.append(np.repeat(event_key(np.asarray(evid)), ntrk))
        n_ev += len(ntrk)
        cache = OUT / "cache" / (C.tag(s, mask, n) + ".npz")
        if not cache.exists():
            if (s, n) in DH.RH.LEFT_OUT:
                v = np.full((int(ntrk.sum()), 3), np.nan, np.float32)        # never read
            else:
                A = uproot.open(p)["Events"].arrays([f"{C.OT_TABLE}_tp_{c}" for c in ("vx", "vy", "vz")])
                v = np.column_stack([ak.to_numpy(ak.flatten(A[f"{C.OT_TABLE}_tp_{c}"])) for c in ("vx", "vy", "vz")]).astype(np.float32)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(str(cache) + ".tmp.npz", vtx=v)
            os.replace(str(cache) + ".tmp.npz", cache)
        vtx.append(np.load(cache)["vtx"])
    T["p_event"], T["p_key"], T["p_vtx"] = np.concatenate(ev), np.concatenate(key), np.concatenate(vtx)
    assert len(T["p_event"]) == len(T["b_pt"]) == len(T["p_vtx"])
    return T


def heads(a):
    """Out-of-fold class probabilities and radius ranges for every track of the files in use."""
    import xgboost as xgb
    from ngtagger.train import trackmva as tm
    for mask in a.masks:
        T = tracks(mask)
        use = ~T["_left_out"]
        X, y, ok = DH.inputs(T, 2), T["x_class"], T["x_class"] >= 0
        fold = (T["p_key"] % 1000003) % N_FOLDS
        P = np.full((len(X), D.n_edges(mask) + 1), np.nan)
        Q = np.full((len(X), len(D.QUANTILES)), np.nan)
        par = dict(DH.XGB, seed=0, objective="multi:softprob", num_class=P.shape[1])
        for wname, w in tm.windows(T, "refit").items():
            for f in range(N_FOLDS):
                tr = np.flatnonzero(use & w & ok & (fold != f))
                te = np.flatnonzero(use & w & (fold == f))
                b = xgb.train(par, xgb.DMatrix(X[tr], label=y[tr]), num_boost_round=DH.ROUNDS)
                P[te] = b.predict(xgb.DMatrix(X[te]))
                Q[te] = D.radius_range(P[te], T["x_radii"][te], mask, D.fit_shape(T["x_r_prod"][tr], y[tr], T["x_radii"][tr], mask))
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OUT / f"heads_{mask}.npz", P=P.astype(np.float32), Q=Q.astype(np.float32), fold=fold.astype(np.int8))
        m = use & ok
        print(C.COLLECTION, mask, "out-of-fold log loss", float(-np.mean(np.log(np.clip(P[m, y[m]], 1e-12, 1)))), flush=True)


def circles(q, pt, phi0, d0):
    kappa = -q * KPT_CMSSW / pt                           # dphi/ds
    x0, y0 = d0 * np.sin(phi0), -d0 * np.cos(phi0)
    return kappa, x0, y0, x0 - np.sin(phi0) / kappa, y0 + np.cos(phi0) / kappa


def path_to(xv, yv, kappa, x0, y0, xc, yc):
    """Signed transverse path length from the track's point of closest approach to the point (xv, yv) on its circle."""
    ang = np.arctan2((x0 - xc) * (yv - yc) - (y0 - yc) * (xv - xc), (x0 - xc) * (xv - xc) + (y0 - yc) * (yv - yc))
    return ang / kappa


def vertex(t1, t2):
    """The common vertex of track pairs: dicts of equal-length arrays q, pt, phi0, d0, z0, tanl. Returns a dict."""
    k1, x01, y01, xc1, yc1 = circles(t1["q"], t1["pt"], t1["phi0"], t1["d0"])
    k2, x02, y02, xc2, yc2 = circles(t2["q"], t2["pt"], t2["phi0"], t2["d0"])
    r1, r2 = 1 / np.abs(k1), 1 / np.abs(k2)
    dx, dy = xc2 - xc1, yc2 - yc1
    d = np.hypot(dx, dy)
    ux, uy = dx / d, dy / d
    cross = (d <= r1 + r2) & (d >= np.abs(r1 - r2))
    a = (d * d + r1 * r1 - r2 * r2) / (2 * d)
    h = np.sqrt(np.clip(r1 * r1 - a * a, 0, None))
    best = None
    for sgn in (1.0, -1.0):
        xv, yv = xc1 + a * ux - sgn * h * uy, yc1 + a * uy + sgn * h * ux
        s1, s2 = path_to(xv, yv, k1, x01, y01, xc1, yc1), path_to(xv, yv, k2, x02, y02, xc2, yc2)
        dz = (t1["z0"] + s1 * t1["tanl"]) - (t2["z0"] + s2 * t2["tanl"])
        cost = np.abs(dz) + 1e3 * ((s1 < 0) | (s2 < 0))        # a crossing behind either track's point of closest approach loses
        cand = (xv, yv, s1, s2, cost)
        best = cand if best is None else tuple(np.where(cost < best[4], c, b) for c, b in zip(cand, best))
    # no crossing: the midpoint of the closest approach on the line of centres (outside each other, or one inside the other)
    outside = d > r1 + r2
    inner_dir = np.where(r1 >= r2, 1.0, -1.0)             # one inside the other: the near points lie beyond the smaller circle's centre
    p1x, p1y = xc1 + np.where(outside, 1.0, inner_dir) * r1 * ux, yc1 + np.where(outside, 1.0, inner_dir) * r1 * uy
    p2x, p2y = xc2 + np.where(outside, -1.0, inner_dir) * r2 * ux, yc2 + np.where(outside, -1.0, inner_dir) * r2 * uy
    mx, my = (p1x + p2x) / 2, (p1y + p2y) / 2
    gap = np.where(cross, 0.0, np.hypot(p1x - p2x, p1y - p2y))
    s1n, s2n = path_to(p1x, p1y, k1, x01, y01, xc1, yc1), path_to(p2x, p2y, k2, x02, y02, xc2, yc2)
    xv, yv = np.where(cross, best[0], mx), np.where(cross, best[1], my)
    s1, s2 = np.where(cross, best[2], s1n), np.where(cross, best[3], s2n)
    z1, z2 = t1["z0"] + s1 * t1["tanl"], t2["z0"] + s2 * t2["tanl"]
    ph1, ph2 = t1["phi0"] + k1 * s1, t2["phi0"] + k2 * s2
    px, py = t1["pt"] * np.cos(ph1) + t2["pt"] * np.cos(ph2), t1["pt"] * np.sin(ph1) + t2["pt"] * np.sin(ph2)
    pz1, pz2 = t1["pt"] * t1["tanl"], t2["pt"] * t2["tanl"]
    p1, p2 = np.hypot(t1["pt"], pz1), np.hypot(t2["pt"], pz2)
    p_sq = px * px + py * py + (pz1 + pz2) ** 2

    def mass(m1, m2):
        e = np.sqrt(p1 * p1 + m1 * m1) + np.sqrt(p2 * p2 + m2 * m2)
        return np.sqrt(np.clip(e * e - p_sq, 0, None))

    R = np.hypot(xv, yv)
    return {"R": R, "z": (z1 + z2) / 2, "dz": np.abs(z1 - z2), "gap": gap, "s1": s1, "s2": s2,
            "cos_point": (xv * px + yv * py) / np.maximum(R * np.hypot(px, py), 1e-12), "pair_pt": np.hypot(px, py),
            "m_pipi": mass(M_PI, M_PI), "m_ppi": np.where(p1 >= p2, mass(M_P, M_PI), mass(M_PI, M_P)), "m_ee": mass(M_E, M_E)}


def pairs(a):
    """The pair candidates of every event (the outer-tracker tracks are the same in every detector scenario: the
    geometry is done once, from the first mask's track list, and checked against the others')."""
    T = tracks(a.masks[0])
    use = ~T["_left_out"]
    sign = np.sign(np.nansum(np.sign(T["b_rInv"][use]) * T["b_tp_charge"][use]))          # the sign of 1/R against the charge
    q = sign * np.sign(T["b_rInv"])
    agree = np.mean((q == T["b_tp_charge"])[use & (T["b_tpIdx"] >= 0) & (T["b_tp_charge"] != 0)])
    v = T["p_vtx"].astype(np.float64)
    has = T["b_tpIdx"] >= 0
    disp = has & (np.hypot(v[:, 0], v[:, 1]) > OTHER_MIN_R)
    out, counts = [], {"events": 0, "opposite_charge_pairs": 0, "within_dR": 0, "kept": 0}
    order = np.flatnonzero(use)
    starts = np.flatnonzero(np.r_[True, np.diff(T["p_event"][order]) != 0, True])
    for lo, hi in zip(starts[:-1], starts[1:]):
        idx = order[lo:hi]
        counts["events"] += 1
        i, j = np.triu_indices(len(idx), 1)
        m = q[idx][i] * q[idx][j] < 0
        counts["opposite_charge_pairs"] += int(m.sum())
        i, j = idx[i[m]], idx[j[m]]
        m = np.hypot(T["b_eta"][i] - T["b_eta"][j], wrap(T["b_phi"][i] - T["b_phi"][j])) < DR_MAX
        i, j = i[m], j[m]
        counts["within_dR"] += len(i)
        tr = lambda k: {"q": q[k], "pt": T["b_pt"][k].astype(np.float64), "phi0": T["b_phi"][k].astype(np.float64),
                        "d0": T["b_d0"][k].astype(np.float64), "z0": T["b_z0"][k].astype(np.float64), "tanl": T["b_tanL"][k].astype(np.float64)}
        with np.errstate(all="ignore"):
            V = vertex(tr(i), tr(j))
        keep = (V["R"] < R_MAX) & (V["dz"] < DZ_KEEP) & (V["gap"] < GAP_KEEP)
        same = has[i] & has[j] & (T["b_tpIdx"][i] != T["b_tpIdx"][j]) & (v[i] == v[j]).all(1)
        kind = np.where(same & T["x_k0s"][i] & T["x_k0s"][j], 1, np.where(same & disp[i], 2, 0)).astype(np.int8)
        out.append({"i": i[keep], "j": j[keep], "kind": kind[keep], **{k: x[keep].astype(np.float32) for k, x in V.items()}})
        counts["kept"] += int(keep.sum())
    P = {k: np.concatenate([o[k] for o in out]) for k in out[0]}
    # the true K0S with both pions tracked (the denominator): one entry per K0S, whatever the number of its tracks
    k = np.flatnonzero(use & T["x_k0s"])
    kid = np.unique(np.column_stack([T["p_event"][k].astype(np.float64), v[k]]), axis=0, return_inverse=True)[1].ravel()
    n_tp = np.array([len(np.unique(T["b_tpIdx"][k[kid == u]])) for u in range(kid.max() + 1)]) if len(k) else np.zeros(0, int)
    both = np.flatnonzero(n_tp == 2)
    first = np.array([k[kid == u][0] for u in both], np.int64)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "pairs.npz", **P, k0s_event=T["p_event"][first], k0s_vtx=v[first], k0s_test=T["_test"][first])
    counts.update(charge_sign_agrees_with_particle=float(agree), k0s_with_a_tracked_pion=int(len(n_tp)), k0s_with_both_pions_tracked=int(len(both)),
                  true_k0s_pairs_kept=int((P["kind"] == 1).sum()), other_true_vertex_pairs_kept=int((P["kind"] == 2).sum()))
    json.dump(counts, open(OUT / "pairs_counts.json", "w"), indent=1)
    print(C.COLLECTION, json.dumps(counts), flush=True)


def analyze(a):
    Pr = dict(np.load(OUT / "pairs.npz"))
    counts = json.load(open(OUT / "pairs_counts.json"))
    res = {"collection": C.COLLECTION, "pairs": counts, "masks": {}}
    T0 = tracks(a.masks[0])
    i, j, kind = Pr["i"], Pr["j"], Pr["kind"]
    test_pair = T0["_test"][i]
    ev_pair = T0["p_event"][i]
    n_events = {"all": counts["events"], "test": int(len(np.unique(T0["p_event"][~T0["_left_out"] & T0["_test"]])))}
    true_r = np.hypot(T0["p_vtx"][i, 0], T0["p_vtx"][i, 1]).astype(np.float64)       # of the first track's particle (the pair's, for a true pair)
    # the preselection: windows at the 98 % point of the true K0S pairs of the non-test events
    fit = (kind == 1) & ~test_pair
    win = {"dz": float(np.quantile(Pr["dz"][fit], CONTAIN)), "gap": float(np.quantile(Pr["gap"][fit], CONTAIN)),
           "cos_point": float(np.quantile(Pr["cos_point"][fit], 1 - CONTAIN)), "dm_pipi": float(np.quantile(np.abs(Pr["m_pipi"][fit] - M_K0S), CONTAIN))}
    pre = (Pr["dz"] < win["dz"]) & (Pr["gap"] <= win["gap"]) & (Pr["cos_point"] > win["cos_point"]) & (np.abs(Pr["m_pipi"] - M_K0S) < win["dm_pipi"])
    k0s_r = np.hypot(Pr["k0s_vtx"][:, 0], Pr["k0s_vtx"][:, 1])
    res["preselection"] = {"windows": win, "n_true_k0s_pairs_fitted_on": int(fit.sum())}

    def found(sel):
        """The true K0S (one per decay) with a pair in `sel`: boolean over the denominator list."""
        s = sel & (kind == 1)
        got = set(zip(ev_pair[s].tolist(), T0["p_vtx"][i[s], 0].tolist()))
        return np.array([(int(e), float(np.float32(x))) in got for e, x in zip(Pr["k0s_event"], Pr["k0s_vtx"][:, 0])], bool)

    def stage(sel, which):
        m = np.ones(len(sel), bool) if which == "all" else test_pair
        d = np.ones(len(k0s_r), bool) if which == "all" else Pr["k0s_test"]
        f = found(sel & m)
        return {"k0s_both_tracked": int(d.sum()), "k0s_found": int(f[d].sum()), "efficiency": float(f[d].mean()) if d.any() else None,
                "random_pairs_per_event": float((sel & m & (kind == 0)).sum() / n_events[which]),
                "other_true_vertex_pairs_per_event": float((sel & m & (kind == 2)).sum() / n_events[which]),
                "efficiency_by_true_radius": [None if not (d & (k0s_r >= lo) & (k0s_r < hi)).any() else
                                              [int(f[d & (k0s_r >= lo) & (k0s_r < hi)].sum()), int((d & (k0s_r >= lo) & (k0s_r < hi)).sum())]
                                              for lo, hi in zip(R_BINS[:-1], R_BINS[1:])]}
    res["stages"] = {w: {"pair kept (R < 60 cm, |dz| < 10 cm, gap < 5 cm)": stage(np.ones(len(i), bool), w), "preselection": stage(pre, w)} for w in ("all", "test")}

    def curve(score, sel, which):
        """Efficiency for true K0S at the given random-pair rates, scanning `score` (larger = more signal-like) inside sel."""
        m = sel & (np.ones(len(sel), bool) if which == "all" else test_pair) & np.isfinite(score)
        d = np.ones(len(k0s_r), bool) if which == "all" else Pr["k0s_test"]
        bkg = np.sort(score[m & (kind == 0)])[::-1]
        out = {}
        for rate in RATES:
            n = int(round(rate * n_events[which]))
            thr = bkg[n] if n < len(bkg) else -np.inf
            f = found(m & (score > thr))
            e = {"threshold": float(thr), "efficiency": float(f[d].mean()), "n_found": int(f[d].sum()), "n": int(d.sum()),
                 "by_true_radius": [[int(f[d & (k0s_r >= lo) & (k0s_r < hi)].sum()), int((d & (k0s_r >= lo) & (k0s_r < hi)).sum())]
                                    for lo, hi in zip(R_BINS[:-1], R_BINS[1:])]}
            out[f"{rate:g} random pairs per event"] = e
        return out

    sig = lambda k: np.abs(T0["b_d0"][k]) / np.maximum(np.sqrt(np.abs(T0["b_cov_d0_d0"][k])), 1e-6)
    base = {"min |d0|/sigma of the two tracks": np.minimum(sig(i), sig(j)), "vertex radius R": Pr["R"].astype(np.float64)}
    res["without_head_quantities"] = {w: {n: curve(s, pre, w) for n, s in base.items()} for w in ("all", "test")}
    for mask in a.masks:
        T = T0 if mask == a.masks[0] else tracks(mask)
        assert np.array_equal(T["b_pt"], T0["b_pt"]), "the outer-tracker tracks differ between detector scenarios"
        H = np.load(OUT / f"heads_{mask}.npz")
        Ph, Q, radii = H["P"].astype(np.float64), H["Q"].astype(np.float64), T["x_radii"].astype(np.float64)
        N = D.n_edges(mask)
        both_disp = np.minimum(Ph[i, 1:].sum(1), Ph[j, 1:].sum(1))
        cls_of_R = lambda k: np.minimum((radii[k] < Pr["R"][:, None]).sum(1), N)
        agree = np.minimum(Ph[i, cls_of_R(i)], Ph[j, cls_of_R(j)])
        lo, hi = lambda k: Q[k, 0] * (1 - RANGE_TOL), lambda k: Q[k, 2] * (1 + RANGE_TOL)
        inside = (Pr["R"] >= lo(i)) & (Pr["R"] <= hi(i)) & (Pr["R"] >= lo(j)) & (Pr["R"] <= hi(j)) & (lo(i) <= hi(j)) & (lo(j) <= hi(i))
        sc = {"min P(class >= 1) of the two tracks": (both_disp, pre),
              "min P(class holding R) of the two tracks": (agree, pre),
              "min P(class >= 1), ranges overlap and hold R": (both_disp, pre & inside)}
        res["masks"][mask] = {w: {n: curve(s, sel, w) for n, (s, sel) in sc.items()} for w in ("all", "test")}
        res["masks"][mask]["ranges overlap and hold R, alone"] = {w: stage(pre & inside, w) for w in ("all", "test")}
        print(C.COLLECTION, mask, "done", flush=True)
    json.dump(res, open(OUT / "report.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("heads", "pairs", "analyze"))
    ap.add_argument("--masks", nargs="+", default=DH.MASKS)
    a = ap.parse_args()
    {"heads": heads, "pairs": pairs, "analyze": analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
