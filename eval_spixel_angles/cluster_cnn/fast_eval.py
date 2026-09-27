"""FAST class-C study: evaluate class-C pseudo-truth variants against the knn donor (approved v4 angle-space emulator), test events.

usage: fast_eval.py <region> <variant> [...]      variant = <cnn>:<draw>
   cnn  : ref (cluster_cnn_v2 pos-hubsig, the current production CNN) | a FastNet tag (emu_fast/preds/<tag>.parquet)
   draw : knn (10-NN donor in the 11 standardised observables, per surface, other event half = the prototype)
          nodonor (reference only: pseudo = CNN) | a1arh | a1mdn (fast_a1.py) | a2 (the FastNet's own ARH head: single network)
Per variant (class-C route applied to every cluster; class T keeps the truth route):
  closure  : C-route output - truth on class-T clusters, per group IL2-4 / IL1 / ID1-4: core RMS about 0 (mode-seeded Gaussian
             core, v2_common.core_fit), full RMS, fraction > 10 deg      (target reference = the class-T route itself)
  PIT      : calibration of the drawn residual for class-T clusters: alpha marginal, beta conditional on the true alpha residual
             (a1/a2 exact from the model; knn = randomised rank among the 10 donor residuals, beta MARGINAL); ECE = mean |F(q) - q|
             over q = 0.05..0.95, tail = P(PIT < 0.025 or > 0.975) (0.05 expected)
  corr     : Pearson / Spearman of the drawn (rA, rB) vs the true residual pairs on class T; quadrant L1 (|r| > 5 deg)
  leak     : AUC(unlinked vs linked) gain of the production output over the strong probe (v2_emu_angle: shape + position + ref CNN
             mean/sigma + CPE); for a non-ref CNN also the probe + that CNN's own mean/sigma ("own")
  emitted  : class-C emitted angle distributions (unlinked; linked with bad truth) vs class T and vs the knn(ref) class C: rms,
             frac |angle| > 60, KS distance to knn(ref)
Outputs: results/fast_eval_<region>.csv (merged by variant), FAST/hists/<region>__<variant>.npz (histograms for fast_plots.py).
"""
import os, sys, json, numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from fast_common import FAST, V2, RES, load_test, classes, groups, respond, closure, auc, sat
from fast_a1 import cnn_source

HB = {"A": np.linspace(-70, 70, 141), "B": np.linspace(-90, 90, 181)}
HR = np.linspace(-60, 60, 241)
KNN_K = 10


def knn_draw_rank(Z, surf, R, clsT, half, valid, rng):
    """the prototype's knn draw + the randomised-rank PIT of the true residual among the 10 donors"""
    n = len(Z); draw = np.full((n, 2), np.nan); pit = np.full((n, 2), np.nan)
    for s in np.unique(surf):
        for hf in (True, False):
            don = np.where(clsT & (half != hf) & (surf == s) & np.isfinite(R).all(1))[0]
            tg = np.where((half == hf) & (surf == s))[0]
            mu, sd = Z[don].mean(0), Z[don].std(0); sd[sd == 0] = 1
            _, idx = cKDTree((Z[don] - mu) / sd).query((Z[tg] - mu) / sd, k=KNN_K, workers=4)
            nb = R[don[idx]]                                            # [len(tg), 10, 2]
            draw[tg] = nb[np.arange(len(tg)), rng.integers(0, KNN_K, len(tg))]
            for j in range(2):
                t = R[tg, j][:, None]
                lt, eq = (nb[..., j] < t).sum(1), (nb[..., j] == t).sum(1)
                pit[tg, j] = np.where(valid[tg], (lt + rng.random(len(tg)) * (eq + 1)) / (KNN_K + 1), np.nan)
    return draw, pit


def pit_stats(p):
    p = p[np.isfinite(p)]
    if len(p) < 100:
        return dict(ece=np.nan, tail=np.nan)
    q = np.linspace(0.05, 0.95, 19)
    return dict(ece=float(np.mean(np.abs(np.array([np.mean(p < x) for x in q]) - q))), tail=float(np.mean((p < 0.025) | (p > 0.975))))


def quad(ra, rb, thr=5.0):
    b = (np.abs(ra) > thr) & (np.abs(rb) > thr)
    return np.array([np.mean(b & (ra > 0) & (rb > 0)), np.mean(b & (ra > 0) & (rb < 0)), np.mean(b & (ra < 0) & (rb > 0)),
                     np.mean(b & (ra < 0) & (rb < 0))])


def ks(x, y):
    x, y = np.sort(x[np.isfinite(x)]), np.sort(y[np.isfinite(y)])
    g = np.r_[x, y]; return float(np.max(np.abs(np.searchsorted(x, g, "right") / len(x) - np.searchsorted(y, g, "right") / len(y))))


def save(rows, sig_rows, csv, reg):
    """merge by variant into the CSVs (called after every variant, so a crash keeps the finished ones)"""
    new = pd.DataFrame(rows)
    old = pd.read_csv(csv) if os.path.exists(csv) else pd.DataFrame()
    if len(old):
        old = old[~old.variant.isin(new.variant.unique())]
    pd.concat([old, new], ignore_index=True).to_csv(csv, index=False)
    sp = os.path.join(RES, f"fast_sigma_{reg}.csv"); S = pd.DataFrame(sig_rows)
    if len(S):
        if os.path.exists(sp):
            o = pd.read_csv(sp); S = pd.concat([o[~o.variant.isin(S.variant.unique())], S], ignore_index=True)
        S.to_csv(sp, index=False)
    return new


if __name__ == "__main__":
    reg, variants = sys.argv[1], sys.argv[2:]
    os.makedirs(os.path.join(FAST, "hists"), exist_ok=True)
    m = load_test(reg); n = len(m); valid, clsT = classes(m); surf, grp = groups(m, reg)
    disc = np.full(n, reg == "disc"); ev = np.sort(m.event.unique()); half = m.event.isin(ev[::2]).values
    tA = np.where(valid, np.degrees(np.arctan(m.tpLocalCotAlpha.values)), 0.0); tB = np.where(valid, np.degrees(np.arctan(m.tpLocalCotBeta.values)), 0.0)
    y = (m.linkClass.values == 1).astype(int); CU = (~clsT) & (y == 1); CL = (~clsT) & (m.linkClass.values == 2)
    oTA, oTB, _, _ = respond(tA, tB, disc, seed=77)
    rA_ref = cnn_source("ref", reg, m)
    cpeX = m.tx_cpe.values * m.pitchX.values * 1e4; cpeY = m.ty_cpe.values * m.pitchY.values * 1e4
    F1 = np.c_[m.sizeX, m.sizeY, m["size"], m.charge, surf, m.globalR, np.abs(m.globalZ), m.localX, m.localY, m.zOutward]
    probe = np.c_[F1, rA_ref[0], rA_ref[1], rA_ref[2], rA_ref[3], cpeX, cpeY]
    csv = os.path.join(RES, f"fast_eval_{reg}.csv")
    base_path = os.path.join(RES, f"fast_eval_{reg}_leakbase.json")
    if os.path.exists(base_path):
        base = json.load(open(base_path))
    else:   # the probe bases (strong = shape + position + ref CNN mean/sigma + CPE; F1 = shape + position only) + the noise
        base = {"strong": auc(probe, y, half, grp=grp), "F1": auc(F1, y, half, grp=grp)}
        sw = auc(probe, y, ~half, grp=grp)
        base["noise_swap_strong"] = {k: sw[k] - base["strong"][k] for k in sw}
        json.dump(base, open(base_path, "w"), indent=1)
    print("leak base", base, flush=True)
    # class-T route reference rows + hists
    rows = []; sig_rows = []
    Hk = {}
    for g in np.unique(grp):
        s = clsT & (grp == g)
        for ang, o_, t_ in (("alpha", oTA, tA), ("beta", oTB, tB)):
            rows.append(dict(variant="classT_route", group=g, angle=ang, **closure(o_[s], t_[s])))
            Hk[f"res_T_{g}_{ang}"] = np.histogram(np.clip(o_[s] - t_[s], HR[0] + 1e-6, HR[-1] - 1e-6), HR)[0]
            e = HB["A" if ang == "alpha" else "B"]
            Hk[f"emit_T_{g}_{ang}"] = np.histogram(np.clip(o_[s], e[0] + 1e-6, e[-1] - 1e-6), e)[0]
    np.savez_compressed(os.path.join(FAST, "hists", f"{reg}__classT_route.npz"), **Hk)
    ref_path = os.path.join(FAST, "hists", f"{reg}__refknn_emit.npz")
    ref_emit = dict(np.load(ref_path)) if os.path.exists(ref_path) else None
    rng = np.random.default_rng(21)
    for var in variants:
        cnn, draw = var.split(":"); draw, _, sd_ = draw.partition("@")        # knn@<seed>: another donor pick (noise estimate)
        hA, hB, sA, sB = cnn_source(cnn, reg, m) if cnn != "ref" else rA_ref
        R = np.c_[tA - hA, tB - hB]
        if draw in ("knn", "nodonor"):
            Z = np.c_[hA, hB, sA, sB, m.sizeX, m.sizeY, np.log(m.charge.clip(lower=1)), m.globalR, np.abs(m.globalZ), m.localX, m.localY]
            d, pit = knn_draw_rank(Z, surf, R, clsT, half, valid, np.random.default_rng(int(sd_ or 21)))
            if draw == "nodonor":
                d = np.zeros_like(d); pit[:] = np.nan
        elif draw in ("a1arh", "a1mdn"):
            P = pd.read_parquet(os.path.join(FAST, "preds", f"a1_{draw[2:]}_{cnn}_{reg}.parquet")).set_index("row").reindex(m.row.values)
            d = np.c_[P.dA.values, P.dB.values]; pit = np.c_[P.pitA.values, P.pitB.values]
        elif draw == "a2":
            P = pd.read_parquet(os.path.join(FAST, "preds", f"{cnn}.parquet")).set_index("row").reindex(m.row.values)
            d = np.c_[P.dA.values, P.dB.values]; pit = np.c_[P.pitA.values, P.pitB.values]
        pA, pB = sat(hA + np.nan_to_num(d[:, 0])), sat(hB + np.nan_to_num(d[:, 1]))
        cA, cB, fA, fB = respond(pA, pB, disc, seed=77)
        prodA, prodB = np.where(clsT, oTA, cA), np.where(clsT, oTB, cB)
        res = {"variant": var}
        a_s = auc(np.c_[probe, prodA, prodB], y, half, grp=grp); a_f = auc(np.c_[F1, prodA, prodB], y, half, grp=grp)
        leak = {g: {"leak_strong": a_s[g] - base["strong"][g], "leak_F1": a_f[g] - base["F1"][g]} for g in a_s}
        if cnn != "ref":
            own = np.c_[probe, hA, hB, sA, sB]; b_own = auc(own, y, half, grp=grp); a_o = auc(np.c_[own, prodA, prodB], y, half, grp=grp)
            for g in a_o:
                leak[g]["leak_own"] = a_o[g] - b_own[g]
        res.update({f"{k}_{g}": v for g in leak for k, v in leak[g].items()})
        H = {}
        for g in np.unique(grp):
            s = clsT & (grp == g) & np.isfinite(d).all(1)
            for ang, j, c_, t_, h_ in (("alpha", 0, cA, tA, hA), ("beta", 1, cB, tB, hB)):
                r = dict(variant=var, group=g, angle=ang, **closure(c_[clsT & (grp == g)], t_[clsT & (grp == g)]))
                r.update({f"pit_{k}": v for k, v in pit_stats(pit[s, j]).items()})
                r["cnn_full"] = closure(h_[clsT & (grp == g)], t_[clsT & (grp == g)])["full"]
                r["cnn_core"] = closure(h_[clsT & (grp == g)], t_[clsT & (grp == g)])["core"]
                e = HB["A" if ang == "alpha" else "B"]
                for lab, sm in (("CU", CU & (grp == g)), ("CL", CL & (grp == g))):
                    x = c_[sm]; r[f"emit_{lab}_rms"] = float(np.sqrt(np.mean(x ** 2))); r[f"emit_{lab}_gt60"] = float(np.mean(np.abs(x) > 60))
                    H[f"emit_{lab}_{g}_{ang}"] = np.histogram(np.clip(x, e[0] + 1e-6, e[-1] - 1e-6), e)[0]
                    if ref_emit is not None:
                        r[f"emit_{lab}_KS_vs_knnref"] = ks(x, ref_emit[f"{lab}_{g}_{ang}"])
                r["emit_T_rms"] = float(np.sqrt(np.mean((oTA if j == 0 else oTB)[clsT & (grp == g)] ** 2)))
                r["emit_CU_KS_vs_T"] = ks(c_[CU & (grp == g)], (oTA if j == 0 else oTB)[clsT & (grp == g)])
                r["sat_C"] = float(np.mean((fA if j == 0 else fB)[~clsT & (grp == g)]))
                H[f"res_C_{g}_{ang}"] = np.histogram(np.clip(c_[clsT & (grp == g)] - t_[clsT & (grp == g)], HR[0] + 1e-6, HR[-1] - 1e-6), HR)[0]
                H[f"pit_{g}_{ang}"] = np.histogram(pit[s, j][np.isfinite(pit[s, j])], np.linspace(0, 1, 21))[0]
                rows.append(r)
            if s.sum() > 100:
                dr, tr_ = d[s], R[s]
                q1, q2 = quad(dr[:, 0], dr[:, 1]), quad(tr_[:, 0], tr_[:, 1])
                rows.append(dict(variant=var, group=g, angle="corr", pearson_draw=float(np.corrcoef(dr[:, 0], dr[:, 1])[0, 1]),
                                 pearson_true=float(np.corrcoef(tr_[:, 0], tr_[:, 1])[0, 1]),
                                 spearman_draw=float(spearmanr(dr[:, 0], dr[:, 1])[0]), spearman_true=float(spearmanr(tr_[:, 0], tr_[:, 1])[0]),
                                 quadL1=float(np.abs(q1 - q2).sum()), quad_draw=json.dumps(q1.round(4).tolist()), quad_true=json.dumps(q2.round(4).tolist())))
                H[f"r2d_draw_{g}"] = np.histogram2d(np.clip(dr[:, 0], -30, 30), np.clip(dr[:, 1], -60, 60), [np.linspace(-30, 30, 61), np.linspace(-60, 60, 61)])[0]
                H[f"r2d_true_{g}"] = np.histogram2d(np.clip(tr_[:, 0], -30, 30), np.clip(tr_[:, 1], -60, 60), [np.linspace(-30, 30, 61), np.linspace(-60, 60, 61)])[0]
        if var == "ref:knn":
            ref_emit = {f"{lab}_{g}_{ang}": (cA if ang == "alpha" else cB)[sm & (grp == g)]
                        for lab, sm in (("CU", CU), ("CL", CL)) for g in np.unique(grp) for ang in ("alpha", "beta")}
            np.savez_compressed(ref_path, **{k: v.astype(np.float32) for k, v in ref_emit.items()})
        for r in rows:
            if r.get("variant") == var:
                r.update({k: v for k, v in leak.get(r["group"], {}).items()}); r.update({f"{k}_region": v for k, v in leak["all"].items()})
        # sigma-LUT question: 68% half-width (q84 - q16)/2 of (output - truth) per surface x |emitted cot| bin, C route vs T route (class T)
        for sv in np.unique(surf):
            for ang, c_, o_, t_ in (("alpha", cA, oTA, tA), ("beta", cB, oTB, tB)):
                for lo, hi in ((0, 0.5), (0.5, 1.07), (1.07, 2), (2, 4), (4, 1e9)):
                    rec = dict(variant=var, surface=int(sv), angle=ang, cot_lo=lo, cot_hi=hi)
                    for lab_, x_ in (("C", c_), ("T", o_)):
                        sm = clsT & (surf == sv) & (np.abs(np.tan(np.radians(x_))) >= lo) & (np.abs(np.tan(np.radians(x_))) < hi)
                        r_ = x_[sm] - t_[sm]
                        rec[f"n_{lab_}"] = int(sm.sum()); rec[f"hw68_{lab_}"] = float(np.subtract(*np.quantile(r_, [0.84, 0.16])) / 2) if sm.sum() > 30 else np.nan
                    sm = ~clsT & (surf == sv); ac = np.abs(np.tan(np.radians(c_[sm])))
                    rec["frac_classC_in_bin"] = float(np.mean((ac >= lo) & (ac < hi)))
                    rec["ratio_C_over_T"] = rec["hw68_C"] / rec["hw68_T"] if rec["hw68_T"] else np.nan
                    sig_rows.append(rec)
        np.savez_compressed(os.path.join(FAST, "hists", f"{reg}__{var.replace(':', '__')}.npz"), **H)
        print(var, json.dumps({k: round(v, 5) for k, v in res.items() if k != "variant"}), flush=True)
        save(rows, sig_rows, csv, reg)
    new = save(rows, sig_rows, csv, reg)
    pd.set_option("display.width", 250)
    print(new[new.angle != "corr"].pivot_table(index="variant", columns=["group", "angle"], values=["full", "gt10"], sort=False).round(3).to_string())
