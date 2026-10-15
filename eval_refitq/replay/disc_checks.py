"""C1b: the checks of the final configuration on file 4 (known-good, with the known-bads) and file 6 (the gate):
D1 (the disc angle import, P0.3 test (i) form), D4 (cut coverage on the three bin families), the D1 / D4
known-bads applied post hoc to the model's rows (c1b/PLAN.md rev 4 §3-§4). D3 and D5 are judged in run_c1b."""
from __future__ import annotations

import numpy as np
from scipy import stats

import angle_import as AI
import cuts as CU
import disc_angles as DA
import gates as G
from ngtagger.spdefs import angles as SA


def d1_pulls(own, imp, geo, variant="model"):
    """The innovation angle pull (n, 2) of every row for one import variant, recomputed from the raw
    ingredients (the model's own prediction): (m' - h) / sqrt(sigma'^2 + HCH), m', sigma' from the variant."""
    surf = own["surface"]
    pred, raw, sig, hch = own["cot_pred"], own["m_raw"], own["sig_pub"], own["hch"]
    facing = own["facing"]
    if variant == "key_from_local":
        rows = imp.rows_known_bad_key(surf, pred[:, 0], pred[:, 1], facing)
    elif variant == "emitted_key":
        rows = imp.rows(surf, raw[:, 0], raw[:, 1])
    elif variant == "global_rows":
        gt = AI.AngleImport(AI.TABLE).table
        orient = gt.orientation(geo["detId"][own["mi"]])
        rows = gt.rows_of("true", surf, SA.key_from_local(surf, pred[:, 0], pred[:, 1], orient), orient)   # facing-keyed
    else:
        rows = imp.rows(surf, pred[:, 0], pred[:, 1])
    x = np.full(raw.shape, np.nan)
    for i, ang in enumerate(SA.ANGLES):
        if variant == "no_rows":
            mm, ss = raw[:, i], sig[:, i]
        elif variant == "global_rows":
            c, w = gt.get("true", ang, "centre", rows), gt.get("true", ang, "width", rows)
            mm, ss = raw[:, i] - c * sig[:, i], w * sig[:, i]
        else:
            mm, ss = imp.calibrate(ang, raw[:, i], sig[:, i], rows,
                                   variant="to_local" if variant == "to_local" else "", facing=facing)
            if variant == "flip_centre":
                c = imp.table.get(DA.KEY, ang, "centre", rows)
                mm = raw[:, i] + c * sig[:, i]
            if variant == "width_div_1.15":
                ss = ss / 1.15
        with np.errstate(invalid="ignore"):
            x[:, i] = (mm - pred[:, i]) / np.sqrt(ss ** 2 + hch[:, i])
    return x, rows


def d1_bins(own, imp, x, by_facing=False):
    """D1's bins: (angle, disc, disc-row key bin[, facing]) over own class-T rows with a clean history (the table's
    population), with the disc table's sd for the widening."""
    out = []
    base = own["table_pop"] & own["clean"] & (own["surface"] > 10)
    for i, ang in enumerate(SA.ANGLES):
        m0 = base & own["has"][:, i] & np.isfinite(x[:, i])
        fac = own["facing"] if by_facing else np.zeros(len(own["facing"]), np.int8)
        keys = np.stack([own["surface"], own["dkeybin"], fac], 1)[m0]
        xs, evs = x[m0, i], own["ev"][m0]
        for k in np.unique(keys, axis=0):
            m = (keys == k).all(1)
            if m.sum() < SA.MIN_FIT:
                continue
            ctr = np.array([np.mean(SA.fine_bin_bounds(int(k[0]), int(k[1])))])
            row = imp.table.rows_of(DA.KEY, np.array([k[0]]), ctr, np.zeros(1, np.int8))[0]
            r = imp.table.doc["keys"][DA.KEY]["rows"][row] if row >= 0 else None
            tsd = (r[ang]["sd_centre"], r[ang]["sd_width"]) if r and r.get(ang) else (0.0, 0.0)
            out.append({"angle": ang, "surface": int(k[0]), "keybin": int(k[1]), "orient": int(k[2]),
                        "x": xs[m], "ev": evs[m], "table_sd": tsd})
    return out


def d1(own, imp, geo, nev, false_veto=False, variants=()):
    """D1 on one file: the model (pooled facings and per facing), its false veto (file 4), the variants (the
    known-bads) and the reported diagnostics (the HCH share, the table-only pull, class C, the key change)."""
    x, rows = d1_pulls(own, imp, geo)
    res = {}
    for lab, byf in (("pooled", False), ("per_facing", True)):
        b = d1_bins(own, imp, x, byf)
        res[lab] = G.e1a_judge(b, nev=nev)
        if false_veto and not byf:
            fv = G.e1a_false_veto(b, res[lab])
            lo = stats.beta.ppf(0.05, fv["vetoed"], fv["resamples"] - fv["vetoed"] + 1) if fv["vetoed"] else 0.0
            res["false_veto"] = dict(fv, lower95=float(lo), material=bool(lo > 0.057))
    res["variants"] = {}
    for v in variants:
        xv, _ = d1_pulls(own, imp, geo, v)
        jp = G.e1a_judge(d1_bins(own, imp, xv), nev=nev)
        jf = G.e1a_judge(d1_bins(own, imp, xv, True), nev=nev)
        res["variants"][v] = {"pooled_fail": jp["n_fail"], "per_facing_fail": jf["n_fail"], "bins": len(jp["bins"]),
                              "fails": jp["n_fail"] + jf["n_fail"] > 0, "facing_only": jp["n_fail"] == 0 < jf["n_fail"]}
    disc = own["table_pop"] & own["clean"] & (own["surface"] > 10)
    w = np.stack([imp.table.get(DA.KEY, a, "width", rows) for a in SA.ANGLES], 1) * own["sig_pub"]
    with np.errstate(invalid="ignore", divide="ignore"):
        share = own["hch"] / w ** 2
    res["hch_share_median"] = [float(np.nanmedian(share[disc, i])) for i in (0, 1)]
    res["table_only_width"] = [float(np.nanmedian(np.abs(own["t1"][disc, i]))) / 0.6745 for i in (0, 1)]
    cc = ~own["classT"] & own["clean"] & (own["surface"] > 10)
    res["class_C"] = {"n": int(cc.sum()), "centre": [float(np.nanmedian(x[cc, i])) for i in (0, 1)],
                      "width": [float(np.nanmedian(np.abs(x[cc, i] - np.nanmedian(x[cc, i]))) / 0.6745) for i in (0, 1)]}
    tk = imp.rows(own["surface"], own["cot_true"][:, 0], own["cot_true"][:, 1])
    res["key_change_fraction"] = float(np.mean(tk[disc] != rows[disc])) if disc.any() else None
    return res


def d4(own, loo, cuts, scen, nev):
    """D4 on one scenario: E1(b) per consumer on the three bin families (the rows carry the cut-bin surface
    code), the rr per number of remaining inner clusters, the class-C gate coverage (a diagnostic) and the
    number of tests."""
    r = G.e1b_all(own, loo, cuts, scen)
    out = {"consumers": r, "material": [], "minor": [], "n_tests": 0}
    for k, v in r.items():
        mat, minor = G.e1b_verdict(scen, k, v, safe_direction_minor=True)
        out["material"] += mat
        out["minor"] += minor
        out["n_tests"] += 1 + len(v["bins"])
    gate, window, rr = CU.populations(own, {k: loo[k] for k in ("own_primary", "clean_track", "n_acc")})
    for dims in ("xy", "xyab"):
        cl = G.e1b_consumer(cuts, "rr", CU.table_key("rr", dims, "calibrated", scen), loo[f"d_{dims}"][rr],
                            loo["surface"][rr], loo["first"][rr], loo["ev"][rr], nev=nev, cls=loo["n_inner"][rr])
        mat, minor, t = G.e1b_class_verdict(scen, dims, cl)
        out[f"rr_per_class_{dims}"] = {"bins": cl["bins"], "t": t}
        out["material"] += mat
        out["minor"] += minor
        out["n_tests"] += len(cl["bins"])
    cc = own["primary"] & ~own["classT"] & own["clean"]
    if cc.any():
        cut = cuts.chi2("gate", "xy", "calibrated", own["surface"][cc], own["first"][cc])
        out["class_C_gate_coverage"] = float(np.mean(own["g_xy"][cc] <= cut))
    return out


def d4_known_bads(own, cuts_975):
    """D4's known-bads on file 4 (post hoc): the 3-sigma Gaussian gate cut and the 97.5 % quantile used as
    the 99 % one; each must fail."""
    gate, _, _ = CU.populations(own, {"own_primary": np.zeros(0, bool), "clean_track": np.zeros(0, bool)})
    x = own["g_xy"][gate]
    out = {"3sigma": {"coverage": float(np.mean(x <= stats.chi2.ppf(0.9973, 2)))}}
    out["3sigma"]["fails"] = out["3sigma"]["coverage"] < 0.985
    r = G.e1b_consumer(cuts_975, "gate", "gate|xy|calibrated", x, own["surface"][gate], own["first"][gate], own["ev"][gate])
    out["q97.5"] = {"pooled": r["pooled"], "max_abs_z": r["max_abs_z"], "t": r["t"],
                    "fails": bool((not r["pooled"]["pass"]) or (r["max_abs_z"] or 0) > r["t"])}
    return out
