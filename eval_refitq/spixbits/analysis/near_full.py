"""spixbits analysis, deliverable C: the near-full-resolution reading (PLAN.md 4a; reviews/stats-review-01.md section 4).

  python near_full.py     -> results/near_full.json, results/near_full.md, results/fig/near_full_<scenario>.png/.svg

Three readings near full resolution, quoted together:
1. measured, upper bound: "network, float" on today's positions (stand-in);
2. model: form B = the measured width scaled by the forced refit's own covariance with the position errors scaled to
   the network (the forecast arm) over the covariance with today's; form A = a straight line of width^2 against the
   single-hit variance s = s_base + step^2/12 through the fine-step points, read at the network's s. Value = form A,
   band between A and B widened by the 68 % bootstrap band; only the interval [B, measured] where the plan says so;
3. direct point above 50 GeV: not run in this scan.

SmartPixels frame: y (12.5 um pitch) is read on d0, x (50 um pitch) on z0.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import core
import lib
from scans import POS_BITS, scan_arm, step_um, save

AXES = {"y": {"q": "d0", "fine_max_step": 18.75, "sensor_today_um": 3.1, "sensor_network_um": 2.0, "var": 0},
        "x": {"q": "z0", "fine_max_step": 75.0, "sensor_today_um": 12.3, "sensor_network_um": 7.6, "var": 1}}


def base_variance():
    """Measured single-hit core RMS^2 of today's positions on barrel layer 1, p >= 40 GeV (AAAA; the cluster table is
    the same in every scenario and IAAA has no layer 1)."""
    for scen in ("AAAA", "AAAI"):
        sc = lib.Scenario(scen)
        if sc.has("network_float"):
            sh = sc.stats("network_float")["single_hit_layer1_p40"]
            return {o: sh[o]["rms_um"] ** 2 for o in ("y", "x")}, {o: sh[o] for o in ("y", "x")}, scen
    return None, None, None


def wls(s, w2, var):
    """Per-replica weighted straight line w2 = a + b s. s (K,), w2 (K, R), var (K,). Returns a, b (R,)."""
    wt = 1.0 / var
    S, Sx, Sxx = wt.sum(), (wt * s).sum(), (wt * s * s).sum()
    Sy, Sxy = (wt[:, None] * w2).sum(0), (wt[:, None] * s[:, None] * w2).sum(0)
    b = (S * Sxy - Sx * Sy) / (S * Sxx - Sx ** 2)
    return (Sy - b * Sx) / S, b


def quad_c(s, w2, var):
    X = np.stack([np.ones_like(s), s, s * s], 1) * (1 / np.sqrt(var))[:, None]
    sol = np.linalg.lstsq(X, w2 / np.sqrt(var)[:, None], rcond=None)[0]
    return sol[2]


def cells_of(sc):
    c = [("prompt, all", "origin", 0)]
    c += [(n, "six", g) for g, n in enumerate(sc.meta["cells"]["six"]["names"]) if n.startswith("prompt")]
    c += [(n, "forward_pooled", g) for g, n in enumerate(sc.meta["cells"]["forward_pooled"]["names"])]
    return c


def analyse(sc, s_base, T, P):
    out = {}
    have_fc = sc.has("forecast_network_float")
    for o, ax in AXES.items():
        q = ax["q"]
        gap = ax["sensor_today_um"] ** 2 - ax["sensor_network_um"] ** 2
        s0, s_net = s_base[o], s_base[o] - gap
        pts = [(b, scan_arm("down", o, b)) for b in POS_BITS[o] if sc.has(scan_arm("down", o, b))]
        # the line is drawn through every point with a fine step; the plan's rule needs 3 of them with >= 98 % of the
        # tracks on the anchor's clusters, otherwise only the interval is the reading and the line is informational
        fine = [(b, a) for b, a in pts if step_um(b) <= ax["fine_max_step"]]
        qualifying = [(b, a) for b, a in fine if sc.stats(a)["same_clusters_share"]["network_float"] >= 0.98]
        excluded = [f"{a} (same clusters as network, float: {sc.stats(a)['same_clusters_share']['network_float']:.3f})" for b, a in fine if (b, a) not in qualifying]
        pulls_bad = [a for b, a in fine if any(abs(v["rms_within_5"] - 1) > 0.15 for v in sc.stats(a)["correct_hit_pull"]["all surfaces"].values())]
        A = {"single_hit_variance_today_um2": s0, "single_hit_variance_network_um2": s_net, "gap_um2": gap,
             "fit_points": ["network_float"] + [a for _, a in fine], "fine_points_below_98pct_same_clusters": excluded,
             "n_qualifying_fine_points": len(qualifying), "cells": {}}
        for label, part, g in cells_of(sc):
            w0, _, r0 = sc.w("network_float", q, part, g)
            d = {"measured_upper_bound_um": lib.val(w0, r0)}
            # form B: covariance forecast on forced refits (the same clusters)
            WB = None
            if have_fc:
                lab = P[part][0] == g
                vf, va = sc.c("forecast_network_float")["forced_var"][lab, ax["var"]], sc.c("network_float")["forced_var"][lab, ax["var"]]
                ok = (va > 0) & (vf > 0)
                fac = float(np.median(np.sqrt(vf[ok] / va[ok])))
                WB = (w0 * fac, r0 * fac)
                d["form_B_covariance_forecast_um"] = lib.val(*WB)
                d["form_B_factor"] = fac
            # form A: straight line in single-hit variance through the fine-step points (and the float anchor at s0)
            s = np.array([s0] + [s0 + step_um(b) ** 2 / 12 for b, _ in fine])
            wp = np.array([w0] + [sc.w(a, q, part, g)[0] for _, a in fine])
            wr = np.stack([r0] + [sc.w(a, q, part, g)[2] for _, a in fine])
            reasons = []
            WA = None
            if len(qualifying) < 3:
                reasons.append(f"fewer than 3 fine-step points with >= 98 % of tracks on unchanged clusters ({len(qualifying)} of {len(fine)} scanned)")
            if len(fine) >= 2:
                var = np.maximum((wr ** 2).var(1), 1e-12)
                a_r, b_r = wls(s, wr ** 2, var)
                a_p, b_p = wls(s, (wp ** 2)[:, None], var)
                a_p, b_p = float(a_p[0]), float(b_p[0])
                d["form_A_slope_um2_per_um2"] = lib.val(b_p, b_r)
                insens = abs(b_p) < 2 * b_r.std()
                d["no_measurable_sensitivity"] = bool(insens)
                wa_p = np.sqrt(max(a_p + b_p * s_net, 0.0))
                wa_r = np.sqrt(np.maximum(a_r + b_r * s_net, 0.0))
                WA = (wa_p, wa_r)
                d["form_A_straight_line_um"] = lib.val(wa_p, wa_r)
                if len(s) >= 4:
                    c_r = quad_c(s, wr ** 2, var)
                    c_p = float(quad_c(s, (wp ** 2)[:, None], var)[0])
                    d["curvature_over_sd"] = c_p / max(c_r.std(), 1e-30)
                    if abs(c_p) > 2 * c_r.std():
                        reasons.append(f"significant curvature ({d['curvature_over_sd']:+.1f} sd)")
                change = abs(wp[int(np.argmax(s))] - w0)
                d["measured_change_over_fit_points_um"] = float(change)
                if WB is not None and abs(wa_p - WB[0]) > change and not insens:
                    reasons.append(f"the two model values are further apart ({abs(wa_p - WB[0]):.2f} um) than the measured change over the fit points ({change:.2f} um)")
            if pulls_bad:
                reasons.append("correct-hit pull RMS outside 1 ± 0.15 at a fit point: " + ", ".join(pulls_bad))
            d["reasons_interval_only"] = reasons
            if d.get("no_measurable_sensitivity"):
                d["reading"] = f"no measurable sensitivity to the {o} step: network value = measured {w0:.1f} um"
                d["value_um"], d["band_um"] = float(w0), [float(lib.band(r0)[0]), float(lib.band(r0)[1])]
            elif WA is not None and WB is not None and not reasons:
                vals = [WA] + ([WB] if WB else [])
                lo = min(lib.band(v[1])[0] for v in vals)
                hi = max(lib.band(v[1])[1] for v in vals)
                d["value_um"], d["band_um"] = float(WA[0]), [float(lo), float(hi)]
                d["reading"] = f"model: {WA[0]:.1f} um, band {lo:.1f}-{hi:.1f} um" + ("" if WB else " (form B missing: forecast arm not complete)") + f"; measured upper bound {w0:.1f} um"
            elif WB is not None:
                d["value_um"], d["band_um"] = None, [float(lib.band(WB[1])[0]), float(lib.band(r0)[1])]
                d["reading"] = f"interval only: {WB[0]:.1f} um (covariance forecast) to {w0:.1f} um (measured stand-in)"
                if WA is not None:
                    d["reading"] += f"; straight line, informational only: {WA[0]:.1f} um"
            else:
                d["value_um"], d["band_um"] = None, None
                d["reading"] = f"not given yet (forecast arm not complete); measured upper bound {w0:.1f} um"
                if WA is not None:
                    d["reading"] += f"; straight line{', informational only' if reasons else ' (meets the rule; band pending the forecast arm)'}: {WA[0]:.1f} um"
            # bits needed at the network's base (model based): numerator measured where step^2/12 >= gap
            if WA is not None and len(pts) == len(POS_BITS[o]):
                s_all = np.array([s0] + [s0 + step_um(b) ** 2 / 12 for b, _ in pts[::-1]])  # ascending in s
                w_all = np.stack([r0] + [sc.w(a, q, part, g)[2] for _, a in pts[::-1]]) ** 2
                bits = [b for b, _ in pts]
                res = {}
                for nm, den in (("form A", WA[1]), ("form B", WB[1] if WB else None)):
                    if den is None:
                        continue
                    R = []
                    for b in bits:
                        st = s_net + step_um(b) ** 2 / 12
                        if st >= s0:  # measured: interpolate width^2 linearly in s between scan points
                            k = np.clip(np.searchsorted(s_all, st) - 1, 0, len(s_all) - 2)
                            f = (st - s_all[k]) / (s_all[k + 1] - s_all[k])
                            num = np.sqrt(w_all[k] * (1 - f) + w_all[k + 1] * f)
                        else:  # below today's base: the straight line
                            num = np.sqrt(np.maximum(a_r + b_r * st, 0.0))
                        R.append(num / den)
                    R = np.stack(R)
                    Rm = np.maximum.accumulate(R[::-1], 0)[::-1]
                    okm = Rm <= 1.10
                    first = np.where(okm.any(0), okm.argmax(0), len(bits))
                    codes = np.array(bits + [bits[-1] + 1])[first]
                    q90, med = np.quantile(codes, 0.9, method="higher"), np.quantile(codes, 0.5, method="higher")
                    f = lambda v: f"more than {bits[-1]}" if v > bits[-1] else str(int(v))
                    res[nm] = {"bits_90pct": f(q90), "bits_median": f(med), "first_measured_bits": max([b for b in bits if s_net + step_um(b) ** 2 / 12 >= s0], default=None)}
                d["bits_needed_network_base_model"] = res
            d["fit"] = {"s_um2": s.tolist(), "w68_um": wp.tolist(), "w68_sd_um": wr.std(1).tolist()}
            A["cells"][label] = d
        out[o] = A
    if sc.has("forecast_bending_float") and sc.has("bending_float"):
        B = {}
        for o, ax in AXES.items():
            lab = P["origin"][0] == 0
            vf, va = sc.c("forecast_bending_float")["forced_var"][lab, ax["var"]], sc.c("bending_float")["forced_var"][lab, ax["var"]]
            ok = (va > 0) & (vf > 0)
            fac = float(np.median(np.sqrt(vf[ok] / va[ok])))
            w0, _, r0 = sc.w("bending_float", ax["q"])
            B[ax["q"]] = {"measured_upper_bound_um": lib.val(w0, r0), "form_B_covariance_forecast_um": lib.val(w0 * fac, r0 * fac), "form_B_factor": fac}
        out["bending_angle_only_float"] = B
    return out


def figure(sc, res):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for axp, (o, ax) in zip(axes, AXES.items()):
        d = res[o]["cells"]["prompt, all"]
        f = d["fit"]
        allp = [(b, scan_arm("down", o, b)) for b in POS_BITS[o] if sc.has(scan_arm("down", o, b)) and step_um(b) <= 4 * ax["fine_max_step"]]
        s0 = res[o]["single_hit_variance_today_um2"]
        xs = [np.sqrt(s0 + step_um(b) ** 2 / 12) for b, _ in allp]
        w = [sc.w(a, ax["q"]) for _, a in allp]
        axp.errorbar(xs, [v[0] for v in w], yerr=[np.std(v[2]) for v in w], fmt="o", label="measured, quantized (down-scan)")
        w0 = d["measured_upper_bound_um"]
        axp.errorbar([np.sqrt(s0)], [w0["value"]], yerr=[w0["sd"]], fmt="s", label="measured: network, float on today's positions (upper bound)")
        sn = np.sqrt(res[o]["single_hit_variance_network_um2"])
        if d.get("band_um"):
            lo, hi = d["band_um"]
            axp.fill_between([sn * 0.97, sn * 1.03], [lo, lo], [hi, hi], facecolor="none", hatch="///", edgecolor="C3",
                             label="model" if d.get("value_um") else "interval (covariance forecast to measured)")
            if d.get("value_um"):
                axp.plot([sn], [d["value_um"]], "_", color="C3", markersize=14)
        axp.set_xscale("log")
        axp.set_xlabel(f"single-hit resolution in {o} on barrel layer 1 [um] (today's {np.sqrt(s0):.1f}, network {sn:.1f})")
        axp.set_ylabel(f"{ax['q']} 68 % half-width [um]")
        axp.set_title(f"{sc.scen}, prompt tracks: {ax['q']} near full resolution in {o}")
        axp.legend(fontsize=7)
    fig.tight_layout()
    save(fig, f"near_full_{sc.scen}")


def main():
    lib.check_disk()
    s_base, sh, src = base_variance()
    if s_base is None:
        raise SystemExit("network_float not summarized yet")
    res = {"provenance": lib.provenance(core.SCENARIOS), "single_hit_today_barrel_layer1_p40": sh, "single_hit_source_scenario": src,
           "direct_point_above_50_GeV": "not run in this scan", "scenarios": {}}
    L = ["## Near-full-resolution reading (PLAN 4a)", "",
         f"Single-hit resolution of today's positions on barrel layer 1 (prompt, p ≥ 40 GeV, {sh['y']['n']} clusters, {src}): core RMS {sh['y']['rms_um']:.1f} µm in y, {sh['x']['rms_um']:.1f} µm in x "
         f"(68 % half-widths {sh['y']['w68_um']:.1f} / {sh['x']['w68_um']:.1f} µm). Network: sensor term 2.0 µm (y) / 7.6 µm (x) in place of today's 3.1 / 12.3 µm.", "",
         "Measured = \"network, float\" on today's positions (an upper bound). Model: form A = straight line of width² against single-hit variance through the fine-step points, read at the network's variance; "
         "form B = measured width × the forced refit's covariance ratio with position errors scaled to the network. The direct point above 50 GeV was not run.", ""]
    for scen in core.SCENARIOS:
        sc = lib.Scenario(scen)
        if sc.meta is None or not sc.has("network_float"):
            continue
        T = core.Truth(scen)
        P = core.partitions(T)
        r = analyse(sc, s_base, T, P)
        res["scenarios"][scen] = r
        figure(sc, r)
        for o, ax in AXES.items():
            L += [f"### {scen}: {ax['q']} against the {o} resolution", "",
                  f"Points of the straight line: {', '.join(r[o]['fit_points'])}." + (f" Below the 98 % same-clusters rule: {'; '.join(r[o]['fine_points_below_98pct_same_clusters'])}." if r[o]["fine_points_below_98pct_same_clusters"] else ""), "",
                  "| cell | measured upper bound [µm] | form A, straight line [µm] | form B, covariance forecast [µm] | reading by the plan's rule | bits needed at the network's base, model (form A / form B denominators; sub-area bits) |", "|---|---|---|---|---|---|"]
            for c, d in r[o]["cells"].items():
                bn = d.get("bits_needed_network_base_model", {})
                bns = " / ".join(f"{v['bits_90pct']} (median {v['bits_median']})" for v in bn.values()) or "n/a"
                rs = d["reading"] + ("; " + "; ".join(d["reasons_interval_only"]) if d["reasons_interval_only"] else "")
                L.append(f"| {c} | {lib.fmt(d['measured_upper_bound_um'])} | {lib.fmt(d.get('form_A_straight_line_um'))} | {lib.fmt(d.get('form_B_covariance_forecast_um'))} | {rs} | {bns} |")
            L.append("")
        if "bending_angle_only_float" in r:
            b = r["bending_angle_only_float"]
            L += [f"{scen}, bending-angle only, float: covariance forecast d0 {lib.fmt(b['d0']['form_B_covariance_forecast_um'])} µm (measured {lib.fmt(b['d0']['measured_upper_bound_um'])}), "
                  f"z0 {lib.fmt(b['z0']['form_B_covariance_forecast_um'])} µm (measured {lib.fmt(b['z0']['measured_upper_bound_um'])}).", ""]
    json.dump(res, open(lib.RESULTS / "near_full.json", "w"), indent=1)
    open(lib.RESULTS / "near_full.md", "w").write("\n".join(L) + "\n")
    print("written near_full")


if __name__ == "__main__":
    main()
