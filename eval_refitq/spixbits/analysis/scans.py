"""spixbits analysis, deliverable B: scans, allocations, bending-angle-only family, controls, "bits needed",
b/c-daughter and displaced-track tables, secondary |d0| separation (PLAN.md sections 5 and 6).

  python scans.py     -> results/scans.json, results/scans_<scenario>.md, results/fig/*.png and *.svg

Reads only the caches of core.py and uses whatever arms are complete; every table says which arms were missing.
SmartPixels frame: x along the 50 um pitch (drives z0), y along the 12.5 um pitch (drives d0), beta = bending angle.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import core
import lib

POS_BITS = {"y": list(range(2, 11)), "x": list(range(2, 9))}
ANG_BITS = [0, 2, 3, 4, 5, 6, 7]  # 0 = not sent
READ = {"y": "d0", "beta": "d0", "x": "z0", "alpha": "z0"}
ALLOCS = ["alloc_4444", "alloc_5533", "alloc_6622", "alloc_5623", "alloc_4633", "alloc_4723", "alloc_5722", "alloc_6703",
          "alloc_6604", "alloc_7702", "alloc_8800"]
FIXED = ["implemented_now", "implemented_now_sizeblind", "implemented_now_newtable", "network_float", "alloc_4444",
         "alloc_4444_wholesensor", "alloc_6604", "alloc_6604_wholesensor", "bending_float", "positions_only_float"]


def step_um(bits):
    return 2400.0 / 2 ** bits


def scan_arm(anchor, out, b):
    """Arm name of a scan point. anchor 'down' (others float) or 'up' (others at 4 bits; its 4-bit point is 4/4/4/4)."""
    if anchor == "up" and b == 4:
        return "alloc_4444"
    return f"{anchor}_{out}_" + ("notsent" if b == 0 else f"{b}b")


def bending_arm(kind, b):
    """Bending-angle-only family from 6/6/0/4: kind 'beta' (beta bits) or 'xy' (x = y bits)."""
    if (kind == "beta" and b == 4) or (kind == "xy" and b == 6):
        return "alloc_6604"
    return f"bending_{kind}_{b}b"


def plain(arm):
    if arm in lib.PLAIN:
        return lib.PLAIN[arm]
    a = lib.arm_label(REG, arm)
    s = a["smartpixels_frame"] if a else arm
    extra = ""
    for tag, txt in (("_globalrange", ", one global angle range"), ("_v3window", ", v3 window unchanged"), ("_flagunusable", ", flagged angles unusable"),
                     ("_allsent", ", all clusters sent"), ("noise_", ", Gaussian noise in place of quantization"), ("forecast_", ", position errors scaled to the network (forecast)")):
        if tag in arm:
            extra += txt
    if a and "whole" in a["position_range"]:
        extra += ", whole sensor"
    return s + extra


def bits_string(arm):
    a = lib.arm_label(REG, arm)
    return a["smartpixels_frame"] if a else ""


def row(sc, arm, part="origin", g=0, refs=("implemented_now", "network_float")):
    d = {"name": plain(arm)}
    for q in ("d0", "z0"):
        w, w95, rp = sc.w(arm, q, part, g)
        d[q] = lib.val(w, rp)
        d[q + "_w95"] = w95
        for ref in refs:
            if sc.has(ref):
                d[f"{q}_ratio_to_{ref}"] = lib.ratio(sc, arm, ref, q, part, g)
    return d


# ------------------------------------------------------------------ bits needed (stats-review-01 section 3)
def bits_needed_reps(sc, arms_by_bits, anchor, q, part, g):
    """b*_r per replica; bits sorted ascending; value = K+1 th entry coded as bmax + 1 ("more than bmax")."""
    bits = sorted(arms_by_bits)
    A = sc.w(anchor, q, part, g)[2]
    R = np.stack([sc.w(arms_by_bits[b], q, part, g)[2] / A for b in bits])  # (K, NREP)
    Rmax = np.maximum.accumulate(R[::-1], axis=0)[::-1]  # running maximum from the finest value
    ok = Rmax <= 1.10
    first = np.where(ok.any(0), ok.argmax(0), len(bits))
    codes = np.array(bits + [bits[-1] + 1])
    point = np.array([sc.w(arms_by_bits[b], q, part, g)[0] for b in bits]) / sc.w(anchor, q, part, g)[0]
    return codes[first], bits, R, point


def bits_label(b, bits, is_angle):
    if b > bits[-1]:
        return f"more than {bits[-1]}"
    if is_angle and b == 0:
        return "not sent"
    return str(int(b))


def bits_needed(sc, arms_by_bits, anchor, q, cells, is_angle):
    """Readout per cell and the simultaneous design number over `design` cells. cells: list of (label, part, g, n)."""
    out = {"cells": {}}
    per_cell = {}
    bits = sorted(arms_by_bits)
    for label, part, g, n in cells:
        if n < 950:  # 950-999: read, tagged as a minor deviation from the 1,000-track rule
            out["cells"][label] = {"n": n, "reading": "not read: fewer than 1,000 tracks"}
            continue
        bs, bits, R, point = bits_needed_reps(sc, arms_by_bits, anchor, q, part, g)
        per_cell[label] = bs
        q90, med = np.quantile(bs, 0.90, method="higher"), np.quantile(bs, 0.50, method="higher")
        d = {"n": n, "bits_90pct": bits_label(q90, bits, is_angle), "bits_median": bits_label(med, bits, is_angle),
             "share_of_replicas_nothing_passes": float((bs > bits[-1]).mean()), "share_of_replicas_coarsest_passes": float((bs == bits[0]).mean()),
             "ratio_curve": {("not sent" if (is_angle and b == 0) else str(b)): {"value": float(point[i]), "sd": float(R[i].std()), "upper90": float(np.quantile(R[i], 0.9))} for i, b in enumerate(bits)}}
        if d["share_of_replicas_nothing_passes"] > 0.10:
            d["reading"] = f"more than {bits[-1]} bits (ratio at {bits[-1]} bits {point[-1]:.3f}, upper 90 % bound {np.quantile(R[-1], 0.9):.3f})"
        elif d["share_of_replicas_coarsest_passes"] >= 0.90:
            d["reading"] = ("not sent passes" if (is_angle and bits[0] == 0) else f"{bits[0]} or fewer")
        else:
            d["reading"] = d["bits_90pct"] + f" (median {d['bits_median']})"
        # non-monotonic feature: a finer value worse than the next coarser one by more than 2 paired sd
        nm = []
        for i in range(len(bits) - 1):
            diff = R[i + 1] - R[i]
            if point[i + 1] - point[i] > 2 * diff.std() and point[i + 1] > point[i]:
                nm.append(f"{bits[i + 1]} bits worse than {bits[i]} by {point[i + 1] - point[i]:.3f} ({(point[i + 1] - point[i]) / diff.std():.1f} sd)")
        d["non_monotonic"] = nm
        out["cells"][label] = d
    def design(labels):
        mx = np.max(np.stack([per_cell[l] for l in labels]), axis=0)
        q90, med = np.quantile(mx, 0.90, method="higher"), np.quantile(mx, 0.50, method="higher")
        return {"bits_90pct": bits_label(q90, bits, is_angle), "bits_median": bits_label(med, bits, is_angle),
                "share_of_replicas_nothing_passes": float((mx > bits[-1]).mean()), "cells": labels}

    six = [l for l, p, *_ in cells if p == "six" and l.startswith("prompt")]
    if all(l in per_cell for l in six):
        out["design_number_six_cells"] = design(six)
        short = [f"{l} (n = {n})" for l, p, g, n in cells if l in six and n < 1000]
        if short:
            out["design_number_six_cells"]["minor_deviation"] = "read with a cell just below 1,000 tracks: " + "; ".join(short)
    else:
        out["design_number_six_cells"] = {"reading": "not given: a design cell has fewer than 950 tracks"}
    four = [l for l in six if "|eta| 0-1.1" in l] + [l for l, p, *_ in cells if p == "forward_pooled"]
    if all(l in per_cell for l in four):
        out["design_number_forward_pooled"] = design(four)
    return out


def read_cells(sc):
    c = [("prompt, all", "origin", 0, sc.meta["origin_counts"][0])]
    for g, (n, k) in enumerate(zip(sc.meta["cells"]["six"]["names"], sc.meta["cells"]["six"]["counts"])):
        c.append((n, "six", g, k))
    for g, (n, k) in enumerate(zip(sc.meta["cells"]["forward_pooled"]["names"], sc.meta["cells"]["forward_pooled"]["counts"])):
        c.append((n, "forward_pooled", g, k))
    return c


# ------------------------------------------------------------------ figures
def save(fig, name):
    d = lib.RESULTS / "fig"
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{name}.png", dpi=110)
    fig.savefig(d / f"{name}.svg")
    plt.close(fig)


def fig_scan(sc, out):
    """d0 and z0 68 % width (prompt, all) against the scanned output, both anchors; 95 % width dashed."""
    pos = out in ("x", "y")
    bits_all = POS_BITS[out] if pos else ANG_BITS
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    drew = False
    for ax, q in zip(axes, ("d0", "z0")):
        for anchor, lab in (("down", "others float (down from network, float)"), ("up", "others at 4 bits (up from 4/4/4/4)")):
            pts = [(b, scan_arm(anchor, out, b)) for b in bits_all if sc.has(scan_arm(anchor, out, b))]
            if len(pts) < 2:
                continue
            drew = True
            xs = [step_um(b) if pos else (1 if b == 0 else b) for b, _ in pts]
            w = [sc.w(a, q) for _, a in pts]
            y = np.array([v[0] for v in w])
            lo = np.array([lib.band(v[2])[0] for v in w])
            hi = np.array([lib.band(v[2])[1] for v in w])
            l, = ax.plot(xs, y, "o-", label=f"68 %: {lab}")
            ax.fill_between(xs, lo, hi, alpha=0.3, color=l.get_color())
            ax.plot(xs, [v[1] for v in w], "x--", color=l.get_color(), label="95 % (plotted, not read)")
        for ref, txt in (("network_float", "network, float"), ("implemented_now", "implemented now")):
            if sc.has(ref):
                v = sc.w(ref, q)
                l = ax.axhline(v[0], linestyle=":", color="k" if ref == "network_float" else "gray", label=f"68 %: {txt}")
        ax.set_yscale("log")
        ax.set_ylabel(f"{q} half-width about zero [um]")
        if pos:
            ax.set_xscale("log")
            ax.invert_xaxis()
            ax.set_xticks([step_um(b) for b in bits_all])
            ax.set_xticklabels([f"{step_um(b):.3g}\n{b}\n{b + 2}" for b in bits_all])
            ax.minorticks_off()
            ax.set_xlabel(f"SmartPixels {out}: step [um] / sub-area bits / whole-sensor bits")
        else:
            ax.set_xticks([1] + bits_all[1:])
            ax.set_xticklabels(["not sent"] + [str(b) for b in bits_all[1:]])
            ax.set_xlabel(f"SmartPixels {out}{' (bending angle)' if out == 'beta' else ''}: bits per link range")
        ax.set_title(f"{sc.scen}, prompt tracks: {q} against {out}")
        ax.legend(fontsize=7)
    fig.tight_layout()
    if drew:
        save(fig, f"scan_{out}_{sc.scen}")
    else:
        plt.close(fig)
    return drew


def fig_scan_cells(sc, out):
    """Ratio to "network, float" of the quantity the output drives, down-scan, six prompt cells."""
    pos = out in ("x", "y")
    bits_all = POS_BITS[out] if pos else ANG_BITS
    pts = [(b, scan_arm("down", out, b)) for b in bits_all if sc.has(scan_arm("down", out, b))]
    if len(pts) < 2:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, q in zip(axes, ("d0", "z0")):
        xs = [step_um(b) if pos else (1 if b == 0 else b) for b, _ in pts]
        for g, name in enumerate(sc.meta["cells"]["six"]["names"]):
            if not name.startswith("prompt"):
                continue
            r = [lib.ratio(sc, a, "network_float", q, "six", g) for _, a in pts]
            ax.errorbar(xs, [v["value"] for v in r], yerr=[[max(v["value"] - v["lo68"], 0) for v in r], [max(v["hi68"] - v["value"], 0) for v in r]],
                        marker="o", capsize=2, label=name.replace("prompt, ", ""))
        ax.axhline(1.10, linestyle=":", color="k")
        ax.set_ylabel(f"{q} half-width / network, float")
        ax.set_ylim(0.9, 2.0)
        if pos:
            ax.set_xscale("log")
            ax.invert_xaxis()
            ax.set_xticks([step_um(b) for b in bits_all])
            ax.set_xticklabels([f"{step_um(b):.3g}\n{b}\n{b + 2}" for b in bits_all])
            ax.minorticks_off()
            ax.set_xlabel(f"SmartPixels {out}: step [um] / sub-area bits / whole-sensor bits")
        else:
            ax.set_xticks([1] + bits_all[1:])
            ax.set_xticklabels(["not sent"] + [str(b) for b in bits_all[1:]])
            ax.set_xlabel(f"SmartPixels {out}{' (bending angle)' if out == 'beta' else ''}: bits per link range")
        ax.set_title(f"{sc.scen}, prompt cells, others float: {q} against {out} (dotted: +10 %)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, f"scan_cells_{out}_{sc.scen}")
    return True


def fig_bending(sc):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    drew = False
    for ax, (kind, bits_all) in zip(axes, (("beta", [2, 3, 4, 5, 6, 7]), ("xy", [4, 5, 6, 7, 8]))):
        for q in ("d0", "z0"):
            for suffix, ls in (("", "o-"), ("_globalrange", "s--")):
                pts = [(b, bending_arm(kind, b) + suffix) for b in bits_all if sc.has(bending_arm(kind, b) + suffix)]
                if len(pts) < 2 or (suffix and kind == "xy"):
                    continue
                drew = True
                r = [lib.ratio(sc, a, "bending_float", q) for _, a in pts]
                ax.errorbar([b for b, _ in pts], [v["value"] for v in r], yerr=[[max(v["value"] - v["lo68"], 0) for v in r], [max(v["hi68"] - v["value"], 0) for v in r]],
                            fmt=ls, capsize=2, label=f"{q}{', one global angle range' if suffix else ''}")
        ax.axhline(1.10, linestyle=":", color="k")
        ax.set_ylabel("half-width / bending-angle only, float")
        ax.set_xlabel("beta (bending angle) bits, x = y = 6 bits (37.5 um)" if kind == "beta" else "x = y sub-area bits (whole-sensor bits = +2), beta 4 bits")
        ax.set_title(f"{sc.scen}, prompt tracks: bending-angle-only family")
        ax.legend(fontsize=7)
    fig.tight_layout()
    if drew:
        save(fig, f"bending_family_{sc.scen}")
    else:
        plt.close(fig)
    return drew


# ------------------------------------------------------------------ tables
def md_rows(rows, refs=("implemented_now", "network_float"), extra=()):
    head = "| arm | d0 [µm] | z0 [µm] | " + " | ".join(f"d0 ÷ {lib.PLAIN[r]} | z0 ÷ {lib.PLAIN[r]}" for r in refs) + "".join(f" | {e}" for e, _ in extra) + " |"
    L = [head, "|---|---|---|" + "---|---|" * len(refs) + "---|" * len(extra)]
    for arm, d in rows.items():
        L.append(f"| {d['name']} | {lib.fmt(d['d0'])} | {lib.fmt(d['z0'])} | " +
                 " | ".join(f"{lib.fmt_band(d.get(f'd0_ratio_to_{r}'))} | {lib.fmt_band(d.get(f'z0_ratio_to_{r}'))}" for r in refs) +
                 "".join(f" | {fn(arm, d)}" for _, fn in extra) + " |")
    return L


def gaussian_control(sc):
    """RMS on forced refits (inside the fixed truncation 5 x the 68 % width of "network, float"), noise / quantized - 1,
    prompt tracks per |eta| range; minor if |D| > max(3 %, 3 sd). 68 % differences on forced refits reported as shape."""
    out = {}
    W = lib.weights(sc.meta["n_events"])
    for o in ("y", "x"):
        for b in (4, 6):
            na, qa = f"noise_{o}_{b}b", f"down_{o}_{b}b"
            if not (sc.has(na) and sc.has(qa)):
                continue
            d = {}
            for q in ("d0", "z0"):
                cn, cq = sc.c(na), sc.c(qa)
                for i in range(3):
                    def rms(c):
                        s2, n = c[f"frms_{q}_s2"][i].astype(float), c[f"frms_{q}_n"][i].astype(float)
                        return np.sqrt(s2.sum() / n.sum()), np.sqrt((W @ s2) / (W @ n))
                    (pn, rn), (pq, rq) = rms(cn), rms(cq)
                    D, sd = pn / pq - 1, float((rn / rq).std())
                    d[f"{q}, {core.ETA_NAMES[i]}"] = {"rms_noise_um": float(pn), "rms_quantized_um": float(pq), "D": float(D), "sd": sd,
                                                     "minor_flag": bool(abs(D) > max(0.03, 3 * sd))}
                r68 = lib.ratio(sc, na, qa, q, forced=True, ref_forced=True)
                d[f"{q}, forced 68 % width, noise / quantized (shape effect)"] = r68
                d[f"{q}, free-running 68 % width, noise / quantized"] = lib.ratio(sc, na, qa, q)
            out[f"{o} step {step_um(b):g} um"] = d
    return out


def main():
    global REG
    REG = core.registry()
    lib.check_disk()
    prov = lib.provenance(core.SCENARIOS)
    res = {"provenance": prov, "scenarios": {}}
    for scen in core.SCENARIOS:
        sc = lib.Scenario(scen)
        if sc.meta is None or not sc.has("network_float"):
            continue
        S = {}
        L = [f"## {scen}: scans, allocations, controls", "",
             f"Run {prov['run_at_local']}. Arms analysed: {prov['scenarios'][scen]['arms_analysed']} of {prov['arms_in_registry']} "
             f"(complete on all five scan files). Prompt tracks unless stated: n = {sc.meta['origin_counts'][0]:,}. "
             "Widths = 68 % of |delivered − true| about zero in µm ± bootstrap sd; ratios with 68 % paired bootstrap bands. In-sample; the angles are placeholders.", ""]
        cells = read_cells(sc)
        # ---- one-at-a-time scans
        S["scans"], S["bits_needed"] = {}, {}
        for out in ("y", "x", "alpha", "beta"):
            pos = out in ("x", "y")
            bits_all = POS_BITS[out] if pos else ANG_BITS
            S["scans"][out] = {}
            for anchor in ("down", "up"):
                arms = {b: scan_arm(anchor, out, b) for b in bits_all}
                have = {b: a for b, a in arms.items() if sc.has(a)}
                missing = [a for b, a in arms.items() if not sc.has(a)]
                rows = {a: row(sc, a) for b, a in have.items()}
                for b, a in have.items():
                    rows[a]["bits"] = b
                    if pos:
                        rows[a].update(step_um=step_um(b), sub_area_bits=b, whole_sensor_bits=b + 2)
                    rows[a]["same_clusters_share"] = sc.stats(a)["same_clusters_share"]
                    rows[a]["hit_pull_rms"] = {k: v["rms_within_5"] for k, v in sc.stats(a)["correct_hit_pull"]["all surfaces"].items()}
                    rows[a]["track_pull_rms"] = {q: sc.stats(a)["track_pull"][q]["rms_within_5"] for q in ("d0", "z0")}
                    rows[a]["single_hit_layer1_p40"] = sc.stats(a)["single_hit_layer1_p40"]
                S["scans"][out][anchor] = {"points": rows, "missing": missing}
                if len(rows) > 1:
                    ttl = "others float (down from \"network, float\")" if anchor == "down" else "others at 4 bits (up from 4/4/4/4)"
                    L += [f"### {scen}: scan of {out}{' (bending angle)' if out == 'beta' else ''}, {ttl}", ""]
                    if missing:
                        L += [f"Missing (units not complete): {', '.join(missing)}", ""]
                    if pos:
                        ex = (("step [µm] / sub-area bits / whole-sensor bits", lambda a, d: f"{d['step_um']:.3g} / {d['sub_area_bits']} / {d['whole_sensor_bits']}"),)
                    else:
                        ex = (("bits", lambda a, d: "not sent" if d["bits"] == 0 else str(d["bits"])),)
                    ex += (("d0 / z0 pull RMS", lambda a, d: f"{d['track_pull_rms']['d0']:.2f} / {d['track_pull_rms']['z0']:.2f}"),)
                    L += md_rows(rows, extra=ex) + [""]
            # bits needed: down-scan, anchor "network, float"
            arms = {b: scan_arm("down", out, b) for b in bits_all}
            if all(sc.has(a) for a in arms.values()):
                bn = bits_needed(sc, arms, "network_float", READ[out], cells, not pos)
                S["bits_needed"][out] = {"quantity": READ[out], "anchor": "network, float", "base": "today's positions (measured)", **bn}
            else:
                S["bits_needed"][out] = {"reading": "not read: the down-scan is not complete", "missing": [a for a in arms.values() if not sc.has(a)]}
            S.setdefault("figures", []).extend([f"scan_{out}_{scen}"] if fig_scan(sc, out) else [])
            S["figures"].extend([f"scan_cells_{out}_{scen}"] if fig_scan_cells(sc, out) else [])
        # bending-angle-only family, anchor "bending-angle only, float"
        if sc.has("bending_float"):
            for kind, bits_all, qs in (("beta", [2, 3, 4, 5, 6, 7], ("d0",)), ("xy", [4, 5, 6, 7, 8], ("d0", "z0"))):
                arms = {b: bending_arm(kind, b) for b in bits_all}
                for q in qs:
                    key = f"bending-angle-only family, {'beta' if kind == 'beta' else 'x = y'} ({q})"
                    if all(sc.has(a) for a in arms.values()):
                        S["bits_needed"][key] = {"quantity": q, "anchor": "bending-angle only, float", "base": "today's positions (measured)",
                                                 **bits_needed(sc, arms, "bending_float", q, cells, False)}
                    else:
                        S["bits_needed"][key] = {"reading": "not read: the family is not complete", "missing": [a for a in arms.values() if not sc.has(a)]}
        L += [f"### {scen}: bits needed to stay within +10 % of the float arm (today's position base, measured)", "",
              "Rule: 68 % width only; in each replica the ratio curve is replaced by its running maximum from the finest value; the smallest passing bit count; "
              "quoted = 90 % quantile over replicas (median in brackets). Position bits are sub-area bits (whole sensor = +2). The design number holds for all its cells at once. "
              "The six-cell design number includes the cell |η| 1.8-2.4, pT ≥ 5 GeV with 995 tracks (5 below the 1,000 rule: minor deviation); the four-cell one pools the two forward |η| ranges as the plan's pooling order says.", "",
              "| output (quantity read) | prompt, all | design number, six prompt cells | design number, forward |η| ranges pooled (four cells) | " + " | ".join(n.replace("prompt, ", "") for n, *_ in cells[1:]) + " |",
              "|---|---|---|" + "---|" * (len(cells) - 1)]
        for out, bn in S["bits_needed"].items():
            if "cells" not in bn:
                L.append(f"| {out} | {bn['reading']} |" + " |" * len(cells))
                continue
            dn = bn["design_number_six_cells"]
            dns = dn.get("reading") or f"{dn['bits_90pct']} (median {dn['bits_median']})"
            dp = bn.get("design_number_forward_pooled")
            dns += " | " + (f"{dp['bits_90pct']} (median {dp['bits_median']})" if dp else "n/a")
            L.append(f"| {out} ({bn['quantity']}) | " + bn["cells"]["prompt, all"]["reading"] + f" | {dns} | " + " | ".join(bn["cells"][n]["reading"] for n, *_ in cells[1:]) + " |")
        nm = [f"- {out}, {c}: {t}" for out, bn in S["bits_needed"].items() if "cells" in bn for c, d in bn["cells"].items() for t in d.get("non_monotonic", [])]
        L += ["", "Non-monotonic steps (a finer value worse than the next coarser one by more than 2 paired sd; reported, not smoothed):", ""] + (nm or ["- none"]) + [""]
        # ---- bending family table
        fam = [bending_arm("beta", b) for b in (2, 3, 4, 5, 6, 7)] + [bending_arm("xy", b) for b in (4, 5, 7, 8)]
        fam = ["bending_float"] + fam + [a + "_globalrange" for a in fam[:6]] + ["alloc_6604_wholesensor"]
        rows = {a: row(sc, a, refs=("implemented_now", "network_float", "bending_float")) for a in fam if sc.has(a)}
        S["bending_family"] = {"points": rows, "missing": [a for a in fam if not sc.has(a)]}
        if len(rows) > 1:
            L += [f"### {scen}: bending-angle-only family (comparison only: whether a one-angle refit is implementable is doubtful)", ""]
            if S["bending_family"]["missing"]:
                L += ["Missing: " + ", ".join(S["bending_family"]["missing"]), ""]
            L += md_rows(rows, refs=("implemented_now", "bending_float")) + [""]
            if fig_bending(sc):
                S["figures"].append(f"bending_family_{scen}")
        # ---- 16-bit allocations with global-range twins
        have = [a for a in ALLOCS if sc.has(a)]
        twins = [a + "_globalrange" for a in ALLOCS if sc.has(a + "_globalrange")]
        rows = {a: row(sc, a) for a in have + twins}
        if have:
            for q in ("d0", "z0"):
                reps = np.stack([sc.w(a, q)[2] for a in have])
                best = np.bincount(reps.argmin(0), minlength=len(have)) / reps.shape[1]
                for a, b in zip(have, best):
                    rows[a][f"{q}_share_of_replicas_best"] = float(b)
        S["allocations_16_bits"] = {"points": rows, "missing": [a for a in ALLOCS if not sc.has(a)] + [a + "_globalrange" for a in ALLOCS[:-1] if not sc.has(a + "_globalrange")]}
        if have:
            L += [f"### {scen}: allocations with 16 bits in total (sub-area), per-link angle ranges, and their one-global-range twins", ""]
            if S["allocations_16_bits"]["missing"]:
                L += ["Missing: " + ", ".join(S["allocations_16_bits"]["missing"]), ""]
            ex = (("share of replicas best among the listed per-link allocations, d0 / z0", lambda a, d: f"{d['d0_share_of_replicas_best']:.2f} / {d['z0_share_of_replicas_best']:.2f}" if "d0_share_of_replicas_best" in d else "–"),)
            L += md_rows(rows, extra=ex) + [""]
        # ---- controls
        S["controls"] = {}
        ctl = []
        for base in ("alloc_4444", "alloc_6604", "network_float", "alloc_4444_wholesensor", "bending_float"):
            for tag, txt in (("_v3window", "v3 window unchanged"), ("_flagunusable", "flagged angles unusable"), ("_allsent", "all clusters sent (no cap of three per sub-area)"), ("_globalrange", "one global angle range")):
                a = base + tag
                if sc.has(a) and sc.has(base):
                    d = {"control": txt, "base": lib.PLAIN.get(base, base)}
                    for q in ("d0", "z0"):
                        d[q] = lib.val(*[sc.w(a, q)[i] for i in (0, 2)])
                        d[q + "_base"] = lib.val(*[sc.w(base, q)[i] for i in (0, 2)])
                        d[q + "_ratio_to_base"] = lib.ratio(sc, a, base, q)
                    d["correct_cluster_attached_innermost"] = lib.frac(sc, a, "inner_correct", "inner_den")
                    d["correct_cluster_attached_innermost_base"] = lib.frac(sc, base, "inner_correct", "inner_den")
                    S["controls"][a] = d
                    ctl.append(f"| {txt} | {d['base']} | {lib.fmt(d['d0'])} | {lib.fmt_band(d['d0_ratio_to_base'])} | {lib.fmt(d['z0'])} | {lib.fmt_band(d['z0_ratio_to_base'])} | "
                               f"{d['correct_cluster_attached_innermost']['value']:.4f} ({d['correct_cluster_attached_innermost_base']['value']:.4f}) |")
        if ctl:
            L += [f"### {scen}: controls (ratio = control ÷ its base arm, same tracks)", "",
                  "| control | base arm | d0 [µm] | d0 ratio | z0 [µm] | z0 ratio | correct cluster attached, innermost (base) |", "|---|---|---|---|---|---|---|"] + ctl + [""]
        g = gaussian_control(sc)
        S["controls"]["gaussian_noise_vs_quantization"] = g
        if g:
            L += [f"### {scen}: Gaussian noise in place of quantization (same single-hit variance), judged in RMS on forced refits", "",
                  "D = RMS(noise) ÷ RMS(quantized) − 1 on the clusters \"implemented now\" accepted, residuals inside 5 × the 68 % width of \"network, float\"; minor if |D| > max(3 %, 3 sd). "
                  "The 68 % width ratio is the expected shape effect (a uniform error has a 68 % width 1.18 × its RMS... of the added term only).", "",
                  "| point | quantity, cell | RMS noise [µm] | RMS quantized [µm] | D ± sd | flag |", "|---|---|---|---|---|---|"]
            for pt, d in g.items():
                for k, v in d.items():
                    if "D" in v:
                        L.append(f"| {pt} | {k} | {v['rms_noise_um']:.1f} | {v['rms_quantized_um']:.1f} | {v['D']:+.3f} ± {v['sd']:.3f} | {'minor' if v['minor_flag'] else ''} |")
                    else:
                        L.append(f"| {pt} | {k} | – | – | ratio {lib.fmt_band(v)} | |")
            L.append("")
        # ---- other origins
        S["other_origins"] = {}
        oarms = [a for a in FIXED + ALLOCS[1:] if sc.has(a)]
        names = sc.meta["cells"]["six"]["names"]
        ocells = [(n, "six", g, sc.meta["cells"]["six"]["counts"][g]) for g, n in enumerate(names) if not n.startswith("prompt")]
        ocells += [(f"{core.ORIGIN_NAMES[o]}, all pT", "origin", o, sc.meta["origin_counts"][o]) for o in (1, 2, 3)]
        for q in ("d0", "z0"):
            L += [f"### {scen}: b/c daughters and displaced tracks, {q} 68 % half-width [µm] (ratio to \"implemented now\" ± sd); |η| inclusive; per cell, not simultaneous", "",
                  "| arm | " + " | ".join(f"{n} (n = {k:,})" for n, _, _, k in ocells) + " |", "|---|" + "---|" * len(ocells)]
            for a in oarms:
                cellsout = []
                for n, part, g, k in ocells:
                    w, _, rp = sc.w(a, q, part, g)
                    r = lib.ratio(sc, a, "implemented_now", q, part, g)
                    S["other_origins"].setdefault(a, {"name": plain(a)}).setdefault(q, {})[n] = {"n": k, "w68_um": lib.val(w, rp), "ratio_to_implemented_now": r, "curve_allowed": k >= 200}
                    cellsout.append(f"{w:.1f} ({r['value']:.3f} ± {r['sd']:.3f})" + ("" if k >= 200 else " [fewer than 200 tracks]"))
                L.append(f"| {plain(a)} | " + " | ".join(cellsout) + " |")
            L.append("")
        # ---- secondary
        S["secondary"] = {}
        L += [f"### {scen}: secondary: separation of b/c-daughter tracks from prompt tracks (area under the curve), and prompt tails", "",
              "Differences are to \"implemented now\" with the paired bootstrap sd. The |d0| separation saturates on long-lived daughters, so it is also given for daughters with true |d0| < 200 µm.", "",
              "| arm | by \\|d0\\| | by \\|d0\\|/σ(d0) | by \\|d0\\|, true \\|d0\\| < 200 µm | by \\|d0\\|/σ(d0), true \\|d0\\| < 200 µm | prompt share with \\|d0 − true\\| > 100 µm |", "|---|---|---|---|---|---|"]
        for a in oarms:
            d = {"name": plain(a)}
            cs = []
            for k in ("abs_d0", "abs_d0_over_sigma", "abs_d0__true_below_200um", "abs_d0_over_sigma__true_below_200um"):
                rp, rr = sc.c(a)[f"auc_{k}_reps"].astype(float), sc.c("implemented_now")[f"auc_{k}_reps"].astype(float)
                pt, pr_ = sc.stats(a)["auc_bc_vs_prompt"][k], sc.stats("implemented_now")["auc_bc_vs_prompt"][k]
                d[k] = {"auc": lib.val(pt, rp), "difference_to_implemented_now": lib.val(pt - pr_, rp - rr)}
                cs.append(f"{pt:.4f} ({pt - pr_:+.4f} ± {(rp - rr).std():.4f})")
            t = lib.frac(sc, a, "tail100", "n")
            d["prompt_share_d0_off_by_more_than_100um"] = t
            S["secondary"][a] = d
            L.append(f"| {plain(a)} | " + " | ".join(cs) + f" | {t['value']:.4f} ± {t['sd']:.4f} |")
        L.append("")
        res["scenarios"][scen] = S
        open(lib.RESULTS / f"scans_{scen}.md", "w").write("\n".join(L) + "\n")
    json.dump(res, open(lib.RESULTS / "scans.json", "w"), indent=1)
    print("written", lib.RESULTS)


if __name__ == "__main__":
    main()
