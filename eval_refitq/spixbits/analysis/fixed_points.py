"""spixbits analysis, deliverable A: the fixed points per scenario (PLAN.md sections 5 and 6).

  python fixed_points.py     -> results/fixed_points.json, fixed_points_<scenario>.md, fixed_points_cross.md

Reads only the caches of core.py; run core.py first. Prompt tracks of the fixed population; widths are 68 % quantiles
of |delivered - true| in um; bands are 68 % paired event-bootstrap bands (1,000 replicas, same weights in every arm).
"""
import json

import numpy as np

import core
import lib

ARMS = ["implemented_now", "implemented_now_sizeblind", "implemented_now_newtable", "network_float", "alloc_4444",
        "alloc_4444_wholesensor", "alloc_6604", "alloc_6604_wholesensor", "bending_float"]
BITS = {"implemented_now": "F/F/F/F", "implemented_now_sizeblind": "F/F/F/F", "implemented_now_newtable": "F/F/F/F",
        "network_float": "F/F/F/F", "alloc_4444": "4/4/4/4", "alloc_4444_wholesensor": "4/4/4/4",
        "alloc_6604": "6/6/0/4", "alloc_6604_wholesensor": "6/6/0/4", "bending_float": "F/F/0/F"}
STEPS = {"alloc_4444": "150 / 150", "alloc_4444_wholesensor": "600 / 600", "alloc_6604": "37.5 / 37.5",
         "alloc_6604_wholesensor": "150 / 150"}


def arm_numbers(sc, arm):
    """All numbers of one arm on the prompt tracks (origin cell 0) and on the six design cells."""
    st = sc.stats(arm)
    d = {"allocation_x_y_alpha_beta": BITS.get(arm), "position_step_x_y_um": STEPS.get(arm, "float")}
    for q in ("d0", "z0"):
        w68, w95, reps = sc.w(arm, q)
        d[f"{q}_w68_um"] = lib.val(w68, reps)
        d[f"{q}_w95_um"] = w95
        d[f"{q}_ratio_to_implemented_now"] = lib.ratio(sc, arm, "implemented_now", q)
        d[f"{q}_ratio_to_network_float"] = lib.ratio(sc, arm, "network_float", q)
        # precision effect: the arm's measurements on the clusters "implemented now" accepted, over "implemented now";
        # hit-choice effect: free-running over forced (in capped arms: hit choice + cap availability)
        d[f"{q}_precision_effect"] = lib.ratio(sc, arm, "implemented_now", q, forced=True)
        d[f"{q}_hit_choice_effect"] = lib.ratio(sc, arm, arm, q, ref_forced=True)
        d[f"{q}_forced_w68_um"] = lib.val(*[sc.w(arm, q, forced=True)[i] for i in (0, 2)])
        cells = {}
        for g, name in enumerate(sc.meta["cells"]["six"]["names"]):
            if not name.startswith("prompt"):
                continue
            w, _, rp = sc.w(arm, q, "six", g)
            cells[name] = {"w68_um": lib.val(w, rp), "ratio_to_implemented_now": lib.ratio(sc, arm, "implemented_now", q, "six", g),
                           "ratio_to_network_float": lib.ratio(sc, arm, "network_float", q, "six", g)}
        d[f"{q}_six_cells"] = cells
    d["refit_fraction"] = lib.frac(sc, arm, "refit", "n")
    d["correct_cluster_attached_innermost"] = lib.frac(sc, arm, "inner_correct", "inner_den")
    d["innermost_detail"] = st["inner"]
    d["hits"] = st["hits"]
    d["same_clusters_share"] = st["same_clusters_share"]
    hp = st["correct_hit_pull"]["all surfaces"]
    d["hit_pull_rms"] = {k: hp[k]["rms_within_5"] for k in hp}
    d["hit_pull_rms_by_region"] = {reg: {k: v[k]["rms_within_5"] for k in v} for reg, v in st["correct_hit_pull"].items()}
    d["hit_pull_q68_over_0.9945"] = {k: hp[k]["q68_over_0.9945"] for k in hp}
    d["d0_pull_rms"] = st["track_pull"]["d0"]["rms_within_5"]
    d["z0_pull_rms"] = st["track_pull"]["z0"]["rms_within_5"]
    d["track_pull_detail"] = st["track_pull"]
    d["prompt_share_d0_off_by_more_than_100um"] = lib.frac(sc, arm, "tail100", "n")
    d["single_hit_layer1_p40"] = st["single_hit_layer1_p40"]
    return d


def flags(scen, res):
    """Things that look wrong, by the brief: an arm better than "implemented now" by more than its band; pull RMS
    outside 1 +- 0.15; a coarser fixed point better than a finer one."""
    out = []
    for arm, d in res.items():
        nm = lib.PLAIN[arm]
        for q in ("d0", "z0"):
            r = d[f"{q}_ratio_to_implemented_now"]
            if arm != "implemented_now" and r["hi68"] < 1.0:
                out.append(f"{scen}: {nm}: {q} is better than 'implemented now' by more than its 68 % band "
                           f"(ratio {lib.fmt_band(r)}; {(1 - r['value']) / max(r['sd'], 1e-9):.1f} sd)")
            for cname, c in d[f"{q}_six_cells"].items():
                r = c["ratio_to_implemented_now"]
                if arm != "implemented_now" and r["value"] + 2 * r["sd"] < 1.0:
                    out.append(f"{scen}: {nm}: {q} better than 'implemented now' by more than 2 sd in {cname} (ratio {lib.fmt_band(r)})")
            p = d[f"{q}_pull_rms"]
            if abs(p - 1) > 0.15:
                out.append(f"{scen}: {nm}: {q} pull RMS {p:.2f} outside 1 ± 0.15")
        for k, p in d["hit_pull_rms"].items():
            if abs(p - 1) > 0.15:
                out.append(f"{scen}: {nm}: correct-hit {k} pull RMS {p:.2f} outside 1 ± 0.15 (all surfaces)")
    for fine, coarse in (("alloc_4444", "alloc_4444_wholesensor"), ("alloc_6604", "alloc_6604_wholesensor"), ("network_float", "alloc_4444"),
                         ("bending_float", "alloc_6604")):
        if fine in res and coarse in res:
            for q in ("d0", "z0"):
                a, b = sc_w(res, fine, q), sc_w(res, coarse, q)
                if b < a:
                    out.append(f"{scen}: non-monotonic: {lib.PLAIN[coarse]} has a smaller {q} width ({b:.1f} um) than {lib.PLAIN[fine]} ({a:.1f} um)")
    return out


def sc_w(res, arm, q):
    return res[arm][f"{q}_w68_um"]["value"]


def table(scen, sc, res, ot):
    L = [f"### {scen}: fixed points, prompt tracks (n = {sc.meta['origin_counts'][0]:,} of {sc.meta['n_tracks']:,} in the population; {sc.meta['n_events']} events)", ""]
    L += ["Widths: 68 % of |delivered − true| about zero, µm, ± bootstrap sd. Ratios with 68 % paired bootstrap bands. "
          "x = along the 50 µm pitch (z in the barrel, drives z0); y = along the 12.5 µm pitch (r-φ, drives d0); β = bending angle.", ""]
    L += ["| quantization scenario | x/y/α/β bits | d0 [µm] | z0 [µm] | d0 ÷ implemented now | z0 ÷ implemented now | d0 ÷ network, float | z0 ÷ network, float | refit fraction |",
          "|---|---|---|---|---|---|---|---|---|"]
    for arm, d in res.items():
        L.append(f"| {lib.PLAIN[arm]} | {d['allocation_x_y_alpha_beta']} | {lib.fmt(d['d0_w68_um'])} | {lib.fmt(d['z0_w68_um'])} | "
                 f"{lib.fmt_band(d['d0_ratio_to_implemented_now'])} | {lib.fmt_band(d['z0_ratio_to_implemented_now'])} | "
                 f"{lib.fmt_band(d['d0_ratio_to_network_float'])} | {lib.fmt_band(d['z0_ratio_to_network_float'])} | {d['refit_fraction']['value']:.4f} |")
    L += [f"| outer tracker only (scale; its d0 is not a measurement) | – | – | {lib.fmt(ot['z0'])} | – | – | – | – | – |", ""]
    L += ["Why it moved, and honesty of the errors. Precision effect = the quantization scenario's measurements and errors on the clusters "
          "\"implemented now\" accepted, ÷ \"implemented now\". Hit-choice effect = free-running ÷ that (in quantization scenarios with the cap of "
          "three clusters per sub-area it includes the cap). Attachment = share of prompt tracks with a crossing cluster on their "
          "innermost visited surface that accepted it. Pull RMS within ±5; hit pulls are the innovation pulls of accepted crossing clusters, all surfaces.", ""]
    L += ["| quantization scenario | d0 precision | d0 hit choice | z0 precision | z0 hit choice | correct cluster attached, innermost | same clusters as implemented now | hit pull RMS y / x / β / α | d0 pull RMS | z0 pull RMS |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for arm, d in res.items():
        hp = d["hit_pull_rms"]
        hps = " / ".join(f"{hp[k]:.2f}" if k in hp else "not used" for k in core.HITQ)
        L.append(f"| {lib.PLAIN[arm]} | {lib.fmt_band(d['d0_precision_effect'])} | {lib.fmt_band(d['d0_hit_choice_effect'])} | "
                 f"{lib.fmt_band(d['z0_precision_effect'])} | {lib.fmt_band(d['z0_hit_choice_effect'])} | "
                 f"{lib.fmt(d['correct_cluster_attached_innermost'], 4)} | {d['same_clusters_share']['implemented_now']:.3f} | {hps} | {d['d0_pull_rms']:.2f} | {d['z0_pull_rms']:.2f} |")
    L += ["", "By cell (six prompt cells; counts " + ", ".join(f"{c:,}" for n, c in zip(sc.meta['cells']['six']['names'], sc.meta['cells']['six']['counts']) if n.startswith('prompt')) + "): width in µm and, in brackets, ratio to \"implemented now\".", ""]
    for q in ("d0", "z0"):
        names = [n for n in sc.meta["cells"]["six"]["names"] if n.startswith("prompt")]
        L += [f"| {q} [µm] | " + " | ".join(n.replace("prompt, ", "") for n in names) + " |", "|---|" + "---|" * len(names)]
        for arm, d in res.items():
            L.append(f"| {lib.PLAIN[arm]} | " + " | ".join(
                f"{d[f'{q}_six_cells'][n]['w68_um']['value']:.1f} ({d[f'{q}_six_cells'][n]['ratio_to_implemented_now']['value']:.3f} ± {d[f'{q}_six_cells'][n]['ratio_to_implemented_now']['sd']:.3f})" for n in names) + " |")
        L.append("")
    return L


def main():
    lib.check_disk()
    lib.RESULTS.mkdir(parents=True, exist_ok=True)
    allres, allflags, cross = {}, [], []
    prov = lib.provenance(core.SCENARIOS)
    for scen in core.SCENARIOS:
        sc = lib.Scenario(scen)
        if sc.meta is None or not sc.has("implemented_now") or not sc.has("network_float"):
            continue
        res = {arm: arm_numbers(sc, arm) for arm in ARMS if sc.has(arm)}
        ot = {q: lib.val(*[sc.w(core.OT_ARM, q)[i] for i in (0, 2)]) for q in ("d0", "z0")}
        fl = flags(scen, res)
        allflags += fl
        allres[scen] = {"population": sc.meta, "quantization_scenarios": res, "outer_tracker_only_z0_w68_um": ot["z0"], "flags": fl}
        L = table(scen, sc, res, ot)
        L += ["Flags:", ""] + ([f"- {f}" for f in fl] or ["- none"]) + [""]
        open(lib.RESULTS / f"fixed_points_{scen}.md", "w").write("\n".join(L))
        cross.append((scen, res))
    json.dump({"provenance": prov, "detector_scenarios": allres}, open(lib.RESULTS / "fixed_points.json", "w"), indent=1)
    L = ["### Fixed points across detector scenarios, prompt tracks: 68 % half-width in µm (ratio to \"implemented now\" of the same detector scenario ± bootstrap sd)", "",
         "| quantization scenario | x/y/α/β bits | " + " | ".join(f"{s} d0" for s, _ in cross) + " | " + " | ".join(f"{s} z0" for s, _ in cross) + " |",
         "|---|---|" + "---|" * (2 * len(cross))]
    for arm in ARMS:
        if not all(arm in r for _, r in cross):
            continue
        cells = []
        for q in ("d0", "z0"):
            for _, r in cross:
                d = r[arm]
                rr = d[f"{q}_ratio_to_implemented_now"]
                cells.append(f"{d[f'{q}_w68_um']['value']:.1f} ({rr['value']:.3f} ± {rr['sd']:.3f})")
        L.append(f"| {lib.PLAIN[arm]} | {BITS[arm]} | " + " | ".join(cells) + " |")
    L += ["", f"Run {prov['run_at_local']}; scan files {prov['scan_files']}; in-sample; the angles are placeholders."]
    open(lib.RESULTS / "fixed_points_cross.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("\nFLAGS\n" + "\n".join(allflags))


if __name__ == "__main__":
    main()
