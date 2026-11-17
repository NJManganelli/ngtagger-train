"""spixbits analysis, deliverable D: the README draft, assembled from the result files of the other steps.

  python readme.py     -> ~/smartpixels/doc-archive/spix-bits-refit-study/README.md

Conclusions are filled from the numbers in results/*.json, so the draft follows whatever units were complete; parts
whose arms are missing say so. Local file only.
"""
import json

import core
import lib

OUT = lib.STUDY / ("README.md" if not core.TAG else f"README_{core.TAG}.md")
REL = "results" if not core.TAG else f"results/{core.TAG}"


def sections(path):
    """{heading: lines} of a results markdown file, split at '### '."""
    out, cur = {}, None
    if not path.exists():
        return out
    for line in open(path).read().split("\n"):
        if line.startswith("### "):
            cur = line[4:]
            out[cur] = []
        elif cur is not None:
            out[cur].append(line)
    return out


def pick(secs, key):
    for h, body in secs.items():
        if key in h:
            return ["#### " + h] + body
    return []


def r3(v):
    return f"{v['value']:.2f}"


def conclusions(fp, sn, nf):
    C = []
    S = fp["detector_scenarios"]
    scs = list(S)

    def per(fn):
        return "; ".join(f"{s}: {fn(S[s]['quantization_scenarios'])}" for s in scs)

    def w(arm, q):
        return lambda A: f"{A['implemented_now'][q + '_w68_um']['value']:.0f} → {A[arm][q + '_w68_um']['value']:.0f} µm (× {r3(A[arm][q + '_ratio_to_implemented_now'])})" if arm in A else "missing"

    C.append("**What a coarse cluster costs (prompt tracks, 68 % half-width, against what is implemented now).**")
    C.append(f"- 16-bit cluster 4/4/4/4 with positions inside the 2.4 mm sub-area (150 µm steps): d0 {per(w('alloc_4444', 'd0'))}. z0 {per(w('alloc_4444', 'z0'))}.")
    C.append(f"- The same 4/4/4/4 over the whole 9.6 mm sensor (600 µm steps): d0 {per(w('alloc_4444_wholesensor', 'd0'))}. z0 {per(w('alloc_4444_wholesensor', 'z0'))}.")
    C.append(f"- Bending-angle only 6/6/0/4 inside the sub-area (37.5 µm steps, no α): d0 {per(w('alloc_6604', 'd0'))}. z0 {per(w('alloc_6604', 'z0'))}. "
             f"Over the whole sensor (150 µm steps): d0 {per(w('alloc_6604_wholesensor', 'd0'))}.")
    C.append("- The cost is largest where today's resolution is best: central tracks with pT ≥ 5 GeV lose the largest factor (cell tables below); "
             "in IAAA, where today's d0 width is already " + f"{S['IAAA']['quantization_scenarios']['implemented_now']['d0_w68_um']['value'] / S['AAAA']['quantization_scenarios']['implemented_now']['d0_w68_um']['value']:.1f}" + " × that of AAAA, the same steps cost a smaller factor." if "IAAA" in S and "AAAA" in S else "")
    A = S[scs[0]]["quantization_scenarios"]
    C.append("")
    C.append("**Why it moves: precision, not hit choice.**")
    C.append(f"- With the clusters held to those \"implemented now\" accepted, 4/4/4/4 already gives × {r3(A['alloc_4444']['d0_precision_effect'])} in d0 ({scs[0]}); letting the refit choose its own clusters adds × {A['alloc_4444']['d0_hit_choice_effect']['value']:.3f}. "
             f"On the whole sensor the hit-choice part is × {A['alloc_4444_wholesensor']['d0_hit_choice_effect']['value']:.3f} and the wrong-cluster share on the innermost surface rises from "
             f"{100 * A['implemented_now']['innermost_detail']['wrong_cluster']:.1f} % to {100 * A['alloc_4444_wholesensor']['innermost_detail']['wrong_cluster']:.1f} %.")
    C.append(f"- Coarser positions do not lose the correct cluster more often: the search window is widened by the quantization variance, and the correct cluster on the innermost surface is attached on "
             f"{100 * A['network_float']['correct_cluster_attached_innermost']['value']:.1f} % of tracks with float positions and {100 * A['alloc_4444']['correct_cluster_attached_innermost']['value']:.1f} % at 4/4/4/4 ({scs[0]}).")
    C.append("")
    C.append("**The network's angles at float precision change little.**")
    C.append(f"- \"network, float\" against \"implemented now\": d0 {per(lambda A: '× ' + r3(A['network_float']['d0_ratio_to_implemented_now']))}; z0 {per(lambda A: '× ' + r3(A['network_float']['z0_ratio_to_implemented_now']))}. "
             f"In {scs[0]} the position error without cluster size alone gives × {A['implemented_now_sizeblind']['d0_ratio_to_implemented_now']['value']:.3f} in d0, and the cap of three clusters per sub-area "
             f"removes {100 * S[scs[0]]['population']['correct_cluster_dropped_by_cap_per_surface']['1'][0]:.1f} % of the correct clusters on barrel layer 1. The new angle-table recipe alone changes nothing measurable "
             f"(× {A['implemented_now_newtable']['d0_ratio_to_implemented_now']['value']:.3f}).")
    C.append(f"- Dropping α at float precision (bending-angle only, float): d0 {per(lambda A: '× ' + f'{A['bending_float']['d0_ratio_to_network_float']['value']:.3f}')} against \"network, float\". "
             "These angles are placeholders (the network on barrel and discs, a CNN stand-in for clusters without angle truth): this is reported, not a statement about the real sensor.")
    # scans
    C.append("")
    C.append("**One output at a time (bits needed to stay within +10 % of \"network, float\", today's position base).**")
    any_bits = False
    for out, txt in (("y", "y (12.5 µm pitch, read on d0)"), ("x", "x (50 µm pitch, read on z0)"), ("beta", "β, the bending angle (read on d0)"), ("alpha", "α (read on z0)")):
        parts = []
        for s in scs:
            bn = sn["detector_scenarios"].get(s, {}).get("bits_needed", {}).get(out, {})
            if "cells" in bn:
                any_bits = True
                dn, dp = bn["design_number_six_cells"], bn.get("design_number_forward_pooled")
                parts.append(f"{s}: prompt total {bn['cells']['prompt, all']['reading']}; design number over six cells {dn.get('bits_90pct', dn.get('reading'))}"
                             + (f", with the forward |η| ranges pooled {dp['bits_90pct']}" if dp else ""))
            else:
                parts.append(f"{s}: not yet available")
        C.append(f"- {txt}: " + "; ".join(parts) + ("." if out in ("alpha", "beta") else " (sub-area bits; whole sensor = +2)."))
    C.append("- Reading of the angle rows: with float positions, the 68 % widths of prompt tracks move by well under 1 % whether α or β is sent at 2-7 bits or not at all, so the +10 % rule passes even for \"not sent\". "
             "This is a statement about this replay and this quantity only: the angles are placeholders, the skip prices are not re-derived, and the widths of prompt tracks are not where angles are expected to act "
             "(hit choice, tails, displaced tracks and the track-MVA heads; the scan tables give attachment, wrong-cluster shares and tails per point). It is reported, not a case for dropping an angle.")
    if not any_bits:
        C.append("- No down-scan was complete on all five files when this draft was made; the curves below show the points that exist.")
    # y curve sentence from whatever exists
    try:
        pts = sn["detector_scenarios"][scs[0]]["scans"]["y"]["down"]["points"]
        if pts:
            C.append(f"- y down-scan, {scs[0]}, d0 ÷ \"network, float\": " + ", ".join(f"{p['step_um']:.3g} µm × {p['d0_ratio_to_network_float']['value']:.2f}" for p in pts.values()) + ".")
        pts = sn["detector_scenarios"][scs[0]]["scans"]["x"]["down"]["points"]
        if pts:
            C.append(f"- x down-scan, {scs[0]}, z0 ÷ \"network, float\": " + ", ".join(f"{p['step_um']:.3g} µm × {p['z0_ratio_to_network_float']['value']:.2f}" for p in pts.values()) + ".")
        for o, q in (("beta", "d0"), ("alpha", "z0")):
            pts = sn["detector_scenarios"][scs[0]]["scans"][o]["down"]["points"]
            if pts:
                C.append(f"- {o} down-scan, {scs[0]}, {q} ÷ \"network, float\": " + ", ".join(f"{'not sent' if p['bits'] == 0 else str(p['bits']) + ' bits'} × {p[q + '_ratio_to_network_float']['value']:.3f}" for p in pts.values()) + ".")
    except KeyError:
        pass
    C.append("")
    endp = any(sn["detector_scenarios"].get(s, {}).get("bits_needed", {}).get("up from 4/4/4/4: y", {}).get("reference_is_unquantized_end_point") for s in scs)
    C.append("**Up from the 4/4/4/4 floor, one output at a time towards the unquantized limit, the other three held at 4 bits (bits needed to stay within +10 % of "
             + ("the unquantized end point of the same output" if endp else "the finest scanned value of the same output; this scan has no unquantized end point") + ").**")
    for out, txt in (("y", "y (read on d0)"), ("x", "x (read on z0)"), ("beta", "β (read on d0)"), ("alpha", "α (read on z0)")):
        parts = []
        for s in scs:
            bn = sn["detector_scenarios"].get(s, {}).get("bits_needed", {}).get(f"up from 4/4/4/4: {out}", {})
            if "cells" in bn:
                dn = bn["design_number_six_cells"]
                parts.append(f"{s}: prompt total {bn['cells']['prompt, all']['reading']}; design number over six cells {dn.get('bits_90pct', dn.get('reading'))}")
            else:
                parts.append(f"{s}: not yet available")
        C.append(f"- {txt}: " + "; ".join(parts) + ".")
    try:
        for o, q in (("y", "d0"), ("x", "z0"), ("beta", "d0"), ("alpha", "z0")):
            pts = sn["detector_scenarios"][scs[0]]["scans"][o]["up"]["points"]
            if len(pts) > 1:
                lab = (lambda p: f"{p['step_um']:.3g} µm") if o in ("x", "y") else (lambda p: "not sent" if p["bits"] == 0 else f"{p['bits']} bits")
                lab2 = lambda p: "unquantized" if p["bits"] == 99 else lab(p)
                C.append(f"- {o} up-scan, {scs[0]}, {q} in µm: " + ", ".join(f"{lab2(p)}: {p[q]['value']:.1f}" for p in sorted(pts.values(), key=lambda p: p["bits"])) + ".")
    except KeyError:
        pass
    C.append("")
    C.append("**Bending-angle-only family (from 6/6/0/4; comparison only, a one-angle refit may not be implementable).**")
    for s in scs:
        bn = sn["detector_scenarios"].get(s, {}).get("bits_needed", {})
        fam = sn["detector_scenarios"].get(s, {}).get("bending_family", {}).get("points", {})
        kb, kd, kz = "bending-angle-only family, beta (d0)", "bending-angle-only family, x = y (d0)", "bending-angle-only family, x = y (z0)"
        if all("cells" in bn.get(k, {}) for k in (kb, kd, kz)):
            C.append(f"- {s}: bits needed against \"bending-angle only, float\": β (x = y at 6 bits) {bn[kb]['cells']['prompt, all']['reading']}; x = y (β at 4 bits) for d0 {bn[kd]['cells']['prompt, all']['reading']}, for z0 {bn[kz]['cells']['prompt, all']['reading']}. "
                     + "d0 by β bits: " + ", ".join(f"{b}: {fam[a]['d0']['value']:.1f} µm" for b, a in ((2, 'bending_beta_2b'), (3, 'bending_beta_3b'), (4, 'alloc_6604'), (5, 'bending_beta_5b'), (6, 'bending_beta_6b'), (7, 'bending_beta_7b')) if a in fam)
                     + "; d0 by x = y bits: " + ", ".join(f"{b}: {fam[a]['d0']['value']:.1f} µm" for b, a in ((4, 'bending_xy_4b'), (5, 'bending_xy_5b'), (6, 'alloc_6604'), (7, 'bending_xy_7b'), (8, 'bending_xy_8b')) if a in fam) + ".")
        else:
            C.append(f"- {s}: not yet available.")
    C.append("- The β reading is taken against a reference with float positions, so \"more than 7 bits\" there measures the 37.5 µm position steps of the family, which no number of β bits can recover; the d0 values by β bits show what β itself changes.")
    C.append("")
    C.append("**Controls.**")
    for s in scs:
        ct = sn["detector_scenarios"].get(s, {}).get("controls", {})
        def cr(a, q="d0"):
            return f"× {ct[a][q + '_ratio_to_base']['value']:.3f}" if a in ct else "missing"
        C.append(f"- {s}, d0 (z0) of the control ÷ its base quantization scenario: v3 window unchanged at 4/4/4/4 {cr('alloc_4444_v3window')} ({cr('alloc_4444_v3window', 'z0')}), at 6/6/0/4 {cr('alloc_6604_v3window')} ({cr('alloc_6604_v3window', 'z0')}); "
                 f"flagged angles unusable at 4/4/4/4 {cr('alloc_4444_flagunusable')} ({cr('alloc_4444_flagunusable', 'z0')}), at 6/6/0/4 {cr('alloc_6604_flagunusable')} ({cr('alloc_6604_flagunusable', 'z0')}); "
                 f"all clusters sent: network, float {cr('network_float_allsent')} ({cr('network_float_allsent', 'z0')}), 4/4/4/4 {cr('alloc_4444_allsent')} ({cr('alloc_4444_allsent', 'z0')}), 4/4/4/4 whole sensor {cr('alloc_4444_wholesensor_allsent')} ({cr('alloc_4444_wholesensor_allsent', 'z0')}); "
                 f"one global angle range at 4/4/4/4 {cr('alloc_4444_globalrange')} ({cr('alloc_4444_globalrange', 'z0')}); no angles at all with float positions ÷ network, float {cr('positions_only_float')} ({cr('positions_only_float', 'z0')}).")
        g = ct.get("gaussian_noise_vs_quantization", {})
        if g:
            fl = [f"{pt}, {k}: D = {v['D']:+.3f} ± {v['sd']:.3f}" for pt, d in g.items() for k, v in d.items() if "D" in v and v["minor_flag"]]
            C.append(f"- {s}, Gaussian noise against quantization at equal single-hit variance (RMS on forced refits): " + ("no comparison beyond max(3 %, 3 sd)." if not fl else "minor flags: " + "; ".join(fl) + "."))
    C.append("")
    C.append("**b/c daughters with pT ≥ 5 GeV: is the d0 loss of \"network, float\" the cap of three clusters per sub-area?**")
    for s in scs:
        pz = sn["detector_scenarios"].get(s, {}).get("bc_daughters_and_the_cap", {})
        c = "b/c daughters, pT >= 5 GeV"
        if "network_float_allsent" in pz:
            f = lambda a, k="ratio_to_implemented_now": lib.fmt_band(pz[a][c][k])
            C.append(f"- {s}: d0 ÷ \"implemented now\" in that cell: network, float {f('network_float')} ({pz['network_float'][c]['d0_w68_um']['value']:.1f} µm against {pz['implemented_now'][c]['d0_w68_um']['value']:.1f} µm); "
                     f"network, float with all clusters sent {f('network_float_allsent')}; implemented now, size-blind {f('implemented_now_sizeblind')}; network, float on forced clusters {pz['network_float'][c]['forced_ratio_to_implemented_now']['value']:.3f}. "
                     f"All sent ÷ capped: {lib.fmt_band(pz['all_sent_over_capped']['network_float'][c])}."
                     + (f" Cap at random ÷ this scan's cap: {lib.fmt_band(pz['random_cap_over_capped']['network_float'][c])}; network, float with the cap at random {f('network_float_caprandom')}." if "random_cap_over_capped" in pz else "")
                     + f" For pT 2-5 GeV: network, float {lib.fmt_band(pz['network_float']['b/c daughters, pT 2-5 GeV']['ratio_to_implemented_now'])}, all clusters sent {lib.fmt_band(pz['network_float_allsent']['b/c daughters, pT 2-5 GeV']['ratio_to_implemented_now'])}.")
    # allocations
    C.append("")
    C.append("**Allocations with 16 bits.**")
    done = False
    for s in scs:
        al = sn["detector_scenarios"].get(s, {}).get("allocations_16_bits", {})
        pts = {k: v for k, v in al.get("points", {}).items() if not k.endswith("_globalrange")}
        if len(pts) >= 11:
            done = True
            bd = min(pts.values(), key=lambda v: v["d0"]["value"])
            bz = min(pts.values(), key=lambda v: v["z0"]["value"])
            C.append(f"- {s}: smallest d0 width: {bd['name']} ({bd['d0']['value']:.1f} µm, × {r3(bd['d0_ratio_to_implemented_now'])} of implemented now; best in {100 * bd['d0_share_of_replicas_best']:.0f} % of replicas); "
                     f"smallest z0 width: {bz['name']} ({bz['z0']['value']:.1f} µm, × {r3(bz['z0_ratio_to_implemented_now'])}; best in {100 * bz['z0_share_of_replicas_best']:.0f} % of replicas).")
    for s in scs:
        al = sn["detector_scenarios"].get(s, {}).get("allocations_16_bits", {}).get("points", {})
        tw = [(k, al[k[:-12]], v) for k, v in al.items() if k.endswith("_globalrange") and k[:-12] in al]
        if tw:
            dd = [(v["d0"]["value"] / b["d0"]["value"], v["z0"]["value"] / b["z0"]["value"], b["name"]) for k, b, v in tw]
            C.append(f"- {s}, one global angle range ÷ per-link ranges over the {len(tw)} allocations with angles: d0 {min(x[0] for x in dd):.3f} to {max(x[0] for x in dd):.3f}, z0 {min(x[1] for x in dd):.3f} to {max(x[1] for x in dd):.3f} "
                     f"(largest d0 change: {max(dd, key=lambda x: abs(x[0] - 1))[2]}).")
    if not done:
        C.append("- The eleven-allocation table is not complete yet; of the two that exist, 6/6/0/4 has the smaller d0 and z0 widths in every detector scenario (fixed-point table).")
    # near full
    C.append("")
    C.append("**Near full resolution (the network's positions are better than today's, which cannot be emulated by adding noise).**")
    for s in scs:
        d = nf["detector_scenarios"].get(s, {})
        if d:
            C.append(f"- {s}: d0 against y: {d['y']['cells']['prompt, all']['reading']}. z0 against x: {d['x']['cells']['prompt, all']['reading']}.")
    C.append("")
    C.append("**Honesty of the errors.**")
    C.append("- Correct-hit pull RMS is inside 1 ± 0.15 in every fixed-point quantization scenario with all surfaces pooled; on the discs the angle pulls of \"implemented now\" are " + f"{A['implemented_now']['hit_pull_rms_by_region']['discs']['beta']:.2f} (β) and {A['implemented_now']['hit_pull_rms_by_region']['discs']['alpha']:.2f} (α) in {scs[0]}" + ", outside the range (with the network's angles: " + f"{A['network_float']['hit_pull_rms_by_region']['discs']['beta']:.2f} and {A['network_float']['hit_pull_rms_by_region']['discs']['alpha']:.2f}" + "). "
             "The d0 and z0 pull RMS are outside 1 ± 0.15 already in \"implemented now\" (" + per(lambda A: f"d0 {A['implemented_now']['d0_pull_rms']:.2f}, z0 {A['implemented_now']['z0_pull_rms']:.2f}") +
             ") and every quantization scenario inherits this; by the letter of the plan every one carries the flag, so the flag does not single one out.")
    return C


def headline(fp, sn):
    S = fp["detector_scenarios"]
    scs = list(S)
    L = ["Prompt tracks, 68 % half-width in µm (ratio to \"implemented now\" of the same detector scenario). In-sample; placeholder angles.", "",
         "| quantization scenario (x/y/α/β bits) | " + " | ".join(f"{s} d0" for s in scs) + " | " + " | ".join(f"{s} z0" for s in scs) + " |", "|---|" + "---|" * (2 * len(scs))]
    rows = [("implemented_now", None), ("network_float", None), ("alloc_4444", None), ("alloc_4444_wholesensor", None), ("alloc_6604", None), ("alloc_6604_wholesensor", None), ("bending_float", None)]
    for arm, _ in rows:
        cs = []
        for q in ("d0", "z0"):
            for s in scs:
                d = S[s]["quantization_scenarios"].get(arm)
                cs.append(f"{d[q + '_w68_um']['value']:.1f} ({d[q + '_ratio_to_implemented_now']['value']:.2f})" if d else "missing")
        L.append(f"| {lib.PLAIN[arm]} ({S[scs[0]]['quantization_scenarios'][arm]['allocation_x_y_alpha_beta']}) | " + " | ".join(cs) + " |")
    # best 16-bit allocations
    for s in scs:
        pts = {k: v for k, v in sn["detector_scenarios"].get(s, {}).get("allocations_16_bits", {}).get("points", {}).items() if not k.endswith("_globalrange")}
        if len(pts) >= 11:
            o = sorted(pts.values(), key=lambda v: v["d0"]["value"])
            L += ["", f"{s}, the eleven 16-bit allocations ordered by d0 width (d0 / z0 in µm): " + "; ".join(f"{v['name'].replace('alpha', 'α').replace('beta', 'β')}: {v['d0']['value']:.1f} / {v['z0']['value']:.1f}" for v in o) + "."]
    L += ["", "Bits needed to stay within +10 % of \"network, float\", one output at a time with the others float (sub-area bits for positions; whole sensor = +2): prompt total / design number over six prompt cells.", "",
          "| output (read on) | " + " | ".join(scs) + " |", "|---|" + "---|" * len(scs)]
    for out, txt in (("y", "y (d0)"), ("x", "x (z0)"), ("beta", "β, bending angle (d0)"), ("alpha", "α (z0)")):
        cs = []
        for s in scs:
            bn = sn["detector_scenarios"].get(s, {}).get("bits_needed", {}).get(out, {})
            cs.append(f"{bn['cells']['prompt, all']['reading']} / {bn['design_number_six_cells'].get('bits_90pct', 'n/a')}" if "cells" in bn else "not available")
        L.append(f"| {txt} | " + " | ".join(cs) + " |")
    return L


LIMITS = """- The angles are placeholders: the barrel-trained PixelAV network stands in on the discs as well; clusters without angle truth keep the CNN stand-in, judged with the network's narrower errors. Nothing here is a statement about the angle quality of the real sensor.
- A particle's secondary clusters (delta rays, fragments) carry the parent's true angle plus a residual, so no angle can reject them here; a real network would not report the parent's direction for them.
- The simulated sensors are 25 × 100 µm, 150 µm thick; the SmartPixels sensor is 12.5 × 50 µm, 100 µm thick. Occupancy and merging are those of the simulated sensor.
- At most three clusters per 2.4 mm sub-area are sent, CAPRULE; clusters straddling a sub-area edge are not modelled.
- Position and angle residuals of one cluster are uncorrelated here (they are correlated in the network).
- Skip prices, seed model and material scale are today's in every quantization scenario; the one bound on the skip price was not run.
- v3's window already loses correct clusters in every quantization scenario; quantization scenarios with quantized positions widen it by the quantization variance and recover some.
- Results are in-sample with respect to the v3 calibrations, the angle ranges and the angle table (500 scan events; ranges and tables fitted on all ten files).
- Positions are built on today's rec-hit positions: near full resolution the measured numbers are an upper bound and the network's own position resolution enters only through the model reading.
- Whether a refit with one angle (the bending-angle-only quantization scenarios) is implementable in a real system is doubtful; they are for comparison only.
- Prompt track collection only: the extended (displaced-capable) collection the taggers read was not run, so the displaced-track rows are those of the prompt collection.
- The d0-sign gate was dropped by the physicist (2026-11-11) as redundant with the first check; a same-sign share is a sanity number only.
- Nothing downstream of the track is rebuilt: no particle flow, jets or tagger numbers.
- First check, accepted by the physicist in the form "cluster choice with cluster-table inputs, arithmetic with CMSSW float positions": the nano's cluster table keeps 16 mantissa bits of the positions, which leaves one window-edge disagreement and an effect of at most 0.026 σ on the state, common to every quantization scenario.
- Forced refits use the clusters "implemented now" accepted even where the cap of three would not have sent them; in capped quantization scenarios the hit-choice effect therefore includes the cap."""

NOT_RUN = """- Skip price re-derived at the 16-bit point (needs the cut fit; a separate run).
- The direct network-position point above 50 GeV (helix + drawn network position residual).
- The extended (displaced-capable) track collection in the replay (everything here is the prompt collection).
- The track-MVA heads (deployed heads scored, heads retrained per quantization scenario).
- The parametrization (correctionlib file per detector scenario, PLAN 7b).
- Bits-needed readings in the 12-cell split (only the prompt total, the six design cells and the pooled forward cells are read); the "both refit" view (the refit fraction is 0.997-1.000 in every fixed-point quantization scenario).
- Stale or pending launch checks of the implementation (planted defects, bit-equal rerun of a unit): see ~/smartpixels/scratch_local/spixbits/RESUME.md."""


def partial_note(prov):
    """A line that marks the write-up as partial and names the groups of the registry that are not complete."""
    reg = core.registry()
    pend = {}
    for s, v in prov["detector_scenarios"].items():
        miss = set(v["quantization_scenarios_missing"]) | set(v["quantization_scenarios_partial"])
        for a in reg:
            if a["name"] in miss:
                pend.setdefault(a["group"], set()).add(s)
    names = {"scan_down": "down-scans", "scan_up": "up-scans", "controls": "controls", "bending": "bending-angle-only family", "allocations": "16-bit allocations", "fixed": "fixed points", "priority": "anchors"}
    if not pend:
        return []
    return ["**PARTIAL WRITE-UP.** Units directory: `" + str(core.UNITS) + "`. Pending (not complete on all five files): "
            + "; ".join(f"{names.get(g, g)} ({', '.join(sorted(ss))})" for g, ss in pend.items()) + ". Sections that need them say \"not yet available\" or \"missing\".", ""]


def main():
    lib.check_disk()
    fp = json.load(open(lib.RESULTS / "fixed_points.json"))
    sn = json.load(open(lib.RESULTS / "scans.json"))
    nf = json.load(open(lib.RESULTS / "near_full.json"))
    prov = sn["provenance"]
    final = core.TAG == "ptcap"
    L = ["# SmartPixels output bits and the L1 track refit: results" + ("" if final else " (first scan, cap of three chosen at random)"), "",
         ("**This is the final write-up of the study.** At most three clusters per 2.4 mm sub-area are sent, the three of highest pT, the pT taken from the bending angle assuming d0 = 0. "
          "The first scan, with the three chosen at random, is kept as a record in `README.md`; the two are compared on the same tracks in section 6a and in `results/compare_ptcap_vs_first_scan.md`."
          if final else
          "**Record of the first scan** (the cap of three clusters per sub-area chosen at random). The final write-up of the study, with the cap ranked by pT from the bending angle, is `README_ptcap.md`; "
          "the two scans are compared on the same tracks in `results/compare_ptcap_vs_first_scan.md`."), "",
         f"Made {prov['run_at_local']} (machine-local time) from the scan units that existed then. The scan was "
         + ("still running" if any(v["quantization_scenarios_analysed"] < prov["quantization_scenarios_in_registry"] for v in prov["detector_scenarios"].values()) else "complete") + ". Specification: `PLAN.md`. "
         "Re-make at any time: `python eval_refitq/spixbits/analysis/run_all.py` in the worktree `ngtagger-train-spixbits` (branch `spixbits`).", "",
         *partial_note(prov),
         "| detector scenario | quantization scenarios complete on all five scan files (of %d) | quantization scenarios present on some files only |" % prov["quantization_scenarios_in_registry"], "|---|---|---|"]
    for s, v in prov["detector_scenarios"].items():
        L.append(f"| {s} | {v['quantization_scenarios_analysed']} | {len(v['quantization_scenarios_partial'])} |")
    L += ["", "Conventions. Everything is in the SmartPixels frame: **x** along the 50 µm pitch (z in the barrel, radial on discs; drives z0), **y** along the 12.5 µm pitch (r-φ; drives d0), "
          "**α** the angle along x, **β** the angle along y, the bending angle. An allocation is written x/y/α/β in bits; F = float, 0 = not sent. Positions are an offset inside the 2.4 mm sub-area "
          "(or the 9.6 mm whole sensor, which has the step of a sub-area code with two bits fewer). Widths are 68 % of |delivered − true| about zero, in µm, on a fixed population "
          "(outer-tracker tracks of the prompt refitted collection with a tracking particle, pT ≥ 2 GeV, |η| < 2.4; a track whose refit does not run keeps its outer-tracker values). "
          "Scan files 5, 7, 8, 9, 10 (500 TTbar events per detector scenario). Bands: paired bootstrap by event, 1,000 replicas, the same event weights in every quantization scenario and detector scenario. "
          "**Results are in-sample and the angles are placeholders** (the network on barrel and discs, a CNN stand-in for clusters without angle truth).", "",
          "## 1. Headline", ""]
    L += headline(fp, sn)
    L += ["", "## 1a. Conclusions", ""]
    L += conclusions(fp, sn, nf)
    L += ["", "## 2. Fixed points", ""]
    cross = lib.RESULTS / "fixed_points_cross.md"
    L += open(cross).read().replace("### ", "#### ").split("\n")
    for s in fp["detector_scenarios"]:
        txt = open(lib.RESULTS / f"fixed_points_{s}.md").read().replace("### ", "#### ")
        L += txt.split("\n")
    L += ["## 3. Scans, one output at a time", "",
          "Curves: 68 % half-width with its bootstrap band, 95 % half-width dashed (plotted, not read). Position axes carry the step, the sub-area bits and the whole-sensor bits. "
          "Full tables per point: `results/scans_<detector scenario>.md`; numbers: `results/scans.json`.", ""]
    for s in sn["detector_scenarios"]:
        figs = sn["detector_scenarios"][s].get("figures", [])
        for f in figs:
            if f.startswith("scan_"):
                L += [f"![{f}]({REL}/fig/{f}.png)", ""]
        secs = sections(lib.RESULTS / f"scans_{s}.md")
        L += pick(secs, "bits needed") + [""]
    L += ["## 4. Allocations with 16 bits, and the bending-angle-only family", ""]
    for s in sn["detector_scenarios"]:
        secs = sections(lib.RESULTS / f"scans_{s}.md")
        L += pick(secs, "allocations with 16 bits") + pick(secs, "bending-angle-only family")
        if f"bending_family_{s}" in sn["detector_scenarios"][s].get("figures", []):
            L += [f"![bending family](" + REL + "/fig/bending_family_{s}.png)", ""]
    L += ["## 5. Controls", ""]
    n0 = len(L)
    for s in sn["detector_scenarios"]:
        secs = sections(lib.RESULTS / f"scans_{s}.md")
        L += pick(secs, "controls (ratio") + pick(secs, "Gaussian noise")
    if len(L) == n0:
        L += ["No control quantization scenario was complete on all five files when this draft was made.", ""]
    L += ["## 6. b/c daughters, displaced tracks, and the |d0| separation", ""]
    for s in sn["detector_scenarios"]:
        secs = sections(lib.RESULTS / f"scans_{s}.md")
        L += pick(secs, "the cap of three clusters per sub-area and the d0 of b/c daughters") + pick(secs, "b/c daughters and displaced tracks, d0") + pick(secs, "b/c daughters and displaced tracks, z0") + pick(secs, "secondary")
    if final:
        L += ["## 6a. The cap of three: ranked by pT against chosen at random, and what the ranking still loses", ""]
        for name in ("cap_look.md",):
            p = lib.RESULTS / name
            if p.exists():
                L += open(p).read().replace("### ", "#### ").split("\n")
        p = lib.STUDY / "results" / "compare_ptcap_vs_first_scan_summary.md"
        if p.exists():
            L += open(p).read().replace("### ", "#### ").split("\n")
        L += ["Full table, every quantization scenario present in both scans: `results/compare_ptcap_vs_first_scan.md`.", ""]
    L += ["## 7. Near full resolution", ""]
    L += open(lib.RESULTS / "near_full.md").read().replace("## Near-full-resolution reading (PLAN 4a)\n", "").replace("### ", "#### ").split("\n")
    for s in nf["detector_scenarios"]:
        L += [f"![near full resolution](" + REL + "/fig/near_full_{s}.png)", ""]
    L += ["## 8. Flags", ""]
    for s, v in fp["detector_scenarios"].items():
        other = [f for f in v["flags"] if "pull RMS" not in f or "d0 pull" not in f and "z0 pull" not in f]
        npull = len(v["flags"]) - len(other)
        L += [f"- {s}: d0 or z0 pull RMS outside 1 ± 0.15 in {npull} quantization scenario × quantity cases (the baseline's own pulls are outside; see conclusions)."] + [f"- {f}" for f in other]
    for s in sn["detector_scenarios"]:
        L += [f"- {s}, outside the fixed points: {f}" for f in sn["detector_scenarios"][s].get("flags", [])]
    cap = next((a["spec"].get("cap") for a in core.registry() if a["name"] == "network_float"), True)
    caprule = ("the three of highest pT, the pT taken from the bending angle assuming d0 = 0" if cap == "pt" else "chosen at random (the chip's real choice is not known)")
    L += ["", "## 9. Limitations", "", LIMITS.replace("CAPRULE", caprule), "", "## 10. Not run", "", NOT_RUN, "",
          "## 11. Files", "",
          "- `results/fixed_points.json`, `results/scans.json`, `results/near_full.json`: all numbers, with the list of quantization scenarios complete, partial and missing when they were made.",
          "- `results/fixed_points_<detector scenario>.md`, `results/fixed_points_cross.md`, `results/scans_<detector scenario>.md`, `results/near_full.md`: tables.",
          "- `results/fig/*.png`, `*.svg`: figures (matplotlib defaults).",
          "- Code: `ngtagger-train-spixbits/eval_refitq/spixbits/analysis/` (core.py summaries and bootstrap per quantization scenario; fixed_points.py; scans.py; near_full.py; readme.py; run_all.py).",
          "- Scan outputs and their schema: `~/smartpixels/scratch_local/spixbits/` (units/, SCHEMA.md, RESUME.md, arms.json). This document was made from the units in `" + str(core.UNITS) + "`.", ""]
    open(OUT, "w").write("\n".join(L))
    print("written", OUT, len(L), "lines")


if __name__ == "__main__":
    main()
