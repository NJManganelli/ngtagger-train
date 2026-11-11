"""spixbits analysis, deliverable D: the README draft, assembled from the result files of the other steps.

  python readme.py     -> ~/smartpixels/doc-archive/spix-bits-refit-study/README.md

Conclusions are filled from the numbers in results/*.json, so the draft follows whatever units were complete; parts
whose arms are missing say so. Local file only.
"""
import json

import core
import lib

OUT = lib.RESULTS.parent / "README.md"


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
    S = fp["scenarios"]
    scs = list(S)

    def per(fn):
        return "; ".join(f"{s}: {fn(S[s]['arms'])}" for s in scs)

    def w(arm, q):
        return lambda A: f"{A['implemented_now'][q + '_w68_um']['value']:.0f} → {A[arm][q + '_w68_um']['value']:.0f} µm (× {r3(A[arm][q + '_ratio_to_implemented_now'])})" if arm in A else "missing"

    C.append("**What a coarse cluster costs (prompt tracks, 68 % half-width, against what is implemented now).**")
    C.append(f"- 16-bit cluster 4/4/4/4 with positions inside the 2.4 mm sub-area (150 µm steps): d0 {per(w('alloc_4444', 'd0'))}. z0 {per(w('alloc_4444', 'z0'))}.")
    C.append(f"- The same 4/4/4/4 over the whole 9.6 mm sensor (600 µm steps): d0 {per(w('alloc_4444_wholesensor', 'd0'))}. z0 {per(w('alloc_4444_wholesensor', 'z0'))}.")
    C.append(f"- Bending-angle only 6/6/0/4 inside the sub-area (37.5 µm steps, no α): d0 {per(w('alloc_6604', 'd0'))}. z0 {per(w('alloc_6604', 'z0'))}. "
             f"Over the whole sensor (150 µm steps): d0 {per(w('alloc_6604_wholesensor', 'd0'))}.")
    C.append("- The cost is largest where today's resolution is best: central tracks with pT ≥ 5 GeV lose the largest factor (cell tables below); "
             "in IAAA, where today's d0 width is already " + f"{S['IAAA']['arms']['implemented_now']['d0_w68_um']['value'] / S['AAAA']['arms']['implemented_now']['d0_w68_um']['value']:.1f}" + " × that of AAAA, the same steps cost a smaller factor." if "IAAA" in S and "AAAA" in S else "")
    A = S[scs[0]]["arms"]
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
            bn = sn["scenarios"].get(s, {}).get("bits_needed", {}).get(out, {})
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
        pts = sn["scenarios"][scs[0]]["scans"]["y"]["down"]["points"]
        if pts:
            C.append(f"- y down-scan, {scs[0]}, d0 ÷ \"network, float\": " + ", ".join(f"{p['step_um']:.3g} µm × {p['d0_ratio_to_network_float']['value']:.2f}" for p in pts.values()) + ".")
        pts = sn["scenarios"][scs[0]]["scans"]["x"]["down"]["points"]
        if pts:
            C.append(f"- x down-scan, {scs[0]}, z0 ÷ \"network, float\": " + ", ".join(f"{p['step_um']:.3g} µm × {p['z0_ratio_to_network_float']['value']:.2f}" for p in pts.values()) + ".")
        for o, q in (("beta", "d0"), ("alpha", "z0")):
            pts = sn["scenarios"][scs[0]]["scans"][o]["down"]["points"]
            if pts:
                C.append(f"- {o} down-scan, {scs[0]}, {q} ÷ \"network, float\": " + ", ".join(f"{'not sent' if p['bits'] == 0 else str(p['bits']) + ' bits'} × {p[q + '_ratio_to_network_float']['value']:.3f}" for p in pts.values()) + ".")
    except KeyError:
        pass
    # allocations
    C.append("")
    C.append("**Allocations with 16 bits.**")
    done = False
    for s in scs:
        al = sn["scenarios"].get(s, {}).get("allocations_16_bits", {})
        pts = {k: v for k, v in al.get("points", {}).items() if not k.endswith("_globalrange")}
        if len(pts) >= 11:
            done = True
            bd = min(pts.values(), key=lambda v: v["d0"]["value"])
            bz = min(pts.values(), key=lambda v: v["z0"]["value"])
            C.append(f"- {s}: smallest d0 width: {bd['name']} ({bd['d0']['value']:.1f} µm, × {r3(bd['d0_ratio_to_implemented_now'])} of implemented now; best in {100 * bd['d0_share_of_replicas_best']:.0f} % of replicas); "
                     f"smallest z0 width: {bz['name']} ({bz['z0']['value']:.1f} µm, × {r3(bz['z0_ratio_to_implemented_now'])}; best in {100 * bz['z0_share_of_replicas_best']:.0f} % of replicas).")
    if not done:
        C.append("- The eleven-allocation table is not complete yet; of the two that exist, 6/6/0/4 has the smaller d0 and z0 widths in every scenario (fixed-point table).")
    # near full
    C.append("")
    C.append("**Near full resolution (the network's positions are better than today's, which cannot be emulated by adding noise).**")
    for s in scs:
        d = nf["scenarios"].get(s, {})
        if d:
            C.append(f"- {s}: d0 against y: {d['y']['cells']['prompt, all']['reading']}. z0 against x: {d['x']['cells']['prompt, all']['reading']}.")
    C.append("")
    C.append("**Honesty of the errors.**")
    C.append("- Correct-hit pull RMS is inside 1 ± 0.15 in every fixed-point arm with all surfaces pooled; on the discs the angle pulls of \"implemented now\" are " + f"{A['implemented_now']['hit_pull_rms_by_region']['discs']['beta']:.2f} (β) and {A['implemented_now']['hit_pull_rms_by_region']['discs']['alpha']:.2f} (α) in {scs[0]}" + ", outside the range (the network arms: " + f"{A['network_float']['hit_pull_rms_by_region']['discs']['beta']:.2f} and {A['network_float']['hit_pull_rms_by_region']['discs']['alpha']:.2f}" + "). "
             "The d0 and z0 pull RMS are outside 1 ± 0.15 already in \"implemented now\" (" + per(lambda A: f"d0 {A['implemented_now']['d0_pull_rms']:.2f}, z0 {A['implemented_now']['z0_pull_rms']:.2f}") +
             ") and every arm inherits this; by the letter of the plan every arm carries the flag, so the flag does not single out an arm.")
    return C


LIMITS = """- The angles are placeholders: the barrel-trained PixelAV network stands in on the discs as well; clusters without angle truth keep the CNN stand-in, judged with the network's narrower errors. Nothing here is a statement about the angle quality of the real sensor.
- A particle's secondary clusters (delta rays, fragments) carry the parent's true angle plus a residual, so no angle can reject them here; a real network would not report the parent's direction for them.
- The simulated sensors are 25 × 100 µm, 150 µm thick; the SmartPixels sensor is 12.5 × 50 µm, 100 µm thick. Occupancy and merging are those of the simulated sensor.
- At most three clusters per 2.4 mm sub-area are sent, chosen at random (the chip's real choice is not known); clusters straddling a sub-area edge are not modelled.
- Position and angle residuals of one cluster are uncorrelated here (they are correlated in the network).
- Skip prices, seed model and material scale are today's in every arm; the one bound on the skip price was not run.
- v3's window already loses correct clusters in every arm; arms with quantized positions widen it by the quantization variance and recover some.
- Results are in-sample with respect to the v3 calibrations, the angle ranges and the angle table (500 scan events; ranges and tables fitted on all ten files).
- Positions are built on today's rec-hit positions: near full resolution the measured numbers are an upper bound and the network's own position resolution enters only through the model reading.
- Whether a refit with one angle (the bending-angle-only arms) is implementable in a real system is doubtful; those arms are for comparison only.
- Nothing downstream of the track is rebuilt: no particle flow, jets or tagger numbers.
- Input precision: the nano's cluster table keeps 16 mantissa bits of the positions; the first check passed in the form the physicist accepted (one window-edge disagreement and an effect of at most 0.026 σ, common to every arm).
- Forced refits use the clusters "implemented now" accepted even where the cap of three would not have sent them; in capped arms the hit-choice effect therefore includes the cap."""

NOT_RUN = """- Skip price re-derived at the 16-bit point (needs the cut fit; a separate run).
- The direct network-position point above 50 GeV (helix + drawn network position residual).
- The extended (displaced-capable) track collection in the replay.
- The track-MVA heads (deployed heads scored, heads retrained per arm).
- The parametrization (correctionlib file per scenario, PLAN 7b).
- Bits-needed readings in the 12-cell split (only the prompt total, the six design cells and the pooled forward cells are read); the "both refit" view (the refit fraction is 0.997-1.000 in every fixed-point arm).
- Stale or pending launch checks of the implementation (planted defects, bit-equal rerun of a unit): see ~/smartpixels/scratch_local/spixbits/RESUME.md."""


def main():
    lib.check_disk()
    fp = json.load(open(lib.RESULTS / "fixed_points.json"))
    sn = json.load(open(lib.RESULTS / "scans.json"))
    nf = json.load(open(lib.RESULTS / "near_full.json"))
    prov = sn["provenance"]
    L = ["# SmartPixels output bits and the L1 track refit: results (DRAFT)", "",
         f"Draft made {prov['run_at_local']} (machine-local time) from the scan units that existed then. The scan was "
         + ("still running" if any(v["arms_analysed"] < prov["arms_in_registry"] for v in prov["scenarios"].values()) else "complete") + ". Specification: `PLAN.md`. "
         "Re-make at any time: `python eval_refitq/spixbits/analysis/run_all.py` in the worktree `ngtagger-train-spixbits` (branch `spixbits`).", "",
         "| scenario | arms complete on all five scan files (of %d) | arms present on some files only |" % prov["arms_in_registry"], "|---|---|---|"]
    for s, v in prov["scenarios"].items():
        L.append(f"| {s} | {v['arms_analysed']} | {len(v['arms_partial'])} |")
    L += ["", "Conventions. Everything is in the SmartPixels frame: **x** along the 50 µm pitch (z in the barrel, radial on discs; drives z0), **y** along the 12.5 µm pitch (r-φ; drives d0), "
          "**α** the angle along x, **β** the angle along y, the bending angle. An allocation is written x/y/α/β in bits; F = float, 0 = not sent. Positions are an offset inside the 2.4 mm sub-area "
          "(or the 9.6 mm whole sensor, which has the step of a sub-area code with two bits fewer). Widths are 68 % of |delivered − true| about zero, in µm, on a fixed population "
          "(outer-tracker tracks of the prompt refitted collection with a tracking particle, pT ≥ 2 GeV, |η| < 2.4; a track whose refit does not run keeps its outer-tracker values). "
          "Scan files 5, 7, 8, 9, 10 (500 TTbar events per scenario). Bands: paired bootstrap by event, 1,000 replicas, the same event weights in every arm and scenario. "
          "**Results are in-sample and the angles are placeholders** (the network on barrel and discs, a CNN stand-in for clusters without angle truth).", "",
          "## 1. Conclusions", ""]
    L += conclusions(fp, sn, nf)
    L += ["", "## 2. Fixed points", ""]
    cross = lib.RESULTS / "fixed_points_cross.md"
    L += open(cross).read().replace("### ", "#### ").split("\n")
    for s in fp["scenarios"]:
        txt = open(lib.RESULTS / f"fixed_points_{s}.md").read().replace("### ", "#### ")
        L += txt.split("\n")
    L += ["## 3. Scans, one output at a time", "",
          "Curves: 68 % half-width with its bootstrap band, 95 % half-width dashed (plotted, not read). Position axes carry the step, the sub-area bits and the whole-sensor bits. "
          "Full tables per point: `results/scans_<scenario>.md`; numbers: `results/scans.json`.", ""]
    for s in sn["scenarios"]:
        figs = sn["scenarios"][s].get("figures", [])
        for f in figs:
            if f.startswith("scan_"):
                L += [f"![{f}](results/fig/{f}.png)", ""]
        secs = sections(lib.RESULTS / f"scans_{s}.md")
        L += pick(secs, "bits needed") + [""]
    L += ["## 4. Allocations with 16 bits, and the bending-angle-only family", ""]
    for s in sn["scenarios"]:
        secs = sections(lib.RESULTS / f"scans_{s}.md")
        L += pick(secs, "allocations with 16 bits") + pick(secs, "bending-angle-only family")
        if f"bending_family_{s}" in sn["scenarios"][s].get("figures", []):
            L += [f"![bending family](results/fig/bending_family_{s}.png)", ""]
    L += ["## 5. Controls", ""]
    n0 = len(L)
    for s in sn["scenarios"]:
        secs = sections(lib.RESULTS / f"scans_{s}.md")
        L += pick(secs, "controls (ratio") + pick(secs, "Gaussian noise")
    if len(L) == n0:
        L += ["No control arm was complete on all five files when this draft was made.", ""]
    L += ["## 6. b/c daughters, displaced tracks, and the |d0| separation", ""]
    for s in sn["scenarios"]:
        secs = sections(lib.RESULTS / f"scans_{s}.md")
        L += pick(secs, "b/c daughters and displaced tracks, d0") + pick(secs, "b/c daughters and displaced tracks, z0") + pick(secs, "secondary")
    L += ["## 7. Near full resolution", ""]
    L += open(lib.RESULTS / "near_full.md").read().replace("## Near-full-resolution reading (PLAN 4a)\n", "").replace("### ", "#### ").split("\n")
    for s in nf["scenarios"]:
        L += [f"![near full resolution](results/fig/near_full_{s}.png)", ""]
    L += ["## 8. Flags", ""]
    for s, v in fp["scenarios"].items():
        other = [f for f in v["flags"] if "pull RMS" not in f or "d0 pull" not in f and "z0 pull" not in f]
        npull = len(v["flags"]) - len(other)
        L += [f"- {s}: d0 or z0 pull RMS outside 1 ± 0.15 in {npull} arm × quantity cases (the baseline's own pulls are outside; see conclusions)."] + [f"- {f}" for f in other]
    for s in sn["scenarios"]:
        L += [f"- {s}, outside the fixed points: {f}" for f in sn["scenarios"][s].get("flags", [])]
    L += ["", "## 9. Limitations", "", LIMITS, "", "## 10. Not run", "", NOT_RUN, "",
          "## 11. Files", "",
          "- `results/fixed_points.json`, `results/scans.json`, `results/near_full.json`: all numbers, with the list of arms complete, partial and missing when they were made.",
          "- `results/fixed_points_<scenario>.md`, `results/fixed_points_cross.md`, `results/scans_<scenario>.md`, `results/near_full.md`: tables.",
          "- `results/fig/*.png`, `*.svg`: figures (matplotlib defaults).",
          "- Code: `ngtagger-train-spixbits/eval_refitq/spixbits/analysis/` (core.py per-arm summaries and bootstrap; fixed_points.py; scans.py; near_full.py; readme.py; run_all.py).",
          "- Scan outputs and their schema: `~/smartpixels/scratch_local/spixbits/` (units/, SCHEMA.md, RESUME.md, arms.json).", ""]
    open(OUT, "w").write("\n".join(L))
    print("written", OUT, len(L), "lines")


if __name__ == "__main__":
    main()
