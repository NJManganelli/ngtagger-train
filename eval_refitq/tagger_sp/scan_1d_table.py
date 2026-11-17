"""The results file of the upward one-parameter scan in the jet tagger (prompt collection), rewritten after every point:
doc-archive/spix-bits-refit-study/results/ptcap/tagger_1d_scan_prompt.md (+ .json).

  python scan_1d_table.py

Reads, per finished point <name> (folder <name>_qheads_enlarged_reencoded under tagger_slimtier_prompt): selection.json
(trials kept per tagger training), paired_vs_implemented_now_refheads_enlarged.json and
paired_vs_alloc_4444_qheads_enlarged_reencoded.json (analyze.py --paired-with: same test events, same bootstrap
replicates). A tagger training with fewer than three kept trials is marked "short" and its number is not quoted.
"""
import json
import os
import time

BASE = "/Volumes/WDMac/smartpixels_scratch/tagger_slimtier_prompt"
OUT = os.path.expanduser("~/smartpixels/doc-archive/spix-bits-refit-study/results/ptcap/tagger_1d_scan_prompt")
REFS = {"implemented now": "implemented_now_refheads_enlarged", "4/4/4/4": "alloc_4444_qheads_enlarged_reencoded"}
PARAMS = {"y": ["5b", "6b", "7b", "8b", "9b", "10b", "11b", "float"], "x": ["5b", "6b", "7b", "8b", "9b", "float"],
          "beta": ["5b", "6b", "7b", "float"], "alpha": ["5b", "6b", "7b", "float"]}
MASKS = ["AAAA", "AAAI", "IAAA"]
SETS = {"baseline": "candidates only", "track": "+ track quantities", "trackmva": "+ track quantities and track-quality heads"}
METRICS = {"b_vs_light_eff@10%": "b efficiency at the 10 % light-jet fake rate", "b_vs_light_eff@1%": "b efficiency at the 1 % light-jet fake rate",
           "b_vs_light_auc": "b-against-light AUC"}


def folder(name):
    return os.path.join(BASE, f"{name}_qheads_enlarged_reencoded")


def bits(param, p):
    b = {"x": 0, "y": 1, "alpha": 2, "beta": 3}[param]
    v = ["4", "4", "4", "4"]
    v[b] = "float" if p == "float" else p[:-1]
    return "/".join(v)


def point(name):
    f = folder(name)
    need = [os.path.join(f, "results_ttbar", "selection.json")] + [os.path.join(f, f"paired_vs_{r}.json") for r in REFS.values()]
    if not all(os.path.exists(p) for p in need):
        return None
    sel = json.load(open(need[0]))["cells"]
    out = {"kept": {k: [v["kept"], v["launched"]] for k, v in sel.items() if not k.endswith("@IIII")}}
    for lab, p in zip(REFS, need[1:]):
        out[lab] = json.load(open(p))["trainings"]
    return out


def cell(P, ref, mask, var, metric):
    if P["kept"].get(f"{var}@{mask}", [0])[0] < 3:
        return "short"
    x = P[ref][f"{var}@{mask}"][metric]
    err = ((x["boot_err"] or 0) ** 2 + x["seed_err"] ** 2) ** 0.5
    return f"{x['this']:.3f} ({x['diff']:+.3f} ± {err:.3f})"


def main():
    pts = {f"up_{p}_{b}": point(f"up_{p}_{b}") for p in PARAMS for b in PARAMS[p]}
    done = [k for k, v in pts.items() if v]
    L = ["# Upward one-parameter scan from 4/4/4/4 in the jet tagger, prompt collection", "",
         f"Updated {time.strftime('%Y-%m-%d %H:%M')} (machine clock, CST). {len(done)} of {len(pts)} points done.", "",
         "Asked by the physicist: \"4/4/4/4 is baseline, then each of x, y, alpha, beta are scanned upwards to their limit, NOT",
         "simultaneously downshifting the other parameters\". Each point raises one output of the SmartPixels cluster and leaves the",
         "other three at exactly the 4/4/4/4 settings (bits for x / y / alpha / beta; \"float\" = that output unquantized). Recipe as",
         "the first pass (`tagger_first_pass_prompt.md`): prompt parametrization, track-quality heads retrained on the scenario's own",
         "tracks of the enlarged sample, candidates re-encoded, slim tagger tier TTbar (8,500 common events, 1,693 test), 10 trials + 10",
         "more where fewer than three are kept, best three averaged, working point = each tagger's own light-jet fake rate.",
         "Entries: value (change ± error); the change is against the reference named in the table title, on the same test events with",
         "the same bootstrap replicates; error = paired event bootstrap ⊕ trial spread of both. \"short\" = fewer than three kept",
         "trials after the top-up: not quoted. Reference values, \"implemented now\" / 4/4/4/4, are in the first-pass file.",
         "SmartPixels angles are placeholders: no design conclusion.", ""]
    for metric, mlab in list(METRICS.items()):
        for ref in REFS:
            for var, vlab in SETS.items():
                if metric != "b_vs_light_eff@10%" and var != "track":
                    continue
                L += [f"## {mlab}, input set \"{vlab}\", change against {ref}", "", "| scenario (x/y/alpha/beta bits) | AAAA | AAAI | IAAA |", "|---|---|---|---|"]
                for p in PARAMS:
                    for b in PARAMS[p]:
                        P = pts[f"up_{p}_{b}"]
                        L.append(f"| {bits(p, b)} (`up_{p}_{b}`) | " + " | ".join(cell(P, ref, m, var, metric) if P else "not run yet" for m in MASKS) + " |")
                L.append("")
    L += ["## Trials kept per tagger training (kept of launched)", "", "| scenario | " + " | ".join(f"{m} {v}" for m in MASKS for v in SETS) + " |",
          "|---|" + "---|" * 9]
    for k in done:
        L.append(f"| `{k}` | " + " | ".join("{} of {}".format(*pts[k]["kept"].get(f"{v}@{m}", ["-", "-"])) for m in MASKS for v in SETS) + " |")
    L += ["", "Per-jet scores of every trial: `<run folder>/preds/`; run folders `/Volumes/WDMac/smartpixels_scratch/tagger_slimtier_prompt/"
          "<scenario>_qheads_enlarged_reencoded/`. Joint classifier test of the parametrization with each scenario's enlarged heads: "
          "`~/smartpixels/scratch_local/spixbits/checks/param_closure_p2_qheads_enlarged_*.json`.", ""]
    notes = OUT + "_notes.md"
    if os.path.exists(notes):
        L += open(notes).read().splitlines()
    open(OUT + ".md", "w").write("\n".join(L) + "\n")
    json.dump({k: v for k, v in pts.items() if v}, open(OUT + ".json", "w"))
    print(f"{len(done)} points -> {OUT}.md")


if __name__ == "__main__":
    main()
