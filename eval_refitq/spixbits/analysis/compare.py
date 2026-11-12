"""spixbits analysis: one scan against another on the same tracks (e.g. the cap of three clusters per sub-area chosen
at random against chosen by pT rank).

  python compare.py TAG [REFERENCE_TAG]     -> results/compare_<TAG>_vs_<reference>.md and .json

Both scans must have been summarized (run_all.py, with --scan-dir/--tag for the tagged one). Reads only the caches.
Every quantization scenario present in both, per detector scenario: d0 and z0 68 % half-widths and the paired ratio
TAG / reference with its 68 % bootstrap band (the replica weights are keyed by event and identical in both).
"""
import json
import sys

import core
import lib


def cache_of(tag):
    return core.CACHE_ROOT / ("cache" if not tag else f"cache_{tag}")


def main(tag, ref=""):
    lib.check_disk()
    reg = {a["name"] for a in core.registry()}
    refname = ref or "first scan"
    res, L = {}, [f"## {tag} against {refname}: same tracks, paired by event", "",
                  "68 % half-width of |delivered − true| in µm; ratio = " + tag + " ÷ " + refname + " with its 68 % paired bootstrap band. In-sample; placeholder angles.", ""]
    for scen in core.SCENARIOS:
        A, B = lib.Scenario(scen, cache_of(tag)), lib.Scenario(scen, cache_of(ref))
        if A.meta is None or B.meta is None:
            continue
        if A.meta["cells"]["six"]["counts"] != B.meta["cells"]["six"]["counts"]:
            L += [f"{scen}: the two scans do not hold the same population; not compared.", ""]
            continue
        names6 = A.meta["cells"]["six"]["names"]
        cells = [("prompt, all", "origin", 0), ("b/c daughters, pT >= 5 GeV", "six", names6.index("b/c daughters, pT >= 5 GeV")),
                 ("b/c daughters, pT 2-5 GeV", "six", names6.index("b/c daughters, pT 2-5 GeV"))]
        res[scen] = {}
        for q in ("d0", "z0"):
            L += [f"### {scen}: {q}", "", "| quantization scenario | " + " | ".join(f"{c}: {tag} / {refname} [µm] | ratio" for c, *_ in cells) + " |", "|---|" + "---|---|" * len(cells)]
            for arm in sorted(a for a in A.arms if a in B.arms and a in reg):
                cs = []
                for c, part, g in cells:
                    wa, wb = A.w(arm, q, part, g), B.w(arm, q, part, g)
                    r = lib.val(wa[0] / wb[0], wa[2] / wb[2])
                    res[scen].setdefault(arm, {}).setdefault(q, {})[c] = {"w68_um": wa[0], "reference_w68_um": wb[0], "ratio": r}
                    cs.append(f"{wa[0]:.1f} / {wb[0]:.1f} | {lib.fmt_band(r)}")
                L.append(f"| {arm} | " + " | ".join(cs) + " |")
            L.append("")
    out = lib.STUDY / "results" / f"compare_{tag}_vs_{(ref or 'first_scan')}"
    json.dump(res, open(str(out) + ".json", "w"), indent=1)
    open(str(out) + ".md", "w").write("\n".join(L) + "\n")
    print("written", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "")
