"""spixbits analysis: what the cap of three clusters per sub-area, ranked by pT from the bending angle, still loses.

  SPIXBITS_SCAN_DIR=<scan dir> SPIXBITS_TAG=ptcap python cap_look.py   -> results/<tag>/cap_look.md, .json, fig/cap_look_<detector scenario>.png/.svg

Reads only the truth files of the scan (never another scan's). Crossing clusters of prompt tracks (particle pT >= 2
GeV) on barrel layer 1: kept or dropped by the cap, against the ranking quantity (the cluster's pT from the network's
unquantized bending angle assuming d0 = 0) and the particle's true pT. The angles are placeholders: a reading only.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import core
import lib
from scans import save


def main():
    lib.check_disk()
    res, L = {}, ["### What the pT-ranked cap still drops on barrel layer 1", "",
                  "Crossing clusters of prompt tracks (true pT ≥ 2 GeV) on barrel layer 1. Ranking quantity = the cluster's pT from the network's unquantized bending angle β, assuming d0 = 0 "
                  "(a cluster with no usable value ranks last). The angles are placeholders, so this is a reading of the emulation, not of the sensor.", "",
                  "Scale to keep in mind (an estimate, not measured here): at the radius of barrel layer 1, about 3 cm, a 2 GeV track is bent by about 0.3 × 3.8 T × 0.03 m / (2 × 2 GeV) ≈ 9 mrad against the beam-line direction, "
                  "and less at higher pT, while the emulated bending-angle residual is tens of mrad wide. A pT computed from one cluster's angle is then set mostly by the residual and comes out far below the true pT; "
                  "the tables below show how far.", ""]
    for scen in core.SCENARIOS:
        parts = [np.load(core.UNITS / "truth" / f"{scen}_f{f:02d}_truth.npz") for f in core.FILES]
        if "crossing_cluster_pt_from_bending_GeV" not in parts[0].files:
            continue
        cat = lambda k: np.concatenate([p[k] for p in parts])
        surf, cc, sent, rnd = cat("slot_surface"), cat("crossing_cluster"), cat("crossing_cluster_sent_under_cap"), cat("crossing_cluster_sent_under_random_cap")
        ptb, nsub = cat("crossing_cluster_pt_from_bending_GeV"), cat("crossing_cluster_n_in_subarea")
        tpt = np.broadcast_to(cat("tp_pt")[:, None], surf.shape)
        prompt = np.broadcast_to((cat("origin") == 0)[:, None], surf.shape)
        m = (surf == 1) & (cc >= 0) & prompt
        if m.sum() < 100:
            L += [f"{scen}: no barrel layer 1 in this detector scenario.", ""]
            continue
        crowded = m & (nsub > 3)
        d = {"n_crossing_clusters": int(m.sum()), "share_in_sub_areas_with_more_than_three": float(crowded.sum() / m.sum()),
             "dropped_pt_ranked": float((m & ~sent).sum() / m.sum()), "dropped_random": float((m & ~rnd).sum() / m.sum()),
             "dropped_pt_ranked_given_more_than_three": float((crowded & ~sent).sum() / max(crowded.sum(), 1)),
             "dropped_random_given_more_than_three": float((crowded & ~rnd).sum() / max(crowded.sum(), 1))}
        rows = {}
        for name, sel in (("kept, sub-area with more than three clusters", crowded & sent), ("dropped", m & ~sent), ("all crossing clusters", m)):
            v, t = ptb[sel], tpt[sel]
            ok = np.isfinite(v)
            rows[name] = {"n": int(sel.sum()), "share_without_a_value": float(1 - ok.mean()), "pt_from_bending_quantiles_16_50_84_GeV": [float(x) for x in np.quantile(v[ok], [0.16, 0.5, 0.84])] if ok.any() else None,
                          "share_pt_from_bending_below_2_GeV": float(((v < 2.0) | ~ok).mean()), "share_pt_from_bending_below_1_GeV": float(((v < 1.0) | ~ok).mean()),
                          "true_pt_quantiles_16_50_84_GeV": [float(x) for x in np.quantile(t, [0.16, 0.5, 0.84])],
                          "median_ratio_pt_from_bending_over_true": float(np.median(v[ok] / t[ok])) if ok.any() else None}
        d["groups"] = rows
        ok = m & np.isfinite(ptb)
        ratio = ptb[ok] / tpt[ok]
        d["pt_from_bending_over_true_pt_quantiles_16_50_84"] = [float(x) for x in np.quantile(ratio, [0.16, 0.5, 0.84])]
        d["share_of_all_crossing_clusters_with_pt_from_bending_below_half_the_true_pt"] = float((ratio < 0.5).mean())
        d["share_without_a_value"] = float(1 - ok.sum() / m.sum())
        dm = m & ~sent
        d["dropped_by_true_pt"] = {k: float((dm & (tpt >= lo) & (tpt < hi)).sum() / max((m & (tpt >= lo) & (tpt < hi)).sum(), 1)) for k, lo, hi in (("2-3 GeV", 2, 3), ("3-5 GeV", 3, 5), ("5-10 GeV", 5, 10), ("> 10 GeV", 10, 1e9))}
        res[scen] = d
        L += [f"**{scen}** ({d['n_crossing_clusters']:,} crossing clusters): {100 * d['share_in_sub_areas_with_more_than_three']:.1f} % sit in a sub-area holding more than three clusters. "
              f"Dropped by the pT-ranked cap: {100 * d['dropped_pt_ranked']:.2f} % of all ({100 * d['dropped_pt_ranked_given_more_than_three']:.1f} % of those in such sub-areas); "
              f"by the cap at random: {100 * d['dropped_random']:.2f} % ({100 * d['dropped_random_given_more_than_three']:.1f} %). "
              f"Ranking quantity ÷ true pT over all crossing clusters: 16 / 50 / 84 % quantiles {d['pt_from_bending_over_true_pt_quantiles_16_50_84'][0]:.2f} / {d['pt_from_bending_over_true_pt_quantiles_16_50_84'][1]:.2f} / {d['pt_from_bending_over_true_pt_quantiles_16_50_84'][2]:.2f}; "
              f"below half the true pT for {100 * d['share_of_all_crossing_clusters_with_pt_from_bending_below_half_the_true_pt']:.1f} %; no value for {100 * d['share_without_a_value']:.1f} %. "
              "Dropped share by true pT: " + ", ".join(f"{k}: {100 * x:.2f} %" for k, x in d["dropped_by_true_pt"].items()) + ".", "",
              "| group | n | pT from bending, 16 / 50 / 84 % [GeV] | below 2 GeV or no value | below 1 GeV or no value | true pT, 16 / 50 / 84 % [GeV] | median pT from bending ÷ true pT |", "|---|---|---|---|---|---|---|"]
        for name, r in rows.items():
            qq = r["pt_from_bending_quantiles_16_50_84_GeV"]
            L.append(f"| {name} | {r['n']:,} | " + (" / ".join(f"{x:.2f}" for x in qq) if qq else "n/a") + f" | {100 * r['share_pt_from_bending_below_2_GeV']:.1f} % | {100 * r['share_pt_from_bending_below_1_GeV']:.1f} % | "
                     + " / ".join(f"{x:.2f}" for x in r["true_pt_quantiles_16_50_84_GeV"]) + f" | {r['median_ratio_pt_from_bending_over_true']:.2f} |")
        L.append("")
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
        bins = np.logspace(-1.5, 2, 50)
        for name, sel in (("kept, sub-area with more than three clusters", crowded & sent), ("dropped", m & ~sent)):
            v = ptb[sel]
            ax[0].hist(np.clip(v[np.isfinite(v)], bins[0], bins[-1]), bins=bins, histtype="step", density=True, label=f"{name} (n = {int(sel.sum()):,})")
            ax[1].hist(np.clip(tpt[sel], bins[0], bins[-1]), bins=bins, histtype="step", density=True, label=name)
        for a, xl in zip(ax, ("cluster pT from the bending angle, d0 = 0 assumed [GeV] (the ranking quantity)", "true particle pT [GeV]")):
            a.set_xscale("log")
            a.set_xlabel(xl)
            a.set_ylabel("crossing clusters, normalized")
            a.axvline(2.0, linestyle=":", color="k")
            a.legend(fontsize=7)
        ax[0].set_title(f"{scen}, barrel layer 1, prompt tracks (true pT >= 2 GeV)")
        fig.tight_layout()
        save(fig, f"cap_look_{scen}")
        L += [f"![cap look](results/{core.TAG}/fig/cap_look_{scen}.png)" if core.TAG else f"![cap look](results/fig/cap_look_{scen}.png)", ""]
    json.dump(res, open(lib.RESULTS / "cap_look.json", "w"), indent=1)
    open(lib.RESULTS / "cap_look.md", "w").write("\n".join(L) + "\n")
    print("written cap_look")


if __name__ == "__main__":
    main()
