"""Binning study: z-histogram granularity for FastHisto (FH) and the E2E NNVtx, paired against
the stock 256-bin configuration on the same test events.

FH  offline float GTT fastHisto (sub2gev code) with nbins bins over the fixed GTT range
    [-20.46912512, 20.46912512] cm (256 -> 0.160 cm, 512 -> 0.080 cm, 1024 -> 0.040 cm) and a
    peak-finder window of `window` consecutive bins (FH_WindowSize; stock 3 = 0.48 cm); the GTT
    eta-binned TVA association is unchanged (it does not depend on the binning).
FHt diagnostic (not GTT): the same peak finder, but z = pT-weighted mean z0 of the tracks in the
    winning window instead of the bin-centre mean; separates "finer peak finding" from "less
    quantisation of the z estimate" (the GTT z is quantised at the bin width).
E2E vertex.py --nbins N --kernel K predictions (soft-histogram layer with N bins over the same
    range, Gaussian sigma = one bin, every pattern Conv1D with kernel K; see vertex.py).

Metrics per configuration (analyze.Cfg, test events f03-f10): PV efficiency |dz| < 0.15 cm and
at 0.05 / 0.10 / 0.30 cm (+ the efficiency-vs-window curve), dz core RMS (iterative 3-sigma clip,
RMS about zero) and full RMS, association AUC, WP efficiency / purity, and the purity at the
association efficiency of the variant's STOCK FH + TVA (one matched operating point for every
configuration). Errors: paired event bootstrap (the analyze.py replicas) of the difference to
the 256-bin baseline of the same method (FH: 256/3; E2E: 256/k3 of the same arm); E2E values
are seed means with the seed std. "Best" = highest PV efficiency on the test events (a mild
optimistic bias; the FH choice on the 200 training-only events f01/f02 is shown next to it).

  pixi run python eval_refitq/nnvtx_sp/binning.py --input-set canon_v2
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as A  # noqa: E402
import nvcommon as nvc  # noqa: E402

FH_SCAN = "256:2,3,4,5;512:3,4,5,6,7,9;1024:3,5,7,9,11,12,13,15"
FHT_SCAN = "256:3;512:3,6;1024:3,12"  # same bin count and the physically equivalent 0.48 cm window
FH_METHODS = ("FH", "FHt")
MLABEL = {**A.MLABEL, "FHt": "FastHisto, track-mean z (diagnostic) + GTT TVA"}
E2E_SCAN = "256:3;512:3,5,7;1024:3,7,13"
EXTRA_WINDOWS = (0.05, 0.10, 0.30)  # cm, PV efficiency at other windows (the natural scale changes)
UM = 1e4
RANGE = 2 * 20.46912512
CN = {256: "#000000", 512: "#0072B2", 1024: "#D55E00"}


def parse_scan(spec):
    out = []
    for part in spec.split(";"):
        nb, ks = part.split(":")
        out += [(int(nb), int(k)) for k in ks.split(",")]
    return out


def width_cm(nb, k):
    return k * RANGE / nb


def extra_effs(cf):
    return {f"eff_{w:.2f}": float(np.mean(np.abs(cf.dz) < w)) for w in EXTRA_WINDOWS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-set", choices=sorted(nvc.OUTDIRS), default="canon_v2")
    ap.add_argument("--outdir", default=None, help="default: <study outdir>/../binning")
    ap.add_argument("--variants", default="V0,V3")
    ap.add_argument("--fh-scan", default=FH_SCAN, help="nbins:window,window;...")
    ap.add_argument("--fht-scan", default=FHT_SCAN, help="FH with track-mean z (diagnostic)")
    ap.add_argument("--e2e-scan", default=E2E_SCAN, help="nbins:kernel,kernel;... (pred files that exist)")
    a = ap.parse_args()
    nvc.configure(a.input_set)
    A.S2A.common.configure(a.input_set)
    outdir = a.outdir or os.path.join(os.path.dirname(nvc.OUTDIR), "binning")
    os.makedirs(outdir, exist_ok=True)
    variants = a.variants.split(",")
    fh_scan, fht_scan, e2e_scan = parse_scan(a.fh_scan), parse_scan(a.fht_scan), parse_scan(a.e2e_scan)
    assert (256, 3) in fh_scan and (256, 3) in e2e_scan, "the 256/3 baseline must be in both scans"

    T, E = nvc.load_dataset()
    folds = nvc.event_folds(E)
    test = folds >= 0
    n_test = int(test.sum())
    genz = E["genz"]
    S = nvc.selections(T)
    cfgs, fh_train_eff, missing, epochs = {}, {}, [], {}
    for v in variants:
        sel, pre, _ = S[v]
        zb = A.fh_z(T, len(test), sel, pre)
        base = A.fh_cfg(T, test, genz, v, zb)
        ref_eff = float(base.passed[base.y.astype(bool)].mean())
        for nb, w in fh_scan:
            z = zb if (nb, w) == (256, 3) else A.fh_z(T, len(test), sel, pre, nb, w)
            cfgs[(v, "FH", nb, w)] = [A.fh_cfg(T, test, genz, v, z, ref_eff)]
            fh_train_eff[(v, nb, w)] = float(np.mean(np.abs(z[~test] - genz[~test]) < A.PV_WINDOW))
        for nb, w in fht_scan:
            z = A.fh_z(T, len(test), sel, pre, nb, w, zmode="tracks")
            cfgs[(v, "FHt", nb, w)] = [A.fh_cfg(T, test, genz, v, z, ref_eff)]
        for nb, k in e2e_scan:
            path = nvc.pred_path(v, nb, k)
            if not os.path.exists(path):
                missing.append(os.path.basename(path))
                continue
            e2e, ep, _ = A.e2e_cfgs(T, test, genz, folds, v, path, ref_eff)
            for arm, lst in e2e.items():
                cfgs[(v, arm, nb, k)] = lst
            epochs[f"{v}_b{nb}_k{k}"] = ep
        print(f"{v}: configs ready", flush=True)
    B, seeds = A.bootstrap_all(cfgs, n_test)
    ext = {key: [extra_effs(cf) for cf in lst] for key, lst in cfgs.items()}

    def base_of(key):
        """The 256-bin baseline of the same method (FHt: the GTT FH 256/3)."""
        return (key[0], "FH" if key[1] == "FHt" else key[1], 256, 3)

    summ = {"input_set": a.input_set, "n_test_events": n_test, "n_boot": A.N_BOOT, "pv_window_cm": A.PV_WINDOW,
            "fh_scan": fh_scan, "e2e_scan": e2e_scan, "missing_pred_files": missing, "epochs": epochs,
            "configs": {}, "best": {}}
    for key in cfgs:
        fhm = key[1] in FH_METHODS
        k = f"{key[0]}:{key[1]}:b{key[2]}:{'w' if fhm else 'k'}{key[3]}"
        d = {"width_cm": width_cm(key[2], key[3]) if fhm else None,
             "kernel_width_cm": width_cm(key[2], key[3]) if not fhm else None,
             "values": {m: dict(zip(("value", "boot_err"), A.val(B, key, m)), seed_sd=A.seed_sd(seeds, key, m))
                        for m in A.METRICS},
             "extra_eff": {w: float(np.mean([e[w] for e in ext[key]])) for w in ext[key][0]},
             "seed_values": seeds[key]}
        if base_of(key) in cfgs and key != base_of(key):
            d["diff_vs_256"] = {m: dict(zip(("value", "boot_err"), A.diff(B, key, base_of(key), m)),
                                        seed_err=A.seed_err_of_diff(seeds, key, base_of(key), m)) for m in A.METRICS}
        if key[1] == "FH":
            d["eff_f01f02_trainonly"] = fh_train_eff[(key[0], key[2], key[3])]
        summ["configs"][k] = d
    # best per (variant, method, nbins) by test PV efficiency (ties -> smaller core RMS)
    for v in variants:
        for m in ("FH", "stock", "mva"):
            for nb in sorted({key[2] for key in cfgs if key[0] == v and key[1] == m}):
                ks = [key for key in cfgs if key[:3] == (v, m, nb)]
                best = max(ks, key=lambda x: (round(B[x]["eff"][0], 6), -B[x]["core_um"][0]))
                r = {"param": best[3], "eff": float(B[best]["eff"][0]), "core_um": float(B[best]["core_um"][0])}
                if m == "FH":
                    r["param_by_f01f02"] = max(ks, key=lambda x: fh_train_eff[(v, nb, x[3])])[3]
                summ["best"][f"{v}:{m}:b{nb}"] = r

    with PdfPages(f"{outdir}/binning.pdf") as pdf:
        A.table_page(pdf, f"Binning study: definitions ({A.S2A.common.INPUT_LABEL}; {n_test} test events)",
                     ["item", "definition"], [
                         ["z range", "[-20.46912512, 20.46912512] cm for every configuration (GTT FH_HistogramParameters)"],
                         ["FH nbins / window", "256 / 512 / 1024 bins (0.160 / 0.080 / 0.040 cm); the peak finder sums "
                                               "`window` consecutive bins (stock 3 = 0.48 cm); z = pT-weighted mean of "
                                               "the window's bin centres; TVA association unchanged"],
                         ["E2E nbins / kernel", "soft-histogram layer with nbins Gaussian bins (sigma = 1 bin) over the same "
                                                "range; kernel = size of all three pattern Conv1D layers (16, 16 filters + "
                                                "1-filter logits; receptive field 3(k-1)+1 bins); no pooling; soft-argmax "
                                                "over all bins; weight/association nets and training unchanged"],
                         ["variants", "V0 = OT tracks; V3 = SP tracks, sub-2 GeV refit tracks admitted (both 2 arms of E2E: "
                                      "stock inputs, + track-MVA scores)"],
                         ["PV eff", f"|z_PV - GenVtx_z| < {A.PV_WINDOW} cm (also 0.05 / 0.10 / 0.30 cm and the curve)"],
                         ["dz core RMS", "iterative 3-sigma clip (about the mean), RMS about zero of the kept events"],
                         ["purity % at FH assoc eff", "association purity with the score threshold set to the association "
                                                      "efficiency of the variant's STOCK FH (256/3) + TVA"],
                         ["errors", "+- paired event bootstrap (diff vs the 256-bin baseline of the same method/arm); "
                                    "[ ] E2E std over 3 seeds / seed error of the difference"],
                         ["missing E2E configs", ", ".join(missing) or "none"]])
        for v in variants:
            for m in ("FH", "FHt", "stock", "mva"):
                keys = sorted([key for key in cfgs if key[:2] == (v, m)], key=lambda x: (x[2], x[3]))
                if not keys:
                    continue
                fhm = m in FH_METHODS
                rows = []
                for key in keys:
                    sd = None if fhm else A.seed_sd
                    r = [key[2], key[3], f"{width_cm(key[2], key[3]):.3f}"]
                    r += [A.fmt(*A.val(B, key, x), x, None if sd is None else sd(seeds, key, x))
                          for x in ("eff", "core_um", "full_um", "auc", "pur_fh")]
                    r += ["/".join(f"{100 * np.mean([e[w] for e in ext[key]]):.1f}" for w in ext[key][0])]
                    if key != base_of(key) and base_of(key) in cfgs:
                        r += [A.fmt(*A.diff(B, key, base_of(key), x), x,
                                    None if fhm else A.seed_err_of_diff(seeds, key, base_of(key), x))
                              for x in ("eff", "core_um", "auc", "pur_fh")]
                    else:
                        r += ["baseline"] * 4
                    if m == "FH":
                        r += [f"{100 * fh_train_eff[(v, key[2], key[3])]:.1f}"]
                    rows.append(r)
                hdr = (["nbins", "window" if fhm else "kernel", "width [cm]", "PV eff %", "dz core [um]",
                        "dz full [um]", "assoc AUC", "pur % @FH eff", "eff % @0.05/0.10/0.30 cm", "d eff %",
                        "d core [um]", "d AUC", "d pur %"] + (["eff % f01/f02 (train-only)"] if m == "FH" else []))
                A.table_page(pdf, f"{nvc.VLABEL[v]}: {MLABEL[m]} vs z binning ({n_test} test events)", hdr, rows,
                             notes=["d = paired difference to the 256-bin baseline (FH and FHt: the GTT FH with 256 bins, "
                                    "3-bin window; E2E: 256 bins, kernel 3, same arm). width = window (FH) or kernel (E2E) "
                                    "x bin width.",
                                    "f01/f02 column: PV eff on the 200 training-only events (independent choice of the "
                                    "window; small sample)." if m == "FH" else
                                    "E2E: seed means; [ ] seed std (values) / seed error (differences)."])
        rows = []
        for k, r in summ["best"].items():
            v, m, nb = k.split(":")
            rows.append([nvc.VLABEL[v], MLABEL[m], nb[1:], r["param"], f"{100 * r['eff']:.1f}", f"{r['core_um']:.0f}",
                         r.get("param_by_f01f02", "-")])
        A.table_page(pdf, "Best configuration per binning (highest test PV efficiency; ties -> smaller core RMS)",
                     ["variant", "method", "nbins", "window / kernel", "PV eff %", "dz core [um]",
                      "FH window chosen on f01/f02"], rows)

        # figures: eff and core RMS vs physical window width (FH), per variant
        fig, axs = plt.subplots(len(variants), 2, figsize=(12, 4.2 * len(variants)), layout="constrained",
                                squeeze=False)
        for i, v in enumerate(variants):
            for nb in sorted({key[2] for key in cfgs if key[:2] == (v, "FH")}):
                ks = sorted([key for key in cfgs if key[:3] == (v, "FH", nb)], key=lambda x: x[3])
                x = [width_cm(nb, key[3]) for key in ks]
                for j, mm, sc in ((0, "eff", 100), (1, "core_um", 1)):
                    y = [B[key][mm][0] * sc for key in ks]
                    e = [np.nanstd(B[key][mm][1:], ddof=1) * sc for key in ks]
                    axs[i, j].errorbar(x, y, yerr=e, marker="o", color=CN[nb], lw=1.5, capsize=2,
                                       label=f"FH {nb} bins ({RANGE / nb:.3f} cm)")
                kt = sorted([key for key in cfgs if key[:3] == (v, "FHt", nb)], key=lambda x: x[3])
                for j, mm, sc in ((0, "eff", 100), (1, "core_um", 1)):
                    axs[i, j].plot([width_cm(nb, key[3]) for key in kt], [B[key][mm][0] * sc for key in kt], marker="s",
                                   mfc="none", ls="none", color=CN[nb], ms=8, label=f"FHt (track-mean z) {nb} bins")
            axs[i, 0].set(xlabel="peak-finder window width [cm]", ylabel=f"PV eff % (|dz| < {A.PV_WINDOW} cm)",
                          title=nvc.VLABEL[v])
            axs[i, 1].set(xlabel="peak-finder window width [cm]", ylabel="dz core RMS [um]", title=nvc.VLABEL[v])
            for ax in axs[i]:
                ax.axvline(0.48, color="k", lw=0.8, ls=":")
                ax.grid(alpha=0.25)
                ax.title.set_fontsize(9)
        fig.suptitle("FastHisto: PV efficiency and dz core RMS vs window width (dotted: stock 3 x 0.16 cm; errors: "
                     "event bootstrap, not paired)", fontsize=10)
        A.legend_below(fig, ncol=3)
        pdf.savefig(fig)
        plt.close(fig)

        # E2E: eff and core vs kernel width, per variant and arm
        e2e_keys = [key for key in cfgs if key[1] in ("stock", "mva")]
        if e2e_keys:
            fig, axs = plt.subplots(len(variants), 2, figsize=(12, 4.2 * len(variants)), layout="constrained",
                                    squeeze=False)
            for i, v in enumerate(variants):
                for arm, ls, mk, mf in (("stock", "--", "s", "none"), ("mva", "-", "o", None)):
                    for nb in sorted({key[2] for key in e2e_keys if key[:2] == (v, arm)}):
                        ks = sorted([key for key in e2e_keys if key[:3] == (v, arm, nb)], key=lambda x: x[3])
                        x = [width_cm(nb, key[3]) for key in ks]
                        for j, mm, sc in ((0, "eff", 100), (1, "core_um", 1)):
                            y = [B[key][mm][0] * sc for key in ks]
                            e = [A.seed_sd(seeds, key, mm) * sc for key in ks]
                            axs[i, j].errorbar(x, y, yerr=e, marker=mk, mfc=mf, ms=7, color=CN[nb], ls=ls, lw=1.5,
                                               capsize=2, label=f"E2E {'+ MVA' if arm == 'mva' else 'stock'}, {nb} bins")
                axs[i, 0].set(xlabel="Conv1D kernel width [cm] (kernel x bin width)", ylabel="PV eff %",
                              title=nvc.VLABEL[v])
                axs[i, 1].set(xlabel="Conv1D kernel width [cm] (kernel x bin width)", ylabel="dz core RMS [um]",
                              title=nvc.VLABEL[v])
                for ax in axs[i]:
                    ax.grid(alpha=0.25)
                    ax.title.set_fontsize(9)
            fig.suptitle("E2E NNVtx: PV efficiency and dz core RMS vs pattern-net kernel width (open squares: stock "
                         "inputs, filled circles: + MVA; error bars: std over 3 seeds)", fontsize=10)
            A.legend_below(fig, ncol=3)
            pdf.savefig(fig)
            plt.close(fig)

        # efficiency vs window curves: baseline + best per binning
        w = np.logspace(-3, 0.5, 70)
        fig, axs = plt.subplots(len(variants), 3, figsize=(16, 4.4 * len(variants)), layout="constrained",
                                squeeze=False, sharey=True)
        for i, v in enumerate(variants):
            for j, m in enumerate(("FH", "stock", "mva")):
                ax = axs[i, j]
                for nb in (256, 512, 1024):
                    b = summ["best"].get(f"{v}:{m}:b{nb}")
                    if b is None:
                        continue
                    for par, ls in ({(3, "-"), (b["param"], "--")} if nb != 256 or m != "FH" else {(3, "-")}):
                        key = (v, m, nb, par)
                        if key not in cfgs or (ls == "--" and par == 3):
                            continue
                        c = np.array([[np.mean(np.abs(cf.dz) < x) for x in w] for cf in cfgs[key]]).mean(0)
                        ax.plot(w * UM, c, color=CN[nb], ls=ls, lw=1.8,
                                label=f"{nb} bins, {'window' if m == 'FH' else 'kernel'} {par}")
                ax.axvline(A.PV_WINDOW * UM, color="k", lw=0.8, ls=":")
                ax.set(xscale="log", xlabel="window |z_PV - z_gen| [um]", title=f"{nvc.VLABEL[v]} | {MLABEL[m]}")
                ax.title.set_fontsize(8)
                ax.grid(alpha=0.25, which="both")
            axs[i, 0].set_ylabel("fraction of events")
        fig.suptitle("PV efficiency vs window: solid = 3-bin window / kernel 3, dashed = best per binning "
                     "(E2E: seed mean)", fontsize=10)
        A.legend_below(fig, ncol=6)
        pdf.savefig(fig)
        plt.close(fig)
    with open(f"{outdir}/summary.json", "w") as fh:
        json.dump(summ, fh, indent=1)
    print(f"wrote {outdir}/binning.pdf and summary.json; missing E2E: {missing}", flush=True)


if __name__ == "__main__":
    main()
