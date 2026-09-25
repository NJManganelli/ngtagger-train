"""Step 1: per-event, per-layer cluster census -- TP-linked vs unlinked, finer truth classes,
and non-angle feature profiles hinting at what unlinked clusters are.

Writes results/counts.json, results/counts_*.md, figs/f1_*.png.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _common as c

RES = c.HERE / "results"
RES.mkdir(exist_ok=True)
c.FIGS.mkdir(exist_ok=True)


def classes(d: pd.DataFrame) -> pd.Series:
    u = d["unlinked"].to_numpy() == 1
    pt = d["tpPt"].to_numpy()
    cf = d["tpChargeFrac"].to_numpy()
    vr = d["tpVr"].to_numpy()
    out = np.full(len(d), "", dtype=object)
    out[u] = "unlinked"
    lk = ~u
    # mutually exclusive, applied in priority order
    shared = lk & (cf < 0.9)
    sec = lk & ~shared & (vr > 1.0)
    out[lk] = "L:pT>=2"
    out[lk & (pt < 2)] = "L:pT 1-2"
    out[lk & (pt < 1)] = "L:pT 0.3-1"
    out[lk & (pt < 0.3)] = "L:pT<0.3"
    out[sec] = "L:secondary (vr>1cm)"
    out[shared] = "L:shared (frac<0.9)"
    return pd.Series(out, index=d.index)


ORDER = ["L:pT>=2", "L:pT 1-2", "L:pT 0.3-1", "L:pT<0.3", "L:secondary (vr>1cm)",
         "L:shared (frac<0.9)", "unlinked"]


def per_event_layer(d: pd.DataFrame, col: str) -> pd.DataFrame:
    g = d.groupby(["evid", "layer", col], observed=True).size().unstack(col, fill_value=0)
    m = g.groupby(level="layer").mean()
    s = g.groupby(level="layer").std()
    return m, s


def main():
    d = c.load()
    nev = d["evid"].nunique()
    d["cls"] = classes(d)
    res = {"n_events": int(nev), "n_clusters": int(len(d))}

    # --- linked vs unlinked per event per layer
    d["lk"] = np.where(d["unlinked"] == 1, "unlinked", "linked")
    m, s = per_event_layer(d, "lk")
    tab = pd.DataFrame({
        "linked/ev": m["linked"].round(1), "unlinked/ev": m["unlinked"].round(1),
        "unlinked frac": (m["unlinked"] / (m["linked"] + m["unlinked"])).round(4),
        "unlinked/ev std": s["unlinked"].round(1)})
    tot = d.groupby("lk").size() / nev
    tab.loc["all"] = [round(tot["linked"], 1), round(tot["unlinked"], 1),
                      round(tot["unlinked"] / tot.sum(), 4), np.nan]
    res["linked_unlinked"] = tab.reset_index().to_dict(orient="list")
    (RES / "counts_linked_unlinked.md").write_text(c.md(tab))

    # --- finer classes per event per layer
    m, _ = per_event_layer(d, "cls")
    m = m[[o for o in ORDER if o in m.columns]]
    m.loc["all"] = m.sum()
    frac = m.div(m.sum(axis=1), axis=0)
    (RES / "counts_classes.md").write_text(
        "Mean clusters per event (classes mutually exclusive, priority: shared > secondary > pT)\n\n"
        + c.md(m.round(1)) + "\n\nFraction of all clusters\n\n" + c.md(frac.round(4)))
    res["classes_per_event"] = m.round(2).reset_index().to_dict(orient="list")

    # --- pileup/bunch-crossing origin (only TPs with pT >= 1 GeV are in L1TTP)
    lk1 = d[(d["unlinked"] == 0) & d["tpBx"].notna()]
    org = np.where(lk1["tpBx"] != 0, "out-of-time PU (bx!=0)",
                   np.where(lk1["tpEvt"] == 0, "hard scatter (bx0,evt0)", "in-time PU (bx0)"))
    o = pd.Series(org).value_counts()
    res["origin_pt_ge1"] = {k: int(v) for k, v in o.items()}
    res["origin_pt_ge1_frac"] = {k: float(v / o.sum()) for k, v in o.items()}
    bx = lk1["tpBx"].value_counts().sort_index()
    res["bx_counts_pt_ge1"] = {int(k): int(v) for k, v in bx.items()}
    pdg = lk1["tpPdgId"].abs().value_counts().head(8)
    res["pdg_pt_ge1"] = {int(k): int(v) for k, v in pdg.items()}

    # --- TP pT spectrum near the TrackingParticle selection edge (ptMinTP = 0.1 GeV default)
    pt = d.loc[d["unlinked"] == 0, "tpPt"].to_numpy()
    res["linked_tpPt_below_0p1_frac"] = float(np.mean(pt < 0.1))
    res["linked_tpPt_0p1_0p12_frac"] = float(np.mean((pt >= 0.1) & (pt < 0.12)))
    res["linked_tpPt_quantiles"] = {q: float(np.quantile(pt, q)) for q in (0.001, 0.01, 0.05, 0.5)}

    # --- feature profiles by class
    feats = ["charge", "size", "sizeX", "sizeY", "qPerPix", "absZ", "localX", "localY"]
    prof = d.groupby("cls")[feats].median().reindex(ORDER)
    prof["size==1"] = d.groupby("cls")["size"].apply(lambda x: (x == 1).mean()).reindex(ORDER)
    prof["sizeX>=6"] = d.groupby("cls")["sizeX"].apply(lambda x: (x >= 6).mean()).reindex(ORDER)
    prof["charge>100k"] = d.groupby("cls")["charge"].apply(lambda x: (x > 1e5).mean()).reindex(ORDER)
    prof["|localX|>0.75cm"] = d.groupby("cls")["localX"].apply(lambda x: (x.abs() > 0.75).mean()).reindex(ORDER)
    (RES / "counts_profiles.md").write_text(c.md(prof.round(4)))
    res["profiles"] = prof.round(5).reset_index().to_dict(orient="list")

    (RES / "counts.json").write_text(json.dumps(res, indent=1, default=float))

    # --- figures: feature shapes, unlinked vs linked subclasses
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.5))
    spec = [("sizeX", np.arange(0.5, 20.5, 1), "cluster sizeX [pixels, r-phi]", False),
            ("sizeY", np.arange(0.5, 20.5, 1), "cluster sizeY [pixels, z]", False),
            ("charge", np.logspace(3.5, 6, 60), "cluster charge [e, as stored]", True),
            ("qPerPix", np.logspace(3, 5, 60), "charge / size [e per fired pixel]", True),
            ("absZ", np.linspace(0, 30, 61), "|global z| [cm]", False),
            ("localX", np.linspace(-0.9, 0.9, 61), "local x on module [cm]", False)]
    show = ["L:pT>=2", "L:pT 0.3-1", "L:pT<0.3", "L:secondary (vr>1cm)", "unlinked"]
    colors = ["#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#d62728"]
    for ax, (f, bins, lab, logx) in zip(axs.flat, spec):
        for cl_, col in zip(show, colors):
            v = d.loc[d["cls"] == cl_, f].to_numpy()
            ax.hist(v, bins=bins, histtype="step", density=True, lw=1.8 if cl_ == "unlinked" else 1.2,
                    color=col, label=cl_)
        ax.set_xlabel(lab)
        ax.set_ylabel("fraction per bin width [1/unit of x]")
        if logx:
            ax.set_xscale("log")
        ax.set_yscale("log" if f in ("sizeX", "sizeY", "charge") else "linear")
    axs[0, 0].legend(fontsize=8)
    fig.suptitle(f"PU200 ttbar, TBPX L1-L4, {nev} events: non-angle features by truth class (unit-normalised)")
    fig.tight_layout()
    fig.savefig(c.FIGS / "f1_features_by_class.png", dpi=110)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(pt, bins=np.logspace(-2, 2, 81), histtype="step", color="k")
    ax.axvline(0.1, color="r", ls="--", label="TP selection ptMinTP = 0.1 GeV (CMSSW default)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("dominant-TP pT [GeV]"); ax.set_ylabel("TP-linked clusters per bin [count]")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(c.FIGS / "f1_linked_tpPt.png", dpi=110)
    print(json.dumps({k: res[k] for k in ("origin_pt_ge1_frac", "linked_tpPt_below_0p1_frac",
                                         "linked_tpPt_0p1_0p12_frac", "bx_counts_pt_ge1")}, indent=1))
    print(c.md(tab)); print(c.md(m.round(1))); print(c.md(prof.round(3)))


if __name__ == "__main__":
    main()
