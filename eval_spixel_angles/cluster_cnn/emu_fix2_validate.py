"""Tests of two angle-source changes on RelVal file 3 (test events; files 1-2 hold the training events). TESTS ONLY.

CHANGE 1: the emu sigma table (spix_emu_sigma_v3.json) is keyed also on the cluster size along the angle's axis.
CHANGE 2: payload mode rebuilt in physical units (spix_angle_response_Conv1D_Full-2bit_v5.json), and the rec-hit producer no
          longer clears hasAlpha/hasBeta at |cot| > 12 (the refit applies its own cap).

  emu_fix2_validate.py --runs DIR --ntuple NTUPLE.root --emu-old OLD_EMU.root [--perf DIR]

DIR holds the nanos (100 events of file 3, refit useAngles = alphaBeta, 4 threads):
  out_emu_v2tab.root     emu, NEW code, the old (v2) sigma table     out_emu_v3tab.root  emu, NEW code, the v3 table
  out_pay_v4_oldcode.root payload v4 (old units), OLD code            out_pay_v4.root     payload v4, NEW code (cap change only)
  out_pay_v5.root        payload v5 (physical units), NEW code
OLD_EMU = the fix-round emu nano of the same events (old code, v2 table): the new code with the v2 table must reproduce it.
NTUPLE = the fix-round ClusterAdcNtuple (emu, emuDebug, first 20 events of file 3): the leak-probe features (tiny-CNN outputs, CPE).
Figures (vector PDF, one shared legend outside the axes, statistics on table pages): figs/emu_fix2_{sigma,leak,refit,payload}.pdf.
Results: results/emu_fix2_validation.json. tpEmuClass is TRUTH-ONLY: used to split, never as a feature.
"""
import argparse, os, sys, json, numpy as np, pandas as pd, uproot, awkward as ak, matplotlib, correctionlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2_common import core_fit
from emu_build_library import load_dump

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = lambda n: os.path.join(HERE, "figs", n)
p = argparse.ArgumentParser()
p.add_argument("--runs", required=True); p.add_argument("--ntuple", required=True); p.add_argument("--emu-old", required=True)
p.add_argument("--perf", default=None)
p.add_argument("--sigma-v2", default="/Volumes/WDMac/smartpixels_scratch/emu_fix/spix_emu_sigma_v2.json")
p.add_argument("--sigma-v3", default="/Volumes/WDMac/smartpixels_scratch/emu_fix2/sigma/sigma_own.json")
a = p.parse_args()
R = lambda t: os.path.join(a.runs, f"out_{t}.root")
deg = lambda c: np.degrees(np.arctan(np.asarray(c, np.float64)))
PC = "L1TSmartPixelsCluster_"
GROUPS = ["IL2-4", "IL1", "ID1-4"]
grp = lambda s: np.where(np.asarray(s) > 10, "ID1-4", np.where(np.asarray(s) == 1, "IL1", "IL2-4"))
SURF = [1, 2, 3, 4, 11, 12, 13, 14]
SNAME = {s: (f"IL{s}" if s < 10 else f"ID{s - 10}") for s in SURF}
OUT = {}
PAL = ["#2a78d6", "#e8742c", "#1f9e76", "#b04a9c", "#c9a227", "#6b4c9a", "#d1495b", "#3d7d7d", "#1b1b1b"]


def table_pages(pdf, df, title, per=30):
    for i in range(0, max(len(df), 1), per):
        t = df.iloc[i:i + per]
        f, ax = plt.subplots(figsize=(min(26, 2 + 1.6 * len(t.columns)), 0.33 * len(t) + 1.4)); ax.axis("off")
        tb = ax.table(cellText=t.astype(str).values, colLabels=list(t.columns), loc="center", cellLoc="left")
        tb.auto_set_font_size(False); tb.set_fontsize(7.5); tb.auto_set_column_width(list(range(len(t.columns))))
        ax.set_title(title + (f" ({i // per + 1})" if len(df) > per else ""), fontsize=10); pdf.savefig(f, bbox_inches="tight"); plt.close(f)


def fmt(df):
    return df.apply(lambda c: c.map(lambda x: f"{x:.4g}" if isinstance(x, (float, np.floating)) else x))


CCOLS = ["surface", "sizeX", "sizeY", "size", "charge", "globalR", "globalZ", "localX", "localY", "sigX", "sigY", "localCotAlpha",
         "localCotBeta", "sigAlpha", "sigBeta", "hasAlpha", "hasBeta", "satAlpha", "satBeta", "tpKey", "tpLocalCotAlpha",
         "tpLocalCotBeta", "tpEmuClass", "detId", "idxInDet"]


def read_clusters(f, cols=CCOLS):
    arr = uproot.open(f)["Events"].arrays([PC + c for c in cols] + ["run", "event"], library="ak")
    n = ak.to_numpy(ak.num(arr[PC + cols[0]]))
    df = pd.DataFrame({c: ak.to_numpy(ak.flatten(arr[PC + c])) for c in cols})
    df["event"] = np.repeat(ak.to_numpy(arr["event"]), n); df["run"] = np.repeat(ak.to_numpy(arr["run"]), n)
    df["group"] = grp(df.surface.values)
    return df.sort_values(["run", "event", "detId", "idxInDet"], kind="stable").reset_index(drop=True)


def compare(f1, f2, only_nonSP):
    """branch-by-branch identity of two nanos on common events (sorted by event: multi-thread output order differs)"""
    t1, t2 = uproot.open(f1)["Events"], uproot.open(f2)["Events"]
    e1 = t1["event"].array(library="np"); e2 = t2["event"].array(library="np")
    o1, o2 = np.argsort(e1, kind="stable"), np.argsort(e2, kind="stable")
    common = sorted(set(e1) & set(e2))
    s1, s2 = o1[np.isin(e1[o1], common)], o2[np.isin(e2[o2], common)]
    diff, same = [], 0
    for k in t1.keys():
        if k not in t2.keys() or (only_nonSP == ("SmartPixels" in k)):
            continue
        b1 = t1[k].array(library="ak")[s1]; b2 = t2[k].array(library="ak")[s2]
        try:
            x1 = ak.to_numpy(ak.flatten(b1, axis=None)); x2 = ak.to_numpy(ak.flatten(b2, axis=None))
            ok = x1.shape == x2.shape and bool(np.all((x1 == x2) | (np.isnan(x1) & np.isnan(x2)) if x1.dtype.kind == "f" else x1 == x2))
        except Exception:
            ok = False
        same += ok
        if not ok:
            diff.append(k)
    return dict(n_common_events=len(common), n_branches_compared=same + len(diff), n_identical=same, differing=diff,
                only_in_one=sorted(set(t2.keys()) ^ set(t1.keys()))[:40])


def pull_rows(D, tag):
    rows = []
    T = (D.tpEmuClass == 1).values
    for s in SURF:
        for v, nm in (("Alpha", "alpha"), ("Beta", "beta")):
            m = (D.surface == s).values & T & (np.abs(D[f"localCot{v}"].values) <= 12) & (D[f"sat{v}"].values == 0)
            pl = (D[f"localCot{v}"].values[m] - D[f"tpLocalCot{v}"].values[m]) / D[f"sig{v}"].values[m]
            q16, q84 = np.quantile(pl, [0.16, 0.84])
            rows.append(dict(table=tag, angle=nm, surface=SNAME[s], n=int(m.sum()), q_width=(q84 - q16) / 2,
                             median=float(np.median(pl)), frac_abs_gt3=float(np.mean(np.abs(pl) > 3))))
    return rows


# ======================= refit =======================
RT, RH = "L1TSmartPixelsTrackSeedlessRefitAAAA_", "L1TSmartPixelsRefitHitSeedlessRefitAAAA_"
TRK = ["pt", "d0", "z0", "spixRefitPerformed", "spixNAcceptedHits", "spixChi2IncXTot", "spixChi2IncYTot",
       "spixChi2IncAlphaTot", "spixChi2IncBetaTot"]
HIT = ["hitAccepted", "hasAlpha", "hasBeta", "selClusterIdx", "selHitClass"]


def refit(f):
    ev = uproot.open(f)["Events"].arrays([RT + c for c in TRK] + [RH + c for c in HIT] + ["L1TTrack_tp_pt", "L1TTrack_tp_d0",
                                          "L1TTrack_tp_z0", PC + "localCotAlpha", PC + "localCotBeta", "event"], library="ak")
    ev = ev[np.argsort(ak.to_numpy(ev["event"]), kind="stable")]
    fl = lambda x: ak.to_numpy(ak.flatten(x))
    T = pd.DataFrame({c: fl(ev[RT + c]) for c in TRK})
    for c in ("tp_pt", "tp_d0", "tp_z0"):
        T["base_" + c] = fl(ev["L1TTrack_" + c])
    T["chi2IT"] = T.spixChi2IncXTot + T.spixChi2IncYTot + T.spixChi2IncAlphaTot + T.spixChi2IncBetaTot
    acc = ev[RH + "hitAccepted"] == 1
    sel = ev[RH + "selClusterIdx"]; sel0 = ak.where(sel >= 0, sel, 0)
    H = pd.DataFrame({"cls": fl(ev[RH + "selHitClass"][acc]), "hasA": fl(ev[RH + "hasAlpha"][acc]).astype(bool),
                      "hasB": fl(ev[RH + "hasBeta"][acc]).astype(bool), "cotA": fl(ev[PC + "localCotAlpha"][sel0][acc]),
                      "cotB": fl(ev[PC + "localCotBeta"][sel0][acc])})
    return T, H


def refit_summary(tag, f):
    T, H = refit(f)
    rows = []
    m = (T.spixRefitPerformed == 1) & (T.base_tp_pt > 0)
    for nm, r in (("d0 [um]", 1e4 * (T.d0 - T.base_tp_d0)), ("z0 [um]", 1e4 * (T.z0 - T.base_tp_z0)),
                  ("pT (reco - true) / true", (T.pt - T.base_tp_pt) / T.base_tp_pt)):
        for lo, hi in ((0, 1e9), (2, 5), (5, 1e9)):
            mm = m & (T.base_tp_pt >= lo) & (T.base_tp_pt < hi); x = r[mm].values
            q16, q84 = np.quantile(x, [0.16, 0.84])
            rows.append(dict(run=tag, quantity=nm, tp_pt=f"[{lo}, {hi if hi < 1e9 else 'inf'})", n=int(mm.sum()),
                             q68_halfwidth=(q84 - q16) / 2, median=float(np.median(x))))
    for nm, x in (("chi2 IT total", T.chi2IT[m]), ("chi2 alpha", T.spixChi2IncAlphaTot[m]), ("chi2 beta", T.spixChi2IncBetaTot[m])):
        rows.append(dict(run=tag, quantity=nm, tp_pt="all", n=int(m.sum()), median=float(np.median(x)), mean=float(np.mean(x)),
                         q90=float(np.quantile(x, 0.9)), q99=float(np.quantile(x, 0.99)), frac_gt_100=float(np.mean(x > 100))))
    cap = dict(run=tag, accepted_hits=len(H), purity_own_TP=float(np.mean(H.cls == 0)), frac_other_TP=float(np.mean(H.cls == 1)),
               frac_no_TP=float(np.mean(H.cls == 2)), hasAlpha_frac=float(H.hasA.mean()), hasBeta_frac=float(H.hasB.mean()),
               cap_rule_violations=int(((H.hasA & (np.abs(H.cotA) > 12)) | (H.hasB & (np.abs(H.cotB) > 12))).sum()),
               accepted_with_abs_cot_gt12=int(((np.abs(H.cotA) > 12) | (np.abs(H.cotB) > 12)).sum()),
               refit_tracks=int(m.sum()))
    return rows, cap, T[m]


def refit_pdf(name, runs, title):
    rows, caps, trk = [], [], {}
    for tag, f in runs:
        r, c, t = refit_summary(tag, f); rows += r; caps.append(c); trk[tag] = t
    RF, CP = pd.DataFrame(rows), pd.DataFrame(caps)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5), layout="constrained")
    specs = [("d0 - true d0 [um]", lambda T: 1e4 * (T.d0 - T.base_tp_d0), np.linspace(-300, 300, 121)),
             ("z0 - true z0 [um]", lambda T: 1e4 * (T.z0 - T.base_tp_z0), np.linspace(-1500, 1500, 121)),
             ("(pT - true) / true", lambda T: (T.pt - T.base_tp_pt) / T.base_tp_pt, np.linspace(-0.3, 0.3, 121)),
             ("chi2 IT total (x + y + alpha + beta)", lambda T: T.chi2IT, np.logspace(-2, 5, 141)),
             ("chi2 alpha", lambda T: T.spixChi2IncAlphaTot, np.logspace(-3, 5, 141)),
             ("chi2 beta", lambda T: T.spixChi2IncBetaTot, np.logspace(-3, 5, 141))]
    for ax, (nm, fn, e) in zip(axes.flat, specs):
        for i, (tag, T) in enumerate(trk.items()):
            x = fn(T).values
            h, _ = np.histogram(np.clip(x, e[0] * (1 + 1e-9) if e[0] > 0 else e[0] + 1e-9, e[-1] * (1 - 1e-9)), bins=e)
            ax.stairs(h / max(h.sum(), 1), e, color=PAL[i], lw=1.6, ls=["-", "--", ":", "-."][i % 4], label=tag)
        ax.set_title(nm, fontsize=10); ax.set_yscale("log"); ax.grid(True, color="#eeeeee"); ax.set_ylabel("fraction of refit tracks")
        ax.set_xlabel(nm + (" (ends = overflow)" if e[0] <= 0 else ""))
        if e[0] > 0:
            ax.set_xscale("log")
    h_, l_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="outside lower center", ncol=len(runs), fontsize=9, frameon=False)
    fig.suptitle(title, fontsize=10)
    with PdfPages(FIG(name)) as pdf:
        pdf.savefig(fig); plt.close(fig)
        table_pages(pdf, fmt(RF), "Refit (TP-matched refit tracks): residual q68 half-widths and medians vs the TP; chi2 summaries")
        table_pages(pdf, fmt(CP), "Accepted refit hits: selected-hit purity (selHitClass 0 = own TP, 1 = other TP, 2 = no TP), "
                                  "has* rates, and the refit |cot| <= 12 cap (violations must be 0)")
    return RF, CP


# ======================= leak =======================
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score


def auc(X, y, half, seed):
    rng = np.random.default_rng(seed); tri = np.where(half)[0]; tri = rng.choice(tri, min(len(tri), 1_000_000), replace=False)
    X = np.where(np.isfinite(X), X, -999.0)
    clf = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=31, learning_rate=0.1, random_state=1).fit(X[tri], y[tri])
    return float(roc_auc_score(y[~half], clf.predict_proba(X[~half])[:, 1]))


NC = ["run", "event", "detId", "idxInDet", "subdet", "layer", "disk", "sizeX", "sizeY", "size", "charge", "globalR", "globalZ",
      "localX", "localY", "originLocalX", "originLocalY", "zOutward", "linkClass", "emuCnnA", "emuCnnB", "emuCnnSigA",
      "emuCnnSigB", "pCotAlpha", "pCotBeta"]
N = uproot.open(a.ntuple)["clusterAdcNtuple/clusters"].arrays(NC, library="pd")
N["surface"] = np.where(N.subdet == 1, N.layer, 10 + N.disk).astype(np.int64); N["group"] = grp(N.surface.values)
N = N[N.surface.isin(SURF)].reset_index(drop=True)
yN = (N.linkClass.values == 1).astype(int); halfN = (N.event.values % 2 == 0)
F1 = np.c_[N.sizeX, N.sizeY, N["size"], N.charge, N.surface, N.globalR, np.abs(N.globalZ), N.localX, N.localY, N.zOutward]
STRONG = np.c_[F1, N.emuCnnA, N.emuCnnB, N.emuCnnSigA, N.emuCnnSigB, (N.localX - N.originLocalX) * 1e4, (N.localY - N.originLocalY) * 1e4]


def probe_gains(extra, label, rows, groups=GROUPS, sel=None):
    """AUC(unlinked vs linked) gain of the extra columns over the F1 and strong probes, per group; even/odd events and swapped"""
    for g in groups:
        gm = (N.group == g).values & (np.ones(len(N), bool) if sel is None else sel)
        r = dict(test=label, group=g, n=int(gm.sum()), frac_unlinked=float(yN[gm].mean()))
        for k, X in (("F1", F1), ("strong", STRONG)):
            b = auc(X[gm], yN[gm], halfN[gm], 3); w_ = auc(np.c_[X, extra][gm], yN[gm], halfN[gm], 3)
            bs = auc(X[gm], yN[gm], ~halfN[gm], 4); ws = auc(np.c_[X, extra][gm], yN[gm], ~halfN[gm], 4)
            r[f"{k}_base"] = b; r[f"{k}_gain"] = w_ - b; r[f"{k}_gain_swapped"] = ws - bs
        rows.append(r); print("leak", r, flush=True)


def sig_eval(path, ang, cot):
    """a sigma table evaluated on the ntuple clusters, inputs by name"""
    c = correctionlib.CorrectionSet.from_file(path)[f"spix_emu_{ang}_sigma"]
    args = [N[i.name].values.astype(np.int64) if i.name in ("surface", "sizeX", "sizeY") else np.abs(cot) for i in c.inputs]
    return c.evaluate(*args)


def nano_probe(D, label, rows):
    base = ["sizeX", "sizeY", "size", "charge", "surface", "globalR", "globalZ", "localX", "localY", "sigX", "sigY"]
    angc = ["localCotAlpha", "localCotBeta", "sigAlpha", "sigBeta", "hasAlpha", "hasBeta", "satAlpha", "satBeta"]
    yn = (D.tpKey.values < 0).astype(int); hn = (D.event.values % 2 == 0)
    for g in GROUPS:
        gm = (D.group == g).values
        b = auc(D[base].values.astype(np.float64)[gm], yn[gm], hn[gm], 5)
        w_ = auc(np.clip(D[base + angc].values.astype(np.float64), -1e6, 1e6)[gm], yn[gm], hn[gm], 5)
        rows.append(dict(test=label, group=g, n=int(gm.sum()), auc_no_angles=b, auc_with_all_angle_columns=w_, gain=w_ - b))
        print("nano leak", rows[-1], flush=True)


# ============================================================================================
# CHANGE 1: the emu sigma table keyed on cluster size
# ============================================================================================
E2, E3 = read_clusters(R("emu_v2tab")), read_clusters(R("emu_v3tab"))
OUT["emu_refactor_check_v2table_newcode_vs_oldcode_SP"] = compare(R("emu_v2tab"), a.emu_old, False)
OUT["emu_v3_vs_v2_SP"] = compare(R("emu_v2tab"), R("emu_v3tab"), False)
OUT["emu_v3_vs_v2_nonSP"] = compare(R("emu_v2tab"), R("emu_v3tab"), True)
same_cot = bool((E2.localCotAlpha.values == E3.localCotAlpha.values).all() and (E2.localCotBeta.values == E3.localCotBeta.values).all())
OUT["emu_v3_emitted_cot_identical_to_v2"] = same_cot
PY = pd.concat([pd.read_csv(os.path.join(HERE, f"results/emu_sigma_pulls_{t}.csv")) for t in ("v2", "own", "both")])
PY["table"] = PY.table.map({"v2": "v2 (surface x |cot|)", "own": "v3 (surface x own size x |cot|)",
                            "both": "alt (surface x sizeX x sizeY x |cot|)"})
PY["surface"] = PY.surface.map(SNAME)
PYs = PY[PY.selection == "|cot|<=12, not saturated"].drop(columns="selection")
PIV = PYs.pivot_table(index=["angle", "surface", "route"], columns="table", values=["q_width", "frac_abs_gt3"], sort=False)
PIV.columns = [f"{v} {t.split(' ')[0]}" for v, t in PIV.columns]
PIV = PIV.reset_index()
SUMM = []
for t, g in PYs.groupby("table"):
    for rt, x in g.groupby("route"):
        SUMM.append(dict(table=t, route=rt, mean_abs_log_qwidth=float(np.mean(np.abs(np.log(x.q_width)))),
                         max_abs_log_qwidth=float(np.max(np.abs(np.log(x.q_width)))), mean_frac_gt3=float(x.frac_abs_gt3.mean()),
                         max_frac_gt3=float(x.frac_abs_gt3.max())))
SUMM = pd.DataFrame(SUMM)
CM = pd.DataFrame(pull_rows(E2, "v2") + pull_rows(E3, "v3"))
CMp = CM.pivot_table(index=["angle", "surface"], columns="table", values=["q_width", "frac_abs_gt3", "median"], sort=False)
CMp.columns = [f"{v} {t}" for v, t in CMp.columns]; CMp = CMp.reset_index()
ST3 = pd.read_csv(os.path.join(HERE, "results/emu_sigma_table_own.csv"))
OUT["pulls_python"] = PYs.to_dict("records"); OUT["pulls_summary"] = SUMM.to_dict("records"); OUT["pulls_cmssw_classT"] = CM.to_dict("records")

with PdfPages(FIG("emu_fix2_sigma.pdf")) as pdf:
    SZ = PY[PY.selection.str.match(r"size[XY] = ")].copy()
    SZ["size"] = SZ.selection.str.extract(r"= (\d+)").astype(int)
    for ang, own in (("alpha", "sizeX"), ("beta", "sizeY")):
        fig, axes = plt.subplots(2, 4, figsize=(18, 8.5), layout="constrained", sharey=True)
        for ax, s in zip(axes.flat, SURF):
            k = 0
            for t in ("v2 (surface x |cot|)", "v3 (surface x own size x |cot|)"):
                for rt, ls in (("T", "-"), ("C", "--")):
                    z = SZ[(SZ.table == t) & (SZ.route == rt) & (SZ.angle == ang) & (SZ.surface == SNAME[s])].sort_values("size")
                    ax.plot(z["size"], z.q_width, color=PAL[k], ls=ls, marker="o", ms=3.5, lw=1.5,
                            label=f"{t}, {'class-T' if rt == 'T' else 'class-C'} route")
                    k += 1
            ax.axhline(1, color="#9e9e9e", lw=0.8); ax.set_yscale("log"); ax.grid(True, color="#eeeeee")
            ax.set_title(f"{SNAME[s]}, cot{ang.capitalize()}", fontsize=10); ax.set_xlabel(f"{own} (8 = 8 or more)")
            ax.set_ylabel("pull q-width (q84 - q16)/2")
        h_, l_ = axes[0, 0].get_legend_handles_labels()
        fig.legend(h_, l_, loc="outside lower center", ncol=2, fontsize=9, frameon=False)
        fig.suptitle(f"Emu sigma table, pulls (emitted - true cot)/sigma vs the cluster size, python reference on the TEST events "
                     f"(class-T clusters through both routes), |cot| <= 12, not saturated", fontsize=10)
        pdf.savefig(fig); plt.close(fig)
    # the v3 table itself: sigma vs |cot| per own-size bin, IL2 and IL1 / ID1 as examples
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), layout="constrained")
    for i, (ang, own) in enumerate((("alpha", "sizeX"), ("beta", "sizeY"))):
        for j, s in enumerate((1, 2, 11)):
            ax = axes[i, j]; g = ST3[(ST3.angle == ang) & (ST3.surface == s)]
            for k, ((lo, hi), gg) in enumerate(g.groupby([f"{own}_lo", f"{own}_hi"], sort=True)):
                if k >= len(PAL):
                    break
                x = np.r_[gg.lo.values, gg.hi.values[-1]]
                ax.stairs(gg.sigma.values, x, color=PAL[k], lw=1.5, label=f"{own} bin {k + 1}")
                ax.text(x[-1], gg.sigma.values[-1], f" {int(lo + 0.5)}-{'' if np.isinf(hi) else int(hi - 0.5)}", fontsize=6, color=PAL[k])
            ST2 = pd.read_csv(os.path.join(HERE, "results/emu_sigma_table_none.csv"))
            g2 = ST2[(ST2.angle == ang) & (ST2.surface == s)]
            ax.stairs(g2.sigma.values, np.r_[g2.lo.values, g2.hi.values[-1]], color="#1b1b1b", lw=2.2, ls=":", label="v2 (no size key)")
            ax.set_xscale("symlog", linthresh=0.05); ax.set_yscale("log"); ax.grid(True, color="#eeeeee")
            ax.set_title(f"{SNAME[s]}, cot{ang.capitalize()} (first {min(len(PAL), g[f'{own}_lo'].nunique())} {own} bins shown)", fontsize=10)
            ax.set_xlabel(f"|reported cot{ang.capitalize()}|"); ax.set_ylabel("sigma(cot) = (q84 - q16)/2")
    h_, l_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="outside lower center", ncol=10, fontsize=9, frameon=False)
    fig.suptitle("v3 sigma table (keyed on the size along the angle's axis) vs v2; labels at the curve ends = the size range", fontsize=10)
    pdf.savefig(fig); plt.close(fig)
    table_pages(pdf, fmt(PIV), "Pulls, python reference on the TEST events (class-T clusters through both routes; 'T+C inclusive' = weighted by the "
                               "test-event class fractions of each (surface, own size) cell), |cot| <= 12, not saturated: q-width and fraction beyond 3", per=34)
    table_pages(pdf, fmt(SUMM), "Summary over the 8 surfaces x 2 angles: mean / max |log q-width| (0 = ideal) and the fraction beyond 3")
    table_pages(pdf, fmt(CMp), "Pulls in CMSSW (nano, file 3, 100 events), class T (TRUTH-ONLY split), |cot| <= 12, not saturated: v2 vs v3 table")
    table_pages(pdf, pd.DataFrame([(k, json.dumps(v)[:300]) for k, v in (("new code + v2 table vs fix-round nano, SP branches",
                                   OUT["emu_refactor_check_v2table_newcode_vs_oldcode_SP"]), ("v3 vs v2 table, SP branches", OUT["emu_v3_vs_v2_SP"]),
                                   ("v3 vs v2 table, non-SP branches", OUT["emu_v3_vs_v2_nonSP"]), ("emitted cot identical", same_cot))],
                                  columns=["check", "result"]), "Checks")

LEAK = []
cotA, cotB = N.pCotAlpha.values, N.pCotBeta.values
angles = np.c_[deg(cotA), deg(cotB)]
probe_gains(angles, "emu: emitted angles only (unchanged by the table)", LEAK)
for tab, path in (("v2", a.sigma_v2), ("v3", a.sigma_v3)):
    probe_gains(np.c_[angles, sig_eval(path, "alpha", cotA), sig_eval(path, "beta", cotB)], f"emu: emitted angles + sigmas ({tab} table)", LEAK)
NANO_LEAK = []
nano_probe(E2, "emu nano, v2 table (all angle columns incl. sigma)", NANO_LEAK)
nano_probe(E3, "emu nano, v3 table (all angle columns incl. sigma)", NANO_LEAK)
EMU_RF, EMU_CP = refit_pdf("emu_fix2_refit.pdf", [("emu, v2 sigma table", R("emu_v2tab")), ("emu, v3 sigma table (size key)", R("emu_v3tab"))],
                           "seedlessRefit AAAA, emu angles, file 3 (100 events): v2 vs v3 sigma table, TP-matched refit tracks")
OUT["refit_emu"] = EMU_RF.to_dict("records"); OUT["refit_emu_hits"] = EMU_CP.to_dict("records")

# ============================================================================================
# CHANGE 2: payload in physical units + no |cot| gate in the producer
# ============================================================================================
PAYS = [("payload v4 (old units), old code", "pay_v4_oldcode"), ("payload v4, new code (cap change only)", "pay_v4"),
        ("payload v5 (physical units), new code", "pay_v5")]
P = {lab: read_clusters(R(t)) for lab, t in PAYS}
OUT["payload_v4_oldcode_vs_newcode_SP"] = compare(R("pay_v4_oldcode"), R("pay_v4"), False)
OUT["payload_v4_oldcode_vs_newcode_nonSP"] = compare(R("pay_v4_oldcode"), R("pay_v4"), True)
d0, d1 = P[PAYS[0][0]], P[PAYS[1][0]]
OUT["payload_cap_change_detail"] = dict(
    same_rows=bool(len(d0) == len(d1) and (d0.detId.values == d1.detId.values).all()),
    cot_identical=bool(np.array_equal(d0.localCotAlpha.values, d1.localCotAlpha.values) and np.array_equal(d0.localCotBeta.values, d1.localCotBeta.values)),
    hasAlpha_changed=int((d0.hasAlpha.values != d1.hasAlpha.values).sum()), hasBeta_changed=int((d0.hasBeta.values != d1.hasBeta.values).sum()),
    hasAlpha_changed_and_abs_cot_gt12=int(((d0.hasAlpha.values != d1.hasAlpha.values) & (np.abs(d1.localCotAlpha.values) > 12)).sum()),
    hasBeta_changed_and_abs_cot_gt12=int(((d0.hasBeta.values != d1.hasBeta.values) & (np.abs(d1.localCotBeta.values) > 12)).sum()))
HAS = []
for lab, D in P.items():
    for g in GROUPS:
        for lk in (True, False):
            m = (D.group == g).values & ((D.tpKey.values >= 0) == lk)
            HAS.append(dict(run=lab, group=g, linked=lk, n=int(m.sum()), hasAlpha=float(D.hasAlpha.values[m].mean()),
                            hasBeta=float(D.hasBeta.values[m].mean()),
                            frac_abs_cotA_gt12=float(np.mean(np.abs(D.localCotAlpha.values[m & (D.hasAlpha.values == 1)]) > 12)) if (m & (D.hasAlpha.values == 1)).any() else np.nan,
                            frac_abs_cotB_gt12=float(np.mean(np.abs(D.localCotBeta.values[m & (D.hasBeta.values == 1)]) > 12)) if (m & (D.hasBeta.values == 1)).any() else np.nan))
HAS = pd.DataFrame(HAS); OUT["payload_has_rates"] = HAS.to_dict("records")

# closure vs the RESCALED dump, per barrel surface (the payload gives disc clusters no angle)
DD = load_dump()
rD = {"A": DD.rA.values, "B": DD.rB.values}
cellD = np.floor(DD.tA.values / 5).astype(int) * 1000 + np.floor(DD.tB.values / 5).astype(int)
CL, CLOS = [], []
e = np.linspace(-30, 30, 121)
with PdfPages(FIG("emu_fix2_payload.pdf")) as pdf:
    for v, nm in (("A", "alpha"), ("B", "beta")):
        fig, axes = plt.subplots(1, 4, figsize=(20, 5.8), layout="constrained")
        for ax, s in zip(axes, (1, 2, 3, 4)):
            D5 = P[PAYS[2][0]]
            m5 = (D5.surface.values == s) & (D5.tpKey.values >= 0) & (D5[f"has{'Alpha' if v == 'A' else 'Beta'}"].values == 1)
            tA, tB = deg(D5.tpLocalCotAlpha.values[m5]), deg(D5.tpLocalCotBeta.values[m5])
            pc = np.floor(tA / 5).astype(int) * 1000 + np.floor(tB / 5).astype(int)
            wP = pd.Series(pc).value_counts(normalize=True); wDd = pd.Series(cellD).value_counts(normalize=True)
            wrow = (pd.Series(cellD).map(wP).fillna(0) / pd.Series(cellD).map(wDd)).values
            covered = float(pd.Series(pc).isin(wDd.index).mean())
            curves = [("SmartPixels Conv1D_Full dump, rescaled (PixelAV angle mix)", rD[v], None, "#9e9e9e", 1.3, "-"),
                      ("dump reweighted to this surface's Phase-2 true-angle mix (5 deg cells)", rD[v], wrow, "#1b1b1b", 1.8, ":")]
            for i, (lab, t) in enumerate(PAYS):
                D = P[lab]; col = "Alpha" if v == "A" else "Beta"
                m = (D.surface.values == s) & (D.tpKey.values >= 0) & (D[f"has{col}"].values == 1)
                curves.append((f"CMSSW {lab}: emitted - true", deg(D[f"localCot{col}"].values[m]) - deg(D[f"tpLocalCot{col}"].values[m]),
                               None, PAL[[0, 3, 1][i]], 1.6, ["-", "--", "-"][i]))
            for lab, r, w, col, lw, ls in curves:
                ok = np.isfinite(r); r_ = r[ok]; w_ = None if w is None else w[ok]
                h, _ = np.histogram(np.clip(r_, e[0] + 1e-6, e[-1] - 1e-6), bins=e, weights=w_)
                ax.stairs(h / max(h.sum(), 1e-12) / np.diff(e), e, color=col, lw=lw, ls=ls, label=lab)
                ww = np.ones(len(r_)) if w_ is None else w_
                CLOS.append(dict(angle=nm, surface=SNAME[s], curve=lab, N=int(len(r_)), core_rms_deg=core_fit(r_, w_)[0],
                                 full_rms_deg=float(np.sqrt(np.average(r_ ** 2, weights=ww))),
                                 frac_abs_gt10deg=float(np.average(np.abs(r_) > 10, weights=ww)),
                                 phase2_cells_covered_by_dump=covered if "reweighted" in lab else ""))
            ax.set_yscale("log"); ax.set_ylim(1e-6, 2); ax.grid(True, color="#eeeeee")
            ax.set_title(f"CMS {nm}{' (bending)' if v == 'A' else ''}, IL{s}", fontsize=10)
            ax.set_xlabel("atan(emitted) - atan(true) [deg] (ends = overflow)"); ax.set_ylabel("density (unit area)")
        h_, l_ = axes[0].get_legend_handles_labels()
        fig.legend(h_, l_, loc="outside lower center", ncol=2, fontsize=9, frameon=False)
        fig.suptitle(f"Payload closure (file 3, 100 events, TP-linked barrel clusters) vs the RESCALED SmartPixels dump: CMS {nm}", fontsize=10)
        pdf.savefig(fig); plt.close(fig)
    CLOS = pd.DataFrame(CLOS); OUT["payload_closure"] = CLOS.to_dict("records")
    table_pages(pdf, fmt(CLOS), "Payload closure statistics [deg]: core RMS (mode-seeded Gaussian core, about 0), full RMS (about 0), fraction beyond "
                                "10 deg; 'covered' = share of the surface's clusters whose (alpha, beta) 5-deg cell has dump rows", per=34)
    # cot-space closure: the payload's own quantity
    rows = []
    for s in (1, 2, 3, 4):
        for lab, _ in PAYS:
            D = P[lab]
            for col in ("Alpha", "Beta"):
                m = (D.surface.values == s) & (D.tpKey.values >= 0) & (D[f"has{col}"].values == 1)
                r = D[f"localCot{col}"].values[m] - D[f"tpLocalCot{col}"].values[m]
                q16, q84 = np.quantile(r, [0.16, 0.84])
                rows.append(dict(surface=SNAME[s], run=lab, angle=col.lower(), n=int(m.sum()), q_width_cot=(q84 - q16) / 2,
                                 rms_cot=float(np.sqrt(np.mean(r ** 2))), median_sigma=float(np.median(D[f"sig{col}"].values[m])),
                                 pull_q_width=float(np.diff(np.quantile(r / D[f"sig{col}"].values[m], [0.16, 0.84]))[0] / 2)))
    CC = pd.DataFrame(rows); OUT["payload_closure_cot"] = CC.to_dict("records")
    table_pages(pdf, fmt(CC), "Payload residuals in cot (the payload's own quantity), TP-linked barrel clusters: q-width, RMS, the published sigma "
                              "(median) and the pull q-width (residual / published sigma)", per=34)
    table_pages(pdf, fmt(HAS), "hasAlpha / hasBeta, TP-linked (tpKey >= 0) vs unlinked, and the share of reported |cot| > 12", per=34)
    table_pages(pdf, pd.DataFrame([(k, json.dumps(OUT[k])[:400]) for k in ("payload_cap_change_detail", "payload_v4_oldcode_vs_newcode_SP",
                                                                              "payload_v4_oldcode_vs_newcode_nonSP")], columns=["check", "result"]),
                "The cap change alone (payload v4, old code vs new code)")

# payload leak: the payload angles joined onto the ntuple clusters (first 20 events of file 3)
key = ["run", "event", "detId", "idxInDet"]
for lab, t in (PAYS[0], PAYS[2]):
    D = P[lab]
    J = N[key].merge(D[key + ["localCotAlpha", "localCotBeta", "sigAlpha", "sigBeta", "hasAlpha", "hasBeta", "localX"]], on=key, how="left")
    matched = J.localCotAlpha.notna().values
    OUT[f"payload_leak_join_{t}"] = dict(ntuple_clusters=len(N), matched=int(matched.sum()),
                                         max_abs_dlocalX=float(np.nanmax(np.abs(J.localX.values - N.localX.values))))
    ext = np.c_[np.where(J.hasAlpha.values == 1, deg(J.localCotAlpha.values), np.nan), np.where(J.hasBeta.values == 1, deg(J.localCotBeta.values), np.nan),
                J.sigAlpha.values, J.sigBeta.values, J.hasAlpha.values, J.hasBeta.values]
    probe_gains(ext, f"{lab}: angles + sigmas + has*", LEAK)
    nano_probe(D, f"{lab} nano (all angle columns)", NANO_LEAK)
LK, NLK = pd.DataFrame(LEAK), pd.DataFrame(NANO_LEAK)
OUT["leak_probes"] = LK.to_dict("records"); OUT["leak_nano"] = NLK.to_dict("records")
with PdfPages(FIG("emu_fix2_leak.pdf")) as pdf:
    table_pages(pdf, fmt(LK), "Leak test (ntuple clusters, file 3, first 20 events): AUC(unlinked vs linked) gain of the added columns over the F1 probe "
                              "(shape + position) and the strong probe (F1 + tiny-CNN mean/sigma + CPE offset); HGB trained on even events, tested on odd "
                              "(and swapped). No-angle convention for the payload: an absent angle = NaN", per=26)
    table_pages(pdf, fmt(NLK), "Leak test (nano, file 3, 100 events): gain of ALL angle columns (cot, sigma, has, sat) over the nano probe", per=26)
PAY_RF, PAY_CP = refit_pdf("emu_fix2_refit_payload.pdf", [(lab, R(t)) for lab, t in PAYS],
                           "seedlessRefit AAAA, payload angles, file 3 (100 events): v4 old code vs v4 new code vs v5, TP-matched refit tracks")
OUT["refit_payload"] = PAY_RF.to_dict("records"); OUT["refit_payload_hits"] = PAY_CP.to_dict("records")

if a.perf:
    rows = []
    for t in ("perf_pay_v4", "perf_pay_v5"):
        js = json.load(open(os.path.join(a.perf, f"timing_{t}.json")))
        tot = js["total"]
        mods = {m["label"]: m for m in js["modules"]}
        rp = mods.get("spixSmartPixelsRecHits", {})
        refit = sum(m["time_real"] for lab, m in mods.items() if lab.startswith("l1tSmartPixelsTrackProducer"))
        wall = [l for l in open(os.path.join(a.perf, f"log_{t}.txt")) if "Elapsed (wall clock)" in l][0].split()[-1].split(":")
        wall = sum(float(x) * 60 ** i for i, x in enumerate(reversed(wall)))
        rss = [l for l in open(os.path.join(a.perf, f"log_{t}.txt")) if "Maximum resident set size" in l][0].split()[-1]
        rows.append(dict(run=t, events=tot["events"], wall_s_per_event_incl_setup=wall / tot["events"],
                         real_s_per_event_summed_over_4_streams=tot["time_real"] / tot["events"] / 1e3,
                         rechit_producer_ms_per_event=rp.get("time_real", np.nan) / max(rp.get("events", 1), 1),
                         refit_producers_ms_per_event=refit / tot["events"], max_rss_GB=int(rss) / 1024 ** 2))
    PF = pd.DataFrame(rows); OUT["performance"] = PF.to_dict("records")
    with PdfPages(FIG("emu_fix2_perf.pdf")) as pdf:
        table_pages(pdf, fmt(PF), "Performance, payload v4 vs v5 (new code), 200 events (files 3 + 4), 4 threads, one job at a time")
json.dump(OUT, open(os.path.join(HERE, "results/emu_fix2_validation.json"), "w"), indent=1,
          default=lambda x: x.item() if hasattr(x, "item") else str(x))
print("done")
