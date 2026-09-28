"""Validation of the fix-round CMSSW emulator (angleSource = "emu": no class-T pT cut, tiny CNN + A1, inclusive sigma LUT,
has* always true) on TEST samples only (RelVal file 3; files 1-2 hold the training events).

  emu_fix_validate.py --nano NANO.root --nano-alpha NANO_ALPHA.root --ntuple NTUPLE.root --v2 V2NANO.root
                      --pay-old PAY_OLD.root --pay-new PAY_NEW.root

  NANO        emu nano, refit useAngles = alphaBeta (the new default)      NANO_ALPHA  the same job with useAngles = alpha
  NTUPLE      ClusterAdcNtuple, angleSource emu, emuDebug (CNN outputs + draws for EVERY cluster)
  V2          cnn_nano_v2 of the same file (the non-SmartPixels tables must be identical)
  PAY_OLD/NEW payload-mode nanos of the same events, before / after the fix round (the refit must be identical)

Figures (vector PDF, one shared legend outside the axes, statistics on table pages): figs/emu_fix_residuals.pdf,
emu_fix_classC_vs_T.pdf, emu_fix_leak.pdf, emu_fix_sigma.pdf, emu_fix_refit.pdf, emu_fix_checks.pdf.
Results: results/emu_fix_validation.json. tpEmuClass / emuClass are TRUTH-ONLY: used here to split, never as a feature.
"""
import argparse, os, sys, json, numpy as np, pandas as pd, uproot, awkward as ak, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2_common import core_fit
from emu_common import load_library, emulate
from emu_build_library import load_dump

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = lambda n: os.path.join(HERE, "figs", n)
p = argparse.ArgumentParser()
for k in ("nano", "nano-alpha", "ntuple", "v2", "pay-old", "pay-new"):
    p.add_argument("--" + k, required=True)
p.add_argument("--library", default="/Volumes/WDMac/smartpixels_scratch/emu_fix/spix_emu_library_v2.root")
a = p.parse_args()
deg = lambda c: np.degrees(np.arctan(np.asarray(c, np.float64)))
PC = "L1TSmartPixelsCluster_"
GROUPS = ["IL2-4", "IL1", "ID1-4"]
grp = lambda s: np.where(np.asarray(s) > 10, "ID1-4", np.where(np.asarray(s) == 1, "IL1", "IL2-4"))
OUT = {}
# fast-study expectations (results/fast_eval_*.csv, tiny:a1arh; class-T route = the prototype)
FAST = {"IL2-4": {"C_alpha": 12.87, "C_beta": 23.26, "T_alpha": 6.90, "T_beta": 15.08, "leak_own": -0.0001, "leak_F1": 0.0108},
        "IL1": {"C_alpha": 11.47, "C_beta": 30.21, "T_alpha": 6.57, "T_beta": 11.79, "leak_own": 0.0022, "leak_F1": 0.0051},
        "ID1-4": {"C_alpha": 9.15, "C_beta": 19.28, "T_alpha": 6.59, "T_beta": 18.25, "leak_own": -0.0001, "leak_F1": 0.0001}}
PROTO_FRAC_T = {"IL1": 0.80, "IL2-4": 0.64, "ID1-4": 0.71}


def table_pages(pdf, df, title, per=30):
    for i in range(0, max(len(df), 1), per):
        t = df.iloc[i:i + per]
        f, ax = plt.subplots(figsize=(min(26, 2 + 1.6 * len(t.columns)), 0.33 * len(t) + 1.4)); ax.axis("off")
        tb = ax.table(cellText=t.astype(str).values, colLabels=list(t.columns), loc="center", cellLoc="left")
        tb.auto_set_font_size(False); tb.set_fontsize(7.5); tb.auto_set_column_width(list(range(len(t.columns))))
        ax.set_title(title + (f" ({i // per + 1})" if len(df) > per else ""), fontsize=10); pdf.savefig(f, bbox_inches="tight"); plt.close(f)


def fmt(df):
    return df.apply(lambda c: c.map(lambda x: f"{x:.4g}" if isinstance(x, (float, np.floating)) else x))


def read_clusters(f, cols):
    arr = uproot.open(f)["Events"].arrays([PC + c for c in cols] + ["run", "event"], library="ak")
    n = ak.to_numpy(ak.num(arr[PC + cols[0]]))
    df = pd.DataFrame({c: ak.to_numpy(ak.flatten(arr[PC + c])) for c in cols})
    df["event"] = np.repeat(ak.to_numpy(arr["event"]), n); df["run"] = np.repeat(ak.to_numpy(arr["run"]), n)
    return df


def stats(r, w=None):
    ok = np.isfinite(r); r = r[ok]; w = None if w is None else w[ok]
    ww = np.ones(len(r)) if w is None else w
    return dict(core_rms=core_fit(r, w)[0], full_rms=float(np.sqrt(np.average(r ** 2, weights=ww))),
                frac_gt10=float(np.average(np.abs(r) > 10, weights=ww)), N=int(len(r)))


# ======================= nano: clusters =======================
COLS = ["surface", "sizeX", "sizeY", "size", "charge", "globalR", "globalZ", "localX", "localY", "sigX", "sigY", "localCotAlpha",
        "localCotBeta", "sigAlpha", "sigBeta", "hasAlpha", "hasBeta", "satAlpha", "satBeta", "tpKey", "tpLocalCotAlpha",
        "tpLocalCotBeta", "tpEmuClass"]
D = read_clusters(a.nano, COLS)
D["group"] = grp(D.surface.values)
T = (D.tpEmuClass == 1).values; C = (D.tpEmuClass == 0).values
L = (D.tpKey >= 0).values
OUT["n_events_nano"] = int(D.event.nunique()); OUT["n_clusters_nano"] = int(len(D))
OUT["class_fractions"] = [dict(group=g, fracT=float(T[D.group == g].mean()), prototype=PROTO_FRAC_T[g],
                               fracC_linked=float((C & L)[D.group == g].mean()), fracC_unlinked=float((C & ~L)[D.group == g].mean()),
                               n=int((D.group == g).sum())) for g in GROUPS]
OUT["has_linked_vs_unlinked"] = [dict(group=g, linked=bool(lk), hasAlpha=float(D.hasAlpha[(D.group == g) & (L == lk)].mean()),
                                      hasBeta=float(D.hasBeta[(D.group == g) & (L == lk)].mean()), n=int(((D.group == g) & (L == lk)).sum()))
                                 for g in GROUPS for lk in (True, False)]
OUT["has_all_one"] = bool((D.hasAlpha == 1).all() and (D.hasBeta == 1).all())
OUT["saturation"] = [dict(group=g, cls=c, n=int(s.sum()), satAlpha_pct=100 * float(D.satAlpha[s].mean()), satBeta_pct=100 * float(D.satBeta[s].mean()),
                          frac_abs_cot_gt12_alpha_pct=100 * float((np.abs(D.localCotAlpha[s]) > 12).mean()),
                          frac_abs_cot_gt12_beta_pct=100 * float((np.abs(D.localCotBeta[s]) > 12).mean()))
                     for g in GROUPS for c, s in (("T", T & (D.group == g).values), ("C", C & (D.group == g).values))]


def nonfinite(f):
    """NaN / inf count in every float branch of the SmartPixels tables"""
    t = uproot.open(f)["Events"]; bad = {}
    for k in t.keys():
        if "SmartPixels" not in k:
            continue
        v = t[k].array(library="ak")
        try:
            x = ak.to_numpy(ak.flatten(v, axis=None))
        except Exception:
            continue
        if x.dtype.kind == "f":
            nb = int((~np.isfinite(x)).sum())
            if nb:
                bad[k] = nb
    return bad


OUT["nonfinite_SP_branches"] = nonfinite(a.nano)

# ======================= residuals vs the target =======================
D_ = load_dump()
rD = {"A": D_.rA.values, "B": D_.rB.values}; dtA, dtB = D_.tA.values, D_.tB.values
cellD = np.floor(dtA / 5).astype(int) * 1000 + np.floor(dtB / 5).astype(int)
tgt = {v: (lambda med: med + 1.5 * (rD[v] - med))(pd.Series(rD[v]).groupby(cellD).transform("median").values) for v in "AB"}
em = {"A": deg(D.localCotAlpha.values), "B": deg(D.localCotBeta.values)}
tr = {"A": deg(D.tpLocalCotAlpha.values), "B": deg(D.tpLocalCotBeta.values)}

# class-C route closure: the ntuple (every cluster has the CMSSW CNN outputs and A1 draws) -> the C route on class-T clusters
NT = uproot.open(a.ntuple)["clusterAdcNtuple/clusters"]
NC = ["run", "event", "detId", "idxInDet", "subdet", "layer", "disk", "sizeX", "sizeY", "size", "charge", "globalR", "globalZ",
      "localX", "localY", "originLocalX", "originLocalY", "zOutward", "linkClass", "tpIdx", "tpCharge", "hxOk", "hxDist",
      "tpLocalCotAlpha", "tpLocalCotBeta", "emuClass", "emuCnnA", "emuCnnB", "emuCnnSigA", "emuCnnSigB", "emuDrawA", "emuDrawB",
      "pCotAlpha", "pCotBeta"]
N = NT.arrays(NC, library="pd")
N["surface"] = np.where(N.subdet == 1, N.layer, 10 + N.disk).astype(np.int64); N["group"] = grp(N.surface.values)
NTm = (N.emuClass == 1).values
lib = load_library(a.library)
eC = emulate(lib, N.surface.values[NTm], np.zeros(NTm.sum(), bool), N.tpLocalCotAlpha.values[NTm], N.tpLocalCotBeta.values[NTm],
             N.emuCnnA.values[NTm], N.emuCnnB.values[NTm], N.emuDrawA.values[NTm], N.emuDrawB.values[NTm],
             N.run.values[NTm], N.event.values[NTm], N.detId.values[NTm], N.idxInDet.values[NTm])
CR = pd.DataFrame({"group": N.group.values[NTm], "rA": eC["outA"] - deg(N.tpLocalCotAlpha.values[NTm]),
                   "rB": eC["outB"] - deg(N.tpLocalCotBeta.values[NTm])})

e = np.linspace(-60, 60, 121); rows = []
fig, axes = plt.subplots(2, 3, figsize=(17, 10), layout="constrained")
for i, v in enumerate("AB"):
    for j, g in enumerate(GROUPS):
        ax = axes[i, j]; gm = (D.group == g).values; sT = gm & T
        keyA = deg(D.tpLocalCotAlpha.values[sT] - 0.312) if g == "ID1-4" else tr["A"][sT]
        pc = np.floor(keyA / 5).astype(int) * 1000 + np.floor(tr["B"][sT] / 5).astype(int)
        wP = pd.Series(pc).value_counts(normalize=True); wDd = pd.Series(cellD).value_counts(normalize=True)
        wrow = (pd.Series(cellD).map(wP).fillna(0) / pd.Series(cellD).map(wDd)).values
        curves = [("SmartPixels dump 1x (PixelAV angle mix)", rD[v], None, "#9e9e9e", 1.3, "-"),
                  ("TARGET: dump, bias kept + 1.5x spread, PixelAV angle mix", tgt[v], None, "#8c6d31", 1.5, "-"),
                  ("TARGET reweighted to the class-T Phase-2 angles (the class-T reference)", tgt[v], wrow, "#1b1b1b", 1.8, ":"),
                  ("CMSSW emu class T: emitted - true", em[v][sT] - tr[v][sT], None, "#e8742c", 1.8, "-"),
                  ("class-C route on class-T clusters (CMSSW CNN + A1 draws; closure)", CR[f"r{v}"].values[CR.group.values == g], None, "#2a78d6", 1.6, "-."),
                  ("CMSSW emu class C linked: emitted - (inconsistent) TP truth", (em[v] - tr[v])[gm & C & L], None, "#b04a9c", 1.2, "--")]
        for lab, r, w, col, lw, ls in curves:
            ok = np.isfinite(r); r_ = r[ok]; w_ = None if w is None else w[ok]
            h, _ = np.histogram(np.clip(r_, e[0] + 1e-6, e[-1] - 1e-6), bins=e, weights=w_)
            ax.stairs(h / max(h.sum(), 1e-12) / np.diff(e), e, color=col, lw=lw, ls=ls, label=lab)
            st = stats(r_, w_)
            ref = ""
            if lab.startswith("class-C route"):
                ref = FAST[g]["C_" + ("alpha" if v == "A" else "beta")]
            elif lab.startswith("CMSSW emu class T"):
                ref = FAST[g]["T_" + ("alpha" if v == "A" else "beta")]
            rows.append(dict(group=g, angle="alpha" if v == "A" else "beta", curve=lab, **st, fast_study_full_rms=ref))
        ax.set_yscale("log"); ax.set_ylim(1e-6, 1); ax.grid(True, color="#eeeeee")
        ax.set_title(f"CMS {'alpha (bending)' if v == 'A' else 'beta'} - {g}", fontsize=10)
        ax.set_xlabel("residual [deg] (ends = overflow)"); ax.set_ylabel("density (unit area)")
h_, l_ = axes[0, 0].get_legend_handles_labels()
fig.legend(h_, l_, loc="outside lower center", ncol=2, fontsize=9, frameon=False)
fig.suptitle("Fix-round CMSSW emu, RelVal file 3 (test events): residuals vs the 1.5x target; class-C route closure", fontsize=10)
R = pd.DataFrame(rows); OUT["residuals"] = R.to_dict("records")
with PdfPages(FIG("emu_fix_residuals.pdf")) as pdf:
    pdf.savefig(fig); plt.close(fig)
    for ang in ("alpha", "beta"):
        table_pages(pdf, fmt(R[R.angle == ang].drop(columns="angle")),
                    f"Residual statistics, CMS {ang} [deg] (core = mode-seeded Gaussian core RMS about 0; full = RMS about 0)")

# ======================= emitted C vs T =======================
ee = np.linspace(-90, 90, 181); rows = []
fig, axes = plt.subplots(2, 3, figsize=(17, 10), layout="constrained")
for i, v in enumerate("AB"):
    for j, g in enumerate(GROUPS):
        ax = axes[i, j]; gm = (D.group == g).values
        for lab, x, col in (("class T: TRUE angle", tr[v][gm & T], "#9e9e9e"), ("class T: EMITTED", em[v][gm & T], "#e8742c"),
                            ("class C linked: EMITTED", em[v][gm & C & L], "#1f9e76"), ("class C unlinked: EMITTED", em[v][gm & C & ~L], "#2a78d6")):
            h, _ = np.histogram(np.clip(x, ee[0] + 1e-6, ee[-1] - 1e-6), bins=ee)
            ax.stairs(h / max(h.sum(), 1) / np.diff(ee), ee, color=col, lw=1.5, label=lab)
            rows.append(dict(group=g, angle="alpha" if v == "A" else "beta", curve=lab, N=len(x), mean=float(np.mean(x)),
                             rms=float(np.sqrt(np.mean(x ** 2))), frac_abs_gt60_pct=100 * float(np.mean(np.abs(x) > 60))))
        ax.set_yscale("log"); ax.set_ylim(1e-6, 0.3); ax.grid(True, color="#eeeeee")
        ax.set_title(f"CMS {'alpha' if v == 'A' else 'beta'} - {g}", fontsize=10); ax.set_xlabel("angle [deg]"); ax.set_ylabel("density (unit area)")
h_, l_ = axes[0, 0].get_legend_handles_labels()
fig.legend(h_, l_, loc="outside lower center", ncol=4, fontsize=9, frameon=False)
fig.suptitle("Fix-round CMSSW emu, file 3: emitted angle distributions, class T vs class C", fontsize=10)
E2 = pd.DataFrame(rows); OUT["emitted"] = E2.to_dict("records")
with PdfPages(FIG("emu_fix_classC_vs_T.pdf")) as pdf:
    pdf.savefig(fig); plt.close(fig)
    table_pages(pdf, fmt(E2), "Emitted angle statistics [deg]")
    table_pages(pdf, fmt(pd.DataFrame(OUT["saturation"])), "Saturation (clamp at +-(90 - 0.01) deg) and |cot| > 12 rates [%], per group x class")

# ======================= leak =======================
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score


def auc(X, y, half, seed):
    rng = np.random.default_rng(seed); tri = np.where(half)[0]; tri = rng.choice(tri, min(len(tri), 1_000_000), replace=False)
    X = np.where(np.isfinite(X), X, -999.0)
    clf = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=31, learning_rate=0.1, random_state=1).fit(X[tri], y[tri])
    return float(roc_auc_score(y[~half], clf.predict_proba(X[~half])[:, 1]))


# (a) the fast-study probes on the ntuple: y = unlinked (simlink, no TP); train on even events, test on odd
y = (N.linkClass.values == 1).astype(int)
half = (N.event.values % 2 == 0)
F1 = np.c_[N.sizeX, N.sizeY, N["size"], N.charge, N.surface, N.globalR, np.abs(N.globalZ), N.localX, N.localY, N.zOutward]
cpe = np.c_[(N.localX - N.originLocalX) * 1e4, (N.localY - N.originLocalY) * 1e4]
strong = np.c_[F1, N.emuCnnA, N.emuCnnB, N.emuCnnSigA, N.emuCnnSigB, cpe]
prod = np.c_[deg(N.pCotAlpha.values), deg(N.pCotBeta.values)]
rows = []
for g in GROUPS:
    gm = (N.group == g).values
    r = dict(group=g, n=int(gm.sum()), frac_unlinked=float(y[gm].mean()))
    for nm, X in (("F1 (shape+position)", F1), ("strong (F1 + tiny-CNN mean/sigma + CPE)", strong)):
        b = auc(X[gm], y[gm], half[gm], 3); w_ = auc(np.c_[X, prod][gm], y[gm], half[gm], 3)
        bs = auc(X[gm], y[gm], ~half[gm], 4); ws = auc(np.c_[X, prod][gm], y[gm], ~half[gm], 4)
        k = "F1" if nm.startswith("F1") else "strong"
        r[f"{k}_base"] = b; r[f"{k}_gain"] = w_ - b; r[f"{k}_gain_swapped_halves"] = ws - bs
    r["fast_study_gain_F1"] = FAST[g]["leak_F1"]; r["fast_study_gain_strong(own CNN)"] = FAST[g]["leak_own"]
    rows.append(r); print("leak", r, flush=True)
LK = pd.DataFrame(rows); OUT["leak_probes"] = LK.to_dict("records")
# (b) the nano columns: all angle columns incl. has / sat / sigma, over the nano probe
base = ["sizeX", "sizeY", "size", "charge", "surface", "globalR", "globalZ", "localX", "localY", "sigX", "sigY"]
angc = ["localCotAlpha", "localCotBeta", "sigAlpha", "sigBeta", "hasAlpha", "hasBeta", "satAlpha", "satBeta"]
yn = (~L).astype(int); hn = (D.event.values % 2 == 0); rows = []
for g in GROUPS:
    gm = (D.group == g).values
    b = auc(D[base].values.astype(np.float64)[gm], yn[gm], hn[gm], 5)
    w_ = auc(np.clip(D[base + angc].values.astype(np.float64), -1e6, 1e6)[gm], yn[gm], hn[gm], 5)
    rows.append(dict(group=g, auc_no_angles=b, auc_with_all_angle_columns=w_, gain=w_ - b, n=int(gm.sum())))
LN = pd.DataFrame(rows); OUT["leak_nano_columns"] = LN.to_dict("records")
with PdfPages(FIG("emu_fix_leak.pdf")) as pdf:
    table_pages(pdf, fmt(LK), "Leak test (ntuple, file 3, 20 events): AUC(unlinked vs linked) gain of the emitted angles over each probe; "
                              "HGB trained on even events, tested on odd (and swapped)")
    table_pages(pdf, fmt(LN), "Leak test (nano, file 3): gain of ALL emu angle columns (cot, sigma, has, sat) over the nano probe")
    table_pages(pdf, fmt(pd.DataFrame(OUT["has_linked_vs_unlinked"])), "hasAlpha / hasBeta, linked (tpKey >= 0) vs unlinked (must be exactly 1)")
    table_pages(pdf, fmt(pd.DataFrame(OUT["class_fractions"])), "Class fractions (TRUTH-ONLY tpEmuClass) vs the prototype (no pT cut)")

# ======================= sigma LUT =======================
ST = pd.read_csv(os.path.join(HERE, "results/emu_sigma_table.csv")); SP = pd.read_csv(os.path.join(HERE, "results/emu_sigma_pulls.csv"))
rows = []
for s in sorted(D.surface.unique()):
    for v, nm in (("A", "alpha"), ("B", "beta")):
        sm = (D.surface == s).values & T
        pc = D[f"localCot{'Alpha' if v == 'A' else 'Beta'}"].values[sm]; tc = D[f"tpLocalCot{'Alpha' if v == 'A' else 'Beta'}"].values[sm]
        sg = D[f"sig{'Alpha' if v == 'A' else 'Beta'}"].values[sm]; pl = (pc - tc) / sg
        for sel, k in (("all", np.ones(len(pl), bool)), ("|cot|<=12", np.abs(pc) <= 12)):
            q16, q84 = np.quantile(pl[k], [0.16, 0.84])
            rows.append(dict(angle=nm, surface=int(s), cls="T (CMSSW nano)", selection=sel, n=int(k.sum()), q_width=(q84 - q16) / 2,
                             median=float(np.median(pl[k])), frac_abs_gt3=float(np.mean(np.abs(pl[k]) > 3))))
PN = pd.DataFrame(rows); OUT["pulls_cmssw_classT"] = PN.to_dict("records")
fig, axes = plt.subplots(1, 2, figsize=(15, 6), layout="constrained")
PAL = {1: "#2a78d6", 2: "#e8742c", 3: "#1f9e76", 4: "#b04a9c", 11: "#c9a227", 12: "#6b4c9a", 13: "#d1495b", 14: "#3d7d7d"}
for ax, nm in zip(axes, ("alpha", "beta")):
    for s, g in ST[ST.angle == nm].groupby("surface"):
        x = np.r_[g.lo.values, g.hi.values[-1]]
        ax.stairs(g.sigma.values, x, color=PAL[s], lw=1.6, label=f"surface {s} ({'IL' if s < 10 else 'ID'}{s % 10})")
    ax.set_xscale("symlog", linthresh=0.05); ax.set_yscale("log"); ax.grid(True, color="#eeeeee")
    ax.set_xlabel(f"|reported cot{nm.capitalize()}| (table clamps above 12)"); ax.set_ylabel(f"sigma(cot{nm.capitalize()}) = (q84 - q16)/2")
    ax.set_title(f"inclusive (T + C) sigma LUT, {nm}", fontsize=10)
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="outside lower center", ncol=8, fontsize=9, frameon=False)
with PdfPages(FIG("emu_fix_sigma.pdf")) as pdf:
    pdf.savefig(fig); plt.close(fig)
    table_pages(pdf, fmt(ST), "sigma LUT bins (training events): sigma = weighted (q84-q16)/2 of the pooled T+C residual [cot]; core/full RMS [cot]; "
                              "per-route widths; wC = class-C weight share", per=34)
    table_pages(pdf, fmt(SP), "Pulls (emitted - true cot)/sigma, python reference on the TEST events (class-T clusters through both routes)", per=34)
    table_pages(pdf, fmt(PN), "Pulls of the CMSSW nano (file 3), class T", per=34)

# ======================= non-SP tables vs v2; payload refit before/after =======================
def compare(f1, f2, only_nonSP):
    t1, t2 = uproot.open(f1)["Events"], uproot.open(f2)["Events"]
    e1 = t1["event"].array(library="np"); e2 = t2["event"].array(library="np")
    o1, o2 = np.argsort(e1, kind="stable"), np.argsort(e2, kind="stable")
    common = sorted(set(e1) & set(e2))
    s1, s2 = o1[np.isin(e1[o1], common)], o2[np.isin(e2[o2], common)]
    diff, same, missing = [], 0, sorted(set(t2.keys()) ^ set(t1.keys()))
    for k in t1.keys():
        if k not in t2.keys() or (only_nonSP and "SmartPixels" in k):
            continue
        if not only_nonSP and "SmartPixels" not in k:
            continue
        b1 = t1[k].array(library="ak")[s1]; b2 = t2[k].array(library="ak")[s2]
        try:
            ok = (ak.all(ak.num(b1, axis=-1) == ak.num(b2, axis=-1)) if b1.ndim > 1 else True)
            if ok:
                x1 = ak.to_numpy(ak.flatten(b1, axis=None)); x2 = ak.to_numpy(ak.flatten(b2, axis=None))
                ok = x1.shape == x2.shape and bool(np.all((x1 == x2) | (np.isnan(x1) & np.isnan(x2)) if x1.dtype.kind == "f" else x1 == x2))
        except Exception:
            ok = False
        if ok:
            same += 1
        else:
            diff.append(k)
    return dict(n_common_events=len(common), n_branches_compared=same + len(diff), n_identical=same, differing=diff,
                branches_only_in_one=missing[:40])


OUT["nonSP_vs_v2"] = compare(a.nano, a.v2, True)
OUT["payload_SP_old_vs_new"] = compare(a.pay_old, a.pay_new, False)
OUT["payload_nonSP_old_vs_new"] = compare(a.pay_old, a.pay_new, True)
print("nonSP vs v2", OUT["nonSP_vs_v2"]); print("payload SP", OUT["payload_SP_old_vs_new"]); print("payload nonSP", OUT["payload_nonSP_old_vs_new"])

# ======================= refit: alphaBeta vs alpha, the |cot| cap =======================
RT, RH = "L1TSmartPixelsTrackSeedlessRefitAAAA_", "L1TSmartPixelsRefitHitSeedlessRefitAAAA_"
TRK = ["pt", "d0", "z0", "spixRefitPerformed", "spixNAcceptedHits", "spixNKFUpdates", "spixMatchedTpKey",
       "spixChi2IncXTot", "spixChi2IncYTot", "spixChi2IncAlphaTot", "spixChi2IncBetaTot"]
HIT = ["trackIdx", "hitAccepted", "hasAlpha", "hasBeta", "selClusterIdx", "selHitClass", "pullAlpha", "pullBeta"]


def refit(f):
    t = uproot.open(f)["Events"]
    ev = t.arrays([RT + c for c in TRK] + [RH + c for c in HIT] + ["L1TTrack_tp_pt", "L1TTrack_tp_d0", "L1TTrack_tp_z0",
                  "L1TTrack_pt", "L1TTrack_genuine", PC + "localCotAlpha", PC + "localCotBeta", "event"], library="ak")
    o = np.argsort(ak.to_numpy(ev["event"]), kind="stable")
    return ev[o]


def flat(x):
    return ak.to_numpy(ak.flatten(x))


rows, caps, trk = [], [], {}
for tag, f in (("alphaBeta (new default)", a.nano), ("alpha", a.nano_alpha)):
    ev = refit(f)
    T_ = pd.DataFrame({c: flat(ev[RT + c]) for c in TRK})
    for c in ("tp_pt", "tp_d0", "tp_z0", "pt", "genuine"):
        T_["base_" + c] = flat(ev["L1TTrack_" + c])
    T_["chi2IT"] = T_.spixChi2IncXTot + T_.spixChi2IncYTot + T_.spixChi2IncAlphaTot + T_.spixChi2IncBetaTot
    trk[tag] = T_
    # accepted hits: purity (selHitClass 0 = the track's own TP) and the |cot| cap rule
    acc = ev[RH + "hitAccepted"] == 1
    sel = ev[RH + "selClusterIdx"]
    ca = ev[PC + "localCotAlpha"][ak.where(sel >= 0, sel, 0)]; cb = ev[PC + "localCotBeta"][ak.where(sel >= 0, sel, 0)]
    hA, hB = flat(ev[RH + "hasAlpha"][acc]).astype(bool), flat(ev[RH + "hasBeta"][acc]).astype(bool)
    cA, cB = flat(ca[acc]), flat(cb[acc]); cls = flat(ev[RH + "selHitClass"][acc])
    caps.append(dict(refit=tag, accepted_hits=int(len(hA)), hasAlpha_rule_violations=int((hA != (np.abs(cA) <= 12)).sum()),
                     hasBeta_rule_violations=int((hB != (np.abs(cB) <= 12)).sum()), alpha_dropped_by_cap=int((~hA).sum()),
                     beta_dropped_by_cap=int((~hB).sum()), purity_own_TP=float(np.mean(cls == 0)), frac_other_TP=float(np.mean(cls == 1)),
                     frac_no_TP=float(np.mean(cls == 2))))
    m = (T_.spixRefitPerformed == 1) & (T_.base_tp_pt > 0)
    for nm, r in (("d0 [um]", 1e4 * (T_.d0 - T_.base_tp_d0)), ("z0 [um]", 1e4 * (T_.z0 - T_.base_tp_z0)),
                  ("pT (reco - true) / true", (T_.pt - T_.base_tp_pt) / T_.base_tp_pt)):
        for pt_lo, pt_hi in ((0, 1e9), (2, 5), (5, 1e9)):
            mm = m & (T_.base_tp_pt >= pt_lo) & (T_.base_tp_pt < pt_hi)
            x = r[mm].values; q16, q84 = np.quantile(x, [0.16, 0.84])
            rows.append(dict(refit=tag, quantity=nm, tp_pt=f"[{pt_lo}, {pt_hi if pt_hi < 1e9 else 'inf'})", n=int(mm.sum()),
                             q68_halfwidth=(q84 - q16) / 2, core_rms=core_fit(x)[0], median=float(np.median(x))))
    for nm, x in (("chi2 IT total", T_.chi2IT[m]), ("chi2 alpha", T_.spixChi2IncAlphaTot[m]), ("chi2 beta", T_.spixChi2IncBetaTot[m])):
        rows.append(dict(refit=tag, quantity=nm, tp_pt="all", n=int(m.sum()), q68_halfwidth=np.nan, core_rms=np.nan,
                         median=float(np.median(x)), mean=float(np.mean(x)), q90=float(np.quantile(x, 0.9)), q99=float(np.quantile(x, 0.99))))
RF = pd.DataFrame(rows); CP = pd.DataFrame(caps)
OUT["refit"] = RF.to_dict("records"); OUT["refit_cap_purity"] = CP.to_dict("records")
fig, axes = plt.subplots(2, 3, figsize=(17, 9.5), layout="constrained")
COL = {"alphaBeta (new default)": "#e8742c", "alpha": "#2a78d6"}
specs = [("d0 [um]", lambda T_: 1e4 * (T_.d0 - T_.base_tp_d0), np.linspace(-300, 300, 121)),
         ("z0 [um]", lambda T_: 1e4 * (T_.z0 - T_.base_tp_z0), np.linspace(-1500, 1500, 121)),
         ("(pT - true) / true", lambda T_: (T_.pt - T_.base_tp_pt) / T_.base_tp_pt, np.linspace(-0.3, 0.3, 121)),
         ("chi2 IT total (x + y + alpha + beta)", lambda T_: T_.chi2IT, np.logspace(-2, 4, 121)),
         ("chi2 alpha", lambda T_: T_.spixChi2IncAlphaTot, np.logspace(-3, 4, 121)),
         ("chi2 beta", lambda T_: T_.spixChi2IncBetaTot, np.logspace(-3, 4, 121))]
for ax, (nm, fn, e) in zip(axes.flat, specs):
    for tag, T_ in trk.items():
        m = (T_.spixRefitPerformed == 1) & (T_.base_tp_pt > 0); x = fn(T_)[m].values
        h, _ = np.histogram(np.clip(x, e[0] * (1 + 1e-9) if e[0] > 0 else e[0] + 1e-9, e[-1] * (1 - 1e-9)), bins=e)
        ax.stairs(h / max(h.sum(), 1), e, color=COL[tag], lw=1.6, label=f"refit useAngles = {tag}")
    ax.set_title(nm, fontsize=10); ax.set_yscale("log"); ax.grid(True, color="#eeeeee"); ax.set_ylabel("fraction of refit tracks")
    if e[0] > 0:
        ax.set_xscale("log")
h_, l_ = axes[0, 0].get_legend_handles_labels()
fig.legend(h_, l_, loc="outside lower center", ncol=2, fontsize=9, frameon=False)
fig.suptitle("seedlessRefit AAAA on the emu nano (file 3, 100 events): useAngles alphaBeta vs alpha, TP-matched refit tracks", fontsize=10)
with PdfPages(FIG("emu_fix_refit.pdf")) as pdf:
    pdf.savefig(fig); plt.close(fig)
    table_pages(pdf, fmt(RF), "Refit residual widths vs the TP (q68 half-width, core RMS, median) and chi2 summaries")
    table_pages(pdf, fmt(CP), "Accepted hits: the |cot| <= 12 cap rule (refit hasAlpha/hasBeta vs the cluster's reported cot) and "
                              "selected-hit purity (selHitClass)")

with PdfPages(FIG("emu_fix_checks.pdf")) as pdf:
    chk = [("events / clusters in the nano", f"{OUT['n_events_nano']} / {OUT['n_clusters_nano']}"),
           ("hasAlpha = hasBeta = 1 for every cluster", OUT["has_all_one"]),
           ("non-finite values in SmartPixels float branches", json.dumps(OUT["nonfinite_SP_branches"]) or "none"),
           ("non-SP tables vs cnn_nano_v2 (same events): identical / compared", f"{OUT['nonSP_vs_v2']['n_identical']} / {OUT['nonSP_vs_v2']['n_branches_compared']}"),
           ("non-SP branches differing vs v2", ", ".join(OUT["nonSP_vs_v2"]["differing"]) or "none"),
           ("payload mode, SP branches before/after the fix round: identical / compared",
            f"{OUT['payload_SP_old_vs_new']['n_identical']} / {OUT['payload_SP_old_vs_new']['n_branches_compared']}"),
           ("payload SP branches differing", ", ".join(OUT["payload_SP_old_vs_new"]["differing"]) or "none"),
           ("payload non-SP branches differing", ", ".join(OUT["payload_nonSP_old_vs_new"]["differing"]) or "none")]
    table_pages(pdf, pd.DataFrame(chk, columns=["check", "result"]), "Fix-round checks (file 3)")
json.dump(OUT, open(os.path.join(HERE, "results/emu_fix_validation.json"), "w"), indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
print("done")
