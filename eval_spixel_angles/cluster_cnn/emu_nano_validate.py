"""Validation of the CMSSW SmartPixels-output emulator (angleSource = "emu") on the regenerated nanos (cnn_nano_v3).

  emu_nano_validate.py --plot-files F03,F04 --all-files F01,...,F10 [--closure-ntuple NTUPLE.root --library LIB.root]

Figures (vector PDF; shared legend outside the axes; statistics on separate table pages):
  figs/emu_nano_residuals.pdf  emitted - true angle [deg] per IL2-4 / IL1 / ID1-4 and CMS alpha / beta: class T (truth-
                               consistent linked), the TARGET = the rescaled SmartPixels Conv1D dump with its per-cell (5x5 deg)
                               bias kept + 1.5 x spread, REWEIGHTED to the class-T Phase-2 true angles (what class T must
                               reproduce), the target on the PixelAV angle mix, the dump 1x; class C linked clusters vs their
                               (inconsistent) TP truth; with --closure-ntuple the class-C ROUTE applied to class-T clusters by
                               the python reference on the CMSSW CNN outputs (the prototype's closure)
  figs/emu_nano_classC_vs_T.pdf emitted angle distributions: class T / class C linked / class C unlinked, + saturation table
  figs/emu_nano_leak.pdf        leak test tables: AUC(unlinked vs linked) without / with the angle columns, hasAlpha/hasBeta
                               linked vs unlinked, saturation rates per surface x class (all files)
Results: results/emu_nano_*.csv|json. tpEmuClass is TRUTH-ONLY; it is used here to split the validation, never as a feature.
"""
import argparse, os, sys, json, numpy as np, pandas as pd, uproot, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2_common import core_fit

HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("--plot-files", required=True); p.add_argument("--all-files", required=True)
p.add_argument("--closure-ntuple", default=""); p.add_argument("--library", default="")
a = p.parse_args()
deg = lambda c: np.degrees(np.arctan(c))
PFX = "L1TSmartPixelsCluster_"
COLS = ["surface", "sizeX", "sizeY", "size", "charge", "globalR", "globalZ", "localX", "localY", "sigX", "sigY", "localCotAlpha",
        "localCotBeta", "sigAlpha", "sigBeta", "hasAlpha", "hasBeta", "satAlpha", "satBeta", "tpKey", "tpLocalCotAlpha",
        "tpLocalCotBeta", "tpEmuClass"]


def read(files, cols):
    import awkward as ak
    out = []
    for i, f in enumerate(files):
        arr = uproot.open(f)["Events"].arrays([PFX + c for c in cols] + ["event"], library="ak")
        n = ak.to_numpy(ak.num(arr[PFX + cols[0]]))
        df = pd.DataFrame({c: ak.to_numpy(ak.flatten(arr[PFX + c])) for c in cols})
        df["event"] = np.repeat(ak.to_numpy(arr["event"]), n) + (i << 32); df["file"] = i
        out.append(df)
    return pd.concat(out, ignore_index=True)


def grp(s):
    return np.where(s > 10, "ID1-4", np.where(s == 1, "IL1", "IL2-4"))


# ---------------- saturation + class fractions over ALL files ----------------
allf = a.all_files.split(",")
S = read(allf, ["surface", "satAlpha", "satBeta", "tpEmuClass", "tpKey", "hasAlpha", "hasBeta"])
S["cls"] = np.where(S.tpEmuClass == 1, "T", np.where(S.tpEmuClass == 0, "C", "none"))
assert (S.cls != "none").all(), "a cluster without an emulator class: not an emu nano?"
sat = S.groupby(["surface", "cls"]).agg(n=("satAlpha", "size"), satAlpha=("satAlpha", "mean"), satBeta=("satBeta", "mean")).reset_index()
sat[["satAlpha", "satBeta"]] *= 100
sat_g = S.assign(group=grp(S.surface.values)).groupby(["group", "cls"]).agg(n=("satAlpha", "size"), satAlpha=("satAlpha", "mean"),
                                                                            satBeta=("satBeta", "mean")).reset_index()
sat_g[["satAlpha", "satBeta"]] *= 100
cls_frac = S.assign(group=grp(S.surface.values)).groupby("group").apply(lambda g: pd.Series({
    "fracT": float((g.cls == "T").mean()), "fracC_linked": float(((g.cls == "C") & (g.tpKey >= 0)).mean()),
    "fracC_unlinked": float(((g.cls == "C") & (g.tpKey < 0)).mean())})).reset_index()
has = S.assign(group=grp(S.surface.values), linked=S.tpKey >= 0).groupby(["group", "linked"]).agg(
    hasAlpha=("hasAlpha", "mean"), hasBeta=("hasBeta", "mean"), n=("hasAlpha", "size")).reset_index()
del S

# ---------------- plots on the plot files ----------------
D = read(a.plot_files.split(","), COLS)
D["group"] = grp(D.surface.values)
T = (D.tpEmuClass == 1).values; CL = (D.tpEmuClass == 0).values & (D.tpKey >= 0).values; CU = (D.tpEmuClass == 0).values & (D.tpKey < 0).values
em = {"A": deg(D.localCotAlpha.values), "B": deg(D.localCotBeta.values)}
tr = {"A": deg(D.tpLocalCotAlpha.values), "B": deg(D.tpLocalCotBeta.values)}

SF = {"cotA": 6.528905408493022, "cotB": 1.8827422609846889}
DP = pd.read_parquet(os.path.expanduser("~/smartpixels/ngtagger-train/2t-Conv1D_Full-2bit_optimized-vars.parquet"))
dtA, dpA = deg(DP.cotBtrue * SF["cotB"]).values, deg(DP.cotB * SF["cotB"]).values
dtB, dpB = deg(DP.cotAtrue * SF["cotA"]).values, deg(DP.cotA * SF["cotA"]).values
rD = {"A": dpA - dtA, "B": dpB - dtB}
cellD = np.floor(dtA / 5).astype(int) * 1000 + np.floor(dtB / 5).astype(int)
tgt = {v: (lambda med: med + 1.5 * (rD[v] - med))(pd.Series(rD[v]).groupby(cellD).transform("median").values) for v in "AB"}

CR = None
if a.closure_ntuple:
    import awkward as ak
    from emu_common import load_library, emulate, PT_MIN, HXDIST_MAX
    t = uproot.open(a.closure_ntuple)["clusterAdcNtuple/clusters"]
    cols = ["run", "event", "detId", "idxInDet", "subdet", "layer", "disk", "sizeX", "sizeY", "charge", "globalR", "globalZ", "localX",
            "localY", "tpLocalCotAlpha", "tpLocalCotBeta", "emuClass", "emuCnnA", "emuCnnB", "emuCnnSigA", "emuCnnSigB"]
    N = t.arrays(cols, library="pd")
    N = N[N.emuClass == 1].reset_index(drop=True)
    surf = np.where(N.subdet == 1, N.layer, 10 + N.disk).astype(np.int64)
    obs = np.c_[N.emuCnnA, N.emuCnnB, N.emuCnnSigA, N.emuCnnSigB, N.sizeX, N.sizeY, np.log(N.charge.clip(lower=1)).values, N.globalR,
                np.abs(N.globalZ.values), N.localX, N.localY]
    e = emulate(load_library(a.library), surf, np.zeros(len(N), bool), N.tpLocalCotAlpha.values, N.tpLocalCotBeta.values,
                N.emuCnnA.values, N.emuCnnB.values, obs, N.run.values, N.event.values, N.detId.values, N.idxInDet.values)
    CR = pd.DataFrame({"group": grp(surf), "rA": e["outA"] - deg(N.tpLocalCotAlpha.values), "rB": e["outB"] - deg(N.tpLocalCotBeta.values)})


def stats(r, w=None):
    ok = np.isfinite(r); r = r[ok]; w = None if w is None else w[ok]
    c = core_fit(r, w)[0]
    ww = np.ones(len(r)) if w is None else w
    return c, float(np.sqrt(np.average(r ** 2, weights=ww))), float(np.average(np.abs(r) > 10, weights=ww)), int(len(r))


GROUPS = ["IL2-4", "IL1", "ID1-4"]
e = np.linspace(-60, 60, 121); rows = []
fig, axes = plt.subplots(2, 3, figsize=(17, 10), layout="constrained")
for i, v in enumerate("AB"):
    for j, g in enumerate(GROUPS):
        ax = axes[i, j]; gm = (D.group == g).values; sT = gm & T
        # the dump cell a class-T cluster is emulated from: on a disc the alpha key is atan(cotAlpha - 0.312)
        keyA = deg(D.tpLocalCotAlpha.values[sT] - 0.312) if g == "ID1-4" else tr["A"][sT]
        pc = np.floor(keyA / 5).astype(int) * 1000 + np.floor(tr["B"][sT] / 5).astype(int)
        wP = pd.Series(pc).value_counts(normalize=True); wDd = pd.Series(cellD).value_counts(normalize=True)
        wrow = (pd.Series(cellD).map(wP).fillna(0) / pd.Series(cellD).map(wDd)).values
        covfrac = wP[wP.index.isin(wDd.index)].sum()
        curves = [("SmartPixels dump 1x (PixelAV angle mix)", rD[v], None, "#9e9e9e", 1.4, "-"),
                  ("TARGET: dump, bias kept + 1.5x spread (PixelAV angle mix)", tgt[v], None, "#1b1b1b", 1.8, "-"),
                  ("TARGET reweighted to the class-T Phase-2 angles (discs: at the alpha lookup key)", tgt[v], wrow, "#1b1b1b", 1.6, ":"),
                  ("CMSSW emu, class T: emitted - true", em[v][sT] - tr[v][sT], None, "#e8742c", 1.8, "-"),
                  ("CMSSW emu, class C linked: emitted - (inconsistent) TP truth", (em[v] - tr[v])[gm & CL], None, "#b04a9c", 1.2, "--")]
        if CR is not None:
            curves.append(("class-C route on class-T clusters (python ref., CMSSW CNN; closure)", CR[f"r{v}"].values[CR.group.values == g],
                           None, "#1f9e76", 1.6, "-."))
        for lab, r, w, col, lw, ls in curves:
            ok = np.isfinite(r); r_ = r[ok]; w_ = None if w is None else w[ok]
            h, _ = np.histogram(np.clip(r_, e[0] + 1e-6, e[-1] - 1e-6), bins=e, weights=w_)
            ax.stairs(h / max(h.sum(), 1e-12) / np.diff(e), e, color=col, lw=lw, ls=ls, label=lab)
            c, f, gt, nn = stats(r_, w_)
            rows.append(dict(group=g, angle="alpha" if v == "A" else "beta", curve=lab, N=nn, core_rms_deg=round(c, 2),
                             full_rms_deg=round(f, 2), frac_gt10=f"{100 * gt:.1f}%",
                             note=f"{100 * covfrac:.0f}% of class-T mass in dump cells" if w is not None else ""))
        ax.set_yscale("log"); ax.set_ylim(1e-6, 1); ax.grid(True, color="#eeeeee")
        ax.set_title(f"CMS {'alpha (bending)' if v == 'A' else 'beta'} - {g}", fontsize=10)
        ax.set_xlabel("residual [deg] (ends = overflow)"); ax.set_ylabel("density (unit area)")
h_, l_ = axes[0, 0].get_legend_handles_labels()
fig.legend(h_, l_, loc="outside lower center", ncol=2, fontsize=9, frameon=False)
fig.suptitle("CMSSW angleSource=emu on cnn_nano_v3 (files: %s): residuals vs the 1.5x target reweighted to the Phase-2 angles" % a.plot_files.split(",")[0].split("/")[-1], fontsize=10)
R = pd.DataFrame(rows); R.to_csv(os.path.join(HERE, "results/emu_nano_residuals.csv"), index=False)


def table_pages(pdf, df, title, fs=8, w=16):
    f2, a2 = plt.subplots(figsize=(w, 0.3 * len(df) + 1.3)); a2.axis("off")
    tb = a2.table(cellText=df.values, colLabels=list(df.columns), loc="center", cellLoc="left", colLoc="left")
    tb.auto_set_font_size(False); tb.set_fontsize(fs); tb.auto_set_column_width(list(range(df.shape[1])))
    a2.set_title(title, fontsize=10); pdf.savefig(f2, bbox_inches="tight"); plt.close(f2)


with PdfPages(os.path.join(HERE, "figs/emu_nano_residuals.pdf")) as pdf:
    pdf.savefig(fig); plt.close(fig)
    for ang in ("alpha", "beta"):
        table_pages(pdf, R[R.angle == ang].drop(columns="angle"), f"Residual statistics - CMS {ang} (core = mode-seeded Gaussian core RMS about 0)")

# emitted angle distributions
ee = {"A": np.linspace(-90, 90, 181), "B": np.linspace(-90, 90, 181)}; rows = []
fig, axes = plt.subplots(2, 3, figsize=(17, 10), layout="constrained")
for i, v in enumerate("AB"):
    for j, g in enumerate(GROUPS):
        ax = axes[i, j]; gm = (D.group == g).values
        for lab, x, col, ls in (("class T: TRUE angle", tr[v][gm & T], "#9e9e9e", "-"), ("class T: EMITTED", em[v][gm & T], "#e8742c", "-"),
                                ("class C linked: EMITTED", em[v][gm & CL], "#1f9e76", "-"), ("class C unlinked: EMITTED", em[v][gm & CU], "#2a78d6", "-")):
            h, _ = np.histogram(np.clip(x, ee[v][0] + 1e-6, ee[v][-1] - 1e-6), bins=ee[v])
            ax.stairs(h / max(h.sum(), 1) / np.diff(ee[v]), ee[v], color=col, ls=ls, lw=1.5, label=lab)
            rows.append(dict(group=g, angle="alpha" if v == "A" else "beta", curve=lab, N=len(x), mean=round(float(np.mean(x)), 2),
                             rms=round(float(np.sqrt(np.mean(x ** 2))), 2), frac_abs_gt60=f"{100 * np.mean(np.abs(x) > 60):.2f}%"))
        ax.set_yscale("log"); ax.set_ylim(1e-6, 0.3); ax.grid(True, color="#eeeeee")
        ax.set_title(f"CMS {'alpha' if v == 'A' else 'beta'} - {g}", fontsize=10); ax.set_xlabel("angle [deg]"); ax.set_ylabel("density (unit area)")
h_, l_ = axes[0, 0].get_legend_handles_labels()
fig.legend(h_, l_, loc="outside lower center", ncol=4, fontsize=9, frameon=False)
fig.suptitle("CMSSW angleSource=emu: emitted angle distributions, class T vs class C", fontsize=10)
E2 = pd.DataFrame(rows); E2.to_csv(os.path.join(HERE, "results/emu_nano_classC_vs_T.csv"), index=False)
with PdfPages(os.path.join(HERE, "figs/emu_nano_classC_vs_T.pdf")) as pdf:
    pdf.savefig(fig); plt.close(fig)
    table_pages(pdf, E2, "Emitted angle statistics")
    sg = sat_g.copy(); sg[["satAlpha", "satBeta"]] = sg[["satAlpha", "satBeta"]].round(4)
    table_pages(pdf, sg.rename(columns={"satAlpha": "satAlpha [%]", "satBeta": "satBeta [%]"}), "Saturation rates, all files, per group x class")

# leak test
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
y = (D.tpKey.values < 0).astype(int)
half = (D.event.values % 2 == 0)
base = ["sizeX", "sizeY", "size", "charge", "surface", "globalR", "globalZ", "localX", "localY", "sigX", "sigY"]
ang = ["localCotAlpha", "localCotBeta", "sigAlpha", "sigBeta", "hasAlpha", "hasBeta", "satAlpha", "satBeta"]
rng = np.random.default_rng(3); leak = []
for g in GROUPS + ["all"]:
    gm = np.ones(len(D), bool) if g == "all" else (D.group == g).values
    res = {}
    for tag, cols in (("no angles", base), ("with the emu angle columns", base + ang)):
        X = D[cols].values.astype(np.float64); X[~np.isfinite(X)] = -999
        tri = np.where(half & gm)[0]; tri = rng.choice(tri, min(len(tri), 1_000_000), replace=False); tei = np.where(~half & gm)[0]
        clf = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=31, learning_rate=0.1, random_state=1).fit(X[tri], y[tri])
        res[tag] = float(roc_auc_score(y[tei], clf.predict_proba(X[tei])[:, 1]))
    leak.append(dict(group=g, auc_no_angles=round(res["no angles"], 4), auc_with_angles=round(res["with the emu angle columns"], 4),
                     gain=round(res["with the emu angle columns"] - res["no angles"], 4), n=int(gm.sum()), frac_unlinked=round(float(y[gm].mean()), 4)))
    print(leak[-1], flush=True)
L = pd.DataFrame(leak); L.to_csv(os.path.join(HERE, "results/emu_nano_leak.csv"), index=False)
hs = has.copy(); hs[["hasAlpha", "hasBeta"]] = hs[["hasAlpha", "hasBeta"]].round(6)
with PdfPages(os.path.join(HERE, "figs/emu_nano_leak.pdf")) as pdf:
    table_pages(pdf, L, "Leak test: HGB AUC(unlinked vs linked), trained on even events, tested on odd (plot files)")
    table_pages(pdf, hs, "hasAlpha / hasBeta, linked (tpKey >= 0) vs unlinked, all files")
    st = sat.copy(); st[["satAlpha", "satBeta"]] = st[["satAlpha", "satBeta"]].round(4)
    table_pages(pdf, st.rename(columns={"satAlpha": "satAlpha [%]", "satBeta": "satBeta [%]"}), "Saturation per surface x class (all files)")
    table_pages(pdf, cls_frac.round(4), "Class fractions (TRUTH-ONLY tpEmuClass), all files")
json.dump({"leak": leak, "saturation_group": sat_g.to_dict("records"), "saturation_surface": sat.to_dict("records"),
           "has": has.to_dict("records"), "class_frac": cls_frac.to_dict("records")},
          open(os.path.join(HERE, "results/emu_nano_validation.json"), "w"), indent=1, default=float)
print(sat_g.to_string()); print(has.to_string()); print(L.to_string())
