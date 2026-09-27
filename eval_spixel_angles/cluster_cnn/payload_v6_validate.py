"""Tests of the PixelAV angle payload v6 (a fine cotBeta axis) against v5. TESTS ONLY.

  payload_v6_validate.py --v5 V5.json --v6 V6.json --dump DUMP.parquet --nano-v5 OUT_V5.root --nano-v6 OUT_V6.root

1. Closure on the rescaled Conv1D_Full-2bit dump the payloads were built from (in-sample: a consistency check of the
   binning, not an independent test). Per signed cotBeta band (the v6 bins) and angle: the dump residual (pred - true)
   vs each payload applied as the producer does (the fused spix_angle_*_shift compound at the true angles, layer 1,
   bLocalY -3.81, prngAcc 1.0): core RMS (mode-seeded Gaussian core, v2_common.core_fit), full RMS, (q84 - q16) / 2 and
   median; pulls (dump residual - bias) / sigma: q-width and median.
2. Nano (RelVal file 3, 100 events, 4 threads, the same config apart from the payload): v5 -> v6 refit (d0 / z0 / pT q68
   vs the TP, chi2 summaries, selected-hit purity), branch identity outside the angle-dependent SmartPixels branches, and
   NaN / inf in every SmartPixels float branch.
Figure (vector PDF, legend outside the axes, statistics on table pages): figs/payload_v6.pdf; results/payload_v6_validation.json.
"""
import argparse, os, sys, json, numpy as np, pandas as pd, uproot, awkward as ak, matplotlib, correctionlib
import pyarrow.parquet as pq
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from v2_common import core_fit
from extract_pixelav_angle_payload import ALPHA_BETA_MAPPING, RESIDUAL_IS_TRUE_MINUS_PRED

p = argparse.ArgumentParser()
for k in ("--v5", "--v6", "--dump", "--nano-v5", "--nano-v6"):
    p.add_argument(k, required=True)
a = p.parse_args()
OUT = {}
COL = {"dump": "#1b1b1b", "v5": "#2a78d6", "v6": "#eb6834"}
LS = {"dump": "-", "v5": "--", "v6": ":"}


def table_pages(pdf, df, title, per=30):
    for i in range(0, max(len(df), 1), per):
        t = df.iloc[i:i + per]
        f, ax = plt.subplots(figsize=(min(26, 2 + 1.6 * len(t.columns)), 0.33 * len(t) + 1.4)); ax.axis("off")
        tb = ax.table(cellText=t.astype(str).values, colLabels=list(t.columns), loc="center", cellLoc="left")
        tb.auto_set_font_size(False); tb.set_fontsize(7.5); tb.auto_set_column_width(list(range(len(t.columns))))
        ax.set_title(title + (f" ({i // per + 1})" if len(df) > per else ""), fontsize=10); pdf.savefig(f, bbox_inches="tight"); plt.close(f)


def fmt(df):
    return df.apply(lambda c: c.map(lambda x: f"{x:.4g}" if isinstance(x, (float, np.floating)) else x))


def beta_edges(path):
    node = [c for c in json.load(open(path))["corrections"] if c["name"] == "spix_angle_beta_sigma"][0]["data"]["content"][0]["value"]
    return node["edges"][node["inputs"].index("cotBeta")]


# ======================= 1. dump closure =======================
have = set(pq.ParquetFile(a.dump).schema_arrow.names)
SA, SB = ALPHA_BETA_MAPPING(have, have, have)
t = pq.read_table(a.dump, columns=[SA["true"], SA["resid"], SB["true"], SB["resid"]]).to_pandas()
sgn = -1.0 if RESIDUAL_IS_TRUE_MINUS_PRED else 1.0
ca, cb = SA["scale"] * t[SA["true"]].values, SB["scale"] * t[SB["true"]].values
RES = {"alpha": sgn * SA["scale"] * t[SA["resid"]].values, "beta": sgn * SB["scale"] * t[SB["resid"]].values}
n = len(t); lay = np.ones(n, np.int64); bly = np.full(n, -3.81); one = np.ones(n)
APP, BIAS, SIG = {}, {}, {}
for v, path in (("v5", a.v5), ("v6", a.v6)):
    cset = correctionlib.CorrectionSet.from_file(path)
    for ang in ("alpha", "beta"):
        APP[v, ang] = cset.compound[f"spix_angle_{ang}_shift"].evaluate(lay, ca, cb, bly, one)
        BIAS[v, ang] = cset[f"spix_angle_{ang}_bias"].evaluate(lay, ca, cb, bly)
        SIG[v, ang] = cset[f"spix_angle_{ang}_sigma"].evaluate(lay, ca, cb, bly)
BE5, BE6 = beta_edges(a.v5), beta_edges(a.v6)
band = np.clip(np.digitize(cb, BE6) - 1, 0, len(BE6) - 2)
deg = lambda c: float(np.degrees(np.arctan(c)))


def wid(x):
    q16, q50, q84 = np.quantile(x, [0.16, 0.5, 0.84])
    return (q84 - q16) / 2, q50


rows = []
for ang in ("alpha", "beta"):
    for j in list(range(len(BE6) - 1)) + ["all"]:
        m = np.ones(n, bool) if j == "all" else band == j
        lab = "all" if j == "all" else f"[{deg(BE6[j]):+.0f}, {deg(BE6[j + 1]):+.0f}]"
        r = dict(angle=ang, beta_band_deg=lab, N=int(m.sum()))
        for src, x in (("dump", RES[ang][m]), ("v5", APP["v5", ang][m]), ("v6", APP["v6", ang][m])):
            w, med = wid(x)
            r[f"{src}_core_rms"] = core_fit(x)[0]; r[f"{src}_full_rms"] = float(np.sqrt(np.mean(x ** 2)))
            r[f"{src}_q_width"] = w; r[f"{src}_median"] = med
        for v in ("v5", "v6"):
            pl = (RES[ang][m] - BIAS[v, ang][m]) / SIG[v, ang][m]
            r[f"{v}_pull_q_width"], r[f"{v}_pull_median"] = wid(pl)
        rows.append(r)
CL = pd.DataFrame(rows); OUT["dump_closure"] = CL.to_dict("records")
OUT["payload_bins"] = {v: dict(cotBeta_edges=e, n_cotBeta=len(e) - 1, file_bytes=os.path.getsize(pth))
                       for v, e, pth in (("v5", BE5, a.v5), ("v6", BE6, a.v6))}


# ======================= 2. nano =======================
PC = "L1TSmartPixelsCluster_"
RT, RH = "L1TSmartPixelsTrackDigiRefitAAAA_", "L1TSmartPixelsRefitHitDigiRefitAAAA_"
TRK = ["pt", "d0", "z0", "spixRefitPerformed", "spixNAcceptedHits", "spixChi2IncXTot", "spixChi2IncYTot",
       "spixChi2IncAlphaTot", "spixChi2IncBetaTot"]
HIT = ["hitAccepted", "selHitClass"]


def refit_summary(tag, f):
    ev = uproot.open(f)["Events"].arrays([RT + c for c in TRK] + [RH + c for c in HIT] + ["L1TTrack_tp_pt", "L1TTrack_tp_d0",
                                          "L1TTrack_tp_z0", "event"], library="ak")
    ev = ev[np.argsort(ak.to_numpy(ev["event"]), kind="stable")]
    fl = lambda x: ak.to_numpy(ak.flatten(x))
    T = pd.DataFrame({c: fl(ev[RT + c]) for c in TRK})
    for c in ("tp_pt", "tp_d0", "tp_z0"):
        T["base_" + c] = fl(ev["L1TTrack_" + c])
    T["chi2IT"] = T.spixChi2IncXTot + T.spixChi2IncYTot + T.spixChi2IncAlphaTot + T.spixChi2IncBetaTot
    cls = fl(ev[RH + "selHitClass"][ev[RH + "hitAccepted"] == 1])
    m = (T.spixRefitPerformed == 1) & (T.base_tp_pt > 0)
    out = []
    for nm, r in (("d0 [um]", 1e4 * (T.d0 - T.base_tp_d0)), ("z0 [um]", 1e4 * (T.z0 - T.base_tp_z0)),
                  ("pT (reco - true) / true", (T.pt - T.base_tp_pt) / T.base_tp_pt)):
        for lo, hi in ((0, 1e9), (2, 5), (5, 1e9)):
            x = r[m & (T.base_tp_pt >= lo) & (T.base_tp_pt < hi)].values
            w, med = wid(x)
            out.append(dict(run=tag, quantity=nm, tp_pt=f"[{lo}, {hi if hi < 1e9 else 'inf'})", n=len(x), q68_halfwidth=w, median=med))
    for nm, x in (("chi2 IT total", T.chi2IT[m]), ("chi2 alpha", T.spixChi2IncAlphaTot[m]), ("chi2 beta", T.spixChi2IncBetaTot[m])):
        out.append(dict(run=tag, quantity=nm, tp_pt="all", n=int(m.sum()), median=float(np.median(x)),
                        q90=float(np.quantile(x, 0.9)), q99=float(np.quantile(x, 0.99)), frac_gt_100=float(np.mean(x > 100))))
    out.append(dict(run=tag, quantity="selected-hit purity (own TP)", tp_pt="all", n=len(cls), median=float(np.mean(cls == 0))))
    return out, T[m]


RF, TR = [], {}
for tag, f in (("v5", a.nano_v5), ("v6", a.nano_v6)):
    r, TR[tag] = refit_summary(tag, f); RF += r
RF = pd.DataFrame(RF); OUT["refit"] = RF.to_dict("records")


def compare(f1, f2):
    t1, t2 = uproot.open(f1)["Events"], uproot.open(f2)["Events"]
    o1, o2 = (np.argsort(x["event"].array(library="np"), kind="stable") for x in (t1, t2))
    same, diff = [], []
    for k in t1.keys():
        if k not in t2.keys():
            continue
        x1 = ak.to_numpy(ak.flatten(t1[k].array(library="ak")[o1], axis=None))
        x2 = ak.to_numpy(ak.flatten(t2[k].array(library="ak")[o2], axis=None))
        ok = x1.shape == x2.shape and bool(np.all((x1 == x2) | (np.isnan(x1) & np.isnan(x2)) if x1.dtype.kind == "f" else x1 == x2))
        (same if ok else diff).append(k)
    return dict(n_compared=len(same) + len(diff), n_identical=len(same), n_differing_SP=sum("SmartPixels" in k for k in diff),
                differing_nonSP=[k for k in diff if "SmartPixels" not in k], differing_SP=diff,
                only_in_one=sorted(set(t1.keys()) ^ set(t2.keys())))


OUT["branch_compare_v5_v6"] = compare(a.nano_v5, a.nano_v6)
t6 = uproot.open(a.nano_v6)["Events"]
FLT = [k for k in t6.keys() if "SmartPixels" in k and t6[k].typename.replace("[]", "") in ("float", "double")]
bad = {}
for k in FLT:
    nb_ = int((~np.isfinite(ak.to_numpy(ak.flatten(t6[k].array(library="ak"), axis=None)))).sum())
    if nb_:
        bad[k] = nb_
OUT["nan_inf_v6"] = dict(n_float_SP_branches=len(FLT), non_finite=bad)

# ======================= figure =======================
cen = [0.5 * (deg(BE6[j]) + deg(BE6[j + 1])) for j in range(len(BE6) - 1)]
C = CL[CL.beta_band_deg != "all"]
with PdfPages(os.path.join(HERE, "figs", "payload_v6.pdf")) as pdf:
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5), layout="constrained")
    for i, ang in enumerate(("alpha", "beta")):
        Ci = C[C.angle == ang]
        for ax, (q, yl) in zip(axes[i], (("q_width", "(q84 - q16) / 2"), ("core_rms", "core RMS"), ("full_rms", "full RMS"))):
            for src in ("dump", "v5", "v6"):
                ax.plot(cen, Ci[f"{src}_{q}"], color=COL[src], ls=LS[src], lw=2, marker="o", ms=4,
                        label={"dump": "rescaled dump (pred - true)", "v5": "payload v5 applied", "v6": "payload v6 applied"}[src])
            for e in BE5:
                ax.axvline(deg(e), color="#cccccc", lw=0.8, zorder=0)
            ax.set_title(f"cot{ang.capitalize()} residual: {yl}", fontsize=10); ax.set_xlabel("true beta [deg] (band centre; grey = v5 edges)")
            ax.set_ylabel(f"{yl} [cot]"); ax.grid(True, color="#eeeeee")
    h_, l_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="outside lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle("PixelAV payload closure on the Conv1D_Full-2bit dump (in-sample), per v6 cotBeta band", fontsize=11)
    pdf.savefig(fig); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    for i, ang in enumerate(("alpha", "beta")):
        Ci = C[C.angle == ang]
        for ax, q in zip(axes[i], ("pull_q_width", "pull_median")):
            for v in ("v5", "v6"):
                ax.plot(cen, Ci[f"{v}_{q}"], color=COL[v], ls=LS[v], lw=2, marker="o", ms=4, label=f"payload {v}")
            ax.axhline(1.0 if q == "pull_q_width" else 0.0, color="#1b1b1b", lw=1, label="ideal")
            ax.set_title(f"cot{ang.capitalize()} pull (dump residual - bias) / sigma: {q.replace('pull_', '')}", fontsize=10)
            ax.set_xlabel("true beta [deg] (band centre)"); ax.grid(True, color="#eeeeee")
    h_, l_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="outside lower center", ncol=3, fontsize=9, frameon=False)
    pdf.savefig(fig); plt.close(fig)
    sel = [0, 5, 9, 11, 12, 14, 18, 23]
    fig, axes = plt.subplots(2, len(sel), figsize=(3.1 * len(sel), 7), layout="constrained")
    for i, ang in enumerate(("alpha", "beta")):
        for ax, j in zip(axes[i], sel):
            m = band == j
            e = np.linspace(-1.0, 1.0, 81) if ang == "beta" else np.linspace(-0.5, 0.5, 81)
            for src, x in (("dump", RES[ang][m]), ("v5", APP["v5", ang][m]), ("v6", APP["v6", ang][m])):
                h, _ = np.histogram(np.clip(x, e[0] + 1e-9, e[-1] - 1e-9), bins=e)
                ax.stairs(h / max(h.sum(), 1), e, color=COL[src], ls=LS[src], lw=1.6, label=src)
            ax.set_yscale("log"); ax.grid(True, color="#eeeeee"); ax.tick_params(labelsize=7)
            ax.set_title(f"cot{ang.capitalize()}, beta [{deg(BE6[j]):+.0f}, {deg(BE6[j + 1]):+.0f}] deg", fontsize=8)
            ax.set_xlabel("pred - true [cot] (ends = overflow)", fontsize=7)
    h_, l_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="outside lower center", ncol=3, fontsize=9, frameon=False)
    pdf.savefig(fig); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout="constrained")
    specs = [("d0 - true d0 [um]", lambda T: 1e4 * (T.d0 - T.base_tp_d0), np.linspace(-300, 300, 121)),
             ("z0 - true z0 [um]", lambda T: 1e4 * (T.z0 - T.base_tp_z0), np.linspace(-1500, 1500, 121)),
             ("chi2 beta", lambda T: T.spixChi2IncBetaTot, np.logspace(-3, 5, 141))]
    for ax, (nm, fn, e) in zip(axes, specs):
        for v in ("v5", "v6"):
            x = fn(TR[v]).values
            h, _ = np.histogram(np.clip(x, e[0] * (1 + 1e-9) if e[0] > 0 else e[0] + 1e-9, e[-1] * (1 - 1e-9)), bins=e)
            ax.stairs(h / max(h.sum(), 1), e, color=COL[v], ls=LS[v], lw=1.6, label=f"payload {v}")
        ax.set_yscale("log"); ax.grid(True, color="#eeeeee"); ax.set_xlabel(nm + " (ends = overflow)"); ax.set_ylabel("fraction of refit tracks")
        if e[0] > 0:
            ax.set_xscale("log")
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="outside lower center", ncol=2, fontsize=9, frameon=False)
    fig.suptitle("Refit (TP-matched refit tracks), RelVal file 3, 100 events", fontsize=10)
    pdf.savefig(fig); plt.close(fig)
    cols = ["angle", "beta_band_deg", "N"] + [f"{s}_{q}" for q in ("q_width", "core_rms", "full_rms", "median") for s in ("dump", "v5", "v6")]
    table_pages(pdf, fmt(CL[cols]), "Dump closure per cotBeta band [cot]: q-width, core RMS, full RMS, median")
    table_pages(pdf, fmt(CL[["angle", "beta_band_deg", "N", "v5_pull_q_width", "v6_pull_q_width", "v5_pull_median", "v6_pull_median"]]),
                "Dump closure: pulls (dump residual - bias) / sigma")
    table_pages(pdf, fmt(RF), "Refit v5 vs v6: residual q68 half-widths and medians vs the TP; chi2; selected-hit purity (median column)")
    bc = OUT["branch_compare_v5_v6"]
    table_pages(pdf, pd.DataFrame([dict(check="branches compared", value=bc["n_compared"]), dict(check="identical", value=bc["n_identical"]),
                                   dict(check="differing SmartPixels", value=bc["n_differing_SP"]),
                                   dict(check="differing non-SmartPixels", value=len(bc["differing_nonSP"])),
                                   dict(check="non-finite values in v6 SmartPixels float branches", value=sum(bad.values())),
                                   dict(check="file bytes v5 / v6", value=f"{os.path.getsize(a.v5)} / {os.path.getsize(a.v6)}")]),
                "Nano checks")
json.dump(OUT, open(os.path.join(HERE, "results", "payload_v6_validation.json"), "w"), indent=1, default=float)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(fmt(CL[cols]).to_string()); print(fmt(CL[["angle", "beta_band_deg", "v5_pull_q_width", "v6_pull_q_width", "v5_pull_median", "v6_pull_median"]]).to_string())
print(fmt(RF).to_string()); print("float SP branches checked:", len(FLT))
print({k: v for k, v in bc.items() if k != "differing_SP"}, "\nSP differing:", bc["differing_SP"]); print("non-finite:", bad)
