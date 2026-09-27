"""Static sigAlpha/sigBeta table of the CMSSW SmartPixels-output emulator (angleSource = "emu"), INCLUSIVE of both classes.

  emu_sigma_table.py build OUT.json     the table (training events)
  emu_sigma_table.py pulls OUT.json     pull checks of the table on the test events

Population (training events = split 0, the class-T clusters, which are the only ones with a consistent truth): every cluster
is emulated through BOTH routes -- the class-T route (truth pseudo-angle) and the class-C route (tiny CNN + A1 draw) -- and
the two are pooled with weights f_T(surface) and f_C(surface) = the production class fractions of that surface on the same
events, so the pooled residuals describe the inclusive (T + C) emitted population.
sigma = (q84 - q16) / 2 of (emitted - true) COT (weighted quantiles), keyed ONLY on what the sensor reports: the surface
code (1-4 = IL1-4, 11-14 = ID1-4) and the |emitted cot| of that angle. Rows whose emitted angle saturated are excluded, and
so is |emitted cot| > CAP (the refit's measAngleMaxAbs): the top bin ends at CAP and the table clamps above it. Adjacent
bins are merged (upwards; a sparse last bin into its predecessor) until each holds >= MIN_N clusters per route.
Also reported per bin (not used by the producer): the weighted core RMS (mode-seeded Gaussian core about 0, v2_common.core_fit)
and full RMS, in cot -> results/emu_sigma_table.csv.
Pulls (emitted - true cot) / sigma on the test events (split 2), per surface and route: q-width (q84 - q16) / 2 and the fraction
with |pull| > 3 -> results/emu_sigma_pulls.csv.
"""
import sys, os, numpy as np, pandas as pd
import correctionlib.schemav2 as cs
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2_common import D_ALL, core_fit
from emu_common import load_library, emulate, load_a1, a1_draw, a1_uniforms, class_t, N_A1_IN
from emu_train_a1 import load_cache, LIB, OUTD

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = 12.0
EDGES = [0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.07, 1.5, 2, 3, 4, 6, CAP]
MIN_N = 500


def class_fractions(split):
    m = pd.read_parquet(os.path.join(D_ALL, "meta.parquet"), columns=["subdet", "layer", "disk", "linkClass", "tpCharge", "hxOk", "hxDist"],
                        filters=[("split", "==", split)])
    m = m[(m.subdet == 1) | ((m.subdet == 2) & (m.disk <= 4))]
    surf = np.where(m.subdet.values == 1, m.layer.values, 10 + m.disk.values)
    T = class_t(m.linkClass.values == 2, m.tpCharge.values, m.hxOk.values, m.hxDist.values)
    return pd.Series(T).groupby(surf).mean().to_dict()


def routes(split):
    """both routes on the class-T clusters of a split: emitted cot, true cot, saturation, surface, route"""
    parts = []
    for reg in ("barrel", "disc"):
        a1 = load_a1(os.path.join(OUTD, f"a1_{reg}.pt")); lib = load_library(LIB)
        C = load_cache(reg, [split])
        for i in range(0, len(C), 500_000):
            c = C.iloc[i:i + 500_000]; n = len(c)
            keys = (c.run.values, c.event.values, c.detId.values, c.idxInDet.values)
            d = a1_draw(a1, c[[f"z{j}" for j in range(N_A1_IN)]].values, a1_uniforms(*keys))
            for route, cls in (("T", True), ("C", False)):
                e = emulate(lib, c.surface.values, np.full(n, cls), c.tcA.values, c.tcB.values, c.z0.values, c.z1.values,
                            d[:, 0], d[:, 1], *keys)
                parts.append(pd.DataFrame({"surface": c.surface.values, "route": route, "cotA": e["cotA"], "cotB": e["cotB"],
                                           "satA": e["satA"], "satB": e["satB"], "tcA": c.tcA.values, "tcB": c.tcB.values}))
            print(reg, "split", split, "emulated", i + n, "/", len(C), flush=True)
    return pd.concat(parts, ignore_index=True)


def wquant(x, w, qs):
    o = np.argsort(x); x, w = x[o], w[o]; c = np.cumsum(w) - 0.5 * w; c /= w.sum()
    return np.interp(qs, c, x)


def merged_bins(cntT, cntC):
    """indices of EDGES kept as bin boundaries: merge upwards until both routes have >= MIN_N, last sparse bin down"""
    keep, accT, accC = [0], 0, 0
    for i in range(len(cntT)):
        accT += cntT[i]; accC += cntC[i]
        if accT >= MIN_N and accC >= MIN_N:
            keep.append(i + 1); accT = accC = 0
    if keep[-1] != len(cntT):
        if len(keep) > 1:
            keep[-1] = len(cntT)
        else:
            keep.append(len(cntT))
    return keep


def build(out):
    fr = class_fractions(0)
    P = routes(0)
    rows, corrs = [], []
    for ang, nm in (("A", "alpha"), ("B", "beta")):
        content = []
        for key in sorted(P.surface.unique()):
            fT = fr[key]
            d = P[(P.surface == key) & ~P[f"sat{ang}"].values]
            pc = d[f"cot{ang}"].values.astype(np.float64); tc = d[f"tc{ang}"].values.astype(np.float64)
            isT = (d.route == "T").values
            nT, nC = isT.sum(), (~isT).sum()
            w = np.where(isT, fT / nT, (1 - fT) / nC)
            ok = np.abs(pc) <= CAP
            b = np.clip(np.digitize(np.abs(pc), EDGES[1:-1]), 0, len(EDGES) - 2)
            cntT = [int((ok & isT & (b == i)).sum()) for i in range(len(EDGES) - 1)]
            cntC = [int((ok & ~isT & (b == i)).sum()) for i in range(len(EDGES) - 1)]
            keep = merged_bins(cntT, cntC)
            edges = [EDGES[k] for k in keep]
            sig = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                s = ok & (np.abs(pc) >= lo) & (np.abs(pc) < hi if hi < CAP else np.abs(pc) <= hi)
                r = (pc - tc)[s]; ww = w[s]
                q16, q84 = wquant(r, ww, [0.16, 0.84]); sg = (q84 - q16) / 2; sig.append(sg)
                rows.append(dict(angle=nm, surface=int(key), lo=lo, hi=hi, nT=int((s & isT).sum()), nC=int((s & ~isT).sum()),
                                 wC=float(ww[~isT[s]].sum() / ww.sum()), sigma=sg, core_rms=core_fit(r, ww)[0],
                                 full_rms=float(np.sqrt(np.average(r ** 2, weights=ww))),
                                 sigma_T_only=float(np.diff(np.quantile(r[isT[s]], [0.16, 0.84]))[0] / 2),
                                 sigma_C_only=float(np.diff(np.quantile(r[~isT[s]], [0.16, 0.84]))[0] / 2)))
            content.append(cs.CategoryItem(key=int(key), value=cs.Binning(nodetype="binning", input="absCot", edges=edges,
                                                                           content=[round(float(v), 6) for v in sig], flow="clamp")))
            print(nm, key, "fT", round(fT, 3), "bins", len(edges) - 1, "sigma", np.round(sig, 4).tolist(), flush=True)
        corrs.append(cs.Correction(
            name=f"spix_emu_{nm}_sigma", version=2,
            description=(f"SmartPixels-output emulator cot{nm.capitalize()} resolution: weighted (q84-q16)/2 of (emitted - true) cot "
                         f"of the INCLUSIVE emulated population (class-T and class-C routes on the class-T clusters of the training "
                         f"events, weighted by the class fractions), by surface code (1-4 IL, 11-14 ID) and |reported cot| <= {CAP} "
                         f"(clamped above). Saturated rows excluded; sparse bins merged (>= {MIN_N} per route). Inputs are "
                         f"sensor-reported only."),
            inputs=[cs.Variable(name="surface", type="int", description="IT surface code: 1-4 = IL1-IL4, 11-14 = ID1-ID4"),
                    cs.Variable(name="absCot", type="real", description=f"|reported cot{nm.capitalize()}|")],
            output=cs.Variable(name="sigma", type="real", description=f"cot{nm.capitalize()} resolution"),
            data=cs.Category(nodetype="category", input="surface", content=content)))
    cset = cs.CorrectionSet(schema_version=2, description="SmartPixels-output emulator angle resolution (sigma) tables, v2 (inclusive)",
                            corrections=corrs)
    open(out, "w").write(cset.model_dump_json(exclude_unset=True, indent=1))
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "results/emu_sigma_table.csv"), index=False)
    pd.DataFrame([dict(surface=k, fracT=v) for k, v in fr.items()]).to_csv(os.path.join(HERE, "results/emu_sigma_classfrac.csv"), index=False)
    print("wrote", out)


def pulls(path):
    import correctionlib
    cset = correctionlib.CorrectionSet.from_file(path)
    P = routes(2)
    rows = []
    for ang, nm in (("A", "alpha"), ("B", "beta")):
        cor = cset[f"spix_emu_{nm}_sigma"]
        sg = cor.evaluate(P.surface.values.astype(np.int64), np.abs(P[f"cot{ang}"].values.astype(np.float64)))
        pull = (P[f"cot{ang}"].values.astype(np.float64) - P[f"tc{ang}"].values) / sg
        for (s, rt), g in pd.DataFrame({"surface": P.surface.values, "route": P.route.values, "pull": pull,
                                        "sat": P[f"sat{ang}"].values, "cap": np.abs(P[f"cot{ang}"].values) <= CAP}).groupby(["surface", "route"]):
            for sel_nm, gg in (("all", g), ("|cot|<=12, not saturated", g[g.cap & ~g.sat])):
                q16, q84 = np.quantile(gg.pull, [0.16, 0.84])
                rows.append(dict(angle=nm, surface=int(s), route=rt, selection=sel_nm, n=len(gg), q_width=(q84 - q16) / 2,
                                 median=float(np.median(gg.pull)), frac_abs_gt3=float(np.mean(np.abs(gg.pull) > 3))))
    R = pd.DataFrame(rows); R.to_csv(os.path.join(HERE, "results/emu_sigma_pulls.csv"), index=False)
    pd.set_option("display.width", 200); print(R.round(4).to_string())


if __name__ == "__main__":
    build(sys.argv[2]) if sys.argv[1] == "build" else pulls(sys.argv[2])
