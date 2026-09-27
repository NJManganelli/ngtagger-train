"""Static sigAlpha/sigBeta table of the CMSSW SmartPixels-output emulator (angleSource = "emu"), INCLUSIVE of both classes.

  emu_sigma_table.py routes                       both routes on the training (split 0) and test (split 2) events -> ROUTES cache
  emu_sigma_table.py build OUT.json [KEY]         the table (training events); KEY = none (v2, default) | own | both (v3)
  emu_sigma_table.py pulls OUT.json [TAG]         pull checks of the table on the test events

Population (training events = split 0, the class-T clusters, which are the only ones with a consistent truth): every cluster
is emulated through BOTH routes -- the class-T route (truth pseudo-angle) and the class-C route (tiny CNN + A1 draw) -- and
the two are pooled with weights f_T and f_C = 1 - f_T, the production class fractions on the same events of the key cell the
cluster falls in (surface x cluster-size bins), so the pooled residuals describe the inclusive (T + C) emitted population.
sigma = (q84 - q16) / 2 of (emitted - true) COT (weighted quantiles), keyed ONLY on what the sensor reports: the surface
code (1-4 = IL1-4, 11-14 = ID1-4), optionally the cluster size(s) (KEY: none = v2, the adopted table; own = sizeX for
alpha, sizeY for beta; both = sizeX and sizeY for either angle; the size keys were studied as v3 and not adopted; ASSUMPTION:
the cluster size counts as sensor-reported) and the |emitted cot| of that angle.
Rows whose emitted angle saturated are excluded, and so is |emitted cot| > CAP (the refit's measAngleMaxAbs): the top cot bin
ends at CAP and the table clamps above it. Bins are chosen from the data, outermost key first: integer sizes are merged
upwards until each bin holds >= MIN_N clusters per route (a sparse last bin into its predecessor; the table clamps sizes
beyond the last bin), then the cot bins inside each size bin the same way.
Also reported per bin (not used by the producer): the weighted core RMS (mode-seeded Gaussian core about 0, v2_common.core_fit)
and full RMS, in cot -> results/emu_sigma_table.csv.
Pulls (emitted - true cot) / sigma on the test events (split 2), per surface and route: q-width (q84 - q16) / 2 and the fraction
with |pull| > 3, plus the inclusive (class-weighted) population and the q-width per own-size bin -> results/emu_sigma_pulls[_TAG].csv.
"""
import sys, os, numpy as np, pandas as pd
import correctionlib.schemav2 as cs
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2_common import D_ALL, core_fit
from emu_common import load_library, emulate, load_a1, a1_draw, a1_uniforms, class_t, N_A1_IN, A1_OBS
from emu_train_a1 import load_cache, LIB, OUTD

HERE = os.path.dirname(os.path.abspath(__file__))
ROUTES = "/Volumes/WDMac/smartpixels_scratch/emu_fix2/cache/routes_s{split}.parquet"
CAP = 12.0
EDGES = [0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.07, 1.5, 2, 3, 4, 6, CAP]
MIN_N = 500
KEYS = {"none": {"A": [], "B": []}, "own": {"A": ["sizeX"], "B": ["sizeY"]},
        "both": {"A": ["sizeX", "sizeY"], "B": ["sizeY", "sizeX"]}}
SIZE_COL = {"sizeX": f"z{A1_OBS.index('sizeX')}", "sizeY": f"z{A1_OBS.index('sizeY')}"}


def class_meta(split):
    """every cluster of the split on the emulated surfaces: surface, sizes, class T (truth, for the class fractions only)"""
    m = pd.read_parquet(os.path.join(D_ALL, "meta.parquet"),
                        columns=["subdet", "layer", "disk", "sizeX", "sizeY", "linkClass", "tpCharge", "hxOk", "hxDist"],
                        filters=[("split", "==", split)])
    m = m[(m.subdet == 1) | ((m.subdet == 2) & (m.disk <= 4))]
    return pd.DataFrame({"surface": np.where(m.subdet.values == 1, m.layer.values, 10 + m.disk.values),
                         "sizeX": m.sizeX.values, "sizeY": m.sizeY.values,
                         "T": class_t(m.linkClass.values == 2, m.tpCharge.values, m.hxOk.values, m.hxDist.values)})


def routes(split):
    """both routes on the class-T clusters of a split: emitted cot, true cot, saturation, surface, sizes, route (cached)"""
    path = ROUTES.format(split=split)
    if os.path.exists(path):
        return pd.read_parquet(path)
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
                                           "satA": e["satA"], "satB": e["satB"], "tcA": c.tcA.values, "tcB": c.tcB.values,
                                           "sizeX": c[SIZE_COL["sizeX"]].values.astype(np.int32),
                                           "sizeY": c[SIZE_COL["sizeY"]].values.astype(np.int32)}))
            print(reg, "split", split, "emulated", i + n, "/", len(C), flush=True)
    P = pd.concat(parts, ignore_index=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    P.to_parquet(path)
    return P


def wquant(x, w, qs):
    o = np.argsort(x); x, w = x[o], w[o]; c = np.cumsum(w) - 0.5 * w; c /= w.sum()
    return np.interp(qs, c, x)


def merged_bins(cntT, cntC):
    """indices of the fine edges kept as bin boundaries: merge upwards until both routes have >= MIN_N, last sparse bin down"""
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


def size_edges(x, isT):
    """half-integer size edges from the data: each bin >= MIN_N per route; clamp beyond the last edge"""
    top = int(max(x.max(), 1))
    cntT = np.bincount(x[isT], minlength=top + 1)[1:]; cntC = np.bincount(x[~isT], minlength=top + 1)[1:]
    return [k + 0.5 for k in merged_bins(cntT, cntC)]


def in_bin(x, lo, hi, first, last):
    return (first | (x >= lo)) & (last | (x < hi))


class Builder:
    def __init__(self, P, M, ang):
        self.P, self.M, self.ang, self.rows = P, M, ang, []

    def node(self, key, sel, msel, keys, ranges):
        """the correctionlib node for the clusters sel (routes) / msel (class-fraction population), keyed on keys"""
        P, ang = self.P, self.ang
        if keys:
            var = keys[0]
            x = P[var].values
            ok = sel & (np.abs(P[f"cot{ang}"].values) <= CAP) & ~P[f"sat{ang}"].values
            isT = (P.route == "T").values
            edges = size_edges(x[ok], isT[ok])
            content = []
            for j, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
                first, last = j == 0, j == len(edges) - 2
                content.append(self.node(key, sel & in_bin(x, lo, hi, first, last),
                                         msel & in_bin(self.M[var].values, lo, hi, first, last), keys[1:],
                                         {**ranges, var: (lo if not first else 0.5, hi if not last else np.inf)}))
            return cs.Binning(nodetype="binning", input=var, edges=edges, content=content, flow="clamp")
        return self.leaf(key, sel, msel, ranges)

    def leaf(self, key, sel, msel, ranges):
        P, ang = self.P, self.ang
        fT = float(self.M["T"].values[msel].mean())
        d = P[sel & ~P[f"sat{ang}"].values]
        pc = d[f"cot{ang}"].values.astype(np.float64); tc = d[f"tc{ang}"].values.astype(np.float64)
        isT = (d.route == "T").values
        nT, nC = isT.sum(), (~isT).sum()
        w = np.where(isT, fT / nT, (1 - fT) / nC)
        ok = np.abs(pc) <= CAP
        b = np.clip(np.digitize(np.abs(pc), EDGES[1:-1]), 0, len(EDGES) - 2)
        cntT = [int((ok & isT & (b == i)).sum()) for i in range(len(EDGES) - 1)]
        cntC = [int((ok & ~isT & (b == i)).sum()) for i in range(len(EDGES) - 1)]
        edges = [EDGES[k] for k in merged_bins(cntT, cntC)]
        sig = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = ok & (np.abs(pc) >= lo) & (np.abs(pc) < hi if hi < CAP else np.abs(pc) <= hi)
            r = (pc - tc)[s]; ww = w[s]
            q16, q84 = wquant(r, ww, [0.16, 0.84]); sg = (q84 - q16) / 2; sig.append(sg)
            self.rows.append(dict(angle="alpha" if ang == "A" else "beta", surface=int(key),
                                  **{f"{v}_lo": ranges[v][0] for v in ranges}, **{f"{v}_hi": ranges[v][1] for v in ranges},
                                  fT=fT, lo=lo, hi=hi, nT=int((s & isT).sum()), nC=int((s & ~isT).sum()),
                                  wC=float(ww[~isT[s]].sum() / ww.sum()), sigma=sg, core_rms=core_fit(r, ww)[0],
                                  full_rms=float(np.sqrt(np.average(r ** 2, weights=ww))),
                                  sigma_T_only=float(np.diff(np.quantile(r[isT[s]], [0.16, 0.84]))[0] / 2),
                                  sigma_C_only=float(np.diff(np.quantile(r[~isT[s]], [0.16, 0.84]))[0] / 2)))
        return cs.Binning(nodetype="binning", input="absCot", edges=edges, content=[round(float(v), 6) for v in sig], flow="clamp")


def build(out, key="none"):
    version = 2 if key == "none" else 3
    M = class_meta(0)
    P = routes(0)
    rows, corrs = [], []
    for ang, nm in (("A", "alpha"), ("B", "beta")):
        B = Builder(P, M, ang)
        keys = KEYS[key][ang]
        content = [cs.CategoryItem(key=int(s), value=B.node(s, (P.surface == s).values, (M.surface == s).values, keys, {}))
                   for s in sorted(P.surface.unique())]
        rows += B.rows
        R = pd.DataFrame(B.rows)
        for s, g in R.groupby("surface"):
            print(nm, s, "leaves", len(g.drop_duplicates([c for c in g.columns if c.endswith(("_lo", "_hi"))] or ["surface"])),
                  "bins", len(g), "sigma range", round(g.sigma.min(), 4), "-", round(g.sigma.max(), 4), flush=True)
        size_txt = {"none": "", "own": f" x cluster size{' along x (sizeX)' if ang == 'A' else ' along y (sizeY)'}",
                    "both": " x cluster sizes (sizeX, sizeY)"}[key]
        corrs.append(cs.Correction(
            name=f"spix_emu_{nm}_sigma", version=version,
            description=(f"SmartPixels-output emulator cot{nm.capitalize()} resolution: weighted (q84-q16)/2 of (emitted - true) cot "
                         f"of the INCLUSIVE emulated population (class-T and class-C routes on the class-T clusters of the training "
                         f"events, weighted by the class fractions of the key cell), by surface code (1-4 IL, 11-14 ID){size_txt} "
                         f"and |reported cot| <= {CAP} (clamped above). Saturated rows excluded; bins merged from the data "
                         f"(>= {MIN_N} per route). Inputs are sensor-reported only"
                         f"{'' if key == 'none' else ' (the cluster size is assumed sensor-reported)'}."),
            inputs=[cs.Variable(name="surface", type="int", description="IT surface code: 1-4 = IL1-IL4, 11-14 = ID1-ID4")]
                   + [cs.Variable(name=v, type="int", description=f"cluster bounding-box extent in pixels, local {v[-1].lower()}")
                      for v in keys]
                   + [cs.Variable(name="absCot", type="real", description=f"|reported cot{nm.capitalize()}|")],
            output=cs.Variable(name="sigma", type="real", description=f"cot{nm.capitalize()} resolution"),
            data=cs.Category(nodetype="category", input="surface", content=content)))
    cset = cs.CorrectionSet(schema_version=2, description=f"SmartPixels-output emulator angle resolution (sigma) tables, v{version} "
                                                          f"(inclusive; key = surface{'' if key == 'none' else ' x cluster size'} x |cot|; "
                                                          f"variant '{key}')", corrections=corrs)
    open(out, "w").write(cset.model_dump_json(exclude_unset=True, indent=1))
    pd.DataFrame(rows).to_csv(os.path.join(HERE, f"results/emu_sigma_table_{key}.csv"), index=False)
    print("wrote", out)


def evaluate(cor, P, ang):
    """sigma for every row, the correction's inputs taken by name"""
    args = [P[i.name].values.astype(np.int64) if i.name in ("surface", "sizeX", "sizeY")
            else np.abs(P[f"cot{ang}"].values.astype(np.float64)) for i in cor.inputs]
    return cor.evaluate(*args)


def pulls(path, tag=""):
    import correctionlib
    cset = correctionlib.CorrectionSet.from_file(path)
    P = routes(2); M = class_meta(2)
    isT = (P.route == "T").values
    rows = []
    for ang, nm in (("A", "alpha"), ("B", "beta")):
        own = KEYS["own"][ang][0]
        pull = (P[f"cot{ang}"].values.astype(np.float64) - P[f"tc{ang}"].values) / evaluate(cset[f"spix_emu_{nm}_sigma"], P, ang)
        good = (np.abs(P[f"cot{ang}"].values) <= CAP) & ~P[f"sat{ang}"].values
        # inclusive weight: each (surface, own size) cell carries its test-event population, split T : C by its class fraction
        cell = M.groupby(["surface", np.minimum(M[own], 30)])["T"].agg(["mean", "size"])
        c = cell.reindex(pd.MultiIndex.from_arrays([P.surface.values, np.minimum(P[own].values, 30)]))
        cnt = pd.Series(1.0, index=P.index).groupby([P.surface.values, np.minimum(P[own].values, 30), P.route.values]).transform("sum").values
        w = np.where(isT, c["mean"].values, 1 - c["mean"].values) * c["size"].values / cnt
        for s in sorted(P.surface.unique()):
            for rt, m in (("T", isT), ("C", ~isT), ("T+C inclusive", np.ones(len(P), bool))):
                for sel_nm, sm in (("all", np.ones(len(P), bool)), ("|cot|<=12, not saturated", good)):
                    k = (P.surface.values == s) & m & sm
                    q16, q84 = wquant(pull[k], w[k], [0.16, 0.84]) if rt.startswith("T+C") else np.quantile(pull[k], [0.16, 0.84])
                    wk = w[k] if rt.startswith("T+C") else np.ones(k.sum())
                    rows.append(dict(table=tag, angle=nm, surface=int(s), route=rt, selection=sel_nm, n=int(k.sum()),
                                     q_width=(q84 - q16) / 2, median=float(np.median(pull[k])),
                                     frac_abs_gt3=float(np.average(np.abs(pull[k]) > 3, weights=wk))))
            for rt, m in (("T", isT), ("C", ~isT)):
                for sz in range(1, 9):
                    k = (P.surface.values == s) & m & good & (np.minimum(P[own].values, 8) == sz)
                    if k.sum() < 100:
                        continue
                    q16, q84 = np.quantile(pull[k], [0.16, 0.84])
                    rows.append(dict(table=tag, angle=nm, surface=int(s), route=rt, selection=f"{own} = {sz}{'+' if sz == 8 else ''}",
                                     n=int(k.sum()), q_width=(q84 - q16) / 2, median=float(np.median(pull[k])),
                                     frac_abs_gt3=float(np.mean(np.abs(pull[k]) > 3))))
    R = pd.DataFrame(rows); R.to_csv(os.path.join(HERE, f"results/emu_sigma_pulls{'_' + tag if tag else ''}.csv"), index=False)
    pd.set_option("display.width", 200)
    print(R[R.selection.isin(["all", "|cot|<=12, not saturated"])].round(4).to_string())


if __name__ == "__main__":
    if sys.argv[1] == "routes":
        routes(0); routes(2)
    elif sys.argv[1] == "build":
        build(sys.argv[2], *(sys.argv[3:4]))
    else:
        pulls(sys.argv[2], *(sys.argv[3:4]))
