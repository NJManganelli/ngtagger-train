"""Static sigAlpha/sigBeta table of the CMSSW SmartPixels-output emulator (angleSource = "emu").

  emu_sigma_table.py LIBRARY.root OUT.json

sigma = (q84 - q16) / 2 of (emitted - true) COT -- the payload's robust core width -- of the EMULATED response on the
class-T clusters (good(): charged TP, pT >= 0.25, helix within 1 cm) of the TRAINING events (split 0), keyed ONLY on
what the sensor reports: the surface code (1-4 = IL1-4, 11-14 = ID1-4) and the |emitted cot| of that angle. Bins with
fewer than MIN_N clusters take the nearest populated bin of the same surface. Also reported per surface (not used by
the producer): the core RMS (emu_common/v2_common core_fit: mode-seeded Gaussian core, about zero) and the full RMS,
in degrees and in cot -> results/emu_sigma_table.csv, results/emu_sigma_widths.csv.
"""
import sys, os, numpy as np, pandas as pd
import correctionlib.schemav2 as cs
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2_common import D_ALL, good, core_fit
from emu_common import load_library, emulate

HERE = os.path.dirname(os.path.abspath(__file__))
EDGES = [0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.07, 1.5, 2, 3, 4, 6, 1e6]
MIN_N = 200
COLS = ["run", "event", "detId", "idxInDet", "subdet", "layer", "disk", "linkClass", "tpPt", "tpCharge",
        "tpLocalCotAlpha", "tpLocalCotBeta", "hxOk", "hxDist"]


def main(libpath, out):
    lib = load_library(libpath)
    parts = []
    for flt in ([("subdet", "==", 1), ("split", "==", 0)], [("subdet", "==", 2), ("disk", "<=", 4), ("split", "==", 0)]):
        m = pd.read_parquet(os.path.join(D_ALL, "meta.parquet"), columns=COLS, filters=flt)
        m = m[good(m)].reset_index(drop=True)
        surf = np.where(m.subdet.values == 1, m.layer.values, 10 + m.disk.values).astype(np.int64)
        for i in range(0, len(m), 500_000):
            c = m.iloc[i:i + 500_000]; s = surf[i:i + 500_000]; n = len(c)
            e = emulate(lib, s, np.ones(n, bool), c.tpLocalCotAlpha.values, c.tpLocalCotBeta.values, None, None, None,
                        c.run.values, c.event.values, c.detId.values, c.idxInDet.values)
            parts.append(pd.DataFrame({"surface": s, "cotA": e["cotA"], "cotB": e["cotB"], "outA": e["outA"], "outB": e["outB"],
                                       "tcA": c.tpLocalCotAlpha.values, "tcB": c.tpLocalCotBeta.values}))
            print("emulated", i + n, "/", len(m), flush=True)
    P = pd.concat(parts, ignore_index=True)
    rows, wrows, corrs = [], [], []
    for ang, nm in (("A", "alpha"), ("B", "beta")):
        content = []
        for key in sorted(P.surface.unique()):
            d = P[P.surface == key]
            pc = d[f"cot{ang}"].values.astype(np.float64); tc = d[f"tc{ang}"].values.astype(np.float64)
            rdeg = d[f"out{ang}"].values - np.degrees(np.arctan(tc))
            wrows.append(dict(angle=nm, surface=int(key), n=len(d), core_deg=core_fit(rdeg)[0], full_deg=float(np.sqrt(np.mean(rdeg ** 2))),
                              q_width_deg=float(np.diff(np.quantile(rdeg, [0.16, 0.84]))[0] / 2),
                              core_cot=core_fit(pc - tc)[0], full_cot=float(np.sqrt(np.mean((pc - tc) ** 2))),
                              q_width_cot=float(np.diff(np.quantile(pc - tc, [0.16, 0.84]))[0] / 2)))
            b = np.digitize(np.abs(pc), EDGES[1:-1])
            sig, cnt = np.full(len(EDGES) - 1, np.nan), np.zeros(len(EDGES) - 1, int)
            for i in range(len(EDGES) - 1):
                r = (pc - tc)[b == i]; cnt[i] = len(r)
                if len(r) >= MIN_N:
                    q16, q84 = np.quantile(r, [0.16, 0.84]); sig[i] = (q84 - q16) / 2
            pop = np.where(np.isfinite(sig))[0]
            filled = np.array([sig[pop[np.argmin(np.abs(pop - i))]] for i in range(len(sig))])
            for i in range(len(sig)):
                rows.append(dict(angle=nm, surface=int(key), lo=EDGES[i], hi=EDGES[i + 1], n=int(cnt[i]), sigma_measured=sig[i], sigma=filled[i]))
            content.append(cs.CategoryItem(key=int(key), value=cs.Binning(nodetype="binning", input="absCot", edges=EDGES,
                                                                           content=[round(float(v), 6) for v in filled], flow="clamp")))
        corrs.append(cs.Correction(
            name=f"spix_emu_{nm}_sigma", version=1,
            description=(f"SmartPixels-output emulator cot{nm.capitalize()} resolution: (q84-q16)/2 of (emitted - true) cot of the "
                         f"emulated response on class-T clusters of the training events, by surface code (1-4 IL, 11-14 ID) and "
                         f"|reported cot|. Static resolution model; inputs are sensor-reported only."),
            inputs=[cs.Variable(name="surface", type="int", description="IT surface code: 1-4 = IL1-IL4, 11-14 = ID1-ID4"),
                    cs.Variable(name="absCot", type="real", description=f"|reported cot{nm.capitalize()}|")],
            output=cs.Variable(name="sigma", type="real", description=f"cot{nm.capitalize()} resolution"),
            data=cs.Category(nodetype="category", input="surface", content=content)))
    cset = cs.CorrectionSet(schema_version=2, description="SmartPixels-output emulator angle resolution (sigma) tables", corrections=corrs)
    open(out, "w").write(cset.model_dump_json(exclude_unset=True, indent=1))
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "results/emu_sigma_table.csv"), index=False)
    W = pd.DataFrame(wrows); W.to_csv(os.path.join(HERE, "results/emu_sigma_widths.csv"), index=False)
    print(W.round(4).to_string()); print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
