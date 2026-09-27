"""Build the data library of the CMSSW SmartPixels-output emulator (angleSource = "emu").

  emu_build_library.py overfit            CNN residual widths, in-sample (training events) vs test events -> emu_cmssw/overfit.json
  emu_build_library.py build OUT.root     the library file

OUT.root (uproot, read by SmartPixelsRecHitProducer through ROOT):
  dump   one row per Conv1D_Full-2bit dump row (RESCALED to physical units, CMS frame = pure swap):
         tA, tB = atan(true cot) [deg] (the 32-NN lookup key), rA, rB = atan(pred) - atan(true) [deg]  (float32)
  donors one row per class-T cluster of the donor events: surface (1-4 IL, 11-14 ID), the 11 observables DONOR_OBS
         (cnnA, cnnB, sigA, sigB [deg], sizeX, sizeY, logq, r, absz, localX, localY [cm]) and resA, resB = true - CNN [deg]
         (float32), sorted by surface (library index = row number within its surface)
  norm   per surface: mu_<obs>, sd_<obs> (float64), the standardisation of the 11 observables (donor mean / std)
  meta   one row: description string
DONOR EVENTS: 25 events of the CNN TRAINING split (split 0 of cluster_cnn_data_all = RelVal file1), every 5th of the 125
sorted training events -- the density of the prototype's library (the other half of the 50 test events), so the class-C
conditioning (and its leak-test result) carries over; not production events (the donors are class T, which never draws).
"""
import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2_common import V2, D_ALL, load, good, deg, core_fit
from emu_common import load_net, normalise, run_net, decode, scalars_v2, DONOR_OBS

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, "emu_cmssw")
N_DONOR_EVENTS_STRIDE = 5
REG = {"barrel": ([("subdet", "==", 1)], "layer", 0), "disc": ([("subdet", "==", 2), ("disk", "<=", 4)], "disk", 10)}


SPDUMP = os.path.expanduser("~/smartpixels/ngtagger-train/2t-Conv1D_Full-2bit_optimized-vars.parquet")
# the dump stores NORMALISED labels; physical = scale x stored (rescale() in smart-pixels-ml/summary-plots/read-all-models-2s.ipynb)
SF = {"cotA": 6.528905408493022, "cotB": 1.8827422609846889}


def load_dump():
    """CMS frame = pure swap: CMS alpha = dump cotB, CMS beta = dump cotA. r = atan(pred) - atan(true) [deg]."""
    d = pd.read_parquet(SPDUMP)
    tA, pA = SF["cotB"] * d.cotBtrue.values, SF["cotB"] * d.cotB.values
    tB, pB = SF["cotA"] * d.cotAtrue.values, SF["cotA"] * d.cotA.values
    return pd.DataFrame({"tA": deg(tA), "tB": deg(tB), "rA": deg(pA) - deg(tA), "rB": deg(pB) - deg(tB)})


def surf_code(m, reg):
    return (m.layer.values if reg == "barrel" else 10 + m.disk.values).astype(np.int64)


def cnn_on(m, reg, rows=None):
    net, nd = load_net(os.path.join(V2, "models", f"{reg}_pos_hubsig"))
    img = np.load(os.path.join(D_ALL, "images.npy"), mmap_mode="r")
    r = m.row.values if rows is None else rows
    S = scalars_v2(m, reg, nd["cols"])
    outs = []
    for i in range(0, len(m), 200_000):
        outs.append(run_net(net, np.asarray(img[r[i:i + 200_000]]), normalise(S[i:i + 200_000], nd)))
    return decode(np.concatenate(outs))


def widths(res):
    res = res[np.isfinite(res)]
    return dict(core=core_fit(res)[0], full=float(np.sqrt(np.mean(res ** 2))), gt10=float(np.mean(np.abs(res) > 10)), n=int(len(res)))


def overfit():
    """pos-hubsig residuals on class-T clusters (good()) of training events (in-sample) vs test events, 200k each"""
    out = {}
    A = os.path.join(V2, "arrays")
    for reg in REG:
        net, nd = load_net(os.path.join(V2, "models", f"{reg}_pos_hubsig"))
        for sp, tag in (("train", "train (in-sample)"), ("test", "test")):
            m = pd.read_parquet(os.path.join(A, f"{reg}_{sp}_meta.parquet"))
            keep = good(m)
            sel = np.sort(np.random.default_rng(3).choice(np.where(keep)[0], min(keep.sum(), 200_000), replace=False))
            m = m.iloc[sel].reset_index(drop=True)
            img = np.asarray(np.load(os.path.join(A, f"{reg}_{sp}_img.npy"), mmap_mode="r")[sel])
            a, b, _, _ = decode(run_net(net, img, normalise(scalars_v2(m, reg, nd["cols"]), nd)))
            tA, tB = deg(m.tpLocalCotAlpha.values), deg(m.tpLocalCotBeta.values)
            s = surf_code(m, reg)
            for g, gs in ([("all", np.ones(len(s), bool))] + [(f"s{k}", s == k) for k in np.unique(s)]):
                out[f"{reg} {tag} {g}"] = {"alpha": widths((a - tA)[gs]), "beta": widths((b - tB)[gs])}
                print(reg, tag, g, "full", {k: round(v["full"], 3) for k, v in out[f"{reg} {tag} {g}"].items()},
                      "core", {k: round(v["core"], 3) for k, v in out[f"{reg} {tag} {g}"].items()}, flush=True)
    json.dump(out, open(os.path.join(OUTD, "overfit.json"), "w"), indent=1)


def build(path):
    import uproot
    D = load_dump()
    dump = {k: D[c].values.astype(np.float32) for k, c in (("tA", "tA"), ("tB", "tB"), ("rA", "rA"), ("rB", "rB"))}
    ev_all = np.unique(pd.read_parquet(os.path.join(D_ALL, "meta.parquet"), columns=["event", "split"],
                                       filters=[("split", "==", 0)]).event.values)
    donor_ev = ev_all[::N_DONOR_EVENTS_STRIDE]
    print("training events", len(ev_all), "donor events", len(donor_ev), flush=True)
    parts = []
    for reg, (flt, L, off) in REG.items():
        m = load(flt + [("split", "==", 0)])
        m = m[good(m) & m.event.isin(donor_ev).values].reset_index(drop=True)
        a, b, sa, sb = cnn_on(m, reg)
        tA, tB = deg(m.tpLocalCotAlpha.values), deg(m.tpLocalCotBeta.values)
        p = pd.DataFrame({"surface": surf_code(m, reg).astype(np.int32), "cnnA": a, "cnnB": b, "sigA": sa, "sigB": sb,
                          "sizeX": m.sizeX.values, "sizeY": m.sizeY.values, "logq": np.log(m.charge.clip(lower=1)).values,
                          "r": m.globalR.values, "absz": np.abs(m.globalZ.values), "localX": m.localX.values,
                          "localY": m.localY.values, "resA": tA - a, "resB": tB - b,
                          "event": m.event.values, "detId": m.detId.values, "row": m.row.values})
        parts.append(p)
        print(reg, "donors", len(p), p.surface.value_counts().sort_index().to_dict(), flush=True)
    P = pd.concat(parts, ignore_index=True).sort_values(["surface", "row"], kind="stable").reset_index(drop=True)
    for c in DONOR_OBS + ["resA", "resB"]:
        P[c] = P[c].astype(np.float32)
    norm = {"surface": []}
    for s, g in P.groupby("surface"):
        norm["surface"].append(s)
        Z = g[DONOR_OBS].values.astype(np.float64)
        mu, sd = Z.mean(0), Z.std(0); sd[sd == 0] = 1
        for j, c in enumerate(DONOR_OBS):
            norm.setdefault(f"mu_{c}", []).append(mu[j]); norm.setdefault(f"sd_{c}", []).append(sd[j])
    trees = {"dump": dump,
             "donors": {"surface": P.surface.values.astype(np.int32), **{c: P[c].values for c in DONOR_OBS + ["resA", "resB"]}},
             "norm": {k: np.asarray(v, np.int32 if k == "surface" else np.float64) for k, v in norm.items()}}
    with uproot.recreate(path, compression=uproot.LZMA(9)) as f:
        for name, cols in trees.items():                     # TTrees (a dict assignment would write an RNTuple)
            f.mktree(name, {k: v.dtype for k, v in cols.items()}).extend(cols)
    P[["surface", "event", "detId", "row"]].to_parquet(os.path.join(V2, "emu_cmssw", "donor_provenance.parquet"))
    json.dump({"donor_events": [int(e) for e in donor_ev], "n_dump": int(len(D)), "n_donors": P.surface.value_counts().sort_index().to_dict()},
              open(os.path.join(OUTD, "library_info.json"), "w"), indent=1, default=int)
    print("wrote", path, os.path.getsize(path), "bytes", flush=True)


if __name__ == "__main__":
    os.makedirs(OUTD, exist_ok=True)
    overfit() if sys.argv[1] == "overfit" else build(sys.argv[2])
