"""Production A1 donor model of the CMSSW SmartPixels-output emulator (angleSource = "emu"), plus the overfit checks.

  emu_train_a1.py cache <region>     class-T clusters of every split (0 train, 1 val, 2 test) with the tiny-CNN outputs
                                     -> FIX/cache/<region>_s<split>.parquet (A1 raw inputs, true angles, keys)
  emu_train_a1.py overfit            tiny-CNN residual widths on class T: training events (in-sample) vs val vs test
                                     -> emu_cmssw/overfit_tiny.json
  emu_train_a1.py train <region> <splits>   A1-arh on the class-T clusters of the given splits (e.g. 0) -> emu_cmssw/a1_<region>.pt
  emu_train_a1.py check <region>     A1 in-sample (its training events) vs out-of-sample (test events): NLL, PIT, and the
                                     class-C route closure (C route applied to class-T test clusters, response included)
                                     -> emu_cmssw/a1_check_<region>.json

The events are those of cluster_cnn_data_all (RelVal file1 events 201-300 and file2 events 6601-6700: split 0 = 125
training, 1 = 25 validation, 2 = 50 test events); the tiny CNN was trained on split-0 class T (fast_train.py). A1 = the
fast-study model (fast_a1.ARH: 15 -> 128 -> 128 trunk, autoregressive 156-bin histogram p(rA | z) p(rB | z, rA)), trained
as fast_a1.fit (10% of the training events = early-stop monitor), on the residual r = true - CNN [deg].
"""
import os, sys, json, time, numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2_common import D_ALL, derived, core_fit
from emu_common import (load_cnn, cnn_scalars, run_cnn, decode, a1_inputs, load_a1, a1_draw, a1_uniforms, class_t,
                        load_library, emulate, REGIONS, N_A1_IN, CNN_DIR)

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, "emu_cmssw")
FIX = "/Volumes/WDMac/smartpixels_scratch/emu_fix"
LIB = os.path.join(FIX, "spix_emu_library_v2.root")
FLT = {"barrel": [("subdet", "==", 1)], "disc": [("subdet", "==", 2), ("disk", "<=", 4)]}
COLS = ["row", "run", "event", "detId", "idxInDet", "subdet", "layer", "disk", "side", "sizeX", "sizeY", "size", "charge",
        "adcSum", "truncX", "truncY", "nPxLost", "globalX", "globalY", "globalZ", "globalR", "globalPhi", "modX", "modY", "modZ",
        "localX", "localY", "zOutward", "axXx", "axXy", "axXz", "axYx", "axYy", "axYz", "axZx", "axZy", "axZz",
        "linkClass", "tpCharge", "tpLocalCotAlpha", "tpLocalCotBeta", "hxOk", "hxDist"]
CLS_T = [("linkClass", "==", 2), ("tpCharge", "!=", 0), ("hxOk", "==", 1), ("hxDist", "<", 1.0)]
deg = lambda c: np.degrees(np.arctan(np.asarray(c, np.float64)))


def cache(reg):
    os.makedirs(os.path.join(FIX, "cache"), exist_ok=True)
    net, nd = load_cnn(CNN_DIR.format(reg=reg))
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu"); net = net.to(dev)
    img = np.load(os.path.join(D_ALL, "images.npy"), mmap_mode="r")
    for sp in (0, 1, 2):
        m = pd.read_parquet(os.path.join(D_ALL, "meta.parquet"), columns=COLS, filters=FLT[reg] + CLS_T + [("split", "==", sp)])
        m = m[class_t(m.linkClass.values == 2, m.tpCharge, m.hxOk, m.hxDist) & (m.tpLocalCotAlpha.values > -900)].reset_index(drop=True)
        m = pd.concat([m, derived(m)], axis=1)
        S = cnn_scalars(m, reg, nd)
        mu = torch.tensor([nd["mu"][c] for c in nd["cols"]], dtype=torch.float32, device=dev)
        sd = torch.tensor([nd["sd"][c] for c in nd["cols"]], dtype=torch.float32, device=dev)
        out = []
        rows = m.row.values
        with torch.no_grad():
            for i in range(0, len(m), 8192):                      # MPS: batches <= 8192
                x = torch.as_tensor(np.asarray(img[rows[i:i + 8192]])).to(dev)
                out.append(net(x, (torch.as_tensor(S[i:i + 8192]).to(dev) - mu) / sd).cpu().numpy())
        a, b, sa, sb = decode(np.concatenate(out))
        lod = m.layer.values if reg == "barrel" else m.disk.values
        Z = a1_inputs(a, b, sa, sb, m.sizeX.values, m.sizeY.values, m.charge.values, m.globalR.values, np.abs(m.globalZ.values),
                      m.localX.values, m.localY.values, lod)
        C = pd.DataFrame(Z, columns=[f"z{j}" for j in range(N_A1_IN)])
        for c in ("row", "run", "event", "detId", "idxInDet"):
            C[c] = m[c].values
        C["surface"] = (REGIONS[reg] + lod).astype(np.int64)
        C["tA"], C["tB"] = deg(m.tpLocalCotAlpha.values), deg(m.tpLocalCotBeta.values)
        C["tcA"], C["tcB"] = m.tpLocalCotAlpha.values, m.tpLocalCotBeta.values
        C.to_parquet(os.path.join(FIX, "cache", f"{reg}_s{sp}.parquet"))
        print(reg, "split", sp, "class T", len(C), "events", C.event.nunique(), flush=True)


def load_cache(reg, splits):
    return pd.concat([pd.read_parquet(os.path.join(FIX, "cache", f"{reg}_s{s}.parquet")) for s in splits], ignore_index=True)


def widths(r):
    r = r[np.isfinite(r)]
    return dict(core=core_fit(r)[0], full=float(np.sqrt(np.mean(r ** 2))), gt10=float(np.mean(np.abs(r) > 10)), n=int(len(r)))


def overfit():
    out = {}
    for reg in REGIONS:
        for sp, tag in ((0, "train (in-sample)"), (1, "val"), (2, "test")):
            C = load_cache(reg, [sp])
            rA, rB = C.tA.values - C.z0.values, C.tB.values - C.z1.values
            for g, gs in [("all", np.ones(len(C), bool))] + [(f"s{k}", C.surface.values == k) for k in sorted(C.surface.unique())]:
                out[f"{reg} {tag} {g}"] = {"alpha": widths(rA[gs]), "beta": widths(rB[gs])}
            print(reg, tag, {k: round(v["full"], 3) for k, v in out[f"{reg} {tag} all"].items()},
                  {k: round(v["core"], 3) for k, v in out[f"{reg} {tag} all"].items()}, flush=True)
    json.dump(out, open(os.path.join(OUTD, "overfit_tiny.json"), "w"), indent=1)


def train(reg, splits):
    from fast_a1 import fit
    C = load_cache(reg, splits)
    Z = C[[f"z{j}" for j in range(N_A1_IN)]].values.astype(np.float64)
    R = np.c_[C.tA.values - C.z0.values, C.tB.values - C.z1.values]
    ok = np.isfinite(Z).all(1) & np.isfinite(R).all(1)
    mu, sd = Z[ok].mean(0), Z[ok].std(0); sd[sd == 0] = 1
    t0 = time.time()
    torch.manual_seed(3)
    net, log = fit("arh", (Z[ok] - mu) / sd, R[ok], C.event.values[ok], seed=7)
    torch.save({"state": net.state_dict(), "mu": mu.astype(np.float32).tolist(), "sd": sd.astype(np.float32).tolist(),
                "splits": list(splits), "events": sorted(int(e) for e in np.unique(C.event.values)), "n": int(ok.sum()),
                "monitor_loss": log}, os.path.join(OUTD, f"a1_{reg}.pt"))
    print(reg, "A1 trained on", int(ok.sum()), "class-T clusters of", C.event.nunique(), "events; monitor loss",
          [round(x, 4) for x in log], round((time.time() - t0) / 60, 1), "min", flush=True)


def check(reg):
    """in-sample vs out-of-sample: CE (NLL) of the true residual pair, PIT calibration, and the C-route closure"""
    from fast_a1 import ARH
    from fast_models import EDGES, pit_from_logits
    from fast_eval import pit_stats
    a1 = load_a1(os.path.join(OUTD, f"a1_{reg}.pt"))
    ck = torch.load(os.path.join(OUTD, f"a1_{reg}.pt"), map_location="cpu")
    lib = load_library(LIB); E = torch.tensor(EDGES, dtype=torch.float32)
    res = {}
    for tag, splits in (("in-sample (A1 training events)", ck["splits"]), ("test events (out-of-sample)", [2])):
        C = load_cache(reg, splits)
        if len(C) > 600_000:
            C = C.iloc[np.sort(np.random.default_rng(5).choice(len(C), 600_000, replace=False))].reset_index(drop=True)
        Z = C[[f"z{j}" for j in range(N_A1_IN)]].values.astype(np.float32)
        R = np.c_[C.tA.values - C.z0.values, C.tB.values - C.z1.values]
        net, mu, sd = a1
        with torch.no_grad():
            h = net.t((torch.from_numpy(Z) - torch.from_numpy(mu)) / torch.from_numpy(sd))
            lA = net.logitsA(h); lB = net.logitsB(h, torch.tensor(R[:, 0], dtype=torch.float32))
            kA = torch.bucketize(torch.tensor(R[:, 0], dtype=torch.float32), E[1:-1], right=True)
            kB = torch.bucketize(torch.tensor(R[:, 1], dtype=torch.float32), E[1:-1], right=True)
            nll = float(torch.nn.functional.cross_entropy(lA, kA) + torch.nn.functional.cross_entropy(lB, kB))
            pA = pit_from_logits(lA.double().numpy(), R[:, 0]); pB = pit_from_logits(lB.double().numpy(), R[:, 1])
        U = a1_uniforms(C.run.values, C.event.values, C.detId.values, C.idxInDet.values)
        d = a1_draw(a1, Z, U)
        e = emulate(lib, C.surface.values, np.zeros(len(C), bool), C.tcA.values, C.tcB.values, C.z0.values, C.z1.values,
                    d[:, 0], d[:, 1], C.run.values, C.event.values, C.detId.values, C.idxInDet.values)
        grp = np.where(C.surface.values > 10, "ID1-4", np.where(C.surface.values == 1, "IL1", "IL2-4"))
        r = {"nll": nll, "n": int(len(C)), "pitA": pit_stats(pA), "pitB": pit_stats(pB)}
        for g in np.unique(grp):
            s = grp == g
            r[g] = {"closure_alpha": widths(e["outA"][s] - C.tA.values[s]), "closure_beta": widths(e["outB"][s] - C.tB.values[s]),
                    "draw_vs_true_q16_84_alpha": [float(np.quantile(d[s, 0], q) - np.quantile(R[s, 0], q)) for q in (0.16, 0.84)],
                    "draw_vs_true_q16_84_beta": [float(np.quantile(d[s, 1], q) - np.quantile(R[s, 1], q)) for q in (0.16, 0.84)]}
        res[tag] = r
        print(reg, tag, json.dumps({g: {k: round(v["full"], 3) for k, v in r[g].items() if k.startswith("closure")}
                                    for g in np.unique(grp)}), "nll", round(nll, 4), "pit", r["pitA"], r["pitB"], flush=True)
    json.dump(res, open(os.path.join(OUTD, f"a1_check_{reg}.json"), "w"), indent=1)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "cache":
        cache(sys.argv[2])
    elif cmd == "overfit":
        overfit()
    elif cmd == "train":
        train(sys.argv[2], [int(s) for s in sys.argv[3].split(",")])
    elif cmd == "check":
        check(sys.argv[2])
