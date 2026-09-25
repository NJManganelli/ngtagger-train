"""Evaluate the disc CNNs (TFPX/TEPX) on TEST events of the barrel+disc dataset (data_all/).

Models: disc (separate disc model), unified (barrel + disc in one model), base (the barrel
model, applied to the barrel rows only, as the reference the unified model must match), and a
summary-feature HGB. Truth = producer convention (tpLocalCot*, helix at the hit). Main set =
"consistent": TP-linked, charged, pT >= 0.25, helix within 1 cm of the hit.

Writes results/disc_*.csv|json and figs/disc_*.png; predictions for every test row go to
WDMac cluster_cnn_preds/predictions_test_all.parquet (row = image index in data_all/).
"""
import os, json, numpy as np, pandas as pd, torch
from models_def import C_LAYER, BASE_COLS, scalars, make
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results"); FG = os.path.join(HERE, "figs")
D = os.path.join(HERE, "data_all")
PRED = "/Volumes/WDMac/smartpixels_scratch/cluster_cnn_preds"
torch.set_num_threads(4)
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # keep batches <= 8192 on MPS
NEED = sorted(set(BASE_COLS + ["row", "subdet", "disk", "ring", "side", "split", "linkClass", "tpPt", "tpCharge",
                               "tpLocalCotAlpha", "tpLocalCotBeta", "hxOk", "hxDist", "tpChargeFrac", "event",
                               "clIdx", "shN", "shCotAlpha", "shCotBeta", "domSimHasTp", "globalZ", "globalR"]))
mi = pd.read_parquet(os.path.join(D, "meta.parquet"), columns=NEED, filters=[("split", "==", 2)]).reset_index(drop=True)
imgs = np.load(os.path.join(D, "images.npy"), mmap_mode="r")
deg = lambda c: np.degrees(np.arctan(c))
smad = lambda x: float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) > 5 else np.nan
DISC = (mi.subdet == 2).values; BAR = ~DISC
print("test rows", len(mi), "disc", DISC.sum(), flush=True)

P = pd.DataFrame(index=mi.index)


def run_cnn(tag, sel):
    nd = json.load(open(os.path.join(HERE, "models", tag, "norm.json")))
    idx = np.where(sel)[0]
    S = ((scalars(mi.iloc[idx], nd["variant"]) - pd.Series(nd["mu"])) / pd.Series(nd["sd"])).values.astype(np.float32)
    net = make(nd["variant"], S.shape[1]); net.load_state_dict(torch.load(os.path.join(HERE, "models", tag, "model.pt")))
    net.to(dev).eval()
    rows = mi.row.values[idx]
    outs = []
    with torch.no_grad():
        for i in range(0, len(idx), 8192):
            X = torch.from_numpy(np.ascontiguousarray(imgs[rows[i:i + 8192]])).to(dev)
            outs.append(net(X, torch.from_numpy(S[i:i + 8192]).to(dev)).cpu().numpy())
    o = np.concatenate(outs)
    c = C_LAYER[mi.layer.values[idx]]
    for k, v in [("cotA", np.where(o[:, 2] > 0, 1, -1) * np.tan(o[:, 0]) - c), ("cotB", np.where(o[:, 3] > 0, 1, -1) * np.tan(o[:, 1])),
                 ("pA", 1 / (1 + np.exp(-o[:, 2]))), ("pB", 1 / (1 + np.exp(-o[:, 3])))]:
        col = np.full(len(mi), np.nan, np.float32); col[idx] = v; P[f"{tag}_{k}"] = col
    print("inferred", tag, len(idx), flush=True)


run_cnn("disc", DISC)
run_cnn("unified", np.ones(len(mi), bool))
run_cnn("base", BAR)

good = lambda m: ((m.linkClass == 2) & (m.tpPt >= 0.25) & (m.tpCharge != 0) & (m.tpLocalCotAlpha > -900)
                  & (m.hxOk == 1) & (m.hxDist < 1.0))
# HGB on summary features, disc train events
FEAT = ["sizeX", "sizeY", "size", "charge", "disk"]
trn = pd.read_parquet(os.path.join(D, "meta.parquet"), columns=FEAT + ["linkClass", "tpPt", "tpCharge", "tpLocalCotAlpha",
                      "tpLocalCotBeta", "hxOk", "hxDist"], filters=[("split", "==", 0), ("subdet", "==", 2), ("linkClass", "==", 2)])
trn = trn[good(trn)].sample(1_000_000, random_state=1)
for ang, col in [("A", "tpLocalCotAlpha"), ("B", "tpLocalCotBeta")]:
    h = HistGradientBoostingRegressor(max_iter=300, max_leaf_nodes=63, random_state=1).fit(trn[FEAT], deg(trn[col]))
    v = np.full(len(mi), np.nan, np.float32); v[DISC] = np.tan(np.radians(h.predict(mi.loc[DISC, FEAT]))); P[f"hgb_cot{ang}"] = v
del trn
print("hgb done", flush=True)
os.makedirs(PRED, exist_ok=True)
pd.concat([mi[["row", "event", "clIdx", "subdet", "layer", "disk", "linkClass"]], P.astype(np.float32)], axis=1).to_parquet(
    os.path.join(PRED, "predictions_test_all.parquet"))


def metrics(pc, tc):
    ok = np.isfinite(pc); pc, tc = pc[ok], tc[ok]
    dd = deg(pc) - deg(tc); dm = np.abs(deg(pc)) - np.abs(deg(tc))
    return dict(n=int(len(pc)), smad_deg=smad(dd), bias_deg=float(np.median(dd)) if len(dd) else np.nan,
                smad_cot=smad(pc - tc), mag_smad_deg=smad(dm),
                sign_acc=float(np.mean(np.sign(pc) == np.sign(tc))) if len(pc) else np.nan)


BINS_B = [0, 0.5, 1.07, 2, 4, np.inf]; LAB_B = ["0-0.5", "0.5-1.07", "1.07-2", "2-4", ">4"]
BINS_D = [0, 0.05, 0.1, 0.25, 0.5, 1.07, np.inf]; LAB_D = ["0-0.05", "0.05-0.1", "0.1-0.25", "0.25-0.5", "0.5-1.07", ">1.07"]
PTB = [0.25, 0.5, 1, 2, np.inf]; PTLAB = ["0.25-0.5", "0.5-1", "1-2", ">2"]
CONS = good(mi).values
rows = []
for region, sel0, models in [("disc", DISC, ["disc", "unified", "hgb"]), ("barrel", BAR, ["base", "unified"])]:
    for ang, col in [("A", "tpLocalCotAlpha"), ("B", "tpLocalCotBeta")]:
        tc = mi[col].values
        groups = [("all", np.ones(len(mi), bool))]
        for bins, labs, nm in ([(BINS_B, LAB_B, "|cot|")] + ([(BINS_D, LAB_D, "|cot|d")] if region == "disc" else [])):
            groups += [(f"{nm} {labs[i]}", (np.abs(tc) >= bins[i]) & (np.abs(tc) < bins[i + 1])) for i in range(len(labs))]
        if region == "disc":
            groups += [(f"disk {d}", (mi.disk == d).values) for d in range(1, 13)]
            groups += [("TFPX", (mi.disk <= 8).values), ("TEPX", (mi.disk >= 9).values)]
            groups += [(f"ring {r}", (mi.ring == r).values) for r in range(1, 6)]
        else:
            groups += [(f"L{l}", (mi.layer == l).values) for l in range(1, 5)]
        groups += [(f"pT {PTLAB[i]}", ((mi.tpPt >= PTB[i]) & (mi.tpPt < PTB[i + 1])).values) for i in range(4)]
        for gname, g in groups:
            for mod in models:
                s = sel0 & CONS & g
                r = metrics(P[f"{mod}_cot{ang}"].values[s], tc[s])
                r.update(region=region, angle="cotAlpha" if ang == "A" else "cotBeta", group=gname, model=mod)
                rows.append(r)
RES = pd.DataFrame(rows); RES.to_csv(os.path.join(R, "disc_resolution.csv"), index=False)

# ---- label contamination on the discs (helix-vs-hit distance; PSimHit where it exists)
lab = []
L = DISC & (mi.linkClass == 2).values & (mi.tpCharge != 0).values & (mi.tpLocalCotAlpha > -900).values
far = (mi.hxOk != 1).values | (mi.hxDist > 1.0).values
for gname, g in [("all", np.ones(len(mi), bool))] + [(f"pT {a}-{b}", ((mi.tpPt >= a) & (mi.tpPt < b)).values)
                                                   for a, b in [(0.1, 0.25), (0.25, 0.5), (0.5, 1), (1, 2), (2, 1e9)]] + \
        [("TFPX", (mi.disk <= 8).values), ("TEPX", (mi.disk >= 9).values)]:
    s = L & g
    sh = s & (mi.shN > 0).values & (mi.shCotAlpha > -900).values & (mi.domSimHasTp == 1).values
    dA = deg(mi.tpLocalCotAlpha.values[sh]) - deg(mi.shCotAlpha.values[sh]); dB = deg(mi.tpLocalCotBeta.values[sh]) - deg(mi.shCotBeta.values[sh])
    lab.append(dict(group=gname, n=int(s.sum()), frac_helix_gt_1cm=float(far[s].mean()), frac_helix_gt_1mm=float(((mi.hxOk != 1).values | (mi.hxDist > 0.1).values)[s].mean()),
                    n_psimhit=int(sh.sum()), helix_vs_psimhit_smad_deg_A=smad(dA), helix_vs_psimhit_smad_deg_B=smad(dB),
                    frac_psimhit_disagree_gt_10deg=float(np.mean((np.abs(dA) > 10) | (np.abs(dB) > 10))) if sh.sum() else np.nan))
pd.DataFrame(lab).to_csv(os.path.join(R, "disc_label_contamination.csv"), index=False)

# ---- leak / tell test on the discs: unlinked (no TP) vs linked, by event halves
te = mi[DISC].copy(); Pt = P[DISC]
ev = np.sort(te.event.unique()); half = te.event.isin(ev[::2]).values
y = (te.linkClass == 1).values.astype(int)
base_feat = te[FEAT].values


def auc(X):
    clf = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=31, learning_rate=0.1, random_state=1)
    sub = np.where(half)[0]; sub = np.random.default_rng(1).choice(sub, min(len(sub), 1_500_000), replace=False)
    clf.fit(X[sub], y[sub]); p = clf.predict_proba(X[~half])[:, 1]
    out = {"all": float(roc_auc_score(y[~half], p))}
    lk = te.linkClass.values[~half]; pt = te.tpPt.values[~half]
    for nm, cs in [("vs linked pT>=2", (lk == 1) | (pt >= 2)), ("vs linked pT<0.3", (lk == 1) | ((lk == 2) & (pt < 0.3))),
                   ("TFPX", te.disk.values[~half] <= 8), ("TEPX", te.disk.values[~half] >= 9)]:
        out[nm] = float(roc_auc_score(y[~half][cs], p[cs]))
    return out


LK = {"non-angle (sizeX,sizeY,size,charge,disk)": auc(base_feat)}
for t in ["disc", "unified"]:
    LK[f"non-angle + CNN[{t}] angles"] = auc(np.c_[base_feat, Pt[f"{t}_cotA"], Pt[f"{t}_cotB"]])
LK["unlinked_frac"] = float(y.mean())
json.dump(LK, open(os.path.join(R, "disc_leak_test.json"), "w"), indent=1)
print(json.dumps(LK, indent=1))

# ---- CNN vs PSimHit on disc signal clusters (linked and unlinked-with-PSimHit)
shr = []
for labn, s in [("disc linked w/ PSimHit pT>=0.25", DISC & (mi.linkClass == 2).values & (mi.shN > 0).values & (mi.tpPt >= 0.25).values & (mi.shCotAlpha > -900).values),
                ("disc unlinked w/ PSimHit", DISC & (mi.linkClass == 1).values & (mi.shN > 0).values & (mi.shCotAlpha > -900).values)]:
    for mod in ["disc", "unified", "truth"]:
        for ang, nm in [("A", "Alpha"), ("B", "Beta")]:
            pc = mi[f"tpLocalCot{nm}"].values if mod == "truth" else P[f"{mod}_cot{ang}"].values
            r = metrics(pc[s], mi[f"shCot{nm}"].values[s]); r.update(sample=labn, model=mod, angle=ang); shr.append(r)
pd.DataFrame(shr).to_csv(os.path.join(R, "disc_psimhit_residuals.csv"), index=False)

# ---- predicted-angle distributions, linked vs unlinked (discs)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
cats = {"unlinked (no TP)": DISC & (mi.linkClass == 1).values, "linked pT<0.25": DISC & (mi.linkClass == 2).values & (mi.tpPt < 0.25).values,
        "linked pT>=2": DISC & (mi.linkClass == 2).values & (mi.tpPt >= 2).values}
for k, (cn, s) in enumerate(cats.items()):
    for j, ang in enumerate(["A", "B"]):
        ax[j].hist(deg(P[f"disc_cot{ang}"].values[s]), bins=np.linspace(-60, 60, 121), histtype="step", density=True, lw=2,
                   color=["#2a78d6", "#eb6834", "#1baf7a"][k], label=f"{cn} (n={s.sum()})")
for j, t in enumerate(["disc CNN predicted alpha [deg]", "disc CNN predicted beta [deg]"]):
    ax[j].set_xlabel(t); ax[j].set_ylabel("density"); ax[j].set_yscale("log"); ax[j].grid(alpha=0.25)
ax[0].legend(fontsize=8, frameon=False); fig.tight_layout(); fig.savefig(os.path.join(FG, "disc_pred_angle_distributions.png"), dpi=110); plt.close(fig)
print("done")
