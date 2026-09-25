"""Evaluate cluster CNNs vs the payload and a summary-feature HGB, on TEST events only.

Writes results/*.json|csv and figs/*.png. Truth = producer convention (tpLocalCot*,
helix-propagated TP direction at the hit). Main evaluation set = "consistent":
TP-linked, charged, pT >= 0.25, helix within 1 cm of the hit (see validate_truth.py);
also reported on "alllabels" (no helix-consistency cut) and "clean" (+ tpChargeFrac >= 0.9).
"""
import os, json, numpy as np, pandas as pd, torch
from models_def import C_LAYER, BASE_COLS, VAR_COLS, scalars, make
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results"); FG = os.path.join(HERE, "figs")
os.makedirs(R, exist_ok=True); os.makedirs(FG, exist_ok=True)
torch.set_num_threads(4)
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # NB: MPS Conv2d is WRONG at batch 65536 (verified vs CPU); keep batches <= 8192
NEED = sorted(set(BASE_COLS + sum(VAR_COLS.values(), []) + [
    "split", "linkClass", "shN", "pHasAlpha", "pHasBeta", "pCotAlpha", "pCotBeta", "tpPt", "tpCharge",
    "tpLocalCotAlpha", "tpLocalCotBeta", "hxOk", "hxDist", "tpChargeFrac", "event", "clIdx", "detId",
    "tx_hx", "ty_hx", "tx_sh", "ty_sh", "tx_cpe", "ty_cpe", "pitchX", "pitchY", "domSimHasTp",
    "shCotAlpha", "shCotBeta", "shPdg", "shPabs", "domSimPt"]))
meta = pd.read_parquet(os.path.join(HERE, "data/meta.parquet"), columns=NEED)
imgs = np.load(os.path.join(HERE, "data/images.npy"), mmap_mode="r")
WX, WY = imgs.shape[1:]
BINS = [0, 0.5, 1.07, 2, 4, np.inf]; BLAB = ["0-0.5", "0.5-1.07", "1.07-2", "2-4", ">4"]
PTB = [0.25, 0.5, 1, 2, np.inf]; PTLAB = ["0.25-0.5", "0.5-1", "1-2", ">2"]
deg = lambda c: np.degrees(np.arctan(c))
smad = lambda x: float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) > 5 else np.nan

# ---------------- CNNs (definitions in models_def.py) ----------------
INF = np.where((meta.split == 2) | ((meta.linkClass == 1) & (meta.shN > 0)))[0]  # test events + all unlinked-with-PSimHit
mi = meta.iloc[INF].reset_index(drop=True)
P = pd.DataFrame(index=mi.index)

def run_cnn(tag):
    nd = json.load(open(os.path.join(HERE, "models", tag, "norm.json")))
    mu, sd = pd.Series(nd["mu"]), pd.Series(nd["sd"])
    S = ((scalars(mi, nd["variant"]) - mu) / sd).values.astype(np.float32)
    net = make(nd["variant"], S.shape[1]); net.load_state_dict(torch.load(os.path.join(HERE, "models", tag, "model.pt"))); net.to(dev).eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(INF), 8192):
            X = torch.from_numpy(np.ascontiguousarray(imgs[INF[i:i + 8192]])).to(dev)
            outs.append(net(X, torch.from_numpy(S[i:i + 8192]).to(dev)).cpu().numpy())
    o = np.concatenate(outs)
    c = C_LAYER[mi.layer.values]
    P[f"{tag}_cotA"] = np.where(o[:, 2] > 0, 1, -1) * np.tan(o[:, 0]) - c
    P[f"{tag}_cotB"] = np.where(o[:, 3] > 0, 1, -1) * np.tan(o[:, 1])
    P[f"{tag}_pA"] = 1 / (1 + np.exp(-o[:, 2])); P[f"{tag}_pB"] = 1 / (1 + np.exp(-o[:, 3]))
    P[f"{tag}_tx"] = o[:, 4]; P[f"{tag}_ty"] = o[:, 5]

CNNS = [t for t in ["base", "bla", "pos", "base_clean", "small"] if os.path.exists(os.path.join(HERE, "models", t, "model.pt"))]
for t in CNNS:
    run_cnn(t); print("inferred", t, flush=True)

# ---------------- payload + HGB on summary features ----------------
P["payload_cotA"] = np.where(mi.pHasAlpha == 1, mi.pCotAlpha, np.nan)
P["payload_cotB"] = np.where(mi.pHasBeta == 1, mi.pCotBeta, np.nan)
FEAT = ["sizeX", "sizeY", "size", "charge", "layer"]
good = lambda m: ((m.linkClass == 2) & (m.tpPt >= 0.25) & (m.tpCharge != 0) & (m.tpLocalCotAlpha > -900)
                  & (m.hxOk == 1) & (m.hxDist < 1.0))
trn = meta[good(meta) & (meta.split == 0)].sample(1_000_000, random_state=1)
for ang, col in [("A", "tpLocalCotAlpha"), ("B", "tpLocalCotBeta")]:
    h = HistGradientBoostingRegressor(max_iter=300, max_leaf_nodes=63, random_state=1).fit(trn[FEAT], deg(trn[col]))
    P[f"hgb_cot{ang}"] = np.tan(np.radians(h.predict(mi[FEAT])))
    h = HistGradientBoostingRegressor(max_iter=300, max_leaf_nodes=63, random_state=1).fit(trn[FEAT], np.abs(deg(trn[col])))
    P[f"hgbmag_cot{ang}"] = np.tan(np.radians(h.predict(mi[FEAT])))  # magnitude-only (sign meaningless)
print("hgb done", flush=True)
PRED = "/Volumes/WDMac/smartpixels_scratch/cluster_cnn_preds"  # large dumps live on the external volume
os.makedirs(PRED, exist_ok=True)
P.astype(np.float32).to_parquet(os.path.join(PRED, "predictions_test.parquet"))
mi[["event", "clIdx", "layer", "linkClass", "tpPt"]].to_parquet(os.path.join(PRED, "predictions_test_keys.parquet"))

# ---------------- resolution tables ----------------
MODELS = ["payload", "hgb", "hgbmag"] + CNNS
def metrics(pc, tc):
    ok = np.isfinite(pc)
    pc, tc = pc[ok], tc[ok]
    dd = deg(pc) - deg(tc); dc = pc - tc; dm = np.abs(deg(pc)) - np.abs(deg(tc))
    return dict(n=int(len(pc)), smad_deg=smad(dd), rms_deg=float(np.sqrt(np.mean(dd ** 2))) if len(dd) else np.nan,
                bias_deg=float(np.median(dd)) if len(dd) else np.nan, smad_cot=smad(dc),
                rms_cot=float(np.sqrt(np.mean(dc ** 2))) if len(dc) else np.nan, bias_cot=float(np.median(dc)) if len(dc) else np.nan,
                sign_acc=float(np.mean(np.sign(pc) == np.sign(tc))) if len(pc) else np.nan, mag_smad_deg=smad(dm))

TEST = (mi.split == 2).values
SETS = {"consistent": TEST & good(mi).values,
        "alllabels": TEST & ((mi.linkClass == 2) & (mi.tpPt >= 0.25) & (mi.tpCharge != 0) & (mi.tpLocalCotAlpha > -900)).values}
SETS["clean"] = SETS["consistent"] & (mi.tpChargeFrac >= 0.9).values
rows = []
for sname, sel in SETS.items():
    for ang, col in [("A", "tpLocalCotAlpha"), ("B", "tpLocalCotBeta")]:
        tc = mi[col].values
        groups = [("all", np.ones(len(mi), bool))]
        groups += [(f"|cot| {BLAB[i]}", (np.abs(tc) >= BINS[i]) & (np.abs(tc) < BINS[i + 1])) for i in range(5)]
        groups += [(f"L{l}", (mi.layer == l).values) for l in range(1, 5)]
        groups += [(f"pT {PTLAB[i]}", ((mi.tpPt >= PTB[i]) & (mi.tpPt < PTB[i + 1])).values) for i in range(4)]
        for gname, g in groups:
            for mod in MODELS:
                s = sel & g
                r = metrics(P[f"{mod}_cot{ang}"].values[s], tc[s])
                r.update(set=sname, angle="cotAlpha" if ang == "A" else "cotBeta", group=gname, model=mod,
                         coverage=float(np.isfinite(P[f"{mod}_cot{ang}"].values[s]).mean()) if s.sum() else np.nan)
                rows.append(r)
RES = pd.DataFrame(rows); RES.to_csv(os.path.join(R, "resolution.csv"), index=False)

# ---------------- positions ----------------
prow = []
tsel = SETS["consistent"]
for lab, sel, tx, ty, extra in [("vs helix (hxDist<1mm)", tsel & (mi.hxDist < 0.1).values, mi.tx_hx.values, mi.ty_hx.values, None),
                               ("vs PSimHit (signal)", tsel & (mi.shN > 0).values & (mi.domSimHasTp == 1).values, mi.tx_sh.values, mi.ty_sh.values, None)]:
    for gname, g in [("all", np.ones(len(mi), bool))] + [(f"pT {PTLAB[i]}", ((mi.tpPt >= PTB[i]) & (mi.tpPt < PTB[i + 1])).values) for i in range(4)] + [(f"L{l}", (mi.layer == l).values) for l in range(1, 5)]:
        s = sel & g
        for mod in ["cpe"] + CNNS:
            px_, py_ = (mi.tx_cpe.values, mi.ty_cpe.values) if mod == "cpe" else (P[f"{mod}_tx"].values, P[f"{mod}_ty"].values)
            dx = (px_[s] - tx[s]) * mi.pitchX.values[s] * 1e4; dy = (py_[s] - ty[s]) * mi.pitchY.values[s] * 1e4
            prow.append(dict(ref=lab, group=gname, model=mod, n=int(s.sum()), x_smad_um=smad(dx), x_rms_um=float(np.sqrt(np.mean(dx ** 2))) if s.sum() else np.nan,
                             x_bias_um=float(np.median(dx)) if s.sum() else np.nan, y_smad_um=smad(dy),
                             y_rms_um=float(np.sqrt(np.mean(dy ** 2))) if s.sum() else np.nan, y_bias_um=float(np.median(dy)) if s.sum() else np.nan))
POS = pd.DataFrame(prow); POS.to_csv(os.path.join(R, "position.csv"), index=False)

# ---------------- residual vs local B / tan(theta_L) ----------------
brow = []
for var in ["bLocalX", "bLocalY", "bLocalZ", "bMag", "tanLAx"]:
    v = mi[var].values
    if var == "tanLAx":
        edges = [-1, 0.1, 1]
    else:
        edges = np.unique(np.quantile(v[tsel], [0, 0.2, 0.4, 0.6, 0.8, 1]))
    for i in range(len(edges) - 1):
        s = tsel & (v >= edges[i]) & (v <= edges[i + 1])
        for mod in [m for m in ["base", "bla"] if m in CNNS] + ["payload"]:
            for ang, col in [("A", "tpLocalCotAlpha"), ("B", "tpLocalCotBeta")]:
                r = metrics(P[f"{mod}_cot{ang}"].values[s], mi[col].values[s])
                brow.append(dict(var=var, lo=float(edges[i]), hi=float(edges[i + 1]), model=mod, angle=ang, n=r["n"], smad_deg=r["smad_deg"], sign_acc=r["sign_acc"]))
pd.DataFrame(brow).to_csv(os.path.join(R, "vs_field.csv"), index=False)

# ---------------- apply to unlinked / OOD ----------------
cats = {"unlinked (no TP)": TEST & (mi.linkClass == 1).values,
        "linked pT<0.25": TEST & (mi.linkClass == 2).values & (mi.tpPt < 0.25).values,
        "linked pT>=2": TEST & (mi.linkClass == 2).values & (mi.tpPt >= 2).values}
dist = {}
col3 = ["#2a78d6", "#eb6834", "#1baf7a"]
if "base" in CNNS:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k, (cn, s) in enumerate(cats.items()):
        for j, ang in enumerate(["A", "B"]):
            v = deg(P[f"base_cot{ang}"].values[s])
            ax[j].hist(v, bins=np.linspace(-90, 90, 91), histtype="step", density=True, lw=2, color=col3[k], label=f"{cn} (n={s.sum()})")
            dist[f"{cn}|cot{ang}"] = dict(n=int(s.sum()), q=[float(x) for x in np.percentile(v, [5, 25, 50, 75, 95])],
                                          frac_abs_gt_45=float(np.mean(np.abs(v) > 45)), frac_pos=float(np.mean(v > 0)))
    for j, t in enumerate(["CNN (base) predicted alpha  [deg, atan cotAlpha]", "CNN (base) predicted beta  [deg, atan cotBeta]"]):
        ax[j].set_xlabel(t); ax[j].set_ylabel("density"); ax[j].grid(alpha=0.25); ax[j].set_yscale("log")
    ax[0].legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FG, "pred_angle_distributions.png"), dpi=120); plt.close(fig)
    # same for the payload (noise-v6 draws for unlinked) for reference
    for cn, s in cats.items():
        for ang in ["A", "B"]:
            v = deg(P[f"payload_cot{ang}"].values[s]); v = v[np.isfinite(v)]
            dist[f"PAYLOAD {cn}|cot{ang}"] = dict(n=int(len(v)), q=[float(x) for x in np.percentile(v, [5, 25, 50, 75, 95])])
    # truth distributions for linked
    for cn in ["linked pT<0.25", "linked pT>=2"]:
        for ang, col in [("A", "tpLocalCotAlpha"), ("B", "tpLocalCotBeta")]:
            v = deg(mi[col].values[cats[cn]]); v = v[np.isfinite(v) & (mi[col].values[cats[cn]] > -900)]
            dist[f"TRUTH {cn}|cot{ang}"] = dict(n=int(len(v)), q=[float(x) for x in np.percentile(v, [5, 25, 50, 75, 95])])
json.dump(dist, open(os.path.join(R, "unlinked_distributions.json"), "w"), indent=1)
ood = []
s0 = TEST & (mi.linkClass == 2).values & (mi.tpPt < 0.25).values & (mi.tpCharge != 0).values & (mi.hxOk == 1).values & (mi.hxDist < 1).values
for mod in ["payload", "hgb"] + CNNS:
    for ang, col in [("A", "tpLocalCotAlpha"), ("B", "tpLocalCotBeta")]:
        r = metrics(P[f"{mod}_cot{ang}"].values[s0], mi[col].values[s0]); r.update(model=mod, angle=ang); ood.append(r)
pd.DataFrame(ood).to_csv(os.path.join(R, "ood_lowpt.csv"), index=False)

# ---------------- PSimHit residuals: unlinked-but-simlinked (signal) ----------------
sh = []
for lab, s in [("unlinked w/ PSimHit (all 200 ev)", ((mi.linkClass == 1) & (mi.shN > 0) & (mi.shCotAlpha > -900)).values),
               ("linked w/ PSimHit pT>=0.25 (test ev)", TEST & (mi.linkClass == 2).values & (mi.shN > 0).values & (mi.tpPt >= 0.25).values & (mi.shCotAlpha > -900).values)]:
    for mod in ["payload"] + CNNS + ["truth"]:
        for ang in ["A", "B"]:
            pc = mi[f"tpLocalCot{'Alpha' if ang == 'A' else 'Beta'}"].values if mod == "truth" else P[f"{mod}_cot{ang}"].values
            tc = mi[f"shCot{'Alpha' if ang == 'A' else 'Beta'}"].values
            for gname, g in [("all", np.ones(len(mi), bool))] + [(f"|cot| {BLAB[i]}", (np.abs(tc) >= BINS[i]) & (np.abs(tc) < BINS[i + 1])) for i in range(5)]:
                r = metrics(pc[s & g], tc[s & g]); r.update(sample=lab, model=mod, angle=ang, group=gname); sh.append(r)
pd.DataFrame(sh).to_csv(os.path.join(R, "psimhit_residuals.csv"), index=False)
um = (mi.linkClass == 1) & (mi.shN > 0)
json.dump(dict(n=int(um.sum()), pdg=pd.Series(np.abs(mi.shPdg[um])).value_counts().head(8).to_dict(),
               pabs_q=[float(x) for x in np.percentile(mi.shPabs[um], [10, 50, 90])],
               domSimPt_q=[float(x) for x in np.nanpercentile(mi.domSimPt[um].where(mi.domSimPt[um] > -900), [10, 50, 90])]),
          open(os.path.join(R, "unlinked_psimhit_population.json"), "w"), indent=1)

# ---------------- leak / tell test ----------------
te = mi[TEST].copy(); Pt = P[TEST]
ev = np.sort(te.event.unique()); half = te.event.isin(ev[::2]).values
y = (te.linkClass == 1).values.astype(int)
base_feat = te[FEAT].values
def auc(X, sel=None):
    clf = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=31, learning_rate=0.1, random_state=1)
    clf.fit(X[half], y[half]); p = clf.predict_proba(X[~half])[:, 1]
    out = {"all": float(roc_auc_score(y[~half], p))}
    lk = te.linkClass.values[~half]; pt = te.tpPt.values[~half]
    for nm, cs in [("vs linked pT>=2", (lk == 1) | (pt >= 2)), ("vs linked pT<0.3", (lk == 1) | ((lk == 2) & (pt < 0.3)))]:
        out[nm] = float(roc_auc_score(y[~half][cs], p[cs]))
    for l in range(1, 5):
        cs = te.layer.values[~half] == l; out[f"L{l}"] = float(roc_auc_score(y[~half][cs], p[cs]))
    return out
LK = {"non-angle (sizeX,sizeY,size,charge,layer)": auc(base_feat),
      "non-angle + payload angles (noise-v6 for unlinked)": auc(np.c_[base_feat, Pt.payload_cotA.fillna(-999), Pt.payload_cotB.fillna(-999)])}
for t in CNNS:
    LK[f"non-angle + CNN[{t}] angles"] = auc(np.c_[base_feat, Pt[f"{t}_cotA"], Pt[f"{t}_cotB"]])
    LK[f"non-angle + CNN[{t}] angles+signprob+xy"] = auc(np.c_[base_feat, Pt[f"{t}_cotA"], Pt[f"{t}_cotB"], Pt[f"{t}_pA"], Pt[f"{t}_pB"], Pt[f"{t}_tx"], Pt[f"{t}_ty"]])
json.dump(LK, open(os.path.join(R, "leak_test.json"), "w"), indent=1)
print(json.dumps(LK, indent=1))

# ---------------- figure: resolution vs |cot| ----------------
cons = RES[(RES.set == "consistent") & RES.group.str.startswith("|cot|")]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
order = ["payload", "hgb", "base", "bla", "pos", "base_clean", "small"]
pal = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#e34948"]
for j, ang in enumerate(["cotAlpha", "cotBeta"]):
    for k, mod in enumerate([m for m in order if m in MODELS]):
        d = cons[(cons.angle == ang) & (cons.model == mod)]
        ax[j].plot(range(5), d.smad_deg.values, marker="o", ms=8, lw=2, color=pal[k], label=mod)
    ax[j].set_xticks(range(5)); ax[j].set_xticklabels(BLAB); ax[j].set_yscale("log"); ax[j].grid(alpha=0.25)
    ax[j].set_xlabel(f"true |{ang}| bin"); ax[j].set_ylabel("sigma_MAD [deg], signed")
ax[0].legend(fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(FG, "resolution_vs_cot.png"), dpi=120); plt.close(fig)

# ---------------- small / full ratio (on-sensor inflation factor) ----------------
if "small" in CNNS and "base" in CNNS:
    rr = []
    for sname in ["consistent", "clean"]:
        for ang in ["cotAlpha", "cotBeta"]:
            for g in ["all"] + [f"|cot| {b}" for b in BLAB] + [f"L{l}" for l in range(1, 5)]:
                d = RES[(RES.set == sname) & (RES.angle == ang) & (RES.group == g)].set_index("model")
                rr.append(dict(set=sname, angle=ang, group=g, small_smad_deg=d.loc["small", "smad_deg"], full_smad_deg=d.loc["base", "smad_deg"],
                               ratio_signed=d.loc["small", "smad_deg"] / d.loc["base", "smad_deg"],
                               ratio_magnitude=d.loc["small", "mag_smad_deg"] / d.loc["base", "mag_smad_deg"],
                               small_sign=d.loc["small", "sign_acc"], full_sign=d.loc["base", "sign_acc"]))
    pd.DataFrame(rr).to_csv(os.path.join(R, "small_over_full.csv"), index=False)

# ---------------- truth-free inflation knob (see inflation.py) ----------------
import inflation as INF_
if "base" in CNNS:
    evs = np.sort(mi.event[TEST].unique()); calib = mi.event.isin(evs[::2]).values  # calibrate on half, verify on the other
    k = INF_.K; infl_rows, tabs = [], {"fine": [], "coarse": [], "layer": []}
    for ang, col in [("A", "tpLocalCotAlpha"), ("B", "tpLocalCotBeta")]:
        pc = P[f"base_cot{ang}"].values; tc = mi[col].values
        cell, coarse, pb = INF_.cells(mi.layer.values, pc, mi.sizeX.values, mi.sizeY.values)
        N = INF_.hash_normal(mi.event.values, mi.clIdx.values, INF_.SALT[ang])
        r = deg(pc) - deg(tc); cs = SETS["consistent"] & calib
        maps = {}
        for lvl, key in [("fine", cell), ("coarse", coarse), ("layer", mi.layer.values)]:
            sig, add = {}, {}
            for c in np.unique(key[cs]):
                m_ = cs & (key == c)
                if m_.sum() < 50:
                    continue
                sig[c] = smad(r[m_]); add[c] = INF_.calibrate_add(r[m_], N[m_], k, smad)
                tabs[lvl].append(dict(angle=ang, key=int(c), n=int(m_.sum()), sigma_deg=sig[c], add_deg=add[c],
                                      naive_add_deg=np.sqrt(k * k - 1) * sig[c]))
            maps[lvl] = (key, sig, add)
        sig = INF_.lookup([(maps[l][0], maps[l][1]) for l in ["fine", "coarse", "layer"]])
        add = INF_.lookup([(maps[l][0], maps[l][2]) for l in ["fine", "coarse", "layer"]])
        th = INF_.inflate(deg(pc), add, mi.event.values, mi.clIdx.values, ang)
        thN = INF_.inflate(deg(pc), np.sqrt(k * k - 1) * sig, mi.event.values, mi.clIdx.values, ang)  # naive formula
        thT = deg(tc) + k * (deg(pc) - deg(tc))            # truth-based alternative, comparison only
        ri, rn, rt = th - deg(tc), thN - deg(tc), thT - deg(tc)
        ver = SETS["consistent"] & ~calib
        fine_sig = maps["fine"][1]
        ratios = [smad(ri[ver & (cell == c)]) / fine_sig[c] for c in np.unique(cell[ver])
                  if (ver & (cell == c)).sum() >= 200 and c in fine_sig]
        ratiosN = [smad(rn[ver & (cell == c)]) / fine_sig[c] for c in np.unique(cell[ver])
                   if (ver & (cell == c)).sum() >= 200 and c in fine_sig]
        row = dict(angle=ang, k=k, nan_frac=float(np.isnan(add).mean()), cells_verified=len(ratios),
                   cell_ratio_median=float(np.median(ratios)), cell_ratio_q10=float(np.percentile(ratios, 10)),
                   cell_ratio_q90=float(np.percentile(ratios, 90)), naive_cell_ratio_median=float(np.median(ratiosN)),
                   all_smad_deg_before=smad(r[ver]), all_smad_deg_after=smad(ri[ver]), all_ratio=smad(ri[ver]) / smad(r[ver]),
                   all_ratio_naive=smad(rn[ver]) / smad(r[ver]), all_ratio_truthbased=smad(rt[ver]) / smad(r[ver]))
        tb = np.digitize(np.abs(tc), BINS[1:-1])
        for i, b in enumerate(BLAB):
            for lab, bb in [("predbin", pb), ("truebin", tb)]:
                m_ = ver & (bb == i)
                row[f"{lab} {b} n"] = int(m_.sum())
                if m_.sum() > 5:
                    row[f"{lab} {b} smad_before"] = smad(r[m_]); row[f"{lab} {b} smad_after"] = smad(ri[m_])
                    row[f"{lab} {b} ratio"] = smad(ri[m_]) / smad(r[m_])
        ul = TEST & (mi.linkClass == 1).values
        row["unlinked_frac_sign_flipped"] = float(np.mean(np.sign(th[ul]) != np.sign(deg(pc)[ul])))
        infl_rows.append(row)
    pd.DataFrame(infl_rows).T.to_csv(os.path.join(R, "inflation_knob.csv"))
    for lvl, rows_ in tabs.items():
        pd.DataFrame(rows_).to_csv(os.path.join(R, f"inflation_table_{lvl}.csv"), index=False)
print("done")
