"""Disc vs barrel cotBeta at MATCHED incidence (user question 2026-10-01).

Why do the disc cotBeta distributions differ from near-normal barrel ones? Is beta defined properly with respect to the
sensor, is cotBeta the right measure, and do flipped sensors confuse the CNN training?

Frames (checked on all 4000 modules, printed on page 1): local x = 25 um pitch, local y = 100 um pitch, w = sensor normal,
thickness 150 um, charge collected at w = -t/2 on every module (Phase2TrackerDigitizerAlgorithm, dir_z = -1).
  barrel: y = -z, w = +-r (zOutward), x = +-phi; Lorentz drift along +x (tanLA 0.202 on L2-L4; L1 = 3D, none)
  disc:   y = -r (radially INWARD) on every module, w = +-z (facing), x = +-phi; no Lorentz drift
A prompt track has cotBeta = -zOutward z / r (barrel) and -facing R / |Z| (disc; facing + = w away from the IP).

Pages
  1 frames; sizeY and P(sizeY >= 2) vs |cotBeta|, sizeX vs cotAlpha + tanLA (truth + clusters only); true-beta distributions
  2 the x-y tilt cue in the pixel images: sign(cov_xy) vs sign(beta) and vs sign((cotAlpha + tanLA) cotBeta)
  3 the representative disc sensor (ID2 ring 3, facing -) vs barrel sensors at matched incidence
  4 flips: the two facings of the same disc ring, and the two zOutward of the same barrel module ring
  5+ two radii per disc (ID1-ID4 x ring 1 / ring 4) vs matched barrel, and a table page
Selection: consistent-linked (TP, charged, helix within 1 cm of the hit), tpPt >= PT_MIN (default 1 GeV). Truth-only and
pixel observables use all 200 events; model outputs exist for the 50 test events only:
  v1 = predictions_test_all.parquet (evaluate_disc.py: factorised |angle| + sign head, NO orientation input;
       "sep" = the disc model on discs and the base model on the barrel, as in figs/disc_pred_angle_distributions)
  v2 = {barrel,disc}_pos_hubsig (orientation + position inputs; the class-C pseudo-truth CNN of the emulator prototype)
  emu = emu_ang.parquet (emulator v4 prototype: class-T emitted = truth + dump response; class-C path on the same clusters)
Matched barrel = barrel clusters (L2-L4 planar, or L1 3D) reweighted in (true beta, true alpha) to the disc sensor.
"""
import os, numpy as np, pandas as pd, pyarrow.dataset as ds, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import wasserstein_distance

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data_all")
P1 = "/Volumes/WDMac/smartpixels_scratch/cluster_cnn_preds/predictions_test_all.parquet"
P2 = "/Volumes/WDMac/smartpixels_scratch/cluster_cnn_v2/preds"
CACHE = "/Volumes/WDMac/smartpixels_scratch/cluster_cnn_v2/disc_vs_barrel"
PT_MIN = float(os.environ.get("PT_MIN", 1.0))
TAG = f"pt{PT_MIN:g}"
deg = lambda c: np.degrees(np.arctan(c))
C_DISC, C_L234, C_L1, C_RAW = "#2a78d6", "#eb6834", "#1baf7a", "#616161"


def load():
    f = os.path.join(CACHE, f"joined_{TAG}.parquet")
    if os.path.exists(f) and not os.environ.get("REBUILD"):
        return pd.read_parquet(f)
    cols = ["row", "event", "split", "subdet", "layer", "disk", "ring", "module", "modZ", "axZz", "zOutward", "linkClass", "tpPt",
            "tpCharge", "tpLocalCotAlpha", "tpLocalCotBeta", "hxOk", "hxDist", "sizeX", "sizeY", "tanLAx"]
    flt = (ds.field("subdet") == 1) | ((ds.field("subdet") == 2) & (ds.field("disk") <= 4))
    m = ds.dataset(os.path.join(DATA, "meta.parquet")).to_table(columns=cols, filter=flt).to_pandas()
    print("barrel + ID1-4 clusters", len(m), flush=True)
    cons = ((m.linkClass == 2) & (m.tpCharge != 0) & (m.tpLocalCotAlpha > -900) & (m.hxOk == 1) & (m.hxDist < 1.0)).values
    m = m[cons & (m.tpPt >= PT_MIN).values].reset_index(drop=True)
    print("selected", len(m), m.subdet.value_counts().to_dict(), flush=True)
    disc = (m.subdet == 2).values
    m["facing"] = np.where(disc, np.sign(m.axZz) * np.sign(m.modZ), m.zOutward).astype(np.int8)
    m["tb"], m["ta"] = deg(m.tpLocalCotBeta).astype(np.float32), deg(m.tpLocalCotAlpha).astype(np.float32)
    # pixel observables from the 24x16 images (row = local x / 25 um, col = local y / 100 um; value = ADC + 1, 0 = empty)
    imgs = np.load(os.path.join(DATA, "images.npy"), mmap_mode="r")
    ix, iy = np.arange(24, dtype=np.float32)[None, :, None], np.arange(16, dtype=np.float32)[None, None, :]
    ymin, cxy = np.empty(len(m), np.float32), np.empty(len(m), np.float32)
    rows = m.row.values
    for i in range(0, len(m), 100_000):
        q = np.asarray(imgs[rows[i:i + 100_000]], dtype=np.float32)
        Q = q.sum((1, 2))
        mx, my = (q * ix).sum((1, 2)) / Q, (q * iy).sum((1, 2)) / Q
        cxy[i:i + len(q)] = (q * (ix - mx[:, None, None]) * (iy - my[:, None, None])).sum((1, 2)) / Q
        qy = np.sort(q.sum(1), axis=1)
        ymin[i:i + len(q)] = qy[:, -2] / (qy[:, -1] + qy[:, -2])
        print("images", i + len(q), "/", len(m), flush=True)
    m["yMinFrac"], m["covXY"] = ymin, cxy
    v1 = pd.read_parquet(P1, columns=["row", "disc_cotB", "base_cotB", "unified_cotB"])
    v1 = v1[v1.row.isin(rows)]
    v1["v1sep"] = np.where(v1.disc_cotB.notna(), v1.disc_cotB, v1.base_cotB)
    m = m.merge(v1[["row", "v1sep", "unified_cotB"]].rename(columns={"unified_cotB": "v1uni"}), on="row", how="left")
    v2 = pd.concat([pd.read_parquet(os.path.join(P2, f"{r}_pos_hubsig.parquet"), columns=["row", "cotB"]) for r in ("barrel", "disc")])
    m = m.merge(v2.rename(columns={"cotB": "v2"}), on="row", how="left")
    e = pd.read_parquet(os.path.join(P2, "emu_ang.parquet"), columns=["row", "clsT", "trueCotB", "emuCotB_T", "CpathCotB_knn"])
    m = m.merge(e.rename(columns={"emuCotB_T": "emuT", "CpathCotB_knn": "emuC"}), on="row", how="left")
    for c in ["v1sep", "v1uni", "v2", "emuT", "emuC", "trueCotB"]:
        m[c] = deg(m[c]).astype(np.float32)
    t = m.trueCotB.notna()
    print("emu truth == ntuple truth:", float(np.nanmax(np.abs(m.trueCotB[t] - m.tb[t]))), "deg; clsT on the selection:",
          float(m.clsT[t].mean()), flush=True)
    os.makedirs(CACHE, exist_ok=True)
    m.to_parquet(f)
    return m


m = load()
DISC, L1, L234 = (m.subdet == 2).values, ((m.subdet == 1) & (m.layer == 1)).values, ((m.subdet == 1) & (m.layer >= 2)).values
TEST = (m.split == 2).values
lay_la = np.where(L234, 0.202, 0.0)
print("selected: disc", DISC.sum(), "L1", L1.sum(), "L2-4", L234.sum(), "| test split", (DISC & TEST).sum(), (L1 & TEST).sum(),
      (L234 & TEST).sum(), flush=True)

TB_E = np.arange(-90, 90.5, 1.0)
TA_E = np.array([-90, -40, -25, -15, -10, -6, -4, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 4, 6, 10, 15, 25, 40, 90.0])


def reweight(ref, pool):
    """weights for the barrel rows `pool` so that their (true beta, true alpha) matches the rows `ref`."""
    Hr = np.histogram2d(m.tb.values[ref], m.ta.values[ref], [TB_E, TA_E])[0]
    tb, ta = m.tb.values[pool], m.ta.values[pool]
    Hb = np.histogram2d(tb, ta, [TB_E, TA_E])[0]
    ib = np.clip(np.digitize(tb, TB_E) - 1, 0, len(TB_E) - 2); ia = np.clip(np.digitize(ta, TA_E) - 1, 0, len(TA_E) - 2)
    w = np.where(Hb[ib, ia] > 0, Hr[ib, ia] / np.maximum(Hb[ib, ia], 1), 0.0)
    cover = float(Hr[Hb > 0].sum() / max(Hr.sum(), 1))
    return w, cover, float(w.sum() ** 2 / max((w ** 2).sum(), 1e-12))


def wquant(x, w, q):
    o = np.argsort(x); x, w = x[o], w[o]; c = np.cumsum(w) / w.sum()
    return float(np.interp(q, c, x))


def core_rms(r, w):
    """RMS about zero inside an iterative 3-sigma clip seeded at the weighted mode (the cluster_cnn METRIC CONVENTION)."""
    if w.sum() <= 0:
        return np.nan
    h, e = np.histogram(r, np.arange(-90, 90.5, 1.0), weights=w); mode = 0.5 * (e[h.argmax()] + e[h.argmax() + 1])
    s, c = 10.0, mode
    for _ in range(20):
        k = np.abs(r - c) < 3 * s
        if w[k].sum() <= 0:
            break
        c = np.average(r[k], weights=w[k]); s2 = np.sqrt(np.average((r[k] - c) ** 2, weights=w[k]))
        if abs(s2 - s) < 1e-3:
            break
        s = s2
    k = np.abs(r - c) < 3 * s
    return float(np.sqrt(np.average(r[k] ** 2, weights=w[k])))


def best_barrel(ref, layers):
    """the single barrel sensor class (layer, module, zOutward) whose true-beta distribution is closest (W1) to `ref`."""
    best = None
    for (l, mod, f), g in m[np.isin(m.layer, layers) & (m.subdet == 1)].groupby(["layer", "module", "facing"]):
        if len(g) < 200:
            continue
        d = wasserstein_distance(m.tb.values[ref], g.tb.values)
        if best is None or d < best[0]:
            best = (d, l, mod, f, g.index.values)
    return best


def step(ax, x, w, bins, color, label, ls="-", lw=1.8, clip=True):
    """normalised step histogram; clip=True piles the overflow into the edge bins."""
    ok = np.isfinite(x)
    if ok.sum() == 0 or w[ok].sum() <= 0:
        return
    h, e = np.histogram(np.clip(x[ok], bins[0], bins[-1] - 1e-6) if clip else x[ok], bins, weights=w[ok])
    ax.stairs(h / h.sum(), e, color=color, ls=ls, lw=lw, label=label)


def samples(ref_mask, test_only):
    """[(label, rows, weights, colour, ls)] for the disc sensor and its matched barrel samples."""
    ref = ref_mask & (TEST if test_only else True)
    out = [("disc", np.where(ref)[0], np.ones(ref.sum()), C_DISC, "-", np.nan, ref.sum())]
    for lab, pool, col in (("barrel L2-L4 (matched)", L234, C_L234), ("barrel L1 3D (matched)", L1, C_L1)):
        p = pool & (TEST if test_only else True)
        w, cov, neff = reweight(ref, p)
        out.append((lab, np.where(p)[0], w, col, "-", cov, neff))
    return out


def stats(rows, w):
    """summary numbers of one (weighted) sample."""
    r = lambda c: m[c].values[rows]
    d = dict(N=len(rows), Neff=float(w.sum() ** 2 / max((w ** 2).sum(), 1e-12)), med_true_beta=wquant(r("tb"), w, 0.5),
             P_sizeY_ge2=float(np.average(r("sizeY") >= 2, weights=w)), mean_sizeX=float(np.average(r("sizeX"), weights=w)))
    s2 = r("sizeY") == 2
    d["yMinFrac_sizeY2"] = float(np.average(r("yMinFrac")[s2], weights=w[s2])) if w[s2].sum() > 0 else np.nan
    for c in ["v1sep", "v1uni", "v2", "emuT", "emuC"]:
        x, ok = r(c), np.isfinite(r(c))
        if w[ok].sum() <= 0:
            continue
        res = x[ok] - r("tb")[ok]
        d[f"{c}_median"] = wquant(x[ok], w[ok], 0.5)
        d[f"{c}_res_median"] = wquant(res, w[ok], 0.5)
        d[f"{c}_res_coreRMS"] = core_rms(res, w[ok])
        d[f"{c}_res_fullRMS"] = float(np.sqrt(np.average(res ** 2, weights=w[ok])))
        d[f"{c}_opp_sign"] = float(np.average(np.sign(x[ok]) != np.sign(r("tb")[ok]), weights=w[ok]))
    return d


def sensor(disk, ring, facing):
    return DISC & (m.disk == disk).values & (m.ring == ring).values & (m.facing == facing).values


B_BETA = np.arange(-60, 60.5, 1.5)
os.makedirs(os.path.join(HERE, "figs"), exist_ok=True); os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
PDF = PdfPages(os.path.join(HERE, f"figs/disc_vs_barrel_beta_{TAG}.pdf"))
PNG_DIR = os.environ.get("PNG_DIR")  # also write each page as a PNG (for viewers without a PDF renderer)


def save(fig):
    PDF.savefig(fig)
    if PNG_DIR:
        os.makedirs(PNG_DIR, exist_ok=True); fig.savefig(os.path.join(PNG_DIR, f"{TAG}_page{PDF.get_pagecount():02d}.png"), dpi=70)
    plt.close(fig)



# ---------------- page 1: frames + cluster size vs incidence (model-free) ----------------
fig, ax = plt.subplots(2, 3, figsize=(19, 10.5), constrained_layout=True)
txt = ("LOCAL FRAMES (all 4000 modules; module axes in global coordinates)\n"
       "every module: pitch x 25 um, pitch y 100 um, thickness 150 um;\n"
       "  charge collected at w = -t/2 (Phase-2 digitizer, dir_z = -1)\n"
       "barrel: y = -z on every module; w = +r or -r (zOutward), half each;\n"
       "  x = +-phi; drift along +x, tanLA 0.202 (L2-L4), 0 (L1 3D)\n"
       "disc ID1-ID4: y = -r (radially inward) on every module;\n"
       "  w = +z or -z (facing), half each per ring; x = +-phi; no drift\n"
       "=> beta = the plane (100 um pitch, normal) = the r-z plane on BOTH\n"
       "prompt cotBeta = -zOutward z/r (barrel), -facing R/|Z| (disc)\n"
       "a flip = a rigid 180 deg rotation (x, w reversed, y kept):\n"
       "  the image <-> local-angle relation is identical for both facings;\n"
       "  only WHICH local angles occur changes (disc: sign(beta) = -facing)")
ax[0, 0].axis("off"); ax[0, 0].text(-0.05, 1, txt, va="top", family="monospace", fontsize=8)
bins = np.arange(0, 1.21, 0.04); c = 0.5 * (bins[1:] + bins[:-1])
for lab, s, col in (("disc ID1-ID4", DISC, C_DISC), ("barrel L2-L4", L234, C_L234), ("barrel L1 (3D)", L1, C_L1)):
    a = np.abs(m.tpLocalCotBeta.values[s]); i = np.digitize(a, bins) - 1; ok = (i >= 0) & (i < len(c))
    n = np.bincount(i[ok], minlength=len(c)); keep = n >= 50
    sy = np.bincount(i[ok], weights=m.sizeY.values[s][ok], minlength=len(c)) / np.maximum(n, 1)
    p2 = np.bincount(i[ok], weights=(m.sizeY.values[s][ok] >= 2), minlength=len(c)) / np.maximum(n, 1)
    ax[0, 1].plot(c[keep], sy[keep], "o-", ms=3, color=col, label=lab); ax[0, 2].plot(c[keep], p2[keep], "o-", ms=3, color=col, label=lab)
ax[0, 1].plot(c, 1 + 1.5 * c, ":", color=C_RAW, label="1 + t|cotβ|/pitch_y (geometry)")
ax[0, 2].plot(c, np.minimum(1.5 * c, 1), ":", color=C_RAW, label="min(t|cotβ|/pitch_y, 1)")
ax[0, 1].set(xlabel="|cotβ| (true)", ylabel="mean sizeY [columns of 100 µm]", title="cluster length along local y vs |cotβ|")
ax[0, 2].set(xlabel="|cotβ| (true)", ylabel="P(sizeY ≥ 2)", title="fraction of clusters spanning ≥ 2 columns")
bx = np.arange(-0.6, 0.61, 0.03); cx = 0.5 * (bx[1:] + bx[:-1])
for lab, s, col in (("disc ID1-ID4", DISC, C_DISC), ("barrel L2-L4", L234, C_L234), ("barrel L1 (3D)", L1, C_L1)):
    u = m.tpLocalCotAlpha.values[s] + lay_la[s]; i = np.digitize(u, bx) - 1; ok = (i >= 0) & (i < len(cx))
    n = np.bincount(i[ok], minlength=len(cx)); keep = n >= 50
    sx = np.bincount(i[ok], weights=m.sizeX.values[s][ok], minlength=len(cx)) / np.maximum(n, 1)
    ax[1, 0].plot(cx[keep], sx[keep], "o-", ms=3, color=col, label=lab)
ax[1, 0].plot(cx, 1 + 6 * np.abs(cx), ":", color=C_RAW, label="1 + t|cotα + tanLA|/pitch_x")
ax[1, 0].set(xlabel="cotα + tanLA (true; tanLA = 0.202 on L2-L4, 0 on L1 and discs)", ylabel="mean sizeX [rows of 25 µm]",
             title="cluster width along local x: the drift only shifts x")
for d, col in zip(range(1, 5), ["#08306b", "#2171b5", "#6baed6", "#9ecae1"]):
    s = DISC & (m.disk == d).values
    step(ax[1, 1], m.tb.values[s], np.ones(s.sum()), B_BETA, col, f"disc ID{d}")
for l, col in zip(range(1, 5), ["#1baf7a", "#f4a261", "#eb6834", "#9c3d10"]):
    s = (m.subdet == 1).values & (m.layer == l).values & (np.abs(m.modZ) < 1).values
    step(ax[1, 1], m.tb.values[s], np.ones(s.sum()), B_BETA, col, f"barrel L{l}, central module (z = 0)", ls="--")
ax[1, 1].set(xlabel="true β [deg] (signed, local frame)", ylabel="fraction", title="incidence: disc rings sit at ±R/|Z| (two facings);\n"
             "central barrel modules are broad (vertex z spread / r)")
for d, col in zip(range(1, 5), ["#08306b", "#2171b5", "#6baed6", "#9ecae1"]):
    for r, ls in ((1, "-"), (4, "--")):
        s = DISC & (m.disk == d).values & (m.ring == r).values
        step(ax[1, 2], np.abs(m.tb.values[s]), np.ones(s.sum()), np.arange(0, 45.5, 0.75), col, f"ID{d} ring {r}", ls=ls)
ax[1, 2].set(xlabel="|true β| [deg]", ylabel="fraction", title="disc |β| per ring (ring 1 R = 5.3 cm, ring 4 R = 13.8 cm)")
for a in ax.flat[1:]:
    a.grid(alpha=0.25); a.legend(fontsize=7, frameon=False)
fig.suptitle(f"Page 1. Frames and cluster size vs incidence (consistent-linked, pT ≥ {PT_MIN:g} GeV, 200 events; truth + clusters only)")
save(fig)

# ---------------- page 2: the tilt cue in the images (model-free) ----------------
fig, ax = plt.subplots(1, 3, figsize=(19, 5.6), constrained_layout=True)
bb = np.array([0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90.0]); cb = 0.5 * (bb[1:] + bb[:-1])
tilt_rows = []
for lab, s, col in (("disc ID1-ID4", DISC, C_DISC), ("barrel L2-L4", L234, C_L234), ("barrel L1 (3D)", L1, C_L1)):
    two = s & (m.sizeX >= 2).values & (m.sizeY >= 2).values & (np.abs(m.covXY) > 1e-6).values
    sc = np.sign(m.covXY.values)
    f_beta = sc == np.sign(m.tpLocalCotBeta.values)
    f_prod = sc == np.sign((m.tpLocalCotAlpha.values + lay_la) * m.tpLocalCotBeta.values)
    i = np.digitize(np.abs(m.tb.values), bb) - 1
    fr, fb, fp = [], [], []
    for k in range(len(cb)):
        a, t = s & (i == k), two & (i == k)
        fr.append(t.sum() / max(a.sum(), 1)); fb.append(f_beta[t].mean() if t.sum() >= 50 else np.nan)
        fp.append(f_prod[t].mean() if t.sum() >= 50 else np.nan)
        tilt_rows.append(dict(sample=lab, abs_beta_lo=bb[k], abs_beta_hi=bb[k + 1], N=int(a.sum()), frac_2D=fr[-1], P_sign_cov_eq_sign_beta=fb[-1],
                              P_sign_cov_eq_sign_product=fp[-1]))
    ax[0].plot(cb, fr, "o-", color=col, label=lab); ax[1].plot(cb, fb, "o-", color=col, label=lab); ax[2].plot(cb, fp, "o-", color=col, label=lab)
pd.DataFrame(tilt_rows).to_csv(os.path.join(HERE, f"results/disc_vs_barrel_beta_tilt_{TAG}.csv"), index=False)
ax[0].set(ylabel="fraction with sizeX ≥ 2 and sizeY ≥ 2", title="clusters that carry any 2-D (x-y) shape")
ax[1].set(ylabel="P( sign cov_xy = sign cotβ )", title="the image tilt alone gives the SIGN of β?\n(0.5 = no information)")
ax[2].set(ylabel="P( sign cov_xy = sign[(cotα + tanLA) cotβ] )", title="the tilt gives the sign of the PRODUCT on every sensor")
for a in ax:
    a.set_xlabel("|true β| [deg]"); a.grid(alpha=0.25); a.legend(fontsize=8, frameon=False)
ax[1].axhline(0.5, color=C_RAW, lw=0.8); ax[1].set_ylim(0.3, 1.02); ax[2].set_ylim(0.3, 1.02)
fig.suptitle("Page 2. A straight track deposits a segment along (cotα + tanLA, cotβ) in the (x, y) image. Without drift the image is symmetric under "
             "(α, β) -> (-α, -β): only sign(αβ) is visible.\nWith drift (L2-L4) cotα + tanLA > 0 for most tracks, so the tilt fixes sign(β). "
             "Charge-weighted covariance of the 24x16 ADC image, clusters with sizeX ≥ 2 and sizeY ≥ 2.", fontsize=10)
save(fig)


# ---------------- page 3: the representative disc sensor vs matched barrel sensors ----------------
def sensor_page(disk, ring, facing, title):
    ref = sensor(disk, ring, facing)
    bestP = best_barrel(np.where(ref)[0], [2, 3, 4]); best1 = best_barrel(np.where(ref)[0], [1])
    fig, ax = plt.subplots(3, 4, figsize=(22, 14), constrained_layout=True)
    S_all, S_test = samples(ref, False), samples(ref, True)
    raw = [(f"best single barrel sensor L{bestP[1]} module {bestP[2]} zOut {bestP[3]:+d} (raw)", bestP[4], C_RAW, "--")]
    bmin, bmax = np.percentile(m.tb.values[ref], [0.5, 99.5]); bz = np.arange(np.floor(bmin) - 6, np.ceil(bmax) + 6.5, 1.0)

    def draw(a, col, bins, S, with_raw=False, mask=None):
        for lab, rows, w, cl, ls, *_ in S:
            k = rows if mask is None else rows[mask(rows)]
            ww = w if mask is None else w[mask(rows)]
            step(a, m[col].values[k], ww, bins, cl, lab, ls)
        if with_raw:
            for lab, rows, cl, ls in raw:
                k = rows if mask is None else rows[mask(rows)]
                step(a, m[col].values[k], np.ones(len(k)), bins, cl, lab, ls, lw=1.2, clip=False)

    draw(ax[0, 0], "tb", bz, S_all, True); ax[0, 0].set(xlabel="true β [deg]", title="incidence β (matched by construction)")
    draw(ax[0, 1], "ta", np.arange(-12, 12.5, 0.5), S_all, True); ax[0, 1].set(xlabel="true α [deg]", title="incidence α (matched)")
    draw(ax[0, 2], "sizeY", np.arange(0.5, 6.5, 1.0), S_all, True); ax[0, 2].set(xlabel="sizeY [columns, 100 µm]", title="cluster length along y")
    draw(ax[0, 3], "sizeX", np.arange(0.5, 8.5, 1.0), S_all, True); ax[0, 3].set(xlabel="sizeX [rows, 25 µm]",
                                                                                  title="cluster width along x (drift on L2-L4)")
    draw(ax[1, 0], "yMinFrac", np.linspace(0, 0.5, 11), S_all, True, mask=lambda r: m.sizeY.values[r] == 2)
    ax[1, 0].set(xlabel="charge fraction in the minority column (sizeY = 2)", title="charge sharing along y")
    pb = np.arange(-60, 60.5, 1.5)
    for a, col, t in ((ax[1, 1], "v1uni", "v1 UNIFIED CNN (same net on barrel and disc)"), (ax[1, 2], "v1sep", "v1 separate CNNs (disc net / barrel net)\n= your disc_pred_angle_distributions"),
                      (ax[1, 3], "v2", "v2 pos-hubsig CNN (orientation + position inputs)"), (ax[2, 0], "emuT", "EMULATOR class T emitted β"),
                      (ax[2, 2], "emuC", "EMULATOR class-C path emitted β (same clusters)")):
        draw(a, col, pb, S_test); a.axvline(np.median(m.tb.values[ref]), color=C_DISC, lw=0.8, ls=":")
        a.set(xlabel="predicted / emitted β [deg]", title=t)
    for lab, rows, w, cl, ls, *_ in S_test:
        step(ax[2, 1], m.emuT.values[rows] - m.tb.values[rows], w, np.arange(-60, 60.5, 1.5), cl, lab, ls)
    ax[2, 1].set(xlabel="emitted − true β [deg]", title="EMULATOR class T residual")
    rows_tab = []
    lines = [f"{'sample':14s} {'N':>7s} {'Neff':>6s} {'P(sY≥2)':>7s} {'v1uni':>6s} {'v1sep':>6s} {'v2':>6s} {'emuT':>6s} {'core':>5s} {'opp':>5s}"]
    for (lab, rows, w, cl, ls, cov, neff), (_, rt, wt, *_rest) in zip(S_all, S_test):
        a, t = stats(rows, w), stats(rt, wt)
        rows_tab.append(dict(sample=lab, disk=disk, ring=ring, facing=facing, cover=cov, **{f"all_{k}": v for k, v in a.items()},
                             **{f"test_{k}": v for k, v in t.items()}))
        short = lab.replace("barrel ", "").replace(" (matched)", "").replace(" 3D", "")
        lines.append(f"{short:14s} {a['N']:7d} {a['Neff']:6.0f} {a['P_sizeY_ge2']:7.3f} {t.get('v1uni_median', np.nan):6.1f} "
                     f"{t.get('v1sep_median', np.nan):6.1f} {t.get('v2_median', np.nan):6.1f} {t.get('emuT_median', np.nan):6.1f} "
                     f"{t.get('emuT_res_coreRMS', np.nan):5.1f} {t.get('emuT_opp_sign', np.nan):5.2f}")
    lines += ["", f"best raw L2-L4 sensor: L{bestP[1]} module {bestP[2]} zOutward {bestP[3]:+d} (W1 {bestP[0]:.1f} deg, N {len(bestP[4])})",
              f"best raw L1 sensor:    L1 module {best1[2]} zOutward {best1[3]:+d} (W1 {best1[0]:.1f} deg, N {len(best1[4])})",
              "v1uni / v1sep / v2 / emuT = median predicted or emitted β [deg]",
              "core = core RMS of emitted − true (mode-seeded 3σ clip, about 0) [deg]",
              "opp = fraction emitted with the opposite sign to the truth",
              f"disc (β, α) weight in cells the barrel covers: L2-L4 {S_all[1][5]:.3f}, L1 {S_all[2][5]:.3f}",
              "P(sY≥2): all 200 events; model columns: the 50 test events"]
    ax[2, 3].axis("off"); ax[2, 3].text(-0.08, 1, "\n".join(lines), va="top", family="monospace", fontsize=7)
    for a in ax.flat[:-1]:
        a.grid(alpha=0.25); a.set_ylabel("fraction")
    ax[0, 0].legend(fontsize=7, frameon=False); ax[1, 1].legend(fontsize=7, frameon=False)
    fig.suptitle(title, fontsize=11)
    save(fig)
    return rows_tab


summary = []
summary += sensor_page(2, 3, -1, f"Page 3. Representative disc sensor ID2 ring 3 (R = 10.5 cm, |Z| = 32.9 cm), facing − (w toward the IP ⇒ β > 0) vs barrel "
                                  f"sensors at MATCHED incidence (reweighted in true β × α). consistent-linked, pT ≥ {PT_MIN:g} GeV")

# ---------------- page 4: flips ----------------
fig, ax = plt.subplots(2, 5, figsize=(24, 9.5), constrained_layout=True)
flip_rows = []
models = [("tb", "TRUE β"), ("v1sep", "v1 separate CNN (no orientation input)"), ("v1uni", "v1 unified CNN (no orientation input)"),
          ("v2", "v2 pos-hubsig (orientation + position)"), ("emuT", "emulator class T")]
for row_i, (lab, base, keyname, keys) in enumerate((("disc ID2 ring 3", DISC & (m.disk == 2).values & (m.ring == 3).values, "facing", (1, -1)),
                                                    ("barrel L1 (3D) module 6 (z = +4.4 cm)", L1 & (m.module == 6).values, "facing", (1, -1)))):
    for j, (col, t) in enumerate(models):
        for f, cl in zip(keys, (C_L234, C_DISC)):
            s = base & (m[keyname] == f).values & (TEST if col != "tb" else True)
            step(ax[row_i, j], m[col].values[s], np.ones(s.sum()), np.arange(-87, 87.5, 1.5), cl,
                 f"{'facing' if row_i == 0 else 'zOutward'} {f:+d} (N {s.sum()})")
        ax[row_i, j].set(title=f"{lab}\n{t}", xlabel="β [deg]"); ax[row_i, j].grid(alpha=0.25); ax[row_i, j].legend(fontsize=7, frameon=False)
for surf, base in [(f"ID{d}", DISC & (m.disk == d).values) for d in range(1, 5)] + [(f"IL{l}", (m.subdet == 1).values & (m.layer == l).values) for l in range(1, 5)]:
    for f in (1, -1):
        s = base & (m.facing == f).values & TEST & (np.abs(m.tb) > 5).values
        d = dict(surface=surf, facing=f, N=int(s.sum()), frac_true_pos=float((m.tb.values[s] > 0).mean()))
        for col in ["v1sep", "v1uni", "v2", "emuT"]:
            x = m[col].values[s]; ok = np.isfinite(x)
            d[f"{col}_frac_pos"] = float((x[ok] > 0).mean()); d[f"{col}_sign_acc"] = float((np.sign(x[ok]) == np.sign(m.tb.values[s][ok])).mean())
        flip_rows.append(d)
FL = pd.DataFrame(flip_rows); FL.to_csv(os.path.join(HERE, f"results/disc_vs_barrel_beta_flips_{TAG}.csv"), index=False)
fig.suptitle("Page 4. FLIPS. Same ring / same module ring, the two orientations: the truth is mirrored (sign(β) = −facing on the discs). A net that cannot see "
             "the orientation and gets no sign from the image\n(discs, L1: page 2) must give the SAME output for both: its sign is then right for one facing "
             "and wrong for the other. Model outputs: 50 test events.", fontsize=10)
save(fig)
fig, a = plt.subplots(figsize=(17, 6.5)); a.axis("off")
a.text(0, 1, "Sign accuracy and fraction predicted positive per surface × orientation (|true β| > 5 deg, test events)\n\n" +
       FL.round(3).to_string(index=False), va="top", family="monospace", fontsize=7.5)
save(fig)

# ---------------- the emulator's disc offset: the SmartPixels dump at the disc key vs the barrel key ----------------
DUMP = os.path.expanduser("~/smartpixels/ngtagger-train/2t-Conv1D_Full-2bit_optimized-vars.parquet")
dmp = pd.read_parquet(DUMP, columns=["cotAtrue", "cotBtrue", "cotA"])
dA, dB, dP = deg(dmp.cotBtrue * 1.8827422609846889).values, deg(dmp.cotAtrue * 6.528905408493022).values, deg(dmp.cotA * 6.528905408493022).values
drows = []
for band, (a0, a1) in (("disc key: alpha_eq = atan(cotA - 0.312), -21..-13 deg (no tilt)", (-21, -13)), ("barrel key near normal: alpha -4..4 deg", (-4, 4))):
    for b0, b1 in ((-30, -20), (-20, -12), (-12, -6), (-6, 0), (0, 6), (6, 12), (12, 20), (20, 30)):
        k = (dA >= a0) & (dA < a1) & (dB >= b0) & (dB < b1)
        drows.append(dict(alpha_band=band, true_beta=f"{b0}..{b1}", N=int(k.sum()), median_emitted_beta=float(np.median(dP[k])),
                          median_residual=float(np.median(dP[k] - dB[k])), opposite_sign=float(np.mean(np.sign(dP[k]) != np.sign(dB[k])))))
DT = pd.DataFrame(drows); DT.to_csv(os.path.join(HERE, "results/disc_vs_barrel_beta_dump_keys.csv"), index=False)
fig, a = plt.subplots(figsize=(15, 7)); a.axis("off")
a.text(0, 1, "The emulator's disc offset. Class T emitted beta = truth + the response of the SmartPixels dump (Conv1D_Full-2bit, CMS axes) drawn at the\n"
       "lookup key. Discs are looked up at alpha_eq = atan(cotA - 0.312), where the PixelAV Lorentz drift cancels (the cluster has NO tilt, as on a\n"
       "real disc); the barrel at the true alpha (~0 near normal: the drift tilts the cluster). Inside the |beta| < ~20 deg plateau the dump leans\n"
       "POSITIVE at the disc key and is centred near 0 at the barrel key. On the discs sign(beta) = -facing, so facing - (beta > 0) gains and\n"
       "facing + (beta < 0) loses: inside the plateau the disc emitted medians sit +1.5 to +6 deg above the matched barrel ones (table pages).\nThe sign carried into the CMS "
       "frame rests on the dump-to-CMS axis map (z_pix = -w_cms); CMSSW itself collects at w = -t/2 on every module. Dump cells hold 40-115 rows.\n\n"
       + DT.round(2).to_string(index=False), va="top", family="monospace", fontsize=8)
save(fig)

# ---------------- pages 5+: two radii per disc ----------------
COLS = [("tb", "true β", None), ("sizeY", "sizeY", np.arange(0.5, 6.5, 1.0)), ("v1uni", "v1 unified CNN β", None), ("v2", "v2 pos-hubsig β", None),
        ("emuT", "emulator class T β", None), ("res", "emulator class T: emitted − true", np.arange(-45, 45.5, 1.5))]
pairs = [(d, r) for d in range(1, 5) for r in (1, 4)]
for page in range(2):
    fig, ax = plt.subplots(4, len(COLS), figsize=(26, 15), constrained_layout=True)
    for i, (d, r) in enumerate(pairs[page * 4:(page + 1) * 4]):
        ref = sensor(d, r, -1)
        S_all, S_test = samples(ref, False), samples(ref, True)
        med = np.median(m.tb.values[ref]); pb = np.arange(-45, max(45, med + 25) + 0.1, 1.5)
        for j, (col, t, bins) in enumerate(COLS):
            S = S_all if col in ("tb", "sizeY") else S_test
            for lab, rows, w, cl, ls, *_ in S:
                x = m.emuT.values[rows] - m.tb.values[rows] if col == "res" else m[col].values[rows]
                step(ax[i, j], x, w, pb if bins is None else bins, cl, lab, ls)
            if col not in ("sizeY", "res"):
                ax[i, j].axvline(med, color=C_DISC, lw=0.8, ls=":")
            ax[i, j].set(title=f"ID{d} ring {r} (R {[5.27, 7.07, 10.49, 13.83][r - 1]} cm), facing −: {t}", xlabel="deg" if col != "sizeY" else "columns")
            ax[i, j].grid(alpha=0.25); ax[i, j].title.set_fontsize(8)
        ax[i, 0].legend(fontsize=6.5, frameon=False)
    fig.suptitle(f"Pages 5-6. Two radii per disc (ring 1 = 5.3 cm, ring 4 = 13.8 cm), facing − (β > 0), vs barrel L2-L4 / L1 at matched incidence. "
                 f"pT ≥ {PT_MIN:g} GeV. sizeY / true β: 200 events; model outputs: 50 test events.", fontsize=11)
    save(fig)

# numbers for every (disk, ring 1/4, facing) sensor
for d, r in pairs:
    for f in (1, -1):
        ref = sensor(d, r, f)
        for (lab, rows, w, cl, ls, cov, neff), (_, rt, wt, *_rest) in zip(samples(ref, False), samples(ref, True)):
            a, t = stats(rows, w), stats(rt, wt)
            summary.append(dict(sample=lab, disk=d, ring=r, facing=f, cover=cov, **{f"all_{k}": v for k, v in a.items()},
                                **{f"test_{k}": v for k, v in t.items()}))
SUM = pd.DataFrame(summary); SUM.to_csv(os.path.join(HERE, f"results/disc_vs_barrel_beta_summary_{TAG}.csv"), index=False)
show = SUM[["disk", "ring", "facing", "sample", "all_N", "all_Neff", "all_med_true_beta", "all_P_sizeY_ge2", "all_yMinFrac_sizeY2",
            "test_v1uni_median", "test_v1sep_median", "test_v1sep_opp_sign", "test_v2_median", "test_emuT_median", "test_emuT_res_coreRMS",
            "test_emuT_res_fullRMS", "test_emuT_opp_sign"]].copy()
show["sample"] = show["sample"].str.replace("barrel ", "").str.replace(" (matched)", "")
show.columns = ["disk", "ring", "fac", "sample", "N", "Neff", "medβ", "P(sY≥2)", "yMin", "v1uni", "v1sep", "v1sep_opp", "v2", "emuT", "emuT_core",
                "emuT_full", "emuT_opp"]
for chunk in range(0, len(show), 54):
    fig, a = plt.subplots(figsize=(17, 11)); a.axis("off")
    a.text(0, 1, "Per disc sensor (disk, ring, facing) vs matched barrel: medians [deg], core / full RMS of emitted − true [deg], opposite-sign fractions\n\n"
           + show.iloc[chunk:chunk + 54].round(3).to_string(index=False), va="top", family="monospace", fontsize=6.6)
    save(fig)
PDF.close()
print(show.round(3).to_string(index=False))
print(FL.round(3).to_string(index=False))
print("wrote", os.path.join(HERE, f"figs/disc_vs_barrel_beta_{TAG}.pdf"))
