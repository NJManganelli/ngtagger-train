"""Shared helpers for the v2 cluster-angle study (orientation/position inputs, signed heads, hybrid, SmartPixels comparison).

All large outputs go to /Volumes/WDMac/smartpixels_scratch/cluster_cnn_v2/ (models/, preds/, arrays/).
Data: data_all/ (barrel + disc images and meta; barrel rows identical to data/, same event split).

Frames (CMSSW module-local, from the ntuple axes):
  barrel: local y = -global z for every module; local z = +r_hat (zOutward = +1) or -r_hat, local x = z cross y flips with it.
  disc:   local y = -r_hat (radially inward), local x = +-phi_hat, local z = +-z_hat (axZz).
Prompt-pointing ("geometric") cot: d = hit - (0,0,0); cotA_geo = (d.axX)/(d.axZ), cotB_geo = (d.axY)/(d.axZ).
"""
import os, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D_ALL = os.path.join(HERE, "data_all")
V2 = "/Volumes/WDMac/smartpixels_scratch/cluster_cnn_v2"
C_LAYER = np.array([0, 0.0, 0.202, 0.202, 0.202], np.float32)
deg = lambda c: np.degrees(np.arctan(c))
smad = lambda x: float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) > 5 else np.nan

GEO_RAW = ["globalX", "globalY", "globalZ", "globalR", "globalPhi", "modX", "modY", "modZ", "localX", "localY", "zOutward",
           "axXx", "axXy", "axXz", "axYx", "axYy", "axYz", "axZx", "axZy", "axZz", "side"]
TRUTH = ["linkClass", "tpPt", "tpCharge", "tpLocalCotAlpha", "tpLocalCotBeta", "hxOk", "hxDist", "tpChargeFrac",
         "tpVx", "tpVy", "tpVz", "shN", "shCotAlpha", "shCotBeta", "shPdg", "shPabs", "domSimHasTp", "domSimPt"]
SHAPE = ["subdet", "layer", "disk", "ring", "sizeX", "sizeY", "size", "charge", "adcSum", "truncX", "truncY", "nPxLost",
         "row", "split", "event", "clIdx", "detId", "pCotAlpha", "pCotBeta", "pHasAlpha", "pHasBeta", "tx_hx", "ty_hx"]


def good(m):
    """truth-consistent labels: TP-linked, charged, pT >= 0.25, helix within 1 cm of the hit"""
    return ((m.linkClass == 2) & (m.tpPt >= 0.25) & (m.tpCharge != 0) & (m.tpLocalCotAlpha > -900)
            & (m.hxOk == 1) & (m.hxDist < 1.0)).values


def derived(m):
    """Rotation-invariant orientation + prompt-pointing features (DataFrame aligned to m)."""
    r = np.hypot(m.modX, m.modY).values
    rx, ry = m.modX.values / r, m.modY.values / r            # module-centre radial unit vector
    fx, fy = -ry, rx                                          # phi_hat
    o = pd.DataFrame(index=m.index)
    o["axZr"] = m.axZx * rx + m.axZy * ry                     # barrel: = zOutward
    o["axXphi"] = m.axXx * fx + m.axXy * fy
    o["axXr"] = m.axXx * rx + m.axXy * ry
    o["axYr"] = m.axYx * rx + m.axYy * ry
    bar = (m.subdet == 1).values
    o["sideZ"] = np.where(bar, np.sign(m.globalZ.values), np.where(m.side.values == 1, -1.0, 1.0))  # disc side 1 = -z
    dx = m.globalX.values; dy = m.globalY.values; dz = m.globalZ.values
    lx = dx * m.axXx + dy * m.axXy + dz * m.axXz; ly = dx * m.axYx + dy * m.axYy + dz * m.axYz
    lz = dx * m.axZx + dy * m.axZy + dz * m.axZz
    o["cotA_geo"] = (lx / lz).clip(-20, 20); o["cotB_geo"] = (ly / lz).clip(-20, 20)
    dn = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    o["pu"], o["pv"], o["pw"] = lx / dn, ly / dn, lz / dn    # unit vector origin->hit in the module frame (u, v, w)
    o["sinPhi"] = np.sin(m.globalPhi); o["cosPhi"] = np.cos(m.globalPhi)
    return o.astype(np.float32)


def load(filters, extra=()):
    cols = sorted(set(GEO_RAW + TRUTH + SHAPE + list(extra)))
    m = pd.read_parquet(os.path.join(D_ALL, "meta.parquet"), columns=cols, filters=filters).reset_index(drop=True)
    return pd.concat([m, derived(m)], axis=1)


# ---------------- metrics ----------------
def metrics(pc, tc):
    ok = np.isfinite(pc) & np.isfinite(tc); pc, tc = pc[ok], tc[ok]
    if len(pc) < 6:
        return dict(n=int(len(pc)))
    dd = deg(pc) - deg(tc); dc = pc - tc
    big = np.abs(deg(tc)) > 10
    return dict(n=int(len(pc)), smad_deg=smad(dd), rms_deg=float(np.sqrt(np.mean(dd ** 2))), bias_deg=float(np.median(dd)),
                smad_cot=smad(dc), rms_cot=float(np.sqrt(np.mean(dc ** 2))), bias_cot=float(np.median(dc)),
                tail5=float(np.mean(np.abs(dd) > 5)), tail10=float(np.mean(np.abs(dd) > 10)),
                sign_acc=float(np.mean(np.sign(pc) == np.sign(tc))),
                opp_sign_true_gt10=float(np.mean(np.sign(pc[big]) != np.sign(tc[big]))) if big.any() else np.nan,
                mag_smad_deg=smad(np.abs(deg(pc)) - np.abs(deg(tc))))


EDGES = {"B": np.arange(-90, 91, 10.0), "A": np.arange(-45, 46, 5.0)}


def confusion(pc, tc, ang):
    """18-bin row-normalised true(row) x pred(col) matrix in degrees, ends = overflow (angle_confusion_surfaces.py)."""
    e = EDGES[ang]; lo, hi = e[0] + 1e-6, e[-1] - 1e-6
    ok = np.isfinite(pc) & np.isfinite(tc)
    H, _, _ = np.histogram2d(np.clip(deg(tc[ok]), lo, hi), np.clip(deg(pc[ok]), lo, hi), bins=[e, e])
    return H / H.sum(1, keepdims=True).clip(min=1), H


def surfaces(m):
    s = [(f"IL{k}", ((m.subdet == 1) & (m.layer == k)).values) for k in (1, 2, 3, 4)]
    s += [(f"ID{k}", ((m.subdet == 2) & (m.disk == k)).values) for k in (1, 2, 3, 4)]
    return s


def core_fit(r, w=None):
    """Robust Gaussian-core estimate of residuals r (deg), immune to mirror side-peaks: seed at the MODE of a smoothed
    histogram with sigma0 = HWHM/1.1774, then iterate mean/std over |r - mu| < 2 sigma with the truncated-Gaussian correction
    (std of N(0,1) truncated at +-2 = 0.8796). Returns (core_rms_about_0 = sqrt(mu^2 + sigma^2), mu, sigma, tail = weighted
    fraction with |r - mu| > 3 sigma). Optional weights w."""
    r = np.asarray(r, float); w = np.ones(len(r)) if w is None else np.asarray(w, float)
    ok = np.isfinite(r) & (w > 0); r, w = r[ok], w[ok]
    if len(r) < 30:
        return np.nan, np.nan, np.nan, np.nan
    lo, hi = np.percentile(r, [0.5, 99.5]); hi = max(hi, lo + 1e-3)
    nb = 400; h, e = np.histogram(r, bins=nb, range=(lo, hi), weights=w)
    hs = np.convolve(h, np.ones(5) / 5, "same"); j = int(np.argmax(hs)); c = 0.5 * (e[1:] + e[:-1])
    half = hs[j] / 2; jl, jr = j, j
    while jl > 0 and hs[jl] > half:
        jl -= 1
    while jr < nb - 1 and hs[jr] > half:
        jr += 1
    mu = c[j]; sg = max((c[jr] - c[jl]) / 2 / 1.1774, (e[1] - e[0]))
    for _ in range(100):
        k = np.abs(r - mu) < 2 * sg
        if w[k].sum() <= 0:
            break
        mu2 = np.average(r[k], weights=w[k]); sg2 = max(np.sqrt(np.average((r[k] - mu2) ** 2, weights=w[k])) / 0.8796, 1e-4)
        if abs(mu2 - mu) < 1e-4 * sg and abs(sg2 - sg) < 1e-4 * sg:
            mu, sg = mu2, sg2; break
        mu, sg = mu2, sg2
    return float(np.hypot(mu, sg)), float(mu), float(sg), float(np.average(np.abs(r - mu) > 3 * sg, weights=w))
