"""PV-(x, y) d0 UPPER-BOUND study of the jet-tagging ladder (eval_refitq/tagger_sp).

Does measuring the track d0 relative to the TRUE primary-vertex transverse position (instead
of the detector origin, which is the beamspot to ~1 um in these samples) help L1 jet b-tagging?
If the truth does not help, no L1 estimate of the PV (x, y) can: this bounds a "PV xy readout"
of the E2E NNVtx before anyone builds one.

  pixi run python eval_refitq/tagger_sp/d0pv.py lut                 # PuppiCand dxy encoding table
  pixi run python eval_refitq/tagger_sp/d0pv.py side --mask AAAA    # per-constituent side cache
  pixi run python eval_refitq/tagger_sp/d0pv.py validate            # track-level validation
  train.py --d0 {A,B,C,D} ...                                       # the tagger variants

Variants (apply() below; everything else -- split, architecture, hyperparameters, matched WPs --
is the ladder's):
  A  the ladder as is (reference, rerun in this study's directory with the same seeding)
  B  every d0 the tagger sees is taken relative to the TRUE PV (GenVtx x, y) of the event:
       d0' = d0 - (x_PV sin(phi) - y_PV cos(phi)),   phi = the constituent TRACK's phi0
     (CMSSW L1 convention d0 = x sin(phi) - y cos(phi) = -dxy(offline); checked on prompt
     hard-scatter TPs: tp_d0 = GenVtx_x sin(phi0) - GenVtx_y cos(phi0) to 0.5 um). Every
     track of the event gets the hard-scatter PV correction -- right only for hard-scatter
     tracks (pile-up tracks come from other vertices). Slots that carry d0:
       trk_d0      (track group, L2/L3)       the SP (refit) track's float d0
       vtxpv_dxy   (vertexdxy_pv group)       identical to trk_d0 (no transverse PV in L1)
       dxy         (baseline group, all L)    the PF/PUPPI candidate's hardware dxy: unsigned,
                   sqrt-like code of the track word's hwD0 = floor(d0 / (1/256 cm)) (13 bit,
                   39 um LSB), dxy = hwDxy / 64 cm; re-derived from d0' through the encoding
                   table measured on the nanos (lut(); exact on every charged constituent)
     Unchanged: trk_sig_d0 (KF covariance), the track-MVA heads (d0 is withheld from them; the
     refit heads' prompt/displaced window is routed by |d0| vs 200 um measured from the ORIGIN,
     left as is -- the routing flips for a fraction of tracks reported by `validate`).
  C  A + the d0 significance trk_d0sig = d0 / sigma_hat(d0) (sigma_hat from the frozen L3
     track-MVA head trkmva_log_sd0 of the same scenario; 0 for constituents without a track).
     Added at L2 too, where it is the only MVA-derived input.
  D  B + C: the significance of the PV-corrected d0.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import awkward as ak
import numpy as np
import uproot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

OUTD = os.path.join(C.OUT, "d0_upper_bound")
D0_LSB_CM = 1.0 / 256        # track word d0 LSB: hwD0 = floor(d0 / LSB) (checked: exact)
DXY_LSB_CM = 1.0 / 64        # PF/PUPPI candidate: dxy = hwDxy * DXY_LSB (checked: exact)
HWD0_BITS = 13
LUT_PATH = os.path.join(OUTD, "puppi_dxy_lut.json")
CAND, JET, LINK = "L1ExtPuppiCand", "L1puppiJetSC4NG", "L1SC4NGJetCands"


def log(*a):
    import resource
    import time

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30
    print(time.strftime("%H:%M:%S"), f"[{rss:.2f} GB]", *a, flush=True)


def pv_proj(x, y, phi):
    """The d0 (L1 convention, d0 = -dxy) that a prompt track of azimuth phi has when it comes
    from a vertex at transverse position (x, y): the correction subtracted in B/D."""
    return x * np.sin(phi) - y * np.cos(phi)


def hwd0(d0_cm):
    """Signed track-word d0 integer (13-bit two's complement range)."""
    lim = 2 ** (HWD0_BITS - 1)
    return np.clip(np.floor(np.asarray(d0_cm, np.float64) / D0_LSB_CM), -lim, lim - 1).astype(np.int64)


# ---------------------------------------------------------------- PuppiCand dxy encoding
def all_files():
    out = []
    for m in C.MASKS:
        man = json.load(open(C.manifest_path(m)))
        out += [(m, s, n, p) for s, n, p in man["files"]]
    return out


def build_lut():
    """Signed hwD0 -> hwDxy of the PF/PUPPI candidates of every study file (charged
    constituents linked to an SP track); fails if the code is not a function of hwD0."""
    seen = {}
    for m, s, n, p in all_files():
        t = uproot.open(p)["Events"]
        A = t.arrays([f"{CAND}_hwDxy", f"{CAND}_l1TrackIdx", f"{C.TRACK_TABLE}_d0"])
        li = A[f"{CAND}_l1TrackIdx"]
        ok = li >= 0
        d0 = ak.to_numpy(ak.flatten(A[f"{C.TRACK_TABLE}_d0"][ak.where(ok, li, 0)][ok]))
        hw = ak.to_numpy(ak.flatten(A[f"{CAND}_hwDxy"][ok])).astype(np.int64)
        for k, v in set(zip(hwd0(d0).tolist(), hw.tolist())):
            if seen.setdefault(k, v) != v:
                raise RuntimeError(f"hwD0 {k} -> hwDxy {seen[k]} and {v}: not a function")
        log(f"lut {m} {s} f{n:02d}: {len(seen)} codes")
    ks = sorted(seen)
    os.makedirs(OUTD, exist_ok=True)
    with open(LUT_PATH, "w") as f:
        json.dump({"what": "PF/PUPPI candidate hwDxy as a function of the linked track's signed hwD0 "
                           "= floor(d0 / (1/256 cm)), measured on every study file; dxy = hwDxy / 64 cm",
                   "hwD0": ks, "hwDxy": [seen[k] for k in ks]}, f)
    log(f"lut: {len(ks)} codes, hwD0 {ks[0]}..{ks[-1]} -> {LUT_PATH}")


_LUT = None


def dxy_code(k):
    """(hwDxy, exact) of signed hwD0 values k. The measured code depends on |hwD0| only
    (checked when the table is loaded), so both signs are merged; an |hwD0| absent from the
    table (only beyond ~296, i.e. |d0| > 1.15 cm) takes the nearest present one (exact=False)."""
    global _LUT
    if _LUT is None:
        d = json.load(open(LUT_PATH))
        t = {}
        for kk, v in zip(d["hwD0"], d["hwDxy"]):
            if t.setdefault(abs(kk), v) != v:
                raise RuntimeError(f"PuppiCand dxy code is not a function of |hwD0| ({kk})")
        ks = np.array(sorted(t), np.int64)
        _LUT = (ks, np.array([t[x] for x in ks], np.int64))
    ks, vs = _LUT
    a = np.abs(k)
    i = np.clip(np.searchsorted(ks, a), 0, len(ks) - 1)
    j = np.clip(i - 1, 0, len(ks) - 1)
    pick = np.where(np.abs(ks[j] - a) < np.abs(ks[i] - a), j, i)
    return vs[pick], ks[pick] == a


# ---------------------------------------------------------------- side cache
def side_path(sample, mask, n):
    return os.path.join(C.cache_dir("d0pv"), C.tag(sample, mask, n) + ".npz")


def build_side(sample, mask, n, fn):
    """Per kept jet of the jet cache (same rows, same 16 slots): the constituent track's phi,
    a has-track flag and the event's GenVtx (x, y). Alignment is checked against the jet
    cache (pt, trk_d0 and the PuppiCand dxy re-derived from the track d0 must be identical)."""
    from ngtagger.data.labels import label_jets
    from ngtagger.data.nano import _gather2, crossref_gather, gather, group_constituents, read_events

    out = side_path(sample, mask, n)
    jc = os.path.join(C.cache_dir("jets"), C.tag(sample, mask, n) + ".npz")
    if os.path.exists(out) and os.path.getmtime(out) > os.path.getmtime(jc):
        return out
    ev = read_events([fn], track_table=C.TRACK_TABLE, extra_track_branches=["phi"])
    gv = uproot.open(fn)["Events"].arrays(["GenVtx_x", "GenVtx_y"], library="np")
    nested = group_constituents(ev, JET, LINK, n_const=C.N_CONST)
    cands, trks = ev[CAND], ev[C.TRACK_TABLE]
    li = _gather2(cands["l1TrackIdx"], nested)
    if not bool(ak.any(ak.flatten(cands["l1TrackIdx"]) >= 0)):
        raise RuntimeError(f"{fn}: {CAND}_l1TrackIdx unfilled (the loader's fallback would apply)")
    gen = {t: ev[t] for t in ("GenJet", "GenVisTau", "GenPart")}
    _, _, _, keep = label_jets(ev[JET], gen, max_dr=0.4)
    k = ak.to_numpy(ak.flatten(keep))

    def pad(x, dt):
        x = ak.fill_none(ak.pad_none(ak.flatten(x, axis=1), C.N_CONST, axis=1, clip=True), 0)
        return ak.to_numpy(x).astype(dt)[k]

    has = pad(li >= 0, bool)
    phi = pad(crossref_gather(cands, trks, nested, "l1TrackIdx", "phi"), np.float32)
    d0 = pad(crossref_gather(cands, trks, nested, "l1TrackIdx", "d0"), np.float32)
    pt = pad(gather(cands, nested, "pt"), np.float32)
    d = np.load(jc)
    X, names = d["X"], [str(x) for x in d["feature_names"]]
    if X.shape[:2] != pt.shape:
        raise RuntimeError(f"{fn}: {pt.shape} jets x slots vs jet cache {X.shape[:2]}")
    for nm, v in (("pt", pt), ("trk_d0", d0)):
        if not np.array_equal(X[..., names.index(nm)], v):
            raise RuntimeError(f"{fn}: {nm} differs from the jet cache (misaligned)")
    code, exact = dxy_code(hwd0(d0[has]))
    if not exact.all() or not np.array_equal(X[..., names.index("dxy")][has],
                                             (code * DXY_LSB_CM).astype(np.float32)):
        raise RuntimeError(f"{fn}: the PuppiCand dxy is not the table's code of the track d0")
    if np.any(X[..., names.index("dxy")][~has] != 0):
        raise RuntimeError(f"{fn}: a constituent without a track has dxy != 0")
    e = d["entry"]
    np.savez_compressed(out, trk_phi=phi, has_trk=has, pv_x=gv["GenVtx_x"][e].astype(np.float64),
                        pv_y=gv["GenVtx_y"][e].astype(np.float64), entry=e)
    log(f"side {C.tag(sample, mask, n)}: {len(e)} jets, {int(has.sum())} constituent tracks; aligned")
    return out


def load_side(mask, samples, D):
    """Side arrays in train.load_mask's row order (manifest files, study acceptance)."""
    man = json.load(open(C.manifest_path(mask)))
    parts = []
    for s in samples:
        for s2, n, _ in man["files"]:
            if s2 != s:
                continue
            j = np.load(os.path.join(C.cache_dir("jets"), C.tag(s, mask, n) + ".npz"))
            acc = (j["pt"] > C.PT_MIN) & (np.abs(j["eta"]) < C.ETA_MAX)
            S = np.load(side_path(s, mask, n))
            if not np.array_equal(S["entry"], j["entry"]):
                raise RuntimeError(f"{C.tag(s, mask, n)}: side cache out of date")
            parts.append({k: S[k][acc] for k in ("trk_phi", "has_trk", "pv_x", "pv_y", "entry")})
    S = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    if not np.array_equal(S["entry"], D["entry"]):
        raise RuntimeError(f"{mask}: side rows do not follow the training rows")
    return S


def extra_columns(mode):
    return ["trk_d0sig"] if mode in ("C", "D") else []


def apply(X, names, S, mode):
    """(X', names') for variant `mode` on the FULL cache tensor (all feature groups)."""
    names = list(names)
    if mode == "A":
        return X, names
    X = X.copy()
    has = S["has_trk"]
    d0 = X[..., names.index("trk_d0")].astype(np.float64)
    if mode in ("B", "D"):
        corr = np.where(has, pv_proj(S["pv_x"][:, None], S["pv_y"][:, None], S["trk_phi"].astype(np.float64)), 0.0)
        d0 = d0 - corr
        for nm in ("trk_d0", "vtxpv_dxy"):
            X[..., names.index(nm)] = np.where(has, d0, X[..., names.index(nm)]).astype(np.float32)
        code, _ = dxy_code(hwd0(d0[has]))
        X[..., names.index("dxy")][has] = (code * DXY_LSB_CM).astype(np.float32)
    if mode in ("C", "D"):
        sig = np.where(has, d0 * 1e4 / np.exp(X[..., names.index("trkmva_log_sd0")].astype(np.float64)), 0.0)
        X = np.concatenate([X, sig[..., None].astype(np.float32)], axis=-1)
        names.append("trk_d0sig")
    return X, names


# ---------------------------------------------------------------- track-level validation
VAL_BR_SP = ["d0", "phi", "eta", "spixRefitPerformed"]
VAL_BR_OT = ["d0", "phi", "eta", "tpFromHardInteraction", "tpIdx", "tp_vx", "tp_vy", "tp_d0", "tp_phi0"]
OT_D0_STEP_UM = 1e4 / 409.6   # the OT (L1TExtTrack) d0 sits on a 24.414 um grid, at half-integer steps
HIST = {"sp": np.arange(-300, 302.5, 2.5), "ot": np.arange(-123, 124) * OT_D0_STEP_UM}   # um
TRUNC = {"sp": 300.0, "ot": 3000.0}                                             # um


def s68(x):
    q = np.percentile(x, [15.865, 84.135])
    return float(0.5 * (q[1] - q[0]))


def shrink(a, b):
    """Signed quadrature difference sqrt|a^2 - b^2| (> 0: b narrower)."""
    return float(np.sqrt(abs(a * a - b * b)) * np.sign(a - b))


def stats(d0a, c, eta, trunc):
    """d0 (um) of A, B = A - c and the wrong sign A + c; c (um). slope_d0_on_corr: least-squares
    slope of d0 on c within |d0| < trunc (+1 if d0 carries the PV offset with this sign, 0 if
    the correction is unrelated to the track, e.g. pile-up); its error is statistical."""
    out = {"n": int(len(d0a)), "mean_corr_um": float(c.mean()), "rms_corr_um": float(c.std())}
    for nm, x in (("A", d0a), ("B", d0a - c), ("wrong_sign", d0a + c)):
        out[f"sigma68_{nm}_um"] = s68(x)
        out[f"median_{nm}_um"] = float(np.median(x))
    out["quadrature_shrink_sigma68_um"] = shrink(out["sigma68_A_um"], out["sigma68_B_um"])
    w = np.abs(d0a) < trunc
    cc, dd = c[w] - c[w].mean(), d0a[w] - d0a[w].mean()
    slope = float((cc * dd).sum() / (cc * cc).sum())
    out["slope_d0_on_corr"] = slope
    out["slope_err"] = float(np.std(dd - slope * cc) / np.sqrt((cc * cc).sum()))
    out["frac_in_slope_window"] = float(w.mean())
    by = {}
    for lo, hi in zip(C.ETA_BINS[:-1], C.ETA_BINS[1:]):
        s = (np.abs(eta) >= lo) & (np.abs(eta) < hi)
        if s.sum() > 50:
            a, b = s68(d0a[s]), s68(d0a[s] - c[s])
            by[f"{lo:g}-{hi:g}"] = {"n": int(s.sum()), "sigma68_A_um": a, "sigma68_B_um": b,
                                    "quadrature_shrink_um": shrink(a, b), "rms_corr_um": float(c[s].std())}
    out["by_abs_eta"] = by
    return out


def validate():
    """Per scenario: prompt hard-scatter tracks (OT-link TP from the hard interaction, produced
    within 1 um of GenVtx) -- d0 spread with and without the true-PV correction, for the SP
    (refit) tracks the tagger sees and for the OT tracks; pile-up tracks (TP not from the hard
    interaction) for the cost of applying the hard-scatter PV to every track."""
    res, hists = {}, {}
    for m in C.MASKS:
        man = json.load(open(C.manifest_path(m)))
        cols = {}
        for s, n, p in man["files"]:
            t = uproot.open(p)["Events"]
            keys = set(t.keys())
            br = [f"{C.TRACK_TABLE}_{b}" for b in VAL_BR_SP if f"{C.TRACK_TABLE}_{b}" in keys] \
                + [f"L1TExtTrack_{b}" for b in VAL_BR_OT] + ["GenVtx_x", "GenVtx_y"]
            A = t.arrays(br)
            ref = A["L1TExtTrack_d0"]
            row = {}
            for b in br:
                a = A[b]
                if b.startswith("GenVtx"):
                    a = ak.broadcast_arrays(a, ref)[0]
                row[b] = ak.to_numpy(ak.flatten(a)).astype(np.float64)
            for b, v in row.items():
                cols.setdefault(b, []).append(v)
        V = {b: np.concatenate(v) for b, v in cols.items()}
        gx, gy = V["GenVtx_x"], V["GenVtx_y"]
        matched = V["L1TExtTrack_tpIdx"] >= 0
        hs = matched & (V["L1TExtTrack_tpFromHardInteraction"] > 0)
        prompt = np.hypot(V["L1TExtTrack_tp_vx"] - gx, V["L1TExtTrack_tp_vy"] - gy) < 1e-4
        sel = {"prompt_hs": hs & prompt, "pileup": matched & ~hs}
        refit = V.get(f"{C.TRACK_TABLE}_spixRefitPerformed", np.zeros(len(gx))) > 0
        R = {"n_tracks": int(len(gx)), "n_matched": int(matched.sum()),
             "frac_hs_tracks_prompt": float(prompt[hs].mean())}
        # sign convention: the TP's own d0 vs the PV projection with the TP's phi0
        ph = sel["prompt_hs"]
        tpc = pv_proj(gx[ph], gy[ph], V["L1TExtTrack_tp_phi0"][ph])
        R["tp_d0_vs_pv_projection"] = {
            "rms_tp_d0_minus_proj_um": float(np.std(V["L1TExtTrack_tp_d0"][ph] - tpc) * 1e4),
            "rms_tp_d0_plus_proj_um": float(np.std(V["L1TExtTrack_tp_d0"][ph] + tpc) * 1e4),
            "rms_proj_um": float(np.std(tpc) * 1e4)}
        # IIII: the SP table is the OT passthrough (identical to "ot"), so only the OT tracks
        tables = (("ot", "L1TExtTrack"),) if m == C.BASELINE_MASK else (("sp", C.TRACK_TABLE), ("ot", "L1TExtTrack"))
        for key, pre in tables:
            d0 = V[f"{pre}_d0"] * 1e4
            c = pv_proj(gx, gy, V[f"{pre}_phi"]) * 1e4
            eta = V[f"{pre}_eta"]
            groups = {"ot": {"all": np.ones(len(gx), bool)}, "sp": {"refit": refit, "not_refit": ~refit}}[key]
            for gname, g in groups.items():
                for sname, s in sel.items():
                    x = s & g
                    if x.sum() < 50:
                        continue
                    lab = f"{key}:{gname}:{sname}"
                    R[lab] = stats(d0[x], c[x], eta[x], TRUNC[key])
                    hists[f"{m}|{lab}"] = np.stack([np.histogram(np.clip(v, HIST[key][0], HIST[key][-1]), HIST[key])[0]
                                                    for v in (d0[x], d0[x] - c[x], d0[x] + c[x])])
            if key == "sp":
                # refit heads' window routing (|d0| <= 200 um from the origin): flips under B
                a, b = np.abs(d0) <= 200, np.abs(d0 - c) <= 200
                R["sp:window_routing_flip_frac"] = float(np.mean(a != b))
        res[m] = R
        log(f"validate {m}: " + json.dumps({k: (v["sigma68_A_um"], v["sigma68_B_um"], v["rms_corr_um"])
                                            for k, v in R.items() if isinstance(v, dict) and "sigma68_A_um" in v}))
    os.makedirs(OUTD, exist_ok=True)
    with open(os.path.join(OUTD, "validation.json"), "w") as f:
        json.dump({"what": "d0 spread of prompt hard-scatter tracks with (B) / without (A) the true-PV "
                           "correction d0 - (x_PV sin phi - y_PV cos phi); wrong_sign = d0 + correction",
                   "prompt_definition": "OT-link TP from the hard interaction, produced < 1 um from GenVtx (x, y)",
                   "pileup_definition": "OT-link TP not from the hard interaction (correction = the HS PV)",
                   "scenarios": res}, f, indent=1)
    np.savez_compressed(os.path.join(OUTD, "validation_hists.npz"), edges_sp=HIST["sp"], edges_ot=HIST["ot"],
                        **{k.replace("|", "__").replace(":", "--"): v for k, v in hists.items()})
    log("validation written")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["lut", "side", "validate"])
    ap.add_argument("--mask", nargs="+", default=C.MASKS, choices=C.MASKS)
    a = ap.parse_args()
    if a.what == "lut":
        build_lut()
    elif a.what == "side":
        for m in a.mask:
            man = json.load(open(C.manifest_path(m)))
            for s, n, p in man["files"]:
                build_side(s, m, n, p)
    else:
        validate()


if __name__ == "__main__":
    main()
