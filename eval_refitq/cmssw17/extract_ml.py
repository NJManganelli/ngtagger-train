"""The ML part of the CMSSW_17 comparison's extraction (plan 3.4.1, 3.4.3): the deployed per-track heads and the
tagger's constituent inputs, computed with the tagger's own code from the canonical nano files of both sides.

    python eval_refitq/cmssw17/extract_ml.py --side d121 --mask AAAA --file 1 2 3 ...

Per (side, mask, file), one npz under $OUT/extract_ml/<side>/<mask>_fNN.npz, merged into the base records by
engine.Side:
- ev_heldout: per event. D121: the head test split (tagger_sp.common.split_table over TTbar files 1-10, the
  heads' training manifest); every other D121 event trained the heads (files 1-2 also trained the cluster CNN).
  CMSSW_17: every event (another sample).
- hd_*: per track of the head collection (OT heads on IIII, SmartPixels refit heads on the SmartPixels masks): the
  six margins (clean, then log sigma_hat of d0, z0, tanL, 1/R, phi in the head units), the window, the OT track's
  pT and eta, and the evaluation truth: matched (the OT track's particle, with the 1 GeV rule of plan 3.4.2),
  clean, and the five residuals in the head units.
- jt_*: per L1puppiJetSC4NG jet: the matched GenJet (dR < 0.4) pT, eta and flavour code, the filled constituents.
- cn_*: per filled constituent (the 16 leading per jet): the deployed tagger's inputs (tagger_sp.common.VARIANTS
  ["joint_ref"]: baseline, track, trkquality, vertexdxy_pv, trackres, trackjoint), with its jet's GenJet cell.

The heads are the deployed reference (lambda 0) boosters of heads_cmssw/work (export_heads.py train); heads parity
shows they equal CMSSW's head columns.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("cmssw17_common", os.path.join(HERE, "common.py"))
CM = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CM)

# the tagger's modules import their own `common` (tagger_sp/common.py): this directory must not shadow it
os.environ["TAGGER_SP_OUT"] = CM.HEADS_WORK          # the deployed heads' fits (joint_heads.jdir)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != HERE]
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tagger_sp"))
import common as TC  # noqa: E402  (tagger_sp/common.py)
import joint_heads as J  # noqa: E402

TAGGER_VARIANT = "joint_ref"
HEAD_SETTING = J.sname("ref", 0.0)


def flat(x, dtype=np.float64):
    import awkward as ak
    return ak.to_numpy(ak.flatten(x)).astype(dtype)


def wrap(dphi):
    return (dphi + np.pi) % (2 * np.pi) - np.pi


def flavour_code(hf, pf, nb):
    """As extract.flavour_code: 0 b with one B hadron, 1 b with two or more, 2 c, 3 light quark, 4 gluon, -1 other."""
    flav = np.full(len(hf), -1, np.int8)
    flav[(hf == 5) & (nb == 1)] = 0
    flav[(hf == 5) & (nb >= 2)] = 1
    flav[hf == 4] = 2
    flav[(hf == 0) & (pf >= 1) & (pf <= 3)] = 3
    flav[(hf == 0) & (pf == 21)] = 4
    return flav


def event_ids(fn):
    import uproot
    with uproot.open(fn) as f:
        return np.column_stack([f["Events"][c].array(library="np").astype(np.int64)
                                for c in ("run", "luminosityBlock", "event")])


def heldout_d121(mask, n):
    """The head test split of D121 file n (entry order), over the heads' training files (TTbar 1-10)."""
    man = json.load(open(TC.manifest_path(mask)))
    files = sorted(int(k) for s, k, _ in man["files"] if s == "ttbar")
    if files != list(CM.D121_FILES):
        raise RuntimeError(f"the heads' TTbar files {files} are not the comparison's {CM.D121_FILES}")
    ev = {k: event_ids(str(CM.d121_path(mask, k))) for k in files}
    return TC.split_table("ttbar", ev)[n]


def mva_ref(mask):
    """The deployed reference heads of a mask as the tagger's trackjoint model (joint_heads.joint_mva)."""
    J.chosen = lambda: {"ref": ("ref", 0.0)}
    return J.joint_mva(mask, "ref")


def heads_part(fn, mask, R):
    import uproot
    import xgboost as xgb

    from ngtagger.train import trackmva as tm

    coll = J.coll_of(mask)
    base_only = coll == "base"
    t = uproot.open(fn)["Events"]
    tm.refuse_truth_inputs(t, TC.FAMILY, "", base_only=base_only)
    br = tm.nano_branches(t.keys(), TC.FAMILY, "", with_truth=True, with_event_id=True, base_only=base_only)
    extra = [f"L1TExtTrack_{c}" for c in J.EXTRA_BASE]
    if not base_only:
        extra += [f"{TC.TRACK_TABLE}_{c}" for c in J.EXTRA_REFIT]
    A = t.arrays(list(dict.fromkeys(br + extra)))
    T, ntrk = tm.tracks_from_batch(A, TC.FAMILY, "", with_truth=True, base_only=base_only)
    for c in J.EXTRA_BASE:
        T["b_" + c] = flat(A[f"L1TExtTrack_{c}"])
    if not base_only:
        for c in J.EXTRA_REFIT:
            T["r_" + c] = flat(A[f"{TC.TRACK_TABLE}_{c}"])
    X = J.features(T, coll)
    F = np.full((len(X), len(J.head_list())), np.nan)
    win = np.full(len(X), -1, np.int8)
    R_res = np.full((len(X), len(J.PARAMS)), np.nan)
    clean = np.zeros(len(X), bool)
    for wi, (wname, w) in enumerate(tm.windows(T, coll).items()):
        d = J.jdir("fits", mask, wname)
        b = xgb.Booster()
        b.load_model(os.path.join(d, HEAD_SETTING + ".json"))
        meta = json.load(open(os.path.join(d, "meta.json")))
        idx = np.flatnonzero(w)
        if len(idx):
            F[idx] = b.predict(xgb.DMatrix(X[idx]), output_margin=True)
        win[idx] = wi
        y, Rw, _ = J.targets(T, coll, k_bend=meta["k_bend"])
        R_res[idx] = Rw[idx]
        clean[idx] = y[idx] > 0
    # the 1 GeV rule (plan 3.4.2): a match to a particle below 1 GeV is no match
    matched = np.nan_to_num(T["b_tp_pt"], nan=-1.0) >= CM.TP_PT_MIN
    R_res[~matched] = np.nan
    R["hd_ev"] = np.repeat(np.arange(len(ntrk)), ntrk).astype(np.int32)
    R["hd_pt"] = T["b_pt"].astype(np.float32)
    R["hd_eta"] = T["b_eta"].astype(np.float32)
    R["hd_win"] = win
    for j in range(F.shape[1]):
        R[f"hd_m{j}"] = F[:, j].astype(np.float32)
    R["hd_matched"] = matched
    R["hd_clean"] = clean
    for j in range(R_res.shape[1]):
        R[f"hd_r{j}"] = R_res[:, j].astype(np.float32)
    R["meta_hd_coll"] = np.array(coll)
    R["meta_hd_heads"] = np.array([h[0] for h in J.head_list()])
    R["meta_hd_setting"] = np.array(HEAD_SETTING)


def jets_part(fn, mask, R, nev):
    import awkward as ak
    import uproot

    from ngtagger.data.features import build_features
    from ngtagger.data.nano import load_jets

    groups = TC.VARIANTS[TAGGER_VARIANT]
    jets, const, _ = load_jets([fn], n_const=TC.N_CONST, feature_groups=groups, track_table=TC.TRACK_TABLE,
                               trackjoint_model=mva_ref(mask))
    X, names = build_features(jets, const, n_const=TC.N_CONST, feature_groups=groups)
    nj = ak.to_numpy(ak.num(jets.pt, axis=1))
    if len(nj) != nev:
        raise RuntimeError(f"{fn}: load_jets gave {len(nj)} events, the file has {nev}")
    jev = np.repeat(np.arange(nev), nj)
    # GenJet match (dR < 0.4, closest) and flavour, as extract.jets_with_genjet
    with uproot.open(fn) as f:
        G = f["Events"].arrays([f"GenJet_{c}" for c in ("pt", "eta", "phi", "hadronFlavour", "partonFlavour",
                                                         "nBHadrons")])
    pairs = ak.cartesian({"j": ak.zip({"eta": jets.eta, "phi": jets.phi}),
                          "g": ak.zip({"eta": G["GenJet_eta"], "phi": G["GenJet_phi"]})}, nested=True)
    dr = np.hypot(pairs.j.eta - pairs.g.eta, wrap(pairs.j.phi - pairs.g.phi))
    imin = ak.argmin(dr, axis=2, keepdims=True)
    ok = flat(ak.fill_none(ak.flatten(dr[imin], axis=2) < 0.4, False), bool)
    gi = flat(ak.fill_none(ak.flatten(imin, axis=2), -1), np.int64)
    ng = ak.to_numpy(ak.num(G["GenJet_pt"]))
    goff = np.concatenate([[0], np.cumsum(ng)[:-1]]).astype(np.int64)
    gg = np.where(ok & (gi >= 0), goff[jev] + gi, -1)
    pick = np.maximum(gg, 0)
    Gf = {c: flat(G[f"GenJet_{c}"]) for c in ("pt", "eta", "hadronFlavour", "partonFlavour", "nBHadrons")}
    gpt = np.where(gg >= 0, Gf["pt"][pick] if len(Gf["pt"]) else np.nan, np.nan)
    geta = np.where(gg >= 0, Gf["eta"][pick] if len(Gf["eta"]) else np.nan, np.nan)
    gfl = np.where(gg >= 0, flavour_code(Gf["hadronFlavour"][pick], np.abs(Gf["partonFlavour"][pick]),
                                         Gf["nBHadrons"][pick]) if len(Gf["pt"]) else -1, -1)
    ifl = names.index("isfilled")
    filled = X[:, :, ifl] > 0
    R["jt_ev"] = jev.astype(np.int32)
    R["jt_pt"] = flat(jets.pt, np.float32)
    R["jt_eta"] = flat(jets.eta, np.float32)
    R["jt_gen_pt"] = gpt.astype(np.float32)
    R["jt_gen_eta"] = geta.astype(np.float32)
    R["jt_flav"] = gfl.astype(np.int8)
    R["jt_nconst"] = filled.sum(1).astype(np.int16)
    jj, kk = np.nonzero(filled)
    R["cn_ev"] = jev[jj].astype(np.int32)
    R["cn_gen_pt"] = gpt[jj].astype(np.float32)
    R["cn_gen_eta"] = geta[jj].astype(np.float32)
    R["cn_flav"] = gfl[jj].astype(np.int8)
    for i, nm in enumerate(names):
        if nm != "isfilled":
            R["cn_" + nm] = X[jj, kk, i].astype(np.float32)
    R["meta_cn_features"] = np.array([nm for nm in names if nm != "isfilled"])
    R["meta_tagger_variant"] = np.array(TAGGER_VARIANT)


def extract_ml(side, mask, n, out):
    fn = str(CM.side_path(side, mask, n))
    t0 = time.time()
    ev = event_ids(fn)
    R = {"meta_side": side, "meta_mask": mask, "meta_file": n, "meta_path": fn,
         "ev_run": ev[:, 0], "ev_lumi": ev[:, 1], "ev_event": ev[:, 2]}
    R["ev_heldout"] = heldout_d121(mask, n) if side == "d121" else np.ones(len(ev), bool)
    heads_part(fn, mask, R)
    jets_part(fn, mask, R, len(ev))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, **R)
    print(f"{side} {mask} f{n:02d}: {len(ev)} events ({int(R['ev_heldout'].sum())} held out), "
          f"{len(R['hd_ev'])} tracks, {len(R['jt_ev'])} jets, {len(R['cn_ev'])} constituents, "
          f"{time.time() - t0:.0f} s -> {out}", flush=True)


def out_path(side, mask, n):
    return os.path.join(str(CM.OUT / "extract_ml"), side, f"{mask}_f{n:02d}.npz")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", required=True, choices=("d121", "c17"))
    ap.add_argument("--mask", required=True, choices=CM.MASKS)
    ap.add_argument("--file", type=int, nargs="+", required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for n in a.file:
        o = out_path(a.side, a.mask, n)
        if os.path.exists(o) and not a.force:
            print("exists:", o)
            continue
        extract_ml(a.side, a.mask, n, o)


if __name__ == "__main__":
    main()
