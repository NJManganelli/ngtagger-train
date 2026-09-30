"""Shared pieces of the E2E NNVtx study on SmartPixels refit tracks (canon_v1 AAAA, PU200 ttbar).

Same data, variants, truth and FastHisto reference as the sub-2 GeV study
(eval_refitq/sub2gev; its offline float GTT fastHisto, selections, TVA windows and
core-RMS convention are imported from there, not copied), now with the GTT
end-to-end NN vertexing chain (ngtagger.train.nnvtx: per-track weight MLP ->
differentiable soft z0 histogram -> Conv1D pattern net -> PV z; per-track
association MLP on |z0 - z_PV| + features) trained without / with the per-track
resolution-quality MVA scores (ngtagger.train.trackmva heads, retrained here).

Variants (as in sub2gev):
  V0  OT tracks (L1TTrack), standard GTT prompt selection (pT >= 2 GeV, ...)
  V1  SmartPixels tracks (L1TSmartPixelsTrack), standard selection  [diagnostic only]
  V3  SmartPixels tracks, standard selection except refitted tracks have no pT floor
Truth: PV = GenVtx_z. A track is hard-scatter (HS) when its TP has (evt, bx) = (0, 0):
V0 via L1TTrack_tpIdx (OT loosely-genuine associator; identical to
L1TTrack_tpFromHardInteraction, checked), V1/V3 via L1TSmartPixelsTrack_tpIdx (weighted
cluster-majority). tpIdx < 0 (unmatched, or pruned |p| <= 1 GeV TP) = not HS.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))


def _load_sub2gev_common():
    import importlib.util
    spec = importlib.util.spec_from_file_location("sub2gev_common", os.path.join(HERE, "..", "sub2gev", "common.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s2 = _load_sub2gev_common()  # eval_refitq/sub2gev/common.py

CANON = s2.CANON
FILES_ALL = s2.FILES_ALL
FILES = [CANON.format(f) for f in FILES_ALL]
TRAIN_ONLY = ["_f01_", "_f02_"]  # in-sample for the cluster CNN: training only, never evaluated
OUTDIR = "/Volumes/WDMac/smartpixels_scratch/nnvtx_sp"
DATASET = OUTDIR + "/dataset.npz"
MVA_DIR = OUTDIR + "/trackmva_canon"          # train-trackmva output (oof scores, report, frozen bundle)
MVA_STEM = MVA_DIR + "/trackmva_canonical_AAAA"
PRED = OUTDIR + "/pred_{variant}.npz"
N_FOLDS = 5
FOLD_SEED = 0
VARIANTS = ("V0", "V1", "V3")
VLABEL = {"V0": "V0 OT tracks", "V1": "V1 SP tracks, 2 GeV floor (diagnostic)",
          "V3": "V3 SP tracks, sub-2 GeV refit tracks admitted"}
MVA_COLL = {"V0": "base", "V1": "refit", "V3": "refit"}  # OT tracks: OT heads; SP tracks: SP heads

PV_WINDOW = s2.PV_WINDOW
core_rms, full_rms, fasthisto, tva_dzmax = s2.core_rms, s2.full_rms, s2.fasthisto, s2.tva_dzmax


def load_dataset(path=DATASET):
    d = np.load(path, allow_pickle=False)
    T = {k[2:]: d[k] for k in d.files if k.startswith("T_")}
    E = {k[2:]: d[k] for k in d.files if k.startswith("E_")}
    return T, E


def selections(T):
    """{variant: (selected mask, prefix of pt/eta/z0 columns, HS truth mask)}; the
    sub2gev selections on this dataset's columns (SP rows carry the OT chi2s/nStubs)."""
    ref = T["r_spixRefitPerformed"] != 0
    com = (T["b_nStubs"], T["b_chi2BendRed"], T["b_chi2ZRed"], T["b_chi2XYRed"])
    s0 = s2.standard_selection(T["b_pt"], T["b_eta"], T["b_z0"], *com)
    s1 = s2.standard_selection(T["r_pt"], T["r_eta"], T["r_z0"], *com)
    s3 = s2.standard_selection(T["r_pt"], T["r_eta"], T["r_z0"], *com, pt_min=np.where(ref, 0.0, 2.0))
    return {"V0": (s0, "b_", T["ot_hs"]), "V1": (s1, "r_", T["sp_hs"]), "V3": (s3, "r_", T["sp_hs"])}


def event_folds(E):
    """The train-trackmva fold assignment (same files, order, seed): -1 = training only."""
    from ngtagger.train.trackmva import event_folds as ef, train_only_mask
    return ef(E["file"], train_only_mask(FILES, TRAIN_ONLY), N_FOLDS, FOLD_SEED)
