"""Shared pieces of the sub-2 GeV SmartPixels refit study (canon_v1 AAAA, PU200 ttbar).

Inputs (canonical L1TSPStudyNano, one row of L1TSmartPixelsTrack per L1TTrack row,
L1TSmartPixelsTrack_l1TrackIdx -> L1TTrack):
  canon_v1/spstudy_AAAA_fNN_100ev.root   GTT fed by the SmartPixels (refit) tracks
  coexist_v1/coexist15_fNN_100ev.root    same events, GTT fed by the OT tracks (L1TTrack)
so L1Vertex of the canonical file is the CMSSW vertex on SmartPixels tracks with the
stock 2 GeV floor, and L1Vertex of the coexist file is the CMSSW OT-only vertex.

The PV finder here is an OFFLINE FLOAT version of the GTT fastHistoEmulation
(L1Trigger/VertexFinder VertexFinder::fastHistoEmulation, l1tVertexProducer_cfi):
256 z bins of 0.15991504 cm over [-20.46912512, 20.46912512] cm, pT-weighted, sliding
window of 3 bins, highest window sum wins, z = window bin centres weighted by the bin
pT sums. The float version skips the word quantisation (z0 word, pT word, integer
truncation of bin sums, 1/x LUT); `quantised=True` re-applies the dominant pieces for the
closure diagnostic only.
"""
from __future__ import annotations

import numpy as np

CANON = "/Volumes/WDMac/smartpixels_scratch/canon_v1/spstudy_AAAA_f{:02d}_100ev.root"
COEXIST = "/Volumes/WDMac/smartpixels_scratch/coexist_v1/coexist15_f{:02d}_100ev.root"
OUTDIR = "/Volumes/WDMac/smartpixels_scratch/sub2gev"
CACHE = OUTDIR + "/cache/tracks_f{:02d}.npz"
FILES_ALL = list(range(1, 11))
FILES_CLEAN = list(range(3, 11))  # f01/f02 are in-sample for the cluster CNN

# GTT fastHisto (l1tVertexProducer_cfi)
FH_MIN, FH_MAX, FH_W = -20.46912512, 20.46912512, 0.15991504
FH_NBINS = 256
FH_WINDOW = 3
Z0_LSB = 2 * 20.46912512 / 4096  # TTTrack_TrackWord::stepZ0 (12-bit z0 word)

# l1tTrackSelectionProducer cutSet (prompt, standard)
SEL = dict(ptMin=2.0, absEtaMax=2.4, absZ0Max=15.0, nStubsMin=4,
           bendChi2Max=2.25, chi2RZMax=5.0, chi2RPhiMax=20.0)

# l1tTrackVertexAssociationProducer cutSet
TVA_ETA = np.array([0.0, 0.7, 1.0, 1.2, 1.6, 2.0, 2.4])
TVA_DZ = np.array([0.37, 0.5, 0.6, 0.75, 1.0, 1.6])

PV_WINDOW = 0.15  # cm: PV "found" if |z_PV - z_gen| < FH_VertexWidth (0.15 cm)

RINV_SAT = (0x3FFF, 0x4000)  # saturated 15-bit 1/R word (+max, -max); 0x4000 = the OOB-LUT case


def core_rms(r, nsig=3.0, maxit=50):
    """Project convention: iterative 3-sigma clip about the mean, RMS about ZERO of the
    kept core (bias counts); returns (core_rms, fraction outside the core)."""
    r = np.asarray(r, float)
    if r.size < 3:
        return np.nan, np.nan
    k = np.ones(r.size, bool)
    for _ in range(maxit):
        mu, sd = r[k].mean(), r[k].std()
        k2 = np.abs(r - mu) < nsig * sd
        if (k2 == k).all() or k2.sum() < 3:
            break
        k = k2
    return float(np.sqrt(np.mean(r[k] ** 2))), float(1 - k.mean())


def full_rms(r):
    r = np.asarray(r, float)
    return float(np.sqrt(np.mean(r ** 2))) if r.size else np.nan


def fasthisto(z0, pt, quantised=False):
    """Float GTT fastHisto on one event's selected tracks -> (z_pv, window_sum, b_max).

    quantised=True (closure diagnostic): z0 rounded to the 12-bit word LSB, pT truncated to
    the 1/32 GeV GTT pT-word LSB, bin sums truncated to integers and saturated at 511
    (link_pt_sum_fixed_t), as in fastHistoEmulation."""
    z0 = np.asarray(z0, float)
    pt = np.asarray(pt, float)
    if quantised:
        z0 = np.floor(z0 / Z0_LSB) * Z0_LSB
        pt = np.floor(pt * 32.0) / 32.0
    idx = np.floor((z0 - FH_MIN) / FH_W).astype(int)
    ok = (idx >= 0) & (idx < FH_NBINS)
    h = np.bincount(idx[ok], weights=pt[ok], minlength=FH_NBINS)
    if quantised:
        h = np.minimum(np.floor(h), 511.0)
    s = h[:-2] + h[1:-1] + h[2:]
    if not np.any(s > 0):
        return np.nan, 0.0, -1
    b = int(np.argmax(s))  # first maximum, as the strict '>' scan in CMSSW
    wb = h[b:b + FH_WINDOW]
    frac = float(np.dot(wb, np.arange(FH_WINDOW)) / wb.sum())
    return FH_MIN + (b + 0.5 + frac) * FH_W, float(s[b]), b


def standard_selection(pt, eta, z0, nstub, chi2bend, chi2rz, chi2rphi, pt_min=SEL["ptMin"]):
    """l1tTrackSelectionProducer prompt cuts on float quantities (pt_min may be an array)."""
    return ((pt >= pt_min) & (np.abs(eta) <= SEL["absEtaMax"]) & (np.abs(z0) <= SEL["absZ0Max"])
            & (nstub >= SEL["nStubsMin"]) & (chi2bend < SEL["bendChi2Max"])
            & (chi2rz < SEL["chi2RZMax"]) & (chi2rphi < SEL["chi2RPhiMax"]))


def tva_dzmax(eta):
    """eta-binned association window of l1tTrackVertexAssociationProducer (|eta| > 2.4 -> 0)."""
    i = np.searchsorted(TVA_ETA, np.abs(eta), side="right") - 1
    out = np.zeros(np.shape(eta))
    ok = (i >= 0) & (i < TVA_DZ.size)
    out[ok] = TVA_DZ[i[ok]]
    return out


def load_cache(files):
    """Concatenate per-file caches; trk['evidx'] becomes the global event row."""
    trk, evt = {}, {}
    off = 0
    for f in files:
        d = np.load(CACHE.format(f))
        for k in d.files:
            v = d[k] + off if k == "evidx" else d[k]
            (evt if k.startswith("ev_") else trk).setdefault(k, []).append(v)
        off += d["ev_event"].size
    trk = {k: np.concatenate(v) for k, v in trk.items()}
    evt = {k: np.concatenate(v) for k, v in evt.items()}
    return trk, evt


def bootstrap(stat_fn, n_events, n_boot=1000, seed=12345):
    """Event bootstrap: stat_fn(idx) -> dict of floats; returns dict of (value, sigma)."""
    rng = np.random.default_rng(seed)
    central = stat_fn(np.arange(n_events))
    reps = {k: [] for k in central}
    for _ in range(n_boot):
        s = stat_fn(rng.integers(0, n_events, n_events))
        for k in central:
            reps[k].append(s[k])
    return {k: (central[k], float(np.nanstd(reps[k], ddof=1))) for k in central}
