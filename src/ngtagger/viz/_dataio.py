"""Nano I/O + track curation for the digiRefit refit-replay visualizer.

Re-anchored to the 5-par PROMPT framing (mem:smartpixels-5par-framing-directive):

  SEED  = 5-par OT-only PROMPT track = the reference ``L1TTrack`` table
          (promptHnpar=5: real d0/z0/phi/rInv/tanL + reduced fit chi2
          ``chi2XYRed``/``chi2ZRed`` + matched-TP ``tpPt``/``tpEta``/``genuine``).
  REFIT = 5-par OT+IT = the PROMPT digiRefit variant tracks
          ``L1TSmartPixelsTrackDigiRefit{AIII,AAII,AAAI,AAAA}`` (NOT the Ext /
          displaced variants) + their per-hit sidecars
          ``L1TSmartPixelsRefitHitDigiRefit{...}``.
  OT stubs = the REAL embedded ``L1TTrackStub`` table (x/y/z, r, layer, isBarrel),
          linked per track by ``trackIdx`` -- replaces the old schematic hitPattern
          placement.
  TRUTH = ``GenVtx_{x,y,z}`` (interim, prompt-only) + ``tpPt``/``tpEta``; see
          :mod:`ngtagger.viz._truth`.

The comparison is: does adding the inner-tracker (IT) hits improve the 5-par prompt
track for b-tagging (impact parameter), vertexing, etc. The 4-par pinned-d0 path is
DROPPED entirely.

Pure uproot/numpy. The real refit-track values are loaded ONLY for the validation
gate; the offline replay itself never consumes them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import uproot

# Variant suffix <-> config string. 'A' = active layer, 'I' = inactive.
# config bit i (MSB=L1) '1' -> 'A'. So 1000->AIII, 1100->AAII, 1110->AAAI, 1111->AAAA.
PRODUCED_CONFIGS = {"1000": "AIII", "1100": "AAII", "1110": "AAAI", "1111": "AAAA"}
ALL_CONFIGS = [
    "1000", "0100", "0010", "0001",
    "1100", "1010", "1001", "0110", "0101", "0011",
    "1110", "1101", "1011", "0111",
    "1111",
]
ANGLE_MODES = ["none", "alpha", "alphaBeta"]

# PROMPT collections (re-anchored; NOT the Ext/displaced twins).
REF = "L1TTrack"                                 # 5-par OT-only prompt seed
SIDE = "L1TSmartPixelsRefitHitDigiRefitAAAA"     # all-layer prompt sidecar
TRK = "L1TSmartPixelsTrackDigiRefit"             # prompt refit + variant suffix
STUB = "L1TTrackStub"                            # real embedded OT stubs
GENVTX = "GenVtx"                                # interim prompt truth vertex

_REF_COLS = ["rInv", "phi", "tanL", "z0", "d0", "pt", "eta", "genuine",
             "tpPt", "tpEta", "tpFromHardInteraction", "tpPdgId", "nStubs",
             "looselyGenuine", "hitPattern", "chi2XYRed", "chi2ZRed"]

# Phase-2 Outer-Tracker barrel mean layer radii [cm] -- kept for the OVERVIEW layer
# guide circles. The per-track stub POSITIONS now come from the real L1TTrackStub
# table (x/y/z), not from these; these only draw the schematic barrel rings.
OT_BARREL_RADII = (25.0, 37.2, 52.2, 68.7, 86.0, 108.6)

_SIDE_COLS = ["trackIdx", "layer", "resX", "resY", "cotAlphaMeas", "cotBetaMeas",
              "parCotAlpha", "parCotBeta", "sigAlpha", "sigBeta",
              "pullX", "pullY", "pullAlpha", "pullBeta",
              "chi2IncRPhi", "chi2IncRZ", "selChi2Margin", "selHitClass",
              "hitAccepted", "hasAlpha", "hasBeta", "windowMult", "windowTruncated"]
_TRK_COLS = ["rInv", "phi", "tanL", "z0", "d0", "pt", "chi2XYRed", "chi2ZRed",
             "spxNAcceptedHits", "spxLayerHitMask", "spxChi2IncRPhiTot",
             "spxChi2IncRZTot", "spxRefitPerformed", "spxSeedCovOK",
             "spxParametrizedSeed", "spxMaxWindowMult", "spxNCrossings",
             "spxNKFUpdates"]
_STUB_COLS = ["trackIdx", "layer", "isBarrel", "r", "x", "y", "z"]


@dataclass
class TrackRecord:
    """One PROMPT reference track + its AAAA per-layer sidecar hits + real OT stubs.

    ``hits`` maps layer (1..4) -> dict of per-hit scalars from the AAAA sidecar
    (only accepted crossings are stored). ``stubs`` is a list of the real OT stub
    dicts (x/y/z, r, layer, isBarrel) for this track. ``real`` maps variant suffix
    -> dict of the produced prompt-refit params/chi2 (validation only). ``genvtx``
    is the generated primary vertex (interim prompt truth).
    """

    event: int
    idx: int
    seed: dict            # rInv, phi0, tanL, z0, d0, pt, eta, chi2XYRed, chi2ZRed
    truth: dict           # genuine, tpPt, tpEta, fromHard, pdgId, nStubs
    hits: dict            # layer -> per-hit dict (IT sidecar)
    stubs: list           # list of real OT stub dicts
    real: dict            # variant -> prompt-refit dict
    genvtx: dict          # x, y, z of the generated PV


@dataclass
class NanoData:
    radii: tuple
    tracks: list = field(default_factory=list)   # flat list of TrackRecord (one per (event, idx))


def _flat(arr, e):
    import awkward as ak
    return ak.to_numpy(arr[e])


def load_nano(nano_file, layer_radii=(3.0, 6.8, 10.9, 16.0), max_events=None):
    """Load every PROMPT reference track + its AAAA sidecar + real OT stubs + PV."""
    t = uproot.open(f"{nano_file}:Events")
    n_ev = t.num_entries if max_events is None else min(max_events, t.num_entries)

    ref = t.arrays([f"{REF}_{c}" for c in _REF_COLS], entry_stop=n_ev)
    side = t.arrays([f"{SIDE}_{c}" for c in _SIDE_COLS], entry_stop=n_ev)
    stub = t.arrays([f"{STUB}_{c}" for c in _STUB_COLS], entry_stop=n_ev)
    trk = {v: t.arrays([f"{TRK}{v}_{c}" for c in _TRK_COLS], entry_stop=n_ev)
           for v in PRODUCED_CONFIGS.values()}
    # GenVtx is a per-event singleton (not jagged): read as flat arrays.
    gv = {c: t[f"{GENVTX}_{c}"].array(entry_stop=n_ev, library="np")
          for c in ("x", "y", "z")}

    data = NanoData(radii=tuple(layer_radii))
    for e in range(n_ev):
        g = _flat(ref[f"{REF}_genuine"], e)
        n_trk = len(g)
        s = {c: _flat(side[f"{SIDE}_{c}"], e) for c in _SIDE_COLS}
        st = {c: _flat(stub[f"{STUB}_{c}"], e) for c in _STUB_COLS}
        tv = {v: {c: _flat(trk[v][f"{TRK}{v}_{c}"], e) for c in _TRK_COLS}
              for v in PRODUCED_CONFIGS.values()}
        rf = {c: _flat(ref[f"{REF}_{c}"], e) for c in _REF_COLS}
        genvtx = dict(x=float(gv["x"][e]), y=float(gv["y"][e]), z=float(gv["z"][e]))

        for i in range(n_trk):
            seed = dict(rInv=float(rf["rInv"][i]), phi0=float(rf["phi"][i]),
                        tanL=float(rf["tanL"][i]), z0=float(rf["z0"][i]),
                        d0=float(rf["d0"][i]), pt=float(rf["pt"][i]), eta=float(rf["eta"][i]),
                        chi2XYRed=float(rf["chi2XYRed"][i]), chi2ZRed=float(rf["chi2ZRed"][i]))
            truth = dict(genuine=bool(rf["genuine"][i]),
                         looselyGenuine=bool(rf["looselyGenuine"][i]),
                         tpPt=float(rf["tpPt"][i]), tpEta=float(rf["tpEta"][i]),
                         fromHard=bool(rf["tpFromHardInteraction"][i]),
                         pdgId=int(rf["tpPdgId"][i]), nStubs=int(rf["nStubs"][i]),
                         hitPattern=int(rf["hitPattern"][i]))
            hits = {}
            rows = np.where(s["trackIdx"] == i)[0]
            for r in rows:
                if not bool(s["hitAccepted"][r]):
                    continue
                L = int(s["layer"][r])
                hits[L] = {c: (float(s[c][r]) if s[c].dtype.kind == "f"
                               else int(s[c][r]) if s[c].dtype.kind in "iu"
                               else bool(s[c][r])) for c in _SIDE_COLS}
            srows = np.where(st["trackIdx"] == i)[0]
            stubs = [dict(layer=int(st["layer"][r]), isBarrel=bool(st["isBarrel"][r]),
                          r=float(st["r"][r]), x=float(st["x"][r]),
                          y=float(st["y"][r]), z=float(st["z"][r])) for r in srows]
            real = {}
            for v in PRODUCED_CONFIGS.values():
                real[v] = dict(rInv=float(tv[v]["rInv"][i]), phi0=float(tv[v]["phi"][i]),
                               tanL=float(tv[v]["tanL"][i]), z0=float(tv[v]["z0"][i]),
                               d0=float(tv[v]["d0"][i]), pt=float(tv[v]["pt"][i]),
                               chi2XYRed=float(tv[v]["chi2XYRed"][i]),
                               chi2ZRed=float(tv[v]["chi2ZRed"][i]),
                               nAcc=int(tv[v]["spxNAcceptedHits"][i]),
                               nUpd=int(tv[v]["spxNKFUpdates"][i]),
                               layerMask=int(tv[v]["spxLayerHitMask"][i]),
                               chi2RPhiTot=float(tv[v]["spxChi2IncRPhiTot"][i]),
                               chi2RZTot=float(tv[v]["spxChi2IncRZTot"][i]),
                               refitPerformed=bool(tv[v]["spxRefitPerformed"][i]),
                               maxWinMult=int(tv[v]["spxMaxWindowMult"][i]))
            data.tracks.append(TrackRecord(event=e, idx=i, seed=seed, truth=truth,
                                           hits=hits, stubs=stubs, real=real, genvtx=genvtx))
    return data


def config_active_layers(config):
    """'1100' -> (True, True, False, False) for layers 1..4."""
    return tuple(c == "1" for c in config)


def hit_class_name(cls):
    return {0: "same-TP (true)", 1: "other-TP (wrong)", 2: "noise", -1: "none"}.get(cls, "?")


def barrel_stub_xy(stubs):
    """Real OT barrel stubs (isBarrel) as (x, y, r, label) for the overview panel."""
    out = []
    for hd in stubs:
        if hd["isBarrel"]:
            out.append((hd["x"], hd["y"], hd["r"], f"OT-L{hd['layer']}"))
    return out


def barrel_stub_rz(stubs):
    """Real OT barrel stubs as (z, r, label) for the r-z overview panel."""
    return [(hd["z"], hd["r"], f"OT-L{hd['layer']}") for hd in stubs if hd["isBarrel"]]
