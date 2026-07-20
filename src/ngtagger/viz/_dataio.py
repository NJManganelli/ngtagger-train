"""Nano I/O + track curation for the digiRefit refit-replay visualizer.

Pure uproot/numpy. Reads the reference track table, the four PRODUCED digiRefit
variant tables (AIII/AAII/AAAI/AAAA = configs 1000/1100/1110/1111, all made at
useAngles=alphaBeta), and the AAAA per-hit sidecar. The AAAA sidecar carries all
four TBPX layers' selected hits, so any of the 15 layer configs is a subset of it.

The real refit-track values are loaded ONLY for the validation gate; the replay
itself never consumes them.
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

REF = "L1TExtTrack"
SIDE = "L1TSmartPixelsExtRefitHitDigiRefitAAAA"   # all-layer sidecar
TRK = "L1TSmartPixelsExtTrackDigiRefit"           # + variant suffix

_REF_COLS = ["rInv", "phi", "tanL", "z0", "d0", "pt", "eta", "genuine",
             "tpPt", "tpFromHardInteraction", "tpPdgId", "nStubs", "looselyGenuine",
             "hitPattern"]

# Phase-2 Outer-Tracker barrel mean layer radii [cm], from the tracklet firmware
# constants (L1Trigger/TrackFindingTracklet Settings.h: irmean_ * rmaxdisk_/4096,
# rmaxdisk_=120): {851,1269,1784,2347,2936,3697} -> these values. Bit i (LSB) of the
# TTTrack hitPattern populates OT barrel layer i+1 (bit 6 = a forward disk slot;
# schematic on barrel tracks). These anchor the OT-only seed's long lever arm.
OT_BARREL_RADII = (24.9, 37.2, 52.3, 68.8, 86.0, 108.3)
_SIDE_COLS = ["trackIdx", "layer", "resX", "resY", "cotAlphaMeas", "cotBetaMeas",
              "parCotAlpha", "parCotBeta", "sigAlpha", "sigBeta",
              "pullX", "pullY", "pullAlpha", "pullBeta",
              "chi2IncRPhi", "chi2IncRZ", "selChi2Margin", "selHitClass",
              "hitAccepted", "hasAlpha", "hasBeta", "windowMult", "windowTruncated"]
_TRK_COLS = ["rInv", "phi", "tanL", "z0", "d0", "pt", "spxNAcceptedHits",
             "spxLayerHitMask", "spxChi2IncRPhiTot", "spxChi2IncRZTot",
             "spxRefitPerformed", "spxSeedCovOK", "spxParametrizedSeed",
             "spxMaxWindowMult", "spxNCrossings"]


@dataclass
class TrackRecord:
    """One reference track + its AAAA per-layer sidecar hits + real refit values.

    ``hits`` maps layer (1..4) -> dict of per-hit scalars from the AAAA sidecar
    (only accepted crossings are stored). ``real`` maps variant suffix -> dict of
    the produced refit params/chi2 (validation only).
    """

    event: int
    idx: int
    seed: dict            # rInv, phi0, tanL, z0, d0, pt, eta
    truth: dict           # genuine, tpPt, fromHard, pdgId, nStubs
    hits: dict            # layer -> per-hit dict
    real: dict            # variant -> refit dict


@dataclass
class NanoData:
    radii: tuple
    tracks: list = field(default_factory=list)   # flat list of TrackRecord (one per (event, idx))


def _flat(arr, e):
    import awkward as ak
    return ak.to_numpy(arr[e])


def load_nano(nano_file, layer_radii=(3.0, 6.8, 10.9, 16.0), max_events=None):
    """Load every reference track and its AAAA sidecar hits + produced-variant refits."""
    import awkward as ak

    t = uproot.open(f"{nano_file}:Events")
    n_ev = t.num_entries if max_events is None else min(max_events, t.num_entries)

    ref = t.arrays([f"{REF}_{c}" for c in _REF_COLS], entry_stop=n_ev)
    side = t.arrays([f"{SIDE}_{c}" for c in _SIDE_COLS], entry_stop=n_ev)
    trk = {v: t.arrays([f"{TRK}{v}_{c}" for c in _TRK_COLS], entry_stop=n_ev)
           for v in PRODUCED_CONFIGS.values()}

    data = NanoData(radii=tuple(layer_radii))
    for e in range(n_ev):
        g = _flat(ref[f"{REF}_genuine"], e)
        n_trk = len(g)
        # sidecar arrays for this event
        s = {c: _flat(side[f"{SIDE}_{c}"], e) for c in _SIDE_COLS}
        # per-variant track arrays
        tv = {v: {c: _flat(trk[v][f"{TRK}{v}_{c}"], e) for c in _TRK_COLS}
              for v in PRODUCED_CONFIGS.values()}
        rf = {c: _flat(ref[f"{REF}_{c}"], e) for c in _REF_COLS}

        # group sidecar rows by trackIdx
        order = np.argsort(s["trackIdx"], kind="stable")
        for i in range(n_trk):
            seed = dict(rInv=float(rf["rInv"][i]), phi0=float(rf["phi"][i]),
                        tanL=float(rf["tanL"][i]), z0=float(rf["z0"][i]),
                        d0=float(rf["d0"][i]), pt=float(rf["pt"][i]), eta=float(rf["eta"][i]))
            truth = dict(genuine=bool(rf["genuine"][i]),
                         looselyGenuine=bool(rf["looselyGenuine"][i]),
                         tpPt=float(rf["tpPt"][i]), fromHard=bool(rf["tpFromHardInteraction"][i]),
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
            real = {}
            for v in PRODUCED_CONFIGS.values():
                real[v] = dict(rInv=float(tv[v]["rInv"][i]), phi0=float(tv[v]["phi"][i]),
                               tanL=float(tv[v]["tanL"][i]), z0=float(tv[v]["z0"][i]),
                               d0=float(tv[v]["d0"][i]), pt=float(tv[v]["pt"][i]),
                               nAcc=int(tv[v]["spxNAcceptedHits"][i]),
                               layerMask=int(tv[v]["spxLayerHitMask"][i]),
                               chi2RPhiTot=float(tv[v]["spxChi2IncRPhiTot"][i]),
                               chi2RZTot=float(tv[v]["spxChi2IncRZTot"][i]),
                               refitPerformed=bool(tv[v]["spxRefitPerformed"][i]),
                               maxWinMult=int(tv[v]["spxMaxWindowMult"][i]))
            data.tracks.append(TrackRecord(event=e, idx=i, seed=seed, truth=truth,
                                           hits=hits, real=real))
    return data


def config_active_layers(config):
    """'1100' -> (True, True, False, False) for layers 1..4."""
    return tuple(c == "1" for c in config)


def hit_class_name(cls):
    return {0: "same-TP (true)", 1: "other-TP (wrong)", 2: "noise", -1: "none"}.get(cls, "?")


def ot_stub_radii(hit_pattern, ot_radii=OT_BARREL_RADII):
    """Decode a TTTrack hitPattern into the OT barrel radii carrying a stub.

    Bit i (LSB) -> OT barrel layer i+1. popcount(hitPattern) == nStubs (verified on
    the sample). Bit 6 is a forward-disk slot with no barrel radius; for barrel
    tracks it is rare and is placed schematically at the outermost barrel radius.
    Returns a list of (radius, layer_label) for the set bits.
    """
    out = []
    for b in range(7):
        if hit_pattern & (1 << b):
            if b < len(ot_radii):
                out.append((ot_radii[b], f"OT-L{b + 1}"))
            else:
                out.append((ot_radii[-1], f"OT-disk(bit{b})"))
    return out
