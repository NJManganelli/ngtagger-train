"""Per-constituent feature engineering: mirrors upstream TrainTagger
baseline_hardware_inputs, computed from L1PFTrkNano candidate + jet tables
(jet-relative quantities recomputed here from the link structure).

ParticleType convention (l1t::PFCandidate::id() stored in L1*PuppiCand_id):
  ChargedHadron=0, Electron=1, NeutralHadron=2, Photon=3, Muon=4
"""

from __future__ import annotations

import awkward as ak
import numpy as np

# Composable feature groups: select via data_config.feature_groups, e.g.
#   feature_groups: [baseline]                      # upstream hardware inputs
#   feature_groups: [baseline, track]               # + track-word floats
#   feature_groups: [baseline, track, trkquality]   # + trkquality BDT score
#   feature_groups: [baseline, cluster]             # + HGCal cluster shapes
FEATURE_GROUPS = {
    "baseline": [
        "pt", "pt_rel", "pt_log", "deta", "dphi", "mass",
        "isPhoton", "isElectronPlus", "isElectronMinus", "isMuonPlus", "isMuonMinus",
        "isNeutralHadron", "isChargedHadronPlus", "isChargedHadronMinus",
        "z0", "dxy", "isfilled", "puppiweight", "emid", "quality",
    ],
    "track": [
        "trk_rInv", "trk_tanL", "trk_z0", "trk_d0",
        "trk_chi2XYRed", "trk_chi2ZRed", "trk_chi2BendRed",
    ],
    # the track-quality GBDT score as a separate, composable input option
    "trkquality": ["trk_trkMVA1"],
    "cluster": ["cl_hOverE", "cl_sigmaRRTot", "cl_absZBarycenter", "cl_eMax", "cl_sigmaZZ"],
    # global fastHisto PV transverse position + significance, one value per
    # event broadcast to every constituent slot (computed at dataset-build
    # time from the extended-track table via ngtagger.train.nnvtx; needs
    # 5-parameter d0, i.e. track_table=L1TExtTrack)
    "vertexdxy": ["vtx_dx", "vtx_dy", "vtx_dxsig", "vtx_dysig"],
}


def resolve_feature_groups(groups: list[str]) -> list[str]:
    names: list[str] = []
    for g in groups:
        if g not in FEATURE_GROUPS:
            raise KeyError(f"unknown feature group '{g}'; known: {sorted(FEATURE_GROUPS)}")
        for n in FEATURE_GROUPS[g]:
            if n not in names:
                names.append(n)
    return names


def _dphi(phi1, phi2):
    d = phi1 - phi2
    return (d + np.pi) % (2 * np.pi) - np.pi


def build_features(jets: ak.Array, constituents: ak.Array, n_const: int = 16,
                   feature_groups: list[str] | None = None):
    """Build the (n_jets, n_const, n_features) tensor.

    jets: (event, jet) records; constituents: (event, jet, cand) records
    aligned with jets. Returns (X float32 array, feature name list).
    """
    names = resolve_feature_groups(feature_groups or ["baseline"])
    c = constituents
    pid = c["id"]
    charge = c["charge"]

    feats = {
        "pt": c.pt,
        "pt_rel": ak.where(ak.broadcast_arrays(jets.pt, c.pt)[0] > 0, c.pt / ak.broadcast_arrays(jets.pt, c.pt)[0], 0.0),
        "pt_log": np.log(ak.where(c.pt > 0, c.pt, 1.0)),
        "deta": c.eta - ak.broadcast_arrays(jets.eta, c.eta)[0],
        "dphi": _dphi(c.phi, ak.broadcast_arrays(jets.phi, c.phi)[0]),
        "mass": c.mass,
        "isPhoton": pid == 3,
        "isElectronPlus": (pid == 1) & (charge > 0),
        "isElectronMinus": (pid == 1) & (charge < 0),
        "isMuonPlus": (pid == 4) & (charge > 0),
        "isMuonMinus": (pid == 4) & (charge < 0),
        "isNeutralHadron": pid == 2,
        "isChargedHadronPlus": (pid == 0) & (charge > 0),
        "isChargedHadronMinus": (pid == 0) & (charge < 0),
        "z0": c.z0,
        "dxy": c.dxy,
        "puppiweight": c.puppiWeight,
        "emid": c.hwEmID,
        "quality": c.hwTkQuality,
    }
    for n in names:
        if n.startswith("trk_") or n.startswith("cl_") or n.startswith("vtx_"):
            if n == "cl_absZBarycenter":
                feats[n] = abs(c["cl_zBarycenter"])
            else:
                feats[n] = c[n]

    # flatten jets across events, pad candidate axis to n_const
    n_feat = len([n for n in names if n != "isfilled"])
    flat = {k: ak.flatten(v, axis=1) for k, v in feats.items() if k in names}
    padded = {k: ak.fill_none(ak.pad_none(v, n_const, axis=1, clip=True), 0.0) for k, v in flat.items()}

    isfilled = ak.fill_none(
        ak.pad_none(ak.ones_like(ak.flatten(c.pt, axis=1)), n_const, axis=1, clip=True), 0.0
    )

    columns = []
    for name in names:
        columns.append(ak.to_numpy(isfilled if name == "isfilled" else padded[name]).astype(np.float32))
    X = np.stack(columns, axis=-1)
    assert X.shape[1:] == (n_const, n_feat + 1), X.shape
    return X, list(names)
