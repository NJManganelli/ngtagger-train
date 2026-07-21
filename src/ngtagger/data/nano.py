"""Read L1PFTrkNano (DPGAnalysis/Phase2L1TNanoAOD extended flavors) directly
into per-jet constituent tensors for tagger training. No intermediate files.

Table layout produced by the L1PFTrkNano tier:
  L1puppiJetSC4NG_*   NGJet collection (jet table the link jetIdx points to)
  L1SC4NGJetCands_*   jet<->constituent association: jetIdx, candIdx, slot, inTagger
  L1ExtPuppiCand_*    deregionized (extended) Puppi candidates + jetIdx/l1TrackIdx/hgcClusterIdx
  L1TExtTrack_*       extended tracker tracks (crossref via l1TrackIdx)
  L1HGCCluster_*      HGCal multiclusters (crossref via hgcClusterIdx)
  GenJet_* / GenVisTau_* / GenPart_*  (withGen flavors) for truth labeling

The link table is written jet-major with ascending slot by construction
(L1JetCandLinkTableProducer), which read_links relies on via run_lengths.
"""

from __future__ import annotations

import awkward as ak
import numpy as np
import uproot

# Candidate-level branches used to build features. hgcClusterIdx and l1TrackIdx
# are OPTIONAL: some fat single-coherent-view nanos ship L1ExtPuppiCand without
# hgcClusterIdx and with an unfilled (all -1) l1TrackIdx. The loader tolerates
# their absence/emptiness and (for l1TrackIdx) recovers the link from the
# row-aligned plain L1PuppiCand tier when available. See load_jets.
CAND_BRANCHES = [
    "pt", "eta", "phi", "mass", "charge", "id", "z0", "dxy",
    "puppiWeight", "hwEmID", "hwTkQuality", "l1TrackIdx", "hgcClusterIdx",
]
# candidate branches that may legitimately be missing on some nanos
CAND_OPTIONAL = ["hgcClusterIdx"]
TRACK_BRANCHES = ["rInv", "tanL", "z0", "d0", "chi2XYRed", "chi2ZRed", "chi2BendRed", "trkMVA1"]
# extra track branches needed only by the "vertexdxy" feature group (global
# fastHisto PV (dx, dy) estimate; requires the 5-par extended track table)
VTX_TRACK_BRANCHES = ["pt", "phi", "z0", "d0"]
CLUSTER_BRANCHES = ["hOverE", "sigmaRRTot", "zBarycenter", "eMax", "sigmaZZ"]
JET_BRANCHES = ["pt", "eta", "phi", "et"]
GEN_BRANCHES = {
    "GenJet": ["pt", "eta", "phi", "partonFlavour", "hadronFlavour"],
    "GenVisTau": ["pt", "eta", "phi", "charge"],
    "GenPart": ["pt", "eta", "phi", "pdgId", "statusFlags"],
}


def _branches(prefix: str, names: list[str]) -> list[str]:
    return [f"{prefix}_{n}" for n in names]


def read_events(
    files: list[str],
    jet_table: str = "L1puppiJetSC4NG",
    link_table: str = "L1SC4NGJetCands",
    cand_table: str = "L1ExtPuppiCand",
    track_table: str = "L1TExtTrack",
    cluster_table: str = "L1HGCCluster",
    with_gen: bool = True,
    max_events: int | None = None,
    extra_track_branches: list[str] | None = None,
    link_cand_table: str | None = None,
    vertex_table: str | None = None,
) -> ak.Array:
    """Load the needed branches from L1PFTrkNano Events trees.

    link_cand_table: optional plain-Puppi tier (e.g. L1PuppiCand) that is
    ROW-ALIGNED with cand_table but carries a filled l1TrackIdx when the
    (deregionized) cand_table's own l1TrackIdx is unfilled (all -1). When set,
    its l1TrackIdx column is read and used to recover the constituent->track
    link. See load_jets.
    """
    # cand branches: drop the ones the file doesn't have (e.g. hgcClusterIdx)
    with uproot.open(f"{files[0]}:Events") as tree0:
        present = set(tree0.keys())
    cand_present = [n for n in CAND_BRANCHES
                    if f"{cand_table}_{n}" in present or n not in CAND_OPTIONAL]
    branches = (
        _branches(jet_table, JET_BRANCHES)
        + [f"{jet_table}_*TagScore*"]
        + _branches(link_table, ["jetIdx", "candIdx", "slot", "inTagger"])
        + _branches(cand_table, cand_present)
        + _branches(track_table, TRACK_BRANCHES + list(extra_track_branches or []))
        + _branches(cluster_table, CLUSTER_BRANCHES)
    )
    if vertex_table:
        branches += [f"{vertex_table}_z0"]
    if with_gen:
        for tab, names in GEN_BRANCHES.items():
            branches += _branches(tab, names)
    events = uproot.concatenate(
        [f"{f}:Events" for f in files],
        filter_name=branches,
        how="zip",
    )
    if max_events is not None:
        events = events[:max_events]

    # The plain-Puppi l1TrackIdx recovery column is read UN-zipped (a lone
    # per-table branch mangles the how="zip" grouping) and attached as a
    # top-level jagged field consumed in load_jets. Skipped when it coincides
    # with cand_table (the link is already in the zipped cand record) or when
    # the file simply doesn't carry that branch.
    if link_cand_table and link_cand_table != cand_table \
            and f"{link_cand_table}_l1TrackIdx" in present:
        aux = uproot.concatenate(
            [f"{f}:Events" for f in files],
            filter_name=[f"{link_cand_table}_l1TrackIdx"],
        )
        col = aux[f"{link_cand_table}_l1TrackIdx"]
        if max_events is not None:
            col = col[:max_events]
        events = ak.with_field(events, col, f"{link_cand_table}_l1TrackIdx")
    return events


def group_constituents(events: ak.Array, jet_table: str, link_table: str, n_const: int = 16) -> ak.Array:
    """Nest the flat link table into per-jet constituent index lists.

    Relies on the producer writing links jet-major with ascending slot.
    Returns (event, jet, constituent)-nested candIdx, truncated to the
    tagger window (slot < n_const) and to valid candIdx.
    """
    links = events[link_table]
    jet_counts = ak.num(events[jet_table].pt, axis=1)

    # run_lengths groups the jet-major link rows per jet
    lengths = ak.run_lengths(links.jetIdx)
    nested_cand = ak.unflatten(links.candIdx, ak.flatten(lengths), axis=1)
    nested_slot = ak.unflatten(links.slot, ak.flatten(lengths), axis=1)

    n_groups = ak.num(nested_cand, axis=1)
    if not ak.all(n_groups == jet_counts):
        raise RuntimeError(
            "link-table groups do not align with the jet table; "
            "was the nano produced with empty jet-table cut and native ordering?"
        )

    keep = (nested_slot < n_const) & (nested_cand >= 0)
    return ak.mask(nested_cand, keep)[~ak.is_none(ak.mask(nested_cand, keep), axis=2)]


def _gather2(arr2d: ak.Array, nested_idx: ak.Array) -> ak.Array:
    """Index an (event, obj) array with an (event, jet, constituent) jagged
    index. awkward rejects direct jagged indexing when the index is one level
    deeper than the array, so gather flat per event and re-nest."""
    counts = ak.num(nested_idx, axis=2)
    flat_idx = ak.flatten(nested_idx, axis=2)
    return ak.unflatten(arr2d[flat_idx], ak.flatten(counts, axis=1), axis=1)


def gather(cands: ak.Array, nested_idx: ak.Array, field: str) -> ak.Array:
    """Gather a candidate field into (event, jet, constituent) nesting."""
    return _gather2(cands[field], nested_idx)


def crossref_gather(cands: ak.Array, other: ak.Array, nested_idx: ak.Array, idx_field: str, field: str, default: float = 0.0) -> ak.Array:
    """Gather a field from a crossref'd table (tracks/clusters) via an idx
    column on the candidates; -1 (no ref) yields `default`."""
    ref_idx = _gather2(cands[idx_field], nested_idx)
    valid = ref_idx >= 0
    safe_idx = ak.where(valid, ref_idx, 0)
    vals = _gather2(other[field], safe_idx)
    return ak.where(valid, vals, default)


def load_jets(
    files: list[str],
    n_const: int = 16,
    feature_groups: list[str] | None = None,
    with_gen: bool = True,
    max_events: int | None = None,
    jet_table: str = "L1puppiJetSC4NG",
    link_table: str = "L1SC4NGJetCands",
    cand_table: str = "L1ExtPuppiCand",
    track_table: str = "L1TExtTrack",
    cluster_table: str = "L1HGCCluster",
    link_cand_table: str | None = "L1PuppiCand",
    refit_config: str | None = None,
    refit_bdt_json: str | None = None,
    vertex_table: str = "L1Vertex",
):
    """Full pipeline: nano files -> (jets awkward record, per-jet constituent
    feature records). Feature engineering and labeling live in features.py /
    labels.py; this returns the aligned building blocks.

    Two Stage-4-specific per-constituent feature groups gather from the
    constituent's tracker track (via the recovered l1TrackIdx into track_table):
      - "refitbdt": the on-the-fly Stage-3 refit-quality conifer score of the
        constituent's SmartPixels-refit track (needs refit_config + refit_bdt_json).
      - "vertexdxy_pv": the transverse IP (dxy) of the constituent track w.r.t.
        the emulated primary vertex (L1Vertex z0 + track d0/z0/phi).

    link_cand_table: plain-Puppi tier row-aligned with cand_table that supplies
    a filled l1TrackIdx when the (deregionized) cand_table's own is unfilled.
    """
    groups = feature_groups or ["baseline"]
    need_trk_gather = any(g in groups for g in ("track", "trkquality", "refitbdt", "vertexdxy_pv"))
    events = read_events(
        files,
        jet_table=jet_table,
        link_table=link_table,
        cand_table=cand_table,
        track_table=track_table,
        cluster_table=cluster_table,
        with_gen=with_gen,
        max_events=max_events,
        extra_track_branches=(VTX_TRACK_BRANCHES if "vertexdxy" in groups else None),
        link_cand_table=link_cand_table if need_trk_gather else None,
        vertex_table=vertex_table if "vertexdxy_pv" in groups else None,
    )
    nested_idx = group_constituents(events, jet_table, link_table, n_const=n_const)
    cands = events[cand_table]

    const = {}
    for name in CAND_BRANCHES:
        if name in CAND_OPTIONAL and name not in cands.fields:
            continue
        const[name] = gather(cands, nested_idx, name)

    # Recover the constituent->track link. The deregionized L1ExtPuppiCand ships
    # l1TrackIdx=-1 in these fat nanos; the row-aligned plain L1PuppiCand carries
    # the filled link, so swap in its column when ours is empty.
    link_idx_field = "l1TrackIdx"
    if need_trk_gather and link_cand_table and f"{link_cand_table}_l1TrackIdx" in _record_fields(events):
        own = cands["l1TrackIdx"] if "l1TrackIdx" in cands.fields else None
        alt = events[f"{link_cand_table}_l1TrackIdx"]
        if own is None or not bool(ak.any(ak.flatten(own) >= 0)):
            # inject alt as a candidate field so crossref_gather can use it
            cands = ak.with_field(cands, alt, "l1TrackIdx_recovered")
            link_idx_field = "l1TrackIdx_recovered"

    if "track" in groups or "trkquality" in groups:
        tracks = events[track_table]
        for name in TRACK_BRANCHES:
            const[f"trk_{name}"] = crossref_gather(cands, tracks, nested_idx, link_idx_field, name)
    if "cluster" in groups:
        clusters = events[cluster_table]
        for name in CLUSTER_BRANCHES:
            const[f"cl_{name}"] = crossref_gather(cands, clusters, nested_idx, "hgcClusterIdx", name)
    if "vertexdxy" in groups:
        # per-event global PV (dx, dy) from the extended-track table,
        # broadcast to every constituent slot (fails loudly on a 4-par table)
        from ngtagger.train.nnvtx import vertex_dxy_features

        vtx = vertex_dxy_features(events[track_table], sanitize=True)
        for name, col in vtx.items():
            const[name] = ak.broadcast_arrays(ak.Array(col), const["pt"])[0]
    if "vertexdxy_pv" in groups:
        # per-constituent-track transverse IP wrt the emulated PV:
        # dxy = d0 - (x_pv sin(phi) - y_pv cos(phi)); with only PV z0 available
        # the transverse PV position is the beamspot (0,0), so this reduces to
        # the track's own d0 -- BUT we also gate on |z0_trk - z0_pv| so the
        # feature is the b-tagging-style displacement of PV-associated tracks.
        d0 = crossref_gather(cands, events[track_table], nested_idx, link_idx_field, "d0")
        z0t = crossref_gather(cands, events[track_table], nested_idx, link_idx_field, "z0")
        pv_z0 = ak.firsts(events[vertex_table]["z0"], axis=1)
        pv_z0 = ak.fill_none(pv_z0, 0.0)
        pv_b = ak.broadcast_arrays(pv_z0, const["pt"])[0]
        const["vtxpv_dxy"] = d0
        const["vtxpv_dz"] = z0t - pv_b
    if "refitbdt" in groups:
        from ngtagger.data.refit_bdt_feature import refit_bdt_per_constituent

        if refit_config is None or refit_bdt_json is None:
            raise ValueError("refitbdt group requires refit_config + refit_bdt_json")
        score = refit_bdt_per_constituent(
            files, events, cands, nested_idx, link_idx_field,
            config=refit_config, conifer_json=refit_bdt_json,
            track_table=track_table, max_events=max_events,
        )
        const["refit_bdt_score"] = score

    constituents = ak.zip(const, depth_limit=3)
    jets = events[jet_table]
    gen = {tab: events[tab] for tab in GEN_BRANCHES} if with_gen else None
    return jets, constituents, gen


def _record_fields(events: ak.Array):
    try:
        return set(events.fields)
    except Exception:
        return set()
