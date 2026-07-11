"""Coffea NanoEvents schema for the extended Phase-2 L1 nano tiers
(L1TrkNano / L1PFNano / L1PFTrkNano).

One schema serves every tier:
  - cross-references are only built for collections present in the file;
    coffea emits a RuntimeWarning when a source collection exists but its
    cross-ref target table is missing (partial tier), and stays silent when
    the source collection is entirely absent (feature of NanoAODSchema's
    warn_missing_crossrefs handling)
  - the jet-constituent link tables get `jet` / `cand` accessors plus the
    derived tagger variables (pt_rel, pt_rel_log, deta, dphi) recomputed
    from the cross-refs
  - candidates get matched_track / matched_cluster / matched_jet accessors
    and uniform track-truth access: `trk_genuine` uses the direct column
    (candidate-only L1PFNano tier) or falls back to the l1TrackIdx
    indirection into the track-table truth (L1PFTrkNano tier)

Usage:
    from coffea.nanoevents import NanoEventsFactory
    from ngtagger.coffea_schema import L1NanoSchema

    events = NanoEventsFactory.from_root({file: "Events"}, schemaclass=L1NanoSchema).events()
    links = events.L1SC4NGJetCands
    tensor_feats = links.pt_rel  # etc.
"""

from __future__ import annotations

import awkward
import numpy
from coffea.nanoevents import NanoAODSchema
from coffea.nanoevents.methods import base, candidate, nanoaod, vector

behavior = {}
behavior.update(nanoaod.behavior)


def _dphi(a, b):
    return (a - b + numpy.pi) % (2 * numpy.pi) - numpy.pi


class _JetCandLinkBase(base.NanoCollection):
    """Common derived variables for jet-constituent link tables; concrete
    subclasses bind the jet/cand target collections."""

    _jet_target = None
    _cand_target = None

    @property
    def jet(self):
        """linked jet (via jetIdx)"""
        return self._events()[self._jet_target]._apply_global_index(self.jetIdxG)

    @property
    def cand(self):
        """linked constituent candidate (via candIdx; masked where -1)"""
        return self._events()[self._cand_target]._apply_global_index(self.candIdxG)

    @property
    def pt_rel(self):
        return self.cand.pt / self.jet.pt

    @property
    def pt_rel_log(self):
        return numpy.log(self.pt_rel)

    @property
    def deta(self):
        return self.cand.eta - self.jet.eta

    @property
    def dphi(self):
        return _dphi(self.cand.phi, self.jet.phi)


@awkward.mixin_class(behavior)
class L1SC4NGJetCandLink(_JetCandLinkBase):
    _jet_target = "L1puppiJetSC4NG"
    _cand_target = "L1ExtPuppiCand"


@awkward.mixin_class(behavior)
class L1SC4JetCandLink(_JetCandLinkBase):
    _jet_target = "L1puppiJetSC4"
    _cand_target = "L1PuppiCand"


class _L1PFCandBase(candidate.PtEtaPhiMCandidate, base.NanoCollection):
    """L1 PF/Puppi candidates: matched-object accessors + uniform truth."""

    _track_target = None
    _jet_target = None

    @property
    def matched_track(self):
        """underlying L1 track (via l1TrackIdx; masked where -1)"""
        return self._events()[self._track_target]._apply_global_index(self.l1TrackIdxG)

    @property
    def matched_cluster(self):
        """HGCal multicluster (via hgcClusterIdx; endcap only, masked where -1)"""
        return self._events().L1HGCCluster._apply_global_index(self.hgcClusterIdxG)

    @property
    def matched_jet(self):
        """first jet clustering this candidate (via jetIdx backref)"""
        return self._events()[self._jet_target]._apply_global_index(self.jetIdxG)

    @property
    def trk_genuine(self):
        """track-truth genuine flag: direct column (candidate-only tier) or
        via the track-table indirection (track tiers)."""
        if "trkGenuine" in self.fields:
            return self.trkGenuine
        return self.matched_track.genuine

    @property
    def trk_tp_pt(self):
        if "trkTpPt" in self.fields:
            return self.trkTpPt
        return self.matched_track.tpPt


behavior.update(awkward._util.copy_behaviors("PtEtaPhiMCandidate", "L1PuppiCandidate", behavior))


@awkward.mixin_class(behavior)
class L1PuppiCandidate(_L1PFCandBase):
    _track_target = "L1TTrack"
    _jet_target = "L1puppiJetSC4"


behavior.update(awkward._util.copy_behaviors("PtEtaPhiMCandidate", "L1ExtPuppiCandidate", behavior))


@awkward.mixin_class(behavior)
class L1ExtPuppiCandidate(_L1PFCandBase):
    _track_target = "L1TExtTrack"
    _jet_target = "L1puppiJetSC4NG"


for _name in ("L1PuppiCandidate", "L1ExtPuppiCandidate"):
    _arr = globals()[f"{_name}Array"]  # created by awkward.mixin_class
    _rec = globals()[f"{_name}Record"]
    _arr.ProjectionClass2D = vector.TwoVectorArray
    _arr.ProjectionClass3D = vector.ThreeVectorArray
    _arr.ProjectionClass4D = _arr
    _arr.MomentumClass = vector.LorentzVectorArray
    _rec.ProjectionClass2D = vector.TwoVectorRecord
    _rec.ProjectionClass3D = vector.ThreeVectorRecord
    _rec.ProjectionClass4D = _rec
    _rec.MomentumClass = vector.LorentzVectorRecord


class L1NanoSchema(NanoAODSchema):
    """NanoAODSchema extended with the L1TrkNano/L1PFNano/L1PFTrkNano
    collections and cross-references.

    Warning semantics across tiers: coffea's stock behavior warns for every
    unresolvable cross-reference, including when the *source* collection is
    entirely absent (a different tier, not a problem). This subclass keeps
    the warning only when the source collection is present but the index
    column or the target table is missing (a genuinely partial file)."""

    def __init__(self, base_form, *args, **kwargs):
        import re
        import warnings as _warnings

        present = {f.split("_")[0] for f in base_form.get("fields", [])}
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            super().__init__(base_form, *args, **kwargs)
        for w in caught:
            msg = str(w.message)
            m = re.match(r"Missing cross-reference index for (\w+) => (\w+)", msg)
            if m and m.group(1).split("_")[0] not in present:
                continue  # whole source collection absent: silent
            _warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)

    mixins = {
        **NanoAODSchema.mixins,
        "L1SC4NGJetCands": "L1SC4NGJetCandLink",
        "L1SC4JetCands": "L1SC4JetCandLink",
        "L1PuppiCand": "L1PuppiCandidate",
        "L1ExtPuppiCand": "L1ExtPuppiCandidate",
        "L1PFCand": "PtEtaPhiMCandidate",
        "L1TTrack": "NanoCollection",
        "L1TExtTrack": "NanoCollection",
        "L1HGCCluster": "NanoCollection",
        "L1puppiJetSC4NG": "NanoCollection",
        "L1puppiJetSC4": "NanoCollection",
    }

    all_cross_references = {
        **NanoAODSchema.all_cross_references,
        # jet <-> constituent association tables
        "L1SC4NGJetCands_jetIdx": "L1puppiJetSC4NG",
        "L1SC4NGJetCands_candIdx": "L1ExtPuppiCand",
        "L1SC4JetCands_jetIdx": "L1puppiJetSC4",
        "L1SC4JetCands_candIdx": "L1PuppiCand",
        # candidate crossrefs
        "L1PuppiCand_jetIdx": "L1puppiJetSC4",
        "L1PuppiCand_l1TrackIdx": "L1TTrack",
        "L1PuppiCand_hgcClusterIdx": "L1HGCCluster",
        "L1ExtPuppiCand_jetIdx": "L1puppiJetSC4NG",
        "L1ExtPuppiCand_l1TrackIdx": "L1TExtTrack",
        "L1ExtPuppiCand_hgcClusterIdx": "L1HGCCluster",
    }

    @classmethod
    def behavior(cls):
        return behavior


def jet_constituents(events, link_table: str = "L1SC4NGJetCands", tagger_window: bool = True):
    """Group the flat link table per jet, returning (event, jet, constituent)
    nested link records (with .cand/.pt_rel/... accessors). Relies on the
    jet-major, slot-ascending ordering guaranteed by the producer."""
    links = events[link_table]
    if tagger_window:
        pass  # filtering applied after grouping to keep jet alignment
    lengths = awkward.flatten(awkward.run_lengths(links.jetIdx))
    nested = awkward.unflatten(links, lengths, axis=1)
    if tagger_window:
        nested = nested[nested.inTagger & (nested.candIdx >= 0)]
    return nested
