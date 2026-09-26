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

import re

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


class _SmartPixelsRefitHitBase(base.NanoCollection):
    """SmartPixels per-hit refit link table: one row per accepted refit hit,
    with a `trackIdx` crossref into the matching variant track table (the same
    row index also indexes the reference L1TTrack/L1TExtTrack, 1:1 aligned).
    Concrete subclasses bind `_track_target` per digiRefit config/variant."""

    _track_target = None

    @property
    def track(self):
        """refit variant track this hit belongs to (via trackIdx)"""
        return self._events()[self._track_target]._apply_global_index(self.trackIdxG)

    @property
    def cluster(self):
        """the selected L1TSmartPixelsCluster row (via selClusterIdx; masked where -1)"""
        return self._events()[CLUSTER_TABLE]._apply_global_index(self.selClusterIdxG)


class _TrackStubBase(base.NanoCollection):
    """L1TTrackStub / L1TExtTrackStub: one row per stub of a track."""

    _track_target = None

    @property
    def track(self):
        """owning track (via trackIdx)"""
        return self._events()[self._track_target]._apply_global_index(self.trackIdxG)

    @property
    def ot_stub(self):
        """the same stub's L1TOTStub row, with its truth (via otStubIdx)"""
        return self._events()["L1TOTStub"]._apply_global_index(self.otStubIdxG)


@awkward.mixin_class(behavior)
class L1TrackStubLink(_TrackStubBase):
    _track_target = "L1TTrack"


@awkward.mixin_class(behavior)
class L1ExtTrackStubLink(_TrackStubBase):
    _track_target = "L1TExtTrack"


@awkward.mixin_class(behavior)
class L1DispVtx(base.NanoCollection):
    """GTT displaced vertex: track-pair accessors via first/secondIndexTrk."""

    @property
    def first_track(self):
        """higher-pt track of the pair"""
        return self._events().L1TExtTrack._apply_global_index(self.firstIndexTrkG)

    @property
    def second_track(self):
        """lower-pt track of the pair"""
        return self._events().L1TExtTrack._apply_global_index(self.secondIndexTrkG)


# ---------------------------------------------------------------------------
# Track-word hw -> float on-demand decoding
# Constants transcribed from DataFormats/L1TrackTrigger/interface/
# TTTrack_TrackWord.h; the hw columns in the file stay exactly as produced
# (raw unsigned bit fields) and these properties compute the undigitized
# double values only when accessed (with a lazy/virtual backend, accessing a
# property loads only the needed hw branch). All ops are vectorized numpy
# over the raw integer columns.
# ---------------------------------------------------------------------------
_TW_MIN_RINV = 0.006
_TW_MIN_PHI0 = 0.7853981696
_TW_MIN_Z0 = 20.46912512
_TW_STEPS = {
    "hwRinv": (15, 2.0 * _TW_MIN_RINV / (1 << 15)),
    "hwPhi": (12, 2.0 * _TW_MIN_PHI0 / (1 << 12)),
    "hwTanl": (16, 1.0 / (1 << 12)),
    "hwZ0": (12, 2.0 * _TW_MIN_Z0 / (1 << 12)),
    "hwD0": (13, 1.0 / (1 << 8)),
}
_TW_BINS = {
    "hwChi2RPhi": numpy.array(
        [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 10.0, 15.0, 20.0, 35.0, 60.0, 200.0]),
    "hwChi2RZ": numpy.array(
        [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0, 10.0, 20.0, 50.0]),
    "hwBendChi2": numpy.array([0.0, 0.75, 1.0, 1.5, 2.25, 3.5, 5.0, 20.0]),
    "hwMVAQuality": numpy.array([0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.750, 0.875]),
}


def _undigitize_signed(bits, n_bits, lsb, offset=0.5):
    """TTTrack_TrackWord::undigitizeSignedValue: two's complement decode,
    then (value + offset) * lsb."""
    v = awkward.values_astype(bits, numpy.int64)
    signed = awkward.where(v >= (1 << (n_bits - 1)), v - (1 << n_bits), v)
    return (awkward.values_astype(signed, numpy.float64) + offset) * lsb


def _bin_lookup(bits, bins):
    def apply(layout, **kwargs):
        if layout.is_numpy:
            return awkward.contents.NumpyArray(bins[numpy.asarray(layout.data, dtype=numpy.int64)])

    return awkward.transform(apply, bits)


class L1TrackWordDecoded(base.NanoCollection):
    """L1 track table behavior: raw hw fields as stored, plus on-demand
    float decoding mirroring the TTTrack_TrackWord get* methods."""

    @property
    def rinvFromHw(self):
        """1/R [1/cm] undigitized from hwRinv"""
        return _undigitize_signed(self.hwRinv, *_TW_STEPS["hwRinv"])

    @property
    def phiFromHw(self):
        """local phi (relative to sector center) undigitized from hwPhi"""
        return _undigitize_signed(self.hwPhi, *_TW_STEPS["hwPhi"])

    @property
    def tanlFromHw(self):
        """tan(lambda) undigitized from hwTanl"""
        return _undigitize_signed(self.hwTanl, *_TW_STEPS["hwTanl"])

    @property
    def z0FromHw(self):
        """z0 [cm] undigitized from hwZ0"""
        return _undigitize_signed(self.hwZ0, *_TW_STEPS["hwZ0"])

    @property
    def d0FromHw(self):
        """d0 [cm] undigitized from hwD0"""
        return _undigitize_signed(self.hwD0, *_TW_STEPS["hwD0"])

    @property
    def chi2RPhiFromHw(self):
        """chi2 r-phi bin value from hwChi2RPhi"""
        return _bin_lookup(self.hwChi2RPhi, _TW_BINS["hwChi2RPhi"])

    @property
    def chi2RZFromHw(self):
        """chi2 r-z bin value from hwChi2RZ"""
        return _bin_lookup(self.hwChi2RZ, _TW_BINS["hwChi2RZ"])

    @property
    def bendChi2FromHw(self):
        """bend chi2 bin value from hwBendChi2"""
        return _bin_lookup(self.hwBendChi2, _TW_BINS["hwBendChi2"])

    @property
    def mvaQualityFromHw(self):
        """track-quality MVA bin value from hwMVAQuality"""
        return _bin_lookup(self.hwMVAQuality, _TW_BINS["hwMVAQuality"])


@awkward.mixin_class(behavior)
class L1TrackWord(L1TrackWordDecoded):
    pass


# ---------------------------------------------------------------------------
# SmartPixels digiRefit variant tables (Phase-3 tier-2 refit study).
# Track tables L1TSmartPixelsTrack<suffix>/L1TSmartPixelsExtTrack<suffix> are
# track-like: they carry the same hw track-word columns as L1TTrack (reuse the
# hw-decode mixin) plus the spix* refit-status extension columns, and are 1:1
# row-aligned with the reference L1TTrack/L1TExtTrack. Hit tables
# L1TSmartPixelsRefitHit<suffix>/L1TSmartPixelsExtRefitHit<suffix> are link
# tables (one row per refit crossing) with a trackIdx crossref into the variant
# track table of the same suffix and a selClusterIdx crossref into
# L1TSmartPixelsCluster (a barrel row; barrel rows are a per-event prefix of
# the unified table, so the row index is also valid there).
#
# Registered PER FILE, by table-name pattern, for every variant present (a
# multi-variant file carries up to 15 activeSP masks x {prompt, Ext}), so no
# config list has to be kept in step with the producer.
# ---------------------------------------------------------------------------
CLUSTER_TABLE = "L1TSmartPixelsCluster"
_SPIX_HIT_RE = re.compile(r"^L1TSmartPixels(Ext)?RefitHit(\w+)$")
_SPIX_TRACK_RE = re.compile(r"^L1TSmartPixels(Ext)?Track(\w+)$")
_OT_STUB_TABLE = "L1TOTStub"
# track-stub table -> its owning track table
_TRACK_STUB_TABLES = {"L1TTrackStub": ("L1TTrack", "L1TrackStubLink"),
                      "L1TExtTrackStub": ("L1TExtTrack", "L1ExtTrackStubLink")}
_HIT_LINK_CLASSES = {}      # track table -> mixin class name


def _hit_link_class(track_table):
    """The refit-hit mixin bound to `track_table`, created once per name."""
    name = _HIT_LINK_CLASSES.get(track_table)
    if name is None:
        name = f"L1SmartPixelsRefitHitLink_{track_table}"
        cls = type(name, (_SmartPixelsRefitHitBase,), {"_track_target": track_table})
        globals()[name] = awkward.mixin_class(behavior)(cls)
        _HIT_LINK_CLASSES[track_table] = name
    return name


def smartpixels_links(fields):
    """(mixins, cross_references) for the SmartPixels and track-stub tables of
    a file with branch names `fields`. A cross-reference is registered only
    when its index column exists, so an older nano without otStubIdx reads
    silently; a present index whose target table is missing still warns."""
    fields = set(fields)
    tables = {f.split("_")[0] for f in fields}
    mixins, xref = {}, {}

    def link(col, target):
        if col in fields:
            xref[col] = target

    for t in sorted(tables):
        m = _SPIX_TRACK_RE.match(t)
        if m:
            mixins[t] = "L1TrackWord"
            continue
        m = _SPIX_HIT_RE.match(t)
        if m:
            trk = f"L1TSmartPixels{m.group(1) or ''}Track{m.group(2)}"
            mixins[t] = _hit_link_class(trk)
            link(f"{t}_trackIdx", trk)
            link(f"{t}_selClusterIdx", CLUSTER_TABLE)
    for stub, (trk, mix) in _TRACK_STUB_TABLES.items():
        if stub in tables:
            mixins[stub] = mix
            link(f"{stub}_trackIdx", trk)
            link(f"{stub}_otStubIdx", _OT_STUB_TABLE)
    return mixins, xref


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
        import warnings as _warnings

        fields = base_form.get("fields", [])
        present = {f.split("_")[0] for f in fields}
        # per-file SmartPixels/track-stub links, on top of the class tables
        mix, xref = smartpixels_links(fields)
        self.mixins = {**type(self).mixins, **mix}
        self.all_cross_references = {**type(self).all_cross_references, **xref}
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
        "L1TTrack": "L1TrackWord",
        "L1TExtTrack": "L1TrackWord",
        "L1HGCCluster": "NanoCollection",
        "L1puppiJetSC4NG": "NanoCollection",
        "L1puppiJetSC4": "NanoCollection",
        "L1DispVertex": "L1DispVtx",
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
        # displaced vertex track-pair crossrefs
        "L1DispVertex_firstIndexTrk": "L1TExtTrack",
        "L1DispVertex_secondIndexTrk": "L1TExtTrack",
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


# ---------------------------------------------------------------------------
# Clusters tier (L1PFTrkNanoSmartPixClusters)
# ---------------------------------------------------------------------------


class _SmartPixelsClusterBase(base.NanoCollection):
    """Untruncated IT clusters.

    A cluster is a detector object that knows nothing about any track; the
    link runs the other way, from a refit hit's selClusterIdx (``hit.cluster``).

    On current nanos the table holds every TBPX cluster, then the ID1-ID4 disc
    clusters, with subdet/layer/disk/side/ring/surface/idxInDet. Barrel-only
    work selects ``surface <= 4``; older files keep the discs in a separate
    L1TSmartPixelsDiscCluster table (ngtagger.data.spix_nano reads both).

    The derived accessors below are the ones that are awkward to recompute
    correctly by hand, because both are per-MODULE quantities and the table is
    flat per event.
    """

    @property
    def n_on_module(self):
        """How many clusters share this cluster's module, broadcast per cluster.

        This is the size of the candidate pool a refit crossing on that module
        faces (measured 20-40 at PU200), as opposed to the post-window
        multiplicity the sidecar reports.
        """
        import awkward as ak
        import numpy as np

        det = self.detId
        counts = ak.num(det)
        flat = ak.to_numpy(ak.flatten(det))
        ev = np.repeat(np.arange(len(counts)), ak.to_numpy(counts))
        key = ev.astype(np.int64) * (1 << 32) + flat.astype(np.int64)
        order = np.argsort(key, kind="stable")
        sk = key[order]
        uniq, inv, cnt = np.unique(sk, return_inverse=True, return_counts=True)
        out = np.empty(len(flat), dtype=np.int32)
        out[order] = cnt[inv]
        return ak.unflatten(out, counts)

    @property
    def charge_rank(self):
        """Rank of this cluster by charge within its module, 0 = highest.

        A SmartPixels sensor cannot read out every cluster in the L1 latency
        budget: the plan is to form outputs only for some number of the
        highest-charge clusters and defer the rest to a Level-1 Accept, at which
        point they are useless for the trigger. So charge rank is the readout
        gate, and "does the cluster the track needs survive rank < N" is a
        first-class efficiency question rather than a diagnostic.
        """
        import awkward as ak
        import numpy as np

        det, q = self.detId, self.charge
        counts = ak.num(det)
        fdet = ak.to_numpy(ak.flatten(det)).astype(np.int64)
        fq = ak.to_numpy(ak.flatten(q)).astype(np.float64)
        ev = np.repeat(np.arange(len(counts)), ak.to_numpy(counts))
        key = ev.astype(np.int64) * (1 << 32) + fdet
        # sort by (module, -charge) so position within a module IS the rank
        order = np.lexsort((-fq, key))
        sk = key[order]
        starts = np.r_[True, sk[1:] != sk[:-1]]
        grp = np.cumsum(starts) - 1
        pos = np.arange(len(sk)) - np.flatnonzero(starts)[grp]
        out = np.empty(len(fdet), dtype=np.int32)
        out[order] = pos
        return ak.unflatten(out, counts)


# awkward.mixin_class registers the behavior under the CLASS NAME, so the class
# must literally be called SmartPixelsCluster to match the mixins entry below --
# registering _SmartPixelsClusterBase would silently give a plain record with no
# charge_rank/n_on_module and no error until first use.
SmartPixelsCluster = awkward.mixin_class(behavior)(
    type("SmartPixelsCluster", (_SmartPixelsClusterBase,), {}))


class L1ClustersNanoSchema(L1NanoSchema):
    """L1NanoSchema plus the untruncated IT cluster table.

    Separate from L1NanoSchema on purpose: the cluster table only exists in the
    L1PFTrkNanoSmartPixClusters tier, and registering its mixin unconditionally
    would make a missing table look like a schema bug rather than the wrong tier.
    """

    mixins = {**L1NanoSchema.mixins, CLUSTER_TABLE: "SmartPixelsCluster"}

    @classmethod
    def behavior(cls):
        return behavior


def require_clusters_tier(events):
    """Fail with an actionable message if this file is not the Clusters tier."""
    if CLUSTER_TABLE not in events.fields:
        raise SystemExit(
            f"this file has no {CLUSTER_TABLE} table, so it is not the Clusters tier.\n"
            "Produce one with:\n"
            "  test/makeSpixConfig.py --pu 200 --tier clusters-truth "
            "--variant digiRefit:1111 --needs-truth -o <out>.py")
    return events[CLUSTER_TABLE]
