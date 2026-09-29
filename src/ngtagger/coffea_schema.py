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
  - TrackingParticle truth links (tpIdx, on every TP-linked table) cross-reference
    L1TTP, with a `tp` accessor; on older key-linked
    nanos the row-convention tpIdx is derived from the key and L1TTP_key
  - candidates get matched_track / matched_cluster / matched_jet accessors
    and uniform track-truth access: `trk_genuine` uses the direct column
    (candidate-only L1PFNano tier) or falls back to the l1TrackIdx
    indirection into the track-table truth (L1PFTrkNano tier)
  - track crossrefs whose target the producer states in the branch doc
    ("... in the <TABLE> table ...", see _DOC_LINKS) follow the doc: in a
    coopt nano the candidate l1TrackIdx points into L1TSmartPixelsTrack /
    L1TSmartPixelsExtTrack, so matched_track returns SmartPixels tracks
    (SmartPixelsTrackLinkWarning). The canonical SmartPixels track tables get
    an `ot_track` accessor (via their l1TrackIdx, the OT track it was built from), and
    L1NanoSchema.candidate_ot_track_via_smartpixels adds the candidate ->
    SmartPixels track -> OT track hop as `matched_ot_track`

Usage:
    from coffea.nanoevents import NanoEventsFactory
    from ngtagger.coffea_schema import L1NanoSchema

    events = NanoEventsFactory.from_root({file: "Events"}, schemaclass=L1NanoSchema).events()
    links = events.L1SC4NGJetCands
    tensor_feats = links.pt_rel  # etc.
"""

from __future__ import annotations

import copy
import re
import warnings

import awkward
import numpy
from coffea.nanoevents import NanoAODSchema, transforms
from coffea.nanoevents.methods import base, candidate, nanoaod, vector
from coffea.nanoevents.util import concat

from ngtagger.data import spix_nano

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
    Concrete subclasses bind `_track_target` per seedlessRefit config/variant."""

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
    """Track -> stub link table (L1TTrackStub, L1TExtTrackStub,
    L1TSmartPixels[Ext]TrackStub): one row per stub of a track, link-only
    (trackIdx, otStubIdx); the stub columns (spix_nano.STUB_COLUMNS) are read
    through ``ot_stub``."""

    _track_target = None

    @property
    def track(self):
        """owning track (via trackIdx)"""
        return self._events()[self._track_target]._apply_global_index(self.trackIdxG)

    @property
    def ot_stub(self):
        """the same stub's L1TOTStub row, with its truth (via otStubIdx)"""
        return self._events()["L1TOTStub"]._apply_global_index(self.otStubIdxG)


def _stub_column(col):
    def get(self):
        return self.ot_stub[col]

    return property(get, doc=f"stub {col} = L1TOTStub[otStubIdx].{col}")


for _col in spix_nano.STUB_COLUMNS:
    setattr(_TrackStubBase, _col, _stub_column(_col))


class _TrackClusterBase(base.NanoCollection):
    """SmartPixels track -> IT cluster link table (L1TSmartPixels[Ext]TrackCluster):
    one row per cluster used by a track (seedlessRefit: the accepted IT hits)."""

    _track_target = None

    @property
    def track(self):
        """owning SmartPixels track (via trackIdx)"""
        return self._events()[self._track_target]._apply_global_index(self.trackIdxG)

    @property
    def cluster(self):
        """the L1TSmartPixelsCluster row (via clusterIdx)"""
        return self._events()[CLUSTER_TABLE]._apply_global_index(self.clusterIdxG)


@awkward.mixin_class(behavior)
class L1DispVtx(base.NanoCollection):
    """GTT displaced vertex: track-pair accessors via first/secondIndexTrk
    (targets bound per file from the branch doc, see _DOC_LINKS)."""

    _first_target = "L1TExtTrack"
    _second_target = "L1TExtTrack"

    @property
    def first_track(self):
        """higher-pt track of the pair"""
        return self._events()[self._first_target]._apply_global_index(self.firstIndexTrkG)

    @property
    def second_track(self):
        """lower-pt track of the pair"""
        return self._events()[self._second_target]._apply_global_index(self.secondIndexTrkG)


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


class _TPLinked(base.NanoCollection):
    """Objects carrying a TrackingParticle truth link into L1TTP."""

    @property
    def tp(self):
        """matched L1TTP row (via tpIdx; None unless the link is a row, i.e. >= 0)"""
        return self._events()[spix_nano.TP_TABLE]._apply_global_index(self.tpIdxG)


@awkward.mixin_class(behavior)
class L1OTStub(_TPLinked):
    """OT stub; `tp` is its principal TP, `tp_inner` / `tp_outer` the owners of its
    inner- / outer-sensor clusters (TTStub clusterRef 0 / 1)."""

    @property
    def tp_inner(self):
        """L1TTP row owning the inner-sensor cluster (via tpIdxInner)"""
        return self._events()[spix_nano.TP_TABLE]._apply_global_index(self.tpIdxInnerG)

    @property
    def tp_outer(self):
        """L1TTP row owning the outer-sensor cluster (via tpIdxOuter)"""
        return self._events()[spix_nano.TP_TABLE]._apply_global_index(self.tpIdxOuterG)


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
class L1TrackWord(L1TrackWordDecoded, _TPLinked):
    pass


class _SmartPixelsTrackBase(L1TrackWord):
    """Canonical (coopt) SmartPixels track table, linked back to the OT track
    it was built from via l1TrackIdx."""

    _ot_track_target = None

    @property
    def ot_track(self):
        """OT track it was built from (via l1TrackIdx; masked where -1, not built from
        an OT track, or -2, OT track product unavailable)"""
        return self._events()[self._ot_track_target]._apply_global_index(self.l1TrackIdxG)


@awkward.mixin_class(behavior)
class L1SmartPixelsTrack(_SmartPixelsTrackBase):
    _ot_track_target = "L1TTrack"


@awkward.mixin_class(behavior)
class L1SmartPixelsExtTrack(_SmartPixelsTrackBase):
    _ot_track_target = "L1TExtTrack"


# ---------------------------------------------------------------------------
# SmartPixels seedlessRefit variant tables (Phase-3 tier-2 refit study).
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
# track -> stub / cluster link table: <track table>Stub, <track table>Cluster
_TRACK_LINK_RE = re.compile(r"^(L1T\w*Track\w*?)(Stub|Cluster)$")
# canonical SmartPixels track table -> the OT track table its l1TrackIdx
# points into (default when the branch doc does not name it)
_SPIX_OT_TRACK_TABLES = {"L1TSmartPixelsTrack": ("L1TTrack", "L1SmartPixelsTrack"),
                         "L1TSmartPixelsExtTrack": ("L1TExtTrack", "L1SmartPixelsExtTrack")}
_LINK_BASES = {"RefitHit": _SmartPixelsRefitHitBase, "Stub": _TrackStubBase,
               "Cluster": _TrackClusterBase}


def _link_class(kind, track_table):
    """The `kind` link-table mixin (RefitHit, Stub, Cluster) bound to
    `track_table`, created once per (kind, name)."""
    name = f"L1{kind}Link_{track_table}"
    if name not in globals():
        cls = type(name, (_LINK_BASES[kind],), {"_track_target": track_table})
        globals()[name] = awkward.mixin_class(behavior)(cls)
    return name


def _hit_link_class(track_table):
    """The refit-hit mixin bound to `track_table`."""
    return _link_class("RefitHit", track_table)


# Key-linked nanos (spix_nano.tp_link_convention != "row"): the row-convention
# link is synthesized per file from the key branch (+ L1TTP key column) by these
# transforms, looked up by name in coffea.nanoevents.transforms. They yield the
# jagged array, so the list node's key is what local2global indexes with.
def _tp_rows_from_keys(stack):
    tp_key = awkward.Array(stack.pop())
    key = awkward.Array(stack.pop())
    stack.append(spix_nano.tp_rows_from_keys(key, tp_key))


def _tp_rows_unjoined(stack):
    stack.append(spix_nano.tp_rows_from_keys(awkward.Array(stack.pop())))


transforms.ngtagger_tp_rows_from_keys = _tp_rows_from_keys
transforms.ngtagger_tp_rows_unjoined = _tp_rows_unjoined


def tp_links(fields):
    """[(table_col, (link branch, L1TTP key branch or None, convention))] for
    every TrackingParticle link of a file with branch names `fields`."""
    fields = set(fields)
    out = []
    for t in sorted({f.split("_")[0] for f in fields}):
        for col in spix_nano.TP_LINK_COLUMNS:
            src = spix_nano.tp_link_source(fields, t, col, required=False)
            if src is not None:
                out.append((f"{t}_{col}", src))
    return out


def with_tp_rows(field_names, contents):
    """(field_names, contents) with every TP link in the row convention: on a
    key-linked file each link gets a form computing tpIdx from its key
    (replacing a v1 key stored under the *Idx name). Unchanged on row files."""
    names, contents = list(field_names), list(contents)
    forms = dict(zip(names, contents))
    for name, (br, tk, conv) in tp_links(names):
        if conv == "row":
            continue
        form = copy.deepcopy(forms[br])
        if tk is None:
            key = concat(forms[br]["form_key"], "!ngtagger_tp_rows_unjoined")
        else:
            key = concat(forms[br]["form_key"], forms[tk]["form_key"], "!ngtagger_tp_rows_from_keys")
        form["form_key"] = key
        form["content"].update(form_key=concat(key, "!content"), primitive="int32", itemsize=4)
        form["content"]["parameters"] = {
            "__doc__": f"tpIdx derived from {br} (a TP key): >= 0 row in L1TTP; -1 not matched; -2 "
                       "no L1TTP table; <= -3 pruned from L1TTP, -3 - tpIdx = the key",
            "typename": "int32_t[]"}
        if name in forms:
            contents[names.index(name)] = form
        else:
            names.append(name)
            contents.append(form)
    return names, contents


def smartpixels_links(fields):
    """(mixins, cross_references) for the SmartPixels and track-stub tables
    and the TrackingParticle links (tpIdx -> L1TTP) of a file with branch
    names `fields`. A cross-reference is registered only
    when its index column exists, so an older nano without otStubIdx reads
    silently; a present index whose target table is missing still warns."""
    fields = set(fields)
    tables = {f.split("_")[0] for f in fields}
    mixins, xref = {}, {}

    def link(col, target):
        if col in fields:
            xref[col] = target

    for t in sorted(tables):
        m = _TRACK_LINK_RE.match(t)
        if m:
            trk, kind = m.groups()
            mixins[t] = _link_class(kind, trk)
            link(f"{t}_trackIdx", trk)
            link(f"{t}_otStubIdx", _OT_STUB_TABLE)
            link(f"{t}_clusterIdx", CLUSTER_TABLE)
            continue
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
    for spx, (trk, mix) in _SPIX_OT_TRACK_TABLES.items():
        if spx in tables:
            mixins[spx] = mix
            link(f"{spx}_l1TrackIdx", trk)
    for name, _ in tp_links(fields):
        xref[name] = spix_nano.TP_TABLE
    for side in ("Inner", "Outer"):
        link(f"{_OT_STUB_TABLE}_tpIdx{side}", spix_nano.TP_TABLE)
    return mixins, xref


class _L1PFCandBase(candidate.PtEtaPhiMCandidate, base.NanoCollection):
    """L1 PF/Puppi candidates: matched-object accessors + uniform truth."""

    _track_target = None
    _ot_track_target = None
    _jet_target = None

    @property
    def matched_track(self):
        """underlying track (via l1TrackIdx; masked where -1/-2): an OT track,
        or a SmartPixels track in a coopt nano (see _DOC_LINKS)"""
        return self._events()[self._track_target]._apply_global_index(self.l1TrackIdxG)

    @property
    def matched_ot_track(self):
        """underlying OT track, through the SmartPixels track when
        matched_track is one (candidate -> SmartPixels track -> OT track via
        both l1TrackIdx hops, masked where either is < 0); the direct link
        otherwise. Needs L1NanoSchema.candidate_ot_track_via_smartpixels."""
        if "otTrackIdxG" not in self.fields:
            raise AttributeError(
                "matched_ot_track needs L1NanoSchema.candidate_ot_track_via_smartpixels = True "
                "(and, for SmartPixels-linked candidates, the SmartPixels track l1TrackIdx column)")
        return self._events()[self._ot_track_target]._apply_global_index(self.otTrackIdxG)

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
    _ot_track_target = "L1TTrack"
    _jet_target = "L1puppiJetSC4"


behavior.update(awkward._util.copy_behaviors("PtEtaPhiMCandidate", "L1ExtPuppiCandidate", behavior))


@awkward.mixin_class(behavior)
class L1ExtPuppiCandidate(_L1PFCandBase):
    _track_target = "L1TExtTrack"
    _ot_track_target = "L1TExtTrack"
    _jet_target = "L1puppiJetSC4NG"


def _wire_candidate(name):
    """Vector projections of the candidate mixin `name`."""
    arr = globals()[f"{name}Array"]  # created by awkward.mixin_class
    rec = globals()[f"{name}Record"]
    arr.ProjectionClass2D = vector.TwoVectorArray
    arr.ProjectionClass3D = vector.ThreeVectorArray
    arr.ProjectionClass4D = arr
    arr.MomentumClass = vector.LorentzVectorArray
    rec.ProjectionClass2D = vector.TwoVectorRecord
    rec.ProjectionClass3D = vector.ThreeVectorRecord
    rec.ProjectionClass4D = rec
    rec.MomentumClass = vector.LorentzVectorRecord


for _name in ("L1PuppiCandidate", "L1ExtPuppiCandidate"):
    _wire_candidate(_name)


def _bound_mixin(mixin, attr, target):
    """The mixin `mixin` with class attribute `attr` (a crossref target) set
    to `target`: `mixin` itself when it already is, else a subclass created
    once per (mixin, attr, target), like the refit-hit link classes."""
    cls = globals()[mixin]
    if getattr(cls, attr) == target:
        return mixin
    name = f"{mixin}_{attr.strip('_')}_{target}"
    if name not in globals():
        is_cand = issubclass(cls, candidate.PtEtaPhiMCandidate)
        if is_cand:
            behavior.update(awkward._util.copy_behaviors("PtEtaPhiMCandidate", name, behavior))
        globals()[name] = awkward.mixin_class(behavior)(type(name, (cls,), {attr: target}))
        if is_cand:
            _wire_candidate(name)
    return name


# Crossrefs whose target the producer states in the branch doc -> the mixin
# class attribute bound to that target. A coopt nano points the candidate
# l1TrackIdx (and possibly the displaced-vertex track pair) into the
# SmartPixels track tables; the doc names the table either way, so the target
# is read from it. Without a parseable doc (older nanos) the schema default
# (all_cross_references / smartpixels_links) stands.
_DOC_LINKS = {
    "L1PuppiCand_l1TrackIdx": "_track_target",
    "L1ExtPuppiCand_l1TrackIdx": "_track_target",
    "L1DispVertex_firstIndexTrk": "_first_target",
    "L1DispVertex_secondIndexTrk": "_second_target",
    "L1TSmartPixelsTrack_l1TrackIdx": "_ot_track_target",
    "L1TSmartPixelsExtTrack_l1TrackIdx": "_ot_track_target",
}
# candidate -> track links, for the SmartPixels warning and the indirect hop
_CAND_TRACK_LINKS = ("L1PuppiCand_l1TrackIdx", "L1ExtPuppiCand_l1TrackIdx")
# "index of the ... track in the <TABLE> table (...)" (candidate, displaced
# vertex) / "row of the <TABLE> track this SmartPixels track was built from"
_DOC_TARGET_RE = re.compile(r"\bin the (\w+) table\b|^row of the (\w+) track\b")


def doc_target(form):
    """Target table named by a branch form's doc (coffea keeps the ROOT branch
    title as __doc__, on the content of a jagged branch), or None."""
    for f in (form, form.get("content") or {}):
        m = _DOC_TARGET_RE.search((f.get("parameters") or {}).get("__doc__") or "")
        if m:
            return m.group(1) or m.group(2)
    return None


class SmartPixelsTrackLinkWarning(UserWarning):
    """A candidate's track crossref points into a SmartPixels track table."""


# candidate -> SmartPixels track -> OT track: compose the two global indices
# (looked up by name in coffea.nanoevents.transforms, like the TP-row ones)
def _compose_global_index_form(first, second):
    form = copy.deepcopy(first)
    form["content"]["form_key"] = concat(
        first["content"]["form_key"], second["content"]["form_key"],
        "!ngtagger_compose_global_index")
    form["content"]["parameters"] = {
        "__doc__": "global OT track index through the SmartPixels track (-1 if either hop is < 0)"}
    return form


def _compose_global_index(stack):
    second = transforms.ensure_array(stack.pop())
    first = transforms.ensure_array(stack.pop())
    out = numpy.full(len(first), -1, dtype=numpy.int64)
    ok = first >= 0  # local2global already masked < 0 and out-of-range rows
    out[ok] = second[first[ok]]
    stack.append(out)


transforms.ngtagger_compose_global_index = _compose_global_index


class L1NanoSchema(NanoAODSchema):
    """NanoAODSchema extended with the L1TrkNano/L1PFNano/L1PFTrkNano
    collections and cross-references.

    Warning semantics across tiers: coffea's stock behavior warns for every
    unresolvable cross-reference, including when the *source* collection is
    entirely absent (a different tier, not a problem). This subclass keeps
    the warning only when the source collection is present but the index
    column or the target table is missing (a genuinely partial file).

    Doc-resolved targets: the crossrefs in _DOC_LINKS take their target from
    the branch doc when it names one (all_cross_references is the default for
    nanos whose doc does not). A candidate l1TrackIdx naming a SmartPixels
    track table emits a SmartPixelsTrackLinkWarning.

    ``candidate_ot_track_via_smartpixels`` (class-level, like
    ``warn_missing_crossrefs``; set it on the class or a subclass): when true,
    candidates get ``matched_ot_track``, the OT track reached through the
    SmartPixels track's l1TrackIdx (the direct l1TrackIdx link when the
    candidate already points at an OT track). If the SmartPixels track table
    has no l1TrackIdx column this warns and matched_ot_track stays
    unavailable."""

    candidate_ot_track_via_smartpixels = False  # If True, synthesize <cand>_otTrackIdxG

    def __init__(self, base_form, *args, **kwargs):
        fields = base_form.get("fields", [])
        forms = dict(zip(fields, base_form.get("contents", [])))
        present = {f.split("_")[0] for f in fields}
        # per-file SmartPixels/track-stub links, on top of the class tables
        mix, xref = smartpixels_links(fields)
        mixins = {**type(self).mixins, **mix}
        xrefs = {**type(self).all_cross_references, **xref}
        for col, attr in _DOC_LINKS.items():
            if col in forms and col in xrefs:
                xrefs[col] = doc_target(forms[col]) or xrefs[col]
                src = col.split("_")[0]
                mixins[src] = _bound_mixin(mixins[src], attr, xrefs[col])
        special = dict(type(self).special_items)
        for col in _CAND_TRACK_LINKS:
            if col not in forms:
                continue
            cand, target = col.split("_")[0], xrefs[col]
            if target.startswith("L1TSmartPixels"):
                warnings.warn(
                    f"{col} points into {target} (per its branch doc): {cand}.matched_track "
                    "returns SmartPixels tracks, not OT tracklet tracks. For the OT track, set "
                    "L1NanoSchema.candidate_ot_track_via_smartpixels = True and use "
                    f"{cand}.matched_ot_track.", SmartPixelsTrackLinkWarning, stacklevel=2)
            if not self.candidate_ot_track_via_smartpixels:
                continue
            hop = f"{target}_l1TrackIdx"
            if hop in xrefs:
                special[f"{cand}_otTrackIdxG"] = (_compose_global_index_form, (col + "G", hop + "G"))
                ot = xrefs[hop]
            elif target.startswith("L1TSmartPixels"):
                warnings.warn(
                    f"candidate_ot_track_via_smartpixels: {col} points into {target}, which has no "
                    f"l1TrackIdx column; {cand}.matched_ot_track is unavailable.",
                    SmartPixelsTrackLinkWarning, stacklevel=2)
                continue
            else:  # already an OT track: the direct link
                special[f"{cand}_otTrackIdxG"] = (copy.deepcopy, (col + "G",))
                ot = target
            mixins[cand] = _bound_mixin(mixins[cand], "_ot_track_target", ot)
        self.mixins, self.all_cross_references, self.special_items = mixins, xrefs, special
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            super().__init__(base_form, *args, **kwargs)
        for w in caught:
            msg = str(w.message)
            m = re.match(r"Missing cross-reference index for (\w+) => (\w+)", msg)
            if m and m.group(1).split("_")[0] not in present:
                continue  # whole source collection absent: silent
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)

    def _build_collections(self, field_names, input_contents):
        return super()._build_collections(*with_tp_rows(field_names, input_contents))

    mixins = {
        **NanoAODSchema.mixins,
        "L1SC4NGJetCands": "L1SC4NGJetCandLink",
        "L1SC4JetCands": "L1SC4JetCandLink",
        "L1PuppiCand": "L1PuppiCandidate",
        "L1ExtPuppiCand": "L1ExtPuppiCandidate",
        "L1PFCand": "PtEtaPhiMCandidate",
        "L1TTrack": "L1TrackWord",
        "L1TExtTrack": "L1TrackWord",
        "L1TOTStub": "L1OTStub",
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
        # (the l1TrackIdx and these targets follow the branch doc, _DOC_LINKS)
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


class _SmartPixelsClusterBase(_TPLinked):
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
            "--variant seedlessRefit:1111 --needs-truth -o <out>.py")
    return events[CLUSTER_TABLE]
