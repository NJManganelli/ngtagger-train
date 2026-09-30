"""On-the-fly per-track MVA scores (ngtagger.train.trackmva) as tagger /
vertexing inputs, analogous to refit_bdt_feature.py.

Two levels:

  per track        ``trackmva_per_track(A, mva)`` scores every row of the
                   bundle's base/refit tables in an uproot batch `A` (read with
                   ``trackmva_branches``); ``trackmva_scores_from_files`` does
                   it over files in event batches. Output: an (event, track)
                   awkward record with one field per head (OUTPUTS). Truth
                   columns are not read, so this works on any nano carrying the
                   refit tables.
  per constituent  ``trackmva_per_constituent`` gathers those per-track scores
                   onto the (event, jet, constituent) nesting through the
                   constituent -> track crossref (l1TrackIdx, as nano.load_jets
                   recovers it). Constituents without a track (l1TrackIdx < 0:
                   neutrals) get SENTINEL.

SENTINEL = -1.0 is outside every head's range (probabilities in [0, 1],
log sigma_hat >= log(1 um) = 0), so a consumer can tell "no track" from any
score. Tracks whose refit was not performed are SCORED, not sentinelled: they
are rows of the refit table, were in the MVA training sample, and
spixRefitPerformed is one of the features (unlike the refit-quality BDT, which
the producer never evaluates on them).

The constituent crossref points into the jet pipeline's track table
(load_jets' track_table, L1TExtTrack by default), so the bundle's family must
be the one whose base OR refit table that is (family "ext" for L1TExtTrack;
"canonical_ext" for L1TSmartPixelsExtTrack, where a canonical/coopt nano's
constituents point); a mismatch raises rather than gathering scores of
unrelated rows. Base and refit rows are 1:1 per event (tracks_from_batch
checks the counts), so the scores of either collection index either table:
a bundle holding only the "base" (OT-only) heads, e.g. for the OT-passthrough
mask IIII, is gathered through the same link.

Jet tables. The current SmartPixels nanos carry no jet / jet-constituent
tables. ``missing_jet_tables`` names what is absent; ``require_jet_tables``
raises JetTablesAbsentError with a message saying so (per-track scores
remain available); load_jets' "trackmva" feature group calls it before
reading anything.
"""

from __future__ import annotations

import awkward as ak
import numpy as np

SENTINEL = -1.0
DEFAULT_JET_TABLES = ("L1puppiJetSC4NG", "L1SC4NGJetCands", "L1ExtPuppiCand")


class JetTablesAbsentError(RuntimeError):
    """The nano has no jet / jet-constituent tables to gather onto."""


def missing_jet_tables(keys, tables=DEFAULT_JET_TABLES) -> list[str]:
    """The tables of `tables` with no branch in `keys`."""
    keys = list(keys)
    return [t for t in tables if not any(k.startswith(t + "_") or k == "n" + t for k in keys)]


def require_jet_tables(keys, tables=DEFAULT_JET_TABLES, where: str = "") -> None:
    miss = missing_jet_tables(keys, tables)
    if miss:
        raise JetTablesAbsentError(
            f"{where + ': ' if where else ''}no jet tables {miss} in this nano, so there is no "
            "jet-constituent nesting to gather track-MVA scores onto. The per-track scores are "
            "still available (ngtagger.data.trackmva_feature.trackmva_scores_from_files); "
            "produce the nano with the jet/constituent tables to use the 'trackmva' tagger "
            "feature group.")


def _as_mva(model):
    from ngtagger.train.trackmva import TrackMVA, load_bundle

    return model if isinstance(model, TrackMVA) else load_bundle(model)


def _coll(mva, coll):
    """None -> "refit" when the bundle has it, else its only collection."""
    if coll is None:
        return "refit" if "refit" in mva.colls else mva.colls[0]
    return coll


def trackmva_branches(keys, model) -> list[str]:
    """The branches ``trackmva_per_track`` needs (online columns only)."""
    from ngtagger.train.trackmva import nano_branches

    mva = _as_mva(model)
    return nano_branches(keys, mva.family, mva.config, with_truth=False, with_event_id=False,
                         base_only=mva.colls == ["base"])


def trackmva_per_track(A, model, coll: str | None = None) -> ak.Array:
    """(event, track) record of the heads for every base/refit row of batch `A`."""
    from ngtagger.train.trackmva import OUTPUTS, tracks_from_batch

    mva = _as_mva(model)
    T, ntrk = tracks_from_batch(A, mva.family, mva.config, with_truth=False,
                                base_only=mva.colls == ["base"])
    P = mva.predict(T, _coll(mva, coll))
    return ak.unflatten(ak.zip({h: P[h].astype(np.float32) for h in OUTPUTS}), ntrk)


def trackmva_scores_from_files(files, model, coll: str | None = None, max_events: int | None = None,
                               step_events: int = 50) -> ak.Array:
    """``trackmva_per_track`` over `files` in order, `step_events` at a time."""
    import uproot

    mva = _as_mva(model)
    parts, n_done = [], 0
    for fn in files:
        t = uproot.open(fn)["Events"]
        br = trackmva_branches(t.keys(), mva)
        stop = t.num_entries if max_events is None else min(t.num_entries, max_events - n_done)
        for start in range(0, max(stop, 0), step_events):
            A = t.arrays(br, entry_start=start, entry_stop=min(start + step_events, stop))
            parts.append(trackmva_per_track(A, mva, coll))
        n_done += max(stop, 0)
        if max_events is not None and n_done >= max_events:
            break
    return ak.concatenate(parts, axis=0)


def trackmva_per_constituent(track_scores: ak.Array, cands: ak.Array, nested_idx: ak.Array,
                             link_idx_field: str = "l1TrackIdx", sentinel: float = SENTINEL,
                             n_tracks=None) -> dict:
    """{head: (event, jet, constituent) float32} gathered from the (event,
    track) `track_scores` via the constituents' `link_idx_field`; `nested_idx`
    is load_jets' (event, jet, constituent) candidate index. Constituents with
    no track (index < 0) get `sentinel`. `n_tracks` (per-event row count of
    the table the crossref points into) is checked against the scores when
    given; an index beyond the scored rows raises."""
    from ngtagger.data.nano import _gather2

    ns = ak.num(track_scores, axis=1)
    if n_tracks is not None and not ak.all(ak.Array(n_tracks) == ns):
        raise RuntimeError("track-MVA scores and the crossref'd track table differ in rows per "
                           "event: the scores are of another track table (family) or events")
    ref_idx = _gather2(cands[link_idx_field], nested_idx)
    valid = ref_idx >= 0
    over = ak.any(ak.flatten(ref_idx, axis=None) >= ak.flatten(
        ak.broadcast_arrays(ns, ref_idx)[0], axis=None))
    if bool(over):
        raise IndexError(f"{link_idx_field} points beyond the scored track rows")
    safe = ak.where(valid, ref_idx, 0)
    out = {}
    for h in track_scores.fields:
        vals = _gather2(track_scores[h], safe)
        out[h] = ak.values_astype(ak.where(valid, vals, np.float32(sentinel)), np.float32)
    return out


def trackmva_jet_features(files, events, cands, nested_idx, link_idx_field: str, model,
                          track_table: str = "L1TExtTrack", coll: str | None = None,
                          max_events: int | None = None, prefix: str = "trkmva_") -> dict:
    """The load_jets 'trackmva' group: {prefix + head: (event, jet, constituent)}.
    `events` is load_jets' event record (for the per-event track counts);
    load_jets checks the jet tables exist (``require_jet_tables``) first.
    `coll` None = "refit" when the bundle has it, else its only collection."""
    mva = _as_mva(model)
    if track_table not in (mva.tables["base"], mva.tables["refit"]):
        raise ValueError(f"the constituent crossref points into {track_table}, but the track-MVA "
                         f"bundle scores {mva.tables['base']} / {mva.tables['refit']} rows (family "
                         f"{mva.family!r}); use the bundle of the matching family")
    coll = _coll(mva, coll)
    scores = trackmva_scores_from_files(files, mva, coll, max_events)
    n_tracks = ak.num(events[track_table], axis=1) if track_table in events.fields else None
    G = trackmva_per_constituent(scores, cands, nested_idx, link_idx_field, n_tracks=n_tracks)
    return {prefix + h: v for h, v in G.items()}
