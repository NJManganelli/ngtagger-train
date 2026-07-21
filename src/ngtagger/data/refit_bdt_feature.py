"""On-the-fly Stage-3 refit-quality BDT score as a per-constituent tagger
feature (mem:smartpixels-bdt-as-tagger-feature-directive).

For each jet-constituent track (recovered via l1TrackIdx into the extended
track table), we assemble its 24 REFIT_BDT_FEATURES v1 vector EXACTLY as the
producer / Stage-3 training does (reusing ngtagger.train.refitquality's
build_spec_dataset), evaluate the deployed Stage-3 conifer JSON with the
bit-faithful conifer_json_walk, and gather the resulting per-track score onto
the (event, jet, constituent) nesting. Passthrough / non-refit / neutral
constituents (l1TrackIdx < 0, or refit not performed) receive a 0.0 sentinel.

The refit variant/hit tables are 1:1 row-aligned with the extended reference
track table (verified: nL1TExtTrack == nRefitVariant per event), so the
l1TrackIdx that points into L1TExtTrack also indexes the refit tables.
"""

from __future__ import annotations

import json

import awkward as ak
import numpy as np


def _config_from_track_table(track_table: str, config: str) -> str:
    return config


def refit_bdt_per_constituent(files, events, cands, nested_idx, link_idx_field,
                              config: str, conifer_json: str,
                              track_table: str = "L1TExtTrack",
                              max_events: int | None = None):
    """Return an (event, jet, constituent) float32 awkward array of the Stage-3
    refit-quality conifer score for each constituent's refit track.

    events/cands/nested_idx come from load_jets; `files` is re-read for the
    refit reference/variant/hit tables (they carry columns not loaded into the
    tagger `events`). The per-event track row order of load_refit_tables matches
    L1TExtTrack (the reference), so l1TrackIdx indexes it directly.
    """
    from ngtagger.data.nano import _gather2
    from ngtagger.train.refitquality import (
        build_spec_dataset,
        conifer_json_walk,
        load_refit_tables,
    )

    # Ext track table -> 5-par seed semantics (track_npar=5, use ref d0 as seed).
    ref, var, hits = load_refit_tables(files, config, track_table, max_events)
    X, _y, _names, aux = build_spec_dataset(
        ref, var, hits, config, label="genuine",
        refit_only=False, require_truth=False,
        seed_npar=5, track_npar=5, spec_version=1,
    )
    with open(conifer_json) as f:
        model = json.load(f)
    margin = conifer_json_walk(model, X)                 # per-track raw logit margin
    score = 1.0 / (1.0 + np.exp(-margin.astype(np.float64)))  # sigmoid -> [0,1]
    # zero out tracks whose refit was not performed (producer never scores them)
    score = np.where(aux["refit_mask"].astype(bool), score, 0.0).astype(np.float32)

    # re-nest the flat per-track score back to (event, track)
    counts = ak.to_numpy(ak.num(ref["genuine"]))
    score_by_event = ak.unflatten(ak.Array(score), counts)

    # gather onto constituents via the recovered l1TrackIdx; <0 -> 0.0
    ref_idx = _gather2(cands[link_idx_field], nested_idx)
    valid = ref_idx >= 0
    safe_idx = ak.where(valid, ref_idx, 0)
    vals = _gather2(score_by_event, safe_idx)
    return ak.where(valid, vals, np.float32(0.0))
