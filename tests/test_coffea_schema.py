"""Schema tests on synthetic nano files: full tier (PFTrk), candidate-only
tier (PF, direct truth columns, missing track tables -> warning), and
track-only tier (no candidate collections -> silent)."""

import warnings

import awkward as ak
import numpy as np
import pytest

uproot = pytest.importorskip("uproot")
pytest.importorskip("coffea")


def _write(tmp_path, name, include_tracks=True, include_cands=True, cand_truth_cols=False):
    """Two events, two jets each, two constituents per jet."""
    tree = {
        "run": np.array([1, 1], dtype=np.uint32),
        "luminosityBlock": np.array([1, 1], dtype=np.uint32),
        "event": np.array([1, 2], dtype=np.uint64),
    }
    if include_cands:
        cand = {
            "pt": [[10.0, 5.0, 8.0, 3.0]] * 2,
            "eta": [[0.1, 0.2, -0.3, -0.4]] * 2,
            "phi": [[0.0, 3.1, -1.0, 1.0]] * 2,
            "mass": [[0.14, 0.0, 0.14, 0.0]] * 2,
            "charge": [[1, 0, -1, 0]] * 2,
            "l1TrackIdx": [[0, -1, 1, -1]] * 2,
            "hgcClusterIdx": [[-1, 0, -1, -1]] * 2,
            "jetIdx": [[0, 0, 1, 1]] * 2,
        }
        if cand_truth_cols:
            cand["trkGenuine"] = [[True, False, False, False]] * 2
            cand["trkTpPt"] = [[9.5, -1.0, -1.0, -1.0]] * 2
        tree["L1ExtPuppiCand"] = ak.zip({k: ak.Array(v) for k, v in cand.items()})
        tree["L1puppiJetSC4NG"] = ak.zip({
            "pt": [[20.0, 12.0]] * 2, "eta": [[0.15, -0.35]] * 2, "phi": [[0.05, -0.5]] * 2,
        })
        tree["L1SC4NGJetCands"] = ak.zip({
            "jetIdx": [[0, 0, 1, 1]] * 2,
            "candIdx": [[0, 1, 2, 3]] * 2,
            "slot": [[0, 1, 0, 1]] * 2,
            "inTagger": [[True, True, True, True]] * 2,
        })
        tree["L1HGCCluster"] = ak.zip({"hOverE": [[0.1]] * 2, "sigmaRRTot": [[0.02]] * 2})
    if include_tracks:
        tree["L1TExtTrack"] = ak.zip({
            "pt": [[9.0, 7.5]] * 2, "eta": [[0.1, -0.3]] * 2, "tanL": [[0.2, -0.6]] * 2,
            "genuine": [[True, False]] * 2, "tpPt": [[9.5, -1.0]] * 2,
        })
    # write NanoAOD-shaped branches: shared n<collection> counters + underscores
    f = uproot.recreate(tmp_path / name)
    types = {k: (v.type if isinstance(v, ak.Array) else v.dtype) for k, v in tree.items()}
    f.mktree("Events", types,
             counter_name=lambda n: "n" + n,
             field_name=lambda outer, inner: inner if outer == "" else outer + "_" + inner)
    f["Events"].extend(tree)
    f.close()
    return str(tmp_path / name)


def _events(path, **kwargs):
    from coffea.nanoevents import NanoEventsFactory

    from ngtagger.coffea_schema import L1NanoSchema

    return NanoEventsFactory.from_root({path: "Events"}, schemaclass=L1NanoSchema, **kwargs).events()


def test_full_tier_crossrefs(tmp_path):
    events = _events(_write(tmp_path, "pftrk.root"))
    links = events.L1SC4NGJetCands

    # link accessors
    assert ak.all(links.jet.pt[0] == [20.0, 20.0, 12.0, 12.0])
    assert ak.all(links.cand.pt[0] == [10.0, 5.0, 8.0, 3.0])
    ptrel = links.pt_rel[0]
    assert abs(ptrel[0] - 10.0 / 20.0) < 1e-6
    dphi = links.dphi[0]
    assert abs(dphi[1] - (3.1 - 0.05)) < 1e-5
    assert ak.all(abs(links.dphi) <= np.pi)  # wrapping convention

    # candidate crossrefs incl. -1 masking
    trk = events.L1ExtPuppiCand.matched_track
    assert trk.pt[0][0] == 9.0
    assert ak.is_none(trk.pt[0], axis=0)[1]  # l1TrackIdx == -1

    # truth via indirection (no direct trkGenuine columns in this tier)
    genuine = events.L1ExtPuppiCand.trk_genuine
    assert genuine[0][0] == True  # noqa: E712
    assert ak.is_none(genuine[0], axis=0)[1]


def test_candidate_only_tier_warns_and_direct_truth(tmp_path):
    path = _write(tmp_path, "pf.root", include_tracks=False, cand_truth_cols=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        events = _events(path)
    assert any("cross-ref" in str(w.message).lower() or "L1TExtTrack" in str(w.message)
               for w in caught), [str(w.message) for w in caught]

    # direct truth columns present -> no indirection needed
    genuine = events.L1ExtPuppiCand.trk_genuine
    assert ak.all(genuine[0] == [True, False, False, False])
    assert events.L1ExtPuppiCand.trk_tp_pt[0][0] == pytest.approx(9.5)


def test_track_only_tier_silent(tmp_path):
    path = _write(tmp_path, "trk.root", include_cands=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        events = _events(path)
    l1_warnings = [w for w in caught if "L1" in str(w.message)]
    assert not l1_warnings, [str(w.message) for w in l1_warnings]
    assert ak.all(events.L1TExtTrack.pt[0] == [9.0, 7.5])


def test_jet_constituents_grouping(tmp_path):
    from ngtagger.coffea_schema import jet_constituents

    events = _events(_write(tmp_path, "group.root"))
    nested = jet_constituents(events)
    assert ak.to_list(ak.num(nested, axis=2)[0]) == [2, 2]
    assert nested.cand.pt[0][1][0] == 8.0
