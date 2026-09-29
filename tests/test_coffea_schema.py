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


def _write_smartpixels(tmp_path, name):
    """Reference L1TTrack + one seedlessRefit variant track table (row-aligned)
    + its per-hit link table with a trackIdx crossref. Two events, 3 tracks."""
    tree = {
        "run": np.array([1, 1], dtype=np.uint32),
        "luminosityBlock": np.array([1, 1], dtype=np.uint32),
        "event": np.array([1, 2], dtype=np.uint64),
    }
    tree["L1TTrack"] = ak.zip({
        "pt": [[10.0, 5.0, 8.0]] * 2, "genuine": [[True, False, True]] * 2,
        "hwTanl": [[100, 200, 300]] * 2,
    })
    tree["L1TSmartPixelsTrackSeedlessRefitAAAA"] = ak.zip({
        "pt": [[10.1, 5.2, 8.05]] * 2, "hwTanl": [[100, 200, 300]] * 2,
        "spixRefitPerformed": [[True, False, True]] * 2,
        "spixNKFUpdates": [[2, 0, 3]] * 2,
    })
    tree["L1TSmartPixelsRefitHitSeedlessRefitAAAA"] = ak.zip({
        "trackIdx": [[0, 0, 2]] * 2, "layer": [[1, 2, 1]] * 2,
        "pullX": [[0.3, -0.2, 0.1]] * 2, "hitAccepted": [[True, True, True]] * 2,
    })
    f = uproot.recreate(tmp_path / name)
    types = {k: (v.type if isinstance(v, ak.Array) else v.dtype) for k, v in tree.items()}
    f.mktree("Events", types, counter_name=lambda n: "n" + n,
             field_name=lambda outer, inner: inner if outer == "" else outer + "_" + inner)
    f["Events"].extend(tree)
    f.close()
    return str(tmp_path / name)


def test_smartpixels_track_and_hit_crossref(tmp_path):
    events = _events(_write_smartpixels(tmp_path, "spx.root"))
    trk = events.L1TSmartPixelsTrackSeedlessRefitAAAA
    # track-word hw decode reused from L1TrackWord mixin
    assert hasattr(trk, "tanlFromHw")
    assert ak.all(trk.spixRefitPerformed[0] == [True, False, True])

    # per-hit link table resolves trackIdx crossref back to the variant track
    hit = events.L1TSmartPixelsRefitHitSeedlessRefitAAAA
    linked = hit.track
    assert ak.all(linked.pt[0] == [10.1, 10.1, 8.05])  # trackIdx 0,0,2
    assert ak.all(abs(hit.pullX[0] - np.array([0.3, -0.2, 0.1])) < 1e-6)


def test_jet_constituents_grouping(tmp_path):
    from ngtagger.coffea_schema import jet_constituents

    events = _events(_write(tmp_path, "group.root"))
    nested = jet_constituents(events)
    assert ak.to_list(ak.num(nested, axis=2)[0]) == [2, 2]
    assert nested.cand.pt[0][1][0] == 8.0


def _write_smartpixels_multi(tmp_path, name):
    """Two refit variants (prompt IIIA, Ext AAAA), a unified cluster table,
    track stubs with otStubIdx into L1TOTStub. One event."""
    tree = {
        "run": np.array([1], dtype=np.uint32),
        "luminosityBlock": np.array([1], dtype=np.uint32),
        "event": np.array([1], dtype=np.uint64),
        "L1TTrack": ak.zip({"pt": [[10.0, 5.0]]}),
        "L1TExtTrack": ak.zip({"pt": [[11.0]]}),
        "L1TSmartPixelsTrackSeedlessRefitIIIA": ak.zip({"pt": [[10.1, 5.1]], "hwTanl": [[1, 2]]}),
        "L1TSmartPixelsExtTrackSeedlessRefitAAAA": ak.zip({"pt": [[11.1]], "hwTanl": [[3]]}),
        "L1TSmartPixelsRefitHitSeedlessRefitIIIA": ak.zip({
            "trackIdx": [[1, 0]], "detId": [[12, 11]], "selClusterIdx": [[2, -1]]}),
        "L1TSmartPixelsExtRefitHitSeedlessRefitAAAA": ak.zip({
            "trackIdx": [[0]], "detId": [[10]], "selClusterIdx": [[0]]}),
        "L1TSmartPixelsCluster": ak.zip({
            "detId": [[10, 11, 12, 20]], "surface": [[1, 2, 4, 11]], "subdet": [[1, 1, 1, 2]]}),
        "L1TOTStub": ak.zip({"detId": [[500, 501, 502]], "bend": [[1.0, 2.0, 3.0]]}),
        "L1TTrackStub": ak.zip({"trackIdx": [[0, 1]], "otStubIdx": [[2, 0]]}),
    }
    f = uproot.recreate(tmp_path / name)
    types = {k: (v.type if isinstance(v, ak.Array) else v.dtype) for k, v in tree.items()}
    f.mktree("Events", types, counter_name=lambda n: "n" + n,
             field_name=lambda outer, inner: inner if outer == "" else outer + "_" + inner)
    f["Events"].extend(tree)
    f.close()
    return str(tmp_path / name)


def test_smartpixels_links_generic(tmp_path):
    from ngtagger.coffea_schema import smartpixels_links

    path = _write_smartpixels_multi(tmp_path, "multi.root")
    fields = uproot.open(f"{path}:Events").keys()
    mix, xref = smartpixels_links(fields)
    # every variant present is registered, whatever its suffix
    assert xref["L1TSmartPixelsRefitHitSeedlessRefitIIIA_trackIdx"] == "L1TSmartPixelsTrackSeedlessRefitIIIA"
    assert xref["L1TSmartPixelsExtRefitHitSeedlessRefitAAAA_trackIdx"] == \
        "L1TSmartPixelsExtTrackSeedlessRefitAAAA"
    assert xref["L1TSmartPixelsRefitHitSeedlessRefitIIIA_selClusterIdx"] == "L1TSmartPixelsCluster"
    assert xref["L1TTrackStub_otStubIdx"] == "L1TOTStub"
    assert xref["L1TTrackStub_trackIdx"] == "L1TTrack"
    assert mix["L1TSmartPixelsExtTrackSeedlessRefitAAAA"] == "L1TrackWord"
    # no index column -> no crossref
    assert "L1TExtTrackStub_otStubIdx" not in smartpixels_links(["L1TExtTrackStub_trackIdx"])[1]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ev = _events(path)
    assert not [w for w in caught if "cross-ref" in str(w.message)], \
        [str(w.message) for w in caught]
    h = ev.L1TSmartPixelsRefitHitSeedlessRefitIIIA
    assert ak.all(h.track.pt[0] == [5.1, 10.1])
    assert h.cluster.detId[0][0] == 12 and ak.is_none(h.cluster.detId[0], axis=0)[1]
    assert ev.L1TSmartPixelsExtRefitHitSeedlessRefitAAAA.track.pt[0][0] == pytest.approx(11.1)
    st = ev.L1TTrackStub
    assert st.detId.tolist() == [[502, 500]]           # L1TOTStub[otStubIdx]
    assert ak.all(st.track.pt[0] == [10.0, 5.0])


def test_real_file_crossrefs():
    """On a real nano: SPIX_NANO=/path/to/file.root pytest tests/test_coffea_schema.py
    -k real_file [SPIX_NANO_EVENTS=3]. Every refit hit table's trackIdx and
    selClusterIdx must land on a row with the same detId; every link-table
    otStubIdx / clusterIdx must land on a row."""
    import os

    path = os.environ.get("SPIX_NANO")
    if not path:
        pytest.skip("set SPIX_NANO to a nano to check its cross-references")
    from coffea.nanoevents import NanoEventsFactory

    from ngtagger.coffea_schema import L1ClustersNanoSchema, smartpixels_links

    nev = int(os.environ.get("SPIX_NANO_EVENTS", "3"))
    fields = uproot.open(f"{path}:Events").keys()
    _, xref = smartpixels_links(fields)
    ev = NanoEventsFactory.from_root({path: "Events"}, schemaclass=L1ClustersNanoSchema,
                                     entry_stop=nev).events()
    checked = []
    for col, target in sorted(xref.items()):
        src, idx = col.split("_", 1)
        tab = ev[src]
        ok = tab[idx] >= 0
        if idx in ("tpIdx", "spixMatchedTpIdx"):
            truth_pt = next((c for c in ("tp_pt", "tpPt") if c in tab.fields), None)
            if truth_pt is not None:
                assert ak.all(abs(tab.tp.pt[ok] - tab[truth_pt][ok]) <= 1e-2 * tab.tp.pt[ok]), col
            checked.append(f"{col} -> {target}: {int(ak.sum(ok))} links")
            continue
        linked = tab.cluster if idx in ("selClusterIdx", "clusterIdx") else (
            tab.ot_stub if idx == "otStubIdx" else tab.ot_track if idx == "l1TrackIdx" else tab.track)
        if idx == "l1TrackIdx":   # every SmartPixels track has its OT track
            assert ak.all(ok) and not ak.any(ak.is_none(linked.pt, axis=1)), col
        if idx == "selClusterIdx":
            assert ak.all(linked.detId[ok] == tab.detId[ok]), col
        if idx in ("otStubIdx", "clusterIdx"):   # link-only tables: every link lands on a row
            assert not ak.any(ak.is_none(linked.detId[ok], axis=1)), col
        if idx == "trackIdx":
            assert ak.all(ak.num(linked.pt[ok], axis=1) == ak.num(tab[idx][ok], axis=1)), col
        checked.append(f"{col} -> {target}: {int(ak.sum(ok))} links")
    print("\n".join(checked))
    assert checked, "no SmartPixels / track-stub cross-references in this file"


def _write_tp_links(tmp_path, name, convention):
    """Two events of TP-linked tables in one link convention: "row" (tpIdx into
    L1TTP), "key" (cnn_nano v2-v4: tpKey + L1TTP_key) or "key_v1" (keys under
    the tpIdx names + L1TTP_idx). L1TTP keeps keys 3, 8 | 9 (5 and 7 pruned)."""
    keys = {"L1TTrack": [[3, -1], [9]], "L1TOTStub": [[8, 5, -1], [7]],
            "L1TSmartPixelsCluster": [[3, 3, 5], [9, -1]],
            "L1TSmartPixelsTrackSeedlessRefitAAAA": [[3, -1], [7]]}
    rows = {"L1TTrack": [[0, -1], [0]], "L1TOTStub": [[1, -8, -1], [-10]],
            "L1TSmartPixelsCluster": [[0, 0, -8], [0, -1]],
            "L1TSmartPixelsTrackSeedlessRefitAAAA": [[0, -1], [-10]]}
    link = rows if convention == "row" else keys
    tp_col = "key" if convention == "key" else "idx"
    tree = {"run": np.array([1, 1], dtype=np.uint32),
            "luminosityBlock": np.array([1, 1], dtype=np.uint32),
            "event": np.array([1, 2], dtype=np.uint64)}
    ttp = {"pt": [[30.0, 80.0], [90.0]]}
    if convention != "row":
        ttp[tp_col] = [[3, 8], [9]]
    tree["L1TTP"] = ak.zip(ttp)
    for t, v in link.items():
        col = "spixMatchedTp" if t.startswith("L1TSmartPixelsTrack") else "tp"
        col += "Key" if convention == "key" else "Idx"
        tree[t] = ak.zip({"pt": [[1.0] * len(e) for e in v], col: v})
    f = uproot.recreate(tmp_path / name)
    types = {k: (v.type if isinstance(v, ak.Array) else v.dtype) for k, v in tree.items()}
    f.mktree("Events", types, counter_name=lambda n: "n" + n,
             field_name=lambda outer, inner: inner if outer == "" else outer + "_" + inner)
    f["Events"].extend(tree)
    f.close()
    return str(tmp_path / name), rows


@pytest.mark.parametrize("convention", ["row", "key", "key_v1"])
def test_tp_links_every_convention(tmp_path, convention):
    from coffea.nanoevents import NanoEventsFactory

    from ngtagger.coffea_schema import L1ClustersNanoSchema, smartpixels_links

    path, rows = _write_tp_links(tmp_path, f"tp_{convention}.root", convention)
    _, xref = smartpixels_links(uproot.open(f"{path}:Events").keys())
    for t in ("L1TTrack", "L1TOTStub", "L1TSmartPixelsCluster"):
        assert xref[f"{t}_tpIdx"] == "L1TTP"
    assert xref["L1TSmartPixelsTrackSeedlessRefitAAAA_spixMatchedTpIdx"] == "L1TTP"
    ev = NanoEventsFactory.from_root({path: "Events"}, schemaclass=L1ClustersNanoSchema).events()
    pt = [[30.0, 80.0], [90.0]]
    for t, want in rows.items():
        tab = ev[t]
        got = tab.spixMatchedTpIdx if t.startswith("L1TSmartPixelsTrack") else tab.tpIdx
        assert got.tolist() == want, t
        assert tab.tp.pt.tolist() == [[pt[e][r] if r >= 0 else None for r in ev_rows]
                                      for e, ev_rows in enumerate(want)], t


_TRK_DOC = ("index of the underlying track in the {} table (-1 if this candidate has no "
            "resolvable track; -2 if the track/PFTrack product is unavailable)")
_SEED_DOC = ("row of the {0} track this SmartPixels track was seeded from (-1 if not seeded "
             "from an {0} track, e.g. joint IT+OT seeding; -2 if the {0} product is unavailable)")


def _write_cand_links(tmp_path, name, coopt, sp_seed_col=True):
    """Two events of L1PuppiCand whose l1TrackIdx points into L1TTrack (old
    nano) or L1TSmartPixelsTrack (coopt nano, whose l1TrackIdx points into
    L1TTrack), plus a displaced vertex pair. Returns (path, branch docs)."""
    tree = {"run": np.array([1, 1], dtype=np.uint32),
            "luminosityBlock": np.array([1, 1], dtype=np.uint32),
            "event": np.array([1, 2], dtype=np.uint64),
            "L1TTrack": ak.zip({"pt": [[10.0, 20.0, 30.0], [40.0]]}),
            "L1TExtTrack": ak.zip({"pt": [[50.0, 60.0], [70.0]]})}
    # coopt: cand -> SP rows [0, -1, 1, -2, 2, 3] | [0, -1]; SP -> OT rows [2, -1, -2, 0] | [0]
    idx = [[0, -1, 1, -2, 2, 3], [0, -1]] if coopt else [[2, -1, 0, -2, 1, 0], [0, -1]]
    tree["L1PuppiCand"] = ak.zip({"pt": [[1.0] * 6, [1.0] * 2], "eta": [[0.0] * 6, [0.0] * 2],
                                  "phi": [[0.0] * 6, [0.0] * 2], "mass": [[0.0] * 6, [0.0] * 2],
                                  "charge": [[0] * 6, [0] * 2], "l1TrackIdx": idx})
    tree["L1DispVertex"] = ak.zip({"firstIndexTrk": [[0], [0]], "secondIndexTrk": [[1], [-1]]})
    docs = {"L1PuppiCand_l1TrackIdx": _TRK_DOC.format("L1TTrack")}
    trk_table = "L1TExtTrack"
    if coopt:
        trk_table = "L1TSmartPixelsExtTrack"
        docs["L1PuppiCand_l1TrackIdx"] = _TRK_DOC.format("L1TSmartPixelsTrack")
        sp = {"pt": [[1.0, 1.1, 1.2, 1.3], [1.4]]}
        if sp_seed_col:
            sp["l1TrackIdx"] = [[2, -1, -2, 0], [0]]
            docs["L1TSmartPixelsTrack_l1TrackIdx"] = _SEED_DOC.format("L1TTrack")
        tree["L1TSmartPixelsTrack"] = ak.zip(sp)
        tree["L1TSmartPixelsExtTrack"] = ak.zip({"pt": [[5.0, 6.0], [7.0]]})
    docs["L1DispVertex_firstIndexTrk"] = f"index of the higher-pt track in the {trk_table} table"
    docs["L1DispVertex_secondIndexTrk"] = f"index of the lower-pt track in the {trk_table} table"
    f = uproot.recreate(tmp_path / name)
    types = {k: (v.type if isinstance(v, ak.Array) else v.dtype) for k, v in tree.items()}
    f.mktree("Events", types, counter_name=lambda n: "n" + n,
             field_name=lambda outer, inner: inner if outer == "" else outer + "_" + inner)
    f["Events"].extend(tree)
    f.close()
    return str(tmp_path / name), docs


def _documented_events(path, docs, via_smartpixels=False):
    """Events read with `docs` as the branch docs (uproot writes the branch
    title as "name[counter]/type", so a synthetic file cannot carry them; coffea
    keeps a real one as the jagged branch content's __doc__)."""
    from coffea.nanoevents import NanoEventsFactory

    from ngtagger.coffea_schema import L1NanoSchema

    class Documented(L1NanoSchema):
        candidate_ot_track_via_smartpixels = via_smartpixels

        def __init__(self, base_form, *args, **kwargs):
            for name, form in zip(base_form["fields"], base_form["contents"]):
                if name in docs:
                    form["content"].setdefault("parameters", {})["__doc__"] = docs[name]
            super().__init__(base_form, *args, **kwargs)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        events = NanoEventsFactory.from_root({path: "Events"}, schemaclass=Documented).events()
    return events, [w for w in caught if w.category.__name__ == "SmartPixelsTrackLinkWarning"]


def test_old_doc_targets_ot_track(tmp_path):
    path, docs = _write_cand_links(tmp_path, "old.root", coopt=False)
    ev, spx_warnings = _documented_events(path, docs, via_smartpixels=True)
    assert not spx_warnings, [str(w.message) for w in spx_warnings]
    want = [[30.0, None, 10.0, None, 20.0, 10.0], [40.0, None]]
    assert ev.L1PuppiCand.matched_track.pt.tolist() == want
    # flag on an old nano: the direct link
    assert ev.L1PuppiCand.matched_ot_track.pt.tolist() == want
    assert ev.L1DispVertex.first_track.pt.tolist() == [[50.0], [70.0]]


def test_smartpixels_doc_targets_smartpixels_track_and_warns(tmp_path):
    path, docs = _write_cand_links(tmp_path, "coopt.root", coopt=True)
    ev, spx_warnings = _documented_events(path, docs)
    assert len(spx_warnings) == 1
    msg = str(spx_warnings[0].message)
    assert "L1TSmartPixelsTrack" in msg and "candidate_ot_track_via_smartpixels" in msg
    assert ev.L1PuppiCand.matched_track.pt.tolist() == [[1.0, None, 1.1, None, 1.2, 1.3], [1.4, None]]
    assert ev.L1DispVertex.first_track.pt.tolist() == [[5.0], [7.0]]
    assert ev.L1DispVertex.second_track.pt.tolist() == [[6.0], [None]]
    # the SmartPixels track -> seeding OT track link (-1/-2 masked)
    assert ev.L1TSmartPixelsTrack.ot_track.pt.tolist() == [[30.0, None, None, 10.0], [40.0]]
    with pytest.raises(AttributeError, match="candidate_ot_track_via_smartpixels"):
        ev.L1PuppiCand.matched_ot_track


def test_candidate_ot_track_via_smartpixels(tmp_path):
    path, docs = _write_cand_links(tmp_path, "coopt_flag.root", coopt=True)
    ev, _ = _documented_events(path, docs, via_smartpixels=True)
    # cand -> SP [0, -1, 1, -2, 2, 3] -> OT [2, x, -1, x, -2, 0]; event 2: SP 0 -> OT 0
    assert ev.L1PuppiCand.matched_ot_track.pt.tolist() == \
        [[30.0, None, None, None, None, 10.0], [40.0, None]]

    path, docs = _write_cand_links(tmp_path, "coopt_noseed.root", coopt=True, sp_seed_col=False)
    ev, spx_warnings = _documented_events(path, docs, via_smartpixels=True)
    assert any("unavailable" in str(w.message) for w in spx_warnings)
    with pytest.raises(AttributeError):
        ev.L1PuppiCand.matched_ot_track


def test_doc_absent_uses_default_targets(tmp_path):
    from ngtagger.coffea_schema import doc_target

    path, _ = _write_cand_links(tmp_path, "nodoc.root", coopt=False)
    ev, spx_warnings = _documented_events(path, {})
    assert not spx_warnings
    fields = uproot.open(f"{path}:Events")
    assert doc_target({"content": {"parameters": {"__doc__": fields["L1PuppiCand_l1TrackIdx"].title}}}) \
        is None
    assert ev.L1PuppiCand.matched_track.pt.tolist() == [[30.0, None, 10.0, None, 20.0, 10.0], [40.0, None]]
    assert ev.L1DispVertex.first_track.pt.tolist() == [[50.0], [70.0]]
    assert type(ev.L1PuppiCand).__name__ == "L1PuppiCandidateArray"
