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
    """Reference L1TTrack + one digiRefit variant track table (row-aligned)
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
    tree["L1TSmartPixelsTrackDigiRefitAAAA"] = ak.zip({
        "pt": [[10.1, 5.2, 8.05]] * 2, "hwTanl": [[100, 200, 300]] * 2,
        "spixRefitPerformed": [[True, False, True]] * 2,
        "spixNKFUpdates": [[2, 0, 3]] * 2,
    })
    tree["L1TSmartPixelsRefitHitDigiRefitAAAA"] = ak.zip({
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
    trk = events.L1TSmartPixelsTrackDigiRefitAAAA
    # track-word hw decode reused from L1TrackWord mixin
    assert hasattr(trk, "tanlFromHw")
    assert ak.all(trk.spixRefitPerformed[0] == [True, False, True])

    # per-hit link table resolves trackIdx crossref back to the variant track
    hit = events.L1TSmartPixelsRefitHitDigiRefitAAAA
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
        "L1TSmartPixelsTrackDigiRefitIIIA": ak.zip({"pt": [[10.1, 5.1]], "hwTanl": [[1, 2]]}),
        "L1TSmartPixelsExtTrackDigiRefitAAAA": ak.zip({"pt": [[11.1]], "hwTanl": [[3]]}),
        "L1TSmartPixelsRefitHitDigiRefitIIIA": ak.zip({
            "trackIdx": [[1, 0]], "detId": [[12, 11]], "selClusterIdx": [[2, -1]]}),
        "L1TSmartPixelsExtRefitHitDigiRefitAAAA": ak.zip({
            "trackIdx": [[0]], "detId": [[10]], "selClusterIdx": [[0]]}),
        "L1TSmartPixelsCluster": ak.zip({
            "detId": [[10, 11, 12, 20]], "surface": [[1, 2, 4, 11]], "subdet": [[1, 1, 1, 2]]}),
        "L1TOTStub": ak.zip({"detId": [[500, 501, 502]], "bend": [[1.0, 2.0, 3.0]]}),
        "L1TTrackStub": ak.zip({"trackIdx": [[0, 1]], "detId": [[502, 500]],
                                "otStubIdx": [[2, 0]]}),
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
    assert xref["L1TSmartPixelsRefitHitDigiRefitIIIA_trackIdx"] == "L1TSmartPixelsTrackDigiRefitIIIA"
    assert xref["L1TSmartPixelsExtRefitHitDigiRefitAAAA_trackIdx"] == \
        "L1TSmartPixelsExtTrackDigiRefitAAAA"
    assert xref["L1TSmartPixelsRefitHitDigiRefitIIIA_selClusterIdx"] == "L1TSmartPixelsCluster"
    assert xref["L1TTrackStub_otStubIdx"] == "L1TOTStub"
    assert xref["L1TTrackStub_trackIdx"] == "L1TTrack"
    assert mix["L1TSmartPixelsExtTrackDigiRefitAAAA"] == "L1TrackWord"
    # no index column -> no crossref (older nanos without otStubIdx stay silent)
    assert "L1TExtTrackStub_otStubIdx" not in smartpixels_links(["L1TExtTrackStub_trackIdx"])[1]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ev = _events(path)
    assert not [w for w in caught if "cross-ref" in str(w.message)], \
        [str(w.message) for w in caught]
    h = ev.L1TSmartPixelsRefitHitDigiRefitIIIA
    assert ak.all(h.track.pt[0] == [5.1, 10.1])
    assert h.cluster.detId[0][0] == 12 and ak.is_none(h.cluster.detId[0], axis=0)[1]
    assert ev.L1TSmartPixelsExtRefitHitDigiRefitAAAA.track.pt[0][0] == pytest.approx(11.1)
    st = ev.L1TTrackStub
    assert ak.all(st.ot_stub.detId[0] == st.detId[0])
    assert ak.all(st.track.pt[0] == [10.0, 5.0])


def test_real_file_crossrefs():
    """On a real nano: SPIX_NANO=/path/to/file.root pytest tests/test_coffea_schema.py
    -k real_file [SPIX_NANO_EVENTS=3]. Every refit hit table's trackIdx and
    selClusterIdx, and every track-stub table's otStubIdx, must land on a row
    with the same detId (and, for stubs, the same bend)."""
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
        linked = tab.cluster if idx == "selClusterIdx" else (
            tab.ot_stub if idx == "otStubIdx" else tab.track)
        if idx in ("selClusterIdx", "otStubIdx"):
            assert ak.all(linked.detId[ok] == tab.detId[ok]), col
        if idx == "otStubIdx":
            assert ak.all(linked.bend[ok] == tab.bend[ok]), col
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
            "L1TSmartPixelsTrackDigiRefitAAAA": [[3, -1], [7]]}
    rows = {"L1TTrack": [[0, -1], [0]], "L1TOTStub": [[1, -8, -1], [-10]],
            "L1TSmartPixelsCluster": [[0, 0, -8], [0, -1]],
            "L1TSmartPixelsTrackDigiRefitAAAA": [[0, -1], [-10]]}
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
    assert xref["L1TSmartPixelsTrackDigiRefitAAAA_spixMatchedTpIdx"] == "L1TTP"
    ev = NanoEventsFactory.from_root({path: "Events"}, schemaclass=L1ClustersNanoSchema).events()
    pt = [[30.0, 80.0], [90.0]]
    for t, want in rows.items():
        tab = ev[t]
        got = tab.spixMatchedTpIdx if t.startswith("L1TSmartPixelsTrack") else tab.tpIdx
        assert got.tolist() == want, t
        assert tab.tp.pt.tolist() == [[pt[e][r] if r >= 0 else None for r in ev_rows]
                                      for e, ev_rows in enumerate(want)], t
