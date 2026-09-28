"""Readers of the link tables and OT-track columns: link-only track -> stub
tables (stub columns through L1TOTStub[otStubIdx]), the SmartPixels track ->
stub / cluster link tables, and the projOT* / spixChi2ITAtOT / spixOTCovOK
columns, on a synthetic nano."""

import warnings

import awkward as ak
import numpy as np
import pytest

uproot = pytest.importorskip("uproot")

from ngtagger.data import spix_nano  # noqa: E402

# one event: OT stubs 0..2; the L1TTrack stubs are OT rows 2 and 0
_OT = {"layer": [[1, 2, 3]], "isBarrel": [[True, True, False]], "detId": [[500, 501, 502]],
       "x": [[1.0, 2.0, 3.0]], "y": [[0.5, 0.6, 0.7]], "z": [[0.1, 0.2, 0.3]],
       "r": [[1.1, 2.1, 3.1]], "phi": [[0.4, 0.3, 0.2]], "bend": [[1.0, 2.0, -3.0]]}
_ROWS = [[2, 0]]


def _write(path):
    link = {"trackIdx": [[0, 1]], "otStubIdx": _ROWS}
    tree = {
        "run": np.array([1], dtype=np.uint32),
        "luminosityBlock": np.array([1], dtype=np.uint32),
        "event": np.array([1], dtype=np.uint64),
        "L1TTrack": ak.zip({"pt": [[10.0, 5.0]]}),
        "L1TOTStub": ak.zip({k: ak.Array(v) for k, v in _OT.items()}),
        "L1TSmartPixelsCluster": ak.zip({"detId": [[10, 11]], "surface": [[1, 2]],
                                         "subdet": [[1, 1]]}),
        "L1TTrackStub": ak.zip(link),
        "L1TSmartPixelsTrackStub": ak.zip(link),
        "L1TSmartPixelsTrackCluster": ak.zip({"trackIdx": [[0, 1]], "clusterIdx": [[1, 0]]}),
        "L1TSmartPixelsTrack": ak.zip({
            "pt": [[10.1, 5.1]], "l1TrackIdx": [[0, 1]], "hwTanl": [[1, 2]],
            "spixChi2ITAtOT": [[1.5, 2.5]], "spixOTCovOK": [[True, False]]}),
        "L1TSmartPixelsRefitHit": ak.zip({
            "trackIdx": [[0, 1]], "detId": [[10, 11]], "selClusterIdx": [[1, -1]],
            "projOTLocalX": [[0.01, 0.02]], "projOTSigX": [[0.001, 0.002]]}),
    }
    f = uproot.recreate(path)
    f.mktree("Events", {k: (v.type if isinstance(v, ak.Array) else v.dtype) for k, v in tree.items()},
             counter_name=lambda n: "n" + n,
             field_name=lambda outer, inner: inner if outer == "" else outer + "_" + inner)
    f["Events"].extend(tree)
    f.close()
    return str(path)


def test_track_stub_columns(tmp_path):
    t = uproot.open(_write(tmp_path / "f.root") + ":Events")
    cols = ["trackIdx", "x", "z", "layer", "bend", "detId"]
    br = spix_nano.track_stub_branches(t.keys(), "L1TTrackStub", cols)
    assert "L1TOTStub_x" in br and "L1TTrackStub_otStubIdx" in br
    S = spix_nano.track_stub_columns(t.arrays(br), "L1TTrackStub", cols)
    assert S["trackIdx"].tolist() == [[0, 1]]
    assert S["x"].tolist() == [[3.0, 1.0]] and S["detId"].tolist() == [[502, 500]]
    assert S["bend"].tolist() == [[-3.0, 1.0]]


def test_table_columns_joins_stubs(tmp_path):
    t = uproot.open(_write(tmp_path / "f.root") + ":Events")
    cols = ["trackIdx", "x", "layer"]
    assert all(spix_nano.has_column(t.keys(), "L1TTrackStub", c) for c in cols)
    A = t.arrays(spix_nano.branch_names(t.keys(), "L1TTrackStub", cols))
    J = spix_nano.table_columns(A, "L1TTrackStub", cols)
    assert J["x"].tolist() == [[3.0, 1.0]] and J["layer"].tolist() == [[3, 1]]


def test_schema(tmp_path):
    pytest.importorskip("coffea")
    from coffea.nanoevents import NanoEventsFactory

    from ngtagger.coffea_schema import L1ClustersNanoSchema

    path = _write(tmp_path / "f.root")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ev = NanoEventsFactory.from_root({path: "Events"}, schemaclass=L1ClustersNanoSchema).events()
    assert not [w for w in caught if "cross-ref" in str(w.message)], [str(w.message) for w in caught]
    spx = ev.L1TSmartPixelsTrack
    assert spx.spixChi2ITAtOT.tolist() == [[1.5, 2.5]]
    assert spx.spixOTCovOK.tolist() == [[True, False]]
    assert ak.all(spx.ot_track.pt[0] == [10.0, 5.0])
    assert ev.L1TSmartPixelsRefitHit.projOTSigX.tolist() == [[0.001, 0.002]]
    st = ev.L1TTrackStub
    assert st.x.tolist() == [[3.0, 1.0]] and st.layer.tolist() == [[3, 1]]
    assert ak.all(st.track.pt[0] == [10.0, 5.0])
    sps = ev.L1TSmartPixelsTrackStub
    assert ak.all(sps.track.pt[0] == [10.1, 5.1]) and sps.z.tolist() == [[0.3, 0.1]]
    spc = ev.L1TSmartPixelsTrackCluster
    assert spc.cluster.detId.tolist() == [[11, 10]]
    assert ak.all(spc.track.pt[0] == [10.1, 5.1])


def test_link_tables_registered():
    from ngtagger.coffea_schema import smartpixels_links

    fields = ["L1TSmartPixelsExtTrackStub_trackIdx", "L1TSmartPixelsExtTrackStub_otStubIdx",
              "L1TSmartPixelsTrackSeedlessRefitAAAACluster_trackIdx",
              "L1TSmartPixelsTrackSeedlessRefitAAAACluster_clusterIdx",
              "L1TSmartPixelsTrackSeedlessRefitAAAA_pt"]
    mix, xref = smartpixels_links(fields)
    assert xref["L1TSmartPixelsExtTrackStub_trackIdx"] == "L1TSmartPixelsExtTrack"
    assert xref["L1TSmartPixelsExtTrackStub_otStubIdx"] == "L1TOTStub"
    assert xref["L1TSmartPixelsTrackSeedlessRefitAAAACluster_trackIdx"] == \
        "L1TSmartPixelsTrackSeedlessRefitAAAA"
    assert xref["L1TSmartPixelsTrackSeedlessRefitAAAACluster_clusterIdx"] == "L1TSmartPixelsCluster"
    assert mix["L1TSmartPixelsTrackSeedlessRefitAAAA"] == "L1TrackWord"


def test_doc_target():
    pytest.importorskip("coffea")
    from ngtagger.coffea_schema import doc_target

    doc = "row of the L1TExtTrack track this SmartPixels track was built from (-1 if none)"
    assert doc_target({"content": {"parameters": {"__doc__": doc}}}) == "L1TExtTrack"
