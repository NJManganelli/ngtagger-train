"""Readers across the 2026-09-27 nano changes: OT-track columns renamed from
"seed" to "OT" (spix_nano.OT_RENAMES), link-only track -> stub tables (stub
columns through L1TOTStub[otStubIdx]) and the SmartPixels track -> stub /
cluster link tables. Each case on an old-layout and a new-layout synthetic nano."""

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


def _write(path, new):
    hit = {"trackIdx": [[0, 1]], "detId": [[10, 11]], "selClusterIdx": [[1, -1]]}
    spx = {"pt": [[10.1, 5.1]], "l1TrackIdx": [[0, 1]], "hwTanl": [[1, 2]]}
    proj = {"LocalX": [[0.01, 0.02]], "SigX": [[0.001, 0.002]]}
    tree = {
        "run": np.array([1], dtype=np.uint32),
        "luminosityBlock": np.array([1], dtype=np.uint32),
        "event": np.array([1], dtype=np.uint64),
        "L1TTrack": ak.zip({"pt": [[10.0, 5.0]]}),
        "L1TOTStub": ak.zip({k: ak.Array(v) for k, v in _OT.items()}),
        "L1TSmartPixelsCluster": ak.zip({"detId": [[10, 11]], "surface": [[1, 2]],
                                         "subdet": [[1, 1]]}),
    }
    word = "OT" if new else "Seed"
    hit.update({f"proj{word}{k}": v for k, v in proj.items()})
    spx.update({f"spixChi2ITAt{word}": [[1.5, 2.5]], f"spix{word}CovOK": [[True, False]]})
    link = {"trackIdx": [[0, 1]], "otStubIdx": _ROWS}
    if new:
        tree["L1TTrackStub"] = ak.zip(link)
        tree["L1TSmartPixelsTrackStub"] = ak.zip(link)
        tree["L1TSmartPixelsTrackCluster"] = ak.zip({"trackIdx": [[0, 1]], "clusterIdx": [[1, 0]]})
    else:
        link.update({c: [[v[0][r] for r in _ROWS[0]]] for c, v in _OT.items()})
        tree["L1TTrackStub"] = ak.zip(link)
    tree["L1TSmartPixelsTrack"] = ak.zip(spx)
    tree["L1TSmartPixelsRefitHit"] = ak.zip(hit)
    f = uproot.recreate(path)
    f.mktree("Events", {k: (v.type if isinstance(v, ak.Array) else v.dtype) for k, v in tree.items()},
             counter_name=lambda n: "n" + n,
             field_name=lambda outer, inner: inner if outer == "" else outer + "_" + inner)
    f["Events"].extend(tree)
    f.close()
    return str(path)


@pytest.mark.parametrize("new", [False, True])
def test_branch_name_resolves_ot_columns(tmp_path, new):
    keys = uproot.open(_write(tmp_path / "f.root", new) + ":Events").keys()
    assert spix_nano.branch_name(keys, "L1TSmartPixelsRefitHit", "projOTLocalX") == \
        f"L1TSmartPixelsRefitHit_proj{'OT' if new else 'Seed'}LocalX"
    assert spix_nano.branch_name(keys, "L1TSmartPixelsTrack", "spixOTCovOK") == \
        f"L1TSmartPixelsTrack_spix{'OT' if new else 'Seed'}CovOK"
    # an old spelling still resolves on a new file
    assert spix_nano.branch_name(keys, "L1TSmartPixelsTrack", "spixChi2ITAtSeed").endswith(
        "OT" if new else "Seed")


@pytest.mark.parametrize("new", [False, True])
def test_track_stub_columns(tmp_path, new):
    t = uproot.open(_write(tmp_path / "f.root", new) + ":Events")
    cols = ["trackIdx", "x", "z", "layer", "bend", "detId"]
    br = spix_nano.track_stub_branches(t.keys(), "L1TTrackStub", cols)
    assert ("L1TOTStub_x" in br) == new and ("L1TTrackStub_x" in br) == (not new)
    S = spix_nano.track_stub_columns(t.arrays(br), "L1TTrackStub", cols)
    assert S["trackIdx"].tolist() == [[0, 1]]
    assert S["x"].tolist() == [[3.0, 1.0]] and S["detId"].tolist() == [[502, 500]]
    assert S["bend"].tolist() == [[-3.0, 1.0]]


@pytest.mark.parametrize("new", [False, True])
def test_schema_reads_both_layouts(tmp_path, new):
    pytest.importorskip("coffea")
    from coffea.nanoevents import NanoEventsFactory

    from ngtagger.coffea_schema import L1ClustersNanoSchema

    path = _write(tmp_path / "f.root", new)
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
    assert ak.all(st.ot_stub.detId[0] == st.detId[0])
    assert ak.all(st.track.pt[0] == [10.0, 5.0])
    if new:
        sps = ev.L1TSmartPixelsTrackStub
        assert ak.all(sps.track.pt[0] == [10.1, 5.1]) and sps.z.tolist() == [[0.3, 0.1]]
        spc = ev.L1TSmartPixelsTrackCluster
        assert spc.cluster.detId.tolist() == [[11, 10]]
        assert ak.all(spc.track.pt[0] == [10.1, 5.1])


def test_link_tables_registered():
    from ngtagger.coffea_schema import smartpixels_links

    fields = ["L1TSmartPixelsExtTrackStub_trackIdx", "L1TSmartPixelsExtTrackStub_otStubIdx",
              "L1TSmartPixelsTrackDigiRefitAAAACluster_trackIdx",
              "L1TSmartPixelsTrackDigiRefitAAAACluster_clusterIdx",
              "L1TSmartPixelsTrackDigiRefitAAAA_pt"]
    mix, xref = smartpixels_links(fields)
    assert xref["L1TSmartPixelsExtTrackStub_trackIdx"] == "L1TSmartPixelsExtTrack"
    assert xref["L1TSmartPixelsExtTrackStub_otStubIdx"] == "L1TOTStub"
    assert xref["L1TSmartPixelsTrackDigiRefitAAAACluster_trackIdx"] == \
        "L1TSmartPixelsTrackDigiRefitAAAA"
    assert xref["L1TSmartPixelsTrackDigiRefitAAAACluster_clusterIdx"] == "L1TSmartPixelsCluster"
    assert mix["L1TSmartPixelsTrackDigiRefitAAAA"] == "L1TrackWord"


@pytest.mark.parametrize("verb", ["built", "seeded"])
def test_doc_target_both_wordings(verb):
    pytest.importorskip("coffea")
    from ngtagger.coffea_schema import doc_target

    doc = f"row of the L1TExtTrack track this SmartPixels track was {verb} from (-1 if none)"
    assert doc_target({"content": {"parameters": {"__doc__": doc}}}) == "L1TExtTrack"


@pytest.mark.parametrize("new", [False, True])
def test_table_columns_joins_stubs(tmp_path, new):
    t = uproot.open(_write(tmp_path / "f.root", new) + ":Events")
    cols = ["trackIdx", "x", "layer"]
    assert all(spix_nano.has_column(t.keys(), "L1TTrackStub", c) for c in cols)
    A = t.arrays(spix_nano.branch_names(t.keys(), "L1TTrackStub", cols))
    J = spix_nano.table_columns(A, "L1TTrackStub", cols)
    assert J["x"].tolist() == [[3.0, 1.0]] and J["layer"].tolist() == [[3, 1]]
