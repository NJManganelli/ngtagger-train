"""SC8 + SC4 pipeline plumbing tests: the jet/link/cand/track tables are config
knobs, not hardcoded names. Builds a synthetic nano and runs the full
prepare_dataset path (read -> group -> features -> gen labels) through the
table-override kwargs.

Two parallel table-name sets share one fixture machinery:
  SC8  L1puppiJetSC8 / L1SC8JetCands / L1PuppiCand / L1TTrack -- SC8 has no
       NG-tagged collection; labels come from gen matching (config
       configs/deepset_hgq2_sc8.yaml).
  SC4  L1puppiJetSC4NG / L1SC4NGJetCands / L1ExtPuppiCand / L1TExtTrack -- the
       default (NG) path; this is the synthetic twin so the default pipeline
       has the same synthetic coverage as SC8."""

import awkward as ak
import numpy as np
import pytest

uproot = pytest.importorskip("uproot")

# SC8 table-name set (no NG collection)
JET_TABLE = "L1puppiJetSC8"
LINK_TABLE = "L1SC8JetCands"
CAND_TABLE = "L1PuppiCand"
TRACK_TABLE = "L1TTrack"

# SC4 (default, NG) table-name set -- the synthetic twin
SC4_TABLES = {
    "jet_table": "L1puppiJetSC4NG", "link_table": "L1SC4NGJetCands",
    "cand_table": "L1ExtPuppiCand", "track_table": "L1TExtTrack",
}
SC8_TABLES = {
    "jet_table": JET_TABLE, "link_table": LINK_TABLE,
    "cand_table": CAND_TABLE, "track_table": TRACK_TABLE,
}


def _write_nano(tmp_path, tables, name, n_events=8):
    """Two jets per event, two constituents each; cand 0/2 track-matched.
    `tables` selects the jet/link/cand/track table NAMES (SC4 vs SC8).

    NOTE: uproot's how="zip" reader groups jagged branches by shared offsets,
    so every collection here carries a pairwise-DISTINCT per-event count
    (jets 2, tracks 3, links 4, cands 5, clusters 6, GenJet 7, GenVisTau 8,
    GenPart 9) or the groups would merge under synthetic-fixture names.
    Extra entries are placed far away in (eta, phi) so matching is unaffected.
    """
    jet_table = tables["jet_table"]
    link_table = tables["link_table"]
    cand_table = tables["cand_table"]
    track_table = tables["track_table"]
    # 5 cands: 4 linked + 1 stray
    cand = {
        "pt": [[30.0, 5.0, 18.0, 3.0, 1.0]] * n_events,
        "eta": [[0.1, 0.2, -0.3, -0.4, 4.0]] * n_events,
        "phi": [[0.0, 0.4, -1.0, -0.6, 3.0]] * n_events,
        "mass": [[0.14, 0.0, 0.14, 0.0, 0.0]] * n_events,
        "charge": [[1, 0, -1, 0, 0]] * n_events,
        "id": [[0, 3, 0, 2, 3]] * n_events,
        "z0": [[0.01, 0.0, -0.02, 0.0, 0.0]] * n_events,
        "dxy": [[0.001, 0.0, -0.002, 0.0, 0.0]] * n_events,
        "puppiWeight": [[1.0, 0.7, 1.0, 0.4, 0.1]] * n_events,
        "hwEmID": [[0, 1, 0, 0, 1]] * n_events,
        "hwTkQuality": [[3, 0, 2, 0, 0]] * n_events,
        "l1TrackIdx": [[0, -1, 1, -1, -1]] * n_events,
        "hgcClusterIdx": [[-1, 0, -1, -1, -1]] * n_events,
    }
    far = lambda n, base: [base + i for i in range(n)]  # noqa: E731
    tree = {
        "run": np.ones(n_events, dtype=np.uint32),
        "luminosityBlock": np.ones(n_events, dtype=np.uint32),
        "event": np.arange(n_events, dtype=np.uint64),
        cand_table: ak.zip({k: ak.Array(v) for k, v in cand.items()}),
        jet_table: ak.zip({
            "pt": [[35.0, 20.0]] * n_events, "eta": [[0.15, -0.35]] * n_events,
            "phi": [[0.05, -0.8]] * n_events, "et": [[35.0, 20.0]] * n_events,
        }),
        link_table: ak.zip({
            "jetIdx": [[0, 0, 1, 1]] * n_events,
            "candIdx": [[0, 1, 2, 3]] * n_events,
            "slot": [[0, 1, 0, 1]] * n_events,
            "inTagger": [[True, True, True, True]] * n_events,
        }),
        track_table: ak.zip({
            "rInv": [[1e-3, -2e-3, 3e-3]] * n_events, "tanL": [[0.2, -0.6, 1.0]] * n_events,
            "z0": [[0.01, -0.02, 0.03]] * n_events, "d0": [[0.001, -0.002, 0.003]] * n_events,
            "chi2XYRed": [[1.1, 2.2, 3.0]] * n_events, "chi2ZRed": [[0.9, 1.5, 2.0]] * n_events,
            "chi2BendRed": [[0.5, 0.7, 0.9]] * n_events, "trkMVA1": [[0.9, 0.4, 0.2]] * n_events,
        }),
        "L1HGCCluster": ak.zip({
            "hOverE": [[0.1] * 6] * n_events, "sigmaRRTot": [[0.02] * 6] * n_events,
            "zBarycenter": [[350.0] * 6] * n_events, "eMax": [[5.0] * 6] * n_events,
            "sigmaZZ": [[1.0] * 6] * n_events,
        }),
        # gen tables for labeling (jet 0 -> b genjet, jet 1 -> gluon genjet;
        # extra gen objects sit at |eta| >= 3, outside the 0.8 match cone)
        "GenJet": ak.zip({
            "pt": [[36.0, 21.0] + [10.0] * 5] * n_events,
            "eta": [[0.16, -0.36] + far(5, 3.0)] * n_events,
            "phi": [[0.06, -0.82] + [3.0] * 5] * n_events,
            "partonFlavour": [[5, 21] + [1] * 5] * n_events,
            "hadronFlavour": [[5, 0] + [0] * 5] * n_events,
        }),
        "GenVisTau": ak.zip({
            "pt": [[12.0] * 8] * n_events, "eta": [far(8, 3.0)] * n_events,
            "phi": [[2.5] * 8] * n_events, "charge": [[1] * 8] * n_events,
        }),
        "GenPart": ak.zip({
            "pt": [[40.0] * 9] * n_events, "eta": [far(9, 3.0)] * n_events,
            "phi": [[-2.0] * 9] * n_events,
            "pdgId": [[11] * 9] * n_events, "statusFlags": [[1] * 9] * n_events,
        }),
    }
    f = uproot.recreate(tmp_path / name)
    types = {k: (v.type if isinstance(v, ak.Array) else v.dtype) for k, v in tree.items()}
    f.mktree("Events", types,
             counter_name=lambda n: "n" + n,
             field_name=lambda outer, inner: inner if outer == "" else outer + "_" + inner)
    f["Events"].extend(tree)
    f.close()
    return str(tmp_path / name)


# Both the default (SC4/NG) and the SC8 table-name sets get identical synthetic
# coverage: the fixture is table-name-agnostic, so the twin is a parametrization.
_TABLE_SETS = [pytest.param(SC4_TABLES, "sc4.root", id="sc4"),
               pytest.param(SC8_TABLES, "sc8.root", id="sc8")]


@pytest.mark.parametrize("tables,fname", _TABLE_SETS)
def test_load_jets_tables(tmp_path, tables, fname):
    from ngtagger.data.nano import load_jets

    path = _write_nano(tmp_path, tables, fname)
    jets, constituents, gen = load_jets(
        [path], n_const=8, feature_groups=["baseline", "track"], **tables,
    )
    assert ak.all(ak.num(jets.pt, axis=1) == 2)
    assert ak.all(ak.num(constituents.pt, axis=2) == 2)
    # track crossref resolved through l1TrackIdx
    assert float(ak.flatten(constituents.trk_trkMVA1, axis=None)[0]) == pytest.approx(0.9)


@pytest.mark.parametrize("tables,fname", _TABLE_SETS)
def test_prepare_dataset_end_to_end(tmp_path, tables, fname):
    from ngtagger.train.trainer import prepare_dataset

    path = _write_nano(tmp_path, tables, fname)
    ds = prepare_dataset(
        [path], n_const=8, feature_groups=["baseline"],
        tables=dict(tables),
        gen_match_dr=0.8, test_fraction=0.25,
    )
    n_jets = len(ds["X_train"]) + len(ds["X_test"])
    assert n_jets == 16  # 8 events x 2 labeled jets, all kept
    assert ds["X_train"].shape[1:] == (8, len(ds["feature_names"]))
    assert ds["y_train"].shape[1] == 8


def test_sc8_config_tables_reach_prepare_dataset(tmp_path, monkeypatch):
    """configs/deepset_hgq2_sc8.yaml table knobs must reach prepare_dataset."""
    import os

    import yaml

    import ngtagger.train.trainer as trainer_mod

    cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "deepset_hgq2_sc8.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    captured = {}

    def fake_prepare(files, n_const, feature_groups, max_events, seed, tables, gen_match_dr,
                     **kwargs):
        captured.update(tables=tables, n_const=n_const, gen_match_dr=gen_match_dr)
        raise _Stop()

    class _Stop(Exception):
        pass

    monkeypatch.setattr(trainer_mod, "prepare_dataset", fake_prepare)
    with pytest.raises(_Stop):
        trainer_mod.run_training(cfg_path, ["dummy.root"], str(tmp_path / "out"))

    assert captured["tables"]["jet_table"] == "L1puppiJetSC8"
    assert captured["tables"]["link_table"] == "L1SC8JetCands"
    assert captured["tables"]["cand_table"] == "L1PuppiCand"
    assert captured["tables"]["track_table"] == "L1TTrack"
    assert captured["n_const"] == cfg["data_config"]["n_constituents"]
    assert captured["gen_match_dr"] == pytest.approx(0.8)
