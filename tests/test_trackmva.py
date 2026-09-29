"""Per-track refit MVAs (ngtagger.train.trackmva) and their on-the-fly scorer /
jet-constituent gather (ngtagger.data.trackmva_feature), on synthetic nanos."""
import json

import awkward as ak
import numpy as np
import pytest

from ngtagger.train import trackmva as tm
from ngtagger.data import trackmva_feature as tf

CFG = "AAAA"
B, R, H = "L1TTrack", f"L1TSmartPixelsTrackSeedlessRefit{CFG}", f"L1TSmartPixelsRefitHitSeedlessRefit{CFG}"
TRUTH_BRANCH_MARKERS = ("genuine", "tp_", "tpFromHardInteraction", "spixMatchedTp", "tpIdx", "tpKey",
                        "selClusterIdx", "L1TTP")


def _synth_tables(n_events=40, seed=0, ntrk_range=(15, 30), links="row"):
    """{table: {col: jagged}} + flat event ids of a small learnable prompt-family nano.
    `links`: "row" (tpIdx into L1TTP), "key" (cnn_nano v2-v4: tpKey + L1TTP_key) or
    "key_no_tp" (keys and no L1TTP table)."""
    rng = np.random.default_rng(seed)
    ntrk = rng.integers(*ntrk_range, size=n_events)
    N = int(ntrk.sum())
    pt = rng.uniform(2, 50, N)
    genuine = (rng.random(N) < 0.9).astype(np.int32)
    tp_pt = np.where(rng.random(N) < 0.97, pt * rng.normal(1, 0.02, N), -1.0)
    tp_d0 = rng.normal(0, 0.001, N) + (rng.random(N) < 0.2) * rng.normal(0, 0.1, N)
    tp_z0 = rng.normal(0, 5, N)
    chi2 = rng.exponential(1, N) * np.where(genuine > 0, 1, 4)
    res_scale = 0.002 * (1 + 10 / pt) * (1 + chi2)          # cm, learnable from inv_pt, chi2
    d0 = tp_d0 + rng.normal(0, 1, N) * res_scale * 20
    z0 = tp_z0 + rng.normal(0, 1, N) * res_scale * 5
    tanl = rng.normal(0, 1, N)
    refit = (rng.random(N) < 0.9).astype(np.int32)
    r_d0 = np.where(refit > 0, tp_d0 + rng.normal(0, 1, N) * res_scale, d0)
    r_z0 = np.where(refit > 0, tp_z0 + rng.normal(0, 1, N) * res_scale, z0)
    tpkey = np.arange(N) + 1000                       # unique TP key per track
    evt_of_trk = np.repeat(np.arange(n_events), ntrk)
    kept = np.arange(N) % 5 != 0                      # the rest fail the L1TTP selection
    ttp_n = np.bincount(evt_of_trk[kept], minlength=n_events)
    tp_row = np.full(N, -1)
    tp_row[kept] = np.arange(kept.sum()) - (np.cumsum(ttp_n) - ttp_n)[evt_of_trk[kept]]
    tp_row = np.where(kept, tp_row, -3 - tpkey)
    by_key = links != "row"
    base = {"pt": pt, "eta": np.arcsinh(tanl), "tanL": tanl, "z0": z0, "d0": d0,
            "trkMVA1": rng.random(N), "chi2XYRed": chi2, "chi2ZRed": rng.exponential(1, N),
            "chi2BendRed": rng.exponential(1, N), "nStubs": rng.integers(4, 7, N),
            "hitPattern": rng.integers(1, 127, N),
            "cov_rInv_rInv": rng.random(N) * 1e-6, "cov_phi_phi": rng.random(N) * 1e-6,
            "cov_tanL_tanL": rng.random(N) * 1e-4, "cov_z0_z0": rng.random(N) * 1e-2,
            "cov_d0_d0": rng.random(N) * 1e-3,
            "genuine": genuine, "tpFromHardInteraction": (rng.random(N) < 0.2).astype(np.int32),
            "tp_pt": tp_pt, "tp_d0": tp_d0, "tp_z0": tp_z0}
    ref = {"pt": pt * rng.normal(1, 0.01, N), "tanL": tanl + rng.normal(0, 1e-3, N), "z0": r_z0,
           "d0": r_d0, "spixRefitPerformed": refit, "spixAnyWindowTruncated": rng.integers(0, 2, N),
           "spixNCrossings": rng.integers(0, 5, N), "spixNAcceptedHits": rng.integers(0, 5, N),
           "spixLayerHitMask": rng.integers(0, 16, N), "spixMaxWindowMult": rng.integers(0, 4, N),
           "spixChi2IncXTot": np.where(refit > 0, rng.exponential(2, N), -999.0),
           "spixChi2IncYTot": rng.exponential(2, N), "spixChi2IncAlphaTot": rng.exponential(2, N),
           "spixChi2ITAtOT": rng.exponential(2, N), "spixChi2ITAtRefit": rng.exponential(2, N),
           "spixShiftChi2": rng.exponential(2, N), "spixLogDetRatio": rng.normal(0, 1, N),
           ("spixMatchedTpKey" if by_key else "tpIdx"): tpkey if by_key else tp_row}
    # hits: 0-4 per track; clusters: one per hit, owned by the track's TP unless "wrong"
    nh = rng.integers(0, 5, N)
    trk_of_hit = np.repeat(np.arange(N), nh)
    toff = np.concatenate([[0], np.cumsum(ntrk)[:-1]])
    evt_of_hit = evt_of_trk[trk_of_hit]
    nh_ev = np.bincount(evt_of_hit, minlength=n_events)
    Hn = len(trk_of_hit)
    wrong = rng.random(Hn) < np.where(genuine[trk_of_hit] > 0, 0.03, 0.5)
    # cluster rows: per event, in hit order (so selClusterIdx = hit's position in the event)
    hoff = np.concatenate([[0], np.cumsum(nh_ev)[:-1]])
    sel = np.arange(Hn) - hoff[evt_of_hit]
    other = 10**6 + np.arange(Hn)                     # a wrong hit's TP: never in L1TTP
    ctp = (np.where(wrong, other, tpkey[trk_of_hit]) if by_key
           else np.where(wrong, -3 - other, tp_row[trk_of_hit]))
    hits = {"hitAccepted": (rng.random(Hn) < 0.9).astype(np.int32),
            "trackIdx": trk_of_hit - toff[evt_of_hit], "hasAlpha": rng.integers(0, 2, Hn),
            "windowMult": rng.integers(1, 4, Hn), "clusterMerged": rng.integers(0, 2, Hn),
            "pullX": rng.normal(0, 1, Hn), "pullY": rng.normal(0, 1, Hn),
            "pullAlpha": rng.normal(0, 1, Hn), "chi2IncX": rng.exponential(1, Hn),
            "chi2IncY": rng.exponential(1, Hn), "chi2IncAlpha": rng.exponential(1, Hn),
            "selClusterIdx": sel}
    J = lambda cols, n: {c: ak.unflatten(np.asarray(v), n) for c, v in cols.items()}  # noqa: E731
    tables = {B: J(base, ntrk), R: J(ref, ntrk), H: J(hits, nh_ev),
              "L1TSmartPixelsCluster": J({("tpKey" if by_key else "tpIdx"): ctp,
                                          "layer": np.ones(Hn, np.int32)}, nh_ev)}
    if links == "key":
        tables["L1TTP"] = J({"key": tpkey[kept]}, ttp_n)
    elif links == "row":
        tables["L1TTP"] = J({"pt": np.abs(tp_pt[kept])}, ttp_n)
    ev = {"run": np.ones(n_events, np.int64), "luminosityBlock": np.ones(n_events, np.int64),
          "event": np.arange(n_events, dtype=np.int64) + 100 * seed}
    return tables, ev


def _batch(tables, ev):
    """An uproot-batch-like awkward record with flat branch names."""
    cols = {f"{t}_{c}": v for t, cs in tables.items() for c, v in cs.items()}
    cols.update({k: ak.Array(v) for k, v in ev.items()})
    return ak.zip(cols, depth_limit=1)


def _write_nano(path, tables, ev):
    import uproot

    # flat NanoAOD-style branch names (each jagged branch gets its own counter)
    with uproot.recreate(path) as f:
        f["Events"] = {**{f"{t}_{c}": v for t, cs in tables.items() for c, v in cs.items()}, **ev}


# ---------------------------------------------------------------------------
def test_second_largest_matches_bruteforce():
    rng = np.random.default_rng(3)
    g = rng.integers(0, 50, 400)
    v = np.round(rng.random(400), 1)                  # ties on purpose
    got = tm._second_largest(v, g, 60)
    for k in range(60):
        x = np.sort(v[g == k])[::-1]
        assert got[k] == (x[1] if len(x) > 1 else 0.0)


def test_n_it_wrong_and_alignment():
    tables, ev = _synth_tables(n_events=6, seed=1)
    A = _batch(tables, ev)
    T, ntrk = tm.tracks_from_batch(A, "prompt", CFG)
    # brute force per event
    exp = []
    for e in range(len(ntrk)):
        tk = ak.to_numpy(tables[R]["tpIdx"][e])
        ct = ak.to_numpy(tables["L1TSmartPixelsCluster"]["tpIdx"][e])
        h = {c: ak.to_numpy(tables[H][c][e]) for c in ("trackIdx", "selClusterIdx", "hitAccepted")}
        n = np.zeros(len(tk))
        for ti, si, a in zip(h["trackIdx"], h["selClusterIdx"], h["hitAccepted"]):
            if a and (ct[si] != tk[ti] or tk[ti] == -1):
                n[ti] += 1
        exp.append(n)
    assert np.array_equal(T["t_n_it_wrong"], np.concatenate(exp))
    assert len(T["b_pt"]) == ntrk.sum()
    bad = dict(tables)
    bad[R] = {c: v[:, :-1] for c, v in tables[R].items()}
    with pytest.raises(RuntimeError, match="row aligned"):
        tm.tracks_from_batch(_batch(bad, ev), "prompt", CFG)


def test_n_it_wrong_unmatched_track():
    """refit tpIdx = -1 (no TP owns a strict majority of the track's cluster weight): every
    accepted hit is wrong, even a noise cluster whose own tpIdx is also -1."""
    tables, ev = _synth_tables(n_events=6, seed=2)
    tk = tables[R]["tpIdx"]
    tables[R]["tpIdx"] = ak.full_like(tk, -1)
    ct = tables["L1TSmartPixelsCluster"]["tpIdx"]
    tables["L1TSmartPixelsCluster"]["tpIdx"] = ak.full_like(ct, -1)
    T, _ = tm.tracks_from_batch(_batch(tables, ev), "prompt", CFG)
    acc = ak.to_numpy(ak.flatten(tables[H]["hitAccepted"]))
    assert T["t_n_it_wrong"].sum() == acc.sum() > 0


def test_key_linked_nano_reads_like_row_linked():
    T, _ = tm.tracks_from_batch(_batch(*_synth_tables(n_events=4, seed=2)), "prompt", CFG)
    K, _ = tm.tracks_from_batch(_batch(*_synth_tables(n_events=4, seed=2, links="key")), "prompt", CFG)
    assert T["t_n_it_wrong"].sum() > 0
    assert np.array_equal(T["t_n_it_wrong"], K["t_n_it_wrong"])
    with pytest.raises(RuntimeError, match="unavailable"):
        tm.tracks_from_batch(_batch(*_synth_tables(n_events=4, seed=2, links="key_no_tp")), "prompt", CFG)


def test_features_withhold_d0_and_z0_value():
    tables, ev = _synth_tables(n_events=10, seed=4)
    T, _ = tm.tracks_from_batch(_batch(tables, ev), "prompt", CFG)
    for coll in tm.COLLS:
        X0, names = tm.feature_matrix(T, coll)
        assert not any("d0" in n for n in names)
        T2 = dict(T)
        # rescale d0 inside the windows and move the whole track along z
        T2["b_d0"] = T["b_d0"] * 0.5
        T2["r_d0"] = T["r_d0"] * 0.99
        T2["b_z0"] = T["b_z0"] + 3.0
        T2["r_z0"] = T["r_z0"] + 3.0
        X1, _ = tm.feature_matrix(T2, coll)
        np.testing.assert_allclose(np.nan_to_num(X0), np.nan_to_num(X1), atol=1e-9)
    # sentinel -999 -> NaN
    assert np.isnan(tm.features(T, "refit")["spixChi2IncXTot"]).any()


def test_event_folds_and_train_only():
    files = ["/a/x_f01_1.root", "/a/x_f02_1.root", "/a/x_f03_1.root"]
    m = tm.train_only_mask(files, ["_f01_", "x_f02_1.root"])
    assert m.tolist() == [True, True, False]
    with pytest.raises(ValueError, match="matches none"):
        tm.train_only_mask(files, ["_f09_"])
    file_idx = np.repeat([0, 1, 2], 10)
    f = tm.event_folds(file_idx, m, n_folds=5, seed=0)
    assert (f[:20] == -1).all() and sorted(f[20:].tolist()) == sorted(list(range(5)) * 2)


def test_train_save_load_and_score(tmp_path):
    paths = []
    for s in range(3):
        tables, ev = _synth_tables(n_events=30, seed=10 + s)
        p = str(tmp_path / f"synth_f0{s + 1}_30ev.root")
        _write_nano(p, tables, ev)
        paths.append(p)
    out = tmp_path / "bundle"
    params = {"max_iter": 8, "learning_rate": 0.3}
    rpath = tm.train_trackmva(paths, str(out), colls=["refit", "base"], train_only=["_f01_"],
                              n_folds=3, crossfit_mode="nested", hgb_params=params,
                              step_events=7, trap=True, log=lambda s: None)
    rep = json.load(open(rpath))
    assert rep["n_events"] == 90 and rep["n_test_events"] == 60
    assert set(rep["refit"]["perf"]) == {"prompt", "displaced"}
    for w in rep["refit"]["perf"].values():
        assert 0 <= w["auc_clean"] <= 1 and 0 <= w["sd0_coverage_68"] <= 1
    assert "with_d0" in rep["refit"]["trap_ablation"]["prompt"]

    stem = out / tm.bundle_stem("prompt", CFG)
    meta = json.load(open(str(stem) + "_meta.json"))
    for k in ("features", "spec_version", "training_files", "event_split", "head_definitions",
              "package_versions", "hgb_params", "model_sha256", "withheld_features"):
        assert k in meta
    assert [f["train_only"] for f in meta["training_files"]] == [True, False, False]
    assert set(meta["event_split"]["fold_by_file"]["synth_f01_30ev.root"]) == {-1}

    oof = np.load(str(stem) + "_oof.npz")
    assert oof["refit_scores"].shape[0] == 3
    assert np.isfinite(oof["refit_scores"]).all()          # nested: every track scored in every slot
    tfold = oof["folds_ev"][oof["track_evt"]]
    assert np.isnan(oof["refit_outer"][tfold < 0]).all()
    assert np.isfinite(oof["refit_outer"][tfold >= 0]).all()

    mva = tm.load_bundle(str(out))
    assert mva.family == "prompt" and mva.colls == ["refit", "base"]
    # scorer on files == direct predict on the reader's flat table
    S = tf.trackmva_scores_from_files(paths, mva, step_events=11)
    T, E = tm.build_dataset(paths, "prompt", CFG, with_truth=False, log=lambda s: None)
    P = mva.predict(T)
    assert ak.num(S, axis=0) == 90
    for h in tm.OUTPUTS:
        np.testing.assert_allclose(ak.to_numpy(ak.flatten(S[h])), P[h].astype(np.float32))
    assert ((P["clean"] >= 0) & (P["clean"] <= 1)).all() and (P["log_sd0"] >= 0).all()
    # scorer reads no truth branch
    import uproot
    br = tf.trackmva_branches(uproot.open(paths[0])["Events"].keys(), mva)
    assert not any(m in b for b in br for m in TRUTH_BRANCH_MARKERS)

    # tampering is detected
    with open(str(stem) + ".joblib", "ab") as f:
        f.write(b"x")
    with pytest.raises(RuntimeError, match="sha256"):
        tm.load_bundle(str(out))


def _jets_fixture():
    # 2 events: event 0 has 3 tracks, event 1 has 2 tracks
    scores = ak.Array([
        [{"clean": 0.1, "d0acc": 0.2}, {"clean": 0.3, "d0acc": 0.4}, {"clean": 0.5, "d0acc": 0.6}],
        [{"clean": 0.7, "d0acc": 0.8}, {"clean": 0.9, "d0acc": 1.0}],
    ])
    cands = ak.Array([
        [{"l1TrackIdx": 2}, {"l1TrackIdx": -1}, {"l1TrackIdx": 0}, {"l1TrackIdx": 1}],
        [{"l1TrackIdx": 1}, {"l1TrackIdx": -1}],
    ])
    # event 0: jet0 = cands [0, 1], jet1 = [3, 2]; event 1: jet0 = [1, 0]
    nested = ak.Array([[[0, 1], [3, 2]], [[1, 0]]])
    return scores, cands, nested


def test_constituent_gather_and_sentinel():
    scores, cands, nested = _jets_fixture()
    G = tf.trackmva_per_constituent(scores, cands, nested, n_tracks=[3, 2])
    flat = lambda a: ak.to_numpy(ak.flatten(a, axis=None))  # noqa: E731
    assert ak.num(G["clean"], axis=2).tolist() == [[2, 2], [2]]
    np.testing.assert_allclose(flat(G["clean"]), [0.5, -1.0, 0.3, 0.1, -1.0, 0.9], rtol=1e-6)
    np.testing.assert_allclose(flat(G["d0acc"]), [0.6, -1.0, 0.4, 0.2, -1.0, 1.0], rtol=1e-6)
    with pytest.raises(RuntimeError, match="rows per"):
        tf.trackmva_per_constituent(scores, cands, nested, n_tracks=[3, 3])
    bad = ak.Array([[{"l1TrackIdx": 7}] * 4, [{"l1TrackIdx": 0}] * 2])
    with pytest.raises(IndexError):
        tf.trackmva_per_constituent(scores, bad, nested)


def test_jet_tables_absent(tmp_path):
    tables, ev = _synth_tables(n_events=3, seed=5)
    p = str(tmp_path / "trackonly.root")
    _write_nano(p, tables, ev)
    import uproot
    keys = uproot.open(p)["Events"].keys()
    assert tf.missing_jet_tables(keys) == list(tf.DEFAULT_JET_TABLES)
    with pytest.raises(tf.JetTablesAbsentError, match="no jet tables"):
        tf.require_jet_tables(keys)
    from ngtagger.data.nano import load_jets
    with pytest.raises(tf.JetTablesAbsentError):
        load_jets([p], feature_groups=["baseline", "trackmva"], trackmva_model=str(tmp_path))
    tf.require_jet_tables(list(keys) + [f"{t}_pt" for t in tf.DEFAULT_JET_TABLES])  # no raise


def test_family_mismatch_raises():
    mva = tm.TrackMVA({"refit": {}}, {"family": "prompt", "config": CFG,
                                      "tables": tm.table_names("prompt", CFG)})
    _, cands, nested = _jets_fixture()
    with pytest.raises(ValueError, match="matching family"):
        tf.trackmva_jet_features(["x.root"], None, cands, nested, "l1TrackIdx", mva,
                                 track_table="L1TExtTrack")


def test_cli_parses():
    from ngtagger import cli

    with pytest.raises(SystemExit):
        cli.main(["train-trackmva", "--help"])
