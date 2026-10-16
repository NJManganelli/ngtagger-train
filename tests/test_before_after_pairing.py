"""WP-D0a before_after: runs are paired by event identity (run, lumi, event) and tpIdx, never by entry number.
Two extracts of the same events in different entry orders (as canon_v3 and canon_v4 hold them) re-index to
identical arrays, give identical figures on one bootstrap, and runs whose event sets differ are refused."""
import sys
from pathlib import Path

import numpy as np
import pytest

R = Path(__file__).resolve().parents[1] / "eval_refitq"
sys.path.insert(0, str(R / "wpd0a"))
sys.path.insert(0, str(R / "combinatorics"))
import before_after as BA  # noqa: E402
import tracklet_topology_cost as M  # noqa: E402

IDS = np.array([[1, 7, 40], [1, 7, 12], [1, 9, 3], [1, 8, 55]], np.int64)
TPS = {0: [0, 1, 3], 1: [0], 2: [0, 1], 3: [2, 4]}          # tpIdx per event (event = row of IDS)
DELIV = {(0, 1), (2, 0), (3, 4)}                            # (event, tpIdx) delivered
N_DR = np.array([5, 1, 3, 4])
N_FAKE = np.array([1, 0, 2, 0])


def run(order, label, fmt=35):
    """An extract of the events IDS[order] in that entry order."""
    ev, tp = [], []
    for entry, e in enumerate(order):
        ev += [entry] * len(TPS[e])
        tp += TPS[e]
    ev, tp = np.array(ev), np.array(tp)
    key = M.tp_key(ev, tp)
    o = np.argsort(key)
    orig = np.asarray(order)[ev]                             # the IDS row of each TP
    Z = {"key": key[o], "event": ev[o].astype(np.int32), "pt": (10.0 * orig + tp)[o].astype(np.float32),
         "B/deliv": np.array([(e, t) in DELIV for e, t in zip(orig, tp)])[o],
         "B/n_dr": N_DR[order], "B/n_fake": N_FAKE[order], "event_ids": IDS[order]}
    meta = {"label": label, "format": fmt, "n_events": len(order), "builds": ["B"], "inputs": ["/nonexistent"]}
    return meta, Z


def test_permuted_runs_pair_identically():
    runs = {"v3": run([0, 1, 2, 3], "v3"), "v4": run([2, 0, 3, 1], "v4")}
    # by entry number the two runs disagree
    assert not np.array_equal(runs["v3"][1]["B/n_dr"], runs["v4"][1]["B/n_dr"])
    assert not np.array_equal(runs["v3"][1]["pt"], runs["v4"][1]["pt"])
    out, ids, _ = BA.by_identity(runs)
    assert np.array_equal(ids, np.unique(IDS, axis=0))
    A, B = out["v3"][1], out["v4"][1]
    assert set(A) == set(B) == {"key", "event", "pt", "B/deliv", "B/n_dr", "B/n_fake"}
    for k in A:
        assert np.array_equal(A[k], B[k]), k
    # one bootstrap over the identities: identical figures and replicas
    W = np.random.default_rng(1).multinomial(4, np.full(4, 0.25), size=50).astype(float)
    fa = BA._ratio(A["B/n_fake"].astype(float), A["B/n_dr"].astype(float), W)
    fb = BA._ratio(B["B/n_fake"].astype(float), B["B/n_dr"].astype(float), W)
    assert fa[0] == fb[0] and np.array_equal(fa[1], fb[1])
    # the same particle is delivered in both
    assert np.array_equal(BA.found_on(B, "B", "deliv", A["key"]), A["B/deliv"])


def test_different_event_sets_refused():
    a = run([0, 1, 2, 3], "a")
    m, Z = run([1, 0, 3, 2], "b")
    Z["event_ids"] = Z["event_ids"].copy()
    Z["event_ids"][0, 2] = 999
    with pytest.raises(SystemExit, match="different event sets"):
        BA.by_identity({"a": a, "b": (m, Z)})


def test_cross_format_must_be_marked():
    runs = {"v3": run([0, 1, 2, 3], "v3", 34), "v4": run([2, 0, 3, 1], "v4", 35)}
    with pytest.raises(SystemExit, match="census formats differ"):
        BA.formats(runs)
    assert BA.formats(runs, cross_format=True) == (34, 35)
