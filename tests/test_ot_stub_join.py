"""The track-stub -> L1TOTStub join: exact via otStubIdx, fuzzy identity as
the fallback, and compare_stub_joins reporting their agreement, on a tiny
synthetic nano."""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import awkward as ak
import numpy as np
import pytest

uproot = pytest.importorskip("uproot")

_DIR = pathlib.Path(__file__).resolve().parents[1] / "eval_refitq" / "combinatorics"


@pytest.fixture(scope="module")
def oo():
    sys.path.insert(0, str(_DIR))
    spec = importlib.util.spec_from_file_location("ot_oracle_t", _DIR / "ot_oracle.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write(path, oo, with_link=True, old_names=False):
    # event 0: OT stubs 0,1,2 ; rows 1 and 2 share one identity (fuzzy must refuse)
    # event 1: OT stubs 0,1
    ot = {"layer": [[1, 2, 2], [3, 1]], "isBarrel": [[1, 1, 1], [1, 1]],
          "detId": [[5, 6, 6], [7, 5]], "x": [[1.0, 2.0, 2.0], [3.0, 4.0]],
          "y": [[0.5, 0.5, 0.5], [0.5, 0.5]], "z": [[0.1, 0.2, 0.2], [0.3, 0.4]],
          "bend": [[1.0, -1.0, -1.0], [0.5, 0.0]],
          ("tpIdx" if old_names else "tpKey"): [[11, 12, 13], [21, -1]],
          "tpPt": [[2.0, 3.0, 3.0], [4.0, -1.0]],
          "tpGenuine": [[1, 1, 1], [1, 0]], "tpCombinatoric": [[0, 0, 0], [0, 0]],
          "tpUnknown": [[0, 0, 0], [0, 1]]}
    # track stubs: ev0 -> OT rows 0 and 2 ; ev1 -> OT rows 1 and 0
    rows = [[0, 2], [1, 0]]
    st = {c: [[ot[c][e][r] for r in rows[e]] for e in range(2)]
          for c in ("layer", "isBarrel", "detId", "x", "y", "z", "bend")}
    st.update({"trackIdx": [[0, 0], [0, 0]], "r": [[1.0, 1.0], [1.0, 1.0]],
               "phi": [[0.0, 0.0], [0.0, 0.0]]})
    if with_link:
        st["otStubIdx"] = rows
    trk = {c: [[0.0], [0.0]] for c in oo.TRK}
    tree = {"L1TOTStub": ak.zip({k: ak.Array(v) for k, v in ot.items()}),
            "L1TTrackStub": ak.zip({k: ak.Array(v) for k, v in st.items()}),
            "L1TTrack": ak.zip({k: ak.Array(v) for k, v in trk.items()})}
    f = uproot.recreate(path)
    f.mktree("Events", {k: v.type for k, v in tree.items()},
             counter_name=lambda n: "n" + n,
             field_name=lambda outer, inner: inner if outer == "" else outer + "_" + inner)
    f["Events"].extend(tree)
    f.close()
    return str(path)


def test_compare_and_exact(tmp_path, oo):
    p = _write(tmp_path / "new.root", oo)
    r = oo.compare_stub_joins(p)
    assert r["n_stubs"] == 4 and r["n_linked"] == 4
    # the shared-identity stub (ev0 -> row 2) is unmatched by the fuzzy join only
    assert r["n_fuzzy_matched"] == 3 and r["n_agree"] == 3 and r["n_disagree"] == 0
    assert r["n_fuzzy_unmatched"] == 1 and r["exact_identity_ok_frac"] == 1.0
    T, S, O = oo.load(p, None)
    tk, flag, hit = oo.stub_truth(S, O)
    assert tk.tolist() == [11, 13, -1, 21] and hit.all()
    assert flag.tolist() == [1, 1, 3, 1]


def test_fallback_on_old_file(tmp_path, oo):
    p = _write(tmp_path / "old.root", oo, with_link=False, old_names=True)
    T, S, O = oo.load(p, None)
    assert oo.OT_LINK not in S
    tk, flag, hit = oo.stub_truth(S, O)
    assert tk.tolist() == [11, -1, -1, 21] and hit.tolist() == [True, False, True, True]
    with pytest.raises(SystemExit):
        oo.compare_stub_joins(p)
