"""The track-stub -> L1TOTStub join (exact, via otStubIdx) on a tiny synthetic
nano with link-only track-stub tables."""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import awkward as ak
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


def _write(path, oo):
    # event 0: OT stubs 0,1,2 ; event 1: OT stubs 0,1
    ot = {"layer": [[1, 2, 2], [3, 1]], "isBarrel": [[1, 1, 1], [1, 1]],
          "detId": [[5, 6, 6], [7, 5]], "x": [[1.0, 2.0, 2.0], [3.0, 4.0]],
          "y": [[0.5, 0.5, 0.5], [0.5, 0.5]], "z": [[0.1, 0.2, 0.2], [0.3, 0.4]],
          "r": [[1.0] * 3, [1.0] * 2], "phi": [[0.0] * 3, [0.0] * 2],
          "bend": [[1.0, -1.0, -1.0], [0.5, 0.0]],
          "tpKey": [[11, 12, 13], [21, -1]],
          "tpPt": [[2.0, 3.0, 3.0], [4.0, -1.0]],
          "tpGenuine": [[1, 1, 1], [1, 0]], "tpCombinatoric": [[0, 0, 0], [0, 0]],
          "tpUnknown": [[0, 0, 0], [0, 1]]}
    # track stubs: ev0 -> OT rows 0 and 2 ; ev1 -> OT rows 1 and 0
    st = {"trackIdx": [[0, 0], [0, 0]], "otStubIdx": [[0, 2], [1, 0]]}
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


def test_exact_join(tmp_path, oo):
    T, S, O = oo.load(_write(tmp_path / "f.root", oo), None)
    assert S["x"].tolist() == [1.0, 2.0, 4.0, 3.0]      # the joined L1TOTStub rows
    tk, flag, hit = oo.stub_truth(S, O)
    assert tk.tolist() == [11, 13, -1, 21] and hit.all()
    assert flag.tolist() == [1, 1, 3, 1]
