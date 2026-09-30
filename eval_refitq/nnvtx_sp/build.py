"""Flatten canon_v1 AAAA f01..f10 into one per-track dataset for the NNVtx study.

The per-track columns are exactly the ones ngtagger.train.trackmva reads for the
"canonical" family (tracks_from_batch: b_ = L1TTrack, r_ = L1TSmartPixelsTrack,
h_ = refit-hit aggregates, truth incl. r_tp_* from L1TTP), in the same file / entry /
row order as `ngtagger train-trackmva` (so its out-of-fold scores align row by row;
checked in vertex.py), plus the vertexing extras:
  r_eta              SP track eta
  b_tpIdx            OT track TP row
  ot_hs, sp_hs       HS truth: TP (evt, bx) == (0, 0) via L1TTrack_tpIdx / L1TSmartPixelsTrack_tpIdx
  E: file, entry, event, genz, ntrk

  pixi run python eval_refitq/nnvtx_sp/build.py
"""
from __future__ import annotations

import os
import sys

import awkward as ak
import numpy as np
import uproot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nvcommon import DATASET, FILES  # noqa: E402

from ngtagger.train import trackmva as tm  # noqa: E402

FAMILY = "canonical"
STEP = 25
EXTRA = ["L1TSmartPixelsTrack_eta", "L1TSmartPixelsTrack_l1TrackIdx", "L1TTrack_tpIdx",
         "L1TTP_evt", "L1TTP_bx", "GenVtx_z", "event"]


def _hs(tpidx, evt, bx, ntrk):
    """HS flag per track: its L1TTP row has (evt, bx) == (0, 0); tpIdx < 0 -> False."""
    ntp = ak.to_numpy(ak.num(evt))
    off = np.repeat(np.concatenate([[0], np.cumsum(ntp)[:-1]]), ntrk)
    ev = ak.to_numpy(ak.flatten(evt))
    b = ak.to_numpy(ak.flatten(bx))
    ok = tpidx >= 0
    g = np.where(ok, off + tpidx, 0)
    return ok & (ev[g] == 0) & (b[g] == 0) if len(ev) else np.zeros(len(tpidx), bool)


def main():
    Ts, Es = [], []
    for fi, fn in enumerate(FILES):
        t = uproot.open(fn)["Events"]
        br = list(dict.fromkeys(tm.nano_branches(t.keys(), FAMILY, "", True) + EXTRA))
        for start in range(0, t.num_entries, STEP):
            A = t.arrays(br, entry_start=start, entry_stop=min(start + STEP, t.num_entries))
            T, ntrk = tm.tracks_from_batch(A, FAMILY, "", True)
            l1 = A["L1TSmartPixelsTrack_l1TrackIdx"]
            assert ak.all(l1 == ak.local_index(l1)), "l1TrackIdx is not the row index"
            T["r_eta"] = ak.to_numpy(ak.flatten(A["L1TSmartPixelsTrack_eta"])).astype(np.float64)
            T["b_tpIdx"] = ak.to_numpy(ak.flatten(A["L1TTrack_tpIdx"])).astype(np.int64)
            T["ot_hs"] = _hs(T["b_tpIdx"], A["L1TTP_evt"], A["L1TTP_bx"], ntrk)
            T["sp_hs"] = _hs(T["r_tpIdx"], A["L1TTP_evt"], A["L1TTP_bx"], ntrk)
            assert np.array_equal(T["ot_hs"], T["b_tpFromHardInteraction"] > 0), "OT HS != tpFromHardInteraction"
            Ts.append(T)
            Es.append({"file": np.full(len(ntrk), fi), "entry": np.arange(start, start + len(ntrk)),
                       "event": ak.to_numpy(A["event"]).astype(np.int64),
                       "genz": ak.to_numpy(A["GenVtx_z"]).astype(np.float64), "ntrk": ntrk})
            del A
        print(f"{os.path.basename(fn)}: {t.num_entries} ev", flush=True)
    T = {k: np.concatenate([t[k] for t in Ts]) for k in Ts[0]}
    E = {k: np.concatenate([e[k] for e in Es]) for k in Es[0]}
    E["evt"] = np.arange(len(E["ntrk"]))
    T["evt"] = np.repeat(E["evt"], E["ntrk"])
    os.makedirs(os.path.dirname(DATASET), exist_ok=True)
    np.savez_compressed(DATASET, **{"T_" + k: v for k, v in T.items()}, **{"E_" + k: v for k, v in E.items()},
                        files=np.array(FILES))
    print(f"{DATASET}: {len(E['evt'])} events, {len(T['evt'])} tracks, "
          f"SP-truth matched {np.mean(T['r_tpIdx'] >= 0):.4f}, OT/SP TP differ {np.mean(T['r_tpIdx'] != T['b_tpIdx']):.4f}",
          flush=True)


if __name__ == "__main__":
    main()
