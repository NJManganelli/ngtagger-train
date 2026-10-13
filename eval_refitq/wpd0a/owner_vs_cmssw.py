"""WP-D0a check: does the census's truth definition hold CMSSW's loosely genuine tracks?

    python eval_refitq/wpd0a/owner_vs_cmssw.py INPUT.root [-n N] -o out.json

On CMSSW's own L1TTrack collection (canon_v3 TTbar file 1 for WP-D0a), every track
whose TTTrackAssociator label is loosely genuine (L1TTrack_looselyGenuine; its TP
is L1TTrack_tpIdx: one TP behind every stub but at most one, under the any-TP rule
of TTStubAssociator.cc:72-87, every contributor of every cluster) is scored under
the census's definitions, applied to its stubs (L1TTrackStub -> L1TOTStub):

  owner       the census majority owner (KF.credited_owner) is CMSSW's TP
  owner-held  ... and holds the track as the census delivers it: n_own >= 3 and
              more than half the stubs (definition sheet truth.delivery)

each under two stub credits: "f28" the census's (M.credited_tps: the stub's
tpIdx and the dominant TP of each of its clusters; definition sheet v2) and "f27"
the stub's own tpIdx only (formats <= 27). The known-good is CMSSW's label itself.
Where the census misses, the reason is classified from the stub credit: the
census credits the TP on fewer than n - 1 of the n stubs (CMSSW's own threshold),
i.e. the TP is not the dominant contributor of a cluster that CMSSW still credits
to it ("dominance"), or the TP passes that threshold and still loses ("rule").
Every number is given for all tracks and for the census's acceptance (every stub
in the barrel, |track eta| <= 1.4). Errors: Wilson 68% intervals and the
event-bootstrap standard deviation (1000 replicas).
Also, as context: tracks CMSSW does NOT call loosely genuine that the census
would deliver (its majority rule is looser than CMSSW's n - 1).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import awkward as ak
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "combinatorics"))
import tracklet_topology_cost as M   # noqa: E402
import kf_emulation as KF            # noqa: E402
from ngtagger.data import spix_nano as SPN   # noqa: E402

TRK = ("looselyGenuine", "genuine", "combinatoric", "unknown", "tpIdx", "nStubs", "pt", "eta")
OTS = M.STUB_TP_COLS + ("isBarrel", "tpGenuine", "tpCombinatoric", "tpUnknown")
N_REP = 1000


def read(path, nev):
    br = ([f"L1TTrack_{c}" for c in TRK] + ["L1TTrackStub_trackIdx", "L1TTrackStub_otStubIdx"]
          + [f"L1TOTStub_{c}" for c in OTS])
    A = uproot.open(f"{path}:Events").arrays(br, entry_stop=nev)
    flat = lambda x: ak.to_numpy(ak.flatten(x))
    nt = ak.to_numpy(ak.num(A["L1TTrack_pt"]))
    no = ak.to_numpy(ak.num(A["L1TOTStub_tpIdx"]))
    T = {c: flat(A[f"L1TTrack_{c}"]) for c in TRK}
    T["event"] = np.repeat(np.arange(len(nt)), nt)
    ev_s = np.repeat(np.arange(len(nt)), ak.to_numpy(ak.num(A["L1TTrackStub_trackIdx"])))
    toff = np.r_[0, np.cumsum(nt)[:-1]]
    ooff = np.r_[0, np.cumsum(no)[:-1]]
    ti = flat(A["L1TTrackStub_trackIdx"]).astype(np.int64)
    si = flat(A["L1TTrackStub_otStubIdx"]).astype(np.int64)
    if (si < 0).any() or (si >= no[ev_s]).any() or (ti < 0).any() or (ti >= nt[ev_s]).any():
        raise SystemExit("L1TTrackStub links out of range")
    O = {c: flat(A[f"L1TOTStub_{c}"]) for c in OTS}
    S = {"track": toff[ev_s] + ti, **{c: O[c][ooff[ev_s] + si] for c in OTS}}
    return T, S, len(nt)


def track_table(track, ntrk, values, fill):
    """values per stub -> (ntrk, width, ...) table in stub order, and its valid mask."""
    o = np.argsort(track, kind="stable")
    g = track[o]
    slot = np.arange(len(g)) - np.searchsorted(g, g, side="left")
    w = int(slot.max()) + 1 if len(g) else 1
    out = np.full((ntrk, w) + values.shape[1:], fill, values.dtype)
    on = np.zeros((ntrk, w), bool)
    out[g, slot] = values[o]
    on[g, slot] = True
    return out, on


def wilson(k, n, z=1.0):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def frac(sel, ok, ev, W):
    """fraction of `sel` that is `ok`, Wilson 68% interval, event-bootstrap sd."""
    num = np.bincount(ev[sel & ok], minlength=W.shape[1]).astype(float)
    den = np.bincount(ev[sel], minlength=W.shape[1]).astype(float)
    k, n = int(num.sum()), int(den.sum())
    rep = (W @ num) / np.maximum(W @ den, 1)
    return {"n": n, "k": k, "frac": k / n if n else None, "wilson68": wilson(k, n),
            "boot_sd": float(rep.std(ddof=1)) if n else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("input")
    ap.add_argument("-n", "--nev", type=int, default=None)
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    T, S, nev = read(a.input, a.nev)
    ntrk = len(T["pt"])
    W = np.random.default_rng(20261009).multinomial(nev, np.full(nev, 1.0 / nev), size=N_REP).astype(float)
    cmssw_tp = T["tpIdx"].astype(np.int64)
    lg = (T["looselyGenuine"] > 0) & SPN.tp_matched(cmssw_tp)
    nst, _ = np.histogram(S["track"], bins=np.arange(ntrk + 1))
    allbar = np.bincount(S["track"], weights=(S["isBarrel"] > 0).astype(float), minlength=ntrk) == nst
    acc = allbar & (np.abs(T["eta"]) <= M.ETA_MATCHED)
    pops = {"all": np.ones(ntrk, bool), "census_acceptance": acc}
    out = {"input": a.input, "n_events": nev, "n_tracks": ntrk,
           "n_loosely_genuine": int(lg.sum()), "n_genuine": int((T["genuine"] > 0).sum()),
           "nStubs_consistent": bool(np.all(T["nStubs"] == nst)),
           "definitions": {"f28": "M.credited_tps(tpIdx, tpIdxInner, tpIdxOuter)",
                           "f27": "M.credited_tps(tpIdx) (genuine stubs only)"}}
    for rule, cols in (("f28", M.STUB_TP_COLS), ("f27", ("tpIdx",))):
        cred = M.credited_tps(*(S[c] for c in cols))
        tps, on = track_table(S["track"], ntrk, cred, -1)
        own, n_own, owned, right = KF.credited_owner(tps, on)
        held = owned & (n_own >= 3) & (n_own > 0.5 * nst)
        # stubs crediting CMSSW's TP, against CMSSW's own threshold n - 1
        k_tp = ((tps == cmssw_tp[:, None, None]).any(-1) & on).sum(1)
        r = {}
        for pn, pm in pops.items():
            sel = lg & pm
            miss = sel & ~(held & (own == cmssw_tp))
            r[pn] = {"owner": frac(sel, own == cmssw_tp, T["event"], W),
                     "owner_held": frac(sel, held & (own == cmssw_tp), T["event"], W),
                     "genuine_owner_held": frac(sel & (T["genuine"] > 0), held & (own == cmssw_tp),
                                                T["event"], W),
                     "missed": {"n": int(miss.sum()),
                                "dominance": int((miss & (k_tp < nst - 1)).sum()),
                                "rule": int((miss & (k_tp >= nst - 1)).sum()),
                                "of_which_owner_right_but_not_held": int((miss & (own == cmssw_tp)).sum())},
                     "not_loosely_genuine_but_held": {
                         "n_tracks": int((pm & ~lg).sum()),
                         "n_held": int((pm & ~lg & held).sum())}}
        out[rule] = r
    json.dump(out, open(a.out, "w"), indent=1)
    for rule in ("f28", "f27"):
        for pn in pops:
            x = out[rule][pn]
            print(f"{rule} {pn:18s} loosely genuine {x['owner_held']['n']:6d}: owner "
                  f"{x['owner']['frac']:.4f}, owner-held {x['owner_held']['frac']:.4f} "
                  f"+- {x['owner_held']['boot_sd']:.4f}; missed {x['missed']}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
