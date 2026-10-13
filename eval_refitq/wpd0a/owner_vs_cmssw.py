"""WP-D0a check: does the census's truth definition hold CMSSW's loosely genuine tracks?

    python eval_refitq/wpd0a/owner_vs_cmssw.py INPUT.root [-n N] -o out.json

Reference: the CMSSW truth-association spec (WDMac wpd0a/cmssw_truth_spec.md, 2026-10-13;
SPEC below). A TTTrack is loosely genuine when exactly one TP is in EITHER cluster's full
contributor list of at least n - 1 of its n stubs and fewer than 2 of them are unknown (SPEC
sec. 3, TTTrackAssociator.cc:37-138; stub membership TTStubAssociator.cc:72-88). The nano
carries one (dominant) TP per OT cluster (SPEC sec. 5-6).

On CMSSW's own collections of the input (canon_v3 TTbar file 1 for WP-D0a), L1TTrack (prompt)
and L1TExtTrack (extended), per collection:

  emulation   CMSSW's loosely-genuine rule re-run on the nano (SPEC sec. 6), a stub holding TP
              T when T is among its credited TPs: the census's credit (M.credited_tps: tpIdx,
              tpIdxInner, tpIdxOuter; "f28") or the SPEC's (tpIdxInner, tpIdxOuter; "io");
              agreement with L1T*Track_looselyGenuine and its tpIdx. This is the dominant-TP
              limit measured on CMSSW's found tracks.
  owner       on CMSSW's loosely genuine tracks, the census majority owner (KF.credited_owner)
              is CMSSW's TP
  owner-held  ... and holds the track as the census delivers it: n_own >= 3 and more than half
              the stubs (definition sheet truth.delivery)

for the census owner under the format-28 credit ("f28") and under the stub's own tpIdx alone
("f27", formats <= 27). A miss is classified from the credit: CMSSW's TP credited on fewer
than n - 1 stubs (it is a non-dominant contributor of some cluster: "dominance") or not
("rule": the census's delivery rule itself, e.g. >= 3 owned stubs). Every number for all
tracks and for the census's acceptance (every stub in the barrel, |track eta| <= 1.4).
Errors: Wilson 68% intervals and the event-bootstrap standard deviation (1000 replicas).
Context: tracks CMSSW does NOT call loosely genuine that the census would deliver (its
majority rule is looser than CMSSW's n - 1).
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
COLLECTIONS = ("L1TTrack", "L1TExtTrack")
RULES = {"f28": M.STUB_TP_COLS, "io": M.STUB_TP_COLS[1:], "f27": M.STUB_TP_COLS[:1]}
N_REP = 1000


def read(path, nev, coll):
    br = ([f"{coll}_{c}" for c in TRK] + [f"{coll}Stub_trackIdx", f"{coll}Stub_otStubIdx"]
          + [f"L1TOTStub_{c}" for c in OTS])
    A = uproot.open(f"{path}:Events").arrays(br, entry_stop=nev)
    flat = lambda x: ak.to_numpy(ak.flatten(x))
    nt = ak.to_numpy(ak.num(A[f"{coll}_pt"]))
    no = ak.to_numpy(ak.num(A["L1TOTStub_tpIdx"]))
    T = {c: flat(A[f"{coll}_{c}"]) for c in TRK}
    T["event"] = np.repeat(np.arange(len(nt)), nt)
    ev_s = np.repeat(np.arange(len(nt)), ak.to_numpy(ak.num(A[f"{coll}Stub_trackIdx"])))
    toff = np.r_[0, np.cumsum(nt)[:-1]]
    ooff = np.r_[0, np.cumsum(no)[:-1]]
    ti = flat(A[f"{coll}Stub_trackIdx"]).astype(np.int64)
    si = flat(A[f"{coll}Stub_otStubIdx"]).astype(np.int64)
    if (si < 0).any() or (si >= no[ev_s]).any() or (ti < 0).any() or (ti >= nt[ev_s]).any():
        raise SystemExit(f"{coll}Stub links out of range")
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


def emulate_loose(tps, on, n_unknown, nst):
    """CMSSW's loosely-genuine rule (SPEC sec. 3) on credited TPs [track x stub x credit]:
    (loose, principal): exactly one TP held by >= n - 1 stubs, fewer than 2 unknown stubs."""
    ntrk = len(tps)
    t, s, k = np.nonzero(on[:, :, None] & SPN.tp_matched(tps))
    key = t.astype(np.int64) * (1 << 32) + (tps[t, s, k].astype(np.int64) + (1 << 31))
    uk, cnt = np.unique(key, return_counts=True)      # credits are distinct per stub
    ut, utp = uk >> 32, (uk & ((1 << 32) - 1)) - (1 << 31)
    cand = cnt >= nst[ut] - 1
    ncand = np.bincount(ut[cand], minlength=ntrk)
    principal = np.full(ntrk, -1, np.int64)
    principal[ut[cand]] = utp[cand]
    loose = (ncand == 1) & (n_unknown < 2)
    return loose, np.where(loose, principal, -1)


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


def one_collection(T, S, W):
    ntrk = len(T["pt"])
    ev = T["event"]
    cmssw_tp = T["tpIdx"].astype(np.int64)
    lg = (T["looselyGenuine"] > 0) & SPN.tp_matched(cmssw_tp)
    nst = np.bincount(S["track"], minlength=ntrk)
    allbar = np.bincount(S["track"], weights=(S["isBarrel"] > 0).astype(float), minlength=ntrk) == nst
    n_unk = np.bincount(S["track"], weights=(S["tpUnknown"] > 0).astype(float), minlength=ntrk)
    pops = {"all": np.ones(ntrk, bool),
            "census_acceptance": allbar & (np.abs(T["eta"]) <= M.ETA_MATCHED)}
    out = {"n_tracks": ntrk, "n_loosely_genuine": int(lg.sum()), "n_genuine": int((T["genuine"] > 0).sum()),
           "nStubs_consistent": bool(np.all(T["nStubs"] == nst)), "emulation": {}}
    for rule, cols in RULES.items():
        cred = M.credited_tps(*(S[c] for c in cols))
        tps, on = track_table(S["track"], ntrk, cred, -1)
        loose_e, prin_e = emulate_loose(tps, on, n_unk, nst)
        if rule != "f27":
            out["emulation"][rule] = {
                "agree": frac(np.ones(ntrk, bool), loose_e == lg, ev, W),
                "cmssw_yes_emu_no": int((lg & ~loose_e).sum()),
                "emu_yes_cmssw_no": int((~lg & loose_e).sum()),
                "principal_agree_when_both_loose": f"{int((lg & loose_e & (prin_e == cmssw_tp)).sum())} / "
                                                   f"{int((lg & loose_e).sum())}"}
        if rule == "io":
            continue
        own, n_own, owned, _ = KF.credited_owner(tps, on)
        held = owned & (n_own >= 3) & (n_own > 0.5 * nst)
        # stubs crediting CMSSW's TP, against CMSSW's own threshold n - 1
        k_tp = ((tps == cmssw_tp[:, None, None]).any(-1) & on).sum(1)
        r = {}
        for pn, pm in pops.items():
            sel = lg & pm
            miss = sel & ~(held & (own == cmssw_tp))
            r[pn] = {"owner": frac(sel, own == cmssw_tp, ev, W),
                     "owner_held": frac(sel, held & (own == cmssw_tp), ev, W),
                     "genuine_owner_held": frac(sel & (T["genuine"] > 0), held & (own == cmssw_tp), ev, W),
                     "missed": {"n": int(miss.sum()),
                                "dominance": int((miss & (k_tp < nst - 1)).sum()),
                                "rule": int((miss & (k_tp >= nst - 1)).sum()),
                                "of_which_owner_right_but_not_held": int((miss & (own == cmssw_tp)).sum())},
                     "not_loosely_genuine_but_held": {"n_tracks": int((pm & ~lg).sum()),
                                                      "n_held": int((pm & ~lg & held).sum())}}
        out[rule] = r
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("input")
    ap.add_argument("-n", "--nev", type=int, default=None)
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    out = {"input": a.input, "spec": "WDMac wpd0a/cmssw_truth_spec.md (2026-10-13)",
           "credits": {k: list(v) for k, v in RULES.items()}}
    for coll in COLLECTIONS:
        T, S, nev = read(a.input, a.nev, coll)
        W = np.random.default_rng(20261009).multinomial(nev, np.full(nev, 1.0 / nev),
                                                        size=N_REP).astype(float)
        out["n_events"] = nev
        out[coll] = one_collection(T, S, W)
    json.dump(out, open(a.out, "w"), indent=1)
    for coll in COLLECTIONS:
        o = out[coll]
        for rule, e in o["emulation"].items():
            print(f"{coll:11s} emulated looselyGenuine ({rule}): agree {e['agree']['frac']:.5f} of "
                  f"{e['agree']['n']}; CMSSW-yes/emu-no {e['cmssw_yes_emu_no']}, emu-yes/CMSSW-no "
                  f"{e['emu_yes_cmssw_no']}; TP agrees {e['principal_agree_when_both_loose']}")
        for rule in ("f28", "f27"):
            for pn in ("all", "census_acceptance"):
                x = o[rule][pn]
                print(f"{coll:11s} {rule} {pn:18s} loosely genuine {x['owner_held']['n']:6d}: owner "
                      f"{x['owner']['frac']:.4f}, owner-held {x['owner_held']['frac']:.4f} "
                      f"+- {x['owner_held']['boot_sd']:.4f}; missed {x['missed']}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
