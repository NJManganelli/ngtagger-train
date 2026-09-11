#!/usr/bin/env python
"""Paired insideOut vs outsideIn A/B on refit hit purity.

WHY RE-RUN. The 75.3% vs 53.9% clean-refit result in mem:smartpixels-v2p6-state
was measured post-cluster-fix on both arms (verified: outsideIn was introduced by
845138a, which POSTDATES ca9dc92 "take refit hit candidates from pixel rec hits
instead of raw digis"), so the cluster fix is not a reason to doubt it. But it
PREDATES two commits that plausibly move it: f629627 (the multiple-scattering
process-noise term Q) and 921e78a (the global-direction sense fix). Q changes hit
selection directly, which is the mechanism the entire outsideIn advantage was
attributed to.

PAIRING IS BY EVENT ID, NOT ROW ORDER. Multi-threaded cmsRun writes events in
COMPLETION order, so the same 40 events come out at different positions in the
two arms (mem:smartpixels-cmssw-git-sparse-checkout-hazard). Comparing row i to
row i would silently compare different events.

selHitClass: 0 = same TP as the track (correct), 1 = other TP (WRONG), 2 = noise.
"""
import argparse, json, re
import awkward as ak
import numpy as np
import uproot

LAYERS = (1, 2, 3, 4)


def load(path):
    t = uproot.open(f"{path}:Events")
    cfg = sorted({m.group(1) for m in
                  (re.match(r"^L1TSmartPixelsRefitHitDigiRefit([A-Za-z0-9]+)_", k)
                   for k in t.keys()) if m})[-1]
    hit = f"L1TSmartPixelsRefitHitDigiRefit{cfg}"
    trk = hit.replace("RefitHit", "Track")
    need = [f"{hit}_{c}" for c in ("trackIdx", "layer", "hitAccepted", "selHitClass")]
    need += [f"{trk}_{c}" for c in ("d0", "z0", "pt", "spixNAcceptedHits")]
    need += ["run", "luminosityBlock", "event"]
    A = t.arrays([k for k in need if k in t.keys()])
    n = ak.to_numpy(ak.num(A[f"{hit}_layer"]))
    ev_id = list(zip(ak.to_numpy(A["run"]).tolist(),
                     ak.to_numpy(A["luminosityBlock"]).tolist(),
                     ak.to_numpy(A["event"]).tolist()))
    return {"hit": hit, "trk": trk, "A": A, "nhit": n, "ev_id": ev_id}


def per_event_purity(S, idx):
    """For one event: clean-refit fraction and per-layer wrong-hit fraction."""
    hit, trk, A = S["hit"], S["trk"], S["A"]
    lay = ak.to_numpy(A[f"{hit}_layer"][idx])
    acc = ak.to_numpy(A[f"{hit}_hitAccepted"][idx])
    cls = ak.to_numpy(A[f"{hit}_selHitClass"][idx])
    ti = ak.to_numpy(A[f"{hit}_trackIdx"][idx]).astype(np.int64)
    use = acc > 0
    lay, cls, ti = lay[use], cls[use], ti[use]
    ntrk = len(ak.to_numpy(A[f"{trk}_pt"][idx]))
    wrong = np.zeros(ntrk, bool)
    nacc = np.zeros(ntrk, int)
    if len(ti):
        np.logical_or.at(wrong, ti, cls == 1)
        np.add.at(nacc, ti, 1)
    has = nacc > 0
    out = {"n_tracks_with_hits": int(has.sum()),
           "n_clean": int((has & ~wrong).sum()),
           "n_accepted": int(len(ti)), "n_wrong": int((cls == 1).sum())}
    for L in LAYERS:
        m = lay == L
        out[f"acc_L{L}"] = int(m.sum())
        out[f"wrong_L{L}"] = int(((cls == 1) & m).sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outsidein", required=True)
    ap.add_argument("--insideout", required=True)
    ap.add_argument("-o", "--out", default="layer_order_ab.json")
    a = ap.parse_args()
    S = {"outsideIn": load(a.outsidein), "insideOut": load(a.insideout)}
    common = sorted(set(S["outsideIn"]["ev_id"]) & set(S["insideOut"]["ev_id"]))
    print(f"events: outsideIn {len(S['outsideIn']['ev_id'])}, "
          f"insideOut {len(S['insideOut']['ev_id'])}, PAIRED {len(common)}")
    if not common:
        raise SystemExit("no common event IDs -- cannot pair")
    R = {"n_paired_events": len(common), "arms": {}}
    for arm, s in S.items():
        pos = {e: i for i, e in enumerate(s["ev_id"])}
        tot = {}
        for e in common:
            for k, v in per_event_purity(s, pos[e]).items():
                tot[k] = tot.get(k, 0) + v
        R["arms"][arm] = {
            "tracks_with_hits": tot["n_tracks_with_hits"],
            "clean_refits": tot["n_clean"],
            "clean_fraction": tot["n_clean"] / max(tot["n_tracks_with_hits"], 1),
            "accepted_hits": tot["n_accepted"],
            "wrong_hits": tot["n_wrong"],
            "wrong_hit_fraction": tot["n_wrong"] / max(tot["n_accepted"], 1),
            "per_layer_wrong_fraction": {
                f"L{L}": tot[f"wrong_L{L}"] / max(tot[f"acc_L{L}"], 1) for L in LAYERS},
        }
    print(f"\n{'arm':12s}{'tracks':>9}{'clean':>9}{'clean frac':>12}"
          f"{'wrong-hit frac':>16}   per-layer wrong L1..L4")
    for arm in ("outsideIn", "insideOut"):
        v = R["arms"][arm]
        pl = " ".join(f"{v['per_layer_wrong_fraction'][f'L{L}']:.3f}" for L in LAYERS)
        print(f"{arm:12s}{v['tracks_with_hits']:9d}{v['clean_refits']:9d}"
              f"{v['clean_fraction']:12.3f}{v['wrong_hit_fraction']:16.3f}   {pl}")
    o, i = R["arms"]["outsideIn"], R["arms"]["insideOut"]
    R["reference_v2p6"] = {"clean_outsideIn": 0.753, "clean_insideOut": 0.539,
                           "note": "mem:smartpixels-v2p6-state, pre-Q and pre-sense-fix"}
    print(f"\nclean fraction: outsideIn {o['clean_fraction']:.3f} vs insideOut "
          f"{i['clean_fraction']:.3f}   (v2.6 recorded 0.753 vs 0.539)")
    json.dump(R, open(a.out, "w"), indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
