"""Validation of the heads converter (PLAN.md section 7 item 2) on "implemented now", extended collection, file 4.

  1. the model reader: the 38 inputs from the nano's stored columns and the margins of the deployed heads against the
     heads' parity reference (first 5 events of AAAA file 4: inputs exact, margins to float rounding);
  2. the converter: the 38 inputs built from the replay's "implemented now" against the inputs from the nano's stored
     CMSSW columns, track by track, and the six head outputs from both.

  python validate_heads.py [--nev N] [--scen AAAA AAAI IAAA]     -> checks/heads_converter_validation.json
"""
import argparse
import json

import numpy as np

import heads as HD
import run_scan as RS
import study as ST
from study import AR, CM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nev", type=int, default=None)
    ap.add_argument("--scen", nargs="+", default=list(ST.SCENARIOS))
    a = ap.parse_args()
    CM.set_collection("extended")
    S = RS.Study()
    res = {"collection": "extended", "file": 4, "scenarios": {}}
    for scen in a.scen:
        path = ST.input_path(scen, 4)
        HD.refuse_truth(path)
        cfg = CM.producer_config(path)
        tools = ST.tools(cfg)
        arm = AR.v3(cfg)
        H = HD.Heads(ST.CACHE / "heads", scen)
        keys = AR.OP._keys(str(path))
        br = list(dict.fromkeys(AR.branches(path) + HD.nano_branches(keys) + ["run", "luminosityBlock", "event"]))
        Xs, Xr, d0s, d0r, ev = [], [], [], [], []
        for start, A in CM.iterate(path, br, chunk=10, nev=a.nev):
            ctx = AR.prepare(A, S.geo)
            out = AR.run_arm(ctx, S.geo, cfg, scen, arm, tools)
            Ts = HD.stored_columns(A)
            Tr = HD.replay_columns(S, ctx, out, Ts, CM.pt_constant(ctx), cfg)
            xs, names = HD.matrix(Ts)
            xr, _ = HD.matrix(Tr)
            Xs.append(xs), Xr.append(xr), d0s.append(Ts["r_d0"]), d0r.append(Tr["r_d0"])
            ev.append(np.stack([np.asarray(A[k])[ctx["tev"]] for k in ("run", "luminosityBlock", "event")], 1))
        Xs, Xr, d0s, d0r, ev = (np.concatenate(v) for v in (Xs, Xr, d0s, d0r, ev))
        ms, mr = H.margins(Xs, d0s), H.margins(Xr, d0r)
        e = {"tracks": int(len(Xs)), "otNPar": cfg.get("otNPar")}
        if scen == "AAAA":                                # the parity reference: the first 5 events, entry order
            P = np.load(ST.CACHE / "heads" / "parity" / "parity_AAAA_f04_5ev_SmartPixels.npz")
            n = len(P["X"])
            same_ev = bool(np.array_equal(P["event_id"], ev[:n]))
            e["parity"] = {"tracks": int(n), "same_events_and_order": same_ev,
                           "inputs_identical": bool(np.array_equal(P["X"].astype(np.float32), Xs[:n])),
                           "inputs_max_abs_difference": float(np.abs(P["X"] - Xs[:n]).max()),
                           "margins_max_abs_difference": float(np.abs(P["margin"] - ms[:n]).max())}
        same = (Xs[:, 22] == Xr[:, 22]) & (Xs[:, 18] == Xr[:, 18]) & (np.abs(Xs[:, 24] - Xr[:, 24]) <= 0.01 * np.abs(Xs[:, 24]) + 1e-6)
        e["tracks_same_clusters_proxy"] = int(same.sum())
        feats = {}
        for i, nme in enumerate(names):
            d = np.abs(Xr[:, i].astype(np.float64) - Xs[:, i])
            rel = d / np.maximum(np.abs(Xs[:, i]), 1e-12)
            okr = (d == 0) | (rel < 1e-3)
            miss = (Xs[:, i] == HD.MISSING) != (Xr[:, i] == HD.MISSING)
            feats[nme] = {"identical_share": float((d == 0).mean()), "within_1e-3_share": float(okr.mean()),
                          "within_1e-3_share_same_clusters": float(okr[same].mean()),
                          "missing_flag_differs_share_same_clusters": float(miss[same].mean()),
                          "median_rel_difference_same_clusters": float(np.median(rel[same]))}
        e["inputs"] = feats
        e["inputs_worst_same_clusters"] = min(feats.items(), key=lambda kv: kv[1]["within_1e-3_share_same_clusters"])[0]
        sc = {}
        for i, h in enumerate(HD.HEADS):
            d = np.abs(mr[:, i] - ms[:, i])
            sc[h] = {"margin_abs_difference_q": {q: float(np.quantile(d[same], float(q))) for q in ("0.5", "0.9", "0.99", "0.999")},
                     "share_beyond_0.05_same_clusters": float((d[same] > 0.05).mean()),
                     "share_beyond_0.05_all_tracks": float((d > 0.05).mean()),
                     "margin_median_stored": float(np.median(ms[:, i])), "margin_median_replay": float(np.median(mr[:, i])),
                     "margin_w68_stored": float(np.diff(np.quantile(ms[:, i], [0.16, 0.84]))[0] / 2)}
        e["head_outputs"] = sc
        e["window_differs_share"] = float(((np.abs(d0s) <= 0.02) != (np.abs(d0r) <= 0.02)).mean())
        res["scenarios"][scen] = e
        print(scen, json.dumps({k: v for k, v in e.items() if k not in ("inputs",)}, indent=1), flush=True)
        print({k: (round(v["within_1e-3_share_same_clusters"], 4), round(v["missing_flag_differs_share_same_clusters"], 4)) for k, v in feats.items()})
    name = "heads_converter_validation" + ("" if a.nev is None else f"_{a.nev}ev") + ".json"
    json.dump(res, open(ST.LOCAL / "checks" / name, "w"), indent=1)


if __name__ == "__main__":
    main()
