"""The "reads nothing the chip does not send" check on the EXTENDED collection (code review round 2, LOW): cluster
size, published errors, class and truth angles blanked after the emulation step; the refit output, the forced refit
and the 38 head inputs and six head outputs must be bit-identical. File 4, the first events.

  python ext_blanking.py [--nev 10]     -> checks/blanking_extended.json
"""
import argparse
import json

import numpy as np

import emulate as EM
import heads as HD
import run_scan as RS
import study as ST
from study import AR, CM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nev", type=int, default=10)
    a = ap.parse_args()
    CM.set_collection("extended")
    S = RS.Study()
    specs = {r["name"]: r["spec"] for r in json.load(open(RS.HERE / "arms.json"))["arms"]}
    res = {"collection": "extended", "nev": a.nev, "differences": {}, "tracks": {}}
    for scen in ("AAAA", "IAAA"):
        path = ST.input_path(scen, 4)
        cfg = CM.producer_config(path)
        tools = ST.tools(cfg)
        H = HD.Heads(ST.CACHE / "heads", scen)
        keys = AR.OP._keys(str(path))
        br = list(dict.fromkeys(AR.branches(path) + HD.nano_branches(keys) + ["run", "luminosityBlock", "event"]
                                + AR.SPN.cluster_branches(keys, list(EM.EXTRA_CLUSTER_COLS), disc_scope="all")))
        bad = {n: set() for n in ("network_float", "alloc_4444", "alloc_6604", "alloc_8800", "down_y_4b")}
        ntrk = 0
        for start, A in CM.iterate(path, br, chunk=10, nev=a.nev):
            ctx = AR.prepare(A, S.geo)
            ntrk += ctx["ntrk"]
            kpt = CM.pt_constant(ctx)
            B = EM.base(ctx, S.geo, A, S.net, S.links, S.LT)
            Ts = HD.stored_columns(A)
            for n in bad:
                arm, t_a = RS.build_arm(AR, cfg, n, specs[n]), S.tools_for(tools, specs[n])
                sent = "beta" in specs[n].get("send", ["alpha", "beta"])
                outs = []
                for c in (EM.emulate(ctx, B, specs[n], S.LT),):
                    for cc in (c, EM.blank_unsent_information(c)):
                        o = AR.run_arm(cc, S.geo, cfg, scen, arm, t_a)
                        X, _ = HD.matrix(HD.replay_columns(S, cc, o, Ts, kpt, cfg, alpha_sent=sent))
                        outs.append((o, X, H.margins(X, o["a"][:, 4])))
                (o1, X1, m1), (o2, X2, m2) = outs
                for k in ("a", "C", "refit"):
                    if not np.array_equal(o1[k], o2[k], equal_nan=True):
                        bad[n].add(k)
                if not np.array_equal(X1, X2):
                    bad[n].add("head inputs")
                if not np.array_equal(m1, m2, equal_nan=True):
                    bad[n].add("head outputs")
        res["differences"][scen] = {n: sorted(v) for n, v in bad.items()}
        res["tracks"][scen] = ntrk
    res["pass"] = not any(v for d in res["differences"].values() for v in d.values())
    json.dump(res, open(ST.LOCAL / "checks" / "blanking_extended.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
