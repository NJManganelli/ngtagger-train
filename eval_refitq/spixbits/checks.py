"""Launch checks of PLAN.md section 8 that are cheap (file 4, AAAA and IAAA, the first events).

  identity   the emulation step with everything float and the angles untouched = the replay without it (bitwise)
  blanking   cluster size, published errors, class and truth angles blanked after the step: output identical (bitwise)
  quantizer  on synthetic uniform input: RMS of (quantized - input) = step / sqrt(12) within 2 %
  d0 sign    (--d0sign, from a 100-event unit output) refit d0 against true d0 on b/c daughters with |d0| > 200 um

  python checks.py [--nev 20]         -> checks/launch_checks.json
"""
import argparse
import json

import numpy as np

import emulate as EM
import run_scan as RS
import study as ST
from study import AR, CM


def same(o1, o2):
    bad = [k for k in ("a", "C", "refit", "nacc") if not np.array_equal(o1[k], o2[k], equal_nan=True)]
    bad += [f"rec.{k}" for k in o1["rec"] if not np.array_equal(o1["rec"][k], o2["rec"][k], equal_nan=True)]
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nev", type=int, default=20)
    a = ap.parse_args()
    res = {"identity": {}, "blanking": {}, "quantizer": {}, "material": []}
    S = RS.Study()
    reg = json.load(open(RS.HERE / "arms.json"))
    specs = {r["name"]: r["spec"] for r in reg["arms"]}
    import fom as FM
    for scen in ("AAAA", "IAAA"):
        path = ST.input_path(scen, ST.DEV_FILE)
        cfg = CM.producer_config(path)
        tools = ST.tools(cfg)
        keys = AR.OP._keys(str(path))
        br = list(dict.fromkeys(AR.branches(path) + ["run", "luminosityBlock", "event"]
                                + AR.SPN.cluster_branches(keys, list(EM.EXTRA_CLUSTER_COLS), disc_scope="all")))
        ident, blank = [], {n: [] for n in ("network_float", "alloc_4444", "alloc_6604", "alloc_4444_flagunusable",
                                            "alloc_4444_globalrange", "noise_y_4b", "bending_float")}
        nfl = 0
        for start, A in CM.iterate(path, br, chunk=10, nev=a.nev):
            ctx = AR.prepare(A, S.geo)
            kpt = CM.pt_constant(ctx)
            B = EM.base(ctx, S.geo, A, S.net, S.links, S.LT)
            base = AR.run_arm(ctx, S.geo, cfg, scen, AR.v3(cfg), tools)
            arm0 = RS.build_arm(AR, cfg, "identity", {"angles": "today", "cap": False})
            ident += same(base, AR.run_arm(EM.emulate(ctx, B, {"angles": "today", "cap": False}, S.LT), S.geo, cfg,
                                           scen, arm0, tools))
            path_, seq = AR.visit_plan(ctx, S.geo, cfg, scen, arm0)
            o = np.flatnonzero(base["rec"]["acc"].any(1))
            sel_ci = np.where(base["rec"]["acc"][o], base["rec"]["ci"][o], -1)
            for n in blank:
                arm = RS.build_arm(AR, cfg, n, specs[n])
                t_a = S.tools_for(tools, specs[n])
                c1 = EM.emulate(ctx, B, specs[n], S.LT)
                c2 = EM.blank_unsent_information(c1)
                o1 = AR.run_arm(c1, S.geo, cfg, scen, arm, t_a)
                blank[n] += same(o1, AR.run_arm(c2, S.geo, cfg, scen, arm, t_a))
                F1 = AR.refit_forced(arm, c1, S.geo, cfg, t_a, o, seq[o], base["rec"]["mi"][o], sel_ci, path_, kpt)
                F2 = AR.refit_forced(arm, c2, S.geo, cfg, t_a, o, seq[o], base["rec"]["mi"][o], sel_ci, path_, kpt)
                if not (np.array_equal(F1.a, F2.a) and np.array_equal(F1.C, F2.C)):
                    blank[n].append("forced")
        res["identity"][scen] = sorted(set(ident))
        res["blanking"][scen] = {n: sorted(set(v)) for n, v in blank.items()}
        if ident:
            res["material"].append(f"identity {scen}: differs in {sorted(set(ident))}")
        for n, v in blank.items():
            if v:
                res["material"].append(f"blanking {scen} {n}: differs in {sorted(set(v))}")
    rng = np.random.default_rng(1)
    for step_um in (600, 150, 37.5, 9.4, 2.3):
        x = rng.uniform(-0.8, 0.8, 100000)
        ph = rng.random(100000)
        x0 = -0.84 + np.floor((x + 0.84) / 0.24) * 0.24
        q = EM.quantize_position(x, x0, step_um * 1e-4, ph)
        r = float(np.sqrt(np.mean((q - x) ** 2)) / (step_um * 1e-4))
        res["quantizer"][f"position {step_um} um"] = {"rms_over_step": r, "max_abs_over_step": float(np.abs(q - x).max() / (step_um * 1e-4))}
        if abs(r / 0.288675 - 1) > 0.02:
            res["material"].append(f"quantizer position {step_um} um: RMS / step {r:.4f}")
    for bits in (2, 4, 7):
        v = rng.uniform(-1.0, 2.0, 100000)
        q, f = EM.quantize_angle(v, np.full(len(v), -0.9), np.full(len(v), 1.9), bits)
        stp = 2.8 / (2 ** bits - 2)
        m = f == 0
        r = float(np.sqrt(np.mean((q[m] - v[m]) ** 2)) / stp)
        res["quantizer"][f"angle {bits} bits"] = {"rms_over_step": r, "flagged": float((~m).mean()),
                                                 "expected_flagged": 0.2 / 3.0}
        if abs(r / 0.288675 - 1) > 0.02:
            res["material"].append(f"quantizer angle {bits} bits: RMS / step {r:.4f}")
    json.dump(res, open(ST.LOCAL / "checks" / "launch_checks.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
