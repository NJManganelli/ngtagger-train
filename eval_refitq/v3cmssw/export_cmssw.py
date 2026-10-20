"""Export the seedless refit v3 payload set for CMSSW from the replay's specification (v3cmssw/PLAN.md §2).

    python eval_refitq/v3cmssw/export_cmssw.py --spec <v3cmssw/spec> --out <CMSSW src>/L1Trigger/Phase3SmartPixels/data/refit_v3/v1 \
        --unit-ref <CMSSW src>/L1Trigger/Phase3SmartPixels/test/refit_v3_unit_reference.json

Every lookup is computed through the reference implementation (the spec's code snapshot, `spec/code/`), so the
payload is the replay's own arithmetic tabulated at every index along each axis, then checked against itself with the
Python correctionlib before anything is written. Output, all in --out:
  rtables.json      correctionlib: R_u (sizeX x |cotAlpha_pred|), R_v (sizeY x |cotBeta_pred|) -> width [cm]
  angles.json       correctionlib: module_orientation (detId -> sign), angle_barrel (surface, sign, key, quantity),
                    angle_disc (surface, key, quantity); quantity c_<a> / w_<a> / mode_<a>, mode 0 no entry,
                    1 entry (m - c sigma, w sigma), 2 entry without a finite centre or width (the angle is unusable)
  skip_price.json   correctionlib: skip_price (dims, cut-surface code, first visit step) with the fallback resolved
  seed_model.json   the seed model, content unchanged except non-finite numbers replaced by +-BIG
  material.json     the barrel planes, the disc-going planes, the scales, the IL1 module and the beam pipe
  manifest.json     sources, sha256, registry ids, the reference commit, the physics-check registry ids
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

BIG = 1.0e30                     # stands for +-inf in every exported file (no non-finite JSON numbers)
SCHEMA = "spix-refit-v3-payload/1"
SURF_BARREL = (1, 2, 3, 4)
SURF_DISC = (11, 12, 13, 14)
CUT_CODES = SURF_BARREL + tuple(s + 100 for s in SURF_BARREL) + SURF_DISC
DIMS = ("xy", "xya", "xyb", "xyab")
ANGLES = ("alpha", "beta")


def _code_paths(spec: Path):
    """Put the spec's code snapshot first on sys.path; the live combinatorics directory supplies only modules the
    snapshot does not carry (common.py imports omnibus_support)."""
    code = spec / "code"
    live = Path(__file__).resolve().parents[1]
    for p in (code / "eval_refitq" / "replay", code / "eval_refitq" / "combinatorics", code / "src"):
        sys.path.insert(0, str(p))
    sys.path.append(str(live / "combinatorics"))
    return (code / "COMMIT").read_text().strip()


def md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def finite(x):
    """JSON-safe: +-inf -> +-BIG, NaN refused (every NaN must have been resolved by the caller)."""
    if isinstance(x, dict):
        return {k: finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [finite(v) for v in x]
    if isinstance(x, float):
        if math.isnan(x):
            raise ValueError("NaN reached the export")
        if math.isinf(x):
            return BIG if x > 0 else -BIG
    return x


def edges_finite(e):
    return [float(min(max(v, -BIG), BIG)) for v in e]


def correction(name, desc, inputs, data, version=1):
    return {"name": name, "description": desc, "version": version, "inputs": inputs,
            "output": {"name": "value", "type": "real"}, "data": data}


def cset(corrs, desc):
    return {"schema_version": 2, "description": desc, "corrections": corrs}


def category(inp, items, default=None):
    out = {"nodetype": "category", "input": inp, "content": [{"key": k, "value": v} for k, v in items]}
    if default is not None:
        out["default"] = default
    return out


def binning(inp, edges, content):
    return {"nodetype": "binning", "input": inp, "edges": edges_finite(edges), "content": content, "flow": "clamp"}


# ---------------------------------------------------------------- position errors (R tables)
def export_rtables(cal):
    corrs = []
    for nm, size_in, cot_in in (("u", "sizeX", "absCotAlphaPred"), ("v", "sizeY", "absCotBetaPred")):
        t = cal["R_tables"][nm]
        W = np.asarray(t["width_cm"], np.float64)
        se, ce = [float(x) for x in t["size_edges"]], [float(x) for x in t["cot_edges"]]
        assert W.shape == (len(se) - 1, len(ce) - 1)
        data = {"nodetype": "multibinning", "inputs": [size_in, cot_in], "edges": [edges_finite(se), edges_finite(ce)],
                "content": [float(v) for v in W.ravel()], "flow": "clamp"}
        corrs.append(correction(f"R_{nm}", f"refit v3 position error, local {'x' if nm == 'u' else 'y'} [cm] "
                                f"(refit_calibration_fit.apply_R; built from {cal['R_tables']['built_from']})",
                                [{"name": size_in, "type": "real"}, {"name": cot_in, "type": "real"}], data))
    return cset(corrs, "SmartPixels refit v3: position-error (R) tables")


def rtable_reference(cal, RCF):
    """(sizeX, sizeY, cotA_pred, cotB_pred) -> (sigX, sigY) through apply_R at every index pair (bin interiors, edges,
    and beyond both ends)."""
    t = cal["R_tables"]
    out = []
    for nm in ("u", "v"):
        se, ce = t[nm]["size_edges"], t[nm]["cot_edges"]
        sizes = sorted(set([0.0] + [float(x) for x in se] + [0.5 * (a + b) for a, b in zip(se[:-1], se[1:])] + [200.0]))
        cots = sorted(set([float(x) for x in ce] + [0.5 * (a + b) for a, b in zip(ce[:-1], ce[1:])] + [150.0]))
        for s in sizes:
            for c in cots:
                for sign in (1.0, -1.0):
                    sz = np.array([s]), np.array([s])
                    ca = np.array([sign * c]) if nm == "u" else np.array([0.0])
                    cb = np.array([sign * c]) if nm == "v" else np.array([0.0])
                    r = RCF.apply_R(t, sz[0], sz[1], ca, cb, np.ones((1, 4)))
                    out.append({"table": nm, "size": s, "cotPred": float(sign * c), "width": float(r[0, 0 if nm == "u" else 1])})
    return out


# ---------------------------------------------------------------- angle tables
def _bin_keys(SA, surface):
    """One key per fine bin of `surface` (inside the bin; the end bins at their finite edge -+ 1) and the bin's
    signed edges."""
    se = SA.signed_edges(SA.abs_edges(surface))
    keys = []
    for b in range(len(se) - 1):
        lo, hi = se[b], se[b + 1]
        keys.append(hi - 1.0 if not np.isfinite(lo) else (lo + 1.0 if not np.isfinite(hi) else 0.5 * (lo + hi)))
    return se, np.asarray(keys, np.float64)


def barrel_entries(AI, SA, imp):
    """{(surface, sign): [per fine bin (c_a, w_a, mode_a, c_b, w_b, mode_b)]} through angle_import.calibrate's
    arithmetic: a module of orientation s, its key s cotB / sqrt(1 + cotA^2); calibrate's nan_to_num (centre 0,
    width 1) for an absent angle of an existing entry."""
    out = {}
    for S in SURF_BARREL:
        se, keys = _bin_keys(SA, S)
        for s in (-1, 1):
            n = len(keys)
            surf = np.full(n, S)
            orient = np.full(n, s, np.int8)
            rows = imp.table.rows_of(AI.KEY, surf, keys, SA.facing(surf, orient))
            vals = []
            for i in range(n):
                e = []
                for ang in ANGLES:
                    if rows[i] < 0:
                        e += [0.0, 1.0, 0]
                        continue
                    c = imp.table.get(AI.KEY, ang, "centre", rows[i:i + 1])
                    w = imp.table.get(AI.KEY, ang, "width", rows[i:i + 1])
                    cl = SA.to_local(ang, c, orient[i:i + 1], surf[i:i + 1])
                    e += [float(np.nan_to_num(cl)[0]), float(np.nan_to_num(w, nan=1.0)[0]), 1]
                vals.append(e)
            out[(S, s)] = (se, vals)
    return out


def disc_entries(DA, SA, dimp):
    """{surface: (edges, [per fine bin (c_a, w_a, mode_a, c_b, w_b, mode_b)])}: DiscImport's rows at the local key;
    mode 2 where the entry's centre or width is not finite (DiscImport.calibrate then gives NaN: the angle drops)."""
    out = {}
    for S in SURF_DISC:
        se, keys = _bin_keys(SA, S)
        n = len(keys)
        rows = dimp.table.rows_of(DA.KEY, np.full(n, S), keys, np.zeros(n, np.int8))
        vals = []
        for i in range(n):
            e = []
            for ang in ANGLES:
                if rows[i] < 0:
                    e += [0.0, 1.0, 0]
                    continue
                c, w = dimp.centre_width(ang, rows[i:i + 1])
                ok = bool(np.isfinite(c[0]) and np.isfinite(w[0]))
                e += [float(c[0]) if ok else 0.0, float(w[0]) if ok else 1.0, 1 if ok else 2]
            vals.append(e)
        out[S] = (se, vals)
    return out


QUANT = ("c_alpha", "w_alpha", "mode_alpha", "c_beta", "w_beta", "mode_beta")


def _quantity_node(e):
    return category("quantity", [(q, float(v)) for q, v in zip(QUANT, e)])


def export_angles(imp, bar, disc):
    o = imp.table.doc["orientation"]
    ori = category("detId", sorted((int(d), float(s)) for d, s in zip(o["det_id"], o["sign"])), default=0.0)
    bnode = category("surface", [(S, category("sign", [(s, binning("key", bar[(S, s)][0],
                                                                    [_quantity_node(e) for e in bar[(S, s)][1]]))
                                                       for s in (-1, 1)])) for S in SURF_BARREL])
    dnode = category("surface", [(S, binning("key", disc[S][0], [_quantity_node(e) for e in disc[S][1]]))
                                 for S in SURF_DISC])
    return cset([
        correction("module_orientation", "barrel module orientation sign from the angle table (0: unknown module, no "
                   "table entry)", [{"name": "detId", "type": "int"}], ori),
        correction("angle_barrel", "barrel angle-table entry at the predicted signed incidence key s cotB / sqrt(1 + "
                   "cotA^2): module-frame centre c (published-sigma units, sign included), width w, mode",
                   [{"name": "surface", "type": "int"}, {"name": "sign", "type": "int"}, {"name": "key", "type": "real"},
                    {"name": "quantity", "type": "string"}], bnode),
        correction("angle_disc", "disc module-frame angle-table entry at the predicted local key sign(cotB) sqrt(cotA^2 "
                   "+ cotB^2), facings pooled: centre c, width w, mode",
                   [{"name": "surface", "type": "int"}, {"name": "key", "type": "real"},
                    {"name": "quantity", "type": "string"}], dnode)],
        "SmartPixels refit v3: angle-error tables (P0.3 v1a1e, barrel global-key and disc module-frame)")


def angle_reference(AI, DA, SA, imp, dimp, geo_modules, rng):
    """Per surface, orientation sign and fine bin: a real module (detId) of that surface and sign and a predicted
    (cotA, cotB) whose key lands in the bin, with the replay's calibrated (m', sigma') for m = 0.3, sigma = 0.05 per
    angle: angle_import.calibrate on the barrel (orientation from the table's map), DiscImport on the discs."""
    o = imp.table.doc["orientation"]
    sign_of = dict(zip((int(d) for d in o["det_id"]), (int(x) for x in o["sign"])))
    dets = {}
    for m in geo_modules:
        if m["subdet"] == "PXB" and int(m["detId"]) in sign_of:
            dets.setdefault((int(m["layer"]), sign_of[int(m["detId"])]), []).append(int(m["detId"]))
        elif m["subdet"] == "PXF" and m.get("disk") in (1, 2, 3, 4):
            dets.setdefault((10 + int(m["disk"]), 0), []).append(int(m["detId"]))
    m0, s0 = 0.3, 0.05
    out = []
    for S in SURF_BARREL:
        se, keys = _bin_keys(SA, S)
        for s in (-1, 1):
            if not dets.get((S, s)):
                continue
            for b, key in enumerate(keys):
                det = int(rng.choice(dets[(S, s)]))
                ca = float(rng.uniform(-0.5, 0.5))
                cb = float(key * math.sqrt(1.0 + ca * ca) / s)
                ma, mb, sa, sb, _ = AI.calibrate(imp, np.array([S]), np.array([det], np.uint32), np.array([ca]),
                                                 np.array([cb]), np.array([m0]), np.array([m0]), np.array([s0]),
                                                 np.array([s0]), 1.0, 1.0, "fixed")
                out.append({"region": "barrel", "surface": S, "sign": s, "detId": det, "cotA": ca, "cotB": cb, "bin": b,
                            "alpha": {"m": float(ma[0]), "sigma": float(sa[0]), "usable": True},
                            "beta": {"m": float(mb[0]), "sigma": float(sb[0]), "usable": True}})
    for S in SURF_DISC:
        se, keys = _bin_keys(SA, S)
        for b, key in enumerate(keys):
            det = int(rng.choice(dets[(S, 0)]))
            ca = float(abs(key) * rng.uniform(0.0, 0.9))
            cb = math.copysign(math.sqrt(max(key * key - ca * ca, 0.0)), key)
            rows = dimp.rows(np.array([S]), np.array([ca]), np.array([cb]))
            e = {"region": "disc", "surface": S, "sign": 0, "detId": det, "cotA": ca, "cotB": cb, "bin": b}
            for ang in ANGLES:
                if rows[0] < 0:
                    e[ang] = {"m": m0, "sigma": s0, "usable": True}
                    continue
                mm, ss = dimp.calibrate(ang, np.array([m0]), np.array([s0]), rows)
                ok = bool(np.isfinite(mm[0]) and np.isfinite(ss[0]))
                e[ang] = {"m": float(mm[0]) if ok else 0.0, "sigma": float(ss[0]) if ok else 0.0, "usable": ok}
            out.append(e)
    for e in out:
        e["m"], e["sigma"] = m0, s0
    return out


def orientation_reference(imp, rng):
    """(detId, sign) for 300 modules of the table's map and 5 detIds outside it (sign 0)."""
    o = imp.table.doc["orientation"]
    ids = np.asarray(o["det_id"], np.int64)
    pick = rng.choice(len(ids), size=min(300, len(ids)), replace=False)
    out = [{"detId": int(ids[i]), "sign": int(o["sign"][i])} for i in pick]
    known = set(ids.tolist())
    extra = [d for d in (1, 303042564 + 7, 344000000, 400000001, 2**31 - 1) if d not in known]
    out += [{"detId": d, "sign": 0} for d in extra]
    want = imp.table.orientation(np.array([e["detId"] for e in out], np.uint32))
    assert [int(x) for x in want] == [e["sign"] for e in out]
    return out


# ---------------------------------------------------------------- skip price
def skip_level(cuts, dims, S, f):
    """Which level of the cut file's fallback chain (cuts.Cuts._lookup) gives P: 0 "S|f", 1 "S", 2 "S-100|f",
    3 "S-100", 4 "*" (the per-path counts of the closure)."""
    tab = cuts.doc["cuts"][f"gate|{dims}|calibrated"]
    chain = [f"{S}|{f}", f"{S}"] + ([f"{S - 100}|{f}", f"{S - 100}"] if S > 100 else [])
    for lev, k in zip((0, 1, 2, 3), chain):
        e = tab.get(k)
        if e is not None and e["value"] is not None and e["n"] >= CU_MIN_N:
            return lev
    return 4


CU_MIN_N = None   # set from the cuts module at export


def export_skip_price(Cuts, cut_path):
    cuts = Cuts.load(cut_path)
    table, levels, ref = [], [], []
    for dims in DIMS:
        codes = []
        lcodes = []
        for S in CUT_CODES:
            firsts, lf = [], []
            for f in (0, 1):
                v = float(cuts.chi2("gate", dims, "calibrated", np.array([S]), np.array([bool(f)]))[0])
                lev = skip_level(cuts, dims, S, f)
                firsts.append((f, v))
                lf.append((f, float(lev)))
                ref.append({"dims": dims, "code": S, "first": f, "value": v, "level": lev})
            codes.append((S, category("first", firsts)))
            lcodes.append((S, category("first", lf)))
        star = float(cuts.doc["cuts"][f"gate|{dims}|calibrated"]["*"]["value"])
        codes.append((0, category("first", [(0, star), (1, star)])))   # code 0: the table's "*" value (a known-bad)
        ref += [{"dims": dims, "code": 0, "first": f, "value": star, "level": 4} for f in (0, 1)]
        table.append((dims, category("code", codes)))
        lcodes.append((0, category("first", [(0, 4.0), (1, 4.0)])))
        levels.append((dims, category("code", lcodes)))
    corr = correction("skip_price", "the two-layer hit choice's skip price P: the 99 % gate cut "
                      "(gate|<dims>|calibrated of the cut file) per cut-surface code (+100: a barrel visit of a "
                      "disc-going track; 0: the table's '*' value, test-only) and first visit step; the cut "
                      "file's fallback chain resolved",
                      [{"name": "dims", "type": "string"}, {"name": "code", "type": "int"},
                       {"name": "first", "type": "int"}], category("dims", table))
    lev = correction("skip_price_level", "the fallback level that gives the skip price (diagnostic, the per-path "
                     "counts): 0 surface|step, 1 surface, 2 barrel surface|step for a disc-going track's barrel "
                     "visit, 3 that barrel surface, 4 the table's '*'",
                     [{"name": "dims", "type": "string"}, {"name": "code", "type": "int"},
                      {"name": "first", "type": "int"}], category("dims", levels))
    return cset([corr, lev], "SmartPixels refit v3: skip price"), ref


# ---------------------------------------------------------------- seed model
def export_seed(doc):
    out = finite(doc)
    out["layout"] = {"theta_per_segment": "[A (9), B (9)] constants, then the same 18 slopes in |eta| - segment centre",
                     "block_entries": "T: 3 log-sigma, 3 partial correlations (12, 13, 23); L: 2 log-sigma, 1 partial "
                                      "correlation", "z_max": 0.999, "log_sigma_clip": 30.0,
                     "segment_centre_cap": 2.6, "eta_last_eps": 1e-9, "big": BIG}
    return out


def seed_reference(SM, theta, doc):
    import jax.numpy as jnp
    seg = {**doc["segments"], "edges": tuple(doc["segments"]["edges"])}
    dom = doc["domain"]
    edges = [float(x) for x in seg["edges"]]
    etas = sorted(set([0.0, 0.3, 0.6, 1.0, 1.2, 1.6, 2.0, 2.4, 2.5, 2.55, 2.59, 2.7, 3.0]
                      + [e + d for e in edges for d in (-1e-9, 0.0, 1e-9)]
                      + [i * dom["eta_cell"] + d for i in range(19) for d in (-1e-7, 1e-7)]))
    etas = [x for x in etas if x >= 0.0]
    ps = [1.0, 1.9, 2.0, 2.1, 3.0, 4.5, 6.0, 10.0, 11.0, 17.0, 26.0, 40.0, 60.0, 500.0]
    out = []
    for e in etas:
        for p in ps:
            ae, pp = np.array([e]), np.array([p])
            sidx = SM.segment_index(seg, ae, np.zeros(1, int))
            pe, _ = SM.p_eff(dom, ae, pp)
            dC = np.asarray(SM.dC_of(jnp.asarray(theta), jnp.asarray(sidx), jnp.asarray(SM.segment_x(seg, ae, sidx)),
                                     jnp.asarray(pe)))[0]
            out.append({"absEta": e, "p": p, "dC": dC.tolist()})
    return out


# ---------------------------------------------------------------- material
def export_material(cal, disc_doc, disc_scale, AR):
    def plane(p):
        return {k: v for k, v in p.items() if k in ("kind", "r", "half_z", "z", "r_ext", "xx0", "disk", "what")}
    barrel = [{"r": (r if not isinstance(r, str) else None), "layer": (int(r[2:]) if isinstance(r, str) else 0),
               "xx0": float(x), "what": w} for r, x, w in cal["planes"]]
    il1 = {"layer": 1, "xx0": float(AR.IL1_MODULE[1]), "what": AR.IL1_MODULE[2]}
    pipe = {"r": float(AR.BEAM_PIPE[0]), "xx0": float(AR.BEAM_PIPE[1]), "what": AR.BEAM_PIPE[2]}
    return finite({"schema": "spix-refit-v3-material/1",
                   "highland": {"c0": 0.0136, "c1": 0.038},
                   "barrel_planes": barrel, "material_scale": float(cal["material_scale"]),
                   "ot_inner_r_cm": float(cal["ot_inner_r_cm"]),
                   "disc_going_planes": [plane(p) for p in disc_doc["planes"]],
                   "disc_material_scale": float(disc_scale), "cylinder_material_scale": float(cal["material_scale"]),
                   "il1_module": il1, "beam_pipe": pipe,
                   "rules": {"barrel_only": "planes with radius in (R_S, R_prev]; layer planes at the layer's mean "
                                            "module-centre radius; kink at the plane radius",
                             "disc_going": "planes crossed within their extent at arc length s(S) < s <= s(prev), "
                                           "kink at the crossing radius; disc X = scale x/X0 / |tanL|",
                             "inner": "barrel-only: calibration planes + IL1 module + beam pipe down to r = 0; "
                                      "disc-going: disc-going planes + the beam pipe (no |z| limit), 0 < s <= "
                                      "s(innermost accepted)"}})


# ---------------------------------------------------------------- tree2
def tree2_reference(AR, rng):
    cases = []
    for n_case in range(400):
        two = n_case % 4 != 3
        x1 = rng.exponential(20.0, 2)
        x1[rng.random(2) < 0.2] = np.inf
        if x1[0] == np.inf:
            x1[1] = np.inf
        P = float(rng.uniform(80.0, 260.0))
        x2 = None
        if two:
            x2 = rng.exponential(60.0, (3, 2))
            x2[rng.random((3, 2)) < 0.25] = np.inf
            for c in range(3):
                if x2[c, 0] == np.inf:
                    x2[c, 1] = np.inf
        sc = AR.tree2_scores(x1[None, :], None if x2 is None else x2[None], np.array([P]))[0]
        cases.append({"x1": finite(x1.tolist()), "x2": None if x2 is None else finite(x2.tolist()), "P": P,
                      "scores": finite(sc.tolist()), "choice": int(np.argmin(sc))})
    return cases


# ---------------------------------------------------------------- self-check
ULP_TOL = 4e-16      # correctionlib's JSON reader (rapidjson, no full-precision flag) can be 1 ulp off a number


def _same(a, b):
    return a == b or abs(a - b) <= ULP_TOL * max(abs(a), abs(b))


def self_check(out_dir, rref, aref, sref, oref):
    import correctionlib
    R = correctionlib.CorrectionSet.from_file(str(out_dir / "rtables.json"))
    for e in rref:
        cor = R["R_u" if e["table"] == "u" else "R_v"]
        got = cor.evaluate(e["size"], abs(e["cotPred"]))
        assert _same(got, e["width"]), (e, got)
    A = correctionlib.CorrectionSet.from_file(str(out_dir / "angles.json"))
    for e in aref:
        for ang in ANGLES:
            if e["region"] == "barrel":
                assert A["module_orientation"].evaluate(e["detId"]) == e["sign"], e
                key = e["sign"] * e["cotB"] / math.sqrt(1.0 + e["cotA"] ** 2)
                q = lambda n: A["angle_barrel"].evaluate(e["surface"], e["sign"], key, f"{n}_{ang}")
            else:
                key = math.copysign(math.sqrt(e["cotA"] ** 2 + e["cotB"] ** 2), e["cotB"])
                q = lambda n: A["angle_disc"].evaluate(e["surface"], key, f"{n}_{ang}")
            mode = int(q("mode"))
            want = e[ang]
            if mode == 2:
                assert not want["usable"], e
                continue
            mm = e["m"] - q("c") * e["sigma"] if mode == 1 else e["m"]
            ss = q("w") * e["sigma"] if mode == 1 else e["sigma"]
            assert want["usable"] and _same(mm, want["m"]) and _same(ss, want["sigma"]), (e, ang, mm, ss)
    for e in oref:
        assert A["module_orientation"].evaluate(e["detId"]) == e["sign"], e
    KS = correctionlib.CorrectionSet.from_file(str(out_dir / "skip_price.json"))
    K, KL = KS["skip_price"], KS["skip_price_level"]
    for e in sref:
        assert _same(K.evaluate(e["dims"], e["code"], e["first"]), e["value"]), e
        assert KL.evaluate(e["dims"], e["code"], e["first"]) == e["level"], e


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--unit-ref", required=True, type=Path)
    ap.add_argument("--set-id", default="v1")
    ap.add_argument("--physics-checks", default="wpc.c1b.j6-gate,wpc.c1b.j5-known-good-a20,wpc.c1b.j4-cut-fit",
                    help="registry ids of the physics checks this payload set passed (PLAN §2 step 4)")
    a = ap.parse_args()
    commit = _code_paths(a.spec)
    import angle_import as AI
    import arms as AR
    import cuts as CU
    global CU_MIN_N
    CU_MIN_N = CU.MIN_N
    import disc_angles as DA
    import refit_calibration_fit as RCF
    import seedmodel as SM
    from ngtagger.spdefs import angles as SA

    pay = a.spec / "payload"
    cfg = json.loads((a.spec / "config_v3.json").read_text())
    prov = json.loads((pay / "PROVENANCE.json").read_text())
    for name, p in prov.items():
        if (pay / name).exists() and sha256(pay / name) != p["sha256"]:
            raise SystemExit(f"{name}: sha256 differs from the spec's PROVENANCE.json")
    cal = json.loads((pay / "refit_calibration_v3.json").read_text())
    seed_doc = json.loads((pay / "seed_model_v1.json").read_text(), parse_constant=lambda c: {
        "Infinity": math.inf, "-Infinity": -math.inf, "NaN": math.nan}[c])
    imp = AI.AngleImport(str(pay / "angle_table_v1a1e_canon_v3.json"))
    dimp = DA.DiscImport(str(pay / "angle_table_v1a1e_disclocal_canon_v3.json"))
    rng = np.random.default_rng(20261020)

    a.out.mkdir(parents=True, exist_ok=True)
    files = {}

    def write(name, doc, sources):
        path = a.out / name
        with open(path, "w") as fh:
            json.dump(doc, fh, indent=1, allow_nan=False)
            fh.write("\n")
        files[name] = {"sha256": sha256(path), "md5": md5(path), "sources": sources}

    write("rtables.json", export_rtables(cal), ["refit_calibration_v3.json"])
    bar, disc = barrel_entries(AI, SA, imp), disc_entries(DA, SA, dimp)
    write("angles.json", export_angles(imp, bar, disc),
          ["angle_table_v1a1e_canon_v3.json", "angle_table_v1a1e_disclocal_canon_v3.json"])
    sp, sref = export_skip_price(CU.Cuts, pay / "wpc_cuts_c1b_v1.json")
    write("skip_price.json", sp, ["wpc_cuts_c1b_v1.json"])
    write("seed_model.json", export_seed(seed_doc), ["seed_model_v1.json"])
    write("material.json", export_material(cal, json.loads((pay / "disc_material_v1.json").read_text()),
                                           cfg["disc_material_scale"], AR),
          ["refit_calibration_v3.json", "disc_material_v1.json", "scalefit_summary.json"])

    rref = rtable_reference(cal, RCF)
    geo_modules = json.loads((pay / "spix_module_geometry_D121.json").read_text())["modules"]
    aref = angle_reference(AI, DA, SA, imp, dimp, geo_modules, rng)
    oref = orientation_reference(imp, rng)
    self_check(a.out, rref, aref, sref, oref)

    manifest = {"schema": SCHEMA, "set_id": a.set_id, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "reference_commit": commit, "spec": str(a.spec), "config": cfg["name"],
                "disc_material_scale": cfg["disc_material_scale"], "skip_price_cov_model": "calibrated",
                "physics_checks": [x for x in a.physics_checks.split(",") if x],
                "files": files,
                "sources": {k: {kk: v[kk] for kk in ("sha256", "registry", "source") if kk in v} for k, v in prov.items()},
                "big": BIG}
    with open(a.out / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=1, allow_nan=False)
        fh.write("\n")

    unit = {"schema": "spix-refit-v3-unitref/1", "reference_commit": commit, "set_id": a.set_id,
            "rtables": rref, "angles": aref, "orientation": oref, "skip_price": sref,
            "seed": seed_reference(SM, np.asarray(seed_doc["theta"]), seed_doc),
            "tree2": tree2_reference(AR, rng)}
    a.unit_ref.parent.mkdir(parents=True, exist_ok=True)
    with open(a.unit_ref, "w") as fh:
        json.dump(finite(unit), fh, allow_nan=False)
        fh.write("\n")
    print(f"payload set {a.set_id} -> {a.out} ({len(files)} files + manifest); unit reference -> {a.unit_ref}")


if __name__ == "__main__":
    main()
