"""The planted defects of the CMSSW_17 comparison (plan 3.4.7): injected into side A of the primary null split (odd
D121 files) after the null check and before the CMSSW_17 production is read. Each defect has a pre-declared expected
outcome: the quantity it must be flagged in, the expected verdict and class, and the quantities derived from it that may
also move. A defect that is not flagged as expected tightens that tolerance (to its effect / 1.5, if the null still
passes) or makes the quantity report-only.

Defects 8 (PUPPI weight) and 9 (head log sigma) act on the tagger inputs and the heads, which the ML part of the
extraction provides (extract_ml.py); they are run when that part exists. Defect 10 (the 1 GeV rule off on one side) replaces side A
by a re-extraction of the odd files without the rule:
    extract.py --side d121 --mask <mask> --file 1 3 5 7 9 --tp-pt-min 0 --out $OUT/extract_no1gev
"""
from __future__ import annotations

import copy
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as CM  # noqa: E402
import stats as ST  # noqa: E402


def _pick(n, frac, salt):
    """A deterministic pseudo-random choice of a fraction of n objects."""
    h = ST.splitmix64(np.arange(n, dtype=np.uint64) ^ np.uint64(salt))
    return (h >> np.uint64(11)).astype(np.float64) / float(1 << 53) < frac


def d_crossings(side):
    for mask in CM.SP_MASKS:
        d = side.m[mask]
        for key in ("sp", "spx"):
            c = d[f"trk_{key}_spix_NCrossings"].astype(np.int64)
            m = _pick(len(c), 0.10, 11)
            d[f"trk_{key}_spix_NCrossings"] = np.where(m, c + 1, c)


def d_refit_off(side):
    for mask in CM.SP_MASKS:
        d = side.m[mask]
        for key in ("sp", "spx"):
            c = d[f"trk_{key}_spix_RefitPerformed"].astype(np.int64)
            m = _pick(len(c), 0.02, 12) & (c > 0)
            d[f"trk_{key}_spix_RefitPerformed"] = np.where(m, 0, c)


def _scale_res(d, key, par, factor, cell=None):
    r = d[f"trk_{key}_res_{par}"].astype(np.float64)
    m = np.isfinite(r)
    if cell is not None:
        pt_i = CM.bin_index(d[f"trk_{key}_tp_pt"], np.asarray(CM.TRK_PT))
        eta_i = CM.bin_index(np.abs(d[f"trk_{key}_tp_eta"]), np.asarray(CM.TRK_ETA))
        m &= (pt_i == cell[0]) & (eta_i == cell[1]) & (d[f"trk_{key}_tp_hs"].astype(np.int64) == cell[2])
    d[f"trk_{key}_res_{par}"] = np.where(m, r * factor, r).astype(np.float32)


def d_d0_one_cell(side):
    # 3 tolerances on the log scale (plan 3.4.7; 0.42, at the gross bound 0.41, so the class is expected MINOR: the
    # excess over the gross bound is not significant), one cell of the AAAA extended SmartPixels tracks:
    # pT 5-10 GeV, |eta| < 0.8, hard scatter
    _scale_res(side.m["AAAA"], "spx", "d0", float(np.exp(3.0 * CM.TOL["core_rms"][1])), cell=(2, 0, 1))


def d_d0_small(side):
    for mask in CM.MASKS:
        for key in (("ot", "otx") if mask == CM.BASELINE_MASK else ("sp", "spx")):
            _scale_res(side.m[mask], key, "d0", 1.05)


def d_z0_bias(side):
    # the OT extended track z0 residual shifted by 0.35 core RMS in |eta| 0.8-1.4 (IIII)
    d = side.m[CM.BASELINE_MASK]
    r = d["trk_otx_res_z0"].astype(np.float64)
    eta_i = CM.bin_index(np.abs(d["trk_otx_tp_eta"]), np.asarray(CM.TRK_ETA))
    m = np.isfinite(r) & (eta_i == 1)
    core = ST.Resid(r[m]).summaries(np.ones(m.sum()))["core"]
    d["trk_otx_res_z0"] = np.where(m, r + 0.35 * core, r).astype(np.float32)


def d_disc_flip(side):
    d = side.m[CM.CLUSTER_MASK]
    m = (d["cl_tpEmuClass"] == 1) & (d["cl_surface"] > 10) & (d["cl_side"] == 2)
    d["cl_localCotAlpha"] = np.where(m, -d["cl_localCotAlpha"], d["cl_localCotAlpha"])


def d_puppi(side):
    """Defect 8: every constituent's PUPPI weight raised by 0.1."""
    for mask in CM.MASKS:
        d = side.m[mask]
        d["cn_puppiweight"] = (d["cn_puppiweight"] + 0.1).astype(np.float32)


def d_head_logsigma_d0(side):
    """Defect 9: the heads' log sigma_hat(d0) raised by 0.15 (between the tolerance and the gross bound), on the
    tracks and on the tagger's head inputs (constituents with a track; -1 = no track)."""
    for mask in CM.MASKS:
        d = side.m[mask]
        d["hd_m1"] = (d["hd_m1"] + 0.15).astype(np.float32)
        x = d["cn_trkjoint_d0"]
        d["cn_trkjoint_d0"] = np.where(x != -1, x + 0.15, x).astype(np.float32)


def d_no_1gev_rule(side):
    """Defect 10: side A is replaced by the odd files extracted without the 1 GeV rule."""
    import engine as EN
    return EN.Side("D121 odd files, 1 GeV rule off", "d121", side.files, base=CM.OUT / "extract_no1gev")


def d_half_clusters(side):
    d = side.m[CM.CLUSTER_MASK]
    for S in CM.SURFACES:
        d[f"ev_ncl_{S}"] = (d[f"ev_ncl_{S}"] // 2).astype(np.int64)


DEFECTS = {
    "crossings_plus1": (d_crossings, "control:*:crossings", "disagrees", "MATERIAL"),
    "refit_off_2pc": (d_refit_off, "refit:*:refit performed", "disagrees", "MATERIAL"),
    "d0_one_cell": (d_d0_one_cell, "res:spx:d0 (AAAA, one cell)", "disagrees", "MINOR"),
    "d0_plus5pc": (d_d0_small, "res:*:d0", "no disagreement", "ok"),
    "z0_bias_ot": (d_z0_bias, "res:otx:z0 mean (|eta| 0.8-1.4)", "disagrees", "MATERIAL"),
    "disc_flip_one_end": (d_disc_flip, "angle:class T:cot alpha (discs, one end)", "disagrees", "MATERIAL"),
    "half_clusters": (d_half_clusters, "mult:SmartPixels clusters", "disagrees", "MATERIAL"),
    "puppi_plus01": (d_puppi, "tagger:*:puppiweight", "disagrees", "MATERIAL"),
    "head_logsigma_d0": (d_head_logsigma_d0, "head:*:log sigma d0", "disagrees", "MINOR"),
    # the plan states the verdict only; hit truth has declared sources, so MATERIAL needs the gross bound
    "no_1gev_rule": (d_no_1gev_rule, "hit:*:hit truth", "disagrees", "MINOR or MATERIAL"),
}


def stage(a, run_compare, null_sides, write):
    out = {}
    ml = os.path.isdir(CM.OUT / "extract_ml" / "d121")       # the ML records exist (as engine.catalogue)
    names = [a.defect] if a.defect else [n for n in DEFECTS if n != "no_1gev_rule"
                                         and (ml or n not in ("puppi_plus01", "head_logsigma_d0"))]
    for name in names:
        fn, where, verdict, cls = DEFECTS[name]
        A, B = null_sides("oddeven")
        A = copy.deepcopy(A) if name != "no_1gev_rule" else A
        A = fn(A) or A
        res = run_compare(A, B, a, label=f"defect:{name}")
        flagged = [r for r in res if r.verdict == "disagrees" or r.cls == "MATERIAL"]
        out[name] = {"expected": {"where": where, "verdict": verdict, "class": cls},
                     "flagged": sorted({(r.quantity, r.mask, r.summary, r.cls) for r in flagged})}
        write(res, str(CM.OUT / "defects" / f"{name}.json"),
              {"defect": name, "expected": out[name]["expected"]})
    summ = CM.OUT / "defects" / "summary.json"
    if summ.exists():           # one defect at a time (--defect) adds to the summary
        old = json.load(open(summ))
        old.update(out)
        out = old
    json.dump(out, open(summ, "w"), indent=1, default=str)


# ---------------------------------------------------------------- the check of each defect -------------------------
# name: (quantities that must be flagged (fnmatch), extra condition on the flagged result, classes accepted,
#        quantities allowed to move as derived from the defect). "flagged" = disagrees or MATERIAL.
EXPECT = {
    "crossings_plus1": (("control:*:crossings",), None, {"MATERIAL"}, ()),
    "refit_off_2pc": (("refit:*:refit performed",), None, {"MATERIAL"},
                      ("refit:*", "hit:*", "res:sp*", "eff:SmartPixels*")),
    "d0_one_cell": (("res:spx:d0",), lambda r: r["mask"] == "AAAA" and list(r["cell"]) == [2, 0, 1],
                    {"MINOR", "MATERIAL"}, ()),
    "d0_plus5pc": ((), None, set(), ()),
    "z0_bias_ot": (("res:otx:z0",), lambda r: r["summary"].startswith("mean"), {"MATERIAL"}, ("res:otx:z0",)),
    "disc_flip_one_end": (("angle:class T:cot alpha",), None, {"MATERIAL"}, ()),
    "half_clusters": (("mult:SmartPixels clusters*",), None, {"MATERIAL"}, ()),
    "puppi_plus01": (("tagger:*:puppiweight",), None, {"MATERIAL"}, ()),
    "head_logsigma_d0": (("head:*:log sigma d0",), None, {"MINOR"},
                         ("head:*:coverage d0*", "tagger:track:trkjoint_d0")),
    "no_1gev_rule": (("hit:*:hit truth",), None, {"MINOR", "MATERIAL"},
                     ("mult:charged particles*", "angle:*", "res:*", "eff:*", "emu:*", "refit:*", "hit:*")),
}


def report(out_dir=None):
    """Pass/fail of each defect run against EXPECT: the expected quantity flagged with an accepted class, and
    nothing flagged outside the expected and derived quantities."""
    import fnmatch
    import glob
    out_dir = str(out_dir or CM.OUT / "defects")
    ok_all = True
    for p in sorted(glob.glob(os.path.join(out_dir, "*.json"))):
        name = os.path.basename(p)[:-5]
        if name not in EXPECT:
            continue
        must, cond, classes, derived = EXPECT[name]
        res = json.load(open(p))["results"]
        flagged = [r for r in res if r["verdict"] == "disagrees" or r["cls"] == "MATERIAL"]
        hit = [r for r in flagged if any(fnmatch.fnmatch(r["quantity"], g) for g in must)
               and (cond is None or cond(r)) and r["cls"] in classes]
        def allowed(r):
            if any(fnmatch.fnmatch(r["quantity"], g) for g in derived):
                return True
            # the target quantity: only its targeted cells (when a condition names them)
            return any(fnmatch.fnmatch(r["quantity"], g) for g in must) and (cond is None or cond(r))
        stray = [r for r in flagged if not allowed(r)]
        passed = (bool(hit) if must else not flagged) and not stray
        ok_all &= passed
        cls = sorted({r["cls"] for r in hit})
        print(f"{name:18s} {'PASS' if passed else 'FAIL'}: flagged {len(flagged)} cells; expected-quantity hits "
              f"{len(hit)} {cls}; strays {len(stray)}")
        for r in stray[:10]:
            print(f"    stray: {r['quantity']} {r['mask']} {r['cell']} {r['summary']} effect {r['effect']:.4g} "
                  f"se {r['se']:.3g} tol {r['tol']:.3g} {r['verdict']} {r['cls']}")
    print("all defects as expected" if ok_all else "NOT all defects as expected")
    return ok_all
