"""Per-track tagger inputs of the PROMPT collection computed off-line (common.COLLECTION == "prompt"): the six
track-quality head outputs, and, with a quantization scenario (common.QUANT), the track inputs that scenario would
have delivered, through the prompt parametrization (doc-archive/spix-bits-refit-study/parametrization/prompt_v2:
spixbits_apply.py with the correctionlib file, the tree maps and the head models of the detector scenario).

Used by build.py as ngtagger.data.nano.load_jets(track_override=TrackOverride(mask)). What it supplies, per track of
the constituent's track table (L1TSmartPixelsTrack):
  always                 head_clean (P(clean)) and head_d0, head_z0, head_tanl, head_rinv, head_phi (log sigma_hat): the
                         slots of the "trackheads" feature group. No CMSSW-computed heads exist on the prompt tables.
  today's tracks,        disp_lo, disp_med, disp_hi (ln of the production-radius range's low, median and high points,
  SmartPixels detector   floor 0.01 cm) and disp_p1 ... disp_p4 (P(class >= k)): the displacement heads, second version
  scenarios              (eval_refitq/spixbits/displacement.py; the "trackdisp" and "trackdispp" feature groups)
  with a quantization    rInv, tanL, z0, d0; cov_d0_d0, cov_z0_z0 (the tagger's sigma(d0), sigma(z0)); spixNAcceptedHits
  scenario               (accepted IT hits); and the heads evaluated on the emulated head inputs.
The candidate's own dxy and z0 (tagger inputs of every cell, the "baseline" group): with common.CANDIDATES ==
"reencoded" they are re-derived from the (shifted) SmartPixels track as the trigger chain derives them (measured on the
slim tier, prompt chain): z0 = 0.05 cm x round-half-towards-zero((w + 1/2) / 5) with the track word w = floor(z0 /
99.9 um) clipped to 12 bits; dxy = hwDxy / 64 cm with hwDxy a measured function of |floor(d0 x 256 / cm)| (13 bits;
the table puppi_dxy_lut_prompt.json beside the parametrization, built by `python quant_shift.py lut`). With "today"
they stay as produced (a control: the tagger then sees today's d0 through the coarse dxy code).
What it does NOT change (weigh this): everything else built from the tracks before the tagger reads them stays as
produced today: the particle-flow and PUPPI candidates' pt / eta / phi, which candidates exist, the PUPPI weights,
the jets and their constituents, the primary vertex (so vtxpv_dz moves only through the track's z0), the OT track's
own chi2 and trkMVA1. Mask IIII (no SmartPixels layers) gets no quantization shift: the OT-only heads on L1TTrack.
Head sets (common.HEADSET): "quantization" = retrained on that quantization scenario's prompt tracks (five TTbar files,
338 training events); "reference" = the "implemented now" heads trained on the same five files (the fair reference of a
"quantization" build, today's tracks only); "quantization_enlarged" / "reference_enlarged" = the same two trained on the
enlarged sample (those five files plus TTbar files 1, 2, 3 and H->tautau; 844 training events; first-pass quantization
scenarios only); "detector" = retrained per detector scenario on today's prompt tracks with the full sample (canon_v5
TTbar + H->tautau).
Truth is used as the parametrization uses it (the OT track's particle: a cell variable and the hit-choice residual);
no truth column becomes a tagger input.
"""
from __future__ import annotations

import json
import os
import sys

import awkward as ak
import numpy as np
import uproot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

from ngtagger.train import trackmva as tm  # noqa: E402

PARAM_DIR = os.environ.get("TAGGER_SP_PARAM_DIR", os.path.expanduser(
    "~/smartpixels/doc-archive/spix-bits-refit-study/parametrization/prompt_v2"))
HEAD_FIELDS = ["head_clean", "head_d0", "head_z0", "head_tanl", "head_rinv", "head_phi"]
# the displacement heads, second version (eval_refitq/spixbits/displacement.py and displacement_heads.py; today's tracks
# of a SmartPixels detector scenario): from the models in heads/displacement_v2/<detector scenario>/<window>/
DISP_FIELDS = ["disp_lo", "disp_med", "disp_hi", "disp_p1", "disp_p2", "disp_p3", "disp_p4"] + [f"disp_v1p{k}" for k in range(1, 5)] + ["disp_v1lnr"]
DISP_FLOOR_CM = 0.01                                    # the radii reach the tagger as ln(max(r, floor))
SHIFT_FIELDS = ["rInv", "tanL", "z0", "d0", "cov_d0_d0", "cov_z0_z0", "spixNAcceptedHits"]
PAR = ("rInv", "phi", "tanL", "z0", "d0")
MISSING = -999.0
_CACHE = {}


def _sha(p):
    import hashlib

    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def shipped(module):
    """A module of eval_refitq/spixbits that is shipped with the parametrization (spixbits_apply: its reference helper;
    displacement: the displacement heads' definitions): imported from this code tree (a lane job's frozen copy may
    import nothing from outside it) and required to be the very file shipped."""
    if module not in _CACHE:
        import importlib

        here = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "spixbits")
        if _sha(os.path.join(here, f"{module}.py")) != _sha(os.path.join(PARAM_DIR, f"{module}.py")):
            raise RuntimeError(f"{module}.py in the code tree differs from the one shipped in {PARAM_DIR}")
        if here not in sys.path:
            sys.path.insert(0, here)
        _CACHE[module] = importlib.import_module(module)
    return _CACHE[module]


def helper():
    return shipped("spixbits_apply")


def base_heads(mask):
    """The OT-only heads (13 inputs, one window): {head: trees} from heads/detector/SmartPixels/<mask>/all/<head>.json."""
    key = ("base", mask)
    if key not in _CACHE:
        SA, out = helper(), {}
        for h in SA.HEADS:
            m = json.load(open(os.path.join(PARAM_DIR, "heads", "detector", "SmartPixels", mask, "all", f"{h}.json")))
            assert m["splitting_convention"] == "<"
            trees = [t[0] for t in m["trees"]]
            n = max(len(t["feature"]) for t in trees)
            pad = lambda k, fill, dt: np.array([list(t[k]) + [fill] * (n - len(t[k])) for t in trees], dt)
            out[h] = {"init": float(m["init_predict"][0]), "feature": pad("feature", -2, np.int64),
                      "threshold": pad("threshold", 0.0, np.float32), "value": pad("value", 0.0, np.float64),
                      "left": pad("children_left", -1, np.int64), "right": pad("children_right", -1, np.int64),
                      "depth": int(m["max_depth"])}
        _CACHE[key] = out
    return _CACHE[key]


def refit_heads(mask, headset, quant):
    key = ("refit", mask, headset, quant)
    if key not in _CACHE:
        SA = helper()
        if headset != "detector":       # per quantization scenario; "reference...": the "implemented now" heads of the same sample
            name = quant if headset.startswith("quantization") else "implemented_now"
            folder = "per_quantization_enlarged" if headset.endswith("_enlarged") else "per_quantization"
            _CACHE[key] = SA.load_heads_npz(os.path.join(PARAM_DIR, "heads", folder, f"{name}_{mask}.npz"))
        else:
            _CACHE[key] = SA.load_heads(os.path.join(PARAM_DIR, "heads", "detector"), mask)
    return _CACHE[key]


def _xgb():
    import xgboost

    return xgboost


def disp_dir(mask, window):
    return os.path.join(PARAM_DIR, "heads", "displacement_v2", mask, window)


def disp_model(mask, window):
    """(the class model, the convention inside a class) of one window."""
    key = ("disp", mask, window)
    if key not in _CACHE:
        b = _xgb().Booster()
        b.load_model(os.path.join(disp_dir(mask, window), "classes.json"))
        shape = np.array(json.load(open(os.path.join(disp_dir(mask, window), "shape.json")))["position_quantiles_by_class"])
        _CACHE[key] = (b, shape)
    return _CACHE[key]


def disp_models_v1(mask, window):
    """The first version's two models, for the test of its tagger gain only: the five classes by fixed radius and the
    regression of ln(radius / cm) (inputs: the heads' 38 + |d0| and |d0| / sigma(d0))."""
    key = ("disp1", mask, window)
    if key not in _CACHE:
        out = []
        for kind in ("band", "radius"):
            b = _xgb().Booster()
            b.load_model(os.path.join(PARAM_DIR, "heads", "displacement", mask, "d0", window, f"{kind}.json"))
            out.append(b)
        _CACHE[key] = tuple(out)
    return _CACHE[key]


def disp_fingerprint(mask):
    """Identifies the displacement models and definitions a jet cache was built with."""
    files = [os.path.join(disp_dir(mask, w), f) for w in ("prompt", "displaced") for f in ("classes.json", "shape.json")]
    return _sha(os.path.join(PARAM_DIR, "displacement.py"))[:8] + "".join(_sha(f)[:4] for f in files)


def matrix(T, coll):
    """The heads' inputs in model order (38 for a refit track, 13 for an OT-only one) with the -999 rule, float32."""
    F = tm.features(T, coll)
    X = np.column_stack([F[n] for n in F]).astype(np.float64)
    X[~np.isfinite(X)] = MISSING
    X[X <= -998] = MISSING
    return X.astype(np.float32)


Z0_WORD_CM, Z0_CAND_CM, DXY_CAND_CM, D0_WORD_PER_CM = 2 * 20.46912 / 4096, 0.05, 1.0 / 64, 256.0     # the track word's z0 step: 99.9 um
LUT_PATH = os.path.join(PARAM_DIR, "puppi_dxy_lut_prompt.json")


def z0_code(z0_cm):
    """The candidate's hwZ0 from the track's z0: the 12-bit track word w = floor(z0 / (2 x 20.46912 cm / 4096)), then
    (w + 1/2) / 5 rounded to the nearest integer, halves towards zero (exact from the word on every candidate checked)."""
    w = np.clip(np.floor(np.asarray(z0_cm, np.float64) / Z0_WORD_CM), -2048, 2047).astype(np.int64)
    mag = np.where(w >= 0, w, -w - 1)
    return np.where(w >= 0, 1, -1) * ((mag + 2) // 5)


def dxy_code(d0_cm):
    """The candidate's hwDxy from the track's d0 through the measured table on |hwD0| (the nearest tabulated value
    beyond the table's end), hwD0 = floor(d0 x 256 / cm) clipped to 13 bits."""
    if "lut" not in _CACHE:
        d = json.load(open(LUT_PATH))
        _CACHE["lut"] = (np.array(d["abs_hwD0"], np.int64), np.array(d["hwDxy"], np.int64))
    ks, vs = _CACHE["lut"]
    k = np.clip(np.floor(np.asarray(d0_cm, np.float64) * D0_WORD_PER_CM), -4096, 4095).astype(np.int64)
    a = np.abs(k)
    i = np.clip(np.searchsorted(ks, a), 0, len(ks) - 1)
    j = np.clip(i - 1, 0, len(ks) - 1)
    return vs[np.where(np.abs(ks[j] - a) < np.abs(ks[i] - a), j, i)]


def build_lut(n_files=12):
    """hwDxy of the prompt PUPPI candidates as a function of |hwD0| of the linked SmartPixels track, measured on the
    first n_files usable slim-tier files of every detector scenario; fails if it is not a function."""
    import slimtier

    seen, n = {}, 0
    for mask in C.MASKS:
        for _, p in slimtier.frozen_files(mask)[:n_files]:
            A = uproot.open(p)["Events"].arrays([f"{C.CAND_TABLE}_hwDxy", f"{C.CAND_TABLE}_l1TrackIdx", f"{C.TRACK_TABLE}_d0"])
            li = A[f"{C.CAND_TABLE}_l1TrackIdx"]
            ok = li >= 0
            d0 = ak.to_numpy(ak.flatten(A[f"{C.TRACK_TABLE}_d0"][ak.where(ok, li, 0)][ok])).astype(np.float64)
            hw = ak.to_numpy(ak.flatten(A[f"{C.CAND_TABLE}_hwDxy"][ok])).astype(np.int64)
            k = np.abs(np.clip(np.floor(d0 * D0_WORD_PER_CM), -4096, 4095).astype(np.int64))
            n += len(k)
            for a, v in set(zip(k.tolist(), hw.tolist())):
                if seen.setdefault(a, v) != v:
                    raise RuntimeError(f"|hwD0| {a} -> hwDxy {seen[a]} and {v}: not a function")
    ks = sorted(seen)
    json.dump({"what": "prompt chain: L1PuppiCand hwDxy as a function of |floor(d0 x 256)| of the linked L1TSmartPixelsTrack; "
                       "dxy = hwDxy / 64 cm", "candidates": n, "files_per_detector_scenario": n_files,
               "abs_hwD0": ks, "hwDxy": [seen[a] for a in ks]}, open(LUT_PATH, "w"))
    print(f"{len(ks)} codes from {n} candidates, |hwD0| up to {ks[-1]} -> {LUT_PATH}")


class TrackOverride:
    cand_from_track = C.CANDIDATES == "reencoded"

    @staticmethod
    def encode_candidate(d0_cm, z0_cm):
        """(dxy, z0) of a candidate [cm] from its track's d0 and z0."""
        return dxy_code(d0_cm) * DXY_CAND_CM, z0_code(z0_cm).astype(np.float32) * np.float32(Z0_CAND_CM)   # as stored

    def __init__(self, mask, quant=None, headset=None):
        self.mask = mask
        self.base_only = mask == C.BASELINE_MASK
        self.quant = C.QUANT if quant is None else quant
        self.headset = C.HEADSET if headset is None else headset
        self.shift = bool(self.quant) and not self.base_only
        self.disp = any(g in C.groups_for(mask) for g in C.DISP_GROUPS)       # today's tracks, SmartPixels detector scenarios
        self.fields = HEAD_FIELDS + (SHIFT_FIELDS if self.shift else []) + (DISP_FIELDS if self.disp else [])

    def columns(self, files, counts, max_events=None):
        if len(files) != 1:
            raise ValueError("TrackOverride: one file per call")
        SA = helper()
        t = uproot.open(files[0])["Events"]
        keys = t.keys()
        ot, sp = C.OT_TABLE, C.TRACK_TABLE
        extra = [f"{ot}_{c}" for c in ("rInv", "phi")] + ["run", "luminosityBlock", "event"]
        if self.disp:
            D = shipped("displacement")
            hit = D.hit_table(sp)
            D.refuse_truth_columns(t, hit)
            hit_cols = [f"{hit}_{c}" for c in D.HIT_COLUMNS + ["layer", "trackIdx"]]
        if self.shift:                                   # the OT track's particle: the parametrization's cell variable
            extra += [f"{ot}_tp_{c}" for c in ("pt", "tanL", "phi0", "z0", "d0", "charge")]
        if not self.base_only:
            extra += [f"{sp}_{c}" for c in ("rInv", "phi", "spixRefitPerformed", *(f"cov_{p}_{p}" for p in PAR))]
        br = list(dict.fromkeys(tm.nano_branches(keys, C.FAMILY, "", False, base_only=self.base_only) + extra))
        A = t.arrays(br + (hit_cols if self.disp else []), entry_stop=max_events)
        T, ntrk = tm.tracks_from_batch(A, C.FAMILY, "", False, self.base_only)
        if not np.array_equal(ntrk, counts):
            raise RuntimeError(f"{files[0]}: the override's tracks are not the track table's")
        flat = lambda b: ak.to_numpy(ak.flatten(A[b])).astype(np.float64)
        sig = lambda m: 1.0 / (1.0 + np.exp(-m))
        if self.base_only:                               # OT-only heads, one window
            X = matrix(T, "base")
            H = base_heads(self.mask)
            M = np.column_stack([SA._margin(H[h], X) for h in SA.HEADS])
        else:
            X = matrix(T, "refit")
            a = np.column_stack([flat(f"{sp}_rInv"), flat(f"{sp}_phi"), T["r_tanL"], T["r_z0"], T["r_d0"]]).astype(np.float64)
            heads = refit_heads(self.mask, self.headset, self.quant)
            if not self.shift:
                M = SA.head_margins(heads, X, a[:, 4])
        out = {}
        if self.shift:
            cset, meta, maps = self.parametrization()
            ev = np.repeat(np.arange(len(ntrk)), ntrk)
            ids = tuple(ak.to_numpy(A[c]).astype(np.int64)[ev] for c in ("run", "luminosityBlock", "event")) \
                + (np.arange(len(ev)) - np.repeat(np.cumsum(ntrk) - ntrk, ntrk),)
            tp_pt = flat(f"{ot}_tp_pt")
            matched = np.nan_to_num(tp_pt, nan=-1.0) > 0
            q, tpt = flat(f"{ot}_tp_charge"), np.where(matched, tp_pt, np.nan)
            truth = np.column_stack([meta["rinv_sign_times_charge"] * q * meta["kpt_GeV_cm"] / tpt, flat(f"{ot}_tp_phi0"),
                                     flat(f"{ot}_tp_tanL"), flat(f"{ot}_tp_z0"), flat(f"{ot}_tp_d0")])
            truth[~matched] = np.nan
            # the cell variable "today's clean margin": the detector-scenario heads on today's inputs, as in the fit
            clean_today = SA.head_margins(refit_heads(self.mask, "detector", ""), X, a[:, 4])[:, 0]
            res = SA.apply(cset, meta, maps, self.quant, heads, a=a,
                           cov_diag=np.column_stack([flat(f"{sp}_cov_{p}_{p}") for p in PAR]),
                           refit=flat(f"{sp}_spixRefitPerformed") > 0, clean_margin=clean_today, inputs=X[:, 13:], ot_inputs=X[:, :13],
                           ot_a=np.column_stack([flat(f"{ot}_rInv"), flat(f"{ot}_phi"), T["b_tanL"], T["b_z0"], T["b_d0"]]),
                           ot_cov_diag=np.column_stack([T[f"b_cov_{p}_{p}"] for p in PAR]), truth=truth, ids=ids)
            M = res["margins"]
            names = meta["head_inputs_14_38"]
            out.update(rInv=res["a"][:, 0], tanL=res["a"][:, 2], z0=res["a"][:, 3], d0=res["a"][:, 4],
                       cov_d0_d0=res["cov_diag"][:, 4], cov_z0_z0=res["cov_diag"][:, 3],
                       spixNAcceptedHits=res["inputs"][:, names.index("spixNAcceptedHits")])
        if self.disp:
            d0 = np.abs(a[:, 4])
            first = np.repeat(np.cumsum(ntrk) - ntrk, ak.to_numpy(ak.num(A[f"{hit}_layer"])))
            slots, radii = D.slot_inputs(self.mask, T["r_tanL"], T["r_z0"], first + flat(f"{hit}_trackIdx").astype(np.int64),
                                         flat(f"{hit}_layer").astype(np.int64), D.hits_from_table({c: flat(f"{hit}_{c}") for c in D.HIT_COLUMNS}))
            Xd = D.assemble(X, d0, np.sqrt(np.abs(flat(f"{sp}_cov_d0_d0"))), slots)
            P, Q = np.zeros((len(X), 5)), np.zeros((len(X), len(D.QUANTILES)))       # P(class 4) stays 0 with three active letters
            P1, lnr1 = np.zeros((len(X), 5)), np.zeros(len(X))                        # the first version's classes and ln r
            for wname, w in (("prompt", d0 <= tm.D0_SPLIT), ("displaced", d0 > tm.D0_SPLIT)):
                if w.any():
                    model, shape = disp_model(self.mask, wname)
                    p = model.predict(_xgb().DMatrix(Xd[w]))
                    P[w, :p.shape[1]] = p
                    Q[w] = D.radius_range(p, radii[w], self.mask, shape)
                    band1, radius1 = disp_models_v1(self.mask, wname)
                    dm1 = _xgb().DMatrix(Xd[w][:, :X.shape[1] + 2])
                    P1[w], lnr1[w] = band1.predict(dm1), radius1.predict(dm1)
            for i, name in enumerate(("lo", "med", "hi")):
                out[f"disp_{name}"] = np.log(np.maximum(Q[:, i], DISP_FLOOR_CM))
            for k in range(4):
                out[f"disp_p{k + 1}"] = P[:, k + 1:].sum(1)
                out[f"disp_v1p{k + 1}"] = P1[:, k + 1:].sum(1)
            out["disp_v1lnr"] = lnr1
        out["head_clean"] = sig(M[:, 0])
        for i, h in enumerate(HEAD_FIELDS[1:], start=1):
            out[h] = M[:, i]
        return {k: np.asarray(v, np.float32) for k, v in out.items()}

    def parametrization(self):
        key = ("param", self.mask)
        if key not in _CACHE:
            SA = helper()
            cset, meta = SA.load(os.path.join(PARAM_DIR, f"spixbits_{self.mask}.json.gz"))
            _CACHE[key] = (cset, meta, SA.load_maps(os.path.join(PARAM_DIR, f"spixbits_{self.mask}_maps.npz")))
        cset, meta, maps = _CACHE[key]
        if self.quant not in meta["quantization_scenarios"]:
            raise KeyError(f"quantization scenario {self.quant!r} is not in the prompt parametrization of {self.mask}")
        return cset, meta, maps


if __name__ == "__main__":
    if sys.argv[1:] == ["lut"]:
        build_lut()
    else:
        raise SystemExit("usage: TAGGER_SP_COLLECTION=prompt TAGGER_SP_TIER=slim SLIMTIER_INVENTORY_DIR=... python quant_shift.py lut")
