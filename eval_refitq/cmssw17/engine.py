"""The comparison engine (plan 3.4.3-3.4.9): data loading, the quantity catalogue, per-cell effects with event
bootstrap errors, shape tests against event permutations, the frozen cell map, verdicts and classification.

A "side" is a set of extracted files (extract.py) of one sample: the CMSSW_17 production, the D121 reference, or a
half of the D121 reference (the null check). Events are indexed globally within a side; every object carries its
event index, so resampling is always by whole events.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as CM  # noqa: E402
import stats as ST  # noqa: E402

SIDE_TAG = {"d121": 1, "c17": 2}
PAR = ("rInv", "phi", "tanL", "z0", "d0")


# ================================================================ data ===============================================
class Side:
    """Concatenated records of one sample (per mask), with global event indices and event uids."""

    def __init__(self, label, side, files, masks=CM.MASKS, base=None):
        self.label, self.side, self.files = label, side, tuple(files)
        base = str(base or CM.OUT / "extract")
        self.m = {}
        for mask in masks:
            parts = []
            for n in self.files:
                p = os.path.join(base, side, f"{mask}_f{n:02d}.npz")
                if not os.path.exists(p):
                    raise FileNotFoundError(p)
                parts.append(dict(np.load(p, allow_pickle=True)))
            self.m[mask] = self._concat(parts)

    @staticmethod
    def _concat(parts):
        out = {}
        evoff = 0
        nev_list = [len(p["ev_run"]) for p in parts]
        keys = set().union(*[set(p) for p in parts])
        for k in sorted(keys):
            if k.startswith("meta_"):
                out[k] = [p.get(k) for p in parts]
                continue
            arrs = []
            evoff = 0
            for p, ne in zip(parts, nev_list):
                if k in p:
                    a = p[k]
                    if k.endswith("_ev"):
                        a = a.astype(np.int64) + evoff
                    elif k.endswith("_trk") or k.endswith("_og"):
                        a = a.astype(np.int64)          # re-based below
                    arrs.append(a)
                evoff += ne
            out[k] = np.concatenate(arrs) if arrs else np.zeros(0)
        # global track indices of refit hits / OT links: rebase per file
        for key in ("sp", "spx"):
            if f"hit_{key}_trk" in out:
                toff, hoff = 0, 0
                tr = []
                for p in parts:
                    nt = len(p.get(f"trk_{key}_pt", []))
                    h = p.get(f"hit_{key}_trk")
                    if h is not None:
                        tr.append(h.astype(np.int64) + toff)
                    toff += nt
                out[f"hit_{key}_trk"] = np.concatenate(tr) if tr else np.zeros(0, np.int64)
        out["ev_file"] = np.concatenate([np.full(ne, int(p["meta_file"])) for p, ne in zip(parts, nev_list)])
        out["ev_uid"] = ST.event_uid(SIDE_TAG.get(parts[0]["meta_side"].item() if hasattr(parts[0]["meta_side"], "item")
                                                  else str(parts[0]["meta_side"]), 3),
                                     out["ev_file"], out["ev_run"], out["ev_lumi"], out["ev_event"])
        out["nev"] = len(out["ev_run"])
        return out

    def subset_events(self, keep_files=None, keep_fn=None, label=None):
        """A new Side holding only the events of the given file numbers, or those keep_fn(d) selects (the null
        splits)."""
        s = Side.__new__(Side)
        s.label = label or f"{self.label}[{','.join(map(str, keep_files))}]"
        s.side, s.files = self.side, tuple(keep_files or ())
        s.m = {}
        for mask, d in self.m.items():
            keep_ev = np.isin(d["ev_file"], keep_files) if keep_fn is None else keep_fn(d)
            newidx = np.cumsum(keep_ev) - 1
            out = {"nev": int(keep_ev.sum())}
            # per-object tables: keep the objects of kept events, renumber events; tracks keep their own indexing
            for k, a in d.items():
                if k == "nev" or k.startswith("meta_"):
                    continue
                if k.startswith("ev_"):
                    out[k] = a[keep_ev]
            for pre in sorted({k.rsplit("_", 1)[0] for k in d if k.endswith("_ev") and not k.startswith("ev_")}):
                ev = d[pre + "_ev"]
                sel = keep_ev[ev]
                for k, a in d.items():
                    if k.startswith(pre + "_") and not k.startswith("ev_") and hasattr(a, "__len__") and len(a) == len(ev):
                        out[k] = a[sel]
                out[pre + "_ev"] = newidx[ev[sel]]
                if pre.startswith("trk_"):
                    out[pre + "_oldidx"] = np.flatnonzero(sel)
            # refit hits point into the track tables: remap
            for key in ("sp", "spx"):
                if f"trk_{key}_oldidx" in out and f"hit_{key}_trk" in out:
                    m = np.full(len(d[f"trk_{key}_ev"]), -1, np.int64)
                    m[out[f"trk_{key}_oldidx"]] = np.arange(len(out[f"trk_{key}_oldidx"]))
                    out[f"hit_{key}_trk"] = m[out[f"hit_{key}_trk"]]
            s.m[mask] = out
        return s


# ================================================================ quantities =========================================
@dataclass
class Q:
    """One quantity: (variable, summary) over objects, cut into cells.
    kind: mean | ratio_sums | frac | resid | quant | shape_only.
    get(d) -> dict(x=values, ev=event index, dims={name: int array}, w=optional object weights,
                   x2=second values for ratio_sums, cats=categories for frac)."""
    name: str
    group: str
    tclass: str
    kind: str
    masks: tuple
    get: object
    dims: tuple = ()
    merge: tuple = ()
    quantiles: tuple = (0.5,)
    shape: bool = False
    reweight: tuple = ()          # (pt, eta) value keys for the in-cell kinematic reweighting of side A
    note: str = ""
    cats: tuple = None            # frac: categories (frozen from D121: >= 5 % share, the rest "other")


def _bin(x, edges):
    return CM.bin_index(x, np.asarray(edges, np.float64))


def _abs_bin(x, edges):
    return _bin(np.abs(x), edges)


def catalogue():
    """The quantity list (plan 3.4.3). Report-only baselines are in group "baseline"."""
    Qs = []
    sp = CM.SP_MASKS
    one = ("IIII",)
    # ---- baselines (reported only) ----
    for k, nm in (("ev_Pileup_nPU", "pileup nPU"), ("ev_Pileup_nTrueInt", "pileup nTrueInt")):
        Qs.append(Q(f"baseline:{nm}", "baseline", "multiplicity", "mean", one,
                    lambda d, k=k: dict(x=d[k], ev=np.arange(d["nev"])), shape=True))
    for k in ("x", "y", "z", "t0"):
        Qs.append(Q(f"baseline:GenVtx {k} RMS", "baseline", "full_rms", "resid", one,
                    lambda d, k=k: dict(x=d[f"ev_GenVtx_{k}"] - np.nanmean(d[f"ev_GenVtx_{k}"]),
                                        ev=np.arange(d["nev"]))))
    Qs.append(Q("baseline:top pT", "baseline", "core_rms", "quant", one,
                lambda d: dict(x=d["gtop_pt"], ev=d["gtop_ev"]), quantiles=(0.5,), shape=True))
    Qs.append(Q("baseline:GenJet pT", "baseline", "core_rms", "quant", one,
                lambda d: dict(x=d["gjet_pt"], ev=d["gjet_ev"],
                               dims=dict(flav=d["gjet_flav"].astype(np.int64), eta=_abs_bin(d["gjet_eta"], CM.JET_ETA))),
                dims=("flav", "eta"), merge=("eta",), quantiles=(0.5,), shape=True))
    for f, nm in enumerate(CM.JET_FLAVOURS):
        Qs.append(Q(f"baseline:GenJets above 30 GeV, {nm}", "baseline", "multiplicity", "mean", one,
                    lambda d, f=f: dict(x=np.bincount(d["gjet_ev"][(d["gjet_flav"] == f) & (d["gjet_pt"] > 30)],
                                                      minlength=d["nev"]), ev=np.arange(d["nev"]))))
    # ---- multiplicities ----
    ev = lambda d: np.arange(d["nev"])  # noqa: E731
    for k, nm in (("ev_n_ot", "OT tracks (prompt)"), ("ev_n_otx", "OT tracks (extended)")):
        Qs.append(Q(f"mult:{nm}", "mult_tracker", "multiplicity", "mean", one,
                    lambda d, k=k: dict(x=d[k], ev=ev(d)), shape=True))
        Qs.append(Q(f"mult:{nm} per pileup interaction", "mult_tracker", "multiplicity", "ratio_sums", one,
                    lambda d, k=k: dict(x=d[k], x2=d["ev_Pileup_nPU"], ev=ev(d))))
    for L in range(1, 7):
        Qs.append(Q(f"mult:OT stubs barrel layer {L}", "mult_tracker", "multiplicity", "mean", one,
                    lambda d, L=L: dict(x=d[f"ev_nstub_b{L}"], ev=ev(d))))
    for D in range(1, 6):
        Qs.append(Q(f"mult:OT stubs disc {D}", "mult_tracker", "multiplicity", "mean", one,
                    lambda d, D=D: dict(x=d[f"ev_nstub_d{D}"], ev=ev(d))))
    for S in CM.SURFACES:
        Qs.append(Q(f"mult:SmartPixels clusters surface {S}", "mult_tracker", "multiplicity", "mean", (CM.CLUSTER_MASK,),
                    lambda d, S=S: dict(x=d[f"ev_ncl_{S}"], ev=ev(d))))
        Qs.append(Q(f"emu:class-T fraction surface {S}", "emu_class", "frac", "ratio_sums", (CM.CLUSTER_MASK,),
                    lambda d, S=S: dict(x=d[f"ev_ncl_{S}_T"], x2=d[f"ev_ncl_{S}_T"] + d[f"ev_ncl_{S}_C"], ev=ev(d)),
                    note="reported against declared difference 10"))
    for key, nm in (("sp", "prompt"), ("spx", "extended")):
        Qs.append(Q(f"mult:refit tracks ({nm})", "mult_tracker", "multiplicity", "mean", sp,
                    lambda d, key=key: dict(x=d[f"ev_n_{key}_refit"], ev=ev(d))))
    for tab in ("L1PFCand", "L1PuppiCand", "L1ExtPuppiCand"):
        for ch in ("ch", "neu"):
            Qs.append(Q(f"mult:{tab} {'charged' if ch == 'ch' else 'neutral'}", "mult_pf_jets", "multiplicity", "mean",
                        CM.MASKS, lambda d, tab=tab, ch=ch: dict(x=d[f"ev_n_{tab}_{ch}"], ev=ev(d)), shape=True))
    for tab in ("L1puppiJetSC4NG", "L1puppiJetSC4", "L1puppiJetSC8"):
        Qs.append(Q(f"mult:{tab} above 30 GeV", "mult_pf_jets", "multiplicity", "mean", CM.MASKS,
                    lambda d, tab=tab: dict(x=d[f"ev_n_{tab}_30"], ev=ev(d))))
    for k, nm in (("ev_nPart_hs", "hard scatter"), ("ev_nPart_pu", "pileup")):
        Qs.append(Q(f"mult:charged particles above 1 GeV, {nm}", "mult_particles", "multiplicity", "mean", one,
                    lambda d, k=k: dict(x=d[k], ev=ev(d))))
    # ---- efficiencies ----
    Qs.append(Q("eff:OT track efficiency", "eff_ot", "efficiency", "mean", one,
                lambda d: dict(x=d["eff_ot_ok"].astype(np.float64), ev=d["eff_ot_ev"],
                               dims=dict(pt=_bin(d["eff_ot_pt"], CM.TRK_PT), eta=_abs_bin(d["eff_ot_eta"], CM.TRK_ETA))),
                dims=("pt", "eta"), merge=("eta", "pt")))
    Qs.append(Q("eff:SmartPixels accepted-hit efficiency", "eff_sp", "efficiency", "mean", sp,
                lambda d: dict(x=d["eff_sp_val"].astype(np.float64), ev=d["eff_sp_ev"],
                               dims=dict(pt=_bin(d["eff_sp_pt"], CM.TRK_PT), eta=_abs_bin(d["eff_sp_eta"], CM.TRK_ETA))),
                dims=("pt", "eta"), merge=("eta", "pt")))
    # ---- residuals to truth ----
    for key, masks, grp in (("ot", one, "res_ot"), ("otx", one, "res_ot"), ("sp", sp, "res_sp"), ("spx", sp, "res_sp")):
        for p in PAR:
            def get(d, key=key, p=p):
                m = np.isfinite(d[f"trk_{key}_res_{p}"])
                if key in ("sp", "spx"):
                    m &= d[f"trk_{key}_spix_RefitPerformed"] > 0
                return dict(x=d[f"trk_{key}_res_{p}"][m].astype(np.float64), ev=d[f"trk_{key}_ev"][m],
                            dims=dict(pt=_bin(d[f"trk_{key}_tp_pt"][m], CM.TRK_PT),
                                      eta=_abs_bin(d[f"trk_{key}_tp_eta"][m], CM.TRK_ETA),
                                      hs=d[f"trk_{key}_tp_hs"][m].astype(np.int64)),
                            rw_pt=d[f"trk_{key}_tp_pt"][m], rw_eta=np.abs(d[f"trk_{key}_tp_eta"][m]))
            Qs.append(Q(f"res:{key}:{p}", grp, "resid", "resid", masks, get, dims=("pt", "eta", "hs"),
                        merge=("eta", "pt", "hs"), shape=True, reweight=("rw_pt", "rw_eta")))
    # ---- refit quality (per SmartPixels track) ----
    for key in ("sp", "spx"):
        def trk_dims(d, key, m):
            return dict(pt=_bin(d[f"trk_{key}_ot_pt"][m], CM.TRK_PT), eta=_abs_bin(d[f"trk_{key}_ot_eta"][m], CM.TRK_ETA))
        Qs.append(Q(f"refit:{key}:refit performed", "refit_quality", "frac", "frac", sp,
                    lambda d, key=key: (lambda m: dict(cat=(d[f"trk_{key}_spix_RefitPerformed"][m] > 0).astype(np.int64),
                                                        ev=d[f"trk_{key}_ev"][m], dims=trk_dims(d, key, m)))(
                        np.isfinite(d[f"trk_{key}_ot_pt"])), dims=("pt", "eta"), merge=("eta", "pt"), cats=(1,)))
        for col, nm in (("NAcceptedHits", "accepted hits"), ("NKFUpdates", "KF updates"), ("Status", "status"),
                        ("LayerHitMask", "layer hit mask"), ("AnyWindowTruncated", "window truncated")):
            Qs.append(Q(f"refit:{key}:{nm}", "refit_quality", "frac", "frac", sp,
                        lambda d, key=key, col=col: (lambda m: dict(cat=d[f"trk_{key}_spix_{col}"][m].astype(np.int64),
                                                                     ev=d[f"trk_{key}_ev"][m], dims=trk_dims(d, key, m)))(
                            d[f"trk_{key}_spix_RefitPerformed"] > 0), dims=("pt", "eta"), merge=("eta", "pt")))
        for col in ("Chi2IncXTot", "Chi2IncYTot", "Chi2IncAlphaTot", "Chi2IncBetaTot"):
            Qs.append(Q(f"refit:{key}:{col}", "refit_quality", "chi2", "quant", sp,
                        lambda d, key=key, col=col: (lambda m: dict(x=d[f"trk_{key}_spix_{col}"][m].astype(np.float64),
                                                                     ev=d[f"trk_{key}_ev"][m], dims=trk_dims(d, key, m)))(
                            (d[f"trk_{key}_spix_RefitPerformed"] > 0) & (d[f"trk_{key}_spix_{col}"] > -900)),
                        dims=("pt", "eta"), merge=("eta", "pt"), quantiles=(0.5, 0.9), shape=True))
        # ---- refit hits ----
        def hit_dims(d, key, m):
            surf = d[f"hit_{key}_layer"][m].astype(np.int64)
            end = np.where(surf > 10, d[f"hit_{key}_side"][m].astype(np.int64), 0)
            return dict(surface=surf * 10 + end, pt=_bin(d[f"hit_{key}_ot_pt"][m], CM.HIT_PT))
        for col in ("chi2IncX", "chi2IncY", "chi2IncAlpha", "chi2IncBeta"):
            Qs.append(Q(f"hit:{key}:{col}", "refit_quality", "chi2", "quant", sp,
                        lambda d, key=key, col=col: (lambda m: dict(x=d[f"hit_{key}_{col}"][m].astype(np.float64),
                                                                     ev=d[f"hit_{key}_ev"][m], dims=hit_dims(d, key, m)))(
                            (d[f"hit_{key}_hitAccepted"] > 0) & (d[f"hit_{key}_{col}"] > -900)),
                        dims=("surface", "pt"), merge=("pt",), quantiles=(0.5, 0.9), shape=True))
        for col in ("pullX", "pullY", "pullAlpha", "pullBeta"):
            Qs.append(Q(f"hit:{key}:{col}", "refit_quality", "resid", "resid", sp,
                        lambda d, key=key, col=col: (lambda m: dict(x=d[f"hit_{key}_{col}"][m].astype(np.float64),
                                                                     ev=d[f"hit_{key}_ev"][m], dims=hit_dims(d, key, m)))(
                            (d[f"hit_{key}_hitAccepted"] > 0) & (d[f"hit_{key}_{col}"] > -900)),
                        dims=("surface", "pt"), merge=("pt",), shape=True))
        Qs.append(Q(f"hit:{key}:two-layer choice", "refit_quality", "frac", "frac", sp,
                    lambda d, key=key: (lambda m: dict(cat=d[f"hit_{key}_kchoice"][m].astype(np.int64),
                                                        ev=d[f"hit_{key}_ev"][m], dims=hit_dims(d, key, m)))(
                        d[f"hit_{key}_kchoice"] >= 0), dims=("surface", "pt"), merge=("pt",)))
        Qs.append(Q(f"hit:{key}:hit truth", "hit_truth", "frac", "frac", sp,
                    lambda d, key=key: (lambda m: dict(cat=d[f"hit_{key}_truth"][m].astype(np.int64),
                                                        ev=d[f"hit_{key}_ev"][m], dims=hit_dims(d, key, m)))(
                        d[f"hit_{key}_truth"] >= 0), dims=("surface", "pt"), merge=("pt",), cats=(0, 1, 2)))
    # ---- controls ----
    for key in ("sp", "spx"):
        def ctl_dims(d, key, m):
            z = d[f"trk_{key}_ot_z0dir"][m]
            zc = np.where(z < -CM.CONTROL_Z0_CUT, 0, np.where(z > CM.CONTROL_Z0_CUT, 2, 1))
            return dict(eta=np.floor(np.abs(d[f"trk_{key}_ot_eta"][m]) / CM.CONTROL_ETA_STEP).astype(np.int64), z0=zc)
        Qs.append(Q(f"control:{key}:crossings", "control", "control_crossings", "mean", sp,
                    lambda d, key=key: (lambda m: dict(x=d[f"trk_{key}_spix_NCrossings"][m].astype(np.float64),
                                                        ev=d[f"trk_{key}_ev"][m], dims=ctl_dims(d, key, m)))(
                        (d[f"trk_{key}_spix_RefitPerformed"] > 0) & np.isfinite(d[f"trk_{key}_ot_eta"])),
                    dims=("eta", "z0"), merge=()))
        Qs.append(Q(f"control:{key}:disc-going", "control", "control_discgoing", "mean", sp,
                    lambda d, key=key: (lambda m: dict(x=d[f"trk_{key}_spix_DiscGoing"][m].astype(np.float64),
                                                        ev=d[f"trk_{key}_ev"][m], dims=ctl_dims(d, key, m)))(
                        (d[f"trk_{key}_spix_RefitPerformed"] > 0) & np.isfinite(d[f"trk_{key}_ot_eta"])
                        & (np.abs(d[f"trk_{key}_ot_eta"]) >= 1.0)),
                    dims=("eta", "z0"), merge=()))
    # ---- angles (the cluster table, read once from AAAA) ----
    inc = (0.0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, np.inf)
    for ang in ("Alpha", "Beta"):
        for cls, grp, tc in ((1, "control", "control_classT"), (0, "angle_classC", "resid")):
            def get(d, ang=ang, cls=cls):
                m = (d["cl_tpEmuClass"] == cls) & (d[f"cl_has{ang}"] > 0)
                surf = d["cl_surface"][m].astype(np.int64)
                end = np.where(surf > 10, d["cl_side"][m].astype(np.int64), 0)
                return dict(x=(d[f"cl_localCot{ang}"][m] - d[f"cl_tpLocalCot{ang}"][m]).astype(np.float64),
                            ev=d["cl_ev"][m], dims=dict(surface=surf * 10 + end,
                                                        inc=_abs_bin(d[f"cl_tpLocalCot{ang}"][m], inc)))
            Qs.append(Q(f"angle:class {'T' if cls else 'C'}:cot {ang.lower()}", grp, tc, "resid", (CM.CLUSTER_MASK,),
                        get, dims=("surface", "inc"), merge=("inc",), shape=(cls == 0)))
    # ---- jets ----
    def jet_dims(d, m):
        return dict(pt=_bin(d["jet_gen_pt"][m], CM.JET_PT), eta=_abs_bin(d["jet_gen_eta"][m], CM.JET_ETA),
                    flav=d["jet_flav"][m].astype(np.int64))
    Qs.append(Q("jet:L1 jet response", "jet_response", "core_rms", "quant", CM.MASKS,
                lambda d: (lambda m: dict(x=(d["jet_pt"][m] / d["jet_gen_pt"][m]).astype(np.float64), ev=d["jet_ev"][m],
                                          dims=jet_dims(d, m)))(np.isfinite(d["jet_gen_pt"]) & (d["jet_flav"] >= 0)),
                dims=("pt", "eta", "flav"), merge=("eta", "pt", "flav"), quantiles=(0.16, 0.5, 0.84)))
    for s in ("bTagScore", "cTagScore", "udsTagScore", "gTagScore", "tau_pTagScore", "tau_nTagScore"):
        Qs.append(Q(f"jet:NG {s}", "ng_scores", "head_pclean", "quant", CM.MASKS,
                    lambda d, s=s: (lambda m: dict(x=d[f"jet_{s}"][m].astype(np.float64), ev=d["jet_ev"][m],
                                                   dims=jet_dims(d, m)))(np.isfinite(d["jet_gen_pt"]) & (d["jet_flav"] >= 0)),
                    dims=("pt", "eta", "flav"), merge=("eta", "pt", "flav"), quantiles=(0.1, 0.5, 0.9), shape=True))
    return Qs


def source_group(group, summary):
    """The declared-source group of a (quantity group, summary) pair (plan 3.4.9)."""
    if group in ("res_ot", "res_sp"):
        return group + ("_mean" if summary == "mean" else "_width")
    return group


def get_objects(q: Q, d):
    """q.get(d) without the objects that fall outside the binning (a dim index of -1)."""
    g = q.get(d)
    if not q.dims:
        return g
    keep = np.ones(len(g["ev"]), bool)
    for nm in q.dims:
        keep &= np.asarray(g["dims"][nm]) >= 0
    if keep.all():
        return g
    out = {}
    for k, v in g.items():
        if k == "dims":
            out["dims"] = {n: np.asarray(a)[keep] for n, a in v.items()}
        elif isinstance(v, np.ndarray) and len(v) == len(keep):
            out[k] = v[keep]
        else:
            out[k] = v
    return out


# ================================================================ per-cell evaluation ================================
def cell_keys(dims, names, level_merge):
    """Cell key per object: the dims' bin indices, with the merged dims set to -9."""
    cols = []
    for nm in names:
        v = np.asarray(dims[nm], np.int64)
        cols.append(np.full(len(v), -9, np.int64) if nm in level_merge else v)
    if not cols:
        return None
    return np.stack(cols, 1)


def summarise(q: Q, sub, w, cat=None):
    """The summaries of one cell for weights w (objects). Returns {summary name: value}."""
    out = {}
    if q.kind == "mean":
        out["mean"] = ST.wmean(sub["x"], w)
    elif q.kind == "ratio_sums":
        s2 = (w * sub["x2"]).sum()
        out["ratio"] = float((w * sub["x"]).sum() / s2) if s2 > 0 else np.nan
    elif q.kind == "frac":
        sw = w.sum()
        for c in q.cats:
            if c == "other":
                m = ~np.isin(sub["cat"], [k for k in q.cats if k != "other"])
            else:
                m = sub["cat"] == c
            out[f"frac[{c}]"] = float(w[m].sum() / sw) if sw > 0 else np.nan
    elif q.kind == "resid":
        r = sub["resid"].summaries(w[sub["finite"]])
        out.update({"mean": r["mean"], "core": r["core"], "full": r["full"], "outside": r["outside"]})
    elif q.kind == "quant":
        o = sub["order"]
        xs = sub["x"][o]
        ws = w[o]
        for qq in q.quantiles:
            out[f"q{int(round(qq * 100))}"] = ST.wquantile_sorted(xs, ws, qq)
    return out


def tol_of(q: Q, summ: str, ref_value: float):
    """(effect kind, tolerance, gross bound) of a summary (plan 3.4.6)."""
    if summ.startswith("frac[") or q.tclass == "frac":
        t, g = CM.frac_tolerance(ref_value if np.isfinite(ref_value) else 0.5)
        return "diff", t, (None if q.group == "control" else g)
    if q.kind == "resid":
        if q.tclass == "control_classT":
            if summ == "core":
                return CM.TOL["control_classT_core"]
            if summ == "mean":
                return CM.TOL["control_classT_mean"]
            return None
        if summ == "core":
            return CM.TOL["core_rms"]
        if summ == "full":
            return CM.TOL["full_rms"]
        if summ == "mean":
            return CM.TOL["mean"]
        if summ == "outside":
            t, g = CM.frac_tolerance(ref_value if np.isfinite(ref_value) else 0.5)
            return "diff", t, g
    if q.kind == "quant" and q.name.startswith("jet:L1 jet response"):
        return CM.TOL["core_rms"]
    return CM.TOL[q.tclass]


def effect(kind, a, b, ref_core=None):
    if not (np.isfinite(a) and np.isfinite(b)):
        return np.nan
    if kind == "logratio":
        return math.log(a / b) if a > 0 and b > 0 else (0.0 if a == b else np.nan)
    if kind == "diff":
        return a - b
    if kind == "diff_corerms":
        return (a - b) / ref_core if ref_core and ref_core > 0 else np.nan
    if kind == "distance":
        return a
    raise ValueError(kind)


@dataclass
class CellResult:
    quantity: str
    group: str
    mask: str
    cell: tuple
    summary: str
    a: float
    b: float
    effect: float
    se: float
    tol: float
    gross: float
    kind: str
    nev_a: int
    nev_b: int
    n_a: int
    n_b: int
    judged: bool
    verdict: str = ""
    cls: str = ""
    p_tol: float = 1.0
    p_gross: float = 1.0
    extra: dict = field(default_factory=dict)


def prepare(q: Q, g, cellmask):
    """The per-cell object subset with the precomputations its summaries need."""
    sub = {k: (v[cellmask] if isinstance(v, np.ndarray) and len(v) == len(cellmask) else v) for k, v in g.items()
           if k != "dims"}
    if q.kind == "resid":
        fin = np.isfinite(sub["x"])
        sub["finite"] = fin
        sub["resid"] = ST.Resid(sub["x"][fin])
    elif q.kind == "quant":
        sub["order"] = np.argsort(sub["x"], kind="stable")
    return sub


def reweight(q: Q, sa, sb, nsub=4):
    """Within a cell, weights for side A objects so that its (pT, |eta|) distribution matches side B (the reference)."""
    if not q.reweight:
        return np.ones(len(sa["ev"]))
    kp, ke = q.reweight
    pa, ea, pb, eb = sa[kp], sa[ke], sb[kp], sb[ke]
    if len(pa) == 0 or len(pb) == 0:
        return np.ones(len(pa))
    lo_p, hi_p = np.nanmin(np.r_[pa, pb]), np.nanmax(np.r_[pa, pb])
    lo_e, hi_e = np.nanmin(np.r_[ea, eb]), np.nanmax(np.r_[ea, eb])
    ep = np.geomspace(max(lo_p, 1e-3), hi_p * (1 + 1e-9), nsub + 1)
    ee = np.linspace(lo_e, hi_e * (1 + 1e-9) + 1e-9, nsub + 1)
    ia = np.clip(np.searchsorted(ep, pa, "right") - 1, 0, nsub - 1) * nsub + np.clip(np.searchsorted(ee, ea, "right") - 1, 0, nsub - 1)
    ib = np.clip(np.searchsorted(ep, pb, "right") - 1, 0, nsub - 1) * nsub + np.clip(np.searchsorted(ee, eb, "right") - 1, 0, nsub - 1)
    ha = np.bincount(ia, minlength=nsub * nsub).astype(float)
    hb = np.bincount(ib, minlength=nsub * nsub).astype(float)
    ratio = np.where(ha > 0, (hb / max(hb.sum(), 1)) / np.maximum(ha / max(ha.sum(), 1), 1e-12), 0.0)
    return ratio[ia]


def boot(q, sub, uid, w0, n_boot):
    """Summaries for the nominal sample (index 0) and n_boot Poisson replicas."""
    res = [summarise(q, sub, w0)]
    for r in range(n_boot):
        w = ST.poisson_weights(uid, r)[sub["ev"]] * w0
        res.append(summarise(q, sub, w))
    return res


# ================================================================ freeze (D121 alone) ================================
def primary_summary(q: Q, cats=None):
    if q.kind == "resid":
        return "core"
    if q.kind == "mean":
        return "mean"
    if q.kind == "ratio_sums":
        return "ratio"
    if q.kind == "frac":
        return f"frac[{(cats or q.cats)[0]}]"
    if q.kind == "quant":
        return f"q{int(round(q.quantiles[len(q.quantiles) // 2] * 100))}"
    raise ValueError(q.kind)


def fine_keys(q: Q, g):
    if not q.dims:
        return np.zeros((len(g["ev"]), 0), np.int64)
    return np.stack([np.asarray(g["dims"][nm], np.int64) for nm in q.dims], 1)


def frac_categories(q: Q, gB):
    if q.cats is not None:
        return tuple(q.cats)
    c, n = np.unique(gB["cat"], return_counts=True)
    share = n / n.sum() if n.sum() else n
    big = tuple(int(x) for x, s in zip(c, share) if s >= 0.05)
    return big + (("other",) if share[[x not in big for x in c]].sum() > 0 else ())


def _summ_names(q, cats):
    if q.kind == "resid":
        return ("mean", "core", "full", "outside") if q.group != "control" else ("mean", "core")
    if q.kind == "mean":
        return ("mean",)
    if q.kind == "ratio_sums":
        return ("ratio",)
    if q.kind == "frac":
        return tuple(f"frac[{c}]" for c in cats)
    return tuple(f"q{int(round(x * 100))}" for x in q.quantiles)


def _effect_kind(q, summ, refv):
    t = tol_of(q, summ, refv)
    return None if t is None else t


def freeze_quantity(q: Q, sideB: Side, mask, n_boot=100):
    """The cell map of a quantity and mask from the D121 reference alone (plan 3.4.4): a cell is judged if its
    predicted error (sqrt(2) x its D121 bootstrap error at the reference's size) is <= tolerance / 2.93 for the
    primary summary, with >= 50 contributing events; otherwise merged in the frozen order, else report-only."""
    d = sideB.m[mask]
    g = get_objects(q, d)
    if q.kind == "frac":
        cats = frac_categories(q, g)
        q = Q(**{**q.__dict__, "cats": cats})
    else:
        cats = None
    fk = fine_keys(q, g)
    uniq, inv = (np.unique(fk, axis=0, return_inverse=True) if fk.shape[1] else (np.zeros((1, 0), np.int64),
                                                                                  np.zeros(len(fk), np.int64)))
    inv = np.asarray(inv).reshape(-1)
    assigned = {}                                   # fine key index -> judged cell id
    cells = []                                      # judged cells: dict(level, members, ref, pred, summaries)
    uid = d["ev_uid"]
    levels = [q.merge[:L] for L in range(len(q.merge) + 1)]
    for lev in levels:
        free = [i for i in range(len(uniq)) if i not in assigned]
        if not free:
            break
        cols = [j for j, nm in enumerate(q.dims) if nm not in lev]
        groups = {}
        for i in free:
            groups.setdefault(tuple(uniq[i, cols]) if cols else (), []).append(i)
        for gk, members in groups.items():
            m = np.isin(inv, members)
            if not m.any():
                continue
            nev = len(np.unique(g["ev"][m]))
            if nev < CM.MIN_EVENTS_PER_CELL:
                continue
            sub = prepare(q, g, m)
            res = boot(q, sub, uid, np.ones(m.sum()), n_boot)
            ok_any = False
            summ = {}
            for sname in _summ_names(q, cats):
                ref = res[0].get(sname, np.nan)
                t = tol_of(q, sname, ref)
                if t is None:
                    continue
                kind, tol, gross = t
                if kind == "diff_corerms":
                    vals = np.array([r[sname] / r["core"] if r.get("core") else np.nan for r in res[1:]])
                elif kind == "logratio":
                    vals = np.log(np.array([r[sname] for r in res[1:]], np.float64))
                else:
                    vals = np.array([r[sname] for r in res[1:]], np.float64)
                se = float(np.nanstd(vals, ddof=1)) if np.isfinite(vals).sum() > 2 else np.nan
                pred = math.sqrt(2) * se
                judged = bool(np.isfinite(pred) and pred <= tol / CM.POWER_DIVISOR)
                if sname == primary_summary(q, cats):
                    ok_any = judged
                summ[sname] = dict(ref=float(ref) if np.isfinite(ref) else None, se_ref=se, pred=pred, tol=tol,
                                   gross=gross, kind=kind, judged=judged)
            if not ok_any:
                continue
            cid = len(cells)
            cells.append(dict(level=list(lev), key=[int(x) for x in gk], dims=[q.dims[j] for j in cols],
                              members=[[int(v) for v in uniq[i]] for i in members], nev_ref=int(nev),
                              n_ref=int(m.sum()), summaries=summ))
            for i in members:
                assigned[i] = cid
    report_only = [[int(v) for v in uniq[i]] for i in range(len(uniq)) if i not in assigned]
    return dict(quantity=q.name, group=q.group, mask=mask, dims=list(q.dims), merge=list(q.merge),
                cats=list(cats) if cats else None, cells=cells, report_only=report_only)


# ================================================================ compare ============================================
def _cell_index(q, g, fz):
    """Per object: the judged cell id from the frozen members (-1: report-only or unseen)."""
    fk = fine_keys(q, g)
    lookup = {}
    for cid, c in enumerate(fz["cells"]):
        for mem in c["members"]:
            lookup[tuple(mem)] = cid
    if fk.shape[1] == 0:
        return np.zeros(len(fk), np.int64) if fz["cells"] else np.full(len(fk), -1, np.int64)
    uniq, inv = np.unique(fk, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    cid = np.array([lookup.get(tuple(u), -1) for u in uniq], np.int64)
    return cid[inv]


def compare_quantity(q: Q, sideA: Side, sideB: Side, mask, fz, n_boot=CM.N_BOOT, n_perm=CM.N_PERM, seed=1,
                     defect=None):
    """Per judged cell: effects with event-bootstrap errors (plan 3.4.5); shapes with event permutations."""
    dA, dB = sideA.m[mask], sideB.m[mask]
    if defect is not None:
        dA = defect(q, mask, dA)
    qq = Q(**{**q.__dict__, "cats": tuple(fz["cats"]) if fz.get("cats") else q.cats})
    gA, gB = get_objects(qq, dA), get_objects(qq, dB)
    cA, cB = _cell_index(qq, gA, fz), _cell_index(qq, gB, fz)
    out = []
    for cid, cell in enumerate(fz["cells"]):
        mA, mB = cA == cid, cB == cid
        nevA = len(np.unique(gA["ev"][mA])) if mA.any() else 0
        nevB = len(np.unique(gB["ev"][mB])) if mB.any() else 0
        sa, sb = prepare(qq, gA, mA), prepare(qq, gB, mB)
        rw = reweight(qq, sa, sb)
        resA = boot(qq, sa, dA["ev_uid"], rw, n_boot)
        resB = boot(qq, sb, dB["ev_uid"], np.ones(mB.sum()), n_boot)
        for sname, fs in cell["summaries"].items():
            kind, tol, gross = fs["kind"], fs["tol"], fs["gross"]
            a, b = resA[0].get(sname, np.nan), resB[0].get(sname, np.nan)
            refc = resB[0].get("core") if kind == "diff_corerms" else None
            eff = effect(kind, a, b, refc)
            reps = [effect(kind, ra.get(sname, np.nan), rb.get(sname, np.nan),
                           rb.get("core") if kind == "diff_corerms" else None) for ra, rb in zip(resA[1:], resB[1:])]
            reps = np.array(reps, np.float64)
            se = float(np.nanstd(reps, ddof=1)) if np.isfinite(reps).sum() > 2 else np.nan
            se *= fz.get("se_inflation", {}).get(f"{kind}:{sname.split('[')[0]}", 1.0)
            judged = fs["judged"] and nevA >= CM.MIN_EVENTS_PER_CELL and nevB >= CM.MIN_EVENTS_PER_CELL
            r = CellResult(q.name, source_group(q.group, sname), mask, tuple(cell["key"]), sname, float(a) if np.isfinite(a) else np.nan,
                           float(b) if np.isfinite(b) else np.nan, eff, se, tol, gross if gross is not None else np.nan,
                           kind, nevA, nevB, int(mA.sum()), int(mB.sum()), judged,
                           extra=dict(level=cell["level"], dims=cell["dims"], pred_se=fs["pred"]))
            r.p_tol = ST.p_exceed(eff, se, tol) if judged else 1.0
            r.p_gross = ST.p_exceed(eff, se, gross) if (judged and gross is not None) else 1.0
            out.append(r)
        if q.shape and len(qq.dims) >= 0 and q.kind in ("mean", "resid", "quant"):
            out.append(shape_cell(qq, gA, gB, mA, mB, mask, cell, nevA, nevB, n_perm, seed + cid))
    return out


def shape_cell(q, gA, gB, mA, mB, mask, cell, nevA, nevB, n_perm, seed):
    """KS distance of one cell against its event-permutation reference (plan 3.4.5)."""
    _, tol, gross = CM.TOL["shape"]
    xa, xb = np.asarray(gA["x"])[mA], np.asarray(gB["x"])[mB]
    ea, eb = np.asarray(gA["ev"])[mA], np.asarray(gB["ev"])[mB]
    # pooled events: A's events then B's
    ua, ia = np.unique(ea, return_inverse=True)
    ub, ib = np.unique(eb, return_inverse=True)
    x = np.r_[xa, xb]
    ev = np.r_[ia, len(ua) + ib]
    side = np.r_[np.ones(len(ua), bool), np.zeros(len(ub), bool)]
    k = ST.KSPerm(x, ev, side)
    D = k.distance()
    perms = k.permutations(n_perm, seed) if len(x) else np.array([np.nan])
    q95 = float(np.nanquantile(perms, 0.95)) if np.isfinite(perms).any() else np.nan
    judged = bool(np.isfinite(q95) and q95 <= tol / 2 and nevA >= CM.MIN_EVENTS_PER_CELL
                  and nevB >= CM.MIN_EVENTS_PER_CELL)
    r = CellResult(q.name, source_group(q.group, "shape"), mask, tuple(cell["key"]), "shape KS", D, np.nan, D, np.nan,
                   tol, gross,
                   "distance", nevA, nevB, int(mA.sum()), int(mB.sum()), judged,
                   extra=dict(level=cell["level"], dims=cell["dims"], q95=q95))
    if judged and np.isfinite(D):
        r.p_tol = float((np.sum(perms >= D - tol) + 1) / (len(perms) + 1))
        r.p_gross = float((np.sum(perms >= D - gross) + 1) / (len(perms) + 1))
        r.extra["agrees"] = bool(D + q95 <= tol)
    return r


def verdicts(results, mixed_rule="sole", alpha=CM.ALPHA):
    """Labels (agrees / disagrees / inconclusive), the MATERIAL family, and the classes (plan 3.4.5, 3.4.9)."""
    # per quantity (variable, summary): Holm on p_tol over the judged cells of all masks
    byq = {}
    for i, r in enumerate(results):
        if r.judged:
            byq.setdefault((r.quantity, r.summary), []).append(i)
    dis = np.zeros(len(results), bool)
    for idx in byq.values():
        rej = ST.holm([results[i].p_tol for i in idx], alpha)
        for i, rj in zip(idx, rej):
            dis[i] = rj
    # the MATERIAL family: controls and no-source quantities (p_tol), declared ones (p_gross) unless payload-exempt
    fam, fam_p = [], []
    for i, r in enumerate(results):
        if not r.judged:
            continue
        src = CM.SOURCES[r.group]
        if src in (("report",),):
            continue
        payload_exempt = set(src) == {CM.PAYLOAD_SOURCE} or (mixed_rule == "any" and CM.PAYLOAD_SOURCE in src)
        if src == ("control",) or not src:
            fam.append(i)
            fam_p.append(r.p_tol)
        elif not payload_exempt and np.isfinite(r.gross):
            fam.append(i)
            fam_p.append(r.p_gross)
    famrej = np.zeros(len(results), bool)
    for i, rj in zip(fam, ST.holm(fam_p, alpha)):
        famrej[i] = rj
    for i, r in enumerate(results):
        if not r.judged:
            r.verdict, r.cls = "report-only", "report"
            continue
        if r.summary == "shape KS":
            ag = r.extra.get("agrees", False)
        else:
            ag = ST.agrees(r.effect, r.se, r.tol)
        r.verdict = "disagrees" if dis[i] else ("agrees" if ag else "inconclusive")
        src = CM.SOURCES[r.group]
        if src == ("report",):
            r.cls = "report"
        elif famrej[i]:
            r.cls = "MATERIAL"
        elif dis[i]:
            r.cls = "MINOR"
        else:
            r.cls = "ok"
        if (np.isfinite(r.gross) and np.isfinite(r.effect) and abs(r.effect) > r.gross and not famrej[i]
                and src not in (("report",),)):
            r.extra["borderline_gross"] = True
    return results
