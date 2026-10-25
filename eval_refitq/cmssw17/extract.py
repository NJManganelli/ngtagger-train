"""Per-file extraction for the CMSSW_17 statistical comparison (plan 3.4): one nano file -> one compact .npz of
records (events, tracks, refit hits, SmartPixels clusters, jets, particles), with the 1 GeV truth rule applied.

    python eval_refitq/cmssw17/extract.py --side d121|c17 --mask AAAA --file 4 [--out DIR] [--force]

Records keep the event's index in the file (`ev`) and the file's event identity table (run, lumi, event), so the
bootstrap and the permutation tests resample whole events. One file at a time (memory < 4 GB).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import awkward as ak
import numpy as np
import uproot

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as CM  # noqa: E402

PAR = ("rInv", "phi", "tanL", "z0", "d0")
SPIX = ("spixNAcceptedHits", "spixNKFUpdates", "spixRefitPerformed", "spixStatus", "spixAnyWindowTruncated",
        "spixLayerHitMask", "spixNCrossings", "spixDiscGoing", "spixMaxWindowMult",
        "spixChi2IncXTot", "spixChi2IncYTot", "spixChi2IncAlphaTot", "spixChi2IncBetaTot")
HIT = ("trackIdx", "layer", "hitAccepted", "kchoice", "rankChi2", "skipPrice", "chi2IncX", "chi2IncY",
       "chi2IncAlpha", "chi2IncBeta", "pullX", "pullY", "pullAlpha", "pullBeta", "selChi2Margin", "windowMult",
       "selClusterIdx", "detId")
CL_ANG = ("surface", "side", "tpEmuClass", "hasAlpha", "hasBeta", "localCotAlpha", "localCotBeta",
          "tpLocalCotAlpha", "tpLocalCotBeta", "tpIdx", "detId")
TP = ("pt", "eta", "phi0", "d0", "z0", "charge", "evt", "bx", "pdgId", "vx", "vy")
NG_SCORES = ("bTagScore", "cTagScore", "udsTagScore", "gTagScore", "tau_pTagScore", "tau_nTagScore",
             "eTagScore", "muTagScore")
COLL = {"ot": ("L1TTrack", None), "otx": ("L1TExtTrack", None),
        "sp": ("L1TSmartPixelsTrack", "L1TTrack"), "spx": ("L1TSmartPixelsExtTrack", "L1TExtTrack")}
HITS = {"sp": "L1TSmartPixelsRefitHit", "spx": "L1TSmartPixelsExtRefitHit"}


def flat(x, dtype=None):
    a = ak.to_numpy(ak.flatten(x)) if x.ndim > 1 else ak.to_numpy(x)
    return a.astype(dtype) if dtype is not None else a


def offsets(x):
    n = ak.to_numpy(ak.num(x)).astype(np.int64)
    return n, np.concatenate([[0], np.cumsum(n)])


def evidx(n):
    return np.repeat(np.arange(len(n)), n)


def branches(keys, mask):
    want = ["run", "luminosityBlock", "event", "Pileup_nPU", "Pileup_nTrueInt", "GenVtx_x", "GenVtx_y", "GenVtx_z",
            "GenVtx_t0", "GenPart_pdgId", "GenPart_pt", "GenPart_statusFlags",
            "GenJet_pt", "GenJet_eta", "GenJet_phi", "GenJet_hadronFlavour", "GenJet_partonFlavour",
            "GenJet_nBHadrons", "L1TOTStub_layer", "L1TOTStub_isBarrel"]
    want += [f"L1TTP_{c}" for c in TP]
    for tab in ("L1TTrack", "L1TExtTrack"):
        want += [f"{tab}_{c}" for c in PAR + ("pt", "eta", "tpIdx")]
        want += [f"{tab}_cov_{p}_{p}" for p in PAR]
    if mask != CM.BASELINE_MASK:
        for tab in ("L1TSmartPixelsTrack", "L1TSmartPixelsExtTrack"):
            want += [f"{tab}_{c}" for c in PAR + ("pt", "eta", "l1TrackIdx") + SPIX]
        for tab in HITS.values():
            want += [f"{tab}_{c}" for c in HIT]
        want += ["L1TSmartPixelsCluster_tpIdx", "L1TSmartPixelsCluster_surface", "L1TSmartPixelsCluster_side",
                 "L1TSmartPixelsCluster_detId"]
        if mask == CM.CLUSTER_MASK:
            want += [f"L1TSmartPixelsCluster_{c}" for c in CL_ANG]
    for tab in ("L1PFCand", "L1PuppiCand", "L1ExtPuppiCand"):
        want += [f"{tab}_charge"]
    want += [f"L1puppiJetSC4NG_{c}" for c in ("pt", "eta", "phi") + NG_SCORES]
    want += ["L1puppiJetSC4_pt", "L1puppiJetSC8_pt"]
    have = set(keys)
    return sorted(set(w for w in want if w in have)), sorted(set(w for w in want if w not in have))


def kconstant(rinv, pt, q):
    """|k| and the sign s with rInv = s * q * |k| / pT, measured on matched OT tracks."""
    ok = (q != 0) & np.isfinite(rinv) & (pt > 0)
    kabs = float(np.median(np.abs(rinv[ok]) * pt[ok]))
    s = float(np.sign(np.median(np.sign(rinv[ok]) * q[ok])))
    return kabs, s


def wrap(dphi):
    return (dphi + np.pi) % (2 * np.pi) - np.pi


def extract(side, mask, n, out, geo=None, tp_pt_min=CM.TP_PT_MIN):
    path = CM.side_path(side, mask, n)
    t0 = time.time()
    t = uproot.open(path)["Events"]
    br, missing = branches(t.keys(), mask)
    A = t.arrays(br)
    R = {"meta_side": side, "meta_mask": mask, "meta_file": n, "meta_path": str(path),
         "meta_missing": np.array(missing, dtype=object)}
    nev = len(A)
    R["ev_run"] = flat(A["run"], np.int64)
    R["ev_lumi"] = flat(A["luminosityBlock"], np.int64)
    R["ev_event"] = flat(A["event"], np.int64)
    for c in ("Pileup_nPU", "Pileup_nTrueInt", "GenVtx_x", "GenVtx_y", "GenVtx_z", "GenVtx_t0"):
        if c in A.fields:
            R["ev_" + c] = flat(A[c], np.float64)

    # ---- TrackingParticles and the 1 GeV rule (plan 3.4.2) ----
    ntp, tpoff = offsets(A["L1TTP_pt"])
    TPc = {c: flat(A[f"L1TTP_{c}"], np.float64) for c in TP}
    # tp_pt_min = 0 switches the 1 GeV rule off (planted defect 10 only)
    tp_ok = (TPc["charge"] != 0) & (TPc["pt"] >= tp_pt_min)
    R["meta_tp_pt_min"] = np.array([tp_pt_min])
    tp_hs = (TPc["evt"] == 0) & (TPc["bx"] == 0)
    tpev = evidx(ntp)
    sel = tp_ok & (np.abs(TPc["eta"]) < 2.4)
    R["ev_nPart_hs"] = np.bincount(tpev[sel & tp_hs], minlength=nev).astype(np.int64)
    R["ev_nPart_pu"] = np.bincount(tpev[sel & ~tp_hs], minlength=nev).astype(np.int64)
    R["ev_nTP_raw"] = ntp

    def tp_global(local, ev):
        """event-local L1TTP index -> global index, -1 when absent or failing the 1 GeV rule."""
        g = np.where(local >= 0, tpoff[ev] + local, -1)
        okg = g >= 0
        g2 = g.copy()
        g2[okg] = np.where(tp_ok[g[okg]], g[okg], -1)
        return g2

    # ---- OT tracks (prompt, extended) ----
    trk = {}
    for key in ("ot", "otx"):
        tab = COLL[key][0]
        nt, toff = offsets(A[f"{tab}_pt"])
        T = {c: flat(A[f"{tab}_{c}"], np.float64) for c in PAR + ("pt", "eta")}
        T["ev"] = evidx(nt)
        T["tp"] = tp_global(flat(A[f"{tab}_tpIdx"], np.int64), T["ev"])
        T["off"] = toff
        T["n"] = nt
        trk[key] = T
        R[f"ev_n_{key}"] = nt
    kabs, ks = kconstant(trk["ot"]["rInv"][trk["ot"]["tp"] >= 0], trk["ot"]["pt"][trk["ot"]["tp"] >= 0],
                         TPc["charge"][trk["ot"]["tp"][trk["ot"]["tp"] >= 0]])
    R["meta_k"] = np.array([kabs, ks])
    # d0 convention (reco against L1TTP_d0): the sign with the narrower residual core (both samples run the same
    # code; recorded, never tuned per side)
    m = trk["otx"]["tp"] >= 0
    dd = trk["otx"]["d0"][m]
    td = TPc["d0"][trk["otx"]["tp"][m]]
    d0s = 1.0 if np.median(np.abs(dd - td)) <= np.median(np.abs(dd + td)) else -1.0
    R["meta_d0sign"] = np.array([d0s])

    def truth_params(g):
        """truth (rInv, phi0, tanL, z0, d0) at global L1TTP indices g (>= 0)."""
        return {"rInv": ks * TPc["charge"][g] * kabs / TPc["pt"][g], "phi": TPc["phi0"][g],
                "tanL": np.sinh(TPc["eta"][g]), "z0": TPc["z0"][g], "d0": d0s * TPc["d0"][g]}

    def save_tracks(key, T, extra=None):
        p = f"trk_{key}_"
        R[p + "ev"] = T["ev"].astype(np.int32)
        for c in PAR + ("pt", "eta"):
            R[p + c] = T[c].astype(np.float32)
        g = T["tp"]
        R[p + "tp"] = g.astype(np.int64)
        R[p + "tp_pt"] = np.where(g >= 0, TPc["pt"][np.maximum(g, 0)], np.nan).astype(np.float32)
        R[p + "tp_eta"] = np.where(g >= 0, TPc["eta"][np.maximum(g, 0)], np.nan).astype(np.float32)
        R[p + "tp_hs"] = np.where(g >= 0, tp_hs[np.maximum(g, 0)], False)
        tru = truth_params(np.maximum(g, 0))
        for c in PAR:
            r = T[c] - tru[c]
            if c == "phi":
                r = wrap(r)
            R[p + "res_" + c] = np.where(g >= 0, r, np.nan).astype(np.float32)
        for k, v in (extra or {}).items():
            R[p + k] = v

    for key in ("ot", "otx"):
        save_tracks(key, trk[key])

    # ---- SmartPixels tracks, refit hits, clusters ----
    if mask != CM.BASELINE_MASK:
        ncl, cloff = offsets(A["L1TSmartPixelsCluster_tpIdx"])
        clev = evidx(ncl)
        cl_tp = tp_global(flat(A["L1TSmartPixelsCluster_tpIdx"], np.int64), clev)
        cl_surf = flat(A["L1TSmartPixelsCluster_surface"], np.int64)
        cl_side = flat(A["L1TSmartPixelsCluster_side"], np.int64)
        cl_det = flat(A["L1TSmartPixelsCluster_detId"], np.int64)
        for S in CM.SURFACES:
            R[f"ev_ncl_{S}"] = np.bincount(clev[cl_surf == S], minlength=nev).astype(np.int64)
        active = active_surfaces(mask)
        for key in ("sp", "spx"):
            tab, ottab = COLL[key]
            nt, toff = offsets(A[f"{tab}_pt"])
            T = {c: flat(A[f"{tab}_{c}"], np.float64) for c in PAR + ("pt", "eta")}
            T["ev"] = evidx(nt)
            l1 = flat(A[f"{tab}_l1TrackIdx"], np.int64)
            ot = trk["ot" if key == "sp" else "otx"]
            og = np.where(l1 >= 0, ot["off"][T["ev"]] + l1, -1)           # global OT track
            T["tp"] = np.where(og >= 0, ot["tp"][np.maximum(og, 0)], -1)   # truth through the OT match
            ex = {c: flat(A[f"{tab}_{c}"]) for c in SPIX}
            ex = {"spix_" + k[4:]: v for k, v in ex.items()}
            ex["ot_eta"] = np.where(og >= 0, ot["eta"][np.maximum(og, 0)], np.nan).astype(np.float32)
            ex["ot_z0dir"] = np.where(og >= 0, ot["z0"][np.maximum(og, 0)] * np.sign(ot["tanL"][np.maximum(og, 0)]),
                                      np.nan).astype(np.float32)
            ex["ot_pt"] = np.where(og >= 0, ot["pt"][np.maximum(og, 0)], np.nan).astype(np.float32)
            ex["og"] = og
            if geo is not None:
                xn, xd = visit_plan_expect(np.stack([ot[c] for c in PAR], 1), geo, mask)
                ex["exp_ncross"] = np.where(og >= 0, xn[np.maximum(og, 0)], -1).astype(np.int16)
                ex["exp_discgoing"] = np.where(og >= 0, xd[np.maximum(og, 0)], False)
            R[f"ev_n_{key}"] = nt
            R[f"ev_n_{key}_refit"] = np.bincount(T["ev"][ex["spix_RefitPerformed"] > 0], minlength=nev)
            save_tracks(key, T, ex)
            # refit hits
            htab = HITS[key]
            nh, _ = offsets(A[f"{htab}_trackIdx"])
            H = {c: flat(A[f"{htab}_{c}"]) for c in HIT}
            hev = evidx(nh)
            hg = toff[hev] + H["trackIdx"].astype(np.int64)                 # global SmartPixels track
            sci = H["selClusterIdx"].astype(np.int64)
            cg = np.where(sci >= 0, cloff[hev] + sci, -1)
            if np.any(cg >= 0):
                bad = (cg >= 0) & (cl_det[np.maximum(cg, 0)] != H["detId"].astype(np.int64))
                if bad.any():
                    raise RuntimeError(f"{path}: {bad.sum()} refit hits whose selected cluster detId differs")
            ctp = np.where(cg >= 0, cl_tp[np.maximum(cg, 0)], -1)
            ttp = T["tp"][hg]
            acc = H["hitAccepted"] > 0
            truth = np.where(~acc | (ttp < 0), -1, np.where(ctp < 0, 2, np.where(ctp == ttp, 0, 1)))
            p = f"hit_{key}_"
            R[p + "ev"] = hev.astype(np.int32)
            R[p + "trk"] = hg.astype(np.int64)
            for c in HIT:
                if c not in ("trackIdx", "selClusterIdx", "detId"):
                    R[p + c] = H[c].astype(np.float32)
            R[p + "truth"] = truth.astype(np.int8)
            R[p + "side"] = np.where(cg >= 0, cl_side[np.maximum(cg, 0)], 0).astype(np.int8)
            R[p + "ot_pt"] = ex["ot_pt"][hg]
            # SmartPixels accepted-hit efficiency (plan 3.4.3): particles matched to an extended OT track
            if key == "spx":
                eff_sp(R, TPc, tp_ok, tp_hs, tpev, nev, cl_tp, cl_surf, active, T, hg, acc, truth)
        # angles: the mask-independent cluster table, read once
        if mask == CM.CLUSTER_MASK:
            K = {c: flat(A[f"L1TSmartPixelsCluster_{c}"]) for c in CL_ANG}
            cls = K["tpEmuClass"].astype(np.int64)
            for S in CM.SURFACES:
                for c, nm in ((1, "T"), (0, "C")):
                    R[f"ev_ncl_{S}_{nm}"] = np.bincount(clev[(cl_surf == S) & (cls == c)], minlength=nev)
            keep = (cl_tp >= 0) & ((cls == 0) | (cls == 1))
            R["cl_ev"] = clev[keep].astype(np.int32)
            for c in ("surface", "side", "tpEmuClass", "hasAlpha", "hasBeta"):
                R["cl_" + c] = K[c][keep].astype(np.int8)
            for c in ("localCotAlpha", "localCotBeta", "tpLocalCotAlpha", "tpLocalCotBeta"):
                R["cl_" + c] = K[c][keep].astype(np.float32)
            R["cl_tp_pt"] = TPc["pt"][cl_tp[keep]].astype(np.float32)

    # ---- per-event multiplicities of PF/Puppi, jets, stubs ----
    for tab in ("L1PFCand", "L1PuppiCand", "L1ExtPuppiCand"):
        if f"{tab}_charge" in A.fields:
            q = A[f"{tab}_charge"]
            R[f"ev_n_{tab}_ch"] = ak.to_numpy(ak.sum(q != 0, axis=1)).astype(np.int64)
            R[f"ev_n_{tab}_neu"] = ak.to_numpy(ak.sum(q == 0, axis=1)).astype(np.int64)
    for tab in ("L1puppiJetSC4NG", "L1puppiJetSC4", "L1puppiJetSC8"):
        if f"{tab}_pt" in A.fields:
            R[f"ev_n_{tab}_30"] = ak.to_numpy(ak.sum(A[f"{tab}_pt"] > 30, axis=1)).astype(np.int64)
    if "L1TOTStub_layer" in A.fields:
        lay = A["L1TOTStub_layer"]
        isb = A["L1TOTStub_isBarrel"]
        for L in range(1, 7):
            R[f"ev_nstub_b{L}"] = ak.to_numpy(ak.sum((isb != 0) & (lay == L), axis=1)).astype(np.int64)
        for D in range(1, 6):
            R[f"ev_nstub_d{D}"] = ak.to_numpy(ak.sum((isb == 0) & (lay == D), axis=1)).astype(np.int64)

    # ---- particle-level baselines: top pT, GenJets ----
    if "GenPart_pdgId" in A.fields:
        pid, fl = A["GenPart_pdgId"], A["GenPart_statusFlags"]
        last = (fl & (1 << 13)) != 0
        top = (abs(pid) == 6) & last
        tpt = A["GenPart_pt"][top]
        nt, _ = offsets(tpt)
        R["gtop_ev"] = evidx(nt).astype(np.int32)
        R["gtop_pt"] = flat(tpt, np.float32)

    # ---- GenJets (particle-level baseline): pT, eta, flavour ----
    if "GenJet_pt" in A.fields:
        ngj, _ = offsets(A["GenJet_pt"])
        R["gjet_ev"] = evidx(ngj).astype(np.int32)
        R["gjet_pt"] = flat(A["GenJet_pt"], np.float32)
        R["gjet_eta"] = flat(A["GenJet_eta"], np.float32)
        R["gjet_flav"] = flavour_code(flat(A["GenJet_hadronFlavour"], np.float64),
                                      np.abs(flat(A["GenJet_partonFlavour"], np.float64)),
                                      flat(A["GenJet_nBHadrons"], np.float64))

    # ---- L1 jets (SC4 NG) with the matched GenJet (plan 3.4.3) ----
    if "L1puppiJetSC4NG_pt" in A.fields and "GenJet_pt" in A.fields:
        jets_with_genjet(R, A)

    # ---- OT track efficiency (charged prompt hard-scatter particles above 2 GeV) ----
    sel = eff_population(TPc, tp_ok, tp_hs)
    matched = np.zeros(len(tp_ok), bool)
    g = trk["otx"]["tp"]
    matched[g[g >= 0]] = True
    R["eff_ot_ev"] = tpev[sel].astype(np.int32)
    R["eff_ot_pt"] = TPc["pt"][sel].astype(np.float32)
    R["eff_ot_eta"] = TPc["eta"][sel].astype(np.float32)
    R["eff_ot_ok"] = matched[sel]

    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp.npz"
    np.savez_compressed(tmp, **R)
    os.replace(tmp, out)
    print(f"{side} {mask} f{n:02d}: {nev} events, {time.time() - t0:.0f} s -> {out}", flush=True)


def active_surfaces(mask):
    """The surfaces an activeSP mask instruments: character k = ILk and IDk (k = 1..4)."""
    out = []
    for k, ch in enumerate(mask, start=1):
        if ch == "A":
            out += [k, 10 + k]
    return set(out)


def eff_population(TPc, tp_ok, tp_hs):
    """The efficiency population: charged hard-scatter particles above 2 GeV within |eta| < 2.4, produced within
    1 cm of the beam line (secondaries born in the material cannot make an OT track)."""
    r = np.hypot(TPc["vx"], TPc["vy"])
    return (tp_ok & tp_hs & (TPc["pt"] > CM.EFF_PT_MIN) & (np.abs(TPc["eta"]) < CM.EFF_ETA_MAX)
            & (r < CM.EFF_PROD_R_MAX))


def eff_sp(R, TPc, tp_ok, tp_hs, tpev, nev, cl_tp, cl_surf, active, T, hg, acc, truth):
    """Per charged hard-scatter particle above 2 GeV whose extended OT track has a SmartPixels track: the share of
    its own active surfaces (surfaces with at least one of its clusters) on which its track accepted its own
    cluster."""
    sel = eff_population(TPc, tp_ok, tp_hs)
    mine = (cl_tp >= 0) & np.isin(cl_surf, list(active))
    key = cl_tp[mine] * 32 + cl_surf[mine]
    ukey = np.unique(key)
    nown = np.bincount(ukey // 32, minlength=len(tp_ok))
    # the particle's SmartPixels track(s): the tracks whose OT match is the particle; right accepted hits per track
    right = acc & (truth == 0)
    nright_trk = np.bincount(hg[right], minlength=len(T["tp"]))
    tp_of_trk = T["tp"]
    best = np.zeros(len(tp_ok), np.int64)
    has = np.zeros(len(tp_ok), bool)
    m = tp_of_trk >= 0
    np.maximum.at(best, tp_of_trk[m], nright_trk[m])
    has[tp_of_trk[m]] = True
    s = sel & has & (nown > 0)
    R["eff_sp_ev"] = tpev[s].astype(np.int32)
    R["eff_sp_pt"] = TPc["pt"][s].astype(np.float32)
    R["eff_sp_eta"] = TPc["eta"][s].astype(np.float32)
    R["eff_sp_val"] = np.minimum(best[s] / nown[s], 1.0).astype(np.float32)


def jets_with_genjet(R, A):
    jpt, jeta, jphi = A["L1puppiJetSC4NG_pt"], A["L1puppiJetSC4NG_eta"], A["L1puppiJetSC4NG_phi"]
    gpt, geta, gphi = A["GenJet_pt"], A["GenJet_eta"], A["GenJet_phi"]
    pairs = ak.cartesian({"j": ak.zip({"eta": jeta, "phi": jphi}), "g": ak.zip({"eta": geta, "phi": gphi})},
                         nested=True)
    dr = np.hypot(pairs.j.eta - pairs.g.eta, wrap(pairs.j.phi - pairs.g.phi))
    imin = ak.argmin(dr, axis=2, keepdims=True)
    drmin = ak.flatten(dr[imin], axis=2)
    gi = ak.flatten(imin, axis=2)
    ok = ak.fill_none(drmin < 0.4, False)
    nj, _ = offsets(jpt)
    R["jet_ev"] = evidx(nj).astype(np.int32)
    R["jet_pt"] = flat(jpt, np.float32)
    R["jet_eta"] = flat(jeta, np.float32)
    for s in NG_SCORES:
        if f"L1puppiJetSC4NG_{s}" in A.fields:
            R[f"jet_{s}"] = flat(A[f"L1puppiJetSC4NG_{s}"], np.float32)
    okf = flat(ok)
    gif = flat(ak.fill_none(gi, -1), np.int64)
    ngj, goff = offsets(gpt)
    jev = R["jet_ev"].astype(np.int64)
    gg = np.where(okf & (gif >= 0), goff[jev] + gif, -1)
    G = {c: flat(A[f"GenJet_{c}"], np.float64) for c in ("pt", "eta", "hadronFlavour", "partonFlavour", "nBHadrons")}
    pick = np.maximum(gg, 0)
    R["jet_gen_pt"] = np.where(gg >= 0, G["pt"][pick], np.nan).astype(np.float32)
    R["jet_gen_eta"] = np.where(gg >= 0, G["eta"][pick], np.nan).astype(np.float32)
    flav = flavour_code(G["hadronFlavour"][pick], np.abs(G["partonFlavour"][pick]), G["nBHadrons"][pick])
    R["jet_flav"] = np.where(gg >= 0, flav, -1).astype(np.int8)


def flavour_code(hf, pf, nb):
    """0 b with one B hadron, 1 b with two or more, 2 c, 3 light quark, 4 gluon, -1 other."""
    flav = np.full(len(hf), -1, np.int8)
    flav[(hf == 5) & (nb == 1)] = 0
    flav[(hf == 5) & (nb >= 2)] = 1
    flav[hf == 4] = 2
    flav[(hf == 0) & (pf >= 1) & (pf <= 3)] = 3
    flav[(hf == 0) & (pf == 21)] = 4
    return flav


def visit_plan_expect(a, geo, mask):
    """Expected crossings and the disc-going flag per OT track from the replay's visit plan (the producer's drPath:
    surfaces within their extent +- 1 cm, the mask's active surfaces)."""
    comb = os.path.join(os.path.dirname(HERE), "combinatorics")
    rep = os.path.join(os.path.dirname(HERE), "replay")
    for p in (comb, rep):
        if p not in sys.path:
            sys.path.insert(0, p)
    import ot_refit_projection as ORP
    from ngtagger.data import spix_nano as SPN
    path = ORP.surface_path(a, geo)
    seq = ORP.visit_sequence(path, ORP.active_surfaces(mask), ORP.PRODUCER_DEFAULTS["layerOrder"])
    ncross = (seq >= 0).sum(1)
    cols = [j for j, S in enumerate(path["surfaces"]) if SPN.is_disc_surface(S)]
    dg = path["in_path"][:, cols].any(1) if cols else np.zeros(len(a), bool)
    return ncross, dg


def load_geometry():
    comb = os.path.join(os.path.dirname(HERE), "combinatorics")
    if comb not in sys.path:
        sys.path.insert(0, comb)
    import ot_projection as OP
    return OP.load_geometry(CM.GEOMETRY)


def out_path(side, mask, n, base=None):
    return os.path.join(str(base or CM.OUT / "extract"), side, f"{mask}_f{n:02d}.npz")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", required=True, choices=("d121", "c17"))
    ap.add_argument("--mask", required=True, choices=CM.MASKS)
    ap.add_argument("--file", type=int, nargs="+", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-visit-plan", action="store_true")
    ap.add_argument("--tp-pt-min", type=float, default=CM.TP_PT_MIN,
                    help="truth pT threshold of the 1 GeV rule; 0 switches it off (planted defect 10, with --out)")
    a = ap.parse_args()
    if a.tp_pt_min != CM.TP_PT_MIN and a.out is None:
        raise SystemExit("--tp-pt-min other than the 1 GeV rule needs --out (never overwrite the comparison's extraction)")
    geo = None if a.no_visit_plan or a.mask == CM.BASELINE_MASK else load_geometry()
    for n in a.file:
        o = out_path(a.side, a.mask, n, a.out)
        if os.path.exists(o) and not a.force:
            print("exists:", o)
            continue
        extract(a.side, a.mask, n, o, geo, a.tp_pt_min)


if __name__ == "__main__":
    main()
