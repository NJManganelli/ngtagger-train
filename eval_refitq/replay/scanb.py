"""WP-C scan, Part B (wpc/scan/PLAN.md §3).

  dcheck  B-pre's known-good on file 4: the d0 pull core (MAD / 0.6745 of (d0 - d0_TP) / sigma(d0)) of disc-going
          against barrel-only prompt tracks per pT bin, INN-D (K = 1), per scenario; they must agree within
          0.1 + t1(N) x the combined bootstrap sigma (N = pT bins x scenarios), else MATERIAL. Reported beside it: the
          pre-fix rule (inner_material_rule="radius", the known-bad). Output: WPC_OUT/scan/bpre/dcheck_f04.json
  run     the B1 arms (and the null control) on --files x every scenario, with K from Part A (--K); per OT track
          (axis 0 = the file's tracks) the figures of merit of B2. Output: WPC_OUT/scan/arms/<scen>_f<NN>.npz
  table   B3: per arm, scenario and track class (barrel-only, disc-going) the figures of merit and their paired
          event-bootstrap differences; the main contrast's GO verdict, the secondary families, the controls.
          Output: WPC_OUT/scan/go_table<tag>.json (--files: which files enter; 4 6 is the reported out-of-fit check)

Every job runs through ops/spheavy --frozen --on-busy wait.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import awkward as ak
from scipy import stats

import arms as AR
import common as CM
import cuts as CU
import fom as FM
import gates as G
from common import OP
from ngtagger.spdefs import angles as SA

OUT = CM.OUT / "scan"
C1B = CM.OUT / "c1b"
PT_BINS = (2.0, 3.0, 5.0, 10.0, np.inf)


def _core(x):
    x = x[np.isfinite(x)]
    return float(np.median(np.abs(x - np.median(x))) / 0.6745) if len(x) >= 30 else np.nan


def dcheck(a):
    with open(C1B / "scale" / "scalefit_summary.json") as fh:
        scale = float(json.load(fh)["scale"])
    geo = OP.load_geometry(CM.GEOMETRY)
    rows = {"model": [], "radius": []}
    t0 = time.time()
    for scen in a.scen:
        path = CM.input_path(scen, CM.PRED_FILE)
        cfg = CM.producer_config(path)
        tools = AR.Tools(cfg, disc=AR.DiscTools(disc_scale=scale))
        arms = {"model": AR.inn_disc(cfg), "radius": AR.inn_disc(cfg, name="INN-D-radius", inner_material_rule="radius")}
        for start, A in CM.iterate(path, AR.branches(path), chunk=a.chunk, nev=a.nev):
            ctx = AR.prepare(A, geo)
            tr = FM.track_truth(ctx)
            for lab, arm in arms.items():
                out = AR.run_arm(ctx, geo, cfg, scen, arm, tools)
                path_, _ = AR.visit_plan(ctx, geo, cfg, scen, arm)
                ok = out["refit"] & tr["pop"] & (tr["origin"] == 0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    pull = (out["a"][:, 4] - tr["d0"]) / np.sqrt(out["C"][:, 4, 4])
                rows[lab].append({"scen": np.full(ok.sum(), CM.SCENARIOS.index(scen)), "ev": (ctx["tev"] + start)[ok],
                                  "dg": AR.disc_going(path_)[ok], "pt": tr["pt"][ok], "pull": pull[ok]})
        print(f"[{time.time() - t0:7.1f}s] {scen}", flush=True)
    reps = G.replicas(nev=100)
    res = {"plan": "wpc/scan/PLAN.md §3 B-pre", "material": [], "bins": {}}
    nb = 0
    for lab in rows:
        R = CM.concat(rows[lab])
        for si, scen in enumerate(CM.SCENARIOS):
            for lo, hi in zip(PT_BINS[:-1], PT_BINS[1:]):
                e = {}
                for cls, m_cls in (("disc_going", R["dg"]), ("barrel_only", ~R["dg"])):
                    m = (R["scen"] == si) & m_cls & (R["pt"] >= lo) & (R["pt"] < hi)
                    x, ev = R["pull"][m], R["ev"][m].astype(np.int64)
                    B = [_core(np.repeat(x, w[ev])) for w in reps[:100]]
                    e[cls] = {"n": int(m.sum()), "core": _core(x), "sd": float(np.nanstd(B))}
                res["bins"][f"{lab}|{scen}|{lo:g}-{hi:g}"] = e
                nb += lab == "model" and np.isfinite(e["disc_going"]["core"]) and np.isfinite(e["barrel_only"]["core"])
    t = G.t_one(max(nb, 1))
    res["t1"] = t
    for k, e in res["bins"].items():
        dg, bo = e["disc_going"], e["barrel_only"]
        if not (np.isfinite(dg["core"]) and np.isfinite(bo["core"])):
            continue
        e["agrees"] = bool(abs(dg["core"] - bo["core"]) <= 0.1 + t * np.hypot(dg["sd"], bo["sd"]))
        if k.startswith("model|") and not e["agrees"]:
            res["material"].append(f"B-pre {k}: d0 pull core disc-going {dg['core']:.3f} vs barrel-only {bo['core']:.3f}")
    os.makedirs(OUT / "bpre", exist_ok=True)
    with open(OUT / "bpre" / "dcheck_f04.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({"material": res["material"]}, indent=1))
    return 0


# ---------------------------------------------------------------- the scan (B1-B3)
F_FLOOR = 0.005                    # B3: the GO floor, 0.5 pt of the primary benefit
H_MARGIN, P_MARGIN = 0.003, 0.002  # B3: the entry gate's E5 margins (H up, P down)
S_IP = 3.0                         # the J3 proxies' IP-significance threshold
JET_DR, JET_PT = 0.4, 20.0         # light gen jets (hadronFlavour 0) for the fake-displaced rate
N_REP = 400


FAMILY = {"PM-ab": "reference", "PM-none": "reference", "INN-D": "main",
          **{k: "angles" for k in ("none", "alpha", "beta", "sel", "upd", "first", "later", "barrel")},
          **{k: "elements" for k in ("cov-producer", "scat-producer", "disc-scale-1", "boxes", "metric-producer",
                                     "R-published")},
          "rr": "procedure", "INN-D-K1": "procedure", "none-K1": "procedure",
          **{k: "k" for k in ("k0.5-fixed", "k0.75-fixed", "k0.5-scaled", "k0.75-scaled")},
          "t9": "tail", "null": "control"}


def arms_of(cfg, K):
    """B1's arms: {name: (arm, family, disc scale or None for the fitted one)}. K: 1 or "tree2" (Part A)."""
    if cfg["useAngles"] != "alphaBeta":
        raise ValueError(f"the PM reference is refit v2 with alphaBeta; the file's producer has {cfg['useAngles']!r}")

    def d(name, **kw):
        return AR.inn_disc(cfg, name=name, K=K, **kw)
    out = {"PM-ab": (AR.pm(cfg, name="PM-ab"), "reference", None),
           "PM-none": (AR.Arm("PM-none", use=frozenset()), "reference", None),
           "INN-D": (d("INN-D"), "main", None),
           "none": (d("none", use=frozenset()), "angles", None),
           "alpha": (d("alpha", use=AR.use_set(("rank", "update"), ("alpha",))), "angles", None),
           "beta": (d("beta", use=AR.use_set(("rank", "update"), ("beta",))), "angles", None),
           "sel": (d("sel", use=AR.use_set(("rank",))), "angles", None),
           "upd": (d("upd", use=AR.use_set(("update",))), "angles", None),
           "first": (d("first", angle_steps="first"), "angles", None),
           "later": (d("later", angle_steps="later"), "angles", None),
           "barrel": (d("barrel", use=AR.use_set(("rank", "update"), AR.ANGLE_NAMES, ("barrel",))), "angles", None),
           "cov-producer": (d("cov-producer", seed_cov="reported"), "elements", None),
           "scat-producer": (d("scat-producer", scattering="producer", inner_material=False), "elements", None),
           "disc-scale-1": (d("disc-scale-1"), "elements", 1.0),
           "boxes": (d("boxes", window="box"), "elements", None),
           "metric-producer": (d("metric-producer", metric="producer"), "elements", None),
           "R-published": (d("R-published", pos_err="published"), "elements", None),
           "rr": (d("rr", rr=True), "procedure", None),
           "k0.5-fixed": (d("k0.5-fixed", k_alpha=0.5, k_beta=0.5), "k", None),
           "k0.75-fixed": (d("k0.75-fixed", k_alpha=0.75, k_beta=0.75), "k", None),
           "k0.5-scaled": (d("k0.5-scaled", k_alpha=0.5, k_beta=0.5, k_mode="scaled"), "k", None),
           "k0.75-scaled": (d("k0.75-scaled", k_alpha=0.75, k_beta=0.75, k_mode="scaled"), "k", None),
           "t9": (d("t9", angle_term="t9"), "tail", None),
           "null": (d("null"), "control", None)}
    if K != 1:
        out["INN-D-K1"] = (AR.inn_disc(cfg, name="INN-D-K1"), "procedure", None)
        out["none-K1"] = (AR.inn_disc(cfg, name="none-K1", use=frozenset()), "procedure", None)
    assert all(FAMILY[k] == f for k, (_, f, _) in out.items())
    return out


def permute_angles(ctx, geo, rng):
    """The null control's input: the measured local angles (values, sigmas, usable flags) of the clusters permuted
    within each (event, surface, key-bin index) group, axis 1 = the clusters of one event. The positions and truth
    are untouched; the angles keep their distribution per bin and lose their link to the track."""
    K = dict(ctx["K"])
    surf = geo["surface"][np.maximum(K["mi"], 0)]
    kb = SA.fine_bin(surf, SA.local_incidence_key(K["localCotAlpha"].astype(np.float64),
                                                   K["localCotBeta"].astype(np.float64)))
    grp = (K["event"].astype(np.int64) * 64 + surf) * 64 + kb.astype(np.int64)
    o = np.lexsort((rng.random(len(grp)), grp))
    srt = np.argsort(grp, kind="stable")
    perm = np.empty(len(grp), np.int64)
    perm[srt] = o                                         # the same group, a random member
    for c in ("localCotAlpha", "localCotBeta", "sigAlpha", "sigBeta", "hasAlpha", "hasBeta", "hasAlpha_raw",
              "hasBeta_raw"):
        K[c] = K[c][perm]
    return {**ctx, "K": K}


def _light_jet_tracks(A, ctx):
    """Per OT track: within dR < 0.4 of a light gen jet (hadronFlavour 0, pT > 20 GeV) of its event."""
    out = np.zeros(ctx["ntrk"], bool)
    jpt, jeta, jphi = (ak.to_numpy(ak.flatten(A[f"GenJet_{c}"])).astype(np.float64) for c in ("pt", "eta", "phi"))
    jfl = ak.to_numpy(ak.flatten(A["GenJet_hadronFlavour"]))
    jev = np.repeat(np.arange(len(A)), ak.to_numpy(ak.num(A["GenJet_pt"])))
    keep = (jfl == 0) & (jpt > JET_PT)
    jev, jeta, jphi = jev[keep], jeta[keep], jphi[keep]
    teta, tphi = ctx["T"]["eta"].astype(np.float64), ctx["T"]["phi"].astype(np.float64)
    for e in np.unique(jev):
        t = np.flatnonzero(ctx["tev"] == e)
        m = jev == e
        dphi = np.angle(np.exp(1j * (tphi[t, None] - jphi[None, m])))
        out[t] = (np.hypot(teta[t, None] - jeta[None, m], dphi) < JET_DR).any(1)
    return out


def track_foms(out, ctx, geo, tr, light):
    """Per OT track (axis 0) the B2 inputs: the innermost active surface's attachability and correct attachment,
    the disc surfaces' (counts), the accepted and own accepted hits, the final d0 / z0 and sigma(d0)."""
    rec, seq = out["rec"], out["seq"]
    n = len(seq)
    K = ctx["K"]
    tp = ctx["trk_tp"]
    left = np.unique(K["tp_global"][K["tp_global"] >= 0] * 64 + geo["surface"][np.maximum(K["mi"], 0)][K["tp_global"] >= 0])
    has_v = seq >= 0
    jlast = np.where(has_v.any(1), has_v.shape[1] - 1 - np.argmax(has_v[:, ::-1], 1), -1)
    inner = np.where(jlast >= 0, seq[np.arange(n), np.maximum(jlast, 0)], -1)

    def attachable(surf):
        return (tp >= 0) & (surf >= 0) & np.isin(tp * 64 + surf, left)
    ci = np.where(rec["acc"], rec["ci"], -1)
    own = (ci >= 0) & (K["tp_global"][np.maximum(ci, 0)] == tp[:, None]) & (tp[:, None] >= 0)
    att_den = attachable(inner)
    att_num = att_den & own[np.arange(n), np.maximum(jlast, 0)] & (jlast >= 0)
    disc = has_v & AR.SPN.is_disc_surface(seq)
    dden = np.zeros(n, np.int64)
    dnum = np.zeros(n, np.int64)
    for j in range(seq.shape[1]):
        m = disc[:, j] & attachable(seq[:, j])
        dden += m
        dnum += m & own[:, j]
    a, C = out["a"], out["C"]
    with np.errstate(invalid="ignore", divide="ignore"):
        sip = np.abs(a[:, 4]) / np.sqrt(C[:, 4, 4])
    return {"att_den": att_den, "att_num": att_num, "dden": dden, "dnum": dnum, "nacc": rec["acc"].sum(1),
            "nown": own.sum(1), "d0": a[:, 4], "z0": a[:, 3], "sip": sip, "refit": out["refit"],
            "pop": tr["pop"], "origin": tr["origin"], "pt": tr["pt"], "d0_tp": tr["d0"], "z0_tp": tr["z0"],
            "d0_ot": ctx["a"][:, 4], "z0_ot": ctx["a"][:, 3], "light": light}


def run(a):
    with open(C1B / "scale" / "scalefit_summary.json") as fh:
        scale = float(json.load(fh)["scale"])
    geo = OP.load_geometry(CM.GEOMETRY)
    cuts = CU.Cuts.load(str(C1B / "cuts" / "wpc_cuts_c1b_v1.json"))
    K = a.K if a.K == "tree2" else 1
    os.makedirs(OUT / "arms", exist_ok=True)
    for f in a.files:
        for scen in a.scen:
            path = CM.input_path(scen, f)
            cfg = CM.producer_config(path)
            assert CM.mask_of(cfg) == scen
            arms = arms_of(cfg, K)
            if a.arms:
                arms = {k: v for k, v in arms.items() if k in a.arms}
            tools = {sc: AR.Tools(cfg, cuts=cuts, disc=AR.DiscTools(disc_scale=sc)) for sc in (scale, 1.0)}
            rng = np.random.default_rng(20261018 + 100 * f + CM.SCENARIOS.index(scen))
            parts, t0, tr_off = {}, time.time(), 0
            br = list(dict.fromkeys(AR.branches(path) + [f"GenJet_{c}" for c in ("pt", "eta", "phi", "hadronFlavour")]))
            for start, A in CM.iterate(path, br, chunk=a.chunk, nev=a.nev):
                ctx = AR.prepare(A, geo)
                tr = FM.track_truth(ctx)
                light = _light_jet_tracks(A, ctx)
                path_, _ = AR.visit_plan(ctx, geo, cfg, scen, AR.inn_disc(cfg))
                dg = AR.disc_going(path_)
                for name, (arm, fam, sc) in arms.items():
                    c = permute_angles(ctx, geo, rng) if name == "null" else ctx
                    out = AR.run_arm(c, geo, cfg, scen, arm, tools[scale if sc is None else sc])
                    d = track_foms(out, ctx, geo, tr, light)
                    d.update(ev=ctx["tev"] + start, dg=dg)
                    parts.setdefault(name, []).append(d)
                tr_off += ctx["ntrk"]
            data = {k: CM.concat(v) for k, v in parts.items()}
            tag = f"{scen}_f{f:02d}" + (f"_{a.tag}" if a.tag else "")
            np.savez_compressed(OUT / "arms" / f"{tag}.npz",
                                **{f"{k}__{fld}": v for k, dd in data.items() for fld, v in dd.items()})
            print(f"[{time.time() - t0:7.1f}s] {scen} f{f:02d}: {tr_off} tracks, {len(arms)} arms", flush=True)
    return 0


def _foms(d, m):
    """Per event (axis 0 of the returned (nev, k) arrays): numerators and denominators of the B2 figures of merit
    over the tracks m: attachment, disc attachment, |dd0| < 100 um (prompt), H (prompt), P, HF attachment, the
    HF IP-significance proxy and the light-jet fake-displaced rate."""
    ok = m & d["pop"]
    prompt = ok & (d["origin"] == 0)
    hf = ok & (d["origin"] == 1) & (np.abs(d["d0_tp"]) > 0.01)
    with np.errstate(invalid="ignore"):
        dd0 = np.abs(d["d0"] - d["d0_tp"]) < 0.01
        harm = FM.harm(d["d0"], d["z0"], d["d0_ot"], d["z0_ot"], {"d0": d["d0_tp"], "z0": d["z0_tp"]})["either"]
        sig = d["sip"] > S_IP
    cols = {"att": (ok & d["att_num"], ok & d["att_den"]),
            "disc_att": (np.where(ok, d["dnum"], 0), np.where(ok, d["dden"], 0)),
            "dd0": (prompt & d["refit"] & dd0, prompt & d["refit"]),
            "H": (prompt & d["refit"] & harm, prompt & d["refit"]),
            "P": (np.where(ok, d["nown"], 0), np.where(ok, d["nacc"], 0)),
            "hf_att": (hf & d["att_num"], hf & d["att_den"]),
            "hf_sip": (hf & d["refit"] & sig, hf & d["refit"]),
            "light_fake": (prompt & d["light"] & d["refit"] & sig, prompt & d["light"] & d["refit"])}
    return cols


def _per_event(d, m, nev):
    ev = d["ev"].astype(np.int64)
    return {k: (np.bincount(ev, weights=n.astype(float), minlength=nev), np.bincount(ev, weights=q.astype(float),
                                                                                     minlength=nev))
            for k, (n, q) in _foms(d, m).items()}


def _diff(A, B, reps):
    """The FoM difference A - B (ratios of per-event sums) and its paired event-bootstrap sd per FoM."""
    out = {}
    for k in A:
        na, da = A[k]
        nb, db = B[k]
        va, vb = na.sum() / max(da.sum(), 1), nb.sum() / max(db.sum(), 1)
        boot = (reps @ na) / np.maximum(reps @ da, 1) - (reps @ nb) / np.maximum(reps @ db, 1)
        out[k] = {"a": va, "b": vb, "diff": va - vb, "sd": float(np.std(boot)), "n": int(da.sum())}
    return out


def _load_arms(files, tag):
    data = {}
    for fi, f in enumerate(files):
        for si, scen in enumerate(CM.SCENARIOS):
            name = f"{scen}_f{f:02d}" + (f"_{tag}" if tag else "")
            z = np.load(OUT / "arms" / f"{name}.npz")
            for k in z.files:
                arm, fld = k.split("__", 1)
                x = z[k]
                if fld == "ev":
                    x = x + fi * 100                     # an event's identity across the files of one scenario
                data.setdefault(scen, {}).setdefault(arm, {}).setdefault(fld, []).append(x)
    return {s: {a: {k: np.concatenate(v) for k, v in d.items()} for a, d in arms.items()} for s, arms in data.items()}


def _inject(d_ref, gain, rng):
    """The positive control: the reference arm with correct attachment added to failing attachable tracks so that
    its attachment rises by `gain` (barrel-only population tracks)."""
    d = {k: v.copy() for k, v in d_ref.items()}
    cand = np.flatnonzero(d["pop"] & ~d["dg"] & d["att_den"] & ~d["att_num"])
    den = (d["pop"] & ~d["dg"] & d["att_den"]).sum()
    k = min(len(cand), int(round(gain * den)))
    d["att_num"][rng.choice(cand, k, replace=False)] = True
    return d


def _go(D, sidak):
    """B3's GO rule on {scen: {class: diffs}} of a main contrast: AAAA's lower bound of the attachment gain on
    barrel-only tracks >= the floor; AAAI and IAAA not worse; the harm margins in every scenario and class; the
    non-prompt guard (HF attachment and the HF IP-significance proxy not worse beyond 3 sigma)."""
    z = sidak
    why = []
    for scen, by in D.items():
        b = by["barrel_only"]["att"]
        lo = b["diff"] - z * b["sd"]
        if scen == "AAAA" and lo < F_FLOOR:
            why.append(f"{scen}: attachment gain lower bound {lo:.4f} < floor {F_FLOOR}")
        if scen != "AAAA" and lo < 0:
            why.append(f"{scen}: attachment gain lower bound {lo:.4f} < 0")
        for cls in ("barrel_only", "disc_going"):
            e = by[cls]
            if e["H"]["diff"] + 1.645 * e["H"]["sd"] > H_MARGIN:
                why.append(f"{scen} {cls}: H upper bound {e['H']['diff'] + 1.645 * e['H']['sd']:.4f} > {H_MARGIN}")
            if -e["P"]["diff"] + 1.645 * e["P"]["sd"] > P_MARGIN:
                why.append(f"{scen} {cls}: P loss upper bound {-e['P']['diff'] + 1.645 * e['P']['sd']:.4f} > {P_MARGIN}")
            for k in ("hf_att", "hf_sip"):
                if e[k]["diff"] < -3 * e[k]["sd"]:
                    why.append(f"{scen} {cls}: {k} worse by {e[k]['diff']:.4f} ({e[k]['diff'] / max(e[k]['sd'], 1e-12):.1f} sd)")
    return {"go": not why, "why": why}


def table(a):
    data = _load_arms(a.files, a.tag)
    nev = 100 * len(a.files)
    reps = G.replicas(nev=nev, n=N_REP, seed=20261018)
    sidak = float(stats.norm.ppf((1 - 0.05) ** (1 / 3)))          # one-sided 95 %, Sidak over the 3 scenarios
    rng = np.random.default_rng(20261018)
    res = {"plan": "wpc/scan/PLAN.md §3 B3", "files": a.files, "material": [], "controls_reported": [],
           "contrasts": {}}

    def contrast(get_a, arm_b):
        """{scen: {class: diffs}} of get_a(scenario's arms) against the scenario's arm_b, paired by event."""
        out = {}
        for scen, arms in data.items():
            A_, B_ = get_a(scen, arms), arms[arm_b]
            out[scen] = {cls: _diff(_per_event(A_, sel, nev), _per_event(B_, sel, nev), reps)
                         for cls, sel in (("barrel_only", ~A_["dg"]), ("disc_going", A_["dg"]))}
        return out

    def arm(name):
        return lambda scen, arms: arms[name]
    main = contrast(arm("INN-D"), "PM-ab")
    res["main"] = {"contrast": "INN-D - PM-ab", **_go(main, sidak), "diffs": main}
    names = sorted({n for arms in data.values() for n in arms} - {"INN-D", "PM-ab", "null"})
    for name in names:
        ref = "PM-ab" if FAMILY[name] == "reference" else "INN-D"
        res["contrasts"][f"{name} - {ref}"] = {"family": FAMILY[name], "diffs": contrast(arm(name), ref)}
    # controls: the null (angles permuted within a key-bin index: misinformation, so it may lose) against no angles
    # must read NO-GO with no gain beyond t1 sd; the positive control (PM-ab with 1.5 floors of attachment injected)
    # must read GO. Judged on the decision scan (every file); reported on a subset
    null = contrast(arm("null"), "none")
    res["control_null"] = {**_go(null, sidak), "diffs": null,
                           "no_gain": all(by["barrel_only"]["att"]["diff"] <= G.t_one(3) * by["barrel_only"]["att"]["sd"]
                                          for by in null.values())}
    pos = contrast(lambda scen, arms: _inject(arms["PM-ab"], 1.5 * F_FLOOR, rng), "PM-ab")
    res["control_positive"] = _go(pos, sidak)
    bad = []
    if res["control_null"]["go"] or not res["control_null"]["no_gain"]:
        bad.append("the null control reads GO or gains beyond t1 sd")
    if not res["control_positive"]["go"]:
        bad.append(f"the positive control does not read GO: {res['control_positive']['why']}")
    res["material" if sorted(a.files) == list(range(1, 11)) else "controls_reported"] = bad
    tag = "_".join(f"{f:02d}" for f in a.files) + (f"_{a.tag}" if a.tag else "")
    with open(OUT / f"go_table_{tag}.json", "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(json.dumps({"go": res["main"]["go"], "why": res["main"]["why"], "material": res["material"]}, indent=1))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("job", choices=("dcheck", "run", "table"))
    ap.add_argument("--scen", nargs="+", default=list(CM.SCENARIOS))
    ap.add_argument("--files", type=int, nargs="+", default=[CM.PRED_FILE])
    ap.add_argument("--K", choices=("1", "tree2"), default="1")
    ap.add_argument("--arms", nargs="+", default=None, help="run: a subset of the arms")
    ap.add_argument("--tag", default="", help="run / table: a suffix for the output files (e.g. the dry run)")
    ap.add_argument("--nev", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=10)
    a = ap.parse_args(argv)
    return {"dcheck": dcheck, "run": run, "table": table}[a.job](a)


if __name__ == "__main__":
    raise SystemExit(main())
