"""The parallel design on today's census (parallel_seed/PLAN.md, 2026-10-21): IT mini-tracks matched to the OT tracks
CMSSW delivers.

The IT side is the census itself, run beforehand by its own command line on one canon_v5 file and build:
  tp_findability.py -i FILE --masks MASK --seed-classes it --targets IL1,IL2,IL3,IL4,ID1,ID2,ID3,ID4 --min-layers 3
      --min-it-layers 3 --export-tracks --angle-cuts CUTS --cache-dir DIR ...
That is the build's IT-only seeds, barrel and SmartPixels discs, followed into its own active IT layers and discs only,
with the census's default angle stages, cut file, fit, acceptance and duplicate removal; truth by the census's rules
(the owner is the particle credited with most hits, a tie gives none; a particle outside L1TTP is a real owner).

Two steps:
  extract  one (census, file) pair -> a small npz, so the census's chunk cache can be deleted afterwards:
    - the IT mini-tracks are the census's exported tracks that survive duplicate removal, refit with the census's own
      fit for their covariance (the census does not export it); the refit must reproduce the exported helix;
    - the OT tracks are the event's L1TTrack collection (CMSSW's track finder), with its covariance and truth match
      (L1TTrack_tpIdx);
    - each track carries its owner's truth helix at the beamline (KF.tp_truth_table), for the resolutions;
    - every event is checked by (run, lumi, event) and its identity stored.
  match    the extracts of one sample and build -> the figures of merit (JSON): each IT mini-track is matched
    one-to-one to an OT track of its event, greedily by the helix-agreement chi2 (the squared differences over the
    summed variances), as arch_parallel.match_systems; a match is right when the IT owner is the OT track's truth
    particle. Per particle, per track (against pT and |eta|), the old scan, hits per IT mini-track, and each side's
    own resolutions and pulls against its owner's truth. Errors come from an event bootstrap.

  python arch_parallel_census.py extract -c CENSUS_DIR -i FILE -o EXTRACT.npz
  python arch_parallel_census.py match -x EXTRACT [EXTRACT ...] -o OUT.json [--label TEXT]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parent))
import census_angles as CA          # noqa: E402
import kf_emulation as KF           # noqa: E402
import ot_chain as OC               # noqa: E402
import tp_findability as TF         # noqa: E402
import tracklet_topology_cost as M  # noqa: E402
from ngtagger.data import spix_nano as SPN  # noqa: E402

PARSETS = {"phi0+z0": ("phi0", "z0"), "phi0+z0+cot": ("phi0", "z0", "cot"),
           "all four": ("phi0", "z0", "cot", "kappa")}
CUTS = (5.0, 10.0, 25.0, 50.0, 200.0, 1e9)
DEFAULT = ("phi0+z0+cot", 25.0)                      # the 16 Sep run's default
PT_BINS = (2.0, 3.0, 5.0, 10.0, np.inf)
ETA_BINS = (0.0, 0.9, 1.4, 2.0, 2.4)
HELIX = ("phi0", "z0", "cot", "kappa")
OT_COLS = ("rInv", "phi", "tanL", "z0", "pt", "tpIdx", "cov_rInv_rInv", "cov_phi_phi", "cov_tanL_tanL", "cov_z0_z0")
IT_KEEP = ("event", "owner", "nhit", "clean", "d_export") + HELIX + tuple(f"var_{p}" for p in HELIX) + \
    tuple(f"true_{p}" for p in HELIX)
OT_KEEP = ("event", "owner", "pt") + HELIX + tuple(f"var_{p}" for p in HELIX) + tuple(f"true_{p}" for p in HELIX)
OLD = {"run": "2026-09-16, AAAA, 400 events, barrel only, census OT tracks, old truth (results/arch_comparison_1000.json)",
       "it_per_event": 76.5, "ot_per_event": 405.0, "match_fraction": 0.628, "right_fraction": 0.972}
NREP = 200


# ---------------------------------------------------------------------------------------------------------- extract
def it_tracks(U, Q, shard, meta, fopt):
    """The IT mini-tracks of one chunk: the standard (2 GeV) entries' exported tracks that survive duplicate removal,
    with the census's fitted helix and, from the refit, its covariance and the refit's agreement with the export."""
    cols = {c: i for i, c in enumerate(meta["track_cols"])}
    pos = {L: i for i, L in enumerate(KF.LAYER_ORDER)}
    parts = []
    for s_i, (lay, soft) in enumerate(zip(meta["seeds"], meta["seed_soft"])):
        if soft or f"trk_{s_i}" not in shard.files:
            continue
        R, G, dr = shard[f"trk_{s_i}"], shard[f"hit_{s_i}"].astype(np.int64), shard[f"hit_dr{s_i}"].astype(bool)
        R, G = R[dr], G[dr]
        if not len(R):
            continue
        gA, gB = G[:, pos[lay[0]]], G[:, pos[lay[1]]]
        trip = (gA, gB, G[:, pos[lay[2]]] if len(lay) == 3 else gB)
        f = KF.fit_tracks(U, Q, trip, gidx=G, **fopt)
        parts.append({"event": R[:, cols["event"]].astype(np.int64), "owner": R[:, cols["own_tpidx"]].astype(np.int64),
                      "nhit": (G >= 0).sum(1).astype(np.int8), "clean": R[:, cols["is_clean"]] > 0,
                      "phi0": f["phi0"], "z0": f["z0"], "cot": f["cot"], "kappa": f["kappa"],
                      **{f"var_{p}": f[f"var_{p}"] for p in HELIX},
                      "d_export": np.max(np.abs(np.stack([f["phi0"] - R[:, cols["phi0"]], f["cot"] - R[:, cols["cot"]],
                                                          f["z0"] - R[:, cols["z0"]]], 1)), 1)})
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]} if parts else None


def ot_tracks(path, ev):
    """The L1TTracks of entries ev (one file): helix, covariance, truth; kappa = rInv / kpt in the census's sign."""
    with uproot.open(path) as f:
        A = f["Events"].arrays([f"L1TTrack_{c}" for c in OT_COLS] + ["run", "luminosityBlock", "event"],
                               entry_start=int(ev[0]), entry_stop=int(ev[-1]) + 1)
    n = np.asarray([len(x) for x in A["L1TTrack_rInv"]])
    T = {c: np.concatenate([np.asarray(x, np.float64) for x in A[f"L1TTrack_{c}"]]) if n.sum() else np.zeros(0)
         for c in OT_COLS}
    T["event"] = np.repeat(np.asarray(ev, np.int64), n)
    kpt = float(np.median(T["pt"] * np.abs(T["rInv"])))
    T.update(phi0=T["phi"], cot=T["tanL"], kappa=T["rInv"] / kpt, var_phi0=T["cov_phi_phi"],
             var_z0=T["cov_z0_z0"], var_cot=T["cov_tanL_tanL"], var_kappa=T["cov_rInv_rInv"] / kpt ** 2,
             owner=T["tpIdx"].astype(np.int64))
    ids = np.stack([np.asarray(A["run"]), np.asarray(A["luminosityBlock"]), np.asarray(A["event"])], 1)
    return T, ids


def with_truth(T, TP):
    """T plus its owner's truth helix at the beamline POCA (KF.tp_truth_table: true_kappa is the UNSIGNED 1/pT), NaN
    for a track with no owner or an owner outside L1TTP."""
    p, has = KF.tp_truth_rows(TP, T["event"], T["owner"])
    for q in HELIX:
        T[f"true_{q}"] = np.where(has, TP[q][p], np.nan) if len(TP["key"]) else np.full(len(has), np.nan)
    return T


def load_pair(census, path):
    """One (census, file) pair: its IT mini-tracks and the file's OT tracks with their truth, events numbered from 0
    in the file, each event checked by (run, lumi, event); returns (IT, OT, n_events, the census's configuration, the
    events' (run, lumi, event))."""
    d = Path(TF.census_dir(census))
    meta = json.load(open(d / "manifest.json"))
    cfg = meta["config"]
    OC.set_discs(cfg["ot_chain"]["discs"], fixes=cfg["ot_chain"]["discs"])
    model = CA.Model(s1_d0max=CA.S1_D0MAX_CM if cfg["ot_chain"]["discs"] else 0.0)
    ang = M.AngleUse(M.DEFAULT_STAGES)
    ang.model = model
    fopt = {"use_angles": ang("kf"), "angles": tuple(sorted(ang.angles["kf"])), "key": ang.keys["kf"],
            "model": ang.model, "d0_prior_cm": cfg["kf_opts"].get("d0_prior_cm", KF.D0_PRIOR_CM)}
    sigz = {int(k): float(v) for k, v in meta["calibration"]["sigz_ot"].items()}
    ITs, OTs, ids_all, evs = [], [], [], []
    for ci, ((U, Q), ev) in enumerate(TF.unified_chunks(path, meta["n_events"], cfg["chunk"], sigz, None, model,
                                                        cfg["it_discs"])):
        TP = KF.tp_truth_table(U)
        IT = it_tracks(U, Q, np.load(d / f"chunk_{ci:05d}.npz"), meta, fopt)
        OT, ids = ot_tracks(path, ev)
        ids_all.append(ids)
        evs.append(np.asarray(ev))
        if IT is not None:
            ITs.append(with_truth(IT, TP))
        OTs.append(with_truth(OT, TP))
        del U, Q
    nev = int(np.concatenate(evs).max()) + 1
    with uproot.open(path) as f:
        own = np.stack([f["Events"][b].array(library="np") for b in ("run", "luminosityBlock", "event")], 1)[:nev]
    assert (np.concatenate(ids_all) == own[np.concatenate(evs)]).all(), f"{path}: event identity differs"
    if not ITs:
        raise SystemExit(f"{census}: no IT mini-tracks")
    IT = {k: np.concatenate([x[k] for x in ITs]) for k in IT_KEEP}
    OT = {k: np.concatenate([x[k] for x in OTs]) for k in OT_KEEP}
    return IT, OT, nev, cfg, own


def extract(a):
    IT, OT, nev, cfg, ids = load_pair(a.census, a.input)
    meta = {"census": a.census, "census_dir": str(TF.census_dir(a.census)), "input": a.input, "n_events": nev,
            "masks": list(cfg["masks"])}
    tmp = a.out + ".part.npz"
    np.savez_compressed(tmp, ids=ids, meta=json.dumps(meta), **{f"it_{k}": v for k, v in IT.items()},
                        **{f"ot_{k}": v for k, v in OT.items()})
    os.replace(tmp, a.out)
    print(f"{a.out}: {nev} events, {len(IT['event'])} IT mini-tracks, {len(OT['event'])} OT tracks; refit vs export "
          f"{np.max(IT['d_export']):.2g}")
    return 0


# ------------------------------------------------------------------------------------------------------------ match
def match(IT, OT, pars, cut, kappa_sign):
    """arch_parallel.match_systems' greedy one-to-one match per event (best chi2 first), on dicts with an event key,
    both sorted by event; the OT curvature in the census's sign convention (kappa_sign)."""
    ia_l, ib_l = [], []
    ea, eb = np.searchsorted(IT["event"], np.arange(IT["event"].max() + 2)), \
        np.searchsorted(OT["event"], np.arange(IT["event"].max() + 2))
    for e in np.unique(IT["event"]):
        ai, bi = np.arange(ea[e], ea[e + 1]), np.arange(eb[e], eb[e + 1])
        if not len(ai) or not len(bi):
            continue
        AA, BB = np.repeat(ai, len(bi)), np.tile(bi, len(ai))
        c2 = np.zeros(len(AA))
        for p in pars:
            ob = OT[p][BB] * (kappa_sign if p == "kappa" else 1.0)
            da = IT[p][AA] - ob
            if p == "phi0":
                da = M.wrap(da)
            c2 += da * da / np.maximum(IT["var_" + p][AA] + OT["var_" + p][BB], 1e-30)
        good = np.flatnonzero(c2 <= cut)
        used_a, used_b = set(), set()
        for k in good[np.argsort(c2[good], kind="stable")]:
            a, b = int(AA[k]), int(BB[k])
            if a in used_a or b in used_b:
                continue
            used_a.add(a)
            used_b.add(b)
            ia_l.append(a)
            ib_l.append(b)
            if len(used_a) == len(ai) or len(used_b) == len(bi):
                break                                   # one side exhausted: no further pair can be taken
    return np.array(ia_l, np.int64), np.array(ib_l, np.int64)


def _frac(num, den, ev, nev, rng, nrep=NREP):
    """sum(num) / sum(den) with its event-bootstrap sd (per-event sums, multinomial replicas)."""
    n = np.bincount(ev, weights=num.astype(float), minlength=nev)
    d = np.bincount(ev, weights=den.astype(float), minlength=nev)
    w = rng.multinomial(nev, np.full(nev, 1.0 / nev), size=nrep)
    boot = (w @ n) / np.maximum(w @ d, 1)
    return {"value": float(n.sum() / max(d.sum(), 1)), "sd": float(np.std(boot)), "den": int(d.sum())}


def _core(x):
    """Half-width of the central 68 % (arch_comparison.rs), NaN under 20 entries."""
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return float("nan")
    q = np.percentile(x, [15.865, 84.135])
    return float(0.5 * (q[1] - q[0]))


def _stats(T, rows):
    """Estimate minus truth, and its pull, per helix parameter over `rows` of T (rows with truth only): the mean, the
    core (central 68 % half-width) and the full RMS. phi0 [rad], z0 [cm], cot; kappa is relative (|kappa| pT_true - 1)
    and kappa_abs absolute (|kappa| - 1 / pT_true [1/GeV], the 16 Sep run's), both with the absolute pull."""
    rows = rows[np.isfinite(T["true_phi0"][rows])]
    d = {"phi0": M.wrap(T["phi0"][rows] - T["true_phi0"][rows]), "z0": T["z0"][rows] - T["true_z0"][rows],
         "cot": T["cot"][rows] - T["true_cot"][rows], "kappa": np.abs(T["kappa"][rows]) - T["true_kappa"][rows]}
    out = {"n": int(len(rows))}
    for q in HELIX + ("kappa_abs",):
        b = q.replace("_abs", "")
        x = d[b] / T["true_kappa"][rows] if q == "kappa" else d[b]
        pull = d[b] / np.sqrt(np.maximum(T[f"var_{b}"][rows], 1e-30))
        out[q] = {"mean": float(np.mean(x)) if len(x) else None, "core": _core(x),
                  "rms": float(np.sqrt(np.mean(x * x))) if len(x) else None,
                  "pull_core": _core(pull), "pull_rms": float(np.sqrt(np.mean(pull * pull))) if len(x) else None}
    return out


def resolutions(T, rows):
    """_stats over all of `rows`, and binned in the owner's true |eta| and pT."""
    teta = np.abs(np.arcsinh(T["true_cot"][rows]))
    tpt = 1.0 / T["true_kappa"][rows]
    res = {"all": _stats(T, rows), "abs_eta": [], "pt": []}
    for lab, x, edges in (("abs_eta", teta, ETA_BINS), ("pt", tpt, PT_BINS)):
        for lo, hi in zip(edges[:-1], edges[1:]):
            res[lab].append({"lo": lo, "hi": None if np.isinf(hi) else hi,
                             **_stats(T, rows[(x >= lo) & (x < hi)])})
    return res


def load_extracts(paths):
    """The extracts of one sample and build, events renumbered in sequence; returns (IT, OT, n_events, meta list,
    the events' (run, lumi, event))."""
    ITs, OTs, metas, ids, nev = [], [], [], [], 0
    for p in paths:
        z = np.load(p)
        m = json.loads(str(z["meta"]))
        IT = {k[3:]: z[k] for k in z.files if k.startswith("it_")}
        OT = {k[3:]: z[k] for k in z.files if k.startswith("ot_")}
        IT["event"] = IT["event"] + nev
        OT["event"] = OT["event"] + nev
        ITs.append(IT)
        OTs.append(OT)
        metas.append(m)
        ids.append(z["ids"])
        nev += m["n_events"]
    if len({tuple(m["masks"]) for m in metas}) != 1:
        raise SystemExit(f"the extracts mix builds: {sorted({tuple(m['masks']) for m in metas})}")
    IT = {k: np.concatenate([x[k] for x in ITs]) for k in ITs[0]}
    OT = {k: np.concatenate([x[k] for x in OTs]) for k in OTs[0]}
    for T in (IT, OT):                                  # event order, stable within an event
        o = np.argsort(T["event"], kind="stable")
        for k in T:
            T[k] = T[k][o]
    return IT, OT, nev, metas, np.concatenate(ids)


def summarize(IT, OT, nev, metas, label):
    """Every figure of merit of one sample and build (the JSON written by `match`)."""
    it_own, ot_own = SPN.tp_matched(IT["owner"]), SPN.tp_matched(OT["owner"])
    # the census curvature's sign against CMSSW's, from the right owners at the loosest phi0+z0+cot match
    ia, ib = match(IT, OT, PARSETS["phi0+z0+cot"], 25.0, 1.0)
    ok = it_own[ia] & (IT["owner"][ia] == OT["owner"][ib])
    sign = 1.0 if np.mean(np.sign(IT["kappa"][ia][ok]) == np.sign(OT["kappa"][ib][ok])) >= 0.5 else -1.0
    rng = np.random.default_rng(20261021)
    nh = IT["nhit"]
    res = {"plan": "parallel_seed/PLAN.md", "label": label, "census": [m["census"] for m in metas],
           "input": [m["input"] for m in metas], "mask": list(metas[0]["masks"]), "n_events": nev,
           "it_per_event": len(IT["event"]) / nev, "ot_per_event": len(OT["event"]) / nev,
           "it_owned": float(np.mean(it_own)), "ot_owned": float(np.mean(ot_own)),
           "refit_vs_export_max": float(np.max(IT["d_export"])), "kappa_sign": sign, "old": OLD,
           "hits": {lab: {"share": float(b.mean()), "owned": float(np.mean(it_own[b])) if b.any() else None,
                          "share_of_fakes": float((b & ~it_own).sum() / max((~it_own).sum(), 1))}
                    for lab, b in (("3", nh == 3), ("4", nh == 4), ("5+", nh >= 5))},
           "scan": {}}
    for pname, pars in PARSETS.items():
        for cut in CUTS:
            ia, ib = match(IT, OT, pars, cut, sign)
            right = it_own[ia] & (IT["owner"][ia] == OT["owner"][ib])
            res["scan"][f"{pname}|{cut:g}"] = {"match_fraction": len(ia) / len(IT["event"]),
                                               "right_fraction": float(right.mean()) if len(ia) else None}
            if (pname, cut) != DEFAULT:
                continue
            m = np.zeros(len(IT["event"]), bool)
            r = np.zeros(len(IT["event"]), bool)
            m[ia], r[ia] = True, right
            pt = np.where(m, 0.0, 1.0 / np.maximum(np.abs(IT["kappa"]), 1e-6))
            pt[ia] = OT["pt"][ib]
            aeta = np.abs(np.arcsinh(IT["cot"]))
            ev = IT["event"]
            D = {"match": _frac(m, np.ones(len(m), bool), ev, nev, rng), "right": _frac(r, m, ev, nev, rng)}
            for lab, x, edges in (("pt", pt, PT_BINS), ("abs_eta", aeta, ETA_BINS)):
                D[lab] = []
                for lo, hi in zip(edges[:-1], edges[1:]):
                    b = (x >= lo) & (x < hi)
                    D[lab].append({"lo": lo, "hi": None if np.isinf(hi) else hi, "match": _frac(m & b, b, ev, nev, rng),
                                   "right": _frac(r & b, m & b, ev, nev, rng)})
            # per particle: the particles with an OT track; those also owning an IT mini-track; matched to each other
            okey = M.tp_key(OT["event"][ot_own], OT["owner"][ot_own])
            ot_keys, first = np.unique(okey, return_index=True)
            it_keys = np.unique(M.tp_key(IT["event"][it_own], IT["owner"][it_own]))
            got = np.unique(M.tp_key(ev[ia][right], IT["owner"][ia][right]))
            has_it, done = np.isin(ot_keys, it_keys), np.isin(ot_keys, got)
            pev = M.tp_key_parts(ot_keys)[0]
            D["per_particle"] = _frac(done, has_it, pev, nev, rng)
            D["per_ot_particle"] = _frac(done, np.ones(len(ot_keys), bool), pev, nev, rng)
            D["ot_particles_with_it"] = _frac(has_it, np.ones(len(ot_keys), bool), pev, nev, rng)
            D["ot_particles_per_event"] = len(ot_keys) / nev
            orow = np.flatnonzero(ot_own)[first]                    # one OT track per particle, for its truth
            tpt, teta = 1.0 / OT["true_kappa"][orow], np.abs(np.arcsinh(OT["true_cot"][orow]))
            for lab, x, edges in (("per_particle_pt", tpt, PT_BINS), ("per_particle_abs_eta", teta, ETA_BINS)):
                D[lab] = []
                for lo, hi in zip(edges[:-1], edges[1:]):
                    b = (x >= lo) & (x < hi)
                    D[lab].append({"lo": lo, "hi": None if np.isinf(hi) else hi,
                                   "per_particle": _frac(done & b, has_it & b, pev, nev, rng)})
            # each side's own resolutions: every owned track (IT also by hit count, and the 3-hit ones with every hit
            # the owner's), and the right pairs (the same particles on both sides)
            D["resolutions"] = {"IT": {"owned": resolutions(IT, np.flatnonzero(it_own)),
                                       "owned_3_hits": resolutions(IT, np.flatnonzero(it_own & (nh == 3))),
                                       "owned_4_or_more_hits": resolutions(IT, np.flatnonzero(it_own & (nh >= 4))),
                                       "clean_3_hits": resolutions(IT, np.flatnonzero(it_own & IT["clean"] & (nh == 3))),
                                       "right_pairs": resolutions(IT, ia[right])},
                                "OT": {"owned": resolutions(OT, np.flatnonzero(ot_own)),
                                       "right_pairs": resolutions(OT, ib[right])}}
            res["default"] = D
    return res


def match_cmd(a):
    IT, OT, nev, metas, ids = load_extracts(a.extracts)
    if len(np.unique(ids, axis=0)) != len(ids):
        raise SystemExit("an event appears twice across the extracts")
    res = {"extracts": a.extracts, **summarize(IT, OT, nev, metas, a.label)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    dd = res["default"]
    print(f"{a.label} {res['mask']} {nev} events: IT {res['it_per_event']:.1f}/ev ({100 * res['it_owned']:.1f} % "
          f"owned), OT {res['ot_per_event']:.1f}/ev; per particle {100 * dd['per_particle']['value']:.2f} +- "
          f"{100 * dd['per_particle']['sd']:.2f} %; matched {100 * dd['match']['value']:.2f} +- "
          f"{100 * dd['match']['sd']:.2f} %, right {100 * dd['right']['value']:.2f} +- {100 * dd['right']['sd']:.2f} %; "
          f"refit vs export {res['refit_vs_export_max']:.2g}; kappa sign {res['kappa_sign']:+.0f}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    x = sub.add_parser("extract", help="one (census, file) pair -> EXTRACT.npz")
    x.add_argument("-c", "--census", required=True, help="the IT-only census's cache directory for this file")
    x.add_argument("-i", "--input", required=True, help="the canon_v5 file the census read")
    x.add_argument("-o", "--out", required=True)
    m = sub.add_parser("match", help="the extracts of one sample and build -> OUT.json")
    m.add_argument("-x", "--extracts", nargs="+", required=True)
    m.add_argument("-o", "--out", required=True)
    m.add_argument("--label", default="", help="the sample's name, recorded in the JSON")
    a = ap.parse_args(argv)
    return extract(a) if a.cmd == "extract" else match_cmd(a)


if __name__ == "__main__":
    raise SystemExit(main())
