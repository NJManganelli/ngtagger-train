"""The parallel design on today's census (parallel_seed/PLAN.md, 2026-10-21): IT mini-tracks matched to the OT tracks
CMSSW delivers.

The IT side is the census itself, run beforehand by its own command line on one canon_v5 file and build:
  tp_findability.py -i FILE --masks MASK --seed-classes it --targets IL1,IL2,IL3,IL4,ID1,ID2,ID3,ID4 --min-layers 3
      --min-it-layers 3 --export-tracks --angle-cuts CUTS --cache-dir DIR ...
That is the build's IT-only seeds, barrel and SmartPixels discs, followed into its own active IT layers and discs only,
with the census's default angle stages, cut file, fit, acceptance and duplicate removal; truth by the census's rules
(the owner is the particle credited with most hits, a tie gives none).

This pass, per census chunk:
  - the IT mini-tracks are the census's exported tracks that survive duplicate removal, refit with the census's own
    fit for their covariance (the census does not export it); the refit must reproduce the exported helix;
  - the OT tracks are the event's L1TTrack collection (CMSSW's track finder), with its covariance and truth match
    (L1TTrack_tpIdx);
  - each IT mini-track is matched one-to-one to an OT track of its event, greedily by the helix-agreement chi2 (the
    squared differences over the summed variances), as arch_parallel.match_systems; a match is right when the IT
    owner is the OT track's truth particle.
Events are taken from the one file both read, and every event is checked by (run, lumi, event).

  python arch_parallel_census.py -c CENSUS_DIR -i FILE -o OUT.json
"""
from __future__ import annotations

import argparse
import json
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

PARSETS = {"phi0+z0": ("phi0", "z0"), "phi0+z0+cot": ("phi0", "z0", "cot"),
           "all four": ("phi0", "z0", "cot", "kappa")}
CUTS = (5.0, 10.0, 25.0, 50.0, 200.0, 1e9)
DEFAULT = ("phi0+z0+cot", 25.0)                      # the 16 Sep run's default
PT_BINS = (2.0, 3.0, 5.0, 10.0, np.inf)
ETA_BINS = (0.0, 0.9, 1.4, 2.0, 2.4)
OT_COLS = ("rInv", "phi", "tanL", "z0", "d0", "pt", "eta", "tpIdx", "cov_rInv_rInv", "cov_phi_phi", "cov_tanL_tanL",
           "cov_z0_z0")
OLD = {"run": "2026-09-16, AAAA, 400 events, barrel only, census OT tracks, old truth (results/arch_comparison_1000.json)",
       "it_per_event": 76.5, "ot_per_event": 405.0, "match_fraction": 0.628, "right_fraction": 0.972}


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
                      "phi0": f["phi0"], "z0": f["z0"], "cot": f["cot"], "kappa": f["kappa"],
                      **{f"var_{p}": f[f"var_{p}"] for p in ("phi0", "z0", "cot", "kappa")},
                      "d_export": np.max(np.abs(np.stack([f["phi0"] - R[:, cols["phi0"]], f["cot"] - R[:, cols["cot"]],
                                                          f["z0"] - R[:, cols["z0"]]], 1)), 1),
                      "entry": np.full(len(R), s_i)})
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


def match(IT, OT, pars, cut, kappa_sign):
    """arch_parallel.match_systems' greedy one-to-one match per event (best chi2 first), on dicts with an event key;
    the OT curvature in the census's sign convention (kappa_sign)."""
    ia_l, ib_l = [], []
    for e in np.unique(IT["event"]):
        ai, bi = np.flatnonzero(IT["event"] == e), np.flatnonzero(OT["event"] == e)
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
    return np.array(ia_l, np.int64), np.array(ib_l, np.int64)


def _frac(num, den, ev, nev, rng, nrep=200):
    """sum(num) / sum(den) with its event-bootstrap sd (per-event sums, multinomial replicas)."""
    n = np.bincount(ev, weights=num.astype(float), minlength=nev)
    d = np.bincount(ev, weights=den.astype(float), minlength=nev)
    w = rng.multinomial(nev, np.full(nev, 1.0 / nev), size=nrep)
    boot = (w @ n) / np.maximum(w @ d, 1)
    return {"value": float(n.sum() / max(d.sum(), 1)), "sd": float(np.std(boot)), "den": int(d.sum())}


def key(e, t):
    """A particle's identity across events: (event, L1TTP index along axis 1 of the event)."""
    return e * (1 << 20) + t


def run(a):
    d = Path(TF.census_dir(a.census))
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
    for ci, ((U, Q), ev) in enumerate(TF.unified_chunks(a.input, meta["n_events"], cfg["chunk"], sigz, None, model,
                                                        cfg["it_discs"])):
        shard = np.load(d / f"chunk_{ci:05d}.npz")
        IT = it_tracks(U, Q, shard, meta, fopt)
        OT, ids = ot_tracks(a.input, ev)
        ids_all.append(ids)
        evs.append(np.asarray(ev))
        if IT is not None:
            ITs.append(IT)
        OTs.append(OT)
        del U, Q
    IT = {k: np.concatenate([x[k] for x in ITs]) for k in ITs[0]}
    OT = {k: np.concatenate([x[k] for x in OTs]) for k in OTs[0]}
    nev = int(np.concatenate(evs).max()) + 1
    with uproot.open(a.input) as f:
        own = np.stack([f["Events"]["run"].array(library="np"), f["Events"]["luminosityBlock"].array(library="np"),
                        f["Events"]["event"].array(library="np")], 1)[:nev]
    assert (np.concatenate(ids_all) == own[np.concatenate(evs)]).all(), "event identity differs between the passes"
    # the census curvature's sign against CMSSW's, from the right owners at the loosest phi0+z0+cot match
    ia, ib = match(IT, OT, PARSETS["phi0+z0+cot"], 25.0, 1.0)
    ok = (IT["owner"][ia] >= 0) & (IT["owner"][ia] == OT["owner"][ib])
    sign = 1.0 if np.mean(np.sign(IT["kappa"][ia][ok]) == np.sign(OT["kappa"][ib][ok])) >= 0.5 else -1.0
    rng = np.random.default_rng(20261021)
    res = {"plan": "parallel_seed/PLAN.md", "census": str(d), "input": a.input, "mask": cfg["masks"],
           "n_events": nev, "it_per_event": len(IT["event"]) / nev, "ot_per_event": len(OT["event"]) / nev,
           "it_owned": float(np.mean(IT["owner"] >= 0)), "ot_owned": float(np.mean(OT["owner"] >= 0)),
           "refit_vs_export_max": float(np.max(IT["d_export"])), "kappa_sign": sign, "old": OLD, "scan": {}}
    for pname, pars in PARSETS.items():
        for cut in CUTS:
            ia, ib = match(IT, OT, pars, cut, sign)
            right = (IT["owner"][ia] >= 0) & (IT["owner"][ia] == OT["owner"][ib])
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
            res["default"] = {"match": _frac(m, np.ones(len(m), bool), ev, nev, rng), "right": _frac(r, m, ev, nev, rng)}
            for lab, x, edges in (("pt", pt, PT_BINS), ("abs_eta", aeta, ETA_BINS)):
                res["default"][lab] = []
                for lo, hi in zip(edges[:-1], edges[1:]):
                    b = (x >= lo) & (x < hi)
                    res["default"][lab].append({"lo": lo, "hi": None if np.isinf(hi) else hi,
                                                "match": _frac(m & b, b, ev, nev, rng),
                                                "right": _frac(r & b, m & b, ev, nev, rng)})
            # per particle: owners of both an IT mini-track and an OT track, matched to each other
            both = np.intersect1d(key(ev[IT["owner"] >= 0], IT["owner"][IT["owner"] >= 0]),
                                  key(OT["event"][OT["owner"] >= 0], OT["owner"][OT["owner"] >= 0]))
            got = np.unique(key(ev[ia][right], IT["owner"][ia][right]))
            res["default"]["per_particle"] = {"particles": int(len(both)),
                                              "matched_right": float(np.isin(both, got).mean()) if len(both) else None}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    dd = res["default"]
    print(f"{cfg['masks']} {nev} events: IT {res['it_per_event']:.1f}/ev ({100 * res['it_owned']:.1f} % owned), OT "
          f"{res['ot_per_event']:.1f}/ev; matched {100 * dd['match']['value']:.1f} +- {100 * dd['match']['sd']:.1f} %, "
          f"right {100 * dd['right']['value']:.1f} +- {100 * dd['right']['sd']:.1f} %; refit vs export "
          f"{res['refit_vs_export_max']:.2g}; kappa sign {sign:+.0f}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-c", "--census", required=True, help="the IT-only census's cache directory")
    ap.add_argument("-i", "--input", required=True, help="the one canon_v5 file the census ran on")
    ap.add_argument("-o", "--out", required=True)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
