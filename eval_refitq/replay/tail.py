"""C1a item 1: localise where the forced replay departs from CMSSW (PLAN §1 item 1, §10 PD5).

On canon_v3 file 4 (the prediction file), for AAAA / AAAI / IAAA, the forced replay (engine.forced)
is run in the variants below and compared crossing by crossing with what the producer recorded:
the OT crossing and cone at the OT state (no drift), the running projection and width (C after
Q, before the update) and the per-scalar pulls. Per track, the first crossing whose prediction,
width or pull departs from the recorded value (PD5 (d) tolerances) is its first divergence.

  variant           hypothesis
  f64               ORP's float64 arithmetic (the baseline the tail was registered on)
  f32               H4: CMSSW's float types
  f32fma            H4: float types with the build's fma contraction (PD5 (a)); the best emulation
  f32fma-cluster    H3: the cluster table's storage precision (PM's input)
  f32fma-ncross+1/-1, f32fma-kpt   H5: the Q bookkeeping (nCross, the field constant)
  f32fma-norelin    the attribution known-bad (no relinearisation term): must read "update"
  ensemble          H1 (PD6 (b), PD7): 50 arithmetic-noise realisations of f32fma (the OT-state residuals
                    propagated), AAAA, the diverging tracks; the distribution-free rank rule

H6 (the float32 input state) is read off the OT crossing at the first-visited surface, which is
computed from the nano's float32 track parameters: bit-identity there means H6 is negligible.

Outputs (WPC_OUT/tail): tail_summary.json; tail_<scen>.npz, the per-crossing offsets at PM's input
precision (f32fma-cluster) with the earlier-max-pull bin, for E4 and E5.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import common as CM
import engine as E
from common import OP

VARIANTS = {
    "f64": E.ForcedOpts(),
    "f32": E.ForcedOpts(crossing="f32"),
    "f32fma": E.ForcedOpts(crossing="f32fma"),
    "f32fma-cluster": E.ForcedOpts(crossing="f32fma", meas="cluster"),
    "f32fma-ncross+1": E.ForcedOpts(crossing="f32fma", ncross_offset=1),
    "f32fma-ncross-1": E.ForcedOpts(crossing="f32fma", ncross_offset=-1),
    "f32fma-kpt": E.ForcedOpts(crossing="f32fma", kpt="cmssw"),
    "f32fma-norelin": E.ForcedOpts(crossing="f32fma", relin=False),
}
BEST, PM_INPUT, KNOWN_BAD = "f32fma", "f32fma-cluster", "f32fma-norelin"
PULL_BINS = (0.0, 3.0, 5.0, 10.0, 30.0, np.inf)
SURFACES = (4, 3, 2, 1, 14, 13, 12, 11)
NAME = {s: (f"ID{s - 10}" if s > 10 else f"IL{s}") for s in SURFACES}
TOL_WIDTH, TOL_PULL = 1e-6, 1e-5           # PD5 (d); positions: bit-identical float32
ANGLE_Q = 2.5e-4                           # PD6 (a): the 12-bit quanta of an angle measurement and its sigma
N_ENSEMBLE = 50
PULLS = ("pullX", "pullY", "pullAlpha", "pullBeta")


def rec_cols(H):
    """The producer's recorded per-crossing values, -999 / 0 sentinels as NaN."""
    def v(c, lo=-900.0):
        x = H[c].astype(np.float64)
        return np.where(x > lo, x, np.nan)
    out = {k: v(c) for k, c in (("u", "projLocalX"), ("v", "projLocalY"), ("cotA", "projCotAlpha"),
                                 ("cotB", "projCotBeta"), ("u_ot", "projOTLocalX"), ("v_ot", "projOTLocalY"))}
    for k, c in (("su", "projSigX"), ("sv", "projSigY"), ("su_ot", "projOTSigX"), ("sv_ot", "projOTSigY")):
        x = H[c].astype(np.float64)
        out[k] = np.where(x > 0, x, np.nan)
    out["pull"] = np.stack([v(c) for c in PULLS], 1)
    m_over_s = [np.abs(v(m) / v(sg)) for m, sg in (("recoCotAlpha", "sigAlpha"), ("recoCotBeta", "sigBeta"))]
    out["pull_tol"] = np.stack([np.full(len(H["layer"]), TOL_PULL)] * 2 +
                               [ANGLE_Q * (np.nan_to_num(ms) + np.abs(np.nan_to_num(out["pull"][:, 2 + i])))
                                + TOL_PULL for i, ms in enumerate(m_over_s)], 1)
    out["f32"] = {k: H[c] for k, c in (("u", "projLocalX"), ("v", "projLocalY"),
                                        ("u_ot", "projOTLocalX"), ("v_ot", "projOTLocalY"))}
    return out


def earlier_max_pull(track, pull_x, pull_y, accepted):
    """Per row: the largest |position pull| CMSSW recorded on the track's EARLIER updates (rows are
    grouped by track in visit order)."""
    p = np.where(accepted, np.nanmax(np.abs(np.stack([pull_x, pull_y], 1)), 1), 0.0)
    p = np.nan_to_num(p)
    out = np.zeros(len(track))
    first = np.r_[True, track[1:] != track[:-1]] if len(track) else np.zeros(0, bool)
    cur = 0.0
    for i in range(len(track)):
        if first[i]:
            cur = 0.0
        out[i] = cur
        cur = max(cur, p[i])
    return out


def compare(rows, rec):
    """Per-row comparison of a forced replay with the record (PD5 (d))."""
    c = {}
    for k in ("u", "v", "cotA", "cotB", "u_ot", "v_ot"):
        c["d" + k] = rows[k] - rec[k]
    for k in ("u", "v", "u_ot", "v_ot"):
        have = np.isfinite(rows[k]) & np.isfinite(rec[k])
        c["exact_" + k] = np.where(have, rows[k].astype(np.float32) == rec["f32"][k], False)
        c["have_" + k] = have
    for k in ("su", "sv", "su_ot", "sv_ot"):
        c["w" + k] = rows[k] / rec[k] - 1.0
    c["dpull"] = rows["pull"] - rec["pull"]
    c["pull_tol"] = rec["pull_tol"]
    return c


def divergence(rows, c):
    """Per row: 3 prediction (state), 2 width (covariance), 1 pull (update), 0 none, -1 no record."""
    have = c["have_u"] & c["have_v"]
    state = have & ~(c["exact_u"] & c["exact_v"])
    width = (np.abs(np.nan_to_num(c["wsu"])) > TOL_WIDTH) | (np.abs(np.nan_to_num(c["wsv"])) > TOL_WIDTH)
    pull = (np.abs(np.nan_to_num(c["dpull"])) > c["pull_tol"]).any(1)
    code = np.where(state, 3, np.where(width, 2, np.where(pull, 1, 0)))
    return np.where(have, code, -1)


def first_divergence(track, code):
    """Per track: (index of the first row with code > 0 or -1, its code)."""
    ntr = int(track.max()) + 1 if len(track) else 0
    first_row = np.full(ntr, -1, np.int64)
    idx = np.flatnonzero(code > 0)
    if len(idx):
        t = track[idx]
        u, i = np.unique(t, return_index=True)
        first_row[u] = idx[i]
    return first_row


def q(x, ps=(50, 90, 99, 99.9)):
    x = np.abs(np.asarray(x, np.float64))
    x = x[np.isfinite(x)]
    return {f"p{p:g}": (float(np.percentile(x, p)) if len(x) else None) for p in ps} | {"n": int(len(x))}


def summarise(rows, rec, c, code, track, emp, accepted):
    """Per-surface and per-pull-bin summary of one variant."""
    out = {"per_surface": {}, "per_pull_bin": {}}
    surf = rows["surface"]
    for S in SURFACES:
        m = surf == S
        if not m.any():
            continue
        e = {"n": int(m.sum())}
        for k in ("u", "v", "u_ot", "v_ot"):
            h = m & c["have_" + k]
            e[f"exact_{k}"] = float(c["exact_" + k][h].mean()) if h.any() else None
        e["du_um"] = q(c["du"][m] * 1e4)
        e["dv_um"] = q(c["dv"][m] * 1e4)
        e["du_sig"] = q(c["du"][m] / rec["su"][m])
        e["dv_sig"] = q(c["dv"][m] / rec["sv"][m])
        e["dcotA"] = q(c["dcotA"][m])
        e["dcotB"] = q(c["dcotB"][m])
        e["width_u"] = q(c["wsu"][m])
        e["width_v"] = q(c["wsv"][m])
        e["width_u_ot"] = q(c["wsu_ot"][m])
        e["width_v_ot"] = q(c["wsv_ot"][m])
        for k, nm in enumerate(PULLS):
            e["d" + nm] = q(c["dpull"][m, k])
        e["code_frac"] = {n: float(np.mean(code[m] == v)) for n, v in
                          (("none", 0), ("pull", 1), ("width", 2), ("state", 3), ("no_record", -1))}
        out["per_surface"][NAME[S]] = e
    later = np.r_[False, track[1:] == track[:-1]] if len(track) else np.zeros(0, bool)
    for lo, hi in zip(PULL_BINS[:-1], PULL_BINS[1:]):
        m = later & (emp >= lo) & (emp < hi)
        out["per_pull_bin"][f"{lo:g}-{hi:g}"] = {
            "frac_of_later": float(m.sum() / max(later.sum(), 1)),
            "du_sig": q(c["du"][m] / rec["su"][m]), "dv_sig": q(c["dv"][m] / rec["sv"][m])}
    fr = first_divergence(track, code)
    hit = fr >= 0
    tracks_with_rows = np.unique(track)
    out["tracks"] = int(len(tracks_with_rows))
    out["tracks_diverging"] = int(hit.sum())
    out["first_divergence"] = {}
    for nm, v in (("pull", 1), ("width", 2), ("state", 3)):
        sel = fr[hit][code[fr[hit]] == v]
        out["first_divergence"][nm] = {
            "n": int(len(sel)),
            "by_surface": {NAME[S]: int((surf[sel] == S).sum()) for S in SURFACES},
            "by_step": {int(s): int((rows["step"][sel] == s).sum()) for s in np.unique(rows["step"][sel])}}
    return out


def run_file(scen, nev, chunk, ensemble):
    """The variants (ensemble False) or the H1 ensemble (ensemble True) on file 4 of one scenario."""
    path = CM.input_path(scen, CM.PRED_FILE)
    cfg = CM.producer_config(path)
    mask = CM.mask_of(cfg)
    assert mask == scen, (mask, scen)
    geo = OP.load_geometry(CM.GEOMETRY)
    parts = {v: [] for v in VARIANTS}
    ens = []
    meta = {"n_d0_floor": 0, "off_path": 0, "n_events": 0}
    toff = 0
    for start, A in CM.iterate(path, E.branches(path), chunk=chunk, nev=nev):
        ctx = OP.prepare_chunk(A, geo)
        H = E.hit_table(A)
        rec = rec_cols(H)
        acc = (H["hitAccepted"] > 0) & (H["selClusterIdx"] >= 0)
        res = {}
        for name, opts in (VARIANTS.items() if not ensemble else ((BEST, VARIANTS[BEST]), (KNOWN_BAD, VARIANTS[KNOWN_BAD]))):
            R = E.forced(ctx, H, geo, cfg, mask, opts)
            rows = R["rows"]
            rows["surface"] = H["layer"].astype(np.int64)
            res[name] = rows
            if name == "f64":
                meta["n_d0_floor"] += R["n_d0_floor"]
                meta["off_path"] += R["off_path"]
            keep = {k: rows[k] for k in ("u", "v", "cotA", "cotB", "su", "sv", "u_ot", "v_ot", "su_ot", "sv_ot",
                                         "pull", "step", "surface")}
            keep["track"] = rows["track"] + toff
            keep["accepted"] = acc
            keep["event"] = H["event"] + start
            for k in ("u", "v", "cotA", "cotB", "su", "sv", "u_ot", "v_ot", "su_ot", "sv_ot"):
                keep["rec_" + k] = rec[k]
            for k in ("u", "v", "u_ot", "v_ot"):
                keep["rec32_" + k] = rec["f32"][k]
            keep["rec_pull"] = rec["pull"]
            keep["rec_pull_tol"] = rec["pull_tol"]
            parts[name].append(keep)
        if ensemble:
            c = compare(res[BEST], rec)
            code = divergence(res[BEST], c)
            trk = res[BEST]["track"]
            div_tracks = np.zeros(ctx["ntrk"], bool)
            div_tracks[trk[code > 0]] = True
            noise = noise_pools(c, H["layer"])
            U = np.full((N_ENSEMBLE, len(trk)), np.nan)
            V = np.full((N_ENSEMBLE, len(trk)), np.nan)
            for i in range(N_ENSEMBLE):
                opts = E.ForcedOpts(crossing="f32fma", noise_seed=1000 * (start + 1) + i)
                Ri = E.forced(ctx, H, geo, cfg, mask, opts, track_mask=div_tracks, noise=noise)["rows"]
                U[i], V[i] = Ri["u"], Ri["v"]
            m = div_tracks[trk]
            ens.append({"U": U[:, m].T, "V": V[:, m].T, "rec_u": rec["u"][m], "rec_v": rec["v"][m],
                        "kb_u": res[KNOWN_BAD]["u"][m], "kb_v": res[KNOWN_BAD]["v"][m],
                        "code": code[m], "surface": H["layer"][m].astype(np.int64), "step": res[BEST]["step"][m]})
        toff += ctx["ntrk"]
        meta["n_events"] += len(A)
    if ensemble:
        E_ = CM.concat(ens)
        E_["U"], E_["V"] = E_["U"].T, E_["V"].T                  # (realisation, row)
        return cfg, meta, None, E_
    return cfg, meta, {v: CM.concat(p) for v, p in parts.items() if p}, None


def noise_pools(c, surface):
    """PD6 (b): the f32fma OT-state residuals of the chunk (prediction u, v; cone widths), per
    surface class, exact zeros included."""
    disc = np.asarray(surface) > 10
    out = {}
    for k, src in (("du", "du_ot"), ("dv", "dv_ot"), ("wu", "wsu_ot"), ("wv", "wsv_ot")):
        x = np.asarray(c[src], np.float64)
        out[k] = {"barrel": x[~disc & np.isfinite(x)], "disc": x[disc & np.isfinite(x)]}
    return out


def ensemble_test(E_):
    """H1 (PD6 (b), PD7): is the recorded prediction inside the arithmetic-noise ensemble? Rows whose
    record diverges under f32fma (code > 0), after the first-visited surface (step > 0). A row is
    outside when the target lies strictly outside [min, max] of the ensemble; exchangeable draws are
    outside with probability 2 / (n_ensemble + 1)."""
    out = {}

    def frac_outside(target_u, target_v, Uu, Vv, rows):
        res = {}
        p0 = 2.0 / (len(Uu) + 1)
        for nm, tgt, X in (("u", target_u, Uu), ("v", target_v, Vv)):
            lo, hi = np.nanmin(X, 0), np.nanmax(X, 0)
            ok = rows & np.isfinite(tgt) & np.isfinite(lo)
            outside = (tgt[ok] < lo[ok]) | (tgt[ok] > hi[ok])
            n = int(ok.sum())
            f = float(outside.mean()) if n else None
            lim = p0 + 3.0 * np.sqrt(p0 * (1 - p0) / max(n, 1))
            res[nm] = {"n": n, "frac_outside": f, "expected": p0, "limit": float(lim),
                       "pass": bool(f is not None and f <= lim)}
        res["pass"] = bool(res["u"]["pass"] and res["v"]["pass"])
        return res

    div = (E_["code"] > 0) & (E_["step"] > 0)
    out["record"] = frac_outside(E_["rec_u"], E_["rec_v"], E_["U"], E_["V"], div)
    out["known_good_realisation0_vs_rest"] = frac_outside(E_["U"][0], E_["V"][0], E_["U"][1:], E_["V"][1:], div)
    out["known_bad_norelin"] = frac_outside(E_["kb_u"], E_["kb_v"], E_["U"], E_["V"], div)
    out["per_surface_record"] = {NAME[S]: frac_outside(E_["rec_u"], E_["rec_v"], E_["U"], E_["V"],
                                                       div & (E_["surface"] == S)) for S in SURFACES
                                 if (div & (E_["surface"] == S)).any()}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scen", nargs="+", default=list(CM.SCENARIOS))
    ap.add_argument("--nev", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--stage", choices=("variants", "ensemble"), default="variants")
    ap.add_argument("--out", default=str(CM.OUT / "tail"))
    a = ap.parse_args(argv)
    import os
    os.makedirs(a.out, exist_ok=True)
    spath = os.path.join(a.out, "tail_summary.json")
    if a.stage == "ensemble":
        with open(spath) as fh:
            summary = json.load(fh)
        for scen in a.scen:
            t0 = time.time()
            cfg, meta, _, ens = run_file(scen, a.nev, a.chunk, ensemble=True)
            summary["scenarios"].setdefault(scen, {})["ensemble_H1"] = {**ensemble_test(ens), "meta": meta,
                                                                         "seconds": round(time.time() - t0, 1)}
            print(f"{scen} ensemble: {meta['n_events']} events, {time.time() - t0:.0f} s", flush=True)
            with open(spath, "w") as fh:
                json.dump(summary, fh, indent=1, default=float)
        return 0
    summary = {"plan": "PLAN.md §1 item 1, §10 PD5-PD7, PD9", "file": CM.PRED_FILE, "variants":
               {k: v.label() for k, v in VARIANTS.items()}, "tolerances": {"positions": "bit-identical float32",
                                                                        "width_rel": TOL_WIDTH, "pull_abs": TOL_PULL},
               "ensemble": {"n": N_ENSEMBLE, "rule": "PD7 rank rule"}, "scenarios": {}}
    for scen in a.scen:
        t0 = time.time()
        cfg, meta, data, _ = run_file(scen, a.nev, a.chunk, ensemble=False)
        S = {"meta": meta, "producer": {k: cfg[k] for k in ("label", "useAngles", "maxKFUpdates",
                                                              "maxHitsPerWindow", "layerOrder")}, "variants": {}}
        for name, d in data.items():
            rows = {k: d[k] for k in ("u", "v", "cotA", "cotB", "su", "sv", "u_ot", "v_ot", "su_ot", "sv_ot",
                                      "pull", "step", "surface")}
            rec = {k: d["rec_" + k] for k in ("u", "v", "cotA", "cotB", "su", "sv", "u_ot", "v_ot", "su_ot", "sv_ot",
                                              "pull")}
            rec["f32"] = {k: d["rec32_" + k] for k in ("u", "v", "u_ot", "v_ot")}
            rec["pull_tol"] = d["rec_pull_tol"]
            c = compare(rows, rec)
            code = divergence(rows, c)
            emp = earlier_max_pull(d["track"], rec["pull"][:, 0], rec["pull"][:, 1], d["accepted"])
            S["variants"][name] = summarise(rows, rec, c, code, d["track"], emp, d["accepted"])
            if name == PM_INPUT:
                bins = np.digitize(emp, PULL_BINS[1:-1])
                np.savez_compressed(os.path.join(a.out, f"tail_{scen}.npz"),
                                    surface=d["surface"], step=d["step"], track=d["track"], event=d["event"],
                                    accepted=d["accepted"], du=c["du"], dv=c["dv"], dcotA=c["dcotA"],
                                    dcotB=c["dcotB"], su=rec["su"], sv=rec["sv"], pull_bin=bins,
                                    earlier_max_pull=emp, code=code)
        S["seconds"] = round(time.time() - t0, 1)
        summary["scenarios"][scen] = S
        print(f"{scen}: {meta['n_events']} events, {S['seconds']} s", flush=True)
        with open(spath, "w") as fh:
            json.dump(summary, fh, indent=1, default=float)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
