"""The real Phase-2 OT tracks, with per-stub truth attached, as the baseline.

WHY THIS EXISTS. Every OT number we quote comes from our own emulation of the
tracklet finder. The nano files also carry the collection CMSSW itself produced,
L1TTrack, so the emulation has an oracle: if our OT-only scenario does not land
close to L1TTrack on efficiency, contamination and resolution, the comparison
against the SmartPixels angles is measuring our reimplementation rather than the
system it is meant to improve on.

WHY THE STUBS HAVE TO BE JOINED. L1TTrack carries per-TRACK truth (genuine /
combinatoric / unknown and a matched tp_*), but no per-STUB truth, so it cannot
say how many of a track's stubs actually belong to the particle that owns it --
which is the axis our whole contamination study is built on. L1TOTStub carries
exactly that, per stub. The two are joined on the FULL stub identity
(event, detId, bend, and x, y, z within a tolerance -- see join_stub_truth).

WHY THE FULL KEY. (event, detId, z) looks sufficient and is not: every stub on a
2S module sits at the module-centre z, so on PU200 48% of OT stubs share that key
with another stub (72-78% per barrel layer). A join that only checks whether SOME
stub matched then passes at ~100% while attaching a neighbour's truth -- it made
"every stub genuine, one TP" read 35% of tracks instead of 89%. With the full key
8 of 1.55 M OT stubs (PU200, 100 events) still share an identity; those are left
UNMATCHED rather than guessed, and the match rate reports them.
"""
from __future__ import annotations
import sys, argparse, json
from pathlib import Path
import numpy as np
import awkward as ak
import uproot
from ngtagger.truth_helix import tp_phi0

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tracklet_topology_cost as M   # noqa: E402
import kf_emulation as KF            # noqa: E402

TRK = ["pt", "eta", "phi", "d0", "z0", "rInv", "tanL", "nStubs", "chi2XYRed",
       "chi2ZRed", "chi2Bend", "trkMVA1", "hwMVAQuality", "genuine", "looselyGenuine",
       "combinatoric", "unknown", "tp_pt", "tp_eta", "tp_phi", "tp_d0",
       "tp_z0", "tp_charge", "tp_vx", "tp_vy", "tp_vz"]
TST = ["layer", "isBarrel", "trackIdx", "detId", "x", "y", "z", "r", "phi", "bend"]
OTS = ["layer", "isBarrel", "detId", "x", "y", "z", "bend", "tpIdx", "tpPt",
       "tpGenuine", "tpCombinatoric", "tpUnknown"]


def _flat(A, pre, cols):
    n = ak.to_numpy(ak.num(A[f"{pre}_{cols[0]}"]))
    D = {c: ak.to_numpy(ak.flatten(A[f"{pre}_{c}"])) for c in cols}
    D["event"] = np.repeat(np.arange(len(n)), n)
    D["_n"] = n
    return D


def load(spec, nev):
    srcs = [f"{p}:Events" for p in M.expand_inputs(spec)]
    keys = ([f"L1TTrack_{c}" for c in TRK] + [f"L1TTrackStub_{c}" for c in TST]
            + [f"L1TOTStub_{c}" for c in OTS])
    A = uproot.concatenate(srcs, keys)
    if nev is not None:
        A = A[:nev]
    return _flat(A, "L1TTrack", TRK), _flat(A, "L1TTrackStub", TST), \
        _flat(A, "L1TOTStub", OTS)


# Two tables, one stub, two float paths: L1TTrackStub and L1TOTStub coordinates
# agree only to 7.6e-6 cm (measured max on PU200; 1.8% of stubs differ at all),
# so equality -- bit-exact or on a rounding grid -- is not a key. The nearest
# DISTINCT stub on a module is a half-strip away (>= 4.5e-3 cm), so a 1e-4 cm
# tolerance sits >10x above the float noise and >40x below the stub spacing.
STUB_MATCH_TOL_CM = 1e-4


def join_stub_truth(S, O):
    """Attach (tpIdx, flag) to every track stub by its full identity.

    A track stub matches the ONE OT stub in the same (event, detId) with the same
    bend and x, y, z each within STUB_MATCH_TOL_CM. Zero candidates, or more than
    one, leaves it UNMATCHED (flag 0, tpIdx -1): a shared identity is reported,
    never resolved by picking one. See the module docstring for why z alone fails.

    Returns (tpidx, flag, matched) aligned to S. flag is 1 genuine /
    2 combinatoric / 3 unknown, matching tp_findability._unify's code so the
    counters mean the same thing on both sides.
    """
    mkey = lambda D: D["event"].astype(np.int64) * (1 << 32) + D["detId"].astype(np.int64)
    ko, ks = mkey(O), mkey(S)
    order = np.argsort(ko, kind="stable")
    lo = np.searchsorted(ko[order], ks, "left")
    n = np.searchsorted(ko[order], ks, "right") - lo      # OT stubs on the module
    si = np.repeat(np.arange(len(ks)), n)                  # (track stub, OT stub) pairs
    oi = order[np.repeat(lo, n) + np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)]
    same = np.asarray(S["bend"])[si] == np.asarray(O["bend"])[oi]
    for c in ("x", "y", "z"):
        same &= np.abs(np.asarray(S[c], np.float64)[si]
                       - np.asarray(O[c], np.float64)[oi]) < STUB_MATCH_TOL_CM
    ncand = np.bincount(si[same], minlength=len(ks))
    src = np.full(len(ks), -1, np.int64)
    src[si[same]] = oi[same]
    hit = ncand == 1
    src = np.where(hit, src, 0)
    tpidx = np.where(hit, O["tpIdx"][src], -1)
    flag = np.where(O["tpGenuine"][src] > 0, 1,
                    np.where(O["tpCombinatoric"][src] > 0, 2, 3))
    return tpidx, np.where(hit, flag, 0), hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-n", "--nev", type=int, default=None)
    ap.add_argument("--ptmin", type=float, default=2.0)
    ap.add_argument("--eta-max", type=float, default=M.ETA_MATCHED)
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()

    T, S, O = load(a.input, a.nev)
    nev = len(T["_n"])
    print(f"{nev} events: {len(T['pt']):,} L1TTrack, {len(S['layer']):,} "
          f"track stubs, {len(O['layer']):,} OT stubs")

    tpidx, flag, hit = join_stub_truth(S, O)
    bar = S["isBarrel"] > 0
    print(f"\n=== the join ===")
    print(f"  track stubs matched to an OT stub: {100 * hit.mean():.2f}% overall, "
          f"{100 * hit[bar].mean():.2f}% of barrel stubs")
    if hit[bar].mean() < 0.999:
        print("  WARNING: the barrel join is not complete; every number below "
              "is conditional on the matched subset")
    print(f"  per-stub truth on matched barrel stubs: "
          f"{100 * (flag[bar & hit] == 1).mean():.1f}% genuine, "
          f"{100 * (flag[bar & hit] == 2).mean():.1f}% combinatoric, "
          f"{100 * (flag[bar & hit] == 3).mean():.1f}% unknown")

    # ---- per-track composition, owner-referenced ------------------------
    off = np.concatenate([[0], np.cumsum(T["_n"])])
    gid = off[S["event"]] + S["trackIdx"]          # global track row per stub
    ok = (S["trackIdx"] >= 0) & bar
    ntrk = len(T["pt"])
    nst = np.zeros(ntrk, np.int32)
    np.add.at(nst, gid[ok], 1)

    # majority owner per track over its GENUINE stubs
    gen = ok & (flag == 1) & (tpidx >= 0)
    owner = np.full(ntrk, -1, np.int64)
    n_own = np.zeros(ntrk, np.int32)
    order = np.lexsort((tpidx[gen], gid[gen]))
    g_s, t_s = gid[gen][order], tpidx[gen][order]
    if len(g_s):
        new = np.r_[True, (g_s[1:] != g_s[:-1]) | (t_s[1:] != t_s[:-1])]
        starts = np.flatnonzero(new)
        cnt = np.diff(np.r_[starts, len(g_s)])
        gg, tt = g_s[starts], t_s[starts]
        best = np.zeros(ntrk, np.int32)
        for g_, t_, c_ in zip(gg, tt, cnt):       # few candidates per track
            if c_ > best[g_]:
                best[g_], owner[g_], n_own[g_] = c_, t_, c_
    right = gen & (tpidx == owner[gid])
    cnt_r = np.zeros(ntrk, np.int32); np.add.at(cnt_r, gid[right], 1)
    cnt_w = np.zeros(ntrk, np.int32); np.add.at(cnt_w, gid[gen & ~right], 1)
    cnt_c = np.zeros(ntrk, np.int32); np.add.at(cnt_c, gid[ok & (flag == 2)], 1)
    cnt_u = np.zeros(ntrk, np.int32); np.add.at(cnt_u, gid[ok & (flag == 3)], 1)

    sel = (T["pt"] >= a.ptmin) & (np.abs(T["eta"]) <= a.eta_max) & (nst >= 4)
    print(f"\n=== {int(sel.sum()):,} real barrel tracks, pT >= {a.ptmin}, "
          f"|eta| <= {a.eta_max}, >= 4 barrel stubs ===")
    tot = nst[sel].sum()
    for lab, v in (("n_ot_right", cnt_r), ("n_ot_wrong", cnt_w),
                   ("n_ot_combinatoric", cnt_c), ("n_ot_unknown", cnt_u)):
        print(f"  {lab:<20s} {v[sel].sum():8d}  {100 * v[sel].sum() / max(tot, 1):5.1f}% "
              f"of stubs   mean/track {v[sel].mean():.3f}")
    nw = cnt_w + cnt_c + cnt_u
    print(f"  clean tracks (0 wrong of any kind): {100 * (nw[sel] == 0).mean():.1f}%")
    for k in range(4):
        m = (nw[sel] == k) if k < 3 else (nw[sel] >= 3)
        print(f"    {'>=3' if k == 3 else k} wrong: {100 * m.mean():5.1f}%")
    print(f"  CMS per-track flags on the same set: "
          f"{100 * (T['genuine'][sel] > 0).mean():.1f}% genuine, "
          f"{100 * (T['combinatoric'][sel] > 0).mean():.1f}% combinatoric, "
          f"{100 * (T['unknown'][sel] > 0).mean():.1f}% unknown")

    # ---- residuals, and the d0 SIGN CONVENTION --------------------------
    print(f"\n=== residuals against the track's own matched particle ===")
    fin = sel & np.isfinite(T["tp_d0"]) & (T["tp_pt"] > 0)
    def rms(x):
        x = x[np.isfinite(x)]
        return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) > 20 else np.nan
    d0f, d0t = T["d0"][fin], T["tp_d0"][fin]
    # THE SIGN TEST HAS TO BE DONE WHERE d0 IS RESOLVED. sigma(d0) here is
    # ~400 um and the prompt d0 spread is the same size, so comparing
    # sigma(d0 - tp_d0) against sigma(d0 + tp_d0) over all tracks is a
    # near-tie whichever way the convention runs. Restricting to tracks whose
    # TRUE |d0| is several sigma out makes it unambiguous.
    print(f"  d0  SIGN, all tracks:  corr = {np.corrcoef(d0f, d0t)[0, 1]:+.4f}"
          f"   sigma(f-t) = {1e4*rms(d0f - d0t):6.1f} um"
          f"   sigma(f+t) = {1e4*rms(d0f + d0t):6.1f} um")
    for lo in (0.05, 0.1, 0.2):
        big = np.abs(d0t) > lo
        if big.sum() < 50:
            continue
        sl = np.nanmedian(d0f[big] / d0t[big])
        print(f"      |tp_d0| > {1e4*lo:5.0f} um (n={int(big.sum()):5d}): "
              f"corr = {np.corrcoef(d0f[big], d0t[big])[0, 1]:+.4f}  "
              f"robust slope = {sl:+.3f}  "
              f"sigma(f-t) = {1e4*rms(d0f[big] - d0t[big]):6.1f} um  "
              f"sigma(f+t) = {1e4*rms(d0f[big] + d0t[big]):6.1f} um")
    big = np.abs(d0t) > 0.1
    verdict = ("SAME" if (big.sum() > 50 and np.nanmedian(d0f[big] / d0t[big]) > 0)
               else "OPPOSITE" if big.sum() > 50 else "UNDETERMINED")
    print(f"      -> L1TTrack_d0 uses the {verdict} sign convention as tp_d0")
    for lab, f_, t_, sc, u in (("z0", T["z0"], T["tp_z0"], 1e4, "um"),
                               ("eta", T["eta"], T["tp_eta"], 1.0, ""),
                               ("phi", T["phi"], tp_phi0(T), 1e3, "mrad")):
        r = f_[fin] - t_[fin]
        if lab == "phi":
            r = np.arctan2(np.sin(r), np.cos(r))
        print(f"  {lab:<4s} sigma = {sc*rms(r):8.3f} {u:<4s} median = {sc*np.median(r[np.isfinite(r)]):+8.3f}")
    rel = (T["pt"][fin] - T["tp_pt"][fin]) / T["tp_pt"][fin]
    print(f"  pT   sigma = {100*rms(rel):8.2f} %    median = {100*np.median(rel):+8.2f} %")
    # READ THIS COLUMN WITH THE CONFOUNDER IN MIND: a track with more stubs has
    # more chances to pick up a wrong one AND is better measured, so a naive
    # split on wrong-hit count mixes contamination with track length and pT.
    # nhit and pT are printed alongside so the confounding is visible rather
    # than being silently absorbed into a "more wrong hits is better" claim.
    print(f"  sigma(d0) by wrong-hit count (with the length/pT confounder shown):")
    for k in range(4):
        m = fin & ((nw == k) if k < 3 else (nw >= 3))
        if m.sum() < 20:
            continue
        print(f"    {'>=3' if k == 3 else k} wrong: n={int(m.sum()):6d}  "
              f"sigma(d0) = {1e4*rms(T['d0'][m] - T['tp_d0'][m]):7.1f} um  "
              f"sigma(z0) = {1e4*rms(T['z0'][m] - T['tp_z0'][m]):7.1f} um  "
              f"mean nstub = {nst[m].mean():4.2f}  median pT = {np.median(T['pt'][m]):5.2f}")

    if a.out:
        json.dump({"n_events": nev, "ptmin": a.ptmin, "eta_max": a.eta_max,
                   "join_barrel_frac": float(hit[bar].mean()),
                   "n_tracks_selected": int(sel.sum()),
                   "stub_frac": {"right": float(cnt_r[sel].sum() / max(tot, 1)),
                                 "wrong": float(cnt_w[sel].sum() / max(tot, 1)),
                                 "combinatoric": float(cnt_c[sel].sum() / max(tot, 1)),
                                 "unknown": float(cnt_u[sel].sum() / max(tot, 1))},
                   "clean_frac": float((nw[sel] == 0).mean()),
                   "sigma_d0_um": float(1e4 * rms(d0f - d0t)),
                   "sigma_z0_um": float(1e4 * rms(T["z0"][fin] - T["tp_z0"][fin])),
                   }, open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
