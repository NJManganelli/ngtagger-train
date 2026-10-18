"""WP-D0b: the D10 census import gate (plan D.0 row "D10 census import"; WDMac wpd0b/PLAN.md sec. 3
and its round-2 amendment B1), and the reports that go with it.

    python import_gate.py keys     KG_CENSUS -o keys.json
    python import_gate.py tolerances CAL_CENSUS KG_CENSUS CUTS -o tol.json
    python import_gate.py judge    CAL_CENSUS TEST_CENSUS CUTS TOL -o gate.json
    python import_gate.py report   G_CENSUS OFF_CENSUS -o endtoend.json

THE CONSUMERS: S1, S2a, S2z, S3 (window, role attach), S4w (window, role follow) and S6 (per build,
pooled over n_meas); with the census's IT discs (WP-Disc, format 36) also the disc consumers S1, S2a, S2z, S4w
(s1d, s2ad, s2zd, s4wd), judged apart, with the offline disc known-bads (census_angles.DISC_KNOWN_BADS), each of
which must turn its own disc consumers red (DISC_KNOWN_BAD_TARGETS). Their records (census_angles.Recorder) carry each correct candidate's statistic
before its own cut; the calibration population is class T, born inside 2.5 cm, the TP above the
stage's floor; every efficiency is particle-level (each TP weighted once per bin, angle_cuts).

  (b-ii)  per bin (the cut's own merged bins; S6 per n): the coverage of the calibration quantile
          against E0 = k / (n_fit + 1), with the beta-binomial null widened to the larger of the exact
          order-statistic variance and the event-bootstrap variance (spdefs.coverage_z); ONE-SIDED
          (amendment of 2026-10-15): a bin fails only for under-coverage, z < -t(N) at the gate's own
          N (every bin of every consumer on that file); over-coverage (z > t(N): a stage keeping more
          than 99 % of what reaches it, the cuts-open calibration's conditioning) is reported, not
          failed. The chi2 over all bins is reported, not judged (over-coverage inflates it).
          MATERIAL when the failing bins hold > 2 % of a consumer's correct particles.
  (b-iii) per consumer (S6 per build pooled over n): the one-sided 95 % upper bound of 0.99 - eff
          within m = 0.5 %; sd from the event bootstrap with the calibration quantiles re-fitted
          inside every replica (250). A cell whose power on KG, Phi(m / sd - 1.645), is below 80 %
          is report-only (amendment B1; the list is fixed by `tolerances`, before G).
  known-bads, each must fail (b-ii) or (b-iii): the per-layer scales at 3 census sigma (gross; S1, S2a,
          S2z, S3, S4w); the cuts fitted at L = 0.98. Report-only (amendment of 2026-10-15; not
          detectable at the cuts-open calibration): S6 with one threshold pooled over n_meas; S3 / S4w
          with two separate 99 % cuts.
  keys    the S4a table key by the paired correct-hit rate of S4a's decisions, the S5 key by the
          paired core resolution of correct tracks (d0, z0, 1/pT), on KG (PLAN.md sec. 1c rules).
  report  the end-to-end particle loss against the all-angles-off census, per build, pT bin and
          genealogy (prompt hard scatter, prompt pile-up, b/c daughter, born beyond IL1).
  dfill   (WP-Disc D-fill) every declared disc entry of the test census fills, and each disc record table's
          records per event lie within 20 % of the known-good's; its known-bads (one disc entry dropped, every
          IT-disc entry dropped) must turn red.

    python import_gate.py dfill    KG_CENSUS TEST_CENSUS -o dfill.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1] / "combinatorics"
sys.path.insert(0, str(HERE))
import angle_cuts as AC              # noqa: E402
import census_angles as CA           # noqa: E402
import tp_findability as TF          # noqa: E402
import tracklet_topology_cost as M   # noqa: E402
from ngtagger.spdefs import angles as SD  # noqa: E402

MARGIN = 0.005
N_REP = 250
SEED = 20261014
LEVEL = CA.LEVEL
CONSUMERS = ("s1", "s2a", "s2z", "s3", "s4w", "s6")
# THE DISC CONSUMERS (WP-Disc, census format 36; wpdisc/PLAN.md sec. 3 D-angle): judged apart from the barrel's,
# each by the same (b-ii) and (b-iii) rules; S3 has no disc target in the menu
DISC_CONSUMERS = ("s1d", "s2ad", "s2zd", "s4wd")
# the offline disc known-bads (census_angles.DISC_KNOWN_BADS) and the disc consumers each must turn red
DISC_KNOWN_BAD_TARGETS = {"facing_flipped": ("s2ad", "s2zd"), "barrel_frame": DISC_CONSUMERS,
                          "barrel_form": ("s2ad", "s2zd"), "global_alpha": ("s2ad",)}
# the record table and the cut-file section of each disc consumer
DISC_TAB = {"s1d": "s1d", "s2ad": "paird", "s2zd": "paird", "s4wd": "wind"}
DISC_BODY = {"s1d": "s1d", "s2ad": "s2ad", "s2zd": "s2zd", "s4wd": "wind"}
MATERIAL_BIN_SHARE = 0.02
POWER_MIN = 0.80
# the known-bads that cannot show at the cuts-open calibration (amendment of 2026-10-15): a cut
# run's S6 has n = 6 or 8 only and the pooled cut falls between those per-n thresholds; S3 / S4w
# keep 99.5 % of what reaches them. Reported, not required red.
KNOWN_BAD_REPORT_ONLY = ("s6_pooled_over_n", "s34_two_99_cuts")


def s6_groups(T):
    """S6's group per track record (census_angles.s6_group: the build, its disc tracks apart with the IT discs)."""
    return T["s6g"] if "s6g" in T else T["build"]


# ==========================================================================
# loading
# ==========================================================================
def load(census):
    d = TF.census_dir(census)
    man = json.load(open(os.path.join(d, "manifest.json")))
    return d, man, CA.load_records(d)


def run_config(man):
    """The run's S5 state and key, its S6 and S3 / S4w angle subsets."""
    ang = M.AngleUse.from_key(man["config"]["angles"])
    s5 = ang.subset("kf") if ang("kf") else "off"
    return {"s5": s5, "key": ang.keys["kf"] if s5 != "off" else "emitted",
            "sub6": ang.subset("accept") if ang("accept") else "ab",
            "win": ang.subset("attach") if ang("attach") else "ab"}


def model_of(man):
    am = man["config"]["angle_model"]
    return CA.Model.from_identity(am)


def n_events(man):
    return int(man["n_events"])


# ==========================================================================
# the test side: per consumer, the records' bins, weights and pass under a cut body
# ==========================================================================
class Test:
    """The consumers' test records of one census, their statistics (computed once) and their bins
    under a cut body; pass() re-thresholds for any body (the bootstrap)."""

    def __init__(self, R, man, cfg, model=None):
        self.R, self.cfg, self.model = R, cfg, model
        self.S = AC.statistics(R, model=model)
        self.ev = {}
        P = R["pair"]
        W = R["win"]
        T = R["trk"]
        self.tab = {"s1": R["s1"], "s2a": P, "s2z": P, "s3": W, "s4w": W, "s6": T}
        self.pop = {"s1": AC.population(R["s1"], "s1"), "s2a": AC.population(P, "pair"),
                    "s2z": AC.population(P, "pair") & np.isfinite(self.S["pair"][1]),
                    "s3": AC.population(W, "win") & (W["role"] == 0),
                    "s4w": AC.population(W, "win") & (W["role"] == 1),
                    "s6": AC.population(T, "trk")}
        self.cpop = {"s1": AC.class_c(R["s1"], "s1"), "s2a": AC.class_c(P, "pair"),
                     "s2z": AC.class_c(P, "pair") & np.isfinite(self.S["pair"][1]),
                     "s3": AC.class_c(W, "win") & (W["role"] == 0),
                     "s4w": AC.class_c(W, "win") & (W["role"] == 1),
                     "s6": AC.class_c(T, "trk")}
        self.c2, self.n6 = AC.s6_parts(T, cfg["s5"], cfg["key"], cfg["sub6"])
        # the disc consumers, when the census recorded them (its IT discs on)
        self.disc = model is not None and all(k in R for k in ("s1d", "paird", "wind"))
        self.consumers = CONSUMERS + (DISC_CONSUMERS if self.disc else ())
        if self.disc:
            for c, kind in DISC_TAB.items():
                Td = R[kind]
                self.tab[c] = Td
                self.pop[c] = AC.population(Td, kind) & ((Td["role"] == 1) if c == "s4wd" else True)
                self.cpop[c] = AC.class_c(Td, kind) & ((Td["role"] == 1) if c == "s4wd" else True)

    def disc_stats(self, model):
        """The disc consumers' statistics of this test's records under `model` (a known-bad's)."""
        return {k: fn(self.R[k], model, *[model.k[a] for a in CA.ANGLES], model.mode)
                for k, fn in (("s1d", AC.stat_s1d), ("paird", AC.stat_paird), ("wind", AC.stat_wind))}

    def bins(self, body):
        """Per consumer: the bin (unit) index of every record in `body` (-1 when none)."""
        R = self.R
        u = {}
        u["s1"] = AC._unit_index([tuple(x) for x in body["s1"]["bins"]],
                                 list(zip(R["s1"]["floor"].tolist(), R["s1"]["row"].tolist())))
        P = R["pair"]
        u["s2a"] = AC._unit_index([tuple(x) for x in body["s2a"]["bins"]],
                                  list(zip(P["floor"].tolist(), P["la"].tolist(), P["lb"].tolist())))
        u["s2z"] = AC._unit_index([tuple(x) for x in body["s2z"]["bins"]],
                                  list(zip(P["floor"].tolist(), P["la"].tolist(), P["lb"].tolist(),
                                           P["A_row"].tolist())))
        W = R["win"]
        uw = AC._unit_index([tuple(x) for x in body["win"]["bins"]],
                            list(zip(W["role"].tolist(), W["seed"].tolist(), W["lc"].tolist(),
                                     W["ptbin"].tolist(), W["floor"].tolist())))
        u["s3"] = u["s4w"] = uw
        T = R["trk"]
        c = self.cfg
        u["s6"] = AC._unit_index([tuple(x) for x in body["s6"]["bins"]],
                                 list(zip(s6_groups(T).tolist(), [c["s5"]] * len(self.n6),
                                          [c["key"]] * len(self.n6), [c["sub6"]] * len(self.n6),
                                          self.n6.tolist())))
        if self.disc:
            S1, P, W = R["s1d"], R["paird"], R["wind"]
            u["s1d"] = AC._unit_index([tuple(x) for x in body["s1d"]["bins"]],
                                      list(zip(S1["floor"].tolist(), S1["row"].tolist())))
            u["s2ad"] = AC._unit_index([tuple(x) for x in body["s2ad"]["bins"]],
                                       list(zip(P["floor"].tolist(), P["la"].tolist(), P["lb"].tolist())))
            u["s2zd"] = np.where(np.isin(P["A_lcode"], M.IT_DISC_CODES), AC._unit_index(
                [tuple(x) for x in body["s2zd"]["bins"]],
                list(zip(P["floor"].tolist(), P["la"].tolist(), P["lb"].tolist(), P["A_row"].tolist()))), -1)
            u["s4wd"] = AC._unit_index([tuple(x) for x in body["wind"]["bins"]],
                                       list(zip(W["role"].tolist(), W["seed"].tolist(), W["lc"].tolist(),
                                                W["ptbin"].tolist(), W["floor"].tolist())))
        return u

    def passes(self, body, units, win_thr=None):
        """Per consumer: whether every record passes `body`'s cut (win_thr overrides S3 / S4w's
        (q_alpha, q_beta) per unit)."""
        out = {}
        st = self.S
        for c, s in (("s1", st["s1"]), ("s2a", st["pair"][0]), ("s2z", st["pair"][1])):
            ui = units[c]
            thr = np.asarray(body[c]["thr"], np.float64)
            t = np.where(ui >= 0, thr[np.maximum(ui, 0)], np.inf)
            out[c] = ~np.isfinite(s) | (s <= t) if c == "s2z" else (s <= t)
        q = np.asarray(body["win"]["thr"][self.cfg["win"]] if win_thr is None else win_thr, np.float64)
        ui = units["s3"]
        qa = np.where(ui >= 0, q[np.maximum(ui, 0), 0], np.inf)
        qb = np.where(ui >= 0, q[np.maximum(ui, 0), 1], np.inf)
        pa, pb = st["win"]
        out["s3"] = out["s4w"] = (np.abs(pa) <= qa) & (np.abs(pb) <= qb)
        ui = units["s6"]
        thr = np.asarray(body["s6"]["thr"], np.float64)
        t = np.where(ui >= 0, thr[np.maximum(ui, 0)], np.inf)
        out["s6"] = (self.n6 == 0) | (self.c2 <= t)
        if self.disc:
            out.update(self.disc_passes(body, units))
        return out

    def disc_passes(self, body, units, stats=None):
        """The disc consumers' pass flags under `body`, from this test's disc statistics (or `stats`: a
        known-bad's, disc_stats)."""
        st = self.S if stats is None else stats
        out = {}
        for c, s in (("s1d", st["s1d"][0]), ("s2ad", st["paird"][0]), ("s2zd", st["paird"][1])):
            ui = units[c]
            thr = np.asarray(body[c]["thr"], np.float64)
            out[c] = s <= np.where(ui >= 0, thr[np.maximum(ui, 0)], np.inf)
        q = np.asarray(body["wind"]["thr"][self.cfg["win"]], np.float64)
        ui = units["s4wd"]
        qa = np.where(ui >= 0, q[np.maximum(ui, 0), 0], np.inf)
        qb = np.where(ui >= 0, q[np.maximum(ui, 0), 1], np.inf)
        pa, pb = st["wind"]
        out["s4wd"] = (np.abs(pa) <= qa) & (np.abs(pb) <= qb)
        return out


def _body_key(c):
    """The cut-file section of consumer c."""
    return "win" if c in ("s3", "s4w") else DISC_BODY.get(c, c)


def groups_of(body, c, units):
    """The bin each record is judged in: the cut's merged group (S2a its own pair; S6 per n)."""
    ui = units[c]
    if c in ("s2a", "s2ad", "s6"):
        return ui
    key = _body_key(c)
    g = np.asarray(body[key]["group"])
    return np.where(ui >= 0, g[np.maximum(ui, 0)], -1)


def n_fit_of(body, c):
    """TPs the calibration quantile of each judged bin was fitted on (per group; S6: its n group)."""
    key = _body_key(c)
    n = np.asarray(body[key]["n_tp"], np.int64)
    if c in ("s2a", "s2ad"):
        return n
    g = np.asarray(body[key]["group"])
    tot = {}
    for gi, ni in zip(g, n):
        tot[gi] = tot.get(gi, 0) + ni
    if c == "s6":
        return np.array([tot[gi] for gi in g])          # per bin (build, n): its group's total
    return tot


def efficiencies(test, body, units, passed, ev_w=None):
    """Per consumer: per judged bin (m, n) particle-weighted, and pooled (S6 per build)."""
    out = {}
    for c in test.consumers:
        T = test.tab[c]
        m = test.pop[c] & (units[c] >= 0)
        g = groups_of(body, c, units)[m]
        w = AC.tp_weights(T["tp_key"][m], units[c][m], None if ev_w is None else ev_w[T["event"][m]])
        fail = ~passed[c][m]
        bins = {}
        if len(g):
            gu, inv = np.unique(g, return_inverse=True)
            nw = np.bincount(inv, weights=w)
            mw = np.bincount(inv, weights=w * fail)
            bins = {int(gi): (float(mi), float(ni)) for gi, mi, ni in zip(gu, mw, nw)}
        if c == "s6":
            pooled = {}
            g6 = s6_groups(T)[m]
            for b in np.unique(g6):
                mb = g6 == b
                pooled[str(b)] = 1.0 - float((w * fail)[mb].sum()) / max(float(w[mb].sum()), 1e-300)
        else:
            pooled = {"all": 1.0 - float((w * fail).sum()) / max(float(w.sum()), 1e-300)}
        out[c] = {"bins": bins, "pooled": pooled}
    return out


# ==========================================================================
# the bootstrap: calibration re-fitted inside every replica
# ==========================================================================
def bootstrap(cal_R, cal_man, model, test, body, cfg, n_cal, n_test, n_rep=N_REP, seed=SEED,
              extra=None):
    """Replica efficiencies (per consumer, per bin and pooled) with CAL and test events resampled
    independently and the cut file re-fitted on each CAL replica (its merges fixed). `extra`:
    {name: fn(body_b) -> (body_b', win_thr or None)} alternative cuts evaluated in the same replicas."""
    rng = np.random.default_rng(seed)
    cal_stats = AC.statistics(cal_R, model=model if test.disc else None)
    cells = sorted({(b[0], b[1], b[2], b[3]) for b in body["s6"]["bins"]
                    if (b[1], b[2], b[3]) == (cfg["s5"], cfg["key"], cfg["sub6"])})
    only = {"s1": None, "s2a": None, "s2z": None, "win": [cfg["win"]], "s6": cells}
    if test.disc:
        only.update({"s1d": None, "s2ad": None, "s2zd": None, "wind": [cfg["win"]]})
    units = test.bins(body)
    reps = []
    for r in range(n_rep):
        wc = rng.multinomial(n_cal, np.full(n_cal, 1.0 / n_cal)).astype(float)
        wt = rng.multinomial(n_test, np.full(n_test, 1.0 / n_test)).astype(float)
        bb, _ = AC.fit(cal_R, cal_man, model, ev_w=wc, only=only, stats=cal_stats)
        bb = _with_merges(bb, body)
        rep = {"nominal": efficiencies(test, body, units, test.passes(bb, units), wt)}
        for name, fn in (extra or {}).items():
            b2, wthr = fn(bb, wc)
            rep[name] = efficiencies(test, body, units, test.passes(b2, units, wthr), wt)
        reps.append(rep)
        if (r + 1) % 25 == 0:
            print(f"  bootstrap {r + 1}/{n_rep}", flush=True)
    return reps


def _with_merges(bb, body):
    """A replica body over the nominal bins (the replica's S6 cells only cover the run's
    configuration; every other key keeps the nominal thresholds, unused by the test)."""
    out = dict(body)
    for k, v in bb.items():
        if k == "s6":
            nom = {tuple(b): i for i, b in enumerate(body["s6"]["bins"])}
            thr = list(body["s6"]["thr"])
            for b, t in zip(v["bins"], v["thr"]):
                thr[nom[tuple(b)]] = t
            out["s6"] = {**body["s6"], "thr": thr}
        elif k in ("win", "wind"):
            out[k] = {**body[k], "thr": {**body[k]["thr"], **v["thr"]}}
        else:
            out[k] = {**body[k], "thr": v["thr"]}
    return out


def sd_from(reps, name, c, cell):
    vals = [r[name][c]["pooled"].get(cell, np.nan) for r in reps]
    return float(np.nanstd(vals, ddof=1))


def bin_sd(reps, name, c, g):
    vals = []
    for r in reps:
        mb = r[name][c]["bins"].get(g)
        if mb and mb[1] > 0:
            vals.append(1.0 - mb[0] / mb[1])
    return float(np.std(vals, ddof=1)) if len(vals) > 2 else np.nan


# ==========================================================================
# the rules
# ==========================================================================
def judge_one(name, eff, reps, body, report_only=(), t_n=None):
    """(b-ii) and (b-iii) of one set of cuts (`name` in the replicas) on one test census."""
    from scipy import stats as st
    out = {"b_iii": {}, "b_ii": {}}
    nbins = sum(len(eff[c]["bins"]) for c in eff)
    t = t_n if t_n is not None else _t(nbins)
    out["t"] = t
    out["n_bins"] = nbins
    for c in eff:
        for cell, e in eff[c]["pooled"].items():
            sd = sd_from(reps, name, c, cell)
            ub = (LEVEL - e) + 1.645 * sd
            ro = f"{c}/{cell}" in report_only
            out["b_iii"][f"{c}/{cell}"] = {"eff": e, "sd": sd, "upper_bound_deficit": ub,
                                           "pass": bool(ub <= MARGIN), "report_only": ro,
                                           "power": float(st.norm.cdf(MARGIN / sd - 1.645)) if sd > 0 else 1.0}
        nf = n_fit_of(body, c)
        zs, fails, overs, share_fail, share_over, tot_n = [], [], [], 0.0, 0.0, 0.0
        for g, (m, n) in eff[c]["bins"].items():
            n_fit = int(nf[g]) if isinstance(nf, np.ndarray) else int(nf.get(g, 0))
            if n_fit < 1 or n < 1:
                continue
            e0 = float(SD.expected_coverage(n_fit, LEVEL))
            var_ex = float(SD.exchangeable_coverage_variance(n_fit, max(n, 1.0), LEVEL))
            sdb = bin_sd(reps, name, c, g)
            var = max(var_ex, sdb * sdb if np.isfinite(sdb) else 0.0)
            z = float(SD.coverage_z(round(m), round(n), e0, var))
            zs.append(z)
            tot_n += n
            if np.isfinite(z) and z < -t:
                fails.append({"bin": int(g), "z": z, "coverage": 1.0 - m / n, "n": n, "e0": e0})
                share_fail += n
            elif np.isfinite(z) and z > t:
                overs.append({"bin": int(g), "z": z, "coverage": 1.0 - m / n, "n": n, "e0": e0})
                share_over += n
        chi2 = float(np.nansum(np.square(zs)))
        dof = int(np.isfinite(zs).sum())
        p = float(st.chi2.sf(chi2, dof)) if dof else 1.0
        out["b_ii"][c] = {"n_bins": dof, "chi2": chi2, "p": p,
                          "failing_bins": fails, "failing_share": share_fail / max(tot_n, 1e-300),
                          "over_bins": overs, "over_share": share_over / max(tot_n, 1e-300),
                          "pass": bool(not fails)}
    return out


def _t(n):
    from ngtagger.spcheck.core import t_family
    return t_family(max(n, 1))


def material(res):
    """The MATERIAL findings of a judgement (PLAN.md sec. 4)."""
    mat = []
    for k, v in res["b_iii"].items():
        if not v["pass"] and not v["report_only"]:
            mat.append(f"(b-iii) {k}: efficiency {v['eff']:.4f}, deficit bound {v['upper_bound_deficit']:.4f}")
    for c, v in res["b_ii"].items():
        if v["failing_share"] > MATERIAL_BIN_SHARE:
            mat.append(f"(b-ii) {c}: failing bins hold {100 * v['failing_share']:.1f} % of its particles")
    return mat


def material_consumers(res):
    """The consumers with a MATERIAL finding in a judgement (material())."""
    out = {k.split("/")[0] for k, v in res["b_iii"].items() if not v["pass"] and not v["report_only"]}
    return out | {c for c, v in res["b_ii"].items() if v["failing_share"] > MATERIAL_BIN_SHARE}


def fails_gate(res):
    """Whether a judgement fails the gate (for the known-bads): by the MATERIAL rule itself, so a known-bad
    is red only where the nominal would be MATERIAL -- a non-report-only (b-iii) cell failing, or under-covering
    (b-ii) bins holding > 2 % of a consumer's particles; one failing bin or a report-only cell is not enough."""
    return bool(material(res))


# ==========================================================================
# known-bads
# ==========================================================================
def legacy_passes(test):
    """Today's per-layer scales at 3 census sigma, on the test's records (S1, S2a, S2z, S3, S4w)."""
    R = test.R
    sa = np.array([M.ALPHA_PULL_SCALE.get(L, 1.0) for L in range(5)])
    sb = np.array([M.BETA_PULL_SCALE.get(L, 1.0) for L in range(5)])

    def legacy(T, p=""):
        lay = np.clip(T[p + "layer"].astype(int), 0, 4)
        r, half, cot = T[p + "r"], T[p + "half_e"], T[p + "cot_e"]
        kap = np.sin(half) / (M.C_BEND * r)
        skap = T[p + "sa_pub"] * sa[lay] * np.abs(np.cos(half)) / (M.C_BEND * r)
        scot = T[p + "sb_pub"] * (1 + cot * cot) * sb[lay]
        return kap, skap, cot, scot, T[p + "z"] - r * cot, r * scot
    out = {}
    T = R["s1"]
    kap, skap, *_ = legacy(T)
    out["s1"] = np.abs(kap) <= 1.0 / T["floor"] + M.NSIG * skap
    P = R["pair"]
    kA, sA, _, _, zA, szA = legacy(P, "A_")
    kB, sB, _, _, zB, szB = legacy(P, "B_")
    itA, itB = AC._it(P["A_layer"]), AC._it(P["B_layer"])
    with np.errstate(all="ignore"):
        okA = ~itA | (np.abs(P["kap_pair"] - kA) <= M.NSIG * sA)
        okB = ~itB | (np.abs(P["kap_pair"] - kB) <= M.NSIG * sB)
        out["s2a"] = okA & okB
        out["s2z"] = ~(itA & itB) | (np.abs(zA - zB) <= M.NSIG * np.hypot(szA, szB))
    W = R["win"]
    kap, skap, cot, scot, _, _ = legacy(W)
    with np.errstate(all="ignore"):
        ok = (np.abs((kap - W["kap_pred"]) / skap) <= M.NSIG) & (np.abs((cot - W["cot_pred"]) / scot) <= M.NSIG)
    out["s3"] = out["s4w"] = ok
    out["s6"] = np.ones(len(test.n6), bool)          # S6's per-hit components are not recorded
    return out


def known_bad_bodies(cal_R, cal_man, model, body, cfg):
    """{name: fn(replica body, CAL event weights) -> (body, win_thr)} of the decision-scale
    known-bads (each re-fitted inside the bootstrap as the nominal is)."""
    cal_stats = AC.statistics(cal_R)

    def l098(bb, wc):
        b2, _ = AC.fit(cal_R, cal_man, model, level=0.98, ev_w=wc,
                       only={"s1": None, "s2a": None, "s2z": None, "win": [cfg["win"]],
                             "s6": sorted({(b[0], b[1], b[2], b[3]) for b in body["s6"]["bins"]
                                           if (b[1], b[2], b[3]) == (cfg["s5"], cfg["key"], cfg["sub6"])})},
                       stats=cal_stats)
        return _with_merges(b2, body), None

    def s6_pooled(bb, wc):
        T = cal_R["trk"]
        pop = AC.population(T, "trk")
        c2, n = AC.s6_parts(T, cfg["s5"], cfg["key"], cfg["sub6"])
        thr = list(bb["s6"]["thr"])
        for i, b in enumerate(body["s6"]["bins"]):
            if (b[1], b[2], b[3]) != (cfg["s5"], cfg["key"], cfg["sub6"]):
                continue
            m = pop & (s6_groups(T) == b[0]) & (n > 0)
            w = AC.tp_weights(T["tp_key"][m], np.zeros(int(m.sum()), np.int64), wc[T["event"][m]])
            thr[i] = AC.wquantile(c2[m], w, LEVEL)        # one raw-chi2 cut for every n
        return {**bb, "s6": {**bb["s6"], "thr": thr}}, None

    def separate(bb, wc):
        qa = np.asarray(bb["win"]["thr"]["a"], np.float64)[:, 0]
        qb = np.asarray(bb["win"]["thr"]["b"], np.float64)[:, 1]
        return bb, np.stack([qa, qb], 1)
    return {"cuts_at_098": l098, "s6_pooled_over_n": s6_pooled, "s34_two_99_cuts": separate}


# ==========================================================================
# commands
# ==========================================================================
def _gate_run(cal, test_c, cuts, cfg_from_test=True, report_only=(), n_rep=N_REP, cross_format=False):
    cal_d, cal_man, cal_R = load(cal)
    t_d, t_man, t_R = load(test_c)
    # the cuts come from the calibration census and are judged on the test census: one format, unless
    # marked cross-format (WP-D0b's own runs: CAL at format 33, KG / G / G2 at 34)
    formats = TF.one_format({"calibration": cal_man, "test": t_man}, cross_format)
    doc = json.load(open(cuts))
    body = {k: doc[k] for k in ("s1", "s2a", "s2z", "win", "s6") + ("s1d", "s2ad", "s2zd", "wind") if k in doc}
    model = model_of(cal_man)
    cfg = run_config(t_man)
    test = Test(t_R, t_man, cfg, model)
    units = test.bins(body)
    passed = test.passes(body, units)
    # the census's own cut decisions are the offline ones (records' `passed`)
    agree = {}
    for c, col in (("s1", "passed"), ("s2a", "passed_a"), ("s2z", "passed_z"), ("s3", "passed"),
                   ("s6", "passed")) + ((("s1d", "passed"), ("s2ad", "passed_a"), ("s2zd", "passed_z"),
                                         ("s4wd", "passed")) if test.disc else ()):
        T = test.tab[c]
        m = units[c] >= 0
        if c == "s3":
            m &= np.isin(T["role"], (0, 1))
        agree[c] = int((passed[c][m] != T[col][m].astype(bool)).sum())
    eff = {"nominal": efficiencies(test, body, units, passed)}
    kbs = known_bad_bodies(cal_R, cal_man, model, body, cfg)
    for name, fn in kbs.items():
        b2, wthr = fn(body, np.ones(n_events(cal_man)))
        eff[name] = efficiencies(test, body, units, test.passes(b2, units, wthr))
    # the legacy per-layer scales exist on the barrel consumers only: the disc consumers keep the nominal cuts
    eff["legacy_3sigma"] = efficiencies(test, body, units, {**passed, **legacy_passes(test)})
    # the disc known-bads (offline: the test records' disc statistics under each, the nominal cuts)
    disc_kb = {}
    if test.disc:
        for kb in CA.DISC_KNOWN_BADS:
            p = dict(passed)
            p.update(test.disc_passes(body, units, test.disc_stats(model.with_disc_known_bad(kb))))
            disc_kb[f"disc_{kb}"] = efficiencies(test, body, units, p)
    eff.update(disc_kb)
    t0 = time.time()
    reps = bootstrap(cal_R, cal_man, model, test, body, cfg, n_events(cal_man), n_events(t_man),
                     n_rep, extra=kbs)
    for r in reps:
        r["legacy_3sigma"] = r["nominal"]          # sd of the legacy known-bad: the nominal's
        for k in disc_kb:
            r[k] = r["nominal"]                    # and of the disc known-bads
    res = {"cal": cal_d, "test": t_d, "formats": formats, "cuts": os.path.realpath(cuts), "config": cfg,
           "n_rep": n_rep, "bootstrap_s": time.time() - t0, "offline_vs_census_disagreements": agree}
    res["nominal"] = judge_one("nominal", eff["nominal"], reps, body, report_only)
    res["known_bad"] = {k: judge_one(k, eff[k], reps, body, report_only)
                        for k in list(kbs) + ["legacy_3sigma"] + list(disc_kb)}
    res["known_bad_red"] = {k: fails_gate(v) for k, v in res["known_bad"].items()}
    # a disc known-bad must turn its own disc consumers red (DISC_KNOWN_BAD_TARGETS)
    for k in disc_kb:
        res["known_bad_red"][k] = bool(material_consumers(res["known_bad"][k]) & set(DISC_KNOWN_BAD_TARGETS[k[5:]]))
    res["disc_consumers"] = list(DISC_CONSUMERS) if test.disc else []
    res["known_bad_report_only"] = list(KNOWN_BAD_REPORT_ONLY)
    res["genealogy"] = genealogy(test, body, units, passed)
    return res


def genealogy(test, body, units, passed):
    """Per consumer, particle-level efficiency per genealogy class and for class C."""
    out = {}
    for c in test.consumers:
        T = test.tab[c]
        base = (units[c] >= 0) & (T["tp_pt"] >= T["floor"])
        if c in ("s3", "s4w", "s4wd"):
            base &= T["role"] == (0 if c == "s3" else 1)
        row = {}
        for code, nm in CA.GEN_NAMES.items():
            m = base & (T["gen"] == code) & (test.pop[c] | (T["gen"] == CA.GEN_BEYOND))
            if c in ("s1", "s3", "s4w"):
                m &= T["is_T"].astype(bool)
            if c in ("s1d", "s4wd"):
                m &= T["lis_T"].astype(bool)
            if m.sum():
                w = AC.tp_weights(T["tp_key"][m], units[c][m])
                row[nm] = {"eff": float((w * passed[c][m]).sum() / w.sum()), "n_tp": float(w.sum())}
        m = test.cpop[c] & (units[c] >= 0)
        if m.sum():
            w = AC.tp_weights(T["tp_key"][m], units[c][m])
            row["class C"] = {"eff": float((w * passed[c][m]).sum() / w.sum()), "n_tp": float(w.sum())}
        out[c] = row
    return out


def cmd_tolerances(a):
    res = _gate_run(a.cal, a.kg, a.cuts, n_rep=a.n_rep, cross_format=a.cross_format)
    ro = sorted(k for k, v in res["nominal"]["b_iii"].items() if v["power"] < POWER_MIN)
    res["report_only"] = ro
    res["rule"] = {"margin": MARGIN, "power_min": POWER_MIN, "material_bin_share": MATERIAL_BIN_SHARE,
                   "level": LEVEL, "n_rep": a.n_rep, "seed": SEED}
    res["material_on_kg"] = material(res["nominal"])
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    _print(res, "KG")
    print(f"report-only cells (power < {POWER_MIN} on KG): {ro}")
    print(f"wrote {a.out}")


def cmd_judge(a):
    tol = json.load(open(a.tol))
    res = _gate_run(a.cal, a.test, a.cuts, report_only=tuple(tol["report_only"]), n_rep=tol["rule"]["n_rep"],
                    cross_format=a.cross_format)
    res["report_only"] = tol["report_only"]
    res["tolerances"] = os.path.realpath(a.tol)
    res["material"] = material(res["nominal"])
    required = {k: v for k, v in res["known_bad_red"].items() if k not in KNOWN_BAD_REPORT_ONLY}
    res["verdict"] = "MATERIAL" if (res["material"] or not all(required.values())) else "pass"
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    _print(res, "TEST")
    print(f"verdict: {res['verdict']}; MATERIAL: {res['material']}; known-bads red: {res['known_bad_red']} "
          f"(report-only: {list(KNOWN_BAD_REPORT_ONLY)})")
    print(f"wrote {a.out}")


def _print(res, lab):
    print(f"[{lab}] offline vs census cut decisions disagree: {res['offline_vs_census_disagreements']}")
    for k, v in res["nominal"]["b_iii"].items():
        print(f"  (b-iii) {k:12s} eff {v['eff']:.4f} +- {v['sd']:.4f}  bound {v['upper_bound_deficit']:+.4f}  "
              f"power {v['power']:.2f}  {'pass' if v['pass'] else 'FAIL'}{' (report-only)' if v['report_only'] else ''}")
    for c, v in res["nominal"]["b_ii"].items():
        print(f"  (b-ii)  {c:5s} {v['n_bins']} bins, failing (under) {len(v['failing_bins'])} "
              f"({100 * v['failing_share']:.2f} % of particles); over-covering {len(v['over_bins'])} "
              f"({100 * v['over_share']:.2f} %; reported); chi2 p {v['p']:.3g} (reported)")
    for k, red in res["known_bad_red"].items():
        print(f"  known-bad {k}: {'red' if red else 'NOT RED'}"
              f"{' (report-only)' if k in KNOWN_BAD_REPORT_ONLY else ''}")


def _core(x):
    """The B.1 core width (estimator.core sigma) of the finite x; NaN below its minimum count."""
    r = SD.core_w(x[np.isfinite(x)])
    return r["sigma"] if r else np.nan


def cmd_keys(a):
    d, man, R = load(a.kg)
    model = model_of(man)
    rng = np.random.default_rng(SEED)
    nev = n_events(man)
    out = {"census": d}
    # S4a: every decision holding a correct candidate, re-scored under both keys
    A = R["arb"]
    C = AC._comp(A)
    ang = M.AngleUse.from_key(man["config"]["angles"])
    w_arb = ang.arb_weight
    scores = {}
    for key in CA.KEYS:
        with np.errstate(all="ignore"):
            if key == "emitted":
                h, sh, lam, sl = CA.corrected_of(C, *[model.k[x] for x in CA.ANGLES], model.mode)
            else:
                rows = model.rows("true", A["layer"], A["cot_pred"])
                ca, wa = model.cw("true", "alpha", rows)
                cb, wb = model.cw("true", "beta", rows)
                h, sh, lam, sl = CA.corrected_of(C, *[model.k[x] for x in CA.ANGLES], model.mode,
                                                 (ca, wa, cb, wb))
            kap, skap, cot, scot, _, _ = CA.represent(A["r"], A["z"], h, sh, lam, sl)
            pa, pb = CA.cand_pulls(kap, skap, cot, scot, A["kap_pred"], A["cot_pred"], A["skap"], A["sct"])
            t = (pa * pa if ang.uses("proj_arb", "alpha") else 0.0) + (pb * pb if ang.uses("proj_arb", "beta") else 0.0)
            scores[key] = A["pos"] + w_arb * np.where(A["dir_ok"], t, 0.0)

    def correct_rate(score, ev_w=None):
        o = np.lexsort((score, A["dec"]))
        first = np.r_[True, A["dec"][o][1:] != A["dec"][o][:-1]]
        pick = o[first]
        w = np.ones(len(pick)) if ev_w is None else ev_w[A["event"][pick]]
        return float((w * A["correct"][pick]).sum() / w.sum())
    rates = {k: correct_rate(s) for k, s in scores.items()}
    diffs = []
    for _ in range(N_REP):
        wt = rng.multinomial(nev, np.full(nev, 1.0 / nev)).astype(float)
        diffs.append(correct_rate(scores["prediction"], wt) - correct_rate(scores["emitted"], wt))
    sd = float(np.std(diffs, ddof=1))
    dlt = rates["prediction"] - rates["emitted"]
    s4a = "prediction" if (dlt > 2 * sd or abs(dlt) <= 2 * sd) else "emitted"
    out["s4a"] = {"correct_hit_rate": rates, "delta_prediction_minus_emitted": dlt, "sd": sd,
                  "n_decisions": int(len(np.unique(A["dec"]))), "choice": s4a,
                  "rule": "higher paired correct-hit rate; a tie (|delta| <= 2 sd) takes prediction"}
    # S5: correct tracks' core resolution under both keys (S5 with both angles)
    T = R["trk"]
    pop = AC.population(T, "trk")
    res = {}
    for key in CA.KEYS:
        ok = pop & T[f"ok_ab_{key}"].astype(bool)
        res[key] = ok
    both = res["emitted"] & res["prediction"]

    def cores(key, ev_w=None):
        idx = np.flatnonzero(both)
        if ev_w is not None:
            idx = np.repeat(idx, ev_w[T["event"][idx]].astype(int))
        return {"d0": _core(T[f"fit_d0_{key}"][idx] - T["tp_d0"][idx]),
                "z0": _core(T[f"fit_z0_{key}"][idx] - T["tp_z0"][idx]),
                "inv_pt": _core(np.abs(T[f"fit_kappa_{key}"][idx]) - 1.0 / T["tp_pt"][idx])}
    nom = {k: cores(k) for k in CA.KEYS}
    rep = {p: [] for p in ("d0", "z0", "inv_pt")}
    for _ in range(N_REP // 2):
        wt = rng.multinomial(nev, np.full(nev, 1.0 / nev)).astype(float)
        ce, cp = cores("emitted", wt), cores("prediction", wt)
        for p in rep:
            rep[p].append(cp[p] - ce[p])
    s5 = {}
    better, worse = [], []
    for p in rep:
        dd = nom["prediction"][p] - nom["emitted"][p]
        sdp = float(np.nanstd(rep[p], ddof=1))
        s5[p] = {"emitted": nom["emitted"][p], "prediction": nom["prediction"][p], "delta": dd, "sd": sdp}
        if dd < -2 * sdp:
            better.append(p)
        if dd > 2 * sdp:
            worse.append(p)
    out["s5"] = {"core_resolution": s5, "n_tracks": int(both.sum()),
                 "choice": "prediction" if (better and not worse) else "emitted",
                 "rule": "prediction only if it improves a parameter beyond 2 sd (paired) and worsens none beyond 2 sd"}
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print(json.dumps({k: {kk: v[kk] for kk in ("choice",)} | ({"rates": v["correct_hit_rate"], "sd": v["sd"]}
                                                               if k == "s4a" else {"cores": v["core_resolution"]})
                      for k, v in out.items() if k != "census"}, indent=1, default=float))
    print(f"wrote {a.out}")


def cmd_report(a):
    """End-to-end particle loss against the all-angles-off census (same events): findable TPs with a
    correct DR-input track (owner the TP, no wrong hit) in one run and not the other."""
    import awkward as ak
    import uproot
    out = {}
    runs = {}
    for lab, c in (("angles", a.g), ("off", a.off)):
        d = TF.census_dir(c)
        C = TF.load(d, with_tracks=True, verbose=False)
        runs[lab] = C
    Cg, Co = runs["angles"], runs["off"]
    formats = TF.one_format(runs, a.cross_format)
    if not np.array_equal(Cg["key"], Co["key"]):
        raise SystemExit("the two censuses' TP tables differ")
    keys = Cg["key"]
    fnd = TF.findable(Cg)
    man = json.load(open(os.path.join(TF.census_dir(a.g), "manifest.json")))
    path = man["inputs"][0]["path"]
    L = uproot.open(f"{path}:Events").arrays(["L1TTP_genFromB", "L1TTP_genFromC", "L1TTP_evt",
                                              "L1TTP_vx", "L1TTP_vy"], entry_stop=int(Cg["n_events"]))
    nt = ak.to_numpy(ak.num(L["L1TTP_evt"]))
    tk = M.tp_key(np.repeat(np.arange(len(nt)), nt), np.arange(int(nt.sum())) - np.repeat(np.cumsum(nt) - nt, nt))
    f = {k: ak.to_numpy(ak.flatten(L[f"L1TTP_{k}"])) for k in ("genFromB", "genFromC", "evt", "vx", "vy")}
    gen_all = np.where(np.hypot(f["vx"], f["vy"]) > CA.BORN_IN_CM, CA.GEN_BEYOND,
                       np.where((f["genFromB"] > 0) | (f["genFromC"] > 0), CA.GEN_HF,
                                np.where(f["evt"] == 0, CA.GEN_PROMPT_HS, CA.GEN_PROMPT_PU)))
    p = np.clip(np.searchsorted(tk, keys), 0, len(tk) - 1)
    gen = np.where(tk[p] == keys, gen_all[p], CA.GEN_NONE)

    def correct_keys(C, build):
        col = {c: j for j, c in enumerate(C["track_cols"])}
        out_k = []
        for e in C["builds"][build]:
            if e not in C["tracks"] or not len(C["tracks"][e]):
                continue
            Tt = C["tracks"][e]
            m = (Tt[:, col["own_tpidx"]] >= 0) & (Tt[:, col["n_wrong"]] == 0)
            out_k.append(M.tp_key(Tt[m, col["event"]].astype(np.int64), Tt[m, col["own_tpidx"]].astype(np.int64)))
        return np.unique(np.concatenate(out_k)) if out_k else np.zeros(0, np.int64)
    for b in Cg["builds"]:
        hg = np.isin(keys, correct_keys(Cg, b))
        ho = np.isin(keys, correct_keys(Co, b))
        row = {}
        for pt_lab, lo, hi in (("pT 1-2", 1.0, 2.0), ("pT 2-5", 2.0, 5.0), ("pT >= 5", 5.0, np.inf)):
            for gcode, gname in CA.GEN_NAMES.items():
                if gcode == CA.GEN_NONE:
                    continue
                m = fnd & (Cg["pt"] >= lo) & (Cg["pt"] < hi) & (np.abs(Cg["eta"]) < 1.1) & (gen == gcode)
                n_off = int((m & ho).sum())
                row[f"{pt_lab} / {gname}"] = {"findable": int(m.sum()), "correct_off": n_off,
                                              "correct_angles": int((m & hg).sum()),
                                              "lost": int((m & ho & ~hg).sum()), "gained": int((m & hg & ~ho).sum()),
                                              "survival": float((m & ho & hg).sum() / max(n_off, 1))}
        out[b] = row
    json.dump({"formats": formats, **out}, open(a.out, "w"), indent=1)
    for b, row in out.items():
        for k, v in row.items():
            if v["correct_off"]:
                print(f"{b} {k:32s} off {v['correct_off']:6d} angles {v['correct_angles']:6d} lost {v['lost']:5d} "
                      f"gained {v['gained']:5d} survival {v['survival']:.4f}")
    print(f"wrote {a.out}")


# ==========================================================================
# D-fill (WP-Disc plan sec. 3): every declared disc entry fills; the disc consumers' records per event within
# DFILL_BAND of the known-good's
# ==========================================================================
DFILL_BAND = 0.20


def dfill_stat(C, R):
    """Per declared disc entry (a census entry with a disc layer) its fitted tracks and TPs found; per disc
    consumer's record table its records per event."""
    ent = {}
    for i, sd in enumerate(C["seeds"]):
        if bool(M.is_disc(sd.layers).any()):
            c = C["counters"].get(i, {})
            ent[f"{i}:{sd.tag}"] = {"fit": float(c.get("fit", 0.0)), "found": float(c.get("found_standalone", 0.0))}
    nev = max(int(C["n_events"]), 1)
    rec = {k: (len(R[k]["event"]) / nev if k in R else 0.0) for k in ("s1d", "paird", "wind")}
    return {"entries": ent, "records_per_event": rec, "n_events": nev}


def dfill_judge(test, kg):
    """D-fill's rule: every declared disc entry of the test census has tracks and found TPs; each disc record
    table's records per event lies within DFILL_BAND of the known-good's (and the known-good has some)."""
    why = [f"entry {e} does not fill ({v['fit']:g} tracks, {v['found']:g} TPs found)"
           for e, v in test["entries"].items() if v["fit"] <= 0 or v["found"] <= 0]
    if not test["entries"]:
        why.append("no declared disc entry")
    for k, v in test["records_per_event"].items():
        ref = kg["records_per_event"].get(k, 0.0)
        if ref <= 0 or abs(v / ref - 1.0) > DFILL_BAND:
            why.append(f"{k}: {v:.1f} records per event against the known-good's {ref:.1f}")
    return not why, why


def _dfill_drop(st, entries):
    """D-fill's known-bads: the test census with `entries` (indices) emptied, their records gone with them (the
    disc pair records are the IT disc seeds' own; a dropped seed's follow records are not separable and stay)."""
    out = {"entries": {e: (dict(v, fit=0.0, found=0.0) if int(e.split(":")[0]) in entries else v)
                       for e, v in st["entries"].items()},
           "records_per_event": dict(st["records_per_event"]), "n_events": st["n_events"]}
    it_disc = [e for e in st["entries"] if any(L in M.IT_DISC_CODES for L in _tag_layers(e))]
    if it_disc and all(int(e.split(":")[0]) in entries for e in it_disc):
        out["records_per_event"]["paird"] = 0.0
    return out


def _tag_layers(e):
    return [CODE_OF[n] for n in e.split(":", 1)[1].split()[0].split("+")]


CODE_OF = {v: k for k, v in TF.NAME.items()}


def cmd_dfill(a):
    out = {}
    stats = {}
    for lab, c in (("kg", a.kg), ("test", a.test)):
        d = TF.census_dir(c)
        stats[lab] = dfill_stat(TF.load(d, verbose=False), CA.load_records(d))
    TF.one_format({"kg": json.load(open(os.path.join(TF.census_dir(a.kg), "manifest.json"))),
                   "test": json.load(open(os.path.join(TF.census_dir(a.test), "manifest.json")))})
    ok, why = dfill_judge(stats["test"], stats["kg"])
    out = {"stage": "D-fill", "kg": a.kg, "test": a.test, "band": DFILL_BAND, **stats, "pass": ok, "why": why}
    idx = [int(e.split(":")[0]) for e in stats["test"]["entries"]]
    it_idx = [int(e.split(":")[0]) for e in stats["test"]["entries"]
              if any(L in M.IT_DISC_CODES for L in _tag_layers(e))]
    kbs = {"one_disc_entry_dropped": idx[:1], "it_disc_entries_dropped": it_idx}
    out["known_bads"] = {}
    for name, drop in kbs.items():
        okb, whyb = dfill_judge(_dfill_drop(stats["test"], set(drop)), stats["kg"])
        out["known_bads"][name] = {"dropped": drop, "red": not okb, "why": whyb}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"D-fill: {'pass' if ok else 'FAIL'}; {why}")
    for name, v in out["known_bads"].items():
        print(f"  known-bad {name}: {'red' if v['red'] else 'NOT RED'}")
    print(f"wrote {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("keys")
    k.add_argument("kg")
    k.add_argument("-o", "--out", required=True)
    t = sub.add_parser("tolerances")
    t.add_argument("cal")
    t.add_argument("kg")
    t.add_argument("cuts")
    t.add_argument("--n-rep", type=int, default=N_REP)
    t.add_argument("-o", "--out", required=True)
    j = sub.add_parser("judge")
    j.add_argument("cal")
    j.add_argument("test")
    j.add_argument("cuts")
    j.add_argument("tol")
    j.add_argument("-o", "--out", required=True)
    r = sub.add_parser("report")
    r.add_argument("g")
    r.add_argument("off")
    r.add_argument("-o", "--out", required=True)
    df = sub.add_parser("dfill")
    df.add_argument("kg")
    df.add_argument("test")
    df.add_argument("-o", "--out", required=True)
    for p in (t, j, r):
        p.add_argument("--cross-format", action="store_true",
                       help="mark the comparison cross-format: censuses of different formats are allowed "
                            "(their formats are reported); refused otherwise")
    a = ap.parse_args()
    {"keys": cmd_keys, "tolerances": cmd_tolerances, "judge": cmd_judge, "report": cmd_report,
     "dfill": cmd_dfill}[a.cmd](a)


if __name__ == "__main__":
    main()
