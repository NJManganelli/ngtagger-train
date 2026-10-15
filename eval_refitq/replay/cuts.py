"""The cutting consumers' 99 % cuts (PLAN §1 item 5, §10 PD1, PD2; decision 2: each cut keeps 99 % of
correct hits in its own statistic, quantile rank k = min(ceil(L (n + 1)), n), spdefs.consumer_quantile).

Consumers and their statistics (all innovation-normalised, so the visit step's prediction error is
divided out; bins also separate the first-visited surface from the later ones):

  gate|<dims>   the forward hit gate: the innovation chi2 over positions + the gate's angle dims
                (dims "xy", "xya", "xyb", "xyab"), population: the OWN PRIMARY (PD1); also K's
                per-empty-surface penalty (PD2)
  rr|<dims>|<cov>|<scen>
                reject-and-refit: the leave-one-out chi2 of an accepted own-primary hit, per scenario
                (PLAN PD15: the pooled cut biased AAAA against IAAA)
  window        the angle window cut: |angle innovation pull| per angle, population: every own class-T
                cluster (an angle-only cut, PD1), bins also by the signed predicted-key bin

Bins: (surface, first-visited or later) for the chi2 consumers, x signed predicted-key bin for the window.
A bin with fewer than MIN_N fit values falls back to the (surface, first / later) pool, then to the
surface, then to everything (flagged per bin in the file). The gate and window cuts are fitted pooled over
the scenarios (PLAN choice 8), the rr cuts per scenario (PD15); separately per covariance model
("producer", "calibrated"), at k = 1.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from ngtagger.spdefs import angles as SA

SCHEMA = "wpc-cuts/2"
LEVEL = 0.99
MIN_N = 200


def _q(stat):
    stat = np.asarray(stat, np.float64)
    stat = stat[np.isfinite(stat)]
    if len(stat) < 1:
        return None
    return float(SA.consumer_quantile(stat, np.zeros(len(stat), np.int64), levels=(LEVEL,), n_bins=1)[0, 0])


def _fit_table(stat, surface, first, keybin=None):
    """{bin key: (value, n, source)} for the consumer's statistic."""
    stat = np.asarray(stat, np.float64)
    ok = np.isfinite(stat)
    stat, surface, first = stat[ok], np.asarray(surface)[ok], np.asarray(first)[ok]
    kb = np.asarray(keybin)[ok] if keybin is not None else None
    out = {}
    glob = _q(stat)
    out["*"] = (glob, int(len(stat)), "all")
    for S in np.unique(surface):
        mS = surface == S
        out[f"{S}"] = (_q(stat[mS]), int(mS.sum()), "surface")
        for f in (True, False):
            m = mS & (first == f)
            out[f"{S}|{int(f)}"] = (_q(stat[m]), int(m.sum()), "surface|step")
            if kb is not None:
                for b in np.unique(kb[m]):
                    mb = m & (kb == b)
                    out[f"{S}|{int(f)}|{int(b)}"] = (_q(stat[mb]), int(mb.sum()), "surface|step|key")
    return {k: {"value": v, "n": n, "level": src} for k, (v, n, src) in out.items()}


class Cuts:
    def __init__(self, doc):
        if doc.get("schema") != SCHEMA:
            raise ValueError(f"not a WP-C cut file ({doc.get('schema')!r})")
        self.doc = doc

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            return cls(json.load(fh))

    def save(self, path):
        tmp = path + f".tmp{os.getpid()}"
        with open(tmp, "w") as fh:
            json.dump(self.doc, fh, indent=1)
        os.replace(tmp, path)

    def _lookup(self, table, keys_chain):
        for k in keys_chain:
            e = table.get(k)
            if e is not None and e["value"] is not None and e["n"] >= MIN_N:
                return e["value"]
        e = table["*"]
        return e["value"]

    def chi2(self, consumer, dims, cov_model, surface, first, scen=None):
        """The consumer's cut per (surface, first); rr needs the scenario (PD15)."""
        tab = self.doc["cuts"][table_key(consumer, dims, cov_model, scen)]
        surface, first = np.atleast_1d(surface), np.atleast_1d(first)
        return np.array([self._lookup(tab, (f"{s}|{int(f)}", f"{s}")) for s, f in zip(surface.tolist(),
                                                                                     first.tolist())])

    def window(self, cov_model, surface, first, keybin_alpha, keybin_beta):
        out = np.full((len(surface), 2), np.inf)
        for i, (ang, kb) in enumerate((("alpha", keybin_alpha), ("beta", keybin_beta))):
            tab = self.doc["cuts"][f"window|{ang}|{cov_model}"]
            out[:, i] = [self._lookup(tab, (f"{s}|{int(f)}|{int(b)}", f"{s}|{int(f)}", f"{s}"))
                         for s, f, b in zip(np.asarray(surface).tolist(), np.asarray(first).tolist(),
                                            np.asarray(kb).tolist())]
        return out


def table_key(consumer, dims, cov_model, scen=None):
    """The cut table's key: rr tables are per scenario (PD15), the gate tables pooled."""
    if consumer == "rr":
        if scen is None:
            raise ValueError("the rr cuts are per scenario (PD15): pass scen")
        return f"rr|{dims}|{cov_model}|{scen}"
    return f"{consumer}|{dims}|{cov_model}"


def populations(R, L, clean=True):
    """The fit populations (PD1, PD8): gate = own primaries (table criterion) with a clean history;
    window = own class-T clusters (table criterion) with a clean history; rr = accepted own primaries on
    tracks whose other accepted hits are all their own. clean=False: without the history condition."""
    h = R["clean"] if clean else np.ones(len(R["clean"]), bool)
    gate = R["primary"] & R["table_pop"] & h
    window = R["table_pop"] & h
    rr = L["own_primary"] & (L["clean_track"] if clean else True)
    return gate, window, rr


def fit(own_by_cov: dict, loo_by_cov: dict, provenance: dict, clean: bool = True) -> Cuts:
    """own_by_cov[cov] = concatenated own-cluster rows (diagnostics.own_rows + mark_primary);
    loo_by_cov[cov] = {"dims": {dims: stat}, "surface", "first", "own_primary", "clean_track", "scen"}."""
    cuts = {}
    for cov in own_by_cov:
        R, L = own_by_cov[cov], loo_by_cov[cov]
        gate, window, rr = populations(R, L, clean)
        for dims in ("xy", "xya", "xyb", "xyab"):
            cuts[f"gate|{dims}|{cov}"] = _fit_table(R[f"g_{dims}"][gate], R["surface"][gate], R["first"][gate])
            for scen in np.unique(L["scen"]):
                m = rr & (L["scen"] == scen)
                cuts[table_key("rr", dims, cov, str(scen))] = _fit_table(L["dims"][dims][m], L["surface"][m],
                                                                       L["first"][m])
        for i, ang in enumerate(("alpha", "beta")):
            cuts[f"window|{ang}|{cov}"] = _fit_table(R["z"][window, i], R["surface"][window], R["first"][window],
                                                     R["keybin"][window, i])
    doc = {"schema": SCHEMA, "level": LEVEL, "rank": "k = min(ceil(L (n + 1)), n) (spdefs.consumer_quantile)",
           "population": "PD1 + PD8: own primaries / own class T with a clean history" if clean else "no history condition",
           "min_n": MIN_N, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "provenance": provenance, "cuts": cuts}
    return Cuts(doc)
