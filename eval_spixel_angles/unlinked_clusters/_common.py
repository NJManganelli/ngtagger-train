"""Shared loader for the unlinked-cluster study.

Two PU200 D121 ttbar samples are used, because no single nano has everything:

  T  itot_tp_f01/f02_100ev.root  newest producer, L1TTP table (bx, evt, pdgId),
     but produced WITHOUT a noiseSet: unlinked clusters carry NO angle
     (hasAlpha = hasBeta = 0, localCot* = -999).
  N  spix_sweep15_500.root (first N events)  produced WITH smarthit_noise_v6.json:
     unlinked clusters carry a synthetic angle drawn from the sizeY-conditioned
     inclusive inverse CDF. No L1TTP table.

Everything is read-only; the flat per-cluster frame is cached in a scratch dir
(env UNLINKED_CACHE, default below), never next to the inputs. The cache names
carry CACHE_VERSION: the frames of the key-linked era (raw_<sample>.parquet,
combined.parquet; TP links joined by collection key) are never reused.

TP links are tpIdx rows into L1TTP (<= -3: matched TP not in L1TTP; unlinked =
not SPN.tp_matched). Key-linked nanos (an L1TTP key column) are refused by
spix_nano.
"""
from __future__ import annotations

import os
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from ngtagger.data import spix_nano as SPN

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"
OTSTUB = Path.home() / "smartpixels/cmssw/work/otstub_arm"
WORK = Path.home() / "smartpixels/cmssw/work"
CACHE = Path(os.environ.get(
    "UNLINKED_CACHE",
    "/private/tmp/claude-501/-Users-nmangane/a13c522f-0b01-442c-b38e-7cf47b0dee18/scratchpad/unlinked_cache"))

SAMPLES = {
    "T": [(OTSTUB / "itot_tp_f01_100ev.root", 100), (OTSTUB / "itot_tp_f02_100ev.root", 100)],
    "N": [(WORK / "spix_sweep15_500.root", 200)],
}
CL = SPN.CLUSTER_TABLE
# bump when the cached frames change meaning (tpidx1: TP links are L1TTP rows)
CACHE_VERSION = "tpidx1"

# Training / payload coverage, SPEC convention (alpha = r-phi bending, beta = along z).
# Conv1D_Full-2bit NN eval dump (the sample the adopted payload is built from):
#   spec alpha <- source cotBtrue in [-1.077, +0.746]; spec beta <- source cotAtrue in [-1.067, +1.069]
TRAIN_ALPHA = (-1.077, 0.746)
TRAIN_BETA = (-1.067, 1.069)
# Payload binning (angle axes, flow=clamp): alpha edges +-0.6, beta edges +-6
PAYLOAD_ALPHA = (-0.6, 0.6)
PAYLOAD_BETA = (-6.0, 6.0)

B_T = 3.8
C_CURV = 0.29979246 * B_T / 100.0   # 1/R [1/cm] = C_CURV / pT[GeV]

NONANGLE = ["size", "sizeX", "sizeY", "charge", "qPerPix", "absZ", "localX", "localY",
            "globalR", "sigX", "sigY"]
ANGLE = ["localCotAlpha", "localCotBeta", "sigAlpha", "sigBeta", "hasAlpha", "hasBeta"]


def _read_file(path: Path, nev: int, with_tp: bool) -> pd.DataFrame:
    t = uproot.open(path)["Events"]
    # every cluster column, TBPX only (the study is per barrel layer)
    cols = [k[len(CL) + 1:] for k in t.keys() if k.startswith(CL + "_")]
    a = t.arrays(SPN.cluster_branches(t.keys(), cols, surfaces=SPN.BARREL) + ["event"],
                 entry_stop=nev)
    J = SPN.cluster_columns(a, cols, surfaces=SPN.BARREL)
    counts = ak.num(J[cols[0]]).to_numpy()
    df = pd.DataFrame({c: ak.flatten(J[c]).to_numpy() for c in cols})
    df["event"] = np.repeat(a["event"].to_numpy(), counts)
    df["ientry"] = np.repeat(np.arange(len(counts)), counts)
    if with_tp and SPN.has_column(t.keys(), "L1TTP", "bx"):
        # the cluster's L1TTP row is its tpIdx (>= 0); other codes match no row
        tk = ["bx", "evt", "pdgId", "pt"]
        b = SPN.table_columns(t.arrays(SPN.branch_names(t.keys(), "L1TTP", tk), entry_stop=nev),
                              "L1TTP", tk)
        nt = ak.num(b["bx"]).to_numpy()
        tp = pd.DataFrame({k: ak.flatten(b[k]).to_numpy() for k in tk})
        tp["ientry"] = np.repeat(np.arange(len(nt)), nt)
        tp["tpIdx"] = (np.arange(len(tp)) - np.repeat(np.cumsum(nt) - nt, nt)).astype(df["tpIdx"].dtype)
        tp = tp.rename(columns={"bx": "tpBx", "evt": "tpEvt", "pdgId": "tpPdgId", "pt": "tpPtTab"})
        df = df.merge(tp, on=["ientry", "tpIdx"], how="left", validate="many_to_one")
    return df


def _load_raw(sample: str) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    cpath = CACHE / f"raw_{sample}_{CACHE_VERSION}.parquet"
    if cpath.exists():
        return pd.read_parquet(cpath)
    parts = []
    for i, (p, nev) in enumerate(SAMPLES[sample]):
        d = _read_file(p, nev, with_tp=(sample == "T"))
        d["ifile"] = i
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(cpath)
    return df


ANGLE_COLS = ["localCotAlpha", "localCotBeta", "sigAlpha", "sigBeta", "hasAlpha", "hasBeta",
              "globalClusterPhi", "globalClusterCotTheta", "sigGlobalClusterPhi",
              "sigGlobalClusterCotTheta"]
KEY = ["event", "detId", "localX", "localY", "charge"]


def load() -> pd.DataFrame:
    """Combined frame = what the CURRENT producer emits with noiseSet=smarthit_noise_v6.

    T and N hold the SAME 200 events and the SAME 5.4M clusters (verified: outer join on
    (event, detId, localX, localY, charge) matches every row, the TP link agrees 100%), in a
    different row order. TP-linked angles are bit-identical between them; only sigAlpha/
    sigBeta differ for ~19% of linked clusters, because the Sep-5 producer (N) looked the
    sigma up at the TRUE angle and the current one (T) at the RECO angle. So: every column
    from T (current producer + L1TTP join), and for UNLINKED rows only, the angle block
    from N (the noise-v6 synthetic draw; its sigma was always keyed on the drawn angle).
    The T-sample angle block is kept as *_T for the no-noiseSet comparison.
    """
    cpath = CACHE / f"combined_{CACHE_VERSION}.parquet"
    if cpath.exists():
        return pd.read_parquet(cpath)
    T = _load_raw("T")
    N = _load_raw("N")
    df = T.merge(N[KEY + ANGLE_COLS], on=KEY, how="left", suffixes=("", "_N"), validate="one_to_one")
    assert df["localCotAlpha_N"].notna().all()
    u = ~SPN.tp_matched(df["tpIdx"].to_numpy())
    for k in ANGLE_COLS:
        df[k + "_T"] = df[k]
        df.loc[u, k] = df.loc[u, k + "_N"].to_numpy()
        df.drop(columns=k + "_N", inplace=True)
    df["evid"] = df["ifile"] * 100000 + df["ientry"]
    df = derive(df)
    df.to_parquet(cpath)
    return df


def derive(df: pd.DataFrame) -> pd.DataFrame:
    df["unlinked"] = (~SPN.tp_matched(df["tpIdx"])).astype(np.int8)
    df["qPerPix"] = df["charge"] / np.maximum(df["size"], 1)
    df["absZ"] = np.abs(df["globalZ"])
    df["tpVr"] = np.where(SPN.tp_matched(df["tpIdx"]), np.hypot(df["tpVx"], df["tpVy"]), np.nan)
    # Implied longitudinal origin of the cluster's reported direction (straight line in r-z,
    # exact for a helix in s-z, s ~ r at these radii): z0 = z - r * cot(theta_dir)
    ok = df["globalClusterCotTheta"] > -900
    df["z0imp"] = np.where(ok, df["globalZ"] - df["globalR"] * df["globalClusterCotTheta"], np.nan)
    df["sigZ0imp"] = np.where(ok, df["globalR"] * df["sigGlobalClusterCotTheta"], np.nan)
    okp = df["globalClusterPhi"] > -900
    dphi = np.angle(np.exp(1j * (df["globalClusterPhi"] - df["globalPhi"])))
    df["dphiDir"] = np.where(okp, dphi, np.nan)
    return df


def roc_auc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    if y.min() == y.max():
        return np.nan
    return roc_auc_score(y, s)


def md(df: pd.DataFrame, fmt: str = "{:.4g}") -> str:
    """Minimal DataFrame -> GitHub markdown (tabulate is not in the pixi env)."""
    def f(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else fmt.format(v)
        return str(v)
    cols = [str(df.index.name or "")] + [str(x) for x in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [f(v) for v in row.to_numpy()]) + " |")
    return "\n".join(lines) + "\n"
