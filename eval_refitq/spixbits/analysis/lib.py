"""spixbits analysis: reading the per-arm caches written by core.py, and the estimators built on the replicas."""
import json
import subprocess
import time
from pathlib import Path

import numpy as np

import core

STUDY = Path.home() / "smartpixels/doc-archive/spix-bits-refit-study"
RESULTS = STUDY / "results" if not core.TAG else STUDY / "results" / core.TAG
MIN_FREE_GB = 14.5


def check_disk():
    import shutil
    free = shutil.disk_usage(Path.home()).free / 1e9
    if free < MIN_FREE_GB:
        raise SystemExit(f"free disk {free:.1f} GB is below {MIN_FREE_GB} GB: nothing written")
    return free


class Scenario:
    """Caches of one scenario, loaded lazily."""

    def __init__(self, scen, cache=None):
        self.scen = scen
        self.dir = (cache or core.CACHE) / scen
        self.meta = json.load(open(self.dir / "_meta.json")) if (self.dir / "_meta.json").exists() else None
        self.arms = sorted(p.stem for p in self.dir.glob("*.npz") if not p.stem.endswith(".tmp")) if self.dir.exists() else []
        self._c, self._s = {}, {}

    def has(self, arm):
        return arm in self.arms

    def c(self, arm):
        if arm not in self._c:
            z = np.load(self.dir / f"{arm}.npz")
            self._c[arm] = {k: z[k] for k in z.files}
        return self._c[arm]

    def stats(self, arm):
        if arm not in self._s:
            self._s[arm] = json.loads(str(self.c(arm)["stats"]))
        return self._s[arm]

    def cell_index(self, part, name):
        return self.meta["cells"][part]["names"].index(name)

    def w(self, arm, q, part="origin", g=0, forced=False):
        """(point W68, point W95, replicas of W68) in um."""
        c = self.c(arm)
        pre = "fw" if forced else "w"
        pt = c[f"{pre}_{part}_{q}_point"][g]
        return float(pt[0]), float(pt[1]), c[f"{pre}_{part}_{q}_reps"][g].astype(np.float64)

    def pe(self, arm, name):
        c = self.c(arm)
        return c["pe"][list(c["pe_names"]).index(name)].astype(np.float64)


_W = {}


def weights(n_events=500):
    """The replica weight matrix (the same as core.weights; depends only on the number of events)."""
    if n_events not in _W:
        rng = np.random.default_rng(core.SEED)
        _W[n_events] = rng.multinomial(n_events, np.full(n_events, 1.0 / n_events), size=core.NREP).astype(np.float64)
    return _W[n_events]


def band(reps):
    """68 % bootstrap band (16 % and 84 % quantiles) and the sd."""
    reps = np.asarray(reps, float)
    lo, hi = np.nanquantile(reps, [0.16, 0.84])
    return float(lo), float(hi), float(np.nanstd(reps))


def val(point, reps):
    lo, hi, sd = band(reps)
    return {"value": float(point), "lo68": lo, "hi68": hi, "sd": sd}


def ratio(sc, arm, ref, q, part="origin", g=0, forced=False, ref_forced=False):
    a = sc.w(arm, q, part, g, forced)
    b = sc.w(ref, q, part, g, ref_forced)
    return val(a[0] / b[0], a[2] / b[2])


def frac(sc, arm, num, den):
    """A share with its bootstrap band from per-event sums."""
    n, d = sc.pe(arm, num), sc.pe(arm, den)
    W = weights(len(n))
    return val(n.sum() / d.sum(), (W @ n) / (W @ d))


def fmt(v, nd=1, pm=True):
    if v is None or not np.isfinite(v["value"]):
        return "n/a"
    s = f"{v['value']:.{nd}f}"
    return s + (f" ± {v['sd']:.{nd}f}" if pm else "")


def fmt_band(v, nd=3):
    if v is None or not np.isfinite(v["value"]):
        return "n/a"
    return f"{v['value']:.{nd}f} [{v['lo68']:.{nd}f}, {v['hi68']:.{nd}f}]"


def provenance(scens):
    """What existed when the analysis ran: per scenario, the arms complete on all five files, and the unit files."""
    disc = core.discover()
    reg = [a["name"] for a in core.registry()]
    out = {"run_at_local": time.strftime("%Y-%m-%d %H:%M %Z"), "scan_files": list(core.FILES), "units_directory": str(core.UNITS), "quantization_scenarios_in_registry": len(reg), "detector_scenarios": {}}
    try:
        out["analysis_commit"] = subprocess.run(["git", "-C", str(Path(__file__).parent), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    for s in scens:
        comp = core.complete(disc, s)
        cached = Scenario(s).arms
        out["detector_scenarios"][s] = {"quantization_scenarios_complete_on_all_files": len(comp), "quantization_scenarios_analysed": len([a for a in cached if a in reg]),
                               "quantization_scenarios_partial": {a: sorted(d) for a, d in disc[s].items() if a not in comp},
                               "quantization_scenarios_missing": [a for a in reg if a not in disc[s]]}
    return out


def arm_label(reg, name):
    for a in reg:
        if a["name"] == name:
            return a
    return None


PLAIN = {
    "implemented_now": "implemented now",
    "implemented_now_sizeblind": "implemented now, size-blind",
    "implemented_now_newtable": "implemented now, new angle table",
    "network_float": "network, float",
    "alloc_4444": "4/4/4/4, sub-area",
    "alloc_4444_wholesensor": "4/4/4/4, whole sensor",
    "alloc_6604": "bending-angle only 6/6/0/4, sub-area",
    "alloc_6604_wholesensor": "bending-angle only 6/6/0/4, whole sensor",
    "bending_float": "bending-angle only, float (F/F/0/F)",
    "positions_only_float": "positions only, float (F/F/0/0)",
    "outer_tracker_only": "outer tracker only (no refit)",
}
