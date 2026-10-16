"""Shared machinery of the fast gate: the item interface, the run context, references,
the identity rule and the family-wise threshold (plan App. C.2 P7, D.0)."""
from __future__ import annotations

import importlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# canon_v4 (C_v4 b85da0c18cd): it carries the DTC frames (L1TOTStubDTC) census format 35 requires, and equals
# canon_v3 file 3 on every branch but the declared bridge columns (closure_vs_cmssw bridge, matched by
# run / lumi / event: the two hold the same events in different orders).
KNOWN_GOOD_INPUT = "/Volumes/WDMac/smartpixels_scratch/canon_v4/spstudy_AAAA_f03_100ev.root"
KNOWN_GOOD_EVENTS = 5
REFS_DIR = Path(__file__).resolve().parent / "refs"
CODE_ROOT = Path(__file__).resolve().parents[3]          # the repo, or the frozen copy's root
REL_TOL = 1e-9


class NotApplicable(Exception):
    """The item cannot run from this code root (its tree is not in the frozen copy)."""


def t_family(n: int, alpha: float = 0.01) -> float:
    """Two-sided per-bin z at which n correct z-scores exceed it with total probability
    alpha (Sidak; plan C.2 P7: 3.89 at 100, 4.01 at 168, 4.33 at 672)."""
    from scipy.stats import norm

    n = max(int(n), 1)
    return float(norm.isf((1.0 - (1.0 - alpha) ** (1.0 / n)) / 2.0))


def event_replicas(n_events: int, n_rep: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_events, np.full(n_events, 1.0 / n_events), size=n_rep)


class Ctx:
    """What an item needs: the input, the event count, where the code lives, a cache."""

    def __init__(self, input_path: str = KNOWN_GOOD_INPUT, events: int = KNOWN_GOOD_EVENTS,
                 code_root: Path = CODE_ROOT, scratch: str | None = None):
        self.input = input_path
        self.events = events
        self.code_root = Path(code_root)
        self.scratch = scratch
        self.cache: dict = {}

    def tree(self, rel: str) -> Path:
        p = self.code_root / rel
        if not p.is_dir():
            raise NotApplicable(f"{rel} is not part of this code root ({self.code_root})")
        return p

    def script_module(self, rel_dir: str, name: str):
        """Import `name` from a script directory of THIS code root (as the scripts do it:
        their directory first on sys.path)."""
        d = str(self.tree(rel_dir))
        if d not in sys.path:
            sys.path.insert(0, d)
        mod = importlib.import_module(name)
        if not os.path.realpath(mod.__file__).startswith(os.path.realpath(d) + os.sep):
            raise RuntimeError(f"{name} imported from {mod.__file__}, not from {d}")
        return mod

    def arrays(self, branches, path: str | None = None, entry_stop: int | None = None):
        import uproot

        path = path or self.input
        key = (path, tuple(sorted(branches)), entry_stop or self.events)
        if key not in self.cache:
            t = uproot.open(path)["Events"]
            self.cache[key] = t.arrays(list(branches), entry_stop=entry_stop or self.events)
        return self.cache[key]

    def keys(self, path: str | None = None):
        import uproot

        path = path or self.input
        k = ("keys", path)
        if k not in self.cache:
            t = uproot.open(path)["Events"]
            self.cache[k] = {name: str(t[name].typename) for name in t.keys()}
        return self.cache[k]


class Item:
    """One fast-gate check. Subclasses set the class attributes and implement compute
    (the statistic with the code as it is), known_bad (the same statistic on the named
    known-bad input or variant) and judge (the rule, against the registered reference)."""

    id = ""
    title = ""
    c4_row = ""                 # the C.4 row (or D.1 item) it implements
    known_bad_desc = ""
    rule_desc = ""
    trees: tuple = ()           # repo trees it imports from besides src/ngtagger

    def compute(self, ctx: Ctx) -> dict:
        raise NotImplementedError

    def known_bad(self, ctx: Ctx) -> dict:
        raise NotImplementedError

    def judge(self, stat: dict, ref: dict | None) -> tuple[bool, str]:
        raise NotImplementedError

    def known_bad_is_red(self, stat: dict, ref: dict | None) -> tuple[bool, str]:
        """True when the rule fails on the known-bad statistic, as it must."""
        ok, why = self.judge(stat, ref)
        return not ok, why

    # references --------------------------------------------------------
    def ref_path(self) -> Path:
        return REFS_DIR / f"{self.id}.json"

    def load_ref(self) -> dict | None:
        p = self.ref_path()
        if not p.exists():
            return None
        with open(p) as f:
            return json.load(f)


# ---------------------------------------------------------------- identity rule
def _close(a, b, rel=REL_TOL, path="") -> list[str]:
    """Differences between two JSON-like structures (floats to rel tolerance)."""
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: present on one side only")
            else:
                out += _close(a[k], b[k], rel, f"{path}/{k}")
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} vs {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                out += _close(x, y, rel, f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            if a is not b:
                out.append(f"{path}: {a} vs {b}")
        elif not (math.isnan(a) and math.isnan(b)) and abs(a - b) > rel * max(1.0, abs(a), abs(b)):
            out.append(f"{path}: {a!r} vs {b!r}")
    elif a != b:
        out.append(f"{path}: {a!r} vs {b!r}")
    return out


def identity(stat: dict, ref: dict | None, keys) -> tuple[bool, list[str]]:
    """The 'identity' rule of D.0: on the reference input the statistic must equal the
    registered known-good value (floats to 1e-9 relative)."""
    if ref is None:
        return False, ["no registered reference (run derive-refs on the known-good first)"]
    diffs = []
    for k in keys:
        diffs += _close(stat.get(k), ref["stat"].get(k), path=k)
    return not diffs, diffs[:20]


def same_input(stat: dict, ref: dict | None) -> bool:
    return ref is not None and stat.get("input") == ref["stat"].get("input") \
        and stat.get("events") == ref["stat"].get("events")


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return [jsonable(v) for v in x.tolist()]
    return x


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
