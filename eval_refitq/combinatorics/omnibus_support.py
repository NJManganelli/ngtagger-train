"""Support for the SmartPixels combinatorics omnibus (spix_combinatorics_omnibus.py).

PROVENANCE FROM THE NANO. A nano does not record the CMSSW build (git) it was made
with; what it does record is every module's configuration, in the ParameterSets
tree. ``pset_blob`` reads that blob, ``module_params`` parses one module's
top-level parameters, ``producer_params`` follows a SmartPixels track table to the
L1SmartPixelsTrackProducer that wrote it (its refit configuration: useAngles,
windows, maxKFUpdates, ...) and ``rechit_params`` finds the SmartPixelsRecHitProducer
(the angle source and the emulator's sigma table, ``emuSigmaSet``). The build hash
must be supplied by the caller (omnibus --build-hash).

THE SIGMA TABLE. ``SigmaTable`` evaluates the emulator's angle-resolution tables
(correctionlib JSON: category on the surface code, binning on |reported cot|,
clamped at the top edge), so a study can compare measured widths with the
sigma the refit was given, bin by bin.

REPORT HELPERS. ``table_pages`` (vector table pages), ``legend_below`` /
``legends_outside`` (legends moved off the data, per the figure format), and the
project's resolution convention ``core_rms`` / ``full_rms`` (eval_refitq/sub2gev).
"""
from __future__ import annotations

import datetime
import hashlib
import importlib.util
import json
import os
import re
import subprocess

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(rel, name):
    sp = importlib.util.spec_from_file_location(name, os.path.join(_HERE, rel))
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


# the project convention: iterative 3-sigma clip about the mean, RMS about zero of
# the kept core (bias counts) -> (core RMS, fraction outside the core)
_S2 = _load(os.path.join("..", "sub2gev", "common.py"), "_sub2gev_common")
core_rms, full_rms = _S2.core_rms, _S2.full_rms


def width68(x):
    """(q84 - q16) / 2: the definition of the emulator sigma table's entries."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 20:
        return float("nan")
    lo, hi = np.percentile(x, [15.865, 84.135])
    return float(0.5 * (hi - lo))


def coverage(pull, ks=(1.0, 2.0, 3.0)):
    """Fraction of |pull| < k for each k (Gaussian: 0.683, 0.954, 0.997)."""
    p = np.asarray(pull, float)
    p = p[np.isfinite(p)]
    return {f"lt{k:g}": (float(np.mean(np.abs(p) < k)) if p.size else float("nan")) for k in ks}


GAUSS_COVERAGE = {"lt1": 0.6827, "lt2": 0.9545, "lt3": 0.9973}


# --------------------------------------------------------------------------
# ParameterSets
# --------------------------------------------------------------------------
_PARAM_RE = re.compile(rb"([A-Za-z0-9_@]+)=\+([A-Za-z])\(((?:[^()]|\([^()]*\))*)\)")


def pset_blob(path):
    """The raw ParameterSets blob of a nano (empty bytes when absent)."""
    import uproot
    with uproot.open(path) as F:
        if "ParameterSets" not in F:
            return b""
        br = F["ParameterSets"]["IdToParameterSetsBlobs"]
        return b"".join(bytes(br.basket(i).data) for i in range(br.num_baskets))


def _record(blob, start):
    """The '<...>' record enclosing position `start` (balanced on '<' / '>')."""
    i = blob.rfind(b"<", 0, start)
    depth = 0
    for j in range(i, len(blob)):
        c = blob[j:j + 1]
        if c == b"<":
            depth += 1
        elif c == b">":
            depth -= 1
            if depth == 0:
                return blob[i:j + 1]
    return blob[i:]


def _decode(kind, raw):
    s = raw.decode("latin-1")
    if kind in "Zz":
        if kind == "z":
            return [x for x in s.strip("{}").replace("\x00", "").split(",") if x]
        return s.replace("\x00", "")
    if kind == "B":
        return s == "true"
    if kind in "IUiuLl":
        return int(s.lstrip("+")) if s.lstrip("+-").isdigit() else s
    if kind == "D":
        return float(s)
    if kind == "d":
        return [float(x) for x in s.strip("{}").split(",") if x]
    if kind == "F":                     # FileInPath: "V001 <relative path> <n> <absolute>"
        parts = s.split()
        return parts[1] if len(parts) > 1 else s
    return s                            # InputTag (t), nested sets (P), ...


def module_params(blob, label):
    """{parameter: value} of the module `label`'s top-level parameters (nested
    parameter sets are kept as raw strings). None when the label is absent."""
    k = blob.find(b"@module_label=+Z(" + label.encode() + b"\x00)")
    if k < 0:
        return None
    rec = _record(blob, k)
    out = {}
    for m in _PARAM_RE.finditer(rec):
        name = m.group(1).decode()
        if name not in out:
            out[name] = _decode(m.group(2).decode(), m.group(3))
    return out


def flat_table_src(blob, table):
    """Label of the producer whose collection the flat table `table` is written from."""
    m = re.search(rb"@module_type=\+Z\(L1TTTrackFlatTableProducer[\x00 ]?\);[^>]*?name=\+Z\("
                  + re.escape(table.encode()) + rb"[\x00 ]?\);[^>]*?src=\+t\(([^:)]+)", blob)
    return m.group(1).decode() if m else None


def producer_params(blob, track_table):
    """Parameters of the L1SmartPixelsTrackProducer behind `track_table`, or None."""
    lab = flat_table_src(blob, track_table)
    return (lab, module_params(blob, lab)) if lab else (None, None)


def rechit_params(blob):
    """(label, parameters) of the SmartPixelsRecHitProducer (angle source, emu tables)."""
    m = re.search(rb"@module_label=\+Z\(([^\x00)]+)\x00\);@module_type=\+Z\(SmartPixelsRecHitProducer", blob)
    if not m:
        return None, None
    lab = m.group(1).decode()
    return lab, module_params(blob, lab)


# producer parameter -> ot_refit_projection.PRODUCER_DEFAULTS key
_REFIT_KEYS = {"seedlessRefitUseAngles": "useAngles", "seedlessRefitMaxKFUpdates": "maxKFUpdates",
               "seedlessRefitMaxHitsPerWindow": "maxHitsPerWindow",
               "seedlessRefitWindowRPhi": "windowRPhi", "seedlessRefitWindowZ": "windowZ",
               "seedlessRefitWindowDiscRPhi": "windowDiscRPhi", "seedlessRefitWindowDiscR": "windowDiscR",
               "seedlessRefitMultScattTerm": "multScattTerm",
               "seedlessRefitApplyProcessNoise": "applyProcessNoise",
               "seedlessRefitLayerOrder": "layerOrder"}


def refit_producer_config(params):
    """The refit configuration (ot_refit_projection.PRODUCER_DEFAULTS keys) of a
    producer's parameters; keys the producer does not carry are omitted."""
    if not params:
        return {}
    return {k: (tuple(params[p]) if isinstance(params[p], list) else params[p])
            for p, k in _REFIT_KEYS.items() if p in params}


def resolve_data_path(rel, search):
    """An absolute path for a CMSSW FileInPath-style relative path, trying each
    directory of `search` (e.g. a CMSSW src). None when not found."""
    if not rel:
        return None
    if os.path.isabs(rel) and os.path.exists(rel):
        return rel
    for d in search:
        p = os.path.join(d, rel)
        if os.path.exists(p):
            return p
    return None


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def git_state(path):
    """{commit, dirty} of the git work tree holding `path` (dirty: that file changed)."""
    d = os.path.dirname(os.path.abspath(path))
    try:
        c = subprocess.run(["git", "-C", d, "rev-parse", "--short=12", "HEAD"],
                           capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", d, "status", "--porcelain", "--", d],
                               capture_output=True, text=True).stdout
        return {"commit": c or None,
                "dirty_files": [l[3:] for l in dirty.splitlines() if l.strip() and l[:2] != "??"]}
    except OSError:
        return {"commit": None, "dirty_files": []}


def peak_memory_gb():
    """(peak RSS, peak footprint) of this process so far, in GiB.

    On macOS the RSS peak undercounts under memory pressure: pages the kernel
    compresses or swaps leave the resident set but still occupy the machine. The
    lifetime peak "physical footprint" (libproc, the number `/usr/bin/time -l`
    prints as "peak memory footprint") counts them; it is None elsewhere."""
    import resource
    import sys
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 30 if sys.platform == "darwin" else 1 << 20)
    if sys.platform != "darwin":
        return rss, None
    import ctypes
    buf = (ctypes.c_uint64 * 64)()          # struct rusage_info_v4 (16-byte uuid, then uint64 fields)
    try:
        if ctypes.CDLL("/usr/lib/libproc.dylib").proc_pid_rusage(os.getpid(), 4, ctypes.byref(buf)) != 0:
            return rss, None
    except OSError:
        return rss, None
    return rss, buf[2 + 28] / (1 << 30)    # ri_lifetime_max_phys_footprint


def file_info(paths):
    out = []
    for p in paths:
        st = os.stat(p)
        out.append({"path": os.path.abspath(p), "bytes": st.st_size,
                    "mtime": datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")})
    return out


# --------------------------------------------------------------------------
# the emulator sigma table
# --------------------------------------------------------------------------
class SigmaTable:
    """The SmartPixels-output emulator's angle-resolution tables (correctionlib JSON,
    e.g. spix_emu_sigma_v2.json): sigma(surface, |reported cot|) per angle, binned in
    |cot| (clamped to the last bin above the top edge)."""

    NAMES = {"alpha": "spix_emu_alpha_sigma", "beta": "spix_emu_beta_sigma"}

    def __init__(self, path):
        self.path = path
        with open(path) as fh:
            J = json.load(fh)
        self.description = J.get("description")
        self.tables = {}
        for c in J["corrections"]:
            ang = next((a for a, n in self.NAMES.items() if c["name"] == n), None)
            if ang is None:
                continue
            per = {}
            for item in c["data"]["content"]:
                node = item["value"]
                per[int(item["key"])] = (np.asarray(node["edges"], float), np.asarray(node["content"], float))
            self.tables[ang] = per
        missing = set(self.NAMES) - set(self.tables)
        if missing:
            raise ValueError(f"{path}: no {sorted(missing)} sigma correction")

    def bins(self, angle, surface):
        """(edges, sigma per bin) of `angle` ("alpha"/"beta") on `surface`."""
        return self.tables[angle][int(surface)]

    def evaluate(self, angle, surface, abscot):
        surface = np.asarray(surface)
        abscot = np.asarray(abscot, float)
        out = np.full(abscot.shape, np.nan)
        for s in np.unique(surface):
            if int(s) not in self.tables[angle]:
                continue
            e, v = self.tables[angle][int(s)]
            m = surface == s
            i = np.clip(np.searchsorted(e, abscot[m], side="right") - 1, 0, len(v) - 1)
            out[m] = v[i]
        return out


# --------------------------------------------------------------------------
# report helpers (vector PDF, legends off the data, table pages)
# --------------------------------------------------------------------------
def legend_below(ax, ncol=2, fontsize=6, **kw):
    """One legend for `ax`, placed BELOW the axes (never over the data)."""
    h, l = ax.get_legend_handles_labels()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    if not h:
        return None
    return ax.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=ncol,
                     fontsize=fontsize, frameon=False, borderaxespad=0.0)


def legends_outside(axes, fontsize=None):
    """Move every legend drawn inside one of `axes` below its axes (same entries)."""
    for ax in np.ravel(axes):
        lg = ax.get_legend()
        if lg is None:
            continue
        fs = fontsize or (lg.get_texts()[0].get_fontsize() if lg.get_texts() else 6)
        n = len(lg.get_texts())
        longest = max((len(t.get_text()) for t in lg.get_texts()), default=0)
        ncol = 1 if longest > 45 else 2 if longest > 22 else 3
        legend_below(ax, ncol=min(ncol, max(1, n)), fontsize=fs)


def table_pages(pdf, title, header, rows, notes=(), per_page=36, fontsize=6.8, highlight=None):
    """Vector table page(s) into an open PdfPages: `rows` (lists of cells, already
    formatted), a title, and notes under the title. highlight(row) -> bool shades a row."""
    import matplotlib.pyplot as plt
    rows = list(rows) or [["(no rows)"] + [""] * (len(header) - 1)]
    for p0 in range(0, len(rows), per_page):
        part = rows[p0:p0 + per_page]
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.text(0.03, 0.975, title + (f"   (page {p0 // per_page + 1})" if len(rows) > per_page else ""),
                 fontsize=11.5, fontweight="bold", va="top")
        if notes:
            fig.text(0.03, 0.945, "\n".join(notes), fontsize=6.6, va="top", family="monospace")
        top = 0.93 - 0.0125 * len(notes)
        ax = fig.add_axes([0.03, 0.03, 0.94, max(top - 0.04, 0.2)])
        ax.axis("off")
        cells = [[str(c) for c in r] for r in part]
        tb = ax.table(cellText=cells, colLabels=header, loc="upper left", cellLoc="right")
        tb.auto_set_font_size(False)
        tb.set_fontsize(fontsize)
        tb.scale(1.0, 1.12)
        # column widths in proportion to the longest entry, filling the page width
        chars = [max([len(str(header[j]))] + [len(r[j]) for r in cells if j < len(r)]) + 2
                 for j in range(len(header))]
        for (i, j), cell in tb.get_celld().items():
            cell.set_width(chars[j] / sum(chars))
        for i, r in enumerate(part, start=1):
            tb[i, 0].set_text_props(ha="left")
            if highlight is not None and highlight(r):
                for j in range(len(header)):
                    tb[i, j].set_facecolor("#f6d2cc")
        pdf.savefig(fig)
        plt.close(fig)


def fmt(v, f="{:.3f}"):
    """A table cell: '-' for None/NaN."""
    if v is None:
        return "-"
    try:
        if not np.isfinite(v):
            return "-"
    except TypeError:
        return str(v)
    return f.format(v)


def jsonable(o):
    """Numpy scalars/arrays -> python, for json.dump."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o
