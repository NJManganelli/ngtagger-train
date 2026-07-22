"""Generic correctionlib schema-v2 ingester for the MVA explorer's regression
panel.

Strategy: never re-implement the evaluator — introspect each correction's JSON
to derive a dense evaluation grid over its *real* inputs (bin edges harvested
from binning/multibinning nodes anywhere in the tree; sensible linspace with
CLI override when an input is only used in formulas), enumerate int/category
inputs as categorical axes, then evaluate with the `correctionlib` python
package on the grid and store the quantized cube.  This handles every node
type (binning, multibinning, category, formula, formularef, transform) by
construction.

Corrections containing non-deterministic nodes (hashprng) cannot be rendered
as a static grid; they are skipped and recorded in the meta, as are compound
corrections whose stack includes such a node (their physical sub-corrections
are exported through the normal path).

EXCEPTION — the deterministic synthesis ENVELOPE: a compound of exactly the
shape [sigma-like deterministic correction, hashprng-stdnormal] with
output_op "*" (the SmartPixels smear factorization) is recorded in the meta
as an `envelopes` entry referencing the exported sigma grid (and the matching
`*_bias` grid when present), so the browser can render bias ± {1,2}·sigma
bands ("synthesis envelope, throw ~ N(bias, sigma) via HashPRNG") without
ever evaluating the raw hash noise.  HashPRNG structures that do not match
this shape keep the plain skip path.
"""

from __future__ import annotations

import gzip
import json
import os

import numpy as np


# --------------------------------------------------------------------------
# JSON loading / tree walking
# --------------------------------------------------------------------------

def load_cset_json(path: str) -> dict:
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f)


def _iter_nodes(node):
    """Yield every dict node (depth-first) in a correction data tree."""
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _iter_nodes(v)
    elif isinstance(node, list):
        for v in node:
            yield from _iter_nodes(v)


def node_types(data) -> set[str]:
    return {n["nodetype"] for n in _iter_nodes(data) if "nodetype" in n}


def _formula_variables(node, corr) -> list[str]:
    if node["nodetype"] == "formula":
        return list(node.get("variables") or [])
    # formularef: variables live on the referenced Correction.generic_formulas
    gf = (corr or {}).get("generic_formulas") or []
    idx = node.get("index", -1)
    if 0 <= idx < len(gf):
        return list(gf[idx].get("variables") or [])
    return []


def introspect_correction(corr: dict, cset_json: dict | None = None) -> dict:
    """Walk one correction and report, per input name:
    edges (sorted union of all binning/multibinning edges keyed on it),
    categories (all category keys), used (appears in any node at all),
    transformed (an upstream `transform` node rewrites it, e.g. |eta|)."""
    edges: dict[str, list[np.ndarray]] = {}
    cats: dict[str, list] = {}
    used: set[str] = set()
    transformed: set[str] = set()
    for n in _iter_nodes(corr["data"]):
        nt = n.get("nodetype")
        if nt == "binning":
            used.add(n["input"])
            edges.setdefault(n["input"], []).append(np.asarray(n["edges"], float))
        elif nt == "multibinning":
            for name, e in zip(n["inputs"], n["edges"]):
                used.add(name)
                edges.setdefault(name, []).append(np.asarray(e, float))
        elif nt == "category":
            used.add(n["input"])
            keys = [c["key"] for c in n["content"]]
            cats.setdefault(n["input"], [])
            for k in keys:
                if k not in cats[n["input"]]:
                    cats[n["input"]].append(k)
        elif nt == "transform":
            used.add(n["input"])
            transformed.add(n["input"])
        elif nt in ("formula", "formularef"):
            used.update(_formula_variables(n, corr))
    merged = {k: _merge_edges(v) for k, v in edges.items()}
    return {"edges": merged, "categories": cats, "used": used,
            "transformed": transformed}


def _merge_edges(edge_lists: list[np.ndarray], tol: float = 1e-9) -> np.ndarray:
    """Sorted union of several edge vectors (deduped with tolerance)."""
    allv = np.sort(np.concatenate(edge_lists))
    keep = np.ones(len(allv), bool)
    keep[1:] = np.diff(allv) > tol * (1.0 + np.abs(allv[1:]))
    return allv[keep]


def centers_of(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (edges[:-1] + edges[1:])


def has_hashprng(corr: dict) -> bool:
    return "hashprng" in node_types(corr["data"])


# --------------------------------------------------------------------------
# Grid construction + evaluation
# --------------------------------------------------------------------------

DEFAULT_LINSPACE = (0.0, 1.0, 25)


def build_axes(corr: dict, cset_json: dict | None = None,
               linspace_overrides: dict | None = None) -> tuple[list[dict], dict]:
    """Return (axes, fixed_inputs).

    axes: ordered like corr['inputs'], one entry per *active* input:
      real: {"name", "kind": "real", "edges": [...], "centers": [...],
             "transformed": bool, "auto_linspace": bool}
      cat:  {"name", "kind": "cat", "categories": [...]}
    fixed_inputs: {name: value} for declared inputs that never influence the
    output (evaluated at a nominal fixed point, excluded from the grid).
    """
    info = introspect_correction(corr, cset_json)
    overrides = linspace_overrides or {}
    axes: list[dict] = []
    fixed: dict = {}
    for inp in corr["inputs"]:
        name, typ = inp["name"], inp["type"]
        if typ in ("int", "string"):
            if name in info["categories"]:
                axes.append({"name": name, "kind": "cat",
                             "categories": info["categories"][name]})
            elif name in info["used"]:
                axes.append({"name": name, "kind": "cat", "categories": [0]})
            else:
                fixed[name] = 0 if typ == "int" else ""
            continue
        # real input
        if name not in info["used"]:
            fixed[name] = 0.0
            continue
        auto = False
        if name in overrides:
            lo, hi, n = overrides[name]
            edges = np.linspace(lo, hi, int(n) + 1)
        elif name in info["edges"]:
            edges = info["edges"][name]
        else:  # used only in formulas: no native binning to harvest
            lo, hi, n = DEFAULT_LINSPACE
            edges = np.linspace(lo, hi, int(n) + 1)
            auto = True
        axes.append({"name": name, "kind": "real",
                     "edges": [float(x) for x in edges],
                     "centers": [float(x) for x in centers_of(np.asarray(edges))],
                     "transformed": name in info["transformed"],
                     "auto_linspace": auto})
    return axes, fixed


def _axis_points(ax: dict) -> np.ndarray:
    if ax["kind"] == "cat":
        return np.asarray(ax["categories"])
    return np.asarray(ax["centers"], float)


def evaluate_on_grid(evaluator, corr: dict, axes: list[dict], fixed: dict) -> np.ndarray:
    """Evaluate a correctionlib Correction on the dense grid defined by axes.
    Returns array of shape tuple(len(points) per axis) in axes order."""
    pts = [_axis_points(a) for a in axes]
    shape = tuple(len(p) for p in pts)
    mesh = np.meshgrid(*pts, indexing="ij") if pts else []
    flat = {a["name"]: m.ravel() for a, m in zip(axes, mesh)}
    n = int(np.prod(shape)) if shape else 1
    args = []
    for inp in corr["inputs"]:
        name, typ = inp["name"], inp["type"]
        if name in flat:
            col = flat[name]
            if typ == "int":
                col = col.astype(np.int64)
            elif typ == "real":
                col = col.astype(np.float64)
            args.append(col)
        else:
            fv = fixed.get(name, 0)
            if typ == "int":
                args.append(np.full(n, int(fv), dtype=np.int64))
            elif typ == "string":
                args.append(str(fv))
            else:
                args.append(np.full(n, float(fv), dtype=np.float64))
    out = np.asarray(evaluator.evaluate(*args), dtype=np.float64)
    return out.reshape(shape) if shape else out.reshape(())


# --------------------------------------------------------------------------
# Whole-file generic export
# --------------------------------------------------------------------------

def export_generic_dataset(json_path: str, dataset_id: str, title: str,
                           out_dir: str, include=None,
                           linspace_overrides: dict | None = None,
                           scale: int | None = None) -> dict:
    """Evaluate every (deterministic) correction in a schema-v2 file on its
    native grid and pack them into one <dataset_id>.bin + meta dict.

    include: optional predicate/name-list restricting which corrections export.
    Returns the meta dict (also written to <dataset_id>_meta.json in out_dir).
    """
    import correctionlib

    from ngtagger.viz.mva_explorer.quantize import LOG10_SCALE, choose_block

    scale = scale or LOG10_SCALE
    cset_json = load_cset_json(json_path)
    cset = correctionlib.CorrectionSet.from_file(json_path)

    if include is None:
        selector = lambda name: True  # noqa: E731
    elif callable(include):
        selector = include
    else:
        wanted = set(include)
        selector = lambda name: name in wanted  # noqa: E731

    blobs: list[bytes] = []
    corr_meta: list[dict] = []
    skipped: list[dict] = []
    offset = 0
    for corr in cset_json.get("corrections", []):
        name = corr["name"]
        if not selector(name):
            continue
        if has_hashprng(corr):
            skipped.append({"name": name,
                            "reason": "contains non-deterministic hashprng node"})
            continue
        axes, fixed = build_axes(corr, cset_json, linspace_overrides)
        values = evaluate_on_grid(cset[name], corr, axes, fixed)
        block, quant = choose_block(values, scale)
        pad = (-offset) % 4  # keep every block 4-byte aligned for JS views
        blobs.append(b"\x00" * pad + block)
        offset += pad
        corr_meta.append({
            "name": name,
            "description": corr.get("description") or "",
            "output": corr.get("output", {}).get("name", "value"),
            "shape": list(values.shape),
            "axes": axes,
            "fixed_inputs": fixed,
            "quant": quant,
            "scale": scale,
            "byte_offset": offset,
            "n_values": int(values.size),
        })
        offset += len(block)

    envelopes: list[dict] = []
    exported = {c["name"] for c in corr_meta}
    by_name = {c["name"]: c for c in cset_json.get("corrections", [])}
    for comp in cset_json.get("compound_corrections", []) or []:
        stack = comp.get("stack", [])
        bad = [s for s in stack if s in by_name and has_hashprng(by_name[s])]
        if not bad:
            continue
        det = [s for s in stack if s not in bad]
        # deterministic-envelope shape: [sigma-like, hashprng stdnormal], output_op "*"
        if (len(stack) == 2 and len(bad) == 1 and len(det) == 1
                and comp.get("output_op") == "*"
                and by_name[bad[0]]["data"].get("distribution") == "stdnormal"
                and det[0] in exported):
            sigma_name = det[0]
            bias_name = (sigma_name.replace("_sigma", "_bias")
                         if "_sigma" in sigma_name else None)
            if bias_name not in exported:
                bias_name = None
            envelopes.append({
                "name": comp["name"],
                "sigma": sigma_name,
                "bias": bias_name,
                "prng": bad[0],
                "description": comp.get("description") or "",
                "label": "synthesis envelope, throw ~ N(bias, sigma) via HashPRNG",
            })
        else:
            skipped.append({"name": comp["name"],
                            "reason": "compound stack includes non-deterministic "
                                      f"node(s): {', '.join(bad)}; its physical "
                                      "sub-corrections are exported individually"})

    meta = {
        "id": dataset_id, "title": title, "type": "generic",
        "source": os.path.basename(json_path),
        "file": f"{dataset_id}.bin",
        "corrections": corr_meta,
        "envelopes": envelopes,
        "skipped": skipped,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{dataset_id}.bin"), "wb") as f:
        for b in blobs:
            f.write(b)
    with open(os.path.join(out_dir, f"{dataset_id}_meta.json"), "w") as f:
        json.dump(meta, f)
    return meta


# --------------------------------------------------------------------------
# Structured (configs x params) export for the smearing presets
# --------------------------------------------------------------------------

def export_structured_smear(json_path: str, dataset_id: str, title: str,
                            out_dir: str, params: list[str] | None = None,
                            kinds: dict | None = None,
                            scale: int | None = None) -> dict:
    """Preset path for the SPIX smearing payloads ({param}_smear_{config} and
    {param}_relative_smear_{config} on a shared TP grid): a single cube of
    shape (nkind, nconfig, nparam, *grid) so the browser can form any
    config-vs-config ratio exactly as eval_spixel's builder does.

    kinds: {"sigma": "{p}_smear_{c}", "relative": "{p}_relative_smear_{c}"}
    Configs use the canonical combinatoric order (0000 first).
    """
    import correctionlib

    from ngtagger.viz.mva_explorer import canonical_config_order
    from ngtagger.viz.mva_explorer.quantize import LOG10_SCALE, quantize_log10_int16

    scale = scale or LOG10_SCALE
    params = params or ["d0", "z0", "pt", "eta", "phi"]
    kinds = kinds or {"sigma": "{p}_smear_{c}",
                      "relative": "{p}_relative_smear_{c}"}
    configs = canonical_config_order()

    cset_json = load_cset_json(json_path)
    by_name = {c["name"]: c for c in cset_json["corrections"]}
    cset = correctionlib.CorrectionSet.from_file(json_path)

    # shared grid: union of harvested edges over every member correction
    kind_names = list(kinds)
    member_names = [kinds[k].format(p=p, c=c)
                    for k in kind_names for p in params for c in configs]
    missing = [n for n in member_names if n not in by_name]
    if missing:
        raise KeyError(f"{json_path}: missing corrections e.g. {missing[:4]}")

    per_input_edges: dict[str, list[np.ndarray]] = {}
    input_order: list[str] = []
    transformed: set[str] = set()
    for n in member_names:
        info = introspect_correction(by_name[n], cset_json)
        for iname, e in info["edges"].items():
            per_input_edges.setdefault(iname, []).append(e)
            if iname not in input_order:
                input_order.append(iname)
        transformed |= info["transformed"]
    axes = []
    for iname in input_order:
        edges = _merge_edges(per_input_edges[iname])
        axes.append({"name": iname, "kind": "real",
                     "edges": [float(x) for x in edges],
                     "centers": [float(x) for x in centers_of(edges)],
                     "transformed": iname in transformed,
                     "auto_linspace": False})
    grid_shape = tuple(len(a["centers"]) for a in axes)
    mesh = np.meshgrid(*[np.asarray(a["centers"]) for a in axes], indexing="ij")
    flat = {a["name"]: m.ravel() for a, m in zip(axes, mesh)}

    cube = np.empty((len(kind_names), len(configs), len(params)) + grid_shape)
    for ik, kind in enumerate(kind_names):
        for ic, cfg in enumerate(configs):
            for ip, par in enumerate(params):
                name = kinds[kind].format(p=par, c=cfg)
                corr = cset[name]
                args = []
                for inp in by_name[name]["inputs"]:
                    if inp["name"] in flat:
                        args.append(flat[inp["name"]])
                    else:  # declared but never binned (e.g. phi_tp): nominal 0
                        args.append(np.zeros(mesh[0].size))
                cube[ik, ic, ip] = np.asarray(
                    corr.evaluate(*args)).reshape(grid_shape)

    if not np.all(np.isfinite(cube)):
        raise ValueError(f"{dataset_id}: non-finite values in smearing cube")
    if np.min(cube) > 0:
        q, quant = quantize_log10_int16(cube, scale), "log10_i16"
    else:  # e.g. CalV1 carries exact-0 sigmas in uncovered bins
        q, quant = cube.astype("<f4"), "f32"

    meta = {
        "id": dataset_id, "title": title, "type": "structured",
        "source": os.path.basename(json_path),
        "file": f"{dataset_id}.bin",
        "quant": quant, "scale": scale,
        "shape": list(q.shape),
        "kinds": kind_names,
        "configs": configs,
        "params": params,
        "axes": axes,
    }
    os.makedirs(out_dir, exist_ok=True)
    q.tofile(os.path.join(out_dir, f"{dataset_id}.bin"))
    with open(os.path.join(out_dir, f"{dataset_id}_meta.json"), "w") as f:
        json.dump(meta, f)
    return meta
