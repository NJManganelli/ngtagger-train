"""The study's definition sheet (working rule 8) and its reference implementations.

    from ngtagger.definitions import SHEET_VERSION, load_sheet, item
    item("fom.harm")["params"]          # {'d0_mm': 0.5, 'z0_mm': 1.0}

    python -m ngtagger.definitions render [VERSION]       # the short human rendering (sheet_<v>.md)
    python -m ngtagger.definitions render-full [VERSION]  # every definition in full
    python -m ngtagger.definitions check  [VERSION]     # structure check

Modules: estimators (the B.1 core), helix (POCA conventions), matched_wp (tie-split working
points), puppi_dxy (the PF/PUPPI dxy code). Nothing here is imported by ngtagger/__init__.py.
"""
from __future__ import annotations

import os

SHEET_VERSION = "v2.2"
HERE = os.path.dirname(os.path.abspath(__file__))
REQUIRED = ("id", "title", "definition", "status")


def sheet_path(version: str = SHEET_VERSION) -> str:
    return os.path.join(HERE, f"sheet_{version}.yaml")


def load_sheet(version: str = SHEET_VERSION) -> dict:
    import yaml

    with open(sheet_path(version)) as f:
        return yaml.safe_load(f)


def items(version: str = SHEET_VERSION) -> dict:
    return {it["id"]: it for sec in load_sheet(version)["sections"] for it in sec["items"]}


def item(item_id: str, version: str = SHEET_VERSION) -> dict:
    return items(version)[item_id]


def check(version: str = SHEET_VERSION) -> list[str]:
    s = load_sheet(version)
    errs, seen = [], set()
    if s.get("version") != version:
        errs.append(f"version field {s.get('version')!r} != {version}")
    for sec in s["sections"]:
        for it in sec["items"]:
            for k in REQUIRED:
                if not str(it.get(k, "")).strip():
                    errs.append(f"{it.get('id')}: missing {k}")
            if it["id"] in seen:
                errs.append(f"duplicate id {it['id']}")
            seen.add(it["id"])
    return errs


def render_markdown(version: str = SHEET_VERSION, full: bool = False) -> str:
    """The human rendering: one line per item (default), or every definition in full."""
    s = load_sheet(version)
    out = [f"# Definition sheet {s['version']} ({s['date']})", "",
           f"One line per definition; the full text, parameters and code pointers are in `sheet_{version}.yaml` "
           "(`python -m ngtagger.definitions render-full`). Generated: edit the YAML, never this file. "
           "Amendments get a new version; every registered result names the version it used.", "",
           "Sources: " + "; ".join(s["sources"]) + ".", ""]
    for sec in s["sections"]:
        out += [f"## {sec['title']}", ""]
        for it in sec["items"]:
            st = it["status"]
            tag = "" if st == "adopted" else f" *[{st}]*"
            body = it["definition"].strip() if full else it.get("summary", it["definition"]).strip()
            out.append(f"- **{it['title']}** (`{it['id']}`){tag}: {body}")
            if full:
                if it.get("params"):
                    out.append("  Parameters: " + "; ".join(f"{k} = {v}" for k, v in it["params"].items()) + ".")
                if it.get("implementation"):
                    out.append("  Code: " + "; ".join(f"`{x}`" for x in it["implementation"]) + ".")
                if it.get("source"):
                    out.append(f"  Source: {it['source']}.")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def main(argv=None) -> int:
    import sys

    a = list(sys.argv[1:] if argv is None else argv)
    what = a[0] if a else "render"
    version = a[1] if len(a) > 1 else SHEET_VERSION
    if what in ("render", "render-full"):
        sys.stdout.write(render_markdown(version, full=what == "render-full"))
        return 0
    if what == "check":
        errs = check(version)
        for e in errs:
            print("ERROR", e)
        print(f"{len(items(version))} items, {len(errs)} problems")
        return 1 if errs else 0
    print(__doc__)
    return 2
