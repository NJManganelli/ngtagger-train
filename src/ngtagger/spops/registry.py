"""The results registry (working rule 6; plan App. C.3-2, C.3-5, D.1).

One JSON file per result in <registry>/results/ (no shared file, so concurrent writers
cannot lose each other's updates; a status change rewrites one file under its own lock).
Each result carries:

  values        [{name, value, error, unit, note}]   value +- error per number
  status        one of STATUSES, with status_note (the reason; required where it matters)
  bias          the bias line (direction and why), as a report must lead with it (C.3-8)
  provenance    inputs with checksums, builds, commits, the tree state with the stored
                patch, the frozen-copy manifest hash, the job id -- from the launcher's
                launch record when the result was produced by a launched job (captured at
                launch, not at registration); "known": false for historical numbers
  definition_sheet   the definition-sheet version the result uses
  history       every status change, appended

<registry>/provenance/<manifest hash>/ holds the manifest and tree.patch of each frozen
copy; <registry>/gates/ the fast-gate results per manifest hash.

    python -m ngtagger.spops.registry list [--status S]
    python -m ngtagger.spops.registry show ID
    python -m ngtagger.spops.registry set-status ID STATUS --note TEXT [--superseded-by ID]
    python -m ngtagger.spops.registry validate
    python -m ngtagger.spops.registry seed FILE.yaml        (historical registrations)
    python -m ngtagger.spops.registry table                 (markdown summary)

Stdlib only, except `seed`, which reads YAML.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import sys
import time

REGISTRY_DIR = os.environ.get("SP_REGISTRY", os.path.join(os.path.expanduser("~"), "smartpixels", "sp_registry"))
SCHEMA = "sp-registry-result/1"
STATUSES = {
    "valid": "measured with known provenance and no known defect",
    "void": "wrong: must not be used (status_note says why)",
    "pathology-affected": "affected by a known pathology (e.g. collapsed trainings) of unknown size",
    "superseded": "replaced by a later measurement or definition (status_note / superseded_by)",
    "refit-v2": "valid for CMSSW refit v2 only; reworked if refit v3 goes ahead",
    "payload-era": "made with the payload-era angle model; not valid for the emulated angles",
    "in-sample": "measured on (part of) the sample its inputs were fitted or trained on",
    "exploratory": "indicative only: provenance, statistics or method not at decision level",
    "not-used": "a correct measurement that is not used for decisions (user decision; status_note)",
}
NOTE_REQUIRED = ("void", "superseded", "pathology-affected", "not-used", "in-sample", "exploratory")
ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,120}$")


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _dirs(reg: str | None) -> dict:
    reg = reg or REGISTRY_DIR
    d = {k: os.path.join(reg, k) for k in ("results", "provenance", "gates", ".locks")}
    for p in d.values():
        os.makedirs(p, exist_ok=True)
    d["root"] = reg
    return d


def sheet_version() -> str | None:
    try:
        from ngtagger.definitions import SHEET_VERSION

        return SHEET_VERSION
    except Exception:
        return None


class RegistryError(ValueError):
    pass


def validate_entry(e: dict) -> list[str]:
    errs = []
    if not ID_RE.match(str(e.get("id", ""))):
        errs.append(f"id {e.get('id')!r} must match {ID_RE.pattern}")
    for k in ("title", "status", "bias"):
        if not str(e.get(k, "")).strip():
            errs.append(f"{e.get('id')}: missing {k}")
    if e.get("status") not in STATUSES:
        errs.append(f"{e.get('id')}: status {e.get('status')!r} not in {sorted(STATUSES)}")
    if e.get("status") in NOTE_REQUIRED and not str(e.get("status_note", "")).strip():
        errs.append(f"{e.get('id')}: status {e.get('status')} needs a status_note")
    vals = e.get("values")
    if not isinstance(vals, list) or not vals:
        errs.append(f"{e.get('id')}: values must be a non-empty list")
    else:
        for v in vals:
            if not isinstance(v, dict) or "name" not in v or "value" not in v:
                errs.append(f"{e.get('id')}: every value needs name and value ({v!r})")
    prov = e.get("provenance")
    if not isinstance(prov, dict) or "known" not in prov:
        errs.append(f"{e.get('id')}: provenance must be a dict with 'known'")
    return errs


def provenance_from_job(job_dir: str) -> dict:
    """The launch-time record of a launched job (launch.json), reduced to provenance."""
    with open(os.path.join(job_dir, "launch.json")) as f:
        L = json.load(f)
    ts = L.get("tree_state", {})
    return {"known": True, "job_id": L.get("job_id"), "job_dir": job_dir, "launched": L.get("launched"),
            "command": L.get("command"), "manifest_hash": L.get("manifest_hash"),
            "commits": {"ngtagger-train": ts.get("commit")}, "tree_state": ts.get("state"),
            "patch": (L.get("provenance_store") or {}).get("patch") or ts.get("patch"),
            "patch_sha256": ts.get("patch_sha256"), "modified_files": ts.get("modified_files"),
            "inputs": [{k: d.get(k) for k in ("declared", "bytes", "mtime", "sha256", "sha256-partial") if k in d}
                       for d in L.get("data", [])],
            "prereg": L.get("prereg")}


def _lock(d: dict, rid: str):
    fd = os.open(os.path.join(d[".locks"], rid + ".lock"), os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def _write(path: str, obj: dict, exclusive: bool) -> None:
    data = (json.dumps(obj, indent=1, default=str) + "\n").encode()
    if exclusive:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def path_of(rid: str, reg: str | None = None) -> str:
    return os.path.join(_dirs(reg)["results"], rid + ".json")


def register(entry: dict, reg: str | None = None, job_dir: str | None = None, overwrite: bool = False,
             by: str | None = None) -> str:
    """Write one result. `job_dir` (default $SP_JOB_DIR) attaches the launcher's launch-time
    provenance. Refuses to overwrite an existing id unless `overwrite`."""
    d = _dirs(reg)
    e = json.loads(json.dumps(entry, default=str))
    e.setdefault("schema", SCHEMA)
    by = by or os.environ.get("SP_REGISTERED_BY") or os.environ.get("USER", "?")
    if job_dir is None:
        job_dir = os.environ.get("SP_JOB_DIR")
    if job_dir and os.path.exists(os.path.join(job_dir, "launch.json")):
        e["provenance"] = {**e.get("provenance", {}), **provenance_from_job(job_dir)}
    e.setdefault("provenance", {"known": False})
    e.setdefault("definition_sheet", sheet_version())
    e["registered"] = now_utc()
    e["registered_by"] = by
    e.setdefault("history", []).append({"t": e["registered"], "status": e.get("status"),
                                        "note": e.get("status_note"), "by": by, "what": "registered"})
    errs = validate_entry(e)
    if errs:
        raise RegistryError("; ".join(errs))
    p = os.path.join(d["results"], e["id"] + ".json")
    fd = _lock(d, e["id"])
    try:
        if os.path.exists(p) and not overwrite:
            raise RegistryError(f"{e['id']} is already registered ({p}); use set-status or overwrite")
        _write(p, e, exclusive=not os.path.exists(p))
    finally:
        os.close(fd)
    return p


def get(rid: str, reg: str | None = None) -> dict:
    with open(path_of(rid, reg)) as f:
        return json.load(f)


def set_status(rid: str, status: str, note: str, reg: str | None = None, by: str | None = None,
               superseded_by: str | None = None) -> dict:
    if status not in STATUSES:
        raise RegistryError(f"status {status!r} not in {sorted(STATUSES)}")
    d = _dirs(reg)
    by = by or os.environ.get("SP_REGISTERED_BY") or os.environ.get("USER", "?")
    fd = _lock(d, rid)
    try:
        e = get(rid, reg)
        e["status"], e["status_note"] = status, note
        if superseded_by:
            e["superseded_by"] = sorted(set(e.get("superseded_by", []) + [superseded_by]))
        e.setdefault("history", []).append({"t": now_utc(), "status": status, "note": note, "by": by,
                                            "what": "status change"})
        errs = validate_entry(e)
        if errs:
            raise RegistryError("; ".join(errs))
        _write(path_of(rid, reg), e, exclusive=False)
    finally:
        os.close(fd)
    return e


def iter_results(reg: str | None = None) -> list[dict]:
    d = _dirs(reg)
    out = []
    for fn in sorted(os.listdir(d["results"])):
        if fn.endswith(".json"):
            with open(os.path.join(d["results"], fn)) as f:
                out.append(json.load(f))
    return out


def store_provenance(reg: str | None, manifest_hash: str, manifest_path: str, patch_path: str,
                     tree_state: dict, untracked_tar: str | None = None) -> dict:
    """Keep a frozen copy's manifest, its launch-time patch and copies of its untracked files
    under provenance/<manifest hash>/ (the job directory may be cleaned; these KB may not)."""
    d = _dirs(reg)
    dst = os.path.join(d["provenance"], manifest_hash)
    os.makedirs(dst, exist_ok=True)
    out = {}
    items = [(manifest_path, "manifest.json", "manifest"),
             (patch_path, f"tree_{tree_state.get('patch_sha256', 'x')[:16]}.patch", "patch")]
    if untracked_tar:
        items.append((untracked_tar, "untracked.tar.gz", "untracked"))
    for src, name, key in items:
        p = os.path.join(dst, name)
        if not os.path.exists(p):
            shutil.copy2(src, p)
        out[key] = p
    return out


def store_gate(reg: str | None, manifest_hash: str, result: dict) -> str:
    d = _dirs(reg)
    p = os.path.join(d["gates"], f"{manifest_hash}.json")
    _write(p, result, exclusive=False)
    return p


# ---------------------------------------------------------------- CLI
def _fmt_value(v: dict) -> str:
    s = f"{v['value']}"
    if v.get("error") not in (None, ""):
        s += f" +- {v['error']}"
    if v.get("unit"):
        s += f" {v['unit']}"
    return f"{v['name']}: {s}"


def cmd_seed(a) -> int:
    import yaml

    with open(a.file) as f:
        doc = yaml.safe_load(f)
    common = doc.get("common", {})
    if a.definition_sheet:
        common.setdefault("definition_sheet", a.definition_sheet)
    n_new = n_skip = 0
    errors = []
    for item in doc["results"]:
        e = {**json.loads(json.dumps(common)), **item}
        e["provenance"] = {**common.get("provenance", {}), **item.get("provenance", {})}
        try:
            register(e, a.registry_dir, job_dir="", overwrite=a.overwrite, by=a.by)
            n_new += 1
        except RegistryError as ex:
            if "already registered" in str(ex):
                n_skip += 1
            else:
                errors.append(str(ex))
    print(f"seeded {n_new}, already present {n_skip}, errors {len(errors)}")
    for e in errors:
        print("  ERROR", e)
    return 1 if errors else 0


def cmd_list(a) -> int:
    for e in iter_results(a.registry_dir):
        if a.status and e["status"] != a.status:
            continue
        print(f"{e['id']:60s} {e['status']:18s} {e['title'][:80]}")
    return 0


def cmd_show(a) -> int:
    print(json.dumps(get(a.id, a.registry_dir), indent=1))
    return 0


def cmd_set_status(a) -> int:
    e = set_status(a.id, a.status, a.note, a.registry_dir, a.by, a.superseded_by)
    print(f"{e['id']}: {e['status']} ({e['status_note']})")
    return 0


def cmd_validate(a) -> int:
    n, bad = 0, 0
    for e in iter_results(a.registry_dir):
        n += 1
        errs = validate_entry(e)
        p = e.get("provenance", {}).get("patch")
        if p and not os.path.exists(p):
            errs.append(f"{e['id']}: stored patch {p} is missing")
        for x in errs:
            bad += 1
            print("INVALID", x)
    print(f"{n} results, {bad} problems")
    return 1 if bad else 0


def cmd_table(a) -> int:
    rows = iter_results(a.registry_dir)
    print("| id | status | title | values | bias |")
    print("|---|---|---|---|---|")
    for e in rows:
        vals = "; ".join(_fmt_value(v) for v in e["values"])[:300]
        st = e["status"] + (f" ({e['status_note']})" if e.get("status_note") else "")
        print(f"| {e['id']} | {st} | {e['title']} | {vals} | {e['bias'][:200]} |")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="registry")
    ap.add_argument("--registry-dir", default=None)
    sub = ap.add_subparsers(dest="what", required=True)
    s = sub.add_parser("seed")
    s.add_argument("file")
    s.add_argument("--overwrite", action="store_true")
    s.add_argument("--by")
    s.add_argument("--definition-sheet", help="sheet version for entries that name none (default: the current one)")
    s = sub.add_parser("list")
    s.add_argument("--status", choices=sorted(STATUSES))
    s = sub.add_parser("show")
    s.add_argument("id")
    s = sub.add_parser("set-status")
    s.add_argument("id")
    s.add_argument("status", choices=sorted(STATUSES))
    s.add_argument("--note", required=True)
    s.add_argument("--superseded-by")
    s.add_argument("--by")
    sub.add_parser("validate")
    sub.add_parser("table")
    a = ap.parse_args(argv)
    return {"seed": cmd_seed, "list": cmd_list, "show": cmd_show, "set-status": cmd_set_status,
            "validate": cmd_validate, "table": cmd_table}[a.what](a)


if __name__ == "__main__":
    sys.exit(main())
