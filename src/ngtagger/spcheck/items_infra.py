"""Fast-gate item on the launcher's import guard (plan App. C.4 row: edits reaching a running
job; patch recorded after the fact; lazy imports missing from a copy).

It tests THIS code root's guard (spops/guard/sitecustomize.py) and the launcher's verdict
function (spops.launch.guard_verdict) on a miniature frozen copy built in a scratch
directory (no git needed, so it also runs from inside a frozen copy). The full launcher,
lock and lane are exercised end to end by `python -m ngtagger.spops.selftest`."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

from .core import Item

PROBE = "src/ngtagger/spops/probe"


def _mini_copy(code_root, dest, remove=()):
    """A writable miniature copy (the code root itself may be a read-only frozen copy)."""
    src = os.path.join(code_root, "src", "ngtagger")
    shutil.copytree(src, os.path.join(dest, "src", "ngtagger"), copy_function=shutil.copyfile,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))
    for d, _, _ in os.walk(dest):
        os.chmod(d, 0o755)
    os.makedirs(os.path.join(dest, ".spguard"))
    shutil.copyfile(os.path.join(src, "spops", "guard", "sitecustomize.py"),
                    os.path.join(dest, ".spguard", "sitecustomize.py"))
    for r in remove:
        os.remove(os.path.join(dest, r))


def _run(code_root, scratch, name, remove=(), unset_pythonpath=False, shared_dir=None):
    from ngtagger.spops.launch import guard_records, guard_verdict

    work = tempfile.mkdtemp(prefix=f"spcheck_guard_{name}_", dir=scratch)
    copy, job = os.path.join(work, "copy"), os.path.join(work, "job")
    os.makedirs(os.path.join(job, "guard"))
    _mini_copy(code_root, copy, remove)
    env = {k: v for k, v in os.environ.items() if not k.startswith("SP_")}
    env.update({"SP_FROZEN_ROOT": copy, "SP_JOB_DIR": job, "SP_GUARD_MODE": "strict",
                "PYTHONPATH": os.pathsep.join([os.path.join(copy, ".spguard"), os.path.join(copy, "src")]),
                "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"})
    if unset_pythonpath:
        env.pop("PYTHONPATH")
    if shared_dir:
        env["SP_PROBE_SHARED_DIR"] = shared_dir
    r = subprocess.run([sys.executable, os.path.join(PROBE, "probe_job.py")], cwd=copy, env=env,
                       capture_output=True, text=True, timeout=300)
    rec = guard_records(job)
    status, code = guard_verdict(r.returncode, rec, expect_guard=True)
    rep = next((json.loads(x[6:]) for x in r.stdout.splitlines() if x.startswith("PROBE ")), {})
    inside = {k: (os.path.realpath(v).startswith(os.path.realpath(copy) + os.sep) if isinstance(v, str)
                  and os.path.isabs(v) else None) for k, v in rep.items() if k in ("ngtagger", "labels", "lazy_helper",
                                                                                   "lazy_pkg")}
    shutil.rmtree(work, ignore_errors=True)
    err = [x for x in r.stderr.splitlines() if "Error" in x][-1:]
    return {"rc": r.returncode, "status": status, "exit_code": code, "modules_inside_copy": inside,
            "violations": [(v["kind"], v["name"]) for v in rec["violations"]], "error": err,
            "guarded_processes": rec["python_processes_guarded"]}


class ImportGuard(Item):
    id = "launcher.import_guard"
    title = "frozen-copy jobs import only from their copy; violations and an unguarded Python fail loudly"
    c4_row = "edits reaching a running job; patch recorded after the fact; lazy imports missing from a copy"
    known_bad_desc = ("(a) PYTHONPATH unset (the job imports the shared src); (b) a lazily imported module removed "
                      "from the copy while the shared tree is on sys.path (import outside any try); (c) a lazily "
                      "imported ngtagger module (train/prediction_dump.py) removed from the copy")
    rule_desc = ("the probe job (eager + lazy imports, as the census and the trainer do) run from a miniature copy "
                 "of this code root: status ok and every module inside the copy; each known-bad ends with a "
                 "non-zero status the launcher reports (98 guard-missing, 97 guard-violation, failed with ImportError)")

    def compute(self, ctx):
        good = _run(ctx.code_root, ctx.scratch, "good")
        return {"input": "probe job", "events": 0, "cases": {"good": good}}

    def known_bad(self, ctx):
        shared = os.path.join(str(ctx.code_root), PROBE)
        return {"input": "probe job", "events": 0, "cases": {
            "pythonpath_unset": _run(ctx.code_root, ctx.scratch, "unset", unset_pythonpath=True),
            "lazy_module_shared_fallback": _run(ctx.code_root, ctx.scratch, "shared",
                                                remove=(f"{PROBE}/probe_lazy_helper.py",), shared_dir=shared),
            "lazy_package_module_missing": _run(ctx.code_root, ctx.scratch, "pkg",
                                                remove=("src/ngtagger/train/prediction_dump.py",))}}

    @staticmethod
    def _case_green(c) -> bool:
        return c["status"] == "ok" and bool(c["modules_inside_copy"]) and all(c["modules_inside_copy"].values())

    def judge(self, stat, ref):
        bad = [f"{n}: {c['status']} (exit {c['exit_code']}) {c['violations'] or c['error']}"
               for n, c in stat["cases"].items() if not self._case_green(c)]
        return not bad, "probe imports only from its copy" if not bad else "; ".join(bad)

    def known_bad_is_red(self, stat, ref):
        """Every known-bad case must fail on its own (not just one of them)."""
        caught = {n: not self._case_green(c) for n, c in stat["cases"].items()}
        why = "; ".join(f"{n}: {c['status']} (exit {c['exit_code']}){' NOT CAUGHT' if not caught[n] else ''}"
                        for n, c in stat["cases"].items())
        return all(caught.values()), why


ITEMS = [ImportGuard()]
