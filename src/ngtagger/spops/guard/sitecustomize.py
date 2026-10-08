"""SmartPixels frozen-copy import guard (plan App. C.3-4).

The launcher copies this file to <copy>/.spguard/sitecustomize.py and puts that directory
first on PYTHONPATH, so every Python process of a frozen-copy job (and its children,
which inherit the environment) runs it at start-up. Active only when SP_FROZEN_ROOT is set.

It checks every module the process imports -- at import time, lazy imports included --
and every source file it executes, and requires each to lie inside the frozen copy or
inside the Python environment (sys.prefix and friends; extra roots in SP_GUARD_ALLOW).
A module from anywhere else (the shared working tree above all) is a violation: the
guard prints a banner, appends a record to $SP_JOB_DIR/guard/violations.jsonl and ends
the process with exit code 97 (os._exit, so no `except` can swallow it). At start it
writes $SP_JOB_DIR/guard/<pid>.start.json, at normal exit <pid>.exit.json: the launcher
fails a job whose Python never ran the guard (PYTHONPATH did not reach it).

It cannot see data files a job reads (the launcher checks declared data files at launch)
and it cannot see an import that fails inside a `try` (nothing is loaded, so there is
nothing to check).  Stdlib only.
"""
import os
import sys


def _install():
    root = os.environ.get("SP_FROZEN_ROOT")
    if not root:
        return
    import json
    import time

    real = os.path.realpath
    root_r = real(root)
    allowed = {root_r}
    for p in (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix):
        if p:
            allowed.add(real(p))
    for p in os.environ.get("SP_GUARD_ALLOW", "").split(os.pathsep):
        if p:
            allowed.add(real(p))
    allowed = tuple(sorted(allowed))
    job_dir = os.environ.get("SP_JOB_DIR")
    gdir = os.path.join(job_dir, "guard") if job_dir else None
    strict = os.environ.get("SP_GUARD_MODE", "strict") == "strict"
    pending = []

    def inside(path):
        rp = real(path)
        return any(rp == a or rp.startswith(a + os.sep) for a in allowed)

    def record(name, obj):
        if not gdir:
            return
        try:
            os.makedirs(gdir, exist_ok=True)
            with open(os.path.join(gdir, name), "a") as f:
                f.write(json.dumps(obj, default=str) + "\n")
        except OSError:
            pass

    def violation(kind, name, path):
        rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "pid": os.getpid(),
               "kind": kind, "name": name, "path": path, "argv": list(sys.argv), "frozen_root": root_r}
        bar = "!" * 100
        sys.stderr.write(f"\n{bar}\nSP IMPORT GUARD VIOLATION: {kind} {name!r} was loaded from\n  {path}\n"
                         f"which lies OUTSIDE the frozen copy {root_r} and the Python environment.\n"
                         f"The job is stopped (exit 97){'' if strict else ' at exit'}.\n{bar}\n")
        sys.stderr.flush()
        record("violations.jsonl", rec)
        if strict:
            try:
                sys.stdout.flush()
            except Exception:
                pass
            os._exit(97)
        pending.append(rec)

    def check_spec(name, spec):
        origin = getattr(spec, "origin", None)
        if origin and origin not in ("built-in", "frozen") and os.path.isabs(origin):
            if not inside(origin):
                violation("module", name, origin)
            return
        locs = list(getattr(spec, "submodule_search_locations", None) or [])
        if origin is None and locs and not any(inside(x) for x in locs):
            violation("namespace package", name, os.pathsep.join(locs))

    class FrozenCopyGuard:
        """First meta-path finder: asks the others, checks what they found."""

        def find_spec(self, name, path=None, target=None):
            for finder in sys.meta_path:
                if finder is self:
                    continue
                fs = getattr(finder, "find_spec", None)
                if fs is None:
                    continue
                spec = fs(name, path, target)
                if spec is not None:
                    check_spec(name, spec)
                    return spec
            return None

        def invalidate_caches(self):
            pass

    def audit(event, args):
        if event == "exec":
            fn = getattr(args[0], "co_filename", "") if args else ""
            if fn and fn.endswith(".py") and os.path.isabs(fn) and not inside(fn):
                violation("exec", fn, fn)
        elif event == "cpython.run_file":
            fn = args[0] if args else ""
            if fn and os.path.isabs(str(fn)) and not inside(str(fn)):
                violation("main script", str(fn), str(fn))

    for m in list(sys.modules.values()):                 # anything loaded before the guard
        f = getattr(m, "__file__", None)
        if f and os.path.isabs(f) and not inside(f):
            violation("module (preloaded)", getattr(m, "__name__", "?"), f)
    if sys.argv and sys.argv[0] and os.path.isfile(sys.argv[0]) and not inside(os.path.abspath(sys.argv[0])):
        violation("main script", sys.argv[0], os.path.abspath(sys.argv[0]))

    sys.meta_path.insert(0, FrozenCopyGuard())
    sys.addaudithook(audit)
    record(f"{os.getpid()}.start.json", {"pid": os.getpid(), "ppid": os.getppid(), "argv": list(sys.argv),
                                          "executable": sys.executable, "frozen_root": root_r,
                                          "allowed": allowed, "sys_path": list(sys.path),
                                          "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})

    import atexit

    def at_exit():
        outside, n_in, n_env = [], 0, 0
        for m in list(sys.modules.values()):
            f = getattr(m, "__file__", None)
            if not f or not os.path.isabs(f):
                continue
            rf = real(f)
            if rf.startswith(root_r + os.sep):
                n_in += 1
            elif inside(f):
                n_env += 1
            else:
                outside.append([getattr(m, "__name__", "?"), f])
        record(f"{os.getpid()}.exit.json", {"pid": os.getpid(), "argv": list(sys.argv),
                                             "modules_in_copy": n_in, "modules_in_env": n_env,
                                             "modules_outside": outside, "violations": pending,
                                             "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        if outside:
            for name, f in outside:
                violation("module (at exit)", name, f)
        if pending:
            sys.stderr.flush()
            os._exit(97)

    atexit.register(at_exit)


def _chain():
    """Run the next sitecustomize on sys.path, if any (this file shadows it)."""
    here = os.path.dirname(os.path.realpath(__file__))
    for p in sys.path:
        if not p or os.path.realpath(p) == here:
            continue
        cand = os.path.join(p, "sitecustomize.py")
        if os.path.isfile(cand):
            import importlib.util

            spec = importlib.util.spec_from_file_location("_sp_chained_sitecustomize", cand)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return


_install()
_chain()
