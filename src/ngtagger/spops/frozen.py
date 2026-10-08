"""Frozen copies of whole trees for long jobs, their manifest, and the launch-time
provenance (plan App. C.3-4, C.3-5).

A frozen copy mirrors the repository layout under <job>/copy/ (so code that finds files
relative to its own __file__, e.g. ``Path(__file__).parents[2] / "eval_mva_explorer/..."``,
still finds them inside the copy). It holds, for every tree given (``src/ngtagger`` always):

  * every TRACKED file, as it is in the working tree at launch (uncommitted edits included;
    the diff against HEAD is stored as tree.patch);
  * the UNTRACKED code files (.py .sh .zsh .bash .yaml .yml .toml .cfg .ini, <= 1 MiB) under
    the tree, unless --no-untracked-code; every other untracked file (outputs, caches, LUT
    dumps) is left out and only counted -- a job that needs one names it with --include;
  * files named with --include (any status), and the import guard (.spguard/sitecustomize.py).

The manifest lists every file with its sha256; its hash identifies the code a job ran
(one fast-gate result serves every job with the same manifest hash).

Stdlib only.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import shutil
import stat
import subprocess
import time

CODE_SUFFIXES = (".py", ".sh", ".zsh", ".bash", ".yaml", ".yml", ".toml", ".cfg", ".ini")
UNTRACKED_CODE_MAX_BYTES = 1 << 20
FULL_HASH_MAX_BYTES = 2 << 30          # larger data files get a partial (head/middle/tail) hash
PARTIAL_CHUNK = 8 << 20
SKIP_NAMES = ("__pycache__", ".DS_Store", ".pytest_cache")
GUARD_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "guard", "sitecustomize.py")
GUARD_DIR = ".spguard"
DEFAULT_TREES = ("src/ngtagger",)
MANIFEST_SCHEMA = "sp-frozen-manifest/1"


class FrozenCopyError(RuntimeError):
    pass


def git(repo: str, *args: str, binary: bool = False):
    r = subprocess.run(["git", "-C", repo, *args], capture_output=True, check=False)
    if r.returncode != 0:
        raise FrozenCopyError(f"git {' '.join(args)}: {r.stderr.decode(errors='replace').strip()}")
    return r.stdout if binary else r.stdout.decode()


def repo_root(path: str) -> str:
    return git(path, "rev-parse", "--show-toplevel").strip()


def sha256_file(path: str, full_max: int = FULL_HASH_MAX_BYTES) -> tuple[str, str]:
    """(hex digest, 'sha256' | 'sha256-partial'). Files above `full_max` are hashed from
    their size plus three 8 MiB chunks (head, middle, tail): enough to tell files apart,
    labelled so nobody mistakes it for a full checksum."""
    size = os.path.getsize(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        if size <= full_max:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
            return h.hexdigest(), "sha256"
        h.update(str(size).encode())
        for off in (0, size // 2, max(0, size - PARTIAL_CHUNK)):
            f.seek(off)
            h.update(f.read(PARTIAL_CHUNK))
    return h.hexdigest(), "sha256-partial"


def _norm_tree(repo: str, t: str) -> str:
    p = os.path.abspath(t if os.path.isabs(t) else os.path.join(repo, t))
    rel = os.path.relpath(p, repo)
    if rel.startswith(".."):
        raise FrozenCopyError(f"{t} is outside the repository {repo}")
    if not os.path.exists(p):
        raise FrozenCopyError(f"{t} does not exist")
    return rel.replace(os.sep, "/")


def _z(out: str) -> list[str]:
    return [x for x in out.split("\0") if x]


def _skip(rel: str, excludes) -> bool:
    parts = rel.split("/")
    return any(p in SKIP_NAMES for p in parts) or any(fnmatch.fnmatch(rel, g) for g in excludes)


def _copy(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def manifest_hash(files: list[dict]) -> str:
    h = hashlib.sha256()
    for f in sorted(files, key=lambda x: x["path"]):
        h.update(f"{f['path']}\t{f['sha256']}\n".encode())
    return h.hexdigest()


def build_copy(repo: str, dest: str, trees=(), includes=(), excludes=(), untracked_code: bool = True,
               remove_for_test=(), read_only: bool = True) -> dict:
    """Build the frozen copy at `dest` (must not exist) and return its manifest."""
    repo = repo_root(repo)
    if os.path.exists(dest):
        raise FrozenCopyError(f"{dest} exists")
    trees = list(dict.fromkeys([_norm_tree(repo, t) for t in (*DEFAULT_TREES, *trees)]))
    modified = set(_z(git(repo, "diff", "--name-only", "-z", "HEAD", "--", *trees)))
    files, missing, excluded = {}, [], {"count": 0, "bytes": 0, "examples": []}
    included = []
    for inc in includes:
        rel0 = _norm_tree(repo, inc)
        src0 = os.path.join(repo, rel0)
        included += [rel0] if os.path.isfile(src0) else [
            os.path.relpath(os.path.join(d, f), repo).replace(os.sep, "/")
            for d, _, fs in os.walk(src0) for f in fs]
    included = [r for r in dict.fromkeys(included) if not _skip(r, excludes)]
    for t in trees:
        for rel in _z(git(repo, "ls-files", "-z", "--", t)):
            if _skip(rel, excludes):
                continue
            src = os.path.join(repo, rel)
            if not os.path.isfile(src):
                missing.append(rel)            # tracked but deleted in the working tree
                continue
            files[rel] = "tracked-modified" if rel in modified else "tracked"
        for rel in _z(git(repo, "ls-files", "-z", "--others", "--exclude-standard", "--", t)):
            if _skip(rel, excludes) or rel in files or rel in included:
                continue
            src = os.path.join(repo, rel)
            size = os.path.getsize(src) if os.path.isfile(src) else 0
            if untracked_code and rel.endswith(CODE_SUFFIXES) and size <= UNTRACKED_CODE_MAX_BYTES:
                files[rel] = "untracked-code"
            else:
                excluded["count"] += 1
                excluded["bytes"] += size
                if len(excluded["examples"]) < 12:
                    excluded["examples"].append(rel)
    for rel in included:
        files.setdefault(rel, "included")
    os.makedirs(dest)
    rows = []
    for rel, origin in sorted(files.items()):
        _copy(os.path.join(repo, rel), os.path.join(dest, rel))
        digest, _ = sha256_file(os.path.join(dest, rel))
        rows.append({"path": rel, "sha256": digest, "bytes": os.path.getsize(os.path.join(dest, rel)),
                     "origin": origin})
    gdst = os.path.join(dest, GUARD_DIR, "sitecustomize.py")
    _copy(GUARD_SRC, gdst)
    rows.append({"path": f"{GUARD_DIR}/sitecustomize.py", "sha256": sha256_file(gdst)[0],
                 "bytes": os.path.getsize(gdst), "origin": "guard"})
    removed = []
    for r in remove_for_test:
        rel = _norm_tree(repo, r) if os.path.exists(os.path.join(repo, r)) else r
        p = os.path.join(dest, rel)
        if os.path.exists(p):
            os.remove(p)
            removed.append(rel)
            rows = [x for x in rows if x["path"] != rel]
    if read_only:
        for d, ds, fs in os.walk(dest):
            for f in fs:
                os.chmod(os.path.join(d, f), stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            os.chmod(d, stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    return {
        "schema": MANIFEST_SCHEMA,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo": repo,
        "commit": git(repo, "rev-parse", "HEAD").strip(),
        "trees": trees,
        "includes": list(includes),
        "excludes": list(excludes),
        "untracked_code": untracked_code,
        "read_only": read_only,
        "n_files": len(rows),
        "bytes": sum(r["bytes"] for r in rows),
        "files": rows,
        "tracked_missing_in_worktree": missing,
        "untracked_left_out": excluded,
        "removed_for_test": removed,
        "manifest_hash": manifest_hash(rows),
    }


def make_writable(dest: str) -> None:
    """Undo read_only (to delete a copy)."""
    for d, ds, fs in os.walk(dest):
        os.chmod(d, stat.S_IRWXU)
        for f in fs:
            os.chmod(os.path.join(d, f), stat.S_IRUSR | stat.S_IWUSR)


def tree_state(repo: str, trees, patch_path: str) -> dict:
    """Commit, branch and the diff of the copied tracked files against HEAD, written to
    `patch_path` (empty diff: 'clean'). Captured at launch, before the job can run."""
    repo = repo_root(repo)
    trees = list(dict.fromkeys([_norm_tree(repo, t) for t in (*DEFAULT_TREES, *trees)]))
    diff = git(repo, "diff", "--binary", "HEAD", "--", *trees, binary=True)
    with open(patch_path, "wb") as f:
        f.write(diff)
    branch = subprocess.run(["git", "-C", repo, "rev-parse", "--abbrev-ref", "HEAD"],
                            capture_output=True, text=True, check=False).stdout.strip()
    return {
        "commit": git(repo, "rev-parse", "HEAD").strip(),
        "branch": branch,
        "trees": trees,
        "state": "clean" if not diff else "dirty",
        "patch": os.path.abspath(patch_path),
        "patch_sha256": hashlib.sha256(diff).hexdigest(),
        "patch_bytes": len(diff),
        "modified_files": _z(git(repo, "diff", "--name-only", "-z", "HEAD", "--", *trees)),
    }


def resolve_data(path: str, repo: str, copy_root: str | None) -> str:
    """Where a job will read `path`: a repository path inside a frozen tree is read from
    the copy (that is what code finding files beside itself does)."""
    repo = os.path.abspath(repo)
    if copy_root is None:
        return os.path.abspath(path)
    if not os.path.isabs(path):
        return os.path.join(copy_root, path)
    rel = os.path.relpath(os.path.abspath(path), repo)
    if not rel.startswith("..") and not rel.startswith(".pixi"):
        return os.path.join(copy_root, rel)
    return path


def check_data(paths, repo: str, copy_root: str | None) -> tuple[list[dict], list[str]]:
    """(records, problems) for the data files a job reads: each must exist where the job
    will read it; size, mtime and checksum are recorded."""
    recs, problems = [], []
    for p in paths:
        where = resolve_data(p, repo, copy_root)
        if not os.path.isfile(where):
            problems.append(f"data file {p} is not present where the job reads it ({where})")
            recs.append({"declared": p, "resolved": where, "present": False})
            continue
        digest, kind = sha256_file(where)
        st = os.stat(where)
        recs.append({"declared": p, "resolved": where, "present": True, "bytes": st.st_size,
                     "mtime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime)),
                     kind: digest})
    return recs, problems


def tar_files(root: str, rels, out: str) -> str:
    """Copies of the untracked files a job imports (C.3-5), as a small tar.gz."""
    import tarfile

    with tarfile.open(out, "w:gz") as t:
        for rel in rels:
            t.add(os.path.join(root, rel), arcname=rel)
    return out


def write_json(path: str, obj) -> None:
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)
