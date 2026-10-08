"""Unit tests of the heavy lane, the frozen copy and the results registry (ngtagger.spops).
The end-to-end launcher scenarios (lock, legacy waits, Docker, known-bads) are in
`python -m ngtagger.spops.selftest`; these need no data, no Docker and no lock."""
import json
import os
import subprocess

import pytest

from ngtagger.spops import frozen as F
from ngtagger.spops import lane as L
from ngtagger.spops import registry as R


def test_bracketed_patterns():
    assert L.bracketed("tp_findability.py") == "[t]p_findability[.]py"
    assert L.bracketed("tagger_sp/train.py") == "[t]agger_sp/train[.]py"
    assert L.bracketed("train-trackmva") == "[t]rain-trackmva"


def test_enter_lane_rechecks_after_taking_the_lock(tmp_path, monkeypatch):
    """A blocker that appears between the scan and the lock must send the launcher back
    to waiting, with the lock released and a recheck-blocked event logged."""
    calls = {"n": 0}
    blocker = [{"kind": "legacy", "pattern": "x", "pid": 1, "cmd": "x"}]

    def fake_scan(policy, exclude=None):
        calls["n"] += 1
        return ([] if calls["n"] in (1, 3, 4) else blocker), {}

    monkeypatch.setattr(L, "scan", fake_scan)
    monkeypatch.setenv("SP_LANE_QUIET", "1")
    lock = L.HeavyLock(str(tmp_path))
    st = L.enter_lane(lock, {"job_id": "t"}, L.Policy(poll_s=0.01), str(tmp_path))
    assert st["polls_waited"] == 1
    ev = [json.loads(x)["ev"] for x in open(tmp_path / "heavy.log")]
    assert ev == ["recheck-blocked", "wait", "acquired"]
    lock.release()
    assert not lock.held_by_anyone()


def test_lock_is_exclusive_and_skip_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "scan", lambda policy, exclude=None: ([], {}))
    monkeypatch.setenv("SP_LANE_QUIET", "1")
    a, b = L.HeavyLock(str(tmp_path)), L.HeavyLock(str(tmp_path))
    assert a.try_acquire({"job_id": "a"})
    assert not b.try_acquire({"job_id": "b"})
    with pytest.raises(L.LaneBusy):
        L.enter_lane(b, {"job_id": "b"}, L.Policy(on_busy="skip"), str(tmp_path))
    a.release()
    assert b.try_acquire({"job_id": "b"})
    b.release()


def _git_repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src" / "ngtagger").mkdir(parents=True)
    (repo / "jobdir" / "out").mkdir(parents=True)
    (repo / "src" / "ngtagger" / "__init__.py").write_text("X = 1\n")
    (repo / "jobdir" / "job.py").write_text("print('job')\n")
    run = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True)
    run("init", "-q")
    run("add", "src/ngtagger/__init__.py", "jobdir/job.py")
    run("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "init")
    (repo / "src" / "ngtagger" / "__init__.py").write_text("X = 2\n")          # dirty tracked file
    (repo / "jobdir" / "helper.py").write_text("Y = 1\n")                      # untracked code
    (repo / "jobdir" / "out" / "big.npz").write_bytes(b"0" * 2000)            # untracked output
    (repo / "jobdir" / "data.txt").write_text("d\n")                           # untracked data
    return repo


def test_frozen_copy_contents_manifest_and_patch(tmp_path):
    repo = _git_repo(tmp_path)
    dest = tmp_path / "job" / "copy"
    man = F.build_copy(str(repo), str(dest), trees=["jobdir"], includes=["jobdir/data.txt"], read_only=False)
    files = {f["path"]: f["origin"] for f in man["files"]}
    assert files["src/ngtagger/__init__.py"] == "tracked-modified"
    assert files["jobdir/job.py"] == "tracked"
    assert files["jobdir/helper.py"] == "untracked-code"
    assert files["jobdir/data.txt"] == "included"
    assert "jobdir/out/big.npz" not in files and man["untracked_left_out"]["count"] == 1
    assert (dest / ".spguard" / "sitecustomize.py").exists()
    assert (dest / "src" / "ngtagger" / "__init__.py").read_text() == "X = 2\n"
    ts = F.tree_state(str(repo), ["jobdir"], str(tmp_path / "job" / "tree.patch"))
    assert ts["state"] == "dirty" and "src/ngtagger/__init__.py" in ts["modified_files"]
    assert "+X = 2" in (tmp_path / "job" / "tree.patch").read_text()
    # the manifest hash moves with any content change
    man2 = F.build_copy(str(repo), str(tmp_path / "c2"), trees=["jobdir"], includes=["jobdir/data.txt"],
                        remove_for_test=["jobdir/helper.py"], read_only=False)
    assert man2["manifest_hash"] != man["manifest_hash"]


def test_data_check_reads_where_the_job_reads(tmp_path):
    repo = _git_repo(tmp_path)
    dest = tmp_path / "copy"
    F.build_copy(str(repo), str(dest), trees=["jobdir"], read_only=False)
    recs, problems = F.check_data([str(repo / "jobdir" / "data.txt")], str(repo), str(dest))
    assert problems and not recs[0]["present"]          # untracked data not included -> refused
    recs, problems = F.check_data([str(repo / "jobdir" / "job.py")], str(repo), str(dest))
    assert not problems and recs[0]["sha256"]


def _entry(i="t.result-one", status="exploratory"):
    return {"id": i, "title": "a result", "values": [{"name": "x", "value": 1.0, "error": 0.1}],
            "status": status, "status_note": "provenance unknown", "bias": "pessimistic: test",
            "provenance": {"known": False}}


def test_registry_one_file_per_result_and_history(tmp_path):
    reg = str(tmp_path / "reg")
    p = R.register(_entry(), reg, job_dir="")
    assert os.path.basename(p) == "t.result-one.json"
    with pytest.raises(R.RegistryError):
        R.register(_entry(), reg, job_dir="")                  # never silently overwritten
    e = R.set_status("t.result-one", "superseded", "replaced by t.result-two", reg, superseded_by="t.result-two")
    assert e["status"] == "superseded" and [h["status"] for h in e["history"]] == ["exploratory", "superseded"]
    assert R.get("t.result-one", reg)["superseded_by"] == ["t.result-two"]


def test_registry_rejects_bad_entries(tmp_path):
    reg = str(tmp_path / "reg")
    for bad in ({**_entry(), "status": "fine"}, {**_entry(), "bias": ""}, {**_entry(), "status_note": ""},
                {**_entry(), "values": []}, {**_entry(), "id": "Bad Id"}):
        with pytest.raises(R.RegistryError):
            R.register(bad, reg, job_dir="")


def test_registry_attaches_launch_time_provenance(tmp_path):
    reg, job = str(tmp_path / "reg"), tmp_path / "job"
    job.mkdir()
    (job / "launch.json").write_text(json.dumps({
        "job_id": "J1", "launched": "t0", "command": ["python", "x.py"], "manifest_hash": "abc",
        "tree_state": {"commit": "c0ffee", "state": "dirty", "patch": "/p/tree.patch", "patch_sha256": "ff",
                       "modified_files": ["a.py"]},
        "data": [{"declared": "in.root", "bytes": 3, "sha256": "aa"}], "prereg": "expect 1"}))
    R.register({**_entry("t.with-job", "valid"), "status_note": ""}, reg, job_dir=str(job))
    prov = R.get("t.with-job", reg)["provenance"]
    assert prov["known"] and prov["commits"]["ngtagger-train"] == "c0ffee" and prov["tree_state"] == "dirty"
    assert prov["inputs"][0]["sha256"] == "aa" and prov["manifest_hash"] == "abc"
