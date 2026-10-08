"""Probe job for the frozen-copy launcher self-test (ngtagger.spops.selftest) and the fast-gate item launcher.import_guard.
Not a pytest module. It imports the way real jobs do: its own directory first on sys.path
(as tp_findability.py:49 does), optionally a hard-coded shared-tree directory appended (as
the 49 scripts with the shared tree's absolute path do), an ngtagger module eagerly, a data
file beside itself (as export_site.py reads plotly.min.js), then two LAZY imports after the
"work" -- outside any try, unless SP_PROBE_SWALLOW=1 (the trainer.py:164-169 pattern)."""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if os.environ.get("SP_PROBE_SHARED_DIR"):
    sys.path.append(os.environ["SP_PROBE_SHARED_DIR"])

import ngtagger  # noqa: E402
from ngtagger.data import labels  # noqa: E402

report = {"argv": sys.argv, "ngtagger": ngtagger.__file__, "labels": labels.__file__}
data = HERE / "probe_data.txt"
report["data"] = data.read_text().strip() if data.exists() else None   # silently None when absent

if os.environ.get("SP_PROBE_SWALLOW") == "1":
    try:
        import probe_lazy_helper
        report["lazy_helper"] = probe_lazy_helper.__file__
    except Exception as e:  # what trainer.py:169 does: print and carry on
        report["lazy_helper"] = f"skipped: {e}"
else:
    import probe_lazy_helper  # noqa: E402
    report["lazy_helper"] = probe_lazy_helper.__file__
    from ngtagger.train import prediction_dump  # noqa: E402
    report["lazy_pkg"] = prediction_dump.__file__

print("PROBE", json.dumps(report))
if os.environ.get("SP_PROBE_OUT"):
    Path(os.environ["SP_PROBE_OUT"]).write_text(json.dumps(report))
