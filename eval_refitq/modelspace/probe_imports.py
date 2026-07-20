"""Import-probe for packages the export/training paths need (Part C audit)."""
import importlib
import sys

MODS = [
    "conifer", "xgboost", "sklearn", "keras", "tensorflow", "hgq", "da4ml",
    "hls4ml", "mlflow", "coffea", "uproot", "awkward", "mdmm", "pquant",
    "onnx", "yaml", "schema",
]

for m in MODS:
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, "__version__", "?")
        print(f"OK    {m:12s} {ver}")
    except Exception as e:
        print(f"FAIL  {m:12s} {type(e).__name__}: {e}")
