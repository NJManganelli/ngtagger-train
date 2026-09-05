#!/usr/bin/env python3
"""Leg 6: deployed v4fixed payload: (a) 'x_cms=-y_pix' present in CorrectionSet description,
all 5 correction descriptions, and cotAlpha/cotBeta variable descriptions; (b) numeric
content (descriptions stripped) identical to eval_spixel_angles/spx_angle_response_Conv1D_Full-2bit.json."""
import json, os

AUD = "/Users/nmangane/smartpixels/ngtagger-train/eval_spixel_angles/convention_audit"
ev = json.load(open(os.path.join(AUD, "evidence.json")))

DEP = "/Users/nmangane/smartpixels/cmssw/work/spxsmoke/spx_angle_response_Conv1D_Full-2bit_v4fixed.json"
REF = "/Users/nmangane/smartpixels/ngtagger-train/eval_spixel_angles/spx_angle_response_Conv1D_Full-2bit.json"
dep = json.load(open(DEP)); ref = json.load(open(REF))

TOK = "x_cms=-y_pix"
desc = dict(cset_description=TOK in dep.get("description", ""))
per_corr, per_var = {}, {}
for c in dep["corrections"]:
    per_corr[c["name"]] = TOK in c.get("description", "")
    for v in c["inputs"]:
        if v["name"] in ("cotAlpha", "cotBeta"):
            per_var.setdefault(c["name"], {})[v["name"]] = TOK in v.get("description", "")

def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if k != "description"}
    if isinstance(o, list):
        return [strip(x) for x in o]
    return o

identical = strip(dep) == strip(ref)
# locate any numeric differences if not identical
diffs = []
if not identical:
    sd, sr = strip(dep), strip(ref)
    def walk(a, b, path):
        if len(diffs) > 10: return
        if type(a) != type(b):
            diffs.append(f"{path}: type {type(a).__name__} vs {type(b).__name__}"); return
        if isinstance(a, dict):
            for k in sorted(set(a) | set(b)):
                if k not in a: diffs.append(f"{path}.{k}: missing in deployed")
                elif k not in b: diffs.append(f"{path}.{k}: missing in ref")
                else: walk(a[k], b[k], f"{path}.{k}")
        elif isinstance(a, list):
            if len(a) != len(b): diffs.append(f"{path}: len {len(a)} vs {len(b)}"); return
            for i, (x, y) in enumerate(zip(a, b)): walk(x, y, f"{path}[{i}]")
        elif a != b:
            diffs.append(f"{path}: {a!r} vs {b!r}")
    walk(sd, sr, "$")

ev["leg6_identity"] = dict(
    token=TOK,
    cset_description_has_token=desc["cset_description"],
    corr_description_has_token=per_corr,
    var_description_has_token=per_var,
    all_descriptions_ok=(desc["cset_description"] and all(per_corr.values())
                         and all(all(v.values()) for v in per_var.values())),
    numeric_identical_to_eval_json=identical,
    first_diffs=diffs[:10],
)
json.dump(ev, open(os.path.join(AUD, "evidence.json"), "w"), indent=1, default=float)
print("cset description token:", desc["cset_description"])
print("per-correction token:", per_corr)
print("per-variable token ok:", all(all(v.values()) for v in per_var.values()))
print("numeric identical:", identical)
if diffs: print("diffs:", diffs[:5])
