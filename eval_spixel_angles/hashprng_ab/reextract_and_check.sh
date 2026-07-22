#!/bin/bash
set -e
cd /Users/nmangane/smartpixels/ngtagger-train
mkdir -p eval_spixel_angles/hashprng_ab/orig_backup
for V in Mlp_Slim-2bit Conv2D_Max-2bit Conv1D_Full-2bit; do
  F=eval_spixel_angles/spx_angle_response_${V}.json
  [ -f eval_spixel_angles/hashprng_ab/orig_backup/$(basename $F) ] || cp $F eval_spixel_angles/hashprng_ab/orig_backup/
  pixi run python eval_spixel_angles/extract_pixelav_angle_payload.py --model-variant $V \
    --out-dir eval_spixel_angles 2t-${V}_optimized-vars.parquet > eval_spixel_angles/hashprng_ab/extract_${V}.log 2>&1
done
pixi run python - <<'PYEOF'
import json
res = {}
for v in ["Mlp_Slim-2bit", "Conv2D_Max-2bit", "Conv1D_Full-2bit"]:
    old = json.load(open(f"eval_spixel_angles/hashprng_ab/orig_backup/spx_angle_response_{v}.json"))
    new = json.load(open(f"eval_spixel_angles/spx_angle_response_{v}.json"))
    oldc = {c["name"]: c for c in old["corrections"]}
    newc = {c["name"]: c for c in new["corrections"]}
    r = {}
    for name in oldc:
        o, n = oldc[name], newc.get(name)
        r[name] = dict(
            present=n is not None,
            data_bit_identical=(n is not None and json.dumps(o["data"], sort_keys=True) == json.dumps(n["data"], sort_keys=True)),
            inputs_identical=(n is not None and o["inputs"] == n["inputs"]),
            output_identical=(n is not None and o["output"] == n["output"]),
        )
    r["_additions"] = sorted(set(newc) - set(oldc))
    r["_compounds"] = sorted(c["name"] for c in new.get("compound_corrections", []))
    res[v] = r
json.dump(res, open("eval_spixel_angles/hashprng_ab/bit_identity_check.json", "w"), indent=1)
ok = all(all(x["data_bit_identical"] and x["inputs_identical"] for k, x in r.items() if not k.startswith("_")) for r in res.values())
print("BIT-IDENTITY:", "PASS" if ok else "FAIL")
for v, r in res.items():
    print(v, "additions:", r["_additions"], "compounds:", r["_compounds"])
PYEOF
