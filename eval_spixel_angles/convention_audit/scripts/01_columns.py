import pyarrow.parquet as pq
import numpy as np
import json

ROOT = "/Users/nmangane/smartpixels/ngtagger-train"
files = [
    f"{ROOT}/2t-Conv1D_Full-2bit_optimized-vars.parquet",
    f"{ROOT}/2t-Mlp_Slim-2bit_optimized-vars.parquet",
    f"{ROOT}/2t-Conv2D_Max-2bit_optimized-vars.parquet",
    f"{ROOT}/part.93.parquet",
]
out = {}
for f in files:
    pf = pq.ParquetFile(f)
    names = [fld.name for fld in pf.schema_arrow]
    # keep non-numeric-named columns (i.e., drop pure-integer pixel columns)
    named = [n for n in names if not n.isdigit()]
    print("=" * 100)
    print(f, "rows:", pf.metadata.num_rows, " total cols:", len(names), " named cols:", len(named))
    print("named:", named)
    if len(named) == 0:
        continue
    tbl = pf.read(columns=named)
    df = tbl.to_pandas()
    rec = {}
    for c in named:
        try:
            x = np.asarray(df[c], dtype=np.float64)
        except Exception:
            print(f"  {c}: non-numeric, sample {df[c].iloc[:3].tolist()}")
            continue
        qs = np.quantile(x, [0.0, 0.01, 0.16, 0.5, 0.84, 0.99, 1.0])
        rec[c] = dict(min=qs[0], q01=qs[1], q16=qs[2], med=qs[3], q84=qs[4], q99=qs[5], max=qs[6],
                      mean=float(x.mean()), std=float(x.std()))
        print(f"  {c:24s} min={qs[0]:+9.4f} q01={qs[1]:+9.4f} med={qs[3]:+9.4f} q99={qs[5]:+9.4f} max={qs[6]:+9.4f} mean={x.mean():+9.4f} std={x.std():9.4f}")
    out[f.split('/')[-1]] = rec

with open(f"{ROOT}/eval_spixel_angles/convention_audit/column_stats.json", "w") as fh:
    json.dump(out, fh, indent=1)
