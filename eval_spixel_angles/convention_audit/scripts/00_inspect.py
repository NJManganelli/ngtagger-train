import pyarrow.parquet as pq
import numpy as np

ROOT = "/Users/nmangane/smartpixels/ngtagger-train"
files = [
    f"{ROOT}/2t-Conv1D_Full-2bit_optimized-vars.parquet",
    f"{ROOT}/2t-Mlp_Slim-2bit_optimized-vars.parquet",
    f"{ROOT}/2t-Conv2D_Max-2bit_optimized-vars.parquet",
    f"{ROOT}/part.93.parquet",
]
for f in files:
    pf = pq.ParquetFile(f)
    print("=" * 80)
    print(f, "rows:", pf.metadata.num_rows)
    print("columns:", [fld.name for fld in pf.schema_arrow])
    # basic stats on first 500k rows for numeric cols
    import pandas as pd
    n = min(pf.metadata.num_rows, 500_000)
    tbl = next(pf.iter_batches(batch_size=n))
    df = tbl.to_pandas() if hasattr(tbl, "to_pandas") else None
    if df is not None:
        num = df.select_dtypes("number")
        stats = num.describe(percentiles=[0.01, 0.5, 0.99]).T
        print(stats[["min", "1%", "50%", "99%", "max"]].to_string())
