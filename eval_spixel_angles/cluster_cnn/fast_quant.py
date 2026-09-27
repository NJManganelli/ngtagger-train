"""FAST class-C study: reduced-precision ONNX variants + their accuracy impact (host venv with onnxruntime 1.26.0 = the CMSSW version).

  venv/bin/python fast_quant.py <model.onnx:kind> ...
For each fp32 model writes <name>_fp16.onnx (onnxruntime.transformers.float16, fp32 I/O kept), <name>_int8dyn.onnx (quantize_dynamic, int8
weights; Conv -> ConvInteger) and <name>_int8qdq.onnx (quantize_static, QDQ, per-channel s8s8, MinMax calibration on 2048 class-C
clusters of the SECOND bench event), then compares every variant with fp32 on all class-C clusters of the bench bundle:
point angles [deg] (|d| RMS, p99, max, fraction > 0.1 / 1 deg), sigma ratio, and (a2 graphs) the drawn pseudo angle (histogram-bin
flips). Output: results/fast_quant_accuracy.csv. (Timing: fast_ort_bench.py quick, in the container.)
"""
import os, sys, numpy as np, pandas as pd, onnx, onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType, QuantFormat, CalibrationDataReader
from onnxruntime.quantization.shape_inference import quant_pre_process
from onnxruntime.transformers import float16   # (onnxconverter-common 1.16 crashes on keep_io_types with these graphs)

FAST = "/Volumes/WDMac/smartpixels_scratch/emu_fast"
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
D = np.load(os.path.join(FAST, "bench", "inputs.npz"))


def feeds(sess, reg, kind, a, b):
    f = {}
    for i in sess.get_inputs():
        if i.name == "image":
            f[i.name] = D[f"{reg}_img"][a:b]
        elif i.name == "scalars":
            f[i.name] = D[f"{reg}_scal_{'ref' if kind == 'ref' else 'fast'}"][a:b]
        elif i.name == "u":
            f[i.name] = D[f"{reg}_u"][a:b]
    return f


class Reader(CalibrationDataReader):
    def __init__(self, sess, reg, kind):
        a = int(D[f"{reg}_ev1"][0]); self.it = iter([feeds(sess, reg, kind, i, i + 256) for i in range(a, a + 2048, 256)])

    def get_next(self):
        return next(self.it, None)


def run(path, reg, kind):
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"]); n = len(D[f"{reg}_img"])
    return np.concatenate([s.run(None, feeds(s, reg, kind, i, min(i + 4096, n)))[0] for i in range(0, n, 4096)])


if __name__ == "__main__":
    rows = []
    for spec in sys.argv[1:]:
        path, kind = spec.rsplit(":", 1); name = os.path.basename(path)[:-5]; reg = "disc" if "disc" in name else "barrel"
        out = os.path.join(FAST, "onnx", "quant"); os.makedirs(out, exist_ok=True)
        pre = os.path.join(out, f"{name}_pre.onnx"); quant_pre_process(path, pre)
        v = {"fp16": os.path.join(out, f"{name}_fp16.onnx"), "int8dyn": os.path.join(out, f"{name}_int8dyn.onnx"),
             "int8qdq": os.path.join(out, f"{name}_int8qdq.onnx")}
        onnx.save(float16.convert_float_to_float16(onnx.load(path), keep_io_types=True), v["fp16"])
        quantize_dynamic(pre, v["int8dyn"], weight_type=QuantType.QInt8)
        s0 = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        quantize_static(pre, v["int8qdq"], Reader(s0, reg, kind), quant_format=QuantFormat.QDQ, per_channel=True,
                        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)
        ref = run(path, reg, kind)
        for tag, p in v.items():
            try:
                o = run(p, reg, kind)
            except Exception as e:
                rows.append(dict(model=name, variant=tag, error=str(e)[:200])); print(name, tag, "FAILED", e, flush=True); continue
            r = dict(model=name, variant=tag, n=len(o), size_kB=os.path.getsize(p) / 1024, fp32_size_kB=os.path.getsize(path) / 1024)
            for j, ang in ((0, "A"), (1, "B")):
                d = np.abs(np.degrees(o[:, j] - ref[:, j]))
                r.update({f"d{ang}_rms": float(np.sqrt(np.mean(d ** 2))), f"d{ang}_p99": float(np.quantile(d, 0.99)), f"d{ang}_max": float(d.max()),
                          f"d{ang}_gt0.1": float(np.mean(d > 0.1)), f"d{ang}_gt1": float(np.mean(d > 1))})
                sr = np.exp(o[:, 2 + j] - ref[:, 2 + j]); r[f"sig{ang}_ratio_p01_p99"] = f"{np.quantile(sr, 0.01):.3f}-{np.quantile(sr, 0.99):.3f}"
                if kind == "a2":
                    dp = np.abs(o[:, 6 + j] - ref[:, 6 + j])
                    r[f"pseudo{ang}_rms"] = float(np.sqrt(np.mean(dp ** 2))); r[f"pseudo{ang}_gt1"] = float(np.mean(dp > 1))
            rows.append(r); print(r, flush=True)
    T = pd.DataFrame(rows); f = os.path.join(RES, "fast_quant_accuracy.csv")
    if os.path.exists(f):
        o = pd.read_csv(f); T = pd.concat([o[~o.model.isin(T.model)], T])
    T.to_csv(f, index=False)
