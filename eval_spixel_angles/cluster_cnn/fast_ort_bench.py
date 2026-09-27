"""FAST class-C study: onnxruntime inference benchmark on REALISTIC class-C batches, meant to run INSIDE the CMSSW container with
the release's own onnxruntime (CMSSW_20_1_0_pre1: ORT 1.26.0, el9_aarch64, MLAS):
  docker exec cmssw-el9-native-1 bash /work/spx_arm_env.sh python3 /work/fast_bench/fast_ort_bench.py <mode> [...]
(copied to /work/fast_bench/ because the container does not mount ngtagger-train; inputs read from /host_volumes = WDMac, read-only).

Inputs: FAST/bench/inputs.npz (fast_export.py inputs): the class-C clusters of 2 test events, sorted by module (event, detId).
We time ONE event's worth per pass (the first event's modules: ~8k barrel + ~13k disc class-C clusters = the fixed class-T definition).
Batching modes: module (one Run per module, as SmartPixelsRecHitProducer), event (one Run for all class-C of the event and region),
chunkN (the event cut in fixed chunks of N).
Session options mirror cms::Ort::ONNXRuntime::defaultSessionOptions (intra_op_num_threads = 1; everything else ORT default:
graph optimisation ORT_ENABLE_ALL, sequential execution, arena allocator) unless swept.
modes:
  sweep  <out.json> <model.onnx:kind> ...   kind = ref | fast | a2 | a1  (the scalars / extra inputs to feed)
         grid: batching x intra {1, 2, 4} (module + event only) x graph opt {disable, basic, extended, all} (module, intra 1 only)
  quick  <out.json> <model.onnx:kind> ...   module + event batching, intra 1, opt all (for fp16 / int8 variants)
  streams <out.json> <model.onnx:kind> <nproc> <intra>   nproc concurrent processes (= CMSSW streams), each runs the module
         batches of the event with its own session; aggregate throughput (total threads nproc x intra <= 4 here)
"""
import os, sys, json, time, numpy as np, onnxruntime as ort

INP = "/host_volumes/WDMac/smartpixels_scratch/emu_fast/bench/inputs.npz"
OPT = {"disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL, "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
       "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED, "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL}


def load_inputs():
    d = np.load(INP); out = {}
    for reg in ("barrel", "disc"):
        out[reg] = dict(img=d[f"{reg}_img"], ref=d[f"{reg}_scal_ref"], fast=d[f"{reg}_scal_fast"], u=d[f"{reg}_u"],
                        starts=d[f"{reg}_starts"], ev1=int(d[f"{reg}_ev1"][0]))
    return out


def feeds(sess, D, kind, a, b):
    names = [i.name for i in sess.get_inputs()]
    f = {}
    for nm in names:
        if nm == "image":
            f[nm] = D["img"][a:b]
        elif nm == "scalars":
            f[nm] = D["ref" if kind == "ref" else "fast"][a:b]
        elif nm == "u":
            f[nm] = D["u"][a:b]
        elif nm == "z":
            f[nm] = np.ascontiguousarray(np.c_[D["fast"][a:b, :11], np.zeros((b - a, 4), np.float32)])
    return f


def event_rows(D):
    """module starts of the FIRST event (rows are sorted by event, then detId)"""
    st = D["starts"]; return st[st <= D["ev1"]]


def batches(D, mode):
    st = event_rows(D); n = int(st[-1])
    if mode == "module":
        return [(int(st[i]), int(st[i + 1])) for i in range(len(st) - 1)]
    if mode == "event":
        return [(0, n)]
    c = int(mode[5:]) if mode.startswith("chunk") else 1
    return [(i, min(i + c, n)) for i in range(0, n, c)]


def session(path, intra=1, opt="all", mode="seq", inter=1):
    so = ort.SessionOptions(); so.intra_op_num_threads = intra; so.inter_op_num_threads = inter
    so.graph_optimization_level = OPT[opt]
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL if mode == "seq" else ort.ExecutionMode.ORT_PARALLEL
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def timeit(sess, D, kind, bl, reps=2):
    fl = [feeds(sess, D, kind, a, b) for a, b in bl]
    for f in fl[: min(len(fl), 50)]:
        sess.run(None, f)                                    # warm-up
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for f in fl:
            sess.run(None, f)
        ts.append(time.perf_counter() - t0)
    n = bl[-1][1] - bl[0][0]
    return dict(n=int(n), nbatch=len(bl), sec=float(min(ts)), us_per_cluster=float(min(ts) / n * 1e6))


def stream_worker(args):
    path, kind, intra, reg, q, bar = args
    D = load_inputs()[reg]; s = session(path, intra); bl = batches(D, "module")
    fl = [feeds(s, D, kind, a, b) for a, b in bl]
    for f in fl[:50]:
        s.run(None, f)
    q.put("ready"); bar.wait(); t0 = time.perf_counter()
    for _ in range(3):
        for f in fl:
            s.run(None, f)
    q.put((time.perf_counter() - t0, 3 * (bl[-1][1] - bl[0][0])))


if __name__ == "__main__":
    mode, outp = sys.argv[1], sys.argv[2]
    res = json.load(open(outp)) if os.path.exists(outp) else []
    D = load_inputs()
    if mode in ("sweep", "quick"):
        for spec in sys.argv[3:]:
            path, kind = spec.rsplit(":", 1); name = os.path.basename(path)
            reg = "disc" if "disc" in name else "barrel"
            grid = []
            for bm in (["module", "event"] if mode == "quick" else ["module", "event", "chunk1", "chunk64", "chunk256", "chunk1024", "chunk4096"]):
                grid.append((bm, 1, "all", "seq"))
            if mode == "sweep":
                for bm in ("module", "event"):
                    grid += [(bm, 2, "all", "seq"), (bm, 4, "all", "seq")]
                grid += [("module", 1, o, "seq") for o in ("disable", "basic", "extended")]
                grid += [("module", 1, "all", "par"), ("event", 4, "all", "par")]
            for bm, intra, opt, em in grid:
                s = session(path, intra, opt, em, inter=(2 if em == "par" else 1))
                r = dict(model=name, region=reg, batching=bm, intra=intra, opt=opt, exec=em, **timeit(s, D[reg], kind, batches(D[reg], bm)))
                print(json.dumps(r), flush=True); res.append(r)
                json.dump(res, open(outp, "w"), indent=1)
    elif mode == "streams":
        import multiprocessing as mp
        path, kind = sys.argv[3].rsplit(":", 1); nproc, intra = int(sys.argv[4]), int(sys.argv[5])
        name = os.path.basename(path); reg = "disc" if "disc" in name else "barrel"
        ctx = mp.get_context("fork"); q = ctx.Queue(); bar = ctx.Barrier(nproc)
        ps = [ctx.Process(target=stream_worker, args=((path, kind, intra, reg, q, bar),)) for _ in range(nproc)]
        [p.start() for p in ps]
        got = [q.get() for _ in range(2 * nproc)]; [p.join() for p in ps]
        tm = [g for g in got if g != "ready"]
        tot_n = sum(n for _, n in tm); wall = max(t for t, _ in tm)
        r = dict(model=name, region=reg, batching="module", streams=nproc, intra=intra, wall=wall, clusters=tot_n,
                 us_per_cluster_aggregate=wall / tot_n * 1e6, us_per_cluster_per_stream=float(np.mean([t / n * 1e6 for t, n in tm])))
        print(json.dumps(r), flush=True); res.append(r); json.dump(res, open(outp, "w"), indent=1)
