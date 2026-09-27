"""Throughput table for the angle sources: steady-state per-event time from the 'Begin processing' timestamps (events
21..N; the first 20 = warm-up, reported separately), wall/CPU/peak RSS from /usr/bin/time -v, the per-module
FastTimerService breakdown, and the emu producer's internal step times. Ratios are to the mode named "payload".

  perf_report.py PERFDIR:mode[:label] [...]   (reads PERFDIR/log_<mode>.txt, PERFDIR/timing_<mode>.json)
-> emu_cmssw/perf_report.json, figs/emu_perf_table.pdf (table pages)"""
import sys, os, re, json, datetime as dt, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WARM = 20
rows, mods, steps = [], [], []
for arg in sys.argv[1:]:
    d, mode, *lab = arg.split(":")
    log = open(os.path.join(d, f"log_{mode}.txt"), errors="replace").read()
    mode = lab[0] if lab else mode
    ts = [dt.datetime.strptime(m, "%d-%b-%Y %H:%M:%S.%f") for m in re.findall(r"Begin processing the \S+ record\..* at (\S+ \S+) UTC", log)]
    ts = np.array([(t - ts[0]).total_seconds() for t in ts])
    n = len(ts)
    wall = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", log).group(1).split(":")
    wall = sum(float(x) * 60 ** i for i, x in enumerate(reversed(wall)))
    user = float(re.search(r"User time \(seconds\): (\S+)", log).group(1)); sysc = float(re.search(r"System time \(seconds\): (\S+)", log).group(1))
    rss = float(re.search(r"Maximum resident set size \(kbytes\): (\d+)", log).group(1)) / 1024 ** 2
    steady = (ts[-1] - ts[WARM]) / (n - 1 - WARM)
    js = json.load(open(os.path.join(d, f"timing_{arg.split(':')[1]}.json")))
    tot = js["total"]
    rows.append({"mode": mode, "events": n, "steady s/event (begin-processing spacing, events > 20)": round(steady, 2),
                 "steady events/s": round(1 / steady, 4), "first 20 events [s]": round(ts[WARM], 1),
                 "job wall [s]": round(wall, 1), "job CPU user+sys [s]": round(user + sysc, 1), "CPU/event (job avg) [s]": round((user + sysc) / n, 2),
                 "peak RSS [GB]": round(rss, 2), "FastTimer event loop real [s]": round(tot.get("time_real", np.nan) / 1000, 1)})
    for m in js["modules"]:
        mods.append({"mode": mode, "module": m["label"], "type": m["type"], "real/event [ms]": m["time_real"] / max(m["events"], 1),
                     "cpu/event [ms]": m["time_thread"] / max(m["events"], 1), "events": m["events"]})
    s = re.search(r"time summed over streams \[s\]:(.*)", log)
    if s:
        for part in s.group(1).split(";"):
            if part.strip():
                k, v = part.strip().rsplit(" ", 1); steps.append({"mode": mode, "step": k, "total [s]": float(v), "per event [ms]": 1000 * float(v) / n})
    s = re.search(r"emu setup (\S+) s", log)
    if s:
        steps.append({"mode": mode, "step": "one-time setup (ONNX sessions, library load, KD-tree builds)", "total [s]": float(s.group(1)), "per event [ms]": np.nan})
R = pd.DataFrame(rows); M = pd.DataFrame(mods); S = pd.DataFrame(steps)
ref = R.set_index("mode")["steady s/event (begin-processing spacing, events > 20)"]
if "payload" in ref:
    R["ratio to payload"] = (R["steady s/event (begin-processing spacing, events > 20)"] / ref["payload"]).round(3)
top = M.sort_values("real/event [ms]", ascending=False).groupby("mode").head(12)
spx = M[M.module.str.startswith("spix")]
out = {"summary": R.to_dict("records"), "top_modules": top.to_dict("records"), "spix_modules": spx.to_dict("records"), "emu_steps": S.to_dict("records")}
json.dump(out, open(os.path.join(HERE, "emu_cmssw", "perf_report.json"), "w"), indent=1, default=float)


def page(pdf, df, title, fs=8):
    f2, a2 = plt.subplots(figsize=(17, 0.3 * len(df) + 1.3)); a2.axis("off")
    tb = a2.table(cellText=df.values, colLabels=[c for c in df.columns], loc="center", cellLoc="left", colLoc="left")
    tb.auto_set_font_size(False); tb.set_fontsize(fs); tb.auto_set_column_width(list(range(df.shape[1])))
    a2.set_title(title, fontsize=10); pdf.savefig(f2, bbox_inches="tight"); plt.close(f2)


with PdfPages(os.path.join(HERE, "figs", "emu_perf_table.pdf")) as pdf:
    page(pdf, R.T.reset_index().rename(columns={"index": "quantity"}), "Throughput, 200 PU200 ttbar events (RelVal files 3+4), 4 threads, arm (cmssw-el9-native-1)")
    t = top.copy(); t[["real/event [ms]", "cpu/event [ms]"]] = t[["real/event [ms]", "cpu/event [ms]"]].round(1)
    page(pdf, t, "Top-12 modules per mode (FastTimerService, all events)")
    t = spx.copy(); t[["real/event [ms]", "cpu/event [ms]"]] = t[["real/event [ms]", "cpu/event [ms]"]].round(1)
    page(pdf, t, "SmartPixels modules (spix*) per mode")
    if len(S):
        t = S.copy(); t[["total [s]", "per event [ms]"]] = t[["total [s]", "per event [ms]"]].round(2)
        page(pdf, t, "SmartPixelsRecHitProducer emu steps (wall time summed over the 4 streams)")
pd.set_option("display.width", 250)
print(R.to_string()); print(spx.round(1).to_string()); print(S.to_string())
