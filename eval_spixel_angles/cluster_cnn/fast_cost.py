"""FAST class-C study: cost model + options table + fast_cnn_tradeoff.pdf / fast_ort.pdf.

Inputs: results/fast_ort_bench.json (fast_ort_bench.py in the CMSSW container: ORT 1.26.0, 4 vCPU aarch64 VM on the M4 Pro =
the machine of the CMSSW timing), results/fast_quant_accuracy.csv, results/fast_eval_{barrel,disc}.csv, FAST/models/*/trainlog.json.

END-TO-END MODEL (per event, 4 streams, CPU summed over streams unless stated):
  measured CMSSW emu job (200 PU200 events, 4 threads, erroneous pT cut: N_C = 34.3k class-C/ev):
    CNN step 14.63 s (= ORT Run + the per-module input assembly: digi map + window + scalars), donor kNN 8.54, dump kNN 0.378,
    helix 0.213, draw 0.182 s; wall 8.23 s/ev. => non-emu wall W0 = 8.23 - 23.94 / 4 = 2.25 s/ev.
  input assembly per class-C cluster t_in = 14.63 s / 34.3k - t_ORT(ref, module batches, intra 1, this bench).
  option: wall = W0 + [N_C (t_in' + t_net + t_draw) + (helix + dump + draw)] / 4, N_C = 21.4k (fixed class T, test events).
    knn donor with the fixed class T: 8.54 x 21.4/34.3 (per-query cost ~ unchanged: log of the library size).
    t_in' = t_in for image networks; t_in(A1 obs only) ~ 0 for the donor model (its inputs are the CNN outputs + cluster scalars).
Payload reference: 2.46 s/ev.
"""
import os, json, glob, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from fast_common import FAST, RES, FIGS
from fast_plots import table_pages, fmt

CMSSW = dict(cnn=14.63, donor=8.54, dump=0.378, helix=0.213, draw=0.182, wall=8.23, nc_old=34.3e3, payload=2.46)
NC = {"barrel": 8283, "disc": 13139}; NC_T = sum(NC.values())
W0 = CMSSW["wall"] - (CMSSW["cnn"] + CMSSW["donor"] + CMSSW["dump"] + CMSSW["helix"] + CMSSW["draw"]) / 4
FIXED = CMSSW["dump"] + CMSSW["helix"] + CMSSW["draw"]
B = pd.DataFrame(json.load(open(os.path.join(RES, "fast_ort_bench.json"))))


def us(model, batching="module", intra=1, opt="all", ex="seq"):
    s = B[(B.model == model) & (B.batching == batching) & (B.intra == intra) & (B.opt == opt) & (B["exec"] == ex)]
    return float(s.us_per_cluster.min()) if len(s) else np.nan


def per_cluster(fmt_, bat="module"):
    """class-C-weighted ORT us/cluster over barrel + disc for a model name pattern ({reg})"""
    return sum(NC[r] * us(fmt_.format(reg=r), bat) for r in NC) / NC_T


if __name__ == "__main__":
    t_ref = per_cluster("spix_emu_cnn_{reg}_v1.onnx")
    t_in = CMSSW["cnn"] / CMSSW["nc_old"] * 1e6 - t_ref                     # us per class-C cluster
    knn_new = CMSSW["donor"] * NC_T / CMSSW["nc_old"]
    rows = []

    def add(name, cnn_us, draw_us, donor_s, note):
        cpu = NC_T * (t_in + cnn_us + draw_us) * 1e-6 + donor_s + FIXED
        rows.append(dict(option=name, cnn_us=round(cnn_us, 1), draw_us=round(draw_us, 2), input_us=round(t_in, 1),
                         donor_knn_s=round(donor_s, 2), emu_cpu_s_per_ev=round(cpu, 2), wall_s_per_ev=round(W0 + cpu / 4, 2),
                         ratio_to_payload=round((W0 + cpu / 4) / CMSSW["payload"], 2), note=note))
    add("CURRENT as measured (pT-cut error, 34.3k C/ev)", np.nan, 0, CMSSW["donor"], "measured 8.23 s/ev")
    rows[-1].update(emu_cpu_s_per_ev=23.94, wall_s_per_ev=8.23, ratio_to_payload=round(8.23 / 2.46, 2))
    add("ref CNN + knn, fixed class T", t_ref, 0, knn_new, "fix only")
    add("ref CNN + A1 arh", t_ref, per_cluster("a1_arh_ref_{reg}.onnx"), 0, "A1 on the current CNN")
    for arch in ("full", "narrow", "dws", "tiny", "proj1d"):
        t_pt = per_cluster(f"{{reg}}_{arch}_point.onnx"); t_a2 = per_cluster(f"{{reg}}_{arch}_a2.onnx")
        t_a1 = per_cluster(f"a1_arh_{{reg}}_{arch}_{{reg}}.onnx")
        add(f"{arch} CNN + knn", t_pt, 0, knn_new, "B only")
        if np.isfinite(t_a1):                                                  # (no A1 was trained on dws)
            add(f"{arch} CNN + A1 arh", t_pt, t_a1, 0, "A1")
        add(f"A2 {arch} (single network)", t_a2, 0, 0, "A2")
    add("A2 tiny fp16 (aarch64 fp16 kernels; x86 untested)", us("barrel_tiny_a2_fp16.onnx"), 0, 0, "barrel timing used for both regions")
    add("tiny CNN fp16 + A1 arh", 0.5 * (us("barrel_tiny_point_fp16.onnx") + us("disc_tiny_point_fp16.onnx")), per_cluster("a1_arh_{reg}_tiny_{reg}.onnx"), 0, "A1, fp16 CNN")
    O = pd.DataFrame(rows); O.to_csv(os.path.join(RES, "fast_cost_options.csv"), index=False)
    pd.set_option("display.width", 250); print(O.to_string(index=False)); print("t_ref", t_ref, "t_in", t_in, "W0", W0)
    # --- figures
    info = []
    for f in glob.glob(os.path.join(FAST, "models", "*", "trainlog.json")):
        i = json.load(open(f))["info"]; info.append(dict(tag=i["tag"], macs=i["point_macs"], params=i["point_params"], total_params=i["total_params"]))
    I = pd.DataFrame(info)
    with PdfPages(os.path.join(FIGS, "fast_ort.pdf")) as pdf:
        fig, ax = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
        order = ["chunk1", "chunk64", "module", "chunk256", "chunk1024", "chunk4096", "event"]
        for k, reg in enumerate(("barrel", "disc")):
            x = ax[k]
            for j, mdl in enumerate(sorted(B[(B.region == reg) & (B.intra == 1) & (B.opt == "all") & (B["exec"] == "seq")].model.unique())):
                s = B[(B.model == mdl) & (B.intra == 1) & (B.opt == "all") & (B["exec"] == "seq")].groupby("batching").us_per_cluster.min().reindex(order)
                x.plot(range(len(order)), s.values, marker="o", label=mdl)
            x.set_xticks(range(len(order)), order, rotation=30); x.set_yscale("log"); x.set_ylabel("us / cluster (1 thread)"); x.set_title(f"{reg}: batching", fontsize=10)
            x.grid(True, color="#eeeeee")
        x = ax[2]; s = B[(B.batching.isin(["module", "event"])) & (B.opt == "all") & (B["exec"] == "seq")]
        for (mdl, bat), g in s.groupby(["model", "batching"]):
            if mdl.startswith(("barrel_full_point", "barrel_tiny_point", "spix_emu_cnn_barrel")) and "_fp16" not in mdl and "int8" not in mdl:
                g = g.groupby("intra").us_per_cluster.min().reset_index(); x.plot(g.intra, g.us_per_cluster, marker="s", label=f"{mdl} [{bat}]")
        x.set_xlabel("intra_op threads"); x.set_ylabel("us / cluster (wall)"); x.set_yscale("log"); x.set_title("intra-op threads", fontsize=10); x.grid(True, color="#eeeeee")
        hl = {}
        for a_ in ax:
            for hh, ll in zip(*a_.get_legend_handles_labels()):
                hl.setdefault(ll, hh)
        fig.legend(list(hl.values()), list(hl.keys()), loc="outside lower center", ncol=4, fontsize=7.5, frameon=False)
        pdf.savefig(fig); plt.close(fig)
        table_pages(pdf, fmt(B.fillna("")), "onnxruntime 1.26.0 (CMSSW_20_1_0_pre1 external), container aarch64, us per class-C cluster", per=30)
        q = os.path.join(RES, "fast_quant_accuracy.csv")
        if os.path.exists(q):
            table_pages(pdf, fmt(pd.read_csv(q).fillna("")), "reduced precision: accuracy vs fp32 (bench bundle, class-C clusters of 2 test events)")
        table_pages(pdf, fmt(O.fillna("")), f"end-to-end estimate (W0 = {W0:.2f} s/ev non-emu wall, input assembly {t_in:.0f} us/cluster, N_C = {NC_T}/ev)")
    with PdfPages(os.path.join(FIGS, "fast_cnn_tradeoff.pdf")) as pdf:
        E = pd.concat([pd.read_csv(os.path.join(RES, f"fast_eval_{r}.csv")).assign(region=r) for r in ("barrel", "disc")])
        E = E[E.angle != "corr"]
        fig, ax = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
        G = [("barrel", "IL2-4"), ("barrel", "IL1"), ("disc", "ID1-4")]
        for j, (reg, g) in enumerate(G):
            for i, ang in enumerate(("alpha", "beta")):
                x = ax[i, j]
                for draw, mk in (("knn", "o"), ("a1arh", "s"), ("a2", "^")):
                    pts = []
                    for _, r in I[I.tag.str.startswith(reg)].iterrows():
                        e = E[(E.variant == f"{r.tag}:{draw}") & (E.group == g) & (E.angle == ang)]
                        if len(e):
                            pts.append((r.macs, e.full.values[0], e.cnn_full.values[0]))
                    if pts:
                        pts = np.array(sorted(pts)); x.plot(pts[:, 0], pts[:, 1], marker=mk, label=f"class-C route full RMS: {draw}")
                        if draw == "knn":
                            x.plot(pts[:, 0], pts[:, 2], marker="x", ls=":", color="#777777", label="CNN point estimate alone: full RMS")
                t = E[(E.variant == "classT_route") & (E.group == g) & (E.angle == ang)]
                if len(t):
                    x.axhline(t.full.values[0], color="#9e9e9e", lw=2, label="class-T route (target)")
                rk = E[(E.variant == "ref:knn") & (E.group == g) & (E.angle == ang)]
                if len(rk):
                    x.axhline(rk.full.values[0], color="#222222", ls="--", label="current: ref CNN + knn")
                x.set_xscale("log"); x.set_xlabel("CNN MACs / cluster"); x.set_ylabel("full RMS vs truth [deg]"); x.set_title(f"{g} {ang}", fontsize=10); x.grid(True, color="#eeeeee")
        hl = {}
        for a_ in ax.flat:
            for hh, ll in zip(*a_.get_legend_handles_labels()):
                hl.setdefault(ll, hh)
        fig.legend(list(hl.values()), list(hl.keys()), loc="outside lower center", ncol=3, fontsize=8, frameon=False)
        pdf.savefig(fig); plt.close(fig)
        table_pages(pdf, fmt(I.sort_values("tag")), "FastNet sizes (point head = the CNN; total incl. the A2 ARH head); MACs per cluster")
