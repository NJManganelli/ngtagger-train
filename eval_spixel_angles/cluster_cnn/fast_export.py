"""(run with the study venv, emu_fast/venv: the pixi env has no `onnx`, which torch.onnx.export needs)
FAST class-C study: ONNX exports + the realistic benchmark input bundle (for fast_ort_bench.py, run in the CMSSW container).

  fast_export.py models <tag> [<tag> ...]    FastNet -> FAST/onnx/<tag>_point.onnx (image, scalars -> out[6], as the production
                                             graph: RAW scalars, normalisation inside) and <tag>_a2.onnx (image, scalars, u[4] ->
                                             out[8]: point 4 + drawn rA, rB + pseudo angles, the single A2 network)
  fast_export.py a1 <src> <region>           A1 arh model -> FAST/onnx/a1_arh_<src>_<region>.onnx (obs[15] raw, u[4] -> r[2])
  fast_export.py inputs                      FAST/bench/inputs.npz: the class-C clusters of 2 test events (barrel + ID1-4), grouped
                                             by module (event, detId) as the producer batches them, with the raw scalars of every model
"""
import os, sys, json, numpy as np, pandas as pd, torch
from fast_models import FastNet, A2Export, raw_scalars, ORIENT, EDGES, NB, renc
from fast_common import FAST, V2, load_test, classes, uniforms

ON = os.path.join(FAST, "onnx"); os.makedirs(ON, exist_ok=True)


def load_fast(tag):
    nd = json.load(open(os.path.join(FAST, "models", tag, "norm.json")))
    net = FastNet(nd["arch"], len(nd["cols"]), nd["wx"], nd["wy"])
    net.load_state_dict(torch.load(os.path.join(FAST, "models", tag, "model.pt"), map_location="cpu")); net.eval()
    return net, nd


class Norm(torch.nn.Module):
    def __init__(self, inner, nd, a2=False):
        super().__init__(); self.inner, self.a2 = inner, a2
        self.register_buffer("mu", torch.tensor([nd["mu"][c] for c in nd["cols"]], dtype=torch.float32))
        self.register_buffer("sd", torch.tensor([nd["sd"][c] for c in nd["cols"]], dtype=torch.float32))

    def forward(self, image, scalars, u=None):
        s = (scalars - self.mu) / self.sd
        return self.inner(image, s, u) if self.a2 else self.inner(image, s)


def scalars_v2(m, region, cols):
    """raw scalar inputs of the retired pos-hubsig CNN (v2_train.feats --inputs pos), for the benchmark bundle"""
    from models_def import scalars as base_scalars
    from v2_common import derived
    if "pu" not in m:
        m = pd.concat([m.reset_index(drop=True), derived(m.reset_index(drop=True))], axis=1)
    s = base_scalars(m, "base" if region == "barrel" else "disc")
    s["is3D"] = ((m.subdet == 1) & (m.layer == 1)).values.astype(np.float32)
    for c in ["axXz", "axYz", "axZz", "axXr", "axYr", "axZr", "sideZ", "pu", "pv", "pw", "globalR", "globalZ", "localX", "localY"]:
        s[c] = m[c].values
    s = s.astype(np.float32)
    assert list(s.columns) == list(cols), (list(s.columns), cols)
    return s.values


def export(mod, args, names, path):
    torch.onnx.export(mod, args, path, input_names=names, output_names=["out"],
                      dynamic_axes={**{k: {0: "N"} for k in names}, "out": {0: "N"}}, opset_version=17, dynamo=False)


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "models":
        for tag in sys.argv[2:]:
            net, nd = load_fast(tag); ns = len(nd["cols"])
            x = torch.zeros(4, 24, 16); x[:, 11:13, 7:9] = 5.0; s = torch.tensor([nd["mu"][c] for c in nd["cols"]]).repeat(4, 1)
            export(Norm(net, nd).eval(), (x, s), ["image", "scalars"], os.path.join(ON, f"{tag}_point.onnx"))
            export(Norm(A2Export(net), nd, a2=True).eval(), (x, s, torch.full((4, 4), 0.5)), ["image", "scalars", "u"],
                   os.path.join(ON, f"{tag}_a2.onnx"))
            print("exported", tag, flush=True)
    elif what == "a1":
        from fast_a1 import ARH, observables
        src, reg = sys.argv[2], sys.argv[3]
        sd = torch.load(os.path.join(FAST, "models", f"a1_arh_{src}_{reg}.pt")); net = ARH(15); net.load_state_dict(sd); net.eval()

        class A1(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.net = net; self.register_buffer("edges", torch.tensor(EDGES, dtype=torch.float32))

            def draw(self, lg, ub, ui):
                c = torch.cumsum(torch.softmax(lg, 1), 1); k = torch.clamp((c < ub[:, None]).sum(1), max=NB - 1)
                return self.edges[k] + ui * (self.edges[k + 1] - self.edges[k])

            def forward(self, z, u):
                h = self.net.t(z); rA = self.draw(self.net.logitsA(h), u[:, 0], u[:, 1])
                rB = self.draw(self.net.logitsB(h, rA), u[:, 2], u[:, 3])
                return torch.stack([rA, rB], 1)
        export(A1().eval(), (torch.zeros(4, 15), torch.full((4, 4), 0.5)), ["z", "u"], os.path.join(ON, f"a1_arh_{src}_{reg}.onnx"))
        print("exported a1", src, reg)
    elif what == "inputs":
        out = {}
        for reg in ("barrel", "disc"):
            m = load_test(reg); valid, clsT = classes(m)
            ev = np.sort(m.event.unique())[:2]
            sel = np.where(m.event.isin(ev).values & ~clsT)[0]
            mm = m.iloc[sel].reset_index(drop=True)
            order = np.lexsort((mm.detId.values, mm.event.values)); mm = mm.iloc[order].reset_index(drop=True); sel = sel[order]
            img = np.asarray(np.load(os.path.join(V2, "arrays", f"{reg}_test_img.npy"), mmap_mode="r")[np.sort(sel)])
            img = img[np.argsort(np.argsort(sel))]                 # rows back in (event, detId) order
            key = mm.event.values.astype(np.int64) * (1 << 32) + mm.detId.values.astype(np.int64)
            _, starts = np.unique(key, return_index=True)
            ref_nd = json.load(open(os.path.join(V2, "models", f"{reg}_pos_hubsig", "norm.json")))
            S_ref = scalars_v2(mm, reg, ref_nd["cols"])
            R = raw_scalars(mm, reg)
            for c in ORIENT:
                R[c] = R[c].round(6)
            fnd = json.load(open(os.path.join(FAST, "models", f"{reg}_full", "norm.json")))
            out[f"{reg}_img"] = img.astype(np.float32); out[f"{reg}_scal_ref"] = S_ref.astype(np.float32)
            out[f"{reg}_scal_fast"] = R[fnd["cols"]].values.astype(np.float32); out[f"{reg}_starts"] = np.r_[starts, len(mm)]
            out[f"{reg}_u"] = uniforms(mm).astype(np.float32); out[f"{reg}_ev1"] = np.array([int((mm.event.values == ev[0]).sum())])
            print(reg, "class-C clusters", len(mm), "in", len(ev), "events;", len(starts), "module batches; median batch",
                  np.median(np.diff(np.r_[starts, len(mm)])), "mean", np.mean(np.diff(np.r_[starts, len(mm)])), flush=True)
        np.savez(os.path.join(FAST, "bench", "inputs.npz"), **out)
