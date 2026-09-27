"""(run with the export venv, /Volumes/WDMac/smartpixels_scratch/emu_fast/venv: the pixi env has no `onnx`)
Export the SmartPixels-output emulator networks to ONNX for SmartPixelsRecHitProducer (angleSource = "emu").

  emu_export_onnx.py cnn MODELDIR OUT.onnx     the tiny CNN (fast_models.FastNet point head)
      inputs  image   float32 [N, 24, 16]  digi ADC + 1 (0 = no pixel), window as in prep.py
              scalars float32 [N, ns]      RAW values of norm.json "cols", in that order, orientation columns rounded to 1e-6
                                           (emu_common.cnn_scalars); the graph applies (s - mu) / sd in float32
      output  out     float32 [N, 6]       atan(cotAlpha), atan(cotBeta) [rad]; log sigma_alpha, log sigma_beta [rad]; tx, ty
  emu_export_onnx.py a1 A1.pt OUT.onnx         the A1 class-C residual model (emu_train_a1.py), draw INSIDE the graph
      inputs  z       float32 [N, 15]      RAW A1 inputs (emu_common.a1_inputs); the graph standardises them
              u       float32 [N, 4]       hash uniforms (emu_common.a1_uniforms)
      output  r       float32 [N, 2]       drawn residuals rA, rB [deg] (pseudo-truth = CNN angle + r)
The CNN scalar column order is printed; the producer's emu*Scalars must repeat it.
"""
import sys, os, json, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emu_common import load_cnn, load_a1, N_A1_IN
from fast_models import EDGES, NB


def export(mod, args, inames, oname, path):
    torch.onnx.export(mod, args, path, input_names=inames, output_names=[oname],
                      dynamic_axes={**{k: {0: "N"} for k in inames}, oname: {0: "N"}}, opset_version=17, dynamo=False)


class CnnGraph(torch.nn.Module):
    def __init__(self, net, nd):
        super().__init__(); self.net = net
        self.register_buffer("mu", torch.tensor([nd["mu"][c] for c in nd["cols"]], dtype=torch.float32))
        self.register_buffer("sd", torch.tensor([nd["sd"][c] for c in nd["cols"]], dtype=torch.float32))

    def forward(self, image, scalars):
        return self.net(image, (scalars - self.mu) / self.sd)


class A1Graph(torch.nn.Module):
    def __init__(self, a1):
        super().__init__(); self.net, mu, sd = a1
        self.register_buffer("mu", torch.from_numpy(mu)); self.register_buffer("sd", torch.from_numpy(sd))
        self.register_buffer("edges", torch.tensor(EDGES, dtype=torch.float32))

    def draw(self, lg, ub, ui):
        c = torch.cumsum(torch.softmax(lg, 1), 1); k = torch.clamp((c < ub[:, None]).sum(1), max=NB - 1)
        return self.edges[k] + ui * (self.edges[k + 1] - self.edges[k])

    def forward(self, z, u):
        h = self.net.t((z - self.mu) / self.sd)
        rA = self.draw(self.net.logitsA(h), u[:, 0], u[:, 1])
        rB = self.draw(self.net.logitsB(h, rA), u[:, 2], u[:, 3])
        return torch.stack([rA, rB], 1)


if __name__ == "__main__":
    what, src, out = sys.argv[1:4]
    if what == "cnn":
        net, nd = load_cnn(src)
        x = torch.zeros(2, 24, 16); x[:, 11:13, 7:9] = 5.0
        s = torch.tensor([nd["mu"][c] for c in nd["cols"]], dtype=torch.float32).repeat(2, 1)
        export(CnnGraph(net, nd).eval(), (x, s), ["image", "scalars"], "out", out)
        print("wrote", out, os.path.getsize(out), "bytes; scalars:", ",".join(nd["cols"]))
    elif what == "a1":
        export(A1Graph(load_a1(src)).eval(), (torch.zeros(2, N_A1_IN), torch.full((2, 4), 0.5)), ["z", "u"], "r", out)
        print("wrote", out, os.path.getsize(out), "bytes")
