"""Export a pos-hubsig cluster CNN to ONNX for SmartPixelsRecHitProducer (angleSource = "emu").

  emu_export_onnx.py MODELDIR OUT.onnx

Graph inputs (RAW; the graph applies the training normalisation (s - mu) / sd in float32 itself):
  image   float32 [N, 24, 16]  digi ADC + 1 (0 = no pixel), window as in prep.py
  scalars float32 [N, ns]      raw values of norm.json "cols", in that order (emu_common.scalars_v2)
output
  out     float32 [N, 6]       atan(cotAlpha), atan(cotBeta) [rad]; log sigma_alpha, log sigma_beta [rad]; tx, ty
decoded as in emu_common.decode. The column order is printed; the producer's emu*Scalars must repeat it.
"""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emu_common import load_net, norm_arrays

mdir, out = sys.argv[1], sys.argv[2]
net, nd = load_net(mdir)
mu, sd = norm_arrays(nd)


class Wrapped(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = net
        self.register_buffer("mu", torch.from_numpy(mu))
        self.register_buffer("sd", torch.from_numpy(sd))

    def forward(self, image, scalars):
        return self.net(image, (scalars - self.mu) / self.sd)


w = Wrapped().eval()
x = torch.zeros(2, 24, 16); x[:, 11:13, 7:9] = 5.0
s = torch.from_numpy(mu).repeat(2, 1)
torch.onnx.export(w, (x, s), out, input_names=["image", "scalars"], output_names=["out"],
                  dynamic_axes={"image": {0: "N"}, "scalars": {0: "N"}, "out": {0: "N"}}, opset_version=13, dynamo=False)
print("wrote", out, os.path.getsize(out), "bytes; scalars:", ",".join(nd["cols"]))
