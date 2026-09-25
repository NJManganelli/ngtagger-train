"""Export a trained cluster CNN to ONNX for SmartPixelsRecHitProducer (angleSource = "cnn").

  export_onnx.py TAG OUT.onnx [MODELDIR]

The graph takes the RAW inputs, so the producer never re-implements the normalisation:
  image   float32 [N, 24, 16]  digi ADC + 1 (0 = no pixel), window as in prep.py
  scalars float32 [N, ns]      raw values of norm.json "cols", in that order
and returns
  out     float32 [N, 6]       atan|f_a|, atan|f_b| (softplus), sign logits a/b, tx, ty
with cotAlpha = sign(out2) * tan(out0) - c_layer and cotBeta = sign(out3) * tan(out1), a
positive sign meaning out > 0 (see apply_all.py). The scalar column order is printed and
stored in the model metadata ("scalars").
"""
import sys, os, json, torch, numpy as np
from models_def import make

tag, out = sys.argv[1], sys.argv[2]
mdir = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
nd = json.load(open(os.path.join(mdir, tag, "norm.json")))
cols = nd["cols"]
net = make(nd["variant"], len(cols))
net.load_state_dict(torch.load(os.path.join(mdir, tag, "model.pt"), map_location="cpu"))
net.eval()


class Wrapped(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer("mu", torch.tensor([nd["mu"][c] for c in cols], dtype=torch.float32))
        self.register_buffer("sd", torch.tensor([nd["sd"][c] for c in cols], dtype=torch.float32))

    def forward(self, image, scalars):
        return self.net(image, (scalars - self.mu) / self.sd)


w = Wrapped(net).eval()
x = torch.zeros(2, 24, 16); x[:, 11:13, 7:9] = 5.0
s = torch.ones(2, len(cols))
torch.onnx.export(w, (x, s), out, input_names=["image", "scalars"], output_names=["out"],
                  dynamic_axes={"image": {0: "N"}, "scalars": {0: "N"}, "out": {0: "N"}}, opset_version=13, dynamo=False)
try:
    import onnx
    m = onnx.load(out)
    m.metadata_props.add(key="scalars", value=",".join(cols))
    m.metadata_props.add(key="variant", value=nd["variant"])
    onnx.save(m, out)
except ImportError:
    pass
print("wrote", out, "variant", nd["variant"], "scalars", ",".join(cols))
