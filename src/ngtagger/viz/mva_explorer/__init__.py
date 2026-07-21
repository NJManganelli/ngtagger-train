"""MVA explorer: one static, offline site that browses (1) correctionlib
regressions, (2) the Stage-3 refit-quality BDT and (3) the Stage-4 jet-tagger
outputs with a shared overlay/range/aggregation UI.

Architecture (mirrors eval_spixel's ratio builder, which this generalizes):
  * python exporters evaluate everything once and write compact typed binaries
    (int16 log10-quantized where values are strictly positive, float32 raw
    otherwise) plus JSON metadata;
  * a pure-JS compute core (site_src/explorer_core.js) does all slicing,
    binning, efficiency/AUC math in the browser;
  * site_src/explorer.html is the self-contained UI (local plotly.min.js, no
    CDN), served with `python -m http.server` from the rendered site dir.

Entry point: python -m ngtagger.viz.mva_explorer  (see __main__.py).
"""

from ngtagger.viz.mva_explorer.quantize import (  # noqa: F401
    LOG10_SCALE,
    dequantize_log10_int16,
    quantize_log10_int16,
)

# The user-canonical SmartPixels config ordering used EVERYWHERE a config
# dropdown appears: 0000 first, then combinatoric order = for k = 1..4 active
# layers, itertools.combinations of the layer positions (TBPX L1 L2 L3 L4):
# 1000 0100 0010 0001 1100 1010 1001 0110 0101 0011 1110 1101 1011 0111 1111.
from itertools import combinations as _combinations


def canonical_config_order(include_baseline: bool = True) -> list[str]:
    order = ["0000"] if include_baseline else []
    for k in range(1, 5):
        for pos in _combinations(range(4), k):
            bits = ["0"] * 4
            for p in pos:
                bits[p] = "1"
            order.append("".join(bits))
    return order
