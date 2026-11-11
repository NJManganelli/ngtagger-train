"""The arm registry (PLAN.md section 5): every arm with its name in both frames and its parameters, written to a JSON
file the runner reads.

An allocation is written x/y/alpha/beta in the SmartPixels frame: bits per output over the 2.4 mm sub-area (positions)
or the link's range (angles); "F" = float, "0" = not sent. In the replay's code frame the SmartPixels x is localY, y is
localX, alpha is the code's "beta" and beta (the bending angle) is the code's "alpha".

  python registry.py [out.json]
"""
import json
import sys

SUBAREA_UM = 2400.0
GROUP_ORDER = ("priority", "fixed", "scan_down", "scan_up", "bending", "allocations", "controls")


def step(bits):
    """Step [um] of a sub-area position code with the given bits (a whole-sensor code with bits + 2)."""
    return None if bits in (None, "F") else SUBAREA_UM / 2 ** bits


def _fmt(v, kind):
    if v is None:
        return "F"
    if v == 0:
        return "0"
    return f"{v:g}um" if kind == "pos" else f"{v}b"


def arm(name, group, x="F", y="F", a="F", b="F", today=False, whole=False, **extra):
    """x, y: sub-area bits or "F" (or a step in um as a float through step_x_um / step_y_um in extra); a, b: bits, "F"
    or 0 (not sent) for the SmartPixels alpha and beta."""
    spec = {"step_x_um": step(x), "step_y_um": step(y)}
    spec.update({k: extra.pop(k) for k in ("step_x_um", "step_y_um") if k in extra})
    if today:
        spec.update(angles="today", cap=False)
    else:
        send = [n for n, v in (("alpha", a), ("beta", b)) if v != 0]
        spec.update(angles="network", cap=True, send=send, pos_err="sizeblind", ang_err="linktable",
                    bits_alpha=a if isinstance(a, int) and a > 0 else None,
                    bits_beta=b if isinstance(b, int) and b > 0 else None, range="link", window_add=True,
                    flag="onesided")
    spec.update(extra)
    sx, sy = spec["step_x_um"], spec["step_y_um"]
    ba = 0 if (not today and "alpha" not in spec["send"]) else spec.get("bits_alpha")
    bb = 0 if (not today and "beta" not in spec["send"]) else spec.get("bits_beta")
    sp = f"x {_fmt(sx, 'pos')} / y {_fmt(sy, 'pos')} / alpha {_fmt(ba, 'ang')} / beta {_fmt(bb, 'ang')}"
    code = f"localY {_fmt(sx, 'pos')} / localX {_fmt(sy, 'pos')} / code beta {_fmt(ba, 'ang')} / code alpha {_fmt(bb, 'ang')}"
    return {"name": name, "group": group, "smartpixels_frame": sp, "code_frame": code,
            "position_range": "whole sensor (9.6 mm)" if whole else "sub-area (2.4 mm)", "spec": spec}


def alloc_name(x, y, a, b):
    return f"alloc_{x}{y}{a}{b}"


def build():
    R = []
    add = lambda *p, **k: R.append(arm(*p, **k))
    # ---- priority: on every file x scenario first
    add("implemented_now", "priority", today=True)
    add("implemented_now_sizeblind", "priority", today=True, pos_err="sizeblind")
    add("implemented_now_newtable", "priority", today=True, ang_err="linktable")
    add("network_float", "priority")
    add("alloc_4444", "priority", 4, 4, 4, 4)
    add("alloc_4444_wholesensor", "priority", 2, 2, 4, 4, whole=True)
    add("alloc_6604", "priority", 6, 6, 0, 4)
    add("bending_float", "priority", "F", "F", 0, "F")
    # ---- the other fixed point over the whole sensor
    add("alloc_6604_wholesensor", "fixed", 4, 4, 0, 4, whole=True)
    # ---- one output at a time, down from "network, float"
    for bits in range(2, 11):
        add(f"down_y_{bits}b", "scan_down", y=bits)
    for bits in range(2, 9):
        add(f"down_x_{bits}b", "scan_down", x=bits)
    for bits in (2, 3, 4, 5, 6, 7, 0):
        add(f"down_alpha_{bits}b" if bits else "down_alpha_notsent", "scan_down", a=bits)
    for bits in (2, 3, 4, 5, 6, 7, 0):
        add(f"down_beta_{bits}b" if bits else "down_beta_notsent", "scan_down", b=bits)
    # ---- one output at a time, up from the 16-bit cluster (the others at 4 bits)
    for bits in range(2, 11):
        if bits != 4:
            add(f"up_y_{bits}b", "scan_up", 4, bits, 4, 4)
    for bits in range(2, 9):
        if bits != 4:
            add(f"up_x_{bits}b", "scan_up", bits, 4, 4, 4)
    for bits in (2, 3, 5, 6, 7, 0):
        add(f"up_alpha_{bits}b" if bits else "up_alpha_notsent", "scan_up", 4, 4, bits, 4)
    for bits in (2, 3, 5, 6, 7, 0):
        add(f"up_beta_{bits}b" if bits else "up_beta_notsent", "scan_up", 4, 4, 4, bits)
    # ---- the bending-angle-only family, from 6/6/0/4
    for bits in (2, 3, 5, 6, 7):
        add(f"bending_beta_{bits}b", "bending", 6, 6, 0, bits)
    for bits in (4, 5, 7, 8):
        add(f"bending_xy_{bits}b", "bending", bits, bits, 0, 4)
    # ---- allocations with 16 bits in total
    allocs = [(5, 5, 3, 3), (6, 6, 2, 2), (5, 6, 2, 3), (4, 6, 3, 3), (4, 7, 2, 3), (5, 7, 2, 2), (6, 7, 0, 3),
              (7, 7, 0, 2), (8, 8, 0, 0)]
    for al in allocs:
        add(alloc_name(*al), "allocations", *al)
    # ---- controls
    for al in [(4, 4, 4, 4), (6, 6, 0, 4)] + [x for x in allocs if x[2] or x[3]]:
        add(alloc_name(*al) + "_globalrange", "controls", *al, range="global")
    for bits in (2, 3, 5, 6, 7):
        add(f"bending_beta_{bits}b_globalrange", "controls", 6, 6, 0, bits, range="global")
    for al in ((4, 4, 4, 4), (6, 6, 0, 4)):
        add(alloc_name(*al) + "_v3window", "controls", *al, window_add=False)
        add(alloc_name(*al) + "_flagunusable", "controls", *al, flag="unusable")
        add(alloc_name(*al) + "_allsent", "controls", *al, cap=False)
    add("network_float_allsent", "controls", cap=False)
    add("alloc_4444_wholesensor_allsent", "controls", 2, 2, 4, 4, whole=True, cap=False)
    add("bending_float_allsent", "controls", "F", "F", 0, "F", cap=False)
    for ax in ("x", "y"):
        for bits in (4, 6):
            add(f"noise_{ax}_{bits}b", "controls", **{ax: bits}, noise=True)
    add("positions_only_float", "controls", "F", "F", 0, 0)
    # the covariance forecast (PLAN 4a): the anchors with the position errors scaled to the network's. The scaled
    # table is the size-blind R (built at pT >= 5 GeV: the sensor alone), so the scale is the network's width over
    # today's sensor-only IL1 width: SmartPixels y 2.0 / 3.1, x 7.6 / 12.3 (code review MEDIUM 3); read on the forced refit
    add("forecast_network_float", "controls", pos_err_scale_x=7.6 / 12.3, pos_err_scale_y=2.0 / 3.1)
    add("forecast_bending_float", "controls", "F", "F", 0, "F", pos_err_scale_x=7.6 / 12.3, pos_err_scale_y=2.0 / 3.1)
    names = [r["name"] for r in R]
    assert len(set(names)) == len(names)
    return {"what": "arm registry of the SmartPixels output-bits refit study (PLAN.md section 5)",
            "not_in_this_scan": ["skip price re-derived at the 16-bit point (needs the cut fit; a separate run)",
                                 "the direct network-position point above 50 GeV (helix + drawn position residual)"],
            "group_order": list(GROUP_ORDER), "arms": R}


if __name__ == "__main__":
    doc = build()
    out = sys.argv[1] if len(sys.argv) > 1 else "arms.json"
    json.dump(doc, open(out, "w"), indent=1)
    from collections import Counter
    print(len(doc["arms"]), dict(Counter(r["group"] for r in doc["arms"])))
