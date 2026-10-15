"""C1b: the material along a disc-going track's path (c1b/PLAN.md rev 4 §1 item 4).

For a disc-going track every plane is ordered by its path position along the helix and charged only when the
helix crosses it within its own extent (no margin), with its own path factor:
  - barrel modules IL1-IL4: cylinders at the layer radius, |z| within the layer's module extent (D121 geometry);
  - passive cylinders (refit_calibration v3's planes): their |z| extent, from the tkLayout sim XML in this order:
    the pixel XML (IT711_2023_05), the OT XML (TiltedTracker404), else the flagged fallback |z| <= 20.07 cm;
  - TFPX disks 1-6: planes at the disc's mean |z|, within its radial module coverage (D121 geometry), x/X0 from
    trackerRecoMaterial.xml (IT711_2023_05), times the disc material scale.
The kink is v3's (refit_calibration_fit.kink_cov / ORP.add_material) at the crossing's transverse radius; Highland
with X = x/X0 x the path factor: sqrt(1 + tanL^2) for a cylinder, sqrt(1 + tanL^2) / |tanL| for a disc. The
half-open convention is v3's: between the previous visited surface (its material included) and the current one
(excluded).
"""
from __future__ import annotations

import json
import re
import time

import numpy as np

from arms import IL1_MODULE
from common import OP, ORP

MATERIAL_DIR = "/Volumes/WDMac/smartpixels_scratch/wpc/c1b/material"
PIXEL_XML = f"{MATERIAL_DIR}/pixel_IT711_2023_05.xml"
OT_XML = f"{MATERIAL_DIR}/tracker_TiltedTracker404.xml"
RECO_XML = f"{MATERIAL_DIR}/trackerRecoMaterial_IT711_2023_05.xml"
TABLE = f"{MATERIAL_DIR}/disc_material_v1.json"
DISKS = (11, 12, 13, 14, 15, 16)                  # TFPX disks 1-6 (7-8 and TEPX: outside |eta| < 2.4)
FALLBACK_HALF_Z = 20.07                             # cm, the barrel module extent, flagged
R_MATCH_MM, MIN_HALF_DZ_MM, MAX_WALL_MM = 1.0, 10.0, 10.0
_TUBS = re.compile(r'<Tubs name="([a-zA-Z]+)R(\d+)Z(\d+)" rMin="([\d.]+)\*mm" rMax="([\d.]+)\*mm" dz="([\d.]+)\*mm"')


def _cylinders(path):
    """tkLayout support / service cylinders of an XML: (rMin, rMax, zlo, zhi) in mm, the z centre from the name."""
    s = open(path).read()
    out = []
    for m in _TUBS.finditer(s):
        zc, rmin, rmax, dz = int(m.group(3)), float(m.group(4)), float(m.group(5)), float(m.group(6))
        if rmax - rmin <= MAX_WALL_MM and dz >= MIN_HALF_DZ_MM:
            out.append((rmin, rmax, zc - dz, zc + dz))
    return out


def _z_extent(r_cm, sources):
    """The |z| extent (cm) of a passive cylinder at r_cm: the union of the matching tkLayout segments, contiguous
    from z = 0 within 2 mm, from the first source that has any; else the fallback."""
    for name, cyl in sources:
        seg = sorted((lo, hi) for rmin, rmax, lo, hi in cyl if rmin - R_MATCH_MM <= 10 * r_cm <= rmax + R_MATCH_MM)
        if not seg:
            continue
        reach = 0.0
        for lo, hi in seg:
            if lo <= reach + 2.0:
                reach = max(reach, hi)
        return reach / 10.0, name
    return FALLBACK_HALF_Z, "fallback |z| <= 20.07 cm (flagged)"


def _disc_rl():
    s = open(RECO_XML).read()
    m = re.search(r'PixelForwardDisk1".*?TrackerRadLength" value="([\d.eE+-]+)"', s, re.S)
    return float(m.group(1))


def build_table(cal, geometry_path, out=TABLE):
    """The plane table for disc-going tracks, with the source of every extent (the provenance)."""
    geo = OP.load_geometry(geometry_path, surfaces=(1, 2, 3, 4) + DISKS)
    sources = [("tkLayout pixel XML (IT711_2023_05)", _cylinders(PIXEL_XML)),
               ("tkLayout OT XML (TiltedTracker404)", _cylinders(OT_XML))]
    planes = [{"kind": "cyl", "r": geo["R"][1], "half_z": geo["halfZ"][1], "xx0": IL1_MODULE[1], "what": IL1_MODULE[2],
               "extent_source": "D121 geometry (module outlines)"}]     # v3 charges IL1 only at the POCA
    for r_spec, xx0, what in cal["planes"]:
        if isinstance(r_spec, str):
            L = int(r_spec[2:])
            planes.append({"kind": "cyl", "r": geo["R"][L], "half_z": geo["halfZ"][L], "xx0": xx0, "what": what,
                           "extent_source": "D121 geometry (module outlines)"})
        else:
            hz, src = _z_extent(float(r_spec), sources)
            planes.append({"kind": "cyl", "r": float(r_spec), "half_z": hz, "xx0": xx0, "what": what, "extent_source": src})
    rl = _disc_rl()
    for d in DISKS:
        planes.append({"kind": "disc", "z": geo["Z"][d], "r_ext": list(geo["rext"][d]), "xx0": rl, "disk": d - 10,
                       "what": f"TFPX disk {d - 10} module", "extent_source": "D121 geometry (module outlines)",
                       "xx0_source": "trackerRecoMaterial.xml IT711_2023_05 (tkLayout RL)"})
    doc = {"schema": "wpc-c1b-material/1", "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "planes": planes, "material_scale_cyl": cal["material_scale"],
           "provenance": {"calibration_planes": "refit_calibration v3", "pixel_xml": PIXEL_XML, "ot_xml": OT_XML,
                          "reco_xml": RECO_XML, "geometry": geometry_path}}
    with open(out, "w") as fh:
        json.dump(doc, fh, indent=1)
    return doc


def crossings(a, planes, cyl_extent=True):
    """(s, r, on) per (track, plane): the transverse arc length to each plane, the crossing radius and whether
    the crossing lies within the plane's extent (no margin; cyl_extent=False: any |z| for the cylinders)."""
    n, k = len(a), len(planes)
    s = np.full((n, k), np.nan)
    r = np.full((n, k), np.nan)
    on = np.zeros((n, k), bool)
    for j, pl in enumerate(planes):
        if pl["kind"] == "cyl":
            sj = OP.cylinder_s(a, pl["r"])
            pos, _ = OP.helix_at(a, np.nan_to_num(sj))
            ok = np.isfinite(sj) & ((np.abs(pos[:, 2]) <= pl["half_z"]) | (not cyl_extent))
            rj = np.full(n, pl["r"])
        else:
            sj = OP.disc_s(a, pl["z"])
            pos, _ = OP.helix_at(a, np.nan_to_num(sj))
            rj = np.hypot(pos[:, 0], pos[:, 1])
            ok = np.isfinite(sj) & (rj >= pl["r_ext"][0]) & (rj <= pl["r_ext"][1])
        s[:, j], r[:, j], on[:, j] = sj, rj, ok
    return s, r, on


def charge(a, C, t, s_cur, s_prev, planes, disc_scale, cyl_scale, kpt, disc_factor="disc", cyl_extent=True):
    """Add the kinks of every plane crossed with s_cur < s <= s_prev (+1e-6) to C[t], each at its crossing radius
    with its path factor (the half-open convention). s_prev = +inf: everything outside the first visit.
    disc_factor "barrel": the C1b known-bad with the cylinder's path factor on the disc planes; cyl_extent=False:
    the known-bad charging the cylinders by radius only (whatever their |z| extent)."""
    if not len(t):
        return
    s, r, on = crossings(a[t], planes, cyl_extent)
    tanl = a[t, 2]
    pt = kpt / np.maximum(np.abs(a[t, 0]), 1e-12)
    p = pt * np.sqrt(1.0 + tanl ** 2)
    sec2 = 1.0 + tanl ** 2
    sec = np.sqrt(sec2)
    for j, pl in enumerate(planes):
        m = on[:, j] & (s[:, j] > s_cur) & (s[:, j] <= s_prev + 1e-6)
        if not m.any():
            continue
        if pl["kind"] == "cyl":
            xx0 = cyl_scale * pl["xx0"]
            path = np.ones(len(t))                       # ORP._highland_theta2 adds sqrt(1 + tanL^2)
        else:
            xx0 = disc_scale * pl["xx0"]
            path = 1.0 / np.maximum(np.abs(tanl), 1e-6) if disc_factor == "disc" else np.ones(len(t))
        th2 = np.where(m, ORP._highland_theta2(xx0 * path, p, tanl), 0.0)
        rs = r[:, j]
        jT = np.zeros((len(t), 5)); jT[:, 1] = sec; jT[:, 4] = +rs * sec
        jL = np.zeros((len(t), 5)); jL[:, 2] = sec2; jL[:, 3] = -rs * sec2
        C[t] += th2[:, None, None] * (jT[:, :, None] * jT[:, None, :] + jL[:, :, None] * jL[:, None, :])
