/* MVA explorer: pure compute core (no DOM, no plotly).
 *
 * Two data families:
 *  - grid datasets (Panel 1, correctionlib): dense cubes, int16 log10-quantized
 *    (value = 10^(q/scale)) or raw float32; sliced/aggregated exactly like
 *    eval_spixel's builder_core.js (this generalizes it: arbitrary axis count,
 *    categorical fixed axes, optional numerator/denominator ratio).
 *  - table datasets (Panels 2+3): float32 row matrices; binned on the fly
 *    (mean/median score +- 16-84% band, working-point efficiency, per-bin
 *    one-vs-rest AUC with average-rank tie handling matching sklearn, score
 *    histograms), with range cuts on any column.
 *
 * Runs in browser and node/JXA (module.exports guard at the bottom).
 */

"use strict";

/* ---------------------------------------------------------------- shared */

function strides(shape) {
  const s = new Array(shape.length);
  let acc = 1;
  for (let i = shape.length - 1; i >= 0; i--) { s[i] = acc; acc *= shape[i]; }
  return s;
}

function binsInRange(centers, lo, hi) {
  const out = [];
  for (let i = 0; i < centers.length; i++)
    if (centers[i] >= lo && centers[i] <= hi) out.push(i);
  return out;
}

function quantile(sorted, q) {
  if (sorted.length === 0) return NaN;
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

function aggregate(values, agg) {
  if (values.length === 0) return { mid: NaN, lo: NaN, hi: NaN };
  const sorted = Float64Array.from(values).sort();
  let mid;
  if (agg === "mean") {
    let s = 0;
    for (const v of values) s += v;
    mid = s / values.length;
  } else {
    mid = quantile(sorted, 0.5);
  }
  return { mid, lo: quantile(sorted, 0.16), hi: quantile(sorted, 0.84) };
}

/* ------------------------------------------------------------------ grid */

/* Decoded value accessor for one grid block (+ optional denominator block of
 * identical layout).  quant: "log10_i16" (scale) | "f32". */
function makeValueFn(arr, quant, scale, base, denBase) {
  if (quant === "log10_i16") {
    if (denBase == null)
      return (off) => Math.pow(10, arr[base + off] / scale);
    return (off) => Math.pow(10, (arr[base + off] - arr[denBase + off]) / scale);
  }
  if (denBase == null) return (off) => arr[base + off];
  return (off) => arr[base + off] / arr[denBase + off];
}

/* Grid slice/aggregate.
 * spec: {
 *   valueFn: (flatGridOffset) -> value,
 *   shape:   grid shape (all grid axes, real AND categorical),
 *   centers: per-axis numeric bin centers (real axes) or null (categorical),
 *   fixedBins: {axisIdx: binIdx} — categorical axes pinned by dropdowns,
 *   xAxis:   axis index, yAxis: axis index or null (2D surface when set),
 *   cuts:    {axisIdx: [lo, hi]} on real axes,
 *   agg:     "median" | "mean",
 *   invert:  bool (flip num/den: value -> 1/value)
 * }
 * 1D -> {dim:1, x, mid[], lo[], hi[], n[]};  2D -> {dim:2, x, y, z[iy][ix]} */
function computeGridEntry(spec) {
  const shape = spec.shape;
  const st = strides(shape);
  const fixed = spec.fixedBins || {};
  const sign = spec.invert ? -1 : 1;
  const vf = spec.invert
    ? (off) => 1 / spec.valueFn(off)
    : spec.valueFn;

  let fixedOff = 0;
  for (const k in fixed) fixedOff += fixed[k] * st[k];

  const ix = spec.xAxis;
  let iy = spec.yAxis == null ? -1 : spec.yAxis;
  if (iy === ix) iy = -1;
  if (fixed[ix] !== undefined) throw new Error("xAxis is a fixed axis");

  const included = shape.map((len, i) => {
    if (fixed[i] !== undefined) return [fixed[i]];
    const cut = (spec.cuts && spec.cuts[i]) || [-Infinity, Infinity];
    if (!spec.centers[i]) return Array.from({ length: len }, (_, b) => b);
    return binsInRange(spec.centers[i], cut[0], cut[1]);
  });

  const restAxes = [];
  for (let i = 0; i < shape.length; i++)
    if (i !== ix && i !== iy && fixed[i] === undefined) restAxes.push(i);

  function gather(plotOff) {
    const vals = [];
    const lists = restAxes.map((r) => included[r]);
    if (lists.some((l) => l.length === 0)) return vals;
    const rst = restAxes.map((r) => st[r]);
    const idx = new Array(restAxes.length).fill(0);
    while (true) {
      let off = fixedOff + plotOff;
      for (let k = 0; k < restAxes.length; k++) off += lists[k][idx[k]] * rst[k];
      const v = vf(off);
      if (Number.isFinite(v)) vals.push(v); // 0/0 ratios in f32 blocks -> skip
      let k = restAxes.length - 1;
      while (k >= 0) {
        idx[k]++;
        if (idx[k] < lists[k].length) break;
        idx[k] = 0;
        k--;
      }
      if (k < 0) break;
    }
    return vals;
  }

  const xBins = included[ix];
  const x = xBins.map((b) => (spec.centers[ix] ? spec.centers[ix][b] : b));

  if (iy < 0) {
    const mid = [], lo = [], hi = [], n = [];
    for (const bx of xBins) {
      const vals = gather(bx * st[ix]);
      const r = aggregate(vals, spec.agg);
      mid.push(r.mid); lo.push(r.lo); hi.push(r.hi); n.push(vals.length);
    }
    return { dim: 1, x, mid, lo, hi, n };
  }

  const yBins = included[iy];
  const y = yBins.map((b) => (spec.centers[iy] ? spec.centers[iy][b] : b));
  const z = [];
  for (const by of yBins) {
    const row = [];
    for (const bx of xBins)
      row.push(aggregate(gather(bx * st[ix] + by * st[iy]), spec.agg).mid);
    z.push(row);
  }
  return { dim: 2, x, y, z };
}

/* ----------------------------------------------------------------- table */

/* table = { data: Float32Array (whole dataset file), ncol,
 *           colIdx: {name: columnIndex}, rowOffset, nRows }  (one row group) */
function makeTable(data, columns, rowOffset, nRows) {
  const colIdx = {};
  columns.forEach((c, i) => { colIdx[c] = i; });
  return { data, ncol: columns.length, colIdx, rowOffset, nRows };
}

function cell(table, row, col) {
  return table.data[(table.rowOffset + row) * table.ncol + col];
}

/* cuts: {colName: [lo, hi]} -> array of selected row indices (0..nRows-1). */
function selectRows(table, cuts) {
  const cutList = [];
  for (const name in cuts || {}) {
    const c = table.colIdx[name];
    if (c === undefined) throw new Error("unknown column " + name);
    cutList.push([c, cuts[name][0], cuts[name][1]]);
  }
  const rows = [];
  for (let r = 0; r < table.nRows; r++) {
    let ok = true;
    for (const [c, lo, hi] of cutList) {
      const v = cell(table, r, c);
      if (!(v >= lo && v <= hi)) { ok = false; break; }
    }
    if (ok) rows.push(r);
  }
  return rows;
}

function columnValues(table, name, rows) {
  const c = table.colIdx[name];
  if (c === undefined) throw new Error("unknown column " + name);
  const out = new Float64Array(rows.length);
  for (let i = 0; i < rows.length; i++) out[i] = cell(table, rows[i], c);
  return out;
}

/* {lo, hi, n, log} -> n+1 edges (log-spaced when log && lo>0). */
function makeEdges(binSpec) {
  const { lo, hi, n } = binSpec;
  const edges = new Array(n + 1);
  if (binSpec.log && lo > 0) {
    const llo = Math.log10(lo), lhi = Math.log10(hi);
    for (let i = 0; i <= n; i++)
      edges[i] = Math.pow(10, llo + ((lhi - llo) * i) / n);
  } else {
    for (let i = 0; i <= n; i++) edges[i] = lo + ((hi - lo) * i) / n;
  }
  return edges;
}

function edgeCenters(edges) {
  const c = new Array(edges.length - 1);
  for (let i = 0; i < c.length; i++) c[i] = 0.5 * (edges[i] + edges[i + 1]);
  return c;
}

/* bin index per value; -1 outside [edges[0], edges[n]] (last edge inclusive). */
function binAssign(x, edges) {
  const n = edges.length - 1;
  const out = new Int32Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const v = x[i];
    if (!(v >= edges[0] && v <= edges[n])) { out[i] = -1; continue; }
    let lo = 0, hi = n;                 // binary search: edges[lo] <= v < edges[lo+1]
    while (lo < hi - 1) {
      const mid = (lo + hi) >> 1;
      if (v >= edges[mid]) lo = mid; else hi = mid;
    }
    out[i] = v === edges[n] ? n - 1 : lo;
  }
  return out;
}

/* y-mode 1: mean/median score +- 16-84% band vs x. */
function binnedScoreStats(x, score, edges, agg) {
  const bins = binAssign(x, edges);
  const nb = edges.length - 1;
  const buckets = Array.from({ length: nb }, () => []);
  for (let i = 0; i < x.length; i++)
    if (bins[i] >= 0) buckets[bins[i]].push(score[i]);
  const centers = edgeCenters(edges);
  const mid = [], lo = [], hi = [], n = [];
  for (let b = 0; b < nb; b++) {
    const r = aggregate(buckets[b], agg);
    mid.push(r.mid); lo.push(r.lo); hi.push(r.hi); n.push(buckets[b].length);
  }
  return { centers, mid, lo, hi, n };
}

/* y-mode 2: working-point efficiency vs x.
 * pos: 0/1 per row.  eff = P(score > cut | pos), mistag = P(score > cut | !pos). */
function binnedEfficiency(x, score, pos, cut, edges) {
  const bins = binAssign(x, edges);
  const nb = edges.length - 1;
  const nPos = new Array(nb).fill(0), nNeg = new Array(nb).fill(0);
  const pPos = new Array(nb).fill(0), pNeg = new Array(nb).fill(0);
  for (let i = 0; i < x.length; i++) {
    const b = bins[i];
    if (b < 0) continue;
    const passed = score[i] > cut;
    if (pos[i]) { nPos[b]++; if (passed) pPos[b]++; }
    else { nNeg[b]++; if (passed) pNeg[b]++; }
  }
  const eff = [], mistag = [];
  for (let b = 0; b < nb; b++) {
    eff.push(nPos[b] > 0 ? pPos[b] / nPos[b] : NaN);
    mistag.push(nNeg[b] > 0 ? pNeg[b] / nNeg[b] : NaN);
  }
  return { centers: edgeCenters(edges), eff, mistag, nPos, nNeg };
}

/* One-vs-rest AUC via Mann-Whitney with average ranks on ties — matches
 * sklearn.roc_auc_score bit-for-bit on finite inputs.  NaN when one class is
 * empty. */
function aucRanked(scores, pos) {
  const n = scores.length;
  const order = Array.from({ length: n }, (_, i) => i)
    .sort((a, b) => scores[a] - scores[b]);
  const ranks = new Float64Array(n);
  let i = 0;
  while (i < n) {
    let j = i;
    while (j + 1 < n && scores[order[j + 1]] === scores[order[i]]) j++;
    const avg = (i + j) / 2 + 1; // 1-based average rank of the tie block
    for (let k = i; k <= j; k++) ranks[order[k]] = avg;
    i = j + 1;
  }
  let nPos = 0, sumPos = 0;
  for (let k = 0; k < n; k++)
    if (pos[k]) { nPos++; sumPos += ranks[k]; }
  const nNeg = n - nPos;
  if (nPos === 0 || nNeg === 0) return NaN;
  return (sumPos - (nPos * (nPos + 1)) / 2) / (nPos * nNeg);
}

/* y-mode 3: per-bin one-vs-rest AUC vs x (with per-bin class counts). */
function binnedAUC(x, score, pos, edges) {
  const bins = binAssign(x, edges);
  const nb = edges.length - 1;
  const bScores = Array.from({ length: nb }, () => []);
  const bPos = Array.from({ length: nb }, () => []);
  for (let i = 0; i < x.length; i++) {
    const b = bins[i];
    if (b < 0) continue;
    bScores[b].push(score[i]);
    bPos[b].push(pos[i]);
  }
  const auc = [], nPos = [], nNeg = [];
  for (let b = 0; b < nb; b++) {
    let np = 0;
    for (const p of bPos[b]) if (p) np++;
    nPos.push(np); nNeg.push(bPos[b].length - np);
    auc.push(aucRanked(bScores[b], bPos[b]));
  }
  return { centers: edgeCenters(edges), auc, nPos, nNeg };
}

/* y-mode 4: score histogram (density = counts / (total * width) when
 * normalize, else raw counts). */
function histogram(values, edges, normalize) {
  const bins = binAssign(values, edges);
  const nb = edges.length - 1;
  const counts = new Array(nb).fill(0);
  let total = 0;
  for (let i = 0; i < values.length; i++)
    if (bins[i] >= 0) { counts[bins[i]]++; total++; }
  const y = [];
  for (let b = 0; b < nb; b++) {
    if (normalize && total > 0) y.push(counts[b] / (total * (edges[b + 1] - edges[b])));
    else y.push(counts[b]);
  }
  return { centers: edgeCenters(edges), y, counts, total };
}

/* Seed-band combiner: curves = [{mid: []}, ...] aligned on the same x grid
 * -> {mid: per-bin mean, lo: min, hi: max, n: contributing curves per bin}. */
function combineCurves(curves, key) {
  key = key || "mid";
  if (curves.length === 0) return { mid: [], lo: [], hi: [], n: [] };
  const nb = curves[0][key].length;
  const mid = [], lo = [], hi = [], n = [];
  for (let b = 0; b < nb; b++) {
    const vals = [];
    for (const c of curves) {
      const v = c[key][b];
      if (v === v) vals.push(v); // skip NaN
    }
    if (vals.length === 0) { mid.push(NaN); lo.push(NaN); hi.push(NaN); n.push(0); continue; }
    let s = 0, mn = Infinity, mx = -Infinity;
    for (const v of vals) { s += v; if (v < mn) mn = v; if (v > mx) mx = v; }
    mid.push(s / vals.length); lo.push(mn); hi.push(mx); n.push(vals.length);
  }
  return { mid, lo, hi, n };
}

/* one-vs-rest positive mask from a label column. */
function positiveMask(labels, positiveValue) {
  const out = new Uint8Array(labels.length);
  for (let i = 0; i < labels.length; i++)
    out[i] = labels[i] === positiveValue ? 1 : 0;
  return out;
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    strides, binsInRange, quantile, aggregate,
    makeValueFn, computeGridEntry,
    makeTable, selectRows, columnValues, makeEdges, edgeCenters, binAssign,
    binnedScoreStats, binnedEfficiency, aucRanked, binnedAUC, histogram,
    combineCurves, positiveMask,
  };
}
