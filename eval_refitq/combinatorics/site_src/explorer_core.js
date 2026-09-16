// Pure compute for the seed-efficiency explorer. No DOM, no Plotly: this file
// runs under node so the binning and duplicate-removal logic can be tested
// without a browser, following the convention in the mva_explorer.
//
// EVERYTHING IS A MASK AND A BIN over two flat typed-array tables. Nothing is
// pre-aggregated per seed, because efficiency for a seed MENU is a set union --
// a TrackingParticle found by three enabled seeds counts once -- and summing
// pre-binned per-seed histograms would double-count it.
'use strict';

// ---- seed selection -------------------------------------------------------
// The bitmap is uint32 word pairs, not uint64. JavaScript has no fast 64-bit
// integer type; BigUint64Array forces BigInt arithmetic, which allocates per
// operation. This mask is ANDed against every TP on every control change.
function seedMaskWords(indices, nWords) {
  const w = new Uint32Array(nWords);
  for (const i of indices) w[i >> 5] |= (1 << (i & 31)) >>> 0;
  return w;
}

function anyBitSet(found, row, nWords, mask) {
  const o = row * nWords;
  for (let k = 0; k < nWords; k++) if ((found[o + k] & mask[k]) !== 0) return true;
  return false;
}

// ---- TP-side cuts ---------------------------------------------------------
// cuts: {col: [min, max]} over TP columns, applied before any binning.
function tpSelect(tp, cols, nRows, cuts) {
  const keep = new Uint8Array(nRows).fill(1);
  for (const name in cuts) {
    const j = cols.indexOf(name);
    if (j < 0) continue;
    const [lo, hi] = cuts[name];
    const nc = cols.length;
    for (let i = 0; i < nRows; i++) {
      if (!keep[i]) continue;
      const v = tp[i * nc + j];
      if (!(v >= lo && v <= hi)) keep[i] = 0;
    }
  }
  return keep;
}

function binIndex(v, edges) {
  if (!(v >= edges[0]) || v > edges[edges.length - 1]) return -1;
  let lo = 0, hi = edges.length - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (v < edges[m]) hi = m; else lo = m; }
  return lo;
}

// ---- efficiency: distinct TPs reached, over a fixed denominator -----------
function binnedEfficiency(tp, cols, nRows, keep, found, nWords, mask,
                          xName, yName, xEdges, yEdges) {
  const nc = cols.length, jx = cols.indexOf(xName), jy = cols.indexOf(yName);
  const nx = xEdges.length - 1, ny = yEdges.length - 1;
  const num = new Float64Array(nx * ny), den = new Float64Array(nx * ny);
  for (let i = 0; i < nRows; i++) {
    if (!keep[i]) continue;
    const bx = binIndex(tp[i * nc + jx], xEdges); if (bx < 0) continue;
    const by = binIndex(tp[i * nc + jy], yEdges); if (by < 0) continue;
    const b = by * nx + bx;
    den[b] += 1;
    if (anyBitSet(found, i, nWords, mask)) num[b] += 1;
  }
  return { num, den, nx, ny };
}

// ---- per-menu duplicate removal over the track table ----------------------
// The table is sorted by tp_key with fakes in a contiguous tp_key < 0 block, so
// this is one linear scan rather than a hash over millions of rows.
//
// LIMITATION, and it is not hideable: this dedupes REAL tracks only, by keeping
// the best rank_score per TrackingParticle among the enabled seeds. Fake tracks
// have no TP to group by, and the real criterion (>= 3 shared hits) needs hit
// lists that are far too large to ship. So fake counts here are PRE-DR and the
// fake rate a selection shows is an OVERESTIMATE relative to a full DR run.
function drSurvivors(trk, cols, nRows, enabled) {
  const nc = cols.length;
  const jk = cols.indexOf('tp_key'), js = cols.indexOf('seed_idx'),
        jr = cols.indexOf('rank_score');
  const out = [];
  let i = 0;
  // fakes first: every enabled-seed fake row survives, see limitation above
  for (; i < nRows && trk[i * nc + jk] < 0; i++) {
    if (enabled[trk[i * nc + js] | 0]) out.push(i);
  }
  while (i < nRows) {
    const key = trk[i * nc + jk];
    let best = -1, bestR = -Infinity, j = i;
    for (; j < nRows && trk[j * nc + jk] === key; j++) {
      if (!enabled[trk[j * nc + js] | 0]) continue;
      const r = trk[j * nc + jr];
      if (r > bestR) { bestR = r; best = j; }
    }
    if (best >= 0) out.push(best);
    i = j;
  }
  return out;
}

// ---- robust spread, for resolution maps ----------------------------------
function robustSigma(vals) {
  if (vals.length < 30) return NaN;
  const a = Float64Array.from(vals).sort();
  const q = (p) => {
    const x = p * (a.length - 1), i = Math.floor(x);
    return a[i] + (a[Math.min(i + 1, a.length - 1)] - a[i]) * (x - i);
  };
  return 0.5 * (q(0.84135) - q(0.15865));
}

function binnedStat(trk, cols, rows, xName, yName, xEdges, yEdges, valueName,
                    stat) {
  const nc = cols.length, jx = cols.indexOf(xName), jy = cols.indexOf(yName),
        jv = valueName ? cols.indexOf(valueName) : -1;
  const nx = xEdges.length - 1, ny = yEdges.length - 1;
  const buckets = new Array(nx * ny);
  const cnt = new Float64Array(nx * ny);
  for (const i of rows) {
    const bx = binIndex(trk[i * nc + jx], xEdges); if (bx < 0) continue;
    const by = binIndex(trk[i * nc + jy], yEdges); if (by < 0) continue;
    const b = by * nx + bx;
    cnt[b] += 1;
    if (jv < 0) continue;
    const v = trk[i * nc + jv];
    if (!Number.isFinite(v)) continue;
    (buckets[b] || (buckets[b] = [])).push(v);
  }
  const val = new Float64Array(nx * ny).fill(NaN);
  for (let b = 0; b < nx * ny; b++) {
    const s = buckets[b];
    if (!s) continue;
    val[b] = stat === 'sigma' ? robustSigma(s)
           : stat === 'mean' ? s.reduce((p, c) => p + c, 0) / s.length
           : s.length;
  }
  return { val, cnt, nx, ny };
}

// fake rate: fraction of surviving tracks with no TrackingParticle, per bin
function binnedFakeRate(trk, cols, rows, xName, yName, xEdges, yEdges, nEvents) {
  const nc = cols.length, jk = cols.indexOf('tp_key');
  const nx = xEdges.length - 1, ny = yEdges.length - 1;
  const fake = new Float64Array(nx * ny), tot = new Float64Array(nx * ny);
  const jx = cols.indexOf(xName), jy = cols.indexOf(yName);
  for (const i of rows) {
    const bx = binIndex(trk[i * nc + jx], xEdges); if (bx < 0) continue;
    const by = binIndex(trk[i * nc + jy], yEdges); if (by < 0) continue;
    const b = by * nx + bx;
    tot[b] += 1;
    if (trk[i * nc + jk] < 0) fake[b] += 1;
  }
  const frac = new Float64Array(nx * ny).fill(NaN);
  const perEv = new Float64Array(nx * ny);
  for (let b = 0; b < nx * ny; b++) {
    if (tot[b] > 0) frac[b] = fake[b] / tot[b];
    perEv[b] = fake[b] / nEvents;
  }
  return { frac, perEv, tot, fake, nx, ny };
}

function linEdges(lo, hi, n) {
  const e = new Float64Array(n + 1);
  for (let i = 0; i <= n; i++) e[i] = lo + (hi - lo) * i / n;
  return e;
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { seedMaskWords, anyBitSet, tpSelect, binIndex,
                     binnedEfficiency, drSurvivors, robustSigma, binnedStat,
                     binnedFakeRate, linEdges };
}
