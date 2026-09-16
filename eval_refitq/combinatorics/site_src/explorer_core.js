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

// ---- per-menu duplicate removal ------------------------------------------
// KEYED ON SHARED HITS, NEVER ON TRUTH. An earlier version grouped tracks by
// tp_key and kept the best rank_score per TrackingParticle. That is an oracle no
// L1 system has, and it distorted in BOTH directions at once: real tracks
// deduplicated better than any implementable DR could manage, fakes not
// deduplicated at all because they have no TP to group by. A fake rate read off
// such a page would mislead about the thing the page most exists to show. It is
// deleted rather than left behind a flag.
//
// What runs instead is the real criterion -- two tracks conflict when they share
// identical hits on >= 3 layers -- over a conflict graph precomputed at export.
// The greedy itself is unchanged from the emulator's: take tracks in descending
// rank_score, accept one if no already-accepted track conflicts with it, and
// kill its neighbours.
//
// `order` is the global descending-rank_score ordering. It is MENU-INDEPENDENT,
// because rank_score is a per-track quantity, so it is computed once at export
// and shipped. Only which tracks are enabled changes per menu.
//
// A DISABLED TRACK MUST NOT KILL ITS NEIGHBOURS -- hence the enabled test before
// the accept, not after. Getting that backwards would let a seed the user has
// switched off go on suppressing tracks.
function drSurvivorsGraph(order, seedOf, indptr, nbr, enabled) {
  const n = order.length;
  const dead = new Uint8Array(n);
  const out = [];
  for (let oi = 0; oi < n; oi++) {
    const t = order[oi];
    if (!enabled[seedOf[t]]) continue;
    if (dead[t]) continue;
    out.push(t);
    for (let k = indptr[t]; k < indptr[t + 1]; k++) dead[nbr[k]] = 1;
  }
  return out;
}

// No conflict graph available (or DR switched off): every enabled track.
function allEnabled(seedOf, nRows, enabled) {
  const out = [];
  for (let i = 0; i < nRows; i++) if (enabled[seedOf[i]]) out.push(i);
  return out;
}

// ---- chunked jobs, so the page can repaint -------------------------------
// A synchronous loop over 12.6M rows blocks the main thread for seconds and the
// browser cannot paint a spinner, let alone a progress bar, while it runs. These
// return a job whose step(budget) does a slice and returns how far it has got,
// so the caller can yield to the event loop between slices. Splitting the work
// costs a few percent; showing nothing for ten seconds costs the user's trust
// that the page is alive.
function drJob(order, seedOf, indptr, nbr, enabled) {
  const n = order.length;
  const dead = new Uint8Array(n);
  const out = [];
  let i = 0;
  return {
    total: n,
    result: out,
    get done() { return i >= n; },
    get at() { return i; },
    step(budget) {
      const end = Math.min(i + budget, n);
      for (; i < end; i++) {
        const t = order[i];
        if (!enabled[seedOf[t]]) continue;
        if (dead[t]) continue;
        out.push(t);
        for (let k = indptr[t]; k < indptr[t + 1]; k++) dead[nbr[k]] = 1;
      }
      return i;
    }
  };
}

// every enabled track, no DR; chunked for the same reason
function allEnabledJob(seedOf, nRows, enabled) {
  const out = [];
  let i = 0;
  return {
    total: nRows,
    result: out,
    get done() { return i >= nRows; },
    get at() { return i; },
    step(budget) {
      const end = Math.min(i + budget, nRows);
      for (; i < end; i++) if (enabled[seedOf[i]]) out.push(i);
      return i;
    }
  };
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

// ---- explicit efficiency accounting ---------------------------------------
// DENOMINATORS ARE THE WHOLE GAME and the page must never leave one implicit.
// Three quantities got confused in analysis and would confuse a reader faster:
//   candidates kept   ~18%  -- a fraction of TRACKS, not an efficiency at all
//   TPs with >=1 cand ~11%  -- 871/ev, dominated by soft particles that donate
//                              one cluster to a fake; not a findable set
//   FINDABLE TPs      ~92%  -- pT >= threshold and >= 4 layers, 118/ev
// There are ~13.7 candidates per TP, so discarding 82% of candidates costs 8%
// of particles. Reporting the first number as "efficiency" understates the
// truth fivefold.
function effByBand(tp, cols, nRows, keep, found, nWords, mask, bands, minLayers) {
  const nc = cols.length, jp = cols.indexOf('pt'), jl = cols.indexOf('n_layers');
  const den = new Float64Array(bands.length), num = new Float64Array(bands.length);
  for (let i = 0; i < nRows; i++) {
    if (!keep[i]) continue;
    if (tp[i * nc + jl] < minLayers) continue;
    const pt = tp[i * nc + jp];
    for (let b = 0; b < bands.length; b++) {
      if (pt >= bands[b][0] && pt < bands[b][1]) {
        den[b] += 1;
        if (anyBitSet(found, i, nWords, mask)) num[b] += 1;
        break;
      }
    }
  }
  return { num, den };
}

function linEdges(lo, hi, n) {
  const e = new Float64Array(n + 1);
  for (let i = 0; i <= n; i++) e[i] = lo + (hi - lo) * i / n;
  return e;
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { seedMaskWords, anyBitSet, tpSelect, binIndex,
                     binnedEfficiency, drSurvivorsGraph, allEnabled,
                     drJob, allEnabledJob,
                     robustSigma, binnedStat, binnedFakeRate, effByBand,
                     linEdges };
}
