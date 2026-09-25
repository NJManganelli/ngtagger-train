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

// A MISSING COLUMN MUST THROW, never return -1. trk[i*nc + (-1)] reads the last
// column of the PREVIOUS row -- valid memory, plausible numbers, silently wrong.
// That is exactly how the track plots came to bin d_z0 of track i-1 as though it
// were pT: the axis menu offered TP-table names and the track table calls the
// same quantities inv_pt, phi0 and nhit.
function need(cols, name, where) {
  const j = cols.indexOf(name);
  if (j < 0) throw new Error(`${where}: no column "${name}" in [${cols.join(',')}]`);
  return j;
}

// An axis may name a track column or a SCORE, which lives in its own uint8
// table. Resolving that here keeps every binning function ignorant of where a
// quantity is stored, and still THROWS on an unknown name rather than reading
// the previous row's last column.
// Quantities the export does NOT store because they are exact functions of
// columns it does. pT and eta residuals belong here: the fit works in 1/pT and
// cot(theta) because those are what a helix is linear in, but neither is
// readable, so the page derives the physical ones instead of shipping 100 MB
// of duplicate columns.
//   d_pt_rel  (pT_fit - pT_true) / pT_true, in %.  1/pT_fit = 1/pT_true + d_kappa
//   d_eta     eta_fit - eta_true.  eta = asinh(cot), so d_eta = d_cot/sqrt(1+cot^2)
const DERIVED = {
  // exact, not an approximation: with kappa = 1/pT and d_kappa = kappa_fit -
  // kappa_true, (pT_fit - pT_true)/pT_true = pT_fit*kappa_true - 1 = -pT_fit*d_kappa
  d_pt_rel: { needs: ['pt', 'd_kappa'], unit: '%',
              f: (pt, dk) => -100 * pt * dk },
  d_eta: { needs: ['d_cot', 'eta'], unit: '',
           f: (dc, eta) => dc / Math.sqrt(1 + Math.sinh(eta) ** 2) },
  // THE CLUSTER'S OWN ANGLE, not its residual. angle_rows stores
  //   res_a = (-c*kappa_cluster) - pred_a ,  res_b = cot_cluster - pred_b
  // so the measurement itself is res + pred. Worth having separately from the
  // residual: the residual says how well the cluster agrees with the track,
  // while this says what the sensor actually reported, and the two answer
  // different questions about a layer.
  meas_a: { needs: ['res_a', 'pred_a'], unit: '', f: (r, p) => r + p },
  meas_b: { needs: ['res_b', 'pred_b'], unit: '', f: (r, p) => r + p },
  // THE ANGULAR FORM OF THE ALPHA RESIDUAL, in radians.
  //
  // res_a is NOT an angle. The KF's r-phi state is phi(r) = x1 + x0*r + x4/r
  // with x0 = -c*kappa, so res_a is the residual of dphi/dr and carries units
  // of rad/cm. Multiplying by the radius recovers the angle, and that is the
  // layer-independent quantity: MEASURED on correctly assigned clusters of
  // clean tracks, sigma(res_a) falls 0.00714 -> 0.00151 from IL1 to IL4, a
  // factor 4.7 that is ENTIRELY the 1/r, while sigma(res_a * r) is flat at
  // 20.4 / 20.7 / 21.2 / 21.8 mrad. So the per-layer spread of res_a is
  // geometry, not sensor performance, and only this form should be compared
  // between layers.
  //
  // r comes from the layer's nominal radius because the angle table stores no
  // radius; the +-5 mm ladder stagger is a ~0.2% effect here. Storing r per
  // cluster in ANGLE_COLS would remove the approximation.
  ang_a_rad: { needs: ['res_a', 'layer'], unit: 'rad',
               f: (ra, L) => ra * (IT_R_NOMINAL[L | 0] || NaN) },
};
const IT_R_NOMINAL = { 1: 2.85, 2: 5.95, 3: 10.33, 4: 14.50 };
function axisGetter(trk, cols, SC, name, where) {
  const j = cols.indexOf(name);
  if (j >= 0) {
    const nc = cols.length;
    return (i) => trk[i * nc + j];
  }
  const k = scoreCol(SC, name);
  if (k >= 0) {
    const nc = SC.cols.length, sc = SC.scale;
    return (i) => SC.data[i * nc + k] / sc;
  }
  const d = DERIVED[name];
  if (d) {
    const nc = cols.length;
    const js = d.needs.map(c => need(cols, c, `${where} (for ${name})`));
    if (js.length === 2) {
      const [a, b] = js;
      return (i) => d.f(trk[i * nc + a], trk[i * nc + b]);
    }
  }
  throw new Error(`${where}: no column, score or derived quantity "${name}"`);
}

function binIndex(v, edges) {
  if (!(v >= edges[0]) || v > edges[edges.length - 1]) return -1;
  let lo = 0, hi = edges.length - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (v < edges[m]) hi = m; else lo = m; }
  return lo;
}

// ---- efficiency: distinct TPs reached, over a fixed denominator -----------
// TWO DEFINITIONS OF "FOUND", AND THE PAGE HOSTS BOTH.
//
//   seed-truthful   the seed's own clusters all belong to the TP. Judges a
//                   SEED: could this pair of layers start a search that reaches
//                   this particle. Fixed at census time, so quality cuts in the
//                   browser do not move it.
//   delivered       some surviving track is majority-owned by the TP. Judges the
//                   SYSTEM: does a usable track come out. Computed here from the
//                   selected track set, so it responds to the seed selection,
//                   duplicate removal, the cuts and the score thresholds.
//
// They disagree because the search adds hits after seeding and can add the
// wrong ones: 45.8% of accepted tracks are majority-owned by a TP other than
// their seed's. Seed-truthful is the more generous of the two.
function bitmapMask(found, nRows, nWords, mask) {
  const m = new Uint8Array(nRows);
  for (let i = 0; i < nRows; i++) m[i] = anyBitSet(found, i, nWords, mask) ? 1 : 0;
  return m;
}

// tp_row is an INDEX into the exported TP table, not a key: (event << 20) |
// tpIdx does not survive float32 (spacing is 128 at event 999), so equality
// matching on tp_key is silently wrong and must never be used.
const OWN_DEFS = {
  majority: 'more than half the hits, and at least 3',
  min3: 'at least 3 hits',
  clean: 'every hit on the track',
};
function deliveredMask(trk, cols, rows, nRows, def) {
  const nc = cols.length;
  const jr = need(cols, 'tp_row', 'deliveredMask');
  const jo = need(cols, 'n_own', 'deliveredMask');
  const jh = need(cols, 'nhit', 'deliveredMask');
  const jw = need(cols, 'n_wrong', 'deliveredMask');
  const m = new Uint8Array(nRows);
  for (const i of rows) {
    const b = i * nc, row = trk[b + jr];
    if (!(row >= 0)) continue;
    const own = trk[b + jo], nh = trk[b + jh];
    const ok = def === 'clean' ? trk[b + jw] === 0
             : def === 'min3' ? own >= 3
             : (own >= 3 && own > 0.5 * nh);
    if (ok) m[row] = 1;
  }
  return m;
}

function binnedEfficiency(tp, cols, nRows, keep, fmask,
                          xName, yName, xEdges, yEdges) {
  const nc = cols.length;
  const jx = need(cols, xName, 'efficiency x');
  const jy = need(cols, yName, 'efficiency y');
  const nx = xEdges.length - 1, ny = yEdges.length - 1;
  const num = new Float64Array(nx * ny), den = new Float64Array(nx * ny);
  for (let i = 0; i < nRows; i++) {
    if (!keep[i]) continue;
    const bx = binIndex(tp[i * nc + jx], xEdges); if (bx < 0) continue;
    const by = binIndex(tp[i * nc + jy], yEdges); if (by < 0) continue;
    const b = by * nx + bx;
    den[b] += 1;
    if (fmask[i]) num[b] += 1;
  }
  return { num, den, nx, ny };
}

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
                    stat, SC) {
  const gx = axisGetter(trk, cols, SC, xName, 'binnedStat x');
  const gy = axisGetter(trk, cols, SC, yName, 'binnedStat y');
  const gv = valueName ? axisGetter(trk, cols, SC, valueName, 'binnedStat value')
                       : null;
  const nx = xEdges.length - 1, ny = yEdges.length - 1;
  const buckets = new Array(nx * ny);
  const cnt = new Float64Array(nx * ny);
  for (const i of rows) {
    const bx = binIndex(gx(i), xEdges); if (bx < 0) continue;
    const by = binIndex(gy(i), yEdges); if (by < 0) continue;
    const b = by * nx + bx;
    cnt[b] += 1;
    if (!gv) continue;
    const v = gv(i);
    if (!Number.isFinite(v)) continue;
    (buckets[b] || (buckets[b] = [])).push(v);
  }
  const val = new Float64Array(nx * ny).fill(NaN);
  for (let b = 0; b < nx * ny; b++) {
    // 'count' has no value column: buckets are never filled for it
    if (stat === 'count') { val[b] = cnt[b]; continue; }
    const q = buckets[b];
    if (!q) continue;
    val[b] = stat === 'sigma' ? robustSigma(q)
           : q.reduce((p, c) => p + c, 0) / q.length;
  }
  return { val, cnt, nx, ny };
}

// Fake definitions, selectable because they differ by more than a factor ten.
// All of them are OWNER-REFERENCED: the truth of a track is the TP holding most
// of its hits, not the TP of its seed cluster.
//   'noise'    tp_key < 0   no hit on the track belongs to any TP
//   'dirty'    n_wrong >= 1 any hit from a TP other than the owner
//   'unusable' n_wrong >= 2 default; the sigma(d0) regime change
const FAKE_DEFS = { noise: 'tp_key', dirty: 'n_wrong', unusable: 'n_wrong' };
function isFake(trk, nc, i, cols, def) {
  if (def === 'noise') return trk[i * nc + cols.indexOf('tp_key')] < 0;
  const w = trk[i * nc + cols.indexOf('n_wrong')];
  return def === 'dirty' ? w >= 1 : w >= 2;
}

function binnedFakeRate(trk, cols, rows, xName, yName, xEdges, yEdges, nEvents,
                        def, SC) {
  def = def || 'unusable';
  const nc = cols.length;
  const jk = need(cols, 'tp_key', 'fakeRate'), jw = need(cols, 'n_wrong', 'fakeRate');
  const nx = xEdges.length - 1, ny = yEdges.length - 1;
  const fake = new Float64Array(nx * ny), tot = new Float64Array(nx * ny);
  const gx = axisGetter(trk, cols, SC, xName, 'fakeRate x');
  const gy = axisGetter(trk, cols, SC, yName, 'fakeRate y');
  for (const i of rows) {
    const bx = binIndex(gx(i), xEdges); if (bx < 0) continue;
    const by = binIndex(gy(i), yEdges); if (by < 0) continue;
    const b = by * nx + bx;
    tot[b] += 1;
    const bad = def === 'noise' ? (trk[i * nc + jk] < 0)
              : def === 'dirty' ? (trk[i * nc + jw] >= 1)
              : (trk[i * nc + jw] >= 2);
    if (bad) fake[b] += 1;
  }
  const frac = new Float64Array(nx * ny).fill(NaN);
  const perEv = new Float64Array(nx * ny);
  for (let b = 0; b < nx * ny; b++) {
    if (tot[b] > 0) frac[b] = fake[b] / tot[b];
    perEv[b] = fake[b] / nEvents;
  }
  return { frac, perEv, tot, fake, nx, ny };
}

// ---- efficiency accounting -------------------------------------------------
// Per TrackingParticle, never per candidate: there are ~13.7 candidates per TP,
// so the two differ by a factor of five and the page must state which it shows.
// Banded efficiency over any TP column. `absolute` bands on |value|, which is
// what the prompt/displaced split needs: a particle is displaced by the
// MAGNITUDE of its true impact parameter, either sign.
function effByBand(tp, cols, nRows, keep, fmask, bands, minLayers,
                   colName, absolute) {
  const nc = cols.length;
  const jv = need(cols, colName || 'pt', 'effByBand');
  const jl = need(cols, 'n_layers', 'effByBand');
  const num = new Float64Array(bands.length), den = new Float64Array(bands.length);
  for (let i = 0; i < nRows; i++) {
    if (!keep[i]) continue;
    if (tp[i * nc + jl] < minLayers) continue;
    let v = tp[i * nc + jv];
    if (absolute) v = Math.abs(v);
    for (let k = 0; k < bands.length; k++) {
      if (v >= bands[k][0] && v < bands[k][1]) {
        den[k] += 1;
        if (fmask[i]) num[k] += 1;
        break;
      }
    }
  }
  return { num, den };
}

// Restrict the TP denominator to prompt or displaced particles, by their TRUE
// impact parameter. Truth is legitimate here: a denominator is a statement
// about which particles existed, not a per-track inference.
function tpSplit(tp, cols, nRows, keep, mode, d0Cut) {
  if (mode === 'all') return keep;
  const nc = cols.length, jd = need(cols, 'd0', 'tpSplit');
  const out = new Uint8Array(nRows);
  for (let i = 0; i < nRows; i++) {
    if (!keep[i]) continue;
    const disp = Math.abs(tp[i * nc + jd]) > d0Cut;
    out[i] = (mode === 'displaced' ? disp : !disp) ? 1 : 0;
  }
  return out;
}

// Per-seed efficiency over bands of a TP column, in ONE pass over the table.
// This is the loss map: which seeds reach which particles, as a function of
// (say) true impact parameter. Calling the single-seed path 30 times would walk
// 2.7M rows 30 times over.
function effGridPerSeed(tp, cols, nRows, keep, found, nWords, seedList, bands,
                        minLayers, colName, absolute) {
  const nc = cols.length;
  const jv = need(cols, colName || 'pt', 'effGridPerSeed');
  const jl = need(cols, 'n_layers', 'effGridPerSeed');
  const ns = seedList.length, nb = bands.length;
  const num = new Float64Array(ns * nb), den = new Float64Array(nb);
  const word = seedList.map(i => i >>> 5), bit = seedList.map(i => 1 << (i & 31));
  for (let i = 0; i < nRows; i++) {
    if (!keep[i]) continue;
    if (tp[i * nc + jl] < minLayers) continue;
    let v = tp[i * nc + jv];
    if (absolute) v = Math.abs(v);
    let b = -1;
    for (let k = 0; k < nb; k++) {
      if (v >= bands[k][0] && v < bands[k][1]) { b = k; break; }
    }
    if (b < 0) continue;
    den[b] += 1;
    const base = i * nWords;
    for (let sIdx = 0; sIdx < ns; sIdx++) {
      if (found[base + word[sIdx]] & bit[sIdx]) num[sIdx * nb + b] += 1;
    }
  }
  return { num, den, ns, nb };
}

function sigmaByWrong(trk, cols, rows) {
  const nc = cols.length;
  const jw = need(cols, 'n_wrong', 'sigmaByWrong');
  const jd = need(cols, 'd_d0', 'sigmaByWrong');
  const buckets = [[], [], []];
  for (const i of rows) {
    const v = trk[i * nc + jd];
    if (!Number.isFinite(v)) continue;
    const w = trk[i * nc + jw];
    buckets[w >= 2 ? 2 : w | 0].push(v);
  }
  return buckets.map(b => ({ n: b.length, sigma_um: 1e4 * robustSigma(b) }));
}

// All three fake definitions in one pass with cached column indices, so the
// panel can show what the choice costs without three scans of the table.
function fakeFractions(trk, cols, rows) {
  const nc = cols.length;
  const jk = need(cols, 'tp_key', 'fakeFractions');
  const jw = need(cols, 'n_wrong', 'fakeFractions');
  let noise = 0, dirty = 0, unusable = 0;
  for (const i of rows) {
    if (trk[i * nc + jk] < 0) noise++;
    const w = trk[i * nc + jw];
    if (w >= 1) dirty++;
    if (w >= 2) unusable++;
  }
  const n = Math.max(rows.length, 1);
  return { n: rows.length, noise: noise / n, dirty: dirty / n,
           unusable: unusable / n };
}

// Track-side selection. Cuts named for columns the track table does not carry
// are skipped rather than failing, so the same cut panel can drive both the TP
// and the track side. scoreCuts is a list of [columnName, minimum] applied
// conjunctively -- the quality discriminants are independent, so more than one
// can be active at a time.
// SCORES LIVE IN THEIR OWN uint8 TABLE, not in the float32 track table: at a
// dozen discriminants the float version would exceed the browser's
// single-buffer limit, and a threshold slider stepping 0.01 cannot use more
// than 8 bits anyway. SC = {data, cols, scale}; thresholds are in [0, 1].
function scoreCol(SC, name) {
  return (SC && name) ? SC.cols.indexOf(name) : -1;
}
function filterTracks(trk, cols, rows, cuts, scoreCuts, SC) {
  const nc = cols.length;
  const lim = [];
  for (const k in cuts) {
    const j = cols.indexOf(k);
    if (j >= 0) lim.push([j, cuts[k][0], cuts[k][1]]);
  }
  const sl = [];
  for (const [name, lo] of (scoreCuts || [])) {
    if (!name || !(lo > 0)) continue;
    const j = scoreCol(SC, name);
    if (j < 0) throw new Error(`filterTracks: no score "${name}"`);
    sl.push([j, lo * SC.scale]);
  }
  const snc = SC ? SC.cols.length : 0;
  const out = [];
  outer:
  for (const i of rows) {
    const b = i * nc;
    for (const [j, lo, hi] of lim) {
      const v = trk[b + j];
      if (!(v >= lo && v <= hi)) continue outer;
    }
    const sb = i * snc;
    for (const [j, lo] of sl) {
      if (!(SC.data[sb + j] >= lo)) continue outer;
    }
    out.push(i);
  }
  return out;
}

function scoreKept(SC, rows, scoreCuts) {
  return (scoreCuts || []).map(([name, lo]) => {
    const j = scoreCol(SC, name);
    if (j < 0 || !(lo > 0)) return { name: name, cut: lo, n: rows.length };
    const nc = SC.cols.length, thr = lo * SC.scale;
    let n = 0;
    for (const i of rows) if (SC.data[i * nc + j] >= thr) n++;
    return { name: name, cut: lo, n: n };
  });
}

// A resolution CURVE split by contamination: sigma of `valueName` against the
// x axis, one series per wrong-hit class, in a single pass. The classes are
// 0, 1, 2 and >=3 hits from a TP other than the majority owner, which is the
// split where sigma(d0) changes regime (50 / 207 / 374 / 324 um measured).
// `rows` is the already-selected set, so the curves move with the cuts and the
// score thresholds like every other panel.
const WRONG_CLASSES = ['0 wrong', '1 wrong', '2 wrong', '>=3 wrong'];
// A CLUSTER-LEVEL RESIDUAL WANTS A CLUSTER-LEVEL CLASS. Splitting a per-cluster
// angle residual by the TRACK's wrong-hit count answers the wrong question: the
// interesting divide is whether THIS cluster belongs to the track it was
// attached to, which is what hit_wrong records. The track's n_wrong stays the
// default because every other residual here is per track.
const HIT_CLASSES = ['cluster correctly assigned', 'cluster wrongly assigned'];
const CLASS_SPECS = {
  n_wrong:   { col: 'n_wrong',   labels: WRONG_CLASSES,
               of: (w) => (w >= 3 ? 3 : (w | 0)) },
  hit_wrong: { col: 'hit_wrong', labels: HIT_CLASSES,
               of: (w) => (w > 0 ? 1 : 0) },
  // Per IT layer. The angles are strongly layer-dependent -- a cluster at
  // IL1's radius sees a much larger incidence angle than one at IL4 -- so a
  // pooled distribution is a mixture of four different things.
  layer: { col: 'layer', labels: ['IL1', 'IL2', 'IL3', 'IL4'],
           of: (L) => Math.min(3, Math.max(0, (L | 0) - 1)) },
};
function classSpec(key) {
  const s = CLASS_SPECS[key || 'n_wrong'];
  if (!s) throw new Error(`no class split "${key}"`);
  return s;
}
function statByWrong(trk, cols, rows, xName, xEdges, valueName, stat, SC) {
  const nc = cols.length;
  const jw = need(cols, 'n_wrong', 'statByWrong');
  const gx = axisGetter(trk, cols, SC, xName, 'statByWrong x');
  const gv = axisGetter(trk, cols, SC, valueName, 'statByWrong value');
  const nx = xEdges.length - 1, ncl = WRONG_CLASSES.length;
  const buckets = new Array(ncl * nx);
  const cnt = new Float64Array(ncl * nx);
  for (const i of rows) {
    const bx = binIndex(gx(i), xEdges); if (bx < 0) continue;
    const w = trk[i * nc + jw];
    const k = w >= 3 ? 3 : (w | 0);
    const b = k * nx + bx;
    cnt[b] += 1;
    const v = gv(i);
    if (!Number.isFinite(v)) continue;
    (buckets[b] || (buckets[b] = [])).push(v);
  }
  const val = new Float64Array(ncl * nx).fill(NaN);
  for (let b = 0; b < ncl * nx; b++) {
    const q = buckets[b];
    if (!q) continue;
    val[b] = stat === 'sigma' ? robustSigma(q)
           : q.reduce((p, c) => p + c, 0) / q.length;
  }
  return { val, cnt, nx, ncl, classes: WRONG_CLASSES };
}

// THE DISTRIBUTION ITSELF, not a width taken from it. A single sigma hides the
// shape, and these shapes are not Gaussian: the contaminated classes have a
// narrow core from the hits that happened to sit near the trajectory plus a
// broad shoulder, so two classes can share a sigma and behave completely
// differently in the tails a tagger has to survive.
//
// Each class is normalised to unit area when `density` is set, because the
// populations differ by a factor of six (1.66M clean against 9.44M with >= 3
// wrong) and raw counts would put the clean shape on the floor.
// Width and centre per class, derived from a FINE internal histogram rather
// than by keeping the values: a 12.5M-row selection would need hundreds of MB
// of JS arrays to sort, while a 4000-bin CDF over +-1 cm gives the 15.9/50/84.1
// percentiles to 5 um -- far below the 50 um it is measuring. Entries beyond
// the fine range are counted, and if a percentile falls among them the width is
// reported as NaN instead of silently clamping to the edge.
const FINE_HALF = 1.0, FINE_BINS = 4000;      // cm, so 5 um resolution
function classWidths(trk, cols, rows, valueName, SC, splitBy) {
  const nc = cols.length;
  const cs = classSpec(splitBy);
  const jw = need(cols, cs.col, 'classWidths');
  const gv = axisGetter(trk, cols, SC, valueName, 'classWidths value');
  const ncl = cs.labels.length;
  const H = new Float64Array(ncl * FINE_BINS);
  const lo = new Float64Array(ncl), hi = new Float64Array(ncl);
  const tot = new Float64Array(ncl);
  const step = 2 * FINE_HALF / FINE_BINS;
  for (const i of rows) {
    const k = cs.of(trk[i * nc + jw]);
    const v = gv(i);
    if (!Number.isFinite(v)) continue;
    tot[k] += 1;
    if (v < -FINE_HALF) { lo[k] += 1; continue; }
    if (v >= FINE_HALF) { hi[k] += 1; continue; }
    H[k * FINE_BINS + ((v + FINE_HALF) / step) | 0] += 1;
  }
  const q = (k, p) => {
    const want = p * tot[k];
    if (want <= lo[k]) return NaN;              // percentile is off-scale low
    if (want >= tot[k] - hi[k]) return NaN;     // ... or off-scale high
    let c = lo[k];
    for (let b = 0; b < FINE_BINS; b++) {
      const n = H[k * FINE_BINS + b];
      if (c + n >= want) {
        const f = n > 0 ? (want - c) / n : 0.5;
        return -FINE_HALF + (b + f) * step;
      }
      c += n;
    }
    return NaN;
  };
  return cs.labels.map((lab, k) => {
    const p16 = q(k, 0.15865), p50 = q(k, 0.5), p84 = q(k, 0.84135);
    return { cls: lab, n: tot[k], median: p50,
             sigma: (Number.isFinite(p16) && Number.isFinite(p84))
                    ? 0.5 * (p84 - p16) : NaN,
             off: tot[k] > 0 ? (lo[k] + hi[k]) / tot[k] : 0 };
  });
}

function histByWrong(trk, cols, rows, valueName, edges, density, SC, splitBy) {
  const nc = cols.length;
  const cs = classSpec(splitBy);
  const jw = need(cols, cs.col, 'histByWrong');
  const gv = axisGetter(trk, cols, SC, valueName, 'histByWrong value');
  const nb = edges.length - 1, ncl = cs.labels.length;
  const h = new Float64Array(ncl * nb);
  const tot = new Float64Array(ncl);
  const under = new Float64Array(ncl), over = new Float64Array(ncl);
  for (const i of rows) {
    const k = cs.of(trk[i * nc + jw]);
    const v = gv(i);
    if (!Number.isFinite(v)) continue;
    tot[k] += 1;
    if (v < edges[0]) { under[k] += 1; continue; }
    if (v >= edges[nb]) { over[k] += 1; continue; }
    const b = binIndex(v, edges);
    if (b >= 0) h[k * nb + b] += 1;
  }
  const out = new Float64Array(ncl * nb);
  for (let k = 0; k < ncl; k++) {
    for (let b = 0; b < nb; b++) {
      const c = h[k * nb + b];
      out[k * nb + b] = density ? (tot[k] > 0 ? c / tot[k] : NaN) : c;
    }
  }
  return { val: out, raw: h, nb, ncl, tot, under, over,
           classes: cs.labels };
}

// SMARTPIXELS CLUSTER ANGLE against the track angle at the same sensor.
//
// The angle table is PER CLUSTER, not per track, so it carries track_row to
// join back. `keepTrack` is a flag array over track rows built from the current
// selection, which is how the cuts and score thresholds reach a per-hit plot.
//
// The split is by hit_wrong -- whether THIS cluster belongs to a TP other than
// the track's majority owner -- which is sharper than the track's total
// wrong-hit count because it separates the contaminating cluster from its
// innocent neighbours on the same track.
const ANGLE_CLASSES = ['cluster is correct', 'cluster is wrong'];
function angleProfile(A, acols, keepTrack, side, xEdges, stat) {
  const nc = acols.length;
  const jr = need(acols, 'track_row', 'angleProfile');
  const jw = need(acols, 'hit_wrong', 'angleProfile');
  const jx = need(acols, 'pred_' + side, 'angleProfile x');
  const jv = need(acols, 'res_' + side, 'angleProfile value');
  const nx = xEdges.length - 1;
  const buckets = new Array(2 * nx);
  const cnt = new Float64Array(2 * nx);
  const n = A.length / nc;
  for (let i = 0; i < n; i++) {
    const b0 = i * nc;
    const t = A[b0 + jr];
    if (keepTrack && !(t >= 0 && keepTrack[t])) continue;
    const bx = binIndex(A[b0 + jx], xEdges); if (bx < 0) continue;
    const k = A[b0 + jw] > 0 ? 1 : 0;
    const b = k * nx + bx;
    cnt[b] += 1;
    const v = A[b0 + jv];
    if (!Number.isFinite(v)) continue;
    (buckets[b] || (buckets[b] = [])).push(v);
  }
  const val = new Float64Array(2 * nx).fill(NaN);
  for (let b = 0; b < 2 * nx; b++) {
    const q = buckets[b];
    if (!q) continue;
    val[b] = stat === 'mean' ? q.reduce((p, c) => p + c, 0) / q.length
                             : robustSigma(q);
  }
  return { val, cnt, nx, classes: ANGLE_CLASSES };
}

// flag array over track rows, so a per-hit table can respect a track selection
function trackFlags(rows, nTrk) {
  const f = new Uint8Array(nTrk);
  for (const i of rows) f[i] = 1;
  return f;
}

function linEdges(lo, hi, n) {
  const e = new Float64Array(n + 1);
  for (let i = 0; i <= n; i++) e[i] = lo + (hi - lo) * i / n;
  return e;
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { seedMaskWords, anyBitSet, tpSelect, binIndex,
                     bitmapMask, deliveredMask, OWN_DEFS, tpSplit, effGridPerSeed,
                     binnedEfficiency, drSurvivorsGraph, allEnabled,
                     drJob, allEnabledJob,
                     robustSigma, binnedStat, binnedFakeRate, effByBand,
                     isFake, FAKE_DEFS, sigmaByWrong, need, linEdges,
                     filterTracks, fakeFractions, scoreKept, scoreCol, axisGetter, DERIVED,
                     statByWrong, histByWrong, classWidths, WRONG_CLASSES,
                     HIT_CLASSES, CLASS_SPECS,
                     angleProfile, trackFlags, ANGLE_CLASSES };
}
