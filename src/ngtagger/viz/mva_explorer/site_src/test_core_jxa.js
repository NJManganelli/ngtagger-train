/* JXA test harness for explorer_core.js (macOS: osascript -l JavaScript).
 * The runner (tests/test_mva_explorer.py or a manual shell) concatenates
 *   explorer_core.js + this file
 * into a temp script and replaces __TESTDATA__ with the fixture path
 * produced by make_core_testdata.py.  Prints per-case results; final line is
 * "ALL PASS" or "<n> FAILURES".
 */

ObjC.import("Foundation");

function readText(path) {
  return ObjC.unwrap(
    $.NSString.stringWithContentsOfFileEncodingError(path, $.NSUTF8StringEncoding, null)
  );
}

var TOL = 1e-9;

function close(a, b) {
  if (a === null || (typeof a === "number" && Number.isNaN(a)))
    return b === null || (typeof b === "number" && Number.isNaN(b));
  if (b === null || (typeof b === "number" && Number.isNaN(b))) return false;
  return Math.abs(a - b) <= TOL * (1 + Math.abs(a) + Math.abs(b));
}

function cmp(got, exp, label) {
  if (Array.isArray(exp)) {
    if (!got || got.length !== exp.length)
      return label + ": length " + (got && got.length) + " vs " + exp.length;
    for (let i = 0; i < exp.length; i++) {
      const r = cmp(got[i], exp[i], label + "[" + i + "]");
      if (r) return r;
    }
    return null;
  }
  if (!close(got, exp)) return label + ": " + got + " vs " + exp;
  return null;
}

function run() {
  const td = JSON.parse(readText("__TESTDATA__"));
  let failures = 0;
  const report = (name, err) => {
    if (err) { failures++; console.log(name + " FAIL: " + err); }
    else console.log(name + " ok");
  };

  /* ---- grid cases ---- */
  const g = td.grid;
  const i16 = Int16Array.from(g.i16);
  const f32 = Float32Array.from(g.f32);
  const st = strides(g.shape);
  g.cases.forEach((c, i) => {
    let spec, res;
    const cuts = {};
    for (const k in c.spec.cuts || {}) cuts[k] = c.spec.cuts[k];
    if (c.dataset === "i16_ratio") {
      spec = {
        valueFn: makeValueFn(i16, "log10_i16", td.scale,
                             c.numFixed * st[0], c.denFixed * st[0]),
        shape: g.shape.slice(1),
        centers: g.centers.slice(1),
        fixedBins: {},
        xAxis: c.spec.xAxis, yAxis: c.spec.yAxis,
        cuts, agg: c.spec.agg, invert: !!c.spec.invert,
      };
    } else {
      const arr = c.dataset === "i16" ? i16 : f32;
      const quant = c.dataset === "i16" ? "log10_i16" : "f32";
      spec = {
        valueFn: makeValueFn(arr, quant, td.scale, 0, null),
        shape: g.shape, centers: g.centers,
        fixedBins: c.spec.fixedBins || {},
        xAxis: c.spec.xAxis, yAxis: c.spec.yAxis,
        cuts, agg: c.spec.agg, invert: !!c.spec.invert,
      };
    }
    try { res = computeGridEntry(spec); }
    catch (e) { report("grid[" + i + "]", "threw " + e); return; }
    let err = res.dim !== c.expected.dim ? "dim " + res.dim + " vs " + c.expected.dim : null;
    if (!err) err = cmp(res.x, c.expected.x, "x");
    if (!err && c.expected.dim === 1)
      err = cmp(res.mid, c.expected.mid, "mid") || cmp(res.lo, c.expected.lo, "lo") ||
            cmp(res.hi, c.expected.hi, "hi") || cmp(res.n, c.expected.n, "n");
    if (!err && c.expected.dim === 2)
      err = cmp(res.y, c.expected.y, "y") || cmp(res.z, c.expected.z, "z");
    report("grid[" + i + "]", err);
  });

  /* ---- table cases ---- */
  const t = td.table;
  const rows = Float32Array.from(t.rows);
  const table = makeTable(rows, t.columns, 0, t.nRows);
  t.cases.forEach((c, i) => {
    const sel = selectRows(table, c.cuts);
    const x = columnValues(table, c.xColumn, sel);
    const s = columnValues(table, c.scoreColumn, sel);
    const lab = columnValues(table, c.labelColumn, sel);
    const pos = positiveMask(lab, c.positiveValue);
    const exp = c.expected;
    let err = null;
    if (!err && exp.stats) {
      const r = binnedScoreStats(x, s, c.edgesX, c.agg);
      err = cmp(r.centers, exp.stats.centers, "stats.centers") ||
            cmp(r.mid, exp.stats.mid, "stats.mid") ||
            cmp(r.lo, exp.stats.lo, "stats.lo") ||
            cmp(r.hi, exp.stats.hi, "stats.hi") ||
            cmp(r.n, exp.stats.n, "stats.n");
    }
    if (!err && exp.efficiency) {
      const r = binnedEfficiency(x, s, pos, c.cut, c.edgesX);
      err = cmp(r.eff, exp.efficiency.eff, "eff") ||
            cmp(r.mistag, exp.efficiency.mistag, "mistag") ||
            cmp(r.nPos, exp.efficiency.nPos, "eff.nPos") ||
            cmp(r.nNeg, exp.efficiency.nNeg, "eff.nNeg");
    }
    if (!err && exp.auc) {
      const r = binnedAUC(x, s, pos, c.edgesX);
      err = cmp(r.auc, exp.auc.auc, "auc") ||
            cmp(r.nPos, exp.auc.nPos, "auc.nPos") ||
            cmp(r.nNeg, exp.auc.nNeg, "auc.nNeg");
    }
    if (!err && exp.aucAll !== undefined)
      err = cmp(aucRanked(s, pos), exp.aucAll, "aucAll");
    if (!err && exp.histogram) {
      const posScores = [];
      for (let k = 0; k < s.length; k++) if (pos[k]) posScores.push(s[k]);
      const r = histogram(posScores, c.edgesScore, true);
      err = cmp(r.y, exp.histogram.y, "hist.y") ||
            cmp(r.counts, exp.histogram.counts, "hist.counts") ||
            cmp(r.total, exp.histogram.total, "hist.total");
    }
    report("table[" + i + "]", err);
  });

  /* ---- synthesisEnvelope ---- */
  {
    const ev = td.envelope;
    const rB = { x: ev.x, mid: ev.bias };
    const rS = { x: ev.x, mid: ev.sigma };
    const checkBands = (got, exp, tag) => {
      let e = null;
      exp.forEach((b, i) => {
        if (!e)
          e = (got[i].k !== b.k ? tag + ".k " + got[i].k + " vs " + b.k : null) ||
              cmp(got[i].lo, b.lo, tag + "[k=" + b.k + "].lo") ||
              cmp(got[i].hi, b.hi, tag + "[k=" + b.k + "].hi");
      });
      return e;
    };
    const r = synthesisEnvelope(rB, rS, ev.ks);
    let err = cmp(r.x, ev.x, "env.x") || cmp(r.mid, ev.expected.mid, "env.mid") ||
              checkBands(r.bands, ev.expected.bands, "env.band");
    if (!err) {
      const r0 = synthesisEnvelope(null, rS, ev.ks);
      err = cmp(r0.mid, ev.expected0.mid, "env0.mid") ||
            checkBands(r0.bands, ev.expected0.bands, "env0.band");
    }
    report("envelope", err);
  }

  /* ---- combineCurves ---- */
  {
    const inCurves = td.combine.input.map((c) => ({
      mid: c.mid.map((v) => (v === null ? NaN : v)),
    }));
    const r = combineCurves(inCurves);
    const exp = td.combine.expected;
    const err = cmp(r.mid, exp.mid, "comb.mid") || cmp(r.lo, exp.lo, "comb.lo") ||
                cmp(r.hi, exp.hi, "comb.hi") || cmp(r.n, exp.n, "comb.n");
    report("combine", err);
  }

  return failures === 0 ? "ALL PASS" : failures + " FAILURES";
}

run();
