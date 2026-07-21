/* JXA smoke of the rendered site data: loads manifest + metas + binaries from
 * __SITEDIR__ exactly as explorer.html would (typed arrays, byte offsets) and
 * runs the compute core on each dataset.  Run via the pytest gate
 * (tests/test_mva_explorer.py::test_site_smoke_jxa) or manually:
 *   cat explorer_core.js smoke_site_jxa.js | sed s|__SITEDIR__|...| > /tmp/s.js
 *   osascript -l JavaScript /tmp/s.js
 * Prints one line per check; final line "SMOKE PASS" or "<n> SMOKE FAILURES".
 */

ObjC.import("Foundation");

var SITE = "__SITEDIR__";

function readText(p) {
  return ObjC.unwrap($.NSString.stringWithContentsOfFileEncodingError(
    p, $.NSUTF8StringEncoding, null));
}

/* NSData -> ArrayBuffer: getBytes is awkward from JXA; use a base64
 * round-trip (fast enough for smoke sizes). */
function readArrayBuffer(p) {
  const data = $.NSData.dataWithContentsOfFile(p);
  if (data.isNil()) throw new Error("cannot read " + p);
  const b64 = ObjC.unwrap(data.base64EncodedStringWithOptions(0));
  const bin = atob_(b64);
  const buf = new ArrayBuffer(bin.length);
  const u8 = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return buf;
}

var B64CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
function atob_(s) {
  s = s.replace(/=+$/, "");
  let out = "";
  let bits = 0, acc = 0;
  for (let i = 0; i < s.length; i++) {
    const v = B64CHARS.indexOf(s[i]);
    if (v < 0) continue;
    acc = (acc << 6) | v;
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out += String.fromCharCode((acc >> bits) & 0xff);
    }
  }
  return out;
}

var failures = 0;
function check(name, cond, detail) {
  if (cond) console.log("ok   " + name);
  else { failures++; console.log("FAIL " + name + (detail ? ": " + detail : "")); }
}

function finite1d(a) { return a.every(v => Number.isFinite(v)); }

function run() {
  const manifest = JSON.parse(readText(SITE + "/manifest.json"));
  check("manifest has regressions", manifest.regressions.length >= 1);

  for (const entry of manifest.regressions) {
    const meta = JSON.parse(readText(SITE + "/" + entry.meta));
    const buf = readArrayBuffer(SITE + "/" + meta.file);
    if (meta.type === "structured") {
      const arr = meta.quant === "log10_i16" ? new Int16Array(buf) : new Float32Array(buf);
      const expected = meta.shape.reduce((a, b) => a * b, 1);
      check(meta.id + " size", arr.length === expected,
            arr.length + " vs " + expected);
      const gridShape = meta.shape.slice(3);
      const gridSize = gridShape.reduce((a, b) => a * b, 1);
      const num = meta.configs.indexOf("1111"), den = meta.configs.indexOf("0000");
      const base = ((0 * meta.configs.length + num) * meta.params.length + 0) * gridSize;
      const denBase = ((0 * meta.configs.length + den) * meta.params.length + 0) * gridSize;
      const r = computeGridEntry({
        valueFn: makeValueFn(arr, meta.quant, meta.scale, base, denBase),
        shape: gridShape, centers: meta.axes.map(a => a.centers),
        fixedBins: {}, xAxis: 0, yAxis: null, cuts: {}, agg: "median",
      });
      check(meta.id + " ratio slice", r.x.length === gridShape[0] &&
            r.mid.some(v => Number.isFinite(v)));
    } else {
      for (const cm of meta.corrections) {
        const view = cm.quant === "log10_i16"
          ? new Int16Array(buf, cm.byte_offset, cm.n_values)
          : new Float32Array(buf, cm.byte_offset, cm.n_values);
        const nReal = cm.axes.filter(a => a.kind === "real").length;
        if (nReal === 0) continue;
        const ix = cm.axes.findIndex(a => a.kind === "real");
        const fixedBins = {};
        cm.axes.forEach((a, i) => { if (a.kind === "cat") fixedBins[i] = 0; });
        const r = computeGridEntry({
          valueFn: makeValueFn(view, cm.quant, cm.scale, 0, null),
          shape: cm.shape,
          centers: cm.axes.map(a => a.kind === "real" ? a.centers : null),
          fixedBins, xAxis: ix, yAxis: null, cuts: {}, agg: "median",
        });
        check(meta.id + "/" + cm.name, r.x.length === cm.shape[ix] && finite1d(r.mid));
      }
    }
  }

  for (const key of ["tkquality", "tagger"]) {
    const entry = manifest[key];
    if (!entry) { console.log("skip " + key + " (not exported)"); continue; }
    const meta = JSON.parse(readText(SITE + "/" + entry.meta));
    const buf = readArrayBuffer(SITE + "/" + meta.file);
    const arr = new Float32Array(buf);
    const total = meta.groups.reduce((a, g) => a + g.n_rows, 0);
    check(meta.id + " size", arr.length === total * meta.columns.length,
          arr.length + " vs " + total * meta.columns.length);
    const g = meta.groups[0];
    const table = makeTable(arr, meta.columns, g.row_offset, g.n_rows);
    const rows = selectRows(table, {});
    const scoreCol = meta.score_columns[0];
    const s = columnValues(table, scoreCol, rows);
    const lab = columnValues(table, meta.label_column, rows);
    const pos = positiveMask(lab, meta.id === "tkq" ? 1 : 0);
    check(meta.id + " scores in [0,1]",
          s.every(v => v >= 0 && v <= 1));
    const auc = aucRanked(s, pos);
    check(meta.id + " global AUC finite", Number.isFinite(auc), String(auc));
    const xcol = Object.keys(meta.x_defaults).find(c => c !== "score");
    const edges = makeEdges(meta.x_defaults[xcol]);
    const x = columnValues(table, xcol, rows);
    const st = binnedScoreStats(x, s, edges, "mean");
    check(meta.id + " binned stats", st.n.reduce((a, b) => a + b, 0) > 0);
    const ef = binnedEfficiency(x, s, pos, 0.5, edges);
    check(meta.id + " efficiency bins", ef.eff.length === edges.length - 1);
    const ba = binnedAUC(x, s, pos, edges);
    check(meta.id + " per-bin AUC", ba.auc.length === edges.length - 1);
    console.log("     " + meta.id + " group " + g.id + ": n=" + g.n_rows +
                " globalAUC(" + scoreCol + ")=" + auc.toFixed(4));
  }

  return failures === 0 ? "SMOKE PASS" : failures + " SMOKE FAILURES";
}

run();
