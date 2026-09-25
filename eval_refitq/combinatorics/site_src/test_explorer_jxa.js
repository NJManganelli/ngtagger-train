// JXA test harness for the explorer page (explorer.html + explorer_core.js).
//
//   osascript -l JavaScript site_src/test_explorer_jxa.js            # tests site_src/
//   osascript -l JavaScript site_src/test_explorer_jxa.js site/       # tests a deployed copy
//
// JXA (macOS JavaScriptCore via osascript), NOT node. It evaluates the page's
// real sources against a small synthetic dataset with the same SHAPE as an
// export, and prints one line per check. What it covers:
//   setup() and menu population; TBL() table switching per scenario; seed
//   groups; the baseline scenarios (delivered efficiency via tp_row, the
//   collection's own score table, score menus following the scenario);
//   quantized-score slider snapping (3-bit levels, byte-rounding case);
//   angle-distribution guards; seed maps; drawInner on every scenario.
// Read the output: a line saying THREW, or an "(expect ...)" not matched, is a
// failure. The expectations are written next to the numbers.
//
// STUB QUIRKS that look like page bugs and are not:
//   * the element stub returns "20" for an unset input, so every CUTS box
//     (pt, eta, phi, d0, z0, n_layers) must be opened explicitly or all tracks
//     are cut -- the snap test does this;
//   * drawInner is async and JXA pumps no microtasks, so a second copy of the
//     page with async/await stripped (HS) is used wherever a throw must surface;
//     paths that really await (the seed-menu scenarios' runJob) are only
//     exercised through the first copy;
//   * window must be the global object (see below), not a stub.
ObjC.import('Foundation');
function rd(p){ return $.NSString.stringWithContentsOfFileEncodingError(p, 4, $()).js; }
// the page sources: this script's own directory, or the first argument
var ARGS = ObjC.deepUnwrap($.NSProcessInfo.processInfo.arguments);
var ME = ARGS.filter(function(a){ return /\.js$/.test(a); }).pop();
var HERE = ME.indexOf('/') >= 0 ? ME.replace(/\/[^\/]*$/, '') : '.';
var REST = ARGS.slice(ARGS.indexOf(ME) + 1);
var SITE = (REST.length ? REST[0] : HERE).replace(/\/$/, '');
var log=[], els={}, plots=[];
var GRPBOX=[], SDBOX=[];
var PANELS=['eff_seed','sd0'];
function mkEl(id){ return { id:id, style:{}, _t:'', _o:[], disabled:true,
  set textContent(v){ this._t=v; }, get textContent(){ return this._t; },
  set innerHTML(v){ this._h=v; }, get innerHTML(){ return this._h||''; },
  set className(v){}, set onclick(f){ this._click=f; },
  add:function(o){ this._o.push(o); }, get options(){ return this._o; },
  remove:function(i){ this._o.splice(i,1); },
  addEventListener:function(n,f){ (this._ev=this._ev||[]).push(f); },
  querySelectorAll:function(){ return []; },
  insertAdjacentHTML:function(){ }, set value(v){ this._v=v; },
  get value(){ return this._v!==undefined ? this._v : (this._o.length? this._o[0].value : '20'); },
  set title(v){}, set checked(v){}, get checked(){ return true; },
  get dataset(){ return {i:'0'}; } }; }
var document={ getElementById:function(id){ return els[id]||(els[id]=mkEl(id)); },
  querySelector:function(){ return {value:'joint'}; },
  querySelectorAll:function(s){
    if (s.indexOf('.cellsel')>=0) return PANELS.map(function(v){
      return {value:v, addEventListener:function(){}}; });
    if (s.indexOf('.grpsel')>=0) return GRPBOX;
    if (s.indexOf('.sd[data-g=')>=0) {
      var g=s.split('"')[1];
      return SDBOX.filter(function(x){ return x.dataset.g===g; });
    }
    if (s.indexOf('.sd')>=0) return SDBOX;
    return [];
  } };
// window MUST BE THE GLOBAL OBJECT, not a stub. The page resolves its core
// as `CORE = module ? module.exports : window`, and in a browser the core's
// top-level function declarations ARE properties of window. Stubbing it as
// {} left CORE empty, so every C.something call failed -- a harness
// artifact that looks exactly like a page bug.
var window=(function(){ return this; })();
function Option(a,b){ return {label:a, value:b}; }
// drawInner's first statement is performance.now(); without this stub it
// throws into a rejected promise that JXA never surfaces, and the harness
// reports 'no synchronous rejection' while nothing has actually run.
var performance={ now:function(){ return 0; } };
var Plotly={ react:function(id,tr,lay){ plots.push({id:id, traces:tr.length,
                                          title:(lay.title&&lay.title.text)||''}); } };
// ---- a synthetic dataset with the same SHAPE as the real one -------------
var NTP=8, NTRK=6, NW=1, NSC=2;
var TPCOLS=['key','event','pt','eta','phi','d0','z0','vr','hit_it','hit_ot','n_layers'];
var TRKCOLS=['seed_idx','tp_key','tp_row','pt','phi','eta','d0','z0','n_layers','nhit',
             'n_own','n_it_right','n_it_wrong','n_ot_right','n_ot_wrong','sig_kf_d0',
             'n_wrong','d_d0','d_z0','d_kappa','d_phi0','d_cot'];
var MAN = { built:'test', n_events:2, source:'synthetic', ptmin:2.0,
  seeds:[{idx:0,tag:'IL1+IL2',layers:[1,2],arity:2,sysclass:0},
         {idx:2,tag:'IL4+OL1',layers:[4,11],arity:2,sysclass:2},
         {idx:3,tag:'IL1+IL2 soft',layers:[1,2],arity:2,sysclass:0,soft:true},
         {idx:1,tag:'OL1+OL2',layers:[11,12],arity:2,sysclass:1}],
  builds:{AAAA:['IL1+IL2','OL1+OL2','IL4+OL1','IL1+IL2 soft']},
  mva:{ mva_clean_prompt:{auc:0.99,features:['a','b'],population:'all',
        target:'x',auc_displaced:0.97,'keep_rate_at_0.9_clean':{prompt:0.8,displaced:0.6}} },
  tp:{file:'tp.bin',cols:TPCOLS,rows:NTP,dtype:'float32'},
  found:{file:'found.bin',rows:NTP,words:NW,dtype:'uint32'},
  found_dr:{file:'found_dr.bin',rows:NTP,words:NW,dtype:'uint32'},
  track:{file:'track.bin',cols:TRKCOLS,rows:NTRK,dtype:'float32',n_no_owner:1},
  score:{file:'score.bin',cols:['mva_clean_prompt','mva_d0_prompt'],rows:NTRK,
         dtype:'uint8',scale:255},
  counters:{'0':{pairs:100,seed_objects:80,before_chi2:40,fit:20,owned:10,cand:5000}},
  funnel_stages:['pairs','seed_objects','before_chi2','fit','owned'],
  funnel_labels:{pairs:'pairs',seed_objects:'seeds',before_chi2:'layer rule',
                 fit:'chi2',owned:'after DR'},
  notes:[] };
function f32(n){ var a=new Float32Array(n); for(var i=0;i<n;i++) a[i]=0; return a; }
var TPB=f32(NTP*TPCOLS.length), TRB=f32(NTRK*TRKCOLS.length);
for (var i=0;i<NTP;i++){ var b=i*TPCOLS.length;
  TPB[b+2]=3+i; TPB[b+3]=0.1*i-0.4; TPB[b+4]=0.2; TPB[b+5]=0.001*(i%4);
  TPB[b+6]=1.0; TPB[b+10]=6; }
for (var i=0;i<NTRK;i++){ var b=i*TRKCOLS.length;
  TRB[b+0]=i%2; TRB[b+1]=i<5?i:-1; TRB[b+2]=i<5?i:-1; TRB[b+3]=4+i; TRB[b+4]=0.1;
  TRB[b+5]=0.2; TRB[b+6]=0.002; TRB[b+7]=1.0; TRB[b+8]=6; TRB[b+9]=6; TRB[b+10]=6-(i%3);
  TRB[b+11]=4; TRB[b+12]=i%2; TRB[b+13]=2; TRB[b+14]=0; TRB[b+15]=0.0016;
  TRB[b+16]=i%3; TRB[b+17]=0.004*(1+(i%3)); TRB[b+18]=0.01; TRB[b+19]=0.005;
  TRB[b+20]=0.001; TRB[b+21]=0.002; }
var FOUND=new Uint32Array(NTP); for (var i=0;i<NTP;i++) FOUND[i]=(i%3)?1:3;
var SCB=new Uint8Array(NTRK*NSC); for (var i=0;i<NTRK*NSC;i++) SCB[i]=40+30*i;
var BUF={ 'manifest.json':MAN, 'tp.bin':TPB.buffer, 'found.bin':FOUND.buffer,
          'found_dr.bin':FOUND.buffer, 'track.bin':TRB.buffer, 'score.bin':SCB.buffer };
function fetch(u,o){
  var key=String(u).split('?')[0];
  var v=BUF[key];
  if (v===undefined) return Promise.reject(new Error('404 '+key));
  var len = (key==='manifest.json') ? 1 : v.byteLength;
  return Promise.resolve({ ok:true, status:200,
    headers:{ get:function(){ return String(len); } },
    json:function(){ return Promise.resolve(v); },
    arrayBuffer:function(){ return Promise.resolve(v); },
    body:null });
}
function AbortController(){ return {signal:{},abort:function(){}}; }
var TQ=[]; function setTimeout(f,t){ TQ.push(f); return TQ.length; }
function clearTimeout(){}
function drainQ(){
  var n=0; while(TQ.length && n<20000){ var f=TQ.shift(); try{ f(); }catch(e){ log.push('timer threw: '+e.message); } n++; }
  // spin the Objective-C run loop so JavaScriptCore drains its microtask queue
  $.NSRunLoop.currentRunLoop.runUntilDate(
    $.NSDate.dateWithTimeIntervalSinceNow(0.01));
}
// READ THE DEPLOYED FILES DIRECTLY. This used to eval served_core.js and
// served_page.js, hand-refreshed snapshots in the scratchpad -- so the harness
// happily passed while testing a page two days old, which is worse than not
// running it. SITE is the deployed directory; the page's inline script is
// extracted here rather than copied by hand.
function inlineScript(html){
  var out=[], i=0;
  while (true) {
    var a=html.indexOf('<script', i); if (a<0) break;
    var b=html.indexOf('>', a); var c=html.indexOf('</script>', b);
    if (b<0 || c<0) break;
    if (html.slice(a,b).indexOf(' src=')<0) out.push(html.slice(b+1,c));
    i=c+9;
  }
  return out.join('\n');
}
// explorer_core.js opens with 'use strict', so under eval its top-level
// function declarations stay inside the eval scope instead of becoming
// globals the way a <script> tag would make them. The core already has a
// module.exports branch for exactly this, and the page's own CORE lookup
// prefers module.exports when it exists -- so define module first and both
// halves resolve through the supported path.
var module={ exports:{} };
try { eval(rd(SITE+'/explorer_core.js')); } catch(e){ log.push('CORE THREW: '+e.message); }
log.push('core exports: '+Object.keys(module.exports).length+' names');
// const/let inside a direct eval are not visible outside it, so the source is
// evaluated with an export hook appended -- same eval, so the bindings are in
// scope at that point.
var H = null;
try {
  eval(inlineScript(rd(SITE+'/explorer.html'))
       + '\n;H = {S:S, setup:setup, buildSeeds:buildSeeds, METRICS:METRICS,'
       + ' SCORE_COLS:SCORE_COLS, TRACK_ONLY:TRACK_ONLY,'
       + ' TBL:TBL, BASELINES:BASELINES, scenario:scenario, drawInner:drawInner,'
       + ' syncGroups:syncGroups};');
} catch(e){ log.push('PAGE THREW: '+e.name+': '+e.message); }
// Async cannot be driven under JXA (no microtask pump), so the synchronous
// seam is exercised directly: load the buffers into S exactly as boot() would,
// then call setup(). This is the part that breaks.
var S = H.S; var setup = H.setup;
S.m = MAN;
S.tp = TPB; S.found = FOUND; S.foundDr = FOUND; S.trk = TRB; S.score = SCB;
S.nTp = NTP; S.nW = NW; S.nTrk = NTRK;
S.canDeliver = MAN.track.cols.indexOf('tp_row') >= 0;
S.SC = { data: SCB, cols: MAN.score.cols, scale: MAN.score.scale };
S.seedOf = new Int32Array(NTRK);
for (var i=0;i<NTRK;i++) S.seedOf[i] = TRB[i*TRKCOLS.length];
try { setup(); log.push('setup() completed'); }
catch(e){ log.push('SETUP THREW: ' + e.name + ': ' + e.message); }
log.push('status: '+els['status'].textContent);
log.push('progress: '+els['progtext'].textContent);
log.push('build options: '+(els['build']?els['build'].options.length:'-'));
log.push('xcol options: '+(els['xcol']?els['xcol'].options.length:'-'));
log.push('mvaA options: '+(els['mvaA']?els['mvaA'].options.length:'-'));
log.push('plots drawn: '+plots.length+' '+JSON.stringify(plots.map(function(p){return p.title;})));
// ---- the baseline scenarios: TBL must swap the active table ---------------
// draw() is async and JXA has no microtask pump, so the synchronous seam is
// what gets exercised: TBL() decides which table every panel then bins.
var BCOLS=['pt','phi','eta','d0','z0','n_layers','nhit','n_ot_right','n_ot_wrong',
           'n_ot_combinatoric','n_ot_unknown','n_wrong','tp_pt','tp_eta','tp_d0',
           'tp_z0','d_d0','d_z0','d_kappa','d_phi0','d_cot','n_own','tp_row'];
var NB=5, BB=new Float32Array(NB*BCOLS.length);
for (var i=0;i<NB;i++){ BB[i*BCOLS.length+BCOLS.indexOf('pt')]=5;
                        BB[i*BCOLS.length+BCOLS.indexOf('d_d0')]=(i-2)*0.01;
                        BB[i*BCOLS.length+BCOLS.indexOf('n_wrong')]=i%4;
                        BB[i*BCOLS.length+BCOLS.indexOf('n_own')]=5-(i%4);
                        BB[i*BCOLS.length+BCOLS.indexOf('nhit')]=5;
                        BB[i*BCOLS.length+BCOLS.indexOf('tp_row')]=i<4?i:-1; }
var BSC1=new Uint8Array(NB*2), BSC2=new Uint8Array(NB*2);
for (var i=0;i<NB*2;i++){ BSC1[i]=Math.round(255*(i%8)/7); BSC2[i]=20+40*i; }
MAN.baseline={file:'baseline.bin',cols:BCOLS,rows:NB,label:'Phase-2 baseline',
  score:{file:'baseline_score.bin',cols:['cmssw_tq_hw3','cmssw_tq_float'],rows:NB,scale:255,
         default:['cmssw_tq_hw3',''],mva:{cmssw_tq_hw3:{what:'3-bit word',levels:8},
         cmssw_tq_float:{what:'float'}}}};
MAN.baseline_emu={file:'baseline_emu.bin',cols:BCOLS,rows:NB,
                  label:'Phase-2 baseline (re-emulated)',
  score:{file:'baseline_emu_score.bin',cols:['mva_tq_prompt','mva_tq_displaced'],rows:NB,scale:255,
         default:['mva_tq_prompt','mva_tq_displaced'],
         mva:{mva_tq_prompt:{auc:0.9,features:['a'],population:'p',target:'t'},
              mva_tq_displaced:{auc:0.94,features:['a'],population:'d',target:'t'}}}};
S.baseline=BB; S.baselineEmu=BB;
var scen='joint';
document.querySelector=function(){ return {value:scen}; };
['joint','baseline','baseline_emu'].forEach(function(sc){
  scen=sc;
  try {
    var tb=H.TBL();
    log.push('TBL('+sc+') -> '+(tb.isBaseline?'BASELINE':'emulation')
             +' table, '+tb.cols.length+' cols, '+tb.n+' rows'
             +(tb.isBaseline&&tb.t!==BB ? '  WRONG BUFFER' : ''));
  } catch(e){ log.push('TBL('+sc+') THREW: '+e.name+': '+e.message); }
});
// ---- seed groups ---------------------------------------------------------
scen='joint';
try {
  H.buildSeeds();
  var html = String(els['seeds'].innerHTML||'');
  var groups = (html.match(/data-g="[a-z]+"/g)||[])
                 .filter(function(v,i,a){return a.indexOf(v)===i;});
  log.push('seed groups rendered: ' + groups.join(' '));
  log.push('  soft group present: ' + (html.indexOf('sub-2-GeV recovery')>=0));
  log.push('  master boxes: ' + (html.match(/class="grpsel"/g)||[]).length);
} catch(e){ log.push('buildSeeds(joint) THREW: '+e.message); }
// toggling the soft group must not touch the others
// stubs must look like elements: buildSeeds attaches listeners to them
function box(g,i){ return {checked:true, indeterminate:false,
  dataset:{g:g,i:i}, addEventListener:function(){}}; }
SDBOX=[box('it','0'), box('ot','1'), box('soft','3')];
GRPBOX=[box('it'), box('ot'), box('soft')];
SDBOX.forEach(function(x){ if(x.dataset.g==='soft') x.checked=false; });
try {
  H.syncGroups();
  log.push('after unticking soft: ' + GRPBOX.map(function(g){
    return g.dataset.g+'='+(g.indeterminate?'partial':(g.checked?'on':'off'));
  }).join(' '));
} catch(e){ log.push('syncGroups THREW: '+e.message); }

scen='baseline';
try { H.buildSeeds(); log.push('buildSeeds(baseline): '+els['seeds'].innerHTML.slice(0,60)); }
catch(e){ log.push('buildSeeds(baseline) THREW: '+e.message); }
// drawInner is async, but on a baseline it never awaits (runJob is only called
// on the seed-menu path), so the body runs to completion synchronously and a
// throw surfaces as a rejected promise. Capture it explicitly.
S.tp=TPB; S.nTp=NTP; S.nW=1; S.found=FOUND; S.foundDr=FOUND;
// A SECOND, SYNCHRONOUS COPY OF THE PAGE. drawInner is async, so any throw
// becomes a rejected promise, and JXA never pumps microtasks -- the handler
// cannot run, so a crash is indistinguishable from success. Re-evaluating the
// source with 'async'/'await' stripped makes the body run inline and throw for
// real. Valid only for paths that never actually await: the baselines take the
// branch that skips runJob. 'joint' does await, so it is expected to misbehave
// here and is included only as a contrast.
var HS = null;
try {
  eval(inlineScript(rd(SITE+'/explorer.html'))
         .replace(/\basync\s+function\b/g, 'function')
         .replace(/\bawait\s+/g, '')
       + '\n;HS = {S:S, setup:setup, drawInner:drawInner, fillScoreMenus:fillScoreMenus, activeSC:activeSC};');
} catch(e){ log.push('SYNC COPY THREW: '+e.name+': '+e.message); }
if (HS) { HS.S.m = MAN; HS.S.tp = TPB; HS.S.found = FOUND; HS.S.foundDr = FOUND;
          HS.S.trk = TRB; HS.S.score = SCB; HS.S.nTp = NTP; HS.S.nW = 1;
          HS.S.nTrk = NTRK; HS.S.baseline = BB; HS.S.baselineEmu = BB;
          HS.S.SC = { data: SCB, cols: MAN.score.cols, scale: MAN.score.scale };
          HS.S.seedOf = S.seedOf;
          HS.S.SCbase = { baseline: {data:BSC1, cols:MAN.baseline.score.cols, scale:255},
                          baseline_emu: {data:BSC2, cols:MAN.baseline_emu.score.cols, scale:255} }; }
// ---- NEW: score menus follow the scenario ---------------------------------
['joint','baseline','baseline_emu','joint'].forEach(function(sc){
  scen=sc;
  try { HS.fillScoreMenus();
    var A=els['mvaA'], B=els['mvaB'];
    log.push('menus('+sc+'): A='+A.value+' B='+(B.value||'off')+' | A options ['
      + A.options.map(function(o){return o.value||'off';}).join(',')+'] | activeSC cols '
      + (HS.activeSC()?HS.activeSC().cols.join(','):'none'));
  } catch(e){ log.push('menus('+sc+') THREW: '+e.name+': '+e.message); }
});
// ---- NEW: quantized slider snaps to real levels ---------------------------
PANELS=['sd0_wrong'];
['pt','eta','phi','d0','z0','n_layers'].forEach(function(c){ els['lo_'+c]=mkEl('lo_'+c); els['lo_'+c].value='-100';
                                        els['hi_'+c]=mkEl('hi_'+c); els['hi_'+c].value='100'; });
[['0.3','level 2 -> 3 tracks (levels 2,4,6)'],['0.2857','exactly 2/7 -> same 3'],
 ['0.9','level 6 -> 1 track'],['0.05','level 0 -> no cut, 5 tracks']].forEach(function(tc){
  scen='baseline'; HS.fillScoreMenus(); els['mvaA'].value='cmssw_tq_hw3'; els['mvaB'].value='';
  els['cutA'].value=tc[0]; els['cutB'].value='0';
  try { HS.drawInner(); } catch(e){ log.push('snap THREW '+e.message); }
  var info=String((els['mvainfo']||{}).innerHTML||'');
  var shown=(info.match(/shown<\/th><td><b>([0-9,]+)/)||[])[1];
  if (tc[0]==='0.05') log.push('DEBUG mvainfo: '+info+' | status: '+els['status'].textContent+' | minlay '+els['minlay'].value);
  log.push('snap cutA='+tc[0]+' -> num '+els['cutA'].value+' step '+els['cutA'].step
    +' shown '+shown+'   (expect '+tc[1]+') status: '+els['status'].textContent.slice(-30));
});
scen='baseline_emu'; HS.fillScoreMenus(); els['cutA'].value='0.37';
try { HS.drawInner(); } catch(e){ log.push('emu THREW '+e.message); }
log.push('non-quantized: cutA '+els['cutA'].value+' step '+els['cutA'].step+' slider step '+els['slideA'].step);
// ---- the angle residual distributions -----------------------------------
// MAN has no angle table, so this exercises the guard; the real check of the
// binning is the JSC unit test against explorer_core directly.
PANELS=['dist_alpha','dist_beta'];
scen='joint'; plots.length=0;
try { if (HS) HS.drawInner(); } catch(e){ log.push('angle dist THREW: '+e.message); }
drainQ();
log.push('angle dists (no angle table): ' + plots.length + ' plotted, '
  + 'panel text: ' + String((els['pp0']||{}).innerHTML||'').slice(0,58));

// ---- the seed maps -------------------------------------------------------
PANELS=['map_it','map_ot','map_all'];
['joint','baseline'].forEach(function(sc){
  scen=sc; plots.length=0; var caught=null;
  try { if (HS) HS.drawInner(); } catch(e){ caught=e; }
  drainQ();
  log.push('seed maps ('+sc+'): ' + (caught ? 'THREW '+caught.message
    : plots.length+' drawn, traces '+plots.map(function(x){return x.traces;}).join('/')));
});
PANELS=['eff_deliv','dist_score','eff_seed','sd0_wrong'];
['baseline','baseline_emu','joint'].forEach(function(sc){
  scen=sc; var caught=null; plots.length=0;
  try { HS.fillScoreMenus(); } catch(e){ log.push('fill THREW '+e.message); }
  try { if (HS) HS.drawInner(); } catch(e){ caught=e; }
  drainQ();
  // Progress is judged by what got WRITTEN, not by a promise settling: JXA has
  // no microtask pump, so a .then callback never runs and 'no rejection' would
  // be indistinguishable from 'never started'.
  var mh = String((els['mvahint']||{}).innerHTML || 'NOT WRITTEN');
  var ma = String((els['mvadefA']||{}).innerHTML || (els['mvadefA']||{})._t || '(empty)');
  log.push('   mvahint: ' + mh.slice(0,52));
  log.push('   mvadefA: ' + (ma===''?'(cleared)':ma.slice(0,52)));
  var ip = String((els['ipbands']||{}).innerHTML || 'NOT WRITTEN');
  var ed = String((els['effdef']||{}).innerHTML || 'NOT WRITTEN');
  log.push('drawInner('+sc+'): '
    + (caught ? 'THREW ' + (caught.name||'') + ': ' + (caught.message||caught) : 'ran')
    + ' | ipbands: ' + ip.slice(0,44)
    + ' | effdef: ' + ed.slice(0,40));
  log.push('   plots: '+plots.length+' '+JSON.stringify(plots.map(function(p){return p.title;}))
    + ' | pp2(eff_seed): '+String((els['pp2']||{}).innerHTML||'').slice(0,50)
    + ' | bands: '+String((els['bands']||{}).innerHTML||'').slice(0,70));
});
log.join('\n');
