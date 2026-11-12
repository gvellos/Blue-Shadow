#!/usr/bin/env python3
"""Build a self-contained analyst UI (ui.html) from the pipeline outputs.
Usage: OUT=./out python build_ui.py   (run pipeline.py or pipeline_real.py first)"""
import base64, json, os

OUT = os.environ.get("OUT", "./out")
load = lambda n: json.load(open(f"{OUT}/{n}"))
contacts, allDet, report = load("unidentified_contacts.json"), load("all_detections.json"), load("report.json")
b64 = lambda p: "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode() if os.path.isfile(p) else ""
for c in contacts:
    c["img"] = b64(f"{OUT}/{c.get('crop', '')}")
    c["img_nir"] = b64(f"{OUT}/{c.get('crop_nir', '')}")
ov = f"{OUT}/overview.jpg"
overview = "data:image/jpeg;base64," + base64.b64encode(open(ov, "rb").read()).decode() if os.path.isfile(ov) else ""
data = json.dumps(dict(contacts=contacts, all=allDet, report=report, overview=overview))

VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
_v = lambda n: open(os.path.join(VENDOR, n), encoding="utf-8").read()
LEAFLET_CSS = "\n".join(_v(n) for n in ("leaflet.css", "MarkerCluster.css", "MarkerCluster.Default.css"))
LEAFLET_JS = "\n".join(_v(n) for n in ("leaflet.js", "leaflet.markercluster.js"))

TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Blue Shadow – analyst queue</title>
<style>/*__LEAFLET_CSS__*/</style>
<style>
:root{--bg:#f6f7f9;--pn:#fff;--tx:#14181f;--mu:#667085;--bd:#dde1e8;--ac:#2563eb;box-sizing:border-box;
padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px);color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0e1116;--pn:#171b22;--tx:#e6e9ef;--mu:#98a2b3;--bd:#2a313c;--ac:#6ea0ff;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#0e1116;--pn:#171b22;--tx:#e6e9ef;--mu:#98a2b3;--bd:#2a313c;--ac:#6ea0ff;color-scheme:dark}
html,body{margin:0;background:var(--bg);color:var(--tx);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
header{padding:12px 16px;display:flex;gap:16px;align-items:baseline;flex-wrap:wrap;border-bottom:1px solid var(--bd);background:var(--pn)}
h1{font-size:16px;margin:0}.mu{color:var(--mu)}
nav button{background:none;border:0;color:var(--mu);padding:6px 10px;cursor:pointer;font:inherit;border-bottom:2px solid transparent}
nav button.on{color:var(--tx);border-color:var(--ac)}
.grid{display:grid;grid-template-columns:280px minmax(0,1fr) 320px;gap:12px;padding:12px}
@media(max-width:1000px){.grid{grid-template-columns:1fr}}
.pn{background:var(--pn);border:1px solid var(--bd);border-radius:10px;padding:10px;min-width:0}
.q{max-height:70vh;overflow:auto}.it{padding:8px;border-radius:8px;cursor:pointer;border:1px solid transparent}
.it:hover{background:var(--bg)}.it.sel{border-color:var(--ac)}.it.dis{opacity:.4}
.chip{display:inline-block;padding:1px 7px;border-radius:99px;font-size:11px;color:#fff}
.btn{border:1px solid var(--bd);background:var(--pn);color:var(--tx);border-radius:8px;padding:6px 10px;cursor:pointer;font:inherit;margin:2px 4px 2px 0}
.btn.on{background:var(--ac);color:#fff;border-color:var(--ac)}
.bar{height:6px;border-radius:3px;background:var(--bd);margin:2px 0 8px}.bar i{display:block;height:100%;border-radius:3px;background:var(--ac)}
table{border-collapse:collapse;width:100%}td,th{padding:6px 8px;border-bottom:1px solid var(--bd);text-align:left}
.sc{overflow-x:auto}svg text{fill:var(--mu);font-size:11px}
kbd{font:12px ui-monospace,monospace}
</style></head><body>
<header><h1>Blue Shadow</h1><span class="mu" id="sum"></span>
<nav><button id="t1" class="on">Queue</button><button id="t2">Size bands &amp; limits</button><button id="t3">Tracks</button><button id="t4">Evaluation</button></nav></header>
<main id="v1" class="grid"><div class="pn q" id="ql"></div><div class="pn"><div id="flt" style="margin-bottom:6px"></div><div id="maptools" style="margin-bottom:6px;display:flex;gap:4px;flex-wrap:wrap;align-items:center"></div><div id="map" style="width:100%;height:72vh;background:var(--bg);border-radius:8px;z-index:0"></div>
<div class="mu" style="margin-top:6px">Green = AIS matched · red = dark candidate · orange = gap / mismatch / no coverage · grey = low-trust or static. Dark candidates are hidden by default — toggle them above; dense views cluster into counts (click to drill in).</div></div><div class="pn" id="card"></div></main>
<main id="v2" class="grid" style="display:none;grid-template-columns:1fr"><div class="pn sc" id="rep"></div></main>
<main id="v3" class="grid" style="display:none;grid-template-columns:1fr"><div class="pn sc" id="trk"></div></main>
<main id="v4" class="grid" style="display:none;grid-template-columns:1fr"><div class="pn sc" id="ev"></div></main>
<script>/*__LEAFLET_JS__*/</script>
<script>
const D=__DATA__;
const COL={AIS_MATCH:"#2ca02c",matched:"#2ca02c",DARK_CANDIDATE:"#d62728",AIS_MISMATCH:"#ff7f0e",POSSIBLE_AIS_GAP:"#ff7f0e",NO_AIS_COVERAGE:"#ff7f0e",NEAR_COAST_CLUTTER:"#888",SMALL_BELOW_LIMIT:"#888",STATIC_STRUCTURE:"#555",SIDELOBE_ARTIFACT:"#bbb"};
const LAT0=55.45,LON0=14.10;
const MS=D.report.map_size||[1500,1500],R=MS[0]/120;
const px=r=>r.map_xy?[r.map_xy[0]/10,r.map_xy[1]/10]:[(r.lon-LON0)*Math.cos(LAT0*Math.PI/180)*111320/10,(LAT0-r.lat)*111320/10];
let sel=null,flt="ALL",st={},showMatched=true,showDark=false;
let mapObj=null,clusterGrp=null,markerById={},lastSig=null,prevSelId=null,imgBounds=null;
try{st=JSON.parse(localStorage.getItem("bs_state")||"{}")}catch(e){}
const save=()=>{try{localStorage.setItem("bs_state",JSON.stringify(st))}catch(e){}};
const state=c=>st[c.id]||"new";
const $=id=>document.getElementById(id);
$("sum").textContent=`${D.report.detections} detections · ${D.report.matched} matched · ${D.report.unidentified} unidentified (${D.report.real?"real scene "+(D.report.scene||"").slice(0,27):"simulated scene"})`;
function filters(){const rs=["ALL",...Object.keys(D.report.reason_breakdown)];
 $("flt").innerHTML=rs.map(r=>`<button class="btn ${flt==r?"on":""}" data-r="${r}">${r=="ALL"?"All":r.replace(/_/g," ").toLowerCase()}${r=="ALL"?"":" ("+D.report.reason_breakdown[r]+")"}</button>`).join("");
 $("flt").querySelectorAll("button").forEach(b=>b.onclick=()=>{flt=b.dataset.r;draw()});}
function list(){const cs=D.contacts.filter(c=>flt=="ALL"||c.reason_code==flt);
 $("ql").innerHTML=cs.map(c=>`<div class="it ${sel&&sel.id==c.id?"sel":""} ${state(c)=="dismissed"?"dis":""}" data-id="${c.id}">
 <b>${c.id.slice(-4)}</b> <span class="chip" style="background:${COL[c.reason_code]}">${c.reason_code.replace(/_/g," ").toLowerCase()}</span>
 <div class="mu">${c.est_length_m} m · conf ${c.confidence}${c.in_sensitive_zone?" · sensitive zone":""}${c.links&&c.links.length?" · 🔗"+c.links.length:""} · ${state(c)}</div></div>`).join("")||"<span class=mu>Nothing here.</span>";
 $("ql").querySelectorAll(".it").forEach(e=>e.onclick=()=>{sel=D.contacts.find(c=>c.id==e.dataset.id);draw()});}
function visible(r){const code=r.status=="matched"?"matched":r.reason_code;
 if(flt!="ALL")return code==flt;
 if(r.status=="matched")return showMatched;
 if(code=="DARK_CANDIDATE")return showDark;
 return true;}
function mapTools(){const nd=(D.report.reason_breakdown&&D.report.reason_breakdown.DARK_CANDIDATE)||0;
 $("maptools").innerHTML=`<button class="btn ${showMatched?'on':''}" data-t="m">matched (${D.report.matched})</button>`+
 (nd?`<button class="btn ${showDark?'on':''}" data-t="d">dark candidates (${nd})</button>`:"")+
 `<button class="btn" data-t="reset">Reset view</button><span class="mu" style="font-size:12px;margin-left:4px">scroll = zoom · drag = pan · click a cluster to drill in</span>`;
 $("maptools").querySelectorAll("button").forEach(b=>b.onclick=()=>{const k=b.dataset.t;
  if(k=="m")showMatched=!showMatched;else if(k=="d")showDark=!showDark;else if(mapObj)mapObj.fitBounds(imgBounds);draw();});}
const mcolor=r=>r.status=="matched"?COL.matched:(COL[r.reason_code]||"#888");
const mstyle=(r,on)=>({radius:on?7:4,color:mcolor(r),weight:on?2:1,fillColor:mcolor(r),fillOpacity:on?.95:.4});
const ll=r=>{const [x,y]=px(r);return [MS[1]-y,x]};
function mk(r){const m=L.circleMarker(ll(r),mstyle(r,sel&&sel.id==r.id));m.__r=r;
 m.on("click",()=>{const c=D.contacts.find(c=>c.id==r.id);if(c){sel=c;draw()}});return m;}
function initMap(){const W=MS[0],H=MS[1];imgBounds=[[0,0],[H,W]];
 mapObj=L.map("map",{crs:L.CRS.Simple,minZoom:-6,maxZoom:6,zoomSnap:0,attributionControl:false});
 if(D.overview)L.imageOverlay(D.overview,imgBounds).addTo(mapObj);
 else{L.rectangle(imgBounds,{stroke:false,fillColor:"#13212b",fillOpacity:1}).addTo(mapObj);
  if(!D.report.map_size){L.rectangle([[0,0],[H,150]],{stroke:false,fillColor:"#8b7d5b",fillOpacity:.5}).addTo(mapObj).bindTooltip("land");
   L.polyline([[0,1300],[H,1300]],{color:"#0aa",dashArray:"6"}).addTo(mapObj).bindTooltip("no AIS coverage");
   L.rectangle([[0,500],[H,650]],{stroke:false,fillColor:"#d62728",fillOpacity:.07}).addTo(mapObj).bindTooltip("sensitive corridor");}}
 mapObj.fitBounds(imgBounds);mapObj.setMaxBounds(L.latLngBounds(imgBounds).pad(.4));
 clusterGrp=L.markerClusterGroup({chunkedLoading:true,maxClusterRadius:45,showCoverageOnHover:false,spiderfyOnMaxZoom:true});
 mapObj.addLayer(clusterGrp);}
function refreshMarkers(){clusterGrp.clearLayers();markerById={};const arr=[];
 D.all.forEach(r=>{if(!visible(r))return;const m=mk(r);markerById[r.id]=m;arr.push(m)});
 clusterGrp.addLayers(arr);}
function highlight(){if(prevSelId&&markerById[prevSelId])markerById[prevSelId].setStyle(mstyle(markerById[prevSelId].__r,false));
 if(sel&&markerById[sel.id]){const m=markerById[sel.id];m.setStyle(mstyle(m.__r,true));m.bringToFront()}
 prevSelId=sel?sel.id:null;}
function map(){if(!mapObj){initMap();lastSig=null}
 const sig=flt+"|"+showMatched+"|"+showDark;
 if(sig!==lastSig){lastSig=sig;refreshMarkers()}
 highlight();}
function card(){const c=sel;if(!c){$("card").innerHTML="<span class=mu>Select a contact from the queue or the map.</span>";return}
 const ev=Object.entries(c.evidence||{}).map(([k,v])=>`<tr><td class="mu">${k}</td><td>${v}</td></tr>`).join("");
 const br=Object.entries(c.confidence_breakdown).map(([k,v])=>`<div class="mu">${k.replace(/_/g," ")} ${v}</div><div class="bar"><i style="width:${v*100}%"></i></div>`).join("");
 const im=(u,t)=>u?`<img src="${u}" title="${t}" width="112" height="112" style="image-rendering:pixelated;border-radius:6px;border:1px solid var(--bd)">`:"";
 $("card").innerHTML=`<div style="display:flex;gap:10px;flex-wrap:wrap">${im(c.img,D.report.sensor?D.report.sensor+" SAR "+(D.report.pol||"")+" (dB)":"true colour")}${im(c.img_nir,"NIR (B08)")}
 <div><b>${c.id}</b><div class="mu">${c.lat}, ${c.lon}</div><div class="mu">${c.time_utc.replace("T"," ").slice(0,16)} UTC</div><div>~${c.est_length_m} m (±${c.est_length_uncertainty_m})</div></div></div>
 <p><span class="chip" style="background:${COL[c.reason_code]}">${c.reason_code.replace(/_/g," ")}</span> ${c.reason_text}</p>
 <table>${ev}<tr><td class="mu">SNR</td><td>${c.snr_db} dB</td></tr><tr><td class="mu">priority</td><td>${c.priority}</td></tr>${c.under_cirrus?'<tr><td class="mu">cirrus</td><td>under thin cirrus</td></tr>':""}</table>
 ${c.nearby_ais&&c.nearby_ais.length?`<h4 style="margin:10px 0 4px">Nearest AIS tracks</h4><table>${c.nearby_ais.map(a=>`<tr><td>${a.mmsi}</td><td>${a.dist_m} m</td><td class="mu">${Math.round(a.age_s/60)} min ago · ${a.fresh?"fresh":"stale"}${a.length_m?" · "+a.length_m+" m":""}</td></tr>`).join("")}</table>`:""}
 <h4 style="margin:10px 0 4px">Confidence ${c.confidence}</h4>${br}
 ${c.links&&c.links.length?`<h4 style="margin:10px 0 4px">Linked observations (tentative) ${c.track_id?`· <span class="mu">${c.track_id}</span>`:""}</h4><table>${c.links.map(l=>`<tr><td>${l.to_id.slice(-9)}</td><td>p ${l.p}</td><td class="mu">${l.dt_hours} h · ${l.dist_km} km · ${l.implied_speed_kn} kn · ${l.kind.replace(/_/g," ")}</td></tr>`).join("")}</table><div class="mu" style="font-size:11px">Links are physics-based candidates only; identity is never assigned.</div>`:""}
 <div>${["investigating","unresolved","dismissed"].map(a=>`<button class="btn ${state(c)==a?"on":""}" data-a="${a}">${{investigating:"Investigate",unresolved:"Keep unresolved",dismissed:"Dismiss"}[a]}</button>`).join("")}</div>
 <div class="mu" style="margin-top:6px">Identity is never assigned automatically. Cross-pass linking is a later module.</div>`;
 $("card").querySelectorAll("[data-a]").forEach(b=>b.onclick=()=>{st[c.id]=b.dataset.a;save();draw()});}
function draw(){filters();mapTools();list();map();card()}
function curveSvg(cv,title){if(!cv||!cv.length)return "";const X=l=>40+(Math.log10(l)-Math.log10(5))/(Math.log10(160)-Math.log10(5))*520,Y=p=>170-p*150;
 const pts=cv.map(c=>`${X(c.length_m)},${Y(c.p_det)}`).join(" ");
 return `<h4>${title}</h4><svg viewBox="0 0 600 200" style="width:100%;max-width:640px"><rect x="40" y="20" width="${X(20)-40}" height="150" fill="#ff7f0e" opacity=".15"/>
 <text x="46" y="36">below Sentinel limit</text><polyline points="${pts}" fill="none" stroke="var(--ac)" stroke-width="2.5"/>
 ${cv.map(c=>`<circle cx="${X(c.length_m)}" cy="${Y(c.p_det)}" r="3.5" fill="var(--ac)"/><text x="${X(c.length_m)-8}" y="188">${c.length_m}</text>`).join("")}
 <text x="4" y="24">1.0</text><text x="4" y="172">0</text></svg>`}
function repReal(){const r=D.report;const rows=r.size_bands.map(b=>`<tr><td>${b.band}</td><td>${b.ais_vessels_in_scene}</td><td>${b.ais_detected}</td><td>${b.recall_vs_ais??"–"}</td><td>${b.recall_95ci?b.recall_95ci.join(" – "):"–"}</td><td>${b.expected_detectable??"–"}</td><td>${b.ais_under_mask??"–"}</td><td>${b.ais_stale??"–"}</td></tr>`).join("");
 const inj=r.injection?`<p class="mu">Injection: ${r.detection_limit_curve.length?r.injection.tiles.length+" clear-water tiles, "+(r.sensor?"ship echo "+r.injection.echo_k+" x sea backscatter per m² of length² (":"hull contrast "+r.injection.hull_contrast_pct+"% reflectance (")+r.injection.hull_contrast_source+")":"skipped"}.</p>`:"";
 const sc=r.sensor?`<p class="mu">Scene: ${r.sensor} ${r.pol||""}, ${r.night?"<b>night pass</b>":"daylight pass"} (sun ${r.sun_elevation_deg}°) · land ${r.scene_stats.land_pct}% · usable ${r.scene_stats.usable_pct}% · pixel ${r.geometry.pixel_m} m, resolution ~${r.geometry.resolution_m} m · Doppler shift correction ${r.geometry.az_shift_applied?"on (median "+r.geometry.median_az_shift_m+" m)":"off"} · ${r.structures_in_scene??0} catalogued structures</p>`:r.scene_stats?`<p class="mu">Scene: water ${r.scene_stats.water_pct}% · cloud masked ${r.scene_stats.cloud_masked_pct}% · thin cirrus ${r.scene_stats.cirrus_pct}% (${r.scene_stats.cirrus_masked?"masked":"kept, flagged"}) · usable ${r.scene_stats.usable_pct}% · ${r.structures_in_scene??0} catalogued structures · ${r.ais_under_mask??0} AIS vessels under mask</p>`:"";
 $("rep").innerHTML=`<p class="mu">${r.note}</p>${sc}<table><tr><th>Size band</th><th>AIS vessels in scene</th><th>Detected</th><th>Recall</th><th>95% CI</th><th>Expected detectable</th><th>Under mask</th><th>Stale AIS</th></tr>${rows}</table><p class="mu">Misses below ~${r.below_limit_m??20} m are expected at this resolution. Wide intervals mean few vessels in that band. "Expected detectable" applies the injection curve to the AIS lengths in that band.</p>
 ${curveSvg(r.detection_limit_curve,"Detection probability vs length (synthetic boats in real water)")}${inj}`}
function rep(){if(D.report.real)return repReal();const r=D.report,rows=r.size_bands.map(b=>`<tr><td>${b.band}</td><td>${b.ais_vessels_in_scene}</td><td>${b.ais_detected}</td><td>${b.recall_vs_ais??"–"}</td><td>${b.expected_detectable}</td><td>${b.sim_truth_vessels}</td><td>${b.sim_truth_detected}</td></tr>`).join("");
 $("rep").innerHTML=`<p class="mu">${r.note} AIS is a biased reference: it only contains cooperative vessels.</p>
 <table><tr><th>Size band</th><th>AIS in scene</th><th>Detected</th><th>Recall</th><th>Expected detectable</th><th>Sim truth</th><th>Sim detected</th></tr>${rows}</table>
 ${curveSvg(r.detection_limit_curve,"Detection probability vs length (injection test)")}`}
function tracks(){const ts=D.report.tracks||[];if(!ts.length){$("trk").innerHTML="<p class=mu>No cross-pass tracks. Run link_contacts.py across two or more passes (or pass --history to pipeline_real.py).</p>";return}
 $("trk").innerHTML=`<p class="mu">${ts.length} track(s) spanning multiple passes. A track chains the same contact across revisits via tentative links; probabilities are capped and never assign identity.</p>`+
 ts.map(t=>`<div style="margin:10px 0;padding:8px;border:1px solid var(--bd);border-radius:8px"><b>${t.track_id}</b> <span class="mu">${t.n_observations} obs · ${(t.first_seen||"").slice(0,10)} → ${(t.last_seen||"").slice(0,10)} · max p ${t.max_link_p??"–"}</span> ${t.resolved_mmsi?`<span class="chip" style="background:#2ca02c">resolved → MMSI ${t.resolved_mmsi}</span>`:`<span class="chip" style="background:#d62728">unresolved</span>`}
 <table><tr><th>Contact</th><th>Time</th><th>Status</th><th>Reason</th><th>Length</th><th>Link p</th></tr>${t.observations.map(o=>`<tr><td>${o.id.slice(-9)}</td><td class="mu">${(o.time_utc||"").replace("T"," ").slice(0,16)}</td><td>${o.status}</td><td>${(o.reason_code||"").replace(/_/g," ").toLowerCase()}</td><td>${o.est_length_m??"–"} m</td><td>${o.link_p??"–"}</td></tr>`).join("")}</table></div>`).join("")}
const TABS={t1:"v1",t2:"v2",t3:"v3",t4:"v4"};
function evalView(){const e=D.report.evaluation;if(!e){$("ev").innerHTML="<p class=mu>No evaluation metrics. Run <code>python eval.py --ui out_eval</code> to score this scene against ground truth.</p>";return}
 const d=e.detection,l=e.linking,th=e.thresholds||{};
 const g=(v,t)=>v>=t?"#2ca02c":"#d62728",pct=v=>(v*100).toFixed(1)+"%";
 const metric=(lab,val,disp,thr)=>`<div style="flex:1;min-width:120px;padding:10px;border:1px solid var(--bd);border-radius:8px"><div class="mu" style="font-size:12px">${lab}</div><div style="font-size:22px;color:${thr==null?"var(--tx)":g(val,thr)}">${disp}</div>${thr!=null?`<div class="mu" style="font-size:11px">min ${pct(thr)}</div>`:""}</div>`;
 let h=`<div style="margin-bottom:10px;padding:8px 12px;border-radius:8px;background:${e.passed?"#1b3a1b":"#3a1b1b"};color:${e.passed?"#7fe07f":"#ff8f8f"}"><b>${e.passed?"PASS":"FAIL"}</b> — all metrics scored against known ground truth${e.failures&&e.failures.length?": "+e.failures.join("; "):""}.</div>`;
 if(d){h+=`<h3 style="margin:6px 0">Detection</h3><div style="display:flex;gap:8px;flex-wrap:wrap">
  ${metric("Recall",d.recall,pct(d.recall),th.detect_recall)}${metric("Precision",d.precision,pct(d.precision),th.detect_precision)}
  ${metric("Status accuracy",d.status_accuracy,pct(d.status_accuracy),th.status_accuracy)}${metric("Length MAE",1,d.length_mae_m+" m",null)}</div>
  <p class="mu" style="margin:8px 0">${d.tp} found · ${d.fn} missed · ${d.fp} false alarms · ${d.masked_leaks} masked-vessel leak(s) · ${d.truth_vessels} truth vessels, ${d.detections} detections.</p>
  <table><tr><th>Scenario</th><th>Detected</th><th>Status correct</th></tr>${Object.keys(d.per_mode).map(m=>`<tr><td>${m}</td><td>${d.per_mode[m].detected}</td><td>${d.per_mode[m].status_ok}</td></tr>`).join("")}</table>
  <p class="mu" style="font-size:12px;margin-top:6px">Scenarios: <b>ok</b> = vessel on AIS (expect matched) · <b>dark</b> = no AIS · <b>gap</b> = stale AIS · <b>mismatch</b> = AIS far from detection · <b>masked</b> = painted under cloud, must not be detected.</p>`;}
 if(l){h+=`<h3 style="margin:14px 0 6px">Cross-pass linking</h3><div style="display:flex;gap:8px;flex-wrap:wrap">
  ${metric("Pairwise precision",l.pair_precision,pct(l.pair_precision),null)}${metric("Pairwise recall",l.pair_recall,pct(l.pair_recall),null)}${metric("F1",l.link_f1,pct(l.link_f1),th.link_f1)}</div>
  <p class="mu" style="margin-top:8px">${l.contacts} contacts over the demo passes grouped into ${l.tracks} track(s); scored on whether each pair of contacts is correctly judged same-vessel or not.</p>`;}
 $("ev").innerHTML=h;}
function tab(id){for(const k in TABS){$(k).className=k==id?"on":"";$(TABS[k]).style.display=k==id?"grid":"none"}if(id=="t1"&&mapObj)setTimeout(()=>mapObj.invalidateSize(),0);if(id=="t2")rep();if(id=="t3")tracks();if(id=="t4")evalView()}
$("t1").onclick=()=>tab("t1");$("t2").onclick=()=>tab("t2");$("t3").onclick=()=>tab("t3");$("t4").onclick=()=>tab("t4");
if(!(D.report.tracks&&D.report.tracks.length))$("t3").style.display="none";
if(!D.report.evaluation)$("t4").style.display="none";
sel=D.contacts[0]||null;draw();
</script></body></html>"""

open(f"{OUT}/ui.html", "w").write(
    TEMPLATE.replace("/*__LEAFLET_CSS__*/", LEAFLET_CSS)
            .replace("/*__LEAFLET_JS__*/", LEAFLET_JS)
            .replace("__DATA__", data))
print(f"wrote {OUT}/ui.html")
