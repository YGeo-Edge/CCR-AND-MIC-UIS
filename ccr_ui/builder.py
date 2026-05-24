import json
import re

INTERNAL = "https://internal.geoedge.com"
MAX_THUMBS = 20

TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CCR Gallery</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:Arial,sans-serif;background:#111;color:#eee;padding:10px}
h1{font-size:16px;margin-bottom:6px}
.stats{font-size:11px;color:#888;margin-bottom:10px}
.controls{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px;align-items:center;position:sticky;top:0;background:#111;z-index:100;padding:6px 0;border-bottom:1px solid #2a2a2a}
#search{flex:1;min-width:180px;padding:5px 8px;background:#222;border:1px solid #444;color:#eee;border-radius:4px;font-size:13px}
.fb{padding:4px 9px;background:#2a2a2a;border:1px solid #444;color:#ccc;border-radius:4px;cursor:pointer;font-size:11px}
.fb.on{background:#0047cc;border-color:#0047cc;color:#fff}
#cd{font-size:11px;color:#888;white-space:nowrap}
.db{border:1px solid #2a2a2a;border-radius:5px;margin-bottom:8px;padding:8px;background:#181818}
.db:hover{border-color:#444}
.db.hi{display:none}
.dh{display:flex;flex-wrap:wrap;align-items:baseline;gap:5px;margin-bottom:6px}
.dt{font-size:13px;font-weight:700;color:#6af;word-break:break-all}
.bg{display:flex;flex-wrap:wrap;gap:3px;align-items:center}
.vc{padding:1px 5px;border-radius:3px;font-size:9px;font-weight:700}
.vc.cf{background:#c8600022;color:#f90;border:1px solid #f904}
.vc.tm{background:#006bc822;color:#6bf;border:1px solid #6bf4}
.bl{padding:1px 5px;border-radius:3px;font-size:9px;font-weight:700;background:#c8000022;color:#f55;border:1px solid #f554}
.qt{font-size:9px;color:#555;font-style:italic}
.cnt{font-size:10px;color:#777}
.more{font-size:9px;color:#f83;margin-left:3px}
.thumbs{display:flex;flex-wrap:wrap;gap:5px}
.tc{width:108px;background:#202020;border-radius:4px;overflow:hidden;border:1px solid #2a2a2a}
.tc img{width:108px;height:72px;object-fit:cover;display:block}
.tm{padding:2px 3px;font-size:8px}
.th{display:block;color:#9cf;word-break:break-all;font-weight:600;margin-bottom:1px;overflow:hidden;max-height:2em}
.tl{display:flex;gap:3px;margin-top:1px}
.tl a{color:#7af;font-size:8px;text-decoration:none}
.ns{font-size:10px;color:#444;padding:4px 0}
</style>
</head>
<body>
<h1>CCR Gallery</h1>
<div class="stats" id="stats"></div>
<div class="controls">
  <input id="search" type="text" placeholder="Search domain..." autofocus>
  <button class="fb on" data-f="all">All</button>
  <button class="fb" data-f="bl">BL</button>
  <button class="fb" data-f="cf">Confiant</button>
  <button class="fb" data-f="tm">TMT</button>
  <button class="fb" data-f="yd">Has data</button>
  <button class="fb" data-f="nd">No data</button>
  <span id="cd"></span>
</div>
<div id="gallery"></div>
<script>
const ROWS=__DATA__;
const INT='https://internal.geoedge.com';
function thumbUrl(h){return`https://geoedge-analytics.s3.amazonaws.com/screenshots/${h.slice(0,2)}/${h.slice(2,4)}/landingthumb_${h}.jpg`}
function jobUrl(id){return id?`${INT}/admin_geinternalpage/analytics/snapshots_job/${id}`:'#'}
function adsUrl(id){return id?`${INT}/admin_geinternalpage/analytics/snapshots_ads?req_rpt_period=all&search_type=ji&search_str=${id}`:'#'}

const gallery=document.getElementById('gallery');
const blocks=[];

ROWS.forEach(([display,query,vendor,bl,thumbs])=>{
  const vc=vendor==='confiant'?'cf':'tm';
  const blBadge=bl?'<span class="bl">BL</span>':'';
  const extraStr=thumbs.length===20?` <span class="more">+more</span>`:'';
  const cnt=`<span class="cnt">(${thumbs.length} shots${extraStr})</span>`;
  const thumbHtml=thumbs.length?thumbs.map(([h,jid,lp])=>{
    const ju=jobUrl(jid),au=adsUrl(jid),src=thumbUrl(h);
    return`<div class="tc"><a href="${ju}" target="_blank"><img loading="lazy" src="${src}" alt="${lp}"></a><div class="tm"><span class="th">${lp}</span><div class="tl"><a href="${ju}" target="_blank">job</a><a href="${au}" target="_blank">ads</a></div></div></div>`;
  }).join(''):'<div class="ns">No screenshots found</div>';

  const div=document.createElement('div');
  div.className='db';
  div.dataset.d=display.toLowerCase();
  div.dataset.q=query.toLowerCase();
  div.dataset.v=vendor;
  div.dataset.bl=bl;
  div.dataset.hd=thumbs.length>0?'1':'0';
  div.innerHTML=`<div class="dh"><span class="dt">${display}</span><div class="bg"><span class="vc ${vc}">${vendor}</span>${blBadge}<span class="qt">${query}</span>${cnt}</div></div><div class="thumbs">${thumbHtml}</div>`;
  gallery.appendChild(div);
  blocks.push(div);
});

const totThumb=ROWS.reduce((s,r)=>s+r[4].length,0);
document.getElementById('stats').textContent=`${ROWS.length} domains • ${totThumb} screenshots shown`;

const searchEl=document.getElementById('search'),cdEl=document.getElementById('cd');
let af='all';
function upd(){
  const q=searchEl.value.trim().toLowerCase();
  let vis=0;
  blocks.forEach(b=>{
    let s=true;
    if(q&&!b.dataset.d.includes(q)&&!b.dataset.q.includes(q))s=false;
    if(af==='bl'&&b.dataset.bl!='1')s=false;
    if(af==='cf'&&b.dataset.v!=='confiant')s=false;
    if(af==='tm'&&b.dataset.v!=='TMT')s=false;
    if(af==='yd'&&b.dataset.hd!='1')s=false;
    if(af==='nd'&&b.dataset.hd!='0')s=false;
    b.classList.toggle('hi',!s);
    if(s)vis++;
  });
  cdEl.textContent=vis+'/'+blocks.length+' shown';
}
searchEl.addEventListener('input',upd);
document.querySelectorAll('.fb').forEach(btn=>{
  btn.addEventListener('click',()=>{
    document.querySelectorAll('.fb').forEach(b=>b.classList.remove('on'));
    btn.classList.add('on');af=btn.dataset.f;upd();
  });
});
upd();
</script>
</body>
</html>'''


def build_gallery(rows, screenshot_data):
    compact = []
    for row in rows:
        display = row["display"]
        query = row["query"]
        vendor = row.get("vendor", "")
        bl = 1 if row.get("should_bl") else 0
        items = screenshot_data.get(display, [])[:MAX_THUMBS]
        thumbs = []
        for it in items:
            m = re.search(r"landingthumb_([0-9a-f]{32})\.jpg", it.get("thumb", ""))
            if not m:
                continue
            h = m.group(1)
            job_id_m = re.search(r"/(\d+)$", it.get("jobHref", ""))
            job_id = job_id_m.group(1) if job_id_m else ""
            lp = it.get("lpHost", "")
            thumbs.append([h, job_id, lp])
        compact.append([display, query, vendor, bl, thumbs])

    data_json = json.dumps(compact, separators=(",", ":"))
    return TEMPLATE.replace("__DATA__", data_json)
