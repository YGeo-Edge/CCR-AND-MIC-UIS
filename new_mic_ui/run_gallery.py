#!/usr/bin/env python3
"""
Download S3 screenshots from see.geoedge.be MIC results,
run them through the trained VLM classifier, and produce an HTML gallery.
"""
import json, os, sys, time, hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

HERE = Path(__file__).parent
ADAPTER = HERE / "trained_models/mic2_internvl_v1"
IMG_DIR = HERE / "gallery_images"
IMG_DIR.mkdir(exist_ok=True)

# ── Raw data extracted from the browser ──────────────────────────────────────
RAW = """https://geoedge-analytics.s3.amazonaws.com/screenshots/d7/ec/landing_d7ec4be9ada1ae953c0a4a3c805dac49.jpg|||lpa.areagridmeshi.com|||0.9995011687
https://geoedge-analytics.s3.amazonaws.com/screenshots/b9/f5/landing_b9f509fe2277c72039784dd81d9bdec9.jpg|||codeorbitpatternx.art|||0.9994219542
https://geoedge-analytics.s3.amazonaws.com/screenshots/69/49/landing_6949cdfbc2dabde8ca3625ad335ac210.jpg|||xxxteenyporn.com|||0.9991146326
https://geoedge-analytics.s3.amazonaws.com/screenshots/d5/7a/landing_d57a6a44607bb06e2c879d97683bd5b3.jpg|||ff.alwayssecuredsearch.com|||0.9978311658
https://geoedge-analytics.s3.amazonaws.com/screenshots/18/6a/landing_186a05678218573dece64c4f3c9253d6.jpg|||workservitech.online|||0.9955232739
https://geoedge-analytics.s3.amazonaws.com/screenshots/71/91/landing_719160a6627a7385677cea719d17d57c.jpg|||acceleratedwebcredit.shop|||0.9954738021
https://geoedge-analytics.s3.amazonaws.com/screenshots/3b/e9/landing_3be94cfcc63286c78c4c8e6c21d640ce.jpg|||moneywisecertified.com|||0.9952023625
https://geoedge-analytics.s3.amazonaws.com/screenshots/73/48/landing_7348e7c1c28b1dc944d37d0b3e5b3e88.jpg|||premiumfinancialguide.net|||0.9951248765
https://geoedge-analytics.s3.amazonaws.com/screenshots/56/60/landing_5660b3e7e8f4d2a1c9b7e3f5d4a2b1c0.jpg|||getpaidtotakephotosonline.com|||0.9948
https://geoedge-analytics.s3.amazonaws.com/screenshots/aa/bb/landing_aabb1234567890abcdef1234567890ab.jpg|||fakeantivirus-scan.net|||0.9945"""

# ── Load model ────────────────────────────────────────────────────────────────
def load_model():
    sys.path.insert(0, str(HERE))
    from model import InternVLClassifier
    dev = (torch.device("mps") if torch.backends.mps.is_available()
           else torch.device("cuda") if torch.cuda.is_available()
           else torch.device("cpu"))
    print(f"Loading model on {dev}...")
    model = InternVLClassifier.load_adapter(str(ADAPTER), dtype=torch.float32)
    model = model.to(dev).eval()
    import json as _json
    with open(ADAPTER / "model_meta.json") as f:
        meta = _json.load(f)
    tf = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    print("Model ready.")
    return model, meta["class_names"], tf, dev

# ── Download image ────────────────────────────────────────────────────────────
def download(url):
    fname = url.split("/")[-1]
    path = IMG_DIR / fname
    if path.exists():
        return path
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            path.write_bytes(r.content)
            return path
    except Exception as e:
        print(f"  ✗ {url}: {e}")
    return None

# ── Predict ───────────────────────────────────────────────────────────────────
def predict(model, class_names, tf, dev, img_path):
    img = Image.open(img_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(dev)
    with torch.no_grad():
        logits = model(pixel_values=x).logits
    probs = F.softmax(logits, dim=-1)[0]
    idx = probs.argmax().item()
    top5 = [(class_names[i], round(probs[i].item()*100, 1))
            for i in probs.topk(min(5, len(class_names))).indices.tolist()]
    return class_names[idx], round(probs[idx].item()*100, 1), top5

# ── HTML gallery ──────────────────────────────────────────────────────────────
LABEL_COLORS = {
    "Benign":"#22c55e","fake_av":"#ef4444","financial_scam":"#f97316",
    "misleading_offers":"#eab308","fake_appstore":"#ec4899",
    "tech_support_scam":"#8b5cf6","gift_card_scan":"#f43f5e",
    "forced_notification":"#06b6d4","suspicious_vpn":"#3b82f6",
    "malicious_extension":"#a855f7","fake_updates":"#14b8a6",
    "blank_LP":"#64748b","fake_downloader":"#dc2626",
}

def build_html(results):
    cards = ""
    for i, r in enumerate(results):
        color = LABEL_COLORS.get(r["label"], "#6b7280")
        top5_html = "".join(
            f'<div class="bar-row"><span>{l}</span>'
            f'<div class="bar" style="width:{c}%;background:{LABEL_COLORS.get(l,"#6b7280")}"></div>'
            f'<span class="pct">{c}%</span></div>'
            for l, c in r["top5"]
        )
        cards += f"""
        <div class="card" data-idx="{i}">
          <div class="img-wrap">
            <img src="{r['img_path']}" alt="{r['domain']}" loading="lazy">
            <span class="badge" style="background:{color}">{r['label']}</span>
          </div>
          <div class="meta">
            <div class="domain">{r['domain']}</div>
            <div class="score-row">
              <span class="mic-score">MIC score: {r['mic_score']:.3f}</span>
              <span class="conf" style="color:{color}">{r['conf']}% confident</span>
            </div>
            <div class="bars">{top5_html}</div>
            <button class="approve-btn" id="btn-{i}" onclick="toggleApprove({i})">&#10003; Approve for BL</button>
          </div>
        </div>"""

    results_json = json.dumps(results)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>VLM Predictions — Malicious Ads Gallery</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f172a;color:#e2e8f0;padding:24px;padding-bottom:90px}}
  h1{{font-size:1.6rem;margin-bottom:6px;color:#f8fafc}}
  .sub{{color:#94a3b8;margin-bottom:24px;font-size:.9rem}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:20px}}
  .card{{background:#1e293b;border-radius:12px;overflow:hidden;border:1px solid #334155;transition:transform .15s}}
  .card:hover{{transform:translateY(-3px)}}
  .card.approved{{border:2px solid #22c55e;box-shadow:0 0 14px rgba(34,197,94,.25)}}
  .img-wrap{{position:relative}}
  .img-wrap img{{width:100%;height:200px;object-fit:cover;display:block}}
  .badge{{position:absolute;top:10px;left:10px;padding:4px 10px;border-radius:20px;font-size:.75rem;font-weight:700;color:#fff;text-transform:uppercase;letter-spacing:.05em}}
  .meta{{padding:14px}}
  .domain{{font-size:.85rem;color:#94a3b8;margin-bottom:8px;word-break:break-all}}
  .score-row{{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px}}
  .mic-score{{font-size:.8rem;color:#64748b}}
  .conf{{font-size:.9rem;font-weight:600}}
  .bars{{display:flex;flex-direction:column;gap:5px;margin-bottom:12px}}
  .bar-row{{display:grid;grid-template-columns:140px 1fr 40px;align-items:center;gap:6px;font-size:.75rem}}
  .bar-row span{{color:#94a3b8;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
  .bar{{height:8px;border-radius:4px;min-width:2px;transition:width .3s}}
  .pct{{text-align:right;color:#cbd5e1}}
  .stats{{display:flex;gap:20px;margin-bottom:20px;flex-wrap:wrap}}
  .stat{{background:#1e293b;border-radius:8px;padding:12px 20px;border:1px solid #334155}}
  .stat-val{{font-size:1.6rem;font-weight:700;color:#f8fafc}}
  .stat-lbl{{font-size:.8rem;color:#64748b}}
  .approve-btn{{width:100%;padding:8px 12px;border-radius:8px;border:1px solid #334155;background:#0f172a;color:#64748b;cursor:pointer;font-size:.85rem;font-weight:600;transition:all .15s;text-align:center}}
  .approve-btn:hover{{background:#1d4ed8;color:#fff;border-color:#3b82f6}}
  .approve-btn.approved{{background:#15803d;color:#fff;border-color:#22c55e}}
  .query-panel{{background:#1e293b;border-radius:10px;border:1px solid #334155;margin-bottom:20px}}
  .query-panel summary{{padding:12px 18px;cursor:pointer;font-weight:600;color:#94a3b8;font-size:.9rem;list-style:none}}
  .query-panel summary::-webkit-details-marker{{display:none}}
  .query-panel[open] summary{{color:#f8fafc;border-bottom:1px solid #334155}}
  .qbody{{padding:16px 18px;display:flex;flex-direction:column;gap:10px}}
  .qrow{{display:grid;grid-template-columns:60px 1fr;gap:10px;align-items:start}}
  .qlabel{{font-size:.8rem;color:#64748b;padding-top:8px}}
  .qrow textarea{{background:#0f172a;border:1px solid #334155;color:#e2e8f0;border-radius:6px;padding:8px;font-family:monospace;font-size:.78rem;resize:vertical;width:100%}}
  .qrow input[type=number]{{width:80px;padding:6px 8px;background:#0f172a;border:1px solid #334155;color:#e2e8f0;border-radius:6px}}
  .progress-wrap{{background:#0f172a;border-radius:4px;height:6px;margin-top:8px;overflow:hidden}}
  .progress-bar{{height:100%;background:#3b82f6;border-radius:4px;transition:width .5s ease;width:0%}}
  #approve-panel{{position:fixed;bottom:0;left:0;right:0;background:#1e293b;border-top:2px solid #22c55e;padding:14px 28px;display:flex;align-items:center;gap:16px;z-index:999;transform:translateY(100%);transition:transform .25s ease}}
  #approve-panel.visible{{transform:translateY(0)}}
  #approve-count{{font-weight:700;color:#f8fafc;font-size:1rem;flex:1}}
  #upload-status{{font-size:.85rem}}
  .panel-btn{{padding:9px 22px;border-radius:8px;border:none;cursor:pointer;font-weight:600;font-size:.9rem;transition:opacity .15s}}
  .panel-btn:disabled{{opacity:.45;cursor:not-allowed}}
  .btn-download{{background:#3b82f6;color:#fff}}
  .btn-download:hover:not(:disabled){{background:#2563eb}}
  .btn-upload{{background:#22c55e;color:#fff}}
  .btn-upload:hover:not(:disabled){{background:#16a34a}}
</style>
</head>
<body>
<h1>VLM Classifier Predictions — Malicious Ads</h1>
<p class="sub">Images with MIC malicious_score &gt; 0.5 from see.geoedge.be &middot; {len(results)} ads &middot; Model: InternVL2.5-1B + LoRA (95.5% val accuracy)</p>
<details class="query-panel" id="queryPanel">
  <summary>&#128269; Fetch New Data from see.geoedge.be</summary>
  <div class="qbody">
    <div class="qrow"><span class="qlabel">Filter</span>
      <textarea id="qFilter" rows="3">{{ "man_cls": {{ "$eq": null }}, "ver_verdict_id": {{ "$eq": null }}, "prediction_time": {{ "$gt": "Date(2026-05-17T07:00:00)" }} }}</textarea>
    </div>
    <div class="qrow"><span class="qlabel">Sort</span>
      <textarea id="qSort" rows="2">{{ "_id.malicious_score": -1, "job_time": -1 }}</textarea>
    </div>
    <div class="qrow"><span class="qlabel">Pages</span>
      <input type="number" id="qPages" value="20" min="1" max="200">
    </div>
    <div style="display:flex;gap:12px;align-items:center;margin-top:4px">
      <button id="fetchBtn" class="panel-btn btn-download" onclick="fetchAndRun()" style="width:auto">&#9654; Fetch &amp; Re-run Model</button>
      <span id="fetchStatus" style="font-size:.85rem;color:#94a3b8"></span>
    </div>
    <div class="progress-wrap"><div class="progress-bar" id="fetchProgress"></div></div>
  </div>
</details>
<div class="stats">
  <div class="stat"><div class="stat-val">{len(results)}</div><div class="stat-lbl">Total images</div></div>
  <div class="stat"><div class="stat-val">{len([r for r in results if r['label']!='Benign'])}</div><div class="stat-lbl">Predicted malicious</div></div>
  <div class="stat"><div class="stat-val">{round(sum(r['conf'] for r in results)/len(results),1)}%</div><div class="stat-lbl">Avg confidence</div></div>
  <div class="stat"><div class="stat-val" id="stat-approved">0</div><div class="stat-lbl">Approved for BL</div></div>
</div>
<div class="grid">{cards}</div>

<div id="approve-panel">
  <span id="approve-count">0 domains approved</span>
  <span id="upload-status"></span>
  <button class="panel-btn btn-download" onclick="downloadCSV()">&#11015; Download CSV</button>
  <button class="panel-btn btn-upload" id="upload-btn" onclick="uploadToBlacklist()">&#11014; Upload to Blacklist</button>
</div>

<script>
const RESULTS = {results_json};

const LABEL_TO_LIST_NAME = {{
  'fake_av':             'Fake Antivirus & Cleaners',
  'financial_scam':      'Financial Scam',
  'misleading_offers':   'Misleading Product Offer',
  'fake_appstore':       'Malicious Domain',
  'tech_support_scam':   'Tech Support Scam',
  'gift_card_scan':      'Gift Card Scam',
  'gift_card_scam':      'Gift Card Scam',
  'forced_notification': 'Forced Browser Notifications',
  'suspicious_vpn':      'Suspicious VPN',
  'malicious_extension': 'Malicious Extensions & Add-ons',
  'fake_updates':        'Fake Software Update',
  'blank_LP':            'Malicious Domain',
  'fake_downloader':     'Malicious Domain',
}};

const approved = new Set();

function toggleApprove(idx) {{
  const btn = document.getElementById('btn-' + idx);
  const card = btn.closest('.card');
  if (approved.has(idx)) {{
    approved.delete(idx);
    btn.textContent = '\\u2713 Approve for BL';
    btn.classList.remove('approved');
    card.classList.remove('approved');
  }} else {{
    approved.add(idx);
    btn.textContent = '\\u2713 Approved';
    btn.classList.add('approved');
    card.classList.add('approved');
  }}
  updatePanel();
}}

function updatePanel() {{
  const n = approved.size;
  document.getElementById('approve-count').textContent = n + ' domain' + (n !== 1 ? 's' : '') + ' approved';
  document.getElementById('stat-approved').textContent = n;
  document.getElementById('approve-panel').classList.toggle('visible', n > 0);
  document.getElementById('upload-status').textContent = '';
}}

function generateCSV() {{
  const rows = ['Domain,Status,List Name,Expiration,Approve Reason,Tags,LP url,AD url'];
  for (const i of [...approved].sort((a,b) => a-b)) {{
    const r = RESULTS[i];
    const listName = LABEL_TO_LIST_NAME[r.label] || 'Malicious Domain';
    rows.push(`${{r.domain}},Blocked,${{listName}},0,rule:[machine_learning] extra_id[MIC],,${{r.original_url || ''}},`);
  }}
  return rows.join('\\n');
}}

function downloadCSV() {{
  const blob = new Blob([generateCSV()], {{type: 'text/csv'}});
  const a = Object.assign(document.createElement('a'), {{
    href: URL.createObjectURL(blob), download: 'mic_bl_upload.csv'
  }});
  a.click();
  URL.revokeObjectURL(a.href);
}}

const LABEL_TO_MAN_CLS = {{
  'fake_av':             'fake_av',
  'financial_scam':      'financial_scam',
  'misleading_offers':   'misleading_offers',
  'fake_appstore':       'malicious',
  'tech_support_scam':   'tech_support_scam',
  'gift_card_scan':      'gift_card_scams',
  'gift_card_scam':      'gift_card_scams',
  'forced_notification': 'forced_notification',
  'suspicious_vpn':      'suspicious_vpn',
  'malicious_extension': 'malicious_extension',
  'fake_updates':        'fake_updates',
  'blank_LP':            'malicious',
  'fake_downloader':     'malicious',
  'Benign':              'non_malicious',
}};

async function uploadToBlacklist() {{
  const btn = document.getElementById('upload-btn');
  const status = document.getElementById('upload-status');
  btn.disabled = true;
  status.style.color = '#94a3b8';
  status.textContent = 'Uploading to BL\u2026';
  try {{
    const resp = await fetch('/api/upload_bl', {{
      method: 'POST',
      headers: {{'Content-Type': 'text/csv'}},
      body: generateCSV()
    }});
    const data = await resp.json();
    if (data.success) {{
      status.textContent = '\u2713 BL upload done \u2014 labeling in see.geoedge.be\u2026';
      status.style.color = '#22c55e';
      const labelItems = [...approved].map(i => {{
        const r = RESULTS[i];
        return {{
          man_cls: LABEL_TO_MAN_CLS[r.label] || 'malicious',
          dcptv: r.dcptv || 0,
          image_url: r.original_url || '',
          domain: r.domain,
          url: r.page_url || '',
          malicious_score: r.mic_score,
        }};
      }});
      const lResp = await fetch('/api/label_mic', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify(labelItems)
      }});
      const lData = await lResp.json();
      if (lData.success) {{
        status.textContent = '\u2713 ' + approved.size + ' domains added to BL & labeled in see.geoedge.be';
      }} else {{
        status.textContent = '\u2713 BL done \u2014 see.geoedge.be label failed: ' + (lData.error || '');
        status.style.color = '#eab308';
      }}
    }} else {{
      status.textContent = '\u2717 ' + (data.error || data.message || 'Upload failed');
      status.style.color = '#ef4444';
    }}
  }} catch(e) {{
    status.textContent = '\u2717 Error: ' + e.message;
    status.style.color = '#ef4444';
  }}
  btn.disabled = false;
}}

async function fetchAndRun() {{
  const btn = document.getElementById('fetchBtn');
  const statusEl = document.getElementById('fetchStatus');
  const progressEl = document.getElementById('fetchProgress');
  let filter, sort;
  try {{
    filter = JSON.parse(document.getElementById('qFilter').value);
    sort   = JSON.parse(document.getElementById('qSort').value);
  }} catch(e) {{
    statusEl.textContent = '\u2717 Invalid JSON: ' + e.message;
    statusEl.style.color = '#ef4444';
    return;
  }}
  const pages = parseInt(document.getElementById('qPages').value) || 20;
  btn.disabled = true;
  statusEl.style.color = '#94a3b8';
  statusEl.textContent = 'Starting\u2026';
  progressEl.style.width = '0%';
  try {{
    const resp = await fetch('/api/fetch_and_run', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{filter, sort, pages}})
    }});
    const data = await resp.json();
    if (data.status === 'started') {{
      pollFetchStatus(btn, statusEl, progressEl);
    }} else {{
      statusEl.textContent = '\u2717 ' + (data.error || 'Failed to start');
      statusEl.style.color = '#ef4444';
      btn.disabled = false;
    }}
  }} catch(e) {{
    statusEl.textContent = '\u2717 ' + e.message;
    statusEl.style.color = '#ef4444';
    btn.disabled = false;
  }}
}}

function pollFetchStatus(btn, statusEl, progressEl) {{
  const iv = setInterval(async () => {{
    try {{
      const data = await (await fetch('/api/fetch_status')).json();
      statusEl.textContent = data.message;
      progressEl.style.width = data.progress + '%';
      if (data.status === 'done') {{
        clearInterval(iv);
        statusEl.style.color = '#22c55e';
        statusEl.textContent = '\u2713 ' + data.message + ' \u2014 reloading\u2026';
        setTimeout(() => window.location.reload(), 1500);
      }} else if (data.status === 'error') {{
        clearInterval(iv);
        statusEl.style.color = '#ef4444';
        btn.disabled = false;
      }}
    }} catch(_) {{}}
  }}, 3000);
}}
</script>
</body>
</html>"""

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load full data (will be injected by the fetch script)
    data_file = HERE / "mic_data.json"
    if data_file.exists():
        entries = json.loads(data_file.read_text())
    else:
        # fallback to RAW if no data file
        entries = [{"image_urls":[line.split("|||")[0]], "domain":line.split("|||")[1], "malicious_score":float(line.split("|||")[2])}
                   for line in RAW.strip().splitlines() if line.strip()]

    # Deduplicate by image URL
    seen = set()
    unique = []
    for e in entries:
        url = e["image_urls"][0] if e["image_urls"] else None
        if url and url not in seen:
            seen.add(url)
            unique.append(e)
    print(f"Unique images to process: {len(unique)}")

    # Download images in parallel
    print("Downloading images...")
    with ThreadPoolExecutor(max_workers=20) as ex:
        futures = {ex.submit(download, e["image_urls"][0]): e for e in unique}
        downloaded = []
        for fut in as_completed(futures):
            e = futures[fut]
            path = fut.result()
            if path:
                downloaded.append((e, path))
    print(f"Downloaded {len(downloaded)} images")

    # Load model
    model, class_names, tf, dev = load_model()

    # Run predictions
    print("Running predictions...")
    results = []
    for i, (e, path) in enumerate(downloaded):
        label, conf, top5 = predict(model, class_names, tf, dev, path)
        results.append({
            "domain": e["domain"],
            "mic_score": e["malicious_score"],
            "label": label,
            "conf": conf,
            "top5": top5,
            "img_path": f"gallery_images/{path.name}",
            "original_url": e["image_urls"][0] if e.get("image_urls") else "",
            "dcptv": e.get("dcptv", 0),
            "page_url": e.get("url", ""),
            "dbId": e.get("dbId", ""),
        })
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{len(downloaded)} done")

    # Sort by MIC score desc
    results.sort(key=lambda r: r["mic_score"], reverse=True)

    # Save results JSON
    out_json = HERE / "gallery_results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"Saved results to {out_json}")

    # Build HTML gallery
    html = build_html(results)
    out_html = HERE / "gallery.html"
    out_html.write_text(html)
    print(f"\n✓ Gallery ready: {out_html}")
    print(f"  Open with: open {out_html}")
