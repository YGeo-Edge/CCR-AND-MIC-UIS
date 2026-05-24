# MIC2 — VLM Malicious Ad Classifier + Review UI

**InternVL2.5-1B + LoRA** fine-tuned on GeoEdge ad screenshots. Classifies landing pages into 13 categories, surfaces results in a browser gallery, and lets analysts approve domains directly to the GeoEdge Blacklist and MIC labeling system with one click.

---

## Features

- **VLM classifier** — InternVL2.5-1B with a LoRA adapter trained on labeled ad screenshots. 13-class output, 95.5% validation accuracy.
- **Gallery UI** — dark-mode card gallery with per-card confidence bars, class label, MIC score, and source URLs.
- **One-click BL approval** — select domains to approve, click _Upload to Blacklist_. Generates a properly formatted CSV and submits it to `internal.geoedge.com` via your existing Chrome session.
- **Auto-labeling on see.geoedge.be** — when approving, simultaneously PUTs the classification to `see.geoedge.be/api/mic/classification`.
- **Fetch & re-run** — built-in MongoDB query panel (filter + sort) to pull fresh data from `see.geoedge.be` and re-run the model without leaving the browser.
- **Floating approval panel** — live count of selected domains, download CSV, or upload directly.

---

## Architecture

```
see.geoedge.be  ──POST /api/mic/classifications──►  gallery_server.py
                                                           │
                                                    saves mic_data.json
                                                           │
                                                    run_gallery.py
                                                     ├─ downloads S3 images
                                                     ├─ runs InternVL2.5-1B + LoRA
                                                     └─ builds gallery.html
                                                           │
                                                    Flask serves gallery
                                                           │
                                               Analyst reviews in browser
                                                           │
                                          ┌────────────────┴────────────────┐
                                   BL CSV upload                    see.geoedge.be PUT
                              internal.geoedge.com              /api/mic/classification
```

Authentication for both internal services is handled via **Playwright** using your existing Chrome session cookies — no passwords stored anywhere.

---

## Classes

| ID | Label | BL List Name |
|----|-------|-------------|
| 0 | Benign | — |
| 1 | fake_av | Fake Antivirus & Cleaners |
| 2 | financial_scam | Financial Scam |
| 3 | misleading_offers | Misleading Product Offer |
| 4 | fake_appstore | Malicious Domain |
| 5 | tech_support_scam | Tech Support Scam |
| 6 | gift_card_scan | Gift Card Scam |
| 7 | forced_notification | Forced Browser Notifications |
| 8 | suspicious_vpn | Suspicious VPN |
| 9 | malicious_extension | Malicious Extensions & Add-ons |
| 10 | fake_updates | Fake Software Update |
| 11 | blank_LP | Malicious Domain |
| 12 | fake_downloader | Malicious Domain |

---

## Setup

### Prerequisites

- Python 3.10+
- Google Chrome installed (for session-based authentication)
- Active sessions in Chrome for `internal.geoedge.com` and `see.geoedge.be`
- CUDA GPU (for EC2 training) or Apple Silicon MPS (for local inference)

### Install

```bash
cd new_mic_ui
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements_ec2.txt
pip install flask playwright requests torch torchvision
playwright install chromium
```

### Run the gallery server

```bash
python gallery_server.py           # starts on http://localhost:8765
# or specify a port:
python gallery_server.py 9000
```

The server auto-serves `gallery.html` (built by `run_gallery.py`). If no gallery exists yet, run the pipeline first.

### Run the full pipeline (fetch → classify → gallery)

**Option A — from the UI:**
Open the gallery, expand _Fetch New Data from see.geoedge.be_, adjust the MongoDB filter and sort, click _Fetch & Re-run Model_. Progress is shown live.

**Option B — from the command line:**

1. Put your input data in `mic_data.json` (array of objects with `domain`, `image_urls`, `malicious_score`, `dcptv`, `url`, `dbId`, `job_time`).
2. Run:
   ```bash
   python run_gallery.py
   ```
   This downloads images to `gallery_images/`, runs inference, and writes `gallery.html`.

---

## Approving domains to the Blacklist

1. Open `http://localhost:8765` in a browser.
2. Click **Approve for BL** on any card. The button turns green and the domain is added to the approval queue.
3. The floating panel at the bottom-right shows how many domains are queued.
4. Click **Upload to Blacklist** — this will:
   - Generate a CSV in the format `internal.geoedge.com` expects (`Domain,Status,List Name,Expiration,Approve Reason,Tags,LP url,AD url`)
   - POST it to `internal.geoedge.com` via Playwright using your Chrome session (existing_status=pending, existing_tags=ignore)
   - PUT each classification to `see.geoedge.be/api/mic/classification`

> **Authentication:** The server copies only your Chrome session cookies to a temporary directory — no singleton conflict, no Keychain prompts. Make sure you are logged into both `internal.geoedge.com` and `see.geoedge.be` in Chrome before uploading.

---

## Model

The LoRA adapter lives in `trained_models/mic2_internvl_v1/`:

| File | Description |
|------|-------------|
| `adapter_model.safetensors` | LoRA weights (~9 MB) |
| `adapter_config.json` | PEFT config |
| `classifier_head.pt` | Linear classification head |
| `model_meta.json` | Class names, base model ID |
| `eval_results.json` | Validation metrics |
| `classification_report.txt` | Per-class precision/recall/F1 |

Base model: `OpenGVLab/InternVL2_5-1B` (downloaded from HuggingFace on first run).

### Training

```bash
python train_cli.py --config configs/mic2_internvl.yaml
```

See `configs/mic2_internvl.yaml` for all hyperparameters. Training was done on an EC2 `g5.xlarge` (A10G GPU).

---

## API Endpoints (Flask server)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serve `gallery.html` |
| `GET` | `/gallery_images/<file>` | Serve a gallery image |
| `POST` | `/api/upload_bl` | Upload CSV body to BL via Playwright |
| `POST` | `/api/label_mic` | PUT labels to see.geoedge.be via Playwright |
| `POST` | `/api/fetch_and_run` | Start background fetch+rerun job |
| `GET` | `/api/fetch_status` | Poll background job progress |

---

## File Overview

| File | Purpose |
|------|---------|
| `gallery_server.py` | Flask server — authentication proxy + static serving |
| `run_gallery.py` | Pipeline: download images → classify → build HTML |
| `model.py` | `InternVLClassifier` — loads base model + LoRA adapter |
| `dataset.py` | Dataset class for training/eval |
| `train.py` / `train_cli.py` | Training loop and CLI entry point |
| `app_inference.py` | Batch inference script |
| `test_inference.py` | Quick single-image inference test |
| `configs/mic2_internvl.yaml` | Training config |
| `mic2_api/` | Standalone FastAPI inference service (Docker) |
| `scan_nn_mlcs.py` | Nearest-neighbour MLC scan utility |
