# CCR Gallery

Turns a CSV of suspicious domains into an HTML screenshot gallery using GeoEdge data.

---

## First-time setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Requires Google Chrome** to be installed on your machine.

---

### 2. Save your GeoEdge credentials

Open your terminal and run these two commands (replace with your actual credentials from the password manager — look for `internal.geoedge.com`):

```bash
echo "export GEOEDGE_USER='your.name'" >> ~/.zshrc
echo "export GEOEDGE_PASS='your_password'" >> ~/.zshrc
source ~/.zshrc
```

**You only need to do this once.** The credentials will be available every time you open a terminal.

---

## Running the tool

```bash
python main.py your_file.csv
```

**First time only:** A Chrome window will open and log you in to GeoEdge automatically. If a Microsoft login screen appears, it will fill in your credentials. If MFA is required, approve it on your phone. Once logged in, the session is saved — future runs happen silently with no browser window.

The gallery file (`ccr_gallery.html`) will open in your browser automatically when done.

---

## CSV format

Your CSV needs at least one column called `host` (or `display` / `domain`) with the domain names.

**Minimal example:**
```csv
host
cleardriftessence.com
paymentsucessfullyapprovedsystems.netlify.app
```

**Full example (all optional columns):**
```csv
host,tld,vendor,should_bl
cleardriftessence.com,cleardriftessence.com,confiant,true
paymentsucessfullyapprovedsystems.netlify.app,netlify.app,confiant,false
buyretailelite.z13.web.core.windows.net,z13.web.core.windows.net,TMT,true
```

| Column | Required | Description |
|--------|----------|-------------|
| `host` (or `display`, `domain`) | ✅ | Full hostname |
| `tld` (or `query`) | optional | Parent domain — auto-computed if missing |
| `vendor` | optional | `confiant` or `TMT` — shows as a colored badge |
| `should_bl` | optional | `true`/`false` — shows as a red BL badge |

---

## Options

```
python main.py input.csv --out my_gallery.html   # save to a custom filename
python main.py input.csv --no-open               # don't auto-open the browser
```
