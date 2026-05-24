# CCR Gallery — Claude Instructions

## What this project does
Generates a self-contained HTML screenshot gallery from a CSV of suspicious domains,
pulling screenshot data from GeoEdge's internal tool (internal.geoedge.com).

## How to run it end-to-end

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
python -m playwright install chromium
```

### Step 2 — Set credentials as environment variables
```bash
export GEOEDGE_USER='yarden.tamam'   # Microsoft SSO username for internal.geoedge.com
export GEOEDGE_PASS='your_password'
```
To persist across sessions, add to `~/.zshrc` and run `source ~/.zshrc`.

### Step 3 — Run
```bash
python main.py input.csv
```

## Authentication details
- The site uses **Microsoft SSO** (Azure AD). There is no form-based login.
- `auth.py` uses **Playwright** to automate the browser-based SSO flow.
- Session is saved to `~/.geoedge_session` (persistent Chromium profile).
- **First run**: opens a visible Chromium window, enters GEOEDGE_USER/PASS via Microsoft login, saves session.
- **Subsequent runs**: reuses saved session silently (headless, no window).
- If session expires (~days depending on org policy), visible browser opens once to re-authenticate.

## CSV column mapping
- `host` → `display` (required)
- `tld` → `query` (optional, auto-computed from display if missing)
- `vendor` → badge color (confiant=orange, TMT=blue)
- `should_bl` → red BL badge

## Search type logic
- `display == query` → `search_type=top_domain_in_requests`
- `display != query` → `search_type=host`

## Files
- `main.py` — CLI entrypoint, CSV parsing, orchestration
- `auth.py` — Playwright SSO login, session caching
- `scraper.py` — fetches screenshot data from GeoEdge API
- `builder.py` — renders the HTML gallery from scraped data
