import re
import sys
import time
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

INTERNAL = "https://internal.geoedge.com"
BASE_PATH = "/admin_geinternalpage/analytics/snapshots_jobs"
COMMON = (
    "req_rpt_period=last30days&job_status=all&no_ads=all&scan_type=-1"
    "&code_type=-1&is_manual=&location=0&emulation_category=-1&location_via=all"
    "&malware_type=0&is_sound=&is_fake=&event_type=-1&is_screenshot=&security_rule="
    "&security_rule_extra_id=0&preview=landing"
)
SUFFIX = (
    "&group=landing_title&rows_limit=500&rows_order="
    "&output_fields%5B%5D=in&output_fields%5B%5D=lu&output_fields%5B%5D=lh&submit=Search"
)
MAX_THUMBS = 20


def _url(query, search_type):
    return (
        f"{INTERNAL}{BASE_PATH}?{COMMON}"
        f"&search_type%5B%5D={search_type}&search_q%5B%5D={quote(query)}"
        f"{SUFFIX}"
    )


def _parse(html):
    soup = BeautifulSoup(html, "html.parser")
    items = []
    for row in soup.select("#tblRows tbody tr"):
        img = row.select_one("label.lp-preview img")
        if not img or not img.get("src") or "s3.amazonaws.com" not in img["src"]:
            continue
        cells = row.find_all("td")
        links = row.select('a[href*="snapshots_job"]')
        m = re.search(r"landingthumb_([0-9a-f]{32})\.jpg", img["src"])
        if not m:
            continue
        items.append({
            "thumb": img["src"],
            "lpHost": cells[10].get_text(strip=True) if len(cells) > 10 else "",
            "jobHref": links[0]["href"] if links else "",
        })
    return items[:MAX_THUMBS]


def scrape_all(rows, cookie):
    """
    Fetch screenshot data for all rows.
    Returns dict keyed by display host -> list of screenshot items.
    """
    session = requests.Session()
    session.headers.update({
        "Cookie": cookie,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    })

    # Separate host-search rows (display != query) from TLD-search rows
    tld_map = {}   # query -> [display, ...]  (deduplicated fetches)
    host_set = []  # [display, ...]  (each needs its own fetch)

    for row in rows:
        display, query = row["display"], row["query"]
        if display == query:
            tld_map.setdefault(query, []).append(display)
        else:
            if display not in host_set:
                host_set.append(display)

    total = len(tld_map) + len(host_set)
    done = 0
    results = {}

    def fetch(query, search_type):
        try:
            r = session.get(_url(query, search_type), timeout=30)
            r.raise_for_status()
            return _parse(r.text)
        except Exception as exc:
            print(f"  WARN [{query}]: {exc}", file=sys.stderr)
            return []

    for query, displays in tld_map.items():
        items = fetch(query, "top_domain_in_requests")
        for d in displays:
            results[d] = items
        done += 1
        print(f"  [{done}/{total}] top_domain:{query}  ({len(items)} shots)")
        time.sleep(0.3)

    for display in host_set:
        items = fetch(display, "host")
        results[display] = items
        done += 1
        print(f"  [{done}/{total}] host:{display}  ({len(items)} shots)")
        time.sleep(0.3)

    return results
