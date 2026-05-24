#!/usr/bin/env python3
"""
Flask server for the MIC gallery.
Serves the gallery HTML/images and proxies BL uploads to internal.geoedge.com
and labeling to see.geoedge.be using the existing Chrome session via Playwright.
"""
import json
import os
import platform
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

HERE = Path(__file__).parent
BL_URL = (
    "https://internal.geoedge.com"
    "/admin_geinternalpage/analytics_malware"
    "/analytics_geoedge_malicious_domain_bulk"
)
SEE_URL = "https://see.geoedge.be/mic_classifications"

# ── Background fetch-and-rerun job state ──────────────────────────────────────
_fetch_job: dict = {"status": "idle", "progress": 0, "message": ""}

app = Flask(__name__)


def _chrome_profile() -> str:
    system = platform.system()
    if system == "Darwin":
        return os.path.expanduser("~/Library/Application Support/Google/Chrome")
    elif system == "Windows":
        return os.path.expanduser("~/AppData/Local/Google/Chrome/User Data")
    return os.path.expanduser("~/.config/google-chrome")


def _make_temp_profile() -> str:
    """
    Copy Chrome session cookies to a fresh temp directory.
    Using a different user-data-dir avoids the ChromeSingleton lock
    (Chrome allows multiple instances with different profile dirs).
    The Keychain encryption key is per macOS user, so copied cookies
    still decrypt fine.
    """
    src = _chrome_profile()
    tmp = tempfile.mkdtemp(prefix="pw-mic-")
    for rel in ["Default/Cookies", "Default/Cookies-wal", "Default/Cookies-shm"]:
        s = os.path.join(src, rel)
        d = os.path.join(tmp, rel)
        if os.path.exists(s):
            os.makedirs(os.path.dirname(d), exist_ok=True)
            try:
                shutil.copy2(s, d)
            except Exception:
                pass
    return tmp


def _launch_ctx(p, tmp_profile: str):
    """Launch a headless Chrome persistent context with the temp profile."""
    return p.chromium.launch_persistent_context(
        tmp_profile,
        headless=True,
        channel="chrome",
        args=["--no-first-run", "--no-default-browser-check"],
        ignore_default_args=["--enable-automation"],
    )


# ── Playwright helpers ────────────────────────────────────────────────────────

def _playwright_upload(csv_path: str) -> dict:
    from playwright.sync_api import sync_playwright
    tmp = _make_temp_profile()
    with sync_playwright() as p:
        try:
            ctx = _launch_ctx(p, tmp)
        except Exception as e:
            shutil.rmtree(tmp, ignore_errors=True)
            return {"success": False, "error": f"Could not launch Chrome: {e}"}
        try:
            page = ctx.new_page()
            page.goto(BL_URL, wait_until="networkidle", timeout=30_000)
            if "microsoftonline" in page.url or "login" in page.url:
                return {"success": False, "error": "Not authenticated — log in to internal.geoedge.com in Chrome first."}
            page.set_input_files("input[name='userfile']", csv_path)
            page.check("input[name='existing_status'][value='pending']")
            page.check("input[name='existing_tags'][value='ignore']")
            btn = (
                page.query_selector("input[value*='Upload']")
                or page.query_selector("button:text('Upload')")
                or page.query_selector("input[type='submit']")
            )
            if not btn:
                return {"success": False, "error": "Upload button not found"}
            btn.click()
            page.wait_for_load_state("networkidle", timeout=30_000)
            err = page.query_selector(".error, .alert-danger, .errorlist")
            if err:
                return {"success": False, "error": err.inner_text().strip()}
            return {"success": True, "message": "Upload successful"}
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            page.close()
            ctx.close()
            shutil.rmtree(tmp, ignore_errors=True)


def _playwright_see_label(label_items: list) -> dict:
    """Call PUT /api/mic/classification for each item via see.geoedge.be session."""
    from playwright.sync_api import sync_playwright
    tmp = _make_temp_profile()
    with sync_playwright() as p:
        try:
            ctx = _launch_ctx(p, tmp)
        except Exception as e:
            shutil.rmtree(tmp, ignore_errors=True)
            return {"success": False, "error": f"Could not launch Chrome: {e}"}
        try:
            page = ctx.new_page()
            page.goto(SEE_URL, wait_until="networkidle", timeout=30_000)
            if "microsoftonline" in page.url or "login" in page.url:
                return {"success": False, "error": "Not authenticated on see.geoedge.be"}
            results = page.evaluate(
                """async (items) => {
                    const out = [];
                    for (const item of items) {
                        try {
                            const r = await fetch('/api/mic/classification', {
                                method: 'PUT',
                                headers: {'Content-Type': 'application/json'},
                                body: JSON.stringify(item)
                            });
                            out.push({domain: item.domain, ok: r.ok, status: r.status});
                        } catch(e) {
                            out.push({domain: item.domain, error: e.message});
                        }
                    }
                    return out;
                }""",
                label_items,
            )
            return {"success": True, "results": results}
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            page.close()
            ctx.close()
            shutil.rmtree(tmp, ignore_errors=True)


def _playwright_fetch_data(filter_json: dict, sort_json: dict, max_pages: int = 20) -> list:
    """Fetch all matching rows from see.geoedge.be/api/mic/classifications."""
    from playwright.sync_api import sync_playwright
    tmp = _make_temp_profile()
    with sync_playwright() as p:
        ctx = _launch_ctx(p, tmp)
        try:
            page = ctx.new_page()
            page.goto(SEE_URL, wait_until="networkidle", timeout=30_000)
            if "microsoftonline" in page.url or "login" in page.url:
                raise RuntimeError("Not authenticated on see.geoedge.be")
            rows = page.evaluate(
                """async ([filter, sort, maxPages]) => {
                    const all = [];
                    for (let pg = 0; pg < maxPages; pg++) {
                        const r = await fetch('/api/mic/classifications', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({filter, page: pg, sort})
                        });
                        const d = await r.json();
                        const rows = d?.data?.rows || [];
                        if (!rows.length) break;
                        all.push(...rows);
                        if (rows.length < (d?.data?.pageSize || 50)) break;
                    }
                    return all;
                }""",
                [filter_json, sort_json, max_pages],
            )
            return rows
        finally:
            page.close()
            ctx.close()
            shutil.rmtree(tmp, ignore_errors=True)


def _run_fetch_and_rerun(filter_json: dict, sort_json: dict, max_pages: int):
    global _fetch_job
    try:
        _fetch_job = {"status": "running", "progress": 5, "message": "Fetching from see.geoedge.be…"}
        rows = _playwright_fetch_data(filter_json, sort_json, max_pages)
        mic_data = [
            {
                "job_time": r.get("job_time", ""),
                "domain": r.get("domain", ""),
                "malicious_score": r.get("malicious_score", 0),
                "image_urls": [r["image_url"]] if r.get("image_url") else [],
                "dcptv": r.get("dcptv", 0),
                "url": r.get("url", ""),
                "dbId": r.get("dbId", ""),
            }
            for r in rows if r.get("image_url") and r.get("domain")
        ]
        (HERE / "mic_data.json").write_text(json.dumps(mic_data))
        _fetch_job = {"status": "running", "progress": 25,
                      "message": f"Fetched {len(mic_data)} entries. Running model…"}
        venv_py = str(HERE / ".venv/bin/python")
        proc = subprocess.Popen(
            [venv_py, str(HERE / "run_gallery.py")],
            cwd=str(HERE), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        for line in proc.stdout:
            line = line.strip()
            if "/" in line:
                try:
                    frac = int(line.split("/")[0].split()[-1]) / int(line.split("/")[1].split()[0])
                    _fetch_job["progress"] = 25 + int(frac * 70)
                except Exception:
                    pass
            _fetch_job["message"] = line
        proc.wait()
        if proc.returncode == 0:
            _fetch_job = {"status": "done", "progress": 100,
                          "message": f"Done — {len(mic_data)} domains processed"}
        else:
            _fetch_job = {"status": "error", "progress": 0, "message": "Model run failed"}
    except Exception as exc:
        _fetch_job = {"status": "error", "progress": 0, "message": str(exc)}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(str(HERE), "gallery.html")


@app.route("/gallery_images/<path:filename>")
def gallery_image(filename):
    return send_from_directory(str(HERE / "gallery_images"), filename)


@app.route("/api/upload_bl", methods=["POST"])
def upload_bl():
    csv_bytes = request.get_data()
    if not csv_bytes:
        return jsonify({"success": False, "error": "Empty CSV"}), 400
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="wb")
    try:
        tmp.write(csv_bytes)
        tmp.close()
        result = _playwright_upload(tmp.name)
    finally:
        os.unlink(tmp.name)
    return jsonify(result)


@app.route("/api/label_mic", methods=["POST"])
def label_mic():
    items = request.get_json()
    if not items:
        return jsonify({"success": False, "error": "No items"}), 400
    result = _playwright_see_label(items)
    return jsonify(result)


@app.route("/api/fetch_and_run", methods=["POST"])
def fetch_and_run():
    global _fetch_job
    if _fetch_job.get("status") == "running":
        return jsonify({"status": "already_running", "message": "A job is already running"})
    body = request.get_json() or {}
    filter_json = body.get("filter", {"malicious_score": {"$gt": 0.9}})
    sort_json = body.get("sort", {"_id.malicious_score": -1})
    max_pages = int(body.get("pages", 20))
    _fetch_job = {"status": "running", "progress": 0, "message": "Starting…"}
    threading.Thread(
        target=_run_fetch_and_rerun,
        args=(filter_json, sort_json, max_pages),
        daemon=True,
    ).start()
    return jsonify({"status": "started"})


@app.route("/api/fetch_status")
def fetch_status():
    return jsonify(_fetch_job)


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f"Gallery running at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
