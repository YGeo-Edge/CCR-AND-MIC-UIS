"""
GeoEdge authentication via Microsoft SSO using Playwright + system Chrome.

First run: opens a visible Chrome window, auto-fills credentials, waits for
MFA approval if needed (up to 3 minutes), then saves the session.

Subsequent runs: reuses the saved session silently (headless).
"""

import os
import sys
from pathlib import Path

INTERNAL    = "https://internal.geoedge.com"
TARGET      = f"{INTERNAL}/admin_geinternalpage/analytics/snapshots_jobs"
LOGIN_URL   = f"{INTERNAL}/admin_geinternalpage/login/"
SESSION_DIR = Path.home() / ".geoedge_session"

LAUNCH_OPTS = dict(
    channel="chrome",
    args=["--no-sandbox", "--disable-dev-shm-usage"],
)


def _cookie_string(context):
    return "; ".join(
        f"{c['name']}={c['value']}"
        for c in context.cookies()
        if "geoedge.com" in c.get("domain", "")
    )


def _is_on_internal(url):
    return "internal.geoedge.com" in url and "login" not in url.lower()


def _session_still_valid(context):
    page = context.new_page()
    try:
        page.goto(TARGET, wait_until="domcontentloaded", timeout=20_000)
        return _is_on_internal(page.url)
    except Exception:
        return False
    finally:
        page.close()


def get_cookie():
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    except ImportError:
        print("  playwright not installed — run: pip install playwright", file=sys.stderr)
        return ""

    user = os.environ.get("GEOEDGE_USER", "")
    pwd  = os.environ.get("GEOEDGE_PASS", "")
    # Ensure full Microsoft UPN
    ms_user = user if "@" in user else f"{user}@geoedge.com"

    with sync_playwright() as p:

        # ── 1. Try saved session silently (headless) ───────────────────────
        if SESSION_DIR.exists():
            ctx = p.chromium.launch_persistent_context(
                str(SESSION_DIR), headless=True, **LAUNCH_OPTS
            )
            if _session_still_valid(ctx):
                cookie = _cookie_string(ctx)
                ctx.close()
                return cookie
            ctx.close()

        # ── 2. Full SSO login in visible browser ───────────────────────────
        if not user or not pwd:
            print("ERROR: GEOEDGE_USER and GEOEDGE_PASS must be set.", file=sys.stderr)
            return ""

        print("  Opening Chrome for GeoEdge SSO login ...")
        print("  If MFA is required, approve it in the browser window.")
        ctx = p.chromium.launch_persistent_context(
            str(SESSION_DIR), headless=False, **LAUNCH_OPTS
        )
        page = ctx.new_page()
        page.goto(LOGIN_URL, wait_until="domcontentloaded")

        try:
            # Microsoft login — enter email
            try:
                page.wait_for_selector("input[type=email]", timeout=12_000)
                page.fill("input[type=email]", ms_user)
                page.keyboard.press("Enter")
            except PWTimeout:
                pass  # might already be past email step (SSO token cached)

            # Enter password
            try:
                page.wait_for_selector("input[type=password]", timeout=12_000)
                page.fill("input[type=password]", pwd)
                page.keyboard.press("Enter")
            except PWTimeout:
                pass  # might not need password (SSO / cached creds)

            # "Stay signed in?" — click Yes
            try:
                page.wait_for_selector("#idBtn_Accept", timeout=8_000)
                page.click("#idBtn_Accept")
            except PWTimeout:
                pass

            # Wait for final redirect to internal.geoedge.com (up to 3 min for MFA)
            print("  Waiting for login to complete (up to 3 min) ...")
            page.wait_for_url(f"{INTERNAL}/**", timeout=180_000)

        except PWTimeout as exc:
            print(f"  Login timed out. Current URL: {page.url}", file=sys.stderr)
            ctx.close()
            return ""

        cookie = _cookie_string(ctx)
        ctx.close()
        print("  Login successful — session saved for future runs.")
        return cookie
