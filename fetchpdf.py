"""
fetch_pdfs.py -- STEP 1: get PDFs onto disk for a date range.

Why semi-automated: the FOS search page's filter fields aren't something I can
inspect from my side, so guessing exact selectors keeps failing. Instead:

  - YOU manually set the date range and click search in a real (visible)
    browser window that this script opens for you.
  - The SCRIPT then takes over: it reads every PDF link on the results page,
    clicks "Next" to page through results, and downloads each PDF -- this
    part doesn't need to know anything about your specific filter UI, it just
    looks for links ending in .pdf, which is a stable pattern across pages.

Install once:
    pip install playwright requests
    playwright install chromium

Run:
    python fetch_pdfs.py --pdf_dir ./pdfs
"""
import argparse
import re
import time
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright

SEARCH_URL = "https://www.financial-ombudsman.org.uk/businesses/resolving-complaint/ombudsman-decisions"
PDF_URL_TEMPLATE = "https://www.financial-ombudsman.org.uk/decision/{ref}.pdf"
REFERENCE_PATTERN = re.compile(r"DRN-\d{5,9}", re.I)
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research-downloader/1.0)"}


def download_pdf(ref: str, pdf_dir: Path, seen: set) -> bool:
    """Download one PDF by reference if we haven't already. Returns True if newly downloaded."""
    ref = ref.upper()
    name = f"{ref}.pdf"
    if name in seen:
        return False
    out_path = pdf_dir / name
    if out_path.exists():
        seen.add(name)
        return False

    url = PDF_URL_TEMPLATE.format(ref=ref)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code == 200 and resp.content[:4] == b"%PDF":
            out_path.write_bytes(resp.content)
            seen.add(name)
            return True
        else:
            print(f"  skipped (not a PDF, status {resp.status_code}): {url}")
    except Exception as e:
        print(f"  failed {url}: {e}")
    return False


def run(pdf_dir: str, max_pages: int):
    pdf_dir = Path(pdf_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    seen = {p.name for p in pdf_dir.glob("*.pdf")}
    print(f"Already have {len(seen)} PDFs in {pdf_dir}.")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # visible so you can interact
        page = browser.new_page()
        page.goto(SEARCH_URL, wait_until="load")

        print("\n" + "=" * 70)
        print("A browser window has opened.")
        print("1. Dismiss the cookie banner if one appears.")
        print("2. Set your date range filter on the page and click Search/Apply.")
        print("3. Wait until you see the list of decision results.")
        print("Then come back here and press Enter to start downloading.")
        print("=" * 70)
        input(">>> Press Enter once results are showing... ")

        total_downloaded = 0
        for page_num in range(1, max_pages + 1):
            # Robustly wait for the page to settle before reading content --
            # clicking "Next" often triggers a client-side (AJAX) update
            # rather than a full page reload, so content can be read mid-update.
            html = None
            for attempt in range(5):
                try:
                    page.wait_for_load_state("networkidle", timeout=8000)
                except Exception:
                    pass
                page.wait_for_timeout(1000)
                try:
                    html = page.content()
                    break
                except Exception:
                    time.sleep(1)
            if html is None:
                print("Could not read page content after retries -- stopping.")
                break

            # Find every decision reference on the page, regardless of how
            # the link/href is structured -- this is more robust than trying
            # to resolve relative hrefs, since we already know the correct
            # direct PDF URL pattern.
            refs = sorted(set(m.upper() for m in REFERENCE_PATTERN.findall(html)))

            print(f"\nPage {page_num}: found {len(refs)} references.")
            new_this_page = 0
            for ref in refs:
                if download_pdf(ref, pdf_dir, seen):
                    new_this_page += 1
                    total_downloaded += 1
                time.sleep(0.3)  # be polite to the server
            print(f"  downloaded {new_this_page} new PDFs (total so far: {total_downloaded})")

            # Try to page forward. If there's no working "Next", we're done.
            next_link = page.get_by_text(re.compile(r"^\s*next\s*$", re.I)).first
            try:
                if next_link.is_visible(timeout=2000) and next_link.is_enabled():
                    next_link.click()
                    try:
                        page.wait_for_load_state("networkidle", timeout=10000)
                    except Exception:
                        pass
                    page.wait_for_timeout(1500)  # let client-side render settle
                else:
                    print("No enabled 'Next' control found -- stopping.")
                    break
            except Exception:
                print("No 'Next' control found -- assuming last page, stopping.")
                break

        browser.close()

    print(f"\nDone. Downloaded {total_downloaded} new PDFs into {pdf_dir}.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_dir", default="./pdfs")
    ap.add_argument("--max_pages", type=int, default=500)
    args = ap.parse_args()
    run(args.pdf_dir, args.max_pages)