"""
discover_references.py

Finds the list of decision references (e.g. DRN-4058666) published in a given
date range, by querying the FOS search results.

This is the ONLY piece that needs a calibration step from you, because the
search page renders results via JavaScript and I can't inspect its live network
traffic from my sandboxed environment. Downloading the actual PDFs afterward
is trivial and needs no calibration -- see download_pdfs.py.

--- One-time calibration (about 5 minutes) ---
1. Open https://www.financial-ombudsman.org.uk/businesses/resolving-complaint/ombudsman-decisions
2. Open DevTools -> Network tab -> filter to Fetch/XHR.
3. Set the "Date from" / "Date to" filters on the page to a narrow test range
   (e.g. one week) and apply.
4. Look at the request(s) that fire. Two outcomes:

   A) You see a JSON response containing reference numbers / dates / links.
      -> Copy that request's URL + query params into API_ENDPOINT and
         API_PARAMS below. Map the field names in `parse_api_response()`.
      -> Run: python3 discover_references.py --mode api --date_from 2025-07-01 --date_to 2026-07-01

   B) No clean JSON -- results are just rendered HTML with no visible API call.
      -> Use the Playwright fallback: `pip install playwright && playwright install chromium`
      -> Run: python3 discover_references.py --mode browser --date_from 2025-07-01 --date_to 2026-07-01
      -> Adjust the CSS selectors marked '# ADAPT' in fetch_via_browser() to
         match what you see when you right-click a result -> Inspect.

Output: references.json -- a flat list of {"reference": "DRN-1234567", "date": "2025-08-01"}
Feed this straight into download_pdfs.py.
"""
import argparse
import json
import re
from pathlib import Path

import requests

REFERENCE_PATTERN = re.compile(r"DRN-\d{5,9}", re.I)

# ---------------------------------------------------------------------------
# OPTION A: JSON API -- fill in after DevTools inspection
# ---------------------------------------------------------------------------
API_ENDPOINT = None  # e.g. "https://www.financial-ombudsman.org.uk/api/decisions/search"
API_PARAMS_TEMPLATE = {
    # e.g. "pageSize": 50, "sortBy": "date desc"
}


def parse_api_response(data: dict) -> list:
    """ADAPT: map the real JSON shape to a list of {"reference":..., "date":...}"""
    return [
        {"reference": item["reference"].upper(), "date": item.get("date")}
        for item in data.get("results", [])
    ]


def fetch_via_api(date_from: str, date_to: str) -> list:
    if not API_ENDPOINT:
        raise SystemExit("Set API_ENDPOINT first -- see the docstring's calibration steps.")
    all_refs = []
    page = 1
    while True:
        params = dict(API_PARAMS_TEMPLATE)
        params.update({"page": page, "dateFrom": date_from, "dateTo": date_to})
        resp = requests.get(API_ENDPOINT, params=params, timeout=30)
        resp.raise_for_status()
        batch = parse_api_response(resp.json())
        if not batch:
            break
        all_refs.extend(batch)
        print(f"  page {page}: {len(batch)} references (running total: {len(all_refs)})")
        page += 1
    return all_refs


# ---------------------------------------------------------------------------
# OPTION B: Playwright fallback -- reads the rendered DOM directly
# ---------------------------------------------------------------------------
def fetch_via_browser(date_from: str, date_to: str, max_pages: int = 200) -> list:
    from playwright.sync_api import sync_playwright

    search_url = "https://www.financial-ombudsman.org.uk/businesses/resolving-complaint/ombudsman-decisions"
    all_refs = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(search_url, wait_until="networkidle")

        try:
            page.click("text=Accept all cookies", timeout=3000)
        except Exception:
            pass

        # ADAPT: these selectors depend on the real filter form field names/ids.
        try:
            page.fill("input[name='dateFrom']", date_from)
            page.fill("input[name='dateTo']", date_to)
            page.click("button:has-text('Filter')")
            page.wait_for_load_state("networkidle")
        except Exception as e:
            print(f"Could not apply date filter automatically ({e}); "
                  f"you may need to adjust the selectors above.")

        for page_num in range(max_pages):
            page.wait_for_timeout(1200)
            body_text = page.inner_text("body")
            found = REFERENCE_PATTERN.findall(body_text)
            for ref in found:
                all_refs.append({"reference": ref.upper(), "date": None})

            next_btn = page.query_selector("text=Next")
            if not next_btn or not next_btn.is_enabled():
                break
            next_btn.click()

        browser.close()

    # de-duplicate while preserving order
    seen = set()
    deduped = []
    for r in all_refs:
        if r["reference"] not in seen:
            seen.add(r["reference"])
            deduped.append(r)
    return deduped


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["api", "browser"], default="browser")
    ap.add_argument("--date_from", required=True, help="YYYY-MM-DD")
    ap.add_argument("--date_to", required=True, help="YYYY-MM-DD")
    ap.add_argument("--out", default="references.json")
    args = ap.parse_args()

    print(f"Discovering references from {args.date_from} to {args.date_to} via {args.mode}...")
    refs = fetch_via_api(args.date_from, args.date_to) if args.mode == "api" else fetch_via_browser(args.date_from, args.date_to)

    Path(args.out).write_text(json.dumps(refs, indent=2))
    print(f"Found {len(refs)} references. Saved to {args.out}")
    print("Next: python3 download_pdfs.py --references references.json --pdf_dir ./pdfs")
