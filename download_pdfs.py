"""
download_pdfs.py

Downloads decision PDFs directly, given a list of references from
discover_references.py. No browser automation needed -- FOS decision PDFs
live at a flat, predictable URL:

    https://www.financial-ombudsman.org.uk/decision/{REFERENCE}.pdf

(confirmed working, e.g. https://www.financial-ombudsman.org.uk/decision/DRN-4058666.pdf)

This script is ready to run as-is. It's incremental: re-running it with an
updated references.json only downloads references not already recorded in
the manifest.

Usage:
    python3 download_pdfs.py --references references.json --pdf_dir ./pdfs
"""
import argparse
import json
import time
from pathlib import Path

import requests

PDF_URL_TEMPLATE = "https://www.financial-ombudsman.org.uk/decision/{ref}.pdf"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research-downloader/1.0)"}


def load_manifest(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"downloaded": {}, "failed": {}}


def save_manifest(path: Path, manifest: dict):
    path.write_text(json.dumps(manifest, indent=2))


def download_all(references: list, pdf_dir: Path, manifest_path: Path,
                  delay_seconds: float = 0.5, retries: int = 2):
    pdf_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(manifest_path)

    todo = [r for r in references if r["reference"] not in manifest["downloaded"]]
    print(f"{len(references)} references total, {len(todo)} not yet downloaded.")

    ok_count = 0
    fail_count = 0

    for i, item in enumerate(todo, 1):
        ref = item["reference"]
        url = PDF_URL_TEMPLATE.format(ref=ref)
        out_path = pdf_dir / f"{ref}.pdf"

        success = False
        last_error = None
        for attempt in range(retries + 1):
            try:
                resp = requests.get(url, headers=HEADERS, timeout=30)
                if resp.status_code == 200 and resp.content[:4] == b"%PDF":
                    out_path.write_bytes(resp.content)
                    success = True
                    break
                else:
                    last_error = f"HTTP {resp.status_code} or non-PDF content"
            except Exception as e:
                last_error = str(e)
            time.sleep(1.0)  # back off before retry

        if success:
            manifest["downloaded"][ref] = {"date": item.get("date"), "file": str(out_path)}
            ok_count += 1
        else:
            manifest["failed"][ref] = last_error
            fail_count += 1
            print(f"  FAILED {ref}: {last_error}")

        if i % 25 == 0:
            print(f"  progress: {i}/{len(todo)} ({ok_count} ok, {fail_count} failed)")
            save_manifest(manifest_path, manifest)  # checkpoint periodically

        time.sleep(delay_seconds)  # be polite to the server

    save_manifest(manifest_path, manifest)
    print(f"Done. Downloaded: {ok_count}. Failed: {fail_count}. "
          f"Already had: {len(references) - len(todo)}.")
    if fail_count:
        print(f"Failed references logged in manifest under 'failed' -- retry those separately.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--references", default="references.json")
    ap.add_argument("--pdf_dir", default="./pdfs")
    ap.add_argument("--manifest", default="download_manifest.json")
    ap.add_argument("--delay", type=float, default=0.5, help="seconds between requests")
    args = ap.parse_args()

    references = json.loads(Path(args.references).read_text())
    download_all(references, Path(args.pdf_dir), Path(args.manifest), delay_seconds=args.delay)
