"""
batch_ingest.py

Watches a folder of downloaded PDFs (populated by the scraper) and parses any
PDF not already recorded in the manifest. This is what makes ingestion
incremental: re-running this script after new PDFs are dropped in only
processes the new ones -- nothing about existing parsed decisions, embeddings,
or the classifier is touched.

Usage:
    python3 batch_ingest.py --pdf_dir ./pdfs --out_dir ./parsed --manifest manifest.json

Outputs:
    - all_decisions.jsonl in out_dir -- ONE file, one JSON object per line,
      containing every parsed decision (input to the embedding step next).
      New runs APPEND new lines rather than rewriting the whole file, so this
      stays cheap even as the file grows to a year's worth of decisions.
    - A review_queue.jsonl listing any decision with parse warnings, so a human
      can spot-check them instead of silently feeding bad data downstream
    - An updated manifest.json recording which source files have been processed
"""
import argparse
import json
from pathlib import Path
from dataclasses import asdict

from parse_sections import parse_decision_pdf


def load_manifest(manifest_path: Path) -> dict:
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)
    return {"processed_files": {}}


def save_manifest(manifest_path: Path, manifest: dict):
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def run(pdf_dir: str, out_dir: str, manifest_path: str):
    pdf_dir = Path(pdf_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)

    combined_path = out_dir / "all_decisions.jsonl"
    review_queue_path = out_dir / "review_queue.jsonl"
    review_entries = []
    new_records = []  # appended to combined_path in one go at the end

    all_pdfs = sorted(pdf_dir.glob("*.pdf"))
    new_pdfs = [p for p in all_pdfs if p.name not in manifest["processed_files"]]

    print(f"Found {len(all_pdfs)} PDFs total, {len(new_pdfs)} new since last run.")

    processed_count = 0
    flagged_count = 0

    for pdf_path in new_pdfs:
        try:
            result = parse_decision_pdf(str(pdf_path))
        except Exception as e:
            manifest["processed_files"][pdf_path.name] = {"status": "error", "error": str(e)}
            review_entries.append({"file": pdf_path.name, "error": str(e)})
            flagged_count += 1
            continue

        new_records.append(asdict(result))

        manifest["processed_files"][pdf_path.name] = {
            "status": "parsed",
            "doc_id": result.doc_id,
            "outcome": result.outcome,
            "warnings": result.parse_warnings,
        }
        processed_count += 1

        # Flag anything with warnings or a missing outcome for human review --
        # these should NOT silently enter the training set for the classifier.
        if result.parse_warnings or result.outcome is None:
            review_entries.append({
                "file": pdf_path.name,
                "doc_id": result.doc_id,
                "warnings": result.parse_warnings,
                "outcome_found": result.outcome,
            })
            flagged_count += 1

    if new_records:
        with open(combined_path, "a") as f:
            for record in new_records:
                f.write(json.dumps(record) + "\n")

    if review_entries:
        with open(review_queue_path, "a") as f:
            for entry in review_entries:
                f.write(json.dumps(entry) + "\n")

    save_manifest(manifest_path, manifest)

    print(f"Parsed: {processed_count}. Flagged for review: {flagged_count}.")
    print(f"All decisions written to: {combined_path}")
    print(f"Manifest updated at {manifest_path}")
    if flagged_count:
        print(f"Review queue: {review_queue_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", default="./pdfs")
    parser.add_argument("--out_dir", default="./parsed")
    parser.add_argument("--manifest", default="./manifest.json")
    args = parser.parse_args()
    run(args.pdf_dir, args.out_dir, args.manifest)