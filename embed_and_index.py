"""
embed_and_index.py

Turns parsed decisions (all_decisions.jsonl) into a searchable vector index.
This is the piece that makes the whole system incremental: new decisions get
embedded and appended to the index -- nothing here ever needs to be "retrained"
when new PDFs show up, unlike an LLM.

Model: sentence-transformers/all-MiniLM-L6-v2 (open-source, ~80MB, runs fine
on CPU). Good default to start with; can be swapped for a larger model like
BAAI/bge-base-en-v1.5 later if retrieval quality needs improving -- just
re-run a full rebuild if you ever change the model (see --rebuild flag).

What gets embedded: "complaint" + "what_happened" text. This is the input a
new case will look like at prediction time (we won't know the outcome or
reasoning for a brand new complaint), so precedent search must be based on
the same fields.

Only decisions with a successfully-extracted outcome are indexed -- rows
still sitting in review_queue.jsonl are excluded until a human confirms them,
so bad parses can't quietly become bad precedents.

Usage:
    python3 embed_and_index.py --decisions ./parsed/all_decisions.jsonl --index_dir ./index

Outputs (in index_dir):
    - faiss.index       -- the vector index itself
    - metadata.jsonl     -- one line per indexed decision, same order as the
                            vectors in faiss.index (row i <-> line i)
    - embed_manifest.json -- tracks which doc_ids are already indexed, so
                            re-running only embeds new decisions
"""
import argparse
import json
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


def load_jsonl(path: Path) -> list:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_manifest(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"indexed_doc_ids": []}


def save_manifest(path: Path, manifest: dict):
    path.write_text(json.dumps(manifest, indent=2))


def build_embed_text(record: dict) -> str:
    complaint = record.get("complaint") or ""
    what_happened = record.get("what_happened") or ""
    return (complaint + "\n" + what_happened).strip()


def run(decisions_path: str, index_dir: str, rebuild: bool):
    decisions_path = Path(decisions_path)
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    index_path = index_dir / "faiss.index"
    metadata_path = index_dir / "metadata.jsonl"
    manifest_path = index_dir / "embed_manifest.json"

    if rebuild:
        for p in (index_path, metadata_path, manifest_path):
            if p.exists():
                p.unlink()
        print("Rebuild requested -- cleared existing index, metadata, and manifest.")

    manifest = load_manifest(manifest_path)
    all_records = load_jsonl(decisions_path)

    # Only index decisions with a usable outcome -- unresolved review_queue
    # items shouldn't become precedents until a human confirms them.
    eligible = [r for r in all_records if r.get("outcome") and not r.get("parse_warnings")]
    skipped = len(all_records) - len(eligible)

    new_records = [r for r in eligible if r["doc_id"] not in manifest["indexed_doc_ids"]]

    print(f"Total parsed: {len(all_records)}. Eligible (clean parse + outcome): {len(eligible)}. "
          f"Skipped (warnings/no outcome): {skipped}.")
    print(f"Already indexed: {len(manifest['indexed_doc_ids'])}. New to embed: {len(new_records)}.")

    if not new_records:
        print("Nothing new to embed. Index is up to date.")
        return

    print(f"Loading embedding model ({MODEL_NAME})...")
    model = SentenceTransformer(MODEL_NAME)

    texts = [build_embed_text(r) for r in new_records]
    print(f"Embedding {len(texts)} new decisions...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    # Normalize so inner-product search behaves as cosine similarity.
    faiss.normalize_L2(embeddings)
    embeddings = embeddings.astype("float32")

    dim = embeddings.shape[1]

    if index_path.exists():
        index = faiss.read_index(str(index_path))
    else:
        index = faiss.IndexFlatIP(dim)  # inner product on normalized vecs = cosine sim

    index.add(embeddings)
    faiss.write_index(index, str(index_path))

    # Metadata rows must stay in the exact same order as vectors were added,
    # since FAISS returns row positions, not IDs.
    with open(metadata_path, "a") as f:
        for r in new_records:
            f.write(json.dumps(r) + "\n")

    manifest["indexed_doc_ids"].extend(r["doc_id"] for r in new_records)
    save_manifest(manifest_path, manifest)

    print(f"Done. Index now has {index.ntotal} vectors total.")
    print(f"Index: {index_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", default="./parsed/all_decisions.jsonl")
    ap.add_argument("--index_dir", default="./index")
    ap.add_argument("--rebuild", action="store_true",
                     help="Wipe and rebuild the index from scratch (e.g. after changing the embedding model)")
    args = ap.parse_args()
    run(args.decisions, args.index_dir, args.rebuild)