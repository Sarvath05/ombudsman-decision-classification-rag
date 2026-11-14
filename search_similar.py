"""
search_similar.py

Quick way to sanity-check retrieval quality: given a new complaint's text,
find the most similar past decisions in the index. This is the same lookup
the RAG explanation stage will use later -- testing it now lets you validate
retrieval quality before building anything on top of it.

Usage:
    python3 search_similar.py --index_dir ./index --query "My bank increased my credit card limit without checking if I could afford it"
"""
import argparse
import json
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


def load_metadata(path: Path) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f]


def search(index_dir: str, query: str, top_k: int):
    index_dir = Path(index_dir)
    index = faiss.read_index(str(index_dir / "faiss.index"))
    metadata = load_metadata(index_dir / "metadata.jsonl")

    model = SentenceTransformer(MODEL_NAME)
    query_vec = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)

    scores, indices = index.search(query_vec.astype("float32"), top_k)

    print(f"\nQuery: {query}\n")
    print(f"Top {top_k} similar precedent decisions:\n")
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
        if idx == -1:
            continue
        record = metadata[idx]
        print(f"{rank}. [{record['reference']}] similarity={score:.3f} outcome={record['outcome']}")
        print(f"   Complaint: {(record.get('complaint') or '')[:150]}...")
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", default="./index")
    ap.add_argument("--query", required=True)
    ap.add_argument("--top_k", type=int, default=5)
    args = ap.parse_args()
    search(args.index_dir, args.query, args.top_k)