"""
predict.py

Takes a new complaint (text) and returns:
  - predicted outcome + confidence, from the trained classifier (0 LLM calls)
  - the most similar precedent decisions, from the FAISS index (0 LLM calls)
  - a confidence tier that decides whether this should be auto-served or
    routed to human review

This is Approach 3 minus the final generation step (the "why" explanation) --
that's the next piece, which will take this script's output as its input.
Everything here is fast, cheap, and involves no LLM calls at all.

Usage:
    python3 predict.py --complaint "Text of the complaint..." --what_happened "Text of what happened..."
"""
import argparse
import json
from pathlib import Path

import numpy as np
import faiss
import joblib
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

# Confidence thresholds for routing -- tune these once you've backtested
# against real outcomes. Starting points, not final values.
HIGH_CONFIDENCE_THRESHOLD = 0.65
MIN_PRECEDENT_SIMILARITY = 0.35  # don't cite a "precedent" that isn't actually similar


def load_metadata(path: Path) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f]


def predict(complaint: str, what_happened: str, index_dir: str, model_dir: str, top_k: int):
    index_dir = Path(index_dir)
    model_dir = Path(model_dir)

    clf = joblib.load(model_dir / "classifier.joblib")
    encoder = joblib.load(model_dir / "label_encoder.joblib")
    index = faiss.read_index(str(index_dir / "faiss.index"))
    metadata = load_metadata(index_dir / "metadata.jsonl")

    embed_model = SentenceTransformer(MODEL_NAME)
    text = (complaint + "\n" + what_happened).strip()
    vec = embed_model.encode([text], convert_to_numpy=True)
    faiss.normalize_L2(vec)
    vec = vec.astype("float32")

    # --- Classification (no LLM call) ---
    proba = clf.predict_proba(vec)[0]
    pred_idx = int(np.argmax(proba))
    predicted_label = encoder.classes_[pred_idx]
    confidence = float(proba[pred_idx])

    confidence_tier = "high" if confidence >= HIGH_CONFIDENCE_THRESHOLD else "low"

    # --- Retrieval (no LLM call) ---
    scores, indices = index.search(vec, top_k)
    precedents = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or score < MIN_PRECEDENT_SIMILARITY:
            continue
        record = metadata[idx]
        precedents.append({
            "reference": record.get("reference"),
            "similarity": round(float(score), 3),
            "outcome": record.get("outcome"),
            "complaint_summary": (record.get("complaint") or "")[:200],
        })

    result = {
        "predicted_outcome": predicted_label,
        "confidence": round(confidence, 3),
        "confidence_tier": confidence_tier,
        "class_probabilities": {
            cls: round(float(p), 3) for cls, p in zip(encoder.classes_, proba)
        },
        "precedents": precedents,
        "status": "predicted, not an actual ombudsman decision",
        "routing": "auto_serve" if confidence_tier == "high" else "route_to_human_review",
    }
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--complaint", required=True)
    ap.add_argument("--what_happened", required=True)
    ap.add_argument("--index_dir", default="./index")
    ap.add_argument("--model_dir", default="./model")
    ap.add_argument("--top_k", type=int, default=5)
    args = ap.parse_args()

    result = predict(args.complaint, args.what_happened, args.index_dir, args.model_dir, args.top_k)
    print(json.dumps(result, indent=2))