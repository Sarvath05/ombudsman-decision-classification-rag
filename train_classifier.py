"""
train_classifier.py

Trains a lightweight classifier on top of the embeddings already sitting in
the FAISS index, to predict the outcome (Upheld / Not upheld / Partly upheld)
of a case from its complaint + what_happened text.

Why a separate classifier instead of just using nearest-neighbor lookup:
a trained model can combine signal across many precedents at once and gives
calibrated-ish probabilities, rather than just "the closest single case".

This is NOT an LLM and NOT the embedding model -- it's a small model trained
in seconds. Re-run this script after your index grows (new decisions added)
to pick up the new labeled examples; it's cheap enough to just retrain from
scratch on the full index each time, unlike the embedding step which is
incremental for cost reasons.

Usage:
    python3 train_classifier.py --index_dir ./index --model_out ./model
"""
import argparse
import json
from pathlib import Path

import numpy as np
import faiss
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score


def load_metadata(path: Path) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_features_and_labels(index_dir: Path):
    index = faiss.read_index(str(index_dir / "faiss.index"))
    metadata = load_metadata(index_dir / "metadata.jsonl")

    n = index.ntotal
    assert n == len(metadata), (
        f"Mismatch: index has {n} vectors but metadata has {len(metadata)} rows -- "
        f"these must stay in sync. Don't edit metadata.jsonl by hand."
    )

    # Reconstruct the stored vectors in the same order they were added.
    X = np.vstack([index.reconstruct(i) for i in range(n)])
    y = [m["outcome"] for m in metadata]
    return X, y, metadata


def run(index_dir: str, model_out: str, test_size: float):
    index_dir = Path(index_dir)
    model_out = Path(model_out)
    model_out.mkdir(parents=True, exist_ok=True)

    X, y_raw, metadata = load_features_and_labels(index_dir)
    print(f"Loaded {len(y_raw)} labeled examples.")

    label_counts = {label: y_raw.count(label) for label in set(y_raw)}
    print(f"Label distribution: {label_counts}")

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    # With very small datasets, stratified splitting can fail if a class has
    # too few examples for both train and test -- fall back to a plain split
    # and warn, rather than crashing.
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
    except ValueError as e:
        print(f"Stratified split failed ({e}) -- using a plain random split instead. "
              f"This usually means you need more examples per class before results are meaningful.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nHoldout accuracy: {acc:.2f} (on {len(y_test)} test examples)")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_, zero_division=0))

    if len(y_raw) < 200:
        print("NOTE: this dataset is very small -- these numbers are a pipeline sanity check, "
              "not a real accuracy estimate. Don't draw conclusions about model quality yet.")

    # Retrain on ALL data for the deployed model (the held-out split above was
    # only for evaluation).
    clf_final = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    clf_final.fit(X, y)

    joblib.dump(clf_final, model_out / "classifier.joblib")
    joblib.dump(encoder, model_out / "label_encoder.joblib")
    print(f"\nSaved trained classifier to {model_out / 'classifier.joblib'}")
    print(f"Saved label encoder to {model_out / 'label_encoder.joblib'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", default="./index")
    ap.add_argument("--model_out", default="./model")
    ap.add_argument("--test_size", type=float, default=0.2)
    args = ap.parse_args()
    run(args.index_dir, args.model_out, args.test_size)