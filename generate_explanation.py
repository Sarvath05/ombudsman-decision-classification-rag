"""
generate_explanation.py

The final piece of Approach 3: takes a new complaint, runs it through the
existing classifier + retrieval pipeline (predict.py), then makes ONE call to
a local Ollama model to draft a "What I've decided - and why" / "My final
decision" style explanation, grounded in the retrieved precedents.

This is the only LLM call in the entire system. Classification and retrieval
above it are both free/instant. Re-running this on a new complaint costs
exactly one local model call -- no API fees, fully offline.

--- One-time setup ---
1. Install Ollama: https://ollama.com/download (Windows installer)
2. Pull a small instruction-tuned model, e.g.:
       ollama pull mistral:7b-instruct
   (If your machine is limited on RAM/GPU, try the lighter:
       ollama pull phi3:mini
   and pass --model phi3:mini below.)
3. Make sure Ollama is running (it usually starts automatically after install;
   otherwise run `ollama serve` in a terminal).

Usage:
    python3 generate_explanation.py --complaint "..." --what_happened "..." --model mistral:7b-instruct
"""
import argparse
import json
import re

import requests

from predict import predict

OLLAMA_URL = "http://localhost:11434/api/generate"

PROMPT_TEMPLATE = """You are drafting a DECISION-SUPPORT explanation for a UK financial dispute case. This is a PREDICTION to assist human review, not a real ombudsman decision.

CASE FACTS (the new complaint):
The complaint: {complaint}
What happened: {what_happened}

A separate classifier has predicted the outcome for this case as: {predicted_label} (confidence: {confidence})
Your job is to explain WHY this outcome is plausible, using the precedent cases below. Do not choose a different outcome yourself -- only justify the one given.

RELEVANT PRECEDENT CASES (paraphrase their reasoning in your own words -- do not copy their wording, and do not invent any facts, figures, or dates beyond what's given here):
{precedents_block}

Write your response in exactly two sections, using these exact headers:

What I've decided - and why
[Your reasoning here, applying the same kind of principles seen in the precedents to the facts of this new case]

My final decision
[A short statement of the outcome, matching: {predicted_label}]

Do not include any monetary figures, statute citations, or specific dates that are not present in the case facts or precedent summaries above.
"""


def build_precedents_block(precedents: list) -> str:
    if not precedents:
        return "(No sufficiently similar precedent cases were found -- reason from general fairness principles only, and note this limitation.)"
    lines = []
    for i, p in enumerate(precedents, 1):
        lines.append(
            f"{i}. [{p['reference']}] Outcome: {p['outcome']}. "
            f"Case summary: {p['complaint_summary']}"
        )
    return "\n".join(lines)


def call_ollama(prompt: str, model: str) -> str:
    resp = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=180,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "").strip()


def extract_numbers(text: str) -> set:
    """Pull out monetary figures and standalone numbers for a hallucination sanity check."""
    return set(re.findall(r"£\s?[\d,]+(?:\.\d+)?|\b\d{3,}\b", text))


def check_for_hallucinated_numbers(generated_text: str, context_text: str) -> list:
    generated_numbers = extract_numbers(generated_text)
    context_numbers = extract_numbers(context_text)
    return sorted(n for n in generated_numbers if n not in context_numbers)


def generate(complaint: str, what_happened: str, index_dir: str, model_dir: str,
             top_k: int, ollama_model: str) -> dict:
    # Step 1-2: classifier + retrieval (no LLM call)
    base_result = predict(complaint, what_happened, index_dir, model_dir, top_k)

    prompt = PROMPT_TEMPLATE.format(
        complaint=complaint,
        what_happened=what_happened,
        predicted_label=base_result["predicted_outcome"],
        confidence=base_result["confidence"],
        precedents_block=build_precedents_block(base_result["precedents"]),
    )

    # Step 3: the one LLM call
    generated_text = call_ollama(prompt, ollama_model)

    # Step 4: cheap post-generation validation (no extra LLM call)
    context_text = complaint + " " + what_happened + " " + build_precedents_block(base_result["precedents"])
    suspicious_numbers = check_for_hallucinated_numbers(generated_text, context_text)

    result = dict(base_result)
    result["generated_explanation"] = generated_text
    result["explanation_validation"] = {
        "suspicious_numbers_not_in_context": suspicious_numbers,
        "flagged_for_review": len(suspicious_numbers) > 0,
    }
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--complaint", required=True)
    ap.add_argument("--what_happened", required=True)
    ap.add_argument("--index_dir", default="./index")
    ap.add_argument("--model_dir", default="./model")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--model", default="mistral:7b-instruct", help="Ollama model name")
    args = ap.parse_args()

    result = generate(args.complaint, args.what_happened, args.index_dir, args.model_dir,
                       args.top_k, args.model)
    print(json.dumps(result, indent=2))