"""
parse_sections.py

Parses a Financial Ombudsman Service (FOS) decision PDF into its four fixed
sections, plus extracted metadata and outcome label.

Design notes:
- Section headers are matched with tolerant regex (dash variants, curly quotes,
  case-insensitivity, optional surrounding whitespace) because PDF text
  extraction often mangles typography.
- This is rule-based, not an LLM call -- it must run cheaply over the entire
  historical corpus and on every new PDF as it's published.
- Output is a flat dict, one per decision, ready to be written to JSON/JSONL
  and later embedded + fed to the classifier.
"""
import re
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import pdfplumber

# Section headers as they appear in FOS decisions. Order matters -- we split
# the full text at each of these, in sequence, from top to bottom.
SECTION_PATTERNS = [
    ("complaint",   r"the\s+complaint"),
    ("what_happened", r"what\s+happened"),
    ("decided_why", r"what\s+i[\u2019']ve\s+decided\s*[\u2013\-\u2014]\s*and\s+why"),
    ("final_decision", r"my\s+final\s+decision"),
]

# Compile tolerant patterns: header must sit on its own-ish line, case-insensitive.
COMPILED_SECTIONS = [
    (name, re.compile(rf"(?im)^\s*{pattern}\s*$"))
    for name, pattern in SECTION_PATTERNS
]

# Outcome extraction: looked for inside the final_decision section text.
# Ordered so more specific phrases are checked before generic ones.
OUTCOME_RULES = [
    ("Partly upheld", re.compile(r"\bpartly\s+uphold\b|\bpartially\s+uphold\b|\bpart\s+uphold\b", re.I)),
    ("Not upheld", re.compile(r"\bdo\s+not\s+uphold\b|\bdon[\u2019']t\s+uphold\b|\bnot\s+upheld\b", re.I)),
    ("Upheld", re.compile(r"\buphold\s+this\s+complaint\b|\bi\s+uphold\b|\bupheld\b", re.I)),
]

REFERENCE_PATTERN = re.compile(r"\bDRN-\d{5,9}\b", re.I)


@dataclass
class ParsedDecision:
    source_file: str
    doc_id: str                 # stable hash, used as the manifest / incremental key
    reference: Optional[str]
    outcome: Optional[str]
    complaint: Optional[str]
    what_happened: Optional[str]
    decided_why: Optional[str]
    final_decision: Optional[str]
    parse_warnings: list


def _extract_text(pdf_path: str) -> str:
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            text_parts.append(t)
    return "\n".join(text_parts)


def _split_sections(full_text: str, warnings: list) -> dict:
    """Find each header's position in the text, then slice the text between
    consecutive headers. Missing headers are recorded as warnings rather than
    silently producing empty sections, so bad PDFs surface for review instead
    of polluting training data."""
    matches = []
    for name, pattern in COMPILED_SECTIONS:
        m = pattern.search(full_text)
        if m:
            matches.append((m.start(), m.end(), name))
        else:
            warnings.append(f"missing_section:{name}")

    matches.sort(key=lambda x: x[0])

    sections = {name: None for name, _ in SECTION_PATTERNS}
    for i, (start, end, name) in enumerate(matches):
        next_start = matches[i + 1][0] if i + 1 < len(matches) else len(full_text)
        sections[name] = full_text[end:next_start].strip()

    return sections


def _extract_outcome(final_decision_text: Optional[str], warnings: list) -> Optional[str]:
    if not final_decision_text:
        warnings.append("no_final_decision_text_for_outcome")
        return None
    for label, pattern in OUTCOME_RULES:
        if pattern.search(final_decision_text):
            return label
    warnings.append("outcome_not_matched_by_rules")
    return None


def _extract_reference(full_text: str) -> Optional[str]:
    m = REFERENCE_PATTERN.search(full_text)
    return m.group(0).upper() if m else None


def parse_decision_pdf(pdf_path: str) -> ParsedDecision:
    warnings = []
    full_text = _extract_text(pdf_path)
    sections = _split_sections(full_text, warnings)
    outcome = _extract_outcome(sections.get("final_decision"), warnings)
    reference = _extract_reference(full_text)

    doc_id = hashlib.sha256(Path(pdf_path).name.encode()).hexdigest()[:16]

    return ParsedDecision(
        source_file=str(pdf_path),
        doc_id=doc_id,
        reference=reference,
        outcome=outcome,
        complaint=sections.get("complaint"),
        what_happened=sections.get("what_happened"),
        decided_why=sections.get("decided_why"),
        final_decision=sections.get("final_decision"),
        parse_warnings=warnings,
    )


def parse_and_save(pdf_path: str, out_dir: str) -> dict:
    result = parse_decision_pdf(pdf_path)
    out_path = Path(out_dir) / f"{result.doc_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(asdict(result), f, indent=2)
    return asdict(result)


if __name__ == "__main__":
    import sys
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "sample_decision.pdf"
    result = parse_decision_pdf(pdf_path)
    print(json.dumps(asdict(result), indent=2))
