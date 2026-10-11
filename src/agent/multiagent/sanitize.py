"""Boundary defence against indirect prompt injection.

Per DS23 S43 slides 17-18: tool output is DATA, never commands.
RoadGuard has three attack vectors:
  1. The user-provided `location` free-text field.
  2. OCR text detected inside the uploaded photo (adversarial signage).
  3. RAG passages (poisoned document in the corpus).

Vectors 1 and 2 are handled by regex neutralization here.
Vector 3 is handled by the dual-LLM pattern in nodes/research.py.
"""

from __future__ import annotations

import re

INJECTION_PATTERNS = [
    re.compile(r"(?i)system\s*:"),
    re.compile(r"(?i)ignore\s+(all\s+)?previous\s+instructions"),
    re.compile(r"(?i)you\s+are\s+now"),
    re.compile(r"(?i)disregard\s+(the\s+)?(above|prior)"),
    re.compile(r"(?i)assistant\s*:"),
    re.compile(r"(?i)new\s+instructions\s*:"),
    re.compile(r"```[\s\S]*?```"),
    re.compile(r"<\|.*?\|>"),
    re.compile(r"\[INST\][\s\S]*?\[/INST\]"),
]

OCR_SUSPICIOUS_MARKERS = (
    "system:",
    "ignore",
    "you are",
    "assistant:",
    "mark as urgent",
    "flag this",
    "override",
    "admin password",
)

MAX_LOCATION_LEN = 200
MAX_OCR_LEN = 100


def sanitize_location(text: str | None) -> str:
    """Vector 1: user-provided location.

    Treat as data. Strip anything that resembles a system directive,
    truncate hard, and never pass through unchanged if a pattern hit.
    """
    if not text:
        return ""
    if len(text) > 500:
        return "[location omitted: too long]"

    cleaned = text
    any_hit = False
    for pat in INJECTION_PATTERNS:
        if pat.search(cleaned):
            any_hit = True
            cleaned = pat.sub("[removed]", cleaned)

    cleaned = cleaned.strip()[:MAX_LOCATION_LEN]

    if any_hit:
        return f"[sanitized] {cleaned}"
    return cleaned


def sanitize_ocr(ocr_text: str | None) -> dict:
    """Vector 2: text detected inside the image.

    Returns a structured record so the Perception agent can decide whether
    to flag the whole case as suspicious. Suspicious OCR triggers an
    approval gate downstream even if the Decision would be `standard`.
    """
    if not ocr_text:
        return {"raw_text": "", "is_suspicious_pattern": False, "sanitized": ""}

    lowered = ocr_text.lower()
    is_suspicious = any(m in lowered for m in OCR_SUSPICIOUS_MARKERS)

    return {
        "raw_text": ocr_text[:MAX_OCR_LEN],
        "is_suspicious_pattern": is_suspicious,
        "sanitized": (
            "[suspicious signage detected - sent to approval gate]"
            if is_suspicious
            else ocr_text[:MAX_OCR_LEN]
        ),
    }


def sanitize_passage_for_quarantine(passage: str) -> str:
    """Vector 3 pre-step: trim a RAG passage before the quarantine LLM.

    The quarantine LLM extracts a JSON summary; the Decision agent
    never sees the raw text. This function only enforces length and
    strips obviously malformed tokens - the quarantine LLM is the real
    defence.
    """
    if not passage:
        return ""
    return passage.replace("\x00", "").strip()[:2000]
