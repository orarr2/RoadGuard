"""Charter-mandated policy fallbacks and the decision-table sanity check."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ALLOWED_ACTIONS = frozenset({"urgent", "standard", "needs_more_evidence", "human_review"})

MIN_CONFIDENCE_FOR_ANY_ACTION = 0.4
CONFIDENCE_FOR_URGENT = 0.6


@dataclass
class FallbackVerdict:
    reason: str
    rationale: str


def check_input_fallbacks(hazard: dict[str, Any]) -> FallbackVerdict | None:
    """Early fallbacks that fire before the LLM loop starts."""
    confidence = float(hazard.get("confidence", 0.0))
    if confidence < MIN_CONFIDENCE_FOR_ANY_ACTION:
        return FallbackVerdict(
            reason="low_confidence",
            rationale=(
                f"Top-1 confidence {confidence:.2f} is below the minimum "
                f"{MIN_CONFIDENCE_FOR_ANY_ACTION:.2f} required to act."
            ),
        )
    cls = hazard.get("class")
    if cls not in {"D00", "D10", "D20", "D40"}:
        return FallbackVerdict(
            reason="unknown_class",
            rationale=f"Unrecognized hazard class: {cls!r}.",
        )
    return None


def check_rag_fallback(procedures: list[dict[str, Any]]) -> FallbackVerdict | None:
    if not procedures:
        return FallbackVerdict(
            reason="empty_rag",
            rationale="No supporting procedure was retrieved from the corpus.",
        )
    return None


def check_tool_error(result: dict[str, Any]) -> FallbackVerdict | None:
    if isinstance(result, dict) and result.get("error") in {
        "tool_exception",
        "tool_timeout",
    }:
        return FallbackVerdict(
            reason=result["error"],
            rationale=f"Tool {result.get('tool')!s} failed: {result.get('message', '')}",
        )
    return None


def decision_table_action(
    hazard: dict[str, Any],
    severity: dict[str, Any],
    has_rag: bool,
) -> str:
    """Return the action the decision table prefers for this input."""
    confidence = float(hazard["confidence"])
    cls = hazard["class"]
    lane_blocked = hazard["lane_blocked"]
    band = severity["band"]

    if confidence < MIN_CONFIDENCE_FOR_ANY_ACTION:
        return "human_review"
    if (
        cls == "D40"
        and lane_blocked in {"partial", "full"}
        and confidence >= CONFIDENCE_FOR_URGENT
    ):
        return "urgent"
    if band == "critical" and confidence >= CONFIDENCE_FOR_URGENT:
        return "urgent"
    if confidence < CONFIDENCE_FOR_URGENT:
        return "needs_more_evidence"
    if band in {"low", "medium"} and has_rag:
        return "standard"
    if not has_rag:
        return "human_review"
    return "standard"


def validate_action(action: str) -> bool:
    return action in ALLOWED_ACTIONS


def final_human_review(reason: str, rationale: str) -> dict[str, Any]:
    """A canonical human_review verdict with no citations."""
    return {
        "action": "human_review",
        "rationale": rationale,
        "priority": 0,
        "cited_sources": [],
        "fallback_reason": reason,
    }
