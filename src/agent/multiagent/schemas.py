"""Handoff schemas between agents.

Per DS23 S44: a handoff is a type-safe interface, not "the model will
understand". These classes define exactly what one agent passes to the next.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from .state import Confidence, HazardProfile


class PerceptionHandoff(TypedDict):
    """Perception -> Research. The research agent sees only this."""

    sender: Literal["perception"]
    recipient: Literal["research"]
    hazard_profile: HazardProfile
    confidence: Confidence
    needs_more_evidence: bool
    missing_fields: list[str]


class ResearchHandoff(TypedDict):
    """Research -> Decision. Decision sees only this plus the profile."""

    sender: Literal["research"]
    recipient: Literal["decision"]
    procedures: list[dict]
    tool_result_ids: list[str]
    coverage: Literal["direct_match", "partial_match", "no_match"]
    confidence: Confidence


class DecisionOutput(TypedDict):
    """Final output of the whole graph. Shape matches the API contract."""

    hazard_id: str
    action: Literal["urgent", "standard", "needs_more_evidence", "human_review"]
    rationale: str
    priority: int
    cited_sources: list[str]
    missing_evidence: list[str]
    trace: list[dict]
    cost_usd: float
    guard_fired: bool


def build_perception_handoff(
    profile: HazardProfile,
    confidence: Confidence,
    needs_more: bool,
    missing: list[str],
) -> PerceptionHandoff:
    return {
        "sender": "perception",
        "recipient": "research",
        "hazard_profile": profile,
        "confidence": confidence,
        "needs_more_evidence": needs_more,
        "missing_fields": missing,
    }


def build_research_handoff(
    procedures: list[dict],
    tool_result_ids: list[str],
    coverage: Literal["direct_match", "partial_match", "no_match"],
    confidence: Confidence,
) -> ResearchHandoff:
    return {
        "sender": "research",
        "recipient": "decision",
        "procedures": procedures,
        "tool_result_ids": tool_result_ids,
        "coverage": coverage,
        "confidence": confidence,
    }
