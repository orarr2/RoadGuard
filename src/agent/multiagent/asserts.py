"""Assert v3 at team level - schema + grounding checks on every handoff.

Per DS23 S44 slide 21: at the kitchen team level, the assert checks
both that the handoff shape is valid AND that every cited source id
actually came back from a tool call. An ungrounded citation is an
instant failure that routes to human_review.
"""

from __future__ import annotations

from .schemas import PerceptionHandoff, ResearchHandoff


class HandoffAssertionError(Exception):
    """Raised when a handoff fails schema or grounding checks."""


def assert_perception_handoff(h: PerceptionHandoff) -> bool:
    """Perception -> Research. Schema only (no tool-grounding yet)."""
    required = {"sender", "recipient", "hazard_profile", "confidence"}
    missing = required - set(h.keys())
    if missing:
        raise HandoffAssertionError(f"perception handoff missing fields: {missing}")

    if h["sender"] != "perception" or h["recipient"] != "research":
        raise HandoffAssertionError(f"wrong direction: {h['sender']} -> {h['recipient']}")

    profile = h.get("hazard_profile") or {}
    if "primary_class" not in profile:
        raise HandoffAssertionError("hazard_profile missing primary_class")

    if h["confidence"] not in {"high", "medium", "low"}:
        raise HandoffAssertionError(f"invalid confidence: {h['confidence']}")

    return True


def assert_research_handoff(h: ResearchHandoff) -> bool:
    """Research -> Decision. Schema check only. Grounding is checked later
    when the Decision agent cites specific ids."""
    required = {"sender", "recipient", "procedures", "tool_result_ids", "coverage"}
    missing = required - set(h.keys())
    if missing:
        raise HandoffAssertionError(f"research handoff missing fields: {missing}")

    if not isinstance(h["procedures"], list):
        raise HandoffAssertionError("procedures must be a list")

    if not isinstance(h["tool_result_ids"], list):
        raise HandoffAssertionError("tool_result_ids must be a list")

    if h["coverage"] == "no_match" and h["procedures"]:
        raise HandoffAssertionError("no_match coverage must have empty procedures")

    return True


UNGROUNDED_ACTIONS = {"human_review", "needs_more_evidence"}


def assert_decision_grounded(
    cited_sources: list[str],
    tool_result_ids: list[str],
    action: str,
) -> bool:
    """Grounding check at the decision step.

    Every cited source must have come back from a tool call this run.
    Two actions can legitimately cite nothing: human_review (we gave up)
    and needs_more_evidence (we are asking the reporter, not deciding).
    """
    if action in UNGROUNDED_ACTIONS:
        ungrounded = [s for s in cited_sources if s not in tool_result_ids]
        if ungrounded:
            raise HandoffAssertionError(
                f"ungrounded cited sources in {action}: {ungrounded}"
            )
        return True

    if not cited_sources:
        raise HandoffAssertionError(
            f"action '{action}' requires at least one cited source (Charter 90% rule)"
        )

    ungrounded = [s for s in cited_sources if s not in tool_result_ids]
    if ungrounded:
        raise HandoffAssertionError(f"ungrounded cited sources: {ungrounded}")

    return True
