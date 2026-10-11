"""Decision Agent: the only agent that can write.

Reads:  hazard_profile, procedures, tool_result_ids, metadata, perception_confidence
Writes: action, rationale, priority, cited_sources
Tools:  check_hazard_severity, request_additional_evidence,
        create_draft_work_order, flag_for_urgent_review

The decision is grounded by assert_decision_grounded: every cited source
must come from tool_result_ids. An ungrounded citation triggers a retry
loop; three failed retries fall through to human_review.
"""

from __future__ import annotations

from ..asserts import HandoffAssertionError, assert_decision_grounded
from ..state import RoadGuardState
from ..tools import DECISION_TOOLS

MAX_GROUNDING_RETRIES = 3


def _decide_rule_based(state: RoadGuardState) -> dict:
    """Deterministic rule tree matching the Agent Contract.

    This is NOT the LLM path. In production the LLM reasons over the
    structured handoff and may propose different wording; this function
    is both the eval baseline and the safe fallback when the LLM output
    fails grounding or the budget guard fires mid-reasoning.
    """
    profile = state.get("hazard_profile") or {}
    md = state.get("metadata") or {}
    tool_result_ids = state.get("tool_result_ids") or []
    coverage = state.get("research_coverage", "no_match")
    perception_conf = state.get("perception_confidence", "medium")

    hazard_class = profile.get("primary_class", "D40")
    confidence = float(profile.get("primary_confidence", 0.0))
    road_type = md.get("road_type", "main")
    lane_blocked = md.get("lane_blocked", "none")

    severity_tool = DECISION_TOOLS["check_hazard_severity"][0](
        hazard_class=hazard_class,
        confidence=confidence,
        road_type=road_type,
        lane_blocked=lane_blocked,
        recurrence_count_30d=profile.get("recurrence_count_30d", 0),
    )
    priority = severity_tool["priority"]

    if perception_conf == "low":
        return {
            "action": "human_review",
            "rationale": (
                f"routed to human: perception={perception_conf} "
                f"(suspicious signage or non-road scene)"
            ),
            "priority": priority,
            "cited_sources": [],
            "missing_evidence": [],
        }

    if 0.4 <= confidence < 0.6 and not profile.get("is_recurring_location"):
        return {
            "action": "needs_more_evidence",
            "rationale": (
                f"confidence {confidence:.2f} in ambiguous band for {hazard_class}"
            ),
            "priority": priority,
            "cited_sources": [],
            "missing_evidence": ["second_angle"],
        }

    if coverage == "no_match":
        return {
            "action": "human_review",
            "rationale": f"no RAG match for {hazard_class} on {road_type}",
            "priority": priority,
            "cited_sources": [],
            "missing_evidence": [],
        }

    if (
        hazard_class == "D40"
        and lane_blocked in ("partial", "full")
        and road_type == "main"
        and confidence >= 0.6
    ):
        return {
            "action": "urgent",
            "rationale": (
                f"D40 pothole on main road with lane_blocked={lane_blocked} "
                f"and confidence {confidence:.2f}"
            ),
            "priority": priority,
            "cited_sources": tool_result_ids[:2],
            "missing_evidence": [],
        }

    return {
        "action": "standard",
        "rationale": f"routine queue for {hazard_class} on {road_type} road",
        "priority": priority,
        "cited_sources": tool_result_ids[:1],
        "missing_evidence": [],
    }


def decision_node(state: RoadGuardState) -> dict:
    """Produce the final decision, grounded in retrieved sources."""
    tool_result_ids = state.get("tool_result_ids") or []

    last_error = ""
    decision: dict | None = None
    for _ in range(MAX_GROUNDING_RETRIES):
        decision = _decide_rule_based(state)
        try:
            assert_decision_grounded(
                cited_sources=decision["cited_sources"],
                tool_result_ids=tool_result_ids,
                action=decision["action"],
            )
            break
        except HandoffAssertionError as e:
            last_error = str(e)
            continue
    else:
        decision = {
            "action": "human_review",
            "rationale": f"grounding failed after {MAX_GROUNDING_RETRIES} retries: {last_error}",
            "priority": 50,
            "cited_sources": [],
            "missing_evidence": [],
        }

    assert decision is not None

    hazard_id = state["hazard_id"]
    if decision["action"] in ("urgent", "standard"):
        DECISION_TOOLS["create_draft_work_order"][0](
            hazard_id=hazard_id,
            priority=decision["priority"],
            suggested_action=decision["action"],
            cited_sources=decision["cited_sources"],
        )
    if decision["action"] == "urgent":
        DECISION_TOOLS["flag_for_urgent_review"][0](
            hazard_id=hazard_id,
            reason=decision["rationale"][:200],
            urgency="immediate",
        )
    if decision["action"] == "needs_more_evidence" and decision["missing_evidence"]:
        DECISION_TOOLS["request_additional_evidence"][0](
            hazard_id=hazard_id,
            missing_field=decision["missing_evidence"][0],
            reason_shown_to_reporter=(
                "A sharper or wider shot would help us triage this correctly."
            ),
        )

    trace_entry = {
        "iter": state.get("iterations", 0) + 1,
        "speaker": "decision",
        "kind": "final",
        "content": decision,
    }

    return {
        **decision,
        "final": True,
        "trace": [trace_entry],
        "iterations": 1,
    }
