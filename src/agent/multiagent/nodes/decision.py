"""Decision Agent: the only agent that can write.

Reads:  hazard_profile, procedures, tool_result_ids, metadata, perception_confidence
Writes: action, rationale, priority, cited_sources
Tools:  check_hazard_severity, request_additional_evidence,
        create_draft_work_order, flag_for_urgent_review

The decision is grounded by assert_decision_grounded: every cited source
must come from tool_result_ids. An ungrounded citation triggers a retry
loop; three failed retries fall through to the rule-based path.

The LLM path (USE_LIVE=1) runs the panel (``use_panel=True``) first. The
panel's verdict must pass the grounding check; otherwise the decision
falls back to the deterministic rule tree, which itself re-runs with
grounding checks.
"""

from __future__ import annotations

import json
import logging

from ..asserts import HandoffAssertionError, assert_decision_grounded
from ..chat import chat, cost_of
from ..node_prompts import DECISION_SYSTEM_PROMPT
from ..state import RoadGuardState
from ..tools import DECISION_TOOLS

LOGGER = logging.getLogger("roadguard.multiagent.decision")
MAX_GROUNDING_RETRIES = 3

_VALID_ACTIONS = {"urgent", "standard", "needs_more_evidence", "human_review"}


def _decide_rule_based(state: RoadGuardState) -> dict:
    """Deterministic rule tree matching the Agent Contract.

    This is the eval baseline AND the safe fallback when the LLM output
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


def _parse_decision_json(text: str) -> dict | None:
    """Pull the decision JSON out of an LLM reply. None on any error."""
    if not text:
        return None
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").split("\n", 1)[-1]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    action = obj.get("action")
    if action not in _VALID_ACTIONS:
        return None
    try:
        priority = int(obj.get("priority", 0))
    except (TypeError, ValueError):
        return None
    cited = obj.get("cited_sources") or []
    missing = obj.get("missing_evidence") or []
    if not isinstance(cited, list) or not isinstance(missing, list):
        return None
    return {
        "action": action,
        "rationale": str(obj.get("rationale") or "")[:500],
        "priority": max(0, min(100, priority)),
        "cited_sources": [str(c) for c in cited][:5],
        "missing_evidence": [str(m) for m in missing][:5],
    }


def _llm_decide(
    state: RoadGuardState,
    rule_hint: dict,
) -> tuple[dict | None, int, int]:
    """Ask the panel for a decision. Returns (decision or None, in_tok, out_tok).

    None means the panel failed or produced an ungroundable answer and the
    caller should fall back to the rule-based path.
    """
    profile = state.get("hazard_profile") or {}
    md = state.get("metadata") or {}
    procedures = state.get("procedures") or []
    tool_result_ids = state.get("tool_result_ids") or []
    coverage = state.get("research_coverage", "no_match")
    perception_conf = state.get("perception_confidence", "medium")

    try:
        user_payload = {
            "hazard_profile": profile,
            "perception_confidence": perception_conf,
            "metadata": {
                "road_type": md.get("road_type"),
                "lane_blocked": md.get("lane_blocked"),
                "location": md.get("location"),
            },
            "research_coverage": coverage,
            "tool_result_ids": tool_result_ids,
            "procedures": [
                {
                    "source_url": p.get("source_url"),
                    "issuing_body": p.get("issuing_body"),
                    "article_ids": p.get("article_ids", []),
                    "quarantined_summary": p.get("quarantined_summary", {}),
                }
                for p in procedures
            ],
            "rule_based_hint": rule_hint,
        }
        messages = [
            {"role": "system", "content": DECISION_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]
        msg, in_tok, out_tok = chat(
            messages, tools=None, temperature=0.0, use_panel=True
        )
    except Exception as exc:
        LOGGER.warning("decision_llm_failed: %s", exc)
        return None, 0, 0

    parsed = _parse_decision_json(msg.content or "")
    if parsed is None:
        return None, in_tok, out_tok

    try:
        assert_decision_grounded(
            cited_sources=parsed["cited_sources"],
            tool_result_ids=tool_result_ids,
            action=parsed["action"],
        )
    except HandoffAssertionError as exc:
        LOGGER.info("decision_llm_ungrounded: %s", exc)
        return None, in_tok, out_tok

    return parsed, in_tok, out_tok


def decision_node(state: RoadGuardState) -> dict:
    """Produce the final decision, grounded in retrieved sources."""
    tool_result_ids = state.get("tool_result_ids") or []

    rule_hint = _decide_rule_based(state)

    llm_decision, in_tok, out_tok = _llm_decide(state, rule_hint)

    decision: dict | None = None
    source: str = "rule_based"
    if llm_decision is not None:
        decision = llm_decision
        source = "llm_panel"
    else:
        last_error = ""
        for _ in range(MAX_GROUNDING_RETRIES):
            candidate = _decide_rule_based(state)
            try:
                assert_decision_grounded(
                    cited_sources=candidate["cited_sources"],
                    tool_result_ids=tool_result_ids,
                    action=candidate["action"],
                )
                decision = candidate
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
        "content": {**decision, "source": source},
    }

    update: dict = {
        **decision,
        "final": True,
        "trace": [trace_entry],
        "iterations": 1,
    }
    if in_tok or out_tok:
        update["tokens_in"] = in_tok
        update["tokens_out"] = out_tok
        update["cost_usd"] = cost_of(in_tok, out_tok)
    return update
