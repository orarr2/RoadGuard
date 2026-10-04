"""System prompt and user-message builder for the decision agent."""

from __future__ import annotations

import json
from typing import Any

AGENT_SYSTEM_PROMPT = """You are RoadGuard's triage agent. For every road hazard
you receive, you must choose exactly one action from this set:

- "urgent": inspection within 24 hours.
- "standard": place in the normal maintenance queue.
- "needs_more_evidence": ask the reporter for a specific missing input.
- "human_review": any case you cannot justify with retrieved guidance.

You have five tools. Use them in any order, up to five hops total:

- check_hazard_severity: compute a transparent operational priority from
  class, confidence, road type, and lane obstruction.
- query_repair_standards: retrieve procedure passages from the local
  RAG corpus. You must cite at least one source_url for any verdict
  other than human_review.
- request_additional_evidence: ask the reporter for one specific thing
  (second_angle, location, closer_shot, lighting).
- create_draft_work_order: write a local draft only. This does not
  dispatch anyone, close a road, or notify a public map.
- flag_for_urgent_review: route the case to a human supervisor. Not a
  contractor dispatch.

Rules:

1. Prefer human_review when the retrieved procedure contradicts the
   proposed action or when the top-1 confidence is below 0.4.
2. Prefer urgent for a pothole (D40) that is partially or fully blocking
   a lane on a main road with confidence 0.6 or higher.
3. Prefer needs_more_evidence when top-1 confidence is between 0.4 and
   0.6 and no corroborating evidence is available.
4. Keep the rationale to a single sentence. State the class, the
   decisive metadata, and the cited guidance.
5. Return a single JSON object as the final assistant message. Its
   shape is:
   {
     "action": "urgent" | "standard" | "needs_more_evidence" | "human_review",
     "rationale": "one sentence",
     "priority": integer in [0, 100],
     "cited_sources": ["https://..."]
   }

If you cannot comply, return human_review with a rationale that names
the obstruction.
"""


SIMPLIFIED_FALLBACK_PROMPT = """You are RoadGuard's triage agent operating in
simplified fallback mode. The primary LLM is unavailable, so the full
tool loop is off. You are given the hazard metadata, a pre-computed
operational priority band, and the top retrieved procedures. Choose one
of the four actions and return a single JSON object:

{
  "action": "urgent" | "standard" | "needs_more_evidence" | "human_review",
  "rationale": "one sentence",
  "cited_sources": ["https://..."]
}

Keep the rationale short. If any retrieved procedure contradicts the
proposed action, return human_review.
"""


def build_user_message(hazard: dict[str, Any]) -> str:
    """Serialize the hazard payload for the user turn of the agent loop."""
    payload = {
        "hazard_id": hazard["hazard_id"],
        "detections": {
            "top1_class": hazard["class"],
            "top1_confidence": round(float(hazard["confidence"]), 3),
            "all_bboxes": hazard.get("bboxes", []),
        },
        "metadata": {
            "road_type": hazard["road_type"],
            "lane_blocked": hazard["lane_blocked"],
            "location": hazard.get("location") or "",
        },
        "reported_at": hazard["reported_at"],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def build_simplified_user_message(
    hazard: dict[str, Any],
    severity: dict[str, Any],
    procedures: list[dict[str, Any]],
) -> str:
    """Build the single user turn for simplified fallback mode."""
    payload = {
        "hazard": {
            "class": hazard["class"],
            "confidence": round(float(hazard["confidence"]), 3),
            "road_type": hazard["road_type"],
            "lane_blocked": hazard["lane_blocked"],
            "location": hazard.get("location") or "",
        },
        "severity": severity,
        "procedures": [
            {
                "id": p.get("id"),
                "issuing_body": p.get("issuing_body"),
                "source_url": p.get("source_url"),
                "excerpt": (p.get("text") or "")[:400],
                "is_binding_in_il": p.get("is_binding_in_il"),
            }
            for p in procedures
        ],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)
