"""flag_for_urgent_review: route the case to a human supervisor.

NOT a contractor dispatch. Writes a row and sends an internal email
summary to the supervisor address. Used when the decision is urgent OR
when the agent cannot justify a non-urgent action with retrieved sources.
"""

from __future__ import annotations

import uuid

SCHEMA = {
    "type": "function",
    "function": {
        "name": "flag_for_urgent_review",
        "description": (
            "Route the case to a human supervisor for review. Writes a "
            "row and sends an internal email summary. This does NOT "
            "contact a contractor, close a road, or publish anything "
            "outside the system. Use for action='urgent' or when the "
            "agent cannot ground its decision in retrieved sources."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "hazard_id": {"type": "string"},
                "reason": {"type": "string", "maxLength": 200},
                "urgency": {"type": "string", "enum": ["immediate", "today", "this_week"]},
            },
            "required": ["hazard_id", "reason", "urgency"],
        },
    },
}


def flag_for_urgent_review(
    hazard_id: str,
    reason: str,
    urgency: str,
    review_queue=None,
    email_client=None,
) -> dict:
    flag_id = f"flag_{uuid.uuid4().hex[:8]}"
    if review_queue is not None:
        review_queue.flag(hazard_id=hazard_id, reason=reason)
    if email_client is not None:
        email_client.send_internal_summary(
            hazard_id=hazard_id, reason=reason, urgency=urgency
        )
    return {
        "ok": True,
        "flag_id": flag_id,
        "hazard_id": hazard_id,
        "urgency": urgency,
        "notified_recipients": "internal only",
    }
