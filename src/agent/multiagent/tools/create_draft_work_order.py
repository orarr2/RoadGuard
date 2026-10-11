"""create_draft_work_order: write a local draft only.

Does NOT dispatch, does NOT notify a public map, does NOT contact a
contractor. Writes a row in SQLite that a human can review/approve.
"""

from __future__ import annotations

import uuid

SCHEMA = {
    "type": "function",
    "function": {
        "name": "create_draft_work_order",
        "description": (
            "Write a LOCAL DRAFT work order to the SQLite database. "
            "This does not dispatch anyone, close a road, or notify "
            "anything outside the system. A human must approve before "
            "the draft becomes a real order. Use only for action in "
            "{urgent, standard}."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "hazard_id": {"type": "string"},
                "priority": {"type": "integer", "minimum": 0, "maximum": 100},
                "suggested_action": {"type": "string", "enum": ["urgent", "standard"]},
                "cited_sources": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["hazard_id", "priority", "suggested_action", "cited_sources"],
        },
    },
}


def create_draft_work_order(
    hazard_id: str,
    priority: int,
    suggested_action: str,
    cited_sources: list[str],
    store=None,
) -> dict:
    draft_id = f"draft_{uuid.uuid4().hex[:8]}"
    if store is not None:
        store.create_draft(
            hazard_id=hazard_id,
            priority=priority,
            suggested_action=suggested_action,
            cited_sources=cited_sources,
        )
    return {
        "ok": True,
        "draft_id": draft_id,
        "hazard_id": hazard_id,
        "priority": priority,
        "suggested_action": suggested_action,
        "awaiting_human_approval": True,
    }
