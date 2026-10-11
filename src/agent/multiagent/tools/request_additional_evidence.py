"""request_additional_evidence: ask the reporter for one specific thing.

Writes a row to SQLite with the request type. Does NOT send a push to
the reporter directly - that happens in the API layer after approval.
"""

from __future__ import annotations

MISSING_FIELD_ENUM = ["second_angle", "location", "closer_shot", "lighting", "wider_shot"]

SCHEMA = {
    "type": "function",
    "function": {
        "name": "request_additional_evidence",
        "description": (
            "Ask the reporter for exactly one specific missing input. "
            "Use when confidence is in [0.4, 0.6] with no corroboration, "
            "or when image_quality.blur_score is below the usable threshold. "
            "Pick ONE missing_field - do not request multiple at once."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "hazard_id": {"type": "string"},
                "missing_field": {"type": "string", "enum": MISSING_FIELD_ENUM},
                "reason_shown_to_reporter": {"type": "string", "maxLength": 140},
            },
            "required": ["hazard_id", "missing_field", "reason_shown_to_reporter"],
        },
    },
}


def request_additional_evidence(
    hazard_id: str,
    missing_field: str,
    reason_shown_to_reporter: str,
    store=None,
) -> dict:
    if missing_field not in MISSING_FIELD_ENUM:
        return {"ok": False, "error": f"invalid missing_field: {missing_field}"}

    request_id = f"req_{hazard_id}_{missing_field}"
    if store is not None:
        store.create_evidence_request(hazard_id, missing_field, reason_shown_to_reporter)

    return {
        "ok": True,
        "request_id": request_id,
        "missing_field": missing_field,
        "message_to_reporter": reason_shown_to_reporter[:140],
    }
