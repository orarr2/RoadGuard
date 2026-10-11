"""check_hazard_severity: transparent rule-based priority.

Not an LLM call. Pure function from (class, confidence, road_type,
lane_blocked, recurrence) to a priority score in [0, 100]. The agent
calls this instead of guessing a number.
"""

from __future__ import annotations

CLASS_BASE_WEIGHT = {"D40": 60, "D20": 40, "D10": 20, "D00": 15}
ROAD_WEIGHT = {"main": 25, "secondary": 15, "parking": 5, "bike_path": 10}
LANE_WEIGHT = {"full": 20, "partial": 10, "none": 0}

SCHEMA = {
    "type": "function",
    "function": {
        "name": "check_hazard_severity",
        "description": (
            "Compute transparent operational priority from hazard class, "
            "YOLO confidence, road type, lane obstruction and recurrence. "
            "Returns an integer in [0, 100]. Use this once before deciding "
            "the action. Do NOT invent priority numbers; call this."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "hazard_class": {"type": "string", "enum": ["D00", "D10", "D20", "D40"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "road_type": {
                    "type": "string",
                    "enum": ["main", "secondary", "bike_path", "parking"],
                },
                "lane_blocked": {"type": "string", "enum": ["none", "partial", "full"]},
                "recurrence_count_30d": {"type": "integer", "minimum": 0},
            },
            "required": ["hazard_class", "confidence", "road_type", "lane_blocked"],
        },
    },
}


def check_hazard_severity(
    hazard_class: str,
    confidence: float,
    road_type: str,
    lane_blocked: str,
    recurrence_count_30d: int = 0,
) -> dict:
    """Transparent priority formula. Return value includes component breakdown."""
    base = CLASS_BASE_WEIGHT.get(hazard_class, 10)
    conf_mult = max(0.5, min(1.0, confidence))
    road = ROAD_WEIGHT.get(road_type, 5)
    lane = LANE_WEIGHT.get(lane_blocked, 0)
    recurrence_bump = min(15, recurrence_count_30d * 5)

    priority = int(base * conf_mult + road + lane + recurrence_bump)
    priority = max(0, min(100, priority))

    return {
        "priority": priority,
        "components": {
            "base_class": base,
            "confidence_multiplier": round(conf_mult, 2),
            "road_bonus": road,
            "lane_bonus": lane,
            "recurrence_bonus": recurrence_bump,
        },
        "formula": "min(100, base * conf_mult + road + lane + recurrence)",
    }
