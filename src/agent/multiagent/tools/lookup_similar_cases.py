"""lookup_similar_cases: vector memory for recurring hazards at same location.

Used by the Perception agent to detect recurring-location hazards that
deserve a priority bump. Backs onto ChromaDB in production; stub returns
a fixed deterministic result for tests.
"""

from __future__ import annotations

SCHEMA = {
    "type": "function",
    "function": {
        "name": "lookup_similar_cases",
        "description": (
            "Find past hazards at the same location within a window. "
            "Returns up to k nearest neighbours with distance, days_ago, "
            "and the resolved_action if available. Use once per hazard "
            "to decide whether this is a recurring issue."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "hazard_class": {"type": "string", "enum": ["D00", "D10", "D20", "D40"]},
                "radius_meters": {
                    "type": "integer",
                    "minimum": 5,
                    "maximum": 500,
                    "default": 50,
                },
                "window_days": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 365,
                    "default": 30,
                },
                "k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
            },
            "required": ["location", "hazard_class"],
        },
    },
}


def lookup_similar_cases(
    location: str,
    hazard_class: str,
    radius_meters: int = 50,
    window_days: int = 30,
    k: int = 5,
    vector_mem=None,
) -> dict:
    if vector_mem is not None:
        hits = vector_mem.find_similar(
            location=location,
            hazard_class=hazard_class,
            radius_m=radius_meters,
            days=window_days,
        )[:k]
    else:
        hits = []

    return {
        "count": len(hits),
        "cases": hits,
        "is_recurring_location": len(hits) >= 3,
    }
