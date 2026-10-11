"""Shared state for the agent graph.

TypedDict with LangGraph reducers. Accumulators use operator.add so that
loop-backs increment counters instead of overwriting them (DS23 S44).
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

Action = Literal["urgent", "standard", "needs_more_evidence", "human_review"]
HazardClass = Literal["D00", "D10", "D20", "D40"]
RoadType = Literal["main", "secondary", "bike_path", "parking"]
LaneBlocked = Literal["none", "partial", "full"]
Confidence = Literal["high", "medium", "low"]


class HazardProfile(TypedDict, total=False):
    """Output of the Perception agent. Consumed by Research + Decision."""

    primary_class: HazardClass
    primary_confidence: float
    secondary_classes: list[dict]
    severity_signal: Literal["minor", "moderate", "severe"]
    severity_rationale: str
    affects_vehicle_path: bool
    is_recurring_location: bool
    recurrence_count_30d: int
    quality_warnings: list[str]


class RoadGuardState(TypedDict, total=False):
    """The one state object passed through all nodes in the graph."""

    hazard_id: str
    yolo_output: dict
    image_quality: dict
    metadata: dict
    sanitized: bool

    hazard_profile: HazardProfile
    perception_confidence: Confidence
    similar_cases: list[dict]

    procedures: list[dict]
    tool_result_ids: list[str]
    research_coverage: Literal["direct_match", "partial_match", "no_match"]

    action: Action
    rationale: str
    priority: int
    cited_sources: list[str]
    missing_evidence: list[str]

    trace: Annotated[list[dict], operator.add]
    iterations: Annotated[int, operator.add]
    cost_usd: Annotated[float, operator.add]
    tokens_in: Annotated[int, operator.add]
    tokens_out: Annotated[int, operator.add]

    needs_retry: bool
    final: bool
    guard_fired: bool


def initial_state(
    hazard_id: str,
    yolo_output: dict,
    image_quality: dict,
    metadata: dict,
) -> RoadGuardState:
    """Build a fresh state at the start of a run."""
    return {
        "hazard_id": hazard_id,
        "yolo_output": yolo_output,
        "image_quality": image_quality,
        "metadata": metadata,
        "sanitized": False,
        "trace": [],
        "iterations": 0,
        "cost_usd": 0.0,
        "tokens_in": 0,
        "tokens_out": 0,
        "final": False,
        "guard_fired": False,
    }
