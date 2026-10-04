"""The five tools exposed to the decision agent.

Each tool declares a JSON schema that the Groq function-calling API
consumes verbatim and a plain Python callable that executes the tool.
Executors receive a ToolContext so the RAG client, the database, and
the mocked external effects can be injected by the API or by tests.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol


HAZARD_CLASS_ENUM = ["D00", "D10", "D20", "D40"]
ROAD_TYPE_ENUM = ["main", "secondary", "bike_path", "parking"]
LANE_BLOCKED_ENUM = ["none", "partial", "full"]
MISSING_FIELD_ENUM = ["second_angle", "location", "closer_shot", "lighting"]


class RAGClient(Protocol):
    """A minimal interface the agent relies on. ChromaDB satisfies it."""

    def query(
        self,
        hazard_class: str,
        road_type: str,
        k: int,
    ) -> list[dict[str, Any]]:
        ...


class WorkOrderStore(Protocol):
    def create_draft(
        self,
        hazard_id: str,
        priority: int,
        suggested_action: str,
    ) -> str:
        ...


class ReviewQueue(Protocol):
    def flag(self, hazard_id: str, reason: str) -> None:
        ...


@dataclass
class ToolContext:
    """Dependencies injected into tool executors."""

    rag: RAGClient | None = None
    work_orders: WorkOrderStore | None = None
    review_queue: ReviewQueue | None = None
    evidence_requests: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class ToolSpec:
    name: str
    description: str
    schema: dict[str, Any]
    executor: Callable[[dict[str, Any], ToolContext], dict[str, Any]]


def _severity_rules(
    hazard_class: str,
    confidence: float,
    road_type: str,
    lane_blocked: str,
) -> dict[str, Any]:
    base_by_class = {"D40": 70, "D20": 55, "D00": 40, "D10": 40}
    base = base_by_class[hazard_class]
    effective_conf = min(0.95, max(0.0, confidence))
    score = base * effective_conf
    road_bonus = {"main": 15, "secondary": 5, "bike_path": 0, "parking": 0}
    lane_bonus = {"none": 0, "partial": 10, "full": 25}
    score += road_bonus[road_type]
    score += lane_bonus[lane_blocked]
    priority = max(0, min(100, int(round(score))))
    if priority >= 80:
        band = "critical"
    elif priority >= 60:
        band = "high"
    elif priority >= 40:
        band = "medium"
    else:
        band = "low"
    rationale = (
        f"class {hazard_class} at confidence {confidence:.2f} on a "
        f"{road_type} road with {lane_blocked} lane obstruction scores "
        f"{priority} ({band})."
    )
    return {"priority": priority, "band": band, "rationale": rationale}


def _check_hazard_severity(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    return _severity_rules(
        hazard_class=args["class"],
        confidence=float(args["confidence"]),
        road_type=args["road_type"],
        lane_blocked=args["lane_blocked"],
    )


def _query_repair_standards(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    if ctx.rag is None:
        return {"passages": [], "note": "rag_client_unavailable"}
    hazard_class = args["hazard_class"]
    road_type = args["road_type"]
    k = int(args.get("k", 3))
    passages = ctx.rag.query(hazard_class=hazard_class, road_type=road_type, k=k)
    return {"passages": passages}


def _request_additional_evidence(
    args: dict[str, Any], ctx: ToolContext
) -> dict[str, Any]:
    missing_field = args["missing_field"]
    hint = args.get("prompt_hint", "")
    prompts_by_field = {
        "second_angle": "Please take a second photograph from a different angle.",
        "location": "Please provide a location string or a nearby landmark.",
        "closer_shot": "Please take a closer photograph of the damaged area.",
        "lighting": "Please retake the photograph with better lighting.",
    }
    prompt_text = prompts_by_field[missing_field]
    if hint:
        prompt_text = f"{prompt_text} {hint}".strip()
    ctx.evidence_requests[missing_field] = {
        "prompt_text": prompt_text,
        "requested_at": time.time(),
    }
    return {"prompt_text": prompt_text, "missing_field": missing_field}


def _create_draft_work_order(
    args: dict[str, Any], ctx: ToolContext
) -> dict[str, Any]:
    hazard_id = args["hazard_id"]
    priority = int(args["priority"])
    suggested_action = args["suggested_action"]
    if ctx.work_orders is None:
        return {
            "work_order_id": f"draft-{uuid.uuid4().hex[:8]}",
            "status": "draft",
            "note": "work_order_store_unavailable",
        }
    work_order_id = ctx.work_orders.create_draft(
        hazard_id=hazard_id,
        priority=priority,
        suggested_action=suggested_action,
    )
    return {"work_order_id": work_order_id, "status": "draft"}


def _flag_for_urgent_review(
    args: dict[str, Any], ctx: ToolContext
) -> dict[str, Any]:
    hazard_id = args["hazard_id"]
    reason = args["reason"]
    if ctx.review_queue is not None:
        ctx.review_queue.flag(hazard_id=hazard_id, reason=reason)
    return {"ok": True}


TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        name="check_hazard_severity",
        description="Compute an operational priority score from transparent rules.",
        schema={
            "type": "object",
            "properties": {
                "class": {"type": "string", "enum": HAZARD_CLASS_ENUM},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "road_type": {"type": "string", "enum": ROAD_TYPE_ENUM},
                "lane_blocked": {"type": "string", "enum": LANE_BLOCKED_ENUM},
            },
            "required": ["class", "confidence", "road_type", "lane_blocked"],
            "additionalProperties": False,
        },
        executor=_check_hazard_severity,
    ),
    ToolSpec(
        name="query_repair_standards",
        description="Retrieve procedure passages from the local RAG corpus.",
        schema={
            "type": "object",
            "properties": {
                "hazard_class": {"type": "string", "enum": HAZARD_CLASS_ENUM},
                "road_type": {"type": "string", "enum": ROAD_TYPE_ENUM},
                "k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 3},
            },
            "required": ["hazard_class", "road_type"],
            "additionalProperties": False,
        },
        executor=_query_repair_standards,
    ),
    ToolSpec(
        name="request_additional_evidence",
        description="Ask the reporter for one specific missing piece of information.",
        schema={
            "type": "object",
            "properties": {
                "missing_field": {"type": "string", "enum": MISSING_FIELD_ENUM},
                "prompt_hint": {"type": "string"},
            },
            "required": ["missing_field"],
            "additionalProperties": False,
        },
        executor=_request_additional_evidence,
    ),
    ToolSpec(
        name="create_draft_work_order",
        description="Create a local draft work order. Not a dispatch.",
        schema={
            "type": "object",
            "properties": {
                "hazard_id": {"type": "string"},
                "priority": {"type": "integer", "minimum": 0, "maximum": 100},
                "suggested_action": {"type": "string"},
            },
            "required": ["hazard_id", "priority", "suggested_action"],
            "additionalProperties": False,
        },
        executor=_create_draft_work_order,
    ),
    ToolSpec(
        name="flag_for_urgent_review",
        description="Route the hazard to a human supervisor. Not a contractor dispatch.",
        schema={
            "type": "object",
            "properties": {
                "hazard_id": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["hazard_id", "reason"],
            "additionalProperties": False,
        },
        executor=_flag_for_urgent_review,
    ),
]


TOOL_REGISTRY: dict[str, ToolSpec] = {spec.name: spec for spec in TOOL_SPECS}


def tool_schemas_for_groq() -> list[dict[str, Any]]:
    """Groq's function-calling schema wraps each tool as `{type, function}`."""
    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.schema,
            },
        }
        for spec in TOOL_SPECS
    ]


def execute_tool_call(
    name: str,
    arguments: dict[str, Any],
    ctx: ToolContext,
    timeout_s: float = 3.0,
) -> dict[str, Any]:
    """Execute a tool by name. Raises KeyError for unknown tools.

    `timeout_s` is informational for now; the tools are local and fast.
    The API layer may enforce it with a thread-level cutoff later.
    """
    if name not in TOOL_REGISTRY:
        raise KeyError(f"unknown_tool: {name}")
    spec = TOOL_REGISTRY[name]
    start = time.monotonic()
    try:
        result = spec.executor(arguments, ctx)
    except Exception as exc:
        return {"error": "tool_exception", "message": str(exc), "tool": name}
    elapsed = time.monotonic() - start
    if elapsed > timeout_s:
        return {
            "error": "tool_timeout",
            "elapsed_s": round(elapsed, 3),
            "tool": name,
            "result": result,
        }
    return result


def compute_severity(
    hazard_class: str,
    confidence: float,
    road_type: str,
    lane_blocked: str,
) -> dict[str, Any]:
    """Public helper used by simplified fallback mode and tests."""
    return _severity_rules(hazard_class, confidence, road_type, lane_blocked)
