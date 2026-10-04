"""Tool schemas and executors."""

import json

import pytest

from agent import tools


def test_tool_registry_covers_five_tools():
    names = {s.name for s in tools.TOOL_SPECS}
    assert names == {
        "check_hazard_severity",
        "query_repair_standards",
        "request_additional_evidence",
        "create_draft_work_order",
        "flag_for_urgent_review",
    }


def test_tool_schemas_are_json_serializable_for_groq():
    schemas = tools.tool_schemas_for_groq()
    assert isinstance(schemas, list) and len(schemas) == 5
    for entry in schemas:
        assert entry["type"] == "function"
        fn = entry["function"]
        assert fn["name"] and fn["description"] and fn["parameters"]
        # JSON schema should round-trip
        assert json.loads(json.dumps(fn["parameters"]))["type"] == "object"


def test_check_hazard_severity_rules_scale_with_confidence():
    low = tools.compute_severity("D40", 0.5, "main", "none")
    high = tools.compute_severity("D40", 0.95, "main", "none")
    assert low["priority"] < high["priority"]


def test_check_hazard_severity_lane_blocked_bonus():
    none = tools.compute_severity("D40", 0.8, "main", "none")
    full = tools.compute_severity("D40", 0.8, "main", "full")
    assert full["priority"] > none["priority"]


def test_check_hazard_severity_band_bands():
    assert tools.compute_severity("D40", 0.95, "main", "full")["band"] == "critical"
    assert tools.compute_severity("D40", 0.8, "parking", "none")["band"] in {"medium", "high"}
    assert tools.compute_severity("D10", 0.5, "parking", "none")["band"] == "low"


def test_request_evidence_stores_prompt():
    ctx = tools.ToolContext()
    out = tools.execute_tool_call(
        "request_additional_evidence",
        {"missing_field": "second_angle", "prompt_hint": "closer please"},
        ctx,
    )
    assert "second_angle" in ctx.evidence_requests
    assert "second_angle" in out["missing_field"]
    assert "closer please" in out["prompt_text"]


def test_create_draft_work_order_uses_injected_store():
    class Store:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def create_draft(self, hazard_id, priority, suggested_action):
            self.calls.append(
                {"hazard_id": hazard_id, "priority": priority, "suggested_action": suggested_action}
            )
            return "wo_abc"

    store = Store()
    ctx = tools.ToolContext(work_orders=store)
    out = tools.execute_tool_call(
        "create_draft_work_order",
        {"hazard_id": "hz1", "priority": 75, "suggested_action": "patch"},
        ctx,
    )
    assert out["work_order_id"] == "wo_abc"
    assert out["status"] == "draft"
    assert store.calls == [{"hazard_id": "hz1", "priority": 75, "suggested_action": "patch"}]


def test_flag_for_urgent_review_notifies_queue():
    class Queue:
        def __init__(self) -> None:
            self.flags: list[tuple[str, str]] = []

        def flag(self, hazard_id, reason):
            self.flags.append((hazard_id, reason))

    queue = Queue()
    ctx = tools.ToolContext(review_queue=queue)
    out = tools.execute_tool_call(
        "flag_for_urgent_review", {"hazard_id": "hz1", "reason": "test"}, ctx
    )
    assert out == {"ok": True}
    assert queue.flags == [("hz1", "test")]


def test_query_repair_standards_without_rag_returns_empty_passages():
    out = tools.execute_tool_call(
        "query_repair_standards",
        {"hazard_class": "D40", "road_type": "main", "k": 3},
        tools.ToolContext(),
    )
    assert out["passages"] == []


def test_unknown_tool_name_raises():
    with pytest.raises(KeyError):
        tools.execute_tool_call("nope", {}, tools.ToolContext())


def test_tool_exception_wrapped_as_result():
    def boom(args, ctx):
        raise RuntimeError("explode")

    tools.TOOL_REGISTRY["__boom__"] = tools.ToolSpec(
        name="__boom__", description="t", schema={}, executor=boom
    )
    try:
        out = tools.execute_tool_call("__boom__", {}, tools.ToolContext())
        assert out["error"] == "tool_exception"
    finally:
        tools.TOOL_REGISTRY.pop("__boom__", None)
