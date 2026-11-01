"""Round 12 (1.11.2026): every tool call carries an optional `why` argument
and the trace prints it as the step's reasoning when the model wrote no
text next to the call."""
import json

import roadguard_agent as ra


def _msg(content=None, tool_calls=None):
    return ra.ChatMessage(finish_reason="tool_calls" if tool_calls else "stop", content=content, tool_calls=tool_calls or [])


def _call(name, args, call_id="c1"):
    return {"id": call_id, "function": {"name": name, "arguments": json.dumps(args)}}


def test_offered_schemas_carry_why_and_the_registry_is_untouched():
    original = ra.PERCEPTION_WORKER_TOOLS["lookup_similar_cases"][1]
    offered = ra._with_why(original)
    assert "why" in offered["function"]["parameters"]["properties"]
    assert "why" not in original["function"]["parameters"]["properties"]
    assert offered["function"]["parameters"]["required"] == original["function"]["parameters"]["required"]
    for schema in ra._decision_tool_schemas():
        assert schema["function"]["parameters"]["properties"]["why"]["type"] == "string"
    assert "why" not in ra.DECISION_TOOLS["check_hazard_severity"][1]["function"]["parameters"]["properties"]


def test_step_thought_prefers_the_message_text_then_the_why_argument():
    assert ra._step_thought("I check the severity first", {"why": "w"}) == "I check the severity first"
    assert ra._step_thought("", {"why": "to learn whether the place recurs"}) == "to learn whether the place recurs"
    assert ra._step_thought(None, {"hazard_class": "D40"}) == ""
    assert ra._step_thought(None, None) == ""


def test_worker_call_with_why_runs_without_it_and_the_step_shows_it(monkeypatch):
    seen = []

    def similar(location: str, hazard_class: str, state: dict | None = None) -> dict:
        seen.append({"location": location, "hazard_class": hazard_class})
        return {"ok": True, "count": 0, "cases": [], "is_recurring_location": False}

    registry = {"lookup_similar_cases": (similar, ra.PERCEPTION_WORKER_TOOLS["lookup_similar_cases"][1])}
    final = json.dumps({"severity_signal": "minor", "confidence": "medium", "needs_more_evidence": False, "missing_fields": []})
    replies = [
        _msg(tool_calls=[_call("lookup_similar_cases", {"location": "x", "hazard_class": "D40", "why": "to learn whether the place recurs"})]),
        _msg(content=final),
    ]
    calls = []

    def fake_chat(messages, tools=None, temperature=0.0, use_panel=False, provider_override=None, **kw):
        calls.append(tools)
        return replies[len(calls) - 1], 10, 5

    monkeypatch.setattr(ra, "chat", fake_chat)
    parsed, _, _, info = ra._react_worker_rounds(
        agent_name="profile_builder", system_prompt="s", user_payload={}, tool_registry=registry,
        state={"hazard_id": "hz_t", "metadata": {}}, provider_override=None, max_hops=2,
    )
    assert parsed["severity_signal"] == "minor"
    assert seen == [{"location": "x", "hazard_class": "D40"}], "why never reaches the function"
    assert info["calls"][0]["ok"] is True
    assert info["steps"][0]["content"]["thought"] == "to learn whether the place recurs"
    assert "why" in calls[0][0]["function"]["parameters"]["properties"], "the model is offered the why field"
    text = "\n".join(ra._entry_lines(info["steps"][0], []))
    assert "to learn whether the place recurs" in text and "(no reasoning text)" not in text


def test_decision_call_with_why_shows_it_as_the_step_reasoning(monkeypatch):
    monkeypatch.setattr(ra, "_decision_messages", lambda state, hint: [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}])
    monkeypatch.setattr(ra, "_run_deadline", lambda: None)
    received = []
    monkeypatch.setattr(ra, "_run_decision_tool", lambda name, args, state, tools_run: received.append(args) or {"ok": True, "priority": 55})
    final = json.dumps({"action": "human_review", "rationale": "x", "priority": 55, "cited_sources": [], "missing_evidence": []})
    replies = [
        _msg(tool_calls=[_call("check_hazard_severity", {"hazard_class": "D40", "confidence": 0.5, "road_type": "main", "lane_blocked": "none", "why": "the priority drives the queue"})]),
        _msg(content=final),
    ]
    n = []
    monkeypatch.setattr(ra, "chat", lambda messages, tools=None, temperature=0.0, **kw: (n.append(1), replies[len(n) - 1], 10, 5)[1:])
    decision, _, _, info = ra._llm_decide_react({"hazard_id": "hz_t", "tool_result_ids": [], "metadata": {}, "budget": None, "iterations": 0}, {"action": "standard"})
    assert decision["action"] == "human_review"
    assert info["steps"][0]["content"]["thought"] == "the priority drives the queue"
    assert received[0]["why"] == "the priority drives the queue", "the Decision dispatcher reads named arguments and ignores why"
    good = {"hazard_class": "D40", "confidence": 0.5, "road_type": "main", "lane_blocked": "none", "why": "x"}
    assert ra._validate_tool_args(ra.DECISION_TOOLS["check_hazard_severity"][1], good) == []


def test_prompts_ask_for_the_why_argument(monkeypatch):
    assert "`why` argument" in ra._build_decision_prompt()
    monkeypatch.setenv("AGENT_WORKER_AUTONOMY", "1")
    assert "`why` argument" in ra._autonomy_prompt("profile_builder", "base")
    monkeypatch.setenv("AGENT_WORKER_AUTONOMY", "0")
    assert ra._autonomy_prompt("profile_builder", "base") == "base"
