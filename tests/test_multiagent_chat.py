"""Chat wrapper (offline stub) + Trace protocol printer."""

import json

from agent.multiagent.chat import ChatMessage, chat, cost_of, estimate_tokens
from agent.multiagent.trace import Trace


def test_offline_chat_without_tools_finishes():
    msg, tin, tout = chat([{"role": "user", "content": "hi"}], tools=None)
    assert isinstance(msg, ChatMessage)
    assert msg.finish_reason == "stop"
    assert tin > 0 and tout > 0


def test_offline_chat_with_tools_emits_tool_call():
    tools = [{"function": {"name": "lookup_similar_cases"}}]
    msg, _, _ = chat([{"role": "user", "content": "look it up"}], tools=tools)
    assert msg.finish_reason == "tool_calls"
    assert msg.tool_calls[0]["function"]["name"] == "lookup_similar_cases"


def test_offline_chat_after_tool_result_finishes():
    msg, _, _ = chat([{"role": "tool", "content": "{}"}], tools=None)
    assert msg.finish_reason == "stop"


def test_offline_chat_decision_context_emits_json():
    conv = [
        {"role": "system", "content": "decision agent"},
        {"role": "user", "content": "procedures to review: ..."},
    ]
    msg, _, _ = chat(conv, tools=None)
    body = json.loads(msg.content)
    assert body["action"] in {"urgent", "standard", "needs_more_evidence", "human_review"}


def test_cost_of_matches_estimate():
    assert cost_of(1_000_000, 0) > 0
    assert cost_of(0, 1_000_000) > cost_of(1_000_000, 0)


def test_estimate_tokens_minimum_one():
    assert estimate_tokens("") == 1


def test_trace_renders_without_raising():
    t = Trace()
    t.handoff(1, "perception", "research", "D40 severe high")
    t.tool(2, "research", "query_repair_standards", {"hazard_class": "D40"}, {"n": 2})
    t.final(3, "decision", {"action": "urgent", "priority": 90})
    t.guard(4, "MAX_ITERATIONS (8) reached")
    t.print()
    rows = t.to_list()
    assert [r["kind"] for r in rows] == ["handoff", "tool", "final", "guard_fired"]
