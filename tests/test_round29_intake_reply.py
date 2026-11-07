"""Round 29 (6.11.2026 audit, F6): a worker's own final answer that arrives
as a tool call is its reply in the wrong envelope and is used as the reply,
whatever the tool's name; only another worker's final keys through a lookup
tool stay a misroute. The Intake agent is offered extract_measurements only
when the text carries a digit."""
import json

import roadguard_agent as ra


def _msg(content=None, tool_calls=None):
    return ra.ChatMessage(finish_reason="tool_calls" if tool_calls else "stop", content=content, tool_calls=tool_calls or [])


def _call(name, args, call_id="c1"):
    return {"id": call_id, "function": {"name": name, "arguments": json.dumps(args)}}


def _run(monkeypatch, agent, replies, registry):
    seen = []

    def fake_chat(messages, tools=None, temperature=0.0, use_panel=False, provider_override=None, **kw):
        seen.append(tools)
        return replies[len(seen) - 1], 10, 5

    monkeypatch.setattr(ra, "chat", fake_chat)
    out = ra._react_worker_rounds(
        agent_name=agent, system_prompt="s", user_payload={"x": 1}, tool_registry=registry,
        state={"hazard_id": "hz_t", "metadata": {}}, provider_override=None, max_hops=2,
    )
    return out, seen


INTAKE_REPLY = {"summary": "pothole near the bus stop", "contradictions": [], "urgency_claimed": False,
                "suspicious": False, "reporter_signals": {"size_cm": None, "depth_cm": None},
                "mentioned": {"hazard_class": "D40", "lane_blocked": None, "road_type": None}}


def test_intake_reply_sent_as_extract_measurements_is_the_reply(monkeypatch):
    """r23, 9 of 9 real reports: the model sent its final JSON as a call to
    extract_measurements and was told it looked like another tool's answer
    until the 90 s cap. Now: used as the reply, nothing runs, one call."""
    ran = []

    def measure(text: str) -> dict:
        ran.append(text)
        return {"ok": True}

    registry = {"extract_measurements": (measure, ra.SIGNAL_WORKER_TOOLS["extract_measurements"][1])}
    (parsed, _, _, info), seen = _run(monkeypatch, "signal_extractor",
                                      [_msg(tool_calls=[_call("extract_measurements", INTAKE_REPLY)])], registry)
    assert parsed == INTAKE_REPLY and len(seen) == 1 and ran == []
    assert info["calls"][0].get("reply_in_tool_call") is True


def test_profile_reply_through_a_lookup_tool_is_the_reply(monkeypatch):
    """Supersedes the round 18 retry: the r11 shape (final JSON as
    lookup_similar_cases) is the worker's own reply, used as such."""
    record = []
    reply = {"severity_signal": "severe", "severity_rationale": "D40 pothole", "confidence": "high",
             "needs_more_evidence": False, "missing_fields": [], "tools_used": []}

    def similar(location: str, hazard_class: str, state: dict | None = None) -> dict:
        record.append(1)
        return {"ok": True, "count": 0, "cases": []}

    registry = {"lookup_similar_cases": (similar, ra.PERCEPTION_WORKER_TOOLS["lookup_similar_cases"][1])}
    (parsed, _, _, info), seen = _run(monkeypatch, "profile_builder",
                                      [_msg(tool_calls=[_call("lookup_similar_cases", reply)])], registry)
    assert parsed == reply and len(seen) == 1 and record == []
    assert info["calls"][0].get("reply_in_tool_call") is True


def test_another_workers_reply_through_a_lookup_tool_is_still_rejected(monkeypatch):
    record = []
    foreign = {"coverage": "direct_match", "confidence": "high", "key_findings": ["x"], "cited_ids": ["p1"]}
    own = {"severity_signal": "minor", "severity_rationale": "D20", "confidence": "medium",
           "needs_more_evidence": False, "missing_fields": []}

    def similar(location: str, hazard_class: str, state: dict | None = None) -> dict:
        record.append(1)
        return {"ok": True, "count": 0, "cases": []}

    registry = {"lookup_similar_cases": (similar, ra.PERCEPTION_WORKER_TOOLS["lookup_similar_cases"][1])}
    replies = [_msg(tool_calls=[_call("lookup_similar_cases", foreign)]), _msg(content=json.dumps(own))]
    (parsed, _, _, info), seen = _run(monkeypatch, "profile_builder", replies, registry)
    assert parsed == own and len(seen) == 2 and record == []
    assert "look like another tool" in (info["calls"][0].get("error") or "")


def test_cross_tool_helper_fires_on_other_workers_keys_only():
    registry = ra.PERCEPTION_WORKER_TOOLS
    own = {"severity_signal": "minor", "severity_rationale": "D20 crack", "confidence": "medium",
           "needs_more_evidence": False}
    foreign = {"coverage": "direct_match", "key_findings": ["x"], "cited_ids": ["p1"]}
    assert ra._worker_tool_call_is_other_tools_reply("profile_builder", "lookup_similar_cases", own, registry) is False
    assert ra._worker_tool_call_is_other_tools_reply("profile_builder", "lookup_similar_cases", foreign, registry) is True
    assert ra._worker_final_in_tool_call("profile_builder", own, registry["lookup_similar_cases"][1]) is True


# ---------------------------------------------------------------- extract_measurements offered on digits only


def _capture_offer(monkeypatch, text):
    monkeypatch.setenv("AGENT_WORKER_REACT", "1")
    monkeypatch.setenv("AGENT_WORKER_AUTONOMY", "1")
    monkeypatch.setenv("AGENT_SIGNAL_LLM", "ollama:fake")
    got = {}

    def fake_worker(**kw):
        got.update(kw)
        return INTAKE_REPLY, 5, 2, {"hops": 0, "tools_run": {}, "calls": []}

    monkeypatch.setattr(ra, "_llm_react_worker", fake_worker)
    parsed, _, _, autonomy = ra._extract_by_llm(
        text, {"location": "x", "road_type": "main", "lane_blocked": "none"}, {}, state={})
    return got, autonomy


def test_text_without_a_digit_is_not_offered_the_measurement_tool(monkeypatch):
    got, autonomy = _capture_offer(monkeypatch, "deep hole near the bus stop")
    assert "extract_measurements" not in got["tool_registry"]
    assert "lookup_prior_reports" in got["tool_registry"]
    assert "call extract_measurements" not in got["system_prompt"]
    assert "size_cm and depth_cm are null" in got["system_prompt"]
    assert "required_not_called:extract_measurements" not in autonomy["violations"]


def test_text_with_a_digit_keeps_the_measurement_tool_required(monkeypatch):
    got, autonomy = _capture_offer(monkeypatch, "hole about 40 cm wide")
    assert "extract_measurements" in got["tool_registry"]
    assert "call extract_measurements" in got["system_prompt"]
    assert "required_not_called:extract_measurements" in autonomy["violations"]
