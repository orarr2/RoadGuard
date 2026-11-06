"""Round 10 (1.11.2026), the owner's loop improvements:
1. Decision receives the two deterministic tool results in its payload.
2. An evidence pass with the same photo and form keeps perception and research.
3. Tool schemas stay in the final Decision round (warm prompt cache); the
   schema descriptions and the payload are shorter.
5. The reason the rule tree decided, and the model's reply, land in the trace.
7. Worker LLM deltas and their effect on the rule tree's verdict are recorded.
"""
import json

import pytest

import roadguard_agent as ra


def _msg(content=None, tool_calls=None):
    return ra.ChatMessage(
        finish_reason="tool_calls" if tool_calls else "stop",
        content=content,
        tool_calls=tool_calls or [],
    )


def _call(name, call_id):
    return {"id": call_id, "function": {"name": name, "arguments": "{}"}}


_STATE = {
    "hazard_profile": {"primary_class": "D40", "primary_confidence": 0.5, "recurrence_count_30d": 1, "severity_signal": "moderate"},
    "metadata": {"road_type": "main", "lane_blocked": "partial", "location": "x"},
    "procedures": [{"source_url": "http://a", "issuing_body": "IRC", "article_ids": ["a1"], "quarantined_summary": {"trimmed_summary": "pothole repair"}}],
    "tool_result_ids": ["a1"],
    "dataset_match": {"found": True, "match": "sha256", "distance": 0, "pic_id": 5, "split": "train", "file_name": "f.jpg",
                      "classes": "D40", "n_labels": 2, "model_prediction": {"top1_class": "D40", "top1_confidence": 0.7}},
}
_HINT = {"action": "standard", "priority": 55, "rationale": "r" * 300, "cited_sources": ["a1", "a2", "a3", "a4"],
         "missing_evidence": [], "image_match_status": {"matched": True}, "image_matched": True}


def test_decision_payload_carries_the_deterministic_tool_results():
    payload = json.loads(ra._decision_messages(_STATE, _HINT)[1]["content"])
    sev = payload["severity_check"]
    expected = ra.check_hazard_severity("D40", 0.5, "main", "partial", 1)
    assert sev["tool"] == "check_hazard_severity" and sev["already_run"] is True
    assert sev["priority"] == expected["priority"] and sev["components"] == expected["components"]
    dm = payload["dataset_match"]
    assert dm["tool"] == "lookup_dataset_image" and dm["already_run"] is True
    assert dm["pic_id"] == 5 and dm["model_prediction"] == "D40 0.70"
    hint = payload["rule_based_hint"]
    assert set(hint) == {"action", "priority", "rationale", "cited_sources", "missing_evidence"}
    assert len(hint["rationale"]) == 240 and hint["cited_sources"] == ["a1", "a2", "a3"]
    assert "source_url" not in payload["procedures"][0]
    without_match = json.loads(ra._decision_messages({**_STATE, "dataset_match": None}, _HINT)[1]["content"])
    assert without_match["dataset_match"] is None
    # Round 13: no reporter text is said explicitly, not left as null.
    assert without_match["reporter_intake"] == {"text_present": False, "note": "no reporter text was sent: there are no reporter claims to weigh or mention"}
    assert "text_present is false" in ra._build_decision_prompt()


def test_decision_prompt_names_the_prefilled_results_and_schemas_are_shorter():
    prompt = ra._build_decision_prompt()
    assert "severity_check" in prompt and "already_run" in prompt
    schemas = ra._decision_tool_schemas()
    assert len(json.dumps(schemas)) < 3600  # 2,933 after round 10 (from 3,485); round 12 adds the why argument to each of the five
    by_name = {s["function"]["name"]: s["function"]["description"] for s in schemas}
    assert "already in the message" in by_name["check_hazard_severity"]
    assert "already in the message" in by_name["lookup_dataset_image"]
    assert by_name["check_hazard_severity"] and by_name["request_additional_evidence"]


def _react_state():
    return {"hazard_id": "hz_t", "tool_result_ids": [], "metadata": {}, "budget": None, "iterations": 0}


def _patch_react(monkeypatch, replies, seen):
    # The loop mechanics are checked with a two-hop cap; production runs
    # with the module's own cap (1 since the Decision hop cut).
    monkeypatch.setattr(ra, "DECISION_REACT_MAX_HOPS", 2)
    monkeypatch.setattr(ra, "_decision_messages", lambda state, hint: [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}])
    monkeypatch.setattr(ra, "_run_decision_tool", lambda name, args, state, tools_run: {"ok": True, "priority": 50})
    monkeypatch.setattr(ra, "_run_deadline", lambda: None)

    def fake_chat(messages, tools=None, temperature=0.0, **kw):
        seen.append((tools, list(messages)))
        return replies[len(seen) - 1], 10, 5

    monkeypatch.setattr(ra, "chat", fake_chat)


def test_decision_react_keeps_the_tools_in_the_final_round_and_nudges(monkeypatch):
    seen = []
    final = json.dumps({"action": "human_review", "rationale": "x", "priority": 50, "cited_sources": [], "missing_evidence": []})
    _patch_react(monkeypatch, [
        _msg(tool_calls=[_call("check_hazard_severity", "c1")]),
        _msg(tool_calls=[_call("lookup_dataset_image", "c2")]),
        _msg(content=final),
    ], seen)
    decision, _, _, info = ra._llm_decide_react(_react_state(), {"action": "standard"})
    assert decision["action"] == "human_review" and info["hops"] == 2
    assert len(seen) == 3
    assert all(tools is not None for tools, _ in seen), "the schemas stay in every round so the prompt prefix stays cached"
    last_user = [m for m in seen[2][1] if m.get("role") == "user"][-1]
    assert "budget" in last_user["content"] and "final JSON" in last_user["content"]
    assert "reply_kind" not in info


def test_decision_react_names_a_model_that_keeps_calling_tools(monkeypatch):
    seen = []
    _patch_react(monkeypatch, [
        _msg(tool_calls=[_call("check_hazard_severity", "c1")]),
        _msg(tool_calls=[_call("lookup_dataset_image", "c2")]),
        _msg(tool_calls=[_call("lookup_dataset_image", "c3")]),
        _msg(tool_calls=[_call("lookup_dataset_image", "c4")]),
    ], seen)
    decision, _, _, info = ra._llm_decide_react(_react_state(), {"action": "standard"})
    assert decision is None and len(seen) == 4
    assert info["reply_kind"] == "tool_call_after_budget"
    assert ra._llm_fallback_reason(info) == "llm_kept_calling_tools"
    budget_msgs = [c["error"] for c in info["calls"] if c.get("error")]
    assert budget_msgs and "budget exhausted" in budget_msgs[0]


def test_decision_react_records_unparseable_and_ungrounded_replies(monkeypatch):
    seen = []
    _patch_react(monkeypatch, [_msg(content="I think the road looks bad but I am not sure")], seen)
    decision, _, _, info = ra._llm_decide_react(_react_state(), {"action": "standard"})
    assert decision is None and info["reply_kind"] == "unparseable"
    assert info["reply_excerpt"].startswith("I think the road")
    assert ra._llm_fallback_reason(info) == "llm_reply_unparseable"

    seen2 = []
    ungrounded = json.dumps({"action": "standard", "rationale": "x", "priority": 50, "cited_sources": ["nope"], "missing_evidence": []})
    _patch_react(monkeypatch, [_msg(content=ungrounded)], seen2)
    decision2, _, _, info2 = ra._llm_decide_react(_react_state(), {"action": "standard"})
    assert decision2 is None and info2["reply_kind"] == "ungrounded"
    assert ra._llm_fallback_reason(info2) == "llm_reply_ungrounded"
    assert ra._llm_fallback_reason({"deadline_hit": True}) == "run_deadline"
    assert ra._llm_fallback_reason({"llm_failed": "TimeoutError"}) == "llm_call_failed:TimeoutError"
    assert ra._llm_fallback_reason(None) == "llm_unavailable_or_unparseable"


def _evidence_state(**overrides):
    md = {"image_sha256": "a" * 64, "lane_blocked": "none", "road_type": "main", "location": "Herzl 10"}
    state = {
        "hazard_profile": {"primary_class": "D40", "severity_signal": "moderate"},
        "procedures": [{"article_ids": ["a1"]}],
        "previous_yolo": {"top1_class": "D40", "top1_confidence": 0.5},
        "yolo_output": {"top1_class": "D40", "top1_confidence": 0.5},
        "previous_metadata": dict(md),
        "metadata": {**md, "location": "Herzl 10, near the bus stop"},
        "evidence_request": {"missing_field": "location"},
        "evidence_round": 0, "iterations": 3, "tokens_in": 100, "tokens_out": 20,
    }
    state.update(overrides)
    return state


def test_evidence_pass_reuses_perception_and_research_when_the_photo_is_unchanged():
    out = ra._await_evidence_node(_evidence_state())
    assert out["evidence_reuse"] is True
    assert out["trace"][0]["content"]["reused"] == ["profile_builder", "rag_retriever"]
    assert out["evidence_round"] == 1

    new_photo = _evidence_state(metadata={"image_sha256": "b" * 64, "lane_blocked": "none", "road_type": "main"})
    out2 = ra._await_evidence_node(new_photo)
    assert out2["evidence_reuse"] is False and out2["trace"][0]["content"]["reused"] == []

    new_class = _evidence_state(yolo_output={"top1_class": "D20", "top1_confidence": 0.5})
    assert ra._await_evidence_node(new_class)["evidence_reuse"] is False

    lane_changed = _evidence_state(metadata={"image_sha256": "a" * 64, "lane_blocked": "partial", "road_type": "main"})
    assert ra._await_evidence_node(lane_changed)["evidence_reuse"] is False

    no_profile = _evidence_state(hazard_profile={})
    assert ra._await_evidence_node(no_profile)["evidence_reuse"] is False


def test_supervisor_routes_a_reuse_pass_straight_to_decision():
    st = {"evidence_reuse": True, "last_speaker": "signal_extractor", "hazard_profile": {"primary_class": "D40"},
          "procedures": [], "next_hop": "profile_builder"}
    assert ra._route_from_supervisor(st) == "decision"
    assert ra._route_from_supervisor({**st, "evidence_reuse": False}) == "profile_builder"
    assert ra._route_from_supervisor({**st, "last_speaker": "decision"}) == "profile_builder"
    assert ra._route_from_supervisor({**st, "hazard_profile": {}}) == "profile_builder"


def test_worker_impact_compares_the_rule_tree_with_and_without_the_worker_llm(monkeypatch):
    def fake_rule(state):
        severe = (state.get("hazard_profile") or {}).get("severity_signal") == "severe"
        return {"action": "urgent" if severe else "standard"}

    monkeypatch.setattr(ra, "_decide_rule_based", fake_rule)
    state = {
        "hazard_profile": {"severity_signal": "severe"}, "perception_confidence": "high", "research_coverage": "direct_match",
        "worker_baseline": {
            "profile": {"severity_signal": "minor", "perception_confidence": "medium", "llm_changed": True},
            "research": {"coverage": "direct_match", "confidence": "high", "llm_changed": False},
        },
    }
    assert ra._worker_impact(state, {"action": "urgent"}) == {
        "profile_llm_changed": True, "research_llm_changed": False,
        "rule_hint_without_worker_llm": "standard", "rule_hint_with_worker_llm": "urgent", "changed_rule_hint": True,
    }
    assert ra._worker_impact({"hazard_profile": {"severity_signal": "severe"}}, {"action": "urgent"}) is None
    assert ra._label_delta({"a": 1, "b": 2}, {"a": 1, "b": 3}) == {"b": {"before": 2, "after": 3}}
    assert ra._label_delta({"a": 1}, {"a": 1}) == {}


def test_trace_lines_show_fallback_reply_worker_impact_deltas_and_reuse():
    dec = {"speaker": "decision", "kind": "final", "content": {
        "action": "standard", "priority": 50, "source": "rule_based", "rationale": "r", "cited_sources": [],
        "fallback_reason": "llm_reply_unparseable",
        "react": {"hops": 2, "calls": [], "reply_kind": "unparseable", "reply_excerpt": "I think the road"},
        "worker_impact": {"profile_llm_changed": True, "research_llm_changed": False,
                          "rule_hint_without_worker_llm": "standard", "rule_hint_with_worker_llm": "standard", "changed_rule_hint": False},
    }}
    text = "\n".join(ra._entry_lines(dec, []))
    assert "fallback_reason=llm_reply_unparseable" in text
    assert "model reply (unparseable): I think the road" in text
    assert "worker impact: profile_llm_changed=True  research_llm_changed=False  rule_hint without/with worker LLM: standard/standard" in text

    ev = {"speaker": "await_evidence", "kind": "evidence_received", "content": {
        "evidence_round": 1, "previous_request": {"missing_field": "second_angle"}, "delta": {},
        "reused": ["profile_builder", "rag_retriever"]}}
    assert "kept from the first pass, straight to Decision" in "\n".join(ra._entry_lines(ev, []))

    prof = {"speaker": "profile_builder", "kind": "handoff_out", "content": {
        "primary_class": "D40", "severity": "severe", "confidence": "medium", "llm_used": True,
        "llm_delta": {"severity_signal": {"before": "minor", "after": "severe"}}}}
    assert "llm changed: severity_signal minor -> severe" in "\n".join(ra._entry_lines(prof, []))
    rag = {"speaker": "rag_retriever", "kind": "handoff_out", "content": {
        "coverage": "direct_match", "n_procedures": 3, "tool_result_ids": ["a1"], "llm_used": True, "llm_delta": {}}}
    assert "llm changed" not in "\n".join(ra._entry_lines(rag, []))
