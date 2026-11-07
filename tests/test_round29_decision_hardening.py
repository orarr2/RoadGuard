"""Round 29 (6.11.2026 audit, F9, F10, F16, F18): Decision sends no tool
schemas below a two-hop cap and repairs a reply that is not valid JSON as
a whole; the supervisor's note reaches Decision, "with high confidence"
is a confidence word and the supervisor call is costed; a priority of 0
is filled after the correction round too; a tool counts as agent value
only when it returned something."""
import json

import roadguard_agent as ra


def _msg(content=None, tool_calls=None):
    return ra.ChatMessage(finish_reason="tool_calls" if tool_calls else "stop", content=content, tool_calls=tool_calls or [])


# ---------------------------------------------------------------- F10: JSON repair


def test_json_repair_cuts_prose_and_closes_what_the_model_left_open():
    assert ra._parse_json_object('Sure, here is the verdict: {"action": "standard", "priority": 40} thanks') == {"action": "standard", "priority": 40}
    assert ra._parse_json_object('{"action": "standard", "priority": 40, "cited_sources": ["p1",],}') == {"action": "standard", "priority": 40, "cited_sources": ["p1"]}
    assert ra._parse_json_object('{"action": "standard", "cited_sources": ["p1"') == {"action": "standard", "cited_sources": ["p1"]}
    assert ra._parse_json_object('{"action": "standard", "rationale": "cut off mid sen') == {"action": "standard", "rationale": "cut off mid sen"}
    assert ra._parse_json_object("no json here") is None
    assert ra._parse_json_object("[1, 2]") is None
    assert ra._parse_json_object('{"action": "standard"}') == {"action": "standard"}


def test_decision_json_inside_prose_and_an_unclosed_fence_parses():
    text = 'Final answer:\n```json\n{"action": "human_review", "priority": 50, "rationale": "r", "cited_sources": [], "missing_evidence": []\n```'
    d = ra._parse_decision_json(text)
    assert d is not None and d["action"] == "human_review" and d["priority"] == 50


# ---------------------------------------------------------------- F10: schemas only with two hops


def _react_state():
    return {"hazard_id": "hz_t", "tool_result_ids": [], "metadata": {}, "budget": None, "iterations": 0}


def _patch(monkeypatch, cap, seen, replies):
    monkeypatch.setattr(ra, "DECISION_REACT_MAX_HOPS", cap)
    monkeypatch.setattr(ra, "_decision_messages", lambda state, hint: [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}])
    monkeypatch.setattr(ra, "_run_decision_tool", lambda name, args, state, tools_run: {"ok": True, "priority": 50})
    monkeypatch.setattr(ra, "_run_deadline", lambda: None)

    def fake_chat(messages, tools=None, temperature=0.0, **kw):
        seen.append(tools)
        return replies[len(seen) - 1], 10, 5

    monkeypatch.setattr(ra, "chat", fake_chat)


FINAL = json.dumps({"action": "human_review", "rationale": "x", "priority": 50, "cited_sources": [], "missing_evidence": []})


def test_no_tool_schemas_below_a_two_hop_cap(monkeypatch):
    seen = []
    _patch(monkeypatch, 1, seen, [_msg(content=FINAL)])
    decision, _, _, info = ra._llm_decide_react(_react_state(), {"action": "standard"})
    assert decision["action"] == "human_review"
    assert seen == [None] and info["tools_offered"] == 0


def test_tool_schemas_stay_with_two_hops(monkeypatch):
    seen = []
    _patch(monkeypatch, 2, seen, [_msg(content=FINAL)])
    decision, _, _, info = ra._llm_decide_react(_react_state(), {"action": "standard"})
    assert decision["action"] == "human_review"
    assert seen[0] and info["tools_offered"] == len(seen[0]) > 0


# ---------------------------------------------------------------- F9: the supervisor note, the regex, the cost


def _decision_state():
    state = ra.initial_state(
        "hz_t", {"top1_class": "D40", "top1_confidence": 0.7, "detections": []}, {},
        {"road_type": "main", "lane_blocked": "none", "location": "x", "reporter_text": ""},
    )
    state.update({
        "hazard_profile": {"primary_class": "D40", "primary_confidence": 0.7, "severity_signal": "moderate",
                           "secondary_classes": [], "recurrence_count_30d": 0},
        "procedures": [{"article_ids": ["a1"], "passage": "pothole repair on main roads", "relevance": {"class_keyword_hit": True}}],
        "tool_result_ids": ["a1"], "research_coverage": "direct_match",
        "research_coverage_detail": {"answered": ["repair_window", "urgency"], "missing": [], "sufficiency": "sufficient_coverage"},
        "perception_confidence": "medium", "reporter_intake": {},
    })
    return state


def test_decision_payload_carries_the_supervisor_note():
    state = _decision_state()
    state["supervisor_plan"] = {"pipeline_concern": "input_ambiguous", "expected_difficulty": "high",
                                "note": "detector_confidence_band is high, no reporter text", "force_action": None}
    hint = {"action": "standard", "priority": 40, "rationale": "r", "cited_sources": [], "missing_evidence": []}
    payload = json.loads(ra._decision_messages(state, hint)[1]["content"])
    assert payload["supervisor_note"] == {"pipeline_concern": "input_ambiguous", "expected_difficulty": "high",
                                          "note": "detector_confidence_band is high, no reporter text"}
    without = json.loads(ra._decision_messages(_decision_state(), hint)[1]["content"])
    assert without["supervisor_note"] is None
    assert "supervisor_note" in ra._build_decision_prompt()


def test_with_high_confidence_is_a_confidence_word():
    assert ra._confidence_word_problems("D40 detected with high confidence", 0.0) == ["confidence_word_high_vs_0.00_low"]
    assert ra._confidence_word_problems("D40 detected with high confidence", 0.9) == []
    assert ra._confidence_word_problems("detected with very low confidence", 0.9) == ["confidence_word_very_low_vs_0.90_high"]


def test_supervisor_call_is_recorded_for_the_run_cost(monkeypatch):
    monkeypatch.setattr(ra, "worker_react_enabled", lambda: True)
    monkeypatch.setattr(ra, "worker_provider", lambda name: "ollama:fake")

    def fake_worker(**kw):
        ra._LLM_LOG.calls.append({"ok": True, "model": "ollama:fake", "latency_s": 4.0, "tokens_in": 100, "tokens_out": 20})
        return ({"pipeline_concern": "none", "note": "detector_confidence_band is high", "expected_difficulty": "low",
                 "force_action": None}, 100, 20, {"hops": 0})

    monkeypatch.setattr(ra, "_llm_react_worker", fake_worker)
    upd = ra._maybe_run_supervisor_agent({
        "yolo_output": {"top1_class": "D40", "top1_confidence": 0.9},
        "metadata": {"road_type": "main", "lane_blocked": "none", "reporter_text": ""},
        "iterations": 0, "trace": [],
    })
    entry = upd["trace"][-1]
    assert entry["kind"] == "supervisor_plan"
    assert entry["content"]["llm"]["calls"] == 1 and entry["content"]["llm"]["tokens_in"] == 100
    assert "llm" not in upd["supervisor_plan"], "the state copy stays the plan itself"
    cost = ra._compute_run_cost(upd["trace"])
    assert cost["llm_calls"] == 1 and cost["tokens_in"] == 100 and cost["tokens_out"] == 20


# ---------------------------------------------------------------- F16: priority 0 after the correction round


def test_zero_priority_is_filled_from_the_severity_check():
    state = {"hazard_profile": {"primary_class": "D00", "primary_confidence": 0.45, "recurrence_count_30d": 0},
             "metadata": {"road_type": "main", "lane_blocked": "none"}}
    d = {"action": "human_review", "priority": 0}
    assert ra._fill_zero_priority(d, state) is True and d["priority"] == 32
    sealed = {"action": "known_image", "priority": 0}
    assert ra._fill_zero_priority(sealed, state) is False and sealed["priority"] == 0
    kept = {"action": "standard", "priority": 40}
    assert ra._fill_zero_priority(kept, state) is False and kept["priority"] == 40


def _stub_side_effect_tools(monkeypatch):
    for name, result in (
        ("create_draft_work_order", {"ok": True, "work_order_id": "wo_t"}),
        ("request_additional_evidence", {"ok": True, "request_id": "er_t"}),
    ):
        schema = ra.DECISION_TOOLS[name][1]
        monkeypatch.setitem(ra.DECISION_TOOLS, name, (lambda result=result, **kw: result, schema))


def test_corrected_reply_with_priority_zero_gets_the_check_priority(monkeypatch):
    """r23 hz_a7a8f0c54b02: the first reply's 0 was filled, the correction
    round's reply carried 0 back in and the trace said
    priority_from=severity_check next to priority=0."""
    _stub_side_effect_tools(monkeypatch)
    monkeypatch.setattr(ra, "_react_enabled", lambda: True)
    monkeypatch.setattr(ra, "_deadline_blocks_hop", lambda d: False)
    first = {"action": "standard", "priority": 0,
             "rationale": "The reporter's claims are consistent with a pothole on the main road.",
             "cited_sources": ["a1"], "missing_evidence": [], "assessment": ""}
    info = {"tools_run": {}, "hops": 0, "calls": [], "steps": [],
            "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "{}"}],
            "reply_text": json.dumps(first), "schemas": None}
    monkeypatch.setattr(ra, "_llm_decide_react", lambda state, rule_hint: (dict(first), 100, 20, info))
    corrected = {"action": "standard", "priority": 0, "rationale": "D40 pothole on a main road, lane none; routine queue.",
                 "cited_sources": ["a1"], "missing_evidence": [], "assessment": ""}
    monkeypatch.setattr(ra, "_correct_decision", lambda state, react_info, problems: (dict(corrected), 10, 5, "ok"))
    upd = ra.decision_node(_decision_state())
    final = [e for e in upd["trace"] if e.get("speaker") == "decision" and e.get("kind") == "final"][-1]
    expected = ra.check_hazard_severity("D40", 0.7, "main", "none", 0)["priority"]
    assert final["content"]["priority"] == expected and expected > 0
    assert upd["priority"] == expected


# ---------------------------------------------------------------- F18: a tool counts only with a result


def test_tool_results_are_judged_by_their_content():
    assert ra._tool_result_informative(json.dumps({"ok": True, "count": 0, "cases": []})) is False
    assert ra._tool_result_informative(json.dumps({"ok": True, "n": 0, "lessons": []})) is False
    assert ra._tool_result_informative({"ok": False, "error": "x"}) is False
    assert ra._tool_result_informative(json.dumps({"ok": True, "count": 2, "cases": [{"id": "hz_1"}]})) is True
    assert ra._tool_result_informative({"ok": True, "quality_verdict": "high", "concerns": []}) is True
    assert ra._tool_result_informative("") is False


def test_profile_tools_count_only_with_a_result():
    handoff = {"primary_class": "D40", "llm_used": True, "llm_delta": {},
               "autonomy": {"hops": 2, "tools_used": ["lookup_similar_cases", "recall_agent_notes"], "violations": []}}
    empty_steps = [
        {"hop": 1, "tool": "lookup_similar_cases", "ok": True, "result": json.dumps({"ok": True, "count": 0, "cases": []})},
        {"hop": 2, "tool": "recall_agent_notes", "ok": True, "result": json.dumps({"ok": True, "n": 0, "lessons": []})},
    ]
    out = ra._justify_profile_builder(empty_steps + [handoff])
    assert out["justified"] is False
    assert "tools_returned_nothing" in out["rebuttals"] and out["reasons"] == ["tools_returned_nothing"]
    full = [{"hop": 1, "tool": "lookup_similar_cases", "ok": True,
             "result": json.dumps({"ok": True, "count": 2, "cases": [{"id": "hz_1"}]})}]
    out2 = ra._justify_profile_builder(full + [handoff])
    assert out2["justified"] is True and "agent_reasoned_with_tools" in out2["reasons"]


def test_rag_queries_count_only_with_passages():
    handoff = {"coverage": "partial_match", "n_procedures": 0, "decision_coverage": {},
               "autonomy": {"hops": 1, "tools_used": ["query_repair_standards"]}}
    empty = [{"hop": 1, "tool": "query_repair_standards", "ok": True,
              "result": json.dumps({"ok": True, "procedures": [], "tool_result_ids": []})}]
    assert ra._justify_rag_retriever(empty + [handoff])["justified"] is False
    full = [{"hop": 1, "tool": "query_repair_standards", "ok": True,
             "result": json.dumps({"ok": True, "procedures": [{"passage": "x"}], "tool_result_ids": ["p1"]})}]
    assert "agent_chose_queries" in ra._justify_rag_retriever(full + [handoff])["reasons"]
