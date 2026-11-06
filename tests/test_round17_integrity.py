"""Round 17 (1.11.2026 evening): the owner's eight items from his r10 review.
Trace entries are stamped when they are created and the reporter's wait is
reported apart from compute; the trace shows every detection; the fact-check
learns class names, severity words, confidence words and time windows no
passage states; a failed fact-check gets one correction round and a verdict
more severe than the rule tree's is replaced; check_hazard_severity fills its
arguments from the state; lookup_similar_cases never returns the case being
triaged; the original photo re-sent is not evidence; priority 0 takes the
severity check; the research line says whether the passages suffice."""
import json

import pytest

import roadguard_agent as ra


def _msg(content=None, tool_calls=None):
    return ra.ChatMessage(
        finish_reason="tool_calls" if tool_calls else "stop", content=content, tool_calls=tool_calls or []
    )


def _state(**over):
    base = {
        "hazard_id": "hz_t",
        "metadata": {"lane_blocked": "none", "road_type": "main", "location": "batch test (x.jpg)"},
        "reporter_intake": {},
        "hazard_profile": {
            "primary_class": "D20",
            "primary_confidence": 0.9,
            "severity_signal": "severe",
            "secondary_classes": [{"class": "D40", "confidence": 0.457}],
            "recurrence_count_30d": 0,
        },
        "research_coverage": "direct_match",
        "research_coverage_detail": {
            "repair_window": False, "urgency": False, "traffic_control": False, "safety_risk": False,
            "answered": [], "missing": ["repair_window", "urgency", "traffic_control", "safety_risk"],
            "sufficiency": "insufficient_coverage",
        },
        "procedures": [],
        "tool_result_ids": [],
    }
    base.update(over)
    return base


# ---------------------------------------------------------------- timestamps


def test_stage_seconds_separate_the_reporters_wait_from_compute():
    t = 1000.0
    trace = [
        {"speaker": "sanitize", "kind": "boundary_defence", "ts": t, "content": {}},
        {"speaker": "profile_builder", "kind": "handoff_out", "ts": t + 20, "content": {}},
        {"speaker": "decision", "kind": "final", "ts": t + 50, "content": {"action": "needs_more_evidence"}},
        {"speaker": "await_evidence", "kind": "evidence_received", "ts": t + 250, "content": {"evidence_round": 1}},
        {"speaker": "decision", "kind": "final", "ts": t + 280, "content": {"action": "human_review"}},
    ]
    stages = ra._stage_seconds(trace)
    assert stages["waiting_evidence"] == 200.0 and "await_evidence" not in stages
    assert stages["profile_builder"] == 20.0 and stages["decision"] == 60.0
    lines = ra.format_trace_lines(
        "hz_t", trace, metadata={}, verdict={"action": "human_review", "priority": 50}, status="done"
    )
    done = [line for line in lines if "[DONE" in line][0]
    assert "elapsed=280s (compute 80s, waiting for evidence 200s)" in done
    stage_line = [line for line in lines if "stage seconds:" in line][0]
    assert "waiting_evidence=200" in stage_line and "await_evidence" not in stage_line
    assert "profile_builder=20" in stage_line and "decision=60" in stage_line


def test_node_entries_are_stamped_when_created():
    state = ra.initial_state(
        "hz_t",
        {"top1_class": "D40", "top1_confidence": 0.7, "detections": []},
        {},
        {"road_type": "main", "lane_blocked": "none", "location": "Herzl 5", "reporter_text": ""},
    )
    update = ra.sanitize_node(state)
    assert all(isinstance(e.get("ts"), float) for e in update["trace"])
    ev = ra._await_evidence_node({**state, "previous_metadata": {}, "hazard_profile": {}, "procedures": None})
    assert isinstance(ev["trace"][0]["ts"], float) and ev["trace"][0]["content"]["declined"] is False


# ---------------------------------------------------------------- detections


def test_model_line_lists_the_other_detections():
    md = {
        "class": "D20", "confidence": 0.885,
        "bboxes": json.dumps([[0, 74, 606, 448, "D20", 0.885], [98, 228, 150, 263, "D40", 0.457], [478, 82, 554, 143, "D00", 0.248]]),
    }
    lines = ra.format_trace_lines("hz_t", [], metadata=md, verdict=None, status="processing")
    model = [line for line in lines if "[MODEL" in line][0]
    assert model.endswith("source=yolo_live  others=D40 0.46,D00 0.25")
    assert ra._other_detections({"class": "D20", "confidence": 0.9, "bboxes": "[]"}) == ""


def test_profile_line_shows_other_boxes_and_the_reason_for_a_severity_change():
    entry = {"speaker": "profile_builder", "kind": "handoff_out", "content": {
        "primary_class": "D20", "severity": "minor", "confidence": "medium", "llm_used": True,
        "others": ["D40 0.46"],
        "llm_delta": {"severity_signal": {"before": "moderate", "after": "minor"}},
        "llm_change_reason": "thin crack, no spalling",
    }}
    text = "\n".join(ra._entry_lines(entry, []))
    assert "others=D40 0.46" in text
    assert "llm changed: severity_signal moderate -> minor  reason: thin crack, no spalling" in text
    assert ra._secondary_labels({"secondary_classes": [{"class": "D40", "confidence": 0.457}, {"class": None}]}) == ["D40 0.46"]


# ---------------------------------------------------------------- fact-check


def test_fact_check_catches_the_r10_wordings():
    dec = {
        "action": "standard", "priority": 60,
        "rationale": "D20 pothole on main road; routine repair queue based on secondary D40 class. No evidence of severe conditions.",
        "assessment": "The detector classifies the defect as D20. The pothole is on a main road.",
    }
    rc = ra._verify_decision_rationale(dec, _state())
    assert "class_name_mismatch:D20_pothole" in rc["rationale_problems"]
    assert "severity_text_minor_vs_profile_severe" in rc["rationale_problems"]
    # D40 is a secondary detection, so "pothole" alone is not a foreign name
    assert not any(p.startswith("class_name_not_in_inputs") for p in rc["problems"])
    assert rc["ok"] is False


def test_fact_check_confidence_words_and_repair_window():
    dec = {
        "action": "standard", "priority": 60,
        "rationale": "D20 alligator cracking at low detector confidence; the standard sets a 24 hour repair window.",
        "assessment": "",
    }
    rc = ra._verify_decision_rationale(dec, _state())
    assert "confidence_word_low_vs_0.90_high" in rc["rationale_problems"]
    assert "repair_window_not_in_sources" in rc["rationale_problems"]
    ok = {
        "action": "standard", "priority": 60,
        "rationale": "D20 alligator cracking at high detector confidence on a main road, no lane blockage; routine queue.",
        "assessment": "medium perception confidence; the standards found do not specify a repair window.",
    }
    assert ra._verify_decision_rationale(ok, _state())["ok"] is True
    minor = _state(hazard_profile={**_state()["hazard_profile"], "severity_signal": "minor"})
    bad = {"action": "standard", "priority": 60, "rationale": "D20 alligator cracking, severe damage.", "assessment": ""}
    assert "severity_text_severe_vs_profile_minor" in ra._verify_decision_rationale(bad, minor)["rationale_problems"]


def test_confidence_band_follows_the_policy_thresholds():
    assert ra._confidence_band(0.3) == "low"
    assert ra._confidence_band(0.5) == "moderate"
    assert ra._confidence_band(0.9) == "high"
    assert ra._confidence_band(None) == "unknown"
    assert ra._confidence_word_problems("high detector confidence", 0.3) == ["confidence_word_high_vs_0.30_low"]
    assert ra._confidence_word_problems("medium perception confidence", 0.3) == []


def test_supervisor_note_confidence_adjective_is_checked_and_the_band_travels():
    payload = {"hazard_class_detected": "D20", "detector_confidence": 0.9, "road_type": "main",
               "lane_blocked": "none", "has_reporter_text": False}
    note, check = ra._verify_supervisor_note(payload, "Hazard class D20, low detector confidence (0.9), no reporter text.")
    assert check == "replaced:detector_confidence_band" and note.startswith("D20 at 0.90")
    _note2, check2 = ra._verify_supervisor_note(payload, "Hazard class D20, high detector confidence (0.9), no reporter text.")
    assert check2 == "ok"


# ---------------------------------------------------------------- payloads


def test_decision_payload_carries_the_band_and_the_sufficiency_note():
    msgs = ra._decision_messages(
        _state(), {"action": "standard", "priority": 60, "rationale": "r", "cited_sources": [], "missing_evidence": []}
    )
    payload = json.loads(msgs[1]["content"])
    assert payload["detector_confidence_band"] == "high"
    assert payload["research_coverage_detail"]["sufficiency"] == "insufficient_coverage"
    assert "do not state one" in payload["research_coverage_detail"]["note"]
    assert "insufficient_coverage" in ra.DECISION_SYSTEM_PROMPT
    assert "detector_confidence_band" in ra.DECISION_SYSTEM_PROMPT
    assert ra._coverage_detail_for_prompt(None) is None
    sufficient = ra._coverage_detail_for_prompt({"answered": ["repair_window", "urgency"], "missing": [], "sufficiency": "sufficient_coverage"})
    assert "note" not in sufficient


def test_decision_coverage_sufficiency_and_the_rag_line():
    procs = [{"passage": "Potholes on arterial roads shall be repaired within 24 hours and treated as urgent."}]
    dc = ra._decision_coverage(procs)
    assert dc["sufficiency"] == "sufficient_coverage" and "repair_window" in dc["answered"]
    thin = ra._decision_coverage([{"passage": "Crack sealing uses rubberised bitumen."}])
    assert thin["sufficiency"] == "insufficient_coverage"
    rag = {"speaker": "rag_retriever", "kind": "handoff_out", "content": {
        "coverage": "direct_match", "n_procedures": 1, "tool_result_ids": ["a"], "llm_used": False, "decision_coverage": dc}}
    assert "sufficiency=sufficient_coverage" in "\n".join(ra._entry_lines(rag, []))
    sentence = ra._deterministic_rationale({"action": "standard"}, _state())
    assert "research coverage direct_match, insufficient_coverage" in sentence


# ---------------------------------------------------------------- tool contracts


def test_severity_tool_fills_missing_arguments_from_the_state_and_still_rejects_wrong_values():
    state = {"hazard_id": "hz_t", "metadata": {"road_type": "main", "lane_blocked": "none"},
             "hazard_profile": {"primary_class": "D40", "primary_confidence": 0.5, "recurrence_count_30d": 0}}
    res = ra._run_decision_tool("check_hazard_severity", {}, state, {})
    assert res["ok"] is True
    assert res["priority"] == ra.check_hazard_severity("D40", 0.5, "main", "none")["priority"]
    assert set(res["args_from_state"]) == {"hazard_class", "confidence", "road_type", "lane_blocked", "recurrence_count_30d"}
    partial = ra._run_decision_tool("check_hazard_severity", {"hazard_class": "D40", "confidence": 0.5}, state, {})
    assert partial["ok"] is True and set(partial["args_from_state"]) == {"road_type", "lane_blocked", "recurrence_count_30d"}
    bad = ra._run_decision_tool("check_hazard_severity", {"hazard_class": "D99"}, state, {})
    assert bad["ok"] is False and "hazard_class must be one of" in bad["error"]
    empty = ra._run_decision_tool("check_hazard_severity", {}, {"hazard_id": "hz_t", "metadata": {}}, {})
    assert empty["ok"] is False and "hazard_class is required" in empty["error"]
    step = {"speaker": "decision", "kind": "react_step", "content": {
        "hop": 1, "tool": "check_hazard_severity", "args": {}, "ok": True, "result": res}}
    assert "args from state: hazard_class, confidence, road_type, lane_blocked, recurrence_count_30d" in "\n".join(ra._entry_lines(step, []))


# ---------------------------------------------------------------- similar cases


def test_similar_cases_exclude_the_current_hazard_and_name_the_outcome(monkeypatch):
    seen = {}

    def fake_hits(location, hazard_class, window_days, k, exclude_hazard_id):
        seen["exclude"] = exclude_hazard_id
        return [{"id": "hz_other", "days_ago": 3.0}]

    monkeypatch.setattr(ra, "_hits_from_db", fake_hits)
    res = ra._tool_lookup_similar_cases_worker("Herzl 5, Holon", "D40", state={"hazard_id": "hz_me", "metadata": {}})
    assert seen["exclude"] == "hz_me" and res["note"] == "prior_cases" and res["count"] == 1
    prior = ra._tool_lookup_prior_reports("Herzl 5, Holon", "D40", 3, state={"hazard_id": "hz_me"})
    assert seen["exclude"] == "hz_me" and prior["note"] == "prior_cases"
    monkeypatch.setattr(ra, "_hits_from_db", lambda *a, **k: [])
    assert ra.lookup_similar_cases("Herzl 5, Holon", "D40")["note"] == "no_prior_cases"
    monkeypatch.setattr(ra, "_hits_from_db", lambda *a, **k: None)
    assert ra.lookup_similar_cases("Herzl 5, Holon", "D40")["note"] == "lookup_failed"
    assert ra.lookup_similar_cases("Tel Aviv", "D40")["note"] == "location_too_coarse"

    class Mem:
        def find_similar(self, **kw):
            return [{"hazard_id": "hz_me", "days_ago": 0.0}, {"hazard_id": "hz_other", "days_ago": 2.0}]

    res = ra.lookup_similar_cases("Herzl 5, Holon", "D40", vector_mem=Mem(), exclude_hazard_id="hz_me")
    assert [c["hazard_id"] for c in res["cases"]] == ["hz_other"] and res["note"] == "prior_cases"


# ---------------------------------------------------------------- evidence


def test_identical_evidence_is_refused_and_a_note_means_declined(monkeypatch):
    prev = {"image_sha256": "a" * 64, "road_type": "main", "lane_blocked": "none", "reporter_text": ""}
    assert ra._evidence_identical(prev, dict(prev)) == (True, False)
    assert ra._evidence_identical(prev, {**prev, "reporter_text": "No other photo is available."}) == (True, True)
    assert ra._evidence_identical(prev, {**prev, "image_sha256": "b" * 64}) == (False, False)
    assert ra._evidence_identical(prev, {**prev, "lane_blocked": "partial"}) == (False, False)

    updates = {}

    class FakeApp:
        def get_state(self, config):
            return type("S", (), {"values": {"metadata": prev, "yolo_output": {"top1_class": "D40"}}})()

        def update_state(self, config, values):
            updates.update(values)

        def invoke(self, _inputs, config=None):
            return {"invoked": True}

    monkeypatch.setattr(ra, "build_graph", lambda: FakeApp())
    monkeypatch.setattr(ra, "thread_waits_for_evidence", lambda hid: True)
    with pytest.raises(ra.IdenticalEvidenceError):
        ra.resume_graph("hz_t", {}, {}, dict(prev))
    assert updates == {}
    ra.resume_graph("hz_t", {}, {}, {**prev, "reporter_text": "No other photo is available."})
    assert updates["evidence_declined"] is True and updates["previous_metadata"] == prev
    updates.clear()
    ra.resume_graph("hz_t", {}, {}, {**prev, "image_sha256": "b" * 64})
    assert updates["evidence_declined"] is False


def test_declined_evidence_goes_to_a_human_without_a_model_pass(monkeypatch):
    calls = []
    monkeypatch.setattr(ra, "_llm_decide_react", lambda *a, **k: calls.append(1) or (None, 0, 0, {}))
    monkeypatch.setattr(ra, "_react_enabled", lambda: True)
    state = ra.initial_state(
        "hz_t", {"top1_class": "D40", "top1_confidence": 0.5, "detections": []}, {},
        {"road_type": "main", "lane_blocked": "none", "location": "batch test (x.jpg)", "reporter_text": "No other photo."},
    )
    state.update({
        "hazard_profile": {"primary_class": "D40", "primary_confidence": 0.5, "severity_signal": "moderate",
                           "secondary_classes": [], "recurrence_count_30d": 0},
        "procedures": [], "tool_result_ids": [], "research_coverage": "no_match",
        "perception_confidence": "medium", "evidence_declined": True, "evidence_round": 1,
    })
    update = ra.decision_node(state)
    assert update["action"] == "human_review" and update["fallback_reason"] == "evidence_declined"
    assert calls == []
    final = [e for e in update["trace"] if e["kind"] == "final"][0]
    assert final["content"]["source"] == "rule_based" and isinstance(final["ts"], float)
    assert "could not provide another photo" in final["content"]["rationale"]
    ev = {"speaker": "await_evidence", "kind": "evidence_received", "content": {
        "evidence_round": 1, "previous_request": {"missing_field": "second_angle"}, "delta": {}, "declined": True}}
    assert "not new evidence" in "\n".join(ra._entry_lines(ev, []))


# ---------------------------------------------------------------- correction and enforcement


def test_fact_check_failure_gets_one_correction_round(monkeypatch):
    reply = json.dumps({
        "action": "standard", "priority": 60,
        "rationale": "D20 alligator cracking on a main road, no lane blockage; routine queue.",
        "cited_sources": ["a1"], "missing_evidence": [], "assessment": "Routine repair.", "reasoning": "x",
    })
    grounded = _state(
        tool_result_ids=["a1"],
        procedures=[{"article_ids": ["a1"], "passage": "alligator crack sealing", "relevance": {"class_keyword_hit": True}}],
    )
    seen = {}

    def fake_chat(messages, tools=None, temperature=0.0, use_panel=False, **kw):
        seen["messages"] = messages
        seen["tools"] = tools
        return _msg(content=reply), 10, 5

    monkeypatch.setattr(ra, "chat", fake_chat)
    monkeypatch.setattr(ra, "_deadline_blocks_hop", lambda d: False)
    info = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "{}"}],
            "reply_text": "{first}", "schemas": [{"function": {"name": "x"}}]}
    corrected, i, o, status = ra._correct_decision(
        grounded, info, ["lane_blockage_not_in_form", "class_name_mismatch:D20_pothole", "confidence_word_low_vs_0.90_high"]
    )
    assert status == "ok" and corrected["action"] == "standard" and (i, o) == (10, 5)
    user = seen["messages"][-1]["content"]
    assert "lane_blocked=none" in user and "D20 alligator crack" in user and "band high" in user
    assert seen["messages"][-2] == {"role": "assistant", "content": "{first}"}
    assert seen["tools"] == info["schemas"]
    assert ra._correct_decision(_state(), {}, ["x"])[3] == "no_conversation"
    monkeypatch.setattr(ra, "_deadline_blocks_hop", lambda d: True)
    assert ra._correct_decision(_state(), info, ["x"])[3] == "run_deadline"


def test_unverified_verdict_more_severe_than_the_tree_is_replaced():
    rule = {"action": "standard", "priority": 57, "rationale": "routine queue", "cited_sources": ["a"], "missing_evidence": []}
    assert ra._enforce_unverified({"action": "urgent", "priority": 80}, rule)["kind"] == "rule_tree_verdict"
    assert ra._enforce_unverified({"action": "standard", "priority": 75}, rule)["kind"] == "rule_tree_verdict"
    assert ra._enforce_unverified({"action": "standard", "priority": 60}, rule)["kind"] == "kept"
    assert ra._enforce_unverified({"action": "human_review", "priority": 57}, rule)["kind"] == "kept"
    assert ra._enforce_unverified({"action": "needs_more_evidence", "priority": 40}, rule)["kind"] == "kept"


def _decision_state(reporter_text=""):
    state = ra.initial_state(
        "hz_t", {"top1_class": "D40", "top1_confidence": 0.7, "detections": []}, {},
        {"road_type": "main", "lane_blocked": "none", "location": "batch test (x.jpg)", "reporter_text": reporter_text},
    )
    state.update({
        "hazard_profile": {"primary_class": "D40", "primary_confidence": 0.7, "severity_signal": "moderate",
                           "secondary_classes": [], "recurrence_count_30d": 0},
        "procedures": [{"article_ids": ["a1"], "passage": "pothole repair on main roads", "relevance": {"class_keyword_hit": True}}],
        "tool_result_ids": ["a1"], "research_coverage": "direct_match",
        "research_coverage_detail": {"repair_window": False, "answered": [], "missing": ["repair_window"], "sufficiency": "insufficient_coverage"},
        "perception_confidence": "medium", "reporter_intake": {},
    })
    return state


def _stub_side_effect_tools(monkeypatch):
    for name, result in (
        ("create_draft_work_order", {"ok": True, "work_order_id": "wo_t"}),
        ("request_additional_evidence", {"ok": True, "request_id": "er_t"}),
    ):
        schema = ra.DECISION_TOOLS[name][1]
        monkeypatch.setitem(ra.DECISION_TOOLS, name, (lambda result=result, **kw: result, schema))


def test_decision_node_replaces_an_unverified_verdict_that_is_more_severe_than_the_tree(monkeypatch):
    _stub_side_effect_tools(monkeypatch)
    monkeypatch.setattr(ra, "_react_enabled", lambda: True)
    monkeypatch.setattr(ra, "_deadline_blocks_hop", lambda d: False)
    bad = {
        "action": "urgent", "priority": 90,
        "rationale": "D40 pothole partially blocking the lane on a main road; dispatch now.",
        "cited_sources": ["a1"], "missing_evidence": [], "assessment": "Dispatch.",
    }
    info = {"tools_run": {}, "hops": 0, "calls": [], "steps": [],
            "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "{}"}],
            "reply_text": json.dumps(bad), "schemas": None}
    monkeypatch.setattr(ra, "_llm_decide_react", lambda state, rule_hint: (dict(bad), 100, 20, info))
    # the correction round answers with the same unsupported claim
    monkeypatch.setattr(ra, "chat", lambda messages, tools=None, temperature=0.0, use_panel=False, **kw: (_msg(content=json.dumps(bad)), 30, 10))
    update = ra.decision_node(_decision_state())
    final = [e for e in update["trace"] if e["kind"] == "final"][0]["content"]
    assert update["action"] == "standard" and update["fallback_reason"] == "rationale_unverified"
    assert final["source"] == "rule_based" and final["floor_applied"] is True
    rc = final["rationale_check"]
    assert rc["ok"] is False and rc["enforcement"] == "rule_tree_verdict"
    assert rc["correction"]["status"] == "ok" and rc["correction"]["recheck_ok"] is False
    assert rc["correction"]["first_verdict"] == {"action": "urgent", "priority": 90}
    assert final["overridden_llm_verdict"]["action"] == "urgent"
    assert update["tokens_in"] == 130 and update["tokens_out"] == 30
    text = "\n".join(ra._entry_lines({"speaker": "decision", "kind": "final", "content": final}, []))
    assert "rationale check: FAIL" in text and "correction round ok" in text
    assert "more severe than the rule tree's and was replaced" in text


def test_decision_node_keeps_a_corrected_verdict(monkeypatch):
    _stub_side_effect_tools(monkeypatch)
    monkeypatch.setattr(ra, "_react_enabled", lambda: True)
    monkeypatch.setattr(ra, "_deadline_blocks_hop", lambda d: False)
    bad = {"action": "standard", "priority": 60,
           "rationale": "D40 pothole; the reporter's claims match the photo.",
           "cited_sources": ["a1"], "missing_evidence": [], "assessment": "Routine."}
    good = {**bad, "rationale": "D40 pothole on a main road at high detector confidence, no reporter text; routine queue.", "reasoning": "x"}
    info = {"tools_run": {}, "hops": 0, "calls": [], "steps": [],
            "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "{}"}],
            "reply_text": json.dumps(bad), "schemas": None}
    monkeypatch.setattr(ra, "_llm_decide_react", lambda state, rule_hint: (dict(bad), 100, 20, info))
    monkeypatch.setattr(ra, "chat", lambda messages, tools=None, temperature=0.0, use_panel=False, **kw: (_msg(content=json.dumps(good)), 30, 10))
    # Sufficient coverage: the round 25 RAG floor must not turn the
    # corrected standard verdict into needs_more_evidence here.
    state = _decision_state()
    state["research_coverage_detail"] = {"repair_window": True, "answered": ["repair_window"], "missing": [], "sufficiency": "sufficient"}
    update = ra.decision_node(state)
    final = [e for e in update["trace"] if e["kind"] == "final"][0]["content"]
    assert update["action"] == "standard" and final["source"] == "llm_react" and "fallback_reason" not in update
    rc = final["rationale_check"]
    assert rc["ok"] is True and rc["corrected"] is True
    assert rc["correction"]["first_problems"] == ["reporter_claims_without_text"]
    assert final["rationale"].startswith("D40 pothole on a main road at high detector confidence")
    text = "\n".join(ra._entry_lines({"speaker": "decision", "kind": "final", "content": final}, []))
    assert "corrected=1" in text and "rationale check: ok after one correction round (first reply failed: reporter_claims_without_text)" in text


def test_model_priority_zero_takes_the_severity_check(monkeypatch):
    _stub_side_effect_tools(monkeypatch)
    monkeypatch.setattr(ra, "_react_enabled", lambda: True)
    monkeypatch.setattr(ra, "_deadline_blocks_hop", lambda d: False)
    zero = {"action": "needs_more_evidence", "priority": 0,
            "rationale": "D40 pothole at high detector confidence; a second angle is needed.",
            "cited_sources": [], "missing_evidence": ["second_angle"], "assessment": "Ask for another photo."}
    info = {"tools_run": {"request_additional_evidence": {"request_id": "er_t", "missing_field": "second_angle", "message_to_reporter": "Another angle please."}},
            "hops": 1, "calls": [], "steps": [], "messages": [], "reply_text": json.dumps(zero), "schemas": None}
    monkeypatch.setattr(ra, "_llm_decide_react", lambda state, rule_hint: (dict(zero), 100, 20, info))
    # 6.11.2026: outside the ambiguous band (0.7) the rule tree says standard,
    # so the model's needs_more_evidence at priority 0 is what reaches the
    # cap: the priority comes from the severity check and the verdict becomes
    # human_review (request cap 0, the reporter is not asked for photos).
    state = _decision_state()
    state["hazard_profile"]["primary_confidence"] = 0.7
    state["yolo_output"]["top1_confidence"] = 0.7
    update = ra.decision_node(state)
    expected = ra.check_hazard_severity("D40", 0.7, "main", "none")["priority"]
    assert update["action"] == "human_review" and update["priority"] == expected and expected > 0
    final = [e for e in update["trace"] if e["kind"] == "final"][0]["content"]
    assert final["priority_from_severity_check"] is True
    assert final["evidence_cap_applied"] is True
    assert "priority_from=severity_check" in "\n".join(ra._entry_lines({"speaker": "decision", "kind": "final", "content": final}, []))
