"""Worker autonomy (AGENT_WORKER_AUTONOMY=1): information gap, declared
and verified tool use, time cap, shared lessons (agent_notes)."""

from __future__ import annotations

import json
import time
import types

import roadguard_agent as ra
import roadguard_api as api

PASSAGE = {
    "passage": "Pothole repair: patch within 48 hours on main roads",
    "article_ids": ["p1"], "hazard_types": ["D40"], "road_types": ["main"],
    "issuing_body": "x", "source_url": "", "date": "", "distance": 0.4,
    "relevance": {"class_keyword_hit": True, "repair_term_hit": True},
}


def _hazard(hid: str, status: str = "open") -> ra.Hazard:
    return ra.Hazard(
        id=hid, user_id="u1", reported_at=time.time(), hazard_class="D40", confidence=0.7,
        road_type="main", lane_blocked="partial", location="Allenby 42", bboxes_json="[]",
        priority=50, agent_decision="human_review", agent_rationale="fixture",
        agent_trace_json="[]", status=status, work_order_id=None,
    )


def _on(monkeypatch):
    monkeypatch.setenv("AGENT_WORKER_AUTONOMY", "1")
    # Round 29 (audit F7): the lessons tool is offered once a lesson exists.
    monkeypatch.setattr(ra, "_agent_notes_count", lambda: 1)
    monkeypatch.delenv("AGENT_WORKER_REACT_MAX_HOPS", raising=False)
    monkeypatch.setenv("AGENT_SIGNAL_LLM", "ollama:fake")
    monkeypatch.setenv("AGENT_PERCEPTION_LLM", "ollama:fake")
    monkeypatch.setenv("AGENT_RESEARCH_LLM", "ollama:fake")


def test_autonomy_flag_implies_react_and_three_hops(monkeypatch):
    monkeypatch.delenv("AGENT_WORKER_REACT", raising=False)
    monkeypatch.delenv("AGENT_WORKER_REACT_MAX_HOPS", raising=False)
    monkeypatch.setenv("AGENT_WORKER_AUTONOMY", "0")
    assert not ra.worker_react_enabled() and ra.worker_react_max_hops() == 1
    monkeypatch.setenv("AGENT_WORKER_AUTONOMY", "1")
    assert ra.worker_react_enabled() and ra.worker_react_max_hops() == 3
    monkeypatch.setenv("AGENT_WORKER_REACT_MAX_HOPS", "2")
    assert ra.worker_react_max_hops() == 2


def test_signal_agent_gets_information_gap_and_numbers_are_verified(monkeypatch):
    _on(monkeypatch)
    seen: dict = {}

    def fake_react(**kw):
        seen.update(kw)
        raw = {
            "hazard_class": "D40", "lane_blocked": "partial", "road_type": None,
            "urgency_claimed": True, "summary": "deep hole",
            "reporter_signals": {"size_cm": 99, "depth_cm": 4, "vehicles_swerving": None, "standing_water": None},
            "tools_used": ["extract_measurements", "lookup_prior_reports"],
        }
        info = {"hops": 1, "tools_run": {"extract_measurements": {
            "ok": True, "size_cm": 30, "depth_cm": None, "vehicles_swerving": True, "standing_water": None}}}
        return raw, 20, 10, info

    monkeypatch.setattr(ra, "_llm_react_worker", fake_react)
    parsed, _in, _out, autonomy = ra._extract_by_llm(
        "a 30 cm hole, cars swerve",
        {"location": "Allenby 42", "road_type": "main", "lane_blocked": "none"},
        {"top1_class": "D40", "top1_confidence": 0.8},
        state={},
    )
    assert "detector" not in seen["user_payload"] and "form" not in seen["user_payload"]
    assert seen["user_payload"]["location"] == "Allenby 42"
    assert "AUTONOMOUS MODE" in seen["system_prompt"]
    assert seen["time_cap_s"] == 90.0  # policy latency.worker_time_cap_s, raised 1.11.2026
    # The ReAct reply is normalised like the single-call reply.
    assert parsed["mentioned"] == {"hazard_class": "D40", "lane_blocked": "partial", "road_type": None}
    sig = parsed["reporter_signals"]
    assert sig["size_cm"] == 30 and sig["depth_cm"] is None and sig["vehicles_swerving"] is True
    assert autonomy["tools_used"] == ["extract_measurements"]
    for v in ("declared_not_run:lookup_prior_reports", "signal_overridden:size_cm", "unverified_number_dropped:depth_cm"):
        assert v in autonomy["violations"]


def test_signal_numbers_without_the_tool_are_recomputed_by_regex(monkeypatch):
    _on(monkeypatch)
    raw = {"hazard_class": None, "lane_blocked": None, "road_type": None, "urgency_claimed": False,
           "summary": "hole", "reporter_signals": {"size_cm": 50, "depth_cm": None,
                                                    "vehicles_swerving": None, "standing_water": None},
           "tools_used": []}
    monkeypatch.setattr(ra, "_llm_react_worker", lambda **_: (raw, 5, 2, {"hops": 0, "tools_run": {}}))
    parsed, _, _, autonomy = ra._extract_by_llm("deep hole here", {"location": ""}, {}, state={})
    assert parsed["reporter_signals"]["size_cm"] is None
    # Round 29 (6.11.2026 audit F6): "deep hole here" carries no digit, so
    # extract_measurements is neither offered nor required; the number the
    # model invented is still dropped by the regex check.
    assert "required_not_called:extract_measurements" not in autonomy["violations"]
    assert "unverified_number_dropped:size_cm" in autonomy["violations"]
    assert autonomy["auto_ran"] == ["extract_measurements"]


def _refine(monkeypatch, raw, tools_run):
    _on(monkeypatch)
    seen: dict = {}

    def fake_react(**kw):
        seen.update(kw)
        return raw, 5, 2, {"hops": len(tools_run), "tools_run": tools_run}

    monkeypatch.setattr(ra, "_llm_react_worker", fake_react)
    base = {"severity_signal": "moderate", "severity_rationale": "r", "is_recurring_location": False}
    refined, _, _ = ra._llm_refine(
        base_profile=base, base_confidence="medium", base_needs_more=False, base_missing=[],
        yolo={}, iq={}, md={"location": "Allenby 42"}, state={},
    )
    return refined, seen


def test_profile_agent_without_quality_tool_gets_it_run_by_the_code(monkeypatch):
    raw = {"severity_signal": "severe", "severity_rationale": "x", "confidence": "high",
           "needs_more_evidence": False, "missing_fields": [], "tools_used": []}
    refined, seen = _refine(monkeypatch, raw, {})
    assert "rule_based_hint" not in seen["user_payload"]
    assert seen["user_payload"]["metadata"]["location"] == "Allenby 42"
    assert "recall_agent_notes" in seen["tool_registry"]
    # The agent read iq={} as usable and the skipped tool was run by the
    # code. Round 18: the LLM may not raise severity, so the "severe" it
    # returned is clamped to the deterministic "moderate" and the violation
    # is on record alongside the required-not-called one.
    assert refined["severity_signal"] == "moderate" and refined["confidence"] == "high"
    assert "fallback_reason" not in refined
    assert refined["autonomy"]["auto_ran"] == ["inspect_image_quality"]
    assert refined["autonomy"]["quality_verdict"] == "high"
    assert "required_not_called:inspect_image_quality" in refined["autonomy"]["violations"]
    assert any("severity_blocked_upgrade:severe_to_moderate" in v for v in refined["autonomy"]["violations"])


def test_profile_agent_confidence_is_capped_by_a_low_quality_verdict(monkeypatch):
    _on(monkeypatch)
    raw = {"severity_signal": "severe", "severity_rationale": "x", "confidence": "high",
           "needs_more_evidence": False, "missing_fields": [], "tools_used": []}
    monkeypatch.setattr(ra, "_llm_react_worker", lambda **_: (raw, 5, 2, {"hops": 0, "tools_run": {}}))
    base = {"severity_signal": "moderate", "severity_rationale": "r", "is_recurring_location": False}
    refined, _, _ = ra._llm_refine(
        base_profile=base, base_confidence="low", base_needs_more=False, base_missing=[],
        yolo={}, iq={"blur_score": 20, "is_road_scene": False}, md={"location": "x"}, state={},
    )
    assert refined["autonomy"]["quality_verdict"] == "low"
    assert refined["confidence"] == "low"
    assert "confidence_capped_by_quality_tool" in refined["autonomy"]["violations"]
    # Round 18: severity upgrade from the LLM is blocked at the moderate base.
    assert refined["severity_signal"] == "moderate"
    assert any("severity_blocked_upgrade" in v for v in refined["autonomy"]["violations"])


def test_profile_agent_with_quality_tool_is_accepted_and_recurrence_checked(monkeypatch):
    raw = {"severity_signal": "severe", "severity_rationale": "x", "confidence": "high",
           "needs_more_evidence": False, "missing_fields": [], "is_recurring_location": True,
           "tools_used": ["inspect_image_quality"]}
    refined, _ = _refine(monkeypatch, raw, {"inspect_image_quality": {"ok": True, "quality_verdict": "high", "concerns": []}})
    # Round 18: severity upgrade is blocked at the base; the recurrence
    # claim contradicting the lookup still ends up in violations.
    assert refined["severity_signal"] == "moderate"
    assert "recurrence_claim_contradicts_lookup" in refined["autonomy"]["violations"]
    assert any("severity_blocked_upgrade:severe_to_moderate" in v for v in refined["autonomy"]["violations"])


def _rag_state():
    return {
        "hazard_profile": {"primary_class": "D40", "severity_signal": "moderate"},
        "metadata": {"road_type": "main", "lane_blocked": "partial", "location": "x"},
        "yolo_output": {"top1_confidence": 0.5},
        "image_quality": {"blur_score": 120, "is_road_scene": True},
        "perception_confidence": "medium",
        "iterations": 3,
    }


def test_rag_agent_retrieval_feeds_the_handoff(monkeypatch):
    _on(monkeypatch)
    monkeypatch.setenv("AGENT_RAG_RETRIEVER_LLM", "1")
    raw = {"coverage": "direct_match", "confidence": "high", "key_findings": ["patch within 48h"],
           "cited_ids": ["p1", "ghost"], "tools_used": ["query_repair_standards"]}
    info = {"hops": 1, "tools_run": {"query_repair_standards": {
        "ok": True, "procedures": [PASSAGE], "tool_result_ids": ["p1"], "coverage": "direct_match"}}}
    monkeypatch.setattr(ra, "_llm_react_worker", lambda **_: (raw, 9, 4, info))
    # Round 29 (audit F8): the deterministic retrieval runs first (cached)
    # and gates the agent; here it finds nothing, so the agent retrieves.
    calls: list = []

    def det(hazard_class, road_type, k=3):
        calls.append((hazard_class, road_type))
        return {"procedures": [], "tool_result_ids": [], "coverage": "no_match"}

    monkeypatch.setitem(ra.RESEARCH_TOOLS, "query_repair_standards", (det, ra.RESEARCH_TOOLS["query_repair_standards"][1]))
    ra._clear_retrieval_cache()
    upd = ra.rag_retriever_node(_rag_state())
    assert calls == [("D40", "main")]
    assert upd["tool_result_ids"] == ["p1"] and upd["research_coverage"] == "direct_match"
    c = upd["trace"][0]["content"]
    assert c["autonomy"]["tools_used"] == ["query_repair_standards"]
    assert c["autonomy"]["source"] == "agent_retrieval"
    assert "fallback" not in c["autonomy"]
    assert "cited_unknown_ids:1" in c["autonomy"]["violations"]
    assert c["key_findings"] == ["patch within 48h"] and c["llm_used"] is True


def test_rag_agent_that_never_retrieves_falls_back_to_the_tool_query(monkeypatch):
    _on(monkeypatch)
    monkeypatch.setenv("AGENT_RAG_RETRIEVER_LLM", "1")
    raw = {"coverage": "direct_match", "confidence": "high", "key_findings": [], "tools_used": []}
    monkeypatch.setattr(ra, "_llm_react_worker", lambda **_: (raw, 9, 4, {"hops": 0, "tools_run": {}}))
    calls: list = []

    def det(hazard_class, road_type, k=3):
        calls.append((hazard_class, road_type))
        # Round 29 (audit F8): a road mismatch grades partial_match, so the
        # research agent still runs; the deterministic result is its fallback.
        return {"procedures": [dict(PASSAGE, road_types=["secondary"])], "tool_result_ids": ["p1"],
                "coverage": "partial_match"}

    monkeypatch.setitem(ra.RESEARCH_TOOLS, "query_repair_standards", (det, ra.RESEARCH_TOOLS["query_repair_standards"][1]))
    ra._clear_retrieval_cache()
    upd = ra.rag_retriever_node(_rag_state())
    assert calls == [("D40", "main")]
    c = upd["trace"][0]["content"]
    assert c["llm_fallback_reason"] == "autonomy_required_tool_not_called"
    assert "required_not_called:query_repair_standards" in c["autonomy"]["violations"]
    assert "no_passages_retrieved" not in c["autonomy"]["violations"]
    assert c["autonomy"]["agent_retrieved"] is False
    assert c["autonomy"]["fallback"].startswith("deterministic_query (")
    assert c["autonomy"]["auto_ran"] == ["query_repair_standards"]
    assert upd["research_coverage"] == "partial_match"  # the tool's coverage, not the agent's direct_match claim


def test_react_worker_stops_at_time_cap(monkeypatch):
    clock = iter([0.0, 0.0, 100.0])
    monkeypatch.setattr(ra.time, "monotonic", lambda: next(clock, 100.0))
    msg = types.SimpleNamespace(
        finish_reason="tool_calls", content=None,
        tool_calls=[{"id": "c1", "function": {"name": "extract_measurements", "arguments": json.dumps({"text": "30 cm"})}}],
    )
    monkeypatch.setattr(ra, "chat", lambda *a, **k: (msg, 3, 1))
    parsed, _in, _out, info = ra._llm_react_worker(
        agent_name="t", system_prompt="s", user_payload={}, tool_registry=ra.SIGNAL_WORKER_TOOLS,
        state={}, max_hops=3, time_cap_s=45,
    )
    assert parsed is None and info["timed_out"] is True
    assert info["hops"] == 1 and "extract_measurements" in info["tools_run"]


def test_agent_notes_roundtrip_and_recall_tool(tmp_path, monkeypatch):
    path = str(tmp_path / "n.db")
    with ra.Database(path) as db:
        db.upsert_user("u1", "a@example.com", "A")
        db.create_hazard(_hazard("hz1"))
        apr = db.create_approval("hz1", "urgent", 90, "r")
        db.add_agent_note(hazard_id="hz1", approval_id=apr, decision="denied", agent_action="urgent",
                          hazard_class="D40", road_type="main", lane_blocked="partial", note="  it was a  shadow ")
        time.sleep(0.01)
        db.add_agent_note(hazard_id="hz2", approval_id="apr_x", decision="approved", agent_action="standard",
                          hazard_class="D40", road_type="secondary")
        notes = db.recall_agent_notes("D40", "main")
        assert [n["road_type"] for n in notes] == ["main", "secondary"]  # same road first, then newest
        assert notes[0]["lesson"] == (
            "a human overturned the agent's urgent verdict for D40 on main (lane partial); note: it was a shadow"
        )
        assert db.recall_agent_notes("D20") == []
    monkeypatch.setenv("ROADGUARD_DB_PATH", path)
    ra.reset_settings_cache()
    try:
        out = ra.recall_agent_notes("D40", "main")
    finally:
        monkeypatch.setenv("ROADGUARD_DB_PATH", ":memory:")
        ra.reset_settings_cache()
    assert out["ok"] and out["n"] == 2 and out["lessons"][0]["road_type"] == "main"
    assert ra.recall_agent_notes("")["ok"] is False


def test_record_decision_lesson_writes_the_note(tmp_path):
    with ra.Database(str(tmp_path / "n.db")) as db:
        db.upsert_user("u1", "a@example.com", "A")
        db.create_hazard(_hazard("hz1"))
        apr = db.create_approval("hz1", "human_review", 50, "r")
        assert db.decide_approval(apr, "approved", "supervisor", "ok")
        assert api.record_decision_lesson(db, apr, "approved", "ok")
        lesson = db.recall_agent_notes("D40", "main")[0]["lesson"]
        assert lesson.startswith("a human confirmed the agent's human_review verdict for D40 on main (lane partial)")
        assert api.record_decision_lesson(db, "apr_missing", "approved", "") is None


def test_lessons_tool_offered_only_under_autonomy(monkeypatch):
    monkeypatch.setattr(ra, "_agent_notes_count", lambda: 1)
    monkeypatch.setenv("AGENT_WORKER_AUTONOMY", "0")
    assert "recall_agent_notes" not in {s["function"]["name"] for s in ra._decision_tool_schemas()}
    assert "recall_agent_notes" not in ra._worker_tools("profile_builder", ra.PERCEPTION_WORKER_TOOLS)
    monkeypatch.setenv("AGENT_WORKER_AUTONOMY", "1")
    assert "recall_agent_notes" in {s["function"]["name"] for s in ra._decision_tool_schemas()}
    assert "recall_agent_notes" in ra._worker_tools("rag_retriever", ra.RESEARCH_WORKER_TOOLS)
    assert "recall_agent_notes" not in ra._worker_tools("signal_extractor", ra.SIGNAL_WORKER_TOOLS)
    assert "recall_agent_notes" in ra._READ_ONLY_DECISION_TOOLS


def test_trace_and_stats_show_autonomy():
    entry = {"iter": 2, "speaker": "profile_builder", "kind": "handoff_out", "ts": time.time(),
             "content": {"to": "rag_retriever", "primary_class": "D40", "severity": "moderate", "confidence": "high",
                         "llm_used": True,
                         "autonomy": {"mode": "autonomous", "tools_used": ["inspect_image_quality"], "declared_tools": [],
                                      "hops": 1, "violations": ["recurrence_claim_contradicts_lookup"]}}}
    lines = ra.format_trace_lines("hz1", [entry], metadata={"class": "D40"}, verdict={"action": "standard", "priority": 10}, status="open")
    # Round 20: the autonomy line carries source=agent_choice when hops>0.
    assert any("autonomy: agent_called=inspect_image_quality  hops=1  source=agent_choice  violations=recurrence_claim_contradicts_lookup" in ln for ln in lines)
    agg = api._aggregate_agent_metrics([{"agent_trace_json": json.dumps([entry]), "class": "D40"}])
    # Round 11 (1.11.2026): tools the agent called, tools the contract ran
    # for it and failed calls are counted apart.
    assert agg["autonomy"] == {"profile_builder": {
        "runs": 1, "tool_calls": 1, "agent_called": 1, "auto_ran": 0, "auto_ran_tools": 0,
        "tool_failures": 0, "violations": 1, "clean_runs": 0,
    }}
