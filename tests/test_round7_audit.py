"""Live audit of the 31.10.2026 night re-run (owner watching the trace):
the per-case report showed 0 tokens and model null, the trace printed a
worker's ReAct steps as empty handoff lines, and the supervisor's note
claimed a blocked lane on open roads. One test per finding."""

from __future__ import annotations

import roadguard_agent as ra
from roadguard_api import _report_trace_summary


def test_report_counts_the_llm_calls_and_tokens_of_a_case():
    trace = [
        {"iter": 0, "speaker": "sanitize", "kind": "boundary_defence", "content": {"ocr_suspicious": False}},
        {"kind": "llm_call", "content": {"model": "ollama:qwen2.5:3b", "latency_s": 22.5, "tokens_in": 1896, "tokens_out": 99, "ok": True}},
        {"iter": 2, "speaker": "profile_builder", "kind": "react_step", "content": {"hop": 1, "tool": "lookup_similar_cases", "ok": True}},
        {"iter": 2, "speaker": "profile_builder", "kind": "handoff_out", "content": {
            "llm_used": True, "llm": {"model": "ollama:qwen2.5:3b", "calls": 1, "tokens_in": 1896, "tokens_out": 99}}},
        {"kind": "llm_call", "content": {"model": "ollama:qwen2.5:3b", "latency_s": 3.6, "tokens_in": 1203, "tokens_out": 34, "ok": True}},
        {"iter": 3, "speaker": "rag_retriever", "kind": "handoff_out", "content": {"llm": {"model": "ollama:qwen2.5:3b", "calls": 1}}},
        {"iter": 4, "speaker": "decision", "kind": "final", "content": {"action": "human_review", "source": "rule_based",
                                                                         "cited_sources": ["catalog:pic_id=4354"]}},
    ]
    timeline, tin, tout, cited, calls = _report_trace_summary(trace)
    assert (tin, tout) == (1896 + 1203, 99 + 34)
    assert [c["speaker"] for c in calls] == ["profile_builder", "rag_retriever"]
    assert all(c["model"] == "ollama:qwen2.5:3b" for c in calls)
    assert cited == ["catalog:pic_id=4354"]
    by_speaker = {t["speaker"]: t for t in timeline if t["speaker"]}
    assert by_speaker["profile_builder"]["llm_used"] is True
    assert by_speaker["rag_retriever"]["llm_used"] is True  # derived from the llm summary
    assert by_speaker["decision"]["source"] == "rule_based"


def test_report_falls_back_to_legacy_fields_without_call_entries():
    trace = [{"iter": 2, "speaker": "profile_builder", "kind": "handoff_out",
              "content": {"llm_used": True, "model": "groq:llama", "tokens_in": 500, "tokens_out": 40}}]
    _, tin, tout, _, calls = _report_trace_summary(trace)
    assert (tin, tout) == (500, 40) and calls[0]["model"] == "groq:llama"


def test_worker_react_step_renders_as_a_step_not_an_empty_handoff():
    entry = {"iter": 2, "speaker": "profile_builder", "kind": "react_step", "content": {
        "hop": 1, "tool": "lookup_similar_cases", "args": {"hazard_class": "D40"}, "ok": True,
        "result": {"ok": True, "matches": 0}, "duration_s": 0.02}}
    lines = ra._entry_lines(entry, [])
    assert "step 1" in lines[0] and "tool attempted: lookup_similar_cases" in lines[1]
    joined = "\n".join(lines)
    assert "class=None" not in joined and "handoff" not in joined


def test_bare_tool_call_entry_renders_as_a_note():
    entry = {"kind": "tool_call", "speaker": "decision", "hop": 0, "tool": "lookup_dataset_image", "ok": True,
             "note": "relaxed catalogue hit pic_id=1024 dist=15"}
    lines = ra._entry_lines(entry, [])
    assert lines == [f"{ra._lbl('decision')} tool: relaxed catalogue hit pic_id=1024 dist=15"]


def test_supervisor_note_is_checked_against_the_fields():
    payload = {"hazard_class_detected": "D40", "detector_confidence": 0.536, "road_type": "main",
               "lane_blocked": "none", "has_reporter_text": False}
    note, check = ra._verify_supervisor_note(payload, "hazard_class is D40, detector_confidence is 0.536, no reporter_text, main lane is blocked")
    assert check == "replaced:lane_blocked" and "lane blocked: none" in note and "D40" in note
    good = "hazard_class_detected is D40, detector_confidence is 0.536, road_type is main, lane_blocked is none, has_reporter_text is false"
    assert ra._verify_supervisor_note(payload, good) == (good, "ok")
    assert ra._verify_supervisor_note(payload, "low confidence, D40 hazard on main road, no reporter text")[1] == "ok"
    assert ra._verify_supervisor_note(payload, "D20 alligator cracking with reporter text")[1] == "replaced:reporter_text,hazard_class"
    assert ra._verify_supervisor_note(payload, "")[1] == "replaced:empty"
    blocked = dict(payload, lane_blocked="partial")
    assert ra._verify_supervisor_note(blocked, "D40 pothole, main lane is blocked")[1] == "ok"


def test_quality_tool_accepts_the_nested_object_small_models_send():
    nested = ra._tool_inspect_image_quality(image_quality={"blur_score": 20, "is_road_scene": False, "brightness_mean": 0.5})
    assert nested["ok"] is True and nested["quality_verdict"] == "low"
    assert set(nested["concerns"]) == {"severe_blur", "not_road_scene"}
    flat = ra._tool_inspect_image_quality(blur_score=150, is_road_scene=True, brightness_mean=0.5)
    assert flat["quality_verdict"] == "high"
    # unknown extras never raise
    assert ra._tool_inspect_image_quality(blur_score=150, extra_field=1)["ok"] is True


def test_react_executor_unpacks_a_single_object_argument(monkeypatch):
    import json as _json

    def tool(a: int = 0, b: int = 0) -> dict:
        return {"ok": True, "sum": a + b}

    replies = iter([
        ra.ChatMessage(finish_reason="tool_calls", content=None, tool_calls=[
            {"id": "c1", "function": {"name": "add", "arguments": _json.dumps({"payload": {"a": 2, "b": 3, "noise": 1}})}}]),
        ra.ChatMessage(finish_reason="stop", content='{"done": true}', tool_calls=[]),
    ])
    monkeypatch.setattr(ra, "chat", lambda *a, **k: (next(replies), 1, 1))
    parsed, _, _, info = ra._llm_react_worker(
        agent_name="profile_builder", system_prompt="s", user_payload={}, state={},
        tool_registry={"add": (tool, {"type": "function", "function": {"name": "add", "parameters": {"type": "object", "properties": {}}}})},
        provider_override="ollama:fake", max_hops=2,
    )
    assert parsed == {"done": True}
    assert info["tools_run"]["add"] == {"ok": True, "sum": 5}


def test_worker_react_steps_reach_the_stored_trace(monkeypatch):
    import json as _json

    def tool(a: int = 0) -> dict:
        return {"ok": True, "a": a}

    replies = iter([
        ra.ChatMessage(finish_reason="tool_calls", content=None, tool_calls=[
            {"id": "c1", "function": {"name": "probe", "arguments": _json.dumps({"a": 1})}}]),
        ra.ChatMessage(finish_reason="stop", content='{"done": true}', tool_calls=[]),
    ])
    monkeypatch.setattr(ra, "chat", lambda *a, **k: (next(replies), 1, 1))
    ra.begin_llm_log()
    parsed, _, _, info = ra._llm_react_worker(
        agent_name="rag_retriever", system_prompt="s", user_payload={}, state={"iterations": 2},
        tool_registry={"probe": (tool, {"type": "function", "function": {"name": "probe", "parameters": {"type": "object", "properties": {}}}})},
        provider_override="ollama:fake", max_hops=1,
    )
    steps = ra.drain_worker_steps()
    assert parsed == {"done": True}
    assert len(steps) == 1 and steps[0]["speaker"] == "rag_retriever" and steps[0]["kind"] == "react_step"
    assert steps[0]["content"]["tool"] == "probe" and steps[0]["iter"] == 3
    assert ra.drain_worker_steps() == []  # drained once


def test_rag_node_trace_carries_the_worker_steps_before_the_handoff(monkeypatch):
    monkeypatch.setenv("AGENT_WORKER_AUTONOMY", "1")
    monkeypatch.setenv("AGENT_RAG_RETRIEVER_LLM", "1")
    monkeypatch.setenv("AGENT_RESEARCH_LLM", "ollama:fake")
    passage = {"passage": "Pothole repair: patch within 48 hours.", "article_ids": ["p1"], "hazard_types": ["D40"],
               "road_types": ["any"], "relevance": {"class_keyword_hit": True, "repair_term_hit": True}}
    raw = {"coverage": "direct_match", "confidence": "high", "key_findings": ["patch within 48h"],
           "cited_ids": ["p1"], "tools_used": ["query_repair_standards"]}
    step = {"iter": 3, "speaker": "rag_retriever", "kind": "react_step",
            "content": {"hop": 1, "tool": "query_repair_standards", "ok": True, "result": "..."}}

    def fake_worker(**_):
        ra.record_worker_step(step)
        return raw, 9, 4, {"hops": 1, "tools_run": {"query_repair_standards": {
            "ok": True, "procedures": [passage], "tool_result_ids": ["p1"], "coverage": "direct_match"}}}

    monkeypatch.setattr(ra, "_llm_react_worker", fake_worker)
    # Round 29 (audit F8): the research LLM runs only when the deterministic
    # retrieval is not direct_match; an empty retrieval keeps it on here.
    monkeypatch.setitem(ra.RESEARCH_TOOLS, "query_repair_standards",
                        (lambda **kw: {"procedures": [], "tool_result_ids": [], "coverage": "no_match"},
                         ra.RESEARCH_TOOLS["query_repair_standards"][1]))
    ra._clear_retrieval_cache()
    state = {"hazard_profile": {"primary_class": "D40", "primary_confidence": 0.5, "severity_signal": "moderate"},
             "metadata": {"road_type": "main", "lane_blocked": "none"}, "yolo_output": {"top1_class": "D40", "top1_confidence": 0.5},
             "image_quality": {"blur_score": 60}, "iterations": 2, "perception_confidence": "medium"}
    upd = ra.rag_retriever_node(state)
    kinds = [(e["speaker"], e["kind"]) for e in upd["trace"]]
    assert kinds == [("rag_retriever", "react_step"), ("rag_retriever", "handoff_out")]


def test_clean_passage_text_trims_cut_words_to_sentences():
    cut = "ety and pavement rideability. Pothole repair operations can usually be divided into two distinct periods. The first period is winter repairs, when temperatu"
    cleaned = ra._clean_passage_text(cut)
    assert cleaned.startswith("Pothole repair operations")
    assert cleaned.endswith("two distinct periods.")
    assert ra._clean_passage_text("Seal the crack. Fill it.") == "Seal the crack. Fill it."
    assert ra._clean_passage_text("") == ""
    # a cut start with no sentence end near by loses only the fragment
    assert ra._clean_passage_text("tion of the surface layer is required here").startswith("of the surface")


def test_rank_semantic_hits_prefers_evidence_over_distance():
    hits = [
        {"article_ids": ["a"], "distance": 0.40, "relevance": {"class_keyword_hit": False, "repair_term_hit": True}},
        {"article_ids": ["b"], "distance": 0.72, "relevance": {"class_keyword_hit": True, "repair_term_hit": True}},
        {"article_ids": ["b"], "distance": 0.72, "relevance": {"class_keyword_hit": True, "repair_term_hit": True}},
        {"article_ids": ["c"], "distance": 0.60, "relevance": {"class_keyword_hit": True, "repair_term_hit": False}},
        {"article_ids": ["d"], "distance": 0.55, "relevance": {"class_keyword_hit": True, "repair_term_hit": True}},
    ]
    ranked = ra._rank_semantic_hits(hits, 3)
    assert [h["article_ids"][0] for h in ranked] == ["d", "b", "c"]


def test_query_repair_standards_merges_two_queries_and_cleans_text():
    class FakeClient:
        def __init__(self):
            self.queries = []

        def query(self, hazard_class, road_type, k):
            return []

        def query_semantic(self, text, k=5):
            self.queries.append(text)
            if "repair standards" in text:
                return [{"passage": "ety and rideability. Pothole patching is done in two periods. The first is winter",
                         "article_ids": ["seg_1"], "distance": 0.72, "hazard_types": [""], "road_types": ["any"]}]
            return [{"passage": "Pothole repair materials: cold mix for winter, hot mix otherwise.",
                     "article_ids": ["seg_2"], "distance": 0.51, "hazard_types": [""], "road_types": ["any"]},
                    {"passage": "ety and rideability. Pothole patching is done in two periods. The first is winter",
                     "article_ids": ["seg_1"], "distance": 0.72, "hazard_types": [""], "road_types": ["any"]}]

    client = FakeClient()
    out = ra.query_repair_standards("D40", "main", k=3, rag_client=client)
    assert len(client.queries) == 2 and "pothole repair" in client.queries[1].lower()
    ids = [p["article_ids"][0] for p in out["procedures"]]
    assert ids == ["seg_2", "seg_1"]  # closer evidence first, duplicate merged
    assert out["procedures"][1]["passage"].startswith("Pothole patching is done")
    assert out["coverage"] == "direct_match"
