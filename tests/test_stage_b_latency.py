"""Stage B: clear-case fast path, no second LLM round on a ReAct miss,
check_system_health no longer offered to Decision, fallback classification."""

from __future__ import annotations

import roadguard_agent as ra
import roadguard_api as api


def test_clear_case_fast_path_only_for_confident_clean_submissions():
    yolo, iq, md = {"top1_confidence": 0.85}, {"blur_score": 150, "is_road_scene": True}, {"reporter_text": ""}
    assert ra._clear_case_fast_path(yolo, iq, md)
    assert ra._clear_case_fast_path({"top1_confidence": 0.6}, iq, md) is None
    assert ra._clear_case_fast_path(yolo, {"blur_score": 40, "is_road_scene": True}, md) is None
    assert ra._clear_case_fast_path(yolo, iq, {"reporter_text": "huge pothole"}) is None
    assert ra._clear_case_fast_path(yolo, iq, {"reporter_text": "", "location_was_scrubbed": True}) is None


def test_fast_path_thresholds_come_from_policy(monkeypatch):
    monkeypatch.setattr(ra, "load_policy", lambda: {"latency": {"fast_path": {"min_detector_confidence": 0.95}}})
    assert ra._clear_case_fast_path({"top1_confidence": 0.9}, {"blur_score": 150}, {}) is None
    assert ra._clear_case_fast_path({"top1_confidence": 0.96}, {"blur_score": 150}, {}) is not None


def test_profile_refine_does_not_retry_single_call_after_react_miss(monkeypatch):
    monkeypatch.setenv("AGENT_WORKER_REACT", "1")
    monkeypatch.setenv("AGENT_PERCEPTION_LLM", "ollama:fake")
    monkeypatch.setattr(ra, "_llm_react_worker", lambda **_: (None, 11, 3, {"hops": 1}))

    def _no_chat(*_a, **_k):
        raise AssertionError("single-call chat() must not run after a ReAct miss")

    monkeypatch.setattr(ra, "chat", _no_chat)
    refined, in_tok, out_tok = ra._llm_refine(
        base_profile={"severity_signal": "moderate", "severity_rationale": "r", "is_recurring_location": False},
        base_confidence="medium", base_needs_more=False, base_missing=[],
        yolo={}, iq={}, md={}, state={},
    )
    assert (refined["severity_signal"], in_tok, out_tok) == ("moderate", 11, 3)
    assert refined["fallback_reason"] == "react_reply_off_contract"


def test_rag_relabel_does_not_retry_single_call_after_react_miss(monkeypatch):
    monkeypatch.setenv("AGENT_WORKER_REACT", "1")
    monkeypatch.setenv("AGENT_RESEARCH_LLM", "ollama:fake")
    monkeypatch.setattr(ra, "_llm_react_worker", lambda **_: (None, 7, 2, {"hops": 0}))
    monkeypatch.setattr(ra, "chat", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no second call")))
    cov, conf, findings, in_tok, out_tok, reason = ra._llm_label(
        "D40", "main", [], [], "direct_match", "high", state={},
    )
    assert (cov, conf, findings, in_tok, out_tok) == ("direct_match", "high", [], 7, 2)
    assert reason == "react_reply_off_contract"


def test_check_system_health_not_offered_to_decision():
    names = {s["function"]["name"] for s in ra._decision_tool_schemas()}
    assert "check_system_health" not in names
    # 1.11.2026: the two tools without a backend left the offer as well.
    assert "rank_similarity_to_catalog" not in names
    assert "check_recent_cluster" not in names
    assert {"check_hazard_severity", "lookup_dataset_image", "request_additional_evidence",
            "create_draft_work_order", "flag_for_urgent_review"} <= names


def test_llm_failure_classification_buckets():
    f = api._classify_llm_failure
    assert f({"error_sample": "ollama_http_503: model not loaded"}) == "http_5xx"
    assert f({"error_sample": "ReadTimeout: timed out after 30s"}) == "timeout"
    assert f({"error_sample": "429 rate_limit_exceeded"}) == "http_429"
    assert f({"error_sample": "ConnectError: connection refused"}) == "connection"
    assert f({"errors": ["JSONDecodeError"]}) == "parse"
    assert f({"errors": ["RuntimeError"]}) == "other"
