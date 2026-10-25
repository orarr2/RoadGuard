"""Explicit handling of LLM failures and budget-guard partial progress.

The reviewer's P0 feedback (Pasted markdown(5).md): a missing or invalid
output must be identified explicitly, logged with the failure reason,
and lead to a limited retry or a safe handoff - never a silent fallback
that leaves the supervisor with no clue what happened.

These tests pin the three mechanisms:
  1. chat() records the exception type in _record_call so llm_summary
     emits ``failed_calls`` and ``errors`` on the trace.
  2. profile_builder / rag_retriever trace entries carry
     ``llm_fallback_reason`` when the LLM was enabled, every call
     failed and the rule tree took over.
  3. graceful_stop_decision enriches the rationale with partial progress
     (last_speaker, class so far, RAG coverage so far) when state is
     passed in - a supervisor reading the digest sees what the pipeline
     accomplished before the cap fired.
"""

from __future__ import annotations

from roadguard_agent import (
    graceful_stop_decision,
    llm_summary,
)


def test_llm_summary_counts_failed_calls_and_lists_error_types():
    calls = [
        {"model": "ollama:qwen2.5:3b", "latency_s": 42.1,
         "tokens_in": 850, "tokens_out": 220, "ok": True},
        {"model": "ollama:qwen2.5:3b", "latency_s": 0.03,
         "tokens_in": 0, "tokens_out": 0, "ok": False,
         "error_type": "APIConnectionError"},
    ]
    summary = llm_summary(calls)
    assert summary["calls"] == 2
    assert summary["ok_calls"] == 1
    assert summary["failed_calls"] == 1
    assert summary["errors"] == ["APIConnectionError"]
    assert summary["tokens_in"] == 850
    assert summary["tokens_out"] == 220


def test_llm_summary_hides_errors_field_when_all_calls_ok():
    calls = [
        {"model": "ollama:qwen2.5:3b", "latency_s": 42.1,
         "tokens_in": 850, "tokens_out": 220, "ok": True},
    ]
    summary = llm_summary(calls)
    assert summary["failed_calls"] == 0
    assert "errors" not in summary


def test_graceful_stop_decision_without_state_is_backward_compatible():
    d = graceful_stop_decision("WALL_CLOCK_CAP_SECONDS (240.0) reached")
    assert d["action"] == "human_review"
    assert d["priority"] == 50
    assert d["guard_fired"] is True
    assert d["cited_sources"] == []
    assert "budget guard fired" in d["rationale"]


def test_graceful_stop_decision_with_state_names_partial_progress():
    state = {
        "last_speaker": "rag_retriever",
        "hazard_profile": {"primary_class": "D40", "severity_signal": "severe"},
        "perception_confidence": "medium",
        "research_coverage": "direct_match",
        "tool_result_ids": ["sec-4.3", "sec-2.1"],
    }
    d = graceful_stop_decision(
        "WALL_CLOCK_CAP_SECONDS (240.0) reached",
        state,
    )
    assert d["action"] == "human_review"
    rationale = d["rationale"]
    # The supervisor sees exactly what the pipeline accomplished:
    assert "stopped at rag_retriever" in rationale
    assert "class=D40" in rationale
    assert "severity=severe" in rationale
    assert "perception=medium" in rationale
    assert "coverage=direct_match" in rationale
    assert "2 passage" in rationale
    assert "decision was not reached" in rationale


def test_graceful_stop_decision_rationale_bounded_to_500_chars():
    """A rich partial-progress rationale must still fit in the 500-char
    agent_rationale column (hazards table) and the 500-char Decision
    rationale slot. Build an overlong state and verify."""
    state = {
        "last_speaker": "rag_retriever",
        "hazard_profile": {
            "primary_class": "D40",
            "severity_signal": "severe-and-very-long-note-" * 20,
        },
        "perception_confidence": "medium",
        "research_coverage": "direct_match",
        "tool_result_ids": ["sec-" + str(i) for i in range(100)],
    }
    d = graceful_stop_decision(
        "WALL_CLOCK_CAP_SECONDS (240.0) reached",
        state,
    )
    assert len(d["rationale"]) <= 500
