"""Round 9 (1.11.2026): a compact Decision prompt, tools without a backend
kept out of the offer, compact tool results for the model, and no retry of
a provider call that ran out of its timeout.

Background: in run r6 the Decision prompt reached 4,044 of the 4,096
context tokens, a call that size takes 100-130 s on the CPU-only VM, the
per-call timeout was 60 s, and the backoff retried the identical call up
to four times before the rule tree decided."""
import json

import pytest

import roadguard_agent as ra


def test_decision_prompt_is_compact_and_keeps_the_contract():
    prompt = ra._build_decision_prompt()
    assert len(prompt) < 9400, len(prompt)  # round 17 added the sufficiency, band and class-name rules (8,932)  # 7,472 after round 9; round 10 added the severity_check and the similar-photo paragraphs; 12,071 before
    for needle in (
        '"urgent"', '"standard"', '"needs_more_evidence"', '"human_review"', '"known_image"',
        "Policy version", "SAFETY", "reporter_claims", "jurisdiction_hint", "dataset_match",
        "lookup_dataset_image", "quality_warnings", '"cited_sources"', "tool_result_ids",
        '"assessment"', '"uncertainty"', "EXAMPLE",
    ):
        assert needle in prompt, needle
    for gone in ("rank_similarity_to_catalog", "check_recent_cluster", "check_system_health"):
        assert gone not in prompt, gone


def test_unwired_tools_are_not_offered_but_still_answer_with_the_stub(monkeypatch):
    monkeypatch.delenv("AGENT_WORKER_AUTONOMY", raising=False)
    names = {s["function"]["name"] for s in ra._decision_tool_schemas()}
    assert names == {
        "check_hazard_severity", "lookup_dataset_image", "request_additional_evidence",
        "create_draft_work_order", "flag_for_urgent_review",
    }
    state = {"hazard_id": "hz_test", "metadata": {}}
    for name in ra._UNWIRED_DECISION_TOOLS:
        result = ra._run_decision_tool(name, {"k": 3}, state, {})
        assert result["ok"] is True and result.get("stub") is True, (name, result)


def test_tool_result_for_model_compacts_the_catalogue_row():
    full = {
        "ok": True, "found": True, "match": "sha256", "distance": 0, "pic_id": 1024,
        "file_name": "firc_roadcrack_1024.jpg", "split": "train", "width": 640, "height": 640,
        "classes": "D40", "n_labels": 20,
        "labels": [{"class": "D40", "bbox": [0.1 * i, 0.2, 0.3, 0.4]} for i in range(20)],
        "model_prediction": {"top1_class": "D40", "top1_confidence": 0.71, "bbox": [1, 2, 3, 4]},
    }
    text = ra._tool_result_for_model("lookup_dataset_image", full)
    compact = json.loads(text)
    assert "labels" not in compact
    assert compact["found"] is True and compact["match"] == "sha256"
    assert compact["pic_id"] == 1024 and compact["classes"] == "D40" and compact["n_labels"] == 20
    assert compact["model_prediction"] == "D40 0.71"
    assert len(text) < len(json.dumps(full)) / 3
    miss = ra._tool_result_for_model("lookup_dataset_image", {"ok": True, "found": False, "reason": "no catalogued image with this fingerprint"})
    assert json.loads(miss)["found"] is False


def test_compact_for_model_caps_lists_strings_and_the_whole_message():
    value = {"items": list(range(12)), "note": "x" * 1000, "nested": {"more": ["a" * 300], "n": 7}}
    compact = ra._compact_for_model(value)
    assert compact["items"][:5] == [0, 1, 2, 3, 4] and compact["items"][-1] == "... 7 more"
    assert len(compact["note"]) == ra._TOOL_RESULT_MAX_STR and compact["note"].endswith("...")
    assert len(compact["nested"]["more"][0]) == ra._TOOL_RESULT_MAX_STR
    assert compact["nested"]["n"] == 7
    wide = {f"k{i}": "y" * 200 for i in range(20)}
    text = ra._tool_result_for_model("check_hazard_severity", wide)
    assert len(text) <= ra._TOOL_RESULT_MAX_CHARS and text.endswith("...")
    small = {"ok": True, "priority": 55, "formula": "min(100, base * conf_mult + road + lane + recurrence)"}
    assert json.loads(ra._tool_result_for_model("check_hazard_severity", small)) == small


def test_backoff_does_not_retry_a_call_that_timed_out(monkeypatch):
    calls: list[str] = []

    def fake_call_provider(spec, messages, tools, temperature, timeout=None):
        calls.append(spec)
        raise TimeoutError("timed out")

    monkeypatch.setattr(ra, "call_provider", fake_call_provider)
    monkeypatch.setattr(ra.time, "sleep", lambda s: pytest.fail(f"slept {s}s before an identical retry"))
    with pytest.raises(TimeoutError):
        ra._call_with_backoff("ollama:fake", [{"role": "user", "content": "x"}], None, 0.0, None)
    assert calls == ["ollama:fake"]


def test_backoff_still_retries_a_transient_5xx(monkeypatch):
    calls: list[str] = []

    def fake_call_provider(spec, messages, tools, temperature, timeout=None):
        calls.append(spec)
        raise RuntimeError("ollama_http_503: service unavailable")

    monkeypatch.setattr(ra, "call_provider", fake_call_provider)
    monkeypatch.setattr(ra.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError):
        ra._call_with_backoff("ollama:fake", [{"role": "user", "content": "x"}], None, 0.0, None)
    assert len(calls) == 1 + len(ra._BACKOFF_SCHEDULE_SECONDS)


def test_fallback_chain_calls_each_distinct_spec_once_after_a_timeout(monkeypatch):
    monkeypatch.setenv("AGENT_LLM_FALLBACK", "ollama:second,ollama:second,ollama:first")
    calls: list[str] = []

    def fake_call_provider(spec, messages, tools, temperature, timeout=None):
        calls.append(spec)
        raise TimeoutError("timed out")

    monkeypatch.setattr(ra, "call_provider", fake_call_provider)
    monkeypatch.setattr(ra.time, "sleep", lambda s: pytest.fail("no sleep expected"))
    with pytest.raises(TimeoutError):
        ra._call_with_fallback("ollama:first", [{"role": "user", "content": "x"}], None, 0.0, None)
    assert calls == ["ollama:first", "ollama:second"]


def test_is_timeout_matches_client_timeouts_only():
    assert ra._is_timeout(TimeoutError("timed out"))

    class ReadTimeout(Exception):
        pass

    assert ra._is_timeout(ReadTimeout("The read operation timed out"))
    assert not ra._is_timeout(RuntimeError("ollama_http_504: gateway timeout"))
    assert not ra._is_timeout(RuntimeError("429 rate_limit_exceeded"))
