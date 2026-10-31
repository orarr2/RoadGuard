"""The run deadline (WALL_CLOCK_CAP_SECONDS from the run's entry) must reach
every provider call made inside a graph node and stop every ReAct loop, so
no call outlives the cap. 31.10.2026 review: a 650 s intake call ran before
the Supervisor's first check started the clock, and a 401 s perception call
retried and fell back with no deadline in sight, in runs capped at 240 s.
"""

from __future__ import annotations

import time

import pytest

import roadguard_agent as ra


@pytest.fixture(autouse=True)
def _clean_deadline():
    ra.clear_deadline()
    yield
    ra.clear_deadline()


def _reply(content: str = "{}", tool_calls: list | None = None) -> ra.ChatMessage:
    return ra.ChatMessage(
        finish_reason="tool_calls" if tool_calls else "stop",
        content=content,
        tool_calls=tool_calls or [],
    )


def test_every_node_call_is_bounded_by_the_run_deadline(monkeypatch):
    monkeypatch.setenv("ROADGUARD_USE_LIVE", "1")
    for key in ("AGENT_LLM_PROVIDER", "AGENT_SIGNAL_LLM", "AGENT_PERCEPTION_LLM", "AGENT_RESEARCH_LLM"):
        monkeypatch.setenv(key, "ollama:fake")
    monkeypatch.setenv("AGENT_RAG_RETRIEVER_LLM", "1")
    monkeypatch.setenv("AGENT_PROFILE_BUILDER_LLM", "1")
    monkeypatch.setenv("AGENT_WORKER_AUTONOMY", "1")
    monkeypatch.delenv("AGENT_LLM_PANEL", raising=False)
    monkeypatch.delenv("AGENT_LLM_FALLBACK", raising=False)
    # A provider timeout far above the cap: a cut timeout proves the
    # deadline reached the call.
    monkeypatch.setenv("AGENT_LLM_TIMEOUT_SECONDS", "1000")
    monkeypatch.setattr(ra, "WALL_CLOCK_CAP_SECONDS", 240.0)
    seen: list[dict] = []

    def fake_call_provider(spec, messages, tools, temperature, timeout=None):
        from langgraph.config import get_config

        seen.append({
            "node": (get_config().get("metadata") or {}).get("langgraph_node"),
            "spec": spec,
            "timeout": timeout,
            "deadline": ra._run_deadline(),
            "in_node": ra._inside_graph_node(),
        })
        return _reply(), 1, 1, "ollama", "fake"

    monkeypatch.setattr(ra, "call_provider", fake_call_provider)
    final = ra.run_graph(
        "hz_deadline_test",
        {
            "top1_class": "D40",
            "top1_confidence": 0.55,
            "top1_margin": 0.1,
            "detections": [{"class": "D40", "confidence": 0.55, "bbox": [0, 0, 10, 10]}],
        },
        {"blur_score": 60, "is_road_scene": True, "brightness_mean": 0.5},
        {
            "road_type": "main",
            "lane_blocked": "none",
            "location": "Allenby 42",
            "reporter_text": "pothole about 40 cm wide",
        },
    )
    assert final["action"] in ("urgent", "standard", "needs_more_evidence", "human_review")
    nodes = {c["node"] for c in seen}
    assert {"signal_extractor", "profile_builder", "rag_retriever", "decision"} <= nodes, nodes
    worker_cap = ra._policy_latency()["worker_time_cap_s"]
    for call in seen:
        assert call["in_node"] is True, call
        assert call["deadline"] is not None, call
        assert call["timeout"] is not None and 0 < call["timeout"] <= 240, call
        if call["node"] == "decision":
            # Decision gets what is left of the run: here nearly the whole cap
            assert call["timeout"] > 200, call
        else:
            # an autonomous worker's call is cut to the worker's own time cap
            assert call["timeout"] <= worker_cap, call


def test_react_worker_does_not_start_a_round_without_headroom(monkeypatch):
    monkeypatch.setattr(ra, "_inside_graph_node", lambda: True)
    ra.set_deadline(time.time() + 2.0)  # under the 5 s hop headroom
    calls: list[int] = []
    monkeypatch.setattr(ra, "chat", lambda *a, **k: calls.append(1) or (_reply(), 1, 1))
    parsed, _, _, info = ra._llm_react_worker(
        agent_name="profile_builder",
        system_prompt="s",
        user_payload={},
        tool_registry={},
        state={},
        provider_override="ollama:fake",
        max_hops=2,
    )
    assert parsed is None
    assert info["deadline_hit"] is True and info["timed_out"] is True
    assert calls == []


def test_worker_time_cap_cuts_each_call_timeout_and_restores_the_deadline(monkeypatch):
    monkeypatch.setenv("ROADGUARD_USE_LIVE", "1")
    monkeypatch.setenv("AGENT_LLM_TIMEOUT_SECONDS", "1000")
    monkeypatch.delenv("AGENT_LLM_FALLBACK", raising=False)
    monkeypatch.setattr(ra, "_inside_graph_node", lambda: True)
    run_deadline = time.time() + 240.0
    ra.set_deadline(run_deadline)
    seen: list[float | None] = []

    def fake_call_provider(spec, messages, tools, temperature, timeout=None):
        seen.append(timeout)
        return _reply('{"severity_signal": "moderate"}'), 1, 1, "ollama", "fake"

    monkeypatch.setattr(ra, "call_provider", fake_call_provider)
    parsed, _, _, info = ra._llm_react_worker(
        agent_name="profile_builder",
        system_prompt="s",
        user_payload={},
        tool_registry={},
        state={},
        provider_override="ollama:fake",
        max_hops=1,
        time_cap_s=20.0,
    )
    assert parsed == {"severity_signal": "moderate"}
    assert seen and all(t is not None and 3.0 <= t <= 20.0 for t in seen), seen
    assert "deadline_hit" not in info
    # the worker's cap never leaks into the rest of the run
    assert ra.current_deadline() == run_deadline


def test_decision_react_stops_at_the_deadline(monkeypatch):
    monkeypatch.setattr(ra, "_inside_graph_node", lambda: True)
    monkeypatch.setattr(ra, "_decision_messages", lambda state, hint: [{"role": "user", "content": "x"}])
    monkeypatch.setattr(ra, "chat", lambda *a, **k: pytest.fail("no call may start without headroom"))
    ra.set_deadline(time.time() + 1.0)
    state = ra.initial_state(
        "hz_deadline_decision",
        {"top1_class": "D40", "top1_confidence": 0.7, "detections": []},
        {"blur_score": 150, "is_road_scene": True},
        {"road_type": "main", "lane_blocked": "none"},
    )
    decision, _, _, info = ra._llm_decide_react(state, {"action": "standard"})
    assert decision is None
    assert info["deadline_hit"] is True and info["hops"] == 0
