"""C1 - LLM-in-every-node wiring and fallback behaviour.

Each node calls chat() when USE_LIVE is on. The Decision node must pass
use_panel=True (panel/committee only fires with that flag). Any parse
error, schema violation, or ungrounded verdict must fall back to the
deterministic rule-based path.
"""

from __future__ import annotations

import json

from agent.multiagent import chat as chat_mod
from agent.multiagent.nodes import decision as decision_mod
from agent.multiagent.nodes import perception as perception_mod
from agent.multiagent.nodes import research as research_mod
from agent.multiagent.nodes.decision import decision_node
from agent.multiagent.nodes.perception import perception_node
from agent.multiagent.nodes.research import research_node
from agent.multiagent.providers import ChatMessage
from agent.multiagent.state import initial_state


def _state_for_perception() -> dict:
    yolo = {
        "top1_class": "D40",
        "top1_confidence": 0.92,
        "top1_margin": 0.4,
        "detections": [
            {"class": "D40", "confidence": 0.92, "area_ratio": 0.09, "bbox": [1, 1, 1, 1]}
        ],
    }
    iq = {
        "blur_score": 180,
        "is_road_scene": True,
        "brightness_mean": 0.6,
        "ocr_sanitized": {"is_suspicious_pattern": False},
    }
    md = {"road_type": "main", "lane_blocked": "partial", "location": "Allenby 42"}
    return initial_state("hz_1", yolo, iq, md)


# ---------------------------------------------------------------------------
# Perception
# ---------------------------------------------------------------------------


def test_perception_uses_llm_when_live(monkeypatch):
    def fake_chat(messages, tools=None, temperature=0.0, use_panel=False):
        assert use_panel is False, "Perception must NOT request the panel"
        return (
            ChatMessage(
                finish_reason="stop",
                content=json.dumps(
                    {
                        "severity_signal": "moderate",
                        "severity_rationale": "llm said moderate",
                        "confidence": "high",
                        "needs_more_evidence": False,
                        "missing_fields": [],
                    }
                ),
            ),
            42,
            7,
        )

    monkeypatch.setattr(perception_mod, "chat", fake_chat)

    out = perception_node(_state_for_perception())
    assert out["hazard_profile"]["severity_signal"] == "moderate"
    assert out["hazard_profile"]["severity_rationale"] == "llm said moderate"
    assert out["perception_confidence"] == "high"
    assert out["tokens_in"] == 42
    assert out["tokens_out"] == 7


def test_perception_falls_back_on_parse_error(monkeypatch):
    def fake_chat(messages, tools=None, temperature=0.0, use_panel=False):
        return ChatMessage(finish_reason="stop", content="not json"), 10, 2

    monkeypatch.setattr(perception_mod, "chat", fake_chat)

    out = perception_node(_state_for_perception())
    # D40 + partial lane blockage -> rule-based says severe
    assert out["hazard_profile"]["severity_signal"] == "severe"


def test_perception_falls_back_on_invalid_enum(monkeypatch):
    def fake_chat(messages, tools=None, temperature=0.0, use_panel=False):
        return (
            ChatMessage(
                finish_reason="stop",
                content=json.dumps(
                    {
                        "severity_signal": "catastrophic",  # not in enum
                        "confidence": "high",
                    }
                ),
            ),
            10,
            2,
        )

    monkeypatch.setattr(perception_mod, "chat", fake_chat)

    out = perception_node(_state_for_perception())
    assert out["hazard_profile"]["severity_signal"] == "severe"  # rule-based


def test_perception_falls_back_on_chat_exception(monkeypatch):
    def fake_chat(messages, tools=None, temperature=0.0, use_panel=False):
        raise RuntimeError("provider down")

    monkeypatch.setattr(perception_mod, "chat", fake_chat)

    out = perception_node(_state_for_perception())
    assert out["hazard_profile"]["severity_signal"] == "severe"
    # No tokens accrued when the call failed
    assert "tokens_in" not in out or out["tokens_in"] == 0


# ---------------------------------------------------------------------------
# Research
# ---------------------------------------------------------------------------


def _state_for_research() -> dict:
    base = _state_for_perception()
    base["hazard_profile"] = {
        "primary_class": "D40",
        "primary_confidence": 0.92,
    }
    return base


def test_research_uses_llm_when_live(monkeypatch):
    def fake_chat(messages, tools=None, temperature=0.0, use_panel=False):
        assert use_panel is False, "Research must NOT request the panel"
        return (
            ChatMessage(
                finish_reason="stop",
                content=json.dumps(
                    {
                        "coverage": "partial_match",
                        "confidence": "medium",
                        "key_findings": ["signage required for lane blockage"],
                    }
                ),
            ),
            25,
            5,
        )

    monkeypatch.setattr(research_mod, "chat", fake_chat)

    out = research_node(_state_for_research())
    # LLM override took effect
    assert out["research_coverage"] == "partial_match"
    # Trace exposes findings
    key_findings = out["trace"][0]["content"]["key_findings"]
    assert key_findings == ["signage required for lane blockage"]


def test_research_falls_back_on_parse_error(monkeypatch):
    def fake_chat(messages, tools=None, temperature=0.0, use_panel=False):
        return ChatMessage(finish_reason="stop", content="```oops"), 10, 2

    monkeypatch.setattr(research_mod, "chat", fake_chat)

    out = research_node(_state_for_research())
    # Rule-based coverage from the tool stands
    assert out["research_coverage"] in {"direct_match", "partial_match", "no_match"}


def test_research_corrects_llm_when_no_procedures(monkeypatch):
    """LLM says direct_match but tool returned nothing - trust the tool."""
    import agent.multiagent.tools as tools_mod

    def empty_tool(hazard_class, road_type, k):
        return {
            "procedures": [],
            "tool_result_ids": [],
            "coverage": "no_match",
        }

    monkeypatch.setitem(
        tools_mod.RESEARCH_TOOLS, "query_repair_standards", (empty_tool, {})
    )

    def fake_chat(messages, tools=None, temperature=0.0, use_panel=False):
        return (
            ChatMessage(
                finish_reason="stop",
                content=json.dumps(
                    {
                        "coverage": "direct_match",
                        "confidence": "high",
                        "key_findings": [],
                    }
                ),
            ),
            5,
            2,
        )

    monkeypatch.setattr(research_mod, "chat", fake_chat)

    out = research_node(_state_for_research())
    assert out["research_coverage"] == "no_match"


# ---------------------------------------------------------------------------
# Decision - panel usage + grounding fallback
# ---------------------------------------------------------------------------


def _state_for_decision() -> dict:
    return {
        "hazard_id": "hz_1",
        "hazard_profile": {
            "primary_class": "D40",
            "primary_confidence": 0.9,
            "severity_signal": "severe",
            "is_recurring_location": False,
            "recurrence_count_30d": 0,
        },
        "metadata": {"road_type": "main", "lane_blocked": "partial"},
        "research_coverage": "direct_match",
        "procedures": [
            {
                "source_url": "example.com",
                "issuing_body": "Netivey Israel",
                "article_ids": ["4.3"],
                "quarantined_summary": {"severity_level": "severe"},
            }
        ],
        "tool_result_ids": ["tr_1", "tr_2"],
        "perception_confidence": "high",
        "trace": [],
        "iterations": 0,
    }


def test_decision_requests_panel(monkeypatch):
    captured: dict = {}

    def fake_chat(messages, tools=None, temperature=0.0, use_panel=False):
        captured["use_panel"] = use_panel
        return (
            ChatMessage(
                finish_reason="stop",
                content=json.dumps(
                    {
                        "action": "urgent",
                        "rationale": "D40 blocks lane on main",
                        "priority": 90,
                        "cited_sources": ["tr_1"],
                        "missing_evidence": [],
                    }
                ),
            ),
            50,
            12,
        )

    monkeypatch.setattr(decision_mod, "chat", fake_chat)

    out = decision_node(_state_for_decision())
    assert captured["use_panel"] is True
    assert out["action"] == "urgent"
    assert out["cited_sources"] == ["tr_1"]
    assert out["trace"][0]["content"]["source"] == "llm_panel"
    assert out["tokens_in"] == 50


def test_decision_falls_back_on_ungrounded_llm(monkeypatch):
    def fake_chat(messages, tools=None, temperature=0.0, use_panel=False):
        return (
            ChatMessage(
                finish_reason="stop",
                content=json.dumps(
                    {
                        "action": "urgent",
                        "rationale": "fake",
                        "priority": 80,
                        "cited_sources": ["invented_id"],  # not in tool_result_ids
                        "missing_evidence": [],
                    }
                ),
            ),
            20,
            5,
        )

    monkeypatch.setattr(decision_mod, "chat", fake_chat)

    out = decision_node(_state_for_decision())
    # Rule-based fallback: D40 + partial + main + conf>=0.6 -> urgent
    assert out["action"] == "urgent"
    assert out["trace"][0]["content"]["source"] == "rule_based"
    # Rule-based cites the real tool ids
    assert set(out["cited_sources"]).issubset({"tr_1", "tr_2"})


def test_decision_falls_back_on_invalid_action(monkeypatch):
    def fake_chat(messages, tools=None, temperature=0.0, use_panel=False):
        return (
            ChatMessage(
                finish_reason="stop",
                content=json.dumps(
                    {
                        "action": "dispatch_tank",  # not a Charter action
                        "rationale": "x",
                        "priority": 50,
                        "cited_sources": [],
                        "missing_evidence": [],
                    }
                ),
            ),
            10,
            3,
        )

    monkeypatch.setattr(decision_mod, "chat", fake_chat)

    out = decision_node(_state_for_decision())
    assert out["action"] in {"urgent", "standard", "needs_more_evidence", "human_review"}
    assert out["trace"][0]["content"]["source"] == "rule_based"


def test_decision_falls_back_on_chat_exception(monkeypatch):
    def fake_chat(messages, tools=None, temperature=0.0, use_panel=False):
        raise TimeoutError("provider timeout")

    monkeypatch.setattr(decision_mod, "chat", fake_chat)

    out = decision_node(_state_for_decision())
    assert out["action"] in {"urgent", "standard", "needs_more_evidence", "human_review"}
    assert out["trace"][0]["content"]["source"] == "rule_based"


def test_decision_clips_priority_to_0_100(monkeypatch):
    def fake_chat(messages, tools=None, temperature=0.0, use_panel=False):
        return (
            ChatMessage(
                finish_reason="stop",
                content=json.dumps(
                    {
                        "action": "urgent",
                        "rationale": "ok",
                        "priority": 999,
                        "cited_sources": ["tr_1"],
                        "missing_evidence": [],
                    }
                ),
            ),
            5,
            2,
        )

    monkeypatch.setattr(decision_mod, "chat", fake_chat)

    out = decision_node(_state_for_decision())
    assert 0 <= out["priority"] <= 100


# ---------------------------------------------------------------------------
# chat() use_panel semantics
# ---------------------------------------------------------------------------


def test_chat_use_panel_true_fires_panel_when_env_set(monkeypatch):
    monkeypatch.setenv("ROADGUARD_USE_LIVE", "1")
    monkeypatch.setenv(
        "AGENT_LLM_PANEL", "groq:a,groq:b"
    )
    monkeypatch.setattr(chat_mod, "USE_LIVE", True)

    from agent.multiagent import providers

    seen: list[tuple[str, str]] = []

    def _factory(name):
        def _fn(messages, tools, temperature, model):
            seen.append((name, model))
            return (
                ChatMessage(
                    finish_reason="stop",
                    content=json.dumps(
                        {
                            "action": "standard",
                            "rationale": "x",
                            "priority": 50,
                            "cited_sources": [],
                        }
                    ),
                ),
                3,
                1,
            )

        return _fn

    monkeypatch.setattr(
        providers,
        "PROVIDERS",
        {n: _factory(n) for n in ("groq", "openai_compat", "ollama")},
    )

    chat_mod.chat([{"role": "user", "content": "x"}], use_panel=True)
    assert len(seen) == 2


def test_chat_default_single_provider_without_panel_flag(monkeypatch):
    monkeypatch.setenv("ROADGUARD_USE_LIVE", "1")
    monkeypatch.setenv("AGENT_LLM_PANEL", "groq:a,groq:b")
    monkeypatch.setattr(chat_mod, "USE_LIVE", True)

    from agent.multiagent import providers

    seen: list[tuple[str, str]] = []

    def _factory(name):
        def _fn(messages, tools, temperature, model):
            seen.append((name, model))
            return ChatMessage(finish_reason="stop", content="ok"), 3, 1

        return _fn

    monkeypatch.setattr(
        providers,
        "PROVIDERS",
        {n: _factory(n) for n in ("groq", "openai_compat", "ollama")},
    )

    chat_mod.chat([{"role": "user", "content": "x"}])
    assert len(seen) == 1
