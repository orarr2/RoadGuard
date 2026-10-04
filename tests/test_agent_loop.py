"""The agent decision loop end to end against a scripted provider."""

import json

from agent import loop, tools


class StubRAG:
    def __init__(self, passages):
        self.passages = passages

    def query(self, hazard_class, road_type, k):
        return self.passages[:k]


def _hazard(**overrides):
    base = {
        "hazard_id": "hz1",
        "class": "D40",
        "confidence": 0.9,
        "bboxes": [],
        "road_type": "main",
        "lane_blocked": "full",
        "location": "x",
        "image_object_key": "x.jpg",
        "reported_at": "2026-10-04T10:00:00Z",
    }
    base.update(overrides)
    return base


def test_input_fallback_bypasses_llm_entirely():
    provider = loop.MockProvider([])  # queue is empty on purpose
    decision = loop.agent_decide(
        hazard=_hazard(confidence=0.2),
        provider=provider,
        ctx=tools.ToolContext(),
    )
    assert decision.action == "human_review"
    assert decision.fallback_reason == "low_confidence"


def test_happy_path_three_hops():
    rag = StubRAG(
        [
            {
                "id": "p1",
                "source_url": "https://x/pothole",
                "text": "procedure",
                "issuing_body": "FHWA",
            }
        ]
    )
    scripted = [
        loop.ProviderMessage(
            finish_reason="tool_calls",
            tool_calls=[
                {
                    "id": "c1",
                    "name": "check_hazard_severity",
                    "arguments": {
                        "class": "D40",
                        "confidence": 0.9,
                        "road_type": "main",
                        "lane_blocked": "full",
                    },
                }
            ],
        ),
        loop.ProviderMessage(
            finish_reason="tool_calls",
            tool_calls=[
                {
                    "id": "c2",
                    "name": "query_repair_standards",
                    "arguments": {"hazard_class": "D40", "road_type": "main", "k": 1},
                }
            ],
        ),
        loop.ProviderMessage(
            finish_reason="stop",
            content=json.dumps(
                {
                    "action": "urgent",
                    "rationale": "pothole blocking a lane on a main road",
                    "priority": 95,
                    "cited_sources": ["https://x/pothole"],
                }
            ),
        ),
    ]
    decision = loop.agent_decide(
        hazard=_hazard(),
        provider=loop.MockProvider(scripted),
        ctx=tools.ToolContext(rag=rag),
    )
    assert decision.action == "urgent"
    assert decision.priority >= 80
    assert "https://x/pothole" in decision.cited_sources


def test_ungrounded_verdict_falls_to_human_review():
    scripted = [
        loop.ProviderMessage(
            finish_reason="stop",
            content=json.dumps(
                {
                    "action": "urgent",
                    "rationale": "no citations",
                    "priority": 90,
                    "cited_sources": [],
                }
            ),
        )
    ]
    decision = loop.agent_decide(
        hazard=_hazard(),
        provider=loop.MockProvider(scripted),
        ctx=tools.ToolContext(rag=StubRAG([])),
    )
    assert decision.action == "human_review"
    assert decision.fallback_reason == "ungrounded"


def test_invalid_action_falls_to_human_review_via_fallback_provider():
    scripted = [
        loop.ProviderMessage(
            finish_reason="stop",
            content=json.dumps(
                {"action": "dispatch_contractor", "rationale": "nope"}
            ),
        )
    ]
    fallback = loop.MockProvider(
        [
            loop.ProviderMessage(
                finish_reason="stop",
                content=json.dumps(
                    {
                        "action": "standard",
                        "rationale": "simplified mode",
                        "cited_sources": ["https://x"],
                    }
                ),
            )
        ]
    )
    decision = loop.agent_decide(
        hazard=_hazard(),
        provider=loop.MockProvider(scripted),
        ctx=tools.ToolContext(
            rag=StubRAG([{"id": "p1", "source_url": "https://x", "text": "x"}])
        ),
        fallback_provider=fallback,
    )
    assert decision.action in {"standard", "human_review"}


def test_hop_budget_exhaustion_returns_human_review():
    tool_calls_endless = [
        loop.ProviderMessage(
            finish_reason="tool_calls",
            tool_calls=[
                {
                    "id": f"c{i}",
                    "name": "check_hazard_severity",
                    "arguments": {
                        "class": "D40",
                        "confidence": 0.9,
                        "road_type": "main",
                        "lane_blocked": "full",
                    },
                }
            ],
        )
        for i in range(6)
    ]
    decision = loop.agent_decide(
        hazard=_hazard(),
        provider=loop.MockProvider(tool_calls_endless),
        ctx=tools.ToolContext(),
        max_hops=3,
    )
    assert decision.action == "human_review"
    assert decision.fallback_reason == "hop_budget_exceeded"


def test_primary_llm_exception_without_fallback():
    class BoomProvider(loop.LLMProvider):
        name = "boom"

        def complete(self, messages, tools_schema):
            raise RuntimeError("api down")

    decision = loop.agent_decide(
        hazard=_hazard(),
        provider=BoomProvider(),
        ctx=tools.ToolContext(),
    )
    assert decision.action == "human_review"
    assert decision.fallback_reason == "primary_llm_unavailable"
