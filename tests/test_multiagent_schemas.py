"""Handoff builders + typed-dict shape."""

from agent.multiagent.schemas import (
    build_perception_handoff,
    build_research_handoff,
)


def test_perception_handoff_shape():
    h = build_perception_handoff({"primary_class": "D40"}, "high", False, [])
    assert h["sender"] == "perception"
    assert h["recipient"] == "research"
    assert h["needs_more_evidence"] is False
    assert h["missing_fields"] == []


def test_perception_handoff_carries_profile():
    profile = {"primary_class": "D20", "severity_signal": "moderate"}
    h = build_perception_handoff(profile, "medium", True, ["closer_shot"])
    assert h["hazard_profile"] == profile
    assert h["confidence"] == "medium"


def test_research_handoff_shape():
    h = build_research_handoff([], [], "no_match", "low")
    assert h["sender"] == "research"
    assert h["recipient"] == "decision"


def test_research_handoff_direct_match_with_procedures():
    procs = [{"article_ids": ["sec-4.3"]}]
    h = build_research_handoff(procs, ["sec-4.3"], "direct_match", "high")
    assert h["coverage"] == "direct_match"
    assert h["procedures"] == procs
    assert h["tool_result_ids"] == ["sec-4.3"]
