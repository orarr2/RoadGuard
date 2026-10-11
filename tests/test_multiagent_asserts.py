"""Assert v3: handoff schema + Decision grounding checks."""

import pytest

from agent.multiagent.asserts import (
    HandoffAssertionError,
    assert_decision_grounded,
    assert_perception_handoff,
    assert_research_handoff,
)
from agent.multiagent.schemas import (
    build_perception_handoff,
    build_research_handoff,
)


def _valid_profile():
    return {
        "primary_class": "D40",
        "primary_confidence": 0.9,
        "secondary_classes": [],
        "severity_signal": "severe",
        "severity_rationale": "pothole with lane obstruction",
        "affects_vehicle_path": True,
        "is_recurring_location": False,
        "recurrence_count_30d": 0,
        "quality_warnings": [],
    }


def test_perception_handoff_valid():
    h = build_perception_handoff(_valid_profile(), "high", False, [])
    assert assert_perception_handoff(h) is True


def test_perception_handoff_missing_primary_class():
    bad = build_perception_handoff(
        {"primary_confidence": 0.9},  # type: ignore[arg-type]
        "high",
        False,
        [],
    )
    with pytest.raises(HandoffAssertionError, match="primary_class"):
        assert_perception_handoff(bad)


def test_perception_handoff_bad_confidence():
    bad = build_perception_handoff(_valid_profile(), "very_high", False, [])  # type: ignore[arg-type]
    with pytest.raises(HandoffAssertionError, match="confidence"):
        assert_perception_handoff(bad)


def test_perception_handoff_wrong_direction():
    h = build_perception_handoff(_valid_profile(), "high", False, [])
    h["sender"] = "decision"  # type: ignore[typeddict-item]
    with pytest.raises(HandoffAssertionError, match="direction"):
        assert_perception_handoff(h)


def test_research_handoff_valid():
    h = build_research_handoff([{"article_ids": ["sec-4.3"]}], ["sec-4.3"], "direct_match", "high")
    assert assert_research_handoff(h) is True


def test_research_handoff_no_match_must_have_empty_procedures():
    h = build_research_handoff([{"article_ids": ["x"]}], ["x"], "no_match", "low")
    with pytest.raises(HandoffAssertionError, match="no_match"):
        assert_research_handoff(h)


def test_research_handoff_procedures_not_list():
    h = build_research_handoff([], [], "direct_match", "high")
    h["procedures"] = "not-a-list"  # type: ignore[typeddict-item]
    with pytest.raises(HandoffAssertionError, match="procedures"):
        assert_research_handoff(h)


def test_grounded_citation_passes():
    assert (
        assert_decision_grounded(
            cited_sources=["sec-4.3"],
            tool_result_ids=["sec-4.3", "sec-2.1"],
            action="urgent",
        )
        is True
    )


def test_ungrounded_citation_fails_for_action_urgent():
    with pytest.raises(HandoffAssertionError, match="ungrounded"):
        assert_decision_grounded(
            cited_sources=["sec-99"],
            tool_result_ids=["sec-4.3"],
            action="urgent",
        )


def test_action_standard_requires_at_least_one_source():
    with pytest.raises(HandoffAssertionError, match="90%"):
        assert_decision_grounded(
            cited_sources=[],
            tool_result_ids=["sec-4.3"],
            action="standard",
        )


def test_human_review_may_cite_nothing():
    assert (
        assert_decision_grounded(
            cited_sources=[],
            tool_result_ids=[],
            action="human_review",
        )
        is True
    )


def test_needs_more_evidence_may_cite_nothing():
    assert (
        assert_decision_grounded(
            cited_sources=[],
            tool_result_ids=[],
            action="needs_more_evidence",
        )
        is True
    )


def test_human_review_cannot_cite_hallucinated():
    with pytest.raises(HandoffAssertionError, match="human_review"):
        assert_decision_grounded(
            cited_sources=["made-up"],
            tool_result_ids=[],
            action="human_review",
        )
