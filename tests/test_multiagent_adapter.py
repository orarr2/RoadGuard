"""Phase-1 adapter: flat hazard dict <-> graph state <-> AgentDecision."""

import pytest

from agent.loop import AgentDecision
from agent.multiagent.adapter import (
    _bboxes_to_detections,
    _count_by_class,
    final_state_to_decision,
    hazard_to_graph_inputs,
    multiagent_decide,
)


def _hz(**overrides):
    base = {
        "hazard_id": "hz_demo",
        "class": "D40",
        "confidence": 0.9,
        "bboxes": [[120, 240, 340, 460, "D40", 0.9]],
        "road_type": "main",
        "lane_blocked": "partial",
        "location": "Allenby St 42",
    }
    base.update(overrides)
    return base


def test_mapping_sets_three_blocks():
    hid, yolo, iq, md = hazard_to_graph_inputs(_hz())
    assert hid == "hz_demo"
    assert yolo["top1_class"] == "D40"
    assert yolo["top1_confidence"] == pytest.approx(0.9)
    assert md["road_type"] == "main"
    assert md["lane_blocked"] == "partial"
    assert iq["is_road_scene"] is True


def test_mapping_without_bboxes_synthesizes_detection():
    hid, yolo, _, _ = hazard_to_graph_inputs(_hz(bboxes=[]))
    assert len(yolo["detections"]) == 1
    assert yolo["detections"][0]["class"] == "D40"


def test_mapping_with_malformed_bbox_rows_is_tolerant():
    hid, yolo, _, _ = hazard_to_graph_inputs(_hz(bboxes=[["bad"], [1, 2, 3, 4, "D20", 0.6]]))
    assert len(yolo["detections"]) == 1
    assert yolo["detections"][0]["class"] == "D20"


def test_count_by_class_tracks_each_known_class():
    detections = [
        {"class": "D40"},
        {"class": "D40"},
        {"class": "D20"},
        {"class": "unknown"},
    ]
    counts = _count_by_class(detections)
    assert counts["D40"] == 2
    assert counts["D20"] == 1
    assert counts["D00"] == 0


def test_bboxes_falls_back_when_rows_unparseable():
    out = _bboxes_to_detections([["x", "y", "z", "w"]], "D10", 0.7)
    assert out == [{"class": "D10", "confidence": 0.7, "area_ratio": 0.0}]


def test_final_state_to_decision_sets_provider_and_trace():
    state = {
        "action": "urgent",
        "rationale": "D40 pothole on main road",
        "priority": 90,
        "cited_sources": ["sec-4.3"],
        "trace": [{"speaker": "decision", "kind": "final"}],
        "guard_fired": False,
    }
    dec = final_state_to_decision(state)
    assert isinstance(dec, AgentDecision)
    assert dec.action == "urgent"
    assert dec.priority == 90
    assert dec.cited_sources == ["sec-4.3"]
    assert dec.provider == "multiagent"
    assert dec.fallback_reason is None


def test_final_state_marks_guard_fired_as_fallback_reason():
    state = {
        "action": "human_review",
        "rationale": "budget guard fired: MAX_ITERATIONS (8) reached",
        "priority": 50,
        "cited_sources": [],
        "trace": [],
        "guard_fired": True,
    }
    dec = final_state_to_decision(state)
    assert dec.fallback_reason == "guard_fired"
    assert dec.action == "human_review"


def test_multiagent_decide_urgent_end_to_end():
    dec = multiagent_decide(_hz())
    assert dec.action == "urgent"
    assert dec.priority >= 60
    assert dec.cited_sources
    assert dec.provider == "multiagent"


def test_multiagent_decide_needs_evidence_for_ambiguous_confidence():
    dec = multiagent_decide(_hz(confidence=0.45, lane_blocked="none", **{"class": "D20"}))
    assert dec.action == "needs_more_evidence"


def test_multiagent_decide_standard_for_secondary_road():
    dec = multiagent_decide(
        _hz(**{"class": "D20"}, road_type="secondary", lane_blocked="none", confidence=0.85)
    )
    assert dec.action == "standard"
    assert dec.cited_sources
