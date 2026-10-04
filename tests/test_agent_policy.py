"""Policy fallbacks and the decision-table action selector."""

from agent import policy, tools


def test_low_confidence_triggers_fallback():
    v = policy.check_input_fallbacks(
        {"class": "D40", "confidence": 0.39, "road_type": "main", "lane_blocked": "full"}
    )
    assert v is not None and v.reason == "low_confidence"


def test_minimum_confidence_passes():
    v = policy.check_input_fallbacks(
        {"class": "D40", "confidence": 0.40, "road_type": "main", "lane_blocked": "full"}
    )
    assert v is None


def test_unknown_class_triggers_fallback():
    v = policy.check_input_fallbacks(
        {"class": "Dxx", "confidence": 0.9, "road_type": "main", "lane_blocked": "full"}
    )
    assert v is not None and v.reason == "unknown_class"


def test_empty_rag_triggers_fallback():
    v = policy.check_rag_fallback([])
    assert v is not None and v.reason == "empty_rag"


def test_rag_with_passages_ok():
    assert policy.check_rag_fallback([{"source_url": "x"}]) is None


def test_tool_timeout_triggers_fallback():
    v = policy.check_tool_error({"error": "tool_timeout", "tool": "foo"})
    assert v is not None and v.reason == "tool_timeout"


def test_tool_exception_triggers_fallback():
    v = policy.check_tool_error({"error": "tool_exception", "tool": "foo", "message": "boom"})
    assert v is not None and v.reason == "tool_exception"


def test_decision_table_urgent_d40_full_main_high_conf():
    hazard = {"class": "D40", "confidence": 0.9, "road_type": "main", "lane_blocked": "full"}
    sev = tools.compute_severity("D40", 0.9, "main", "full")
    assert policy.decision_table_action(hazard, sev, has_rag=True) == "urgent"


def test_decision_table_needs_more_evidence_borderline():
    hazard = {"class": "D40", "confidence": 0.55, "road_type": "secondary", "lane_blocked": "partial"}
    sev = tools.compute_severity("D40", 0.55, "secondary", "partial")
    assert policy.decision_table_action(hazard, sev, has_rag=True) == "needs_more_evidence"


def test_decision_table_standard_routine():
    hazard = {"class": "D00", "confidence": 0.78, "road_type": "bike_path", "lane_blocked": "none"}
    sev = tools.compute_severity("D00", 0.78, "bike_path", "none")
    assert policy.decision_table_action(hazard, sev, has_rag=True) == "standard"


def test_decision_table_human_review_without_rag():
    hazard = {"class": "D20", "confidence": 0.72, "road_type": "main", "lane_blocked": "partial"}
    sev = tools.compute_severity("D20", 0.72, "main", "partial")
    assert policy.decision_table_action(hazard, sev, has_rag=False) == "human_review"


def test_validate_action_allows_four_values():
    for a in ["urgent", "standard", "needs_more_evidence", "human_review"]:
        assert policy.validate_action(a) is True
    for bad in ["", "dispatch", "unknown", "URGENT"]:
        assert policy.validate_action(bad) is False


def test_final_human_review_shape():
    v = policy.final_human_review("empty_rag", "no procedure")
    assert v["action"] == "human_review"
    assert v["priority"] == 0
    assert v["cited_sources"] == []
    assert v["fallback_reason"] == "empty_rag"
