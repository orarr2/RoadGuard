"""The five tools exposed to the three agents (least-privilege mapping)."""

from agent.multiagent.tools import (
    DECISION_TOOLS,
    PERCEPTION_TOOLS,
    RESEARCH_TOOLS,
    check_hazard_severity,
    create_draft_work_order,
    flag_for_urgent_review,
    lookup_similar_cases,
    query_repair_standards,
    request_additional_evidence,
)


def test_least_privilege_mapping():
    assert set(PERCEPTION_TOOLS) == {"lookup_similar_cases"}
    assert set(RESEARCH_TOOLS) == {"query_repair_standards"}
    assert set(DECISION_TOOLS) == {
        "check_hazard_severity",
        "request_additional_evidence",
        "create_draft_work_order",
        "flag_for_urgent_review",
    }


def test_severity_priority_bounded():
    for cls in ["D00", "D10", "D20", "D40"]:
        r = check_hazard_severity(cls, 0.9, "main", "full", 5)
        assert 0 <= r["priority"] <= 100


def test_severity_recurrence_bump_capped():
    low = check_hazard_severity("D40", 0.9, "main", "full", 0)
    high = check_hazard_severity("D40", 0.9, "main", "full", 100)
    assert high["components"]["recurrence_bonus"] == 15
    assert high["priority"] >= low["priority"]


def test_severity_low_confidence_halves_base():
    r = check_hazard_severity("D40", 0.3, "main", "full")
    assert r["components"]["confidence_multiplier"] == 0.5


def test_severity_schema_exposes_all_params():
    schema = check_hazard_severity.__globals__["SCHEMA"]
    params = schema["function"]["parameters"]["properties"]
    for key in ("hazard_class", "confidence", "road_type", "lane_blocked", "recurrence_count_30d"):
        assert key in params


def test_query_repair_known_pair_returns_direct_match():
    r = query_repair_standards("D40", "main")
    assert r["coverage"] == "direct_match"
    assert "sec-4.3" in r["tool_result_ids"]


def test_query_repair_unknown_pair_returns_no_match():
    r = query_repair_standards("D10", "bike_path")
    assert r["coverage"] == "no_match"
    assert r["tool_result_ids"] == []
    assert r["procedures"] == []


def test_query_repair_k_limits_results():
    r = query_repair_standards("D40", "main", k=1)
    assert len(r["procedures"]) == 1


def test_lookup_similar_stub_returns_empty_without_mem():
    r = lookup_similar_cases("Allenby St 42", "D40")
    assert r == {"count": 0, "cases": [], "is_recurring_location": False}


def test_lookup_similar_with_injected_memory_marks_recurrence():
    class FakeMem:
        def find_similar(self, **_):
            return [{"id": 1}, {"id": 2}, {"id": 3}]

    r = lookup_similar_cases("Begin Blvd 15", "D40", vector_mem=FakeMem())
    assert r["count"] == 3
    assert r["is_recurring_location"] is True


def test_create_draft_work_order_returns_awaiting_approval():
    r = create_draft_work_order("h-1", 55, "standard", ["sec-5.1"])
    assert r["awaiting_human_approval"] is True
    assert r["draft_id"].startswith("draft_")


def test_create_draft_work_order_hits_injected_store():
    class FakeStore:
        def __init__(self):
            self.calls = []

        def create_draft(self, **kw):
            self.calls.append(kw)

    s = FakeStore()
    create_draft_work_order("h-9", 70, "urgent", ["sec-4.3"], store=s)
    assert len(s.calls) == 1
    assert s.calls[0]["hazard_id"] == "h-9"


def test_flag_for_urgent_review_returns_flag_id():
    r = flag_for_urgent_review("h-2", "lane blocked", "immediate")
    assert r["flag_id"].startswith("flag_")


def test_flag_for_urgent_review_uses_injected_queue_and_email():
    class FakeQueue:
        def __init__(self):
            self.flagged = []

        def flag(self, **kw):
            self.flagged.append(kw)

    class FakeEmail:
        def __init__(self):
            self.sent = []

        def send_internal_summary(self, **kw):
            self.sent.append(kw)

    q, e = FakeQueue(), FakeEmail()
    flag_for_urgent_review("h-3", "test", "today", review_queue=q, email_client=e)
    assert q.flagged[0]["hazard_id"] == "h-3"
    assert e.sent[0]["urgency"] == "today"


def test_request_additional_evidence_rejects_unknown_field():
    r = request_additional_evidence("h-1", "weather", "please send a photo in the sun")
    assert r["ok"] is False
    assert "invalid" in r["error"]


def test_request_additional_evidence_truncates_message():
    long_msg = "x" * 500
    r = request_additional_evidence("h-1", "closer_shot", long_msg)
    assert len(r["message_to_reporter"]) <= 140
