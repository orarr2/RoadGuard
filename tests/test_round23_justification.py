"""Round 23 (4.11.2026): the owner's demand to see, per every image,
whether EACH of the six agents was justified (yes / no + reason), the
total run cost, and WHICH cases the agents outperformed the rule tree
or the human. Validates the per-case helpers and the aggregate the API
returns under /stats/kpi, /stats/agents and /stats/advantage.
"""
import json

import pytest

import roadguard_agent as ra
from roadguard_api import _aggregate_agent_metrics


# ---------------------------------------------------------------- per-agent justified


def _sanitize_entry(**fields):
    return {"speaker": "sanitize", "kind": "handoff_out",
            "content": {"has_text": False, **fields}}


def _signal_entry(**fields):
    return {"speaker": "signal_extractor", "kind": "handoff_out",
            "content": fields}


def _supervisor_entry(**fields):
    return {"speaker": "supervisor", "kind": "handoff_out",
            "content": fields}


def _profile_entry(**fields):
    base = {"primary_class": "D40", "severity": "moderate",
            "confidence": "medium", "llm_used": True,
            "autonomy": {"tools_used": [], "violations": [], "hops": 0}}
    base.update(fields)
    return {"speaker": "profile_builder", "kind": "handoff_out", "content": base}


def _rag_entry(**fields):
    base = {"coverage": "none", "n_procedures": 0, "llm_used": True,
            "autonomy": {"tools_used": [], "violations": [], "hops": 0}}
    base.update(fields)
    return {"speaker": "rag_retriever", "kind": "handoff_out", "content": base}


def _decision_final_entry(**fields):
    base = {"action": "standard", "priority": 50, "source": "llm_react",
            "rationale": "r", "cited_sources": [],
            "worker_impact": {
                "profile_llm_changed": False, "research_llm_changed": False,
                "rule_hint_without_worker_llm": "standard",
                "rule_hint_with_worker_llm": "standard",
                "changed_rule_hint": False,
            }}
    base.update(fields)
    return {"speaker": "decision", "kind": "final", "content": base}


def test_sanitize_justified_on_any_scrub():
    j = ra._justify_sanitize([{"location_scrubbed": True}])
    assert j["justified"] is True
    assert "location_scrubbed" in j["reasons"]


def test_sanitize_not_justified_without_scrub():
    j = ra._justify_sanitize([{"location_scrubbed": False,
                               "ocr_suspicious": False,
                               "reporter_text_suspicious": False}])
    assert j["justified"] is False
    assert "nothing_to_scrub" in j["reasons"]


def test_signal_extractor_justified_on_mentioned_claims():
    j = ra._justify_signal_extractor([{
        "has_text": True,
        "mentioned": {"hazard_class": "D40", "lane_blocked": None, "road_type": None},
        "urgency_claimed": False,
        "reporter_signals": {},
        "contradictions": [],
    }])
    assert j["justified"] is True
    assert "extracted_claims_from_text" in j["reasons"]


def test_signal_extractor_not_justified_without_text():
    j = ra._justify_signal_extractor([{"has_text": False}])
    assert j["justified"] is False


def test_profile_justified_on_catalogue_hit():
    content = _profile_entry(dataset_lookup={"found": True})["content"]
    j = ra._justify_profile_builder([content])
    assert j["justified"] is True
    assert "catalogue_hit" in j["reasons"]


def test_profile_not_justified_when_llm_blocked():
    content = _profile_entry(
        dataset_lookup={"found": False},
        llm_delta={"severity_signal": {"before": "severe", "after": "minor"}},
        autonomy={"tools_used": [], "violations": ["severity_blocked_downgrade:minor_to_severe"], "hops": 0},
    )["content"]
    j = ra._justify_profile_builder([content])
    assert j["justified"] is False
    assert "llm_blocked_by_gate" in j["reasons"] or j["reasons"] == ["no_signal_added"]
    assert j["rebuttals"]


def test_rag_justified_on_direct_match_with_sufficient_coverage():
    content = _rag_entry(coverage="direct_match", n_procedures=3,
                         decision_coverage={"sufficiency": "sufficient_coverage",
                                            "answered": ["repair_window"], "missing": []})["content"]
    j = ra._justify_rag_retriever([content])
    assert j["justified"] is True
    assert "direct_match_retrieved" in j["reasons"]
    assert "sufficient_coverage" in j["reasons"]


def test_decision_justified_on_changed_rule_hint():
    final = _decision_final_entry(
        action="human_review", source="llm_react",
        worker_impact={"profile_llm_changed": True, "research_llm_changed": False,
                       "rule_hint_without_worker_llm": "standard",
                       "rule_hint_with_worker_llm": "human_review",
                       "changed_rule_hint": True},
        cited_sources=[{"url": "x", "name": "doc"}],
        rationale_check={"ok": True},
    )
    summary = ra._build_agent_value_summary([final])
    assert summary["changed_rule_hint"] is True
    pa = summary["per_agent_justified"]
    assert pa["decision"]["justified"] is True
    assert "workers_moved_rule_tree" in pa["decision"]["reasons"]


def test_decision_not_justified_on_rule_based_fallback():
    final = _decision_final_entry(source="rule_based")
    summary = ra._build_agent_value_summary([final])
    pa = summary["per_agent_justified"]
    assert pa["decision"]["justified"] is False
    assert "llm_fell_back_to_tree" in (pa["decision"].get("rebuttals") or [])


# ---------------------------------------------------------------- per-run cost


def test_run_cost_sums_calls_and_tokens():
    trace = [
        {"speaker": "profile_builder", "kind": "handoff_out", "ts": 100.0,
         "content": {"llm": {"calls": 2, "tokens_in": 500, "tokens_out": 120}}},
        {"speaker": "rag_retriever", "kind": "handoff_out", "ts": 110.0,
         "content": {"llm": {"calls": 1, "tokens_in": 300, "tokens_out": 80}}},
        {"speaker": "decision", "kind": "final", "ts": 150.0,
         "content": {"action": "standard", "source": "llm_react", "priority": 50,
                     "llm": {"calls": 1, "tokens_in": 1800, "tokens_out": 250}}},
    ]
    cost = ra._compute_run_cost(trace)
    assert cost["llm_calls"] == 4
    assert cost["tokens_in"] == 2600
    assert cost["tokens_out"] == 450
    assert cost["wall_seconds"] == 50.0
    assert cost["cost_self_hosted_usd"] == 0.0
    # Expected: (2600/1000)*0.0003 + (450/1000)*0.0006 = 0.00078 + 0.00027 = 0.00105
    assert abs(cost["cost_commercial_equivalent_usd"] - 0.00105) < 1e-6


def test_run_cost_with_no_llm_calls_returns_zero():
    trace = [{"speaker": "decision", "kind": "final", "ts": 100.0,
              "content": {"action": "known_image", "source": "catalog_match", "priority": 10}}]
    cost = ra._compute_run_cost(trace)
    assert cost["llm_calls"] == 0
    assert cost["tokens_in"] == 0
    assert cost["cost_commercial_equivalent_usd"] == 0.0


# ---------------------------------------------------------------- summary carries per_agent + cost


def test_summary_carries_per_agent_and_cost():
    trace = [
        _sanitize_entry(location_scrubbed=True),
        _signal_entry(has_text=True,
                      mentioned={"hazard_class": "D40", "lane_blocked": None, "road_type": None},
                      urgency_claimed=False, reporter_signals={}, contradictions=[]),
        _profile_entry(dataset_lookup={"found": True}),
        _rag_entry(coverage="direct_match", n_procedures=3,
                   decision_coverage={"sufficiency": "sufficient_coverage",
                                      "answered": ["repair_window"], "missing": []}),
        _decision_final_entry(action="human_review", source="llm_react",
                              worker_impact={"profile_llm_changed": True, "research_llm_changed": False,
                                             "rule_hint_without_worker_llm": "standard",
                                             "rule_hint_with_worker_llm": "human_review",
                                             "changed_rule_hint": True},
                              cited_sources=[{"url": "u", "name": "n"}],
                              rationale_check={"ok": True},
                              llm={"calls": 1, "tokens_in": 1000, "tokens_out": 150}),
    ]
    s = ra._build_agent_value_summary(trace)
    assert "per_agent_justified" in s
    pa = s["per_agent_justified"]
    assert set(pa.keys()) == {"sanitize", "signal_extractor", "supervisor",
                              "profile_builder", "rag_retriever", "decision"}
    assert pa["sanitize"]["justified"] is True
    assert pa["signal_extractor"]["justified"] is True
    assert pa["profile_builder"]["justified"] is True
    assert pa["rag_retriever"]["justified"] is True
    assert pa["decision"]["justified"] is True
    # supervisor did not run on this trace
    assert pa["supervisor"]["ran"] is False
    assert s["agents_justified_count"] == 5
    assert s["agents_total"] == 6
    assert s["cost"]["llm_calls"] == 1
    assert s["cost"]["tokens_in"] == 1000


# ---------------------------------------------------------------- trace footer


def test_trace_footer_renders_per_agent_lines_and_cost():
    trace = [
        _sanitize_entry(location_scrubbed=True),
        _decision_final_entry(action="human_review", source="llm_react",
                              worker_impact={"profile_llm_changed": True, "research_llm_changed": False,
                                             "rule_hint_without_worker_llm": "standard",
                                             "rule_hint_with_worker_llm": "human_review",
                                             "changed_rule_hint": True},
                              cited_sources=[{"url": "u", "name": "n"}],
                              rationale_check={"ok": True},
                              llm={"calls": 1, "tokens_in": 500, "tokens_out": 100}),
    ]
    lines = ra.format_trace_lines(
        "hz_test", trace,
        metadata={"class": "D40", "confidence": 0.6, "road_type": "main"},
        verdict={"action": "human_review", "priority": 50}, status="open",
    )
    body = "\n".join(lines)
    assert "agent justified:" in body
    assert "sanitize" in body and "decision" in body
    assert "YES" in body  # at least one agent was justified
    assert "run cost:" in body
    assert "tokens_in=500" in body
    assert "ollama_self_hosted=$0.00" in body
    assert "commercial_equivalent=$" in body


# ---------------------------------------------------------------- aggregate


def test_aggregate_returns_agents_justified_and_cost_total():
    class Row:
        def __init__(self, data):
            self._d = data
        def __getitem__(self, k):
            return self._d[k]
        def keys(self):
            return list(self._d.keys())

    trace = [
        _sanitize_entry(location_scrubbed=True),
        _decision_final_entry(action="human_review", source="llm_react",
                              worker_impact={"profile_llm_changed": True, "research_llm_changed": False,
                                             "rule_hint_without_worker_llm": "standard",
                                             "rule_hint_with_worker_llm": "human_review",
                                             "changed_rule_hint": True},
                              cited_sources=[{"url": "u", "name": "n"}],
                              rationale_check={"ok": True},
                              llm={"calls": 1, "tokens_in": 500, "tokens_out": 100}),
    ]
    summary = ra._build_agent_value_summary(trace)
    row = Row({
        "id": "hz_1", "class": "D40", "agent_decision": "human_review",
        "agent_trace_json": json.dumps(trace),
        "agent_value_json": json.dumps(summary),
    })
    out = _aggregate_agent_metrics([row])
    aj = out["agents_justified"]
    assert aj["sanitize"]["justified"] == 1
    assert aj["sanitize"]["runs"] == 1
    assert aj["decision"]["justified"] == 1
    assert aj["decision"]["runs"] == 1
    assert out["cost_total"]["llm_calls"] == 1
    assert out["cost_total"]["tokens_in"] == 500
    assert "hz_1" in out["advantage_over_tree_ids"]
