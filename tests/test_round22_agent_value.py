"""Round 22 (4.11.2026): per-case agent value -- is the LLM / agent actually
changing what the deterministic rule tree alone would have decided?
A: the trace footer shows rule_tree / with_workers / llm / final and the
   change flags, built from the Decision final entry.
B: the DB stores the same summary on hazards.agent_value_json so
   SELECT WHERE json_extract(agent_value_json,'$.changed_rule_hint')=1
   answers the question in one SQL hop.
C: /stats/agents (and /digest/latest's KPI strip) carry the aggregate
   counters -- runs, changed_rule_hint, profile_llm_changed,
   research_llm_changed, floor_applied, enforcement_rule_tree_verdict.
"""
import json

import pytest

import roadguard_agent as ra
from roadguard_api import _aggregate_agent_metrics


# ---------------------------------------------------------------- A: summary from trace


def _decision_final(
    *,
    action="standard",
    source="llm_react",
    rule_tree_without="standard",
    rule_tree_with="standard",
    profile_changed=False,
    research_changed=False,
    floor_applied=False,
    floor_reason=None,
    overridden=None,
    enforcement=None,
):
    content = {
        "action": action,
        "priority": 50,
        "source": source,
        "rationale": "r",
        "cited_sources": [],
        "worker_impact": {
            "profile_llm_changed": bool(profile_changed),
            "research_llm_changed": bool(research_changed),
            "rule_hint_without_worker_llm": rule_tree_without,
            "rule_hint_with_worker_llm": rule_tree_with,
            "changed_rule_hint": rule_tree_without != rule_tree_with,
        },
    }
    if floor_applied:
        content["floor_applied"] = True
        content["floor_reason"] = floor_reason or "unnamed"
    if overridden:
        content["overridden_llm_verdict"] = overridden
    if enforcement is not None:
        content["rationale_check"] = {"ok": False, "problems": ["x"], "enforcement": enforcement}
    return {"speaker": "decision", "kind": "final", "content": content}


def test_summary_none_on_missing_decision_final():
    assert ra._build_agent_value_summary([]) is None
    assert ra._build_agent_value_summary([{"speaker": "sanitize", "kind": "handoff_out", "content": {}}]) is None
    assert ra._build_agent_value_summary(None) is None


def test_summary_worker_llm_changed_tree():
    """The classic r21 case: rule tree alone says standard, worker LLMs
    push it to human_review, final stays human_review."""
    final = _decision_final(
        action="human_review",
        source="llm_react",
        rule_tree_without="standard",
        rule_tree_with="human_review",
        profile_changed=True,
    )
    s = ra._build_agent_value_summary([final])
    assert s["rule_tree_action"] == "standard"
    assert s["rule_tree_with_worker_action"] == "human_review"
    assert s["llm_proposed_action"] == "human_review"
    assert s["final_action"] == "human_review"
    assert s["changed_rule_hint"] is True
    assert s["profile_llm_changed"] is True
    assert s["research_llm_changed"] is False
    assert s["decision_source"] == "llm_react"
    assert s["floor_applied"] is False
    assert s["enforcement"] is None


def test_summary_floor_overrides_llm_verdict():
    """The LLM proposed standard; the phash_class_mismatch floor fired
    and the final verdict is human_review. The summary carries BOTH the
    model's rejected text (via overridden_llm_verdict) and the floor."""
    final = _decision_final(
        action="human_review",
        source="llm_react",
        rule_tree_without="standard",
        rule_tree_with="standard",
        floor_applied=True,
        floor_reason="phash_class_mismatch",
        overridden={"action": "standard", "priority": 50, "rationale": "catalogue neighbour"},
    )
    s = ra._build_agent_value_summary([final])
    assert s["llm_proposed_action"] == "standard"
    assert s["final_action"] == "human_review"
    assert s["floor_applied"] is True
    assert s["floor_reason"] == "phash_class_mismatch"


def test_summary_rule_based_path_has_no_llm_action():
    final = _decision_final(
        action="standard",
        source="rule_based",
        rule_tree_without="standard",
        rule_tree_with="standard",
    )
    s = ra._build_agent_value_summary([final])
    assert s["decision_source"] == "rule_based"
    assert s["llm_proposed_action"] is None
    assert s["final_action"] == "standard"
    assert s["changed_rule_hint"] is False


# ---------------------------------------------------------------- B: DB column populated and queryable


def test_create_hazard_derives_agent_value_json(tmp_path):
    """A hazard inserted without a pre-computed agent_value_json still
    ends up with the summary stored in the DB column."""
    db_path = tmp_path / "test.db"
    with ra.Database(str(db_path)) as db:
        db.upsert_user("u1", "u@example.com", "u")
        trace = [_decision_final(
            action="human_review",
            source="llm_react",
            rule_tree_without="standard",
            rule_tree_with="human_review",
            profile_changed=True,
        )]
        h = ra.Hazard(
            id="hz_r22_a", user_id="u1", reported_at=1.0, hazard_class="D40",
            confidence=0.7, road_type="main", lane_blocked="none", location=None,
            bboxes_json="[]", priority=50, agent_decision="human_review",
            agent_rationale="r", agent_trace_json=json.dumps(trace),
            status="open", work_order_id=None,
        )
        db.create_hazard(h)
        out = db._conn.execute("SELECT agent_value_json FROM hazards WHERE id='hz_r22_a'").fetchone()
    stored = json.loads(out["agent_value_json"])
    assert stored["changed_rule_hint"] is True
    assert stored["rule_tree_action"] == "standard"
    assert stored["final_action"] == "human_review"


def test_update_hazard_verdict_refreshes_agent_value_json(tmp_path):
    """A verdict update after resumed evidence refreshes the column."""
    db_path = tmp_path / "test.db"
    with ra.Database(str(db_path)) as db:
        db.upsert_user("u1", "u@example.com", "u")
        initial_trace = [_decision_final(
            action="needs_more_evidence",
            source="rule_based",
            rule_tree_without="needs_more_evidence",
            rule_tree_with="needs_more_evidence",
        )]
        h = ra.Hazard(
            id="hz_r22_b", user_id="u1", reported_at=1.0, hazard_class="D40",
            confidence=0.7, road_type="main", lane_blocked="none", location=None,
            bboxes_json="[]", priority=20, agent_decision="needs_more_evidence",
            agent_rationale="r", agent_trace_json=json.dumps(initial_trace),
            status="open", work_order_id=None,
        )
        db.create_hazard(h)
        resumed_trace = [_decision_final(
            action="human_review",
            source="llm_react",
            rule_tree_without="standard",
            rule_tree_with="human_review",
            profile_changed=True,
        )]
        db.update_hazard_verdict(
            "hz_r22_b", priority=50, agent_decision="human_review",
            agent_rationale="r2", agent_trace_json=json.dumps(resumed_trace),
            cited_sources_json="[]", status="open", work_order_id=None,
        )
        row = db._conn.execute("SELECT agent_value_json FROM hazards WHERE id='hz_r22_b'").fetchone()
    stored = json.loads(row["agent_value_json"])
    assert stored["final_action"] == "human_review"
    assert stored["changed_rule_hint"] is True


# ---------------------------------------------------------------- C: aggregate counters


def _row(trace, decision="standard", clazz="D40"):
    """A fake hazards row shape _aggregate_agent_metrics expects."""
    class Row:
        def __init__(self, data):
            self._d = data

        def __getitem__(self, key):
            return self._d[key]

        def keys(self):
            return list(self._d.keys())

    return Row({
        "id": "x", "class": clazz, "agent_decision": decision,
        "agent_trace_json": json.dumps(trace),
    })


def test_aggregate_counts_changed_and_source():
    rows = [
        # case 1: worker LLM moved tree, LLM path
        _row([_decision_final(
            action="human_review", source="llm_react",
            rule_tree_without="standard", rule_tree_with="human_review",
            profile_changed=True)], decision="human_review"),
        # case 2: nothing changed, LLM path
        _row([_decision_final(
            action="standard", source="llm_react",
            rule_tree_without="standard", rule_tree_with="standard")],
             decision="standard"),
        # case 3: rule_based path, no LLM
        _row([_decision_final(
            action="standard", source="rule_based",
            rule_tree_without="standard", rule_tree_with="standard")],
             decision="standard"),
        # case 4: catalog_match short-circuit
        _row([_decision_final(
            action="known_image", source="catalog_match",
            rule_tree_without="known_image", rule_tree_with="known_image")],
             decision="known_image"),
        # case 5: floor fired, LLM overridden by tree
        _row([_decision_final(
            action="human_review", source="llm_react",
            rule_tree_without="standard", rule_tree_with="standard",
            floor_applied=True, floor_reason="phash_class_mismatch",
            overridden={"action": "standard", "priority": 45, "rationale": "x"})],
             decision="human_review"),
    ]
    out = _aggregate_agent_metrics(rows)
    av = out["agent_value"]
    assert av["runs"] == 5
    assert av["changed_rule_hint"] == 1
    assert av["profile_llm_changed"] == 1
    assert av["research_llm_changed"] == 0
    assert av["decision_llm_react"] == 3
    assert av["decision_rule_based"] == 1
    assert av["decision_catalog_match"] == 1
    assert av["floor_applied"] == 1
    # case 5: LLM proposed standard, rule tree without worker = standard,
    # so llm_more_severe_than_tree stays 0 and the floor still fired.
    # case 1: LLM proposed human_review, rule tree without worker = standard,
    # so llm_more_severe_than_tree = 1.
    assert av["llm_more_severe_than_tree"] == 1
    assert av["llm_less_severe_than_tree"] == 0


# ---------------------------------------------------------------- D: trace footer


def test_format_trace_lines_footer_includes_agent_value_when_run_finished():
    trace = [_decision_final(
        action="human_review", source="llm_react",
        rule_tree_without="standard", rule_tree_with="human_review",
        profile_changed=True)]
    lines = ra.format_trace_lines(
        "hz_x", trace,
        metadata={"class": "D40", "confidence": 0.6, "road_type": "main"},
        verdict={"action": "human_review", "priority": 50},
        status="open",
    )
    body = "\n".join(lines)
    assert "agent value:" in body
    assert "rule_tree=standard" in body
    assert "with_workers=human_review" in body
    assert "llm=human_review" in body
    assert "final=human_review" in body
    assert "changed_rule_hint" in body
    assert "profile_llm_changed" in body


def test_format_trace_lines_footer_omitted_while_processing():
    """Round in progress -- no DONE line and no agent value footer yet."""
    trace = [_decision_final(action="standard")]
    lines = ra.format_trace_lines(
        "hz_x", trace,
        metadata={"class": "D40", "confidence": 0.6, "road_type": "main"},
        status="processing",
    )
    body = "\n".join(lines)
    assert "agent value:" not in body
