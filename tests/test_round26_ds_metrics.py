"""Round 26 (5.11.2026): the 20 data-science metrics the owner asked for
in the Agent value tab. Ground truth comes from in-system worker feedback
(agent_feedback), not an external labeled catalog. Metrics that need
feedback return None (shown as 'pending' in the UI); metrics derivable
from the trace alone populate with any n>=1 hazard.
"""
import json

import pytest

import roadguard_api as rg


def _row(**kwargs):
    """Sqlite3.Row stand-in that supports both subscript and keys()."""
    class _R(dict):
        def keys(self):
            return list(super().keys())
        def __getitem__(self, k):
            return super().__getitem__(k)
    return _R(kwargs)


def _hazard(hid, action, priority=50, av=None, trace=None, confidence=0.3, cls="D40"):
    return _row(
        id=hid, class_=cls, confidence=confidence,
        priority=priority, agent_decision=action,
        agent_value_json=json.dumps(av or {}),
        agent_trace_json=json.dumps(trace or []),
    )


def _fb(hid, kind, verdict):
    return _row(hazard_id=hid, kind=kind, verdict=verdict,
                agree=1 if verdict == "agree" else 0)


# ---------------------------------------------------------------- shape

def test_empty_window_returns_zero_shape():
    out = rg._compute_ds_metrics_block([], [])
    assert out["n_hazards"] == 0
    assert out["feedback_agreement"]["hazard_valid_rate"] is None
    assert out["ds_metrics"]["confusion_matrix"] == [[0, 0, 0]] * 4
    assert out["ds_metrics"]["precision_macro_pct"] is None
    assert out["ds_metrics"]["roc_auc"] is None


def test_no_feedback_populates_uplift_and_cost_only():
    rows = [_hazard("h1", "standard", av={
        "rule_tree": "standard", "final": "standard", "llm": "-",
        "cost": {"llm_calls": 2, "tokens_in": 300, "tokens_out": 20,
                 "wall_seconds": 5.0, "cost_commercial_equivalent_usd": 0.001},
    })]
    out = rg._compute_ds_metrics_block(rows, [])
    assert out["uplift"]["rule_tree_eq_final_rate"] == 100.0
    assert out["cost_ext"]["tokens_in_mean"] == 300.0
    assert out["feedback_agreement"]["hazard_valid_rate"] is None


# ---------------------------------------------------------------- M1-M5


def test_hazard_valid_rate():
    rows = [_hazard(f"h{i}", "standard", av={"rule_tree": "standard",
                                             "final": "standard", "llm": "-"})
            for i in range(4)]
    fb = [_fb("h0", "hazard", "agree"), _fb("h1", "hazard", "agree"),
          _fb("h2", "hazard", "disagree"), _fb("h3", "hazard", "not_relevant")]
    out = rg._compute_ds_metrics_block(rows, fb)
    assert out["feedback_agreement"]["hazard_valid_rate"] == 50.0
    # M2 disagree-rate is rendered from hazard_counts in JS; check counts.
    hc = out["feedback_agreement"]["hazard_counts"]
    assert hc == {"agree": 2, "disagree": 1, "not_relevant": 1}


def test_priority_correct_rate():
    rows = [_hazard("h0", "standard")]
    fb = [_fb("h0", "priority", "ok"), _fb("h0", "priority", "too_low")]
    # latest wins per (hazard, kind), so the input to the function must
    # already be the LATEST verdict per pair. This mirrors the SQL.
    out = rg._compute_ds_metrics_block(rows, fb[-1:])
    assert out["feedback_agreement"]["priority_correct_rate"] == 0.0


def test_safety_floor_precision_only_for_floor_fired_with_feedback():
    rows = [
        _hazard("fir", "human_review", av={
            "rule_tree": "standard", "final": "human_review", "llm": "-",
            "flags": "floor:phash_class_mismatch", "floor_applied": True,
        }),
        _hazard("not_fir", "standard", av={
            "rule_tree": "standard", "final": "standard", "llm": "-",
        }),
    ]
    fb = [_fb("fir", "hazard", "agree"), _fb("not_fir", "hazard", "agree")]
    out = rg._compute_ds_metrics_block(rows, fb)
    assert out["feedback_agreement"]["safety_floor_n"] == 1
    assert out["feedback_agreement"]["safety_floor_precision"] == 100.0


def test_llm_uplift_precision_only_for_flipped_cases():
    rows = [
        _hazard("flipped", "human_review", av={
            "rule_tree": "standard", "final": "human_review",
            "llm": "human_review", "flags": "research_llm_changed",
        }),
        _hazard("no_flip", "standard", av={
            "rule_tree": "standard", "final": "standard", "llm": "-",
        }),
    ]
    fb = [_fb("flipped", "hazard", "agree"), _fb("no_flip", "hazard", "disagree")]
    out = rg._compute_ds_metrics_block(rows, fb)
    assert out["feedback_agreement"]["llm_uplift_n"] == 1
    assert out["feedback_agreement"]["llm_uplift_precision"] == 100.0


# ---------------------------------------------------------------- M6-M9


def test_uplift_rates_and_clean_wins():
    rows = [
        _hazard("a", "standard", av={"rule_tree": "standard", "final": "standard", "llm": "-"}),
        _hazard("b", "human_review", av={"rule_tree": "standard", "final": "human_review",
                                         "llm": "human_review",
                                         "flags": "research_llm_changed"}),
        _hazard("c", "human_review", av={"rule_tree": "standard", "final": "human_review",
                                         "llm": "standard",
                                         "flags": "floor:phash_class_mismatch",
                                         "floor_applied": True}),
    ]
    out = rg._compute_ds_metrics_block(rows, [])
    assert out["uplift"]["rule_tree_eq_final_rate"] == pytest.approx(33.3, abs=0.1)
    assert out["uplift"]["floor_fire_rate"] == pytest.approx(33.3, abs=0.1)
    assert out["uplift"]["llm_flip_rate"] == pytest.approx(33.3, abs=0.1)
    # Only 'b' is a clean win: LLM flipped and no floor fired.
    assert out["uplift"]["llm_clean_wins"] == 1


# ---------------------------------------------------------------- M10-M13


def test_cost_p95_and_mean():
    rows = []
    for i in range(10):
        rows.append(_hazard(f"h{i}", "standard", av={
            "rule_tree": "standard", "final": "standard", "llm": "-",
            "cost": {"llm_calls": i + 1, "tokens_in": 100 * (i + 1),
                     "tokens_out": 10 * (i + 1), "wall_seconds": float(i + 1),
                     "cost_commercial_equivalent_usd": 0.001 * (i + 1)},
        }))
    out = rg._compute_ds_metrics_block(rows, [])
    assert out["cost_ext"]["tokens_in_mean"] == 550.0
    assert out["cost_ext"]["tokens_in_p95"] in (900, 1000)
    assert out["cost_ext"]["llm_calls_per_case_mean"] == 5.5


# ---------------------------------------------------------------- M14-M15


def test_justified_histogram():
    rows = [
        _hazard("h0", "standard", av={"rule_tree": "s", "final": "s",
            "per_agent_justified": {"a": {"justified": True}, "b": {"justified": True}}}),
        _hazard("h1", "standard", av={"rule_tree": "s", "final": "s",
            "per_agent_justified": {"a": {"justified": True}, "b": {"justified": True},
                                    "c": {"justified": True}}}),
        _hazard("h2", "standard", av={"rule_tree": "s", "final": "s",
            "per_agent_justified": {}}),
    ]
    out = rg._compute_ds_metrics_block(rows, [])
    assert out["justified_histogram"][2] == 1
    assert out["justified_histogram"][3] == 1
    assert out["justified_histogram"][0] == 1


# ---------------------------------------------------------------- M16-M18 confusion + precision


def test_confusion_matrix_shape_and_per_class_precision():
    rows = [
        _hazard("h1", "standard"),
        _hazard("h2", "standard"),
        _hazard("h3", "human_review"),
    ]
    fb = [_fb("h1", "hazard", "agree"), _fb("h2", "hazard", "disagree"),
          _fb("h3", "hazard", "agree")]
    out = rg._compute_ds_metrics_block(rows, fb)
    cm = out["ds_metrics"]["confusion_matrix"]
    assert len(cm) == 4 and all(len(row) == 3 for row in cm)
    rows_lbl = out["ds_metrics"]["confusion_rows"]
    cols_lbl = out["ds_metrics"]["confusion_cols"]
    std_i = rows_lbl.index("standard")
    hr_i = rows_lbl.index("human_review")
    agr_j = cols_lbl.index("agree")
    dis_j = cols_lbl.index("disagree")
    assert cm[std_i][agr_j] == 1
    assert cm[std_i][dis_j] == 1
    assert cm[hr_i][agr_j] == 1
    pc = out["ds_metrics"]["per_class_precision"]
    assert pc["standard"]["precision_pct"] == 50.0
    assert pc["human_review"]["precision_pct"] == 100.0
    assert out["ds_metrics"]["precision_macro_pct"] == 75.0


# ---------------------------------------------------------------- M19 ROC-AUC


def test_roc_auc_higher_priority_on_agreed_gives_auc_1():
    rows = [_hazard("h1", "standard", priority=80),
            _hazard("h2", "standard", priority=20)]
    fb = [_fb("h1", "hazard", "agree"), _fb("h2", "hazard", "disagree")]
    out = rg._compute_ds_metrics_block(rows, fb)
    assert out["ds_metrics"]["roc_auc"] == 1.0


def test_roc_auc_none_when_single_class():
    rows = [_hazard("h1", "standard", priority=80)]
    fb = [_fb("h1", "hazard", "agree")]
    out = rg._compute_ds_metrics_block(rows, fb)
    assert out["ds_metrics"]["roc_auc"] is None


# ---------------------------------------------------------------- M20 grounding


def test_grounding_score_citation_in_rag_passages():
    trace = [
        {"speaker": "rag_retriever", "content": {
            "procedures": [{"sec": "sec-1.2"}, {"sec": "sec-4.3"}]}},
        {"speaker": "decision", "kind": "final", "content": {
            "action": "standard", "cited_sources": ["sec-1.2"]}},
    ]
    rows = [_hazard("h1", "standard", trace=trace)]
    out = rg._compute_ds_metrics_block(rows, [])
    assert out["ds_metrics"]["grounding_n"] == 1
    assert out["ds_metrics"]["grounding_score_pct"] == 100.0


def test_grounding_score_citation_missing_from_passages():
    trace = [
        {"speaker": "rag_retriever", "content": {
            "procedures": [{"sec": "sec-9.9"}]}},
        {"speaker": "decision", "kind": "final", "content": {
            "action": "standard", "cited_sources": ["sec-1.2"]}},
    ]
    rows = [_hazard("h1", "standard", trace=trace)]
    out = rg._compute_ds_metrics_block(rows, [])
    assert out["ds_metrics"]["grounding_n"] == 1
    assert out["ds_metrics"]["grounding_score_pct"] == 0.0


# ---------------------------------------------------------------- real schema

def test_real_schema_action_suffix_keys_resolve():
    """The actual agent_value_json in the DB uses *_action suffixes
    (rule_tree_action, final_action, llm_proposed_action), not the
    trace-render short names. The compute must read both."""
    rows = [
        _hazard("h1", "human_review", av={
            "rule_tree_action": "human_review",
            "final_action": "human_review",
            "llm_proposed_action": "human_review",
            "floor_applied": False,
        }),
        _hazard("h2", "human_review", av={
            "rule_tree_action": "standard",
            "final_action": "human_review",
            "llm_proposed_action": "standard",
            "floor_applied": True,
            "floor_reason": "phash_class_mismatch",
        }),
    ]
    out = rg._compute_ds_metrics_block(rows, [])
    assert out["uplift"]["rule_tree_eq_final_rate"] == 50.0
    assert out["uplift"]["floor_fire_rate"] == 50.0
    # LLM did not move the rule tree on either case.
    assert out["uplift"]["llm_flip_rate"] == 0.0
