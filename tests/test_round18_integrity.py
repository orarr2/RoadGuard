"""Round 18 (2.11.2026 morning): the owner's six items from his r11 review.
A: signal extractor refuses to describe a hazard from a "no new info" text and
drops any LLM claim not supported by the keyword tables.
B: profile builder blocks LLM severity upgrades and flags rationale claims
that the actual numbers do not back.
C: a tool call whose arguments carry a different tool's final-answer shape is
rejected as a failed tool call, not accepted as a reply.
D: Decision payload carries source metadata; the fact-check flags urgency
wording when the retrieved standards did not answer that question.
E: a catalog-sealed decision reads source=catalog_match on the trace, not a
false llm_unavailable_or_unparseable label.
F: on an evidence_declined pass, the signal extractor skips the LLM entirely."""
import json

import pytest

import roadguard_agent as ra


# ---------------------------------------------------------------- A: signal extractor strict


def test_no_new_info_text_short_circuits_the_llm_entirely(monkeypatch):
    calls = []
    monkeypatch.setattr(ra, "_extract_by_llm", lambda *a, **k: calls.append(1) or (None, 0, 0, None))
    state = {
        "metadata": {"reporter_text": "No other photo is available, this is the only picture I took.",
                     "road_type": "main", "lane_blocked": "none", "location": "x"},
        "yolo_output": {"top1_class": "D40", "top1_confidence": 0.5, "detections": []},
    }
    update = ra.signal_extractor_node(state)
    assert calls == []
    entry = update["trace"][0]["content"]
    assert entry["skip_reason"] == "no_new_info"
    assert entry["source"] == "skipped:no_new_info"
    assert entry["mentioned"] == {"hazard_class": None, "lane_blocked": None, "road_type": None}
    assert entry["urgency_claimed"] is False
    assert all(v is None for v in entry["reporter_signals"].values())


@pytest.mark.parametrize(
    "text,expect_fired",
    [
        ("No other photo is available.", True),
        ("This is the only picture I took.", True),
        ("No more info to add.", True),
        ("I cannot provide another photo.", True),
        ("Nothing else to report.", True),
        ("Only one photo was taken.", True),
        ("Huge pothole, cars swerving everywhere!", False),
        ("", False),
    ],
)
def test_text_is_no_new_info_catches_the_common_phrases(text, expect_fired):
    assert ra._text_is_no_new_info(text) is expect_fired


def test_verify_llm_claims_drops_fields_not_in_the_text():
    parsed = {
        "mentioned": {"hazard_class": "D40", "lane_blocked": "partial", "road_type": "main"},
        "urgency_claimed": True,
        "summary": "s",
        "reporter_signals": ra._empty_signals(),
    }
    cleaned = ra._verify_llm_claims(dict(parsed, mentioned=dict(parsed["mentioned"])), "No other photo is available.")
    assert cleaned["mentioned"] == {"hazard_class": None, "lane_blocked": None, "road_type": None}
    assert cleaned["urgency_claimed"] is False
    assert set(cleaned["dropped_unsupported"]) == {"hazard_class", "lane_blocked", "road_type", "urgency_claimed"}


def test_verify_llm_claims_keeps_fields_the_text_does_say():
    text = "Huge pothole on the main road, cars swerving into the opposite lane, urgent!"
    parsed = {
        "mentioned": {"hazard_class": "D40", "lane_blocked": "partial", "road_type": "main"},
        "urgency_claimed": True,
    }
    cleaned = ra._verify_llm_claims(dict(parsed, mentioned=dict(parsed["mentioned"])), text)
    assert cleaned["mentioned"] == {"hazard_class": "D40", "lane_blocked": "partial", "road_type": "main"}
    assert cleaned["urgency_claimed"] is True
    assert "dropped_unsupported" not in cleaned


def test_signal_extractor_node_marks_source_as_llm_strict_when_claims_dropped(monkeypatch):
    monkeypatch.setattr(ra, "_extract_by_llm", lambda *a, **k: (
        {"mentioned": {"hazard_class": "D40", "lane_blocked": "partial", "road_type": "main"},
         "urgency_claimed": True, "summary": "s", "reporter_signals": ra._empty_signals()},
        5, 2, None,
    ))
    state = {
        "metadata": {"reporter_text": "Took a photo of something.",
                     "road_type": "main", "lane_blocked": "none", "location": "x"},
        "yolo_output": {"top1_class": "D40", "top1_confidence": 0.5, "detections": []},
    }
    update = ra.signal_extractor_node(state)
    content = update["trace"][0]["content"]
    assert content["source"] == "llm_strict"
    assert content["dropped_unsupported"] == ["hazard_class", "lane_blocked", "road_type", "urgency_claimed"]
    assert content["mentioned"] == {"hazard_class": None, "lane_blocked": None, "road_type": None}


def test_signal_extractor_node_keeps_source_llm_when_everything_is_grounded(monkeypatch):
    monkeypatch.setattr(ra, "_extract_by_llm", lambda *a, **k: (
        {"mentioned": {"hazard_class": "D40", "lane_blocked": "partial", "road_type": "main"},
         "urgency_claimed": True, "summary": "pothole", "reporter_signals": ra._empty_signals()},
        5, 2, None,
    ))
    state = {
        "metadata": {"reporter_text": "Urgent pothole on main road, cars swerving everywhere.",
                     "road_type": "main", "lane_blocked": "partial", "location": "x"},
        "yolo_output": {"top1_class": "D40", "top1_confidence": 0.5, "detections": []},
    }
    update = ra.signal_extractor_node(state)
    content = update["trace"][0]["content"]
    assert content["source"] == "llm"
    assert "dropped_unsupported" not in content


# ---------------------------------------------------------------- B: profile builder numeric gates


def _refine_with_llm(monkeypatch, raw, base_sev, yolo=None, iq=None, md=None):
    monkeypatch.setattr(ra, "worker_react_enabled", lambda: True)
    monkeypatch.setattr(ra, "worker_provider", lambda name: "ollama:test")
    monkeypatch.setattr(ra, "worker_autonomy_enabled", lambda: False)

    def fake_react(**_kw):
        return raw, 5, 2, {"hops": 0, "tools_run": {"inspect_image_quality": {"ok": True, "quality_verdict": "high"}}}

    monkeypatch.setattr(ra, "_llm_react_worker", fake_react)
    base = {"severity_signal": base_sev, "severity_rationale": "r", "is_recurring_location": False}
    return ra._llm_refine(
        base_profile=base, base_confidence="medium", base_needs_more=False, base_missing=[],
        yolo=yolo or {"top1_confidence": 0.5, "detections": [{"area_ratio": 0.02}]},
        iq=iq or {"blur_score": 120, "is_road_scene": True}, md=md or {"lane_blocked": "none", "location": "x"},
        state={},
    )


def test_severity_upgrade_from_llm_is_blocked_at_the_base(monkeypatch):
    raw = {"severity_signal": "severe", "severity_rationale": "D40 pothole, routine triage",
           "confidence": "medium", "needs_more_evidence": False, "missing_fields": []}
    refined, _, _ = _refine_with_llm(monkeypatch, raw, base_sev="minor")
    assert refined["severity_signal"] == "minor"
    assert any("severity_blocked_upgrade:severe_to_minor" in v for v in refined["autonomy"].get("violations", []))


def test_severity_downgrade_from_llm_needs_a_positive_signal(monkeypatch):
    """Round 21: a severity downgrade from the LLM is accepted only when the
    rationale names a positive low-severity cue (small size, surface crack,
    etc.); without that, the deterministic base severity stands."""
    with_signal = {"severity_signal": "minor",
                   "severity_rationale": "D40 small defect, surface crack only, routine",
                   "confidence": "medium", "needs_more_evidence": False, "missing_fields": []}
    refined, _, _ = _refine_with_llm(monkeypatch, with_signal, base_sev="moderate")
    assert refined["severity_signal"] == "minor"
    assert "severity_blocked_downgrade" not in str(refined.get("autonomy") or {})

    without_signal = {"severity_signal": "minor",
                      "severity_rationale": "D40 pothole, routine triage",
                      "confidence": "medium", "needs_more_evidence": False, "missing_fields": []}
    refined2, _, _ = _refine_with_llm(monkeypatch, without_signal, base_sev="moderate")
    assert refined2["severity_signal"] == "moderate"
    assert any("severity_blocked_downgrade:minor_to_moderate" in v for v in refined2["autonomy"]["violations"])


def test_rationale_confidence_claim_above_actual_is_flagged():
    rationale, problems = ra._verify_perception_rationale(
        "D40 with confidence >= 0.6 on main road",
        yolo={"top1_confidence": 0.3, "detections": []},
        iq={}, md={"lane_blocked": "none"},
    )
    assert any(p.startswith("confidence_claim_") for p in problems)


def test_rationale_area_ratio_claim_above_actual_is_flagged():
    rationale, problems = ra._verify_perception_rationale(
        "D40 with area_ratio>=0.05 severe",
        yolo={"top1_confidence": 0.9, "detections": [{"area_ratio": 0.01}]},
        iq={}, md={"lane_blocked": "none"},
    )
    assert any(p.startswith("area_ratio_claim_") for p in problems)


def test_rationale_lane_blockage_claim_when_lane_none_is_flagged():
    rationale, problems = ra._verify_perception_rationale(
        "D40 with partial lane blockage, severe",
        yolo={"top1_confidence": 0.9, "detections": [{"area_ratio": 0.1}]},
        iq={}, md={"lane_blocked": "none"},
    )
    assert "lane_blockage_claim_but_form_says_none" in problems


def test_grounded_rationale_passes_untouched():
    rationale, problems = ra._verify_perception_rationale(
        "D40 pothole, no reporter text, routine triage",
        yolo={"top1_confidence": 0.9, "detections": [{"area_ratio": 0.1}]},
        iq={}, md={"lane_blocked": "none"},
    )
    assert problems == []


# ---------------------------------------------------------------- C: ReAct cross-tool reply


def test_worker_tool_call_is_other_tools_reply_detects_cross_tool_schema():
    registry = ra.PERCEPTION_WORKER_TOOLS
    # Round 29 (6.11.2026 audit F6): the helper fires on ANOTHER worker's
    # final keys (here rag_retriever's) riding a lookup tool; the worker's
    # own final keys are its reply and are accepted before this check.
    bad_args = {
        "coverage": "direct_match",
        "key_findings": ["patch within 48h"],
        "cited_ids": ["p1"],
        "confidence": "medium",
    }
    assert ra._worker_tool_call_is_other_tools_reply(
        "profile_builder", "lookup_similar_cases", bad_args, registry,
    ) is True
    own_args = {
        "severity_signal": "minor",
        "severity_rationale": "D20 crack",
        "confidence": "medium",
        "needs_more_evidence": False,
    }
    assert ra._worker_tool_call_is_other_tools_reply(
        "profile_builder", "lookup_similar_cases", own_args, registry,
    ) is False


def test_worker_tool_call_is_other_tools_reply_false_for_actual_input():
    registry = ra.PERCEPTION_WORKER_TOOLS
    good_args = {"hazard_class": "D40", "location": "Herzl 5"}
    assert ra._worker_tool_call_is_other_tools_reply(
        "profile_builder", "lookup_similar_cases", good_args, registry,
    ) is False


def test_worker_tool_call_is_other_tools_reply_false_for_non_lookup_tool():
    # Round 18 fires only for the lookup/query tool prefixes. A tool called
    # "create_draft_work_order" is a write tool and its own schema handling
    # stays with _worker_final_in_tool_call.
    registry = ra.DECISION_TOOLS
    args = {"severity_signal": "minor", "severity_rationale": "D20 crack",
            "confidence": "medium", "needs_more_evidence": False}
    assert ra._worker_tool_call_is_other_tools_reply(
        "profile_builder", "create_draft_work_order", args, registry,
    ) is False


# ---------------------------------------------------------------- D: RAG source metadata + urgency fact-check


def _decision_state(answered_urgency=False, procedures=None):
    base = ra.initial_state(
        "hz_t", {"top1_class": "D40", "top1_confidence": 0.9, "detections": []}, {},
        {"road_type": "main", "lane_blocked": "none", "location": "x", "reporter_text": ""},
    )
    base.update({
        "hazard_profile": {"primary_class": "D40", "primary_confidence": 0.9, "severity_signal": "moderate",
                           "secondary_classes": [], "recurrence_count_30d": 0},
        "procedures": procedures or [{"article_ids": ["a1"], "source_url": "https://example/standard",
                                      "issuing_body": "DOT", "date": "2024-05-01",
                                      "quarantined_summary": {"text": "pothole repair standards"},
                                      "relevance": {"class_keyword_hit": True}}],
        "tool_result_ids": ["a1"], "research_coverage": "direct_match",
        "research_coverage_detail": {"repair_window": True, "urgency": answered_urgency,
                                     "traffic_control": False, "safety_risk": False,
                                     "answered": ["repair_window"] + (["urgency"] if answered_urgency else []),
                                     "missing": [],
                                     "sufficiency": "insufficient_coverage"},
        "perception_confidence": "medium", "reporter_intake": {},
    })
    return base


def test_decision_payload_carries_source_metadata():
    msgs = ra._decision_messages(
        _decision_state(),
        {"action": "standard", "priority": 60, "rationale": "r", "cited_sources": [], "missing_evidence": []},
    )
    payload = json.loads(msgs[1]["content"])
    procs = payload["procedures"]
    assert procs and procs[0]["source"]["url"] == "https://example/standard"
    assert procs[0]["source"]["date"] == "2024-05-01"


def test_urgency_wording_without_sources_is_flagged():
    state = _decision_state(answered_urgency=False)
    decision = {
        "action": "urgent", "priority": 80, "cited_sources": ["a1"], "missing_evidence": [],
        "rationale": "D40 pothole on main road; urgent dispatch recommended.",
        "assessment": "",
    }
    check = ra._verify_decision_rationale(decision, state)
    assert "urgency_not_in_sources" in check["problems"]


def test_urgency_wording_with_sources_is_allowed():
    state = _decision_state(answered_urgency=True)
    decision = {
        "action": "urgent", "priority": 80, "cited_sources": ["a1"], "missing_evidence": [],
        "rationale": "D40 pothole on main road; urgent per the retrieved standard.",
        "assessment": "",
    }
    check = ra._verify_decision_rationale(decision, state)
    assert "urgency_not_in_sources" not in check["problems"]


def test_repair_window_wording_still_requires_the_source_answer():
    state = _decision_state(answered_urgency=False)
    state["research_coverage_detail"]["repair_window"] = False
    state["research_coverage_detail"]["answered"] = []
    decision = {
        "action": "standard", "priority": 55, "cited_sources": ["a1"], "missing_evidence": [],
        "rationale": "D40 pothole; the standard sets a 24 hour repair window.",
        "assessment": "",
    }
    check = ra._verify_decision_rationale(decision, state)
    assert "repair_window_not_in_sources" in check["problems"]


# ---------------------------------------------------------------- E: catalog-sealed fallback label


def test_catalog_sealed_decision_reads_source_catalog_match(monkeypatch):
    # Stub the side-effect tools so decision_node runs end-to-end
    for name, result in (("create_draft_work_order", {"ok": True, "work_order_id": "wo_t"}),
                         ("request_additional_evidence", {"ok": True, "request_id": "er_t"})):
        schema = ra.DECISION_TOOLS[name][1]
        monkeypatch.setitem(ra.DECISION_TOOLS, name, (lambda result=result, **kw: result, schema))
    monkeypatch.setattr(ra, "_react_enabled", lambda: True)
    state = ra.initial_state(
        "hz_t", {"top1_class": "D40", "top1_confidence": 0.73, "detections": []}, {},
        {"road_type": "main", "lane_blocked": "none", "location": "x",
         "image_sha256": "a" * 64, "image_phash": "ff" * 8},
    )
    state.update({
        "hazard_profile": {"primary_class": "D40", "primary_confidence": 0.73, "severity_signal": "moderate",
                           "secondary_classes": [], "recurrence_count_30d": 0},
        "procedures": [], "tool_result_ids": [], "research_coverage": "direct_match",
        "perception_confidence": "medium", "reporter_intake": {},
        # A sealed catalog match: sha256 identity, same class, confidence floor met.
        "dataset_match": {
            "found": True, "match": "sha256", "pic_id": 1024, "classes": "D40",
            "model_prediction": {"top1_class": "D40", "top1_confidence": 0.73},
            "stored_confidence": 0.73, "distance": 0, "split": "train", "file_name": "x.jpg",
            "n_labels": 1,
        },
    })
    # No LLM calls should happen on the catalog path
    monkeypatch.setattr(ra, "_llm_decide_react", lambda *a, **k: (_ for _ in ()).throw(AssertionError("LLM called on catalog seal")))
    update = ra.decision_node(state)
    final = [e for e in update["trace"] if e["kind"] == "final"][0]["content"]
    assert update["action"] == "known_image"
    assert final["source"] == "catalog_match"
    assert "fallback_reason" not in final
    assert "fallback_reason" not in update


# ---------------------------------------------------------------- F: signal extractor skipped on evidence_declined


def test_signal_extractor_skipped_on_evidence_declined(monkeypatch):
    calls = []
    monkeypatch.setattr(ra, "_extract_by_llm", lambda *a, **k: calls.append(1) or (None, 0, 0, None))
    state = {
        "evidence_declined": True,
        "metadata": {"reporter_text": "Huge pothole, cars swerving, urgent!",
                     "road_type": "main", "lane_blocked": "none", "location": "x"},
        "yolo_output": {"top1_class": "D40", "top1_confidence": 0.5, "detections": []},
    }
    update = ra.signal_extractor_node(state)
    assert calls == []
    entry = update["trace"][0]["content"]
    assert entry["skip_reason"] == "evidence_declined"
    assert entry["source"] == "skipped:evidence_declined"
