"""Round 19 regression tests (3.11.2026 night): the five scenarios the owner
named in his r12 review. Each test replays one of the problem patterns from
r11 or r12 and asserts the round 17-19 guardrails hold.

The owner's five scenarios, quoted:
1. D40 at confidence 0.47 and lane none: the profiler cannot claim confidence
   above the threshold or lane blockage.
2. D40 at confidence 0.54: cannot raise severity based on conditions that are
   not in the inputs.
3. Phash neighbour with a different catalog label: used for comparison only,
   not for validating this photo's class.
4. Sha256 identical with a contradicting catalog label: HUMAN_REVIEW.
5. RAG returned insufficient_coverage: cannot cite a repair window or urgency
   as a fact in the Decision rationale."""
import json

import pytest

import roadguard_agent as ra


# ---------------------------------------------------------------- 1. profile builder cannot upgrade on invented numbers


def test_regression_1_profile_builder_cannot_claim_confidence_above_actual():
    """D40 at 0.47 with lane none: the LLM's rationale that says
    confidence>=0.6 and partial lane blockage is flagged in Python."""
    rationale, problems = ra._verify_perception_rationale(
        "D40 with partial lane blockage, area_ratio>=0.05 and confidence>=0.6 - severe per policy",
        yolo={"top1_confidence": 0.47, "detections": [{"area_ratio": 0.01}]},
        iq={}, md={"lane_blocked": "none"},
    )
    assert any(p.startswith("confidence_claim_0.60_vs_actual_0.47") for p in problems)
    assert any(p.startswith("area_ratio_claim_0.050_vs_actual_0.010") for p in problems)
    assert "lane_blockage_claim_but_form_says_none" in problems


# ---------------------------------------------------------------- 2. severity upgrade blocked at the base


def test_regression_2_severity_upgrade_from_the_llm_is_blocked_in_python(monkeypatch):
    """D40 at 0.54 with the LLM returning severe: Python clamps it to the
    deterministic base. The owner's hz_5288d2aad4fb from r11."""
    monkeypatch.setattr(ra, "worker_react_enabled", lambda: True)
    monkeypatch.setattr(ra, "worker_provider", lambda name: "ollama:test")
    monkeypatch.setattr(ra, "worker_autonomy_enabled", lambda: False)
    raw = {"severity_signal": "severe", "severity_rationale": "D40 pothole severe", "confidence": "medium",
           "needs_more_evidence": False, "missing_fields": []}
    monkeypatch.setattr(ra, "_llm_react_worker", lambda **_kw: (raw, 5, 2, {"hops": 0, "tools_run": {"inspect_image_quality": {"ok": True, "quality_verdict": "high"}}}))
    base = {"severity_signal": "moderate", "severity_rationale": "r", "is_recurring_location": False}
    refined, _, _ = ra._llm_refine(
        base_profile=base, base_confidence="medium", base_needs_more=False, base_missing=[],
        yolo={"top1_confidence": 0.54, "detections": [{"area_ratio": 0.03}]},
        iq={"blur_score": 120, "is_road_scene": True},
        md={"lane_blocked": "none", "location": "x"}, state={},
    )
    assert refined["severity_signal"] == "moderate"  # LLM wanted severe
    assert any("severity_blocked_upgrade:severe_to_moderate" in v for v in refined["autonomy"]["violations"])


# ---------------------------------------------------------------- 3. phash neighbour is not evidence about this photo


def test_regression_3_phash_neighbor_pic_id_cannot_be_cited_as_evidence():
    """r12 hz_472b131e8248: live D20 at 0.26, phash neighbour pic_id=10179
    labelled D00. The model's rationale that cites pic_id=10179 is flagged."""
    state = {
        "hazard_profile": {"primary_class": "D20", "primary_confidence": 0.26, "secondary_classes": []},
        "metadata": {"road_type": "main", "lane_blocked": "none"},
        "yolo_output": {"top1_class": "D20", "top1_confidence": 0.26, "detections": []},
        "reporter_intake": {},
        "research_coverage": "direct_match",
        "research_coverage_detail": {"repair_window": False, "urgency": False, "answered": [],
                                     "missing": ["repair_window", "urgency"],
                                     "sufficiency": "insufficient_coverage"},
        "dataset_match": {
            "found": True, "match": "phash", "distance": 14, "pic_id": 10179,
            "classes": "D00", "model_prediction": {"top1_class": "D00", "top1_confidence": 0.32},
        },
    }
    dec = {
        "action": "standard", "priority": 55,
        "cited_sources": [], "missing_evidence": [],
        "rationale": "D20 crack on main road; cataloged pic_id=10179 is consistent with the image.",
        "assessment": "",
    }
    rc = ra._verify_decision_rationale(dec, state)
    assert any(p.startswith("phash_neighbor_cited_as_evidence:pic_id=10179") for p in rc["problems"])
    # The payload itself names the rule in words, as a defence in depth.
    ctx = ra._catalogue_context_for_prompt(state["dataset_match"])
    # Round 20: payload hides the pic_id entirely instead of just forbidding it.
    assert "HIDDEN" in ctx["similar_catalogue_photo"]["note"]


# ---------------------------------------------------------------- 4. sha256 identity with a contradicting catalog label → human_review


def test_regression_4_sha256_match_with_contradicting_label_goes_to_human_review():
    """A catalogued file whose stored class differs from the live detection
    is routed to a human (round 11 known_image contradiction floor)."""
    state = {
        "hazard_profile": {"primary_class": "D20", "primary_confidence": 0.6, "severity_signal": "moderate"},
        "metadata": {"road_type": "main", "lane_blocked": "none", "location": "x"},
        "yolo_output": {"top1_class": "D20", "top1_confidence": 0.6, "detections": []},
        "reporter_intake": {},
        "research_coverage": "direct_match",
        "tool_result_ids": ["a1"],
        "procedures": [],
        "perception_confidence": "medium",
        "dataset_match": {
            "found": True, "match": "sha256", "distance": 0, "pic_id": 467,
            "classes": "D00",  # catalog says D00, but live detection is D20
            "model_prediction": {"top1_class": "D00", "top1_confidence": 0.9},
            "stored_confidence": 0.9, "split": "train", "file_name": "x.jpg", "n_labels": 1,
        },
    }
    decision = ra._decide_rule_based(state)
    assert decision["action"] == "human_review"
    assert "catalog" in decision["rationale"].lower() or "contradict" in decision["rationale"].lower()


# ---------------------------------------------------------------- 5. RAG insufficient_coverage: no repair window / urgency facts


def test_regression_5_insufficient_coverage_blocks_repair_window_and_urgency_claims():
    """When the retrieved standards did not answer the repair-window or
    urgency questions, the Decision rationale that states a window or an
    urgency is flagged by the fact-check."""
    state = {
        "hazard_profile": {"primary_class": "D40", "primary_confidence": 0.9, "secondary_classes": []},
        "metadata": {"road_type": "main", "lane_blocked": "none"},
        "yolo_output": {"top1_class": "D40", "top1_confidence": 0.9, "detections": []},
        "reporter_intake": {},
        "research_coverage": "direct_match",
        "research_coverage_detail": {
            "repair_window": False, "urgency": False, "traffic_control": False, "safety_risk": False,
            "answered": [], "missing": ["repair_window", "urgency"],
            "sufficiency": "insufficient_coverage",
        },
    }
    # Both wordings should trip
    dec_window = {
        "action": "standard", "priority": 60, "cited_sources": [], "missing_evidence": [],
        "rationale": "D40 pothole; the retrieved standard sets a 24 hour repair window.",
        "assessment": "",
    }
    rc_w = ra._verify_decision_rationale(dec_window, state)
    assert "repair_window_not_in_sources" in rc_w["problems"]

    dec_urgent = {
        "action": "urgent", "priority": 80, "cited_sources": [], "missing_evidence": [],
        "rationale": "D40 pothole; urgent dispatch recommended per the standard.",
        "assessment": "",
    }
    rc_u = ra._verify_decision_rationale(dec_urgent, state)
    assert "urgency_not_in_sources" in rc_u["problems"]
