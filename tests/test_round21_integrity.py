"""Round 21 (3.11.2026 late night): the owner's four items from his r13 review
on PREVENTION, not detection.
A: a phash neighbour whose catalog label disagrees with the live detector
   is routed to human_review by the rule tree (the Decision agent and the
   rule tree both see this).
B: the operator trace no longer prints the model's rejected text under the
   FAIL line; the deterministic sentence stands alone.
C: the Profile Builder's LLM may not downgrade severity without a positive
   low-severity cue in its rationale.
D: the deterministic rationale sentence, when research was insufficient,
   lists the questions the standards did not answer."""
import roadguard_agent as ra


# ---------------------------------------------------------------- A: phash class mismatch floor


def _state_with_phash(live_class, live_conf, catalog_classes, pic_id=10179, distance=8):
    return {
        "hazard_profile": {"primary_class": live_class, "primary_confidence": live_conf,
                           "severity_signal": "moderate", "recurrence_count_30d": 0},
        "metadata": {"road_type": "main", "lane_blocked": "none"},
        "yolo_output": {"top1_class": live_class, "top1_confidence": live_conf, "detections": []},
        "reporter_intake": {},
        "research_coverage": "direct_match",
        "procedures": [],
        "tool_result_ids": [],
        "perception_confidence": "medium",
        "dataset_match": {
            "found": True, "match": "phash", "distance": distance, "pic_id": pic_id,
            "classes": catalog_classes,
            "model_prediction": {"top1_class": catalog_classes.split(",")[0], "top1_confidence": 0.3},
            "stored_confidence": 0.3, "split": "train", "file_name": "x.jpg", "n_labels": 1,
        },
    }


def test_phash_neighbor_class_mismatch_routes_to_human_review():
    """r13 hz_770ac7c30461: live D20, catalog neighbour D40, verdict STANDARD.
    Round 21 says human_review."""
    state = _state_with_phash(live_class="D20", live_conf=0.6, catalog_classes="D40")
    decision = ra._decide_rule_based(state)
    assert decision["action"] == "human_review"
    assert decision["floor_applied"] is True
    assert decision["floor_reason"] == "phash_class_mismatch"
    assert "phash" in decision["rationale"].lower()
    assert "D40" in str(decision["floor_trigger"])


def test_phash_neighbor_beyond_the_floor_radius_is_context_only():
    """6.11.2026 audit F5: r23 sent 22 of 94 cases to a human on neighbours
    at distance 12-16; beyond policy known_image.phash_floor_max_distance
    (10) the neighbour is cited but does not decide."""
    state = _state_with_phash(live_class="D20", live_conf=0.7, catalog_classes="D40", distance=14)
    decision = ra._decide_rule_based(state)
    assert decision.get("floor_reason") != "phash_class_mismatch"
    assert decision["action"] == "standard"
    assert "catalog:pic_id=10179" in decision["cited_sources"]


def test_phash_neighbor_class_agreement_lets_the_tree_decide():
    """When the catalog neighbour's class agrees with the live detector,
    phash is supporting context; the rule tree picks the normal verdict."""
    state = _state_with_phash(live_class="D40", live_conf=0.7, catalog_classes="D40")
    decision = ra._decide_rule_based(state)
    # Not forced into human_review by the phash floor; the regular tree stands.
    assert decision["action"] in {"standard", "urgent"}
    assert decision.get("floor_reason") != "phash_class_mismatch"


def test_sha256_match_is_unaffected_by_the_phash_floor():
    """A sha256 identity match is handled by _known_image_decision; the
    phash floor must not touch it."""
    state = _state_with_phash(live_class="D20", live_conf=0.7, catalog_classes="D40")
    state["dataset_match"]["match"] = "sha256"
    state["dataset_match"]["distance"] = 0
    decision = ra._decide_rule_based(state)
    # sha256 + class disagreement is caught by _known_image_decision as the
    # catalog_detector_contradiction floor, not the round 21 phash floor.
    assert decision.get("floor_reason") in {"known_image_catalog_detector_contradiction", None}
    assert decision["action"] == "human_review"


# ---------------------------------------------------------------- B: operator trace hides rejected model text


def test_trace_hides_model_rationale_and_assessment_under_fail_line():
    failed = {"speaker": "decision", "kind": "final", "content": {
        "action": "standard", "priority": 45, "source": "llm_react",
        "rationale": "routine queue for D40", "cited_sources": [],
        "rationale_check": {
            "ok": False,
            "problems": ["reporter_claims_without_text"],
            "model_rationale": "The reporter's claims are consistent with the pothole description",
            "model_assessment": "Reporter saw cars swerving",
        },
    }}
    text = "\n".join(ra._entry_lines(failed, []))
    assert "rationale check: FAIL" in text
    assert "deterministic sentence shown (model text kept on rationale_check for the audit)" in text
    assert "model rationale:" not in text
    assert "model assessment:" not in text
    assert "The reporter's claims are consistent" not in text


# ---------------------------------------------------------------- C: profile builder downgrade needs a signal


def _refine_with_llm(monkeypatch, raw, base_sev, yolo=None, iq=None, md=None):
    monkeypatch.setattr(ra, "worker_react_enabled", lambda: True)
    monkeypatch.setattr(ra, "worker_provider", lambda name: "ollama:test")
    monkeypatch.setattr(ra, "worker_autonomy_enabled", lambda: False)
    monkeypatch.setattr(ra, "_llm_react_worker", lambda **_kw: (raw, 5, 2, {"hops": 0, "tools_run": {"inspect_image_quality": {"ok": True, "quality_verdict": "high"}}}))
    base = {"severity_signal": base_sev, "severity_rationale": "r", "is_recurring_location": False}
    return ra._llm_refine(
        base_profile=base, base_confidence="medium", base_needs_more=False, base_missing=[],
        yolo=yolo or {"top1_confidence": 0.6, "detections": [{"area_ratio": 0.03}]},
        iq=iq or {"blur_score": 120, "is_road_scene": True},
        md=md or {"lane_blocked": "none", "location": "x"}, state={},
    )


def test_profile_downgrade_without_positive_signal_is_blocked(monkeypatch):
    """A rationale that says only "routine triage" without a low-severity
    cue does not justify a downgrade; the base severity stands."""
    raw = {"severity_signal": "minor", "severity_rationale": "D40, no past reports, routine",
           "confidence": "medium", "needs_more_evidence": False, "missing_fields": []}
    refined, _, _ = _refine_with_llm(monkeypatch, raw, base_sev="severe")
    assert refined["severity_signal"] == "severe"
    assert any("severity_blocked_downgrade:minor_to_severe" in v for v in refined["autonomy"]["violations"])


def test_profile_downgrade_with_a_positive_signal_is_accepted(monkeypatch):
    """A rationale that names a low-severity cue (small size, surface crack)
    is accepted."""
    raw = {"severity_signal": "minor", "severity_rationale": "D20, surface crack only, narrow crack, light severity",
           "confidence": "medium", "needs_more_evidence": False, "missing_fields": []}
    refined, _, _ = _refine_with_llm(monkeypatch, raw, base_sev="moderate")
    assert refined["severity_signal"] == "minor"
    violations_str = str(refined.get("autonomy") or {})
    assert "severity_blocked_downgrade" not in violations_str


# ---------------------------------------------------------------- D: deterministic sentence names missing questions


def test_deterministic_sentence_names_missing_research_questions():
    state = {
        "hazard_profile": {"primary_class": "D40", "primary_confidence": 0.7},
        "metadata": {"road_type": "main", "lane_blocked": "none"},
        "reporter_intake": {},
        "research_coverage": "direct_match",
        "research_coverage_detail": {
            "sufficiency": "insufficient_coverage",
            "answered": ["urgency"],
            "missing": ["repair_window", "traffic_control", "safety_risk"],
        },
    }
    sentence = ra._deterministic_rationale({"action": "standard"}, state)
    assert "insufficient_coverage" in sentence
    assert "standards did not answer" in sentence
    assert "repair_window" in sentence


def test_deterministic_sentence_omits_the_note_when_coverage_is_sufficient():
    state = {
        "hazard_profile": {"primary_class": "D40", "primary_confidence": 0.7},
        "metadata": {"road_type": "main", "lane_blocked": "none"},
        "reporter_intake": {},
        "research_coverage": "direct_match",
        "research_coverage_detail": {
            "sufficiency": "sufficient_coverage",
            "answered": ["repair_window", "urgency"],
            "missing": [],
        },
    }
    sentence = ra._deterministic_rationale({"action": "standard"}, state)
    assert "standards did not answer" not in sentence
