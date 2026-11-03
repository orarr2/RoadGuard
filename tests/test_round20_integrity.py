"""Round 20 (3.11.2026 night): the owner's four items from his r13 review.
A: the Decision prompt's `similar_catalogue_photo` block carries nothing
   that identifies the neighbour (no pic_id, labels or prediction).
B: the Supervisor payload uses the `hazard_class` key the tool requires
   (r13 hz_0973521216da: 'hazard_class is required' every call).
C: when the fact-check names an input the inputs do not carry, the state
   fields that would otherwise feed that claim to a downstream agent are
   cleared.
D: the autonomy line tags its source: `agent_choice` when the agent chose
   a tool and `deterministic_fallback` when the code ran a tool for it."""
import json

import roadguard_agent as ra


# ---------------------------------------------------------------- A: phash neighbour hidden from prompt


def test_phash_neighbor_payload_hides_pic_id_labels_and_prediction():
    """A phash match delivers only the fact that an approximate match
    exists; the pic_id, labels and recorded prediction are absent so the
    LLM cannot copy them as evidence about this photo."""
    match = {
        "found": True, "match": "phash", "distance": 14, "pic_id": 10179,
        "classes": "D00", "model_prediction": {"top1_class": "D00", "top1_confidence": 0.32},
        "stored_confidence": 0.32, "split": "train", "file_name": "x.jpg", "n_labels": 1,
    }
    ctx = ra._catalogue_context_for_prompt(match)
    sim = ctx["similar_catalogue_photo"]
    assert set(sim.keys()) == {"note"}, f"extra identifying keys: {sorted(sim.keys())}"
    assert "HIDDEN" in sim["note"]
    assert "pic_id" in sim["note"]  # the rule is explained to the model


def test_sha256_match_still_delivers_the_full_dataset_match_block():
    """A sha256 identity delivers the full catalogue block as before — the
    hiding is only for approximate (phash) neighbours."""
    match = {
        "found": True, "match": "sha256", "distance": 0, "pic_id": 1024,
        "classes": "D40", "model_prediction": {"top1_class": "D40", "top1_confidence": 0.73},
        "stored_confidence": 0.73, "split": "train", "file_name": "x.jpg", "n_labels": 1,
    }
    ctx = ra._catalogue_context_for_prompt(match)
    assert ctx["similar_catalogue_photo"] is None
    assert ctx["dataset_match"] is not None
    assert ctx["dataset_match"]["pic_id"] == 1024


# ---------------------------------------------------------------- B: supervisor payload uses hazard_class


def test_supervisor_payload_uses_hazard_class_key(monkeypatch):
    """The supervisor's assess_input_complexity tool needs `hazard_class`;
    the payload used to send `hazard_class_detected` and failed every call."""
    monkeypatch.setattr(ra, "worker_provider", lambda name: "ollama:test")
    monkeypatch.setattr(ra, "worker_react_enabled", lambda: True)
    monkeypatch.setattr(ra, "worker_autonomy_enabled", lambda: False)
    seen = {}

    def fake_react(**kw):
        seen.update(kw)
        return ({"pipeline_concern": "none", "note": "ok", "expected_difficulty": "simple",
                 "force_action": None}, 5, 2, {"hops": 0, "tools_run": {}})

    monkeypatch.setattr(ra, "_llm_react_worker", fake_react)
    state = {
        "yolo_output": {"top1_class": "D20", "top1_confidence": 0.528, "detections": []},
        "metadata": {"road_type": "main", "lane_blocked": "none", "reporter_text": "",
                     "reporter_text_suspicious": False},
        "iterations": 0, "trace": [],
    }
    update = ra._maybe_run_supervisor_agent(state)
    # The supervisor was called with a payload that carries hazard_class.
    payload = seen.get("user_payload") or {}
    assert payload.get("hazard_class") == "D20"
    # The old key stays for back-compat with the note check.
    assert payload.get("hazard_class_detected") == "D20"
    # The tool itself accepts the key and does not error out.
    out = ra._tool_assess_input_complexity(hazard_class=payload["hazard_class"],
                                           detector_confidence=payload["detector_confidence"],
                                           has_reporter_text=payload["has_reporter_text"],
                                           lane_blocked=payload["lane_blocked"])
    assert out["ok"] is True and out["difficulty"] in {"simple", "mid", "hard"}


# ---------------------------------------------------------------- C: fact-check cleans state


def test_scrub_clears_reporter_intake_on_reporter_claims_without_text():
    state = {
        "reporter_intake": {
            "text": "",
            "mentioned": {"hazard_class": "D40", "lane_blocked": "partial", "road_type": "main"},
            "urgency_claimed": True,
            "summary": "s",
        },
        "reporter_signals": {"size_cm": 50, "depth_cm": 10, "vehicles_swerving": True, "standing_water": False},
    }
    cleared = ra._scrub_state_on_fact_check_failure(state, ["reporter_claims_without_text"])
    assert "reporter_intake.mentioned" in cleared
    assert "reporter_intake.urgency_claimed" in cleared
    assert "reporter_signals" in cleared
    assert state["reporter_intake"]["mentioned"] == {"hazard_class": None, "lane_blocked": None, "road_type": None}
    assert state["reporter_intake"]["urgency_claimed"] is False
    assert all(v is None for v in state["reporter_signals"].values())


def test_scrub_clears_lane_blocked_on_lane_blockage_not_in_form():
    state = {
        "reporter_intake": {
            "text": "pothole",
            "mentioned": {"hazard_class": "D40", "lane_blocked": "partial", "road_type": "main"},
            "urgency_claimed": False,
        },
        "reporter_signals": {"size_cm": None, "depth_cm": None, "vehicles_swerving": None, "standing_water": None},
    }
    cleared = ra._scrub_state_on_fact_check_failure(state, ["lane_blockage_not_in_form"])
    assert cleared == ["reporter_intake.mentioned.lane_blocked"]
    assert state["reporter_intake"]["mentioned"]["lane_blocked"] is None
    # Other mentioned fields and urgency_claimed stay as they were.
    assert state["reporter_intake"]["mentioned"]["hazard_class"] == "D40"


def test_scrub_is_a_no_op_when_no_matching_problem():
    state = {
        "reporter_intake": {"mentioned": {"hazard_class": "D40"}, "urgency_claimed": True},
        "reporter_signals": {"size_cm": 50},
    }
    cleared = ra._scrub_state_on_fact_check_failure(state, ["known_image_claim_without_sha256_match"])
    assert cleared == []
    assert state["reporter_intake"]["mentioned"]["hazard_class"] == "D40"
    assert state["reporter_intake"]["urgency_claimed"] is True


# ---------------------------------------------------------------- D: autonomy source tag


def test_autonomy_line_tags_agent_choice_when_hops_positive():
    entry = {"speaker": "profile_builder", "kind": "handoff_out", "content": {
        "primary_class": "D40", "severity": "minor", "confidence": "low", "llm_used": True,
        "autonomy": {"tools_used": ["inspect_image_quality"], "hops": 2, "violations": []},
    }}
    lines = ra._entry_lines(entry, [])
    auto_line = next(line for line in lines if "autonomy:" in line)
    assert "source=agent_choice" in auto_line


def test_autonomy_line_tags_deterministic_fallback_when_only_auto_ran():
    entry = {"speaker": "rag_retriever", "kind": "handoff_out", "content": {
        "coverage": "direct_match", "n_procedures": 3, "tool_result_ids": ["a1"], "llm_used": True,
        "autonomy": {"tools_used": [], "hops": 0, "auto_ran": ["query_repair_standards"],
                     "fallback": "deterministic_query (3 passages)", "violations": []},
    }}
    lines = ra._entry_lines(entry, [])
    auto_line = next(line for line in lines if "autonomy:" in line)
    assert "source=deterministic_fallback" in auto_line


def test_autonomy_line_keeps_explicit_source_when_already_set():
    entry = {"speaker": "rag_retriever", "kind": "handoff_out", "content": {
        "coverage": "direct_match", "n_procedures": 3, "tool_result_ids": ["a1"], "llm_used": True,
        "autonomy": {"tools_used": ["query_repair_standards"], "hops": 1,
                     "source": "agent_retrieval", "violations": []},
    }}
    lines = ra._entry_lines(entry, [])
    auto_line = next(line for line in lines if "autonomy:" in line)
    assert "source=agent_retrieval" in auto_line
    assert "source=agent_choice" not in auto_line
